use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use super::{
    VisionImpl, VisionRequest, VisionResponse, VisionRuntime,
    RegisteredVision, VisionRegistryKey,
};
use crate::config::{DeviceSelection, NeuronConfig, ResourceTracker};
use crate::llm::{ChatResult, ModelInfo, NeuronError};

#[cfg(feature = "candle")]
use crate::CandleYoloRuntime;

use crate::OrtVisionRuntime;

pub struct VisionRuntimeRegistry {
    pub registered: Vec<RegisteredVision>,
    loaded: Arc<tokio::sync::Mutex<HashMap<VisionRegistryKey, Arc<dyn VisionRuntime>>>>,
    resources: Mutex<ResourceTracker>,
}

impl Clone for VisionRuntimeRegistry {
    fn clone(&self) -> Self {
        Self {
            registered: self.registered.clone(),
            loaded: self.loaded.clone(),
            resources: Mutex::new(ResourceTracker::default()),
        }
    }
}

impl VisionRuntimeRegistry {
    pub fn new(_config: Arc<NeuronConfig>) -> Self {
        Self {
            registered: Vec::new(),
            loaded: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            resources: Mutex::new(ResourceTracker::default()),
        }
    }

    pub fn register(
        &mut self,
        model_id: String,
        impl_id: VisionImpl,
        device: DeviceSelection,
        required_memory_bytes: u64,
    ) -> Result<(), NeuronError> {
        let mut res = self.resources.lock().unwrap();
        if !res.can_load(required_memory_bytes, &device) {
            let available = match &device {
                DeviceSelection::Cpu | DeviceSelection::Auto => res.available_cpu_bytes(),
                _ => res.available_gpu_bytes(),
            };
            return Err(NeuronError::new(format!(
                "InsufficientResources: model={model_id}, required={required_memory_bytes}, available={available}"
            )));
        }
        res.allocate(required_memory_bytes, &device)
            .map_err(NeuronError::new)?;

        self.registered.push(RegisteredVision {
            model_id,
            impl_id,
            device,
            required_memory_bytes,
        });
        Ok(())
    }

    pub async fn runtime_for(&self, model_id: &str) -> Result<Arc<dyn VisionRuntime>, NeuronError> {
        let entry = self
            .registered
            .iter()
            .find(|r| r.model_id == model_id)
            .ok_or_else(|| NeuronError::new(format!("ModelNotRegistered: {model_id}")))?;

        let key = VisionRegistryKey {
            model_id: entry.model_id.clone(),
            impl_id: entry.impl_id,
            device: entry.device,
        };

        let mut loaded = self.loaded.lock().await;
        if let Some(rt) = loaded.get(&key) {
            return Ok(rt.clone());
        }

        let rt: Arc<dyn VisionRuntime> = match entry.impl_id {
            #[cfg(feature = "candle")]
            VisionImpl::CandleYolo => {
                let loaded_box: Box<dyn VisionRuntime> = CandleYoloRuntime::load(&entry.model_id, &entry.device)
                    .await
                    .map_err(|e| {
                        NeuronError::new(format!("CandleYoloRuntime load failed: {e}"))
                    })?;
                Arc::from(loaded_box)
            }
            #[cfg(feature = "candle")]
            VisionImpl::CandleRtDetr | VisionImpl::CandleDetr => {
                return Err(NeuronError::not_implemented(format!(
                    "{:?} backend: architecture not yet wired (v1.0 only CandleYoloRuntime is implemented)",
                    entry.impl_id
                )));
            }
            VisionImpl::Ort => {
                let loaded_box: Box<dyn VisionRuntime> = OrtVisionRuntime::load(
                    &entry.model_id,
                    &entry.device,
                )
                .await
                .map_err(|e| {
                    NeuronError::new(format!("OrtVisionRuntime load failed: {e}"))
                })?;
                Arc::from(loaded_box)
            }
        };

        loaded.insert(key, rt.clone());
        Ok(rt)
    }

    pub fn list_registered(&self) -> Vec<String> {
        self.registered.iter().map(|r| r.model_id.clone()).collect()
    }

    pub fn default_runtime(&self) -> Arc<dyn VisionRuntime> {
        match self.registered.first() {
            Some(entry) => {
                let id = entry.model_id.clone();
                match tokio::runtime::Handle::try_current() {
                    Ok(handle) => match handle.block_on(async { self.runtime_for(&id).await }) {
                        Ok(rt) => rt,
                        Err(e) => {
                            eprintln!("[neuron] vision: failed to load default runtime for {id}: {e}");
                            Arc::new(UnavailableVisionRuntime)
                        }
                    },
                    Err(_) => Arc::new(UnavailableVisionRuntime),
                }
            }
            None => Arc::new(UnavailableVisionRuntime),
        }
    }
}

/// Fallback runtime returned when no vision model is registered. Every call
/// returns `NeuronError::not_initialized`. Same pattern as the other capabilities'
/// `Unavailable*Runtime` stubs.
struct UnavailableVisionRuntime;

#[async_trait]
impl VisionRuntime for UnavailableVisionRuntime {
    async fn load(_: &str, _: &DeviceSelection) -> ChatResult<Box<dyn VisionRuntime>>
    where
        Self: Sized,
    {
        Err(NeuronError::new("NoRegisteredModel: register at least one Vision model"))
    }
    async fn run(&self, _req: VisionRequest) -> ChatResult<VisionResponse> {
        Err(NeuronError::new("NoRegisteredModel: register at least one Vision model"))
    }
    fn list_models(&self) -> Vec<ModelInfo> {
        vec![]
    }
    fn name(&self) -> &str {
        "unavailable"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::VisionTask;

    #[tokio::test]
    async fn test_runtime_for_unregistered_returns_error() {
        let config = Arc::new(NeuronConfig::new());
        let reg = VisionRuntimeRegistry::new(config);
        let result = reg.runtime_for("nonexistent").await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.message.contains("ModelNotRegistered"));
    }

    #[test]
    fn test_default_runtime_returns_unavailable_when_no_registered() {
        let config = Arc::new(NeuronConfig::new());
        let reg = VisionRuntimeRegistry::new(config);
        let rt = reg.default_runtime();
        assert_eq!(rt.name(), "unavailable");
    }

    #[tokio::test]
    async fn test_unavailable_vision_runtime_detect_returns_error() {
        let rt = UnavailableVisionRuntime;
        let req = VisionRequest {
            model: "yolov8n".into(),
            task: VisionTask::Detect,
            image: super::super::VisionImageSource::ImageUrl {
                image_url: super::super::VisionImageUrl { url: "x".into(), detail: None },
            },
            confidence_threshold: None,
            iou_threshold: None,
            max_detections: None,
            classes: None,
            lora_adapters: None,
        };
        let result = rt.run(req).await;
        assert!(result.is_err());
    }
}
