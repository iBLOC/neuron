use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures_util::stream::BoxStream;

use super::{LlmRuntime, LlmImpl, ModelInfo, NeureError, RegisteredLlm, RegistryKey};
use crate::config::{DeviceSelection, NeureConfig, ResourceTracker};
use crate::llm::{ChatRequest, ChatResponse, ChatChunk};

pub struct LlmRuntimeRegistry {
    pub registered: Vec<RegisteredLlm>,
    loaded: Arc<tokio::sync::Mutex<HashMap<RegistryKey, Arc<dyn LlmRuntime>>>>,
    resources: Mutex<ResourceTracker>,
}

impl LlmRuntimeRegistry {
    pub fn new(_config: Arc<NeureConfig>) -> Self {
        Self {
            registered: Vec::new(),
            loaded: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            resources: Mutex::new(ResourceTracker::default()),
        }
    }

    pub fn register(
        &mut self,
        model_id: String,
        impl_id: LlmImpl,
        device: DeviceSelection,
        required_memory_bytes: u64,
    ) -> Result<(), NeureError> {
        // Check resources before registering
        let mut res = self.resources.lock().unwrap();
        if !res.can_load(required_memory_bytes, &device) {
            let available = match &device {
                DeviceSelection::Cpu | DeviceSelection::Auto => res.available_cpu_bytes(),
                _ => res.available_gpu_bytes(),
            };
            return Err(NeureError::new(format!(
                "InsufficientResources: model={model_id}, required={required_memory_bytes}, available={available}"
            )));
        }
        res.allocate(required_memory_bytes, &device)
            .map_err(NeureError::new)?;

        self.registered.push(RegisteredLlm {
            model_id,
            impl_id,
            device,
            required_memory_bytes,
        });
        Ok(())
    }

    pub async fn runtime_for(&self, model_id: &str) -> Result<Arc<dyn LlmRuntime>, NeureError> {
        let entry = self
            .registered
            .iter()
            .find(|r| r.model_id == model_id)
            .ok_or_else(|| {
                NeureError::new(format!("ModelNotRegistered: {model_id}"))
            })?;

        let key = RegistryKey {
            model_id: entry.model_id.clone(),
            impl_id: entry.impl_id,
            device: entry.device,
        };

        let mut loaded = self.loaded.lock().await;
        if let Some(rt) = loaded.get(&key) {
            return Ok(rt.clone());
        }

        let rt: Arc<dyn LlmRuntime> = match entry.impl_id {
            #[cfg(feature = "candle")]
            LlmImpl::Candle => {
                let loaded = super::CandleLlmRuntime::load(&entry.model_id, &entry.device)
                    .await
                    .map_err(|e| {
                        NeureError::new(format!("CandleLlmRuntime load failed: {e}"))
                    })?;
                Arc::from(loaded)
            }
            #[cfg(feature = "litert")]
            LlmImpl::Litert => {
                let loaded = super::LitertLlmRuntime::load(&entry.model_id, &entry.device)
                    .await
                    .map_err(|e| {
                        NeureError::new(format!("LitertLlmRuntime load failed: {e}"))
                    })?;
                Arc::from(loaded)
            }
            #[cfg(feature = "mistralrs")]
            LlmImpl::MistralRs => {
                let loaded = super::MistralRsLlmRuntime::load(&entry.model_id, &entry.device)
                    .await
                    .map_err(|e| {
                        NeureError::new(format!("MistralRsLlmRuntime load failed: {e}"))
                    })?;
                Arc::from(loaded)
            }
        };

        loaded.insert(key, rt.clone());
        Ok(rt)
    }

    pub fn list_registered(&self) -> Vec<String> {
        self.registered.iter().map(|r| r.model_id.clone()).collect()
    }

    /// Return the first registered model, or an unavailable stub if
    /// nothing is registered. The stub returns errors for every
    /// method — callers must have at least one model registered to
    /// get a functional runtime through this path.
    pub fn default_runtime(&self) -> Arc<dyn LlmRuntime> {
        match self.registered.first() {
            Some(entry) => {
                let id = entry.model_id.clone();
                match tokio::runtime::Handle::try_current() {
                    Ok(handle) => match handle.block_on(async { self.runtime_for(&id).await }) {
                        Ok(rt) => rt,
                        Err(e) => {
                            eprintln!("[neure] llm: failed to load default runtime for {id}: {e}");
                            Arc::new(UnavailableLlmRuntime)
                        }
                    },
                    Err(_) => Arc::new(UnavailableLlmRuntime),
                }
            }
            None => Arc::new(UnavailableLlmRuntime),
        }
    }
}

struct UnavailableLlmRuntime;

#[async_trait]
impl LlmRuntime for UnavailableLlmRuntime {
    async fn load(_: &str, _: &DeviceSelection) -> crate::llm::ChatResult<Box<dyn LlmRuntime>>
    where
        Self: Sized,
    {
        Err(NeureError::new(UNV_MSG))
    }
    async fn chat(&self, _: ChatRequest) -> crate::llm::ChatResult<ChatResponse> {
        Err(NeureError::new(UNV_MSG))
    }
    async fn chat_stream(&self, _: ChatRequest) -> crate::llm::ChatResult<BoxStream<'static, ChatChunk>> {
        Err(NeureError::new(UNV_MSG))
    }
    fn list_models(&self) -> Vec<ModelInfo> {
        vec![]
    }
    fn name(&self) -> &str {
        "unavailable"
    }
}

const UNV_MSG: &str = "NoRegisteredModel: register at least one LLM model";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DeviceSelection;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_runtime_for_unregistered_returns_error() {
        let config = Arc::new(NeureConfig::new());
        let reg = LlmRuntimeRegistry::new(config);
        let result = reg.runtime_for("nonexistent").await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.message.contains("ModelNotRegistered"));
    }

    #[test]
    fn test_default_runtime_empty_returns_unavailable() {
        let config = Arc::new(NeureConfig::new());
        let reg = LlmRuntimeRegistry::new(config);
        let rt = reg.default_runtime();
        assert_eq!(rt.name(), "unavailable");
    }
}