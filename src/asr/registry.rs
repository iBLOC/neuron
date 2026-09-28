use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use super::{AsrImpl, RegisteredAsr, AsrRegistryKey, Transcription};
use crate::config::{DeviceSelection, NeuronConfig, ResourceTracker};
use crate::asr::AsrRuntime;
use crate::llm::NeuronError;

#[cfg(feature = "candle")]
use crate::WhisperAsrRuntime;
#[cfg(feature = "tract")]
use crate::asr::TractSenseVoiceRuntime;
#[cfg(feature = "sherpa")]
use crate::asr::SherpaAsrRuntime;

pub struct AsrRuntimeRegistry {
    pub registered: Vec<RegisteredAsr>,
    loaded: Arc<tokio::sync::Mutex<HashMap<AsrRegistryKey, Arc<dyn AsrRuntime>>>>,
    resources: Mutex<ResourceTracker>,
}

impl Clone for AsrRuntimeRegistry {
    fn clone(&self) -> Self {
        Self {
            registered: self.registered.clone(),
            loaded: self.loaded.clone(),
            resources: Mutex::new(ResourceTracker::default()),
        }
    }
}

impl AsrRuntimeRegistry {
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
        impl_id: AsrImpl,
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

        self.registered.push(RegisteredAsr {
            model_id,
            impl_id,
            device,
            required_memory_bytes,
        });
        Ok(())
    }

    pub async fn runtime_for(&self, model_id: &str) -> Result<Arc<dyn AsrRuntime>, NeuronError> {
        let entry = self
            .registered
            .iter()
            .find(|r| r.model_id == model_id)
            .ok_or_else(|| {
                NeuronError::new(format!("ModelNotRegistered: {model_id}"))
            })?;

        let key = AsrRegistryKey {
            model_id: entry.model_id.clone(),
            impl_id: entry.impl_id,
            device: entry.device,
        };

        let mut loaded = self.loaded.lock().await;
        if let Some(rt) = loaded.get(&key) {
            return Ok(rt.clone());
        }

        let rt: Arc<dyn AsrRuntime> = match entry.impl_id {
            #[cfg(feature = "candle")]
            AsrImpl::Candle => {
                let loaded = WhisperAsrRuntime::load(&entry.model_id, &entry.device)
                    .await
                    .map_err(|e| {
                        NeuronError::new(format!("WhisperAsrRuntime load failed: {e}"))
                    })?;
                Arc::from(loaded)
            }
            #[cfg(feature = "tract")]
            AsrImpl::Tract => {
                let loaded = TractSenseVoiceRuntime::load(&entry.model_id, &entry.device)
                    .await
                    .map_err(|e| {
                        NeuronError::new(format!("TractSenseVoiceRuntime load failed: {e}"))
                    })?;
                Arc::from(loaded)
            }
            #[cfg(feature = "sherpa")]
            AsrImpl::Sherpa => {
                let loaded = SherpaAsrRuntime::load(&entry.model_id, &entry.device)
                    .await
                    .map_err(|e| {
                        NeuronError::new(format!("SherpaAsrRuntime load failed: {e}"))
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

    pub fn default_runtime(&self) -> Arc<dyn AsrRuntime> {
        match self.registered.first() {
            Some(entry) => {
                let id = entry.model_id.clone();
                match tokio::runtime::Handle::try_current() {
                    Ok(handle) => match handle.block_on(async { self.runtime_for(&id).await }) {
                        Ok(rt) => rt,
                        Err(e) => {
                            eprintln!("[neuron] asr: failed to load default runtime for {id}: {e}");
                            Arc::new(UnavailableAsrRuntime)
                        }
                    },
                    Err(_) => Arc::new(UnavailableAsrRuntime),
                }
            }
            None => Arc::new(UnavailableAsrRuntime),
        }
    }
}

struct UnavailableAsrRuntime;

#[async_trait]
impl AsrRuntime for UnavailableAsrRuntime {
    async fn load(_: &str, _: &DeviceSelection) -> crate::llm::ChatResult<Box<dyn AsrRuntime>>
    where
        Self: Sized,
    {
        Err(NeuronError::new("NoRegisteredModel: register at least one ASR model"))
    }
    async fn transcribe(&self, _audio: &[u8], _lang: Option<&str>) -> crate::llm::ChatResult<Transcription> {
        Err(NeuronError::new("NoRegisteredModel: register at least one ASR model"))
    }
    fn name(&self) -> &str {
        "unavailable"
    }
}

