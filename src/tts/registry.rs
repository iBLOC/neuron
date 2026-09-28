use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use super::{TtsImpl, RegisteredTts, TtsRegistryKey, TtsAudio, VoiceInfo};
use crate::llm::NeuronError;
use crate::config::{DeviceSelection, NeuronConfig, ResourceTracker};
use crate::tts::TtsRuntime;

pub struct TtsRuntimeRegistry {
    pub registered: Vec<RegisteredTts>,
    loaded: Arc<tokio::sync::Mutex<HashMap<TtsRegistryKey, Arc<dyn TtsRuntime>>>>,
    resources: Mutex<ResourceTracker>,
}

impl Clone for TtsRuntimeRegistry {
    fn clone(&self) -> Self {
        Self {
            registered: self.registered.clone(),
            loaded: self.loaded.clone(),
            resources: Mutex::new(ResourceTracker::default()),
        }
    }
}

impl TtsRuntimeRegistry {
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
        impl_id: TtsImpl,
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

        self.registered.push(RegisteredTts {
            model_id,
            impl_id,
            device,
            required_memory_bytes,
        });
        Ok(())
    }

    pub async fn runtime_for(&self, model_id: &str) -> Result<Arc<dyn TtsRuntime>, NeuronError> {
        let entry = self
            .registered
            .iter()
            .find(|r| r.model_id == model_id)
            .ok_or_else(|| NeuronError::model_not_registered(model_id))?;

        let key = TtsRegistryKey {
            model_id: entry.model_id.clone(),
            impl_id: entry.impl_id,
            device: entry.device,
        };

        let mut loaded = self.loaded.lock().await;
        if let Some(rt) = loaded.get(&key) {
            return Ok(rt.clone());
        }

        // Each match arm handles its own return path so the wildcard
        // doesn't make the post-match code unreachable.
        match entry.impl_id {
            #[cfg(feature = "voxcpm")]
            TtsImpl::Burn => {
                // Both burn TTS runtimes share the `burn` engine impl;
                // the catalog id picks the model generation.
                let result = if entry.model_id == "voxcpm2" {
                    super::VoxCpm2TtsRuntime::load(&entry.model_id, &entry.device).await
                } else {
                    super::VoxCpmTtsRuntime::load(&entry.model_id, &entry.device).await
                };
                // `load` returns a boxed trait object; the cache stores
                // `Arc<dyn TtsRuntime>`. Coerce via `Arc::from` (the blanket
                // `impl<T: ?Sized> From<Box<T>> for Arc<T>`).
                let rt: Arc<dyn TtsRuntime> =
                    Arc::from(result.map_err(|e| {
                        NeuronError::new(format!("burn TTS load failed: {e}"))
                    })?);
                loaded.insert(key, rt.clone());
                return Ok(rt);
            }
            #[allow(unreachable_patterns)]
            _ => {
                Err(NeuronError::new("TTS implementation 'burn' requires building neuron with --features voxcpm".to_string()))
            }
        }
    }

    pub fn list_registered(&self) -> Vec<String> {
        self.registered.iter().map(|r| r.model_id.clone()).collect()
    }

    pub fn default_runtime(&self) -> Arc<dyn TtsRuntime> {
        match self.registered.first() {
            Some(entry) => {
                let id = entry.model_id.clone();
                match tokio::runtime::Handle::try_current() {
                    Ok(handle) => match handle.block_on(async { self.runtime_for(&id).await }) {
                        Ok(rt) => rt,
                        Err(e) => {
                            eprintln!("[neuron] tts: failed to load default runtime for {id}: {e}");
                            Arc::new(UnavailableTtsRuntime)
                        }
                    },
                    Err(_) => Arc::new(UnavailableTtsRuntime),
                }
            }
            None => Arc::new(UnavailableTtsRuntime),
        }
    }
}

struct UnavailableTtsRuntime;

#[async_trait]
impl TtsRuntime for UnavailableTtsRuntime {
    async fn load(_: &str, _: &DeviceSelection) -> crate::llm::ChatResult<Box<dyn TtsRuntime>>
    where
        Self: Sized,
    {
        Err(NeuronError::new("NoRegisteredModel: register at least one TTS model"))
    }
    async fn synthesize(&self, _text: &str, _voice: Option<&str>) -> crate::llm::ChatResult<TtsAudio> {
        Err(NeuronError::new("NoRegisteredModel: register at least one TTS model"))
    }
    fn list_voices(&self) -> Vec<VoiceInfo> {
        vec![]
    }
    fn name(&self) -> &str {
        "unavailable"
    }
}

