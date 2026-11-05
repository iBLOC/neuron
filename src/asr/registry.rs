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
    pub registered: Arc<Mutex<Vec<RegisteredAsr>>>,
    loaded: Arc<tokio::sync::Mutex<HashMap<AsrRegistryKey, Arc<dyn AsrRuntime>>>>,
    resources: Arc<Mutex<ResourceTracker>>,
}

impl Clone for AsrRuntimeRegistry {
    fn clone(&self) -> Self {
        Self {
            registered: self.registered.clone(),
            loaded: self.loaded.clone(),
            resources: self.resources.clone(),
        }
    }
}

impl AsrRuntimeRegistry {
    pub fn new(_config: Arc<NeuronConfig>) -> Self {
        Self {
            registered: Arc::new(Mutex::new(Vec::new())),
            loaded: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            resources: Arc::new(Mutex::new(ResourceTracker::default())),
        }
    }

    pub fn register(
        &self,
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

        self.registered.lock().unwrap().push(RegisteredAsr {
            model_id,
            impl_id,
            device,
            required_memory_bytes,
        });
        Ok(())
    }

    pub async fn runtime_for(&self, model_id: &str) -> Result<Arc<dyn AsrRuntime>, NeuronError> {
        let entry = {
            let registered = self.registered.lock().unwrap();
            registered
                .iter()
                .find(|r| r.model_id == model_id)
                .cloned()
                .ok_or_else(|| {
                NeuronError::new(format!("ModelNotRegistered: {model_id}"))
            })?
        };

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
        self.registered
            .lock()
            .unwrap()
            .iter()
            .map(|r| r.model_id.clone())
            .collect()
    }

    /// A shortcut handle that resolves the first registered ASR model on
    /// **first use** rather than at construction time.
    ///
    /// The eager variant cannot work: `ServerState::new` runs inside the
    /// host's tokio runtime, where blocking on the load panics with
    /// "Cannot start a runtime from within a runtime" (measured 2026-11-05:
    /// the panic came from `handle.block_on` inside `default_runtime`
    /// below -- so an embedded neuron that registered an ASR engine simply
    /// **died at boot**). The proxy defers the load to the first async call,
    /// so the weights are still materialized on demand and the shortcut
    /// still resolves to a working runtime -- the same shape as the TTS
    /// registry's lazy shortcut.
    pub fn lazy_default_runtime(self: &Arc<Self>) -> Arc<dyn AsrRuntime> {
        let first = self.registered.lock().unwrap().first().cloned();
        match first {
            Some(entry) => {
                let model_id = entry.model_id.clone();
                Arc::new(LazyAsrRuntime {
                    registry: self.clone(),
                    model_id,
                    resolved: Mutex::new(None),
                })
            }
            None => Arc::new(UnavailableAsrRuntime),
        }
    }

    pub fn default_runtime(&self) -> Arc<dyn AsrRuntime> {
        let first = self.registered.lock().unwrap().first().cloned();
        match first {
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

/// An ASR runtime handle that resolves the registry's default model on
/// the first call and delegates everything to it afterwards.
///
/// The resolved runtime is cached in the registry itself, so the second
/// call is a hash lookup; the local cache only serves the `name` accessor.
struct LazyAsrRuntime {
    registry: Arc<AsrRuntimeRegistry>,
    model_id: String,
    resolved: Mutex<Option<Arc<dyn AsrRuntime>>>,
}

impl LazyAsrRuntime {
    async fn resolve(&self) -> Result<Arc<dyn AsrRuntime>, NeuronError> {
        let rt = self.registry.runtime_for(&self.model_id).await?;
        *self.resolved.lock().unwrap() = Some(rt.clone());
        Ok(rt)
    }
}

#[async_trait]
impl AsrRuntime for LazyAsrRuntime {
    async fn load(
        _model: &str,
        _device: &DeviceSelection,
    ) -> crate::llm::ChatResult<Box<dyn AsrRuntime>>
    where
        Self: Sized,
    {
        Err(NeuronError::new(
            "the default ASR shortcut is not loadable -- request a registered model id instead",
        ))
    }

    async fn transcribe(
        &self,
        audio: &[u8],
        lang: Option<&str>,
    ) -> crate::llm::ChatResult<Transcription> {
        self.resolve().await?.transcribe(audio, lang).await
    }

    fn name(&self) -> &str {
        "lazy-asr-default"
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

#[cfg(all(test, feature = "candle"))]
mod tests {
    use super::*;

    fn registry_with_one_engine() -> Arc<AsrRuntimeRegistry> {
        let config = Arc::new(NeuronConfig::new());
        let registry = Arc::new(AsrRuntimeRegistry::new(config));
        registry
            .register(
                "whisper-model".to_string(),
                AsrImpl::Candle,
                DeviceSelection::Cpu,
                1_000_000,
            )
            .expect("register an ASR engine");
        registry
    }

    /// The lazy shortcut must be constructible **inside** a running runtime --
    /// that is exactly where `ServerState::new` builds it.
    #[tokio::test]
    async fn test_lazy_default_runtime_construction_inside_runtime_does_not_panic() {
        let registry = registry_with_one_engine();
        let shortcut = registry.lazy_default_runtime();
        assert_eq!(shortcut.name(), "lazy-asr-default");
    }

    /// **反向验证**：the eager path must still blow up inside a runtime --
    /// otherwise "the lazy path works" would prove nothing (the two
    /// symptoms have to be **different**, otherwise a red reading is
    /// uninterpretable).
    #[tokio::test]
    async fn test_eager_default_runtime_inside_runtime_panics() {
        let registry = registry_with_one_engine();
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = registry.default_runtime();
        }))
        .is_err();
        assert!(
            panicked,
            "eager default_runtime 必须在 runtime 内 panic（两个症状必须不同）"
        );
    }
}

