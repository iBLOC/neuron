//! Registry for image-generation runtimes. Mirrors the other
//! capability registries: registered entries + a loaded-instance
//! cache keyed by `(model_id, impl_id, device)` + a resource tracker,
//! with lazy loading on first request.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use super::{GeneratedImage, ImageGenImpl, ImageGenParams, ImageGenRegistryKey, RegisteredImageGen};
use crate::config::{DeviceSelection, NeuronConfig, ResourceTracker};
use crate::llm::{ModelInfo, NeuronError};
use crate::imagegen::ImageGenRuntime;

pub struct ImageGenRuntimeRegistry {
    /// The registered models.
    ///
    /// Shared behind an `Arc<Mutex<_>>` so that a **clone** of the
    /// registry sees the same list: `ServerState` keeps one copy for
    /// the handlers and one inside the registration lock, and a model
    /// registered at runtime must reach both. (The other capability
    /// registries still copy their list on clone — see
    /// `src/imagegen/registry_sharing_test.rs`.)
    pub registered: Arc<Mutex<Vec<RegisteredImageGen>>>,
    loaded: Arc<tokio::sync::Mutex<HashMap<ImageGenRegistryKey, Arc<dyn ImageGenRuntime>>>>,
    resources: Mutex<ResourceTracker>,
}

impl Clone for ImageGenRuntimeRegistry {
    fn clone(&self) -> Self {
        Self {
            registered: self.registered.clone(),
            loaded: self.loaded.clone(),
            resources: Mutex::new(ResourceTracker::default()),
        }
    }
}

/// The device's total memory in bytes, if it can be determined.
///
/// The shared `ResourceTracker` ships a hard-coded 8 GB GPU / 16 GB
/// CPU budget — a placeholder from the 8 GB edge tier this crate
/// started on. A 24 GB card therefore refuses to register a 17 GB
/// model, and the refusal surfaces as a confusing
/// "ModelNotRegistered". `nvidia-smi` is queried once at startup (the
/// model puller already shells out to `huggingface-cli`, so the
/// pattern is not new); a failure is not fatal — the default budget
/// is used and the registration error says what to do.
fn device_total_bytes(device: &DeviceSelection) -> Option<u64> {
    match device {
        DeviceSelection::Nvidia => {
            let output = std::process::Command::new("nvidia-smi")
                .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
                .output()
                .ok()?;
            let text = String::from_utf8(output.stdout).ok()?;
            let mib: u64 = text.lines().next()?.trim().parse().ok()?;
            // Leave a slice for the CUDA context and the activations.
            Some(mib * 1024 * 1024 / 10 * 9)
        }
        _ => None,
    }
}

impl ImageGenRuntimeRegistry {
    pub fn new(config: Arc<NeuronConfig>) -> Self {
        let resources = match device_total_bytes(&config.device) {
            Some(gpu) => ResourceTracker::new(16_000_000_000, gpu),
            None => ResourceTracker::default(),
        };
        Self {
            registered: Arc::new(Mutex::new(Vec::new())),
            loaded: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            resources: Mutex::new(resources),
        }
    }

    pub fn register(
        &self,
        model_id: String,
        impl_id: ImageGenImpl,
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

        self.registered.lock().unwrap().push(RegisteredImageGen {
            model_id,
            impl_id,
            device,
            required_memory_bytes,
        });
        Ok(())
    }

    pub async fn runtime_for(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn ImageGenRuntime>, NeuronError> {
        let entry = {
            let registered = self.registered.lock().unwrap();
            registered
                .iter()
                .find(|r| r.model_id == model_id)
                .cloned()
                .ok_or_else(|| NeuronError::model_not_registered(model_id))?
        };

        let key = ImageGenRegistryKey {
            model_id: entry.model_id.clone(),
            impl_id: entry.impl_id,
            device: entry.device,
        };

        let mut loaded = self.loaded.lock().await;
        if let Some(rt) = loaded.get(&key) {
            return Ok(rt.clone());
        }

        match entry.impl_id {
            #[cfg(feature = "candle")]
            ImageGenImpl::MossImage => {
                let result = super::MossImageRuntime::load(&entry.model_id, &entry.device).await;
                let rt: Arc<dyn ImageGenRuntime> = Arc::from(result.map_err(|e| {
                    NeuronError::new(format!("moss-image load failed: {e}"))
                })?);
                loaded.insert(key, rt.clone());
                return Ok(rt);
            }
            #[cfg(test)]
            ImageGenImpl::Mock => {
                let rt: Arc<dyn ImageGenRuntime> = Arc::new(super::mock::MockImageGenRuntime::new());
                loaded.insert(key, rt.clone());
                return Ok(rt);
            }
            #[allow(unreachable_patterns)]
            _ => Err(NeuronError::new(
                "image generation requires building neuron with the candle feature".to_string(),
            )),
        }
    }

    /// A shortcut handle that resolves the first registered model on
    /// first use. See [`crate::tts::registry`]'s lazy default for why
    /// the eager variant cannot work inside `ServerState::new`.
    pub fn lazy_default_runtime(self: &Arc<Self>) -> Arc<dyn ImageGenRuntime> {
        let first = self.registered.lock().unwrap().first().cloned();
        match first {
            Some(entry) => {
                let model_id = entry.model_id.clone();
                Arc::new(LazyImageGenRuntime {
                    registry: self.clone(),
                    model_id,
                    resolved: Mutex::new(None),
                })
            }
            None => Arc::new(UnavailableImageGenRuntime),
        }
    }

    pub fn list_registered(&self) -> Vec<String> {
        self.registered
            .lock()
            .unwrap()
            .iter()
            .map(|r| r.model_id.clone())
            .collect()
    }
}

/// A proxy that resolves the registry's default model on the first
/// call and delegates everything afterwards.
struct LazyImageGenRuntime {
    registry: Arc<ImageGenRuntimeRegistry>,
    model_id: String,
    resolved: Mutex<Option<Arc<dyn ImageGenRuntime>>>,
}

impl LazyImageGenRuntime {
    async fn resolve(&self) -> Result<Arc<dyn ImageGenRuntime>, NeuronError> {
        let rt = self.registry.runtime_for(&self.model_id).await?;
        *self.resolved.lock().unwrap() = Some(rt.clone());
        Ok(rt)
    }
}

#[async_trait]
impl ImageGenRuntime for LazyImageGenRuntime {
    async fn load(_model: &str, _device: &DeviceSelection) -> Result<Box<dyn ImageGenRuntime>, NeuronError>
    where
        Self: Sized,
    {
        Err(NeuronError::new(
            "the default imagegen shortcut is not loadable — request a registered model id instead",
        ))
    }

    async fn generate(&self, params: &ImageGenParams) -> Result<GeneratedImage, NeuronError> {
        self.resolve().await?.generate(params).await
    }

    fn list_models(&self) -> Vec<ModelInfo> {
        match self.resolved.lock().unwrap().as_ref() {
            Some(rt) => rt.list_models(),
            None => Vec::new(),
        }
    }

    fn name(&self) -> &str {
        "lazy-imagegen-default"
    }
}

struct UnavailableImageGenRuntime;

#[async_trait]
impl ImageGenRuntime for UnavailableImageGenRuntime {
    async fn load(_: &str, _: &DeviceSelection) -> Result<Box<dyn ImageGenRuntime>, NeuronError>
    where
        Self: Sized,
    {
        Err(NeuronError::new(
            "NoRegisteredModel: register at least one imagegen model (NEURON_IMAGEGEN_RUNTIME)",
        ))
    }

    async fn generate(&self, _params: &ImageGenParams) -> Result<GeneratedImage, NeuronError> {
        Err(NeuronError::new(
            "NoRegisteredModel: register at least one imagegen model (NEURON_IMAGEGEN_RUNTIME)",
        ))
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

    #[tokio::test]
    async fn test_lazy_default_without_registration_is_unavailable() {
        let registry = Arc::new(ImageGenRuntimeRegistry::new(Arc::new(NeuronConfig::default())));
        let shortcut = registry.lazy_default_runtime();
        assert_eq!(shortcut.name(), "unavailable");
        let err = shortcut
            .generate(&ImageGenParams::default())
            .await
            .unwrap_err();
        assert!(err.message.contains("NoRegisteredModel"), "got: {}", err.message);
    }

    #[tokio::test]
    async fn test_lazy_default_construction_inside_runtime_does_not_panic() {
        let registry = Arc::new(ImageGenRuntimeRegistry::new(Arc::new(NeuronConfig::default())));
        let _shortcut = registry.lazy_default_runtime();
    }
}
