use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use super::{EmbeddingImpl, EmbeddingRequest, EmbeddingResponse, NeuronError, RegisteredEmbedding, EmbeddingRegistryKey};
use crate::config::{DeviceSelection, NeuronConfig, ResourceTracker};
use crate::embedding::EmbeddingRuntime;
use crate::llm::ModelInfo;

#[cfg(feature = "candle")]
use crate::embedding::candle::MiniLmL6V2EmbeddingRuntime;

pub struct EmbeddingRuntimeRegistry {
    pub registered: Vec<RegisteredEmbedding>,
    loaded: Arc<tokio::sync::Mutex<HashMap<EmbeddingRegistryKey, Arc<dyn EmbeddingRuntime>>>>,
    resources: Mutex<ResourceTracker>,
}

impl Clone for EmbeddingRuntimeRegistry {
    fn clone(&self) -> Self {
        Self {
            registered: self.registered.clone(),
            loaded: self.loaded.clone(),
            resources: Mutex::new(ResourceTracker::default()),
        }
    }
}

impl EmbeddingRuntimeRegistry {
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
        impl_id: EmbeddingImpl,
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

        self.registered.push(RegisteredEmbedding {
            model_id,
            impl_id,
            device,
            required_memory_bytes,
        });
        Ok(())
    }

    pub async fn runtime_for(&self, model_id: &str) -> Result<Arc<dyn EmbeddingRuntime>, NeuronError> {
        let entry = self
            .registered
            .iter()
            .find(|r| r.model_id == model_id)
            .ok_or_else(|| {
                NeuronError::new(format!("ModelNotRegistered: {model_id}"))
            })?;

        let key = EmbeddingRegistryKey {
            model_id: entry.model_id.clone(),
            impl_id: entry.impl_id,
            device: entry.device,
        };

        let mut loaded = self.loaded.lock().await;
        if let Some(rt) = loaded.get(&key) {
            return Ok(rt.clone());
        }

        let rt: Arc<dyn EmbeddingRuntime> = match entry.impl_id {
            #[cfg(feature = "candle")]
            EmbeddingImpl::Candle => {
                let loaded: Box<dyn EmbeddingRuntime> = MiniLmL6V2EmbeddingRuntime::load(&entry.model_id, &entry.device)
                    .await
                    .map_err(|e| {
                        NeuronError::new(format!("MiniLmL6V2EmbeddingRuntime load failed: {e}"))
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

    pub fn default_runtime(&self) -> Arc<dyn EmbeddingRuntime> {
        match self.registered.first() {
            Some(entry) => {
                let id = entry.model_id.clone();
                match tokio::runtime::Handle::try_current() {
                    Ok(handle) => match handle.block_on(async { self.runtime_for(&id).await }) {
                        Ok(rt) => rt,
                        Err(e) => {
                            eprintln!("[neuron] embedding: failed to load default runtime for {id}: {e}");
                            Arc::new(UnavailableEmbeddingRuntime)
                        }
                    },
                    Err(_) => Arc::new(UnavailableEmbeddingRuntime),
                }
            }
            None => Arc::new(UnavailableEmbeddingRuntime),
        }
    }
}

struct UnavailableEmbeddingRuntime;

#[async_trait]
impl EmbeddingRuntime for UnavailableEmbeddingRuntime {
    async fn load(_: &str, _: &DeviceSelection) -> crate::llm::ChatResult<Box<dyn EmbeddingRuntime>>
    where
        Self: Sized,
    {
        Err(NeuronError::new("NoRegisteredModel: register at least one Embedding model"))
    }
    async fn embed(&self, _req: EmbeddingRequest) -> crate::llm::ChatResult<EmbeddingResponse> {
        Err(NeuronError::new("NoRegisteredModel: register at least one Embedding model"))
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
    async fn test_runtime_for_unregistered_returns_error() {
        let config = Arc::new(NeuronConfig::new());
        let reg = EmbeddingRuntimeRegistry::new(config);
        let result = reg.runtime_for("nonexistent").await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.message.contains("ModelNotRegistered"));
    }
}
