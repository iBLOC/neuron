use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures_util::stream::BoxStream;

use super::{LlmRuntime, LlmImpl, ModelInfo, NeuronError, RegisteredLlm, RegistryKey};
use crate::config::{DeviceSelection, NeuronConfig, ResourceTracker};
use crate::llm::{ChatRequest, ChatResponse, ChatChunk};

pub struct LlmRuntimeRegistry {
    pub registered: Vec<RegisteredLlm>,
    // std (not tokio) mutex: the load itself runs *outside* the lock
    // and re-checks on insert, so this guard never crosses an await.
    // That is what lets `unload` be synchronous — the scheduler
    // evicts models from lease-Drop contexts, where blocking on a
    // tokio mutex is not allowed.
    loaded: Arc<std::sync::Mutex<HashMap<RegistryKey, Arc<dyn LlmRuntime>>>>,
    resources: Mutex<ResourceTracker>,
}

impl Clone for LlmRuntimeRegistry {
    fn clone(&self) -> Self {
        Self {
            registered: self.registered.clone(),
            loaded: self.loaded.clone(),
            resources: Mutex::new(ResourceTracker::default()),
        }
    }
}

impl LlmRuntimeRegistry {
    pub fn new(_config: Arc<NeuronConfig>) -> Self {
        Self {
            registered: Vec::new(),
            loaded: Arc::new(std::sync::Mutex::new(HashMap::new())),
            resources: Mutex::new(ResourceTracker::default()),
        }
    }

    pub fn register(
        &mut self,
        model_id: String,
        impl_id: LlmImpl,
        device: DeviceSelection,
        required_memory_bytes: u64,
    ) -> Result<(), NeuronError> {
        // Check resources before registering
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

        self.registered.push(RegisteredLlm {
            model_id,
            impl_id,
            device,
            required_memory_bytes,
        });
        Ok(())
    }

    pub async fn runtime_for(&self, model_id: &str) -> Result<Arc<dyn LlmRuntime>, NeuronError> {
        let entry = self
            .registered
            .iter()
            .find(|r| r.model_id == model_id)
            .ok_or_else(|| {
                NeuronError::new(format!("ModelNotRegistered: {model_id}"))
            })?;

        let key = RegistryKey {
            model_id: entry.model_id.clone(),
            impl_id: entry.impl_id,
            device: entry.device,
        };

        {
            let loaded = self.loaded.lock().unwrap();
            if let Some(rt) = loaded.get(&key) {
                return Ok(rt.clone());
            }
        }

        // Load outside the cache lock: two concurrent first requests
        // for the same model may both load; the insert below keeps
        // whichever lands first and drops the loser.
        let rt: Arc<dyn LlmRuntime> = match entry.impl_id {
            #[cfg(feature = "candle")]
            LlmImpl::Candle => {
                let loaded = super::CandleLlmRuntime::load(&entry.model_id, &entry.device)
                    .await
                    .map_err(|e| {
                        NeuronError::new(format!("CandleLlmRuntime load failed: {e}"))
                    })?;
                Arc::from(loaded)
            }
            #[cfg(feature = "litert")]
            LlmImpl::Litert => {
                let loaded = super::LitertLlmRuntime::load(&entry.model_id, &entry.device)
                    .await
                    .map_err(|e| {
                        NeuronError::new(format!("LitertLlmRuntime load failed: {e}"))
                    })?;
                Arc::from(loaded)
            }
            #[cfg(feature = "mistralrs")]
            LlmImpl::MistralRs => {
                let loaded = super::MistralRsLlmRuntime::load(&entry.model_id, &entry.device)
                    .await
                    .map_err(|e| {
                        NeuronError::new(format!("MistralRsLlmRuntime load failed: {e}"))
                    })?;
                Arc::from(loaded)
            }
        };

        let mut loaded = self.loaded.lock().unwrap();
        // Another task may have inserted while we were loading —
        // keep the first-loaded runtime, drop ours.
        if let Some(existing) = loaded.get(&key) {
            return Ok(existing.clone());
        }
        loaded.insert(key, rt.clone());
        Ok(rt)
    }

    /// Drop the cached runtime(s) for `model_id` — the weights free
    /// when the last `Arc` reference (held only by in-flight
    /// requests) goes away. Returns how many cache entries were
    /// removed. Synchronous so the scheduler can call it from
    /// lease-drop contexts.
    pub fn unload(&self, model_id: &str) -> usize {
        let mut loaded = self.loaded.lock().unwrap();
        let removed: Vec<RegistryKey> = loaded
            .keys()
            .filter(|k| k.model_id == model_id)
            .cloned()
            .collect();
        // Drop the runtimes after releasing the cache lock —
        // deallocating GB of tensors is not something to do while
        // holding it.
        let dropped: Vec<Arc<dyn LlmRuntime>> =
            removed.iter().filter_map(|k| loaded.remove(k)).collect();
        drop(loaded);
        drop(dropped);
        removed.len()
    }

    pub fn is_registered(&self, model_id: &str) -> bool {
        self.registered.iter().any(|r| r.model_id == model_id)
    }

    pub fn is_loaded(&self, model_id: &str) -> bool {
        self.loaded
            .lock()
            .unwrap()
            .keys()
            .any(|k| k.model_id == model_id)
    }

    pub fn loaded_model_ids(&self) -> Vec<String> {
        let loaded = self.loaded.lock().unwrap();
        let mut ids: Vec<String> = loaded.keys().map(|k| k.model_id.clone()).collect();
        ids.sort();
        ids.dedup();
        ids
    }

    pub fn list_registered(&self) -> Vec<String> {
        self.registered.iter().map(|r| r.model_id.clone()).collect()
    }

    /// Return a placeholder for the first registered model, or an
    /// unavailable stub if nothing is registered.
    ///
    /// Why this returns a stub instead of loading the registered
    /// model eagerly: the previous implementation called
    /// `Handle::try_current().block_on(self.runtime_for(&id))` here,
    /// which panics with "Cannot start a runtime from within a
    /// runtime" when `ServerState::new` is invoked from inside a
    /// tokio runtime (any host that uses `#[tokio::main]` + the
    /// `run_embedded` async API, including all `examples/` smoke
    /// harnesses). The actual lazy load happens on `runtime_for`'s
    /// path, which is async and called by HTTP handlers — so the
    /// public contract is preserved: any *registered* model id
    /// still resolves to a working runtime via
    /// `runtime_for(model).await`, and only the (already-broken)
    /// "state.llm fallback for unregistered model ids" path
    /// receives a stub. That path was always expected to fail; the
    /// stub just makes the failure explicit instead of panicking on
    /// startup.
    pub fn default_runtime(&self) -> Arc<dyn LlmRuntime> {
        match self.registered.first() {
            Some(_) => Arc::new(UnavailableLlmRuntime),
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
        Err(NeuronError::new(UNV_MSG))
    }
    async fn chat(&self, _: ChatRequest) -> crate::llm::ChatResult<ChatResponse> {
        Err(NeuronError::new(UNV_MSG))
    }
    async fn chat_stream(&self, _: ChatRequest) -> crate::llm::ChatResult<BoxStream<'static, ChatChunk>> {
        Err(NeuronError::new(UNV_MSG))
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
        let config = Arc::new(NeuronConfig::new());
        let reg = LlmRuntimeRegistry::new(config);
        let result = reg.runtime_for("nonexistent").await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.message.contains("ModelNotRegistered"));
    }

    #[test]
    fn test_default_runtime_empty_returns_unavailable() {
        let config = Arc::new(NeuronConfig::new());
        let reg = LlmRuntimeRegistry::new(config);
        let rt = reg.default_runtime();
        assert_eq!(rt.name(), "unavailable");
    }
}