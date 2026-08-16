use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use super::{NeureError, RegisteredRerank, RerankImpl, RerankRegistryKey, RerankRequest, RerankResponse};
use crate::config::{DeviceSelection, NeureConfig, ResourceTracker};
use crate::llm::ModelInfo;
use crate::rerank::RerankRuntime;

pub struct RerankRuntimeRegistry {
    pub registered: Vec<RegisteredRerank>,
    loaded: Arc<tokio::sync::Mutex<HashMap<RerankRegistryKey, Arc<dyn RerankRuntime>>>>,
    resources: Mutex<ResourceTracker>,
}

impl RerankRuntimeRegistry {
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
        impl_id: RerankImpl,
        device: DeviceSelection,
        required_memory_bytes: u64,
    ) -> Result<(), NeureError> {
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

        self.registered.push(RegisteredRerank {
            model_id,
            impl_id,
            device,
            required_memory_bytes,
        });
        Ok(())
    }

    pub async fn runtime_for(&self, model_id: &str) -> Result<Arc<dyn RerankRuntime>, NeureError> {
        let entry = self
            .registered
            .iter()
            .find(|r| r.model_id == model_id)
            .ok_or_else(|| {
                NeureError::new(format!("ModelNotRegistered: {model_id}"))
            })?;

        let key = RerankRegistryKey {
            model_id: entry.model_id.clone(),
            impl_id: entry.impl_id,
            device: entry.device,
        };

        let mut loaded = self.loaded.lock().await;
        if let Some(rt) = loaded.get(&key) {
            return Ok(rt.clone());
        }

        let rt: Arc<dyn RerankRuntime> = match entry.impl_id {
            #[cfg(feature = "candle")]
            RerankImpl::Bge | RerankImpl::Candle => {
                let loaded = super::CandleRerankRuntime::load(&entry.model_id, &entry.device)
                    .await
                    .map_err(|e| {
                        NeureError::new(format!("BgeRerankRuntime load failed: {e}"))
                    })?;
                Arc::from(loaded)
            }
            #[cfg(feature = "candle")]
            RerankImpl::Mxbai => {
                let loaded: Box<dyn RerankRuntime> = super::MxbaiRerankRuntime::load(&entry.model_id, &entry.device)
                    .await
                    .map_err(|e| {
                        NeureError::new(format!("MxbaiRerankRuntime load failed: {e}"))
                    })?;
                Arc::from(loaded)
            }
            #[cfg(feature = "candle")]
            RerankImpl::Jina => {
                let loaded = super::JinaRerankRuntime::load(&entry.model_id, &entry.device)
                    .await
                    .map_err(|e| {
                        NeureError::new(format!("JinaRerankRuntime load failed: {e}"))
                    })?;
                Arc::from(loaded)
            }
            RerankImpl::Cohere => {
                let loaded = super::CohereRerankRuntime::load(&entry.model_id, &entry.device)
                    .await
                    .map_err(|e| {
                        NeureError::new(format!("CohereRerankRuntime load failed: {e}"))
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

    pub fn default_runtime(&self) -> Arc<dyn RerankRuntime> {
        match self.registered.first() {
            Some(entry) => {
                let id = entry.model_id.clone();
                match tokio::runtime::Handle::try_current() {
                    Ok(handle) => match handle.block_on(async { self.runtime_for(&id).await }) {
                        Ok(rt) => rt,
                        Err(e) => {
                            eprintln!("[neure] rerank: failed to load default runtime for {id}: {e}");
                            Arc::new(UnavailableRerankRuntime)
                        }
                    },
                    Err(_) => Arc::new(UnavailableRerankRuntime),
                }
            }
            None => Arc::new(UnavailableRerankRuntime),
        }
    }
}

struct UnavailableRerankRuntime;

#[async_trait]
impl RerankRuntime for UnavailableRerankRuntime {
    async fn load(_: &str, _: &DeviceSelection) -> crate::llm::ChatResult<Box<dyn RerankRuntime>>
    where
        Self: Sized,
    {
        Err(NeureError::new("NoRegisteredModel: register at least one Rerank model"))
    }
    async fn rerank(&self, _req: RerankRequest) -> crate::llm::ChatResult<RerankResponse> {
        Err(NeureError::new("NoRegisteredModel: register at least one Rerank model"))
    }
    fn list_models(&self) -> Vec<ModelInfo> {
        vec![]
    }
    fn name(&self) -> &str {
        "unavailable"
    }
}

