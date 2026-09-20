use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tokio::sync::Mutex as AsyncMutex;

use super::{Chronos2Runtime, RegisteredChronos2, StubChronos2Runtime};
use crate::config::{DeviceSelection, ResourceTracker};
use crate::llm::NeureError;

pub struct Chronos2Registry {
    pub registered: Vec<RegisteredChronos2>,
    loaded: Arc<AsyncMutex<HashMap<String, Arc<dyn Chronos2Runtime>>>>,
    resources: Mutex<ResourceTracker>,
}

impl Clone for Chronos2Registry {
    fn clone(&self) -> Self {
        Self {
            registered: self.registered.clone(),
            loaded: self.loaded.clone(),
            resources: Mutex::new(ResourceTracker::default()),
        }
    }
}

impl Chronos2Registry {
    pub fn new() -> Self {
        Self {
            registered: Vec::new(),
            loaded: Arc::new(AsyncMutex::new(HashMap::new())),
            resources: Mutex::new(ResourceTracker::default()),
        }
    }

    /// Register a chronos2 model. Records it as a stub by default
    /// so the registry is non-empty even before the candle port
    /// lands; the real backend registers through
    /// `register_candle_runtime` (TODO: next commit).
    pub fn register(
        &mut self,
        model_id: String,
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
        self.registered.push(RegisteredChronos2 {
            model_id,
            device,
            required_memory_bytes,
        });
        Ok(())
    }

    /// Resolve a runtime for the given model id. v0 path:
    /// when `NEURE_CHRONOS2_MODEL_PATH/<id>` exists on disk,
    /// the candle runtime loads it. Otherwise we fall back to
    /// the stub so the 503 path stays observable for unit
    /// tests / dev environments that don't ship a checkpoint.
    pub async fn runtime_for(&self, model: &str) -> Arc<dyn Chronos2Runtime> {
        let mut g = self.loaded.lock().await;
        if let Some(rt) = g.get(model) {
            return rt.clone();
        }
        // The candle_runtime path is only available with `--features chronos2`.
        // Default builds (without that feature) always fall back to the stub,
        // which is the documented behavior for the stub-only `Sprint 3 first commit`.
        #[cfg(feature = "chronos2")]
        let real_path = crate::chronos2::env_discovery()
            .filter(|(id, _)| id == model)
            .map(|(_, path)| path);
        #[cfg(not(feature = "chronos2"))]
        let real_path: Option<std::path::PathBuf> = None;

        let rt: Arc<dyn Chronos2Runtime> = if let Some(path) = real_path {
            #[cfg(feature = "chronos2")]
            {
                match tokio::task::spawn_blocking({
                    let p = path.clone();
                    move || crate::chronos2::candle_runtime::CandleChronos2Runtime::load(&p, crate::config::NeureConfig::default_device())
                })
                .await
                {
                    Ok(Ok(rt)) => {
                        tracing::info!(model = %model, path = %path.display(), "loaded Chronos2 model");
                        Arc::new(rt) as Arc<dyn Chronos2Runtime>
                    }
                    _ => {
                        tracing::warn!(model = %model, path = %path.display(), "failed to load Chronos2 model; falling back to stub");
                        Arc::new(StubChronos2Runtime::new(model))
                    }
                }
            }
            #[cfg(not(feature = "chronos2"))]
            {
                let _ = path;
                Arc::new(StubChronos2Runtime::new(model))
            }
        } else {
            Arc::new(StubChronos2Runtime::new(model))
        };
        g.insert(model.to_string(), rt.clone());
        rt
    }
}

impl Default for Chronos2Registry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chronos2::{Chronos2Error, ForecastRequest};

    #[tokio::test]
    async fn runtime_for_returns_stub_for_unloaded_model() {
        let reg = Chronos2Registry::new();
        let rt = reg.runtime_for("missing-model").await;
        let resp = rt
            .forecast(ForecastRequest {
                model: "missing-model".into(),
                series: vec![1.0, 2.0, 3.0],
                horizon: 2,
                method: crate::chronos2::ForecastMethod::Mean,
            })
            .await;
        assert!(matches!(resp.unwrap_err(), Chronos2Error::NotImplemented(_)));
    }

    #[tokio::test]
    async fn runtime_for_caches_first_look_up() {
        let reg = Chronos2Registry::new();
        let a = reg.runtime_for("x").await;
        let b = reg.runtime_for("x").await;
        assert!(Arc::ptr_eq(&a, &b));
    }
}
