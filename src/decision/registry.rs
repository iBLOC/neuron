//! Model registry for the decision capability.
//!
//! Same shape as [`crate::rerank::registry`]: a list of registered models, a
//! lazy loaded-instance cache, and a [`ResourceTracker`] guarding admission.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use super::{
    DecisionImpl, DecisionRegistryKey, DecisionRequest, DecisionResponse, DecisionRuntime,
    RegisteredDecision,
};
use crate::config::{DeviceSelection, NeuronConfig, ResourceTracker};
use crate::llm::{ModelInfo, NeuronError};

pub struct DecisionRuntimeRegistry {
    pub registered: Vec<RegisteredDecision>,
    loaded: Arc<tokio::sync::Mutex<HashMap<DecisionRegistryKey, Arc<dyn DecisionRuntime>>>>,
    resources: Mutex<ResourceTracker>,
}

impl Clone for DecisionRuntimeRegistry {
    fn clone(&self) -> Self {
        Self {
            registered: self.registered.clone(),
            loaded: self.loaded.clone(),
            resources: Mutex::new(ResourceTracker::default()),
        }
    }
}

impl DecisionRuntimeRegistry {
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
        impl_id: DecisionImpl,
        device: DeviceSelection,
        required_memory_bytes: u64,
        model_path: String,
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

        self.registered.push(RegisteredDecision {
            model_id,
            impl_id,
            device,
            required_memory_bytes,
            model_path,
        });
        Ok(())
    }

    pub async fn runtime_for(&self, model_id: &str) -> Result<Arc<dyn DecisionRuntime>, NeuronError> {
        let entry = self
            .registered
            .iter()
            .find(|r| r.model_id == model_id)
            .ok_or_else(|| NeuronError::new(format!("ModelNotRegistered: {model_id}")))?;

        let key = DecisionRegistryKey {
            model_id: entry.model_id.clone(),
            impl_id: entry.impl_id,
            device: entry.device,
        };

        let mut loaded = self.loaded.lock().await;
        if let Some(rt) = loaded.get(&key) {
            return Ok(rt.clone());
        }

        let rt: Arc<dyn DecisionRuntime> = match entry.impl_id {
            #[cfg(feature = "tract")]
            DecisionImpl::Tract => {
                let loaded = super::tract_session::TractDecisionRuntime::load(
                    &entry.model_id,
                    &entry.device,
                )
                .await
                .map_err(|e| NeuronError::new(format!("TractDecisionRuntime load failed: {e}")))?;
                Arc::from(loaded)
            }
        };

        loaded.insert(key, rt.clone());
        Ok(rt)
    }

    /// Look up where a registered model's prepped ONNX graph lives.
    pub fn model_path(&self, model_id: &str) -> Option<String> {
        self.registered
            .iter()
            .find(|r| r.model_id == model_id)
            .map(|r| r.model_path.clone())
    }

    pub fn list_registered(&self) -> Vec<String> {
        self.registered.iter().map(|r| r.model_id.clone()).collect()
    }

    /// Enumerate the registered models without loading any of them.
    ///
    /// `/v1/models` is a pure listing, so it must not pay for a graph load.
    /// The runtime trait's own `list_models` is only reachable once a model
    /// is loaded, which is why the registry -- not the shortcut handle -- is
    /// the source of truth here.
    pub fn list_models(&self) -> Vec<ModelInfo> {
        self.registered
            .iter()
            .map(|r| ModelInfo::new(r.model_id.clone(), "decision"))
            .collect()
    }

    /// Return a placeholder for the first registered model, or an
    /// unavailable stub if nothing is registered.
    ///
    /// Loading the model here would require blocking on
    /// `runtime_for`, which panics with "Cannot start a runtime from
    /// within a runtime" whenever `ServerState::new` runs inside a
    /// tokio runtime (every `#[tokio::main]` host and every smoke
    /// example). See `LlmRuntimeRegistry::default_runtime` for the
    /// full rationale — the lazy load happens on `runtime_for`'s
    /// async path, which the HTTP handlers call.
    pub fn default_runtime(&self) -> Arc<dyn DecisionRuntime> {
        match self.registered.first() {
            Some(_) => Arc::new(UnavailableDecisionRuntime),
            None => Arc::new(UnavailableDecisionRuntime),
        }
    }
}

/// Error stub returned when no decision model is registered.
///
/// Per the "no echo stubs" rule every inference call against an unregistered
/// model must surface a clear error rather than fabricate output.
struct UnavailableDecisionRuntime;

#[async_trait]
impl DecisionRuntime for UnavailableDecisionRuntime {
    async fn load(_: &str, _: &DeviceSelection) -> crate::llm::ChatResult<Box<dyn DecisionRuntime>>
    where
        Self: Sized,
    {
        Err(NeuronError::new("NoRegisteredModel: register at least one Decision model"))
    }

    async fn decide(&self, _req: DecisionRequest) -> crate::llm::ChatResult<DecisionResponse> {
        Err(NeuronError::new("NoRegisteredModel: register at least one Decision model"))
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

    fn registry() -> DecisionRuntimeRegistry {
        DecisionRuntimeRegistry::new(Arc::new(NeuronConfig::default()))
    }

    #[test]
    fn test_registry_starts_empty() {
        let reg = registry();
        assert!(reg.list_registered().is_empty());
        assert!(reg.model_path("anything").is_none());
    }

    #[test]
    #[cfg(feature = "tract")]
    fn test_registry_register_then_list_registered() {
        let mut reg = registry();
        let impl_id = DecisionImpl::Tract;

        reg.register(
            "laya".to_string(),
            impl_id,
            DeviceSelection::Cpu,
            1024,
            "/tmp/laya.onnx".to_string(),
        )
        .unwrap();

        assert_eq!(reg.list_registered(), vec!["laya".to_string()]);
        assert_eq!(reg.model_path("laya").as_deref(), Some("/tmp/laya.onnx"));
        assert!(reg.model_path("other").is_none());
    }

    #[test]
    #[cfg(feature = "tract")]
    fn test_registry_register_rejects_excessive_memory() {
        let mut reg = registry();
        let impl_id = DecisionImpl::Tract;

        let err = reg
            .register(
                "huge".to_string(),
                impl_id,
                DeviceSelection::Cpu,
                u64::MAX,
                "/tmp/huge.onnx".to_string(),
            )
            .unwrap_err();
        assert!(err.message.contains("InsufficientResources"), "{}", err.message);
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_registry_runtime_for_unregistered_is_model_not_registered() {
        let reg = registry();
        let err = reg.runtime_for("ghost").await.err().expect("must fail");
        assert!(err.message.contains("ModelNotRegistered"), "{}", err.message);
    }

    #[tokio::test]
    async fn test_registry_default_runtime_without_registration_is_unavailable() {
        let reg = registry();
        let rt = reg.default_runtime();
        assert_eq!(rt.name(), "unavailable");
    }

    #[tokio::test]
    async fn test_registry_unavailable_decide_is_error() {
        let reg = registry();
        let rt = reg.default_runtime();
        let req = DecisionRequest::new("m", serde_json::json!("state"), vec![]);
        let err = rt.decide(req).await.err().expect("must fail");
        assert!(err.message.contains("NoRegisteredModel"), "{}", err.message);
    }
}
