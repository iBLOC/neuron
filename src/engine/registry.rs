use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

use crate::capabilities::Capability;

pub struct EngineRegistry<T: ?Sized + Send + Sync + 'static> {
    by_model: RwLock<HashMap<String, Arc<T>>>,
    default_shortcut: RwLock<Option<Arc<T>>>,
    _marker: PhantomData<T>,
}

impl<T: ?Sized + Send + Sync + 'static> EngineRegistry<T> {
    pub fn new() -> Self {
        Self { by_model: RwLock::new(HashMap::new()),
               default_shortcut: RwLock::new(None),
               _marker: PhantomData }
    }
    pub fn register(&self, model_id: String, engine: Arc<T>) {
        self.by_model.write().unwrap().insert(model_id, engine);
    }
    pub fn lookup(&self, model_id: &str) -> Option<Arc<T>> {
        self.by_model.read().unwrap().get(model_id).cloned()
    }
    pub fn set_default(&self, engine: Arc<T>) {
        *self.default_shortcut.write().unwrap() = Some(engine);
    }
    pub fn default(&self) -> Option<Arc<T>> {
        self.default_shortcut.read().unwrap().clone()
    }
    pub fn list(&self) -> Vec<String> {
        self.by_model.read().unwrap().keys().cloned().collect()
    }
}

impl<T: ?Sized + Send + Sync + 'static> Default for EngineRegistry<T> {
    fn default() -> Self { Self::new() }
}

pub struct CapabilityRegistries {
    pub llm: Arc<EngineRegistry<dyn crate::engine::AnyCapabilityEngine>>,
    pub tts: Arc<EngineRegistry<dyn crate::engine::AnyCapabilityEngine>>,
    pub asr: Arc<EngineRegistry<dyn crate::engine::AnyCapabilityEngine>>,
    pub rerank: Arc<EngineRegistry<dyn crate::engine::AnyCapabilityEngine>>,
    pub embedding: Arc<EngineRegistry<dyn crate::engine::AnyCapabilityEngine>>,
    pub vision: Arc<EngineRegistry<dyn crate::engine::AnyCapabilityEngine>>,
    pub decision: Arc<EngineRegistry<dyn crate::engine::AnyCapabilityEngine>>,
}

impl CapabilityRegistries {
    pub fn new() -> Self {
        Self {
            llm: Arc::new(EngineRegistry::new()),
            tts: Arc::new(EngineRegistry::new()),
            asr: Arc::new(EngineRegistry::new()),
            rerank: Arc::new(EngineRegistry::new()),
            embedding: Arc::new(EngineRegistry::new()),
            vision: Arc::new(EngineRegistry::new()),
            decision: Arc::new(EngineRegistry::new()),
        }
    }

    pub fn lookup(&self, cap: Capability, model_id: &str) -> Option<Arc<dyn crate::engine::AnyCapabilityEngine>> {
        let reg: &EngineRegistry<dyn crate::engine::AnyCapabilityEngine> = match cap {
            Capability::Llm => &self.llm,
            Capability::Tts => &self.tts,
            Capability::Asr => &self.asr,
            Capability::Rerank => &self.rerank,
            Capability::Embedding => &self.embedding,
            Capability::Vision => &self.vision,
            Capability::Decision => &self.decision,
        };
        reg.lookup(model_id)
    }

    pub fn lookup_default(&self, cap: Capability) -> Option<Arc<dyn crate::engine::AnyCapabilityEngine>> {
        match cap {
            Capability::Llm => self.llm.default(),
            Capability::Tts => self.tts.default(),
            Capability::Asr => self.asr.default(),
            Capability::Rerank => self.rerank.default(),
            Capability::Embedding => self.embedding.default(),
            Capability::Vision => self.vision.default(),
            Capability::Decision => self.decision.default(),
        }
    }

    pub fn model_id_from_canonical(req: &crate::canonical::CanonicalRequest) -> Option<String> {
        match req {
            crate::canonical::CanonicalRequest::Llm(r) => Some(r.model.clone()),
            crate::canonical::CanonicalRequest::Vision(r) => Some(r.model.clone()),
            _ => None,
        }
    }
}

impl Default for CapabilityRegistries {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical::{CanonicalLlmRequest, CanonicalLlmResponse};
    use crate::engine::{AnyCapabilityEngine, LlmRuntime};
    use async_trait::async_trait;
    use futures_util::stream::BoxStream;

    struct MockEngine { id: String }
    #[async_trait]
    impl LlmRuntime for MockEngine {
        async fn execute(&self, _req: CanonicalLlmRequest)
            -> crate::llm::ChatResult<CanonicalLlmResponse> { unimplemented!() }
        async fn execute_stream(&self, _req: CanonicalLlmRequest)
            -> crate::llm::ChatResult<BoxStream<'static, crate::canonical::CanonicalLlmStreamEvent>> { unimplemented!() }
        fn capabilities(&self) -> &crate::capabilities::ModelCapabilities { unimplemented!() }
        fn name(&self) -> &str { &self.id }
    }

    #[test]
    fn test_engine_registry_register_lookup() {
        let reg: EngineRegistry<dyn AnyCapabilityEngine> = EngineRegistry::new();
        assert!(reg.lookup("x").is_none());
        let e: Arc<dyn AnyCapabilityEngine> = Arc::new(MockEngine { id: "m".into() });
        reg.register("x".into(), e);
        assert!(reg.lookup("x").is_some());
    }

    #[test]
    fn test_capability_registries_dispatch_by_cap() {
        let regs = CapabilityRegistries::new();
        let e: Arc<dyn AnyCapabilityEngine> = Arc::new(MockEngine { id: "llm".into() });
        regs.llm.register("qwen3".into(), e);
        assert!(regs.lookup(Capability::Llm, "qwen3").is_some());
        assert!(regs.lookup(Capability::Tts, "qwen3").is_none());
        assert!(regs.lookup(Capability::Llm, "missing").is_none());
    }
}