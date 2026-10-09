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
    /// Remove a registered engine. Returns whether it was present.
    /// Used by the model scheduler to evict weights; the engine frees
    /// when the last `Arc` (in-flight requests only) drops.
    pub fn unregister(&self, model_id: &str) -> bool {
        self.by_model.write().unwrap().remove(model_id).is_some()
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
    pub ocr: Arc<EngineRegistry<dyn crate::engine::AnyCapabilityEngine>>,
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
            ocr: Arc::new(EngineRegistry::new()),
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
            Capability::Ocr => &self.ocr,
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
            Capability::Ocr => self.ocr.default(),
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

    // -- unregister / Arc-drop invariants ----------------------------------
    //
    // `unregister` is the canonical-engine side of the scheduler's
    // drain-then-swap policy. The contract is:
    //   1. unregister(model_id) returns true iff the id was registered.
    //   2. After unregister, subsequent lookup(model_id) returns None.
    //   3. unregister does NOT eagerly drop weights — the engine Arc
    //      stays alive until the last clone (e.g. an in-flight HTTP
    //      request) drops. Verified by holding an extra clone and
    //      checking `lookup` from before vs after unregister.
    //   4. A second unregister of the same id is idempotent (false).
    //   5. list() reflects the post-unregister view.

    #[test]
    fn test_unregister_returns_true_when_present_false_otherwise() {
        let reg: EngineRegistry<dyn AnyCapabilityEngine> = EngineRegistry::new();
        let e: Arc<dyn AnyCapabilityEngine> = Arc::new(MockEngine { id: "m".into() });
        reg.register("a".into(), e);
        assert!(reg.unregister("a"));
        assert!(!reg.unregister("a"), "second unregister is idempotent (false)");
        assert!(!reg.unregister("never-registered"));
    }

    #[test]
    fn test_unregister_makes_lookup_return_none_but_keeps_existing_arc_alive() {
        let reg: EngineRegistry<dyn AnyCapabilityEngine> = EngineRegistry::new();
        let e: Arc<dyn AnyCapabilityEngine> = Arc::new(MockEngine { id: "m".into() });
        reg.register("a".into(), e);
        // Caller may still hold an Arc (e.g. an in-flight HTTP handler
        // cloning the engine after `register` returned). unregister
        // does NOT take that Arc away from the caller.
        let held = reg.lookup("a").expect("registered");
        assert_eq!(held.name(), "m");
        // After unregister, the registry has no record of "a" anymore.
        assert!(reg.unregister("a"));
        assert!(reg.lookup("a").is_none());
        // The Arc the caller held is still valid — they can finish
        // the request it was handed into without panicking on a
        // dangling trait object.
        assert_eq!(held.name(), "m");
    }

    #[test]
    fn test_unregister_then_re_register_under_same_id() {
        // After eviction the scheduler may admit a different engine
        // under the same id later (e.g. a different weight file
        // backed by a different runtime). Verify the registry
        // accepts a fresh registration under a previously-evicted
        // id without state from the old engine bleeding through.
        let reg: EngineRegistry<dyn AnyCapabilityEngine> = EngineRegistry::new();
        reg.register("a".into(), Arc::new(MockEngine { id: "first".into() }));
        assert!(reg.unregister("a"));
        assert!(reg.lookup("a").is_none());
        reg.register("a".into(), Arc::new(MockEngine { id: "second".into() }));
        let after = reg.lookup("a").expect("re-registered");
        assert_eq!(after.name(), "second");
        assert_eq!(reg.list(), vec!["a".to_string()]);
    }

    #[test]
    fn test_unregister_isolates_per_capability_registry() {
        // unregister on the LLM engine table must NOT touch the
        // TTS / ASR / … tables — capability isolation is the
        // scheduler's load-bearing invariant.
        let regs = CapabilityRegistries::new();
        regs.llm.register("shared-id".into(), Arc::new(MockEngine { id: "llm".into() }));
        regs.tts.register("shared-id".into(), Arc::new(MockEngine { id: "tts".into() }));

        assert!(regs.llm.unregister("shared-id"));
        assert!(regs.lookup(Capability::Llm, "shared-id").is_none());
        // TTS entry is untouched.
        assert_eq!(
            regs.lookup(Capability::Tts, "shared-id").unwrap().name(),
            "tts"
        );
        // Unregistering the same id from the TTS table now returns true.
        assert!(regs.tts.unregister("shared-id"));
    }

    #[test]
    fn test_set_default_is_independent_of_unregister() {
        // unregister removes a model_id mapping but does not touch
        // the default_shortcut. This matters for the scheduler's
        // "registered model id is gone, fall back to the default
        // engine" path — the default survives any unregister.
        let reg: EngineRegistry<dyn AnyCapabilityEngine> = EngineRegistry::new();
        let e_default: Arc<dyn AnyCapabilityEngine> = Arc::new(MockEngine { id: "default".into() });
        reg.set_default(e_default);
        reg.register("a".into(), Arc::new(MockEngine { id: "a".into() }));
        assert!(reg.unregister("a"));
        assert!(reg.default().is_some(), "default survives unregister");
        assert_eq!(reg.default().unwrap().name(), "default");
    }
}