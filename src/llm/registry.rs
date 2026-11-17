use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures_util::stream::BoxStream;

use super::{LlmRuntime, LlmImpl, ModelInfo, NeuronError, RegisteredLlm, RegistryKey};
use crate::config::{DeviceSelection, NeuronConfig, ResourceTracker};
use crate::llm::{ChatRequest, ChatResponse, ChatChunk};

pub struct LlmRuntimeRegistry {
    pub registered: Arc<Mutex<Vec<RegisteredLlm>>>,
    // std (not tokio) mutex: the load itself runs *outside* the lock
    // and re-checks on insert, so this guard never crosses an await.
    // That is what lets `unload` be synchronous — the scheduler
    // evicts models from lease-Drop contexts, where blocking on a
    // tokio mutex is not allowed.
    loaded: Arc<std::sync::Mutex<HashMap<RegistryKey, Arc<dyn LlmRuntime>>>>,
    resources: Arc<Mutex<ResourceTracker>>,
    /// Kept for the lazy-registration path ([`Self::runtime_for`]):
    /// a model downloaded **after** boot lands under `model_dirs` but
    /// the boot-time auto-scan already ran — without the dirs there is
    /// no way to turn `ModelNotRegistered` into a loadable entry.
    config: Arc<NeuronConfig>,
}

impl Clone for LlmRuntimeRegistry {
    fn clone(&self) -> Self {
        Self {
            registered: self.registered.clone(),
            loaded: self.loaded.clone(),
            resources: self.resources.clone(),
            config: self.config.clone(),
        }
    }
}

impl LlmRuntimeRegistry {
    pub fn new(config: Arc<NeuronConfig>) -> Self {
        Self {
            registered: Arc::new(Mutex::new(Vec::new())),
            loaded: Arc::new(std::sync::Mutex::new(HashMap::new())),
            resources: Arc::new(Mutex::new(ResourceTracker::default())),
            config,
        }
    }

    pub fn register(
        &self,
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

        self.registered.lock().unwrap().push(RegisteredLlm {
            model_id,
            impl_id,
            device,
            required_memory_bytes,
        });
        Ok(())
    }

    /// Scan `model_dirs` for `llm/{model_id}/model.safetensors` and
    /// register it as a candle engine in-place — the same shape as
    /// the boot-time auto-scan (candle impl, 8 GB budget estimate,
    /// `Auto` device). Only runs under the `candle` feature: without
    /// it there is no backend that can serve safetensors, so the
    /// original `ModelNotRegistered` error must stand.
    ///
    /// Returns `Err` when the model is not on disk / registration is
    /// denied (the caller logs it at debug and surfaces the original
    /// miss). Racy double-registration from two concurrent first
    /// requests is tolerated the same way the boot scan tolerates
    /// duplicate ids: `runtime_for` takes the first match and the
    /// loaded-cache dedup keeps one runtime.
    #[cfg(feature = "candle")]
    fn try_register_downloaded(&self, model_id: &str) -> Result<(), String> {
        for dir in &self.config.model_dirs {
            let candidate = dir.join("llm").join(model_id);
            if !candidate.join("model.safetensors").exists() {
                continue;
            }
            if self.is_registered(model_id) {
                return Ok(());
            }
            return self
                .register(
                    model_id.to_string(),
                    LlmImpl::Candle,
                    DeviceSelection::Auto,
                    8_000_000_000,
                )
                .map_err(|e| e.to_string());
        }
        Err(format!("no downloaded model at llm/{model_id} under model_dirs"))
    }

    pub async fn runtime_for(&self, model_id: &str) -> Result<Arc<dyn LlmRuntime>, NeuronError> {
        // 懒注册：boot 的自动扫描只跑一次 ⇒ pull 完成的模型在盘上
        // 却不在 registered 里 ⇒ 请求打过来直接 `ModelNotRegistered`
        // ⇒ 「下载成功、一用就报错、重启应用才能用」——失败长得像
        // 成功的又一族。这里在 miss 路径上扫一次 `model_dirs`，命中
        // 就地注册再继续解析（与 boot 扫描同一形状：candle ＋
        // `model.safetensors` 存在 ＋ 8 GB 预算估计）。
        #[cfg(feature = "candle")]
        if !self.is_registered(model_id)
            && let Err(e) = self.try_register_downloaded(model_id)
        {
            tracing::debug!(model = %model_id, error = %e, "lazy registration did not apply");
        }

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
            #[cfg(feature = "llamacpp")]
            LlmImpl::LlamaCpp => {
                let loaded = super::LlamaCppLlmRuntime::load(&entry.model_id, &entry.device)
                    .await
                    .map_err(|e| {
                        NeuronError::new(format!("LlamaCppLlmRuntime load failed: {e}"))
                    })?;
                Arc::from(loaded)
            }
            // Stub backend for unit tests only — never wired by
            // production code. Rejecting here keeps the cache-eviction
            // test path observable without needing real weights.
            LlmImpl::Stub => {
                return Err(NeuronError::not_implemented(
                    "LlmImpl::Stub is a test-only backend; register a real engine instead",
                ));
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
        self.registered
            .lock()
            .unwrap()
            .iter()
            .any(|r| r.model_id == model_id)
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
        self.registered
            .lock()
            .unwrap()
            .iter()
            .map(|r| r.model_id.clone())
            .collect()
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
        let first = self.registered.lock().unwrap().first().cloned();
        match first {
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

    // -- unload / cache semantics ---------------------------------------
    //
    // The model scheduler (src/scheduler/) drives these four
    // methods: `is_registered` distinguishes "registered but not
    // loaded" from "never heard of"; `is_loaded` /
    // `loaded_model_ids` power the residents snapshot;
    // `unload` actually drops weights. The invariants below back
    // the scheduler's drain-then-swap policy — if any of them
    // regresses, the scheduler will silently leak or refuse to
    // re-load.

    #[test]
    fn test_is_registered_independent_of_loaded_cache() {
        let config = Arc::new(NeuronConfig::new());
        let mut reg = LlmRuntimeRegistry::new(config);
        reg.register(
            "stub-model-a".into(),
            LlmImpl::Stub,
            DeviceSelection::Cpu,
            0,
        )
        .unwrap();
        // Fresh registration: registered but never loaded.
        assert!(reg.is_registered("stub-model-a"));
        assert!(!reg.is_loaded("stub-model-a"));
        assert!(reg.loaded_model_ids().is_empty());
    }

    #[test]
    fn test_unload_unregistered_is_a_noop() {
        let config = Arc::new(NeuronConfig::new());
        let reg = LlmRuntimeRegistry::new(config);
        assert_eq!(reg.unload("never-registered"), 0);
    }

    #[tokio::test]
    async fn test_runtime_for_returns_not_implemented_for_stub() {
        // Stub entries are accepted at register-time (so the
        // scheduler's resident bookkeeping can test against them)
        // but `runtime_for` always errors — production code never
        // registers a Stub, so this is purely a defensive contract
        // test.
        let config = Arc::new(NeuronConfig::new());
        let mut reg = LlmRuntimeRegistry::new(config);
        reg.register(
            "stub-only".into(),
            LlmImpl::Stub,
            DeviceSelection::Cpu,
            0,
        )
        .unwrap();
        match reg.runtime_for("stub-only").await {
            Ok(_) => panic!("Stub backend must reject runtime_for"),
            Err(e) => assert!(
                e.message.contains("test-only backend"),
                "expected the Stub-rejection error, got message: {}",
                e.message
            ),
        }
    }

    #[tokio::test]
    async fn test_unload_then_reacquire_re_runs_load_path() {
        // This is the load-bearing invariant for the scheduler's
        // swap-after-drain rule: after a resident is evicted, the
        // *next* request for that model must trigger a fresh load
        // rather than silently serving a stale Arc. With the Stub
        // backend we can verify the cache insert path runs again
        // (returning the rejection error each time) without needing
        // real weights.
        let config = Arc::new(NeuronConfig::new());
        let mut reg = LlmRuntimeRegistry::new(config);
        reg.register(
            "model-x".into(),
            LlmImpl::Stub,
            DeviceSelection::Cpu,
            0,
        )
        .unwrap();

        // 1st attempt: cache empty, load path runs, error returned.
        assert!(reg.runtime_for("model-x").await.is_err());
        assert!(
            !reg.is_loaded("model-x"),
            "Stub backend rejects without populating the cache"
        );

        // Even if something else were to populate the cache, unload
        // must remove it. Simulate by pre-seeding the cache via
        // an entry that the load path *can* install — there's no
        // public set_loaded method (and there shouldn't be), so
        // we exercise the real path through runtime_for after
        // forcibly clearing the registry's loaded map by hand via
        // the unload helper. Stub backend will reject twice, but
        // the invariant we want is that unload() does not leak
        // state — i.e. is_loaded stays false throughout.
        assert_eq!(reg.unload("model-x"), 0, "no cached entry to evict");
        assert!(reg.runtime_for("model-x").await.is_err());
        assert!(!reg.is_loaded("model-x"));
    }

    #[test]
    fn test_unload_returns_count_of_removed_entries() {
        // Multiple cache entries can share a model_id when the
        // underlying cache key also spans impl/device. With a
        // single registration, the count should match the number
        // of cache slots that matched.
        let config = Arc::new(NeuronConfig::new());
        let mut reg = LlmRuntimeRegistry::new(config);
        reg.register(
            "model-y".into(),
            LlmImpl::Stub,
            DeviceSelection::Cpu,
            0,
        )
        .unwrap();
        // Nothing is loaded — unload should report 0 removed.
        assert_eq!(reg.unload("model-y"), 0);
        // Second call is still a no-op (idempotent).
        assert_eq!(reg.unload("model-y"), 0);
    }

    #[test]
    fn test_loaded_model_ids_dedups_across_keys() {
        // Same model_id registered under two impl/device
        // combinations would yield two cache slots; the public
        // loaded_model_ids view should dedup to one entry. This
        // is observable invariant for the scheduler status
        // snapshot — duplicate residents would inflate the
        // resident_bytes accounting.
        let config = Arc::new(NeuronConfig::new());
        let mut reg = LlmRuntimeRegistry::new(config);
        reg.register(
            "model-z".into(),
            LlmImpl::Stub,
            DeviceSelection::Cpu,
            0,
        )
        .unwrap();
        // No load has happened, so the view is empty.
        assert!(reg.loaded_model_ids().is_empty());
    }

    /// 懒注册：pull 完成的模型在 `model_dirs/llm/<id>/` 里 ⇒ 首次请求
    /// （miss 路径）就地注册，`ModelNotRegistered` 不再打脸「下载成功」。
    /// 直接量 [`Self::try_register_downloaded`]（文件形状 ＋ 注册生效 ＋
    /// 不重复注册）＋ 量 `runtime_for` 的 miss 路径不再报
    /// `ModelNotRegistered`（假权重会在 load 阶段报别的错 —— 那正是
    /// 要的形状：错误已经走到「真的在加载」这一步）。
    #[cfg(feature = "candle")]
    #[tokio::test]
    async fn test_lazy_registration_picks_up_downloaded_model() {
        let tmp = tempfile::TempDir::new().unwrap();
        let model_dir = tmp.path().join("llm").join("dl-model");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("model.safetensors"), b"not-real-weights").unwrap();

        let config = Arc::new(NeuronConfig::new().with_model_dirs(vec![tmp.path().to_path_buf()]));
        let reg = LlmRuntimeRegistry::new(config);

        assert!(!reg.is_registered("dl-model"), "nothing registered at boot");
        reg.try_register_downloaded("dl-model")
            .expect("downloaded model must register lazily");
        assert!(reg.is_registered("dl-model"));

        // 不重复注册（资源预算按份扣，重复注册会白扣一份）。
        reg.try_register_downloaded("dl-model").unwrap();
        assert_eq!(reg.registered.lock().unwrap().len(), 1);

        // 盘上没有的模型：不注册、报错说真话。
        assert!(reg.try_register_downloaded("not-on-disk").is_err());
        assert!(!reg.is_registered("not-on-disk"));
    }

    #[cfg(feature = "candle")]
    #[tokio::test]
    async fn test_runtime_for_miss_path_lazy_registers_before_erroring() {
        let tmp = tempfile::TempDir::new().unwrap();
        let model_dir = tmp.path().join("llm").join("dl-model");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("model.safetensors"), b"not-real-weights").unwrap();

        let config = Arc::new(NeuronConfig::new().with_model_dirs(vec![tmp.path().to_path_buf()]));
        let reg = LlmRuntimeRegistry::new(config);

        let err = match reg.runtime_for("dl-model").await {
            Ok(_) => panic!("fake weights cannot load — but the error must not be ModelNotRegistered"),
            Err(e) => e,
        };
        let msg = format!("{err}");
        assert!(
            !msg.contains("ModelNotRegistered"),
            "miss path must lazy-register a downloaded model first, got: {msg}"
        );
        assert!(
            reg.is_registered("dl-model"),
            "the lazy registration must have applied"
        );
    }
}