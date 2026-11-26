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
    /// ⚠️ **模型库根**（刀 16 续同源，2026-11-26）：加载面必须读**下载面／
    /// 注册面用的同一份** ⇒ ⇒ 否则嵌脑形态里宿主给的 `<data_dir>/models`
    /// 只被用来下载，加载时却去别处找 ⇒ ⇒ **引擎注册上了、权重找不到**
    /// （失败长得像「模型没下载」⇒ ⇒ 而盘上**明明有**那份权重）。
    ///
    /// ⚠️ 此前这一格是 `_config`（**收了又丢**）⇒ ⇒ 加载面只能去读 env ⇒ ⇒
    /// **两个真相源** ⇒ ⇒ 同一个模型在同一个决策里被判成两个相反的结论。
    config: Arc<NeuronConfig>,
}

impl Clone for AsrRuntimeRegistry {
    fn clone(&self) -> Self {
        Self {
            registered: self.registered.clone(),
            loaded: self.loaded.clone(),
            resources: self.resources.clone(),
            config: self.config.clone(),
        }
    }
}

impl AsrRuntimeRegistry {
    pub fn new(config: Arc<NeuronConfig>) -> Self {
        Self {
            registered: Arc::new(Mutex::new(Vec::new())),
            loaded: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            resources: Arc::new(Mutex::new(ResourceTracker::default())),
            config,
        }
    }

    /// **加载面要递下去的那份根**（＝ `config.model_dirs`）—— 判据钉这一格。
    pub fn model_roots(&self) -> &[std::path::PathBuf] {
        &self.config.model_dirs
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

    /// 这枚模型**此刻注册了吗**（注册表是**运行期内存** ⇒ ⇒ 脑一重启
    /// 就空 ⇒ ⇒ 「盘上有」与「注册着」是**两件事**）。
    fn is_registered(&self, model_id: &str) -> bool {
        self.registered
            .lock()
            .unwrap()
            .iter()
            .any(|r| r.model_id == model_id)
    }

    /// ⚠️⚠️ **盘上有 ⇒ ⇒ 自动注册**（2026-11-26）—— **LLM 那条腿同形**
    /// （`llm/registry.rs::try_register_downloaded`；父仓闸 N6 钉的就是
    /// 「下载完即用，不必重启应用」）⇒ ⇒ **ASR 这条腿当时没跟上**
    /// ⇒ ⇒ 后果：**脑一重启，下载好的 ASR 权重就废** ⇒ ⇒ 而客户端的
    /// 自动注册**只在下完那一刻**做一次 ⇒ ⇒ 「模型明明在盘上」却报
    /// `ModelNotRegistered` ⇒ ⇒ **失败长得像「模型没下载」**。
    ///
    /// ⚠️ **注册形状与端脑那一面逐字同形**（`telencephalon
    /// src/api/neuron_public.rs`）：`DeviceSelection::Auto` ＋
    /// `required_memory_bytes = 0`（那个数在端脑侧就是占位）⇒ ⇒ 两面
    /// 不许各算各的预算。
    ///
    /// ⚠️ **引擎不猜** ⇒ ⇒ 从**盘上事实**推（`infer_asr_impl_from_dir`）
    /// ⇒ ⇒ 推不出来 ⇒ ⇒ **不注册**（⇒ ⇒ `ModelNotRegistered` 照旧）。
    fn try_register_downloaded(&self, model_id: &str) -> Result<(), String> {
        if self.is_registered(model_id) {
            return Ok(());
        }
        for dir in &self.config.model_dirs {
            let candidate = dir.join("asr").join(model_id);
            if !candidate.is_dir() {
                continue;
            }
            let Some(impl_id) = super::infer_asr_impl_from_dir(&candidate) else {
                continue;
            };
            return self
                .register(model_id.to_string(), impl_id, DeviceSelection::Auto, 0)
                .map_err(|e| e.message.clone());
        }
        Err(format!("no downloaded ASR model for {model_id}"))
    }

    /// ⚠️ **盘上第一枚能装的 ASR 模型**（2026-11-26）—— 「默认引擎」那一格
    /// 的兜底：**注册表是空的**（脑重启后）⇒ ⇒ 但**盘上有** ⇒ ⇒ 把它
    /// 注册上 ⇒ ⇒ 「下载完就能用」跨得过重启。
    ///
    /// ⚠️ **为什么要单独一格**：端脑的解析序是「显式 id → 叶区覆盖 →
    /// 全局默认 → **第一个已注册引擎**」⇒ ⇒ **最后那一条**走的是
    /// `lazy_default_runtime()` ⇒ ⇒ **它不会去问 `runtime_for`** ⇒ ⇒
    /// 只在 miss 路径上懒注册 ⇒ ⇒ **默认那条路永远落空** ⇒ ⇒ 必须两处
    /// 都接（这一刀踩的就是这个）。
    fn first_downloaded_asr_model(&self) -> Option<String> {
        for dir in &self.config.model_dirs {
            let asr_root = dir.join("asr");
            let Ok(entries) = std::fs::read_dir(&asr_root) else {
                continue;
            };
            let mut candidates: Vec<String> = entries
                .filter_map(|e| e.ok())
                .filter(|e| e.path().is_dir())
                .filter_map(|e| e.file_name().to_str().map(|s| s.to_string()))
                .filter(|id| super::infer_asr_impl_from_dir(&asr_root.join(id)).is_some())
                .collect();
            candidates.sort();
            if let Some(first) = candidates.first().cloned()
                && self.try_register_downloaded(&first).is_ok()
            {
                return Some(first);
            }
        }
        None
    }

    pub async fn runtime_for(&self, model_id: &str) -> Result<Arc<dyn AsrRuntime>, NeuronError> {
        let entry = {
            let registered = self.registered.lock().unwrap();
            registered.iter().find(|r| r.model_id == model_id).cloned()
        };
        // ⚠️⚠️ **没注册 ⇒ ⇒ 先试「盘上有 ⇒ 自动注册」**（LLM 那条腿同形；
        // 见 `try_register_downloaded` 的口径）⇒ ⇒ 脑重启后下载好的权重
        // 仍然能用 ⇒ ⇒ 而不是一个「模型明明在盘上」的 `ModelNotRegistered`。
        let entry = match entry {
            Some(e) => e,
            None => {
                let _ = self.try_register_downloaded(model_id);
                let registered = self.registered.lock().unwrap();
                registered
                    .iter()
                    .find(|r| r.model_id == model_id)
                    .cloned()
                    .ok_or_else(|| NeuronError::new(format!("ModelNotRegistered: {model_id}")))?
            }
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
                // ⚠️ **把【注册面用的那份根】原样递给加载面**（刀 16 续同源）
                // ⇒ ⇒ 注册刚在 `config.model_dirs` 里找到这份权重 ⇒ ⇒ 加载
                // 就必须读**同一份** ⇒ ⇒ 否则「引擎注册上了、权重找不到」。
                let loaded = WhisperAsrRuntime::load_with_roots(
                    &entry.model_id,
                    &entry.device,
                    &self.config.model_dirs,
                )
                    .await
                    .map_err(|e| {
                        NeuronError::new(format!("WhisperAsrRuntime load failed: {e}"))
                    })?;
                Arc::from(loaded)
            }
            #[cfg(feature = "tract")]
            AsrImpl::Tract => {
                let loaded = TractSenseVoiceRuntime::load_with_roots(
                    &entry.model_id,
                    &entry.device,
                    &self.config.model_dirs,
                )
                    .await
                    .map_err(|e| {
                        NeuronError::new(format!("TractSenseVoiceRuntime load failed: {e}"))
                    })?;
                Arc::from(loaded)
            }
            #[cfg(feature = "sherpa")]
            AsrImpl::Sherpa => {
                let loaded = SherpaAsrRuntime::load_with_roots(
                    &entry.model_id,
                    &entry.device,
                    &self.config.model_dirs,
                )
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
        // ⚠️⚠️ **注册表是空的 ⇒ ⇒ 先看盘上有没有能装的模型**（2026-11-26）
        // ⇒ ⇒ 脑重启后「第一个已注册引擎」那一格**不许**空着 ⇒ ⇒
        // 下载好的权重跨得过重启 ⇒ ⇒ 「下载完就能用」是真的。
        // ⇒ ⇒ 而端脑解析序的**最后一条**正是走这里 ⇒ ⇒ 它**不会**去问
        // `runtime_for` ⇒ ⇒ 必须在这一格自己接上。
        if self.registered.lock().unwrap().is_empty() {
            let _ = self.first_downloaded_asr_model();
        }
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

    /// ⚠️⚠️ **盘上有 ⇒ ⇒ 自动注册**（2026-11-26）—— 判据 F4-a。
    ///
    /// ⚠️ **为什么**：注册表是**运行期内存** ⇒ ⇒ 脑一重启就空 ⇒ ⇒ 而客户端
    /// 的自动注册**只在下完那一刻**做一次 ⇒ ⇒ 「模型明明在盘上」却报
    /// `ModelNotRegistered` ⇒ ⇒ **失败长得像「模型没下载」**。
    #[test]
    fn a_downloaded_asr_model_gets_registered_on_demand() {
        let root = std::env::temp_dir().join(format!("neuron-asr-auto-{}", uuid::Uuid::new_v4()));
        let model_dir = root.join("asr").join("auto-probe-model");
        std::fs::create_dir_all(&model_dir).unwrap();
        // ⚠️ **candle 的 whisper 布局**（目录上**看着**就是一份能装的权重）。
        std::fs::write(model_dir.join("config.json"), b"{}").unwrap();
        std::fs::write(model_dir.join("tokenizer.json"), b"{}").unwrap();
        std::fs::write(model_dir.join("model.safetensors"), b"fake").unwrap();
        let config = Arc::new(NeuronConfig::new().with_model_dirs(vec![root.clone()]));
        let registry = AsrRuntimeRegistry::new(config);
        assert!(
            registry.list_registered().is_empty(),
            "这条判据的前提：注册表一开始是空的"
        );
        registry
            .try_register_downloaded("auto-probe-model")
            .expect("盘上有一份能装的权重 ⇒ ⇒ 应当注册得上");
        assert_eq!(
            registry.list_registered(),
            vec!["auto-probe-model".to_string()]
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// ⚠️ **盘上没有 ⇒ ⇒ 仍然是 `ModelNotRegistered`**（2026-11-26）——
    /// 判据 F4-b。
    ///
    /// ⚠️ **为什么要有这一格**：懒注册不许把「没有」说成「有」⇒ ⇒
    /// 「判不了 ≠ 没有」这条不许破。
    #[tokio::test]
    async fn a_missing_model_still_reports_model_not_registered() {
        let root = std::env::temp_dir().join(format!("neuron-asr-none-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(root.join("asr")).unwrap();
        let config = Arc::new(NeuronConfig::new().with_model_dirs(vec![root.clone()]));
        let registry = Arc::new(AsrRuntimeRegistry::new(config));
        let err = match registry.runtime_for("nothing-here").await {
            Ok(_) => panic!("一个不存在的模型不许能加载"),
            Err(e) => e,
        };
        assert!(
            err.message.contains("ModelNotRegistered"),
            "盘上没有 ⇒ ⇒ 必须还是 ModelNotRegistered，got: {}",
            err.message
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// ⚠️ **引擎不猜**（2026-11-26）—— 判据 F4-c。
    #[test]
    fn a_half_baked_model_dir_does_not_get_registered() {
        let root = std::env::temp_dir().join(format!("neuron-asr-half-{}", uuid::Uuid::new_v4()));
        let model_dir = root.join("asr").join("half-probe-model");
        std::fs::create_dir_all(&model_dir).unwrap();
        // ⚠️ **只有权重、没有 config/tokenizer** ⇒ ⇒ 推不出引擎 ⇒ ⇒
        // **不许注册**（⇒ ⇒ 「引擎注册上了、权重找不到」那一族不许回来）。
        std::fs::write(model_dir.join("model.safetensors"), b"fake").unwrap();
        let config = Arc::new(NeuronConfig::new().with_model_dirs(vec![root.clone()]));
        let registry = AsrRuntimeRegistry::new(config);
        assert!(
            registry
                .try_register_downloaded("half-probe-model")
                .is_err()
        );
        assert!(registry.list_registered().is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

    /// ⚠️⚠️ **「默认引擎」那一格不许空着**（2026-11-26）—— 判据 F4-d。
    ///
    /// ⚠️ **为什么**：端脑的解析序最后一条是「**第一个已注册引擎**」⇒ ⇒
    /// 它走的是 `lazy_default_runtime()` ⇒ ⇒ **不会去问 `runtime_for`**
    /// ⇒ ⇒ 只在 miss 路径懒注册 ⇒ ⇒ 默认那条路永远落空 ⇒ ⇒ 必须两处都接。
    #[test]
    fn the_default_engine_falls_back_to_a_downloaded_model() {
        let root = std::env::temp_dir().join(format!("neuron-asr-def-{}", uuid::Uuid::new_v4()));
        let model_dir = root.join("asr").join("default-probe-model");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("config.json"), b"{}").unwrap();
        std::fs::write(model_dir.join("tokenizer.json"), b"{}").unwrap();
        std::fs::write(model_dir.join("model.safetensors"), b"fake").unwrap();
        let config = Arc::new(NeuronConfig::new().with_model_dirs(vec![root.clone()]));
        let registry = Arc::new(AsrRuntimeRegistry::new(config));
        assert!(registry.list_registered().is_empty());
        let shortcut = registry.lazy_default_runtime();
        // ⚠️ **名字不是 "unavailable"** ⇒ ⇒ 这一格**不是**空的。
        assert_eq!(shortcut.name(), "lazy-asr-default");
        // ⚠️ **注册表此刻必须已经有那枚引擎了**。
        assert_eq!(
            registry.list_registered(),
            vec!["default-probe-model".to_string()],
            "默认引擎那一格必须落到盘上那枚模型"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// ⚠️ **从盘上事实推引擎**（2026-11-26）—— 判据 F4-e（纯函数）。
    #[test]
    fn the_asr_impl_is_inferred_from_the_files_on_disk() {
        let root = std::env::temp_dir().join(format!("neuron-asr-imp-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        // ⚠️ **空目录 ⇒ ⇒ 推不出来**。
        assert_eq!(crate::asr::infer_asr_impl_from_dir(&root), None);
        // ⚠️ **只有 config/tokenizer、没有权重 ⇒ ⇒ 还是推不出来**。
        std::fs::write(root.join("config.json"), b"{}").unwrap();
        std::fs::write(root.join("tokenizer.json"), b"{}").unwrap();
        assert_eq!(crate::asr::infer_asr_impl_from_dir(&root), None);
        std::fs::write(root.join("model.safetensors"), b"fake").unwrap();
        #[cfg(feature = "candle")]
        assert_eq!(
            crate::asr::infer_asr_impl_from_dir(&root),
            Some(super::AsrImpl::Candle)
        );
        let _ = std::fs::remove_dir_all(&root);
    }

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

    /// ⚠️ **注册表不许把 config 丢掉**（2026-11-26）—— 此前 `new(_config)`
    /// **收了又丢** ⇒ ⇒ 加载面只能去读 env ⇒ ⇒ **两个真相源** ⇒ ⇒ 嵌脑形态
    /// 里「引擎注册上了、权重找不到」。
    #[test]
    fn asr_registry_keeps_the_config_model_dirs() {
        let dir = std::env::temp_dir().join(format!("neuron-asr-roots-{}", uuid::Uuid::new_v4()));
        let config = Arc::new(NeuronConfig::new().with_model_dirs(vec![dir.clone()]));
        let registry = AsrRuntimeRegistry::new(config);
        assert_eq!(registry.model_roots(), &[dir]);
    }

    /// **贯通判据**：注册面递下来的那份根 ⇒ ⇒ 必须**真的到达加载面** ⇒ ⇒
    /// 加载面报的错里要出现**那个目录下的真实路径**（而不是「找不到 / 试过」）。
    ///
    /// ⚠️ 两句话**形状不同** ⇒ ⇒ 可判：根没到 ⇒ 错误是「not found ... tried
    /// ...」；根到了 ⇒ 错误是「config.json not found in <那个目录>」。
    #[tokio::test]
    async fn asr_registry_load_path_reaches_the_config_model_dirs() {
        let root = std::env::temp_dir().join(format!("neuron-asr-wire-{}", uuid::Uuid::new_v4()));
        let model_dir = root.join("asr").join("wire-probe-model");
        std::fs::create_dir_all(&model_dir).unwrap();
        let config = Arc::new(NeuronConfig::new().with_model_dirs(vec![root.clone()]));
        let registry = Arc::new(AsrRuntimeRegistry::new(config));
        registry
            .register(
                "wire-probe-model".to_string(),
                AsrImpl::Candle,
                DeviceSelection::Cpu,
                1_000_000,
            )
            .expect("register");
        // ⚠️ `expect_err` 要求 `Ok` 侧实现 `Debug` ⇒ ⇒ `dyn AsrRuntime`
        // 不实现 ⇒ ⇒ 改成 match（判据不许逼着生产代码加 Debug 约束）。
        let err = match registry.runtime_for("wire-probe-model").await {
            Ok(_) => panic!("an empty model dir must not load"),
            Err(e) => e,
        };
        let msg = err.message.clone();
        assert!(
            msg.contains(&model_dir.display().to_string()),
            "the load path must reach the config's model dir; got: {msg}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }
}

