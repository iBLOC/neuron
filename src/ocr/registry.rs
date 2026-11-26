use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use super::{
    OcrImpl, OcrRegistryKey, OcrRequest, OcrResponse, OcrRuntime, RegisteredOcr,
};
use crate::config::{DeviceSelection, NeuronConfig, ResourceTracker};
use crate::llm::{ChatResult, ModelInfo, NeuronError};

pub struct OcrRuntimeRegistry {
    pub registered: Arc<Mutex<Vec<RegisteredOcr>>>,
    loaded: Arc<tokio::sync::Mutex<HashMap<OcrRegistryKey, Arc<dyn OcrRuntime>>>>,
    resources: Arc<Mutex<ResourceTracker>>,
}

impl Clone for OcrRuntimeRegistry {
    fn clone(&self) -> Self {
        Self {
            registered: self.registered.clone(),
            loaded: self.loaded.clone(),
            resources: self.resources.clone(),
        }
    }
}

impl OcrRuntimeRegistry {
    pub fn new(_config: Arc<NeuronConfig>) -> Self {
        Self {
            registered: Arc::new(Mutex::new(Vec::new())),
            loaded: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            resources: Arc::new(Mutex::new(ResourceTracker::default())),
        }
    }

    pub fn register(
        &self,
        model_id: String,
        impl_id: OcrImpl,
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

        self.registered.lock().unwrap().push(RegisteredOcr {
            model_id,
            impl_id,
            device,
            required_memory_bytes,
        });
        Ok(())
    }

    pub async fn runtime_for(&self, model_id: &str) -> Result<Arc<dyn OcrRuntime>, NeuronError> {
        let entry = {
            let registered = self.registered.lock().unwrap();
            registered
                .iter()
                .find(|r| r.model_id == model_id)
                .cloned()
                .ok_or_else(|| NeuronError::new(format!("ModelNotRegistered: {model_id}")))?
        };

        let key = OcrRegistryKey {
            model_id: entry.model_id.clone(),
            impl_id: entry.impl_id,
            device: entry.device,
        };

        let mut loaded = self.loaded.lock().await;
        if let Some(rt) = loaded.get(&key) {
            return Ok(rt.clone());
        }

        // Lazy load happens outside the cache map; the insert below fires
        // only on success so a failed load is retried on the next request
        // instead of poisoning the cache.
        let rt: Arc<dyn OcrRuntime> = match entry.impl_id {
            #[cfg(any(feature = "tract", feature = "ort"))]
            OcrImpl::PaddleOcr => {
                let loaded_box: Box<dyn OcrRuntime> =
                    super::PaddleOcrRuntime::load(&entry.model_id, &entry.device)
                        .await
                        .map_err(|e| NeuronError::new(format!("PaddleOcrRuntime load failed: {e}")))?;
                Arc::from(loaded_box)
            }
            #[cfg(not(any(feature = "tract", feature = "ort")))]
            OcrImpl::PaddleOcr => {
                return Err(NeuronError::not_implemented(
                    "PaddleOcr pipeline requires the `tract` or `ort` feature (rebuild with \
                     --features tract); the graphs must also be prepped with \
                     scripts/prep-ppocr-onnx.py",
                ));
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

    /// Registered entries (not loaded instances), so a catalog listing never
    /// pays for a pipeline load. Same shape as the decision registry's.
    pub fn list_models(&self) -> Vec<ModelInfo> {
        self.registered
            .lock()
            .unwrap()
            .iter()
            .map(|r| ModelInfo::new(r.model_id.clone(), "ocr"))
            .collect()
    }

    /// Placeholder for the shortcut field. Never loads: blocking on
    /// `runtime_for` here panics with "Cannot start a runtime from within a
    /// runtime" whenever `ServerState::new` runs inside a tokio runtime
    /// (every `#[tokio::main]` host). The lazy load happens on
    /// `runtime_for`'s async path, which the HTTP handlers call; the
    /// shortcut only serves unregistered-model fallbacks. Same shape as the
    /// decision registry's `default_runtime`.
    pub fn default_runtime(&self) -> Arc<dyn OcrRuntime> {
        Arc::new(UnavailableOcrRuntime)
    }

    /// **注册表里第一枚** —— 也就是「默认的那一枚」。
    ///
    /// ⚠️ **为什么需要这条路**（2026-11-25 真机读数）：
    /// `/v1/models` **只列上游目录条目**（真机：41 条，asr／embedding／
    /// imagegen／llm／rerank／tts／vision）⇒ ⇒ **运行时注册的 OCR 模型
    /// 一个都不在里面** ⇒ ⇒ 客户端**学不到** `model_id` ⇒ ⇒ 客户端只能
    /// 打 `model: ''` ⇒ ⇒ 端脑得接住这一脚 ⇒ ⇒ 否则**探针式**的 OCR
    /// 那一支**结构性不可达**（永远 501）。
    ///
    /// ⚠️ **为什么能在这里加载**（而 `default_runtime()` 不能）：
    /// 调用点在 **HTTP 处理器**里（**不在** `ServerState::new` 里）⇒ ⇒
    /// 不会撞上「从一个 runtime 内部启动另一个 runtime」的 panic ⇒ ⇒
    /// 这一条是**异步**的，处理器 `await` 它。
    ///
    /// 一枚都没注册 ⇒ ⇒ `Err`（⇒ 调用方退到不可用桩 ⇒ ⇒ 501）。
    pub async fn runtime_for_default(&self) -> Result<Arc<dyn OcrRuntime>, NeuronError> {
        let first = {
            let registered = self.registered.lock().unwrap();
            registered.first().cloned()
        };
        let entry = first.ok_or_else(|| {
            NeuronError::new("NoRegisteredModel: register at least one OCR model")
        })?;
        self.runtime_for(&entry.model_id).await
    }
}

/// Fallback runtime returned when no OCR model is registered. Every call
/// returns `NeuronError::not_initialized`. Same pattern as the other
/// capabilities' `Unavailable*Runtime` stubs.
struct UnavailableOcrRuntime;

#[async_trait]
impl OcrRuntime for UnavailableOcrRuntime {
    async fn load(_: &str, _: &DeviceSelection) -> ChatResult<Box<dyn OcrRuntime>>
    where
        Self: Sized,
    {
        Err(NeuronError::not_initialized(
            "NoRegisteredModel: register at least one OCR model",
        ))
    }
    async fn run(&self, _req: OcrRequest) -> ChatResult<OcrResponse> {
        Err(NeuronError::not_initialized(
            "NoRegisteredModel: register at least one OCR model",
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
    use crate::vision::VisionImageSource;

    #[tokio::test]
    async fn test_runtime_for_unregistered_returns_error() {
        let config = Arc::new(NeuronConfig::new());
        let reg = OcrRuntimeRegistry::new(config);
        let result = reg.runtime_for("nonexistent").await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.message.contains("ModelNotRegistered"));
    }

    #[test]
    fn test_default_runtime_returns_unavailable_when_no_registered() {
        let config = Arc::new(NeuronConfig::new());
        let reg = OcrRuntimeRegistry::new(config);
        let rt = reg.default_runtime();
        assert_eq!(rt.name(), "unavailable");
    }

    #[tokio::test]
    async fn test_runtime_for_default_with_nothing_registered_says_so() {
        let config = Arc::new(NeuronConfig::new());
        let reg = OcrRuntimeRegistry::new(config);
        let err = reg.runtime_for_default().await.err().unwrap();
        assert!(err.message.contains("NoRegisteredModel"), "{err:?}");
    }

    #[tokio::test]
    async fn test_runtime_for_default_reaches_the_registered_model() {
        let config = Arc::new(NeuronConfig::new());
        let mut reg = OcrRuntimeRegistry::new(config);
        reg.register(
            "ppocrv5-mobile".into(),
            OcrImpl::PaddleOcr,
            DeviceSelection::Cpu,
            64 * 1024 * 1024,
        )
        .expect("register within default resource budget");
        // ⚠️ **权重不在** ⇒ ⇒ 加载必然失败 ⇒ ⇒ 但**失败的那一步**是关键：
        // 它必须已经**越过「没注册」那一关** ⇒ ⇒ 错误里**不许**再出现
        // NoRegisteredModel ⇒ ⇒ 这才证明「默认那一枚」选对了。
        let err = reg.runtime_for_default().await.err().unwrap();
        assert!(
            !err.message.contains("NoRegisteredModel"),
            "default runtime must try the registered model, got: {err:?}"
        );
    }

    #[tokio::test]
    async fn test_registered_paddleocr_lazy_load_returns_not_implemented() {
        let config = Arc::new(NeuronConfig::new());
        let mut reg = OcrRuntimeRegistry::new(config);
        reg.register(
            "ppocrv5-mobile".into(),
            OcrImpl::PaddleOcr,
            DeviceSelection::Cpu,
            64 * 1024 * 1024,
        )
        .expect("register within default resource budget");
        assert_eq!(reg.list_registered(), vec!["ppocrv5-mobile".to_string()]);
        let err = reg.runtime_for("ppocrv5-mobile").await.err().unwrap();
        #[cfg(not(any(feature = "tract", feature = "ort")))]
        {
            // Scaffold contract: without any executor the load surfaces a
            // clear 501.
            assert_eq!(err.error_type, "not_implemented", "{err:?}");
        }
        #[cfg(any(feature = "tract", feature = "ort"))]
        {
            // With an executor wired, the load proceeds to model discovery
            // and fails on missing weights (nothing registered on disk in CI).
            assert_ne!(err.error_type, "not_implemented", "{err:?}");
        }
    }

    #[tokio::test]
    async fn test_unavailable_ocr_runtime_run_errors() {
        let rt = UnavailableOcrRuntime;
        let req = OcrRequest {
            model: "ppocrv5-mobile".into(),
            image: VisionImageSource::ImageUrl {
                image_url: crate::vision::VisionImageUrl { url: "x".into(), detail: None },
            },
            languages: None,
            det_thresh: None,
            det_box_thresh: None,
            det_unclip_ratio: None,
            det_max_side: None,
            rec_batch_size: None,
        };
        let result = rt.run(req).await;
        assert!(result.is_err());
        assert_eq!(result.err().unwrap().error_type, "not_initialized");
    }

    #[test]
    fn test_insufficient_resources_rejected() {
        let config = Arc::new(NeuronConfig::new());
        let mut reg = OcrRuntimeRegistry::new(config);
        let err = reg
            .register(
                "too-big".into(),
                OcrImpl::PaddleOcr,
                DeviceSelection::Cpu,
                u64::MAX,
            )
            .err()
            .expect("u64::MAX exceeds the default CPU budget");
        assert!(err.message.contains("InsufficientResources"), "{err:?}");
    }
}
