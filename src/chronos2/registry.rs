use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tokio::sync::Mutex as AsyncMutex;

use super::{Chronos2Error, Chronos2Runtime, RegisteredChronos2};
use crate::config::{DeviceSelection, ResourceTracker};
use crate::llm::NeuronError;

/// ⚠ **`registered` 与 `resources` 都住在 `Arc<Mutex<_>>` 里** ——
/// `ServerState` 会 clone 出一份放进动态注册锁（`chronos2_registry_lock`），
/// 处理器与端脑读的是另一份（`chronos2_registry`）。两份**必须看到同一张表**：
/// clone 一旦深拷贝 `registered` ⇒ ⇒ 运行时注册的模型落进锁里那份，
/// 处理器与端脑那格永远看不见（「注册上了，但谁也列不出它」）。
/// 与其余 capability 注册表同形（见 `src/imagegen/registry_sharing_test.rs`）。
pub struct Chronos2Registry {
    pub registered: Arc<Mutex<Vec<RegisteredChronos2>>>,
    loaded: Arc<AsyncMutex<HashMap<String, Arc<dyn Chronos2Runtime>>>>,
    resources: Arc<Mutex<ResourceTracker>>,
}

impl Clone for Chronos2Registry {
    fn clone(&self) -> Self {
        Self {
            registered: self.registered.clone(),
            loaded: self.loaded.clone(),
            resources: self.resources.clone(),
        }
    }
}

impl Chronos2Registry {
    pub fn new() -> Self {
        Self {
            registered: Arc::new(Mutex::new(Vec::new())),
            loaded: Arc::new(AsyncMutex::new(HashMap::new())),
            resources: Arc::new(Mutex::new(ResourceTracker::default())),
        }
    }

    /// Register a chronos2 model. Records it as a stub by default
    /// so the registry is non-empty even before the candle port
    /// lands; the real backend registers through
    /// `register_candle_runtime` (TODO: next commit).
    pub fn register(
        &self,
        model_id: String,
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
        self.registered.lock().unwrap().push(RegisteredChronos2 {
            model_id,
            device,
            required_memory_bytes,
        });
        Ok(())
    }

    /// Registered chronos2 models, for `/v1/models`.
    ///
    /// Sourced from the registry rather than the `state.chronos2`
    /// shortcut: the shortcut is a lazy-load handle, and listing through
    /// it would pay for a model load just to answer a directory request.
    pub fn list_models(&self) -> Vec<crate::llm::ModelInfo> {
        self.registered
            .lock()
            .unwrap()
            .iter()
            .map(|r| crate::llm::ModelInfo::new(r.model_id.clone(), "chronos2"))
            .collect()
    }

    /// 已注册的模型 id（族标准形状 —— asr/tts/vision/... 同名同形）。
    ///
    /// ⚠ 端脑的 `registered.forecast` 那一格走这里 ⇒ ⇒
    /// 动态注册（锁里那份 registry）注册上去的 id **必须**出现在
    /// 这里 —— 那一格曾在 clone 深拷贝下分叉（见 struct 文档）。
    pub fn list_registered(&self) -> Vec<String> {
        self.registered
            .lock()
            .unwrap()
            .iter()
            .map(|r| r.model_id.clone())
            .collect()
    }

    /// Resolve a runtime for the **named** model.
    ///
    /// ⚠ **点名的那枚 id 决定跑哪一枚 checkpoint** ⇒ ⇒
    /// 点名一枚**没装**的 ⇒ ⇒ **报错**（点名 id ＋
    /// 说清为什么）⇒ ⇒ **绝不**拿别的 checkpoint
    /// 顶上（落 stub ＝「服务了一枚没人要的
    /// 权重」＝失败长得像成功）。
    ///
    /// ⚠ The old shape answered **every** miss with the **stub** runtime
    /// ⇒ ⇒ the HTTP layer said `501` with a canned *"chronos2 candle port
    /// is not yet implemented"* note ⇒ ⇒ that note named neither the model
    /// nor the reason ⇒ ⇒ a caller who mistyped an id could not tell
    /// "you asked for a checkpoint that is not installed" from "neuron cannot
    /// do time series at all". **A miss is now an error, not a stub.**
    pub async fn runtime_for(
        &self,
        model: &str,
    ) -> Result<Arc<dyn Chronos2Runtime>, Chronos2Error> {
        {
            let g = self.loaded.lock().await;
            if let Some(rt) = g.get(model) {
                return Ok(rt.clone());
            }
        }

        // ① 选 checkpoint —— **按点名的 id 选**；找不到
        //    ⇒ ⇒ 报错（带 id 带搜过的路径）。
        let path = crate::chronos2::resolve_model_path_with_reason(model).map_err(|reason| {
            Chronos2Error::ModelNotAvailable(format!(
                "chronos2 checkpoint unavailable for model '{model}': {reason}"
            ))
        })?;

        // ② 这个 build 里有没有 chronos2 后端？
        #[cfg(not(feature = "chronos2"))]
        {
            let _ = path;
            Err(Chronos2Error::NotImplemented(format!(
                "chronos2 backend is not compiled into this neuron build (rebuild with --features chronos2); model '{model}' was not served"
            )))
        }

        // ③ 加载 —— 加载失败也**报错**（点名 id ＋
        //    路径 ＋ 原因），**不落 stub**。
        #[cfg(feature = "chronos2")]
        {
            let loaded = tokio::task::spawn_blocking({
                let p = path.clone();
                let model = model.to_string();
                move || -> Result<Arc<dyn Chronos2Runtime>, String> {
                    let device = crate::config::NeuronConfig::default_device();
                    // ⚠ **Two checkpoint layouts exist here.** The released
                    // Amazon Chronos-Bolt layout (patch embeddings) is served by
                    // the upstream-compatible runtime; the vendored port only
                    // fits the synthetic fixtures it was written for — pointing
                    // it at a real checkpoint is what used to send every request
                    // to the stub.
                    if crate::chronos2::bolt::is_released_layout(&p) {
                        crate::chronos2::bolt::BoltRuntime::load(&model, &p, device)
                            .map(|rt| Arc::new(rt) as Arc<dyn Chronos2Runtime>)
                            .map_err(|e| e.to_string())
                    } else {
                        crate::chronos2::candle_runtime::CandleChronos2Runtime::load(&p, device)
                            .map(|rt| Arc::new(rt) as Arc<dyn Chronos2Runtime>)
                            .map_err(|e| e.to_string())
                    }
                }
            })
            .await;

            let rt: Arc<dyn Chronos2Runtime> = match loaded {
                // ⚠ The loader already hands back an
                // `Arc<dyn Chronos2Runtime>` ⇒ ⇒ wrapping it a second
                // time asks for `Arc<dyn Chronos2Runtime>: Chronos2Runtime`
                // ⇒ ⇒ which the trait does not (and should not) implement.
                Ok(Ok(rt)) => rt,
                Ok(Err(load_err)) => {
                    return Err(Chronos2Error::Runtime(format!(
                        "chronos2 checkpoint for model '{model}' at {} failed to load: {load_err}",
                        path.display()
                    )));
                }
                Err(join_err) => {
                    return Err(Chronos2Error::Runtime(format!(
                        "chronos2 checkpoint for model '{model}' at {} failed to load: the loader task failed: {join_err}",
                        path.display()
                    )));
                }
            };
            tracing::info!(model = %model, path = %path.display(), "loaded Chronos2 model");
            // ⚠ 只缓存**真的被加载过**的那一枚 ——
            // 一次失败不得把「失败」冻结成
            // 「永远失败」（旧形式把 stub 也缓存了）。
            self.loaded.lock().await.insert(model.to_string(), rt.clone());
            Ok(rt)
        }
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

    /// ⚠ **点名一枚没装的 checkpoint ⇒ ⇒ 报错，并且
    /// 报错里**有那枚 id**。
    ///
    /// ⚠ 旧形式这里是「落 stub → 预测失败→ 501」
    /// ⇒ ⇒ 现在它是「按点名的 id 报错」。
    #[tokio::test]
    async fn runtime_for_names_the_checkpoint_it_could_not_find() {
        let reg = Chronos2Registry::new();
        let asked = "definitely-not-an-installed-chronos2-checkpoint";
        let result = reg.runtime_for(asked).await;
        let err = match result {
            Ok(_) => panic!("a checkpoint that is not installed must not resolve to a runtime"),
            Err(e) => e,
        };
        let text = err.to_string();
        assert!(
            text.contains(asked),
            "the error must name the checkpoint that was asked for: {text}"
        );
        assert!(
            !text.contains("chronos-bolt-small"),
            "the error must not name another checkpoint: {text}"
        );
    }

    /// ⚠ 一次失败**不得把「失败」冻结成「永远
    /// 失败」** —— 两次查阅给出的必须是同一条
    /// 话（而不是「第一次报错、、「第二次
    /// 第一次那枚模型又能用了」」）。
    #[tokio::test]
    async fn runtime_for_reports_a_miss_the_same_way_every_time() {
        let reg = Chronos2Registry::new();
        let asked = "definitely-not-an-installed-chronos2-checkpoint";
        let first = reg.runtime_for(asked).await.err();
        let second = reg.runtime_for(asked).await.err();
        let first = first.expect("the first look-up must miss");
        let second = second.expect("the second look-up must miss");
        assert_eq!(first.to_string(), second.to_string());
    }
}
