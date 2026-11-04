//! Engine trait surface. Phase 1 implements only the `LlmRuntime`
//! capability. TtsRuntime / AsrRuntime / etc. are added in future phases.

use async_trait::async_trait;
use futures_util::stream::BoxStream;

use crate::canonical::{
    CanonicalLlmRequest, CanonicalLlmResponse, CanonicalLlmStreamEvent,
};
use crate::capabilities::ModelCapabilities;
use crate::config::DeviceSelection;

#[async_trait]
pub trait LlmRuntime: Send + Sync {
    async fn execute(&self, req: CanonicalLlmRequest)
        -> crate::llm::ChatResult<CanonicalLlmResponse>;

    async fn execute_stream(&self, req: CanonicalLlmRequest)
        -> crate::llm::ChatResult<BoxStream<'static, CanonicalLlmStreamEvent>>;

    /// Multimodal execute — same as `execute` plus a list of
    /// decoded image/audio attachments. Default impl returns
    /// `not_implemented`; the llama.cpp backend (`LlmImpl::LlamaCpp`)
    /// overrides this when an mmproj GGUF was loaded.
    async fn execute_with_media(
        &self,
        _req: CanonicalLlmRequest,
        _media: Vec<crate::canonical::MediaAttachment>,
    ) -> crate::llm::ChatResult<CanonicalLlmResponse> {
        Err(crate::llm::NeuronError::not_implemented(
            "this LlmRuntime does not support media input; \
             the llamacpp backend with an mmproj GGUF does",
        ))
    }

    /// Multimodal execute-stream. Same default `not_implemented`
    /// fallback.
    async fn execute_with_media_stream(
        &self,
        _req: CanonicalLlmRequest,
        _media: Vec<crate::canonical::MediaAttachment>,
    ) -> crate::llm::ChatResult<BoxStream<'static, CanonicalLlmStreamEvent>> {
        Err(crate::llm::NeuronError::not_implemented(
            "this LlmRuntime does not support media input; \
             the llamacpp backend with an mmproj GGUF does",
        ))
    }

    fn capabilities(&self) -> &ModelCapabilities;
    fn name(&self) -> &str;

    /// 这个引擎是不是**远端**的（HTTP 转发到别的机器/服务）。
    ///
    /// ⚠️ **为什么要有这条**（真机 2026-11-04 实证）：远端引擎**不占本机
    /// 内存**（它只是一条 HTTP 转发）⇒ ⇒ 模型调度器（内存预算 / 闲置
    /// 回收）**不该管它** ⇒ ⇒ 而一旦它进了调度器的 resident 表，闲置
    /// `idle_evict_secs`（默认 120 秒）之后 reaper 会把它 `unregister`
    /// 掉 ⇒ ⇒ 「手机借 PC 的模型」这条链**每闲置 2 分钟就断一次**
    /// （实测：答完 9×9 闲置 150 秒，下一条立刻报
    /// 「no engine registered for model 'llm/qwen2.5-0.5b'」）。
    fn is_remote(&self) -> bool {
        false
    }
}

#[async_trait]
pub trait LoadableLlmRuntime: LlmRuntime {
    async fn load(model_id: &str, device: &DeviceSelection)
        -> crate::llm::ChatResult<Box<dyn LlmRuntime>>
    where Self: Sized;
}

pub trait AnyCapabilityEngine: LlmRuntime {}
impl<T: LlmRuntime + ?Sized> AnyCapabilityEngine for T {}

pub mod bridge;
pub mod privacy;
pub mod registry;
pub mod remote;