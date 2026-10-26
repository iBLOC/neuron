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
pub mod registry;
pub mod remote;