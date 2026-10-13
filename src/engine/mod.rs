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