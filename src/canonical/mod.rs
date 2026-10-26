pub mod content;
pub mod sampling;
pub mod tool;
pub mod types;
pub mod usage;

pub use content::{
    AudioBlock, AudioSource, ContentBlock, DocumentBlock,
    ImageBlock, ImageSource, MediaAttachment, ReasoningBlock, TextBlock,
    ToolResultBlock, ToolUseBlock, VideoBlock, VideoSource,
};

/// The default media marker llama.cpp's `mtmd` module uses
/// (`<__media__>`). The OpenAI→canonical translator inserts this
/// string at each image block's position in the text stream so a
/// multimodal backend can split text+image tokens correctly.
///
/// This is `pub` (not just crate-private) so downstream hosts that
/// build their own `CanonicalLlmRequest` can produce a multimodal
/// prompt without re-importing llama-cpp-2.
pub fn mtmd_marker() -> &'static str {
    "<__media__>"
}
pub use sampling::{CacheControl, CacheType, SamplingParams};
pub use tool::{ToolCall, ToolDefinition};
pub use types::{
    CanonicalAsrRequest, CanonicalAsrResponse, CanonicalEmbeddingRequest,
    CanonicalEmbeddingResponse, CanonicalLlmRequest, CanonicalLlmResponse,
    CanonicalLlmStreamEvent, CanonicalRequest, CanonicalResponse,
    CanonicalRerankRequest, CanonicalRerankResponse, CanonicalStreamEvent,
    CanonicalTtsRequest, CanonicalTtsResponse, ContentDelta, MessageRole, SystemBlock,
};
pub use usage::{StopReason, UsageInfo};

pub use types::CanonicalMessage;