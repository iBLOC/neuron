//! Capability discriminator: Llm / Tts / Asr / Rerank / Embedding.
//!
//! Re-exported from `crate::models::registry::EngineType` (existing type).
//! The plugin architecture uses this as the dispatch key for both
//! adapters and engine registries.

pub use crate::models::registry::EngineType as Capability;

pub mod modality;

pub use modality::Modality;

/// 「LLM 能不能输入音频」的可判事实（**纯函数模块** ⇒ ⇒ 默认档也能编、也能判
/// ⇒ ⇒ 那条 llama.cpp 腿要 libclang ⇒ ⇒ 判据写在它里面就永远跑不到）。
pub mod audio_input;
pub use audio_input::AudioInputState;

/// **四种载荷**（文本／图片／音频／视频）的规范名与**协议层**口径
/// （**单一真相源** ⇒ ⇒ 别处不许各写各的载荷清单）。
pub mod payloads;
pub use payloads::{PayloadSupport, PAYLOADS};

pub mod api_style;
pub use api_style::ApiStyle;

pub mod model_caps;
pub use model_caps::ModelCapabilities;

pub mod catalog;
pub use catalog::CapabilityCatalog;

#[derive(Debug, Clone, Default)]
pub struct AdapterCapabilities {
    pub supports_streaming: bool,
    pub supports_tools: bool,
    pub supports_vision: bool,
    pub supports_audio: bool,
    pub supports_video: bool,
    pub supports_prompt_caching: bool,
    pub supports_extended_thinking: bool,
}