use std::collections::BTreeSet;
use serde::{Deserialize, Serialize};

use super::{ApiStyle, Modality};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct ModelCapabilities {
    pub engine_impl: String,
    pub model_id: String,
    pub input_modalities: BTreeSet<Modality>,
    pub output_modalities: BTreeSet<Modality>,
    pub api_styles: BTreeSet<ApiStyle>,
    pub supports_streaming: bool,
    pub supports_tools: bool,
    pub supports_prompt_caching: bool,
    pub supports_extended_thinking: bool,
    pub context_window: Option<u32>,
    pub max_output_tokens: Option<u32>,
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_is_empty() {
        let caps = ModelCapabilities::default();
        assert!(caps.input_modalities.is_empty());
        assert!(!caps.supports_streaming);
    }

    #[test]
    fn test_serde_round_trip() {
        let caps = ModelCapabilities {
            engine_impl: "candle".into(),
            model_id: "qwen3-0.6b".into(),
            input_modalities: [Modality::TextInput].into_iter().collect(),
            output_modalities: [Modality::TextOutput].into_iter().collect(),
            api_styles: [ApiStyle::openai_chat()].into_iter().collect(),
            supports_streaming: true,
            supports_tools: true,
            ..Default::default()
        };
        let json = serde_json::to_string(&caps).unwrap();
        let back: ModelCapabilities = serde_json::from_str(&json).unwrap();
        assert_eq!(back.engine_impl, "candle");
        assert_eq!(back.supports_streaming, true);
        assert_eq!(back.api_styles.len(), 1);
    }
}