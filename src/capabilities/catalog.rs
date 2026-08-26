use std::collections::HashMap;
use std::sync::RwLock;

use super::{ApiStyle, ModelCapabilities, Modality};

/// Canonical default model advertised by the platform. Exposed
/// at `/v1/models` as the first entry and tagged with `"is_default": true`
/// so external OpenAI/Anthropic-compatible clients can pick it as
/// their default when no model is specified.
///
/// MiniCPM5-1B (OpenBMB, Apache 2.0) is a 1B LlamaForCausalLM model
/// with built-in hybrid reasoning (<think>) and 131K context. It
/// reaches 1B-class open-source SOTA while fitting on 2-core 4 GB
/// edge devices via the candle (Llama) runtime.
pub const DEFAULT_MODEL_ID: &str = "minicpm5-1b";

pub struct CapabilityCatalog {
    by_model: RwLock<HashMap<String, ModelCapabilities>>,
}

impl Default for CapabilityCatalog {
    fn default() -> Self {
        Self::new()
    }
}

impl CapabilityCatalog {
    pub fn new() -> Self {
        Self { by_model: RwLock::new(HashMap::new()) }
    }

    pub fn with_defaults() -> Self {
        let cat = Self::new();
        for caps in default_capabilities() {
            cat.register(caps);
        }
        cat
    }

    pub fn register(&self, caps: ModelCapabilities) {
        let mut map = self.by_model.write().unwrap();
        map.insert(caps.model_id.clone(), caps);
    }

    pub fn lookup(&self, model_id: &str) -> Option<ModelCapabilities> {
        self.by_model.read().unwrap().get(model_id).cloned()
    }

    pub fn can_serve(&self, model_id: &str, style: &ApiStyle) -> bool {
        self.lookup(model_id)
            .map(|c| c.api_styles.contains(style))
            .unwrap_or(false)
    }

    pub fn list(&self) -> Vec<String> {
        self.by_model.read().unwrap().keys().cloned().collect()
    }
}

/// Return the default set of (engine_impl, model_id, capabilities) entries
/// covering all engines shipped with neure. Public so host processes can:
/// 1. Call `CapabilityCatalog::with_defaults()` for in-memory use
/// 2. Serialize the Vec<ModelCapabilities> to a config file for persistence
/// 3. On startup, read the config file and call `CapabilityCatalog::register(...)`
///    on each entry to rebuild the catalog
///
/// `CapabilityCatalog` itself wraps runtime state (`RwLock<HashMap>`) which
/// cannot be safely serialized/deserialized across threads, so the
/// serialization boundary is on `Vec<ModelCapabilities>` instead.
pub fn default_capabilities() -> Vec<ModelCapabilities> {
    use Modality::*;
    use Modality::BoundingBox;
    vec![
        cap("candle", "qwen2.5-0.5b", &[TextInput], &[TextOutput],
            &[ApiStyle::openai_chat()], true, true, false, false),
        cap("candle", "qwen3-0.6b", &[TextInput], &[TextOutput],
            &[ApiStyle::openai_chat()], true, true, false, false),
        // Platform default — MiniCPM5-1B (OpenBMB, Apache 2.0).
        // 1B LlamaForCausalLM fitting on 2-core 4 GB edge devices.
        cap("candle", DEFAULT_MODEL_ID, &[TextInput], &[TextOutput],
            &[ApiStyle::openai_chat(), ApiStyle::anthropic_messages()],
            true, true, false, false),
        cap("candle", "llama-3-8b", &[TextInput], &[TextOutput],
            &[ApiStyle::openai_chat()], true, true, false, false),
        cap("candle", "phi-3-mini", &[TextInput], &[TextOutput],
            &[ApiStyle::openai_chat()], true, true, false, false),
        cap("candle", "mistral-7b", &[TextInput], &[TextOutput],
            &[ApiStyle::openai_chat()], true, true, false, false),
        cap("candle", "chatglm-3-6b", &[TextInput], &[TextOutput],
            &[ApiStyle::openai_chat()], true, true, false, false),

        cap("candle", "whisper-base", &[AudioInput], &[TextOutput],
            &[ApiStyle::openai_audio()], false, false, false, false),

        cap("candle", "bge-reranker-base", &[TextInput], &[EmbeddingOutput],
            &[ApiStyle::openai_rerank()], false, false, false, false),
        cap("candle", "mxbai-rerank-base-v2", &[TextInput], &[EmbeddingOutput],
            &[ApiStyle::openai_rerank()], false, false, false, false),
        cap("candle", "jina-reranker-base-v2", &[TextInput], &[EmbeddingOutput],
            &[ApiStyle::openai_rerank()], false, false, false, false),

        cap("candle", "all-minilm-l6-v2", &[TextInput], &[EmbeddingOutput],
            &[ApiStyle::openai_embeddings()], false, false, false, false),

        cap("burn", "voxcpm-0.5b", &[TextInput], &[AudioOutput],
            &[ApiStyle::openai_audio()], true, false, false, false),

        cap("candle", "yolov8n", &[ImageInput], &[BoundingBox],
            &[ApiStyle::openai_vision()], false, false, false, false),
        cap("candle", "yolov8s", &[ImageInput], &[BoundingBox],
            &[ApiStyle::openai_vision()], false, false, false, false),
        cap("candle", "yolov11n", &[ImageInput], &[BoundingBox],
            &[ApiStyle::openai_vision()], false, false, false, false),
        // RT-DETR (transformer-based, no NMS needed)
        cap("candle", "rtdetr-r50", &[ImageInput], &[BoundingBox],
            &[ApiStyle::openai_vision()], false, false, false, false),
        // DETR (original transformer detector)
        cap("candle", "detr-resnet50", &[ImageInput], &[BoundingBox],
            &[ApiStyle::openai_vision()], false, false, false, false),
        // RF-DETR (Roboflow DETR with DINOv2 backbone, query-based)
        cap("ort", "rf-detr-base", &[ImageInput], &[BoundingBox],
            &[ApiStyle::openai_vision()], false, false, false, false),
        cap("ort", "rf-detr-large", &[ImageInput], &[BoundingBox],
            &[ApiStyle::openai_vision()], false, false, false, false),
        // Grounding DINO (open-vocabulary detection, text-prompted)
        cap("ultralytics", "grounding-dino-base", &[ImageInput], &[BoundingBox],
            &[ApiStyle::openai_vision()], false, false, false, false),
        // Florence-2 (vision-language, text-prompted)
        cap("ort", "florence-2-base", &[ImageInput], &[BoundingBox],
            &[ApiStyle::openai_vision()], false, false, false, false),
        // Mage-VL (codec-native streaming VLM, image+video understanding)
        // Apache-2.0 license. Uses the neure-native mage_vl vendor tree
        // (candle port of MageViT + Qwen3-4B) for inference; register
        // via NEURE_LLM_RUNTIME=mage_vl.
        cap("mage_vl", "microsoft/Mage-VL", &[ImageInput, TextInput], &[TextOutput],
            &[ApiStyle::openai_chat(), ApiStyle::anthropic_messages()],
            true, true, false, false),

        cap("litert", "gemma-3n-e2b-it", &[TextInput], &[TextOutput],
            &[ApiStyle::openai_chat(), ApiStyle::anthropic_messages()], true, false, false, false),
        cap("litert", "gemma-3n-e4b-it", &[TextInput], &[TextOutput],
            &[ApiStyle::openai_chat(), ApiStyle::anthropic_messages()], true, false, false, false),
        cap("litert", "gemma-3-12b-it", &[TextInput], &[TextOutput],
            &[ApiStyle::openai_chat(), ApiStyle::anthropic_messages()], true, false, false, false),
        // Platform default — Gemma 4 E4B (Google, 4B effective
        // params). Small enough for embedded / mobile / dev tier
        // while still scoring well on general chat and instruction
        // following. Source: https://www.modelscope.cn/models/google/gemma-4-E4B
        cap("litert", "gemma-4-e2b-it", &[TextInput], &[TextOutput],
            &[ApiStyle::openai_chat(), ApiStyle::anthropic_messages()], true, false, false, false),
        cap("litert", DEFAULT_MODEL_ID, &[TextInput], &[TextOutput],
            &[ApiStyle::openai_chat(), ApiStyle::anthropic_messages()],
            true, false, false, false),
        cap("litert", "gemma-4-12b-it", &[TextInput], &[TextOutput],
            &[ApiStyle::openai_chat(), ApiStyle::anthropic_messages()], true, false, false, false),

        ModelCapabilities {
            engine_impl: "mistralrs".into(),
            model_id: "Qwen/Qwen3-0.6B".into(),
            input_modalities: [TextInput, ToolInput].into_iter().collect(),
            output_modalities: [TextOutput].into_iter().collect(),
            api_styles: [ApiStyle::openai_chat(), ApiStyle::anthropic_messages()]
                .into_iter().collect(),
            supports_streaming: true,
            supports_tools: true,
            supports_prompt_caching: true,
            supports_extended_thinking: true,
            ..Default::default()
        },
    ]
}

#[allow(clippy::too_many_arguments)]
fn cap(
    engine_impl: &str,
    model_id: &str,
    input: &[Modality],
    output: &[Modality],
    styles: &[ApiStyle],
    streaming: bool,
    tools: bool,
    caching: bool,
    thinking: bool,
) -> ModelCapabilities {
    ModelCapabilities {
        engine_impl: engine_impl.into(),
        model_id: model_id.into(),
        input_modalities: input.iter().copied().collect(),
        output_modalities: output.iter().copied().collect(),
        api_styles: styles.iter().cloned().collect(),
        supports_streaming: streaming,
        supports_tools: tools,
        supports_prompt_caching: caching,
        supports_extended_thinking: thinking,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_with_defaults_has_candle_llms() {
        let cat = CapabilityCatalog::with_defaults();
        assert!(cat.lookup("qwen3-0.6b").is_some());
        assert!(cat.lookup("whisper-base").is_some());
        assert!(cat.lookup("voxcpm-0.5b").is_some());
        assert!(cat.lookup("gemma-3n-e2b-it").is_some());
        assert!(cat.lookup("Qwen/Qwen3-0.6B").is_some());
    }

    #[test]
    fn test_with_defaults_has_vision_models() {
        let cat = CapabilityCatalog::with_defaults();
        // YOLO family
        assert!(cat.lookup("yolov8n").is_some());
        assert!(cat.lookup("yolov8s").is_some());
        assert!(cat.lookup("yolov11n").is_some());
        // Transformer-based detectors
        assert!(cat.lookup("rtdetr-r50").is_some());
        assert!(cat.lookup("detr-resnet50").is_some());
        // RF-DETR (Roboflow DETR, query-based)
        assert!(cat.lookup("rf-detr-base").is_some());
        assert!(cat.lookup("rf-detr-large").is_some());
        // Open-vocabulary / vision-language
        assert!(cat.lookup("grounding-dino-base").is_some());
        assert!(cat.lookup("florence-2-base").is_some());
    }

    #[test]
    fn test_vision_models_have_correct_modalities() {
        let cat = CapabilityCatalog::with_defaults();
        // YOLO models take images and produce bounding boxes
        let yolov8n = cat.lookup("yolov8n").unwrap();
        assert!(yolov8n.input_modalities.contains(&Modality::ImageInput));
        assert!(yolov8n.output_modalities.contains(&Modality::BoundingBox));
        assert!(yolov8n.api_styles.contains(&ApiStyle::openai_vision()));
    }

    #[test]
    fn test_can_serve_openai_chat() {
        let cat = CapabilityCatalog::with_defaults();
        assert!(cat.can_serve("qwen3-0.6b", &ApiStyle::openai_chat()));
        assert!(!cat.can_serve("whisper-base", &ApiStyle::openai_chat()));
        assert!(cat.can_serve("whisper-base", &ApiStyle::openai_audio()));
    }

    #[test]
    fn test_can_serve_anthropic() {
        let cat = CapabilityCatalog::with_defaults();
        assert!(cat.can_serve("Qwen/Qwen3-0.6B", &ApiStyle::anthropic_messages()));
        assert!(!cat.can_serve("qwen3-0.6b", &ApiStyle::anthropic_messages()));
    }

    #[test]
    fn test_register_override() {
        let cat = CapabilityCatalog::new();
        cat.register(ModelCapabilities {
            model_id: "test-model".into(),
            engine_impl: "host".into(),
            api_styles: [ApiStyle::openai_chat()].into_iter().collect(),
            ..Default::default()
        });
        cat.register(ModelCapabilities {
            model_id: "test-model".into(),
            engine_impl: "host-v2".into(),
            api_styles: [ApiStyle::anthropic_messages()].into_iter().collect(),
            ..Default::default()
        });
        let lookup = cat.lookup("test-model").unwrap();
        assert_eq!(lookup.engine_impl, "host-v2");
    }

    #[test]
    fn test_list_returns_all_model_ids() {
        let cat = CapabilityCatalog::with_defaults();
        let models = cat.list();
        assert!(models.contains(&"qwen3-0.6b".to_string()));
        assert!(models.len() > 5);
    }
}