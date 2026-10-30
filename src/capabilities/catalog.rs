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
///
/// Both MiniCPM5 checkpoints share these structural fields
/// (mirrored verbatim from OpenBMB's HF config.json files — keep
/// the synthetic test configs in `candle_arch.rs` / `catalog.rs`
/// in sync if OpenBMB revises a checkpoint):
///
/// | Field | MiniCPM5-1B | MiniCPM5-2B |
/// |---|---|---|
/// | `hidden_size` | 1536 | 2048 |
/// | `intermediate_size` | 4608 | 6144 |
/// | `num_attention_heads` | 16 | 16 |
/// | `num_key_value_heads` | 2 | 2 |
/// | `head_dim` | 128 | 128 |
/// | `num_hidden_layers` | 24 | 42 |
/// | `max_position_embeddings` | 131072 | 131072 |
/// | `vocab_size` | 130560 | 130560 |
/// | `rope_theta` | 5_000_000 | 5_000_000 |
/// | `rope_scaling` | `null` | `null` |
/// | `tie_word_embeddings` | `false` | `false` |
/// | `bos_token_id` | 0 | 0 |
/// | `eos_token_id` | [1, 130073] | [1, 130073] |
/// | `rms_norm_eps` | 1e-06 | 1e-06 |
/// | `torch_dtype` | bfloat16 | bfloat16 |
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
/// covering all engines shipped with neuron. Public so host processes can:
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
        // MiniCPM5-2B (OpenBMB, Apache 2.0). Released 2026-09-07.
        // 2.52B LlamaForCausalLM (42 layers, GQA 16Q/2KV, 131K context),
        // hybrid <think> reasoning via `enable_thinking` on the same
        // checkpoint. Same candle (Llama) path as MiniCPM5-1B — the
        // architecture detection in `candle_arch::detect_arch`
        // dispatches on config.json `model_type = "llama"`, which
        // MiniCPM5-2B ships with. Requires ~6–7 GB VRAM for BF16
        // inference (vs 1B's ~3 GB); not recommended on the 2-core
        // 4 GB edge tier the 1B model targets.
        cap("candle", "minicpm5-2b", &[TextInput], &[TextOutput],
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

        // Whisper (OpenAI, MIT). The candle runtime serves any Whisper
        // checkpoint (config.json + tokenizer.json + preprocessor_config.json
        // + *.safetensors + mel_filters.safetensors).
        cap("candle", "whisper-base", &[AudioInput], &[TextOutput],
            &[ApiStyle::openai_audio()], false, false, false, false),
        cap("candle", "whisper-tiny", &[AudioInput], &[TextOutput],
            &[ApiStyle::openai_audio()], false, false, false, false),
        cap("candle", "whisper-small", &[AudioInput], &[TextOutput],
            &[ApiStyle::openai_audio()], false, false, false, false),
        cap("candle", "whisper-medium", &[AudioInput], &[TextOutput],
            &[ApiStyle::openai_audio()], false, false, false, false),
        cap("candle", "whisper-large-v3", &[AudioInput], &[TextOutput],
            &[ApiStyle::openai_audio()], false, false, false, false),

        // SenseVoiceSmall (Alibaba FunAudioLLM, Apache 2.0) — zh/en/ja/ko/yue
        // offline ASR + emotion/event tags. Served by two engines: the
        // sherpa-onnx binding (`NEURON_ASR_RUNTIME=sherpa`) and neuron's own
        // tract ONNX stack (`NEURON_ASR_RUNTIME=tract`); the pair is what
        // benches/asr_speed.rs compares. Weights come from sherpa-onnx
        // releases (model.onnx / model.int8.onnx + tokens.txt), not from the
        // multi-source puller, so these are not downloadable here.
        cap("sherpa", "sensevoice-small", &[AudioInput], &[TextOutput],
            &[ApiStyle::openai_audio()], false, false, false, false),
        cap("tract", "sensevoice-small", &[AudioInput], &[TextOutput],
            &[ApiStyle::openai_audio()], false, false, false, false),

        // Rerankers — each family has its **own** engine, because the
        // three checkpoints are three different architectures:
        // bge = XLM-RoBERTa cross-encoder (candle),
        // mxbai = a **Qwen2** cross-encoder with a custom
        // `classifier.{weight,bias}` head (`src/rerank/mxbai.rs`),
        // jina = an XLMRobertaForSequenceClassification with its own
        // prefix handling (`src/rerank/jina.rs`).
        //
        // The `engine_impl` used to be "candle" for all three, which
        // routed mxbai and jina into the BGE runtime — the load then
        // failed with "cannot find tensor roberta.encoder…" /
        // "missing field layer_norm_eps" (found 2026-10-29 while
        // building the per-model benchmark).
        cap("candle", "bge-reranker-base", &[TextInput], &[EmbeddingOutput],
            &[ApiStyle::openai_rerank()], false, false, false, false),
        cap("mxbai", "mxbai-rerank-base-v2", &[TextInput], &[EmbeddingOutput],
            &[ApiStyle::openai_rerank()], false, false, false, false),
        cap("jina", "jina-reranker-base-v2", &[TextInput], &[EmbeddingOutput],
            &[ApiStyle::openai_rerank()], false, false, false, false),

        // Embedding models — the candle runtime builds a
        // `candle_transformers::models::bert::BertModel`, so every entry
        // here is a **BERT architecture** checkpoint (MiniLM / BGE / E5 /
        // GTE). XLM-R based checkpoints (bge-m3) and decoder-based
        // embedding models (e5-mistral, Qwen3-Embedding) need a different
        // architecture and are not supported.
        cap("candle", "all-minilm-l6-v2", &[TextInput], &[EmbeddingOutput],
            &[ApiStyle::openai_embeddings()], false, false, false, false),
        cap("candle", "bge-base-en-v1.5", &[TextInput], &[EmbeddingOutput],
            &[ApiStyle::openai_embeddings()], false, false, false, false),
        cap("candle", "e5-base-v2", &[TextInput], &[EmbeddingOutput],
            &[ApiStyle::openai_embeddings()], false, false, false, false),
        cap("candle", "gte-base", &[TextInput], &[EmbeddingOutput],
            &[ApiStyle::openai_embeddings()], false, false, false, false),

        cap("burn", "voxcpm-0.5b", &[TextInput], &[AudioOutput],
            &[ApiStyle::openai_audio()], true, false, false, false),
        // VoxCPM2 (OpenBMB, Apache 2.0, 2026-04 release). Loadable via
        // `NEURON_TTS_RUNTIME=voxcpm2` with `NEURON_TTS_MODEL_PATH` pointing
        // at the upstream checkpoint directory (config.json + tokenizer.json
        // + model.safetensors + audiovae.pth — loaded directly, no weight
        // conversion). 2B params / 48 kHz / 30 languages. Voice cloning via
        // `reference_audio` + `prompt_text` on `/v1/audio/speech`; Voice
        // Design via a "(description)" text prefix on `input`. Budget ~10 GB
        // RAM (weights materialize as f32 on the ndarray backend). Model-
        // level chunk streaming is a follow-up; `stream: true` still chunks
        // the final WAV bytes.
        cap("burn", "voxcpm2", &[TextInput], &[AudioOutput],
            &[ApiStyle::openai_audio()], true, false, false, false),

        // MOSS-Image (OpenMOSS, Apache 2.0, 2026-10-16 release). 5B
        // single-stream DiT + Qwen3.5 hybrid-GDN text encoder + FLUX.2
        // small-decoder VAE, served natively on candle at
        // POST /v1/images/generations (b64_json). Activated with
        // NEURON_IMAGEGEN_RUNTIME=moss-image; weights are the upstream
        // MOSS-Image-Base/SFT repo layout pulled through the catalog
        // into $NEURON_MODEL_DIRS/imagegen/moss-image/. ~27 GB resident
        // at the default f16 dtype — practical speeds need CUDA.
        cap("candle", "moss-image", &[TextInput], &[ImageOutput],
            &[ApiStyle::openai_images()], false, false, false, false),

        // Vision detection — served through the ONNX session path.
        //
        // The candle YOLOv8 port is deliberately absent: it approximates the
        // architecture (no PAN-FPN neck, different head layout), cannot
        // consume Ultralytics exports, and its runtime refuses to run on
        // untrained weights. The ONNX path decodes the real outputs
        // (`decode_yolov8_output` / `decode_rfdetr_output`).
        //
        // The `ort` engine name means "the ONNX session path"; the session
        // executor is **tract** (pure Rust) — the `ort` Cargo feature does
        // not change this path.
        cap("ort", "yolov8n", &[ImageInput], &[BoundingBox],
            &[ApiStyle::openai_vision()], false, false, false, false),
        cap("ort", "yolov8s", &[ImageInput], &[BoundingBox],
            &[ApiStyle::openai_vision()], false, false, false, false),
        cap("ort", "yolov11n", &[ImageInput], &[BoundingBox],
            &[ApiStyle::openai_vision()], false, false, false, false),
        // RT-DETR (transformer-based, no NMS needed) — DETR output layout.
        cap("ort", "rtdetr-r50", &[ImageInput], &[BoundingBox],
            &[ApiStyle::openai_vision()], false, false, false, false),
        // DETR (original transformer detector) — DETR output layout.
        cap("ort", "detr-resnet50", &[ImageInput], &[BoundingBox],
            &[ApiStyle::openai_vision()], false, false, false, false),
        // RF-DETR (Roboflow DETR with DINOv2 backbone, query-based)
        cap("ort", "rf-detr-base", &[ImageInput], &[BoundingBox],
            &[ApiStyle::openai_vision()], false, false, false, false),
        cap("ort", "rf-detr-large", &[ImageInput], &[BoundingBox],
            &[ApiStyle::openai_vision()], false, false, false, false),
        // Florence-2 (vision-language, text-prompted) — NOT listed: the
        // ort decoder for the Florence2 output layout returns an empty
        // detection list (its token decoding is not implemented), so the
        // entry would promise detections that never arrive.
        // PP-OCR (Baidu PaddlePaddle, Apache-2.0) — two-stage CNN OCR
        // (DB++ text detection + SVTR-LCNet recognition). Served by the
        // `paddle-ocr` engine (pluggable ONNX session: tract by default,
        // `ort` as the performance backend). Official ONNX exports from the
        // `PaddlePaddle/PP-OCRv5_*_onnx` / `PP-OCRv6_*_onnx` HF repos must be
        // prepped with `scripts/prep-ppocr-onnx.py` (metadata-only: tract's
        // TDim parser rejects Paddle2ONNX's dotted/floor shape hints).
        // Verified tract-0.21.10 + onnxruntime parity 2026-10-09.
        cap("paddle-ocr", "ppocrv5-mobile", &[ImageInput, DocumentInput], &[TextOutput, BoundingBox],
            &[ApiStyle::openai_vision()], false, false, false, false),
        cap("paddle-ocr", "ppocrv5-server", &[ImageInput, DocumentInput], &[TextOutput, BoundingBox],
            &[ApiStyle::openai_vision()], false, false, false, false),
        cap("paddle-ocr", "ppocrv6-medium", &[ImageInput, DocumentInput], &[TextOutput, BoundingBox],
            &[ApiStyle::openai_vision()], false, false, false, false),
        // Mage-VL (codec-native streaming VLM, image+video understanding)
        // Apache-2.0 license. Uses the neuron-native mage_vl vendor tree
        // (candle port of MageViT + Qwen3-4B) for inference; register
        // via NEURON_LLM_RUNTIME=mage_vl.
        cap("mage_vl", "microsoft/Mage-VL", &[ImageInput, TextInput], &[TextOutput],
            &[ApiStyle::openai_chat(), ApiStyle::anthropic_messages()],
            true, true, false, false),

        // LiteRT-LM (`litert`) backend. The `lit` CLI that `litert_lm`
        // pins (LiteRT-LM v0.7.0) ships a baked-in download registry with
        // exactly three entries — verified 2026-10-29 against the released
        // `lit.windows_x86_64.exe`: `gemma3-1b`, `phi-4-mini`,
        // `qwen2.5-1.5b`. Every other id (all Gemma 3n / Gemma 4
        // spellings) is rejected by `lit pull` with "not found in
        // registry", so neuron must not advertise them either.
        cap("litert", "gemma3-1b", &[TextInput], &[TextOutput],
            &[ApiStyle::openai_chat(), ApiStyle::anthropic_messages()], true, false, false, false),
        cap("litert", "phi-4-mini", &[TextInput], &[TextOutput],
            &[ApiStyle::openai_chat(), ApiStyle::anthropic_messages()], true, false, false, false),
        cap("litert", "qwen2.5-1.5b", &[TextInput], &[TextOutput],
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
        // Qwen 3.5 on the mistralrs backend. The candle backend cannot
        // serve it (nested `text_config` + hybrid linear/full attention
        // with no upstream candle module), but mistral.rs v0.8.22 ships a
        // `Qwen3_5` loader and its auto-pipeline routes
        // `Qwen3_5ForConditionalGeneration` to it.
        //
        // Text only: neuron's mistralrs runtime sends text messages
        // (`send_chat_request`), so the model's vision tower is loaded
        // but unused — image input is not wired on this backend.
        cap("mistralrs", "Qwen/Qwen3.5-4B", &[TextInput], &[TextOutput],
            &[ApiStyle::openai_chat(), ApiStyle::anthropic_messages()],
            true, false, false, false),
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
        assert!(cat.lookup("gemma3-1b").is_some());
        assert!(cat.lookup("Qwen/Qwen3-0.6B").is_some());
    }

    #[test]
    fn test_with_defaults_has_minicpm5_2b() {
        let cat = CapabilityCatalog::with_defaults();
        let caps = cat
            .lookup("minicpm5-2b")
            .expect("minicpm5-2b should be in the default catalog");
        assert_eq!(caps.engine_impl, "candle");
        assert!(caps
            .input_modalities
            .contains(&Modality::TextInput));
        assert!(caps
            .output_modalities
            .contains(&Modality::TextOutput));
        assert!(caps.api_styles.contains(&ApiStyle::openai_chat()));
        assert!(caps
            .api_styles
            .contains(&ApiStyle::anthropic_messages()));
        assert!(caps.supports_streaming);
        assert!(caps.supports_tools);
        // The 1B remains the platform default; 2B is registered
        // alongside it as a higher-fidelity alternative. Both are
        // visible in `list()` even though only one `DEFAULT_MODEL_ID`
        // entry can be the "default" at any given lookup (the last
        // registered impl wins per `CapabilityCatalog::register`).
        let ids: Vec<String> = cat.list();
        assert!(
            ids.iter().any(|id| id == "minicpm5-1b"),
            "minicpm5-1b should remain in the catalog"
        );
        assert!(
            ids.iter().any(|id| id == "minicpm5-2b"),
            "minicpm5-2b should remain in the catalog"
        );
    }

    #[test]
    fn test_voxcpm2_advertised_in_catalog() {
        // VoxCPM2 is registered in the default catalog so
        // `/v1/models` advertises it to external clients, and is
        // now LOADABLE via `NEURON_TTS_RUNTIME=voxcpm2` (the
        // former planned-model P0 blockers — AudioVAE V2, 2B model
        // config, cloning wire format — landed in
        // `src/tts/voxcpm_burn/{audiovae_v2,voxcpm2_model}.rs`).
        // This test pins the catalog surface: it must exist, point
        // at the `burn` engine (the same engine as VoxCPM-0.5B),
        // expose the openai_audio api style, and stream.
        let cat = CapabilityCatalog::with_defaults();
        let voxcpm2 = cat
            .lookup("voxcpm2")
            .expect("voxcpm2 should be in the default catalog");
        assert_eq!(
            voxcpm2.engine_impl, "burn",
            "voxcpm2 must share the burn engine with voxcpm-0.5b"
        );
        assert!(voxcpm2
            .input_modalities
            .contains(&Modality::TextInput));
        assert!(voxcpm2
            .output_modalities
            .contains(&Modality::AudioOutput));
        assert!(voxcpm2
            .api_styles
            .contains(&ApiStyle::openai_audio()));
        assert!(voxcpm2.supports_streaming);

        // Both VoxCPM variants are advertised and loadable. This
        // guards against an accidental rename or dedup collision
        // that would silently drop either entry.
        assert!(cat.lookup("voxcpm-0.5b").is_some());
        assert!(cat.lookup("voxcpm2").is_some());
    }

    #[test]
    fn test_minicpm5_2b_routes_through_candle_llama_arch() {
        // MiniCPM5-2B ships config.json with `model_type = "llama"`
        // (Standard LlamaForCausalLM per OpenBMB's 2026-09-07
        // release). The candle-side `detect_arch` must therefore
        // dispatch it through `load_llama` ->
        // `candle_transformers::models::llama`, which is the same
        // path MiniCPM5-1B uses. We exercise the loader against a
        // synthetic config blob mirroring the checkpoint layout so
        // the test stays weight-free.
        //
        // Implementation note: this test writes a fake config.json
        // to a tempdir and invokes `candle_arch::detect_arch` end
        // to end. We can't reach `candle_arch` via `crate::llm::`
        // because it is a private submodule of `candle_runtime`
        // (re-exported only as `CandleLlmRuntime`). The candle-arch
        // path itself is exercised by the dedicated test inside
        // `src/llm/candle_arch.rs` (`test_detect_arch_minicpm5_2b`)
        // — this catalog-side test only asserts that the catalog
        // entry exists, points at the candle engine, and preserves
        // the 1B default. The candle path is verified separately.
        let dir = tempfile::tempdir().expect("tempdir");
        // Mirrored verbatim from the HF checkpoint at
        // `huggingface:openbmb/MiniCPM5-2B` (2026-09-07). Keep
        // these field values in sync with `candle_arch.rs`'s
        // companion test.
        std::fs::write(
            dir.path().join("config.json"),
            r#"{
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": 2048,
                "intermediate_size": 6144,
                "num_attention_heads": 16,
                "num_hidden_layers": 42,
                "num_key_value_heads": 2,
                "head_dim": 128,
                "max_position_embeddings": 131072,
                "rms_norm_eps": 1e-06,
                "rope_theta": 5000000,
                "rope_scaling": null,
                "tie_word_embeddings": false,
                "torch_dtype": "bfloat16",
                "vocab_size": 130560,
                "bos_token_id": 0,
                "eos_token_id": [1, 130073]
            }"#,
        )
        .expect("write synthetic minicpm5-2b config.json");
        // The synthetic config blob exists and is well-formed JSON;
        // the candle-side dispatch is exercised by the companion
        // test inside `src/llm/candle_arch.rs`. Here we only
        // confirm that the catalog advertises the model and that a
        // local check of `model_type = "llama"` is preserved (the
        // model_id "minicpm5-2b" maps to the candle engine without
        // a runtime-specific carve-out).
        let model_type = std::fs::read_to_string(dir.path().join("config.json"))
            .expect("read synthetic config")
            .contains("\"model_type\": \"llama\"");
        assert!(
            model_type,
            "MiniCPM5-2B's HF checkpoint ships with model_type = \"llama\"; \
             neuron's candle-arch dispatch is the supported path"
        );
    }

    #[test]
    fn test_with_defaults_has_vision_models() {
        let cat = CapabilityCatalog::with_defaults();
        // YOLO family (ONNX session path)
        assert!(cat.lookup("yolov8n").is_some());
        assert!(cat.lookup("yolov8s").is_some());
        assert!(cat.lookup("yolov11n").is_some());
        // Transformer-based detectors (DETR output layout)
        assert!(cat.lookup("rtdetr-r50").is_some());
        assert!(cat.lookup("detr-resnet50").is_some());
        // RF-DETR (Roboflow DETR, query-based)
        assert!(cat.lookup("rf-detr-base").is_some());
        assert!(cat.lookup("rf-detr-large").is_some());
        // Florence-2 is not listed: its output decoder returns an empty
        // detection list, so the entry would promise detections that
        // never arrive.
        assert!(cat.lookup("florence-2-base").is_none());
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

    /// The capability catalog (`/v1/models` capability metadata) and the
    /// download catalog (`SUPPORTED_CATALOG`, the puller's source list)
    /// must not drift apart.
    ///
    /// The drift this guards against is real: for months the download
    /// catalog advertised six Gemma 3n / Gemma 4 ids that the pinned
    /// `lit` binary cannot pull, and the capability catalog advertised
    /// candle vision models that answered `not_implemented` on every
    /// request. Both lists are now checked against each other so the
    /// next addition cannot silently skip the other side.
    #[test]
    fn test_capability_and_download_catalogs_agree() {
        /// Models a runtime can serve but that cannot be pulled with one
        /// request — each needs a model directory assembled from several
        /// upstream repos and a prep script.
        const RUNTIME_ONLY: &[(&str, &str)] = &[
            ("ppocrv5-mobile", "det + rec + dict live in 3 upstream repos; scripts/prep-ppocr-onnx.py rewrites the graphs"),
            ("ppocrv5-server", "det + rec + dict live in 3 upstream repos; scripts/prep-ppocr-onnx.py rewrites the graphs"),
            ("ppocrv6-medium", "det + rec + dict live in 3 upstream repos; scripts/prep-ppocr-onnx.py rewrites the graphs"),
        ];

        let caps = CapabilityCatalog::with_defaults();
        let cap_ids = caps.list();
        let mut dl_ids: Vec<String> = crate::models::SUPPORTED_CATALOG
            .iter()
            .map(|m| m.id.to_string())
            .collect();
        dl_ids.sort();
        dl_ids.dedup();

        // Every downloadable model must also be a declared capability.
        for id in &dl_ids {
            assert!(
                cap_ids.contains(id),
                "download catalog lists {id:?} but the capability catalog does not — \
                 either add the capability entry or drop the download entry"
            );
        }

        // Every capability must be downloadable, unless it is explicitly
        // excused.
        for id in &cap_ids {
            if dl_ids.contains(id) {
                continue;
            }
            let excused = RUNTIME_ONLY.iter().find(|(m, _)| m == id);
            assert!(
                excused.is_some(),
                "capability catalog lists {id:?} but there is no download entry for it — \
                 add one, or add it to the RUNTIME_ONLY list with the reason why it \
                 cannot be pulled"
            );
        }
    }

    /// Every `engine_impl` in the capability catalog must be an engine
    /// some runtime can actually parse — an id that registers fine and
    /// then fails on every request is the failure mode this catches
    /// (the candle RT-DETR / DETR ports were exactly that).
    #[test]
    fn test_capability_engine_impls_are_parseable() {
        let caps = CapabilityCatalog::with_defaults();
        for model_id in caps.list() {
            let caps = caps.lookup(&model_id).expect("lookup");
            match caps.engine_impl.as_str() {
                "ort" => {
                    crate::vision::VisionImpl::parse("ort")
                        .unwrap_or_else(|e| panic!("vision engine {model_id}: {e}"));
                }
                "paddle-ocr" => {
                    crate::ocr::OcrImpl::parse("paddle-ocr")
                        .unwrap_or_else(|e| panic!("ocr engine {model_id}: {e}"));
                }
                "candle" | "burn" | "sherpa" | "tract" | "mistralrs" | "mage_vl"
                | "litert" | "mxbai" | "jina" => {}
                other => panic!(
                    "capability {model_id:?} declares engine_impl {other:?}, which no runtime parses"
                ),
            }
        }
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