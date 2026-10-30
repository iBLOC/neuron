use std::collections::HashMap;
use std::sync::Arc;

use crate::models::registry::EngineType;
use crate::models::registry::Registry;
use crate::models::source::SourceRegistry;

/// Resolved source-override configuration, derived from
/// `NeuronConfig`. Consumed by `SourceRegistry::resolve_for` to
/// pick a source for a given (engine, model_id) at pull time.
#[derive(Debug, Clone, Default)]
pub struct CatalogConfig {
    pub default_source_id: String,
    pub per_engine_source: HashMap<EngineType, String>,
    pub per_model_source: HashMap<String, String>,
}

impl CatalogConfig {
    /// Build a `CatalogConfig` from the relevant fields of
    /// `NeuronConfig`. The source-endpoint map is intentionally
    /// not copied here — endpoints live on the `SourceRegistry`
    /// itself (set when sources are constructed).
    pub fn from_neuron_config(cfg: &crate::config::NeuronConfig) -> Self {
        Self {
            default_source_id: cfg.default_source_id.clone(),
            per_engine_source: cfg.per_engine_source.clone(),
            per_model_source: cfg.per_model_source.clone(),
        }
    }

    /// Resolve which source id a pull should use.
    /// Lookup order: per-model > per-engine > default.
    pub fn resolve_source_id(&self, engine: EngineType, model_id: &str) -> String {
        let key = format!("{}/{}", engine.as_str(), model_id);
        if let Some(id) = self.per_model_source.get(&key) {
            return id.clone();
        }
        if let Some(id) = self.per_engine_source.get(&engine) {
            return id.clone();
        }
        self.default_source_id.clone()
    }
}

/// A model family that neuron code can load and serve. Hardcoded
/// at build time and matched against the on-disk registry to
/// compute `available: bool` on the `/v1/models` response.
#[derive(Debug, Clone)]
pub struct SupportedModel {
    pub id: &'static str,
    pub engine: EngineType,
    pub engine_impl: &'static str,
    pub default_repo: &'static str,
    pub source_id: &'static str,
    pub families: &'static [&'static str],
}

/// The full list of models neuron currently knows how to serve.
/// Single static slice so it can be referenced from tests and
/// from the `/v1/models` handler without rebuilding.
pub const SUPPORTED_CATALOG: &[SupportedModel] = &[
    // HuggingFace models
    SupportedModel {
        id: "qwen2.5-0.5b",
        engine: EngineType::Llm,
        engine_impl: "candle",
        default_repo: "huggingface:Qwen/Qwen2.5-0.5B-Instruct",
        source_id: "huggingface",
        families: &["qwen2"],
    },
    // MOSS-Image (OpenMOSS text-to-image, Apache 2.0). The whole
    // pipeline — 5B DiT + Qwen3.5 text encoder + FLUX.2 VAE — lives
    // in one repo (~27 GB); both sources dedup into a single catalog
    // entry offering "download from HF / ModelScope".
    SupportedModel {
        id: "moss-image",
        engine: EngineType::ImageGen,
        engine_impl: "moss-image",
        default_repo: "huggingface:OpenMOSS-Team/MOSS-Image-Base",
        source_id: "huggingface",
        families: &["moss-image"],
    },
    SupportedModel {
        id: "moss-image",
        engine: EngineType::ImageGen,
        engine_impl: "moss-image",
        default_repo: "modelscope:OpenMOSS-Team/MOSS-Image-Base",
        source_id: "modelscope",
        families: &["moss-image"],
    },
    SupportedModel {
        id: "qwen3-0.6b",
        engine: EngineType::Llm,
        engine_impl: "candle",
        default_repo: "huggingface:Qwen/Qwen3-0.6B",
        source_id: "huggingface",
        families: &["qwen3"],
    },
    SupportedModel {
        id: "llama-3-8b",
        engine: EngineType::Llm,
        engine_impl: "candle",
        default_repo: "huggingface:meta-llama/Meta-Llama-3-8B",
        source_id: "huggingface",
        families: &["llama"],
    },
    SupportedModel {
        id: "phi-3-mini",
        engine: EngineType::Llm,
        engine_impl: "candle",
        default_repo: "huggingface:microsoft/Phi-3-mini-4k-instruct",
        source_id: "huggingface",
        families: &["phi3"],
    },
    // Whisper (OpenAI, MIT). The candle runtime reads config.json +
    // tokenizer.json + preprocessor_config.json + *.safetensors, and it
    // ALSO needs `mel_filters.safetensors` in the same directory — that
    // file lives in the `lmz/candle-whisper` space, not in the model
    // repos, so it is a manual extra step after the pull.
    SupportedModel {
        id: "whisper-base",
        engine: EngineType::Asr,
        engine_impl: "candle",
        default_repo: "huggingface:openai/whisper-base",
        source_id: "huggingface",
        families: &["whisper"],
    },
    SupportedModel {
        id: "whisper-tiny",
        engine: EngineType::Asr,
        engine_impl: "candle",
        default_repo: "huggingface:openai/whisper-tiny",
        source_id: "huggingface",
        families: &["whisper"],
    },
    SupportedModel {
        id: "whisper-small",
        engine: EngineType::Asr,
        engine_impl: "candle",
        default_repo: "huggingface:openai/whisper-small",
        source_id: "huggingface",
        families: &["whisper"],
    },
    SupportedModel {
        id: "whisper-medium",
        engine: EngineType::Asr,
        engine_impl: "candle",
        default_repo: "huggingface:openai/whisper-medium",
        source_id: "huggingface",
        families: &["whisper"],
    },
    SupportedModel {
        id: "whisper-large-v3",
        engine: EngineType::Asr,
        engine_impl: "candle",
        default_repo: "huggingface:openai/whisper-large-v3",
        source_id: "huggingface",
        families: &["whisper"],
    },
    SupportedModel {
        id: "bge-reranker-base",
        engine: EngineType::Rerank,
        engine_impl: "candle",
        default_repo: "huggingface:BAAI/bge-reranker-base",
        source_id: "huggingface",
        families: &["xlm-roberta"],
    },
    // mxbai-rerank-base-v2 (mixedbread, Apache-2.0) — cross-encoder
    // reranker, served by the `mxbai` runtime
    // (NEURON_RERANK_RUNTIME=mxbai).
    SupportedModel {
        id: "mxbai-rerank-base-v2",
        engine: EngineType::Rerank,
        engine_impl: "mxbai",
        default_repo: "huggingface:mixedbread-ai/mxbai-rerank-base-v2",
        source_id: "huggingface",
        families: &["bert", "mxbai"],
    },
    // jina-reranker-base-v2 (jinaai, CC-BY-NC-4.0) — XLMRoberta
    // cross-encoder, served by the `jina` runtime
    // (NEURON_RERANK_RUNTIME=jina). The repo layout is exactly what the
    // runtime expects: config.json + tokenizer.json + model.safetensors.
    SupportedModel {
        id: "jina-reranker-base-v2",
        engine: EngineType::Rerank,
        engine_impl: "jina",
        default_repo: "huggingface:jinaai/jina-reranker-v2-base-multilingual",
        source_id: "huggingface",
        families: &["xlm-roberta", "jina"],
    },
    // SenseVoiceSmall (Alibaba FunAudioLLM, Apache-2.0) — zh/en/ja/ko/yue
    // ASR with emotion + event tags. The repo below is the sherpa-onnx
    // packaging (model.int8.onnx + tokens.txt) that the `sherpa` runtime
    // reads directly. The `tract` runtime needs the prepped graph from
    // `scripts/prep-sensevoice-onnx.py` instead.
    SupportedModel {
        id: "sensevoice-small",
        engine: EngineType::Asr,
        engine_impl: "sherpa",
        default_repo: "huggingface:csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09",
        source_id: "huggingface",
        families: &["sensevoice"],
    },
    // VoxCPM2 (OpenBMB, Apache-2.0, 2026-04) — 2B / 48 kHz / 30 languages.
    // The upstream checkpoint is loaded directly (config.json +
    // tokenizer.json + model.safetensors + audiovae.pth); no conversion.
    SupportedModel {
        id: "voxcpm2",
        engine: EngineType::Tts,
        engine_impl: "burn",
        default_repo: "huggingface:openbmb/VoxCPM2",
        source_id: "huggingface",
        families: &["voxcpm2"],
    },
    // MiniCPM5-2B (OpenBMB, Apache-2.0) — the 2B sibling of the platform
    // default; same LlamaForCausalLM path, ~6-7 GB resident at BF16.
    SupportedModel {
        id: "minicpm5-2b",
        engine: EngineType::Llm,
        engine_impl: "candle",
        default_repo: "huggingface:openbmb/MiniCPM5-2B",
        source_id: "huggingface",
        families: &["llama", "minicpm5"],
    },
    // ChatGLM3-6B (THUDM) — the `chatglm` architecture family.
    SupportedModel {
        id: "chatglm-3-6b",
        engine: EngineType::Llm,
        engine_impl: "candle",
        default_repo: "huggingface:THUDM/chatglm3-6b",
        source_id: "huggingface",
        families: &["chatglm"],
    },
    // Mistral 7B Instruct v0.3 — gated on HuggingFace; set HF_TOKEN to
    // pull it.
    SupportedModel {
        id: "mistral-7b",
        engine: EngineType::Llm,
        engine_impl: "candle",
        default_repo: "huggingface:mistralai/Mistral-7B-Instruct-v0.3",
        source_id: "huggingface",
        families: &["mistral"],
    },
    // Mage-VL (Microsoft, Apache-2.0) — codec-native streaming VLM. The
    // repo carries the LLM shards + `streammind_gate.safetensors` that
    // the mage_vl runtime loads.
    SupportedModel {
        id: "microsoft/Mage-VL",
        engine: EngineType::Llm,
        engine_impl: "mage_vl",
        default_repo: "huggingface:microsoft/Mage-VL",
        source_id: "huggingface",
        families: &["mage-vl"],
    },
    SupportedModel {
        id: "voxcpm-0.5b",
        engine: EngineType::Tts,
        engine_impl: "burn",
        default_repo: "huggingface:openbmb/VoxCPM-0.5B",
        source_id: "huggingface",
        families: &["voxcpm"],
    },
    // Embedding models. The candle runtime builds a
    // `candle_transformers::models::bert::BertModel`, so any **BERT
    // architecture** checkpoint works (config.json + tokenizer.json +
    // model.safetensors). That covers the MiniLM / BGE / E5 / GTE
    // families below — but NOT the XLM-R based ones (bge-m3) or the
    // decoder-based embedding models (e5-mistral, Qwen3-Embedding).
    SupportedModel {
        id: "all-minilm-l6-v2",
        engine: EngineType::Embedding,
        engine_impl: "candle",
        default_repo: "huggingface:sentence-transformers/all-MiniLM-L6-v2",
        source_id: "huggingface",
        families: &["bert", "minilm"],
    },
    SupportedModel {
        id: "bge-base-en-v1.5",
        engine: EngineType::Embedding,
        engine_impl: "candle",
        default_repo: "huggingface:BAAI/bge-base-en-v1.5",
        source_id: "huggingface",
        families: &["bert", "bge"],
    },
    SupportedModel {
        id: "e5-base-v2",
        engine: EngineType::Embedding,
        engine_impl: "candle",
        default_repo: "huggingface:intfloat/e5-base-v2",
        source_id: "huggingface",
        families: &["bert", "e5"],
    },
    SupportedModel {
        id: "gte-base",
        engine: EngineType::Embedding,
        engine_impl: "candle",
        default_repo: "huggingface:thenlper/gte-base",
        source_id: "huggingface",
        families: &["bert", "gte"],
    },
    // LiteRT-LM (`litert`) family.
    //
    // The `lit` CLI that `litert_lm` pins (LiteRT-LM v0.7.0) has a
    // baked-in download registry with exactly three entries, verified
    // 2026-10-29 against the released `lit.windows_x86_64.exe`:
    //   gemma3-1b  → litert-community/Gemma3-1B-IT
    //   phi-4-mini → litert-community/Phi-4-mini-instruct
    //   qwen2.5-1.5b → litert-community/Qwen2.5-1.5B-Instruct
    // `lit pull` answers "model '<id>' not found in registry" for
    // everything else — including all Gemma 3n / Gemma 4 spellings —
    // so those ids must not be advertised as downloadable.
    //
    // NOTE: the litert runtime pulls through `lit pull` (its own cache),
    // not through neuron's multi-source puller; these entries document
    // which ids the engine can serve and where the weights live.
    SupportedModel {
        id: "gemma3-1b",
        engine: EngineType::Llm,
        engine_impl: "litert",
        default_repo: "huggingface:litert-community/Gemma3-1B-IT",
        source_id: "huggingface",
        families: &["gemma3"],
    },
    SupportedModel {
        id: "phi-4-mini",
        engine: EngineType::Llm,
        engine_impl: "litert",
        default_repo: "huggingface:litert-community/Phi-4-mini-instruct",
        source_id: "huggingface",
        families: &["phi4"],
    },
    SupportedModel {
        id: "qwen2.5-1.5b",
        engine: EngineType::Llm,
        engine_impl: "litert",
        default_repo: "huggingface:litert-community/Qwen2.5-1.5B-Instruct",
        source_id: "huggingface",
        families: &["qwen2"],
    },
    // HF Mirror — China-friendly CDN that fronts the same HF repos
    // without the gating/access checks. Same wire format, so the
    // HF download logic works unchanged.
    SupportedModel {
        id: "gemma3-1b",
        engine: EngineType::Llm,
        engine_impl: "litert",
        default_repo: "hf-mirror:litert-community/Gemma3-1B-IT",
        source_id: "hf-mirror",
        families: &["gemma3"],
    },
    SupportedModel {
        id: "phi-4-mini",
        engine: EngineType::Llm,
        engine_impl: "litert",
        default_repo: "hf-mirror:litert-community/Phi-4-mini-instruct",
        source_id: "hf-mirror",
        families: &["phi4"],
    },
    SupportedModel {
        id: "qwen2.5-1.5b",
        engine: EngineType::Llm,
        engine_impl: "litert",
        default_repo: "hf-mirror:litert-community/Qwen2.5-1.5B-Instruct",
        source_id: "hf-mirror",
        families: &["qwen2"],
    },
    // ModelScope models
    SupportedModel {
        id: "qwen2.5-0.5b",
        engine: EngineType::Llm,
        engine_impl: "candle",
        default_repo: "modelscope:Qwen/Qwen2.5-0.5B-Instruct",
        source_id: "modelscope",
        families: &["qwen2"],
    },
    SupportedModel {
        id: "qwen3-0.6b",
        engine: EngineType::Llm,
        engine_impl: "candle",
        default_repo: "modelscope:Qwen/Qwen3-0.6B",
        source_id: "modelscope",
        families: &["qwen3"],
    },
    SupportedModel {
        id: "whisper-base",
        engine: EngineType::Asr,
        engine_impl: "candle",
        default_repo: "modelscope:openai-whisper/whisper-base",
        source_id: "modelscope",
        families: &["whisper"],
    },
    SupportedModel {
        id: "bge-reranker-base",
        engine: EngineType::Rerank,
        engine_impl: "candle",
        default_repo: "modelscope:BAAI/bge-reranker-base",
        source_id: "modelscope",
        families: &["xlm-roberta"],
    },
    // MiniCPM 5 1B — 1B-class SOTA from OpenBMB, LlamaForCausalLM
    // architecture, Apache 2.0. On-device / edge deployment friendly
    // with built-in hybrid reasoning (<think>). Platform default.
    // ModelScope: https://modelscope.cn/models/OpenBMB/MiniCPM5-1B
    SupportedModel {
        id: "minicpm5-1b",
        engine: EngineType::Llm,
        engine_impl: "candle",
        default_repo: "modelscope:OpenBMB/MiniCPM5-1B",
        source_id: "modelscope",
        families: &["llama", "minicpm5"],
    },
    // MiniCPM5-2B (OpenBMB, Apache-2.0) — ModelScope mirror.
    SupportedModel {
        id: "minicpm5-2b",
        engine: EngineType::Llm,
        engine_impl: "candle",
        default_repo: "modelscope:OpenBMB/MiniCPM5-2B",
        source_id: "modelscope",
        families: &["llama", "minicpm5"],
    },
    // VoxCPM2 (OpenBMB, Apache-2.0) — ModelScope mirror.
    SupportedModel {
        id: "voxcpm2",
        engine: EngineType::Tts,
        engine_impl: "burn",
        default_repo: "modelscope:OpenBMB/VoxCPM2",
        source_id: "modelscope",
        families: &["voxcpm2"],
    },
    // Vision models — served through the ONNX session path (`ort`).
    //
    // The vision runtime loads a *prepared* model directory
    // (`model.onnx` + `config.json`); no upstream repo ships that layout,
    // so a pull here only fetches the source checkpoint (the Ultralytics
    // `.pt` weights / the DETR `model.safetensors`). Converting it is a
    // manual step — see `docs/howto/vision-tasks.md` for the export
    // commands and the config.json fields the decoders read.
    //
    // The candle YOLOv8 port is deliberately NOT listed: it is an
    // approximation of the architecture (no PAN-FPN neck, different head
    // layout) and cannot consume Ultralytics exports, so it would only
    // ever fail at load time.
    SupportedModel {
        id: "yolov8n",
        engine: EngineType::Vision,
        engine_impl: "ort",
        default_repo: "huggingface:Ultralytics/YOLOv8:yolov8n.pt",
        source_id: "huggingface",
        families: &["yolo"],
    },
    SupportedModel {
        id: "yolov8s",
        engine: EngineType::Vision,
        engine_impl: "ort",
        default_repo: "huggingface:Ultralytics/YOLOv8:yolov8s.pt",
        source_id: "huggingface",
        families: &["yolo"],
    },
    SupportedModel {
        id: "yolov11n",
        engine: EngineType::Vision,
        engine_impl: "ort",
        default_repo: "huggingface:Ultralytics/YOLO11:yolo11n.pt",
        source_id: "huggingface",
        families: &["yolo"],
    },
    // RT-DETR (transformer-based, no NMS needed) — ONNX session path.
    SupportedModel {
        id: "rtdetr-r50",
        engine: EngineType::Vision,
        engine_impl: "ort",
        default_repo: "huggingface:PekingU/rtdetr_r50vd",
        source_id: "huggingface",
        families: &["rtdetr"],
    },
    // DETR (original transformer detector) — ONNX session path.
    SupportedModel {
        id: "detr-resnet50",
        engine: EngineType::Vision,
        engine_impl: "ort",
        default_repo: "huggingface:facebook/detr-resnet-50",
        source_id: "huggingface",
        families: &["detr"],
    },
    // RF-DETR (Roboflow DETR with DINOv2 backbone) — ONNX session path.
    SupportedModel {
        id: "rf-detr-base",
        engine: EngineType::Vision,
        engine_impl: "ort",
        default_repo: "huggingface:Roboflow/rf-detr-base",
        source_id: "huggingface",
        families: &["rf-detr"],
    },
    SupportedModel {
        id: "rf-detr-large",
        engine: EngineType::Vision,
        engine_impl: "ort",
        default_repo: "huggingface:Roboflow/rf-detr-large",
        source_id: "huggingface",
        families: &["rf-detr"],
    },
    // Florence-2 (vision-language, text-prompted) is intentionally absent:
    // the ort decoder for the Florence2 output layout returns an empty
    // detection list (its token decoding is not implemented), so the model
    // cannot actually serve detections today.
    // mistralrs runtime — high-throughput inference for Qwen3.
    SupportedModel {
        id: "Qwen/Qwen3-0.6B",
        engine: EngineType::Llm,
        engine_impl: "mistralrs",
        default_repo: "huggingface:Qwen/Qwen3-0.6B",
        source_id: "huggingface",
        families: &["qwen3"],
    },
    // Qwen 3.5 on the mistralrs runtime. The candle backend rejects
    // Qwen 3.5 (nested `text_config` + hybrid linear/full attention),
    // but mistral.rs v0.8.22 ships the Qwen3_5 loader. Text chat only —
    // neuron's mistralrs runtime does not send image attachments.
    SupportedModel {
        id: "Qwen/Qwen3.5-4B",
        engine: EngineType::Llm,
        engine_impl: "mistralrs",
        default_repo: "huggingface:Qwen/Qwen3.5-4B",
        source_id: "huggingface",
        families: &["qwen3_5"],
    },
];

/// One source for a `CatalogEntry`. A single model can be hosted on
/// multiple repositories (HuggingFace + ModelScope + ...); each
/// becomes one `SourceRef` inside the `sources` array.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SourceRef {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    /// `huggingface:Qwen/Qwen3-0.6B` — passed verbatim to the
    /// puller as `reference`.
    pub reference: String,
}

/// One row in the `/v1/models` response. Combines a hardcoded
/// `SupportedModel` with the on-disk state (if any) returned by
/// `Registry::get`. Multiple `SupportedModel` entries that share
/// the same `(engine, id)` are merged into a single
/// `CatalogEntry` whose `sources` lists every available repo.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CatalogEntry {
    pub id: String,
    pub engine: String,
    pub engine_impl: &'static str,
    pub capabilities: Vec<String>,
    pub available: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_count: Option<u32>,
    /// All repositories on which this model is offered. Empty
    /// when the model is only on disk (no catalog match).
    pub sources: Vec<SourceRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_repo: Option<String>,
    /// Convenience: first source id (kept for backwards-compatible
    /// filter behavior on `/v1/models?source=...`).
    pub source: String,
}

/// Glue between the hardcoded supported model list, a
/// multi-directory on-disk registry, and a source registry. The
/// `Catalog` is owned by `ServerState` and consulted by the
/// `/v1/models` handler and the source-resolver.
pub struct Catalog {
    registry: Registry,
    sources: Arc<SourceRegistry>,
    config: CatalogConfig,
    supported: &'static [SupportedModel],
}

impl Catalog {
    pub fn new(registry: Registry, sources: Arc<SourceRegistry>, config: CatalogConfig) -> Self {
        Self {
            registry,
            sources,
            config,
            supported: SUPPORTED_CATALOG,
        }
    }

    pub fn registry(&self) -> &Registry {
        &self.registry
    }

    pub fn sources(&self) -> &SourceRegistry {
        &self.sources
    }

    pub fn config(&self) -> &CatalogConfig {
        &self.config
    }

    pub fn supported(&self) -> &[SupportedModel] {
        self.supported
    }

    /// Build the list of `CatalogEntry` for the `/v1/models` endpoint.
    /// Each entry pairs a `SupportedModel` with the on-disk state
    /// (if any) returned by `Registry::get`. Multiple
    /// `SupportedModel` entries with the same `(engine, id)` —
    /// i.e. the same model offered by different repositories —
    /// are merged into one `CatalogEntry` whose `sources` array
    /// carries one `SourceRef` per repo.
    pub fn entries(&self) -> Vec<CatalogEntry> {
        // Group SupportedModel rows by (engine, id) so that the same
        // model hosted on HuggingFace and ModelScope shows up as a
        // single CatalogEntry.
        let mut groups: std::collections::BTreeMap<(String, String), CatalogEntry> =
            std::collections::BTreeMap::new();

        for s in self.supported {
            let engine_str = s.engine.as_str().to_string();
            let id_str = s.id.to_string();
            let key = (engine_str.clone(), id_str.clone());
            let on_disk = self.registry.get(s.engine, s.id);
            let source = self.sources.get(s.source_id);

            let source_ref = SourceRef {
                id: s.source_id.to_string(),
                endpoint: source.and_then(|s| s.base_url().map(String::from)),
                reference: s.default_repo.to_string(),
            };

            let entry = groups.entry(key).or_insert_with(|| CatalogEntry {
                id: format!("{}/{}", engine_str, id_str),
                engine: engine_str.clone(),
                engine_impl: s.engine_impl,
                capabilities: vec![engine_str.clone()],
                available: on_disk.is_some(),
                path: on_disk
                    .as_ref()
                    .map(|m| m.path.to_string_lossy().to_string()),
                size_bytes: on_disk.as_ref().map(|m| m.size_bytes),
                file_count: on_disk.as_ref().map(|m| m.file_count),
                sources: Vec::new(),
                default_repo: None,
                source: String::new(),
            });

            entry.sources.push(source_ref);
            if entry.default_repo.is_none() {
                entry.default_repo = Some(s.default_repo.to_string());
            }
            if entry.source.is_empty() {
                entry.source = s.source_id.to_string();
            }
        }

        // Also surface on-disk models that have no catalog entry.
        let mut seen_ids: std::collections::HashSet<(String, String)> =
            groups.keys().cloned().collect();
        for m in self.registry.list() {
            let key = (m.engine.as_str().to_string(), m.id.clone());
            if !seen_ids.contains(&key) {
                groups.insert(
                    key.clone(),
                    CatalogEntry {
                        id: format!("{}/{}", m.engine.as_str(), m.id),
                        engine: m.engine.as_str().to_string(),
                        engine_impl: "",
                        capabilities: vec![m.engine.as_str().to_string()],
                        available: true,
                        path: Some(m.path.to_string_lossy().to_string()),
                        size_bytes: Some(m.size_bytes),
                        file_count: Some(m.file_count),
                        sources: Vec::new(),
                        default_repo: None,
                        source: String::new(),
                    },
                );
                seen_ids.insert(key);
            }
        }

        groups.into_values().collect()
    }

    /// Returns `(supported_count, available_count)`.
    pub fn counts(&self) -> (usize, usize) {
        let entries = self.entries();
        let supported = entries.len();
        let available = entries.iter().filter(|e| e.available).count();
        (supported, available)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_uses_default_when_no_override() {
        let cfg = CatalogConfig {
            default_source_id: "huggingface".to_string(),
            ..Default::default()
        };
        assert_eq!(
            cfg.resolve_source_id(EngineType::Llm, "anything"),
            "huggingface"
        );
    }

    #[test]
    fn test_resolve_per_engine_takes_priority_over_default() {
        let mut per_engine = HashMap::new();
        per_engine.insert(EngineType::Llm, "modelscope".to_string());
        let cfg = CatalogConfig {
            default_source_id: "huggingface".to_string(),
            per_engine_source: per_engine,
            ..Default::default()
        };
        assert_eq!(cfg.resolve_source_id(EngineType::Llm, "x"), "modelscope");
        assert_eq!(cfg.resolve_source_id(EngineType::Tts, "y"), "huggingface");
    }

    #[test]
    fn test_resolve_per_model_takes_priority_over_per_engine() {
        let mut per_engine = HashMap::new();
        per_engine.insert(EngineType::Llm, "modelscope".to_string());
        let mut per_model = HashMap::new();
        per_model.insert("llm/qwen2.5-0.5b".to_string(), "private-hub".to_string());
        let cfg = CatalogConfig {
            default_source_id: "huggingface".to_string(),
            per_engine_source: per_engine,
            per_model_source: per_model,
        };
        assert_eq!(
            cfg.resolve_source_id(EngineType::Llm, "qwen2.5-0.5b"),
            "private-hub"
        );
        assert_eq!(
            cfg.resolve_source_id(EngineType::Llm, "other-llm"),
            "modelscope"
        );
    }

    // -- Catalog --

    use crate::models::source::SourceRegistry;
    use std::collections::HashMap;
    use tempfile::TempDir;

    #[test]
    fn test_catalog_entries_lists_all_supported() {
        let tmp = TempDir::new().unwrap();
        let registry = Registry::new(tmp.path().to_path_buf());
        let sources = Arc::new(SourceRegistry::with_defaults());
        let cfg = CatalogConfig {
            default_source_id: "huggingface".to_string(),
            per_engine_source: HashMap::new(),
            per_model_source: HashMap::new(),
        };
        let catalog = Catalog::new(registry, sources, cfg);
        let entries = catalog.entries();
        // After dedup by (engine, id) the count matches unique pairs.
        let unique_count: std::collections::HashSet<_> =
            SUPPORTED_CATALOG.iter().map(|m| (m.engine, m.id)).collect();
        assert_eq!(entries.len(), unique_count.len());
        assert!(entries.iter().all(|e| !e.available));
    }

    #[test]
    fn test_catalog_marks_available_when_on_disk() {
        use std::fs;
        let tmp = TempDir::new().unwrap();
        let model_dir = tmp.path().join("llm").join("qwen2.5-0.5b");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("config.json"), b"{}").unwrap();

        let registry = Registry::new(tmp.path().to_path_buf());
        let sources = Arc::new(SourceRegistry::with_defaults());
        let cfg = CatalogConfig {
            default_source_id: "huggingface".to_string(),
            per_engine_source: HashMap::new(),
            per_model_source: HashMap::new(),
        };
        let catalog = Catalog::new(registry, sources, cfg);
        let (supported, available) = catalog.counts();
        assert!(supported >= 7);
        assert_eq!(available, 1);
    }

    #[test]
    fn test_supported_catalog_includes_4_engines() {
        use std::collections::HashSet;
        let engines: HashSet<EngineType> = SUPPORTED_CATALOG.iter().map(|m| m.engine).collect();
        assert!(engines.contains(&EngineType::Llm));
        assert!(engines.contains(&EngineType::Tts));
        assert!(engines.contains(&EngineType::Asr));
        assert!(engines.contains(&EngineType::Rerank));
    }

    #[test]
    fn test_catalog_entries_surfaces_source_endpoint_from_registry() {
        // Build a SourceRegistry with a custom HF endpoint, then
        // confirm every entry backed by `source_id = "huggingface"`
        // carries the endpoint on its `sources[*].endpoint` field.
        use std::collections::HashMap;
        let tmp = TempDir::new().unwrap();
        let registry = Registry::new(tmp.path().to_path_buf());
        let mut endpoints = HashMap::new();
        endpoints.insert(
            "huggingface".to_string(),
            "https://hf-mirror.com".to_string(),
        );
        let sources = Arc::new(SourceRegistry::with_endpoints(&endpoints));
        let cfg = CatalogConfig {
            default_source_id: "huggingface".to_string(),
            per_engine_source: HashMap::new(),
            per_model_source: HashMap::new(),
        };
        let catalog = Catalog::new(registry, sources, cfg);
        let entries = catalog.entries();
        assert!(!entries.is_empty());
        for e in &entries {
            // Walk each source on the entry; HF entries must carry
            // the configured endpoint on their SourceRef.
            let hf = e.sources.iter().find(|s| s.id == "huggingface");
            if let Some(s) = hf {
                assert_eq!(
                    s.endpoint.as_deref(),
                    Some("https://hf-mirror.com"),
                    "entry {} should carry the configured HF endpoint",
                    e.id
                );
            }
        }
    }

    #[test]
    fn test_catalog_entries_omits_source_endpoint_when_registry_has_none() {
        // Default SourceRegistry (no endpoint) should produce entries
        // whose source refs have `endpoint: None` — exercised because
        // the field is `skip_serializing_if = "Option::is_none"`, which
        // means a bug here would silently drop the field instead of
        // failing loudly.
        let tmp = TempDir::new().unwrap();
        let registry = Registry::new(tmp.path().to_path_buf());
        let sources = Arc::new(SourceRegistry::with_defaults());
        let cfg = CatalogConfig {
            default_source_id: "huggingface".to_string(),
            per_engine_source: HashMap::new(),
            per_model_source: HashMap::new(),
        };
        let catalog = Catalog::new(registry, sources, cfg);
        let entries = catalog.entries();
        for e in &entries {
            for s in &e.sources {
                // hf-mirror always carries a default endpoint
                // (https://hf-mirror.com) in the default registry.
                if s.id == "hf-mirror" {
                    assert!(
                        s.endpoint.is_some(),
                        "hf-mirror should have a default endpoint for {}",
                        e.id
                    );
                    continue;
                }
                assert!(
                    s.endpoint.is_none(),
                    "default registry should yield endpoint: None, got {:?} for {}",
                    s.endpoint,
                    e.id
                );
            }
        }
    }
}
