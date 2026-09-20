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
    SupportedModel {
        id: "whisper-base",
        engine: EngineType::Asr,
        engine_impl: "candle",
        default_repo: "huggingface:openai/whisper-base",
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
    SupportedModel {
        id: "voxcpm-0.5b",
        engine: EngineType::Tts,
        engine_impl: "burn",
        default_repo: "huggingface:openbmb/VoxCPM-0.5B",
        source_id: "huggingface",
        families: &["voxcpm"],
    },
    SupportedModel {
        id: "all-minilm-l6-v2",
        engine: EngineType::Embedding,
        engine_impl: "candle",
        default_repo: "huggingface:sentence-transformers/all-MiniLM-L6-v2",
        source_id: "huggingface",
        families: &["bert", "minilm"],
    },
    // Google Gemma family.
    // - Gemma 4 (E2B/E4B/12B): Apache 2.0 — fully permissive.
    // - Gemma 3n / Gemma 3: gated on HuggingFace (requires license
    //   agreement in the user's HF account). Use `litert` runtime.
    SupportedModel {
        id: "gemma-4-e2b-it",
        engine: EngineType::Llm,
        engine_impl: "litert",
        default_repo: "huggingface:google/gemma-4-E2B-it",
        source_id: "huggingface",
        families: &["gemma4"],
    },
    SupportedModel {
        id: "gemma-4-e4b-it",
        engine: EngineType::Llm,
        engine_impl: "litert",
        default_repo: "huggingface:google/gemma-4-E4B-it",
        source_id: "huggingface",
        families: &["gemma4"],
    },
    SupportedModel {
        id: "gemma-4-12b-it",
        engine: EngineType::Llm,
        engine_impl: "litert",
        default_repo: "huggingface:google/gemma-4-12B-it",
        source_id: "huggingface",
        families: &["gemma4"],
    },
    // HF Mirror — China-friendly CDN that fronts the same HF repos
    // without the gating/access checks. Same wire format, so the
    // HF download logic works unchanged.
    SupportedModel {
        id: "gemma-4-e2b-it",
        engine: EngineType::Llm,
        engine_impl: "litert",
        default_repo: "hf-mirror:google/gemma-4-E2B-it",
        source_id: "hf-mirror",
        families: &["gemma4"],
    },
    SupportedModel {
        id: "gemma-4-e4b-it",
        engine: EngineType::Llm,
        engine_impl: "litert",
        default_repo: "hf-mirror:google/gemma-4-E4B-it",
        source_id: "hf-mirror",
        families: &["gemma4"],
    },
    SupportedModel {
        id: "gemma-4-12b-it",
        engine: EngineType::Llm,
        engine_impl: "litert",
        default_repo: "hf-mirror:google/gemma-4-12B-it",
        source_id: "hf-mirror",
        families: &["gemma4"],
    },
    SupportedModel {
        id: "gemma-3n-e2b-it",
        engine: EngineType::Llm,
        engine_impl: "litert",
        default_repo: "huggingface:google/gemma-3n-E2B-it",
        source_id: "huggingface",
        families: &["gemma3n"],
    },
    SupportedModel {
        id: "gemma-3n-e4b-it",
        engine: EngineType::Llm,
        engine_impl: "litert",
        default_repo: "huggingface:google/gemma-3n-E4B-it",
        source_id: "huggingface",
        families: &["gemma3n"],
    },
    SupportedModel {
        id: "gemma-3-12b-it",
        engine: EngineType::Llm,
        engine_impl: "litert",
        default_repo: "huggingface:google/gemma-3-12b-it",
        source_id: "huggingface",
        families: &["gemma3"],
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
    // Vision models — YOLO family via candle-yolo.
    SupportedModel {
        id: "yolov8n",
        engine: EngineType::Vision,
        engine_impl: "candle-yolo",
        default_repo: "huggingface:Ultralytics/YOLOv8:yolov8n.pt",
        source_id: "huggingface",
        families: &["yolo"],
    },
    SupportedModel {
        id: "yolov8s",
        engine: EngineType::Vision,
        engine_impl: "candle-yolo",
        default_repo: "huggingface:Ultralytics/YOLOv8:yolov8s.pt",
        source_id: "huggingface",
        families: &["yolo"],
    },
    SupportedModel {
        id: "yolov11n",
        engine: EngineType::Vision,
        engine_impl: "candle-yolo",
        default_repo: "huggingface:Ultralytics/YOLO11:yolo11n.pt",
        source_id: "huggingface",
        families: &["yolo"],
    },
    // RT-DETR (transformer-based, no NMS needed) — candle-rtdetr.
    SupportedModel {
        id: "rtdetr-r50",
        engine: EngineType::Vision,
        engine_impl: "candle-rtdetr",
        default_repo: "huggingface:PekingU/rtdetr_r50vd",
        source_id: "huggingface",
        families: &["rtdetr"],
    },
    // DETR (original transformer detector) — candle-detr.
    SupportedModel {
        id: "detr-resnet50",
        engine: EngineType::Vision,
        engine_impl: "candle-detr",
        default_repo: "huggingface:facebook/detr-resnet-50",
        source_id: "huggingface",
        families: &["detr"],
    },
    // RF-DETR (Roboflow DETR with DINOv2 backbone) — ONNX Runtime.
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
    // Grounding DINO (open-vocabulary, text-prompted) — ultralytics CLI.
    SupportedModel {
        id: "grounding-dino-base",
        engine: EngineType::Vision,
        engine_impl: "ultralytics",
        default_repo: "huggingface:IDEA-Research/grounding-dino-base",
        source_id: "huggingface",
        families: &["grounding-dino"],
    },
    // Florence-2 (vision-language, text-prompted) — ONNX Runtime.
    SupportedModel {
        id: "florence-2-base",
        engine: EngineType::Vision,
        engine_impl: "ort",
        default_repo: "huggingface:microsoft/Florence-2-base",
        source_id: "huggingface",
        families: &["florence"],
    },
    // mistralrs runtime — high-throughput inference for Qwen3.
    SupportedModel {
        id: "Qwen/Qwen3-0.6B",
        engine: EngineType::Llm,
        engine_impl: "mistralrs",
        default_repo: "huggingface:Qwen/Qwen3-0.6B",
        source_id: "huggingface",
        families: &["qwen3"],
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
