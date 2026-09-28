//! Real Candle-backed Rerank runtime for Jina Reranker v2.
//!
//! Loads an `XLMRobertaForSequenceClassification` model (e.g. jinaai/jina-reranker-base-v2
//! or jinaai/jina-reranker-m0) and produces [0, 1]-normalized relevance scores via sigmoid.
//!
//! Jina's distinguishing feature is configurable query / document prefixes, read
//! from `config.json` (fields `query_prefix` and `document_prefix`, both optional).
//! When the field is absent the prefix is empty (same as BGE's behavior).
//!
//! Enable with: `NEURON_RERANK_RUNTIME=jina` and set `NEURON_RERANK_MODEL_PATH`
//! to a directory containing config.json, tokenizer.json, and model.safetensors.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::xlm_roberta::{
    Config as XLMRobertaConfig, XLMRobertaForSequenceClassification,
};
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::config::{ensure_dir, DeviceSelection};
use crate::llm::{ChatResult, ModelInfo, NeuronError};

use super::{RerankRequest, RerankResponse, RerankResult, RerankRuntime, RerankUsage};

const DEFAULT_MAX_LENGTH: usize = 512;

pub struct JinaRerankRuntime {
    inner: Arc<Mutex<Option<LoadedModel>>>,
}

struct LoadedModel {
    model: Arc<XLMRobertaForSequenceClassification>,
    tokenizer: Tokenizer,
    query_prefix: String,
    document_prefix: String,
    device: Device,
}

/// Jina v2 config.json shape. We deserialize the bare minimum needed
/// for runtime config; the rest of the standard XLMRobertaConfig is
/// reconstructed by reading the file as `XLMRobertaConfig` directly.
#[derive(Debug, Deserialize)]
struct JinaConfigOverlay {
    #[serde(default)]
    query_prefix: String,
    #[serde(default)]
    document_prefix: String,
    #[serde(default)]
    num_labels: Option<usize>,
}

impl JinaRerankRuntime {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(None)),
        }
    }

    fn resolve_model_path(model: &str) -> Result<PathBuf, String> {
        let path = match crate::config::env_var_result("RERANK_MODEL_PATH") {
            Ok(p) => PathBuf::from(p),
            Err(_) => {
                return Err(format!(
                    "JinaRerankRuntime: set NEURON_RERANK_MODEL_PATH to a directory containing \
                     config.json + tokenizer.json + *.safetensors for model '{}'. \
                     Download jinaai/jina-reranker-v2-base-multilingual from HuggingFace and set the path.",
                    model
                ));
            }
        };
        ensure_dir(&path, "NEURON_RERANK_MODEL_PATH")?;
        let config_path = path.join("config.json");
        let tokenizer_path = path.join("tokenizer.json");
        let has_weights = std::fs::read_dir(&path)
            .map(|entries| {
                entries.filter_map(|e| e.ok()).any(|e| {
                    e.path()
                        .extension()
                        .map(|ext| ext == "safetensors")
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false);

        if !config_path.exists() {
            return Err(format!("config.json not found in {}", path.display()));
        }
        if !tokenizer_path.exists() {
            return Err(format!("tokenizer.json not found in {}", path.display()));
        }
        if !has_weights {
            return Err(format!("No .safetensors files found in {}", path.display()));
        }
        Ok(path)
    }

    fn map_device(device: &DeviceSelection) -> Result<Device, String> {
        match device {
            DeviceSelection::Cpu => Ok(Device::Cpu),
            DeviceSelection::Nvidia => {
                #[cfg(feature = "cuda")]
                {
                    Device::new_cuda(0).map_err(|e| format!("cuda: {}", e))
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err("CUDA not enabled. Rebuild with --features cuda".to_string())
                }
            }
            DeviceSelection::Apple => {
                #[cfg(feature = "metal")]
                {
                    Device::new_metal(0).map_err(|e| format!("metal: {}", e))
                }
                #[cfg(not(feature = "metal"))]
                {
                    Err("Metal not enabled. Rebuild with --features metal".to_string())
                }
            }
            DeviceSelection::Auto | DeviceSelection::Vulkan => Ok(Device::Cpu),
        }
    }
}

impl Default for JinaRerankRuntime {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl RerankRuntime for JinaRerankRuntime {
    async fn load(model: &str, device: &DeviceSelection) -> ChatResult<Box<dyn RerankRuntime>>
    where
        Self: Sized,
    {
        let path = Self::resolve_model_path(model).map_err(NeuronError::not_implemented)?;
        let dev = Self::map_device(device).map_err(NeuronError::not_implemented)?;

        let config_path = path.join("config.json");
        let config_data = std::fs::read_to_string(&config_path)
            .map_err(|e| NeuronError::not_implemented(format!("read config.json: {}", e)))?;
        let config: XLMRobertaConfig = serde_json::from_str(&config_data)
            .map_err(|e| NeuronError::not_implemented(format!("parse config.json: {}", e)))?;
        // Jina v2 stores its optional query/document prefixes + a
        // sometimes-non-default num_labels in the same config.json.
        // We re-parse as a flat overlay to pick those up. Unknown
        // fields are ignored.
        let overlay: JinaConfigOverlay = serde_json::from_str(&config_data)
            .map_err(|e| NeuronError::not_implemented(format!("parse jina overlay: {}", e)))?;

        let tokenizer_path = path.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| NeuronError::not_implemented(format!("load tokenizer: {}", e)))?;

        let safetensors_path = path.join("model.safetensors");
        let vb = unsafe {
            crate::candle_compat::from_mmaped_safetensors(&[&safetensors_path], DType::F32, &dev)
        }
        .map_err(|e| NeuronError::not_implemented(format!("load safetensors: {}", e)))?;

        // Jina v2 ships with num_labels=1 for the binary rerank head.
        // Fall back to 1 if the overlay doesn't carry it.
        let num_labels = overlay.num_labels.unwrap_or(1);
        let model = XLMRobertaForSequenceClassification::new(num_labels, &config, vb)
            .map_err(|e| NeuronError::not_implemented(format!("build model: {}", e)))?;

        let runtime = JinaRerankRuntime::new();
        *runtime.inner.lock().unwrap() = Some(LoadedModel {
            model: Arc::new(model),
            tokenizer,
            query_prefix: overlay.query_prefix,
            document_prefix: overlay.document_prefix,
            device: dev,
        });
        Ok(Box::new(runtime))
    }

    async fn rerank(&self, req: RerankRequest) -> ChatResult<RerankResponse> {
        if req.documents.is_empty() {
            return Err(NeuronError::invalid_input("documents cannot be empty"));
        }

        let (model, tokenizer, query_prefix, document_prefix, device) = {
            let inner = self.inner.lock().unwrap();
            let loaded = inner.as_ref().ok_or_else(|| {
                NeuronError::not_initialized("JinaRerankRuntime not loaded. Call load() first.")
            })?;
            (
                Arc::clone(&loaded.model),
                loaded.tokenizer.clone(),
                loaded.query_prefix.clone(),
                loaded.document_prefix.clone(),
                loaded.device.clone(),
            )
        };

        let return_docs = req.return_documents.unwrap_or(true);
        let mut results = Vec::with_capacity(req.documents.len());

        for (i, doc) in req.documents.iter().enumerate() {
            // Jina v2 applies the prefixes BEFORE tokenization.
            let prefixed_query = format!("{query_prefix}{}", req.query);
            let prefixed_doc = format!("{document_prefix}{doc}");

            let encoding = tokenizer
                .encode(prefixed_query.as_str(), true)
                .map_err(|e| NeuronError::not_implemented(format!("tokenize query: {}", e)))?;
            let doc_encoding = tokenizer
                .encode(prefixed_doc.as_str(), true)
                .map_err(|e| NeuronError::not_implemented(format!("tokenize doc: {}", e)))?;

            let mut input_ids: Vec<u32> = encoding.get_ids().to_vec();
            let mut attention_mask: Vec<u32> = encoding.get_attention_mask().to_vec();
            let mut token_type_ids: Vec<u32> = encoding.get_type_ids().to_vec();

            input_ids.extend(doc_encoding.get_ids());
            attention_mask.extend(doc_encoding.get_attention_mask());
            token_type_ids.extend(doc_encoding.get_type_ids());

            if input_ids.len() > DEFAULT_MAX_LENGTH {
                input_ids.truncate(DEFAULT_MAX_LENGTH);
                attention_mask.truncate(DEFAULT_MAX_LENGTH);
                token_type_ids.truncate(DEFAULT_MAX_LENGTH);
            }

            let input_tensor = Tensor::new(input_ids.as_slice(), &device)
                .map_err(|e| NeuronError::not_implemented(format!("input tensor: {}", e)))?
                .unsqueeze(0)
                .map_err(|e| NeuronError::not_implemented(format!("unsqueeze input: {}", e)))?;
            let mask_tensor = Tensor::new(attention_mask.as_slice(), &device)
                .map_err(|e| NeuronError::not_implemented(format!("mask tensor: {}", e)))?
                .unsqueeze(0)
                .map_err(|e| NeuronError::not_implemented(format!("unsqueeze mask: {}", e)))?;
            let token_type_tensor = Tensor::new(token_type_ids.as_slice(), &device)
                .map_err(|e| NeuronError::not_implemented(format!("token_type tensor: {}", e)))?
                .unsqueeze(0)
                .map_err(|e| NeuronError::not_implemented(format!("unsqueeze token_type: {}", e)))?;

            let output = model
                .forward(&input_tensor, &mask_tensor, &token_type_tensor)
                .map_err(|e| NeuronError::not_implemented(format!("forward: {}", e)))?;

            let logits: Vec<Vec<f32>> = output
                .to_vec2::<f32>()
                .map_err(|e| NeuronError::not_implemented(format!("to_vec2: {}", e)))?;
            // num_labels=1 path: logits[0][0] is the single relevance logit.
            // num_labels=2 path (older jina v1): use the "entailment" logit
            // at index 1, since jina's class 0 is "non-relevant" and
            // class 1 is "relevant". After sigmoid that gives [0, 1].
            let logit = if logits.first().map(|r| r.len()) == Some(1) {
                logits[0][0]
            } else {
                logits
                    .first()
                    .and_then(|row| row.get(1).copied())
                    .ok_or_else(|| {
                        NeuronError::not_implemented(
                            "Jina output has unexpected shape: expected [1,1] or [1,2]".to_string(),
                        )
                    })?
            };
            let score = 1.0 / (1.0 + (-logit).exp());

            results.push(RerankResult {
                index: i,
                relevance_score: score,
                document: if return_docs {
                    Some(doc.clone())
                } else {
                    None
                },
            });
        }

        results.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.index.cmp(&b.index))
        });

        if let Some(k) = req.top_n {
            results.truncate(k.min(results.len()));
        }

        let usage = RerankUsage::estimate(&req.query, &req.documents);
        Ok(RerankResponse::new(req.model, results, usage))
    }

    fn list_models(&self) -> Vec<ModelInfo> {
        let mut info = ModelInfo::new("jina-reranker-base-v2", "neuron-jina");
        info.capabilities = Some(vec!["rerank".to_string()]);
        vec![info]
    }

    fn name(&self) -> &str {
        "jina-rerank"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jina_name() {
        let runtime = JinaRerankRuntime::new();
        assert_eq!(runtime.name(), "jina-rerank");
    }

    #[test]
    fn test_jina_list_models_returns_jina_reranker_base_v2() {
        let runtime = JinaRerankRuntime::new();
        let models = runtime.list_models();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "jina-reranker-base-v2");
    }

    #[test]
    fn test_jina_config_overlay_deserializes_with_no_prefixes() {
        let json = r#"{"query_prefix": "", "document_prefix": ""}"#;
        let overlay: JinaConfigOverlay = serde_json::from_str(json).unwrap();
        assert_eq!(overlay.query_prefix, "");
        assert_eq!(overlay.document_prefix, "");
    }

    #[test]
    fn test_jina_config_overlay_deserializes_with_prefixes() {
        let json = r#"{"query_prefix": "Represent this sentence: ", "document_prefix": ""}"#;
        let overlay: JinaConfigOverlay = serde_json::from_str(json).unwrap();
        assert_eq!(overlay.query_prefix, "Represent this sentence: ");
    }

    #[test]
    fn test_jina_config_overlay_ignores_unknown_fields() {
        // jina v2's real config.json has many fields (architectures,
        // id2label, layer_norm_eps, etc.). Verify we ignore them.
        let json = r#"{
            "query_prefix": "Q: ",
            "model_type": "xlm-roberta",
            "num_labels": 1,
            "architectures": ["XLMRobertaForSequenceClassification"],
            "id2label": {"0": "LABEL_0", "1": "LABEL_1"}
        }"#;
        let overlay: JinaConfigOverlay = serde_json::from_str(json).unwrap();
        assert_eq!(overlay.query_prefix, "Q: ");
        assert_eq!(overlay.num_labels, Some(1));
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_jina_load_without_env_returns_not_implemented() {
        unsafe { std::env::remove_var("NEURON_RERANK_MODEL_PATH") };
        let result = JinaRerankRuntime::load("jina-reranker-base-v2", &DeviceSelection::Cpu).await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert_eq!(err.error_type, "not_implemented");
        assert!(err.message.contains("NEURON_RERANK_MODEL_PATH"));
    }

    #[tokio::test]
    async fn test_jina_rerank_returns_not_initialized() {
        let runtime = JinaRerankRuntime::new();
        let req = RerankRequest::new("jina-reranker-base-v2", "q", vec!["a".to_string()]);
        let result = runtime.rerank(req).await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert_eq!(err.error_type, "not_initialized");
    }
}
