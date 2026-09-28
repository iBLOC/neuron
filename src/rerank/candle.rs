//! Real Candle-backed Rerank runtime for BGE-reranker-style cross-encoders.
//!
//! Loads a `XLMRobertaForSequenceClassification` model (e.g. BAAI/bge-reranker-base)
//! and produces [0, 1]-normalized relevance scores via sigmoid.
//!
//! Enable with: `NEURON_RERANK_RUNTIME=candle` and set `NEURON_RERANK_MODEL_PATH`
//! to a directory containing config.json, tokenizer.json, and model.safetensors.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use candle::{Device, DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::xlm_roberta::{
    Config as XLMRobertaConfig, XLMRobertaForSequenceClassification,
};
use tokenizers::Tokenizer;

use crate::config::{ensure_dir, DeviceSelection};
use crate::llm::{ChatResult, ModelInfo, NeuronError};

use super::{RerankRequest, RerankResponse, RerankResult, RerankRuntime, RerankUsage};

const DEFAULT_MAX_LENGTH: usize = 512;

pub struct CandleRerankRuntime {
    inner: Arc<Mutex<Option<LoadedModel>>>,
}

struct LoadedModel {
    model: Arc<XLMRobertaForSequenceClassification>,
    tokenizer: Tokenizer,
    device: Device,
}

impl CandleRerankRuntime {
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
                    "CandleRerankRuntime: set NEURON_RERANK_MODEL_PATH to a directory containing \
                     config.json + tokenizer.json + *.safetensors for model '{}'. \
                     Download BAAI/bge-reranker-base from HuggingFace and set the path.",
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

impl Default for CandleRerankRuntime {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl RerankRuntime for CandleRerankRuntime {
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

        let tokenizer_path = path.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| NeuronError::not_implemented(format!("load tokenizer: {}", e)))?;

        let safetensors_path = path.join("model.safetensors");
        let vb = unsafe {
            crate::candle_compat::from_mmaped_safetensors(&[&safetensors_path], DType::F32, &dev)
        }
        .map_err(|e| NeuronError::not_implemented(format!("load safetensors: {}", e)))?;

        let num_labels = 1;
        let model = XLMRobertaForSequenceClassification::new(num_labels, &config, vb)
            .map_err(|e| NeuronError::not_implemented(format!("build model: {}", e)))?;

        let runtime = CandleRerankRuntime::new();
        *runtime.inner.lock().unwrap() = Some(LoadedModel { model: Arc::new(model), tokenizer, device: dev });
        Ok(Box::new(runtime))
    }

    async fn rerank(&self, req: RerankRequest) -> ChatResult<RerankResponse> {
        if req.documents.is_empty() {
            return Err(NeuronError::invalid_input("documents cannot be empty"));
        }

        let (model, tokenizer, device) = {
            let inner = self.inner.lock().unwrap();
            let loaded = inner.as_ref().ok_or_else(|| {
                NeuronError::not_initialized("CandleRerankRuntime not loaded. Call load() first.")
            })?;
            (Arc::clone(&loaded.model), loaded.tokenizer.clone(), loaded.device.clone())
        };

        let return_docs = req.return_documents.unwrap_or(true);
        let mut results = Vec::with_capacity(req.documents.len());

        for (i, doc) in req.documents.iter().enumerate() {
            let encoding = tokenizer
                .encode(req.query.as_str(), true)
                .map_err(|e| NeuronError::not_implemented(format!("tokenize query: {}", e)))?;
            let doc_encoding = tokenizer
                .encode(doc.as_str(), true)
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
            let logit = logits
                .first()
                .and_then(|row| row.first())
                .copied()
                .ok_or_else(|| {
                    NeuronError::not_implemented(String::from("Failed to extract logit from model output"))
                })?;

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
        let mut info = ModelInfo::new("bge-reranker-base", "neuron-candle");
        info.capabilities = Some(vec!["rerank".to_string()]);
        vec![info]
    }

    fn name(&self) -> &str {
        "candle-rerank"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[serial_test::serial]
    fn test_resolve_model_path_without_env_returns_useful_error() {
        unsafe { std::env::remove_var("NEURON_RERANK_MODEL_PATH") };
        let result = CandleRerankRuntime::resolve_model_path("bge-reranker-base");
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(
            err.contains("NEURON_RERANK_MODEL_PATH"),
            "Error should mention NEURON_RERANK_MODEL_PATH, got: {}",
            err
        );
    }

    #[test]
    #[serial_test::serial]
    fn test_resolve_model_path_with_valid_env_path_returns_ok() {
        let dir = std::env::temp_dir().join(format!(
            "neuron-rerank-resolve-ok-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        std::fs::write(dir.join("config.json"), b"{}").expect("write config");
        std::fs::write(dir.join("tokenizer.json"), b"{}").expect("write tokenizer");
        std::fs::write(dir.join("model.safetensors"), b"fake").expect("write weights");
        unsafe { std::env::set_var("NEURON_RERANK_MODEL_PATH", &dir) };

        let result = CandleRerankRuntime::resolve_model_path("bge-reranker-base");
        let _ = std::fs::remove_dir_all(&dir);
        unsafe { std::env::remove_var("NEURON_RERANK_MODEL_PATH") };

        assert!(result.is_ok(), "expected Ok, got: {result:?}");
        assert_eq!(result.unwrap(), dir);
    }

    #[test]
    fn test_candle_rerank_name() {
        let runtime = CandleRerankRuntime::new();
        assert_eq!(runtime.name(), "candle-rerank");
    }

    #[test]
    fn test_candle_rerank_list_models_returns_bge_reranker_base() {
        let runtime = CandleRerankRuntime::new();
        let models = runtime.list_models();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "bge-reranker-base");
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_candle_rerank_rerank_returns_not_initialized() {
        let runtime = CandleRerankRuntime::new();
        let req = RerankRequest::new("bge-reranker-base", "q", vec!["a".to_string()]);
        let result = runtime.rerank(req).await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert_eq!(err.error_type, "not_initialized");
    }
}