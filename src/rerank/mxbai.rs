//! Real Candle-backed Rerank runtime for mxbai-rerank-base-v2.
//!
//! Loads a Qwen2 cross-encoder (mixedbread-ai/mxbai-rerank-base-v2) with
//! a custom `Linear(hidden_size, 1)` classification head stored at
//! `classifier.weight` / `classifier.bias` in the same `model.safetensors`.
//! The architecture is built from `candle_transformers::models::qwen2`,
//! NOT a built-in `ForSequenceClassification` variant.
//!
//! Per the mxbai model card, the query must be prefixed with
//! `"Represent this sentence to search relevant passages: "` BEFORE
//! tokenization. Documents are passed through unchanged.
//!
//! Enable with: `NEURON_RERANK_RUNTIME=mxbai` and set `NEURON_RERANK_MODEL_PATH`
//! to a directory containing config.json, tokenizer.json, and model.safetensors.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use candle::Module;
use candle::{DType, Device, Tensor};
use candle_nn::{linear, Linear, VarBuilder};
use candle_transformers::models::qwen2::{Config as Qwen2Config, Model as Qwen2Model};
use tokenizers::Tokenizer;

use crate::config::{ensure_dir, DeviceSelection};
use crate::llm::{ChatResult, ModelInfo, NeuronError};

use super::{RerankRequest, RerankResponse, RerankResult, RerankRuntime, RerankUsage};

/// mxbai-rerank-base-v2's required query prefix, per the model card.
/// Document side is unprefixed.
const MXBAI_QUERY_PREFIX: &str = "Represent this sentence to search relevant passages: ";

/// Default cap on sequence length. mxbai's Qwen2 backbone was trained
/// with up to 8192 token context; for the rerank task a 512-token budget
/// matches BGE's default and the typical max-length for rerank
/// benchmarks. Users who need longer can override via a future config knob.
const DEFAULT_MAX_LENGTH: usize = 512;

pub struct MxbaiRerankRuntime {
    inner: Arc<Mutex<Option<LoadedModel>>>,
}

struct LoadedModel {
    model: Qwen2Model,
    classifier: Linear,
    tokenizer: Tokenizer,
    device: Device,
}

impl MxbaiRerankRuntime {
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
                    "MxbaiRerankRuntime: set NEURON_RERANK_MODEL_PATH to a directory containing \
                     config.json + tokenizer.json + *.safetensors for model '{}'. \
                     Download mixedbread-ai/mxbai-rerank-base-v2 from HuggingFace and set the path.",
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

    /// Apply the mxbai query prefix to a raw query, then concat with doc.
    /// Cap at `DEFAULT_MAX_LENGTH` tokens after tokenization.
    fn build_input_ids(
        tokenizer: &Tokenizer,
        query: &str,
        doc: &str,
    ) -> Result<(Vec<u32>, Vec<u32>), String> {
        let prefixed = format!("{MXBAI_QUERY_PREFIX}{query}");
        let q_enc = tokenizer
            .encode(prefixed.as_str(), true)
            .map_err(|e| format!("tokenize query: {e}"))?;
        let d_enc = tokenizer
            .encode(doc, true)
            .map_err(|e| format!("tokenize doc: {e}"))?;

        let mut ids: Vec<u32> = q_enc.get_ids().to_vec();
        let mut mask: Vec<u32> = q_enc.get_attention_mask().to_vec();
        ids.extend_from_slice(d_enc.get_ids());
        mask.extend_from_slice(d_enc.get_attention_mask());

        if ids.len() > DEFAULT_MAX_LENGTH {
            ids.truncate(DEFAULT_MAX_LENGTH);
            mask.truncate(DEFAULT_MAX_LENGTH);
        }
        Ok((ids, mask))
    }

    /// Score one (query, doc) pair. Pure sync — called inside the
    /// lock guard so the model stays Send/Sync safe.
    fn score_pair(
        model: &mut Qwen2Model,
        classifier: &Linear,
        tokenizer: &Tokenizer,
        device: &Device,
        query: &str,
        doc: &str,
    ) -> Result<f32, String> {
        let (ids, mask) = Self::build_input_ids(tokenizer, query, doc)?;
        let seq_len = ids.len();
        let input_ids = Tensor::new(ids.as_slice(), device)
            .map_err(|e| format!("ids tensor: {e}"))?
            .unsqueeze(0)
            .map_err(|e| format!("unsqueeze ids: {e}"))?;
        let attn_mask = Tensor::new(mask.as_slice(), device)
            .map_err(|e| format!("mask tensor: {e}"))?
            .unsqueeze(0)
            .map_err(|e| format!("unsqueeze mask: {e}"))?;
        // Qwen2 forward signature: (input_ids, seqlen_offset, attn_mask).
        // seqlen_offset=0 disables the KV cache path (single forward).
        let hidden = model
            .forward(&input_ids, 0, Some(&attn_mask))
            .map_err(|e| format!("qwen2 forward: {e}"))?;
        // hidden: [1, seq_len, hidden_size]. Take the last token.
        let last = hidden
            .narrow(1, seq_len - 1, 1)
            .map_err(|e| format!("narrow last token: {e}"))?
            .squeeze(1)
            .map_err(|e| format!("squeeze seq: {e}"))?;
        // last: [1, hidden_size] -> classifier -> [1, 1] -> [1] -> scalar
        let logit = classifier
            .forward(&last)
            .map_err(|e| format!("classifier forward: {e}"))?
            .squeeze(0)
            .map_err(|e| format!("squeeze batch: {e}"))?
            .to_scalar::<f32>()
            .map_err(|e| format!("to_scalar: {e}"))?;
        Ok(1.0 / (1.0 + (-logit).exp()))
    }
}

impl Default for MxbaiRerankRuntime {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl RerankRuntime for MxbaiRerankRuntime {
    async fn load(model: &str, device: &DeviceSelection) -> ChatResult<Box<dyn RerankRuntime>>
    where
        Self: Sized,
    {
        let path = Self::resolve_model_path(model).map_err(NeuronError::not_implemented)?;
        let dev = Self::map_device(device).map_err(NeuronError::not_implemented)?;

        let config_path = path.join("config.json");
        let config_data = std::fs::read_to_string(&config_path)
            .map_err(|e| NeuronError::not_implemented(format!("read config.json: {}", e)))?;
        let config: Qwen2Config = serde_json::from_str(&config_data)
            .map_err(|e| NeuronError::not_implemented(format!("parse Qwen2Config: {}", e)))?;

        let tokenizer_path = path.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| NeuronError::not_implemented(format!("load tokenizer: {}", e)))?;

        let safetensors_path = path.join("model.safetensors");
        let vb = unsafe {
            crate::candle_compat::from_mmaped_safetensors(&[&safetensors_path], DType::F32, &dev)
        }
        .map_err(|e| NeuronError::not_implemented(format!("load safetensors: {}", e)))?;

        // Qwen2 backbone: weights are stored under the "model" prefix
        // in mxbai's safetensors. Build a sub-VarBuilder for it.
        let model_vb = vb.pp("model");
        let mut qwen = Qwen2Model::new(&config, model_vb)
            .map_err(|e| NeuronError::not_implemented(format!("build Qwen2 model: {}", e)))?;
        // Clear any pre-existing KV cache from a previous load.
        qwen.clear_kv_cache();

        // Classification head: `Linear(hidden_size, 1)`. mxbai stores
        // it under "classifier"; some forks use "score". Try both.
        let classifier = match linear(config.hidden_size, 1, vb.pp("classifier")) {
            Ok(c) => c,
            Err(_) => linear(config.hidden_size, 1, vb.pp("score")).map_err(|e| {
                NeuronError::not_implemented(format!(
                    "build classifier (tried 'classifier' and 'score' prefixes): {e}"
                ))
            })?,
        };

        let runtime = MxbaiRerankRuntime::new();
        *runtime.inner.lock().unwrap() = Some(LoadedModel {
            model: qwen,
            classifier,
            tokenizer,
            device: dev,
        });
        Ok(Box::new(runtime))
    }

    async fn rerank(&self, req: RerankRequest) -> ChatResult<RerankResponse> {
        if req.documents.is_empty() {
            return Err(NeuronError::invalid_input("documents cannot be empty"));
        }

        // Hold the lock for the entire forward loop. All forward
        // work is sync, so no .await happens while the guard is
        // alive — keeping the model Send/Sync safe.
        let mut guard = self.inner.lock().unwrap();
        let loaded = guard.as_mut().ok_or_else(|| {
            NeuronError::not_initialized("MxbaiRerankRuntime not loaded. Call load() first.")
        })?;
        let LoadedModel {
            model,
            classifier,
            tokenizer,
            device,
        } = loaded;

        let return_docs = req.return_documents.unwrap_or(true);
        let mut results = Vec::with_capacity(req.documents.len());
        for (i, doc) in req.documents.iter().enumerate() {
            // Clear KV cache between docs to prevent cross-doc attention bleed.
            model.clear_kv_cache();
            let score = Self::score_pair(model, classifier, tokenizer, device, &req.query, doc)
                .map_err(NeuronError::not_implemented)?;
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
        drop(guard);

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
        let mut info = ModelInfo::new("mxbai-rerank-base-v2", "neuron-mxbai");
        info.capabilities = Some(vec!["rerank".to_string()]);
        vec![info]
    }

    fn name(&self) -> &str {
        "mxbai-rerank"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mxbai_name() {
        let runtime = MxbaiRerankRuntime::new();
        assert_eq!(runtime.name(), "mxbai-rerank");
    }

    #[test]
    fn test_mxbai_list_models_returns_mxbai_rerank_base_v2() {
        let runtime = MxbaiRerankRuntime::new();
        let models = runtime.list_models();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "mxbai-rerank-base-v2");
    }

    #[test]
    fn test_mxbai_query_prefix_constant_matches_model_card() {
        // The prefix MUST match the official mxbai model card; if upstream
        // changes the wording, this test breaks and forces a re-check.
        assert_eq!(
            MXBAI_QUERY_PREFIX,
            "Represent this sentence to search relevant passages: "
        );
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_mxbai_load_without_env_returns_not_implemented() {
        unsafe { std::env::remove_var("NEURON_RERANK_MODEL_PATH") };
        let result = MxbaiRerankRuntime::load("mxbai-rerank-base-v2", &DeviceSelection::Cpu).await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert_eq!(err.error_type, "not_implemented");
        assert!(err.message.contains("NEURON_RERANK_MODEL_PATH"));
    }

    #[tokio::test]
    async fn test_mxbai_rerank_returns_not_initialized() {
        let runtime = MxbaiRerankRuntime::new();
        let req = RerankRequest::new("mxbai-rerank-base-v2", "q", vec!["a".to_string()]);
        let result = runtime.rerank(req).await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert_eq!(err.error_type, "not_initialized");
    }
}
