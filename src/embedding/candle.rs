//! Candle-based [`MiniLmL6V2EmbeddingRuntime`] — the real
//! implementation behind `sentence-transformers/all-MiniLM-L6-v2`.
//!
//! Loading requires `NEURON_EMBEDDING_MODEL_PATH` to point at a
//! directory containing `config.json` + `tokenizer.json` +
//! `*.safetensors`. Without that env var, [`load`] returns a
//! `not_implemented` error so the embedded server still starts and
//! the echo runtime stays usable.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer};

use crate::config::{ensure_dir, DeviceSelection};
use crate::embedding::{
    EmbeddingData, EmbeddingRequest, EmbeddingResponse, EmbeddingRuntime, EmbeddingUsage,
    EmbeddingVector, EncodingFormat,
};
use crate::llm::{ChatResult, ModelInfo, NeuronError};

pub struct MiniLmL6V2EmbeddingRuntime {
    inner: Arc<Mutex<Option<LoadedModel>>>,
    models: Vec<ModelInfo>,
}

struct LoadedModel {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl MiniLmL6V2EmbeddingRuntime {
    pub fn new() -> Self {
        let mut info = ModelInfo::new("all-minilm-l6-v2", "neuron");
        info.capabilities = Some(vec!["embedding".to_string()]);
        Self {
            inner: Arc::new(Mutex::new(None)),
            models: vec![info],
        }
    }

    fn resolve_model_path(model: &str) -> Result<PathBuf, String> {
        let _ = model;
        let path = match crate::config::env_var_result("EMBEDDING_MODEL_PATH") {
            Ok(p) => PathBuf::from(p),
            Err(_) => {
                return Err(format!(
                    "MiniLmL6V2EmbeddingRuntime: set NEURON_EMBEDDING_MODEL_PATH to a \
                     directory containing config.json + tokenizer.json + \
                     *.safetensors. Download `sentence-transformers/all-MiniLM-L6-v2` \
                     from HuggingFace and set the path. Requested model: '{model}'"
                ));
            }
        };
        ensure_dir(&path, "NEURON_EMBEDDING_MODEL_PATH")?;
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
            return Err(format!(
                "No .safetensors files found in {}",
                path.display()
            ));
        }
        Ok(path)
    }

    fn map_device_selection(device: &DeviceSelection) -> Result<Device, String> {
        match device {
            DeviceSelection::Cpu => Ok(Device::Cpu),
            DeviceSelection::Nvidia => {
                #[cfg(feature = "cuda")]
                {
                    Device::new_cuda(0).map_err(|e| format!("cuda: {e}"))
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err("CUDA not enabled. Rebuild with --features cuda".to_string())
                }
            }
            DeviceSelection::Apple => {
                #[cfg(feature = "metal")]
                {
                    Device::new_metal(0).map_err(|e| format!("metal: {e}"))
                }
                #[cfg(not(feature = "metal"))]
                {
                    Err("Metal not enabled. Rebuild with --features metal".to_string())
                }
            }
            DeviceSelection::Vulkan | DeviceSelection::Auto => {
                // Auto falls back to CPU; Vulkan not yet wired.
                Ok(Device::Cpu)
            }
        }
    }

    /// Mean-pool the last hidden state with the attention mask, then
    /// L2-normalize — the standard sentence-transformers pipeline
    /// for all-MiniLM-L6-v2. Returns a `[batch, hidden]` tensor.
    fn mean_pool(
        last_hidden: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor, String> {
        // Expand [batch, seq] mask to [batch, seq, 1] for broadcast
        // mul against [batch, seq, hidden] hidden states.
        let mask3d = attention_mask
            .unsqueeze(2)
            .map_err(|e| format!("unsqueeze mask: {e}"))?
            .to_dtype(last_hidden.dtype())
            .map_err(|e| format!("cast mask dtype: {e}"))?;
        let masked = last_hidden
            .broadcast_mul(&mask3d)
            .map_err(|e| format!("apply mask: {e}"))?;
        let summed = masked
            .sum(1)
            .map_err(|e| format!("sum over seq: {e}"))?;
        let counts = attention_mask
            .sum(1)
            .map_err(|e| format!("sum mask: {e}"))?
            .to_dtype(last_hidden.dtype())
            .map_err(|e| format!("cast counts dtype: {e}"))?
            .clamp(1e-9, f64::MAX)
            .map_err(|e| format!("clamp counts: {e}"))?
            .unsqueeze(1)
            .map_err(|e| format!("unsqueeze counts: {e}"))?;
        let mean = summed
            .broadcast_div(&counts)
            .map_err(|e| format!("div by counts: {e}"))?;
        // L2 normalize per row.
        let norm_sq = mean
            .sqr()
            .map_err(|e| format!("sqr: {e}"))?
            .sum(1)
            .map_err(|e| format!("sum sqr: {e}"))?;
        let norm = norm_sq
            .sqrt()
            .map_err(|e| format!("sqrt: {e}"))?
            .clamp(1e-12, f64::MAX)
            .map_err(|e| format!("clamp norm: {e}"))?
            .unsqueeze(1)
            .map_err(|e| format!("unsqueeze norm: {e}"))?;
        mean.broadcast_div(&norm).map_err(|e| format!("l2 div: {e}"))
    }
}

impl Default for MiniLmL6V2EmbeddingRuntime {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl EmbeddingRuntime for MiniLmL6V2EmbeddingRuntime {
    async fn load(
        model: &str,
        device: &DeviceSelection,
    ) -> ChatResult<Box<dyn EmbeddingRuntime>>
    where
        Self: Sized,
    {
        let path = Self::resolve_model_path(model).map_err(NeuronError::not_implemented)?;
        let device = Self::map_device_selection(device).map_err(NeuronError::new)?;
        let config: BertConfig = serde_json::from_slice(
            &std::fs::read(path.join("config.json"))
                .map_err(|e| NeuronError::new(format!("read config.json: {e}")))?,
        )
        .map_err(|e| NeuronError::new(format!("parse BertConfig: {e}")))?;
        let tokenizer = Tokenizer::from_file(path.join("tokenizer.json"))
            .map_err(|e| NeuronError::new(format!("load tokenizer: {e}")))?;
        // Enable batch-longest padding so `encode_batch` produces
        // encodings of identical length, ready for a single forward.
        // For all-MiniLM-L6-v2 the [PAD] token has id 0, matching
        // the model's pad embedding.
        let mut tokenizer = tokenizer;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: PaddingDirection::Right,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            pad_type_id: 0,
            pad_to_multiple_of: None,
        }));
        let dtype = DType::F32;
        let weights_path = path.join("model.safetensors");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[weights_path],
                dtype,
                &device,
            )
        }
        .map_err(|e| NeuronError::new(format!("load weights: {e}")))?;
        let bert = BertModel::load(vb, &config)
            .map_err(|e| NeuronError::new(format!("build BertModel: {e}")))?;
        let runtime = Self::new();
        {
            let mut guard = runtime.inner.lock().unwrap();
            *guard = Some(LoadedModel {
                model: bert,
                tokenizer,
                device,
            });
        }
        Ok(Box::new(runtime))
    }

    async fn embed(&self, req: EmbeddingRequest) -> ChatResult<EmbeddingResponse> {
        let texts = req.texts();
        if texts.is_empty() || (texts.len() == 1 && texts[0].is_empty()) {
            return Err(NeuronError::invalid_input("input cannot be empty"));
        }
        let format = match &req.encoding_format {
            None => EncodingFormat::Float,
            Some(s) => EncodingFormat::parse(s).map_err(NeuronError::invalid_input)?,
        };
        let mut guard = self.inner.lock().unwrap();
        let loaded = match guard.as_mut() {
            Some(l) => l,
            None => {
                return Err(NeuronError::not_initialized(
                    "MiniLmL6V2EmbeddingRuntime: call `load()` first or set \
                     NEURON_EMBEDDING_RUNTIME=candle via run_embedded()",
                ));
            }
        };

        // Batched encode → single forward → per-row mean-pool.
        let encodings = loaded
            .tokenizer
            .encode_batch(texts.clone(), true)
            .map_err(|e| NeuronError::new(format!("tokenize batch: {e}")))?;
        let batch = encodings.len();
        let seq_len = encodings[0].get_ids().len();

        let mut all_ids: Vec<u32> = Vec::with_capacity(batch * seq_len);
        let mut all_mask: Vec<u32> = Vec::with_capacity(batch * seq_len);
        let mut all_type_ids: Vec<u32> = Vec::with_capacity(batch * seq_len);
        for enc in &encodings {
            all_ids.extend_from_slice(enc.get_ids());
            all_mask.extend_from_slice(enc.get_attention_mask());
            all_type_ids.extend_from_slice(enc.get_type_ids());
        }

        let ids_t = Tensor::new(all_ids.as_slice(), &loaded.device)
            .map_err(|e| NeuronError::new(format!("ids tensor: {e}")))?
            .reshape(&[batch, seq_len])
            .map_err(|e| NeuronError::new(format!("reshape ids: {e}")))?;
        let mask_t = Tensor::new(all_mask.as_slice(), &loaded.device)
            .map_err(|e| NeuronError::new(format!("mask tensor: {e}")))?
            .reshape(&[batch, seq_len])
            .map_err(|e| NeuronError::new(format!("reshape mask: {e}")))?;
        let type_t = Tensor::new(all_type_ids.as_slice(), &loaded.device)
            .map_err(|e| NeuronError::new(format!("type_ids tensor: {e}")))?
            .reshape(&[batch, seq_len])
            .map_err(|e| NeuronError::new(format!("reshape type_ids: {e}")))?;

        let out = loaded
            .model
            .forward(&ids_t, &type_t, Some(&mask_t))
            .map_err(|e| NeuronError::new(format!("bert forward: {e}")))?;
        let pooled = Self::mean_pool(&out, &mask_t).map_err(NeuronError::new)?;
        let rows: Vec<Vec<f32>> = pooled
            .to_vec2()
            .map_err(|e| NeuronError::new(format!("to_vec2: {e}")))?;

        let data: Vec<EmbeddingData> = rows
            .into_iter()
            .enumerate()
            .map(|(i, vec)| EmbeddingData {
                object: "embedding".to_string(),
                index: i,
                embedding: EmbeddingVector::encode(&vec, format),
            })
            .collect();
        let usage = EmbeddingUsage::estimate(&texts);
        Ok(EmbeddingResponse::new(req.model, data, usage))
    }

    fn list_models(&self) -> Vec<ModelInfo> {
        self.models.clone()
    }

    fn name(&self) -> &str {
        "candle-embedding"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[serial_test::serial]
    async fn test_candle_embedding_load_without_env_returns_not_implemented() {
        unsafe { std::env::remove_var("NEURON_EMBEDDING_MODEL_PATH") };
        let result =
            MiniLmL6V2EmbeddingRuntime::load("all-minilm-l6-v2", &DeviceSelection::Cpu).await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert_eq!(err.error_type, "not_implemented");
        assert!(err.message.contains("NEURON_EMBEDDING_MODEL_PATH"));
    }

    #[tokio::test]
    async fn test_candle_embedding_embed_returns_not_initialized() {
        let runtime = MiniLmL6V2EmbeddingRuntime::new();
        let req = EmbeddingRequest::new("all-minilm-l6-v2", "hello");
        let result = runtime.embed(req).await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert_eq!(err.error_type, "not_initialized");
    }

    #[test]
    fn test_candle_embedding_name() {
        let runtime = MiniLmL6V2EmbeddingRuntime::new();
        assert_eq!(runtime.name(), "candle-embedding");
    }

    #[test]
    fn test_candle_embedding_list_models_returns_all_minilm_l6_v2() {
        let runtime = MiniLmL6V2EmbeddingRuntime::new();
        let models = runtime.list_models();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "all-minilm-l6-v2");
        assert_eq!(
            models[0].capabilities.as_ref().unwrap(),
            &vec!["embedding".to_string()]
        );
    }

    #[test]
    fn test_resolve_model_path_missing_env() {
        unsafe { std::env::remove_var("NEURON_EMBEDDING_MODEL_PATH") };
        let err = MiniLmL6V2EmbeddingRuntime::resolve_model_path("all-minilm-l6-v2")
            .unwrap_err();
        assert!(err.contains("NEURON_EMBEDDING_MODEL_PATH"));
    }

    #[test]
    fn test_resolve_model_path_nonexistent_dir() {
        unsafe {
            std::env::set_var(
                "NEURON_EMBEDDING_MODEL_PATH",
                "/this/path/does/not/exist/12345",
            );
        }
        let err = MiniLmL6V2EmbeddingRuntime::resolve_model_path("all-minilm-l6-v2")
            .unwrap_err();
        assert!(err.contains("does not exist"));
        unsafe { std::env::remove_var("NEURON_EMBEDDING_MODEL_PATH") };
    }
}
