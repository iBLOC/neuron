//! Candle-based [`EmbeddingGemma2Runtime`] — the runtime behind
//! `NEURON_EMBEDDING_RUNTIME=gemma2`, serving Google's
//! `google/embeddinggemma-2` (Apache-2.0, 2026-10) natively on candle.
//!
//! Model directory contract (like MiniLM's):
//!
//! - `NEURON_EMBEDDING_MODEL_PATH` — a directory with `config.json` +
//!   `tokenizer.json` + `model.safetensors` (the upstream HF repo
//!   layout, loaded directly — no conversion), **or**
//! - absent, the default location
//!   `<NEURON_MODEL_DIRS root>/gemma2/embeddinggemma-2/`.
//!
//! Pipeline (mirrors the sentence-transformers wrapper the model
//! card documents): tokenize → bidirectional text tower (vendored in
//! [`super::text_model`]) → masked mean pool over the per-token 768-d
//! projections → L2 normalize → optional Matryoshka truncation to
//! `dimensions` (512 / 256 / 128 are the card-recommended sizes; any
//! `1..=768` slice is accepted) with re-normalization.
//!
//! Compute dtype: f32 on CPU, bf16 on CUDA — the checkpoint ships
//! bf16 and the model card explicitly warns against f16. 24 layers ×
//! 512 hidden as f32 ≈ 1.1 GB of RAM on CPU.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use candle::{DType, Device, Tensor};
use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy, TruncationDirection, TruncationParams, TruncationStrategy, Tokenizer};

use crate::config::{ensure_dir, DeviceSelection};
use crate::embedding::{
    EmbeddingData, EmbeddingRequest, EmbeddingResponse, EmbeddingRuntime, EmbeddingUsage,
    EmbeddingVector, EncodingFormat,
};
use crate::llm::{ChatResult, ModelInfo, NeuronError};

use super::config::TextConfig;
use super::text_model::TextModel;

/// Default maximum sequence length for tokenization (truncation).
/// The model accepts up to 8192 tokens per the model card, but the
/// bidirectional attention masks cost `batch × seq² × 4 bytes`; 2048
/// keeps a 4-element batch under ~64 MB of mask. Override with
/// `NEURON_EMBEDDING_GEMMA2_MAX_SEQ`.
const DEFAULT_MAX_SEQ: u32 = 2048;

pub struct EmbeddingGemma2Runtime {
    inner: Arc<Mutex<Option<LoadedModel>>>,
    models: Vec<ModelInfo>,
}

struct LoadedModel {
    model: TextModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl EmbeddingGemma2Runtime {
    pub fn new() -> Self {
        let mut info = ModelInfo::new("embeddinggemma-2", "neuron");
        info.capabilities = Some(vec!["embedding".to_string()]);
        Self {
            inner: Arc::new(Mutex::new(None)),
            models: vec![info],
        }
    }

    /// Resolve the model directory: `NEURON_EMBEDDING_MODEL_PATH`
    /// wins, else the default `gemma2/embeddinggemma-2` subtree under
    /// the first model-dir root.
    fn resolve_model_path(model: &str) -> Result<PathBuf, String> {
        let path = match crate::config::env_var_result("EMBEDDING_MODEL_PATH") {
            Ok(p) => PathBuf::from(p),
            Err(_) => crate::config::NeuronConfig::resolved_model_dir()
                .join("gemma2")
                .join(model),
        };
        ensure_dir(&path, "NEURON_EMBEDDING_MODEL_PATH")?;
        let config_path = path.join("config.json");
        let tokenizer_path = path.join("tokenizer.json");
        let weights_path = path.join("model.safetensors");
        if !config_path.exists() {
            return Err(format!("config.json not found in {}", path.display()));
        }
        if !tokenizer_path.exists() {
            return Err(format!("tokenizer.json not found in {}", path.display()));
        }
        if !weights_path.exists() {
            return Err(format!(
                "model.safetensors not found in {} (EmbeddingGemma 2 ships one unsharded safetensors file)",
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

    /// f32 on CPU, bf16 on CUDA — the checkpoint ships bf16 and the
    /// model card forbids f16; f32 is the safe CPU compute dtype.
    fn compute_dtype(device: &Device) -> DType {
        match device {
            Device::Cuda(_) => DType::BF16,
            _ => DType::F32,
        }
    }

    fn max_seq_tokens() -> u32 {
        crate::config::env_var("EMBEDDING_GEMMA2_MAX_SEQ")
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(DEFAULT_MAX_SEQ)
    }

    /// Masked mean pool over the per-token embeddings, then L2
    /// normalize — the sentence-transformers `1_Pooling` +
    /// `2_Normalize` steps. `[batch, seq, dim]` → `[batch, dim]`.
    fn pool_and_normalize(last_hidden: &Tensor, attention_mask: &Tensor) -> Result<Tensor, String> {
        let mask3d = attention_mask
            .unsqueeze(2)
            .map_err(|e| format!("unsqueeze mask: {e}"))?
            .to_dtype(last_hidden.dtype())
            .map_err(|e| format!("cast mask dtype: {e}"))?;
        let masked = last_hidden
            .broadcast_mul(&mask3d)
            .map_err(|e| format!("apply mask: {e}"))?;
        let summed = masked.sum(1).map_err(|e| format!("sum over seq: {e}"))?;
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
        l2_normalize(&mean)
    }
}

/// Per-row L2 normalize with an epsilon guard.
fn l2_normalize(xs: &Tensor) -> Result<Tensor, String> {
    let norm_sq = xs
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
    xs.broadcast_div(&norm).map_err(|e| format!("l2 div: {e}"))
}

/// Matryoshka truncation: keep the first `dims` components, then
/// re-normalize (the model card's recipe for 512 / 256 / 128).
fn truncate_dims(xs: Tensor, dims: usize) -> Result<Tensor, String> {
    let narrowed = xs
        .narrow(1, 0, dims)
        .map_err(|e| format!("truncate dims: {e}"))?;
    l2_normalize(&narrowed)
}

impl Default for EmbeddingGemma2Runtime {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl EmbeddingRuntime for EmbeddingGemma2Runtime {
    async fn load(
        model: &str,
        device: &DeviceSelection,
    ) -> ChatResult<Box<dyn EmbeddingRuntime>>
    where
        Self: Sized,
    {
        let path = Self::resolve_model_path(model).map_err(NeuronError::not_implemented)?;
        let device = Self::map_device_selection(device).map_err(NeuronError::new)?;
        // The checkpoint's config.json is the composite shape
        // (text_config + vision_config + audio_config); the text
        // tower reads the text_config object.
        let raw = std::fs::read(path.join("config.json"))
            .map_err(|e| NeuronError::new(format!("read config.json: {e}")))?;
        let json: serde_json::Value =
            serde_json::from_slice(&raw).map_err(|e| NeuronError::new(format!("parse config.json: {e}")))?;
        let text_config = json
            .get("text_config")
            .cloned()
            .ok_or_else(|| NeuronError::new("config.json has no text_config"))?;
        let config: TextConfig = serde_json::from_value(text_config)
            .map_err(|e| NeuronError::new(format!("parse EmbeddingGemma2 TextConfig: {e}")))?;
        config
            .validate()
            .map_err(NeuronError::new)?;

        let mut tokenizer = Tokenizer::from_file(path.join("tokenizer.json"))
            .map_err(|e| NeuronError::new(format!("load tokenizer: {e}")))?;
        // Batch-longest right padding (pad id 0, matching the
        // checkpoint's pad token) so one forward serves the batch;
        // right truncation to keep the seq² mask cost bounded.
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: PaddingDirection::Right,
            pad_id: config.pad_token_id,
            pad_token: "<pad>".to_string(),
            pad_type_id: 0,
            pad_to_multiple_of: None,
        }));
        tokenizer
            .with_truncation(Some(TruncationParams {
                strategy: TruncationStrategy::LongestFirst,
                max_length: Self::max_seq_tokens() as usize,
                stride: 0,
                direction: TruncationDirection::Right,
            }))
            .map_err(|e| NeuronError::new(format!("configure truncation: {e}")))?;

        let dtype = Self::compute_dtype(&device);
        let weights_path = path.join("model.safetensors");
        // Root vb; the text tower lives under `language_model.`.
        let vb = unsafe {
            crate::candle_compat::from_mmaped_safetensors(&[weights_path], dtype, &device)
        }
        .map_err(|e| NeuronError::new(format!("load weights: {e}")))?;
        let model = TextModel::new(&config, vb.pp("language_model"))
            .map_err(|e| NeuronError::new(format!("build EmbeddingGemma2 TextModel: {e}")))?;

        let runtime = Self::new();
        {
            let mut guard = runtime.inner.lock().unwrap();
            *guard = Some(LoadedModel {
                model,
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
        if let Some(dims) = req.dimensions {
            if dims == 0 {
                return Err(NeuronError::invalid_input("dimensions must be >= 1"));
            }
        }
        let mut guard = self.inner.lock().unwrap();
        let loaded = match guard.as_mut() {
            Some(l) => l,
            None => {
                return Err(NeuronError::not_initialized(
                    "EmbeddingGemma2Runtime: call `load()` first or set \
                     NEURON_EMBEDDING_RUNTIME=gemma2 via run_embedded()",
                ));
            }
        };

        // Batched encode → single forward → masked mean pool.
        let encodings = loaded
            .tokenizer
            .encode_batch(texts.clone(), true)
            .map_err(|e| NeuronError::new(format!("tokenize batch: {e}")))?;
        let batch = encodings.len();
        let seq_len = encodings[0].get_ids().len();

        let mut all_ids: Vec<u32> = Vec::with_capacity(batch * seq_len);
        let mut all_mask: Vec<u32> = Vec::with_capacity(batch * seq_len);
        for enc in &encodings {
            all_ids.extend_from_slice(enc.get_ids());
            all_mask.extend_from_slice(enc.get_attention_mask());
        }

        let ids_t = Tensor::new(all_ids.as_slice(), &loaded.device)
            .map_err(|e| NeuronError::new(format!("ids tensor: {e}")))?
            .reshape(&[batch, seq_len])
            .map_err(|e| NeuronError::new(format!("reshape ids: {e}")))?;
        let mask_t = Tensor::new(all_mask.as_slice(), &loaded.device)
            .map_err(|e| NeuronError::new(format!("mask tensor: {e}")))?
            .reshape(&[batch, seq_len])
            .map_err(|e| NeuronError::new(format!("reshape mask: {e}")))?;

        let has_pad = all_mask.iter().any(|m| *m == 0);
        let per_token = loaded
            .model
            .forward(&ids_t, if has_pad { Some(&mask_t) } else { None })
            .map_err(|e| NeuronError::new(format!("embeddinggemma2 forward: {e}")))?;
        let pooled = Self::pool_and_normalize(&per_token, &mask_t).map_err(NeuronError::new)?;
        let pooled = match req.dimensions.map(|d| d as usize) {
            Some(dims) if dims < pooled.dim(1).map_err(|e| NeuronError::new(e.to_string()))? => {
                truncate_dims(pooled, dims).map_err(NeuronError::new)?
            }
            _ => pooled,
        };
        let rows: Vec<Vec<f32>> = pooled
            .to_dtype(DType::F32)
            .map_err(|e| NeuronError::new(format!("cast rows: {e}")))?
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
        "gemma2-embedding"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[serial_test::serial]
    async fn test_gemma2_load_with_bad_path_errors() {
        // Force the env-var path to a nonexistent directory so the
        // test does not depend on whether the real checkpoint happens
        // to be downloaded on this machine.
        unsafe { std::env::set_var("NEURON_EMBEDDING_MODEL_PATH", "/this/path/does/not/exist/eg2") };
        let result =
            EmbeddingGemma2Runtime::load("embeddinggemma-2", &DeviceSelection::Cpu).await;
        unsafe { std::env::remove_var("NEURON_EMBEDDING_MODEL_PATH") };
        assert!(result.is_err(), "a nonexistent model dir must error cleanly");
        let err = result.err().unwrap();
        assert_eq!(err.error_type, "not_implemented");
    }

    #[tokio::test]
    async fn test_gemma2_embed_returns_not_initialized() {
        let runtime = EmbeddingGemma2Runtime::new();
        let req = EmbeddingRequest::new("embeddinggemma-2", "hello");
        let result = runtime.embed(req).await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert_eq!(err.error_type, "not_initialized");
    }

    #[test]
    fn test_gemma2_name() {
        let runtime = EmbeddingGemma2Runtime::new();
        assert_eq!(runtime.name(), "gemma2-embedding");
    }

    #[test]
    fn test_gemma2_list_models() {
        let runtime = EmbeddingGemma2Runtime::new();
        let models = runtime.list_models();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "embeddinggemma-2");
        assert_eq!(
            models[0].capabilities.as_ref().unwrap(),
            &vec!["embedding".to_string()]
        );
    }

    #[test]
    fn test_gemma2_compute_dtype_by_device() {
        assert_eq!(EmbeddingGemma2Runtime::compute_dtype(&Device::Cpu), DType::F32);
    }

    #[test]
    fn test_gemma2_max_seq_env_override() {
        unsafe { std::env::set_var("NEURON_EMBEDDING_GEMMA2_MAX_SEQ", "512") };
        assert_eq!(EmbeddingGemma2Runtime::max_seq_tokens(), 512);
        unsafe { std::env::set_var("NEURON_EMBEDDING_GEMMA2_MAX_SEQ", "0") };
        assert_eq!(
            EmbeddingGemma2Runtime::max_seq_tokens(),
            DEFAULT_MAX_SEQ,
            "0 must fall back to the default, not disable truncation"
        );
        unsafe { std::env::remove_var("NEURON_EMBEDDING_GEMMA2_MAX_SEQ") };
    }

    /// Real-checkpoint end-to-end: `cargo test --lib
    /// real_weights_embed -- --ignored --nocapture`
    ///
    /// Needs the checkpoint at the default model location
    /// (`<model root>/gemma2/embeddinggemma-2/`) or
    /// `NEURON_EMBEDDING_MODEL_PATH` pointing at it. Asserts the
    /// response shape, L2 normalization, that related texts embed
    /// closer than unrelated ones, the MRL `dimensions` truncation,
    /// and that a padded batch member matches its solo embedding.
    #[tokio::test]
    #[serial_test::serial]
    #[ignore]
    async fn real_weights_embed() {
        unsafe { std::env::remove_var("NEURON_EMBEDDING_MODEL_PATH") };
        let rt = EmbeddingGemma2Runtime::load("embeddinggemma-2", &DeviceSelection::Cpu)
            .await
            .expect("load real checkpoint");

        // 1. Shape + normalization.
        let req = EmbeddingRequest::new(
            "embeddinggemma-2",
            vec![
                "The cat sat on the mat.".to_string(),
                "A cat rested on a rug.".to_string(),
                "Quantum entanglement links particle states across distance.".to_string(),
            ],
        );
        let resp = rt.embed(req).await.expect("embed");
        assert_eq!(resp.data.len(), 3);
        let vecs: Vec<Vec<f32>> = resp
            .data
            .iter()
            .map(|d| match &d.embedding {
                EmbeddingVector::Float(v) => v.clone(),
                other => panic!("expected float encoding, got {other:?}"),
            })
            .collect();
        for v in &vecs {
            assert_eq!(v.len(), 768, "native MRL width");
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-2, "L2-normalized, got {norm}");
        }
        let cos = |a: &[f32], b: &[f32]| -> f32 {
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            dot // inputs are unit vectors
        };
        let related = cos(&vecs[0], &vecs[1]);
        let unrelated = cos(&vecs[0], &vecs[2]);
        println!("cos(cat, cat-rug) = {related:.4}, cos(cat, quantum) = {unrelated:.4}");
        assert!(
            related > unrelated,
            "related texts must embed closer: {related:.4} vs {unrelated:.4}"
        );
        assert!(related > 0.3, "same-topic pair suspiciously far: {related:.4}");

        // 2. MRL truncation via `dimensions`.
        let mut req = EmbeddingRequest::new("embeddinggemma-2", "The cat sat on the mat.");
        req.dimensions = Some(256);
        let resp = rt.embed(req).await.expect("embed dims=256");
        let v = match &resp.data[0].embedding {
            EmbeddingVector::Float(v) => v.clone(),
            other => panic!("expected float encoding, got {other:?}"),
        };
        assert_eq!(v.len(), 256);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-2, "re-normalized after truncation, got {norm}");

        // 3. Padded batch member ≈ solo embedding (padding must not
        // pollute the bidirectional attention or the mean pool).
        let solo = {
            let req = EmbeddingRequest::new("embeddinggemma-2", "The cat sat on the mat.");
            let resp = rt.embed(req).await.expect("embed solo");
            match &resp.data[0].embedding {
                EmbeddingVector::Float(v) => v.clone(),
                other => panic!("expected float encoding, got {other:?}"),
            }
        };
        let req = EmbeddingRequest::new(
            "embeddinggemma-2",
            vec![
                "The cat sat on the mat.".to_string(),
                "A much longer sentence that forces right padding onto the batch sibling, \
                 because BatchLongest pads every member to the longest input in the batch."
                    .to_string(),
            ],
        );
        let resp = rt.embed(req).await.expect("embed batch");
        let paired = match &resp.data[0].embedding {
            EmbeddingVector::Float(v) => v.clone(),
            other => panic!("expected float encoding, got {other:?}"),
        };
        let batch_cos = cos(&solo, &paired);
        println!("cos(solo, padded-batch-member) = {batch_cos:.6}");
        assert!(
            batch_cos > 0.999,
            "padding must not perturb the embedding: {batch_cos:.6}"
        );
    }
}
