//! Real Candle-backed Rerank runtime for mxbai-rerank-base-v2.
//!
//! **Scoring model (rewritten 2026-10-30 to match the released
//! checkpoint).** mxbai-rerank-base-v2 is a *generative* Qwen2 (0.5B,
//! 24 layers, hidden 896, tied embeddings) that scores by answering
//! `1` (relevant) or `0` (not relevant) after a relevance-judgment
//! prompt. The checkpoint ships ONLY the backbone — no `classifier.*`
//! head (the model card's `classifier.weight` description does not
//! match the released `model.safetensors`) — plus
//! `1_LogitScore/config.json` naming the verdict tokens
//! (`true_token_id: 16` = "1", `false_token_id: 15` = "0"). The score
//! is `sigmoid(logit["1"] − logit["0"])` at the last prompt position.
//!
//! The prompt mirrors the checkpoint's `chat_template.jinja` (Qwen chat
//! framing + the fixed "search relevance expert" instruction); verdict
//! token ids are looked up from the tokenizer with the
//! `1_LogitScore` values as fallback.
//!
//! History: the first implementation assumed a regression head and
//! failed with "build classifier (tried 'classifier' and 'score'
//! prefixes)"; the `.pp("model")` double-prefix bug before that was
//! fixed 2026-10-29 (candle's Qwen2Model adds the prefix itself). Both
//! were found by `benches/model_bench.rs` on real weights.
//!
//! Enable with: `NEURON_RERANK_RUNTIME=mxbai` and set `NEURON_RERANK_MODEL_PATH`
//! to a directory containing config.json, tokenizer.json, and model.safetensors.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use candle::{DType, Device, Tensor};
use candle_transformers::models::qwen2::{Config as Qwen2Config, ModelForCausalLM};
use tokenizers::Tokenizer;

use crate::config::{ensure_dir, DeviceSelection};
use crate::llm::{ChatResult, ModelInfo, NeuronError};

use super::{RerankRequest, RerankResponse, RerankResult, RerankRuntime, RerankUsage};

/// Cap on the tokenized prompt length (Qwen2 supports far more; this
/// matches the rerank-typical budget and keeps per-doc latency bounded).
const DEFAULT_MAX_LENGTH: usize = 512;

/// Fallback verdict token ids when the tokenizer has no single-token
/// `"1"` / `"0"` entries. These match the checkpoint's own
/// `1_LogitScore/config.json`.
const FALLBACK_TOKEN_YES: u32 = 16;
const FALLBACK_TOKEN_NO: u32 = 15;

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

/// The relevance prompt, mirroring the checkpoint's
/// `chat_template.jinja` with both slots filled.
fn build_prompt(query: &str, doc: &str) -> String {
    format!(
        "<|im_start|>system\n\
         You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n\
         <|im_start|>user\n\
         query: {query}\n\
         document: {doc}\n\
         You are a search relevance expert who evaluates how well documents match search \
         queries. For each query-document pair, carefully analyze the semantic relationship \
         between them, then provide your binary relevance judgment (0 for not relevant, \
         1 for relevant).\n\
         Relevance:<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

pub struct MxbaiRerankRuntime {
    inner: Arc<Mutex<Option<LoadedModel>>>,
}

struct LoadedModel {
    model: ModelForCausalLM,
    tokenizer: Tokenizer,
    /// Verdict token ids: `1` (relevant) and `0` (not relevant).
    token_yes: u32,
    token_no: u32,
    device: Device,
}

impl MxbaiRerankRuntime {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(None)),
        }
    }

    fn resolve_model_dir(model: &str) -> Result<PathBuf, String> {
        if let Ok(p) = crate::config::env_var_result("RERANK_MODEL_PATH") {
            return Ok(PathBuf::from(p));
        }
        // Fall back to the standard model library layout
        // (<root>/rerank/<id>) so a pulled model just works.
        let root = crate::config::NeuronConfig::resolved_model_dir();
        let dir = root.join("rerank").join(model);
        if dir.is_dir() {
            return Ok(dir);
        }
        Err(format!(
            "MxbaiRerankRuntime: set NEURON_RERANK_MODEL_PATH to a directory containing \
             config.json + tokenizer.json + model.safetensors for model '{model}' \
             (mixedbread-ai/mxbai-rerank-base-v2); looked in {}",
            dir.display()
        ))
    }

    /// Look up the verdict token ids. Prefers the checkpoint's own
    /// `1_LogitScore/config.json`; falls back to a tokenizer lookup and
    /// finally to the documented constants.
    fn resolve_verdict_tokens(dir: &std::path::Path, tokenizer: &Tokenizer) -> (u32, u32) {
        let logit_cfg = dir.join("1_LogitScore").join("config.json");
        if let Ok(bytes) = std::fs::read_to_string(&logit_cfg) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&bytes) {
                let yes = v.get("true_token_id").and_then(|x| x.as_u64());
                let no = v.get("false_token_id").and_then(|x| x.as_u64());
                if let (Some(yes), Some(no)) = (yes, no) {
                    return (yes as u32, no as u32);
                }
            }
        }
        let from_tok = |s: &str, fallback: u32| {
            tokenizer
                .token_to_id(s)
                .unwrap_or(fallback)
        };
        (from_tok("1", FALLBACK_TOKEN_YES), from_tok("0", FALLBACK_TOKEN_NO))
    }

    /// Score one (query, doc) pair. Pure sync — called inside the
    /// lock guard so the model stays Send/Sync safe.
    fn score_pair(
        model: &mut ModelForCausalLM,
        tokenizer: &Tokenizer,
        token_yes: u32,
        token_no: u32,
        device: &Device,
        query: &str,
        doc: &str,
    ) -> Result<f32, String> {
        let prompt = build_prompt(query, doc);
        // add_special_tokens=false: the prompt already carries the Qwen
        // chat framing (<|im_start|> / <|im_end|> are vocab entries).
        let encoding = tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|e| format!("tokenize prompt: {e}"))?;
        let mut ids: Vec<u32> = encoding.get_ids().to_vec();
        if ids.len() > DEFAULT_MAX_LENGTH {
            ids.truncate(DEFAULT_MAX_LENGTH);
        }
        let input_ids = Tensor::new(ids.as_slice(), device)
            .map_err(|e| format!("ids tensor: {e}"))?
            .unsqueeze(0)
            .map_err(|e| format!("unsqueeze ids: {e}"))?;

        // Single full-sequence forward, no KV cache (seqlen_offset=0).
        // `ModelForCausalLM` applies the tied lm_head and returns LOGITS
        // `[1, seq, vocab]`; the bare `Qwen2Model` returns hidden states —
        // the first cut of this scorer read hidden dims 15/16 as "logits"
        // and produced meaningless scores (found 2026-10-30 by the
        // generation probe in examples/mxbai_score_check.rs).
        let logits = model
            .forward(&input_ids, 0)
            .map_err(|e| format!("qwen2 forward: {e}"))?;
        // `ModelForCausalLM::forward` already narrows to the last position:
        // logits is `[1, 1, vocab]`.
        let last = logits
            .squeeze(1)
            .map_err(|e| format!("squeeze seq: {e}"))?
            .squeeze(0)
            .map_err(|e| format!("squeeze batch: {e}"))?;
        let logit_yes = last
            .get(token_yes as usize)
            .map_err(|e| format!("gather yes token: {e}"))?
            .to_scalar::<f32>()
            .map_err(|e| format!("yes to_scalar: {e}"))?;
        let logit_no = last
            .get(token_no as usize)
            .map_err(|e| format!("gather no token: {e}"))?
            .to_scalar::<f32>()
            .map_err(|e| format!("no to_scalar: {e}"))?;
        // P(relevant) over the two verdict tokens.
        Ok(1.0 / (1.0 + (-(logit_yes - logit_no)).exp()))
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
        let dir = Self::resolve_model_dir(model)
            .map_err(NeuronError::not_implemented)
            .map_err(|e| NeuronError::new(format!("MxbaiRerankRuntime load failed: {e}")))?;
        let dev = map_device(device).map_err(NeuronError::not_implemented)?;

        ensure_dir(&dir, "NEURON_RERANK_MODEL_PATH").map_err(NeuronError::not_implemented)?;

        let config_path = dir.join("config.json");
        let config_data = std::fs::read_to_string(&config_path)
            .map_err(|e| NeuronError::not_implemented(format!("read config.json: {e}")))?;
        let mut config: Qwen2Config = serde_json::from_str(&config_data)
            .map_err(|e| NeuronError::not_implemented(format!("parse config.json: {e}")))?;


        let tokenizer_path = dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| NeuronError::not_implemented(format!("load tokenizer: {e}")))?;

        let (token_yes, token_no) = Self::resolve_verdict_tokens(&dir, &tokenizer);

        let safetensors_path = dir.join("model.safetensors");
        let vb = unsafe {
            crate::candle_compat::from_mmaped_safetensors(&[&safetensors_path], DType::F32, &dev)
        }
        .map_err(|e| NeuronError::not_implemented(format!("load safetensors: {e}")))?;

        // The checkpoint's keys are `model.*` and candle's Qwen2Model adds
        // that prefix itself — pass the UNPREFIXED VarBuilder (an extra
        // `.pp("model")` here made the loader look for
        // `model.model.embed_tokens.weight`).
        let mut qwen = ModelForCausalLM::new(&config, vb.clone())
            .map_err(|e| NeuronError::not_implemented(format!("build Qwen2 model: {e}")))?;
        qwen.clear_kv_cache();

        let runtime = MxbaiRerankRuntime {
            inner: Arc::new(Mutex::new(None)),
        };
        *runtime.inner.lock().unwrap() = Some(LoadedModel {
            model: qwen,
            tokenizer,
            token_yes,
            token_no,
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
            tokenizer,
            token_yes,
            token_no,
            device,
        } = loaded;

        let return_docs = req.return_documents.unwrap_or(true);
        let mut results = Vec::with_capacity(req.documents.len());
        for (i, doc) in req.documents.iter().enumerate() {
            // Clear KV cache between docs to prevent cross-doc attention bleed.
            model.clear_kv_cache();
            let score = Self::score_pair(model, tokenizer, *token_yes, *token_no, device, &req.query, doc)
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
        "mxbai"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_prompt_matches_chat_template() {
        let p = build_prompt("what is a apple?", "An apple is a fruit.");
        assert!(p.starts_with("<|im_start|>system\n"));
        assert!(p.contains("query: what is a apple?"));
        assert!(p.contains("document: An apple is a fruit."));
        assert!(p.ends_with("Relevance:<|im_end|>\n<|im_start|>assistant\n"));
    }

    #[test]
    fn test_build_prompt_escapes_nothing_but_embeds_both_slots() {
        let p = build_prompt("q", "d");
        assert!(p.contains("query: q\ndocument: d"));
    }
}
