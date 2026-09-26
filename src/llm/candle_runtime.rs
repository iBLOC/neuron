use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use candle_core::{Device, DType, IndexOp};
use candle_nn::VarBuilder;
use candle_transformers::models::{chatglm, llama, mistral, phi3, qwen2, qwen3};
use futures_util::stream::BoxStream;
use tokenizers::Tokenizer;

use super::{ChatMessage, ChatRequest, ChatResponse, Choice, DeltaMessage, LlmRuntime, ModelInfo};
use crate::config::{ensure_dir, DeviceSelection};
use crate::llm::NeuronError;

#[path = "candle_arch.rs"]
mod candle_arch;
use candle_arch::{detect_arch, detect_model_type, CausalArch};

#[cfg(feature = "flash-attn")]
mod vendor;

pub struct CandleLlmRuntime {
    inner: Arc<Mutex<Option<LoadedModel>>>,
}

enum CausalModel {
    Qwen2(qwen2::ModelForCausalLM),
    Qwen3(qwen3::ModelForCausalLM),
    #[cfg(feature = "flash-attn")]
    Qwen2Flash(vendor::qwen2::ModelForCausalLM),
    #[cfg(feature = "flash-attn")]
    Qwen3Flash(vendor::qwen3::ModelForCausalLM),
    Llama {
        model: llama::Llama,
        cache: llama::Cache,
    },
    Phi3(phi3::Model),
    Mistral(mistral::Model),
    ChatGlm(chatglm::Model),
}

impl CausalModel {
    fn forward(&mut self, input: &candle_core::Tensor, offset: usize) -> candle_core::Result<candle_core::Tensor> {
        match self {
            CausalModel::Qwen2(m) => m.forward(input, offset),
            CausalModel::Qwen3(m) => m.forward(input, offset),
            #[cfg(feature = "flash-attn")]
            CausalModel::Qwen2Flash(m) => m.forward(input, offset),
            #[cfg(feature = "flash-attn")]
            CausalModel::Qwen3Flash(m) => m.forward(input, offset),
            CausalModel::Llama { model, cache } => model.forward(input, offset, cache),
            CausalModel::Phi3(m) => m.forward(input, offset),
            CausalModel::Mistral(m) => m.forward(input, offset),
            CausalModel::ChatGlm(m) => m.forward(input),
        }
    }
}

/// Resolve the `NEURON_USE_FLASH_ATTN` opt-in for FlashAttention-2.
///
/// `flash-attn` requires the `candle-flash-attn` crate (built against
/// CUDA) and a CUDA device at runtime; the sdpa path is correct for
/// CPU and Metal. We default to `false` and only flip the Llama
/// `Config::use_flash_attn` field when the user explicitly opts in.
///
/// Accepted truthy values: `1`, `true`, `TRUE`, `yes`.
/// Accepted falsy values: `0`, `false`, `FALSE`, `no`.
/// Anything else (or unset) returns `None`, falling back to `false`.
pub(crate) fn parse_use_flash_attn() -> Option<bool> {
    crate::config::env_var("USE_FLASH_ATTN").and_then(|v| {
        match v.as_str() {
            "1" | "true" | "TRUE" | "yes" => Some(true),
            "0" | "false" | "FALSE" | "no" => Some(false),
            _ => None,
        }
    })
}

struct LoadedModel {
    model: CausalModel,
    tokenizer: Tokenizer,
    eos_token_id: u32,
    device: Device,
    /// Held for the Llama path so `chat` can rebuild the KV cache
    /// from scratch at the start of every request — candle's
    /// `llama::Cache` carries the position offset from the
    /// previous request's decode loop and the next call would
    /// otherwise panic on a broadcast mismatch. `None` for the
    /// non-Llama architectures (their caches are model-internal
    /// or absent).
    llama_cfg: Option<llama::Config>,
    /// Storage dtype the weights were materialised in. The Llama KV
    /// cache has to be rebuilt with this same dtype, and keeping the
    /// checkpoint's native dtype (rather than always upcasting to
    /// `F32`) is what lets a 2B model fit in RAM on a 16 GB host.
    dtype: DType,
}

/// Storage dtype used to materialise Llama-family weights.
///
/// `NEURON_LLM_DTYPE` accepts `f32` (default — the historical
/// behaviour), `bf16` and `f16`. Anything unrecognised falls back to
/// `f32` with a warning rather than failing the load: a typo in an
/// env var should not take a whole model down.
fn resolve_weight_dtype() -> DType {
    match crate::config::env_var("LLM_DTYPE").as_deref() {
        Some(raw) => match raw.trim().to_ascii_lowercase().as_str() {
            "f32" | "float32" | "fp32" => DType::F32,
            "bf16" | "bfloat16" => DType::BF16,
            "f16" | "float16" | "fp16" | "half" => DType::F16,
            other => {
                eprintln!(
                    "[neuron] NEURON_LLM_DTYPE={other:?} not recognised \
                     (want f32 | bf16 | f16); falling back to f32"
                );
                DType::F32
            }
        },
        None => DType::F32,
    }
}

impl CandleLlmRuntime {
    pub fn new() -> Self {
        Self { inner: Arc::new(Mutex::new(None)) }
    }

    /// Detect the causal LM architecture from a model's `config.json`.
    /// Thin re-export of [`detect_arch`] so callers can keep using
    /// `CandleLlmRuntime::detect_arch(...)` for backward compatibility.
    pub fn detect_arch(model_dir: &std::path::Path) -> Result<(CausalArch, String), String> {
        detect_arch(model_dir)
    }

    /// Detect only the `model_type` string from `config.json`.
    /// Thin re-export of [`detect_model_type`] so callers can keep
    /// using `CandleLlmRuntime::detect_model_type(...)` for
    /// backward compatibility.
    pub fn detect_model_type(model_dir: &std::path::Path) -> Result<String, String> {
        detect_model_type(model_dir)
    }

    pub fn resolve_model_path(model: &str) -> Result<PathBuf, String> {
        if let Ok(p) = crate::config::env_var_result("LLM_MODEL_PATH") {
            let path = PathBuf::from(p);
            ensure_dir(&path, "NEURON_LLM_MODEL_PATH")?;
            return Ok(path);
        }

        // No explicit env override: look the model up by id in the model
        // library that /v1/models/pull populates
        // (`~/.neuron/models/llm/<id>`), so any downloaded model is
        // servable without extra configuration.
        let root = crate::config::NeuronConfig::resolved_model_dir();
        let candidates = [root.join("llm").join(model), root.join(model)];
        for candidate in &candidates {
            if candidate.is_dir() {
                return Self::validate_model_dir(candidate);
            }
        }
        Err(format!(
            "CandleLlmRuntime: model '{model}' not found under {} (set NEURON_LLM_MODEL_PATH \
             to a directory containing config.json + tokenizer.json + *.safetensors)",
            root.display()
        ))
    }

    /// Verify a directory holds a servable model (config + tokenizer +
    /// at least one .safetensors shard) and return it.
    fn validate_model_dir(path: &std::path::Path) -> Result<PathBuf, String> {
        let path = path.to_path_buf();
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

    fn map_device_selection(device: &DeviceSelection) -> Result<Device, String> {
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
            DeviceSelection::Auto => {
                // Try CUDA first, then Metal, then fall back to CPU
                #[cfg(feature = "cuda")]
                {
                    if let Ok(dev) = Device::new_cuda(0) {
                        return Ok(dev);
                    }
                }
                #[cfg(feature = "metal")]
                {
                    if let Ok(dev) = Device::new_metal(0) {
                        return Ok(dev);
                    }
                }
                Ok(Device::Cpu)
            }
            DeviceSelection::Vulkan => Ok(Device::Cpu),
        }
    }

    /// Rebuild the Llama KV cache so a generation starts from a clean
    /// position axis.
    ///
    /// candle's `llama::Cache` is stateful — its `k_v` tensors carry the
    /// position offset of the previous request's decode loop — so
    /// reusing one across requests triggers a `cannot broadcast [N, N] to
    /// [1, H, N, D]` panic on the next forward pass. Every generation
    /// entry point must therefore reset it first, which is why this is a
    /// shared helper rather than inlined at each call site: `chat` and
    /// `chat_stream` previously diverged here, and the streaming path
    /// failed silently (empty SSE body) whenever it ran after a non-
    /// streaming request.
    ///
    /// Non-Llama architectures use model-internal caches (or none) and
    /// need no reset.
    fn reset_llama_cache(m: &mut LoadedModel) -> Result<(), NeuronError> {
        if let (CausalModel::Llama { model: _, cache }, Some(cfg)) =
            (&mut m.model, m.llama_cfg.as_ref())
        {
            *cache = llama::Cache::new(true, m.dtype, cfg, &m.device)
                .map_err(|e| NeuronError::not_implemented(format!("rebuild Llama cache: {e}")))?;
        }
        Ok(())
    }

    fn render_chatml_prompt(messages: &[ChatMessage]) -> String {
        let mut out = String::new();
        for msg in messages {
            out.push_str(&format!("<|im_start|>{}\n", msg.role));
            out.push_str(&msg.content);
            out.push_str("<|im_end|>\n");
        }
        out.push_str("<|im_start|>assistant\n");
        out
    }

    // SentencePiece uses U+2581 (▁) to mark word boundaries; replace with a space.
    fn decode_token_for_stream(token: &str) -> String {
        token.replace('\u{2581}', " ")
    }

    /// Append `token` to `generated`, re-decode the whole sequence, and
    /// return the byte offset the caller has already emitted alongside the
    /// text that is new since then.
    ///
    /// The vocabulary string for a single id is raw byte-level BPE — `Ġ`
    /// for a space, `Ċ` for a newline — which is not user-visible text, so
    /// it must never be sent to a client. Decoding the accumulated ids
    /// instead is what makes the streaming body match the non-streaming one
    /// byte for byte.
    fn decode_new_suffix(
        tokenizer: &Tokenizer,
        generated: &mut Vec<u32>,
        text_so_far: &mut String,
        token: u32,
    ) -> (usize, String) {
        generated.push(token);
        let old_len = text_so_far.len();
        *text_so_far = match tokenizer.decode(generated, true) {
            Ok(s) => s,
            Err(_) => {
                // Degrade to the vocab string rather than dropping the token.
                let raw = tokenizer.id_to_token(token).unwrap_or_default();
                format!("{text_so_far}{}", Self::decode_token_for_stream(&raw))
            }
        };
        let suffix = text_so_far.get(old_len..).unwrap_or_default().to_string();
        (old_len, suffix)
    }

    /// Sample a token id from `logits` with temperature-scaled softmax and
    /// optional top-p (nucleus) and/or top-k truncation.
    ///   * `temperature <= 0` falls back to argmax.
    ///   * `top_p >= 1.0` (or `<= 0.0`) means no nucleus truncation.
    ///   * `0.0 < top_p < 1.0` keeps the smallest top-K set whose
    ///     cumulative probability meets or exceeds `top_p`.
    ///   * `top_k == 0` (or >= logits.len()) means no top-k truncation.
    ///   * `top_k >= 1` keeps the K tokens with the largest probability.
    ///   * top-p and top-k compose: truncate by both, then sample within
    ///     the intersection.
    fn sample_token_from_logits(
        logits: &[f32],
        temperature: f32,
        top_p: f32,
        top_k: u32,
    ) -> u32 {
        if logits.is_empty() {
            return 0;
        }
        if temperature <= 0.0 {
            return logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
        }
        // Subtract the max logit before exp to keep the softmax numerically
        // stable for large logit values.
        let max_logit = logits
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let probs: Vec<f32> = logits
            .iter()
            .map(|&l| ((l - max_logit) / temperature).exp())
            .collect();
        let sum: f32 = probs.iter().sum();
        if !sum.is_finite() || sum <= 0.0 {
            return 0;
        }
        let probs: Vec<f32> = probs.iter().map(|p| p / sum).collect();

        let mut indexed: Vec<(usize, f32)> =
            probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        let k_cutoff = if (top_k as usize) < indexed.len() && top_k > 0 {
            top_k as usize
        } else {
            indexed.len()
        };
        let mut candidates: Vec<(usize, f32)> = indexed.into_iter().take(k_cutoff).collect();

        if !(0.0 < top_p && top_p < 1.0) {
            if top_p <= 0.0 {
                return candidates.first().map(|(i, _)| *i as u32).unwrap_or(0);
            }
            let truncated_sum: f32 = candidates.iter().map(|(_, p)| p).sum();
            if truncated_sum <= 0.0 {
                return candidates.first().map(|(i, _)| *i as u32).unwrap_or(0);
            }
            let r: f32 = fastrand::f32() * truncated_sum;
            let mut acc = 0.0_f32;
            for (i, (_, p)) in candidates.iter().enumerate() {
                acc += p;
                if r < acc {
                    return candidates[i].0 as u32;
                }
            }
            return candidates.last().map(|(i, _)| *i as u32).unwrap_or(0);
        }

        let mut cumsum = 0.0_f32;
        let mut p_cutoff = candidates.len();
        for (i, (_, p)) in candidates.iter().enumerate() {
            cumsum += p;
            if cumsum >= top_p {
                p_cutoff = i + 1;
                break;
            }
        }
        candidates.truncate(p_cutoff);
        let truncated_sum: f32 = candidates.iter().map(|(_, p)| p).sum();
        if truncated_sum <= 0.0 {
            return candidates.first().map(|(i, _)| *i as u32).unwrap_or(0);
        }
        let r: f32 = fastrand::f32() * truncated_sum;
        let mut acc = 0.0_f32;
        for (i, (_, p)) in candidates.iter().enumerate() {
            acc += p;
            if r < acc {
                return candidates[i].0 as u32;
            }
        }
        candidates.last().map(|(i, _)| *i as u32).unwrap_or(0)
    }

    /// If any of `stops` is a non-empty suffix of `text`, return the byte
    /// offset of the first match (i.e., the length of the prefix that does
    /// NOT contain the stop sequence). Empty stop strings are ignored.
    fn check_stop_suffix(text: &str, stops: &[String]) -> Option<usize> {
        for stop in stops {
            if !stop.is_empty() && text.ends_with(stop.as_str()) {
                return Some(text.len() - stop.len());
            }
        }
        None
    }

}

impl Default for CandleLlmRuntime {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl super::LlmRuntime for CandleLlmRuntime {
    async fn load(model: &str, device: &DeviceSelection) -> Result<Box<dyn LlmRuntime>, NeuronError>
    where
        Self: Sized,
    {
        let path = Self::resolve_model_path(model).map_err(NeuronError::not_implemented)?;
        let dev = Self::map_device_selection(device).map_err(NeuronError::not_implemented)?;

        let (arch, model_type) = detect_arch(&path)
            .map_err(NeuronError::not_implemented)?;
        eprintln!(
            "[neuron] CandleLlmRuntime: detected model_type={model_type} in {}",
            path.display()
        );

        // NEURON_USE_FLASH_ATTN: opt-in to FlashAttention-2 for the
        // architectures that support it. Upstream candle-transformers
        // 0.9.2 ships the field for Llama 2/3 + Mistral; Qwen 2/3
        // support is added in src/llm/vendor/ (requires the
        // --features flash-attn build + a CUDA device at runtime).
        // See `parse_use_flash_attn` for accepted values.
        let use_flash_attn = parse_use_flash_attn().unwrap_or(false);
        if use_flash_attn {
            match &arch {
                CausalArch::Qwen2(_) | CausalArch::Qwen3(_) => {
                    #[cfg(feature = "flash-attn")]
                    eprintln!(
                        "[neuron] CandleLlmRuntime: Qwen2/3 use_flash_attn=true (requires CUDA)"
                    );
                    #[cfg(not(feature = "flash-attn"))]
                    eprintln!(
                        "[neuron] WARNING: NEURON_USE_FLASH_ATTN=1 is set but flash-attn feature \
                         not enabled; re-build with --features flash-attn. Using sdpa."
                    );
                }
                CausalArch::Mistral(_) => {
                    eprintln!(
                        "[neuron] CandleLlmRuntime: use_flash_attn=true (requires --features flash-attn and CUDA)"
                    );
                }
                _ => {
                    eprintln!(
                        "[neuron] CandleLlmRuntime: use_flash_attn=true (requires --features flash-attn and CUDA)"
                    );
                }
            }
        }

        let safetensors_files: Vec<PathBuf> = std::fs::read_dir(&path)
            .map_err(|e| NeuronError::not_implemented(format!("read dir: {e}")))?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "safetensors")
                    .unwrap_or(false)
            })
            .map(|e| e.path())
            .collect();

        if safetensors_files.is_empty() {
            return Err(NeuronError::not_implemented(
                "No safetensors files found".to_string(),
            ));
        }

        let safetensors_paths: Vec<&std::path::Path> =
            safetensors_files.iter().map(|p| p.as_path()).collect();

        // `bf16` halves peak weight memory versus the f32 default, which
        // is the difference between fitting and paging out on a 16 GB host.
        let weight_dtype = resolve_weight_dtype();
        eprintln!(
            "[neuron] CandleLlmRuntime: weights dtype = {weight_dtype:?} (NEURON_LLM_DTYPE=f32|bf16|f16)"
        );

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(safetensors_paths.as_slice(), weight_dtype, &dev)
        }
        .map_err(|e| NeuronError::not_implemented(format!("load weights: {e}")))?;

        let model = match &arch {
            CausalArch::Qwen2(cfg) => {
                #[cfg(feature = "flash-attn")]
                if use_flash_attn {
                    let flash_cfg: vendor::qwen2::Config = cfg.clone().into();
                    let m = vendor::qwen2::ModelForCausalLM::new(&flash_cfg, vb)
                        .map_err(|e| NeuronError::not_implemented(format!("build Qwen2 (flash): {e}")))?;
                    return Ok(CausalModel::Qwen2Flash(m));
                }
                let m = qwen2::ModelForCausalLM::new(cfg, vb)
                    .map_err(|e| NeuronError::not_implemented(format!("build Qwen2: {e}")))?;
                CausalModel::Qwen2(m)
            }
            CausalArch::Qwen3(cfg) => {
                #[cfg(feature = "flash-attn")]
                if use_flash_attn {
                    let flash_cfg: vendor::qwen3::Config = cfg.clone().into();
                    let m = vendor::qwen3::ModelForCausalLM::new(&flash_cfg, vb)
                        .map_err(|e| NeuronError::not_implemented(format!("build Qwen3/3.5 (flash): {e}")))?;
                    return Ok(CausalModel::Qwen3Flash(m));
                }
                let m = qwen3::ModelForCausalLM::new(cfg, vb)
                    .map_err(|e| NeuronError::not_implemented(format!("build Qwen3/3.5: {e}")))?;
                CausalModel::Qwen3(m)
            }
            CausalArch::Llama(cfg) => {
                let inner = llama::LlamaConfig::clone(cfg)
                    .into_config(use_flash_attn);
                let m = llama::Llama::load(vb, &inner)
                    .map_err(|e| NeuronError::not_implemented(format!("build Llama: {e}")))?;
                let cache = llama::Cache::new(
                    true,
                    weight_dtype,
                    &inner,
                    &dev,
                )
                .map_err(|e| NeuronError::not_implemented(format!("build Llama cache: {e}")))?;
                CausalModel::Llama { model: m, cache }
            }
            CausalArch::Phi3(cfg) => {
                let m = phi3::Model::new(cfg, vb)
                    .map_err(|e| NeuronError::not_implemented(format!("build Phi3: {e}")))?;
                CausalModel::Phi3(m)
            }
            CausalArch::Mistral(cfg) => {
                let mut cfg = cfg.clone();
                if use_flash_attn {
                    cfg.use_flash_attn = true;
                }
                let m = mistral::Model::new(&cfg, vb)
                    .map_err(|e| NeuronError::not_implemented(format!("build Mistral: {e}")))?;
                CausalModel::Mistral(m)
            }
            CausalArch::ChatGlm(cfg) => {
                let m = chatglm::Model::new(cfg, vb)
                    .map_err(|e| NeuronError::not_implemented(format!("build ChatGLM: {e}")))?;
                CausalModel::ChatGlm(m)
            }
        };

        let tokenizer = Tokenizer::from_file(path.join("tokenizer.json"))
            .map_err(|e| NeuronError::not_implemented(format!("load tokenizer: {e}")))?;

        // EOS ids: <|im_end|> for Qwen (ChatML turn terminator);
        // <|end_of_text|> for Llama 3; <|endoftext|> for Phi-3;
        // </s> for Mistral 7B; <|user|> for ChatGLM 3 (the chat-tuned
        // turn boundary). chat_stream also breaks on secondary ids
        // (e.g. <|endoftext|> 151643 for Qwen) for defense-in-depth.
        let eos_token_id = match &arch {
            CausalArch::Qwen2(_) | CausalArch::Qwen3(_) => 151_645u32,
            CausalArch::Llama(_) => 128_001u32,
            CausalArch::Phi3(_) => 32_000u32,
            CausalArch::Mistral(_) => 2u32,
            CausalArch::ChatGlm(_) => 150_021u32,
        };

        let llama_cfg = match &arch {
            CausalArch::Llama(cfg) => Some(llama::LlamaConfig::clone(cfg).into_config(use_flash_attn)),
            _ => None,
        };
        let loaded = LoadedModel { model, tokenizer, eos_token_id, device: dev, llama_cfg, dtype: weight_dtype };
        let runtime = CandleLlmRuntime::new();
        *runtime.inner.lock().unwrap() = Some(loaded);

        Ok(Box::new(runtime))
    }

    async fn chat(&self, req: ChatRequest) -> Result<ChatResponse, NeuronError> {
        let (tokenizer, eos_token_id) = {
            let inner = self.inner.lock().unwrap();
            let m = inner.as_ref().ok_or_else(|| {
                NeuronError::not_initialized("CandleLlmRuntime not loaded. Call load() first.")
            })?;
            (m.tokenizer.clone(), m.eos_token_id)
        };

        // Reset the Llama KV cache at the start of every generation.
        {
            let mut inner = self.inner.lock().unwrap();
            let m = inner.as_mut().ok_or_else(|| {
                NeuronError::not_initialized("CandleLlmRuntime not loaded. Call load() first.")
            })?;
            Self::reset_llama_cache(m)?;
        }

        let prompt = Self::render_chatml_prompt(&req.messages);
        let encoding = tokenizer
            .encode(prompt.as_str(), true)
            .map_err(|e| NeuronError::not_implemented(format!("tokenize: {e}")))?;
        let prompt_token_ids: Vec<u32> = encoding.get_ids().to_vec();
        let prompt_tokens = prompt_token_ids.len() as u32;
        let max_new_tokens = req.max_tokens.unwrap_or(256).min(2048) as usize;

        // Real token generation: greedy argmax with the model's built-in KV cache.
        let mut inner = self.inner.lock().unwrap();
        let m = inner.as_mut().ok_or_else(|| {
            NeuronError::not_initialized("CandleLlmRuntime not loaded. Call load() first.")
        })?;

        let input_tensor = candle_core::Tensor::from_vec(
            prompt_token_ids.clone(),
            (1, prompt_token_ids.len()),
            &m.device,
        )
        .map_err(|e| NeuronError::not_implemented(format!("build input tensor: {e}")))?;
        eprintln!(
            "[debug] candle chat: prompt_tokens={}, device={:?}",
            prompt_token_ids.len(),
            m.device
        );

        let logits = m
            .model
            .forward(&input_tensor, 0)
            .map_err(|e| NeuronError::not_implemented(format!("forward pass 1: {e}")))?;
        eprintln!(
            "[debug] candle chat: forward(1) logits dims = {} (prompt_len={})",
            logits.dims().len(),
            prompt_token_ids.len()
        );
        // Candle's Llama::forward returns `[batch, vocab]` (the
        // last-token logits are already sliced), so a single `i(0)`
        // is enough — no need for the 3-element `(0, dim(1)-1, ..)`
        // tuple the chat loop used previously.
        let last_logits = logits
            .i(0)
            .map_err(|e| NeuronError::not_implemented(format!("index last logits: {e}")))?;
        let mut next_token = last_logits
            .argmax(candle_core::D::Minus1)
            .map_err(|e| NeuronError::not_implemented(format!("argmax: {e}")))?
            .to_scalar::<u32>()
            .map_err(|e| NeuronError::not_implemented(format!("to scalar: {e}")))?;

        let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
        // The prompt forward already yields the first generated token, so the
        // loop below contributes at most `max_new_tokens - 1` more. `max_tokens`
        // caps the total output; it is not additive with that first token.
        if max_new_tokens > 0 && next_token != eos_token_id {
            generated.push(next_token);
        }

        for offset in (prompt_token_ids.len() + 1)..(prompt_token_ids.len() + max_new_tokens) {
            if next_token == eos_token_id {
                break;
            }
            let next_input = candle_core::Tensor::from_vec(
                vec![next_token],
                (1, 1),
                &m.device,
            )
            .map_err(|e| NeuronError::not_implemented(format!("build next input: {e}")))?;
            let logits = m
                .model
                .forward(&next_input, offset)
                .map_err(|e| NeuronError::not_implemented(format!("forward pass {}: {}", offset, e)))?;
            // Same single-batch indexing: candle Llama::forward
            // returns `[batch, vocab]` even for the single-token
            // step.
            let last_logits = logits
                .i(0)
                .map_err(|e| NeuronError::not_implemented(format!("index last: {e}")))?;
            next_token = last_logits
                .argmax(candle_core::D::Minus1)
                .map_err(|e| NeuronError::not_implemented(format!("argmax: {e}")))?
                .to_scalar::<u32>()
                .map_err(|e| NeuronError::not_implemented(format!("to scalar: {e}")))?;
            if next_token != eos_token_id {
                generated.push(next_token);
            }
        }

        let output_text = tokenizer
            .decode(&generated, true)
            .unwrap_or_else(|_| format!("[decoding failed: {} tokens generated]", generated.len()));

        let completion_tokens = generated.len() as u32;

        Ok(ChatResponse {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: req.model,
            choices: vec![Choice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: output_text,
                },
                finish_reason: Some(if next_token == eos_token_id { "stop".to_string() } else { "length".to_string() }),
            }],
            usage: Some(super::Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            }),
        })
    }

    async fn chat_stream(
        &self,
        req: ChatRequest,
    ) -> super::ChatResult<futures_util::stream::BoxStream<'static, super::ChatChunk>> {
        let (tokenizer, eos_token_id) = {
            let inner = self.inner.lock().unwrap();
            let m = inner.as_ref().ok_or_else(|| {
                NeuronError::not_initialized("CandleLlmRuntime not loaded. Call load() first.")
            })?;
            (m.tokenizer.clone(), m.eos_token_id)
        };

        // Pre-encode the prompt so encoding errors return before the stream is
        // created (the stream itself has no way to surface an error mid-flight
        // because its Item type is `ChatChunk`, not `Result<ChatChunk, _>`).
        let prompt = Self::render_chatml_prompt(&req.messages);
        let prompt_ids: Vec<u32> = tokenizer
            .encode(prompt.as_str(), true)
            .map_err(|e| NeuronError::not_implemented(format!("tokenize: {e}")))?
            .get_ids()
            .to_vec();
        let prompt_len = prompt_ids.len();
        let max_new_tokens = req.max_tokens.unwrap_or(256).min(2048) as usize;
        let stop_seqs: Vec<String> = req.stop.clone().unwrap_or_default();
        // 0.0 means "no sampling" — the helper falls back to argmax.
        let temperature: f32 = req.temperature.unwrap_or(0.0);
        // 1.0 means "no nucleus truncation" (full distribution).
        let top_p: f32 = req.top_p.unwrap_or(1.0);
        // 0 means "no top-k truncation" (full vocab).
        let top_k: u32 = req.top_k.unwrap_or(0);

        let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        let created = chrono::Utc::now().timestamp();
        let model_id = req.model.clone();
        let object = "chat.completion.chunk".to_string();

        let (tx, rx) = tokio::sync::mpsc::channel::<super::ChatChunk>(16);
        let inner = self.inner.clone();

        std::thread::spawn(move || {
            // Hold the model lock for the entire generation: the candle Qwen
            // models carry an internal KV cache (ConcatKvCache) as part of
            // `self`, so concurrent forward passes would corrupt it. Holding
            // the lock end-to-end keeps the cache consistent.
            let mut guard = match inner.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            let m = match guard.as_mut() {
                Some(loaded) => loaded,
                None => {
                    let _ = tx.blocking_send(super::ChatChunk {
                        id: id.clone(),
                        object: object.clone(),
                        created,
                        model: model_id.clone(),
                        choices: vec![super::ChunkChoice {
                            index: 0,
                            delta: DeltaMessage { role: None, content: None },
                            finish_reason: Some("length".to_string()),
                        }],
                    });
                    return;
                }
            };

            // Same reset `chat()` performs: the Llama KV cache is
            // stateful, so a stream that follows a non-stream request
            // would otherwise fail its first forward and yield a 200
            // with a completely empty SSE body.
            if let Err(e) = Self::reset_llama_cache(m) {
                eprintln!("[candle chat_stream] cache reset failed: {e}");
                let _ = tx.blocking_send(super::ChatChunk {
                    id: id.clone(),
                    object: object.clone(),
                    created,
                    model: model_id.clone(),
                    choices: vec![super::ChunkChoice {
                        index: 0,
                        delta: DeltaMessage { role: None, content: None },
                        finish_reason: Some("length".to_string()),
                    }],
                });
                return;
            }

            let _ = tx.blocking_send(super::ChatChunk {
                id: id.clone(),
                object: object.clone(),
                created,
                model: model_id.clone(),
                choices: vec![super::ChunkChoice {
                    index: 0,
                    delta: DeltaMessage {
                        role: Some("assistant".to_string()),
                        content: None,
                    },
                    finish_reason: None,
                }],
            });

            let input = match candle_core::Tensor::from_vec(
                prompt_ids.clone(),
                (1, prompt_len),
                &m.device,
            ) {
                Ok(t) => t,
                Err(e) => {
                    // The stream's Item type is `ChatChunk`, not
                    // `Result<_, _>`, so a failure here can only end the
                    // stream. Log it: a silent empty SSE body is
                    // indistinguishable from a server that generated
                    // nothing, and cost hours to diagnose once already.
                    eprintln!("[candle chat_stream] input tensor failed: {e}");
                    return;
                }
            };
            let logits = match m.model.forward(&input, 0) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("[candle chat_stream] prompt forward failed: {e}");
                    return;
                }
            };
            // candle's Llama::forward returns `[batch, vocab]` — the
            // last-token logits are already sliced — so a single `i(0)`
            // is enough. The 3-tuple `(0, dim(1)-1, ..)` form this loop
            // used previously panicked with "index out of bounds: the
            // len is 2 but the index is 2" against a 2-D tensor.
            let last = match logits.i(0) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("[candle chat_stream] index prompt logits: {e}");
                    return;
                }
            };
            let logits_vec: Vec<f32> = match last.to_vec1() {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("[candle chat_stream] read prompt logits: {e}");
                    return;
                }
            };
            let mut next_token =
                Self::sample_token_from_logits(
                    &logits_vec, temperature, top_p, top_k,
                );
            let mut finished = next_token == eos_token_id;
            let mut count: usize = 0;
            let mut text_so_far = String::new();
            let mut generated: Vec<u32> = Vec::new();

            if !finished {
                let (old_len, decoded) = Self::decode_new_suffix(
                    &m.tokenizer,
                    &mut generated,
                    &mut text_so_far,
                    next_token,
                );

                if let Some(prefix_len) = Self::check_stop_suffix(&text_so_far, &stop_seqs) {
                    let emit_len = prefix_len.saturating_sub(old_len);
                    if emit_len > 0 {
                        let _ = tx.blocking_send(super::ChatChunk {
                            id: id.clone(),
                            object: object.clone(),
                            created,
                            model: model_id.clone(),
                            choices: vec![super::ChunkChoice {
                                index: 0,
                                delta: DeltaMessage {
                                    role: None,
                                    content: Some(text_so_far[old_len..old_len + emit_len].to_string()),
                                },
                                finish_reason: None,
                            }],
                        });
                    }
                    finished = true;
                } else {
                    let _ = tx.blocking_send(super::ChatChunk {
                        id: id.clone(),
                        object: object.clone(),
                        created,
                        model: model_id.clone(),
                        choices: vec![super::ChunkChoice {
                            index: 0,
                            delta: DeltaMessage {
                                role: None,
                                content: Some(decoded),
                            },
                            finish_reason: None,
                        }],
                    });
                }
                count += 1;
            }

            for offset in (prompt_len + 1)..(prompt_len + 1 + max_new_tokens) {
                if finished {
                    break;
                }
                if count >= max_new_tokens {
                    break;
                }

                let input = match candle_core::Tensor::from_vec(
                    vec![next_token],
                    (1, 1),
                    &m.device,
                ) {
                    Ok(t) => t,
                    Err(_) => return,
                };
                let logits = match m.model.forward(&input, offset) {
                    Ok(t) => t,
                    Err(_) => return,
                };
                // Same `[batch, vocab]` shape as the prompt forward.
                let last = match logits.i(0) {
                    Ok(t) => t,
                    Err(e) => {
                        eprintln!("[candle chat_stream] index step logits: {e}");
                        return;
                    }
                };
                let logits_vec: Vec<f32> = match last.to_vec1() {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("[candle chat_stream] read step logits: {e}");
                        return;
                    }
                };
                next_token =
                    Self::sample_token_from_logits(
                        &logits_vec, temperature, top_p, top_k,
                    );

                if next_token == eos_token_id {
                    finished = true;
                } else {
                    let (old_len, decoded) = Self::decode_new_suffix(
                        &m.tokenizer,
                        &mut generated,
                        &mut text_so_far,
                        next_token,
                    );

                    if let Some(prefix_len) = Self::check_stop_suffix(&text_so_far, &stop_seqs) {
                        let emit_len = prefix_len.saturating_sub(old_len);
                        if emit_len > 0 {
                            let _ = tx.blocking_send(super::ChatChunk {
                                id: id.clone(),
                                object: object.clone(),
                                created,
                                model: model_id.clone(),
                                choices: vec![super::ChunkChoice {
                                    index: 0,
                                    delta: DeltaMessage {
                                        role: None,
                                        content: Some(text_so_far[old_len..old_len + emit_len].to_string()),
                                    },
                                    finish_reason: None,
                                }],
                            });
                        }
                        finished = true;
                    } else {
                        let _ = tx.blocking_send(super::ChatChunk {
                            id: id.clone(),
                            object: object.clone(),
                            created,
                            model: model_id.clone(),
                            choices: vec![super::ChunkChoice {
                                index: 0,
                                delta: DeltaMessage {
                                    role: None,
                                    content: Some(decoded),
                                },
                                finish_reason: None,
                            }],
                        });
                    }
                    count += 1;
                }
            }

            let _ = tx.blocking_send(super::ChatChunk {
                id: id.clone(),
                object: object.clone(),
                created,
                model: model_id.clone(),
                choices: vec![super::ChunkChoice {
                    index: 0,
                    delta: DeltaMessage { role: None, content: None },
                    finish_reason: Some(if finished {
                        "stop".to_string()
                    } else {
                        "length".to_string()
                    }),
                }],
            });
        });

        let stream = async_stream::stream! {
            let mut rx = rx;
            while let Some(chunk) = rx.recv().await {
                yield chunk;
            }
        };

        Ok(Box::pin(stream))
    }

    fn list_models(&self) -> Vec<ModelInfo> {
        vec![
            ModelInfo::new("qwen3-0.6b", "neuron-candle"),
            ModelInfo::new("qwen3.5-0.6b", "neuron-candle"),
            ModelInfo::new("qwen3.5-4b", "neuron-candle"),
            ModelInfo::new("qwen2.5-0.5b", "neuron-candle"),
            ModelInfo::new("qwen2.5-1.5b", "neuron-candle"),
            ModelInfo::new("qwen2.5-7b", "neuron-candle"),
            ModelInfo::new("llama-3.2-1b", "neuron-candle"),
            ModelInfo::new("llama-3.2-3b", "neuron-candle"),
            ModelInfo::new("llama-3.1-8b", "neuron-candle"),
            ModelInfo::new("llama-2-7b", "neuron-candle"),
            ModelInfo::new("phi-3-mini-4k", "neuron-candle"),
            ModelInfo::new("phi-3-small-8k", "neuron-candle"),
            ModelInfo::new("phi-3-medium-4k", "neuron-candle"),
            ModelInfo::new("mistral-7b-v0.1", "neuron-candle"),
            ModelInfo::new("chatglm3-6b", "neuron-candle"),
        ]
    }

fn name(&self) -> &str {
        "candle"
    }
}

#[async_trait]
impl crate::engine::LlmRuntime for CandleLlmRuntime {
    async fn execute(&self, req: crate::canonical::CanonicalLlmRequest)
        -> crate::llm::ChatResult<crate::canonical::CanonicalLlmResponse>
    {
        let chat_req = crate::llm::translators::canonical_to_chat_request(&req)
            .map_err(NeuronError::new)?;
        let chat_resp = self.chat(chat_req).await?;
        Ok(crate::llm::translators::chat_response_to_canonical(&req.model, chat_resp))
    }

    async fn execute_stream(&self, req: crate::canonical::CanonicalLlmRequest)
        -> crate::llm::ChatResult<BoxStream<'static, crate::canonical::CanonicalLlmStreamEvent>>
    {
        let chat_req = crate::llm::translators::canonical_to_chat_request(&req)
            .map_err(NeuronError::new)?;
        let stream = self.chat_stream(chat_req).await?;
        use futures_util::StreamExt;
        Ok(Box::pin(stream.filter_map(|chunk| async move {
            chunk.choices.into_iter().next()
                .and_then(|c| c.delta.content)
                .filter(|s| !s.is_empty())
                .map(|t| crate::canonical::CanonicalLlmStreamEvent::ContentBlockDelta {
                    index: 0,
                    delta: crate::canonical::ContentDelta::TextDelta(t),
                })
        })))
    }

    fn capabilities(&self) -> &crate::capabilities::ModelCapabilities {
        static CAPS: std::sync::LazyLock<crate::capabilities::ModelCapabilities> =
            std::sync::LazyLock::new(|| crate::capabilities::ModelCapabilities {
                engine_impl: "candle".into(),
                model_id: "candle-default".into(),
                input_modalities: [crate::capabilities::Modality::TextInput].into_iter().collect(),
                output_modalities: [crate::capabilities::Modality::TextOutput].into_iter().collect(),
                api_styles: [crate::capabilities::ApiStyle::openai_chat()].into_iter().collect(),
                supports_streaming: true,
                supports_tools: true,
                ..Default::default()
            });
        &CAPS
    }

    fn name(&self) -> &str { "candle" }
}

#[cfg(test)]
#[path = "candle_runtime_tests.rs"]
mod tests;
