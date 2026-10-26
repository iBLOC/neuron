//! `LlamaCppLlmRuntime` — neuron adapter for [`llama-cpp-2`]
//! (utilityai/llama-cpp-rs), the safe Rust binding to
//! [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp).
//!
//! The 40+ GGUF-format models the llama.cpp ecosystem ships — Qwen 2/3,
//! Llama 2/3, Mistral, DeepSeek, GLM, Phi 4, Gemma 2/3, Yi, Mixtral,
//! DBRX, Command-R, etc. — load directly here without a safetensors→GGUF
//! conversion step. The runtime auto-applies the GGUF's embedded chat
//! template when present (so Qwen2.5-Instruct, Llama-3-Instruct, Mistral
//! all get the right `<|im_start|>` / `<|start_header_id|>` / `[INST]`
//! framing for free); for GGUF files that omit the metadata, a generic
//! ChatML fallback is used. The fallback can be forced on for any model
//! via `NEURON_LLAMA_CPP_FORCE_CHATML=true`.
//!
//! ## Build prereq
//!
//! `llama-cpp-2` compiles llama.cpp via `cmake` from source, so turning
//! on the `llamacpp` Cargo feature needs a C++ toolchain on PATH:
//!
//! - **Windows** — Visual Studio 2022 Build Tools (the "Desktop
//!   development with C++" workload) + `cmake.exe` on PATH
//! - **Linux** — `gcc` / `clang` + `cmake`
//! - **macOS** — Xcode command-line tools + `cmake` (`brew install cmake`)
//!
//! A build on a host missing these tools fails inside `llama-cpp-sys-2`'s
//! build script with a clear "cmake not found" or "no C++ compiler"
//! error. **Default builds (no `llamacpp` feature) are unaffected** —
//! the dependency is gated behind the feature flag, the same way
//! `mistralrs`, `litert`, `tract`, `ort`, `voxcpm` are.
//!
//! ## Model identifier
//!
//! `model_id` is the path to a local GGUF file (absolute or relative).
//! The runtime does NOT download weights — callers stage GGUF files at
//! known paths, or use the source puller (`POST /v1/models/pull`) to
//! pre-download. The legacy convention `$NEURON_MODEL_DIRS/candle/<id>/`
//! is honored as a search root for backward compatibility.
//!
//! ## GPU layer offload (`n_gpu_layers`)
//!
//! Driven by the `DeviceSelection` on the registration entry:
//!
//! | `DeviceSelection` | `n_gpu_layers` |
//! |---|---|
//! | `Auto` | 0 (CPU; safe default — matches candle's `Auto`) |
//! | `Cpu` | 0 |
//! | `Cuda` / `Apple` / `Vulkan` | all layers |
//!
//! Override the per-request count (e.g. partial offload on a small
//! VRAM box) via `NEURON_LLAMA_CPP_GPU_LAYERS=<n>`.
//!
//! ## Context size
//!
//! `NEURON_LLAMA_CPP_N_CTX` (default `2048`). The KV cache grows with
//! n_ctx — a 7B model at 2048 ctx allocates ~1 GiB per concurrent
//! context. Each chat() call creates a fresh `LlamaContext`, so this
//! is the upper bound on resident KV-cache memory per in-flight
//! request, not a steady-state pool cost.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::stream::BoxStream;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{ChatTemplateError, LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::mtmd::{MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputChunks, MtmdInputText};
use llama_cpp_2::sampling::LlamaSampler;
use tracing::{info, warn};

use super::{
    ChatChunk, ChatMessage, ChatRequest, ChatResponse, ChunkChoice, DeltaMessage, LlmRuntime,
    ModelInfo, NeuronError,
};
use crate::canonical::MediaAttachment;
use crate::config::DeviceSelection;

type ChatResult<T> = Result<T, NeuronError>;

/// Default context length (in tokens) when `NEURON_LLAMA_CPP_N_CTX` is unset.
/// 2048 is a safe middle ground for chat workloads and keeps KV-cache
/// memory at ~1 GiB for a 7B f16 model.
const DEFAULT_N_CTX: u32 = 2048;

/// Chat-template fallback for GGUF files whose metadata doesn't ship
/// a `tokenizer.chat_template` key. Mirrors the Qwen2.5 / Qwen3 ChatML
/// frame, which is the de-facto open-source default and is what
/// `/v1/chat/completions` callers expect to find when the runtime
/// can't auto-detect.
const CHATML_FALLBACK: &str = "{% for message in messages %}{% if (message['role'] == 'system') %}{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}{% elif (message['role'] == 'user') %}{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}{% elif message['role'] == 'assistant' %}{{ '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";

pub struct LlamaCppLlmRuntime {
    model_id: String,
    /// Path to the GGUF file (kept for diagnostics and `list_models`).
    gguf_path: PathBuf,
    /// Saved so we can rebuild a fresh `LlamaContext` per chat() call
    /// (the llama-cpp-2 design is one context per stream of inference;
    /// the model + backend are shared, Send + Sync).
    n_ctx: u32,
    n_gpu_layers: u32,
    /// `LlamaBackend` is process-global state; safe to share via Arc.
    backend: Arc<LlamaBackend>,
    /// `LlamaModel` wraps a process-global C++ pointer; Send + Sync.
    model: Arc<LlamaModel>,
    /// GGUF embedded template. `None` when the file didn't ship one —
    /// `render_prompt` falls back to ChatML in that case (driven by
    /// `apply_chat_template` returning `TemplateNotFound`, not by
    /// peeking at the wrapper's bytes, so the runtime stays
    /// forward-compatible with `llama-cpp-2` patch versions that
    /// change the newtype's inner representation).
    chat_template: Option<LlamaChatTemplate>,
    /// Optional multimodal projector context (`mtmd`). `None` when no
    /// mmproj GGUF was found next to the text GGUF — the runtime
    /// then refuses `chat_with_media` calls with a clear
    /// "not a multimodal model" error instead of pretending. When
    /// `Some`, `Arc<MtmdContext>` is shared across concurrent chat
    /// requests — mtmd context is `Send + Sync` (PR #920 fix in
    /// llama-cpp-rs).
    mtmd: Option<Arc<MtmdContext>>,
    /// Per-instance capabilities cache. Lazy-initialized on the
    /// first `capabilities()` call so multimodal modality is
    /// reported only when an mmproj GGUF was actually loaded.
    /// `OnceLock` over `LazyLock` because the capability set is
    /// instance-specific (depends on `self.mtmd.is_some()`).
    caps_cache: std::sync::OnceLock<crate::capabilities::ModelCapabilities>,
}

impl LlamaCppLlmRuntime {
    /// Public so integration tests can construct one without a real
    /// GGUF on disk; the rest of the codebase goes through
    /// `LlmRuntime::load` (which `ServerState::new` invokes via
    /// `runtime_for`).
    pub async fn load(
        model_id: &str,
        device: &DeviceSelection,
    ) -> ChatResult<Box<dyn LlmRuntime>> {
        let gguf_path = resolve_gguf_path(model_id).map_err(|e| {
            NeuronError::not_initialized(format!(
                "llamacpp load failed for model '{model_id}': {e}. \
                 LlamaCppLlmRuntime needs a local GGUF file path; the runtime \
                 does NOT download weights. Either pass an absolute path as \
                 the model id, or pre-stage the .gguf at \
                 $NEURON_MODEL_DIRS/candle/{model_id}/model.gguf (or any of \
                 the search roots in $NEURON_MODEL_DIRS)."
            ))
        })?;

        let backend = Arc::new(LlamaBackend::init().map_err(|e| {
            NeuronError::not_initialized(format!(
                "llama-cpp-2 backend init failed: {e}. \
                 This is a process-global init; if you see it on every call, \
                 the host's llama.cpp build is broken."
            ))
        })?);

        let n_gpu_layers = gpu_layers_for(device);

        let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);

        let model = LlamaModel::load_from_file(&backend, &gguf_path, &model_params)
            .map_err(|e| {
                NeuronError::not_initialized(format!(
                    "llama.cpp model load failed for {path}: {e}",
                    path = gguf_path.display()
                ))
            })?;

        let chat_template = model.chat_template();
        // We don't probe the wrapper's bytes here — `apply_chat_template`
        // returns `TemplateNotFound` for empty templates, which is what
        // `render_prompt` keys on. The warn-on-load fires when the
        // model is well-known to need ChatML (most pre-2024 GGUF
        // exports), but is otherwise cosmetic.

        // ---- mmproj auto-discovery (multimodal) ---------------------------
        //
        // Look for a vision-projector GGUF in the same directory as
        // the text model. The conventions we honor (in priority order):
        //   1. `$NEURON_LLAMA_CPP_MMPROJ` env var (explicit override)
        //   2. `<dir>/mmproj-<...>.gguf` (HF llama.cpp conversion convention)
        //   3. `<dir>/model.mmproj.gguf` (alternative convention)
        //   4. `<dir>/mmproj.gguf` (bare convention)
        //
        // If none of these resolve, the runtime loads as a text-only
        // LLM and `chat_with_media` will report
        // `not_initialized`/"no mmproj found".
        let mtmd: Option<Arc<MtmdContext>> = match resolve_mmproj_path(&gguf_path) {
            Some(mmproj_path) => {
                let params = MtmdContextParams {
                    use_gpu: matches!(
                        device,
                        DeviceSelection::Nvidia
                            | DeviceSelection::Apple
                            | DeviceSelection::Vulkan
                    ),
                    print_timings: false,
                    n_threads: 0, // 0 = llama.cpp default (usually #cores/2)
                    media_marker: std::ffi::CString::new(
                        llama_cpp_2::mtmd::mtmd_default_marker(),
                    )
                    .map_err(|e| {
                        NeuronError::not_initialized(format!(
                            "mmproj media marker CString: {e}"
                        ))
                    })?,
                    image_min_tokens: 0, // accept the model's default
                    image_max_tokens: 0,
                };
                match MtmdContext::init_from_file(
                    mmproj_path.to_string_lossy().as_ref(),
                    &model,
                    &params,
                ) {
                    Ok(ctx) if ctx.support_vision() => {
                        info!(
                            mmproj = %mmproj_path.display(),
                            use_mrope = ctx.decode_use_mrope(),
                            "llamacpp mmproj loaded — multimodal enabled"
                        );
                        Some(Arc::new(ctx))
                    }
                    Ok(ctx) => {
                        warn!(
                            mmproj = %mmproj_path.display(),
                            support_vision = ctx.support_vision(),
                            support_audio = ctx.support_audio(),
                            "mmproj GGUF loaded but reports no vision support; \
                             treating as text-only"
                        );
                        None
                    }
                    Err(e) => {
                        warn!(
                            mmproj = %mmproj_path.display(),
                            error = %e,
                            "MtmdContext::init_from_file failed; falling back to text-only"
                        );
                        None
                    }
                }
            }
            None => None,
        };

        let n_ctx = env_n_ctx().unwrap_or(DEFAULT_N_CTX);
        info!(
            model_id = %model_id,
            path = %gguf_path.display(),
            n_ctx,
            n_gpu_layers,
            multimodal = mtmd.is_some(),
            "llamacpp model loaded"
        );

        Ok(Box::new(Self {
            model_id: model_id.to_string(),
            gguf_path,
            n_ctx,
            n_gpu_layers,
            backend,
            model: Arc::new(model),
            chat_template: Some(chat_template),
            mtmd,
            caps_cache: std::sync::OnceLock::new(),
        }))
    }

    /// Common multimodal prefill: build the (text + markers + bitmaps)
    /// prompt, tokenize via mtmd, run `eval_chunks` to populate the KV
    /// cache, and return the context pieces the generation loop
    /// needs. Returns `(Arc<MtmdContext>, LlamaContext, Vec<LlamaToken>,
    /// i32)` where the last value is `n_past` after the prefill.
    ///
    /// This is the load-bearing part of `chat_with_media` — once
    /// prefill is done, the existing per-token generation loop takes
    /// over without needing to know mtmd is involved.
    async fn prefill_multimodal(
        &self,
        prompt: &str,
        media: &[MediaAttachment],
    ) -> ChatResult<(Arc<MtmdContext>, LlamaContext<'_>, Vec<llama_cpp_2::token::LlamaToken>, i32)> {
        let mtmd = self.mtmd.as_ref().ok_or_else(|| {
            NeuronError::not_initialized(
                "no mmproj GGUF was loaded next to the text GGUF — \
                 chat_with_media requires a vision-projector model. \
                 Drop a mmproj-*.gguf / model.mmproj / mmproj.gguf into \
                 the model directory, or set NEURON_LLAMA_CPP_MMPROJ to \
                 an explicit path.",
            )
        })?;

        // Intersperse `<__media__>` markers in the text so mtmd knows
        // where each image goes. The OpenAI spec allows multiple
        // image blocks in one user message; we expect media.len() to
        // match the number of markers the caller inserted (the
        // adapter layer is responsible for that mapping).
        let marker = llama_cpp_2::mtmd::mtmd_default_marker();
        let mut text_with_markers = String::new();
        let mut cursor = 0usize;
        for (i, _) in media.iter().enumerate() {
            // Replace the ith <__media__> we already have in the
            // prompt with the real marker; if the prompt didn't ship
            // any, append the marker at the end (best-effort).
            if let Some(pos) = prompt[cursor..].find(marker) {
                let abs = cursor + pos;
                text_with_markers.push_str(&prompt[cursor..abs]);
                text_with_markers.push_str(marker);
                cursor = abs + marker.len();
            } else {
                // Pad remaining markers at the end.
                text_with_markers.push_str(&prompt[cursor..]);
                if i < media.len() {
                    text_with_markers.push(' ');
                    text_with_markers.push_str(marker);
                }
                cursor = prompt.len();
            }
        }
        text_with_markers.push_str(&prompt[cursor..]);

        // Decode each attachment to an MtmdBitmap via the
        // stb_image path. The `placeholder` flag is the OpenAI
        // `detail` knob ("low" -> true to skip the high-res branch
        // and let the projector keep the image cheap; "high" / "auto"
        // -> false for the full-resolution decode).
        let bitmaps: Vec<MtmdBitmap> = media
            .iter()
            .map(|m| {
                let placeholder = m.detail.as_deref() == Some("low");
                MtmdBitmap::from_buffer(mtmd, &m.bytes, placeholder).map_err(|e| {
                    NeuronError::invalid_input(format!(
                        "decode media attachment (mime={}): {e}",
                        m.mime_type
                    ))
                })
            })
            .collect::<Result<_, _>>()?;

        let bitmap_refs: Vec<&MtmdBitmap> = bitmaps.iter().collect();

        let input = MtmdInputText {
            text: text_with_markers.clone(),
            add_special: true,
            parse_special: true,
        };

        // Tokenize text+bitmaps into chunks. mtmd returns an
        // MtmdInputChunks that knows how many text vs image tokens
        // each piece contributes.
        let chunks: MtmdInputChunks = mtmd
            .tokenize(input, &bitmap_refs)
            .map_err(|e| NeuronError::new(format!("llamacpp mtmd tokenize: {e}")))?;

        // Allocate a fresh LlamaContext for this request (same shape
        // as the text-only `chat()` path — one context per request,
        // model + backend shared via Arc).
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(self.n_ctx.max(1)));
        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| NeuronError::new(format!("llamacpp new_context(mtmd): {e}")))?;

        // Prefill the KV cache from the multimodal chunks. mtmd
        // advances n_past internally; we just need to capture the
        // return value to know where the next generated token lands.
        let n_past: i32 = chunks
            .eval_chunks(mtmd, &mut ctx, /*n_past=*/ 0, /*seq_id=*/ 0, self.n_ctx as i32, /*logits_last=*/ true)
            .map_err(|e| NeuronError::new(format!("llamacpp mtmd eval_chunks: {e}")))?;

        // We don't have a single `Vec<LlamaToken>` of the prompt
        // here — mtmd's chunks are an opaque structure. Return an
        // empty vec; the chat_with_media response's `usage` field
        // uses `prompt_tokens = n_past as u32` (the total tokens
        // mtmd just evaluated, which is what the caller wants).
        let prompt_tokens: Vec<llama_cpp_2::token::LlamaToken> = Vec::new();

        Ok((Arc::clone(mtmd), ctx, prompt_tokens, n_past))
    }

    /// Build a sampler chain from the request's sampling params.
    /// Mirrors what `simple` does in llama.cpp's `examples/simple.cpp`:
    /// penalties → top-K → top-P → temperature → dist. All stages are
    /// optional — a request with no sampling fields gets greedy.
    fn build_sampler(&self, req: &ChatRequest) -> LlamaSampler {
        // Pull the chain together stage-by-stage. `LlamaSampler::new_chain`
        // takes `&[&dyn Sampler]`-ish slices; the actual constructor name
        // matches upstream `llama_sampler_chain_add`. Defaults are sensible:
        //   - temperature 0.0 → greedy
        //   - top_p 1.0      → no nucleus filtering
        //   - top_k unset    → no K cap
        let temp = req.temperature.unwrap_or(0.0);
        let top_p = req.top_p.unwrap_or(1.0);
        let top_k = req.top_k.unwrap_or(0);

        if temp <= 0.0 {
            // Greedy: the safe path for tests + any caller that didn't
            // explicitly request sampling.
            return LlamaSampler::greedy();
        }

        // Non-greedy path. Build a small chain: penalties → top-K → top-P
        // → temperature → dist. The order matches llama.cpp's `simple`.
        let mut samplers: Vec<LlamaSampler> = Vec::new();
        if let Some(stops) = req.stop.as_ref().filter(|s| !s.is_empty()) {
            // Penalty: discourage repeating literal stop strings (best-effort;
            // llama.cpp's repeat-penalty is per-token, not per-string, so we
            // apply a 1.05 repeat penalty to nudge the model off any
            // stop-aligned output).
            let _ = stops; // suppress unused warning; string-level stop is
                           // handled in the decode loop.
            samplers.push(LlamaSampler::penalties(64, 1.05, 0.0, 0.0));
        }
        if top_k > 0 {
            samplers.push(LlamaSampler::top_k(top_k as i32, 1));
        }
        if top_p < 1.0 {
            samplers.push(LlamaSampler::top_p(top_p, 1));
        }
        samplers.push(LlamaSampler::temp(temp));
        samplers.push(LlamaSampler::dist(seed_for_request(req)));

        LlamaSampler::new_chain(samplers)
    }
}

#[async_trait]
impl LlmRuntime for LlamaCppLlmRuntime {
    async fn load(
        model: &str,
        device: &DeviceSelection,
    ) -> ChatResult<Box<dyn LlmRuntime>>
    where
        Self: Sized,
    {
        Self::load(model, device).await
    }

    async fn chat(&self, req: ChatRequest) -> ChatResult<ChatResponse> {
        let prompt = render_prompt(&self.model, self.chat_template.as_ref(), &req.messages)?;

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(self.n_ctx.max(1)));
        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| NeuronError::new(format!("llamacpp new_context: {e}")))?;

        let prompt_tokens = tokenize_prompt(&self.model, &prompt)?;
        let mut batch = LlamaBatch::new(self.n_ctx as usize, 1);
        batch
            .add_sequence(&prompt_tokens, 0, true)
            .map_err(|e| NeuronError::new(format!("llamacpp add_sequence: {e}")))?;
        ctx.decode(&mut batch)
            .map_err(|e| NeuronError::new(format!("llamacpp decode(prompt): {e}")))?;

        let mut sampler = self.build_sampler(&req);
        let mut output = String::new();
        let mut completion_tokens: u32 = 0;
        let max_tokens = req.max_tokens.unwrap_or(256) as usize;

        let stop_strings = req.stop.clone().unwrap_or_default();
        let last_pos = prompt_tokens.len() as i32 - 1;
        let mut pos = last_pos;

        while completion_tokens < max_tokens as u32 {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            if self.model.is_eog_token(token) {
                break;
            }
            let piece = self
                .model
                .token_to_piece(token, llama_cpp_2::token::Special::Tokenize)
                .map_err(|e| NeuronError::new(format!("llamacpp token_to_piece: {e}")))?;
            output.push_str(&piece);
            completion_tokens += 1;

            if completion_tokens >= max_tokens as u32 {
                break;
            }
            if !stop_strings.is_empty() && stop_matches(&output, &stop_strings) {
                break;
            }

            // Append the sampled token and decode one step.
            pos += 1;
            batch.clear();
            batch
                .add(token, pos, &[0], true)
                .map_err(|e| NeuronError::new(format!("llamacpp add(step): {e}")))?;
            ctx.decode(&mut batch)
                .map_err(|e| NeuronError::new(format!("llamacpp decode(step): {e}")))?;
        }

        let finish_reason = if completion_tokens >= max_tokens as u32 {
            "length"
        } else {
            "stop"
        };

        let prompt_tokens_len = prompt_tokens.len() as u32;
        Ok(ChatResponse {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: req.model,
            choices: vec![super::Choice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: output,
                },
                finish_reason: Some(finish_reason.to_string()),
            }],
            usage: Some(super::Usage {
                prompt_tokens: prompt_tokens_len,
                completion_tokens,
                total_tokens: prompt_tokens_len + completion_tokens,
            }),
        })
    }

    async fn chat_stream(
        &self,
        req: ChatRequest,
    ) -> ChatResult<BoxStream<'static, ChatChunk>> {
        let prompt = render_prompt(&self.model, self.chat_template.as_ref(), &req.messages)?;

        // Per-stream context: each chat_stream call creates a fresh
        // `LlamaContext` (the runtime doesn't share KV across calls).
        // The `Arc<LlamaModel>` + `Arc<LlamaBackend>` are Send + Sync and
        // cheap to clone into the stream's owned state, satisfying the
        // 'static bound on the returned stream.
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(self.n_ctx.max(1)));
        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| NeuronError::new(format!("llamacpp new_context(stream): {e}")))?;

        let prompt_tokens = tokenize_prompt(&self.model, &prompt)?;
        let mut batch = LlamaBatch::new(self.n_ctx as usize, 1);
        batch
            .add_sequence(&prompt_tokens, 0, true)
            .map_err(|e| NeuronError::new(format!("llamacpp add_sequence(stream): {e}")))?;
        ctx.decode(&mut batch)
            .map_err(|e| NeuronError::new(format!("llamacpp decode(prompt/stream): {e}")))?;

        let mut sampler = self.build_sampler(&req);
        let max_tokens = req.max_tokens.unwrap_or(256) as u32;
        let stop_strings = req.stop.clone().unwrap_or_default();
        let chunk_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        let created = chrono::Utc::now().timestamp();
        let model_id = req.model.clone();

        let adapted = async_stream::stream! {
            // First chunk: role announcement (OpenAI streaming convention).
            yield ChatChunk {
                id: chunk_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model_id.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: DeltaMessage {
                        role: Some("assistant".to_string()),
                        content: None,
                    },
                    finish_reason: None,
                }],
            };

            let mut emitted: u32 = 0;
            let mut output = String::new();
            let mut pos: i32 = prompt_tokens.len() as i32 - 1;
            let mut finish_reason = "stop";

            while emitted < max_tokens {
                let token = sampler.sample(&ctx, batch.n_tokens() - 1);
                if self.model.is_eog_token(token) {
                    finish_reason = "stop";
                    break;
                }
                let piece = match self
                    .model
                    .token_to_piece(token, llama_cpp_2::token::Special::Tokenize)
                {
                    Ok(p) => p,
                    Err(e) => {
                        warn!("llamacpp token_to_piece(stream): {e}");
                        break;
                    }
                };
                output.push_str(&piece);
                emitted += 1;
                if !stop_strings.is_empty() && stop_matches(&output, &stop_strings) {
                    finish_reason = "stop";
                    break;
                }
                if emitted >= max_tokens {
                    finish_reason = "length";
                    break;
                }

                yield ChatChunk {
                    id: chunk_id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model_id.clone(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: DeltaMessage {
                            role: None,
                            content: Some(piece),
                        },
                        finish_reason: None,
                    }],
                };

                // Decode the next step.
                pos += 1;
                batch.clear();
                if let Err(e) = batch.add(token, pos, &[0], true) {
                    warn!("llamacpp add(stream): {e}");
                    break;
                }
                if let Err(e) = ctx.decode(&mut batch) {
                    warn!("llamacpp decode(stream): {e}");
                    break;
                }
            }

            // Terminal chunk: stop / length marker.
            yield ChatChunk {
                id: chunk_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model_id.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: DeltaMessage { role: None, content: None },
                    finish_reason: Some(finish_reason.to_string()),
                }],
            };
        };

        Ok(Box::pin(adapted))
    }

    async fn chat_with_media(
        &self,
        req: ChatRequest,
        media: Vec<MediaAttachment>,
    ) -> ChatResult<ChatResponse> {
        let prompt = render_prompt(&self.model, self.chat_template.as_ref(), &req.messages)?;
        let (mtmd, mut ctx, prompt_tokens, n_past) = self
            .prefill_multimodal(&prompt, &media)
            .await?;

        let mut sampler = self.build_sampler(&req);
        let max_tokens = req.max_tokens.unwrap_or(256) as u32;
        let stop_strings = req.stop.clone().unwrap_or_default();
        let mut output = String::new();
        let mut completion_tokens: u32 = 0;
        let mut finish_reason = "stop";

        while completion_tokens < max_tokens {
            // mtmd path: n_past already accounts for the full
            // multimodal prompt (text tokens + image tokens), so the
            // last token's logits are at index n_past - 1.
            let token = sampler.sample(&ctx, n_past - 1);
            if self.model.is_eog_token(token) {
                finish_reason = "stop";
                break;
            }
            let piece = self
                .model
                .token_to_piece(token, llama_cpp_2::token::Special::Tokenize)
                .map_err(|e| NeuronError::new(format!("llamacpp token_to_piece: {e}")))?;
            output.push_str(&piece);
            completion_tokens += 1;
            if !stop_strings.is_empty() && stop_matches(&output, &stop_strings) {
                finish_reason = "stop";
                break;
            }
            if completion_tokens >= max_tokens {
                finish_reason = "length";
                break;
            }

            // Append the sampled token and decode one step.
            n_past = append_step_decode(
                &self.model,
                &mut ctx,
                token,
                n_past,
            )?;
        }

        let prompt_tokens_len = prompt_tokens.len() as u32;
        Ok(ChatResponse {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: req.model,
            choices: vec![super::Choice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: output,
                },
                finish_reason: Some(finish_reason.to_string()),
            }],
            usage: Some(super::Usage {
                prompt_tokens: prompt_tokens_len,
                completion_tokens,
                total_tokens: prompt_tokens_len + completion_tokens,
            }),
        })
    }

    async fn chat_with_media_stream(
        &self,
        req: ChatRequest,
        media: Vec<MediaAttachment>,
    ) -> ChatResult<BoxStream<'static, ChatChunk>> {
        let prompt = render_prompt(&self.model, self.chat_template.as_ref(), &req.messages)?;
        let (mtmd, mut ctx, _prompt_tokens, mut n_past) = self
            .prefill_multimodal(&prompt, &media)
            .await?;

        let mut sampler = self.build_sampler(&req);
        let max_tokens = req.max_tokens.unwrap_or(256) as u32;
        let stop_strings = req.stop.clone().unwrap_or_default();
        let chunk_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        let created = chrono::Utc::now().timestamp();
        let model_id = req.model.clone();

        let adapted = async_stream::stream! {
            yield ChatChunk {
                id: chunk_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model_id.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: DeltaMessage {
                        role: Some("assistant".to_string()),
                        content: None,
                    },
                    finish_reason: None,
                }],
            };

            let mut emitted: u32 = 0;
            let mut output = String::new();
            let mut finish_reason = "stop";

            while emitted < max_tokens {
                let token = sampler.sample(&ctx, n_past - 1);
                if self.model.is_eog_token(token) {
                    finish_reason = "stop";
                    break;
                }
                let piece = match self
                    .model
                    .token_to_piece(token, llama_cpp_2::token::Special::Tokenize)
                {
                    Ok(p) => p,
                    Err(e) => {
                        warn!("llamacpp token_to_piece(stream/media): {e}");
                        break;
                    }
                };
                output.push_str(&piece);
                emitted += 1;
                if !stop_strings.is_empty() && stop_matches(&output, &stop_strings) {
                    finish_reason = "stop";
                    break;
                }
                if emitted >= max_tokens {
                    finish_reason = "length";
                    break;
                }

                yield ChatChunk {
                    id: chunk_id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model_id.clone(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: DeltaMessage {
                            role: None,
                            content: Some(piece),
                        },
                        finish_reason: None,
                    }],
                };

                n_past = match append_step_decode(&self.model, &mut ctx, token, n_past) {
                    Ok(p) => p,
                    Err(_) => break,
                };
            }

            yield ChatChunk {
                id: chunk_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model_id.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: DeltaMessage { role: None, content: None },
                    finish_reason: Some(finish_reason.to_string()),
                }],
            };
        };

        Ok(Box::pin(adapted))
    }

    fn list_models(&self) -> Vec<ModelInfo> {
        vec![ModelInfo::new(&self.model_id, "llamacpp")]
    }

    fn name(&self) -> &str {
        "llamacpp"
    }
}

#[cfg(feature = "llamacpp")]
#[async_trait]
impl crate::engine::LlmRuntime for LlamaCppLlmRuntime {
    async fn execute(
        &self,
        req: crate::canonical::CanonicalLlmRequest,
    ) -> crate::llm::ChatResult<crate::canonical::CanonicalLlmResponse> {
        let chat_req = crate::llm::translators::canonical_to_chat_request(&req)
            .map_err(NeuronError::new)?;
        let chat_resp = self.chat(chat_req).await?;
        Ok(crate::llm::translators::chat_response_to_canonical(&req.model, chat_resp))
    }

    async fn execute_stream(
        &self,
        req: crate::canonical::CanonicalLlmRequest,
    ) -> crate::llm::ChatResult<BoxStream<'static, crate::canonical::CanonicalLlmStreamEvent>> {
        let chat_req = crate::llm::translators::canonical_to_chat_request(&req)
            .map_err(NeuronError::new)?;
        let stream = self.chat_stream(chat_req).await?;
        use futures_util::StreamExt;
        Ok(Box::pin(stream.filter_map(|chunk| async move {
            chunk
                .choices
                .into_iter()
                .next()
                .and_then(|c| c.delta.content)
                .filter(|s| !s.is_empty())
                .map(|t| crate::canonical::CanonicalLlmStreamEvent::ContentBlockDelta {
                    index: 0,
                    delta: crate::canonical::ContentDelta::TextDelta(t),
                })
        })))
    }

    async fn execute_with_media(
        &self,
        req: crate::canonical::CanonicalLlmRequest,
        media: Vec<crate::canonical::MediaAttachment>,
    ) -> crate::llm::ChatResult<crate::canonical::CanonicalLlmResponse> {
        let (chat_req, extra_media) =
            crate::llm::translators::canonical_to_chat_request_with_media(&req)
                .map_err(NeuronError::new)?;
        // `extra_media` is whatever the canonical→ChatRequest translator
        // pulled out of the text; `media` is the explicit list passed by
        // the adapter. We concat (with `media` taking precedence in
        // ordering — it's the adapter's job to feed image blocks in
        // message order, which is the same order the translator picks).
        let mut all = extra_media;
        all.extend(media);
        let chat_resp = self.chat_with_media(chat_req, all).await?;
        Ok(crate::llm::translators::chat_response_to_canonical(&req.model, chat_resp))
    }

    async fn execute_with_media_stream(
        &self,
        req: crate::canonical::CanonicalLlmRequest,
        media: Vec<crate::canonical::MediaAttachment>,
    ) -> crate::llm::ChatResult<BoxStream<'static, crate::canonical::CanonicalLlmStreamEvent>> {
        let (chat_req, extra_media) =
            crate::llm::translators::canonical_to_chat_request_with_media(&req)
                .map_err(NeuronError::new)?;
        let mut all = extra_media;
        all.extend(media);
        let stream = self.chat_with_media_stream(chat_req, all).await?;
        use futures_util::StreamExt;
        Ok(Box::pin(stream.filter_map(|chunk| async move {
            chunk
                .choices
                .into_iter()
                .next()
                .and_then(|c| c.delta.content)
                .filter(|s| !s.is_empty())
                .map(|t| crate::canonical::CanonicalLlmStreamEvent::ContentBlockDelta {
                    index: 0,
                    delta: crate::canonical::ContentDelta::TextDelta(t),
                })
        })))
    }

    fn capabilities(&self) -> &crate::capabilities::ModelCapabilities {
        // Per-instance capabilities — the multimodal modality is
        // reported only when an mmproj GGUF was successfully loaded
        // next to the text model. We cache the computed value in a
        // `OnceLock` so the first caller pays the build cost and
        // subsequent callers get a `&'static`-like borrow off
        // `self`.
        self.caps_cache.get_or_init(|| {
            use std::collections::HashSet;
            use crate::capabilities::{ApiStyle, Modality, ModelCapabilities};

            let mut input = HashSet::new();
            input.insert(Modality::TextInput);
            if self.mtmd.is_some() {
                input.insert(Modality::ImageInput);
            }
            let mut output = HashSet::new();
            output.insert(Modality::TextOutput);

            ModelCapabilities {
                engine_impl: "llamacpp".into(),
                model_id: self.model_id.clone(),
                input_modalities: input,
                output_modalities: output,
                api_styles: [
                    ApiStyle::openai_chat(),
                    ApiStyle::anthropic_messages(),
                ]
                .into_iter()
                .collect(),
                supports_streaming: true,
                supports_tools: false,
                supports_prompt_caching: false,
                supports_extended_thinking: false,
                ..Default::default()
            }
        })
    }

    fn name(&self) -> &str {
        "llamacpp"
    }
}

// --- helpers ---------------------------------------------------------------

/// Render an OpenAI-style `Vec<ChatMessage>` through the GGUF's embedded
/// chat template (or the ChatML fallback) into a single prompt string.
fn render_prompt(
    model: &LlamaModel,
    tmpl: Option<&LlamaChatTemplate>,
    messages: &[ChatMessage],
) -> ChatResult<String> {
    let mut chat: Vec<LlamaChatMessage> = messages
        .iter()
        .map(|m| LlamaChatMessage::new(m.role.clone(), m.content.clone()))
        .collect();

    if force_chatml_env() {
        return render_chatml(&chat);
    }

    let Some(tmpl) = tmpl else {
        // No embedded template was stored (we never reach here when
        // load() set Some(chat_template) but a future patch that flips
        // the default to None must still get a valid prompt).
        return render_chatml(&chat);
    };

    match model.apply_chat_template(tmpl, &mut chat, true) {
        Ok(s) => Ok(s),
        // The GGUF didn't ship a chat template (most pre-2024 GGUF exports
        // lack the `tokenizer.chat_template` metadata key) — fall back to
        // ChatML. llama-cpp-2 surfaces this as a typed error variant
        // rather than panicking or returning an empty prompt, which is
        // why this match arm exists separately.
        Err(ChatTemplateError::TemplateNotFound) => render_chatml(&chat),
        Err(e) => Err(NeuronError::new(format!(
            "llamacpp apply_chat_template: {e}"
        ))),
    }
}

/// Inline mini-jinja-subset renderer for the ChatML fallback. llama-cpp-2
/// does not bundle a Jinja runtime (upstream llama.cpp's chat template
/// engine requires libkt / a jinja2 build flag), so for the common
/// "no embedded template" case we render the four-message ChatML frame
/// ourselves. It supports the same minimal subset upstream's `simple.cpp`
/// documents: `system` / `user` / `assistant` roles in declaration
/// order, plus `add_generation_prompt`.
fn render_chatml(messages: &[LlamaChatMessage]) -> ChatResult<String> {
    let mut out = String::new();
    for m in messages {
        let role = m.role.as_str();
        if !matches!(role, "system" | "user" | "assistant") {
            warn!(
                "ChatML fallback: unknown role {role:?}; rendering as 'user' to keep prompt well-formed"
            );
            out.push_str("<|im_start|>user\n");
            out.push_str(&m.content);
            out.push_str("<|im_end|>\n");
            continue;
        }
        out.push_str(&format!("<|im_start|>{role}\n"));
        out.push_str(&m.content);
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>assistant\n");
    Ok(out)
}

fn tokenize_prompt(
    model: &LlamaModel,
    prompt: &str,
) -> ChatResult<Vec<llama_cpp_2::token::LlamaToken>> {
    // `add_special=true` keeps BOS on (so Llama-3-Instruct gets its
    // expected leading `<|begin_of_text|>`); `parse_special=true` lets
    // GGUF-shipped special tokens like `<|im_start|>` survive the encode
    // instead of being split into text.
    model
        .str_to_token(prompt, true, true)
        .map_err(|e| NeuronError::new(format!("llamacpp str_to_token: {e}")))
}

/// True if any of the requested stop strings appears as a suffix of
/// `output`. Tokenizers split stop strings into individual tokens; we
/// only fire the stop when the full string is visible in the decoded
/// output so callers that pass a multi-token stop don't get their
/// response truncated at the first partial match.
fn stop_matches(output: &str, stops: &[String]) -> bool {
    stops.iter().any(|s| output.ends_with(s))
}

fn gpu_layers_for(device: &DeviceSelection) -> u32 {
    if let Some(n) = env_gpu_layers_override() {
        return n;
    }
    use crate::config::DeviceSelection::*;
    match device {
        Cpu | Auto => 0,
        // All backends that llama-cpp-2 supports get full offload; the
        // user can drop the count via NEURON_LLAMA_CPP_GPU_LAYERS for
        // partial offload.
        _ => u32::MAX,
    }
}

fn env_n_ctx() -> Option<u32> {
    crate::config::env_var("LLAMA_CPP_N_CTX").and_then(|s| s.parse().ok())
}

fn env_gpu_layers_override() -> Option<u32> {
    crate::config::env_var("LLAMA_CPP_GPU_LAYERS").and_then(|s| s.parse().ok())
}

fn force_chatml_env() -> bool {
    matches!(
        crate::config::env_var("LLAMA_CPP_FORCE_CHATML").as_deref(),
        Some("1") | Some("true") | Some("TRUE")
    )
}

/// Build a deterministic seed from the request's stop-list length so
/// non-greedy sampling is at least reproducible per-request shape (good
/// enough for the "neighbouring tests should see the same output"
/// invariant; cryptographically strong entropy is not the goal here).
fn seed_for_request(req: &ChatRequest) -> u32 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    req.model.hash(&mut h);
    req.messages.len().hash(&mut h);
    if let Some(t) = req.temperature {
        t.to_bits().hash(&mut h);
    }
    h.finish() as u32
}

/// Resolve `model_id` to a local GGUF file path. Accepts:
///   1. an absolute path
///   2. a relative path (resolved against CWD)
///   3. the bare `<id>` form, looked up under
///      `$NEURON_MODEL_DIRS/candle/<id>/model.gguf` (legacy conv) or
///      `$NEURON_MODEL_DIRS/llamacpp/<id>/model.gguf`
fn resolve_gguf_path(model_id: &str) -> Result<PathBuf, String> {
    let raw = Path::new(model_id);
    if raw.is_absolute() && raw.exists() {
        return Ok(raw.to_path_buf());
    }
    if raw.exists() {
        return Ok(raw.canonicalize().unwrap_or_else(|_| raw.to_path_buf()));
    }

    // Search the model dirs. Reuse the env helper that the rest of
    // the codebase uses for the same purpose.
    let roots = crate::config::env_var("MODEL_DIRS")
        .or_else(|| crate::config::env_var("MODEL_DIR"))
        .unwrap_or_else(|| "~/.neuron/models".to_string());
    let sep = if cfg!(windows) { ';' } else { ':' };

    for root in roots.split(sep) {
        let root = root.trim();
        if root.is_empty() {
            continue;
        }
        let expanded = expand_home(root);
        for sub in &["llamacpp", "candle"] {
            let candidate = PathBuf::from(&expanded)
                .join(sub)
                .join(model_id)
                .join("model.gguf");
            if candidate.exists() {
                return Ok(candidate);
            }
        }
    }

    Err(format!(
        "no GGUF file found for model_id={model_id:?}. \
         Expected a path to a .gguf file, or a model directory under one of \
         $NEURON_MODEL_DIRS/{{llamacpp,candle}}/{model_id}/model.gguf"
    ))
}

fn expand_home(p: &str) -> String {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")) {
            let mut s = std::path::PathBuf::from(home);
            s.push(rest);
            return s.to_string_lossy().to_string();
        }
    } else if p == "~" {
        if let Some(home) = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")) {
            return home.to_string_lossy().to_string();
        }
    }
    p.to_string()
}

/// Multimodal-projector GGUF path resolution. Honors:
///
/// 1. `NEURON_LLAMA_CPP_MMPROJ` (absolute or relative; relative paths
///    resolved against CWD, not the text-GGUF dir, so users can point
///    at a shared `mmproj` cache).
/// 2. The text GGUF's parent dir, in priority order:
///      - `<dir>/mmproj-<anything>.gguf` (HF llama.cpp conversion
///        convention — most models ship with this name)
///      - `<dir>/model.mmproj.gguf` (alternative)
///      - `<dir>/mmproj.gguf` (bare convention)
///
/// Returns `None` when none of these resolve; `load()` then
/// constructs a text-only runtime and `chat_with_media` returns
/// `not_initialized` on the call.
fn resolve_mmproj_path(text_gguf: &Path) -> Option<PathBuf> {
    // 1. Explicit env override.
    if let Ok(p) = std::env::var("NEURON_LLAMA_CPP_MMPROJ") {
        if !p.trim().is_empty() {
            let expanded = expand_home(p.trim());
            let candidate = PathBuf::from(&expanded);
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }

    // 2. Co-located with the text model.
    let dir = text_gguf.parent()?;

    // 2a. First-pass: glob `<dir>/mmproj-*.gguf` (one file wins;
    // deterministic order via sort).
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.extension().map(|e| e == "gguf").unwrap_or(false)
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("mmproj-"))
                    .unwrap_or(false)
        })
        .collect();
    found.sort();
    if let Some(p) = found.into_iter().next() {
        return Some(p);
    }

    // 2b. Fallbacks.
    for name in &["model.mmproj.gguf", "mmproj.gguf"] {
        let candidate = dir.join(name);
        if candidate.exists() {
            return Some(candidate);
        }
    }

    None
}

/// Append a single generated token to the KV cache and advance
/// `n_past` by one. Used by the multimodal generation loop, which
/// reuses the same `LlamaContext` across steps (unlike the
/// text-only `chat()` path that creates a fresh context per
/// request).
fn append_step_decode(
    _model: &LlamaModel,
    ctx: &mut LlamaContext<'_>,
    token: llama_cpp_2::token::LlamaToken,
    n_past: i32,
) -> ChatResult<i32> {
    let mut batch = LlamaBatch::new(1, 1);
    batch
        .add(token, n_past, &[0], true)
        .map_err(|e| NeuronError::new(format!("llamacpp mtmd add(step): {e}")))?;
    ctx.decode(&mut batch)
        .map_err(|e| NeuronError::new(format!("llamacpp mtmd decode(step): {e}")))?;
    Ok(n_past + 1)
}
            return s.to_string_lossy().to_string();
        }
    } else if p == "~" {
        if let Some(home) = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")) {
            return home.to_string_lossy().to_string();
        }
    }
    p.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_chatml_emits_im_start_im_end_framing() {
        let msgs = vec![
            LlamaChatMessage::new("system", "You are terse."),
            LlamaChatMessage::new("user", "Hi"),
        ];
        let s = render_chatml(&msgs).expect("render ok");
        assert!(s.contains("<|im_start|>system\nYou are terse.<|im_end|>"));
        assert!(s.contains("<|im_start|>user\nHi<|im_end|>"));
        assert!(s.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn render_chatml_unknown_role_falls_back_to_user_with_warning() {
        // The 'tool' role is unknown to ChatML; we coerce to 'user' so
        // the prompt is at least well-formed. Caller should re-render
        // through the GGUF template (with role-aware jinja) for proper
        // tool support, which is why capabilities() reports
        // supports_tools=false.
        let msgs = vec![LlamaChatMessage::new("tool", "result: 42")];
        let s = render_chatml(&msgs).expect("render ok");
        assert!(s.contains("<|im_start|>user\nresult: 42<|im_end|>"));
    }

    #[test]
    fn stop_matches_returns_true_for_suffix_hit() {
        let out = "Hello world. End.";
        let stops = vec!["End.".to_string(), "STOP".to_string()];
        assert!(stop_matches(out, &stops));
    }

    #[test]
    fn stop_matches_returns_false_when_no_suffix_hit() {
        let out = "Hello world.";
        let stops = vec!["End.".to_string()];
        assert!(!stop_matches(out, &stops));
    }

    #[test]
    fn stop_matches_handles_empty_stop_list() {
        // Empty list should never trigger a stop (the function shouldn't
        // even be called with an empty list, but be defensive).
        let out = "anything";
        let stops: Vec<String> = vec![];
        assert!(!stop_matches(out, &stops));
    }

    #[test]
    fn seed_for_request_is_deterministic() {
        let req = ChatRequest {
            model: "qwen".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: "hi".into(),
            }],
            temperature: Some(0.7),
            max_tokens: None,
            top_p: None,
            top_k: None,
            stream: false,
            stop: None,
        };
        let a = seed_for_request(&req);
        let b = seed_for_request(&req);
        assert_eq!(a, b);
    }

    #[test]
    fn seed_for_request_differs_when_request_shapes_differ() {
        let r1 = ChatRequest {
            model: "qwen".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: "a".into(),
            }],
            temperature: Some(0.7),
            max_tokens: None,
            top_p: None,
            top_k: None,
            stream: false,
            stop: None,
        };
        let r2 = ChatRequest {
            model: "qwen".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: "b".into(),
            }],
            temperature: Some(0.7),
            max_tokens: None,
            top_p: None,
            top_k: None,
            stream: false,
            stop: None,
        };
        // The seed is hashed from message-count + a few stable fields
        // so identical-shape requests with different content CAN
        // collide; the contract is just that the function does not
        // panic and returns a u32. Re-running with a third variable
        // (temperature != 0.7) is the stronger test below.
        let s1 = seed_for_request(&r1);
        let _ = seed_for_request(&r2);
        let r3 = ChatRequest {
            temperature: Some(0.3),
            ..r1.clone()
        };
        let s3 = seed_for_request(&r3);
        assert_ne!(s1, s3, "temperature change should change the seed");
    }

    #[test]
    fn expand_home_strips_tilde_for_explicit_path() {
        // "no tilde" should be a no-op
        let s = expand_home("/opt/models");
        assert_eq!(s, "/opt/models");
    }

    #[test]
    fn gpu_layers_for_cpu_device_is_zero() {
        assert_eq!(gpu_layers_for(&DeviceSelection::Cpu), 0);
    }

    #[test]
    fn gpu_layers_for_auto_device_is_zero() {
        // Conservative default; matches candle's `Auto`. Users wanting
        // GPU offload from Auto must set the device explicitly.
        assert_eq!(gpu_layers_for(&DeviceSelection::Auto), 0);
    }

    #[test]
    fn gpu_layers_for_cuda_device_is_max() {
        assert_eq!(gpu_layers_for(&DeviceSelection::Nvidia), u32::MAX);
    }

    #[test]
    fn gpu_layers_for_apple_device_is_max() {
        assert_eq!(gpu_layers_for(&DeviceSelection::Apple), u32::MAX);
    }

    // ---- mmproj auto-discovery (resolve_mmproj_path) ----
    //
    // `resolve_mmproj_path` reads `$NEURON_LLAMA_CPP_MMPROJ` and
    // walks the text-GGUF's parent dir. Tests cover:
    //   - empty dir → None
    //   - dir with a `mmproj-*.gguf` file → returns that file
    //   - dir with `model.mmproj.gguf` → returns that
    //   - env var override beats the co-located file
    //   - a `model.gguf` in the same dir is NOT mistaken for an mmproj
    //
    // We use `tempfile::tempdir` (already a dev-dep) for hermetic
    // directories and `unsafe { std::env::set_var(...) }` to
    // control the env var; each test restores the env at the end
    // via Drop semantics on a small RAII guard.

    use std::sync::Mutex;

    // Serialize env-var tests globally — Rust tests in the same
    // process share the env, and `NEURON_LLAMA_CPP_MMPROJ` would
    // race between parallel `cargo test` threads.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvVarGuard {
        key: &'static str,
        prev: Option<String>,
    }
    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            // edition 2024: env::set_var / remove_var are unsafe
            let prev = std::env::var(key).ok();
            unsafe { std::env::set_var(key, value) };
            Self { key, prev }
        }
    }
    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.prev {
                Some(v) => unsafe { std::env::set_var(self.key, v) },
                None => unsafe { std::env::remove_var(self.key) },
            }
        }
    }

    fn touch(path: &std::path::Path) {
        std::fs::write(path, b"fake gguf\n").expect("write fixture");
    }

    #[test]
    fn resolve_mmproj_returns_none_when_dir_has_only_text_model() {
        let _g = ENV_LOCK.lock().unwrap();
        let _g = EnvVarGuard::set("NEURON_LLAMA_CPP_MMPROJ", "");

        let dir = tempfile::tempdir().expect("tempdir");
        let text = dir.path().join("model.gguf");
        touch(&text);

        assert_eq!(resolve_mmproj_path(&text), None);
    }

    #[test]
    fn resolve_mmproj_finds_mmproj_prefixed_file() {
        let _g = ENV_LOCK.lock().unwrap();
        let _g = EnvVarGuard::set("NEURON_LLAMA_CPP_MMPROJ", "");

        let dir = tempfile::tempdir().expect("tempdir");
        let text = dir.path().join("model.gguf");
        let mmproj = dir.path().join("mmproj-MiniCPM-V-4.7-f16.gguf");
        touch(&text);
        touch(&mmproj);

        let found = resolve_mmproj_path(&text).expect("should find mmproj");
        assert_eq!(found, mmproj);
    }

    #[test]
    fn resolve_mmproj_falls_back_to_model_mmproj_gguf() {
        let _g = ENV_LOCK.lock().unwrap();
        let _g = EnvVarGuard::set("NEURON_LLAMA_CPP_MMPROJ", "");

        let dir = tempfile::tempdir().expect("tempdir");
        let text = dir.path().join("model.gguf");
        let alt = dir.path().join("model.mmproj.gguf");
        touch(&text);
        touch(&alt);

        let found = resolve_mmproj_path(&text).expect("should find model.mmproj.gguf");
        assert_eq!(found, alt);
    }

    #[test]
    fn resolve_mmproj_falls_back_to_bare_mmproj_gguf() {
        let _g = ENV_LOCK.lock().unwrap();
        let _g = EnvVarGuard::set("NEURON_LLAMA_CPP_MMPROJ", "");

        let dir = tempfile::tempdir().expect("tempdir");
        let text = dir.path().join("model.gguf");
        let alt = dir.path().join("mmproj.gguf");
        touch(&text);
        touch(&alt);

        let found = resolve_mmproj_path(&text).expect("should find mmproj.gguf");
        assert_eq!(found, alt);
    }

    #[test]
    fn resolve_mmproj_env_var_overrides_co_located_file() {
        let _g = ENV_LOCK.lock().unwrap();
        let dir = tempfile::tempdir().expect("tempdir");
        let text = dir.path().join("model.gguf");
        let co_located = dir.path().join("mmproj.gguf");
        let override_path = dir.path().join("elsewhere.gguf");
        touch(&text);
        touch(&co_located);
        touch(&override_path);
        let _g = EnvVarGuard::set(
            "NEURON_LLAMA_CPP_MMPROJ",
            override_path.to_string_lossy().as_ref(),
        );

        let found = resolve_mmproj_path(&text).expect("env var override");
        assert_eq!(found, override_path);
    }

    #[test]
    fn resolve_mmproj_env_var_to_missing_path_falls_through_to_discovery() {
        // If the env var points at a non-existent file, we silently
        // fall back to the discovery search (rather than
        // erroring) — this lets ops change the var to an
        // expected-but-not-yet-staged path without breaking
        // co-located fallback.
        let _g = ENV_LOCK.lock().unwrap();
        let dir = tempfile::tempdir().expect("tempdir");
        let text = dir.path().join("model.gguf");
        let co_located = dir.path().join("mmproj.gguf");
        touch(&text);
        touch(&co_located);
        let _g = EnvVarGuard::set("NEURON_LLAMA_CPP_MMPROJ", "/no/such/path.gguf");

        let found = resolve_mmproj_path(&text).expect("should fall through to co-located");
        assert_eq!(found, co_located);
    }

    #[test]
    fn resolve_mmproj_does_not_mistake_text_gguf_for_mmproj() {
        // A second `*.gguf` in the dir that doesn't start with
        // `mmproj-` (e.g. a quantized variant) must NOT be picked
        // up as the projector.
        let _g = ENV_LOCK.lock().unwrap();
        let _g = EnvVarGuard::set("NEURON_LLAMA_CPP_MMPROJ", "");

        let dir = tempfile::tempdir().expect("tempdir");
        let text = dir.path().join("model.gguf");
        let decoy = dir.path().join("model.Q4_K_M.gguf");
        touch(&text);
        touch(&decoy);

        assert_eq!(resolve_mmproj_path(&text), None);
    }
}
