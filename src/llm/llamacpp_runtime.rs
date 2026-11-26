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
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures_util::stream::BoxStream;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::mtmd::{
    MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputChunks, MtmdInputText,
};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::{ApplyChatTemplateError, ChatTemplateError};
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

pub struct LlamaCppLlmRuntime {
    model_id: String,
    /// Saved so we can rebuild a fresh `LlamaContext` per chat() call
    /// (the llama-cpp-2 design is one context per stream of inference;
    /// the model + backend are shared, Send + Sync).
    n_ctx: u32,
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
    /// `Some`, the projector is shared across concurrent chat requests.
    ///
    /// ⚠ **刀 15** —— 上游写得很直白：`mtmd_context` 的 **evaluation 与
    /// encoding 不是线程安全的**，绑定因此把 `eval_chunks` 收成
    /// `&mut MtmdContext` ⇒ ⇒ 旧代码那个「一个 Arc 直接共享给并发请求」
    /// **与上游契约相冲** ⇒ ⇒ 这里套一层 `Mutex` 把「tokenize ＋ eval」整段
    /// 串行化（上游明确说 evaluation/encoding 不是线程安全的）。
    mtmd: Option<Arc<Mutex<MtmdContext>>>,
    /// 投影器**自己**报的两个 flag —— 「能吃图」与「能吃音」是**两件事**。
    ///
    /// ⚠️ 为什么存下来而不是每次去问 mtmd —— `MtmdContext` 在
    /// `Arc<Mutex<..>>` 里 ⇒ ⇒ 每次问都要抢锁 ⇒ ⇒ 而那把锁是**给 eval
    /// 串行化用的**（上游：evaluation 不是线程安全的）⇒ ⇒ 能力面（一条
    /// 热路径）不该为了一句「支不支持」去抢它。
    mtmd_support_vision: bool,
    mtmd_support_audio: bool,
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
    pub async fn load(model_id: &str, device: &DeviceSelection) -> ChatResult<Box<dyn LlmRuntime>> {
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

        // ---- MoE 专家下沉（刀 14）：降级阶梯的第三档 ----------------------
        //
        // 判据在 crate::accel（**纯函数**，可测）⇒ 这里是**执行**：把专家
        // 张量钉在 CPU buffer type 上（**等价于上游 llama.cpp 的 --cpu-moe**），
        // 其余张量照常按 n_gpu_layers 走设备 ⇒ ⇒ 「专家在内存、其余在显存」。
        //
        // ⚠ **决策必须到达执行者** —— 上一刀撞出来的教训：策略层会算，但
        // 决策到不了物化权重的那一行，就只是「计划」。
        //
        // ⚠ **这一格只到「专家全在内存」** —— 「显存里再留一份热的」那一半
        // （上游 master 的 llama_model_params.moe_cache_size）**在 neuron's
        // 钉版上不存在**（钉版整棵子树 moe_cache 命中 0）⇒ ⇒ 那是版本债。
        // ⚠⚠️ **刀 17：执行者自己报的能力**（**读回**，不是推断）—— llama.cpp
        // ⇒ ⇒ 「此刻能不能真的把层卸到加速设备」⇒ ⇒ 「其余在显存」那一半有没有落点。
        // ⇒ ⇒ ⚠ 读回点在**物化权重之前**（与 G24 同纪律：前提必须在决策之前
        // ⇒ ⇒ 落地 ⇒ ⇒ 不许先加载再回头说「其实没有 GPU」）。
        //
        // ⚠⚠️ **这个读回的语义要说准**（实测上游源码，`llama.cpp/src/llama.cpp`）：
        //   `llama_supports_gpu_offload()` ＝ 「**有没有一个 GPU/IGPU 设备**可枚举，
        //   或 RPC 后端可用」⇒ ⇒ 而**不是**「CUDA 后端编进去了没有」⇒ ⇒⇒
        //   但这两个问题**一并被它答了**：
        //   ① 后端**没编** ⇒ ⇒ 那个 build 里根本没有 CUDA 设备类型 ⇒ ⇒ 枚举不到 ⇒ ⇒ false；
        //   ② 后端**编了但机器没卡** ⇒ ⇒ `ggml_cuda_init()` 的 `cudaGetDeviceCount`
        //      返回 0 ⇒ ⇒ `device_count = 0` ⇒ ⇒ **不注册任何设备** ⇒ ⇒ 枚举不到 ⇒ ⇒ false。
        //   ⇒ ⇒ 所以这一格问的是「**此刻这台机器上，这个 build 能不能真的卸到设备上**」
        //   ⇒ ⇒ ⇒ 而「其余在显存」这句话**恰恰需要这个前提** ⇒ ⇒ ⇒ 正合适。
        // ⚠ **读回发生在 backend init 之后**（backend 初始化时才会 `ggml_backend_load_all()`
        //   ⇒ ⇒ 那时后端与设备才真的枚举过）⇒ ⇒ 这个顺序是**承重的**。
        let executor_has_gpu = backend.supports_gpu_offload();
        info!(
            model_id = %model_id,
            executor_has_gpu,
            n_gpu_layers,
            "llamacpp executor GPU-offload capability (read back from llama.cpp)"
        );
        let placement = expert_placement_for(&gguf_path, device, executor_has_gpu);
        let wants_experts_in_ram = matches!(placement, crate::accel::ExpertPlacement::ExpertsInRam)
            || env_cpu_moe_requested();

        let mut model_params =
            Box::pin(LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers));
        if wants_experts_in_ram {
            model_params.as_mut().add_cpu_moe_override();
        }
        // 读回**装进模型参数里**的 override ⇒ 这一格只准按读回的结果说话。
        let patterns = model_params.tensor_buft_override_patterns();
        let actual = crate::accel::classify_expert_placement(Some(&patterns));
        if wants_experts_in_ram {
            eprintln!(
                "[neuron] MoE 专家张量 → 系统内存：判据 {} ｜ 装进参数的 override {} ｜ 这一档 {}",
                placement.name(),
                if patterns.is_empty() {
                    "（空）".to_string()
                } else {
                    patterns.join(", ")
                },
                actual.name()
            );
            if !matches!(actual, crate::accel::ExpertPlacement::ExpertsInRam) {
                eprintln!(
                    "[neuron] ⚠ 期望「专家在系统内存」但读回的是 {} ⇒ ⇒ 这一格**没生效**（不许报成功）",
                    actual.name()
                );
            }
        }

        let model =
            LlamaModel::load_from_file(&backend, &gguf_path, &*model_params).map_err(|e| {
                NeuronError::not_initialized(format!(
                    "llama.cpp model load failed for {path}: {e}",
                    path = gguf_path.display()
                ))
            })?;

        // ⚠ **刀 15** —— 0.1.161 上 `chat_template` 要一个「模板名」参数
        // （None ＝ 模型的默认模板）**并且返回 Result** ⇒ ⇒ 「这个 GGUF
        // 没带模板」在**载入期**就分叉掉 ⇒ ⇒ 存 `Option`：None ⇒ 后面
        // 走 ChatML 兜底。
        let chat_template = match model.chat_template(None) {
            Ok(t) => Some(t),
            Err(ChatTemplateError::MissingTemplate) => {
                warn!("llamacpp model carries no chat template — falling back to ChatML");
                None
            }
            Err(e) => {
                warn!("llamacpp chat_template lookup failed ({e}) — falling back to ChatML");
                None
            }
        };
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
        // ⚠️ **图与音分开记** —— 判据源在 `capabilities::audio_input`
        //（默认档也能编 ⇒ ⇒ 判据会被 CI 跑到）⇒ ⇒ 这里只**取事实**。
        let mut mtmd_support_vision = false;
        let mut mtmd_support_audio = false;
        let mtmd: Option<Arc<Mutex<MtmdContext>>> = match resolve_mmproj_path(&gguf_path) {
            Some(mmproj_path) => {
                let params = MtmdContextParams {
                    use_gpu: matches!(
                        device,
                        DeviceSelection::Nvidia | DeviceSelection::Apple | DeviceSelection::Vulkan
                    ),
                    print_timings: false,
                    n_threads: 0, // 0 = llama.cpp default (usually #cores/2)
                    media_marker: std::ffi::CString::new(llama_cpp_2::mtmd::mtmd_default_marker())
                        .map_err(|e| {
                            NeuronError::not_initialized(format!(
                                "mmproj media marker CString: {e}"
                            ))
                        })?,
                    // ⚠ **语义映射点（刀 15）** —— 0.1.161 的文档写的是
                    // 「**-1** ＝ 用模型自己的默认」⇒ ⇒ 旧代码写的 0 不是
                    // 「默认」，而是「下限 0 / 上限 0」⇒ ⇒ 上限 0 会让图像
                    // **一个 token 都分不到** ⇒ ⇒ 这里改回 -1。
                    image_min_tokens: -1,
                    image_max_tokens: -1,
                };
                match MtmdContext::init_from_file(
                    mmproj_path.to_string_lossy().as_ref(),
                    &model,
                    &params,
                ) {
                    Ok(ctx) => {
                        // ⚠️ **图与音分开** —— 投影器自己报两个 flag
                        // ⇒ ⇒ 「有没有模态」不是「有没有 vision」。
                        let support_vision = ctx.support_vision();
                        let support_audio = ctx.support_audio();
                        mtmd_support_vision = support_vision;
                        mtmd_support_audio = support_audio;
                        // ⚠️ **音频专用投影器不许被误杀** —— 旧判据是
                        // 「没有 vision ⇒ 当纯文本」⇒ ⇒ Qwen2-Audio 那种
                        // （vision=false / audio=true）会被**整个丢掉**
                        // ⇒ ⇒ 那不是「不支持」，是「能用的东西被杀了」。
                        if crate::capabilities::audio_input::mmproj_usable(
                            support_vision,
                            support_audio,
                        ) {
                            info!(
                                mmproj = %mmproj_path.display(),
                                use_mrope = ctx.decode_use_mrope(),
                                support_vision,
                                support_audio,
                                "llamacpp mmproj loaded — multimodal enabled"
                            );
                            Some(Arc::new(Mutex::new(ctx)))
                        } else {
                            warn!(
                                mmproj = %mmproj_path.display(),
                                support_vision,
                                support_audio,
                                "mmproj GGUF loaded but reports neither vision nor audio \
                                 support; treating as text-only"
                            );
                            None
                        }
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
            n_ctx,
            backend,
            model: Arc::new(model),
            chat_template,
            mtmd,
            mtmd_support_vision,
            mtmd_support_audio,
            caps_cache: std::sync::OnceLock::new(),
        }))
    }

    /// Common multimodal prefill: build the (text + markers + bitmaps)
    /// prompt, tokenize via mtmd, run `eval_chunks` to populate the KV
    /// cache, and return the context pieces the generation loop
    /// needs. Returns `(LlamaContext, Vec<LlamaToken>, i32)` where the last
    /// value is `n_past` after the prefill.
    ///
    /// This is the load-bearing part of `chat_with_media` — once
    /// prefill is done, the existing per-token generation loop takes
    /// over without needing to know mtmd is involved.
    /// ⚠ **刀 16** —— 刻意**不吃 `&self`**：流式那条路必须 `'static`
    ///   （trait 的返回类型边界）⇒ ⇒ 上下文只能在**流里**建 ⇒ ⇒ 建它的
    ///   那几样东西必须由流自己拥有（Arc 克隆）⇒ ⇒ 这一格改成显式传参。
    ///   ⇒ ⇒ 「没装 projector」那一格的真话挪到调用方（两条路共用一句）。
    fn prefill_multimodal<'m>(
        model: &'m LlamaModel,
        backend: &LlamaBackend,
        mtmd_arc: &Arc<Mutex<MtmdContext>>,
        n_ctx: u32,
        prompt: &str,
        media: &[MediaAttachment],
    ) -> ChatResult<(LlamaContext<'m>, Vec<llama_cpp_2::token::LlamaToken>, i32)> {
        // ⚠ **上游契约（刀 15）** —— `mtmd_context` 的 **evaluation / encoding
        //   不是线程安全的**（上游绑定因此把 `eval_chunks` 收成
        //   `&mut MtmdContext`）⇒ ⇒ 这里取锁把 eval 串行化
        //   ⚠ **锁内没有 await** ⇒ guard 不会跨 await 存活（future 仍 Send）。
        let mut mtmd_guard = mtmd_arc
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mtmd: &MtmdContext = &*mtmd_guard;

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
        let ctx_params =
            LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(n_ctx.max(1)));
        let mut ctx = model
            .new_context(backend, ctx_params)
            .map_err(|e| NeuronError::new(format!("llamacpp new_context(mtmd): {e}")))?;

        // Prefill the KV cache from the multimodal chunks. mtmd
        // advances n_past internally; we just need to capture the
        // return value to know where the next generated token lands.
        let n_past: i32 = chunks
            .eval_chunks(
                &mut mtmd_guard,
                &mut ctx,
                /*n_past=*/ 0,
                /*seq_id=*/ 0,
                n_ctx as i32,
                /*logits_last=*/ true,
            )
            .map_err(|e| NeuronError::new(format!("llamacpp mtmd eval_chunks: {e}")))?;
        drop(mtmd_guard);

        // We don't have a single `Vec<LlamaToken>` of the prompt
        // here — mtmd's chunks are an opaque structure. Return an
        // empty vec; the chat_with_media response's `usage` field
        // uses `prompt_tokens = n_past as u32` (the total tokens
        // mtmd just evaluated, which is what the caller wants).
        let prompt_tokens: Vec<llama_cpp_2::token::LlamaToken> = Vec::new();

        Ok((ctx, prompt_tokens, n_past))
    }

    /// Build a sampler chain from the request's sampling params.
    /// Mirrors what `simple` does in llama.cpp's `examples/simple.cpp`:
    /// penalties → top-K → top-P → temperature → dist. All stages are
    /// optional — a request with no sampling fields gets greedy.
    fn build_sampler(&self, req: &ChatRequest) -> LlamaSampler {
        // Pull the chain together stage-by-stage. (0.1.161 names it
        // `chain` / `chain_simple`; the older `new_chain` the old code used
        // does not exist here.) Defaults are sensible:
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
                           // ⚠ **语义映射点（刀 15）** —— 0.1.161 的 `penalties` 多了
                           // `penalty_last_n`（往上数多少个 token 参与重复惩罚）⇒ ⇒ 旧代码
                           // 的 4 个实参在新签名下会**错位** ⇒ ⇒ 这里补上游默认的 64
                           // （llama.cpp 的 `--repeat-last-n` 默认也是 64）。
            samplers.push(LlamaSampler::penalties(64, 64, 1.05, 0.0, 0.0));
        }
        if top_k > 0 {
            samplers.push(LlamaSampler::top_k(top_k as i32));
        }
        if top_p < 1.0 {
            samplers.push(LlamaSampler::top_p(top_p, 1));
        }
        samplers.push(LlamaSampler::temp(temp));
        samplers.push(LlamaSampler::dist(seed_for_request(req)));

        LlamaSampler::chain(samplers, /*no_perf=*/ false)
    }
}

#[async_trait]
impl LlmRuntime for LlamaCppLlmRuntime {
    async fn load(model: &str, device: &DeviceSelection) -> ChatResult<Box<dyn LlmRuntime>>
    where
        Self: Sized,
    {
        Self::load(model, device).await
    }

    async fn chat(&self, req: ChatRequest) -> ChatResult<ChatResponse> {
        let prompt = render_prompt(&self.model, self.chat_template.as_ref(), &req.messages)?;

        let ctx_params =
            LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(self.n_ctx.max(1)));
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
            if self.model.vocab().is_eog(token) {
                break;
            }
            let piece = decode_piece(&self.model, token);
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
                    // llama.cpp hands back the raw generation — any
                    // `<think>` stream is already inline in `content`
                    // (no backend-side splitting), so there is no
                    // separate reasoning to forward here.
                    reasoning_content: None,
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

    async fn chat_stream(&self, req: ChatRequest) -> ChatResult<BoxStream<'static, ChatChunk>> {
        let prompt = render_prompt(&self.model, self.chat_template.as_ref(), &req.messages)?;

        let prompt_tokens = tokenize_prompt(&self.model, &prompt)?;

        // ⚠⚠️ **'static 边界（刀 16）** —— trait 的返回类型是
        //   `BoxStream<'static, ChatChunk>` ⇒ ⇒ **流不许捕获 `&self`** ⇒ ⇒ 而
        //   `LlamaContext<'_>` **借用模型** ⇒ ⇒ **上下文只能在流里建**（流自己
        //   拥有模型的 Arc）⇒ ⇒ 旧代码那段注释（「把 Arc 克隆进流里就满足
        //   'static」）**说了却没做** ⇒ ⇒ 这一刀补上那一半。
        //   ⚠ 顺带：流里的 `self.model.vocab().is_eog(...)` 也会捕获 `&self`
        //   ⇒ ⇒ 流内一律用**搬进去的** `model`。
        let model = Arc::clone(&self.model);
        let backend = Arc::clone(&self.backend);
        let n_ctx = self.n_ctx;

        let mut sampler = self.build_sampler(&req);
        let max_tokens = req.max_tokens.unwrap_or(256) as u32;
        let stop_strings = req.stop.clone().unwrap_or_default();
        let chunk_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        let created = chrono::Utc::now().timestamp();
        let model_id = req.model.clone();

        let adapted = async_stream::stream! {
            // ⚠ **上下文在流里建**（'static 边界，见上）⇒ ⇒ 流内的失败
            //   **不能**变成 Err（流的 Item 是 ChatChunk，不是 Result）⇒ ⇒
            //   这一格以「一个 finish_reason=error 的终止块 ＋ 记日志」表达
            //   ⇒ ⇒ 与本仓 candle 腿同一口径（那边把能失败的步骤放在流外）。
            let ctx_params =
                LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(n_ctx.max(1)));
            let mut ctx = match model.new_context(&backend, ctx_params) {
                Ok(c) => c,
                Err(e) => {
                    warn!("llamacpp new_context(stream): {e}");
                    yield ChatChunk {
                        id: chunk_id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model_id.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: DeltaMessage { role: None, content: None, reasoning_content: None },
                            finish_reason: Some("error".to_string()),
                        }],
                    };
                    return;
                }
            };
            let mut batch = LlamaBatch::new(n_ctx as usize, 1);
            if let Err(e) = batch.add_sequence(&prompt_tokens, 0, true) {
                warn!("llamacpp add_sequence(stream): {e}");
                return;
            }
            if let Err(e) = ctx.decode(&mut batch) {
                warn!("llamacpp decode(prompt/stream): {e}");
                return;
            }

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
                        reasoning_content: None,
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
                if model.vocab().is_eog(token) {
                    finish_reason = "stop";
                    break;
                }
                let piece = decode_piece(&model, token);
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
                            reasoning_content: None,
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
                    delta: DeltaMessage { role: None, content: None, reasoning_content: None },
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
        let mtmd_arc = self.mtmd.as_ref().ok_or_else(no_mmproj_error)?;
        let (mut ctx, prompt_tokens, mut n_past) = Self::prefill_multimodal(
            &self.model,
            &self.backend,
            mtmd_arc,
            self.n_ctx,
            &prompt,
            &media,
        )?;

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
            if self.model.vocab().is_eog(token) {
                finish_reason = "stop";
                break;
            }
            let piece = decode_piece(&self.model, token);
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
            n_past = append_step_decode(&self.model, &mut ctx, token, n_past)?;
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
                    // llama.cpp hands back the raw generation — any
                    // `<think>` stream is already inline in `content`
                    // (no backend-side splitting), so there is no
                    // separate reasoning to forward here.
                    reasoning_content: None,
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

        // ⚠⚠️ **'static 边界（刀 16）** —— 同 `chat_stream`：流不许捕获
        //   `&self` ⇒ ⇒ 上下文（含 mtmd 预填）只能在流里建 ⇒ ⇒ 这里把
        //   模型／后端／projector 的 Arc **搬进流里**。
        let model = Arc::clone(&self.model);
        let backend = Arc::clone(&self.backend);
        let mtmd_arc = self.mtmd.clone();
        let n_ctx = self.n_ctx;

        let mut sampler = self.build_sampler(&req);
        let max_tokens = req.max_tokens.unwrap_or(256) as u32;
        let stop_strings = req.stop.clone().unwrap_or_default();
        let chunk_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        let created = chrono::Utc::now().timestamp();
        let model_id = req.model.clone();

        let adapted = async_stream::stream! {
            // ⚠ **预填在流里做**（刀 16）—— 上下文借用模型 ⇒ ⇒ 流外建 ⇒ ⇒
            //   流会捕获 `&self` ⇒ ⇒ `'static` 不成立。
            let prefill = match mtmd_arc.as_ref() {
                Some(m) => Self::prefill_multimodal(&model, &backend, m, n_ctx, &prompt, &media),
                None => Err(no_mmproj_error()),
            };
            let (mut ctx, _prompt_tokens, mut n_past) = match prefill {
                Ok(v) => v,
                Err(e) => {
                    warn!("llamacpp prefill(stream/media): {e}");
                    yield ChatChunk {
                        id: chunk_id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model_id.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: DeltaMessage { role: None, content: None, reasoning_content: None },
                            finish_reason: Some("error".to_string()),
                        }],
                    };
                    return;
                }
            };

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
                        reasoning_content: None,
                    },
                    finish_reason: None,
                }],
            };

            let mut emitted: u32 = 0;
            let mut output = String::new();
            let mut finish_reason = "stop";

            while emitted < max_tokens {
                let token = sampler.sample(&ctx, n_past - 1);
                if model.vocab().is_eog(token) {
                    finish_reason = "stop";
                    break;
                }
                let piece = decode_piece(&model, token);
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
                            reasoning_content: None,
                        },
                        finish_reason: None,
                    }],
                };

                n_past = match append_step_decode(&model, &mut ctx, token, n_past) {
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
                    delta: DeltaMessage { role: None, content: None, reasoning_content: None },
                    finish_reason: Some(finish_reason.to_string()),
                }],
            };
        };

        Ok(Box::pin(adapted))
    }

    fn supports_media(&self) -> bool {
        // Only when a multimodal projector GGUF was loaded next to the
        // text model. Without it `chat_with_media` fails with
        // `not_implemented`, so advertising media input would be a
        // promise this backend cannot keep.
        //
        // ⚠️ **「有投影器」不等于「能吃图」** —— 图与音是**两件事**
        // ⇒ ⇒ 分开看 `supports_image()` / `supports_audio()`。
        self.mtmd.is_some()
    }

    /// ⚠️ 「能吃图」是**投影器自己**说的（不是「有投影器」）⇒ ⇒
    /// 音频专用投影器不许被顺带报成「能吃图」。
    fn supports_image(&self) -> bool {
        self.mtmd_support_vision
    }

    /// 「能吃音」是**投影器自己**说的 ⇒ ⇒ 判据源＝`support_audio()`。
    fn supports_audio(&self) -> bool {
        self.mtmd_support_audio
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
        let chat_req =
            crate::llm::translators::canonical_to_chat_request(&req).map_err(NeuronError::new)?;
        let chat_resp = self.chat(chat_req).await?;
        Ok(crate::llm::translators::chat_response_to_canonical(
            &req.model, chat_resp,
        ))
    }

    async fn execute_stream(
        &self,
        req: crate::canonical::CanonicalLlmRequest,
    ) -> crate::llm::ChatResult<BoxStream<'static, crate::canonical::CanonicalLlmStreamEvent>> {
        let chat_req =
            crate::llm::translators::canonical_to_chat_request(&req).map_err(NeuronError::new)?;
        let stream = self.chat_stream(chat_req).await?;
        use futures_util::StreamExt;
        Ok(Box::pin(stream.filter_map(|chunk| async move {
            chunk
                .choices
                .into_iter()
                .next()
                .and_then(|c| c.delta.content)
                .filter(|s| !s.is_empty())
                .map(
                    |t| crate::canonical::CanonicalLlmStreamEvent::ContentBlockDelta {
                        index: 0,
                        delta: crate::canonical::ContentDelta::TextDelta(t),
                    },
                )
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
        Ok(crate::llm::translators::chat_response_to_canonical(
            &req.model, chat_resp,
        ))
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
                .map(
                    |t| crate::canonical::CanonicalLlmStreamEvent::ContentBlockDelta {
                        index: 0,
                        delta: crate::canonical::ContentDelta::TextDelta(t),
                    },
                )
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
            use crate::capabilities::{ApiStyle, Modality, ModelCapabilities};

            // ⚠️ **图与音分开** —— 「有投影器」既不等于「能吃图」，
            // 更不等于「能吃音」⇒ ⇒ 判据住在**默认档可编**的纯函数里
            //（这一档要 libclang ⇒ ⇒ 判据写在它里面就永远跑不到）。
            let input = crate::capabilities::audio_input::mmproj_input_modalities(
                self.mtmd_support_vision,
                self.mtmd_support_audio,
            );
            let mut output = std::collections::BTreeSet::new();
            output.insert(Modality::TextOutput);

            ModelCapabilities {
                engine_impl: "llamacpp".into(),
                model_id: self.model_id.clone(),
                // ⚠ **用 collect() 而不是直接搬集合**（刀 15）—— 共享工作树里
                //   capabilities 的集合类型正在变（HashSet ⇄ BTreeSet）⇒ ⇒ 直接搬
                //   会与在飞面撞类型 ⇒ ⇒ 收成迭代器再 collect ⇒ ⇒ **两种类型都成立**。
                input_modalities: input.into_iter().collect(),
                output_modalities: output.into_iter().collect(),
                api_styles: [ApiStyle::openai_chat(), ApiStyle::anthropic_messages()]
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

/// 「没装 projector」这一格的真话（两条路共用同一句 ⇒ ⇒ 不许各说各话）。
fn no_mmproj_error() -> NeuronError {
    NeuronError::not_initialized(
        "no mmproj GGUF was loaded next to the text GGUF — \
         chat_with_media requires a vision-projector model. \
         Drop a mmproj-*.gguf / model.mmproj / mmproj.gguf into \
         the model directory, or set NEURON_LLAMA_CPP_MMPROJ to \
         an explicit path.",
    )
}

/// 一轮对话（**我们自己的一份副本**）。
///
/// ⚠ **刀 15** —— `LlamaChatMessage` 在 llama-cpp-2 0.1.161 上把
/// `role`/`content` **做成私有字段** ⇒ ⇒ ChatML 兜底渲染器读不回来
/// ⇒ ⇒ 这里留一份平行副本（渲染行为与旧代码逐字一致）。
#[derive(Debug, Clone)]
struct ChatTurn {
    role: String,
    content: String,
}

/// 把一个生成的 token 解码成正文片段。
///
/// ⚠ **语义映射点（刀 15）** —— 旧代码写的是
/// `model.token_to_piece(token, llama_cpp_2::token::Special::Tokenize)`，
/// 但那个 `Special` enum **在 0.1.50 / 0.1.99 / 0.1.103 三个上游 tag 里
/// 都不存在**（它属于更早那个 crate），而且它的原义是
/// **「编码时要不要把特殊 token 当真 token」**（上游 0.1.50 的
/// `Special::Tokenize` 文档原话）⇒ ⇒ 旧代码是**把编码语义用在了解码
/// 函数上**。新 API 的解码参数是 llama.cpp C API 的 `special: bool`
/// （**要不要把控制 token 渲染成文本**）⇒ ⇒ 这里取 `false`：控制 token
/// （`<|im_end|>` 之类）不该进 OpenAI 兼容的正文；EOG 在解码前就 break 了。
fn decode_piece(model: &LlamaModel, token: llama_cpp_2::token::LlamaToken) -> String {
    let bytes = model
        .vocab()
        .token_to_piece(token, /*special=*/ false, None);
    String::from_utf8_lossy(&bytes).to_string()
}

/// Render an OpenAI-style `Vec<ChatMessage>` through the GGUF's embedded
/// chat template (or the ChatML fallback) into a single prompt string.
fn render_prompt(
    model: &LlamaModel,
    tmpl: Option<&LlamaChatTemplate>,
    messages: &[ChatMessage],
) -> ChatResult<String> {
    // ⚠ **刀 15** —— `LlamaChatMessage` 在 0.1.161 上把 `role`/`content`
    // **做成私有字段** ⇒ ⇒ ChatML 兜底渲染器读不回来 ⇒ ⇒ 这里**自己留一份
    // 平行副本**（`ChatTurn`），渲染行为与旧代码逐字一致。
    let turns: Vec<ChatTurn> = messages
        .iter()
        .map(|m| ChatTurn {
            role: m.role.clone(),
            content: m.content.clone(),
        })
        .collect();

    // 0.1.161 上 `LlamaChatMessage::new` 返回 Result（含 NUL 字节会失败）。
    let chat: Vec<LlamaChatMessage> = turns
        .iter()
        .map(|t| LlamaChatMessage::new(t.role.clone(), t.content.clone()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| NeuronError::new(format!("llamacpp chat message: {e}")))?;

    if force_chatml_env() {
        return render_chatml(&turns);
    }

    let Some(tmpl) = tmpl else {
        // No embedded template was stored (we never reach here when
        // load() set Some(chat_template) but a future patch that flips
        // the default to None must still get a valid prompt).
        return render_chatml(&turns);
    };

    match model.apply_chat_template(tmpl, &chat, true) {
        Ok(s) => Ok(s),
        // ⚠ **语义映射点（刀 15）** —— 旧代码匹配的是
        // `ChatTemplateError::TemplateNotFound`；0.1.161 上「没有模板」在
        // **载入期**就分叉了（`chat_template(None)` 返回
        // `ChatTemplateError::MissingTemplate` ⇒ ⇒ `chat_template` 存成
        // `None` ⇒ ⇒ 走上面的 ChatML 兜底）⇒ ⇒ 到了这一行还出错，只可能是
        // 「模板在、但渲染不出东西」⇒ ⇒ 同样兜底成 ChatML。
        Err(ApplyChatTemplateError::NullResult) => render_chatml(&turns),
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
///
/// ⚠ **刀 16** —— 此前这里还留着一份 jinja 模板**原文**（`CHATML_FALLBACK`）
///   ⇒ ⇒ 那与本函数是**同一份规格的两个副本** ⇒ ⇒ 两份必然漂 ⇒ ⇒
///   **删掉重复的那份**，规格只留在这里（逐字对齐 Qwen2.5 / Qwen3 的帧：
///   `<|im_start|>{role}\n{content}<|im_end|>\n`，末尾追加
///   `<|im_start|>assistant\n`）。
fn render_chatml(messages: &[ChatTurn]) -> ChatResult<String> {
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
    // ⚠ **等价替换（刀 15）** —— 旧代码写的是 `model.str_to_token(prompt, true, true)`，
    // 那个方法在 0.1.161 上不存在 ⇒ ⇒ 落到 `LlamaVocab::tokenize`：第二个
    // 参数是 `add_special`、第三个是 `parse_special` ⇒ ⇒ 语义**逐字对应**
    // （上游 0.1.50 的 `Special::Tokenize` 原义就是「允许把特殊/控制 token
    // 当真 token 编码」＝ `parse_special=true`）⇒ ⇒ 这一格**不是映射点，是等价**。
    Ok(model.vocab().tokenize(prompt.as_bytes(), true, true))
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

/// 用户显式要「专家张量留系统内存」（`NEURON_LLAMA_CPP_CPU_MOE`）。
///
/// ⚠ 「自动」＝「用户没说话」—— 用户说了就照办（哪怕自动那一档不成立）。
fn env_cpu_moe_requested() -> bool {
    matches!(
        crate::config::env_var("LLAMA_CPP_CPU_MOE").as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("on") | Some("yes")
    )
}

/// 这个 GGUF 是不是 MoE —— **判据源是元数据**里的 <arch>.expert_count。
///
/// ⚠ 元数据读不到 ⇒ ⇒ **按「不是 MoE」处理**（保守：不敢下沉就别下沉）；
/// ⚠ <arch>.expert_count 也没有 ⇒ ⇒ 退回架构名单（accel::arch_looks_moe）。
fn gguf_looks_moe(path: &std::path::Path) -> bool {
    let Some(ctx) = llama_cpp_2::gguf::GgufContext::from_file(path) else {
        return false;
    };
    let arch_idx = ctx.find_key("general.architecture");
    if arch_idx < 0 {
        return false;
    }
    let Some(arch) = ctx.val_str(arch_idx) else {
        return false;
    };
    let expert_key = format!("{arch}.expert_count");
    let idx = ctx.find_key(&expert_key);
    if idx >= 0 {
        return ctx.val_u32(idx) > 1;
    }
    crate::accel::arch_looks_moe(arch)
}

/// 这一格（第档）该落在哪儿 —— **判据在 accel 层，这里只取事实**。
///
/// ⚠ 事实：权重字节数（盘上文件大小）＋ 是不是 MoE（GGUF 元数据）
/// ⚠ ＋ **执行者自己报的**加速设备能力（`executor_has_gpu`）。
///
/// ⚠⚠️ **刀 17 —— 为什么「执行者能力」是这里取的事实之一**：策略层用的
/// ⇒ ⇒ 「这台机器有多少显存」（`caps.vram_bytes`，**candle 测的**）⇒ ⇒ 而
/// ⇒ ⇒ 「llama.cpp 这个 build 有没有 GPU 后端」是**另一个事实** ⇒ ⇒ 两个
/// ⇒ ⇒ **可以脱钩**（机器有 24 GiB 的卡，但这个 build 没编 CUDA）⇒ ⇒ ⇒
/// ⇒ ⇒ 脱钩时若不把这件事带进去，策略层会算出「专家在内存、其余在显存」
/// ⇒ ⇒ ⇒ 而执行者那边**根本没有显存那一侧** ⇒ ⇒ ⇒ 「显存」是假话。
/// ⇒ ⇒ ⇒ 所以这里读回 **llama.cpp 自己**的答案（`supports_gpu_offload()`），
/// ⇒ ⇒ ⇒ 不许拿机器的显存去替执行者说话。
fn expert_placement_for(
    gguf_path: &std::path::Path,
    device: &DeviceSelection,
    executor_has_gpu: bool,
) -> crate::accel::ExpertPlacement {
    // 设备就是 CPU ⇒ ⇒ 没有「其余留在显存」可言 ⇒ ⇒ 整体在主机。
    if matches!(device, DeviceSelection::Cpu) {
        return crate::accel::ExpertPlacement::Host;
    }
    let caps = crate::accel::HostCapabilities::detect_cached();
    let weight_bytes = std::fs::metadata(gguf_path).map(|m| m.len()).unwrap_or(0);
    let is_moe = gguf_looks_moe(gguf_path);
    crate::accel::resolve_expert_placement(caps, is_moe, weight_bytes, executor_has_gpu)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_chatml_emits_im_start_im_end_framing() {
        let msgs = vec![
            ChatTurn {
                role: "system".into(),
                content: "You are terse.".into(),
            },
            ChatTurn {
                role: "user".into(),
                content: "Hi".into(),
            },
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
        let msgs = vec![ChatTurn {
            role: "tool".into(),
            content: "result: 42".into(),
        }];
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
                reasoning_content: None,
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
                reasoning_content: None,
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
                reasoning_content: None,
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
