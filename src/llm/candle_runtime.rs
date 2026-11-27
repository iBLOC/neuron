use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use candle::{DType, Device, IndexOp};
use candle_transformers::models::{chatglm, mistral, phi3, qwen2, qwen3};

// The vendored llama (explicit `head_dim`) is upstream-compatible when
// the field is absent, so every llama dispatch goes through it — see
// `vendor/llama_hd.rs` for why the fork exists (MiniCPM5-1B).
use futures_util::stream::BoxStream;
use tokenizers::Tokenizer;
use vendor::llama_hd as llama;

use super::{ChatMessage, ChatRequest, ChatResponse, Choice, DeltaMessage, LlmRuntime, ModelInfo};
use crate::config::{DeviceSelection, ensure_dir};
use crate::llm::NeuronError;

#[path = "candle_arch.rs"]
mod candle_arch;
use candle_arch::{CausalArch, detect_arch, detect_model_type};

#[path = "vendor/mod.rs"]
mod vendor;

pub struct CandleLlmRuntime {
    inner: Arc<Mutex<Option<LoadedModel>>>,
}

enum CausalModel {
    Qwen2(qwen2::ModelForCausalLM),
    Qwen3(qwen3::ModelForCausalLM),
    #[cfg(feature = "flash-attn")]
    Qwen2Flash(vendor::qwen2::ModelForCausalLM),
    Llama {
        model: llama::Llama,
        cache: llama::Cache,
    },
    Phi3(phi3::Model),
    Mistral(mistral::Model),
    ChatGlm(chatglm::Model),
}

impl CausalModel {
    fn forward(&mut self, input: &candle::Tensor, offset: usize) -> candle::Result<candle::Tensor> {
        match self {
            CausalModel::Qwen2(m) => m.forward(input, offset),
            CausalModel::Qwen3(m) => m.forward(input, offset),
            #[cfg(feature = "flash-attn")]
            CausalModel::Qwen2Flash(m) => m.forward(input, offset),
            CausalModel::Llama { model, cache } => model.forward(input, offset, cache),
            CausalModel::Phi3(m) => m.forward(input, offset),
            CausalModel::Mistral(m) => m.forward(input, offset),
            CausalModel::ChatGlm(m) => m.forward(input),
        }
    }
}

/// What `NEURON_USE_FLASH_ATTN` asked for.
///
/// The raw value of an unrecognized spelling is kept instead of being
/// folded into "off": a typo must not cost the operator the acceleration
/// they asked for **in silence**.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FlashAttnRequest {
    /// The knob was not set at all.
    NotSet,
    /// FlashAttention-2 was asked for.
    Enabled,
    /// FlashAttention-2 was explicitly turned off.
    Disabled,
    /// The knob was set to a spelling we do not recognize.
    Unknown(String),
}

/// The candle architectures, named.
///
/// The message has to say **which** architecture has no FlashAttention
/// path — "FA is off" alone is not actionable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FlashAttnArch {
    Qwen2,
    Qwen3,
    Llama,
    Phi3,
    Mistral,
    ChatGlm,
}

impl FlashAttnArch {
    /// Whether this architecture has a FlashAttention path at all.
    ///
    /// Qwen2 goes through the vendored FA-capable module
    /// (`src/llm/vendor/qwen2.rs`); Qwen3 goes through candle-transformers'
    /// own upstream `use_flash_attn` path (the vendored copy was a leftover
    /// from an older pin and has been deleted). Llama and Mistral carry
    /// candle-transformers' own `use_flash_attn` field. **Phi3 and
    /// ChatGlm have no such field** — the knob is not plumbed into
    /// them at all, so they must be reported as "not covered" instead
    /// of "enabled" (the old code printed "use_flash_attn=true" for
    /// them while never passing the flag anywhere).
    pub(crate) fn supports_flash_attn(self) -> bool {
        matches!(
            self,
            FlashAttnArch::Qwen2
                | FlashAttnArch::Qwen3
                | FlashAttnArch::Llama
                | FlashAttnArch::Mistral
        )
    }

    /// The architecture name, for the operator-facing message.
    pub(crate) fn name(self) -> &'static str {
        match self {
            FlashAttnArch::Qwen2 => "qwen2",
            FlashAttnArch::Qwen3 => "qwen3",
            FlashAttnArch::Llama => "llama",
            FlashAttnArch::Phi3 => "phi3",
            FlashAttnArch::Mistral => "mistral",
            FlashAttnArch::ChatGlm => "chatglm",
        }
    }
}

/// The effective FlashAttention state of the candle leg.
///
/// "asked for" and "in use" are two different facts, and only the
/// second one may reach the kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FlashAttnStatus {
    /// Never asked for ⇒ the sdpa path.
    NotRequested,
    /// Explicitly turned off ⇒ the sdpa path.
    DisabledByEnv,
    /// The spelling was not recognized ⇒ the sdpa path, and the value
    /// is reported.
    UnknownValue,
    /// Asked for, but this build has no `flash-attn` feature ⇒ sdpa.
    FeatureNotCompiled,
    /// Asked for, but this architecture has no FlashAttention path ⇒ sdpa.
    ArchNotCovered,
    /// Asked for, feature and architecture are fine, but the device is
    /// not CUDA ⇒ sdpa.
    DeviceNotCuda,
    /// Asked for, feature / architecture / device are all fine, but the
    /// weights are in a dtype the FlashAttention kernels do not accept
    /// (they need f16/bf16 — **not** f32) ⇒ sdpa.
    ///
    /// ⚠ 这一条是**真机撞出来的**：f32 权重下 FA 会被照送进内核，然后在
    /// forward pass 里炸成 `flash-attn is only supported for f16/bf16
    /// (F32)` ⇒ ⇒ 那时模型**已经加载**、显存**已经占了** ⇒ ⇒ 一个只在
    /// 内核里才暴露的前提，就得**在建模之前**查。
    DtypeNotSupported,
    /// FlashAttention-2 is actually in use.
    Enabled,
}

impl FlashAttnStatus {
    /// Whether FlashAttention-2 is **in use** — as opposed to merely
    /// requested.
    ///
    /// Only this flag may reach the kernels: every other status means
    /// the sdpa path runs, so the model must be built accordingly.
    pub(crate) fn is_enabled(self) -> bool {
        matches!(self, FlashAttnStatus::Enabled)
    }
}

/// Resolve `NEURON_USE_FLASH_ATTN`.
///
/// Accepted truthy values: `1`, `true`, `yes`, `on`.
/// Accepted falsy values: `0`, `false`, `no`, `off`.
/// Case-insensitive; surrounding whitespace is ignored.
/// Unset ⇒ `NotSet`. Anything else ⇒ `Unknown(raw)` — the same
/// spellings the mistralrs leg's `NEURON_MISTRALRS_MTP` knob accepts,
/// so the two legs speak one language.
pub(crate) fn parse_use_flash_attn() -> FlashAttnRequest {
    let Some(raw) = crate::config::env_var("USE_FLASH_ATTN") else {
        return FlashAttnRequest::NotSet;
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => FlashAttnRequest::Enabled,
        "0" | "false" | "no" | "off" => FlashAttnRequest::Disabled,
        _ => FlashAttnRequest::Unknown(raw),
    }
}

/// The architecture of a loaded model, in the terms the FlashAttention
/// message needs.
pub(crate) fn flash_attn_arch_of(arch: &CausalArch) -> FlashAttnArch {
    match arch {
        CausalArch::Qwen2(_) => FlashAttnArch::Qwen2,
        CausalArch::Qwen3(_) => FlashAttnArch::Qwen3,
        CausalArch::Llama(_) => FlashAttnArch::Llama,
        CausalArch::Phi3(_) => FlashAttnArch::Phi3,
        CausalArch::Mistral(_) => FlashAttnArch::Mistral,
        CausalArch::ChatGlm(_) => FlashAttnArch::ChatGlm,
    }
}

/// Everything FlashAttention-2 needs, in one place.
///
/// **四个前提全齐**才行。第四个（`dtype_supported`）是**真机撞出来的**：
/// f32 权重下 FA 会被照送进内核，然后在 forward pass 里炸成
/// `flash-attn is only supported for f16/bf16 (F32)` ⇒ ⇒ 那时模型**已经
/// 加载**、显存**已经占了**、加速**已经把服务带下去了** ⇒ ⇒ 一个只在
/// 内核里才暴露的前提，就得**在建模之前**查。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FlashAttnPrereqs {
    /// 本 build 编了 `flash-attn` feature。
    pub feature_compiled: bool,
    /// 这个架构在 candle 腿里有 FA 路径。
    pub arch_covered: bool,
    /// 设备是 CUDA。
    pub device_is_cuda: bool,
    /// 权重 dtype 是 FA 内核认的（f16/bf16 —— **不是** f32）。
    pub dtype_supported: bool,
}

/// Decide what the candle leg will actually do.
///
/// The order matters: a missing feature is reported before a missing
/// architecture, and a missing architecture before a missing device —
/// each is a different thing to fix, and the first one found is the
/// cheapest to explain. The dtype comes last: it is the **newest**
/// prerequisite, and the one the operator can fix with a single knob.
pub(crate) fn resolve_flash_attn(
    request: &FlashAttnRequest,
    prereqs: FlashAttnPrereqs,
) -> FlashAttnStatus {
    match request {
        FlashAttnRequest::NotSet => FlashAttnStatus::NotRequested,
        FlashAttnRequest::Disabled => FlashAttnStatus::DisabledByEnv,
        FlashAttnRequest::Unknown(_) => FlashAttnStatus::UnknownValue,
        FlashAttnRequest::Enabled => {
            if !prereqs.feature_compiled {
                return FlashAttnStatus::FeatureNotCompiled;
            }
            if !prereqs.arch_covered {
                return FlashAttnStatus::ArchNotCovered;
            }
            if !prereqs.device_is_cuda {
                return FlashAttnStatus::DeviceNotCuda;
            }
            if !prereqs.dtype_supported {
                return FlashAttnStatus::DtypeNotSupported;
            }
            FlashAttnStatus::Enabled
        }
    }
}

/// The operator-facing line for a FlashAttention status.
///
/// Returns `None` when there is nothing to report (never asked for,
/// explicitly off, or genuinely enabled). **Every degraded state must
/// produce a line that names the missing prerequisite** — a silent
/// fallback is the failure mode this whole thing exists to kill.
pub(crate) fn flash_attn_status_message(
    status: FlashAttnStatus,
    request: &FlashAttnRequest,
    arch: FlashAttnArch,
) -> Option<String> {
    match status {
        FlashAttnStatus::NotRequested | FlashAttnStatus::DisabledByEnv => None,
        FlashAttnStatus::Enabled => None,
        FlashAttnStatus::UnknownValue => {
            let raw = match request {
                FlashAttnRequest::Unknown(raw) => raw,
                _ => "",
            };
            Some(format!(
                "NEURON_USE_FLASH_ATTN={raw:?} is not a recognized value — accepted spellings are \
                 on/off (1/0, true/false, yes/no). The CUDA FlashAttention-2 path was NOT \
                 enabled."
            ))
        }
        FlashAttnStatus::FeatureNotCompiled => Some(
            "NEURON_USE_FLASH_ATTN asked for the CUDA FlashAttention-2 path, but this build has no `flash-attn` \
             feature (build with --features flash-attn). The CUDA FlashAttention-2 path is NOT in use."
                .to_string(),
        ),
        FlashAttnStatus::ArchNotCovered => Some(format!(
            "NEURON_USE_FLASH_ATTN asked for the CUDA FlashAttention-2 path, but the {} architecture has no \
             FlashAttention path in the candle runtime (covered: qwen2, qwen3, llama, mistral). \
             The CUDA FlashAttention-2 path is NOT in use.",
            arch.name()
        )),
        FlashAttnStatus::DeviceNotCuda => Some(
            "NEURON_USE_FLASH_ATTN asked for the CUDA FlashAttention-2 path, but the device is not CUDA (the \
             FlashAttention kernels are CUDA-only). The CUDA FlashAttention-2 path is NOT in use."
                .to_string(),
        ),
        FlashAttnStatus::DtypeNotSupported => Some(
            "NEURON_USE_FLASH_ATTN asked for the CUDA FlashAttention-2 path, but the weights are f32 — the \
             FlashAttention kernels only support f16/bf16. Set NEURON_LLM_DTYPE=bf16 (or f16). The CUDA \
             FlashAttention-2 path is NOT in use."
                .to_string(),
        ),
    }
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
fn resolve_weight_dtype(override_dtype: Option<crate::accel::DTypeChoice>) -> DType {
    // **一处解析，两处消费**：策略层（`accel::requested_dtype`，记账
    // 用）与这条腿（真的物化）读的是**同一个**旋钮 ⇒ ⇒ 不许一个说 f32、
    // 另一个说 bf16 —— 那会让记账比真实占用**少算一半**。
    //
    // ⚠ **条目带来的档位优先**（2026-11-25 刀 12）：策略层算出来的 dtype
    // （可能是「装不下 ⇒ 无损降到 bf16」的结果）**必须到达这里** ⇒ ⇒
    // 否则「自动降级」只是一句计划 —— 运行时仍按 f32 物化 ⇒ ⇒ 模型
    // 照样装不下 ⇒ ⇒ **那个降级是假的**。
    //
    // 认不出来的值由 `requested_dtype()` 出声（原文照旧），这里只做映射。
    let chosen = override_dtype.or_else(crate::accel::requested_dtype);
    match chosen {
        Some(crate::accel::DTypeChoice::F32) => DType::F32,
        Some(crate::accel::DTypeChoice::BF16) => DType::BF16,
        Some(crate::accel::DTypeChoice::F16) => DType::F16,
        None => DType::F32,
    }
}

impl CandleLlmRuntime {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(None)),
        }
    }

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

    /// The final-position logits row `[vocab]` from a forward result.
    ///
    /// Candle models differ in what `forward` returns: the vendored
    /// Llama slices to `[batch, vocab]`, while upstream Qwen2/3 keep
    /// every position `[batch, seq, vocab]`. Handle both — assuming
    /// 2-D everywhere made every Qwen request fail with
    /// "to scalar: unexpected rank" / empty streams.
    fn last_position_logits(logits: &candle::Tensor) -> candle::Result<candle::Tensor> {
        match logits.dims().len() {
            2 => logits.i(0),
            3 => {
                let seq = logits.dim(1)?;
                logits.i((0, seq - 1))
            }
            rank => Err(candle::Error::Msg(format!(
                "unexpected logits rank {rank} (expected 2 or 3)"
            ))),
        }
    }

    /// **纯函数**：在给定的根里找 LLM 权重目录 —— 布局是
    /// `<root>/llm/<model_id>`（模型库布局，`/v1/models/pull` 落的那份）
    /// **或** `<root>/<model_id>`（直接指路的那份）。
    ///
    /// ⚠️ **判据住在纯函数里**（零环境依赖）⇒ ⇒ 「布局对不对」与「根从哪儿来」
    /// 是**两件可分别证的事** ⇒ ⇒ 不许把 env 读塞进来。
    pub fn find_llm_model_dir(
        roots: &[std::path::PathBuf],
        model_id: &str,
    ) -> Option<std::path::PathBuf> {
        for root in roots {
            for candidate in [root.join("llm").join(model_id), root.join(model_id)] {
                if candidate.is_dir() {
                    return Some(candidate);
                }
            }
        }
        None
    }

    /// 按**给定的根**解析模型目录（`NEURON_LLM_MODEL_PATH` 仍然优先）。
    ///
    /// ⚠️ **根从哪儿来**（2026-11-26 刀 16 续）：**注册面与加载面同一个真相源**
    /// ⇒ ⇒ 注册面（boot 扫描／懒注册）走 `config.model_dirs`，加载面
    /// 就必须走**同一份** ⇒ ⇒ 否则嵌脑形态里宿主给的目录只被用来**注册**，
    /// 加载时却去 `./.neuron/models` 找 ⇒ ⇒ **引擎注册上了、权重找不到**
    /// （真机逐字：`model 'qwen3-0.6b' not found under ./.neuron/models`
    /// ⇒ ⇒ 失败长得像「模型没下载」）⇒ ⇒ 而盘上**明明有**那份权重。
    pub fn resolve_model_path_in(
        model: &str,
        roots: &[std::path::PathBuf],
    ) -> Result<PathBuf, String> {
        if let Ok(p) = crate::config::env_var_result("LLM_MODEL_PATH") {
            let path = PathBuf::from(p);
            ensure_dir(&path, "NEURON_LLM_MODEL_PATH")?;
            return Ok(path);
        }

        // No explicit env override: look the model up by id in the model
        // library that /v1/models/pull populates
        // (`<root>/llm/<id>`), so any downloaded model is
        // servable without extra configuration.
        if let Some(found) = Self::find_llm_model_dir(roots, model) {
            return Self::validate_model_dir(&found);
        }
        let tried: Vec<String> = roots
            .iter()
            .flat_map(|r| [r.join("llm").join(model), r.join(model)])
            .map(|p| p.display().to_string())
            .collect();
        Err(format!(
            "CandleLlmRuntime: model '{model}' not found under {} (set NEURON_LLM_MODEL_PATH \
             to a directory containing config.json + tokenizer.json + *.safetensors; the embedded \
             slot is NEURON_MODELS_DIR = <data_dir>/models)",
            if tried.is_empty() {
                "(no model root configured)".to_string()
            } else {
                tried.join(", ")
            }
        ))
    }

    /// 按**进程级真相源**（`NeuronConfig::model_roots()`）解析模型目录。
    ///
    /// ⚠️ 那一格认 `NEURON_MODELS_DIR`（**嵌脑那一格**：端脑启动时把宿主
    /// 给的 `<data_dir>/models` 指到那儿）⇒ ⇒ 与 ASR 那条腿**同一口径**
    /// （那条腿早就有这个根）⇒ ⇒ 三条腿不许各说各话。
    pub fn resolve_model_path(model: &str) -> Result<PathBuf, String> {
        let roots = crate::config::NeuronConfig::model_roots();
        Self::resolve_model_path_in(model, &roots)
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
                // Try CUDA first, then Metal, then fall back to CPU. The
                // failure is logged rather than swallowed: a silent CPU
                // fallback turns a 30 tok/s GPU model into a 0.3 tok/s
                // CPU model with nothing but a benchmark to point at it.
                #[cfg(feature = "cuda")]
                {
                    match Device::new_cuda(0) {
                        Ok(dev) => return Ok(dev),
                        Err(e) => eprintln!(
                            "[neuron] CandleLlmRuntime: CUDA init failed, falling back: {e}"
                        ),
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

    /// Make the CUDA primary context current on the calling thread.
    ///
    /// cudarc retains the device's **primary** context and binds it to
    /// whichever thread created the `Device` — the context is shared, but
    /// "current" is per-thread state. The streaming path runs the
    /// generation on a freshly spawned thread, so without this call every
    /// CUDA forward on that thread fails with
    /// `CUDA_ERROR_INVALID_CONTEXT, invalid device context` (observed
    /// 2026-10-29 while benchmarking the 3090: the SSE body came back
    /// empty while the non-streaming path worked).
    ///
    /// No-op for the CPU device.
    fn bind_cuda_context(device: &Device) {
        #[cfg(feature = "cuda")]
        if matches!(device, Device::Cuda(_)) {
            // Creating the device here *binds* the device's primary
            // context to this thread (cudarc retains the primary
            // context and calls `bind_to_thread` on it). The context is
            // shared across threads, so the weights allocated by the
            // loading thread stay valid — only the "current context"
            // (per-thread state) needed fixing.
            let _ = Device::new_cuda(0);
        }
        let _ = device;
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
        match &mut m.model {
            CausalModel::Llama { model: _, cache } => {
                if let Some(cfg) = m.llama_cfg.as_ref() {
                    *cache = llama::Cache::new(true, m.dtype, cfg, &m.device).map_err(|e| {
                        NeuronError::not_implemented(format!("rebuild Llama cache: {e}"))
                    })?;
                }
            }
            // Qwen-family caches live inside the model and are never
            // cleared by the forward path itself. Without this, a second
            // request appends to the previous request's cache and the
            // new candle rev's attention fails on
            // broadcast_add(mask, scores): lhs [1, 16, 22, kv_grown] vs
            // rhs [1, 1, 22, 22].
            CausalModel::Qwen2(m) => m.clear_kv_cache(),
            #[cfg(feature = "flash-attn")]
            CausalModel::Qwen2Flash(m) => m.clear_kv_cache(),
            CausalModel::Qwen3(m) => m.clear_kv_cache(),
            CausalModel::Phi3(m) => m.clear_kv_cache(),
            CausalModel::Mistral(m) => m.clear_kv_cache(),
            // candle-transformers' ChatGlm exposes no cache-reset API;
            // multi-request reuse for that arch behaves as before.
            CausalModel::ChatGlm(_) => {}
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
    fn sample_token_from_logits(logits: &[f32], temperature: f32, top_p: f32, top_k: u32) -> u32 {
        if logits.is_empty() {
            return 0;
        }
        if temperature <= 0.0 {
            return logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
        }
        // Subtract the max logit before exp to keep the softmax numerically
        // stable for large logit values.
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
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
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

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
    /// ⚠ 不带 dtype 的那一半 —— 走全局 env（`NEURON_LLM_DTYPE`）⇒ ⇒ 与
    /// 策略层的起始档一致（刀 12：条目带来的档位由 `load_with_dtype` 负责）。
    async fn load(model: &str, device: &DeviceSelection) -> Result<Box<dyn LlmRuntime>, NeuronError>
    where
        Self: Sized,
    {
        let roots = crate::config::NeuronConfig::model_roots();
        Self::load_with_dtype(
            model,
            device,
            crate::accel::requested_dtype().unwrap_or(crate::accel::DTypeChoice::F32),
            &roots,
        )
        .await
    }

    /// ⚠ **带 dtype 的那一半** —— 条目里的那一档在这里**真的**物化
    /// （刀 12：决策必须到达执行者）。
    async fn load_with_dtype(
        model: &str,
        device: &DeviceSelection,
        dtype: crate::accel::DTypeChoice,
        roots: &[std::path::PathBuf],
    ) -> Result<Box<dyn LlmRuntime>, NeuronError>
    where
        Self: Sized,
    {
        // ⚠️ **按【给定的根】解析**（2026-11-26 刀 16 续）—— 那份根就是
        // **注册面**用的那份（registry 把 `config.model_dirs` 递下来）
        // ⇒ ⇒ 两面**结构上**同源 ⇒ ⇒ 「引擎注册上了、权重找不到」不再可能。
        let path =
            Self::resolve_model_path_in(model, roots).map_err(NeuronError::not_implemented)?;
        let dev = Self::map_device_selection(device).map_err(NeuronError::not_implemented)?;

        let (arch, model_type) = detect_arch(&path).map_err(NeuronError::not_implemented)?;
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
        //
        // "asked for" and "in use" are two different facts, and only
        // the second one may reach the kernels. The flag handed to the
        // model builder is therefore the EFFECTIVE one: the requested
        // one used to build a FlashAttention model even when this
        // build has no `flash-attn` feature or the device is not CUDA —
        // and the old warning named only the missing feature, never
        // the missing device (so a CPU run silently ran sdpa, and a
        // CPU run with the feature compiled would have failed inside
        // the attention kernel instead).
        // 权重 dtype 必须**先**解析出来 —— FlashAttention 的四个前提里有
        // 一个是 dtype（内核只认 f16/bf16），而它得在**建模之前**查。
        //
        // `bf16` halves peak weight memory versus the f32 default, which
        // is the difference between fitting and paging out on a 16 GB host.
        let weight_dtype = resolve_weight_dtype(Some(dtype));
        eprintln!(
            "[neuron] CandleLlmRuntime: weights dtype = {weight_dtype:?} (条目/策略给的档位优先；否则 NEURON_LLM_DTYPE=f32|bf16|f16)"
        );

        // ---- 「这一档跑得起来吗」—— 在**加载权重之前**就判 ----
        //
        // ⚠ **「装得下」不等于「跑得起来」** —— 真机撞过（qwen3-0.6b，CUDA，
        // 本 build 编了 `flash-attn`）：f32 **加载成功**（显存真占着
        // 2 460 MiB）⇒ ⇒ 然后炸在 forward pass（`flash-attn is only
        // supported for f16/bf16 (F32)`）⇒ ⇒ 那是「加载成功、生成炸」——
        // **服务会倒在最不该倒的地方**（用户要等到开口说话才看到）。
        //
        // ⚠ 所以这里在 **load 期**就拒 ⇒ ⇒ 服务不倒，用户拿到的是**一句真话**：
        // 哪一档跑不了、为什么、怎么改 ⇒ ⇒ 不许「先加载上，炸了再说」。
        //
        // ⚠ **判据与策略层是同一个函数**（[`crate::accel::dtype_runnable_for_arch`]）
        // ⇒ ⇒ 两边不许各说各话。
        if !crate::accel::dtype_runnable_for_arch(
            &model_type,
            dtype,
            dev.is_cuda(),
            cfg!(feature = "flash-attn"),
        ) {
            return Err(NeuronError::invalid_input(format!(
                "[neuron] 这一档（{}）**跑不起来** —— {} 族 ＋ CUDA 设备 ＋ 本 build 编了 \
                 `flash-attn` ⇒ ⇒ 注意力会走 **CUDA flash 内核** ⇒ ⇒ 那颗内核**只认 \
                 f16/bf16** ⇒ ⇒ f32 会**加载成功、前向炸**（不是加载时）⇒ ⇒ \
                 请显式设 `NEURON_LLM_DTYPE=bf16`（或 f16）",
                dtype.name(),
                model_type,
            )));
        }

        let fa_request = parse_use_flash_attn();
        let fa_arch = flash_attn_arch_of(&arch);
        let fa_status = resolve_flash_attn(
            &fa_request,
            FlashAttnPrereqs {
                feature_compiled: cfg!(feature = "flash-attn"),
                arch_covered: fa_arch.supports_flash_attn(),
                device_is_cuda: dev.is_cuda(),
                // FA 内核只接受 f16/bf16 ⇒ ⇒ 默认的 f32 **不能**进内核
                // （真机实测：f32 下会炸在 forward pass —— 那时模型已加载、
                // 显存已占）⇒ ⇒ 所以这里在 load 期就把它挡下来。
                dtype_supported: !matches!(weight_dtype, DType::F32),
            },
        );
        let use_flash_attn = fa_status.is_enabled();
        if use_flash_attn {
            // The positive half of the very same fact. "Asked for" is
            // not "in use", so the state that IS in use has to be
            // observable too — otherwise the only trace of FA in a log
            // is the complaint about its absence, and a reader cannot
            // tell "on" from "never asked".
            eprintln!(
                "[neuron] CandleLlmRuntime: FlashAttention-2 IS IN USE (feature compiled, device is CUDA, arch={}, dtype={:?})",
                fa_arch.name(),
                weight_dtype
            );
        }
        if let Some(msg) = flash_attn_status_message(fa_status, &fa_request, fa_arch) {
            // eprintln!, not tracing: neuron installs no tracing
            // subscriber of its own, so a tracing event is a silent
            // sink in a bare host — and "I asked for FA and got sdpa"
            // is exactly the message that must not be lost.
            eprintln!("[neuron] ERROR: {msg}");
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

        let vb = unsafe {
            crate::candle_compat::from_mmaped_safetensors(
                safetensors_paths.as_slice(),
                weight_dtype,
                &dev,
            )
        }
        .map_err(|e| NeuronError::not_implemented(format!("load weights: {e}")))?;

        let model = match &arch {
            CausalArch::Qwen2(cfg) => {
                // The FA build of Qwen2 is the **vendored** module:
                // upstream candle-transformers has no FlashAttention path
                // for qwen2 at all.
                //
                // The flag that reaches the model builder is the EFFECTIVE
                // one, and both branches have to fold into a single
                // expression: the old code `return`ed a `CausalModel`
                // out of a function whose return type is
                // `Box<dyn LlmRuntime>` (E0308) — the `flash-attn`
                // feature had never been compiled in this repo.
                #[cfg(feature = "flash-attn")]
                let model = if use_flash_attn {
                    let flash_cfg: vendor::qwen2::Config = cfg.clone().into();
                    let m = vendor::qwen2::ModelForCausalLM::new(&flash_cfg, vb).map_err(|e| {
                        NeuronError::not_implemented(format!("build Qwen2 (flash): {e}"))
                    })?;
                    CausalModel::Qwen2Flash(m)
                } else {
                    let m = qwen2::ModelForCausalLM::new(cfg, vb)
                        .map_err(|e| NeuronError::not_implemented(format!("build Qwen2: {e}")))?;
                    CausalModel::Qwen2(m)
                };
                #[cfg(not(feature = "flash-attn"))]
                let model = {
                    let m = qwen2::ModelForCausalLM::new(cfg, vb)
                        .map_err(|e| NeuronError::not_implemented(format!("build Qwen2: {e}")))?;
                    CausalModel::Qwen2(m)
                };
                model
            }
            CausalArch::Qwen3(cfg) => {
                // Qwen3 needs NO vendored module: the pinned
                // candle-transformers (rev 66a8cf1, 0.11) already ships a
                // FlashAttention path for qwen3 — the fused **CPU** kernel
                // when the `flash-attn` feature is off, and the **CUDA**
                // FA2 kernel when it is on and the device is a GPU.
                //
                // The vendored copy was a leftover from an older pin
                // (0.9.2), it duplicated what upstream now does, and it
                // never compiled against the pinned candle-flash-attn
                // API (`softmax_scale` changed from `Option<f64>` to
                // `f32`).
                let m = qwen3::ModelForCausalLM::new(cfg, vb)
                    .map_err(|e| NeuronError::not_implemented(format!("build Qwen3/3.5: {e}")))?;
                CausalModel::Qwen3(m)
            }
            CausalArch::Llama(cfg) => {
                let inner = llama::LlamaConfig::clone(cfg).into_config(use_flash_attn);
                let m = llama::Llama::load(vb, &inner)
                    .map_err(|e| NeuronError::not_implemented(format!("build Llama: {e}")))?;
                let cache = llama::Cache::new(true, weight_dtype, &inner, &dev)
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
            CausalArch::Llama(cfg) => {
                Some(llama::LlamaConfig::clone(cfg).into_config(use_flash_attn))
            }
            _ => None,
        };
        let loaded = LoadedModel {
            model,
            tokenizer,
            eos_token_id,
            device: dev,
            llama_cfg,
            dtype: weight_dtype,
        };
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

        // The caller may run on a different worker thread than the one
        // that loaded the model; re-bind the CUDA context before touching
        // the device (see `bind_cuda_context`).
        {
            let inner = self.inner.lock().unwrap();
            if let Some(m) = inner.as_ref() {
                Self::bind_cuda_context(&m.device);
            }
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
        // Reset the KV cache INSIDE the generation critical section.
        // Resetting in a separate lock scope (the previous shape of
        // this code) left a window where another request's generation
        // could append entries after our reset and before our forward:
        // the next prompt then attended over stale positions and the
        // attention mask (built for prompt_len) failed broadcast_add —
        // observed live as `lhs: [1, 14, 77, 135], rhs: [1, 1, 77, 77]`
        // where 135 = 77 fresh + 58 leaked from a concurrent request.
        Self::reset_llama_cache(m)?;

        let input_tensor = candle::Tensor::from_vec(
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
        // Candle models differ in what `forward` returns: the vendored
        // Llama slices to `[batch, vocab]`, while upstream Qwen2/3 keep
        // every position `[batch, seq, vocab]`. Reduce to the last
        // position's row either way — assuming 2-D makes every Qwen
        // request fail (argmax/to_scalar rank errors, empty streams).
        let last_logits = Self::last_position_logits(&logits)
            .map_err(|e| NeuronError::not_implemented(format!("last-position logits: {e}")))?;
        let mut next_token = last_logits
            .argmax(candle::D::Minus1)
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
            let next_input = candle::Tensor::from_vec(vec![next_token], (1, 1), &m.device)
                .map_err(|e| NeuronError::not_implemented(format!("build next input: {e}")))?;
            let logits = m.model.forward(&next_input, offset).map_err(|e| {
                NeuronError::not_implemented(format!("forward pass {}: {}", offset, e))
            })?;
            // Same single-batch reduction as the prompt forward.
            let last_logits = Self::last_position_logits(&logits)
                .map_err(|e| NeuronError::not_implemented(format!("last-position logits: {e}")))?;
            next_token = last_logits
                .argmax(candle::D::Minus1)
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
                    reasoning_content: None,
                },
                finish_reason: Some(if next_token == eos_token_id {
                    "stop".to_string()
                } else {
                    "length".to_string()
                }),
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
            // The CUDA primary context is bound per thread, and this
            // generation runs on a thread that did not create the device.
            // Without this, every CUDA forward here fails with
            // "invalid device context" and the SSE body comes back empty
            // (observed 2026-10-29 while benchmarking the RTX 3090).
            {
                let inner = inner.lock().unwrap();
                if let Some(m) = inner.as_ref() {
                    Self::bind_cuda_context(&m.device);
                }
            }
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
                            delta: DeltaMessage {
                                role: None,
                                content: None,
                                reasoning_content: None,
                            },
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
                        delta: DeltaMessage {
                            role: None,
                            content: None,
                            reasoning_content: None,
                        },
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
                        reasoning_content: None,
                    },
                    finish_reason: None,
                }],
            });

            let input =
                match candle::Tensor::from_vec(prompt_ids.clone(), (1, prompt_len), &m.device) {
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
            // Reduce 2-D `[batch, vocab]` or 3-D `[batch, seq, vocab]`
            // forward output to the last position's row (see
            // last_position_logits).
            let last = match Self::last_position_logits(&logits) {
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
                Self::sample_token_from_logits(&logits_vec, temperature, top_p, top_k);
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
                                    content: Some(
                                        text_so_far[old_len..old_len + emit_len].to_string(),
                                    ),
                                    reasoning_content: None,
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
                                reasoning_content: None,
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

                let input = match candle::Tensor::from_vec(vec![next_token], (1, 1), &m.device) {
                    Ok(t) => t,
                    Err(_) => return,
                };
                let logits = match m.model.forward(&input, offset) {
                    Ok(t) => t,
                    Err(_) => return,
                };
                // Same reduction as the prompt forward.
                let last = match Self::last_position_logits(&logits) {
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
                next_token = Self::sample_token_from_logits(&logits_vec, temperature, top_p, top_k);

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
                                        content: Some(
                                            text_so_far[old_len..old_len + emit_len].to_string(),
                                        ),
                                        reasoning_content: None,
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
                                    reasoning_content: None,
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
                    delta: DeltaMessage {
                        role: None,
                        content: None,
                        reasoning_content: None,
                    },
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
        // Only families the candle runtime can actually load. Qwen3.5 is
        // deliberately absent: its config is nested under `text_config`
        // and its text tower is a hybrid linear/full-attention stack that
        // upstream candle-transformers has no module for — `detect_arch`
        // rejects it at config parse. Qwen3.5 is served by the mistralrs
        // backend instead (see `src/models/catalog.rs`).
        vec![
            ModelInfo::new("qwen3-0.6b", "neuron-candle"),
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

    fn capabilities(&self) -> &crate::capabilities::ModelCapabilities {
        static CAPS: std::sync::LazyLock<crate::capabilities::ModelCapabilities> =
            std::sync::LazyLock::new(|| crate::capabilities::ModelCapabilities {
                engine_impl: "candle".into(),
                model_id: "candle-default".into(),
                input_modalities: [crate::capabilities::Modality::TextInput]
                    .into_iter()
                    .collect(),
                output_modalities: [crate::capabilities::Modality::TextOutput]
                    .into_iter()
                    .collect(),
                api_styles: [crate::capabilities::ApiStyle::openai_chat()]
                    .into_iter()
                    .collect(),
                supports_streaming: true,
                supports_tools: true,
                ..Default::default()
            });
        &CAPS
    }

    fn name(&self) -> &str {
        "candle"
    }
}

#[cfg(test)]
#[path = "candle_runtime_tests.rs"]
mod tests;
