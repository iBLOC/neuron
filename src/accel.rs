//! 设备环境 ⇒ 加速策略。
//!
//! 这个模块回答的是**「这台机器跑得动什么、跑多快」** —— 与
//! `crate::capabilities` 的「模型能吃什么」是**两条轴，不许混**（那个
//! catalog 自己就写明过这一点）。
//!
//! 三层，三个类型：
//! * `HostCapabilities` —— **环境**：设备种类、内存、显存、核数、编进来的 feature；
//! * `ModelFacts` —— **模型**：架构、权重字节数、权重格式、盘上 dtype；
//! * `AccelStrategy` —— **策略**：设备、dtype、FlashAttention、引擎 ＋ **理由** ＋ **拿不到的东西**。
//!
//! 三条纪律（这一刀的全部意义）：
//! * **「最优」是启发式，不是实测** —— 策略给的是「按环境推出来的建议」，每个
//!   选择都带 `reasons`，每个拿不到的东西都进 `unavailable` ⇒ ⇒ **不许把它
//!   当成「测出来的最优」**；
//! * **测不到就说测不到** —— 内存／显存是 `Option`：`None` 表示「没测到」，
//!   **不是「有 0 字节」** ⇒ ⇒ 策略在缺数据时必须退回保守选择并说明；
//! * **策略层不许自己读环境** —— `resolve_strategy` 是纯函数，环境由调用方
//!   喂进来 ⇒ ⇒ 可测、可复现、不会在并行测试里抢全局状态。
//!
//! ⇒ ⇒ 还有第四条，来自这一刀本身：**谁填 `HostCapabilities.device` 就决定了
//! 结论** —— 库里的 `HostCapabilities::detect` 只能说出「**这个 build 能用**的
//! 设备」，而探针能说出「**这台机器真有**的设备」⇒ ⇒ 两者对不上时，**那个差
//! 距本身就是读数**（本机就有一张 24 GB 的卡，而这个 build 连 `cuda` 都没编）。
//!
//! ⇒ ⇒ 而第五条，来自「显存测不到」的**代价**（2026-11-25 那一刀量出来的）：
//! **兜底常量不许被当成真值** —— 真机 24 GiB 的卡上，一个**真放得下**的 3B
//! 模型被 8 GB 的兜底常量**拒了** ⇒ ⇒ 所以显存必须**真的测**，而且
//! **从哪儿测到的**（CUDA driver API 的 CUDA 序号／nvidia-smi 的物理序号）
//! 必须写在脸上（`vram_source`）—— 测不到就说测不到，不许猜。

use crate::config::DeviceSelection;
use crate::llm::LlmImpl;
use std::process::Command;

/// 设备种类。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    /// **没测到**（这个 build 没有能问的腿）。
    Unknown,
    Cpu,
    Cuda,
    Metal,
}

impl DeviceKind {
    pub fn name(self) -> &'static str {
        match self {
            DeviceKind::Unknown => "unknown",
            DeviceKind::Cpu => "cpu",
            DeviceKind::Cuda => "cuda",
            DeviceKind::Metal => "metal",
        }
    }

    /// 调度器记账用的平台键。
    pub fn platform_key(self) -> &'static str {
        match self {
            DeviceKind::Cuda => "cuda",
            DeviceKind::Metal => "metal",
            DeviceKind::Cpu | DeviceKind::Unknown => "cpu",
        }
    }
}

/// 显存读数是**从哪儿来的**。
///
/// 「测到的」与「兜底的」必须能分开看 —— 否则一个兜底常量会被当成真值：
/// 上一刀量出来过它的代价（8 GB 的 GPU 兜底在 24 GiB 的卡上，把一个**真放得下**
/// 的 3B 模型**拒了**）⇒ ⇒ 所以这一栏不是装饰，是**证词**。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VramSource {
    /// CUDA driver API（进程内）：**按 CUDA 序号** ⇒ ⇒ 与运行时真正打开的
    /// 那张卡**同一个口径**（尊重 `CUDA_VISIBLE_DEVICES`）。
    CudaDriver,
    /// `nvidia-smi`（外部进程）：**按物理序号** ⇒ ⇒ 多卡机器上与 CUDA 序号
    /// 未必一致（CUDA 默认 `FASTEST_FIRST`）⇒ ⇒ 所以它是次选口径。
    NvidiaSmi,
    /// **没测到**（不是 0 字节）。
    Unmeasured,
}

impl VramSource {
    /// 人话版（探针／日志直接用）。
    pub fn name(self) -> &'static str {
        match self {
            VramSource::CudaDriver => "cuda-driver（进程内，按 CUDA 序号）",
            VramSource::NvidiaSmi => "nvidia-smi（外部进程，按物理序号）",
            VramSource::Unmeasured => "没测到",
        }
    }

    /// 这个读数是**测到的**吗？
    pub fn is_measured(self) -> bool {
        !matches!(self, VramSource::Unmeasured)
    }
}

/// 这个 build **编译进来**了哪些能力。
///
/// 「编译进来」与「装上了」是两回事 —— 编译进来的东西在本机**一定**能用，
/// 没编译进来的东西**一定**用不了（哪怕机器上装着）。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompiledFeatures {
    pub candle: bool,
    pub cuda: bool,
    pub metal: bool,
    pub flash_attn: bool,
    pub mistralrs: bool,
    pub llamacpp: bool,
}

impl CompiledFeatures {
    /// 本 build 实际编进来的 feature。
    pub const fn current() -> Self {
        Self {
            candle: cfg!(feature = "candle"),
            cuda: cfg!(feature = "cuda"),
            metal: cfg!(feature = "metal"),
            flash_attn: cfg!(feature = "flash-attn"),
            mistralrs: cfg!(feature = "mistralrs"),
            llamacpp: cfg!(feature = "llamacpp"),
        }
    }
}

/// 这台机器的环境。
///
/// `ram_bytes`／`vram_bytes`／`cores` 都是 `Option`：**`None`＝没测到**，
/// 不是 0。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostCapabilities {
    /// 这个 build **能用**的设备。
    ///
    /// ⚠ **不是**「这台机器有什么」—— 那件事在 `machine_device` 里 ⇒ ⇒
    /// **两个字段，不是一个**（2026-11-25 实测撞出来的，见下）。
    pub device: DeviceKind,
    /// 这台机器**真有**的加速设备（**与本 build 编了什么无关**）。
    ///
    /// ⚠ **为什么必须是两个字段** —— 探测设备种类的那条腿是 **feature 门控**
    /// 的（`#[cfg(feature = "cuda")]`）⇒ ⇒ 在**没编 `cuda`** 的 build 里，
    /// 「这台机器有卡」这件事被**读成了「机器没卡」** ⇒ ⇒ 于是
    /// 「GPU 那条路开不了（要 `--features cuda`）」那句话
    /// **在真实路径上永远说不出口**（那个分支不可达）⇒ ⇒ 用户只看到 CPU，
    /// **一句真话都没有**。
    ///
    /// ⇒ ⇒ 所以：**容量与能力是两个字段**（显存那一刀已经这么切过一回），
    /// **设备与能力也是两个字段** —— 「机器有卡」是机器的事实，
    /// 「build 能不能用」是 build 的能力。
    pub machine_device: DeviceKind,
    /// 主机物理内存（字节）—— **总量**。
    pub ram_bytes: Option<u64>,
    /// **现在可用**的内存（字节）—— 与 [`Self::ram_bytes`] **不是同一件事**。
    ///
    /// ⚠⚠️ **容量轴（`pool_of`）用的是这一栏，不是总量** —— 真机撞出来的
    /// （刀 16）：rk3566 上 `MemTotal` 1.91 GiB 而 `MemAvailable` 1.04 GiB
    /// ⇒ ⇒ 差 54% ⇒ ⇒ 用总量 ⇒ ⇒ 策略说「装得下」⇒ ⇒ 真加载 ⇒ ⇒
    /// **被 OOM killer 杀掉** ⇒ ⇒ 「装不下」这格在真机上说假话。
    ///
    /// ⚠ `None` ＝ **没测到**（不是 0）⇒ ⇒ 回落规则见 `pool_of`。
    pub ram_available_bytes: Option<u64>,
    /// 显存（字节）。
    pub vram_bytes: Option<u64>,
    /// 显存**从哪儿测到的**（`vram_bytes == None` ⇒ `Unmeasured`）。
    ///
    /// ⚠ 这一栏不许省 —— 上一刀的代价就是「一个兜底常量被当成真值」：
    /// 8 GB 的 GPU 兜底在 24 GiB 的卡上把一个**真放得下**的 3B 模型**拒了**。
    pub vram_source: VramSource,
    /// 测到几张加速卡（driver／`nvidia-smi` 都能数）。
    pub vram_devices: usize,
    /// 逻辑核数。
    pub cores: Option<usize>,
    /// 本 build 编译进来的能力。
    pub features: CompiledFeatures,
}

impl Default for HostCapabilities {
    fn default() -> Self {
        Self {
            device: DeviceKind::Unknown,
            machine_device: DeviceKind::Unknown,
            ram_bytes: None,
            ram_available_bytes: None,
            vram_bytes: None,
            vram_source: VramSource::Unmeasured,
            vram_devices: 0,
            cores: None,
            features: CompiledFeatures::current(),
        }
    }
}

impl HostCapabilities {
    /// 探测「**这个 build 能用**的」环境 ＋「**这台机器真有的**」显存。
    ///
    /// 设备种类只能问 candle（它是唯一知道怎么开设备的腿）；内存按平台读；
    /// 核数用 `available_parallelism`；**显存走 [`detect_vram`]**（CUDA
    /// driver API 优先，`nvidia-smi` 次之，都不行才 `None` ＋ `Unmeasured`）。
    ///
    /// ⇒ ⇒ 注意这个函数**只**代表「这个 build 能用」，不代表「这台机器有」——
    /// 「机器有 24 GB 显存但 build 没编 cuda」这件事，只有探针（拿 OS 的说法
    /// 对账）才看得出来。
    /// ⇒ ⇒ 但**显存**是例外：它是**机器的容量**，不是 build 的能力 ⇒ ⇒
    /// 所以它单独测，而且**测法写在脸上**（`vram_source`）—— 不许让一个
    /// 兜底常量冒充真值。
    pub fn detect() -> Self {
        let device = detect_device_kind();
        let vram = detect_vram();
        Self {
            device,
            machine_device: detect_machine_device(),
            ram_bytes: ram_bytes_platform(),
            ram_available_bytes: ram_available_bytes_platform(),
            vram_bytes: vram.bytes,
            vram_source: vram.source,
            vram_devices: vram.devices,
            cores: std::thread::available_parallelism().ok().map(|n| n.get()),
            features: CompiledFeatures::current(),
        }
    }

    /// 探测一次，缓存结果。
    ///
    /// 探测会问 candle 能不能开 CUDA（要初始化 CUDA 上下文）⇒ ⇒ 不便宜；
    /// 而注册面要对**每个**模型目录问一次 ⇒ ⇒ 必须缓存。
    pub fn detect_cached() -> &'static HostCapabilities {
        static DETECTED: std::sync::OnceLock<HostCapabilities> = std::sync::OnceLock::new();
        DETECTED.get_or_init(HostCapabilities::detect)
    }

    /// 把 `Auto` 解析成一个**具体**设备。
    ///
    /// 「Auto」不是一个设备，是一个**问句** ⇒ ⇒ 记账与策略都必须先把它
    /// 问成答案（否则「Auto」在内存账本里会被当成 CPU，显存那一份就白放着）。
    pub fn resolved_device(&self) -> DeviceSelection {
        match self.device {
            DeviceKind::Cuda if self.features.cuda => DeviceSelection::Nvidia,
            DeviceKind::Metal if self.features.metal => DeviceSelection::Apple,
            DeviceKind::Cuda | DeviceKind::Metal => DeviceSelection::Cpu,
            DeviceKind::Cpu => DeviceSelection::Cpu,
            DeviceKind::Unknown => DeviceSelection::Auto,
        }
    }
}

/// 权重格式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    Safetensors,
    Gguf,
    Unknown,
}

/// 模型的事实（从盘上直接读出来的，不是配置里声称的）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelFacts {
    pub model_id: String,
    /// `config.json` 的 `model_type`（qwen2／qwen3／llama／qwen3_5…）。
    pub arch: String,
    /// 盘上权重的字节数。
    pub weight_bytes: u64,
    pub format: WeightFormat,
    /// 盘上 dtype 的字节数（4=f32，2=bf16/f16）。
    pub on_disk_bytes: Option<u8>,
}

impl ModelFacts {
    /// 直接从模型目录读事实。
    pub fn from_dir(dir: &std::path::Path) -> Result<ModelFacts, String> {
        let raw = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| format!("read {}: {e}", dir.join("config.json").display()))?;
        let v: serde_json::Value =
            serde_json::from_str(&raw).map_err(|e| format!("parse config.json: {e}"))?;
        let arch = v
            .get("model_type")
            .and_then(|m| m.as_str())
            .unwrap_or("unknown")
            .to_string();
        let on_disk_bytes = v
            .get("torch_dtype")
            .and_then(|m| m.as_str())
            .and_then(|s| match s {
                "float32" | "f32" => Some(4u8),
                "bfloat16" | "bf16" | "float16" | "f16" => Some(2u8),
                "float64" | "f64" => Some(8u8),
                _ => None,
            });

        let mut weight_bytes = 0u64;
        let mut format = WeightFormat::Unknown;
        if let Ok(rd) = std::fs::read_dir(dir) {
            for e in rd.flatten() {
                let name = e.file_name();
                let is_st = name.to_string_lossy().ends_with(".safetensors");
                let is_gguf = name.to_string_lossy().ends_with(".gguf");
                if !is_st && !is_gguf {
                    continue;
                }
                format = if is_st {
                    WeightFormat::Safetensors
                } else {
                    WeightFormat::Gguf
                };
                if let Ok(md) = e.metadata() {
                    weight_bytes = weight_bytes.saturating_add(md.len());
                }
            }
        }
        let model_id = dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();
        Ok(ModelFacts {
            model_id,
            arch,
            weight_bytes,
            format,
            on_disk_bytes,
        })
    }
}

/// 权重 dtype。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DTypeChoice {
    F32,
    BF16,
    F16,
}

impl DTypeChoice {
    pub fn name(self) -> &'static str {
        match self {
            DTypeChoice::F32 => "f32",
            DTypeChoice::BF16 => "bf16",
            DTypeChoice::F16 => "f16",
        }
    }

    /// 解析用户显式要求的 dtype（`f32`／`bf16`／`f16` 及其别名）。
    ///
    /// 认不出来就是 `None` —— **不猜**：调用方要么回退到默认并出声，
    /// 要么按「用户没说话」处理。
    pub fn from_env_value(raw: &str) -> Option<DTypeChoice> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "f32" | "float32" | "fp32" => Some(DTypeChoice::F32),
            "bf16" | "bfloat16" => Some(DTypeChoice::BF16),
            "f16" | "float16" | "fp16" | "half" => Some(DTypeChoice::F16),
            _ => None,
        }
    }
}

/// 加速策略。
///
/// `reasons` 说「为什么这么选」，`unavailable` 说「**要了但拿不到**的」——
/// 两个都要有：只有结论没有理由的策略，出了事没法追。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccelStrategy {
    /// 具体设备（`Auto` 只在「设备没测到」时保留）。
    pub device: DeviceSelection,
    pub dtype: DTypeChoice,
    /// **FA2 的前提**齐不齐（feature ＋ CUDA 设备 ＋ 架构覆盖）。
    ///
    /// ⚠ **这不是「会不会开」** —— 开不开由运行时的旋钮决定（用户没要
    /// 就不开；这是 FA 那一刀定下的纪律：**请求 ≠ 生效**）⇒ ⇒ 所以这里
    /// 说的是「**能不能**开」，不是「**开了**」。
    pub flash_attn_available: bool,
    /// 能跑这个模型的引擎；`None` ＝ **没有引擎能读它**。
    pub engine: Option<LlmImpl>,
    /// 按 `dtype` 物化后**真的要占**的字节（一个数字，一处算）。
    ///
    /// 注册面记账用的就是这个数 —— 旧路径写死 8 GB（0.92 GiB 的模型也
    /// 按 8 GB 记），那是**编出来的**数字，不是从盘上读出来的。
    pub required_bytes: u64,
    pub reasons: Vec<String>,
    pub unavailable: Vec<String>,
}

/// candle 腿覆盖的架构（`model_type`）。
pub fn candle_supported_arch(arch: &str) -> bool {
    matches!(
        arch,
        "qwen2" | "qwen3" | "llama" | "phi3" | "mistral" | "chatglm"
    )
}

/// 有 FlashAttention 路径的架构（`model_type`）。
///
/// 与 candle 腿的 FA 覆盖一致：qwen2／qwen3 走 vendored 补丁，llama 走
/// `vendor/llama_hd.rs`，mistral 走上游字段；**phi3／chatglm 没有这个字段**。
pub fn fa_supported_arch(arch: &str) -> bool {
    matches!(arch, "qwen2" | "qwen3" | "llama" | "mistral")
}

/// **qwen3 族**（`config.json` 的 `model_type`）。
///
/// ⚠ 与 `candle_arch.rs` 的分派**对齐**：这三个拼写都落到 `load_qwen3`
/// （`qwen3_5`／`qwen3.5` 会报「配置形状不支持」⇒ ⇒ 但**仍然是这一族**）。
fn qwen3_family(arch: &str) -> bool {
    matches!(arch, "qwen3" | "qwen3_5" | "qwen3.5")
}

/// 权重 dtype 的**前提**：这一档**跑不跑得起来**。
///
/// ⚠ 与「装不装得下」是**两条轴**，不许混：
/// · **装得下** ＝ 容量（显存／内存够不够）；
/// · **跑得起来** ＝ 前提（这一档在这个族／这个设备／这个 build 上能不能执行）。
///
/// ⇒ ⇒ 只判容量，就会放行一个「**装得下、一生成就炸**」的档 —— 真机撞过：
/// qwen3-0.6b 在 f32 下**加载成功**（显存真占着 2 460 MiB）⇒ ⇒ 然后炸在
/// forward pass（`flash-attn is only supported for f16/bf16 (F32)`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DtypePrereq {
    /// 没有前提：这一档跑得起来。
    None,
    /// 只认 f16/bf16：f32 会在**前向**炸（内核不收 f32）。
    HalfOnly,
}

/// 这一档的**前提**是什么（`DtypePrereq::None` ＝ 没有前提 ⇒ 跑得起来）。
///
/// ⚠ **单一真相源** —— 策略层（候选搜索）与运行时（load 期兜底）读的是
/// **同一个函数** ⇒ ⇒ 两边不许各说各话。
///
/// **真机读数**（2026-11-25，qwen3-0.6b，CUDA 设备，本 build 编了 `flash-attn`）：
/// f32 ⇒ **加载成功**、**生成炸在 forward pass** ⇒ ⇒ 这一格不是「装不下」，
/// 是「**跑不起来**」。
///
/// **源码级**（钉版 candle-transformers **0.11.0** rev `66a8cf1` 的
/// `src/models/qwen3.rs`）：注意力分派是
/// `#[cfg(not(feature="flash-attn"))]` CPU ⇒ `forward_cpu_flash_attn`／
/// `#[cfg(feature="flash-attn")]` 非 CPU ⇒ `forward_flash_attn`／
/// 否则 ⇒ `forward_standard_attn` ⇒ ⇒ **CUDA ＋ 编了 flash-attn** ⇒ ⇒
/// 走的是 candle-flash-attn 的 **CUDA** 内核 ⇒ ⇒ 那颗内核**只认 f16/bf16**
/// （`candle-flash-attn/src/lib.rs`：`bail!("flash-attn is only supported for f16/bf16")`）。
///
/// ⇒ ⇒ **雷区精确到**「**qwen3 族 ＋ CUDA 设备 ＋ 本 build 编了 `flash-attn`
/// ＋ f32**」—— **不是**「qwen3 一律不能 f32」：
/// · CPU ＋ 没编 flash-attn ⇒ candle-nn 的 **CPU** flash 内核
///   （`flash_attn<T: WithDType + Float>`）⇒ ⇒ **dtype 泛型**，f32 明确在列
///   ⇒ ⇒ 跑得起来；
/// · CPU ＋ 编了 flash-attn ⇒ `forward_standard_attn` ⇒ ⇒ 跑得起来；
/// · CUDA ＋ 没编 flash-attn ⇒ `forward_standard_attn` ⇒ ⇒ 跑得起来。
///
/// ⚠ **约束不许推广** —— qwen2／llama 族实测 f32 生成成功（vendored qwen2
/// 有闪／非闪两路；llama 走 `into_config(use_flash_attn)`）⇒ ⇒ 判据只到
/// 量过的那一族。
pub fn dtype_prereq_for_arch(
    arch: &str,
    dtype: DTypeChoice,
    device_is_cuda: bool,
    flash_attn_compiled: bool,
) -> DtypePrereq {
    if !matches!(dtype, DTypeChoice::F32) {
        // f16／bf16：到处都跑得起来。
        return DtypePrereq::None;
    }
    if !qwen3_family(arch) {
        // 约束只到量过的那一族。
        return DtypePrereq::None;
    }
    if device_is_cuda && flash_attn_compiled {
        return DtypePrereq::HalfOnly;
    }
    DtypePrereq::None
}

/// **这一族、这个设备、这个 build，跑得起来吗？**（dtype 前提这一维）
///
/// ⇒ ⇒ 策略层用它筛候选，运行时用它兜底 —— **同一个判据**。
pub fn dtype_runnable_for_arch(
    arch: &str,
    dtype: DTypeChoice,
    device_is_cuda: bool,
    flash_attn_compiled: bool,
) -> bool {
    dtype_prereq_for_arch(arch, dtype, device_is_cuda, flash_attn_compiled) == DtypePrereq::None
}

/// 「这一档跑不起来」的**成因**（人话）—— 进 reason／unavailable。
///
/// ⚠ 只在**真的有**前提时才有话说 ⇒ ⇒ 调用方先问 `dtype_prereq_for_arch`。
pub fn dtype_prereq_why(arch: &str) -> String {
    format!(
        "{} 族 ＋ CUDA 设备 ＋ 本 build 编了 `flash-attn` ⇒ ⇒ 注意力会走 **CUDA flash 内核** \
         ⇒ ⇒ 那颗内核**只认 f16/bf16** ⇒ ⇒ f32 会**加载成功、前向炸**",
        arch
    )
}

/// 权重按目标 dtype 物化后的字节数。
pub fn materialized_bytes(model: &ModelFacts, dtype: DTypeChoice) -> u64 {
    let on_disk = model.on_disk_bytes.unwrap_or(2).max(1) as u64;
    let target = match dtype {
        DTypeChoice::F32 => 4u64,
        DTypeChoice::BF16 | DTypeChoice::F16 => 2u64,
    };
    model.weight_bytes.saturating_mul(target) / on_disk
}

/// 用户**显式**要的（旋钮）—— 策略先听用户的，再谈自动。
///
/// 「自动」的意思是「**用户没说话**」，不是「**可以不听用户的**」。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StrategyRequest {
    /// 显式设备（`None` ＝ 用户没说话，由环境决定）。
    pub device: Option<DeviceSelection>,
    /// 显式权重 dtype（`None` ＝ 用户没说话）。
    pub dtype: Option<DTypeChoice>,
    /// 显式要／不要 FlashAttention（`None` ＝ 用户没说话）。
    ///
    /// ⚠ **只有用户提了才说话** —— 没人提 FA 时，策略对 FA **完全沉默**
    /// （注册日志不许被一条没人提的要求刷屏）；提了才回答「这条路开不
    /// 开得了，缺的是哪一个前提」。
    pub flash_attn: Option<bool>,
}

/// 注册面要的那份决策：**腿 ＋ 设备 ＋ 要占多少字节**。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimePlan {
    pub model_id: String,
    pub engine: LlmImpl,
    pub device: DeviceSelection,
    pub dtype: DTypeChoice,
    /// 按 `dtype` 物化后**真的要占**的字节。
    pub required_bytes: u64,
    pub reasons: Vec<String>,
    pub unavailable: Vec<String>,
}

/// 用户显式要的 FlashAttention（`NEURON_USE_FLASH_ATTN`）。
///
/// `None` ＝ **用户没说话**（或说了个认不出来的值 —— 认不出来就**不猜**，
/// 让运行时自己出声）；`Some(true)` ＝ 要；`Some(false)` ＝ 明确不要。
pub fn requested_flash_attn() -> Option<bool> {
    match crate::config::env_var("USE_FLASH_ATTN") {
        Some(raw) => match raw.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        },
        None => None,
    }
}

/// 用户显式要的权重 dtype（`NEURON_LLM_DTYPE`）。
///
/// **一处解析，两处消费**：策略层（记账）与 candle 腿（真的物化）读的是
/// 同一个旋钮 ⇒ ⇒ 不许一个说 f32、另一个说 bf16。
pub fn requested_dtype() -> Option<DTypeChoice> {
    match crate::config::env_var("LLM_DTYPE") {
        Some(raw) => match DTypeChoice::from_env_value(&raw) {
            Some(d) => Some(d),
            None => {
                eprintln!(
                    "[neuron] NEURON_LLM_DTYPE={raw:?} not recognised \
                     (want f32 | bf16 | f16); falling back to f32"
                );
                None
            }
        },
        None => None,
    }
}

/// 这台机器的**调度器平台键**（`cpu`／`cuda`／`metal`）。
///
/// 探测一次就缓存 —— 探测要问 candle 能不能开 CUDA ⇒ ⇒ 不便宜，
/// 而调度器每个周期都要看这个键。
pub fn detected_platform_key() -> &'static str {
    static KEY: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
    KEY.get_or_init(|| HostCapabilities::detect_cached().device.platform_key())
}

/// 把「盘上的事实」＋「这台机器」＋「用户的旋钮」变成一份**可执行的**计划。
///
/// 注册面（启动扫描／懒注册）直接用它：引擎腿、设备、要占的字节 —— 全部
/// 从**读出来的事实**算，不写死。
pub fn plan_with(
    caps: &HostCapabilities,
    dir: &std::path::Path,
    req: &StrategyRequest,
) -> Result<RuntimePlan, String> {
    let facts = ModelFacts::from_dir(dir).map_err(|e| format!("读不出模型事实：{e}"))?;
    let strategy = resolve_strategy(caps, &facts, req);
    let Some(engine) = strategy.engine else {
        let why = if strategy.unavailable.is_empty() {
            "这个 build 没有任何一条腿能读它".to_string()
        } else {
            strategy.unavailable.join("；")
        };
        return Err(why);
    };
    Ok(RuntimePlan {
        model_id: facts.model_id.clone(),
        engine,
        device: strategy.device,
        dtype: strategy.dtype,
        required_bytes: strategy.required_bytes,
        reasons: strategy.reasons,
        unavailable: strategy.unavailable,
    })
}

/// 同 [`plan_with`]，但自己探一次环境（一次性调用用；批量用 `plan_with`）。
pub fn plan_for_dir(dir: &std::path::Path, req: &StrategyRequest) -> Result<RuntimePlan, String> {
    plan_with(HostCapabilities::detect_cached(), dir, req)
}

/// 按环境与模型事实算出加速策略。
///
/// **纯函数** —— 不读环境、不读全局状态 ⇒ ⇒ 可测、可复现。
pub fn resolve_strategy(
    caps: &HostCapabilities,
    model: &ModelFacts,
    req: &StrategyRequest,
) -> AccelStrategy {
    let mut reasons: Vec<String> = Vec::new();
    let mut unavailable: Vec<String> = Vec::new();

    // ---- 1. 设备 ----
    //
    // **用户显式指定的设备优先** —— 策略只在「用户没说话」时才替它做主；
    // 用户说了而这台机器做不到 ⇒ ⇒ **说出来**，不偷偷换一个设备替他跑
    // （那是「说假话」的另一种形状：用户要 GPU，你给他 CPU，他以为是 GPU）。
    let mut device = match req.device {
        Some(DeviceSelection::Auto) | None => match caps.device {
            DeviceKind::Cuda if caps.features.cuda => {
                reasons.push("设备：探测到 CUDA 设备且本 build 编了 `cuda` ⇒ 走 GPU".to_string());
                DeviceSelection::Nvidia
            }
            DeviceKind::Cuda => {
                unavailable.push(
                    "设备：探测到 CUDA 设备，但本 build **没有编 `cuda` feature** ⇒ GPU 那条路开不了（要 `--features cuda`）".to_string(),
                );
                reasons.push("设备：GPU 用不了 ⇒ 走 CPU".to_string());
                DeviceSelection::Cpu
            }
            DeviceKind::Metal if caps.features.metal => {
                reasons.push("设备：探测到 Metal 设备且本 build 编了 `metal` ⇒ 走 GPU".to_string());
                DeviceSelection::Apple
            }
            DeviceKind::Metal => {
                unavailable.push(
                    "设备：探测到 Metal 设备，但本 build **没有编 `metal` feature** ⇒ GPU 那条路开不了".to_string(),
                );
                reasons.push("设备：GPU 用不了 ⇒ 走 CPU".to_string());
                DeviceSelection::Cpu
            }
            DeviceKind::Cpu => {
                // ⚠ **机器有卡 ≠ 这个 build 能用** —— 而「有没有卡」是**机器的
                // 事实**，「能不能用」是**build 的能力** ⇒ ⇒ 两个字段
                // （2026-11-25 实测撞出来的）。
                //
                // ⚠ 旧版这里只有一句「没有可用的加速设备 ⇒ 走 CPU」⇒ ⇒ 在
                // **没编 `cuda`** 的出厂 build 里，探测设备的那条腿被
                // `#[cfg(feature = "cuda")]` 门控掉了 ⇒ ⇒ `caps.device`
                // 永远是 `Cpu` ⇒ ⇒ **「GPU 那条路开不了」那句话在真实路径
                // 上不可达** ⇒ ⇒ 用户只看到 CPU，**一句真话都没有**。
                match caps.machine_device {
                    DeviceKind::Cuda => {
                        unavailable.push(
                            "设备：这台机器**有** NVIDIA 卡（nvidia-smi 测到），但本 build **没有编 `cuda` feature** ⇒ GPU 那条路开不了（要 `--features cuda`）".to_string(),
                        );
                        reasons.push(
                            "设备：机器**有**卡，但这个 build 编不进 GPU ⇒ 走 CPU".to_string(),
                        );
                    }
                    DeviceKind::Metal => {
                        unavailable.push(
                            "设备：这台机器**有** Apple/Metal 设备，但本 build **没有编 `metal` feature** ⇒ GPU 那条路开不了（要 `--features metal`）".to_string(),
                        );
                        reasons.push(
                            "设备：机器**有**卡，但这个 build 编不进 GPU ⇒ 走 CPU".to_string(),
                        );
                    }
                    _ => {
                        reasons.push("设备：没有可用的加速设备 ⇒ 走 CPU".to_string());
                    }
                }
                DeviceSelection::Cpu
            }
            DeviceKind::Unknown => {
                reasons.push(
                    "设备：设备种类**没测到**（这个 build 没有能问的腿）⇒ 保留 `Auto`，由运行时自己探"
                        .to_string(),
                );
                // ⚠ 机器的事实**照样**说得出来（那台腿不需要编任何 feature）
                // ⇒ ⇒ 「机器有卡」不许被这个 build 的沉默一起吞掉。
                if caps.machine_device == DeviceKind::Cuda {
                    unavailable.push(
                        "设备：这台机器**有** NVIDIA 卡（nvidia-smi 测到），但这个 build 连 candle 都没编 ⇒ GPU 那条路开不了".to_string(),
                    );
                } else if caps.machine_device == DeviceKind::Metal {
                    unavailable.push(
                        "设备：这台机器**有** Apple/Metal 设备，但这个 build 连 candle 都没编 ⇒ GPU 那条路开不了".to_string(),
                    );
                }
                DeviceSelection::Auto
            }
        },
        Some(d) => {
            reasons.push(format!(
                "设备：用户显式指定了 {d:?} ⇒ 照办（策略不替你改主意）"
            ));
            match d {
                DeviceSelection::Nvidia if !caps.features.cuda => unavailable.push(
                    if caps.machine_device == DeviceKind::Cuda {
                        "设备：你要的是 NVIDIA，但本 build **没有编 `cuda` feature** ⇒ 这条路开不了（要 `--features cuda`）".to_string()
                    } else {
                        "设备：你要的是 NVIDIA，但本 build **没有编 `cuda` feature**，而且这台机器**也没测到** NVIDIA 卡 ⇒ 这条路开不了".to_string()
                    },
                ),
                DeviceSelection::Apple if !caps.features.metal => unavailable.push(
                    "设备：你要的是 Apple/Metal，但本 build **没有编 `metal` feature** ⇒ 这条路开不了".to_string(),
                ),
                _ => {}
            }
            d
        }
    };

    // ---- 1b. 池 ----
    //
    // ⚠ **池必须提前算** —— 「装不下时会**无损降**」那句话**只有在池测到了
    // 的时候才是真的** ⇒ ⇒ 所以它得在 dtype 那段**之前**就知道池测没测到
    // （刀 16 真机撞出来的：Android 的 `target_os` 是 `"android"` ⇒ ⇒
    // 内存轴恒为 `None` ⇒ ⇒ **降级窗整个不存在** ⇒ ⇒ 那句话是**假话**）。
    let pool_of = |d: DeviceSelection| -> Option<(u64, &'static str)> {
        match d {
            DeviceSelection::Nvidia | DeviceSelection::Apple => {
                caps.vram_bytes.map(|v| (v, "显存"))
            }
            // ⚠⚠️ **内存池用「现在可用」，不是「总量」** —— 真机撞出来的
            // （刀 16）：rk3566 上 MemTotal 1.91 GiB 而 MemAvailable 1.04 GiB
            // ⇒ ⇒ 差 54% ⇒ ⇒ 用总量 ⇒ ⇒ 策略说「1.40 GiB 的 bf16 装得下」
            // ⇒ ⇒ 真加载 ⇒ ⇒ **被 OOM killer 杀掉**（逐字：
            // `oom_reaper: reaped process 25746 (dtype_probe)`）⇒ ⇒
            // ⇒ 「装得下」这格**在真机上说假话**。
            //
            // ⚠ **回落顺序**：可用 ⇒ ⇒ 总量 ⇒ ⇒ `None`（没测到）。
            //   ⇒ 总量是**乐观上界** ⇒ ⇒ 只在「可用测不到」时用它 ⇒ ⇒ 两级都
            //   ⇒ 测不到 ⇒ ⇒ 池**判不了** ⇒ ⇒ 不许猜（刀 10 纪律③）。
            _ => caps
                .ram_available_bytes
                .or(caps.ram_bytes)
                .map(|v| (v, "内存")),
        }
    };

    // ---- 2. dtype ----
    //
    // ⚠ **这一段改过一次口径**（2026-11-25）：旧版写「加速卡 ⇒ bf16」
    // ⇒ ⇒ 那是**策略的愿望**，不是**运行时会做的事** —— candle 腿的
    // `resolve_weight_dtype()` 在 `NEURON_LLM_DTYPE` 没设时是 **f32**
    // ⇒ ⇒ 策略若按 bf16 记账，就会比真实占用**少算一半** ⇒ ⇒ 所以现在
    // 策略跟着**运行时**走：用户显式要求就照办；没要求就照运行时的默认
    // （f32）；加速卡上想省显存 ⇒ **建议**（带旋钮名），不替它改。
    let mut dtype = match req.dtype {
        Some(d) => {
            reasons.push(format!(
                "dtype：用户显式要求 {}（`NEURON_LLM_DTYPE`）⇒ 照办",
                d.name()
            ));
            d
        }
        None => {
            // ⚠ **承诺只在池测到时成立** —— 池测不到 ⇒ ⇒ **降级窗不存在**
            // ⇒ ⇒ 不许说「装不下时会无损降」⇒ ⇒ 不许说假话（刀 16）。
            if pool_of(device).is_some() {
                reasons.push(
                    "dtype：用户没要求（`NEURON_LLM_DTYPE` 没设）⇒ ⇒ 起始按运行时的默认 f32 算 ⇒ ⇒ 装不下时会**无损降**到 bf16（见降级段）"
                        .to_string(),
                );
            } else {
                reasons.push(
                    "dtype：用户没要求（`NEURON_LLM_DTYPE` 没设）⇒ ⇒ 起始按运行时的默认 f32 算 ⇒ ⇒ **但这个池没测到** ⇒ ⇒ 「装不下时**无损降**到 bf16」这条路**不存在**（降级候选一个都不生成）⇒ ⇒ 装不装得下**判不了**（不许猜）"
                        .to_string(),
                );
                unavailable.push(
                    "dtype：这个池**没测到**（内存／显存）⇒ ⇒ 这一档**装不装得下判不了** ⇒ ⇒ 不许假装降级窗存在 ⇒ ⇒ 也不许假装装得下"
                        .to_string(),
                );
            }
            DTypeChoice::F32
        }
    };

    // 装得下吗 —— 按**实际会用的** dtype 算，不是按策略的愿望算。
    let mut need = materialized_bytes(model, dtype);

    // ---- 2b. 自动降级：**先降精度，再换设备** ----
    //
    // ⚠ 这一格此前是**缺的**：策略层知道「装不下」（进 unavailable），但**不会
    // 自己降级** ⇒ ⇒ 注册面于是拿着「GPU ＋ f32」去注册 ⇒ ⇒ 要么注册被拒
    // （`InsufficientResources`），要么在 load 时才炸 ⇒ ⇒
    // **「按设备自动降级」此前只有一半**：会**诊断**，不会**降级**。
    //
    // ⚠ **为什么精度排在设备前面**（2026-11-25 刀 12 订正刀 10 的口径）：
    // · **降到 bf16** 在「盘上就是每参数 2 字节（bf16／f16）」时是**无损**的
    //   —— 那份数据本来就是 2 字节，f32 只是**上采样**出来的 ⇒ ⇒ 落回 bf16
    //   **一个字节都不丢** ⇒ ⇒ 代价为零；
    // · **换设备**（GPU ⇒ CPU）⇒ ⇒ 代价是**速度**（真机实测 GPU 是 CPU 的
    //   6.0×，反过来就是掉一个数量级）；
    // ⇒ ⇒ 所以能靠精度解决的问题，**不许**先动设备 ⇒ ⇒ 设备的退让排在
    //     精度的后面 —— 这不是「降级得更狠」，是**降级得更便宜**。
    //
    // ⚠ **降级的三条纪律**（与刀 10 那条同形）：
    // ① **只退不升**（f32 ⇒ ⇒ bf16 合法；**升精度不自动**；用户**显式**
    //    指定了设备／dtype ⇒ ⇒ **不许偷偷换** —— 那是「说假话」的另一种形状）；
    // ② **只在「装得下」时才降** —— 降完还是装不下 ⇒ ⇒ **不降级**（降级
    //    救不回来 ⇒ 照实报「装不下」）；
    // ③ **池没测到 ⇒ 不许猜** —— 探不到显存 ⇒ ⇒ 既不许当它装得下，也不许当
    //    它装不下 ⇒ ⇒ 不降级（保守：留在原样，让账本去拒）。
    //
    // ⚠ `pool_of` **已经提前算好了**（见 1b 段）⇒ ⇒ 池测没测到，dtype 那段
    //    就已经知道了 ⇒ ⇒ 不许到这儿才发现「降级窗其实不存在」。

    // ⚠ **自动降精度只在「无损」时做** —— 盘上就是 2 字节/参数 ⇒ ⇒ bf16 是
    //    它本来的样子 ⇒ ⇒ 落回 bf16 不丢任何信息 ⇒ ⇒ 才可以自动做。
    //    ⚠️ 盘上是 f32（每参数 4 字节）或量化（1 字节）⇒ ⇒ 降到 bf16 **会
    //    丢精度** ⇒ ⇒ **不许自动做** ⇒ ⇒ 那只是**建议**（见下面的 unavailable
    //    分支）—— 「悄悄换精度」是另一种形状的「说假话」。
    let lossless_bf16 =
        req.dtype.is_none() && dtype == DTypeChoice::F32 && model.on_disk_bytes == Some(2);

    let start_device = device;
    let start_dtype = dtype;

    // ---- 「跑得起来吗」这一条腿（与「装得下」并列的第二条）----
    //
    // ⚠ **这是这一刀的全部意义**：候选必须**同时**满足两件事 ——
    // ① **装得下**（容量：显存／内存够不够）；
    // ② **跑得起来**（前提：这一档在这个族／这个设备／这个 build 上能不能执行）。
    //
    // ⚠ **只判 ① 会放行「装得下、一生成就炸」的档** —— 真机撞过：qwen3-0.6b
    // 在 f32 下**加载成功**（显存真占着 2 460 MiB）⇒ ⇒ 然后炸在 forward pass
    // ⇒ ⇒ 那是「加载成功、生成炸」—— 服务会倒在最不该倒的地方。
    //
    // ⚠ **判据是单一真相源**：[`dtype_runnable_for_arch`] —— 运行时（load 期
    // 兜底）读的是**同一个**函数 ⇒ ⇒ 两边不许各说各话。
    let runnable = |dev: DeviceSelection, dt: DTypeChoice| -> bool {
        dtype_runnable_for_arch(
            &model.arch,
            dt,
            matches!(dev, DeviceSelection::Nvidia),
            caps.features.flash_attn,
        )
    };
    let start_runnable = runnable(start_device, start_dtype);

    // 候选序列＝**优先级**：从「最贴近用户要的」排到「最省的」。
    //
    // ⚠ **降级候选只在「起始池测到了」时才存在** —— 探不到显存 ⇒ ⇒ 既
    // 不许当它装得下，也不许当它装不下 ⇒ ⇒ 那就没有「降级」的依据
    // （刀 10 的纪律③）⇒ ⇒ 保持原样，让账本去拒。
    let start_pool = pool_of(start_device);
    let mut candidates: Vec<(DeviceSelection, DTypeChoice)> = vec![(start_device, start_dtype)];
    if start_pool.is_some() {
        if lossless_bf16 {
            candidates.push((start_device, DTypeChoice::BF16));
        }
        if req.device.is_none()
            && matches!(
                start_device,
                DeviceSelection::Nvidia | DeviceSelection::Apple
            )
        {
            candidates.push((DeviceSelection::Cpu, start_dtype));
            if lossless_bf16 {
                candidates.push((DeviceSelection::Cpu, DTypeChoice::BF16));
            }
        }
        // ⚠ **「跑不起来」的起始档 ⇒ ⇒ 补一条「同一设备上跑得起来」的档** ——
        // f16／bf16 在哪儿都跑得起来 ⇒ ⇒ 这是**唯一**的出路。
        //
        // ⚠ **但用户显式要了 dtype ⇒ ⇒ 不许偷偷换**（那是「说假话」的另一种
        // 形状）⇒ ⇒ 照实报「这一档跑不起来」＋ 指出出路 ⇒ ⇒ 由运行时在
        // **load 期**拒（服务不倒），不许「加载成功、生成炸」。
        //
        // ⚠ **这条例外盖过「自动降精度只在无损时做」** —— 「跑不起来」比
        // 「丢精度」更硬：丢精度还能出结果，跑不起来连输出都没有 ⇒ ⇒ 但
        // 理由文案必须**照实**说「这一档会丢精度」，不许顺手写成「无损」。
        if req.dtype.is_none() && !start_runnable {
            let already = candidates
                .iter()
                .any(|(d, dt)| *d == start_device && runnable(*d, *dt));
            if !already {
                for alt in [DTypeChoice::BF16, DTypeChoice::F16] {
                    if runnable(start_device, alt) {
                        candidates.insert(1, (start_device, alt));
                        break;
                    }
                }
            }
        }
    }

    // 取**第一个「装得下」且「跑得起来」**的候选
    // （池没测到 ⇒ 那个候选**判不了** ⇒ 跳过 ⇒ 不猜）。
    let mut picked: Option<(DeviceSelection, DTypeChoice, u64, &'static str)> = None;
    // ⚠ 「装得下但**跑不起来**」的那一档 ⇒ ⇒ 要**单独**记下来报，不许混进
    // 「装不下」—— 两件事的**出路不一样**。
    let mut blocked_by_prereq: Option<(DeviceSelection, DTypeChoice, u64)> = None;
    for (dev, dt) in &candidates {
        let Some((pool, pool_name)) = pool_of(*dev) else {
            continue;
        };
        let need_here = materialized_bytes(model, *dt);
        if need_here > pool {
            continue; // 装不下 ⇒ ⇒ 不是候选
        }
        if !runnable(*dev, *dt) {
            blocked_by_prereq = Some((*dev, *dt, need_here));
            continue; // 装得下，但**跑不起来** ⇒ ⇒ 也不是候选
        }
        picked = Some((*dev, *dt, need_here, pool_name));
        break;
    }

    if let Some((dev, dt, need_here, pool_name)) = picked {
        if dt != start_dtype {
            if start_runnable {
                reasons.push(format!(
                    "dtype：{pool_name}装不下 f32（要 {orig} 字节，而{pool_name}只有 {pool} 字节）⇒ ⇒ **自动降到 bf16**（要 {need_here} 字节）—— 盘上就是每参数 2 字节 ⇒ ⇒ **不丢精度**",
                    orig = materialized_bytes(model, start_dtype),
                    pool = pool_of(dev).map(|p| p.0).unwrap_or(0),
                ));
            } else {
                // ⚠ **「跑不起来」不是「装不下」** ⇒ ⇒ 两句话不许混。
                reasons.push(format!(
                    "dtype：起始档 {start} **跑不起来**（{why}）⇒ ⇒ 自动落到 {dt}（要 {need_here} 字节）—— {lossless}",
                    start = start_dtype.name(),
                    why = dtype_prereq_why(&model.arch),
                    dt = dt.name(),
                    need_here = need_here,
                    lossless = if model.on_disk_bytes == Some(2) {
                        "盘上就是每参数 2 字节 ⇒ ⇒ **不丢精度**"
                    } else {
                        "⚠ 盘上不是每参数 2 字节 ⇒ ⇒ 这一档**会丢精度**（但 f32 那一档**根本跑不起来** ⇒ ⇒ 丢精度还能出结果，跑不起来连输出都没有）"
                    },
                ));
            }
        }
        if dev != start_device {
            reasons.push(format!(
                "设备：{why} ⇒ ⇒ **自动退 CPU**（{ram_note}）",
                why = if start_runnable {
                    format!(
                        "显存装不下（要 {need_here} 字节，而显存只有 {vram} 字节）",
                        vram = caps.vram_bytes.unwrap_or(0)
                    )
                } else {
                    format!(
                        "GPU 上这一档**跑不起来**（{why}）",
                        why = dtype_prereq_why(&model.arch)
                    )
                },
                ram_note = if start_runnable {
                    format!("内存 {ram} 字节装得下", ram = caps.ram_bytes.unwrap_or(0))
                } else {
                    format!(
                        "CPU 上走的是**另一条**注意力路 ⇒ ⇒ 这一档跑得起来；内存 {ram} 字节装得下",
                        ram = caps.ram_bytes.unwrap_or(0)
                    )
                },
            ));
        }
        device = dev;
        dtype = dt;
        need = need_here;
    }

    // ⚠ **「装得下但跑不起来」要单独报** —— 不许混进「装不下」：
    // 装不下的出路是「换设备／降精度」，跑不起来的出路只有「换 dtype」。
    if picked.is_none()
        && let Some((_dev, dt, need_here)) = blocked_by_prereq
    {
        unavailable.push(format!(
                "dtype：{dt} 这一档**装得下**（要 {need_here} 字节）但**跑不起来**（{why}）⇒ ⇒ 这一族要显式设 `NEURON_LLM_DTYPE=bf16`（或 f16）才跑得起来",
                dt = dt.name(),
                need_here = need_here,
                why = dtype_prereq_why(&model.arch),
        ));
    }

    // 池按**最终**的设备算（降级之后就不是显存池了）。
    //
    // ⚠⚠️ **必须走 `pool_of`** —— 内存池是「**现在可用**」而不是「总量」
    //   （刀 16 真机撞出来的：rk3566 上 MemTotal 1.91 GiB 而 MemAvailable
    //   1.04 GiB）⇒ ⇒ 这里若直接用 `caps.ram_bytes` ⇒ ⇒ **同一个池两套值**
    //   ⇒ ⇒ 前面判「装不下」用可用、这里判「装不下」用总量 ⇒ ⇒
    //   ⇒ **同一档在同一个决策里被判出两个相反的结论**。
    let on_gpu = matches!(device, DeviceSelection::Nvidia | DeviceSelection::Apple);
    let (pool, pool_name) = if on_gpu {
        (caps.vram_bytes, "显存")
    } else {
        (caps.ram_available_bytes.or(caps.ram_bytes), "内存")
    };
    if let Some(pool) = pool
        && need > pool
    {
        unavailable.push(format!(
            "dtype：权重按 {} 物化要 {need} 字节，而{pool_name}只有 {pool} 字节 ⇒ 装不下",
            dtype.name()
        ));
        // 给一条**可操作**的出路（是建议，不是偷偷改）。
        // ⚠ 只建议**更省**的那一档 —— 建议「升精度」是荒谬的。
        //
        // ⚠⚠️ **出路必须「真的装得下」** —— 真机撞出来的（刀 16）：
        //   rk3566 上可用内存 1.04 GiB，而 bf16 物化 1.40 GiB ⇒ ⇒
        //   ⇒ 「要装下得设 bf16」这句话是**假话**（设了也装不下）⇒ ⇒
        //   ⇒ 不许给一条走不通的出路。
        //
        // ⚠⚠️ **「会不会丢精度」要按盘上事实说** —— 盘上**就是** 2 字节/参数
        //   （`on_disk_bytes == Some(2)`）⇒ ⇒ bf16 是它本来的样子 ⇒ ⇒
        //   ⇒ **不丢精度** ⇒ ⇒ 照实说「不丢精度」⇒ ⇒ 而「自动降级为什么
        //   没做」要另找原因（**装完还是装不下**）⇒ ⇒ 不许拿「会丢精度」
        //   当挡箭牌（那是**说假话**）。
        if dtype == DTypeChoice::F32 {
            let alt = DTypeChoice::BF16;
            let alt_bytes = materialized_bytes(model, alt);
            if alt_bytes <= pool {
                let lossless = model.on_disk_bytes == Some(2);
                reasons.push(format!(
                        "dtype：要装下得显式设 `NEURON_LLM_DTYPE={}`（要 {alt_bytes} 字节）—— {lossless_note} ⇒ ⇒ 所以只**建议**，不自动改",
                        alt.name(),
                        lossless_note = if lossless {
                            "这一档**不丢精度**（盘上就是每参数 2 字节 ⇒ ⇒ bf16 是它本来的样子）"
                        } else {
                            "这一档**会丢精度**（盘上不是 2 字节/参数）"
                        },
                    ));
            }
        }
    }
    if on_gpu && dtype == DTypeChoice::F32 {
        reasons.push(
            "dtype：加速卡上 bf16 更省显存，但**自动降精度只在无损时做**（盘上就是每参数 2 字节）⇒ ⇒ 这一档是**建议**（`NEURON_LLM_DTYPE=bf16`），不是偷偷改"
                .to_string(),
        );
    }

    // ---- 3. FlashAttention ----
    //
    // **只说「能不能」，不说「开不开」** —— 开不开由运行时的旋钮决定
    // （用户没要就不开）⇒ ⇒ 所以用户**没提**时，这里**不**产 reason／
    // unavailable：没人提的要求，不许在注册日志里刷屏。
    //
    // ⚠ 口径**对齐运行时的降级顺序**：feature → 架构 → 设备 → **dtype**
    // （缺哪个是不同的东西要修，先找到的那个最便宜解释）。dtype 这一条
    // 是**真机撞出来的**：FA 内核只认 f16/bf16，而运行时的默认 dtype 是
    // f32 ⇒ ⇒ 默认档下「FA 可用」是**假话** ⇒ ⇒ 现在它进这个判定。
    let fa_arch_ok = fa_supported_arch(&model.arch);
    // 内核只接受 f16/bf16 —— f32 进不去。
    let fa_dtype_ok = !matches!(dtype, DTypeChoice::F32);
    let flash_attn_available = caps.features.flash_attn
        && matches!(device, DeviceSelection::Nvidia)
        && fa_arch_ok
        && fa_dtype_ok;

    // 用户**明确要了** FA ⇒ ⇒ 那才轮到策略回答「这条路开不开得了」，
    // 并点名缺的那一个前提（**只报缺的那个**，不替用户改主意）。
    if req.flash_attn == Some(true) {
        if flash_attn_available {
            reasons.push(
                "FA：你要了，前提齐（feature ＋ CUDA ＋ 架构 ＋ dtype）⇒ ⇒ 但**开不开**仍由运行时的旋钮决定"
                    .to_string(),
            );
        } else {
            let mut missing: Vec<&str> = Vec::new();
            if !caps.features.flash_attn {
                missing.push("本 build 没编 `flash-attn` feature（要 `--features flash-attn`）");
            }
            if !matches!(device, DeviceSelection::Nvidia) {
                missing.push("设备不是 CUDA（FA 内核是 CUDA-only）");
            }
            if !fa_arch_ok {
                missing.push("这个架构在 candle 腿里没有 FA 路径");
            }
            if !fa_dtype_ok {
                missing.push(
                    "权重 dtype 是 f32（FA 内核只认 f16/bf16 ⇒ ⇒ 显式设 `NEURON_LLM_DTYPE=bf16`）",
                );
            }
            unavailable.push(format!(
                "FA：你要了，但这条路开不了 ⇒ ⇒ {}",
                missing.join("；")
            ));
        }
    }

    // ---- 4. 引擎 ----
    let engine = resolve_engine(caps, model, &mut reasons, &mut unavailable);

    AccelStrategy {
        device,
        dtype,
        flash_attn_available,
        engine,
        required_bytes: need,
        reasons,
        unavailable,
    }
}

/// 选引擎。
///
/// `LlmImpl` 的每个变体都跟着 feature 走 ⇒ ⇒ 这里必须按 feature 分支，
/// 否则没编那条腿的 build 连编译都过不了。
fn resolve_engine(
    caps: &HostCapabilities,
    model: &ModelFacts,
    reasons: &mut Vec<String>,
    unavailable: &mut Vec<String>,
) -> Option<LlmImpl> {
    match model.format {
        WeightFormat::Gguf => {
            #[cfg(feature = "llamacpp")]
            {
                if caps.features.llamacpp {
                    reasons.push("引擎：权重是 GGUF ⇒ llamacpp".to_string());
                    return Some(LlmImpl::LlamaCpp);
                }
            }
            unavailable
                .push("引擎：权重是 GGUF，但本 build 没编 `llamacpp` ⇒ 没有引擎能读它".to_string());
            None
        }
        WeightFormat::Safetensors => {
            #[cfg(feature = "candle")]
            {
                if caps.features.candle && candle_supported_arch(&model.arch) {
                    reasons.push(format!(
                        "引擎：safetensors ＋ 架构 {} 在 candle 腿的覆盖里 ⇒ candle",
                        model.arch
                    ));
                    return Some(LlmImpl::Candle);
                }
            }
            #[cfg(feature = "mistralrs")]
            {
                if caps.features.mistralrs {
                    reasons.push(format!(
                        "引擎：safetensors ＋ 架构 {} ⇒ mistralrs 腿的 loader",
                        model.arch
                    ));
                    return Some(LlmImpl::MistralRs);
                }
            }
            unavailable.push(format!(
                "引擎：架构 {} 在本 build 已编的腿里没有 loader（candle 覆盖：qwen2／qwen3／llama／phi3／mistral／chatglm）",
                model.arch
            ));
            None
        }
        WeightFormat::Unknown => {
            unavailable
                .push("引擎：权重格式认不出来（既不是 safetensors 也不是 gguf）".to_string());
            None
        }
    }
}

/// 设备种类：能问 candle 就问 candle。
fn detect_device_kind() -> DeviceKind {
    #[cfg(feature = "candle")]
    {
        #[cfg(feature = "cuda")]
        if candle::Device::new_cuda(0).is_ok() {
            return DeviceKind::Cuda;
        }
        #[cfg(feature = "metal")]
        if candle::Device::new_metal(0).is_ok() {
            return DeviceKind::Metal;
        }
        DeviceKind::Cpu
    }
    #[cfg(not(feature = "candle"))]
    DeviceKind::Unknown
}

/// 这台机器**真有**的加速设备 —— **与本 build 编了什么无关**。
///
/// 腿（**都不需要编任何 feature** ⇒ ⇒ 所以「机器有卡」这件事在**每一个**
/// build 里都问得到，包括没编 `cuda` 的那个）：
/// 1. `nvidia-smi`（外部进程）—— 见 ≥1 张 NVIDIA 卡；
/// 2. macOS 的 `system_profiler` —— 见 Apple GPU（Metal）；
/// 3. 都不行 ⇒ **Windows／Linux：`Cpu`**（真的只有 CPU），
///    **macOS：`Unknown`**（探不到 ≠ 没有 ⇒ 不许说成「只有 CPU」）。
///
/// ⚠ **与 `detect_device_kind()` 的区别**：那一条腿是 **feature 门控**的
/// ⇒ ⇒ 在没编 `cuda` 的 build 里它只会说「CPU」⇒ ⇒ 机器的事实就丢了。
pub fn detect_machine_device() -> DeviceKind {
    if nvidia_smi_gpu_count() > 0 {
        return DeviceKind::Cuda;
    }
    #[cfg(target_os = "macos")]
    {
        if macos_has_apple_gpu() {
            DeviceKind::Metal
        } else {
            DeviceKind::Unknown
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        DeviceKind::Cpu
    }
}

/// 纯函数：数 `nvidia-smi --query-gpu=index --format=csv,noheader` 的行数。
///
/// ⚠ **一张卡就是一张卡** —— 哪怕 `memory.total` 是 `[N/A]`（Windows WDDM
/// 下某些列就是它）⇒ ⇒ 那张卡**仍然存在** ⇒ ⇒ 计数**不受**显存可读性影响。
/// ⇒ ⇒ 这条腿与「显存测不测得到」**解耦** ⇒ ⇒ 所以「机器有卡」与
/// 「显存测到了」是两个读数（不许拿其中一个冒充另一个）。
///
/// ⚠ 负控：非数字的行（`nvidia-smi` 的报错）**不是**一张卡。
pub fn parse_nvidia_smi_devices(csv: &str) -> usize {
    csv.lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .filter(|l| {
            l.split(',')
                .next()
                .and_then(|f| f.trim().parse::<u32>().ok())
                .is_some()
        })
        .count()
}

/// 问 `nvidia-smi`：这台机器有几张 NVIDIA 卡（**不需要编 `cuda`**）。
fn nvidia_smi_gpu_count() -> usize {
    for exe in nvidia_smi_candidates() {
        let Ok(out) = Command::new(&exe)
            .args(["--query-gpu=index", "--format=csv,noheader"])
            .output()
        else {
            continue;
        };
        if !out.status.success() {
            continue;
        }
        let n = parse_nvidia_smi_devices(&String::from_utf8_lossy(&out.stdout));
        if n > 0 {
            return n;
        }
    }
    0
}

/// macOS：这台机器有没有 Apple GPU（Metal）。
#[cfg(target_os = "macos")]
fn macos_has_apple_gpu() -> bool {
    let Ok(out) = Command::new("system_profiler")
        .args(["SPDisplaysDataType"])
        .output()
    else {
        return false;
    };
    parse_macos_metal_support(&String::from_utf8_lossy(&out.stdout))
}

/// 纯函数：macOS 的 `system_profiler SPDisplaysDataType` 输出里有没有 Metal。
///
/// ⚠ **只认「Metal Support: Yes」这一行** —— 同一份报告里也可能列着别的 GPU
/// （eGPU／虚拟显示）⇒ ⇒ 不许拿「有显卡」当「有 Metal」。
pub fn parse_macos_metal_support(report: &str) -> bool {
    report.lines().any(|l| {
        let t = l.trim();
        let Some((k, v)) = t.split_once(':') else {
            return false;
        };
        k.trim() == "Metal Support" && v.trim().eq_ignore_ascii_case("Yes")
    })
}

/// 主机物理内存。
///
/// ⚠⚠️ **Android 必须与 Linux 同腿**（2026-11-26 刀 16，真机撞出来的）——
/// ⇒ ⇒ **Android 的 `target_os` 是 `"android"`，不是 `"linux"`** ⇒ ⇒ 只写
/// `#[cfg(target_os = "linux")]` 的话 ⇒ ⇒ **手机这一轴恒为 `None`** ⇒ ⇒
///
/// 后果**不是**「少一个读数」，是**降级窗整个消失**：
/// · 策略层的容量轴（`pool_of` ⇒ `caps.ram_bytes`）在手机上**恒为 `None``
///   ⇒ ⇒ `start_pool.is_some()` 为假 ⇒ ⇒ **降级候选一个都不生成** ⇒ ⇒
///   ⇒ 策略只会挑「起始档」⇒ ⇒ 一台 1.91 GiB 的手机会拿到 **f32（2.40 GiB）**
///   ⇒ ⇒ **在 load 期 OOM**（不是「装不下」被拒，是**被杀**）；
/// · 而 `pool_of` 返回 `None` 时那一档**判不了容量** ⇒ ⇒ 策略连「装不下」
///   都说不出来 ⇒ ⇒ 「失败长得像成功」的那一族。
///
/// ⇒ ⇒ 真机读数（rk3566 / 1.91 GiB）：`MemTotal: 2004240 kB` ⇒ ⇒ 修好后
/// 内存轴**活过来** ⇒ ⇒ 降级窗**非空** ⇒ ⇒ f32 装不下 ⇒ ⇒ 无损落 bf16。
///
/// ⇒ ⇒ `/proc/meminfo` 在 Android 上是 `-r--r--r--`（**全局可读**）⇒ ⇒
/// App 进程（u0_a177）也读得到 ⇒ ⇒ 这一腿在手机上是**真的**。
fn ram_bytes_platform() -> Option<u64> {
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        let mut s: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
        s.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        let ok = unsafe { GlobalMemoryStatusEx(&mut s) };
        if ok != 0 {
            Some(s.ullTotalPhys as u64)
        } else {
            None
        }
    }
    // ⚠ **`android` 与 `linux` 同一腿** —— 见上面的说明。
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        let Ok(raw) = std::fs::read_to_string("/proc/meminfo") else {
            return None;
        };
        parse_meminfo_total_bytes(&raw)
    }
    #[cfg(not(any(windows, target_os = "linux", target_os = "android")))]
    None
}

/// **现在可用**的内存（字节）—— 与 [`ram_bytes_platform`]（物理内存总量）
/// **不是同一件事**。
///
/// ⚠⚠️ **为什么容量池该用这个，而不是总量** —— 真机撞出来的（刀 16）：
/// rk3566 手机上 `MemTotal` 1.91 GiB 而 `MemAvailable` **1.04 GiB**
/// ⇒ ⇒ 差 **54%** ⇒ ⇒ ⇒ 用总量 ⇒ ⇒ 策略说「1.40 GiB 的 bf16 装得下」
/// ⇒ ⇒ 真加载 ⇒ ⇒ **被 OOM killer 杀掉**（逐字：
/// `oom_reaper: reaped process 25746 (dtype_probe)`）⇒ ⇒ 「装得下」这格
/// **在真机上说假话** ⇒ ⇒ 「失败长得像成功」的又一族。
///
/// ⚠ **Windows 上同一口径** —— `GlobalMemoryStatusEx` 的 `ullAvailPhys`
/// 就是「现在可用」⇒ ⇒ 三个平台**同一语义**（不是只修手机）。
fn ram_available_bytes_platform() -> Option<u64> {
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        let mut s: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
        s.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        let ok = unsafe { GlobalMemoryStatusEx(&mut s) };
        if ok != 0 {
            Some(s.ullAvailPhys as u64)
        } else {
            None
        }
    }
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        let Ok(raw) = std::fs::read_to_string("/proc/meminfo") else {
            return None;
        };
        parse_meminfo_available_bytes(&raw)
    }
    #[cfg(not(any(windows, target_os = "linux", target_os = "android")))]
    None
}

/// 解析 `/proc/meminfo` 里的 `MemTotal:`（kB ⇒ 字节）。
///
/// ⚠ **纯函数** ⇒ ⇒ 判据能**直接钉它**（不许只靠 cfg 字符串那一格）。
/// ⚠ Android 的 `/proc/meminfo` 与 Linux 同格式（真机实测逐字相同）。
pub fn parse_meminfo_total_bytes(raw: &str) -> Option<u64> {
    parse_meminfo_kb(raw, "MemTotal:").map(|kb| kb.saturating_mul(1024))
}

/// 解析 `/proc/meminfo` 里的 `MemAvailable:`（kB ⇒ 字节）。
///
/// ⚠⚠️ **这是「现在能用多少」，不是「机器有多少」** —— 两个是**不同的数**
/// ⇒ ⇒ 真机实测（rk3566）：`MemTotal` 1.91 GiB 而 `MemAvailable` **1.04 GiB**
/// ⇒ ⇒ 差 **54%** ⇒ ⇒ ⇒ **用总量当容量池 ⇒ ⇒ 必然说假话**。
///
/// ⚠ **为什么容量池该用这个** —— 「装不装得下」问的是「**现在**能不能塞进去」
/// ⇒ ⇒ 而手机上系统本身 ＋ 别的 App 已经吃掉一大半 ⇒ ⇒ 用总量 ⇒ ⇒
/// 策略说「装得下」⇒ ⇒ 真加载 ⇒ ⇒ **OOM killer 来收割**（真机逐字：
/// `oom_reaper: reaped process 25746 (dtype_probe), now anon-rss:0kB`）。
///
/// ⚠ **为什么不是 `MemFree:`** —— `MemFree` **不含**可回收的 page cache 与
/// slab ⇒ ⇒ 它**低估**可用量 ⇒ ⇒ Linux 的 `MemAvailable` 才是内核自己算的
/// 「还能分配多少」（内核的 `available` 估算）⇒ ⇒ 口径一致。
pub fn parse_meminfo_available_bytes(raw: &str) -> Option<u64> {
    parse_meminfo_kb(raw, "MemAvailable:").map(|kb| kb.saturating_mul(1024))
}

/// 从 `/proc/meminfo` 里取某一行（kB）。
///
/// ⚠ **纯函数**（刀 16）—— 键名是**参数** ⇒ ⇒ 同一份解析逻辑供
/// `MemTotal:`（总量）与 `MemAvailable:`（可用）共用 ⇒ ⇒ 两处不许各写一遍。
fn parse_meminfo_kb(raw: &str, key: &str) -> Option<u64> {
    for line in raw.lines() {
        if let Some(rest) = line.strip_prefix(key) {
            let kb: u64 = rest.trim().trim_end_matches(" kB").trim().parse().ok()?;
            return Some(kb);
        }
    }
    None
}

/// 一次显存读数。
///
/// `bytes == None` ⇒ **没测到**（不是 0 字节）；`source` 说**从哪儿测到的**。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VramReading {
    /// 显存容量（字节）—— **设备 0 那一张**的容量，不是所有卡的合计。
    pub bytes: Option<u64>,
    /// 测法。
    pub source: VramSource,
    /// 测到几张卡。
    pub devices: usize,
}

/// 测显存 —— **能测就测，测不到就说测不到**。
///
/// 三步，顺序有讲究：
/// 1. **CUDA driver API**（编了 `cuda` 时）—— 进程内、**按 CUDA 序号**
///    ⇒ ⇒ 与运行时真正打开的那张卡**同一个口径**（尊重 `CUDA_VISIBLE_DEVICES`）；
/// 2. **`nvidia-smi`** —— 外部进程、**按物理序号** ⇒ ⇒ 多卡机器上与 CUDA
///    序号未必一致（CUDA 默认 `FASTEST_FIRST`）⇒ ⇒ 所以它是次选；
/// 3. 都不行 ⇒ **`None` ＋ `Unmeasured`** ⇒ ⇒ 账本那一半退回兜底常量，
///    而且**自报家门**（不许把兜底当真值）。
///
/// ⚠ **问的是「一张卡的容量」，不是所有卡的合计** —— 一个模型只能待在一张
/// 卡上 ⇒ ⇒ 把两张卡加成一个池，会让账本放行一个**根本装不下**的模型。
pub fn detect_vram() -> VramReading {
    #[cfg(feature = "cuda")]
    {
        if let Some((bytes, devices)) = vram_via_cuda_driver() {
            return VramReading {
                bytes: Some(bytes),
                source: VramSource::CudaDriver,
                devices,
            };
        }
    }
    if let Some((bytes, devices)) = vram_via_nvidia_smi() {
        return VramReading {
            bytes: Some(bytes),
            source: VramSource::NvidiaSmi,
            devices,
        };
    }
    tracing::warn!(
        "[neuron] 显存**测不到**（CUDA driver API 与 nvidia-smi 都没给出容量）\
         ⇒ ⇒ 账本的 GPU 那一半会退回兜底常量（那个数**不是**测到的）"
    );
    VramReading {
        bytes: None,
        source: VramSource::Unmeasured,
        devices: 0,
    }
}

/// 走 CUDA driver API 问显存（只有编了 `cuda` 的 build 有这条腿）。
///
/// **按 CUDA 序号**取设备 0 —— 与 `candle::Device::new_cuda(0)` 打开的是
/// **同一张卡** ⇒ ⇒ 这就是「用运行时的眼睛量显存」。
#[cfg(feature = "cuda")]
fn vram_via_cuda_driver() -> Option<(u64, usize)> {
    use candle::cuda_backend::cudarc::driver::CudaContext;
    let count = CudaContext::device_count().ok()?;
    if count <= 0 {
        return None;
    }
    let ctx = CudaContext::new(0).ok()?;
    let total = ctx.total_mem().ok()?;
    Some((total as u64, count as usize))
}

/// 走 `nvidia-smi` 问显存（外部进程 ⇒ 不需要编 `cuda` 也能问）。
fn vram_via_nvidia_smi() -> Option<(u64, usize)> {
    for exe in nvidia_smi_candidates() {
        let Ok(out) = Command::new(&exe)
            .args([
                "--query-gpu=index,memory.total",
                "--format=csv,noheader,nounits",
            ])
            .output()
        else {
            continue;
        };
        if !out.status.success() {
            continue;
        }
        if let Some(r) = parse_nvidia_smi_vram(&String::from_utf8_lossy(&out.stdout)) {
            return Some(r);
        }
    }
    None
}

/// `nvidia-smi` 的候选路径：PATH 上的 ＋ CUDA 安装目录（`CUDA_PATH`）。
///
/// Windows 上 `nvidia-smi` **常常不在 PATH 里**（它装在
/// `%CUDA_PATH%\bin`）⇒ ⇒ 只试 PATH 会白问一次。
fn nvidia_smi_candidates() -> Vec<std::path::PathBuf> {
    let mut v = vec![std::path::PathBuf::from("nvidia-smi")];
    if let Ok(cuda_path) = std::env::var("CUDA_PATH")
        && !cuda_path.trim().is_empty()
    {
        let exe = if cfg!(windows) {
            "nvidia-smi.exe"
        } else {
            "nvidia-smi"
        };
        v.push(std::path::Path::new(&cuda_path).join("bin").join(exe));
    }
    v
}

/// 解析 `nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits`。
///
/// 返回 **（设备 0 的容量（字节），卡数）**。
///
/// ⚠ **不是总和** —— 一个模型只能待在一张卡上 ⇒ ⇒ 取**运行时用的那一张**。
/// ⚠ `[N/A]`（Windows WDDM 下某些列就是它）**不是 0** ⇒ ⇒ 那一行直接跳过
///    （测不到的那张卡不许被当成 0 字节的卡）。
///
/// **纯函数** ⇒ ⇒ 可测（判据里喂假输出，不 spawn 进程）。
pub fn parse_nvidia_smi_vram(csv: &str) -> Option<(u64, usize)> {
    let mut rows: Vec<(u32, u64)> = Vec::new();
    for line in csv.lines() {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        let mut fields = t.split(',').map(|f| f.trim());
        let Some(idx) = fields.next().and_then(|f| f.parse::<u32>().ok()) else {
            continue;
        };
        let Some(mib) = fields.next().and_then(parse_mib) else {
            continue;
        };
        rows.push((idx, mib));
    }
    if rows.is_empty() {
        return None;
    }
    let devices = rows.len();
    let picked = rows
        .iter()
        .find(|(i, _)| *i == 0)
        .or_else(|| rows.iter().min_by_key(|(i, _)| *i))
        .map(|(_, mib)| *mib)?;
    Some((picked.saturating_mul(1024 * 1024), devices))
}

/// 解析一列 MiB（`24576`／`24576 MiB`）；`[N/A]` 之类**不是 0，是测不到**。
fn parse_mib(field: &str) -> Option<u64> {
    let t = field.trim();
    if t.is_empty() || t.starts_with('[') {
        return None;
    }
    let t = t.trim_end_matches("MiB").trim();
    if t.is_empty() {
        return None;
    }
    t.parse::<u64>().ok()
}

// ---- MoE 专家张量放在哪一档（2026-11-25 刀 14）----------------------------
//
// ⚠ **这一格是「第三档」** —— 降级阶梯此前只有两格：「全 GPU」→ 「全退 CPU」。
// 中间那格（**专家张量留在系统内存，其余留在显存**）此前**不存在** ⇒ ⇒ 于是
// 「装不下」只有一个出口：把**整个模型**挪到 CPU（代价：速度 —— 真机实测
// GPU 是 CPU 的 6.0×）。
//
// ⇒ ⇒ 「专家下沉」那一格在 llama.cpp 侧是**现成的**（等价于上游的 --cpu-moe：
// 把 blk.N.ffn_(up|down|gate|gate_up)_(ch|)exps 钉在 CPU buffer type 上，
// 其余张量照常按 n_gpu_layers 走设备）⇒ ⇒ **零 C API 改动**。
//
// ⚠ **本档只到「专家全在内存」** —— 「显存里再留一份热的」那一半（上游
// master 的 llama_model_params.moe_cache_size）**在 neuron's 钉版上不存在**
// （钉版整棵子树 moe_cache 命中 0）⇒ ⇒ 那是**版本债**，不是缺功能。

/// MoE 专家张量放在哪一档。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertPlacement {
    /// **判不了**（没读回／池子没测到）—— 不许当成「都在设备上」。
    Unknown,
    /// 全在加速设备上（装得下；非 MoE 权重也归这一档 —— 没有专家可下沉）。
    Device,
    /// **专家张量在系统内存**，其余在加速设备（**第三档**）。
    ExpertsInRam,
    /// **部分**张量在主机内存，但命中的不是专家（不是第三档，别冒充）。
    Mixed,
    /// 整体在主机内存（两池都不够 ⇒ 退 CPU）。
    Host,
}

impl ExpertPlacement {
    /// 这一档的名字（给日志与账本用）。
    pub fn name(self) -> &'static str {
        match self {
            Self::Unknown => "unknown（判不了）",
            Self::Device => "device（全在加速设备）",
            Self::ExpertsInRam => "experts-in-ram（专家在系统内存，其余在加速设备）",
            Self::Mixed => "mixed（部分张量在主机内存，但不是专家那一档）",
            Self::Host => "host（整体在主机内存）",
        }
    }
}

/// llama.cpp 的专家张量正则（上游 LLM_FFN_EXPS_REGEX 的逐字拷贝）。
///
/// ⚠ **必须逐字抄上游** —— 执行者读回来的 override 模式要拿它来判
/// 「这一格是不是真的把专家挪到了内存」⇒ ⇒ 两边**必须**是同一个模式。
pub const LLAMA_EXPERT_TENSOR_PATTERN: &str = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps";

/// 已知是 MoE 的架构（general.architecture 的取值）—— **兜底名单**。
///
/// ⚠ 这不是判据源 —— 真判据是 GGUF 元数据里的 <arch>.expert_count（> 1 就是
/// MoE）⇒ ⇒ 名单只用来回答「元数据读不到时，这一条**算不算** MoE」⇒ ⇒
/// 名单外的架构**一律按「不是 MoE」处理**（保守：不敢下沉就别下沉）。
pub fn arch_looks_moe(arch: &str) -> bool {
    matches!(
        arch,
        "qwen3moe" | "qwen3next" | "qwen2moe" | "mixtral" | "deepseek2" | "olmoe" | "grokmoe"
    )
}

/// 降级阶梯的第三档：**专家张量放哪**。
///
/// **纯函数** —— 不读环境、不读盘 ⇒ ⇒ 可测、可复现。
///
/// 判据（按顺序）：
/// 1. **执行者没有加速设备后端** ⇒ ⇒ Host（**见下面那道前提**）；
/// 2. **不是 MoE** ⇒ ⇒ 没有专家可下沉 ⇒ Device；
/// 3. **权重装得下显存** ⇒ ⇒ Device；
/// 4. **装不下显存、但装得下内存** ⇒ ⇒ ExpertsInRam（**第三档**）；
/// 5. **两池都不够** ⇒ ⇒ Host。
///
/// ⚠ **第 4 条为什么不需要「专家占多少字节」** —— 「整个模型装得下内存」
/// ⇒ ⇒ **专家当然也装得下内存** ⇒ ⇒ 上界就够用，不必去算专家那部分的
/// 字节数（那要读一整套张量表）。
///
/// ⚠⚠️ **刀 17（2026-11-25）—— 第 4 条前面还有一道「执行者前提」**：
/// 「其余留在显存」这句话，**只在执行者真的有加速设备后端时**才成立 ⇒ ⇒
/// ⇒ 而「有没有那个后端」是**执行者那个 build 的事实**，与「这台机器有多少
/// ⇒ ⇒ 显存」（`caps.vram_bytes`，candle 测的）**不是同一件事** ⇒ ⇒ 两个
/// ⇒ ⇒ 可以脱钩 ⇒ ⇒ 脱钩时策略层会算出 `ExpertsInRam`，而执行者那边
/// ⇒ ⇒ **一个 GPU 后端都没有** ⇒ ⇒ `with_n_gpu_layers(u32::MAX)` 是
/// ⇒ ⇒ **空指令** ⇒ ⇒ 「其余在显存」那一半**无处安放** ⇒ ⇒ ⇒ 真档是
/// ⇒ ⇒ **Host**（整盘在主机内存，因为压根没有「显存那一侧」）。
///
/// ⇒ ⇒ 所以第三个事实是**必需的**：调用方必须把**执行者自己报的**能力
/// ⇒ ⇒ （llama.cpp 的 `LlamaBackend::supports_gpu_offload()`）传进来 ⇒ ⇒
/// ⇒ ⇒ **不许拿机器的显存去替执行者说话**。
///
/// ⚠⚠️ **那个读回的语义要说准**（实测上游源码，`llama.cpp/src/llama.cpp`）：
///   它答的是「**此刻有没有一个 GPU/IGPU 设备可枚举**（或 RPC 后端可用）」
///   ⇒ ⇒ **不是**「CUDA 后端编进去了没有」⇒ ⇒⇒ 但这两个问题**一并被它答了**：
///   ① 后端**没编** ⇒ ⇒ 那个 build 里没有 CUDA 设备类型 ⇒ ⇒ 枚举不到 ⇒ ⇒ false；
///   ② 后端**编了但机器没卡** ⇒ ⇒ `ggml_cuda_init()` 的 `cudaGetDeviceCount` 返回 0
///      ⇒ ⇒ `device_count = 0` ⇒ ⇒ **不注册任何设备** ⇒ ⇒ 枚举不到 ⇒ ⇒ false。
///   ⇒ ⇒ 所以它问的正是「**此刻这台机器上，这个 build 能不能真的卸到设备上**」
///   ⇒ ⇒ ⇒ 而「其余在显存」这句话**恰恰需要这个前提** ⇒ ⇒ ⇒ **正合适**。
///
/// ⚠ **为什么不是「Unknown」而是「Host」** —— 「判不了」与「判定了没有」
/// ⇒ ⇒ 是两件事：这里**判定了**（执行者明确报「不支持卸载」）⇒ ⇒ 整盘在
/// ⇒ ⇒ 主机 ⇒ ⇒ **Host**（一个确定的档）⇒ ⇒ 而 `Unknown` 留给「没测到」。
pub fn resolve_expert_placement(
    caps: &HostCapabilities,
    is_moe: bool,
    weight_bytes: u64,
    executor_has_gpu: bool,
) -> ExpertPlacement {
    // ⚠⚠️ **刀 17：执行者前提** —— 没有加速设备后端 ⇒ ⇒ 「在设备上」这一档
    // ⇒ ⇒ 与「其余在显存」那一档**都不成立** ⇒ ⇒ 整盘在主机。
    //
    // ⚠ 这一条**排在最前**：它比「是不是 MoE」还要靠前 ⇒ ⇒ 因为
    // ⇒ ⇒ 「全在加速设备上」这句话本身也需要**有那个设备**才成立 ⇒ ⇒
    // ⇒ ⇒ 连非 MoE 权重也不能报 Device（那会是一句假话）。
    // ⇒ ⇒ （设备选择是 CPU 时同理 ⇒ ⇒ 已在调用方提前判掉。）
    if !executor_has_gpu {
        return ExpertPlacement::Host;
    }
    if !is_moe {
        return ExpertPlacement::Device;
    }
    // 池子没测到 ⇒ ⇒ **判不了**，不许拿兜底常量当真值。
    let (Some(vram), Some(ram)) = (caps.vram_bytes, caps.ram_bytes) else {
        return ExpertPlacement::Unknown;
    };
    if weight_bytes <= vram {
        return ExpertPlacement::Device;
    }
    if weight_bytes <= ram {
        ExpertPlacement::ExpertsInRam
    } else {
        ExpertPlacement::Host
    }
}

/// 把**装进模型参数里的** override 模式翻译成「这一格是哪一档」。
///
/// ⚠ None ＝ **没读回** ⇒ ⇒ Unknown（判不了）；
///   Some(&[]) ＝ 读回了、确实一个 override 都没有 ⇒ ⇒ Device。
///
/// ⚠ **权威等级**：它证明的是「**我们要求了什么**」（llama.cpp 的加载器按
/// 这份 override 分配张量），**不是**「专家真的落在 CPU 上」⇒ ⇒ 后者要靠
/// 加载日志／显存读数来证（别拿这一条冒充）。
pub fn classify_expert_placement(patterns: Option<&[String]>) -> ExpertPlacement {
    let Some(patterns) = patterns else {
        return ExpertPlacement::Unknown;
    };
    if patterns.is_empty() {
        return ExpertPlacement::Device;
    }
    let is_expert = |p: &String| p.contains("ffn_") && p.contains("exps");
    if patterns.iter().any(is_expert) {
        ExpertPlacement::ExpertsInRam
    } else {
        ExpertPlacement::Mixed
    }
}

#[cfg(test)]
#[path = "accel_tests.rs"]
mod tests;
