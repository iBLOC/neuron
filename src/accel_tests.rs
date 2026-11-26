//! `resolve_strategy` 的判据。
//!
//! 全部是**纯函数**判据：不读环境、不读全局状态 ⇒ ⇒ 不需要
//! `serial_test`，也不会在并行测试里抢环境变量。
//!
//! 负控是这个模块的重点：**「有 CUDA 设备」与「能用 CUDA」是两回事**，
//! 「编译进来」与「装上了」也是两回事 —— 策略层必须把后者说出来，
//! 而且**不许**在理由里假装前者成立。
//!
//! ⚠ 2026-11-25 补的那几条（策略接进运行时那一刀逼出来的）：
//! · **用户显式要的优先**（设备／dtype）—— 「自动」＝「用户没说话」，
//!   不是「可以不听用户的」；
//! · **dtype 跟运行时走** —— 策略不许把「加速卡应该用 bf16」这种
//!   **愿望**说成会发生的事（运行时的默认是 f32）；
//! · **预算不许再是写死的 8 GB** —— 那是编出来的数字。

use super::*;

/// 造一份「模型事实」：盘上 bf16、权重若干字节。
fn model(arch: &str, weight_bytes: u64) -> ModelFacts {
    ModelFacts {
        model_id: "test-model".to_string(),
        arch: arch.to_string(),
        weight_bytes,
        format: WeightFormat::Safetensors,
        on_disk_bytes: Some(2),
    }
}

/// 造一份「环境」。
fn caps(device: DeviceKind, features: CompiledFeatures) -> HostCapabilities {
    HostCapabilities {
        device,
        machine_device: device,
        ram_bytes: Some(16_000_000_000),
        // ⚠ 可用内存：默认与总量同值 ⇒ ⇒ 「总量口径」与「可用口径」在判据里
        //   **默认一致** ⇒ ⇒ 于是只有专门改这一栏的判据才看得出差别。
        ram_available_bytes: Some(16_000_000_000),
        vram_bytes: Some(24_000_000_000),
        vram_source: VramSource::CudaDriver,
        vram_devices: 1,
        cores: Some(12),
        features,
    }
}

/// 造一份「池子大小由参数说了算」的环境 —— **降级窗**就靠它造。
///
/// ⚠ 真实机器上这个窗**常常是空的**：内存比显存小的机器（GPU 机）⇒ ⇒
/// 「退 CPU」救不了任何模型 ⇒ ⇒ 那一格只能**判据级**验。
fn caps_with_pools(
    device: DeviceKind,
    features: CompiledFeatures,
    ram: Option<u64>,
    vram: Option<u64>,
) -> HostCapabilities {
    let mut c = caps(device, features);
    c.ram_bytes = ram;
    // ⚠⚠️ **「可用内存」必须与「总量内存」同步** —— 容量池现在**优先**用
    //   「可用」⇒ ⇒ 只改 `ram_bytes` 会让判据**测不到池**（判据与
    //   现实脱节）⇒ ⇒ 「把池设为没测到」这条路必须两栏一起置 `None`。
    c.ram_available_bytes = ram;
    c.vram_bytes = vram;
    c
}

/// 全 feature 都关的环境。
fn no_features() -> CompiledFeatures {
    CompiledFeatures {
        candle: false,
        cuda: false,
        metal: false,
        flash_attn: false,
        mistralrs: false,
        llamacpp: false,
    }
}

/// 默认请求：**用户没说话**。
fn req() -> StrategyRequest {
    StrategyRequest::default()
}

/// 造一份「机器**有**卡、但这个 build 编不进 GPU」的环境。
///
/// 这就是**出厂形态**：没编 `cuda` ⇒ 探测设备的那条腿被门控掉 ⇒
/// `device` 是 CPU，而 `machine_device` 才是机器的真话。
fn caps_machine_gpu_only(machine: DeviceKind) -> HostCapabilities {
    let mut c = caps(DeviceKind::Cpu, no_features());
    c.machine_device = machine;
    c
}

// ---------------------------------------------------------------------------
// 刀 16（2026-11-26）：**手机那条腿** —— 真机撞出来的洞。
//
// ⚠ 洞的形状：**「装不下 ⇒ 无损降级」这条路在手机上整个不存在** ⇒ ⇒
// 而且它**说得像存在**（策略层照旧说「装不下时会无损降」）⇒ ⇒
// ⇒ 「失败长得像成功」的那一族。
//
// ⚠ 根因（源码级）：Android 的 `target_os` 是 `"android"`，**不是**
//   `"linux"` ⇒ ⇒ `ram_bytes_platform()` 的 `/proc/meminfo` 那条腿
//   **在手机上恒为 `None`** ⇒ ⇒ 容量轴瞎 ⇒ ⇒ 降级窗空。
// ---------------------------------------------------------------------------

/// 判据①：**内存轴在 Android 上必须是活的** —— 平台分支不许漏掉 android。
///
/// ⚠ 这一格**钉的是 cfg 分支本身**（不是钉某台机器的读数）⇒ ⇒
/// 判据不能只跑在 Windows 上 ⇒ ⇒ 用**源码文本**做静态判据：
/// `accel.rs` 里 `/proc/meminfo` 那条腿的 cfg **必须**含 `android`。
#[test]
fn test_android_shares_the_meminfo_leg_with_linux() {
    // ⚠ 真机读数（rk3566 / Android 11）：`MemTotal: 2004240 kB` ⇒ ⇒
    // `/proc/meminfo` 存在且**全局可读**（`-r--r--r--`，App uid 也读得到）
    // ⇒ ⇒ 这一腿在手机上是**真的** ⇒ ⇒ cfg 漏掉 android 就是漏掉了真数据。
    let src = include_str!("accel.rs");
    // ⚠ 窗口要**含 cfg 属性本身** ⇒ ⇒ 从 `/proc/meminfo` 往**前**找最近
    //   那条 `#[cfg(...)]` ⇒ ⇒ 只看后面的函数体是不够的（cfg 在上面）。
    //
    // ⚠⚠️ **但不能停在文档注释里** —— 本文件的注释里**引用过**旧写法
    //   （`#[cfg(target_os = "linux")]`）⇒ ⇒ 最近的 `#[cfg(` 可能是**注释里
    //   那一句** ⇒ ⇒ 判据必须**跳过注释行**，只看**真正的属性**。
    //   ⇒ ⇒ 判据源不许有第二份真相：这里直接读**函数体里那行**。
    let meminfo_at = src
        .find("std::fs::read_to_string(\"/proc/meminfo\")")
        .expect("accel.rs must read /proc/meminfo");
    let head = &src[..meminfo_at];
    // 往上找最近的**非注释**的 `#[cfg(`（注释行以 `//` 开头 ⇒ 跳过）。
    let cfg_at = head
        .rfind("\n    #[cfg(")
        .expect("the /proc/meminfo leg must be behind a #[cfg(...)] attribute");
    let cfg_line = &head[cfg_at..];
    let cfg_line = &cfg_line[..cfg_line.find(']').expect("cfg attribute must close") + 1];
    assert!(
        cfg_line.contains("android"),
        "the /proc/meminfo leg's cfg must cover android too \
         (target_os=android is NOT linux) — found: {}",
        cfg_line
    );
}

/// 判据②：**纯解析函数** —— `/proc/meminfo` 的 `MemTotal:` 解析。
///
/// ⚠ 负控：`MemFree:` / `MemAvailable:` **不许**被当成 `MemTotal:`
///   （那会把「可用内存」当「物理内存」⇒ ⇒ 降级窗算错）。
#[test]
fn test_parse_meminfo_total_bytes_reads_memtotal_only() {
    let raw =
        "MemTotal:        2004240 kB\nMemFree:         38596 kB\nMemAvailable:    928392 kB\n";
    assert_eq!(
        parse_meminfo_total_bytes(raw),
        Some(2_004_240 * 1024),
        "MemTotal must win over MemFree/MemAvailable"
    );
    // 负控：没有 MemTotal ⇒ 测不到 ⇒ None（不许拿别的行凑）
    assert_eq!(parse_meminfo_total_bytes("MemFree: 100 kB\n"), None);
    // 负控：空输入
    assert_eq!(parse_meminfo_total_bytes(""), None);
}

/// 判据③：**池测不到 ⇒ 不许说「装不下时会无损降」**。
///
/// ⚠ 这是「说假话」的那一格 ⇒ ⇒ 池没测到 ⇒ ⇒ 承诺不许出现在理由里。
#[test]
fn test_unmeasured_pool_does_not_promise_a_lossless_downgrade() {
    let mut c = caps_with_pools(DeviceKind::Cpu, no_features(), None, None);
    c.features.candle = true;
    let s = resolve_strategy(&c, &model("qwen2", 8_000_000_000), &req());
    assert!(
        !s.reasons.iter().any(|r| r.contains("装不下时会**无损降**")),
        "an unmeasured pool has NO downgrade window — promising one is a lie: {:?}",
        s.reasons
    );
    assert!(
        s.unavailable.iter().any(|u| u.contains("没测到")),
        "must say the pool was not measured: {:?}",
        s.unavailable
    );
}

/// 判据④：**池测不到 ⇒ 降级窗真的不存在**（不是「存在但没选中」）。
///
/// ⚠ 这一格钉的是**候选搜索**的结果 ⇒ ⇒ 「装了 bf16 候选但没选中」与
/// 「**根本没有** bf16 候选」是两件事 ⇒ ⇒ 前者是选择问题，后者是洞。
#[test]
fn test_unmeasured_pool_produces_no_downgrade_candidate() {
    let mut c = caps_with_pools(DeviceKind::Cpu, no_features(), None, None);
    c.features.candle = true;
    // 盘上 bf16、8 GB 权重 ⇒ f32 物化 16 GB ⇒ ⇒ 池 12 GB ⇒ **装不下** ⇒ ⇒ 会落 bf16。
    // ⚠ 池必须**小于** 16 GB：等于时 `need > pool` 为假 ⇒ ⇒ 不降级（那是另一条路）。
    let measured = caps_with_pools(DeviceKind::Cpu, no_features(), Some(12_000_000_000), None);
    let measured_strategy = resolve_strategy(&measured, &model("qwen2", 8_000_000_000), &req());
    assert_eq!(
        measured_strategy.dtype,
        DTypeChoice::BF16,
        "with a measured pool the lossless downgrade must happen"
    );

    let unmeasured_strategy = resolve_strategy(&c, &model("qwen2", 8_000_000_000), &req());
    assert_eq!(
        unmeasured_strategy.dtype,
        DTypeChoice::F32,
        "with an unmeasured pool there is NO downgrade ⇒ stay on the start dtype"
    );
    assert!(
        !unmeasured_strategy
            .unavailable
            .iter()
            .any(|u| u.contains("装不下")),
        "must NOT claim 'does not fit' when the pool was never measured: {:?}",
        unmeasured_strategy.unavailable
    );
}

/// 判据⑤：**手机那台机器的形状** —— 1.91 GiB 内存 ＋ 2.40 GiB 的 f32 物化。
///
/// ⚠ 这一格是**真机形状**（rk3566：MemTotal 2 004 240 kB）⇒ ⇒ 桌面机
///   （15.91 GiB）**永远给不出**这一格 ⇒ ⇒ 它的降级窗是**空的**。
/// ⚠ 这也钉住了「手机 CPU 路**没有**那颗雷」这个说法的**前提**：
///   CPU 设备 ＋ 没编 flash-attn ⇒ ⇒ f32 的**前提**成立（跑得起来）
///   ⇒ ⇒ 所以在手机上，**唯一**能救它的是**容量**，不是 dtype。
#[test]
fn test_phone_sized_ram_downgrades_qwen3_f32_to_bf16() {
    let mut c = caps_with_pools(
        DeviceKind::Cpu,
        no_features(),
        Some(2_004_240 * 1024), // 真机：rk3566 的 MemTotal
        None,
    );
    c.features.candle = true;
    // 盘上 bf16、1.45 GB ⇒ f32 物化 2.90 GB > 1.91 GiB ⇒ ⇒ 降 bf16。
    let s = resolve_strategy(&c, &model("qwen3", 1_433_700_000), &req());
    assert_eq!(
        s.dtype,
        DTypeChoice::BF16,
        "a 1.91 GiB phone must not be handed a 2.9 GB f32 materialization"
    );
    assert_eq!(s.device, DeviceSelection::Cpu);
    assert!(
        s.reasons
            .iter()
            .any(|r| r.contains("自动降到 bf16") && r.contains("不丢精度")),
        "the downgrade must say it is lossless: {:?}",
        s.reasons
    );
    // ⚠ **CPU 路 ⇒ f32 的前提成立** ⇒ ⇒ 「跑不起来」那一条不许出现。
    assert!(
        !s.unavailable.iter().any(|u| u.contains("跑不起来")),
        "CPU + no flash-attn => the f32 prereq holds — the landmine is CUDA-only: {:?}",
        s.unavailable
    );
}

// ---------------------------------------------------------------------------
// 刀 16 第二批：容量池用「现在可用」，不是「总量」。
//
// ⚠ 真机逐字（rk3566 / Android 11）：`MemTotal: 2004240 kB` 而
//   `MemAvailable: 1089804 kB` ⇒ ⇒ **差 54%** ⇒ ⇒ 用总量 ⇒ ⇒ 策略说
//   「1.40 GiB 的 bf16 装得下」⇒ ⇒ 真加载 ⇒ ⇒ **被 OOM killer 杀掉**
//   （`oom_reaper: reaped process 25746 (dtype_probe)`）⇒ ⇒
//   ⇒ 「装得下」这格**在真机上说假话**。
// ---------------------------------------------------------------------------

/// 判据⑥：**容量池用「可用内存」** —— 不是总量。
///
/// ⚠ 这一格钉的是**池的取值** ⇒ ⇒ 桌面机（15.9 GiB 总量／4.8 GiB 可用）
///   **也**受影响 ⇒ ⇒ ⇒ 改口径不是「只修手机」。
#[test]
fn test_capacity_pool_uses_available_memory_not_total() {
    let mut c = caps_with_pools(
        DeviceKind::Cpu,
        no_features(),
        Some(16_000_000_000), // 总量
        None,
    );
    c.features.candle = true;
    // ⚠ **盘上 9 GB** ⇒ ⇒ f32 物化 **18 GB** > 总量 16 GB ⇒ ⇒ 装不下 ⇒ ⇒ 降级；
    //   ⇒ bf16 物化 **9 GB** ≤ 16 GB ⇒ ⇒ 装得下 ⇒ ⇒ 降级窗**非空**。
    //   （⚠ 「等于」不算「装不下」⇒ ⇒ 盘上 8 GB ⇒ f32 物化 16 GB ＝ 总量 ⇒ ⇒ **不**降级。）
    let m = model("qwen2", 9_000_000_000);
    let by_total = resolve_strategy(&c, &m, &req());
    assert_eq!(
        by_total.dtype,
        DTypeChoice::BF16,
        "total-memory pool downgrades"
    );

    // ⚠ **可用只有 4 GB** ⇒ ⇒ f32（18 GB）与 bf16（9 GB）**都装不下**
    //   ⇒ ⇒ 策略必须照实说「装不下」⇒ ⇒ 不许装得下。
    c.ram_available_bytes = Some(4_000_000_000);
    let by_available = resolve_strategy(&c, &m, &req());
    assert!(
        by_available
            .unavailable
            .iter()
            .any(|u| u.contains("装不下")),
        "with only 4 GiB available, an 8 GB bf16 model does NOT fit — must say so: {:?}",
        by_available.unavailable
    );
    assert_ne!(
        by_available.dtype,
        DTypeChoice::BF16,
        "must not hand out a dtype that does not fit in the available memory"
    );
}

/// 判据⑦：**`mem_probe` 的 `/proc` 腿也必须认 android**。
///
/// ⚠ 同一个洞的**第二处** —— 手机的调度器预算会落进 **12 GiB 兜底常量**
///   ⇒ ⇒ 在 1.91 GiB 的手机上超预算 6 倍 ⇒ ⇒ 调度器以为有地方 ⇒ ⇒
///   ⇒ 让装不下的模型留着 ⇒ ⇒ OOM killer 来收割。
#[test]
fn test_scheduler_mem_probe_covers_android() {
    let src = include_str!("scheduler/mem_probe.rs");
    let at = src
        .find("fn total() -> u64")
        .expect("mem_probe.rs must have a total()");
    let head = &src[..at];
    let cfg_at = head
        .rfind("\n#[cfg(")
        .expect("the /proc leg must be behind a #[cfg(...)] attribute");
    let line = &head[cfg_at..];
    let line = &line[..line.find(']').expect("cfg attribute must close") + 1];
    assert!(
        line.contains("android"),
        "the scheduler mem_probe /proc leg must cover android too — found: {}",
        line
    );
}

/// 判据⑧：**`MemAvailable` 的解析** —— 与 `MemTotal` 严格分开。
#[test]
fn test_parse_meminfo_available_bytes_reads_available_only() {
    let raw = [
        "MemTotal:        2004240 kB",
        "MemFree:          24480 kB",
        "MemAvailable:    1089804 kB",
    ]
    .join("\n");
    assert_eq!(parse_meminfo_available_bytes(&raw), Some(1_089_804 * 1024));
    // ⚠ 负控：没有 MemAvailable ⇒ 测不到 ⇒ None（不许拿 MemTotal 顶）
    assert_eq!(parse_meminfo_available_bytes("MemTotal: 2004240 kB"), None);
}

/// 判据⑨：**同一个池不许有两套值** —— 「装不下」的判断必须前后一致。
///
/// ⚠ 真机撞出来的（刀 16）：候选搜索那段走 `pool_of`（**现在可用**）
///   ⇒ ⇒ 而末尾「装不下」那段曾直接用 `caps.ram_bytes`（**总量**）
///   ⇒ ⇒ **同一档在同一个决策里被判出两个相反的结论**
///   ⇒ ⇒ 「失败长得像成功」＋「自相矛盾」的合一。
#[test]
fn test_the_fit_verdict_uses_the_same_pool_as_the_candidate_search() {
    // ⚠ 总量 16 GB／**可用 4 GB** ⇒ ⇒ 两个池差 4 倍 ⇒ ⇒ 差值能被抓到。
    let mut c = caps_with_pools(DeviceKind::Cpu, no_features(), Some(16_000_000_000), None);
    c.ram_available_bytes = Some(4_000_000_000);
    c.features.candle = true;
    // 盘上 9 GB ⇒ f32 物化 18 GB ⇒ 两者都装不下。
    let s = resolve_strategy(&c, &model("qwen2", 9_000_000_000), &req());
    // 末尾那段必须**也**按「可用 4 GB」判 ⇒ ⇒ 说装不下。
    assert!(
        s.unavailable.iter().any(|u| u.contains("装不下")),
        "the final fit verdict must use the same (available) pool: {:?}",
        s.unavailable
    );
    // ⚠ **不许说「要装下得设 bf16」** —— bf16 物化 9 GB > 可用 4 GB ⇒ ⇒
    //   ⇒ 那条出路**走不通** ⇒ ⇒ 不许给。
    assert!(
        !s.reasons.iter().any(|r| r.contains("要装下得显式设")),
        "must not offer an exit that does not fit: {:?}",
        s.reasons
    );
}

/// 判据⑩：**「会丢精度」的说法必须按盘上事实**。
///
/// ⚠ 盘上**就是** 2 字节/参数 ⇒ ⇒ bf16 是它本来的样子 ⇒ ⇒ **不丢精度**
///   ⇒ ⇒ 照实说「不丢精度」⇒ ⇒ 不许拿「会丢精度」当挡箭牌。
#[test]
fn test_the_precision_claim_follows_the_disk_facts_not_a_guess() {
    let mut c = caps_with_pools(DeviceKind::Cpu, no_features(), Some(16_000_000_000), None);
    c.features.candle = true;
    // ⚠⚠️ **那句话只在「自动降级没发生」时才出现** —— 而自动降级被禁的
    //   条件是「盘上**不是** 2 字节/参数」⇒ ⇒ 于是：
    //   · 盘上 2 字节 ⇒ ⇒ 自动降级**会**发生 ⇒ ⇒ 那句话**不该**出现
    //     （它出现就说明降级逻辑与建议逻辑**各说各话**）；
    //   · 盘上 4 字节 ⇒ ⇒ 自动降级被禁 ⇒ ⇒ 选了 f32（装不下）⇒ ⇒
    //     ⇒ 那句话出现 ⇒ ⇒ 此时「**会丢精度**」是**对的**（bf16 ≠ 盘上的 f32）。
    //
    // ⇒ ⇒ 判据⑩分两半：① 盘上 2 字节 ⇒ ⇒ **不许**出现那句话；
    //   ② 盘上 4 字节 ⇒ ⇒ 那句话出现且**必须**说「会丢精度」。
    //
    // ⚠ 这一格钉的是「**建议与降级不许各说各话**」—— 两个分支由**同一份盘上
    //   事实**驱动 ⇒ ⇒ 不许一个说「无损」、另一个说「会丢」。

    // ① 盘上 2 字节 ⇒ ⇒ 自动降级会发生 ⇒ ⇒ 那句话不该出现。
    let mut c2 = caps_with_pools(DeviceKind::Cpu, no_features(), Some(16_000_000_000), None);
    c2.features.candle = true;
    c2.ram_available_bytes = Some(12_000_000_000);
    let s2 = resolve_strategy(&c2, &model("qwen2", 7_000_000_000), &req());
    assert_eq!(
        s2.dtype,
        DTypeChoice::BF16,
        "on-disk 2 bytes/param => the lossless downgrade must happen"
    );
    assert!(
        !s2.reasons.iter().any(|r| r.contains("要装下得显式设")),
        "the downgrade already happened — no 'set the knob' advice needed: {:?}",
        s2.reasons
    );

    // ② 盘上 4 字节 ⇒ ⇒ 自动降级被禁 ⇒ ⇒ 那句话出现，且**必须**说「会丢精度」。
    let mut c4 = caps_with_pools(DeviceKind::Cpu, no_features(), Some(16_000_000_000), None);
    c4.features.candle = true;
    c4.ram_available_bytes = Some(12_000_000_000);
    // ⚠ **盘上 4 字节 ⇒ 物化字节要重算**：`materialized_bytes` 是
    //   `weight_bytes × target / on_disk` ⇒ ⇒ 盘上 4 字节时「盘上 7 GB」
    //   ⇒ ⇒ f32 物化 **7 GB**（不是 14 GB！）⇒ ⇒ bf16 物化 **3.5 GB**
    //   ⇒ ⇒ 都装得下（可用 12 GB）⇒ ⇒ 那句话**不该**出现 ⇒ ⇒
    //   ⇒ 要让它出现，得让 **f32 物化 > 可用** ⇒ ⇒ 盘上 20 GB ⇒ ⇒
    //   ⇒ f32 物化 20 GB > 12 GB ⇒ ⇒ 装不下；bf16 物化 10 GB ≤ 12 GB ⇒ ⇒
    //   ⇒ 装得下 ⇒ ⇒ 出路出现。
    let mut m4 = model("qwen2", 20_000_000_000);
    m4.on_disk_bytes = Some(4); // 盘上是 f32 ⇒ ⇒ bf16 确实丢精度
    let s4 = resolve_strategy(&c4, &m4, &req());
    assert_eq!(
        s4.dtype,
        DTypeChoice::F32,
        "a lossy downgrade must not happen automatically"
    );
    let advice = s4
        .reasons
        .iter()
        .find(|r| r.contains("要装下得显式设"))
        .expect("a bf16 exit should be offered when bf16 does fit");
    assert!(
        advice.contains("会丢精度"),
        "on-disk 4 bytes/param => bf16 IS lossy — say so: {}",
        advice
    );
    assert!(
        !advice.contains("不丢精度"),
        "must not claim losslessness where there is loss: {}",
        advice
    );
}

#[test]
fn test_machine_has_gpu_but_build_lacks_cuda_feature_says_it_out() {
    let s = resolve_strategy(
        &caps_machine_gpu_only(DeviceKind::Cuda),
        &model("qwen2", 1_000_000_000),
        &req(),
    );
    assert_eq!(s.device, DeviceSelection::Cpu);
    assert!(
        s.unavailable
            .iter()
            .any(|u| u.contains("NVIDIA") && u.contains("cuda")),
        "必须点名：机器有卡但 build 没编 cuda；got: {:?}",
        s.unavailable
    );
    // ⚠ 负控：**不许**说成「没有可用的加速设备」⇒ 那是在说机器没卡（假话）。
    assert!(
        !s.reasons.iter().any(|r| r.contains("没有可用的加速设备")),
        "机器有卡 ⇒ 不许说成没有加速设备；got: {:?}",
        s.reasons
    );
}

#[test]
fn test_machine_really_has_no_gpu_keeps_the_plain_cpu_reason() {
    let s = resolve_strategy(
        &caps_machine_gpu_only(DeviceKind::Cpu),
        &model("qwen2", 1_000_000_000),
        &req(),
    );
    assert_eq!(s.device, DeviceSelection::Cpu);
    assert!(
        s.reasons.iter().any(|r| r.contains("没有可用的加速设备")),
        "got: {:?}",
        s.reasons
    );
    // ⚠ 负控：机器**真的**没卡 ⇒ 不许喊 NVIDIA（不许乱喊＝另一种假话）。
    assert!(
        s.unavailable.iter().all(|u| !u.contains("NVIDIA")),
        "机器没卡 ⇒ 不许喊 NVIDIA；got: {:?}",
        s.unavailable
    );
}

#[test]
fn test_machine_device_is_a_field_of_its_own() {
    // 「机器有卡」与「build 能不能用」必须能**同时**成立且互不覆盖。
    let mut c = caps(DeviceKind::Cpu, no_features());
    c.machine_device = DeviceKind::Cuda;
    assert_eq!(c.device, DeviceKind::Cpu);
    assert_eq!(c.machine_device, DeviceKind::Cuda);

    let mut f = no_features();
    f.cuda = true;
    let mut d = caps(DeviceKind::Cuda, f);
    d.machine_device = DeviceKind::Cpu;
    assert_eq!(d.device, DeviceKind::Cuda);
    assert_eq!(d.machine_device, DeviceKind::Cpu);
}

#[test]
fn test_parse_nvidia_smi_devices_counts_cards_even_when_memory_unreadable() {
    assert_eq!(parse_nvidia_smi_devices("0\n1\n"), 2);
    assert_eq!(parse_nvidia_smi_devices(""), 0);
    assert_eq!(parse_nvidia_smi_devices("   \n\n"), 0);
    // ⚠ 负控：`nvidia-smi` 的报错行**不是**一张卡。
    assert_eq!(
        parse_nvidia_smi_devices("Unable to determine the device handle\n"),
        0
    );
}

#[test]
fn test_parse_macos_metal_support_only_accepts_yes() {
    let yes = "      Chipset Model: Apple M3 Pro\n      Metal Support: Yes\n";
    assert!(parse_macos_metal_support(yes));
    let no = "      Chipset Model: Apple M3 Pro\n      Metal Support: No\n";
    assert!(!parse_macos_metal_support(no));
    // ⚠ 负控：报告里列着别的 GPU ≠ 有 Metal。
    assert!(!parse_macos_metal_support(
        "      Chipset Model: NVIDIA GeForce RTX 4090\n"
    ));
}

/// 盘上 4e9 字节（bf16 ⇒ 2e9 参数）⇒ f32 物化 **8e9 字节**（≈7.45 GiB）。
///
/// ⚠ 口径：materialized_bytes 是「weight_bytes × 4 ÷ on_disk_bytes」，
/// 而本模块的 model() 助手把 on_disk_bytes 记成 **2** ⇒ ⇒ 要 8e9 字节
/// 就得给 **4e9** 的盘上权重。
fn heavy_model() -> ModelFacts {
    model("qwen2", 4_000_000_000)
}

/// 盘上就是 **f32**（每参数 4 字节）⇒ ⇒ 落到 bf16 **会丢精度** ⇒ ⇒
/// 只许**建议**，不许自动改。
fn f32_on_disk_model(weight_bytes: u64) -> ModelFacts {
    ModelFacts {
        model_id: "test-model".to_string(),
        arch: "qwen2".to_string(),
        weight_bytes,
        format: WeightFormat::Safetensors,
        on_disk_bytes: Some(4),
    }
}

#[test]
fn test_vram_too_small_downgrades_precision_first_and_keeps_the_gpu() {
    let mut f = no_features();
    f.cuda = true;
    // 显存 4 GB，内存 64 GB ⇒ ⇒ 8 GB 的权重（f32 物化）装不进显存。
    let caps = caps_with_pools(
        DeviceKind::Cuda,
        f,
        Some(64_000_000_000),
        Some(4_000_000_000),
    );
    let s = resolve_strategy(&caps, &heavy_model(), &req());
    // ⚠ 刀 12 **订正刀 10 的口径**：**先降精度、保留 GPU** ——
    // 盘上就是每参数 2 字节 ⇒ ⇒ 落到 bf16 **不丢精度**（那份数据本来就是
    // 2 字节，f32 只是**上采样**出来的）⇒ ⇒ 而退 CPU 要掉一个数量级的速度。
    assert_eq!(
        s.device,
        DeviceSelection::Nvidia,
        "能靠精度解决 ⇒ ⇒ 不该动设备"
    );
    assert_eq!(s.dtype, DTypeChoice::BF16, "应当无损降到 bf16");
    assert!(
        s.reasons.iter().any(|r| r.contains("自动降到 bf16")),
        "降级必须说清楚；got: {:?}",
        s.reasons
    );
    // 记账也跟着降（required_bytes ＝ 按 bf16 物化的字节）。
    assert_eq!(
        s.required_bytes,
        materialized_bytes(&heavy_model(), DTypeChoice::BF16)
    );
    // ⚠ 降级之后**不许**再留一条「装不下」的结论（降完装得下）。
    assert!(
        !s.unavailable.iter().any(|u| u.contains("装不下")),
        "降级之后不该还说装不下；got: {:?}",
        s.unavailable
    );
}

#[test]
fn test_falls_back_to_cpu_when_precision_cannot_save_it() {
    let mut f = no_features();
    f.cuda = true;
    // 显存只有 3 GB ⇒ ⇒ 连 bf16（8 GB）都装不下 ⇒ ⇒ 省精度救不了 ⇒ ⇒
    // 这时才退 CPU（f32 的 16 GB 装得进 64 GB 内存）。
    let caps = caps_with_pools(
        DeviceKind::Cuda,
        f,
        Some(64_000_000_000),
        Some(3_000_000_000),
    );
    let s = resolve_strategy(&caps, &model("qwen2", 8_000_000_000), &req());
    assert_eq!(
        s.device,
        DeviceSelection::Cpu,
        "省精度救不了 ⇒ ⇒ 必须退 CPU"
    );
    assert_eq!(
        s.dtype,
        DTypeChoice::F32,
        "没有降精度（用户没要求 ⇒ ⇒ 保持 f32）"
    );
    assert!(
        s.reasons.iter().any(|r| r.contains("自动退 CPU")),
        "退设备必须说清楚；got: {:?}",
        s.reasons
    );
    assert!(
        !s.unavailable.iter().any(|u| u.contains("装不下")),
        "内存装得下 ⇒ ⇒ 不许还说装不下；got: {:?}",
        s.unavailable
    );
}

#[test]
fn test_lossy_precision_downgrade_is_only_ever_suggested() {
    let mut f = no_features();
    f.cuda = true;
    // 盘上是 f32（每参数 4 字节）⇒ ⇒ 落到 bf16 **会丢精度** ⇒ ⇒ 不许自动改。
    let caps = caps_with_pools(
        DeviceKind::Cuda,
        f,
        Some(64_000_000_000),
        Some(3_000_000_000),
    );
    let m = f32_on_disk_model(4_000_000_000); // f32 物化 ＝ 4e9
    let s = resolve_strategy(&caps, &m, &req());
    assert_eq!(s.dtype, DTypeChoice::F32, "会丢精度的降级 ⇒ ⇒ 不许自动做");
    assert!(
        !s.reasons.iter().any(|r| r.contains("自动降到 bf16")),
        "悄悄换精度是另一种形状的「说假话」；got: {:?}",
        s.reasons
    );
    // ⇒ ⇒ 它只能退到 CPU（f32 的 4e9 装得进 64 GB 内存）。
    assert_eq!(s.device, DeviceSelection::Cpu);
}

#[test]
fn test_explicit_dtype_is_not_silently_swapped() {
    let mut f = no_features();
    f.cuda = true;
    let caps = caps_with_pools(
        DeviceKind::Cuda,
        f,
        Some(64_000_000_000),
        Some(4_000_000_000),
    );
    let s = resolve_strategy(
        &caps,
        &heavy_model(),
        &StrategyRequest {
            device: None,
            dtype: Some(DTypeChoice::F32),
            flash_attn: None,
        },
    );
    assert_eq!(
        s.dtype,
        DTypeChoice::F32,
        "用户显式要 f32 ⇒ ⇒ 不许偷偷换精度"
    );
    assert!(
        !s.reasons.iter().any(|r| r.contains("自动降到 bf16")),
        "got: {:?}",
        s.reasons
    );
}

#[test]
fn test_both_pools_too_small_does_not_downgrade() {
    let mut f = no_features();
    f.cuda = true;
    // 两个池都只有 4 GB ⇒ ⇒ **连 bf16（8 GB）都装不下** ⇒ ⇒ 哪儿都装不下
    // ⇒ ⇒ **不降级**（降级救不回来）。
    //
    // ⚠ 刀 12 订正：这条原本用 4e9 盘上权重（f32 8 GB／bf16 4 GB）⇒ ⇒
    // 那个 bf16 **装得进** 4 GB 的显存 ⇒ ⇒ 「不降级」的前提已经不成立
    // ⇒ ⇒ 现在换成更重的权重，让「连省精度也救不了」这件事**真的**成立。
    let caps = caps_with_pools(
        DeviceKind::Cuda,
        f,
        Some(4_000_000_000),
        Some(4_000_000_000),
    );
    let s = resolve_strategy(&caps, &model("qwen2", 8_000_000_000), &req());
    assert_eq!(
        s.device,
        DeviceSelection::Nvidia,
        "两个池都装不下 ⇒ ⇒ 不许假装降级成功"
    );
    assert!(
        s.unavailable.iter().any(|u| u.contains("装不下")),
        "必须照实报装不下；got: {:?}",
        s.unavailable
    );
}

#[test]
fn test_explicit_nvidia_is_not_silently_swapped_to_cpu() {
    let mut f = no_features();
    f.cuda = true;
    let caps = caps_with_pools(
        DeviceKind::Cuda,
        f,
        Some(64_000_000_000),
        Some(4_000_000_000),
    );
    let req = StrategyRequest {
        device: Some(DeviceSelection::Nvidia),
        ..Default::default()
    };
    let s = resolve_strategy(&caps, &heavy_model(), &req);
    // ⚠ 用户**显式**说了 ⇒ ⇒ 不许偷偷换设备（那是「说假话」的另一种形状）。
    assert_eq!(
        s.device,
        DeviceSelection::Nvidia,
        "用户显式指定 ⇒ ⇒ 不许偷偷换"
    );
    // ⚠ 刀 12 订正：精度**可以**自动降（用户没要求 dtype ⇒ ⇒ 且无损）⇒ ⇒
    // 所以这里**不再**是「照实报装不下」，而是「不许退 CPU」。
    assert!(
        !s.reasons.iter().any(|r| r.contains("自动退 CPU")),
        "显式指定了设备 ⇒ ⇒ 不许退 CPU；got: {:?}",
        s.reasons
    );
    assert_eq!(
        s.dtype,
        DTypeChoice::BF16,
        "盘上是 2 字节/参数 ⇒ ⇒ 无损降精度是允许的"
    );
}

#[test]
fn test_unmeasured_vram_does_not_downgrade() {
    let mut f = no_features();
    f.cuda = true;
    // 显存**没测到** ⇒ ⇒ 不许猜（既不许当它装得下，也不许当它装不下）。
    let caps = caps_with_pools(DeviceKind::Cuda, f, Some(64_000_000_000), None);
    let s = resolve_strategy(&caps, &heavy_model(), &req());
    assert_eq!(
        s.device,
        DeviceSelection::Nvidia,
        "池没测到 ⇒ ⇒ 不降级（不许猜）"
    );
    assert!(
        !s.reasons.iter().any(|r| r.contains("自动退 CPU")),
        "没测到就不许降级；got: {:?}",
        s.reasons
    );
    // ⚠ 刀 12：同一个护栏也管**精度** —— 池没测到 ⇒ ⇒ 没有「降」的依据。
    assert!(
        !s.reasons.iter().any(|r| r.contains("自动降到 bf16")),
        "没测到就不许降精度；got: {:?}",
        s.reasons
    );
}

#[test]
fn test_cuda_device_with_cuda_feature_goes_to_gpu() {
    let mut f = no_features();
    f.cuda = true;
    let s = resolve_strategy(
        &caps(DeviceKind::Cuda, f),
        &model("qwen2", 1_000_000_000),
        &req(),
    );
    assert_eq!(s.device, DeviceSelection::Nvidia);
    assert!(
        s.reasons.iter().any(|r| r.contains("CUDA")),
        "got: {:?}",
        s.reasons
    );
}

#[test]
fn test_cuda_device_without_cuda_feature_is_reported_not_assumed() {
    // **负控**：这台机器有 CUDA 设备，但这个 build 没编 cuda
    // ⇒ ⇒ 策略必须说「GPU 那条路开不了」，而且**不许**把设备选成 GPU。
    let s = resolve_strategy(
        &caps(DeviceKind::Cuda, no_features()),
        &model("qwen2", 1_000_000_000),
        &req(),
    );
    assert_eq!(s.device, DeviceSelection::Cpu);
    assert!(
        s.unavailable.iter().any(|u| u.contains("cuda")),
        "must name the missing feature: {:?}",
        s.unavailable
    );
    assert!(
        !s.reasons.iter().any(|r| r.contains("走 GPU")),
        "must not claim the GPU is used: {:?}",
        s.reasons
    );
}

#[test]
fn test_unknown_device_keeps_auto() {
    let s = resolve_strategy(
        &caps(DeviceKind::Unknown, no_features()),
        &model("qwen2", 1),
        &req(),
    );
    assert_eq!(s.device, DeviceSelection::Auto);
    assert!(
        s.reasons.iter().any(|r| r.contains("没测到")),
        "got: {:?}",
        s.reasons
    );
}

#[test]
fn test_dtype_follows_the_runtime_default_not_a_wish() {
    // ⚠ 口径改过一次：旧版写「加速卡 ⇒ bf16」⇒ ⇒ 那是**策略的愿望**，
    // 不是**运行时会做的事** —— candle 腿的 `resolve_weight_dtype()` 在
    // `NEURON_LLM_DTYPE` 没设时是 **f32** ⇒ ⇒ 策略若按 bf16 记账，
    // 就会比真实占用**少算一半** ⇒ ⇒ 所以策略跟着运行时走。
    let cpu = resolve_strategy(
        &caps(DeviceKind::Cpu, no_features()),
        &model("qwen2", 1_000_000_000),
        &req(),
    );
    assert_eq!(cpu.dtype, DTypeChoice::F32);

    let mut f = no_features();
    f.cuda = true;
    let gpu = resolve_strategy(
        &caps(DeviceKind::Cuda, f),
        &model("qwen2", 1_000_000_000),
        &req(),
    );
    assert_eq!(
        gpu.dtype,
        DTypeChoice::F32,
        "the runtime's default is f32 — the strategy must not silently switch to bf16"
    );
    assert!(
        gpu.reasons.iter().any(|r| r.contains("运行时的默认")),
        "must say where the dtype comes from: {:?}",
        gpu.reasons
    );
}

#[test]
fn test_gpu_recommends_bf16_but_only_as_a_advice() {
    // 加速卡上 bf16 更省 ⇒ ⇒ 那是**建议**（带旋钮名），不是替用户改主意。
    let mut f = no_features();
    f.cuda = true;
    let s = resolve_strategy(
        &caps(DeviceKind::Cuda, f),
        &model("qwen2", 1_000_000_000),
        &req(),
    );
    assert!(
        s.reasons.iter().any(|r| r.contains("NEURON_LLM_DTYPE")),
        "the advice must name the knob: {:?}",
        s.reasons
    );
}

#[test]
fn test_dtype_is_auto_downgraded_when_that_downgrade_loses_nothing() {
    // ⚠ **口径更新（刀 12）**：旧版这里断言「f32 装不下 ⇒ dtype 不许变」
    // ⇒ ⇒ 那时的降级**只有「换设备」一种**，代价是**速度**；现在多了
    // 「降精度」这一档 ⇒ ⇒ 盘上就是每参数 2 字节（bf16）时，f32 只是
    // **上采样**出来的 ⇒ ⇒ 落回 bf16 **一个字节都不丢** ⇒ ⇒ 自动降是
    // **对的**，而且**必须说清为什么**（不丢精度）。
    //
    // ⇒ ⇒ 「不许偷偷换精度」那条纪律**没有被取消**，它被**收窄**成
    //     「**会丢精度**时不许偷偷换」—— 那一条由
    //     `test_lossy_precision_downgrade_is_only_ever_suggested` 钉着；
    //     「用户显式要了 dtype 时不许换」由
    //     `test_explicit_dtype_is_not_silently_swapped` 钉着。
    let mut c = caps(DeviceKind::Cpu, no_features());
    c.ram_bytes = Some(16_000_000_000);
    // 盘上 16e9 字节（每参数 2 字节）⇒ ⇒ f32 物化 32e9（装不进 16e9 的内存）
    // ⇒ ⇒ bf16 物化 16e9（装得进）⇒ ⇒ 这就是降级窗。
    let s = resolve_strategy(&c, &model("qwen2", 16_000_000_000), &req());
    assert_eq!(s.dtype, DTypeChoice::BF16, "无损的降精度 ⇒ ⇒ 该降就得降");
    assert!(
        s.reasons
            .iter()
            .any(|r| r.contains("自动降到 bf16") && r.contains("不丢精度")),
        "自动降必须说清为什么（无损）；got: {:?}",
        s.reasons
    );
    assert!(
        !s.unavailable.iter().any(|u| u.contains("装不下")),
        "bf16 装得下 ⇒ ⇒ 不许还说装不下；got: {:?}",
        s.unavailable
    );
}

#[test]
fn test_model_that_cannot_fit_at_all_is_reported() {
    // 连「换 dtype 救回来」都不行 ⇒ ⇒ 只能报不可用。
    let mut c = caps(DeviceKind::Cpu, no_features());
    c.ram_bytes = Some(1_000_000_000);
    // ⚠⚠️ **两栏都要设**（刀 16 口径变更）—— 容量池现在**优先**用「现在可用」
    //   ⇒ ⇒ 只设 `ram_bytes` ⇒ ⇒ 这一栏还是 helper 的默认值（16 GiB）⇒ ⇒
    //   ⇒ 这一格**测不到它想测的池** ⇒ ⇒ 判据与现实脱节。
    c.ram_available_bytes = Some(1_000_000_000);
    let s = resolve_strategy(&c, &model("qwen2", 16_000_000_000), &req());
    assert!(
        s.unavailable.iter().any(|u| u.contains("装不下")),
        "must say it does not fit: {:?}",
        s.unavailable
    );
}

#[test]
fn test_flash_attn_availability_needs_feature_device_arch_and_dtype_together() {
    // 量的是「**能不能**开」（前提齐不齐），**不是**「会不会开」——
    // 开不开由运行时的旋钮决定（用户没要就不开）⇒ ⇒ 字段叫
    // `flash_attn_available` 就是为了不许人把它读成「开了」。
    //
    // ⚠ 2026-11-25：**前提从三个变成四个** —— dtype 也是前提（FA 内核
    // 只认 f16/bf16，而运行时的默认是 f32）⇒ ⇒ 默认 dtype 下「FA 可用」
    // 是**假话** ⇒ ⇒ 真机撞出来的。
    let mut f = no_features();
    f.candle = true;
    // 三缺一 ⇒ 不可用。
    let no_feature = resolve_strategy(&caps(DeviceKind::Cuda, f), &model("qwen2", 1), &req());
    assert!(!no_feature.flash_attn_available);

    let mut f2 = no_features();
    f2.flash_attn = true;
    let no_gpu = resolve_strategy(&caps(DeviceKind::Cpu, f2), &model("qwen2", 1), &req());
    assert!(!no_gpu.flash_attn_available);

    let mut f3 = no_features();
    f3.flash_attn = true;
    f3.cuda = true;
    let bad_arch = resolve_strategy(&caps(DeviceKind::Cuda, f3), &model("phi3", 1), &req());
    assert!(!bad_arch.flash_attn_available);

    let mut f4 = no_features();
    f4.flash_attn = true;
    f4.cuda = true;
    // 三个前提齐 —— 但 dtype 还是默认的 f32 ⇒ ⇒ 仍然**不可用**。
    let f32_dtype = resolve_strategy(&caps(DeviceKind::Cuda, f4), &model("qwen2", 1), &req());
    assert!(
        !f32_dtype.flash_attn_available,
        "f32 权重进不了 FA 内核 —— 默认 dtype 下说「可用」是假话"
    );

    // 显式 bf16 ⇒ 四个前提齐 ⇒ 真的能开。
    let all_good = resolve_strategy(
        &caps(DeviceKind::Cuda, f4),
        &model("qwen2", 1),
        &StrategyRequest {
            device: None,
            dtype: Some(DTypeChoice::BF16),
            flash_attn: None,
        },
    );
    assert!(all_good.flash_attn_available);
}

#[test]
fn test_explicit_flash_attn_request_names_the_missing_dtype_knob() {
    // 用户**明确要了** FA ⇒ ⇒ 策略必须回答「开不开得了，缺的是哪一个
    // 前提」⇒ ⇒ 缺 dtype 时必须点名 `NEURON_LLM_DTYPE=bf16`。
    let mut f = no_features();
    f.candle = true;
    f.flash_attn = true;
    f.cuda = true;
    let s = resolve_strategy(
        &caps(DeviceKind::Cuda, f),
        &model("qwen2", 1),
        &StrategyRequest {
            device: None,
            dtype: None,
            flash_attn: Some(true),
        },
    );
    // 「要了但拿不到」⇒ ⇒ 答案在 `unavailable` 里（不是 `reasons`）——
    // `reasons` 是「为什么这么选」，`unavailable` 是「你要的没拿到」。
    assert!(
        s.unavailable
            .iter()
            .any(|u| u.contains("FA") && u.contains("NEURON_LLM_DTYPE=bf16")),
        "must answer the question that was asked: {:?}",
        s.unavailable
    );
}

#[test]
fn test_flash_attn_stays_silent_when_nobody_asked() {
    // 用户没提 FA ⇒ ⇒ 策略**完全沉默**（注册日志不许被一条没人提的
    // 要求刷屏）⇒ ⇒ 连「dtype 缺不缺」都不提。
    let mut f = no_features();
    f.candle = true;
    f.flash_attn = true;
    f.cuda = true;
    let s = resolve_strategy(&caps(DeviceKind::Cuda, f), &model("qwen2", 1), &req());
    assert!(
        !s.reasons.iter().any(|r| r.contains("FA")),
        "got: {:?}",
        s.reasons
    );
    assert!(
        !s.unavailable.iter().any(|u| u.contains("FA")),
        "got: {:?}",
        s.unavailable
    );
}

#[test]
fn test_strategy_says_nothing_about_flash_attn_when_nobody_asked() {
    // 用户没提 FA ⇒ ⇒ 策略既不报「开」也不报「拿不到」⇒ ⇒ 注册日志不会
    // 被一条没人提的要求刷屏（与 candle 腿的降级纪律同一条：
    // 没要／明确关／真在用 ⇒ 三种态都不出声）。
    let mut f = no_features();
    f.flash_attn = true;
    f.cuda = true;
    let s = resolve_strategy(&caps(DeviceKind::Cuda, f), &model("qwen2", 1), &req());
    assert!(!s.reasons.iter().any(|r| r.contains("FlashAttention")));
    assert!(!s.unavailable.iter().any(|u| u.contains("FlashAttention")));
}

#[test]
fn test_every_strategy_carries_a_reason() {
    // 「只有结论没有理由的策略，出了事没法追」⇒ ⇒ 每个状态都至少一条理由。
    for device in [
        DeviceKind::Unknown,
        DeviceKind::Cpu,
        DeviceKind::Cuda,
        DeviceKind::Metal,
    ] {
        for arch in ["qwen2", "qwen3", "llama", "phi3", "chatglm"] {
            let s = resolve_strategy(&caps(device, no_features()), &model(arch, 1), &req());
            assert!(
                !s.reasons.is_empty(),
                "device={device:?} arch={arch} produced no reason"
            );
        }
    }
}

#[test]
fn test_arch_coverage_lists_match_the_implementations() {
    // candle 腿：qwen2／qwen3／llama／phi3／mistral／chatglm
    assert!(candle_supported_arch("qwen2"));
    assert!(candle_supported_arch("chatglm"));
    assert!(!candle_supported_arch("qwen3_5"));
    // FA：qwen2／qwen3／llama／mistral（phi3／chatglm 根本没这个字段）
    assert!(fa_supported_arch("qwen2"));
    assert!(fa_supported_arch("mistral"));
    assert!(!fa_supported_arch("phi3"));
    assert!(!fa_supported_arch("chatglm"));
}

#[test]
fn test_materialized_bytes_scales_with_dtype() {
    let m = model("qwen2", 2_000_000_000);
    assert_eq!(materialized_bytes(&m, DTypeChoice::BF16), 2_000_000_000);
    assert_eq!(materialized_bytes(&m, DTypeChoice::F32), 4_000_000_000);
}

#[test]
fn test_resolved_device_never_stays_auto_when_the_device_is_known() {
    // 「Auto」在记账里被当成 CPU 的那个坑，就从这里开始 ⇒ ⇒
    // 设备已知时，解析结果**不许**还是 Auto。
    let mut c = caps(DeviceKind::Cpu, no_features());
    assert_ne!(c.resolved_device(), DeviceSelection::Auto);
    c.device = DeviceKind::Cuda;
    assert_ne!(c.resolved_device(), DeviceSelection::Auto);
    c.device = DeviceKind::Unknown;
    assert_eq!(c.resolved_device(), DeviceSelection::Auto);
}

#[test]
fn test_model_facts_are_read_off_the_disk_not_claimed() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    std::fs::write(
        dir.join("config.json"),
        r#"{"model_type":"qwen2","torch_dtype":"bfloat16"}"#,
    )
    .unwrap();
    std::fs::write(dir.join("model.safetensors"), vec![0u8; 4096]).unwrap();
    let f = ModelFacts::from_dir(dir).unwrap();
    assert_eq!(f.arch, "qwen2");
    assert_eq!(f.format, WeightFormat::Safetensors);
    assert_eq!(f.on_disk_bytes, Some(2));
    assert_eq!(f.weight_bytes, 4096);
}

#[test]
fn test_gguf_engine_follows_the_compiled_legs_not_the_runtime_flag() {
    // **负控**：运行时把 features.llamacpp 设成 true 也**造不出** llamacpp 腿
    // —— 腿在编译期就在不在 ⇒ ⇒ 引擎只能有编译进来的那些。
    let mut f = no_features();
    f.llamacpp = true;
    let mut m = model("qwen2", 1);
    m.format = WeightFormat::Gguf;
    let s = resolve_strategy(&caps(DeviceKind::Cpu, f), &m, &req());
    #[cfg(feature = "llamacpp")]
    {
        assert!(s.engine.is_some());
    }
    #[cfg(not(feature = "llamacpp"))]
    {
        assert!(s.engine.is_none());
        assert!(
            s.unavailable.iter().any(|u| u.contains("GGUF")),
            "must name the format: {:?}",
            s.unavailable
        );
    }
    // 不管有没有那条腿，「没编」时必须说出原因。
    let s2 = resolve_strategy(&caps(DeviceKind::Cpu, no_features()), &m, &req());
    assert!(s2.engine.is_none());
    assert!(
        s2.unavailable.iter().any(|u| u.contains("GGUF")),
        "must name the format: {:?}",
        s2.unavailable
    );
}

#[test]
fn test_runtime_feature_flag_cannot_invent_a_leg() {
    // **负控（与上一条同一形状）**：运行时把 features.candle 设成 true，
    // 也**造不出** candle 腿 ⇒ ⇒ 引擎只看编译进来的腿。
    let mut f = no_features();
    f.candle = true;
    let s = resolve_strategy(&caps(DeviceKind::Cpu, f), &model("qwen2", 1), &req());
    #[cfg(not(feature = "candle"))]
    {
        assert!(
            s.engine.is_none(),
            "no candle leg compiled ⇒ no candle engine"
        );
    }
    #[cfg(feature = "candle")]
    {
        assert!(s.engine.is_some());
    }
}

// ===========================================================================
// 2026-11-25 补：策略**接进运行时**那一刀逼出来的判据
// ===========================================================================

#[test]
fn test_explicit_device_request_wins_over_detection() {
    // 用户说 NVIDIA ⇒ ⇒ 策略**不许**替他换成 CPU（哪怕这台机器的 build
    // 没编 cuda）⇒ ⇒ 但必须把「这条路开不了」说出来。
    let s = resolve_strategy(
        &caps(DeviceKind::Cpu, no_features()),
        &model("qwen2", 1),
        &StrategyRequest {
            device: Some(DeviceSelection::Nvidia),
            dtype: None,
            flash_attn: None,
        },
    );
    assert_eq!(s.device, DeviceSelection::Nvidia);
    assert!(
        s.unavailable.iter().any(|u| u.contains("cuda")),
        "must name the missing feature instead of silently switching: {:?}",
        s.unavailable
    );
    assert!(
        s.reasons.iter().any(|r| r.contains("显式")),
        "must say the user's choice was honoured: {:?}",
        s.reasons
    );
}

#[test]
fn test_explicit_dtype_request_wins_over_the_default() {
    let s = resolve_strategy(
        &caps(DeviceKind::Cpu, no_features()),
        &model("qwen2", 1_000_000_000),
        &StrategyRequest {
            device: None,
            dtype: Some(DTypeChoice::BF16),
            flash_attn: None,
        },
    );
    assert_eq!(s.dtype, DTypeChoice::BF16);
    assert!(
        s.reasons.iter().any(|r| r.contains("NEURON_LLM_DTYPE")),
        "must say where the dtype came from: {:?}",
        s.reasons
    );
}

#[test]
fn test_required_bytes_matches_the_materialized_dtype() {
    // 一个数字，一处算 ⇒ ⇒ 策略里的字节数必须等于按它自己的 dtype
    // 物化出来的字节数（不许两处各算一遍）。
    let m = model("qwen2", 2_000_000_000);
    let s = resolve_strategy(&caps(DeviceKind::Cpu, no_features()), &m, &req());
    assert_eq!(
        s.required_bytes,
        materialized_bytes(&m, s.dtype),
        "required_bytes must follow the dtype the strategy chose"
    );
}

#[test]
fn test_required_bytes_is_not_a_flat_constant() {
    // **负控**：旧的注册路径给**每个**模型都记 8 GB（0.92 GiB 的模型也
    // 按 8 GB 记）⇒ ⇒ 那是编出来的数字 ⇒ ⇒ 现在必须从盘上算。
    let m = model("qwen2", 988_000_000);
    let s = resolve_strategy(&caps(DeviceKind::Cpu, no_features()), &m, &req());
    assert_ne!(s.required_bytes, 8_000_000_000);
    assert_eq!(s.required_bytes, materialized_bytes(&m, DTypeChoice::F32));
}

#[test]
fn test_dtype_env_parser_agrees_with_the_runtime() {
    // **一处解析，两处消费** —— 解析器认得的拼法必须与运行时一致。
    assert_eq!(DTypeChoice::from_env_value("f32"), Some(DTypeChoice::F32));
    assert_eq!(DTypeChoice::from_env_value("F32"), Some(DTypeChoice::F32));
    assert_eq!(DTypeChoice::from_env_value("bf16"), Some(DTypeChoice::BF16));
    assert_eq!(
        DTypeChoice::from_env_value("bfloat16"),
        Some(DTypeChoice::BF16)
    );
    assert_eq!(DTypeChoice::from_env_value("f16"), Some(DTypeChoice::F16));
    assert_eq!(DTypeChoice::from_env_value("half"), Some(DTypeChoice::F16));
    // 认不出来 ⇒ **不猜** ⇒ None（调用方出声并回退）。
    assert_eq!(DTypeChoice::from_env_value("onn"), None);
    assert_eq!(DTypeChoice::from_env_value("  "), None);
}

#[test]
fn test_detected_platform_key_matches_the_detected_device() {
    // 调度器的平台键必须是**探测**出来的那个，不是写死的。
    assert_eq!(
        detected_platform_key(),
        HostCapabilities::detect().device.platform_key()
    );
    assert!(
        ["cpu", "cuda", "metal"].contains(&detected_platform_key()),
        "unexpected platform key: {}",
        detected_platform_key()
    );
}

#[test]
fn test_scheduler_platform_falls_back_to_the_detected_value() {
    // 旧口径：`platform` 没填 ⇒ "cpu" ⇒ ⇒ 一台跑在显存上的模型在
    // **cpu 档**记账 ⇒ ⇒ 而且 `observes_rss()` 会去观测一个**不含权重**
    // 的进程 RSS ⇒ ⇒ 现在空 ⇒ 探测值。
    let cfg = crate::scheduler::SchedulerConfig::default();
    assert_eq!(cfg.effective_platform(), detected_platform_key());
    assert_eq!(cfg.observes_rss(), cfg.effective_platform() == "cpu");
    // 显式填的仍然说了算。
    let mut explicit = cfg.clone();
    explicit.platform = "cuda".to_string();
    assert_eq!(explicit.effective_platform(), "cuda");
    assert!(!explicit.observes_rss());
}

#[test]
fn test_plan_reads_the_facts_off_the_disk() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("tiny-model");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("config.json"),
        r#"{"model_type":"qwen2","torch_dtype":"bfloat16"}"#,
    )
    .unwrap();
    std::fs::write(dir.join("model.safetensors"), vec![0u8; 4096]).unwrap();

    let plan = plan_with(&caps(DeviceKind::Cpu, no_features()), &dir, &req());
    match plan {
        Ok(p) => {
            assert_eq!(p.model_id, "tiny-model");
            assert_eq!(p.dtype, DTypeChoice::F32);
            // 4096 字节的盘上权重（bf16）⇒ f32 物化 ⇒ 8192 字节。
            assert_eq!(p.required_bytes, 8192);
            assert_eq!(p.device, DeviceSelection::Cpu);
        }
        Err(e) => {
            // 这个 build 没有能读它的腿 ⇒ ⇒ 失败**必须说清原因**。
            assert!(
                e.contains("架构") || e.contains("qwen2"),
                "the refusal must name the reason: {e}"
            );
        }
    }
}

#[test]
fn test_plan_refuses_a_model_no_leg_can_read() {
    // 架构不在任何已编腿的覆盖里 ⇒ ⇒ 计划失败 ⇒ ⇒ 注册面**注册期**
    // 就说清，而不是留一个注定在 load 时才炸的条目。
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("weird-model");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("config.json"),
        r#"{"model_type":"totally-unknown-arch","torch_dtype":"bfloat16"}"#,
    )
    .unwrap();
    std::fs::write(dir.join("model.safetensors"), vec![0u8; 16]).unwrap();

    // 本 build 若有 mistralrs 腿，它按 HF repo id／本地目录都能读 ⇒
    // 那时计划**允许**（引擎是编译进来的那条腿）；否则必须拒绝并说清。
    match plan_with(&caps(DeviceKind::Cpu, no_features()), &dir, &req()) {
        Ok(p) => {
            assert!(
                ["candle", "mistralrs", "llamacpp"].contains(&p.engine.as_str()),
                "the engine must be a compiled leg: {}",
                p.engine.as_str()
            );
        }
        Err(e) => {
            assert!(
                e.contains("totally-unknown-arch") || e.contains("架构"),
                "the refusal must name the architecture: {e}"
            );
        }
    }
}

// ===========================================================================
// 2026-11-25 补：**显存真的测得到**那一刀逼出来的判据
// ===========================================================================

#[test]
fn test_parse_nvidia_smi_vram_reads_device_zero() {
    // 正常输出 ⇒ ⇒ 测得到，而且**单位是字节**（不是 MiB 混进账本）。
    assert_eq!(
        parse_nvidia_smi_vram("0, 24576\n"),
        Some((24576 * 1024 * 1024, 1))
    );
}

#[test]
fn test_parse_nvidia_smi_vram_never_sums_cards() {
    // ⚠ **负控**：两张 24 GiB 的卡 ⇒ ⇒ **不是** 48 GiB 的池 ⇒ ⇒ 一个模型只能
    // 待在一张卡上 ⇒ ⇒ 求和会让账本放行一个**根本装不下**的模型。
    let (bytes, devices) = parse_nvidia_smi_vram("0, 24576\n1, 24576\n").unwrap();
    assert_eq!(bytes, 24576 * 1024 * 1024, "must be ONE card, not the sum");
    assert_eq!(devices, 2, "but both cards are counted");
}

#[test]
fn test_parse_nvidia_smi_vram_skips_na_rows_instead_of_reading_zero() {
    // ⚠ **负控**：「[N/A]」**不是 0**（Windows WDDM 下某些列就是它）⇒ ⇒
    // 那一行跳过，不许变成「一张 0 字节的卡」。
    let (bytes, devices) = parse_nvidia_smi_vram("0, [N/A]\n1, 24576\n").unwrap();
    assert_eq!(bytes, 24576 * 1024 * 1024);
    assert_eq!(devices, 1, "the unreadable row is not a card we know");
}

#[test]
fn test_parse_nvidia_smi_vram_on_garbage_is_none_not_zero() {
    // 测不到 ⇒ ⇒ `None`（**不是 0 字节**）—— 这一刀的第一条纪律。
    assert_eq!(parse_nvidia_smi_vram(""), None);
    assert_eq!(parse_nvidia_smi_vram("[N/A]\n"), None);
    assert_eq!(
        parse_nvidia_smi_vram("Failed to initialize NVML: Driver/library version mismatch"),
        None
    );
}

#[test]
fn test_parse_nvidia_smi_vram_accepts_the_mib_suffix() {
    assert_eq!(
        parse_nvidia_smi_vram("0, 24576 MiB\n"),
        Some((24576 * 1024 * 1024, 1))
    );
}

#[test]
fn test_vram_source_says_where_the_number_came_from() {
    assert!(VramSource::CudaDriver.is_measured());
    assert!(VramSource::NvidiaSmi.is_measured());
    assert!(!VramSource::Unmeasured.is_measured());
    assert!(VramSource::Unmeasured.name().contains("没测到"));
}

#[test]
fn test_detect_vram_never_reports_zero_bytes() {
    let r = detect_vram();
    match r.bytes {
        Some(b) => {
            assert!(b > 0, "测到的显存不可能是 0 字节");
            assert!(r.source.is_measured());
        }
        None => {
            assert_eq!(r.source, VramSource::Unmeasured, "没测到就必须说没测到");
        }
    }
}

#[test]
fn test_detected_capabilities_report_the_vram_source_consistently() {
    let caps = HostCapabilities::detect();
    match caps.vram_bytes {
        Some(b) => {
            assert!(b > 0);
            assert!(
                caps.vram_source.is_measured(),
                "a measured number must say where it came from: {:?}",
                caps.vram_source
            );
        }
        None => assert_eq!(
            caps.vram_source,
            VramSource::Unmeasured,
            "没有数就必须说没测到，不许留一个假的来源"
        ),
    }
}

#[test]
fn test_ledger_gpu_pool_is_the_measured_value_when_measured() {
    // 账本的显存池**必须就是测到的那个数** ⇒ ⇒ 不许再是 8 GB 的兜底常量。
    let caps = HostCapabilities::detect_cached();
    let ledger = crate::config::ResourceTracker::detect();
    match caps.vram_bytes {
        Some(v) => {
            assert_eq!(
                ledger.total_gpu_bytes(),
                v,
                "账本的显存池必须就是测到的那个数"
            );
            assert_eq!(ledger.gpu_source(), crate::config::TotalsSource::Measured);
        }
        None => {
            assert_eq!(ledger.gpu_source(), crate::config::TotalsSource::Fallback);
            assert_ne!(ledger.total_gpu_bytes(), 0);
        }
    }
}

#[test]
fn test_ledger_says_fallback_when_it_cannot_measure() {
    // 兜底不许被当真值 ⇒ ⇒ 默认账本（两个数都是写死的）必须自报 Fallback。
    let rt = crate::config::ResourceTracker::default();
    assert_eq!(rt.totals_source(), crate::config::TotalsSource::Fallback);
    assert_eq!(rt.cpu_source(), crate::config::TotalsSource::Fallback);
    assert_eq!(rt.gpu_source(), crate::config::TotalsSource::Fallback);
}

#[test]
fn test_measured_vram_does_not_change_the_device_choice() {
    // **负控**：测到显存 ≠ 这个 build 能用 GPU ⇒ ⇒ 设备选择不许被显存带跑。
    let s = resolve_strategy(
        &caps(DeviceKind::Cuda, no_features()),
        &model("qwen2", 1),
        &req(),
    );
    assert_eq!(
        s.device,
        DeviceSelection::Cpu,
        "a 24 GiB card exists, but this build has no cuda feature"
    );
}

#[test]
fn test_the_fit_check_uses_the_measured_pool_not_the_fallback_constant() {
    // **这一刀的正题**：3B 级（盘上 bf16 6 GiB ⇒ f32 物化 12 GiB）在 24 GiB
    // 的池上装得下 ⇒ ⇒ 策略不许说「装不下」⇒ ⇒ 而在 8 GiB 的池上必须说。
    let mut f = no_features();
    f.cuda = true;
    let mut c = caps(DeviceKind::Cuda, f);
    let m = model("qwen2", 6 * 1024 * 1024 * 1024);
    assert_eq!(
        materialized_bytes(&m, DTypeChoice::F32),
        12 * 1024 * 1024 * 1024
    );

    c.vram_bytes = Some(24 * 1024 * 1024 * 1024);
    // ⚠⚠️ **两栏都要设**（刀 16 口径变更）—— 容量池优先用「现在可用」。
    c.ram_available_bytes = c.ram_bytes;
    let fits = resolve_strategy(&c, &m, &req());
    assert!(
        !fits.unavailable.iter().any(|u| u.contains("装不下")),
        "12 GiB fits in a 24 GiB pool: {:?}",
        fits.unavailable
    );

    // ⚠ **口径更新（刀 10 ＋ 刀 12）**：
    // · 刀 10：现在「装不下」会先问一句「**CPU 装得下吗**」—— 装得下就
    //   **自动退 CPU** ⇒ ⇒ 所以要让这一格仍然说「装不下」，**内存也得不够**
    //   （这里给 4 GiB）；
    // · 刀 12：**降精度排在换设备前面** ⇒ ⇒ 只把池子调小**不够**了 ——
    //   6 GiB 的权重（盘上 bf16）降到 bf16 只要 6 GiB ⇒ ⇒ 在 8 GiB 的显存上
    //   **装得下**（而且**留在 GPU**）⇒ ⇒ 要说「装不下」，得连 bf16 都装不下
    //   （14 GiB 的权重）。
    c.vram_bytes = Some(8 * 1024 * 1024 * 1024);
    c.ram_bytes = Some(4 * 1024 * 1024 * 1024);
    // ⚠⚠️ **两栏都要设**（刀 16 口径变更）—— 容量池优先用「现在可用」。
    c.ram_available_bytes = c.ram_bytes;
    let heavy = model("qwen2", 14 * 1024 * 1024 * 1024);
    let tight = resolve_strategy(&c, &heavy, &req());
    assert!(
        tight.unavailable.iter().any(|u| u.contains("装不下")),
        "12 GiB does not fit in an 8 GiB pool (and the CPU pool is smaller too): {:?}",
        tight.unavailable
    );
}

// ---- 刀 14：第三档（专家下沉）的判据 ----------------------------------------
//
// ⚠ 这一组的重点是**两件事不许混**：
// · 「判得了」（池子测到了 ⇒ 能比大小）与「判不了」（没测到 ⇒ 不猜）；
// · 「我们要求了什么」（override 装进参数了）与「专家真的落在 CPU 上」
//   （那要加载日志／显存读数）—— 后者**不由这一组证明**。

/// 造一份「池子由参数说了算、且设备是 CUDA」的环境。
fn caps_pools(ram: Option<u64>, vram: Option<u64>) -> HostCapabilities {
    // ⚠ 用 `..current()` 展开 —— 这个 struct 的字段别的会话还在加
    // （上一刀就撞过：少 candle／metal 两个字段 ⇒ E0063）⇒ ⇒ 写死字段列表
    // 迟早再撞一次。
    let features = CompiledFeatures {
        cuda: true,
        ..CompiledFeatures::current()
    };
    let mut c = caps(DeviceKind::Cuda, features);
    c.ram_bytes = ram;
    c.vram_bytes = vram;
    c.vram_source = VramSource::CudaDriver;
    c
}

#[test]
fn moe_that_fits_the_vram_stays_on_the_device() {
    let c = caps_pools(Some(64 * 1024 * 1024 * 1024), Some(24 * 1024 * 1024 * 1024));
    let p = resolve_expert_placement(&c, true, 20 * 1024 * 1024 * 1024, true);
    assert_eq!(p, ExpertPlacement::Device, "20 GiB fits in a 24 GiB pool");
}

#[test]
fn moe_that_does_not_fit_the_vram_but_fits_the_ram_sinks_its_experts() {
    // ⚠ **这是第三格** —— 此前只有「全 GPU」与「全退 CPU」两格。
    let c = caps_pools(Some(64 * 1024 * 1024 * 1024), Some(24 * 1024 * 1024 * 1024));
    let p = resolve_expert_placement(&c, true, 40 * 1024 * 1024 * 1024, true);
    assert_eq!(
        p,
        ExpertPlacement::ExpertsInRam,
        "40 GiB does not fit in a 24 GiB VRAM pool but fits in a 64 GiB RAM pool ⇒ ⇒ the experts sink into system memory while the rest stays on the GPU"
    );
}

#[test]
fn moe_that_fits_neither_pool_goes_all_the_way_to_the_host() {
    let c = caps_pools(Some(16 * 1024 * 1024 * 1024), Some(24 * 1024 * 1024 * 1024));
    let p = resolve_expert_placement(&c, true, 40 * 1024 * 1024 * 1024, true);
    assert_eq!(p, ExpertPlacement::Host);
}

#[test]
fn a_non_moe_weight_has_no_experts_to_sink() {
    let c = caps_pools(Some(16 * 1024 * 1024 * 1024), Some(24 * 1024 * 1024 * 1024));
    let p = resolve_expert_placement(&c, false, 40 * 1024 * 1024 * 1024, true);
    assert_eq!(p, ExpertPlacement::Device);
}

#[test]
fn an_unmeasured_pool_is_not_a_zero_pool() {
    // ⚠ 池子**没测到** ⇒ ⇒ 判不了 ⇒ ⇒ **不许**当成「0 ⇒ 装不下 ⇒ 退 CPU」。
    let c = caps_pools(None, None);
    let p = resolve_expert_placement(&c, true, 40 * 1024 * 1024 * 1024, true);
    assert_eq!(p, ExpertPlacement::Unknown);
    let c2 = caps_pools(Some(64 * 1024 * 1024 * 1024), None);
    assert_eq!(
        resolve_expert_placement(&c2, true, 1, true),
        ExpertPlacement::Unknown
    );
}

#[test]
fn the_third_rung_needs_the_executor_to_actually_have_a_gpu_backend() {
    // ⚠⚠️ **刀 17（2026-11-25）** —— 这一条是**双向断言**：
    //   ① 执行者**有**加速设备后端 ⇒ ⇒ 「装不下显存、装得下内存」⇒ ⇒ 第三档；
    //   ② 执行者**没有**加速设备后端 ⇒ ⇒ 第三档**不成立** ⇒ ⇒ 整盘在主机。
    //
    // ⚠ **为什么②是真问题而不是假设** —— 策略层的「显存」是 **candle 测的**
    //   （`caps.vram_bytes`），而「llama.cpp 这个 build 有没有 GPU 后端」是
    //   **另一个事实** ⇒ ⇒ 两个可以脱钩（机器有卡，但这个 build 没编 CUDA）
    //   ⇒ ⇒ 脱钩时若不把执行者的答案带进去 ⇒ ⇒ 策略层会算出「其余在显存」
    //   ⇒ ⇒ ⇒ 而执行者那边**根本没有显存那一侧** ⇒ ⇒ ⇒ 「显存」是假话。
    let c = caps_pools(Some(64 * 1024 * 1024 * 1024), Some(24 * 1024 * 1024 * 1024));
    let weight = 40 * 1024 * 1024 * 1024;

    // ① 有后端 ⇒ 第三档（与既有语义一致）。
    assert_eq!(
        resolve_expert_placement(&c, true, weight, true),
        ExpertPlacement::ExpertsInRam,
        "40 GiB does not fit in a 24 GiB VRAM pool but fits in a 64 GiB RAM pool ⇒ ⇒ the experts sink into system memory while the rest stays on the GPU"
    );

    // ② 没有后端 ⇒ 第三档**不成立** ⇒ ⇒ 整盘在主机。
    assert_eq!(
        resolve_expert_placement(&c, true, weight, false),
        ExpertPlacement::Host,
        "without an accelerator backend on the executor there is no device side at all ⇒ ⇒ the third rung does not exist ⇒ ⇒ everything stays on the host"
    );

    // ⚠ **连「装得下显存」那一档也不成立** —— 「全在加速设备上」这句话
    //   本身就需要**有那个设备** ⇒ ⇒ 没设备时连 Device 都不能报（那是假话）。
    assert_eq!(
        resolve_expert_placement(&c, true, 20 * 1024 * 1024 * 1024, false),
        ExpertPlacement::Host,
        "a weight that would fit the VRAM still has no device to sit on ⇒ ⇒ Host, not Device"
    );

    // ⚠ **非 MoE 也一样** —— 「全在加速设备上」对非 MoE 权重同样需要设备。
    assert_eq!(
        resolve_expert_placement(&c, false, 20 * 1024 * 1024 * 1024, false),
        ExpertPlacement::Host,
        "no executor GPU backend ⇒ ⇒ nothing can sit on a device ⇒ ⇒ Host even for a non-MoE weight"
    );
}

#[test]
fn the_read_back_alone_says_which_rung_it_is() {
    // 没读回 ⇒ 判不了。
    assert_eq!(classify_expert_placement(None), ExpertPlacement::Unknown);
    // 读回了、确实一个 override 都没有 ⇒ 全在设备。
    assert_eq!(
        classify_expert_placement(Some(&[])),
        ExpertPlacement::Device
    );
    // 读回了专家那条 ⇒ 第三档。
    let experts = vec![LLAMA_EXPERT_TENSOR_PATTERN.to_string()];
    assert_eq!(
        classify_expert_placement(Some(&experts)),
        ExpertPlacement::ExpertsInRam
    );
    // 读回了别的（不是专家）⇒ **不冒充**第三档。
    let others = vec!["blk\\.\\d+\\.attn_q.*".to_string()];
    assert_eq!(
        classify_expert_placement(Some(&others)),
        ExpertPlacement::Mixed
    );
}

#[test]
fn the_expert_pattern_is_the_upstream_one_verbatim() {
    // ⚠ 逐字抄上游 LLM_FFN_EXPS_REGEX —— 两边不是一个模式，就判不出这一格。
    assert_eq!(
        LLAMA_EXPERT_TENSOR_PATTERN,
        "\\.ffn_(up|down|gate|gate_up)_(ch|)exps"
    );
    assert!(LLAMA_EXPERT_TENSOR_PATTERN.contains("exps"));
    assert!(LLAMA_EXPERT_TENSOR_PATTERN.contains("ffn_"));
}

#[test]
fn the_moe_arch_list_is_a_fallback_not_a_verdict() {
    assert!(arch_looks_moe("qwen3moe"));
    assert!(arch_looks_moe("qwen3next"));
    // 名单外 ⇒ ⇒ 按「不是 MoE」处理（保守：不敢下沉就别下沉）。
    assert!(!arch_looks_moe("qwen2"));
    assert!(!arch_looks_moe("llama"));
}

// ---- 「跑得起来吗」这一条腿（刀 15）----
//
// ⚠ 这一刀的**全部意义**：候选必须**同时**满足「装得下」（容量）与
// 「跑得起来」（前提）⇒ ⇒ 只判容量的策略会放行一个「**加载成功、前向炸**」
// 的档 —— 那是真机撞出来的（qwen3-0.6b ＋ CUDA ＋ 编了 `flash-attn`）。

/// 造一份「CUDA 设备 ＋ 编了 flash-attn」的环境 —— **雷区那一格**。
fn caps_cuda_flash() -> HostCapabilities {
    let mut f = no_features();
    f.candle = true;
    f.cuda = true;
    f.flash_attn = true;
    caps(DeviceKind::Cuda, f)
}

#[test]
fn qwen3_f32_on_cuda_with_flash_attn_is_not_runnable() {
    // ⚠ 真机撞出来的那一格：qwen3 族 ＋ CUDA ＋ 编了 flash-attn ＋ f32
    // ⇒ ⇒ **加载成功、前向炸** ⇒ ⇒ 判据必须说「跑不起来」。
    assert!(!dtype_runnable_for_arch(
        "qwen3",
        DTypeChoice::F32,
        true,
        true
    ));
    // f16／bf16：到处都跑得起来。
    assert!(dtype_runnable_for_arch(
        "qwen3",
        DTypeChoice::BF16,
        true,
        true
    ));
    assert!(dtype_runnable_for_arch(
        "qwen3",
        DTypeChoice::F16,
        true,
        true
    ));
}

#[test]
fn the_constraint_stays_inside_the_cell_that_was_measured() {
    // ⚠ **约束不许推广** —— CPU 上（编了／没编 flash-attn）都跑得起来；
    // CUDA 上**没编** flash-attn ⇒ ⇒ 走标准注意力 ⇒ ⇒ 也跑得起来。
    assert!(dtype_runnable_for_arch(
        "qwen3",
        DTypeChoice::F32,
        false,
        true
    ));
    assert!(dtype_runnable_for_arch(
        "qwen3",
        DTypeChoice::F32,
        false,
        false
    ));
    assert!(dtype_runnable_for_arch(
        "qwen3",
        DTypeChoice::F32,
        true,
        false
    ));
    // ⚠ qwen2／llama 实测 f32 生成成功 ⇒ ⇒ 不许被这条约束扫到。
    assert!(dtype_runnable_for_arch(
        "qwen2",
        DTypeChoice::F32,
        true,
        true
    ));
    assert!(dtype_runnable_for_arch(
        "llama",
        DTypeChoice::F32,
        true,
        true
    ));
    // qwen3.5 的两个拼写仍然是这一族（配置形状不支持是**另一件事**）。
    assert!(!dtype_runnable_for_arch(
        "qwen3_5",
        DTypeChoice::F32,
        true,
        true
    ));
    assert!(!dtype_runnable_for_arch(
        "qwen3.5",
        DTypeChoice::F32,
        true,
        true
    ));
}

#[test]
fn the_strategy_lands_on_a_runnable_dose_not_merely_a_loadable_one() {
    // ⚠ **这一刀的核心**：策略不能只判「装得下」⇒ ⇒ 必须同时判「跑得起来」。
    let s = resolve_strategy(&caps_cuda_flash(), &model("qwen3", 1_000_000_000), &req());
    assert_eq!(
        s.dtype,
        DTypeChoice::BF16,
        "f32 这一档跑不起来 ⇒ ⇒ 不能选它"
    );
    assert_eq!(
        s.device,
        DeviceSelection::Nvidia,
        "换的是**精度**，不是设备"
    );
    assert!(
        s.reasons.iter().any(|r| r.contains("跑不起来")),
        "必须说清是「跑不起来」，不许混进「装不下」；got: {:?}",
        s.reasons
    );
    assert!(
        !s.unavailable.iter().any(|u| u.contains("装不下")),
        "装得下就不是「装不下」；got: {:?}",
        s.unavailable
    );
}

#[test]
fn an_explicit_f32_on_a_pinned_gpu_is_reported_not_swapped() {
    // ⚠ 用户**显式**要了 f32 **且**把设备钉在 GPU ⇒ ⇒ 不许偷偷换档 ⇒ ⇒
    // 照实说「跑不起来」＋ 指出出路（运行时会在 **load 期**拒）。
    let req = StrategyRequest {
        device: Some(DeviceSelection::Nvidia),
        dtype: Some(DTypeChoice::F32),
        flash_attn: None,
    };
    let s = resolve_strategy(&caps_cuda_flash(), &model("qwen3", 1_000_000_000), &req);
    assert_eq!(s.dtype, DTypeChoice::F32, "显式指定的档不许偷偷换");
    assert_eq!(
        s.device,
        DeviceSelection::Nvidia,
        "显式指定的设备不许偷偷换"
    );
    assert!(
        s.unavailable
            .iter()
            .any(|u| u.contains("跑不起来") && u.contains("bf16")),
        "必须照实报「跑不起来」并指出出路；got: {:?}",
        s.unavailable
    );
}

#[test]
fn an_explicit_f32_with_a_free_device_falls_back_to_cpu_not_to_another_dtype() {
    // ⚠ 用户显式要了 f32，**但没钉设备** ⇒ ⇒ 设备是策略的权限 ⇒ ⇒
    // ⇒ 退 CPU（CPU 上 f32 跑得起来）⇒ ⇒ **dtype 不许动**。
    let req = StrategyRequest {
        device: None,
        dtype: Some(DTypeChoice::F32),
        flash_attn: None,
    };
    let s = resolve_strategy(&caps_cuda_flash(), &model("qwen3", 1_000_000_000), &req);
    assert_eq!(s.dtype, DTypeChoice::F32, "显式要的档不许偷偷换");
    assert_eq!(s.device, DeviceSelection::Cpu, "GPU 上跑不起来 ⇒ ⇒ 退 CPU");
    assert!(
        s.reasons.iter().any(|r| r.contains("跑不起来")),
        "退设备的原因必须说清是「跑不起来」；got: {:?}",
        s.reasons
    );
}

#[test]
fn the_two_measured_families_are_not_swept_by_the_new_constraint() {
    // ⚠ **回归**：qwen2／llama 实测 f32 生成成功 ⇒ ⇒ 策略不许动它们。
    for arch in ["qwen2", "llama"] {
        let s = resolve_strategy(&caps_cuda_flash(), &model(arch, 1_000_000_000), &req());
        assert_eq!(
            s.dtype,
            DTypeChoice::F32,
            "{arch} 实测 f32 能跑 ⇒ ⇒ 不许被扫到"
        );
        assert!(
            !s.reasons.iter().any(|r| r.contains("跑不起来")),
            "{arch} 不该被说成跑不起来；got: {:?}",
            s.reasons
        );
    }
}

#[test]
fn whatever_the_strategy_picks_is_either_runnable_or_reported() {
    // ⚠ **单一真相源**：策略选出来的那一档 ⇒ ⇒ 用**同一个**判据复查 ⇒ ⇒
    // ⇒ 要么跑得起来，要么**被照实报出来** —— 不许出现「选了一档跑不起来的
    // 档却什么都不说」。
    for arch in ["qwen3", "qwen2", "llama"] {
        for d in [DTypeChoice::F32, DTypeChoice::BF16] {
            let req = StrategyRequest {
                device: Some(DeviceSelection::Nvidia),
                dtype: Some(d),
                flash_attn: None,
            };
            let s = resolve_strategy(&caps_cuda_flash(), &model(arch, 1_000_000_000), &req);
            let runnable = dtype_runnable_for_arch(arch, s.dtype, true, true);
            if !runnable {
                assert!(
                    s.unavailable.iter().any(|u| u.contains("跑不起来")),
                    "{arch}/{} 跑不起来 ⇒ ⇒ 必须报出来；got: {:?}",
                    d.name(),
                    s.unavailable
                );
            }
        }
    }
}
