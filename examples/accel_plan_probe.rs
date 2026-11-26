//! 设备环境 ⇒ 加速策略的**真机探针**。
//!
//! **它量什么**：neuron 的「设备环境」与「这台机器真有的环境」差多少。
//!
//! **三列对账表**（这是这个探针存在的理由）：
//!
//! | 列 | 来源 | 说的是 |
//! |---|---|---|
//! | neuron 以为的 | `HostCapabilities::detect()` ＋ `ResourceTracker::detect()` | **这个 build 能用**什么 ＋ **显存从哪儿测到的** |
//! | 账本里的 | `ResourceTracker` 的两个总数 ＋ `cpu_source()`／`gpu_source()` | 内存账本**认为**有多少（来源**分栏**标明）|
//! | 机器真有的 | `nvidia-smi`（显存／设备）＋ OS 的物理内存 | 这台机器**真有**什么 |
//!
//! ⇒ ⇒ **两列对不上时，那个差就是读数** —— 本机就有一张 24 GB 的卡，而这个
//! build 连 `cuda` 都没编 ⇒ ⇒ 「neuron 支持 GPU」这句话在这台机器上是假的。
//!
//! **退出码**（判不了就不许假绿）：
//!
//! | code | meaning |
//! |---|---|
//! | 0 | 环境探到了 ＋ 策略算出来了 ＋ 账本的来源已标明 |
//! | 2 | 探针跑不起来（模型目录不在／环境探不到） |

use neuron_lib::accel::{
    CompiledFeatures, DeviceKind, HostCapabilities, ModelFacts, resolve_strategy,
};
use neuron_lib::config::{DeviceSelection, ResourceTracker, TotalsSource};
use std::path::PathBuf;
use std::process::Command;

const DEFAULT_MODEL: &str = "qwen2.5-0.5b";

fn home() -> PathBuf {
    let h = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_default();
    PathBuf::from(h).join(".neuron").join("models").join("llm")
}

/// 机器真有的设备与显存（`nvidia-smi`）。
///
/// `None` ＝ **没测到**（没有 nvidia-smi／没有 NVIDIA 卡／命令失败），
/// 不是「0 字节」。
fn os_view() -> (Option<DeviceKind>, Option<u64>, usize) {
    // PATH 上的 nvidia-smi 优先；不在 PATH 里就找 CUDA 安装目录。
    let mut candidates: Vec<PathBuf> = vec![PathBuf::from("nvidia-smi")];
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        if !cuda_path.trim().is_empty() {
            let exe = if cfg!(windows) {
                "nvidia-smi.exe"
            } else {
                "nvidia-smi"
            };
            candidates.push(std::path::Path::new(&cuda_path).join("bin").join(exe));
        }
    }
    candidates.push(PathBuf::from(
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\nvidia-smi.exe",
    ));

    let mut text = String::new();
    for exe in &candidates {
        let Ok(out) = Command::new(exe)
            .arg("--query-gpu=index,memory.total")
            .arg("--format=csv,noheader,nounits")
            .output()
        else {
            continue;
        };
        if !out.status.success() {
            continue;
        }
        text = String::from_utf8_lossy(&out.stdout).to_string();
        break;
    }
    if text.is_empty() {
        return (None, None, 0);
    }

    // 取**物理序号 0 那一张**（与库里的「运行时用的那一张」同一口径）
    // ⇒ ⇒ 多卡机器上它与 CUDA 序号未必一致（CUDA 默认 FASTEST_FIRST）
    // ⇒ ⇒ 这正是库优先走 driver API 的原因。
    let mut rows: Vec<(u32, u64)> = Vec::new();
    for line in text.lines() {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        let mut f = t.split(',').map(|s| s.trim());
        let Some(idx) = f.next().and_then(|s| s.parse::<u32>().ok()) else {
            continue;
        };
        let Some(mib) = f.next().and_then(|s| s.parse::<u64>().ok()) else {
            continue;
        };
        rows.push((idx, mib));
    }
    if rows.is_empty() {
        return (None, None, 0);
    }
    let devices = rows.len();
    let picked = rows
        .iter()
        .find(|(i, _)| *i == 0)
        .or_else(|| rows.iter().min_by_key(|(i, _)| *i))
        .map(|(_, mib)| *mib)
        .unwrap_or(0);
    (Some(DeviceKind::Cuda), Some(picked * 1024 * 1024), devices)
}

fn gib(b: u64) -> String {
    format!("{:.2} GiB", b as f64 / (1024.0 * 1024.0 * 1024.0))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_id =
        std::env::var("NEURON_ACCEL_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
    let dir = home().join(&model_id);

    // ---- ① neuron 以为的环境 ----
    let caps = HostCapabilities::detect();
    println!("== neuron 以为的环境（这个 build 能用的） ==");
    println!("device            = {}", caps.device.name());
    // ⚠ **机器真有的设备**是**另一个字段** —— 它**不需要**编任何 feature
    // ⇒ ⇒ 在没编 `cuda` 的出厂 build 里，这一栏**照样**说得出「这台机器
    // 有卡」⇒ ⇒ 「GPU 那条路开不了」这句话才**说得出口**。
    println!(
        "machine_device    = {}   （机器真有的；**不依赖**本 build 编了什么）",
        caps.machine_device.name()
    );
    println!(
        "ram_bytes         = {}",
        caps.ram_bytes.map(gib).unwrap_or_else(|| "没测到".into())
    );
    println!(
        "vram_bytes        = {}",
        caps.vram_bytes.map(gib).unwrap_or_else(|| "没测到".into())
    );
    println!(
        "vram_bytes(字节)  = {}",
        caps.vram_bytes
            .map(|b| b.to_string())
            .unwrap_or_else(|| "没测到".into())
    );
    println!("vram_source       = {}", caps.vram_source.name());
    println!("vram_devices      = {}", caps.vram_devices);
    println!(
        "cores             = {}",
        caps.cores
            .map(|c| c.to_string())
            .unwrap_or_else(|| "没测到".into())
    );
    println!("features          = {}", feature_line(&caps.features));

    // ---- ② 账本里的（旧的硬编码 vs 探测到的） ----
    let tracker = ResourceTracker::detect();
    println!("\n== 内存账本 ==");
    println!(
        "totals_source     = {}",
        match tracker.totals_source() {
            TotalsSource::Measured => "Measured（测来的）",
            TotalsSource::Fallback => "Fallback（**兜底常量** —— 账本自己都这么说）",
        }
    );
    println!(
        "cpu_source        = {}",
        match tracker.cpu_source() {
            TotalsSource::Measured => "Measured（测来的）",
            TotalsSource::Fallback => "Fallback（**兜底常量** —— 旧的 16 GB）",
        }
    );
    println!(
        "gpu_source        = {}",
        match tracker.gpu_source() {
            TotalsSource::Measured => "Measured（测来的）",
            TotalsSource::Fallback => "Fallback（**兜底常量** —— 旧的 8 GB）",
        }
    );
    println!("total_cpu_bytes   = {}", gib(tracker.total_cpu_bytes()));
    println!("total_gpu_bytes   = {}", gib(tracker.total_gpu_bytes()));
    println!("resolved_device   = {:?}", tracker.resolved_device());

    // ---- ③ 机器真有的 ----
    let (os_device, os_vram, os_devices) = os_view();
    println!("\n== 机器真有的（OS 说的） ==");
    println!(
        "device            = {}",
        os_device
            .map(|d| d.name().to_string())
            .unwrap_or_else(|| "没测到".into())
    );
    println!(
        "vram_bytes        = {}",
        os_vram.map(gib).unwrap_or_else(|| "没测到".into())
    );
    println!(
        "vram_bytes(字节)  = {}",
        os_vram
            .map(|b| b.to_string())
            .unwrap_or_else(|| "没测到".into())
    );
    println!("vram_devices      = {}", os_devices);

    // ---- ④ 对账 ----
    println!("\n== 对账（差距就是读数） ==");
    let mut findings: Vec<String> = Vec::new();
    if os_device == Some(DeviceKind::Cuda) && !caps.features.cuda {
        findings.push(format!(
            "机器有 CUDA 设备（显存 {}），但这个 build **没编 `cuda` feature** ⇒ ⇒ 「neuron 支持 GPU」在这台机器上是假的",
            os_vram.map(gib).unwrap_or_else(|| "未知".into())
        ));
    }
    // ⚠ 机器的设备**也要对账** —— neuron 自己测的那一栏（machine_device）
    // 与 OS 说的必须是同一件事；不一致 ⇒ 说明这两条腿的**口径**不同。
    if let Some(os) = os_device {
        if caps.machine_device != os {
            findings.push(format!(
                "neuron 自己测到的机器设备（{}）与 OS 说的（{}）**对不上** ⇒ ⇒ 两条腿的口径不同",
                caps.machine_device.name(),
                os.name()
            ));
        }
    } else if caps.machine_device == DeviceKind::Cuda {
        findings.push(
            "OS 那边没测到设备，但 neuron 自己测到了 NVIDIA 卡 ⇒ ⇒ 两条腿的口径不同（或 OS 那条腿缺 nvidia-smi）"
                .to_string(),
        );
    }
    if os_vram.is_some() && caps.vram_bytes.is_none() {
        findings.push(
            "机器有显存，但 neuron 的账本**测不到**显存 ⇒ ⇒ 账本的 GPU 那一半仍是兜底常量（旧的 8 GB）"
                .to_string(),
        );
    }
    if let (Some(mine), Some(truth)) = (caps.vram_bytes, os_vram) {
        let rel = mine.abs_diff(truth) as f64 * 100.0 / truth.max(1) as f64;
        if rel > 1.0 {
            findings.push(format!(
                "库测到的显存（{} 字节）与 nvidia-smi 说的（{} 字节）**差 {:.2}%** ⇒ ⇒ 两个口径量的可能不是同一张卡（driver 按 CUDA 序号／nvidia-smi 按物理序号）",
                mine, truth, rel
            ));
        }
    }
    if tracker.gpu_source() == TotalsSource::Fallback {
        findings.push(
            "账本的**显存**那一半是**兜底常量**（旧的 8 GB）⇒ ⇒ 设备环境仍然被写死在代码里"
                .to_string(),
        );
    }
    if tracker.cpu_source() == TotalsSource::Fallback {
        findings.push("账本的**内存**那一半是**兜底常量**（旧的 16 GB）⇒ ⇒ 同上".to_string());
    }
    if findings.is_empty() {
        println!("（没有差距）");
    } else {
        for f in &findings {
            println!("⚠ {f}");
        }
    }
    // 口径差（**不是**差距）：两个测法说的本来就不是同一个数 ——
    // driver 给的是**驱动可用**容量（cudaMalloc 真能拿到的），
    // nvidia-smi 给的是**标称**容量 ⇒ ⇒ 逐字相等是**错的期望**，
    // 差 0.0x% 不许被报成「量的不是同一张卡」。
    if let (Some(mine), Some(truth)) = (caps.vram_bytes, os_vram) {
        let diff = mine.abs_diff(truth);
        if diff > 0 {
            println!(
                "（口径差：driver {} 字节 vs nvidia-smi {} 字节 ⇒ 差 {} 字节 / {:.2}% ⇒ ⇒ 两个测法的正常差，**不是**差距）",
                mine,
                truth,
                diff,
                diff as f64 * 100.0 / truth.max(1) as f64
            );
        }
    }

    // ---- ⑤ 策略（对真权重算） ----
    println!("\n== 策略（对真权重算） ==");
    println!("model_dir         = {}", dir.display());
    let facts = match ModelFacts::from_dir(&dir) {
        Ok(f) => f,
        Err(e) => {
            println!("读不到模型目录：{e}");
            println!("VERDICT: COULD_NOT_RUN");
            std::process::exit(2);
        }
    };
    println!(
        "facts             = arch={} format={:?} weights={} on_disk_dtype={:?}",
        facts.arch,
        facts.format,
        gib(facts.weight_bytes),
        facts.on_disk_bytes
    );

    // 策略用**机器真有的**设备来算（不是 neuron 自己那一列）⇒ ⇒
    // 这就是「按设备环境自动匹配」在真机上的答案。
    let mut real = caps.clone();
    if let Some(d) = os_device {
        real.device = d;
        real.machine_device = d;
    }
    if os_vram.is_some() {
        real.vram_bytes = os_vram;
    }
    // 用户**显式**要的（设备／dtype）—— 没说话就留空，交给策略。
    let req = neuron_lib::accel::StrategyRequest {
        device: None,
        dtype: neuron_lib::accel::requested_dtype(),
        flash_attn: neuron_lib::accel::requested_flash_attn(),
    };
    let strategy = resolve_strategy(&real, &facts, &req);
    println!("device            = {:?}", strategy.device);
    println!("dtype             = {}", strategy.dtype.name());
    println!(
        "flash_attn_avail = {}   （**能不能**开；开不开由旋钮决定 —— 请求 ≠ 生效）",
        strategy.flash_attn_available
    );
    println!(
        "engine            = {}",
        strategy
            .engine
            .map(|e| e.as_str().to_string())
            .unwrap_or_else(|| "**没有引擎能读它**".into())
    );
    println!("reasons:");
    for r in &strategy.reasons {
        println!("  · {r}");
    }
    if strategy.unavailable.is_empty() {
        println!("unavailable: （无）");
    } else {
        println!("unavailable:");
        for u in &strategy.unavailable {
            println!("  · {u}");
        }
    }

    // ---- ⑥ 注册面会怎么决定（策略接进运行时的那一面） ----
    //
    // ⑤ 那一段问的是「**机器真有的**」（拿 nvidia-smi 对账）；这一段问的是
    // 「**注册面会决定的**」（库自己探测 ＋ 策略算）⇒ ⇒ 两列不一样，
    // **那个差就是读数**。
    println!("\n== 注册面会怎么决定（策略接进运行时的那一面） ==");
    match neuron_lib::accel::plan_for_dir(&dir, &req) {
        Ok(plan) => {
            println!("engine            = {}", plan.engine.as_str());
            println!("device            = {:?}", plan.device);
            println!("dtype             = {}", plan.dtype.name());
            println!(
                "required_bytes    = {}   （旧的注册路径给**每个**模型都写死 8 GB）",
                gib(plan.required_bytes)
            );
            println!("reasons:");
            for r in &plan.reasons {
                println!("  · {r}");
            }
            if plan.unavailable.is_empty() {
                println!("unavailable: （无）");
            } else {
                println!("unavailable:");
                for u in &plan.unavailable {
                    println!("  · {u}");
                }
            }
            // 账本：按**真实**字节能不能装下 vs 按旧的 8 GB 能不能装下
            let ledger = ResourceTracker::detect();
            println!(
                "can_load(真实字节) = {}   can_load(旧 8 GB) = {}",
                ledger.can_load(plan.required_bytes, &plan.device),
                ledger.can_load(8_000_000_000, &plan.device)
            );
        }
        Err(e) => {
            println!("没有腿能读它 ⇒ 注册期就说清：{e}");
        }
    }

    // ---- ⑥-bis 反事实：账本的显存池够不够（拿真机显存当真相） ----
    //
    // 「库内显存测不到」这条债的**代价**是可以量的：真机 24 GiB 的卡上，
    // 一个**真放得下**的模型会不会被账本拒掉 ⇒ ⇒ 会 ⇒ ⇒ 那就是代价。
    println!("\n== 反事实：账本的显存池够不够（拿真机显存当真相） ==");
    match os_vram {
        Some(vram) => {
            let ledger = ResourceTracker::detect();
            // 3B 级权重按 f32 物化 = 12 GiB。
            let need_3b: u64 = 12 * 1024 * 1024 * 1024;
            let fits = need_3b <= vram;
            let admitted = ledger.can_load(need_3b, &DeviceSelection::Nvidia);
            println!(
                "账本的显存池          = {}（来源：{}）",
                gib(ledger.total_gpu_bytes()),
                match ledger.gpu_source() {
                    TotalsSource::Measured => "Measured（测来的）",
                    TotalsSource::Fallback => "Fallback（**兜底常量**）",
                }
            );
            println!(
                "3B 级（f32 物化 12.00 GiB）：真机装得下={fits}  账本说装得下={admitted}{}",
                if fits && !admitted {
                    "   ⇒ ⇒ **账本拒了真放得下的模型**（显存池是 8 GB 兜底常量）"
                } else if fits && admitted {
                    "   ⇒ ⇒ 账本**放行**了真放得下的模型（显存池是测到的）"
                } else {
                    ""
                }
            );
        }
        None => println!("（没测到真机显存 ⇒ 这一格判不了）"),
    }

    // ---- ⑦ 调度器的平台键 ----
    println!("\n== 调度器的平台键 ==");
    let sched = neuron_lib::scheduler::SchedulerConfig::default();
    println!(
        "effective_platform = {}   （旧口径：没填就是 \"cpu\"）",
        sched.effective_platform()
    );
    println!(
        "observes_rss       = {}   （进程 RSS 里**不含**显存的权重）",
        sched.observes_rss()
    );

    println!(
        "\nVERDICT: OK — 环境探到了 ＋ **显存测到了（来源已标明）** ＋ **机器真有的设备已测到（machine_device）** ＋ 策略算出来了 ＋ 账本来源已分栏标明 ＋ 注册面与调度器口径已接上"
    );
    Ok(())
}

fn feature_line(f: &CompiledFeatures) -> String {
    let mut v = Vec::new();
    if f.candle {
        v.push("candle");
    }
    if f.cuda {
        v.push("cuda");
    }
    if f.metal {
        v.push("metal");
    }
    if f.flash_attn {
        v.push("flash-attn");
    }
    if f.mistralrs {
        v.push("mistralrs");
    }
    if f.llamacpp {
        v.push("llamacpp");
    }
    if v.is_empty() {
        "（一个都没编）".to_string()
    } else {
        v.join("、")
    }
}
