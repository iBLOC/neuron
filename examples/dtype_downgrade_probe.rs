//! 「自动降级」的**真机探针**（刀 12）—— 降级窗在这台机器上到底开在哪。
//!
//! **它量什么**：策略层拿到「这台机器真的显存／内存」 ＋ 「盘上真的是
//! bf16 的权重」 ⇒ ⇒ 会算出什么设备／什么精度 ⇒ ⇒ 降级窗开在哪一段。
//!
//! ⚠ **它量的不是「能不能真加载」** —— 它是**纯函数**（`resolve_strategy`）
//! 在**真机参数**上的输出 ⇒ ⇒ 属于**策略级真机读数**，不是真加载读数。
//!
//! **两种模式**：
//! · 默认：**扫**一组参数量 ⇒ ⇒ 打出「降级窗」开在哪一段（合成事实 ＋ 真机参数）；
//! · `NEURON_DTYPE_PROBE_DIR=<模型目录>`：读**真模型目录**的事实
//!   （`ModelFacts::from_dir`）⇒ ⇒ 真权重 ＋ 真机参数。
//!
//! **退出码**：0 ＝ 探到了；2 ＝ 探针跑不起来。

use neuron_lib::accel::{
    DTypeChoice, DeviceKind, HostCapabilities, ModelFacts, StrategyRequest, WeightFormat,
    materialized_bytes, resolve_strategy,
};
use neuron_lib::config::DeviceSelection;
use std::path::PathBuf;
use std::process::Command;

fn gib(b: u64) -> String {
    format!("{:.2}", b as f64 / (1024.0 * 1024.0 * 1024.0))
}

/// 机器真有的设备与显存（nvidia-smi）—— 与 `accel_plan_probe` 同一口径。
fn os_view() -> (Option<DeviceKind>, Option<u64>) {
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
        let text = String::from_utf8_lossy(&out.stdout).to_string();
        let mut best: Option<u64> = None;
        for line in text.lines() {
            let t = line.trim();
            if t.is_empty() {
                continue;
            }
            let mut f = t.split(',').map(|s| s.trim());
            let _idx = f.next();
            if let Some(mib) = f.next().and_then(|s| s.parse::<u64>().ok()) {
                best = Some(mib * 1024 * 1024);
                break;
            }
        }
        if best.is_some() {
            return (Some(DeviceKind::Cuda), best);
        }
    }
    (None, None)
}

fn synth(params_b: f64) -> ModelFacts {
    // 盘上是 bf16（每参数 2 字节）⇒ ⇒ 降回 bf16 是**无损**的。
    ModelFacts {
        model_id: format!("synthetic-{}B", params_b),
        arch: "qwen2".to_string(),
        weight_bytes: (params_b * 1e9) as u64,
        format: WeightFormat::Safetensors,
        on_disk_bytes: Some(2),
    }
}

fn row(
    caps: &HostCapabilities,
    facts: &ModelFacts,
    req: &StrategyRequest,
) -> (String, &'static str) {
    let s = resolve_strategy(caps, facts, req);
    let f32b = materialized_bytes(facts, DTypeChoice::F32);
    let bf16b = materialized_bytes(facts, DTypeChoice::BF16);
    let line = format!(
        "{:<18} f32={:>7} GiB  bf16={:>7} GiB  ⇒  {:<7} {:<4}  占 {:>7} GiB",
        facts.model_id,
        gib(f32b),
        gib(bf16b),
        format!("{:?}", s.device),
        s.dtype.name(),
        gib(s.required_bytes)
    );
    let verdict = if s.unavailable.iter().any(|u| u.contains("装不下")) {
        "⇒ ⇒ 装不下（照实报）"
    } else if s.dtype == DTypeChoice::BF16 && s.device != DeviceSelection::Cpu {
        "⇒ ⇒ **降精度救了这个模型，而且**没动设备"
    } else if s.dtype == DTypeChoice::BF16 {
        "⇒ ⇒ 降精度救了这个模型（设备也换了）"
    } else {
        "⇒ ⇒ 原样就装得下（没降级）"
    };
    (line, verdict)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let caps = HostCapabilities::detect();
    let (os_device, os_vram) = os_view();

    println!("== 真机参数（这个 build 探测到的 ＋ OS 说的） ==");
    println!("  device(build)     = {}", caps.device.name());
    println!("  machine_device    = {}", caps.machine_device.name());
    println!(
        "  ram_bytes         = {}",
        caps.ram_bytes.map(gib).unwrap_or_else(|| "没测到".into())
    );
    println!(
        "  vram_bytes        = {}",
        caps.vram_bytes.map(gib).unwrap_or_else(|| "没测到".into())
    );
    println!(
        "  os_device/os_vram = {} / {}",
        os_device
            .map(|d| d.name().to_string())
            .unwrap_or_else(|| "没测到".into()),
        os_vram.map(gib).unwrap_or_else(|| "没测到".into())
    );

    // 策略按**机器真有的**算（跟 `accel_plan_probe` 同一口径）。
    let mut real = caps.clone();
    if let Some(d) = os_device {
        real.device = d;
        real.machine_device = d;
    }
    if os_vram.is_some() {
        real.vram_bytes = os_vram;
    }

    let req = StrategyRequest {
        device: None,
        dtype: None,
        flash_attn: None,
    };

    // ---- 真模型目录模式 ----
    if let Ok(dir) = std::env::var("NEURON_DTYPE_PROBE_DIR") {
        let dir = PathBuf::from(dir);
        println!("\n== 真模型目录 ==");
        println!("  dir = {}", dir.display());
        let facts = match ModelFacts::from_dir(&dir) {
            Ok(f) => f,
            Err(e) => {
                println!("读不到模型目录：{e}");
                println!("VERDICT: COULD_NOT_RUN");
                std::process::exit(2);
            }
        };
        println!(
            "  facts: arch={} format={:?} 盘上权重={} GiB on_disk_bytes={:?}",
            facts.arch,
            facts.format,
            gib(facts.weight_bytes),
            facts.on_disk_bytes
        );
        let (line, verdict) = row(&real, &facts, &req);
        println!("  {line}");
        println!("  {verdict}");
        let s = resolve_strategy(&real, &facts, &req);
        println!("  reasons:");
        for r in &s.reasons {
            println!("    · {r}");
        }
        if !s.unavailable.is_empty() {
            println!("  unavailable:");
            for u in &s.unavailable {
                println!("    · {u}");
            }
        }
        println!("\nVERDICT: OK — 真权重 ＋ 真机参数 ⇒ 策略已算");
        return Ok(());
    }

    // ---- 扫描模式 ----
    println!("\n== 降级窗（盘上 bf16；真机参数 ＋ 合成权重事实） ==");
    let mut window_open = false;
    let mut window_closed = false;
    for params_b in [
        0.5f64, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 12.0, 14.0, 16.0, 20.0, 27.0, 32.0, 70.0,
    ] {
        let facts = synth(params_b);
        let (line, verdict) = row(&real, &facts, &req);
        println!("  {line}");
        println!("      {verdict}");
        if verdict.contains("降精度救了") {
            window_open = true;
        }
        if verdict.contains("装不下") {
            window_closed = true;
        }
    }
    println!();
    println!(
        "  降级窗在这台机器上**开**过 ＝ {}；**关**上（两池都不够）＝ {}",
        window_open, window_closed
    );
    println!("\nVERDICT: OK — 策略级真机读数（**不是**真加载读数）");
    Ok(())
}
