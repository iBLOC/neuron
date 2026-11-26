//! **设备对照台**：同一份真权重，CPU 腿 vs GPU 腿。
//!
//! **它量什么**：策略说「该走哪条腿」到底值不值 —— 同一权重、同一提示、
//! 同一 token 上限，**两条腿各跑一次**，比 tok/s。
//!
//! **而且**：父进程在子进程跑的时候**用 nvidia-smi 采样** ⇒ ⇒ 两列：
//! · **显存总占用的涨幅**（跑腿前后差值）—— ⚠ Windows/WDDM 下 nvidia-smi
//!   的**每进程**显存列是 `[N/A]`（实测）⇒ ⇒ 只能靠总量涨幅；
//! · **这个 pid 有没有 CUDA 上下文**（它在不在 compute-apps 列表里）——
//!   每进程的字节数读不到，**在不在列表里**读得到。
//! ⇒ ⇒ 「权重真的在显存里」是**量到的**，不是推断的。
//!
//! **为什么父子两个进程**：每条腿要各自初始化设备（CUDA 上下文按线程绑
//! 定，见 candle_runtime 的 bind_cuda_context）⇒ ⇒ 两条腿跑在同一个进程
//! 里会互相污染 ⇒ ⇒ 父进程只做三件事：**跑腿 ＋ 收读数 ＋ 采样显存**。
//!
//! **子进程走产品路径**（CandleLlmRuntime::load ＋ chat）——
//! 不是直接调 candle ⇒ ⇒ 量到的是**用户真正拿到的东西**。
//!
//! **退出码**（判不了就不许假绿）：
//!
//! | code | meaning |
//! |------|---------|
//! | 0 | 两条腿都有答案 ⇒ ⇒ **可比较** |
//! | 1 | 有一条腿跑了但没出答案（且不是被平台拒绝） |
//! | 2 | 跑不起来（没有权重／子进程崩了） |
//! | 3 | 有一条腿被**平台拒绝**（例如这个 build 没编 cuda ⇒ ⇒ 「CUDA not enabled」） |
//!
//! 环境变量：
//!
//! * NEURON_BENCH_MODEL_DIR —— 权重目录（默认 ~/.neuron/models/llm/qwen2.5-0.5b）；
//! * NEURON_BENCH_PROMPT —— 两条腿答同一道题；
//! * NEURON_BENCH_MAX_TOKENS —— 生成上限（默认 64）；
//! * NEURON_BENCH_LEGS —— 跑哪几条腿（默认 cpu,nvidia）。
//!
//! 运行：
//!
//! cargo run --release --example accel_device_bench --features cuda

use neuron_lib::LlmRuntime;
use neuron_lib::accel::{HostCapabilities, ModelFacts};
use neuron_lib::config::DeviceSelection;
use neuron_lib::llm::{CandleLlmRuntime, ChatMessage, ChatRequest};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

const FACTS_PREFIX: &str = "NEURON_BENCH_FACTS ";
const METRICS_PREFIX: &str = "NEURON_BENCH_METRICS ";
const ANSWER_BEGIN: &str = "NEURON_BENCH_ANSWER_BEGIN";
const ANSWER_END: &str = "NEURON_BENCH_ANSWER_END";

/// 贪心解码（temperature 0）⇒ ⇒ 两条腿答的是同一道题、可逐字比。
const DEFAULT_PROMPT: &str =
    "What is the smallest prime number greater than 5? Reply with just the number.";

fn default_dir() -> PathBuf {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_default();
    PathBuf::from(home)
        .join(".neuron")
        .join("models")
        .join("llm")
        .join("qwen2.5-0.5b")
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

/// 这条腿要哪台设备。
fn leg_device(name: &str) -> DeviceSelection {
    match name {
        "cpu" => DeviceSelection::Cpu,
        "nvidia" => DeviceSelection::Nvidia,
        "auto" => DeviceSelection::Auto,
        other => panic!("unknown leg {other:?} (use cpu,nvidia,auto)"),
    }
}

/// 权重目录的**事实**（从盘上读，不声称）。
fn checkpoint_facts(dir: &PathBuf) -> String {
    match ModelFacts::from_dir(dir) {
        Ok(f) => format!(
            "arch={} format={:?} weight_bytes={} on_disk_dtype={:?}",
            f.arch, f.format, f.weight_bytes, f.on_disk_bytes
        ),
        Err(e) => format!("facts_unavailable={e}"),
    }
}

// ---------------------------------------------------------------------------
// 子进程：一条腿
// ---------------------------------------------------------------------------

async fn run_child() -> Result<(), Box<dyn std::error::Error>> {
    let dir = PathBuf::from(env_or(
        "NEURON_BENCH_MODEL_DIR",
        &default_dir().to_string_lossy(),
    ));
    let prompt = env_or("NEURON_BENCH_PROMPT", DEFAULT_PROMPT);
    let max_tokens: u32 = std::env::var("NEURON_BENCH_MAX_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let leg = env_or("NEURON_BENCH_DEVICE", "cpu");
    let sel = leg_device(&leg);

    // 事实先行：加载被拒时，父进程仍要知道「是权重的问题还是平台的问题」。
    println!("{FACTS_PREFIX}{}", checkpoint_facts(&dir));

    let load_started = Instant::now();
    let runtime: Box<dyn LlmRuntime> = CandleLlmRuntime::load(&dir.to_string_lossy(), &sel).await?;
    let load_s = load_started.elapsed().as_secs_f64();

    let req = ChatRequest {
        model: dir.to_string_lossy().into_owned(),
        messages: vec![ChatMessage {
            role: "user".into(),
            content: prompt,
            reasoning_content: None,
        }],
        temperature: Some(0.0),
        max_tokens: Some(max_tokens),
        top_p: None,
        top_k: None,
        stream: false,
        stop: None,
    };
    let answer_started = Instant::now();
    let resp = runtime.chat(req).await?;
    let answer_s = answer_started.elapsed().as_secs_f64();
    let completion_tokens = resp
        .usage
        .as_ref()
        .map(|u| u.completion_tokens)
        .unwrap_or(0);
    let text = resp
        .choices
        .first()
        .map(|c| c.message.content.clone())
        .unwrap_or_default();

    println!(
        "{METRICS_PREFIX}load_s={load_s:.3} answer_s={answer_s:.3} completion_tokens={completion_tokens}"
    );
    println!("{ANSWER_BEGIN}");
    println!("{text}");
    println!("{ANSWER_END}");
    Ok(())
}

// ---------------------------------------------------------------------------
// 父进程：跑腿 ＋ 收读数 ＋ 采样显存
// ---------------------------------------------------------------------------

/// 这张卡当前**总共**被占了多少 MiB。
///
/// ⚠ Windows/WDDM 下 nvidia-smi 的**每进程**显存列是 `[N/A]`（实测）
/// ⇒ ⇒ 证明「权重进了显存」只能靠**总占用的涨幅**（跑腿前后的差值）。
fn gpu_used_mib() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| l.trim().parse::<u64>().ok())
        .max()
}

/// 这个 pid 有没有 CUDA 上下文（它在 nvidia-smi 的 compute-apps 里吗）。
///
/// ⚠ 每进程的**字节数**在 Windows 上读不到（`[N/A]`）⇒ ⇒ 但
/// **在不在列表里**读得到 ⇒ ⇒ 这一列就是「这条腿真的开过 GPU」的证据。
fn has_cuda_context(pid: u32) -> bool {
    let Ok(out) = std::process::Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
    else {
        return false;
    };
    if !out.status.success() {
        return false;
    }
    String::from_utf8_lossy(&out.stdout).lines().any(|line| {
        line.split(',')
            .next()
            .map(|p| p.trim().parse::<u32>().ok() == Some(pid))
            .unwrap_or(false)
    })
}

/// 采样一条腿跑的时候：**显存总占用的涨幅** ＋ **它有没有 CUDA 上下文**。
fn sample_gpu_usage(pid: u32, baseline: Option<u64>, stop: Arc<AtomicBool>) -> (u64, bool) {
    let mut peak = baseline.unwrap_or(0);
    let mut ctx = false;
    while !stop.load(Ordering::Relaxed) {
        if let Some(used) = gpu_used_mib() {
            peak = peak.max(used);
        }
        if !ctx {
            ctx = has_cuda_context(pid);
        }
        std::thread::sleep(Duration::from_millis(400));
    }
    let delta = peak.saturating_sub(baseline.unwrap_or(0));
    (delta, ctx)
}

struct Leg {
    name: String,
    stdout: String,
    stderr: String,
    code: Option<i32>,
    facts: String,
    answer: Option<String>,
    load_s: Option<f64>,
    answer_s: Option<f64>,
    completion_tokens: Option<u32>,
    /// 跑这条腿期间，显存总占用涨了多少 MiB（0 ＝ 没涨）。
    vram_delta_mib: u64,
    /// 这个进程有没有 CUDA 上下文（⇒ 它真的开过 GPU）。
    cuda_context: bool,
}

fn parse_leg(
    name: &str,
    out: std::process::Output,
    vram_delta_mib: u64,
    cuda_context: bool,
) -> Leg {
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    let facts = stdout
        .lines()
        .find(|l| l.starts_with(FACTS_PREFIX))
        .map(|l| l.trim_start_matches(FACTS_PREFIX).to_string())
        .unwrap_or_default();
    let metrics = stdout
        .lines()
        .find(|l| l.starts_with(METRICS_PREFIX))
        .map(|l| l.trim_start_matches(METRICS_PREFIX).to_string())
        .unwrap_or_default();
    let mut load_s = None;
    let mut answer_s = None;
    let mut completion_tokens = None;
    for field in metrics.split_whitespace() {
        if let Some(v) = field.strip_prefix("load_s=") {
            load_s = v.parse().ok();
        } else if let Some(v) = field.strip_prefix("answer_s=") {
            answer_s = v.parse().ok();
        } else if let Some(v) = field.strip_prefix("completion_tokens=") {
            completion_tokens = v.parse().ok();
        }
    }
    let answer = stdout
        .split(ANSWER_BEGIN)
        .nth(1)
        .and_then(|tail| tail.split(ANSWER_END).next())
        .map(|s| s.trim().to_string());
    Leg {
        name: name.to_string(),
        stdout,
        stderr,
        code: out.status.code(),
        facts,
        answer,
        load_s,
        answer_s,
        completion_tokens,
        vram_delta_mib,
        cuda_context,
    }
}

fn tok_s(leg: &Leg) -> Option<f64> {
    let tokens = leg.completion_tokens? as f64;
    let secs = leg.answer_s?;
    if tokens > 0.0 && secs > 0.0 {
        Some(tokens / secs)
    } else {
        None
    }
}

fn print_tail(leg: &Leg) {
    let mut lines: Vec<&str> = leg
        .stderr
        .lines()
        .chain(leg.stdout.lines())
        .filter(|l| !l.trim().is_empty())
        .collect();
    let tail = if lines.len() > 14 {
        lines.split_off(lines.len() - 14)
    } else {
        lines
    };
    for l in tail {
        println!("  | {l}");
    }
}

fn run_parent() -> Result<(), Box<dyn std::error::Error>> {
    let exe = std::env::current_exe()?;
    let dir = env_or("NEURON_BENCH_MODEL_DIR", &default_dir().to_string_lossy());
    let legs: Vec<String> = env_or("NEURON_BENCH_LEGS", "cpu,nvidia")
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    println!("设备对照台（真权重 ＋ 产品路径）");
    println!("  weights : {dir}");
    let caps = HostCapabilities::detect();
    println!(
        "  build   : device={} features={:?}（这个 build 能用的）",
        caps.device.name(),
        caps.features
    );
    println!();

    let mut results: Vec<Leg> = Vec::new();
    for leg_name in &legs {
        println!("running leg {leg_name} …");
        let baseline = gpu_used_mib();
        let mut child = std::process::Command::new(&exe)
            .arg("--child")
            .env("NEURON_BENCH_DEVICE", leg_name)
            .env("NEURON_BENCH_MODEL_DIR", &dir)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()?;
        let pid = child.id();
        let stop = Arc::new(AtomicBool::new(false));
        let sampler_stop = stop.clone();
        let sampler = std::thread::spawn(move || sample_gpu_usage(pid, baseline, sampler_stop));
        let out = child.wait_with_output()?;
        stop.store(true, Ordering::Relaxed);
        let (delta, ctx) = sampler.join().unwrap_or((0, false));
        let leg = parse_leg(leg_name, out, delta, ctx);
        println!("leg {leg_name} finished\n");
        results.push(leg);
    }

    for leg in &results {
        println!("── leg: {} ──", leg.name);
        println!("  exit        : {:?}", leg.code);
        if !leg.facts.is_empty() {
            println!("  checkpoint  : {}", leg.facts);
        }
        if let Some(v) = leg.load_s {
            println!("  load_s      : {v:.2}");
        }
        if let Some(v) = leg.answer_s {
            println!("  answer_s    : {v:.2}");
        }
        if let Some(v) = leg.completion_tokens {
            println!("  completion  : {v} tokens");
        }
        match tok_s(leg) {
            Some(v) => println!("  tok/s       : {v:.2}"),
            None => println!("  tok/s       : (no reading)"),
        }
        println!(
            "  vram_delta  : +{} MiB（跑这条腿期间，显存总占用的涨幅）",
            leg.vram_delta_mib
        );
        println!(
            "  cuda_ctx    : {}（这个 pid 在不在 nvidia-smi 的 compute-apps 里）",
            if leg.cuda_context { "yes" } else { "no" }
        );
        match &leg.answer {
            Some(a) => println!("  answer      : {a:?}"),
            None => println!("  answer      : (none — 这条腿没出答案)"),
        }
        println!();
    }

    // ---- 判定 ----
    let cpu = results.iter().find(|l| l.name == "cpu");
    let gpu = results.iter().find(|l| l.name == "nvidia");

    if let Some(gpu) = gpu {
        if gpu.answer.is_none() {
            let refusal = format!("{} {}", gpu.stderr, gpu.stdout);
            if refusal.contains("CUDA not enabled") || refusal.contains("cuda:") {
                println!("VERDICT: BLOCKED — GPU 腿被这个 build 拒绝了（没编 cuda feature）");
                println!("  ⇒ ⇒ 「这台机器有 GPU」与「这个 build 能用 GPU」是两回事。");
                print_tail(gpu);
                std::process::exit(3);
            }
            println!("VERDICT: FAIL — GPU 腿没出答案");
            print_tail(gpu);
            std::process::exit(1);
        }
    }

    let (Some(cpu), Some(gpu)) = (cpu, gpu) else {
        println!("VERDICT: cannot decide — 只跑了一条腿，没有对照");
        std::process::exit(2);
    };
    if cpu.answer.is_none() {
        println!("VERDICT: cannot decide — CPU 腿没出答案，对照不成立");
        print_tail(cpu);
        std::process::exit(1);
    }
    match (tok_s(cpu), tok_s(gpu)) {
        (Some(a), Some(b)) => {
            println!(
                "tok/s      : cpu {a:.2}  →  nvidia {b:.2}   ({:.1}×)",
                b / a
            );
        }
        _ => {
            println!("VERDICT: cannot decide — 有一边没有 tok/s 读数");
            std::process::exit(2);
        }
    }
    println!(
        "vram_delta : cpu +{} MiB  →  nvidia +{} MiB",
        cpu.vram_delta_mib, gpu.vram_delta_mib
    );
    println!(
        "cuda_ctx   : cpu {}  →  nvidia {}",
        if cpu.cuda_context { "yes" } else { "no" },
        if gpu.cuda_context { "yes" } else { "no" }
    );
    if cpu.answer == gpu.answer {
        println!("answers    : identical（同一道题、贪心解码 ⇒ 两条腿答得一样）");
    } else {
        println!("answers    : DIFFERENT");
        println!("  cpu   : {:?}", cpu.answer);
        println!("  nvidia: {:?}", gpu.answer);
    }
    println!();
    println!("VERDICT: OK — 两条腿都跑通了，对照成立");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args().any(|a| a == "--child") {
        return run_child().await;
    }
    run_parent()
}
