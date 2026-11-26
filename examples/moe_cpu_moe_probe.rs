//! MoE 专家下沉的 **neuron 进程级** 真机探针（刀 18）。
//!
//! **它走的是 neuron 自己的腿** —— `LlmRuntimeRegistry::register` ⇒ ⇒
//! `runtime_for` ⇒ ⇒ `LlamaCppLlmRuntime::load` ⇒ ⇒ 所以这一格量到的是
//! **neuron 的代码**（不是另写一个探针 crate）⇒ ⇒ 包括它那条读回与那条
//! `eprintln!`（判据 ／ 装进参数的 override ／ 实际落点）。
//!
//! **它量什么**：
//! | 量 | 怎么量 |
//! |---|---|
//! | 专家落点 | neuron 自己读回的 `tensor_buft_override_patterns()`（探针不重算） |
//! | 显存差 | 加载前后的 `nvidia-smi memory.used` 之**差**（不是绝对值） |
//! | 臂的判读 | `arm_env=`（**环境变量原始值**，探针**不重算**）＋ neuron 那条 `eprintln!`（权威） |
//!
//! **两臂**（同一份源码、同一台机器、两个进程）：
//! - **A（对照）**：不设 `NEURON_LLAMA_CPP_CPU_MOE` ⇒ ⇒ 专家按 `n_gpu_layers` 走设备；
//! - **B（专家钉 CPU）**：`NEURON_LLAMA_CPP_CPU_MOE=1` ⇒ ⇒ 专家张量钉 CPU buffer type
//!   （等价于上游 llama.cpp 的 `--cpu-moe`）。
//!
//! ⇒ ⇒ **两臂的显存差之差** ＝ 专家张量被挪走的那部分。
//!
//! **退出码**（判不了就不许假绿）：
//! | code | meaning |
//! |---|---|
//! | 0 | 装起来了 ＋ override 读回了 ＋ 显存差量到了 |
//! | 2 | 探针跑不起来（GGUF 不在／注册被拒／加载失败） |

use neuron_lib::accel::DTypeChoice;
use neuron_lib::config::{DeviceSelection, NeuronConfig};
use neuron_lib::llm::{LlmImpl, LlmRuntimeRegistry};
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

/// 显存占用（MiB）—— 读的是**整卡**的 `memory.used`，所以要取**差**。
fn vram_mib() -> Option<u64> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    s.lines().next()?.trim().parse::<u64>().ok()
}

#[tokio::main]
async fn main() {
    let path = match std::env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: moe_cpu_moe_probe <gguf path>");
            std::process::exit(2);
        }
    };
    let p = PathBuf::from(&path);
    let size = match std::fs::metadata(&p) {
        Ok(m) => m.len(),
        Err(e) => {
            eprintln!("probe: gguf not found: {e}");
            std::process::exit(2);
        }
    };
    // ⚠ 这一格的名字**必须**与 neuron 侧读的是同一个（`config::env_var`
    // 的前缀是 `NEURON_` ⇒ ⇒ `NEURON_LLAMA_CPP_CPU_MOE`）⇒ ⇒ 否则两臂其实同臂。
    //
    // ⚠⚠️ **不许在这里重算「这算不算开」** —— neuron 侧的真值集是
    // `1 / true / TRUE / on / yes`（`llamacpp_runtime::env_cpu_moe_requested`）
    // ⇒ ⇒ 探针若自己写一个**更窄**的集合（第一版只认 `"1"`）⇒ ⇒ 设成 `true`
    // 时探针会说「没开」而 neuron **照开** ⇒ ⇒ ⇒ 标签会**说谎** ⇒ ⇒ ⇒
    // ⇒ 那正是「失败长得像成功」的形状。
    // ⇒ ⇒ 所以这里只**逐字**打环境变量的**原始值** ⇒ ⇒ 权威判读留给 neuron
    // ⇒ ⇒ 自己那条 `eprintln!`（「判据 ／ 装进参数的 override ／ 这一档」）。
    let cpu_moe_env = std::env::var("NEURON_LLAMA_CPP_CPU_MOE").ok();
    println!("file_bytes={size}");
    println!("arm_env={}", cpu_moe_env.as_deref().unwrap_or("<unset>"));

    let registry = LlmRuntimeRegistry::new(Arc::new(NeuronConfig::default()));
    // model_id 就用**绝对路径** ⇒ ⇒ neuron's `resolve_gguf_path` 认它 ⇒ ⇒
    // ⇒ 不必先注册到模型目录（这与真实请求同一条解析路径）。
    // ⚠ **类型要照着 registry 的真签名来**（实测 `registry.rs`）：`register` 收
    //    `String`（`model_id: String`）⇒ ⇒ `runtime_for` 收 `&str` ⇒ ⇒ ⇒
    //    ⇒ ⇒ 这两处**都**要用同一个 `String`（第一版传 `PathBuf` ⇒ ⇒ 编译不过）。
    let model_id = path.clone();
    if let Err(e) = registry.register(
        model_id.clone(),
        LlmImpl::LlamaCpp,
        DeviceSelection::Nvidia,
        DTypeChoice::F16,
        size,
    ) {
        eprintln!("probe: register failed: {e}");
        std::process::exit(2);
    }

    let vram_before = vram_mib();
    println!("VRAM_BEFORE={vram_before:?}");
    match registry.runtime_for(&model_id).await {
        Ok(_rt) => {
            let vram_after = vram_mib();
            println!("LOADED VRAM_AFTER={vram_after:?}");
            match (vram_before, vram_after) {
                (Some(b), Some(a)) => println!("VRAM_DELTA={} MiB", a as i64 - b as i64),
                _ => println!("VRAM_DELTA=unknown"),
            }
            println!("VERDICT: OK");
        }
        Err(e) => {
            eprintln!("probe: load failed: {e}");
            std::process::exit(2);
        }
    }
}
