//! Probe for the candle leg's `NEURON_USE_FLASH_ATTN` knob.
//!
//! **What it measures**: whether "I asked for FlashAttention-2" and
//! "FlashAttention-2 is in use" are the same fact. They are not — the
//! request needs three prerequisites (the `flash-attn` build feature, an
//! architecture that has an FA path, and a CUDA device) — and the
//! failure mode this probe exists to kill is the **silent** one: a build
//! whose feature is missing (or whose device is not CUDA) quietly
//! running the default attention path while the operator believes FA is
//! on.
//!
//! **How**: four runs of the PRODUCT path (`CandleLlmRuntime::load` +
//! `LlmRuntime::chat`, real weights) through a child process, one per
//! spelling of the knob, with the child's stderr read by the parent.
//!
//! | case | knob | expected |
//! |------|------|----------|
//! | unset | not set | silent — no complaint, no in-use marker |
//! | off | `off` | silent |
//! | on | `1` | **exactly one** of {complaint, in-use marker} |
//! | unknown | `onn` | a complaint quoting `onn`, no in-use marker |
//!
//! The "on" row is the measurement: the complaint and the in-use
//! marker are **mutually exclusive**, and the complaint has to name the
//! missing prerequisite. A run where both appear (or neither does) is
//! a failed run, not a green one.
//!
//! Every case also has to end with **the model answering**: a degraded
//! acceleration must not take the model down with it. The acceleration
//! is missing; the service must not be.
//!
//! **Exit codes** (a probe that cannot decide must not look green):
//!
//! | code | meaning |
//! |------|---------|
//! | 0 | every spelling behaved as documented |
//! | 1 | at least one spelling behaved differently |
//! | 2 | the probe could not run (no weights / child crashed) |
//!
//! Environment:
//!
//! * `NEURON_FA_PROBE_MODEL` — model id under `~/.neuron/models/llm`
//!   (default `qwen2.5-0.5b`).
//! * `NEURON_FA_PROBE_MAX_TOKENS` — generation cap (default 16).
//! * `NEURON_FA_PROBE_DEVICE` — `cpu` (default) or `nvidia`. `nvidia`
//!   needs the `cuda` feature; without it the child cannot open a CUDA
//!   device and the probe reports "could not run" (exit 2) rather
//!   than pretending.
//!
//! Run:
//!
//! ```bash
//! cargo run --release --example flash_attn_status_probe
//! ```

use neuron_lib::config::DeviceSelection;
use neuron_lib::llm::{CandleLlmRuntime, ChatMessage, ChatRequest, LlmRuntime};
use std::path::Path;
use std::process::Command;

/// Sentinels the parent parses out of the child's stdout.
const ANSWER_BEGIN: &str = "NEURON_FA_ANSWER_BEGIN";
const ANSWER_END: &str = "NEURON_FA_ANSWER_END";

/// The prefix every operator-facing degradation line carries.
const ERROR_MARKER: &str = "[neuron] ERROR:";

/// The positive half of the same fact: the state that IS in use.
const IN_USE_MARKER: &str = "FlashAttention-2 IS IN USE";

const DEFAULT_MODEL: &str = "qwen2.5-0.5b";

/// What a case must show.
#[derive(Clone, Copy)]
enum Expect {
    /// No complaint and no in-use marker.
    Silent,
    /// A complaint containing this substring — and no in-use marker.
    Complaint(&'static str),
    /// The in-use marker — and no complaint.
    InUse,
}

/// One spelling of the knob and what it owes the operator.
struct Case {
    label: &'static str,
    /// `None` ⇒ the knob is not set at all.
    value: Option<&'static str>,
    expect: Expect,
}

/// Whether the dtype the child will materialise the weights in is one
/// the FlashAttention kernels accept.
///
/// They take **f16/bf16 only** — **not** f32 ⇒ ⇒ and the runtime's
/// default dtype is f32 ⇒ ⇒ so in that regime the probe must expect a
/// **complaint** (naming `NEURON_LLM_DTYPE=bf16`), not an in-use
/// marker. This is the fourth prerequisite, and it was found the hard
/// way: with f32 weights the FA path was entered and then died inside
/// the forward pass.
fn dtype_takes_flash_attn() -> bool {
    match std::env::var("NEURON_LLM_DTYPE") {
        Ok(raw) => matches!(
            raw.trim().to_ascii_lowercase().as_str(),
            "bf16" | "bfloat16" | "f16" | "fp16" | "half"
        ),
        Err(_) => false,
    }
}

/// The device the probe drives the model on.
fn device() -> DeviceSelection {
    match std::env::var("NEURON_FA_PROBE_DEVICE")
        .unwrap_or_default()
        .as_str()
    {
        "nvidia" | "cuda" | "gpu" => DeviceSelection::Nvidia,
        _ => DeviceSelection::Cpu,
    }
}

/// The four spellings, and what each of them owes the operator.
///
/// The "on" case depends on the build AND the device, and the probe
/// is compiled with the same features as the crate — so `cfg!` is the
/// honest source of the first, and the requested device is the honest
/// source of the second.
fn cases() -> Vec<Case> {
    let feature = cfg!(feature = "flash-attn");
    let cuda_feature = cfg!(feature = "cuda");
    let on_cuda = matches!(device(), DeviceSelection::Nvidia) && cuda_feature;
    let dtype_ok = dtype_takes_flash_attn();
    let on_expect = if !feature {
        // No FA feature in this build ⇒ the missing prerequisite is
        // the feature itself.
        Expect::Complaint("flash-attn")
    } else if !on_cuda {
        // The feature is there, the device is not ⇒ the missing
        // prerequisite is the device.
        Expect::Complaint("device is not CUDA")
    } else if !dtype_ok {
        // Feature and device are there, but the weights are in a dtype
        // the FA kernels refuse ⇒ ⇒ the missing prerequisite is the
        // **dtype** ⇒ ⇒ and the complaint has to name the knob.
        Expect::Complaint("NEURON_LLM_DTYPE=bf16")
    } else {
        // Everything is present ⇒ FA really is in use, and the run
        // must say so.
        Expect::InUse
    };
    vec![
        Case {
            label: "unset",
            value: None,
            expect: Expect::Silent,
        },
        Case {
            label: "on",
            value: Some("1"),
            expect: on_expect,
        },
        Case {
            label: "unknown",
            value: Some("onn"),
            expect: Expect::Complaint("onn"),
        },
        Case {
            label: "off",
            value: Some("off"),
            expect: Expect::Silent,
        },
    ]
}

/// The child: the PRODUCT path with real weights.
async fn run_child() -> Result<(), Box<dyn std::error::Error>> {
    let model =
        std::env::var("NEURON_FA_PROBE_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
    let max_tokens: u32 = std::env::var("NEURON_FA_PROBE_MAX_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    let dev = device();

    let runtime = CandleLlmRuntime::load(&model, &dev).await?;

    let req = ChatRequest {
        model: model.clone(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: "Reply with the single word OK.".to_string(),
            reasoning_content: None,
        }],
        temperature: Some(0.0),
        max_tokens: Some(max_tokens),
        top_p: None,
        top_k: None,
        stream: false,
        stop: None,
    };
    let resp = runtime.chat(req).await?;
    let text = resp
        .choices
        .first()
        .map(|c| c.message.content.clone())
        .unwrap_or_default();
    println!("{ANSWER_BEGIN}{text}{ANSWER_END}");
    Ok(())
}

/// Run one case and judge it.
///
/// `Err` means the probe **cannot decide** (the child could not even
/// run); `Ok(false)` means it ran and behaved differently than the
/// contract says.
fn run_case(exe: &Path, case: &Case) -> Result<(bool, bool, String), String> {
    let mut cmd = Command::new(exe);
    cmd.arg("--child");
    // Both the current and the legacy prefix have to go, or a stale
    // value in the parent environment would decide the case.
    cmd.env_remove("NEURON_USE_FLASH_ATTN");
    cmd.env_remove("NEURE_USE_FLASH_ATTN");
    if let Some(v) = case.value {
        cmd.env("NEURON_USE_FLASH_ATTN", v);
    }
    let out = cmd.output().map_err(|e| format!("spawn failed: {e}"))?;
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();

    if !out.status.success() {
        let tail: String = stderr.lines().rev().take(3).collect::<Vec<_>>().join(" | ");
        return Err(format!(
            "child exited with {:?}; last stderr: {}",
            out.status.code(),
            tail
        ));
    }

    let complaint = stderr
        .lines()
        .find(|l| l.contains(ERROR_MARKER) && l.contains("FlashAttention"))
        .unwrap_or("")
        .to_string();
    let complained = !complaint.is_empty();
    let in_use = stderr.lines().any(|l| l.contains(IN_USE_MARKER));
    let answered = stdout.contains(ANSWER_BEGIN) && stdout.contains(ANSWER_END);

    let mut problems: Vec<String> = Vec::new();
    match case.expect {
        Expect::Silent => {
            if complained {
                problems.push(format!("unexpected complaint: {complaint}"));
            }
            if in_use {
                problems.push("unexpected in-use marker".to_string());
            }
        }
        Expect::Complaint(marker) => {
            if !complained {
                problems.push(format!("no complaint (expected one mentioning `{marker}`)"));
            } else if !complaint.contains(marker) {
                problems.push(format!("complaint lacks `{marker}`: {complaint}"));
            }
            if in_use {
                problems.push("in-use marker next to a complaint".to_string());
            }
        }
        Expect::InUse => {
            if !in_use {
                problems.push("no in-use marker".to_string());
            }
            if complained {
                problems.push(format!("complaint next to an in-use marker: {complaint}"));
            }
        }
    }
    // A degradation must not take the model down with it.
    if !answered {
        problems.push("model did not answer".to_string());
    }

    let detail = if problems.is_empty() {
        if complained {
            complaint.clone()
        } else if in_use {
            "in-use marker seen".to_string()
        } else {
            "silent, as documented".to_string()
        }
    } else {
        problems.join("; ")
    };
    Ok((problems.is_empty(), answered, detail))
}

fn run_parent() -> Result<(), Box<dyn std::error::Error>> {
    let exe = std::env::current_exe()?;
    let model =
        std::env::var("NEURON_FA_PROBE_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
    println!("flash_attn_status_probe: model={model}");
    println!(
        "flash_attn_status_probe: build has the `flash-attn` feature = {}",
        cfg!(feature = "flash-attn")
    );
    println!(
        "flash_attn_status_probe: build has the `cuda` feature = {}",
        cfg!(feature = "cuda")
    );
    println!("flash_attn_status_probe: device={:?}", device());
    println!(
        "flash_attn_status_probe: weight dtype = {}（FA 内核只认 f16/bf16 ⇒ ⇒ 这一档的期望由它决定）",
        std::env::var("NEURON_LLM_DTYPE").unwrap_or_else(|_| "f32（运行时的默认）".to_string())
    );

    let mut all_ok = true;
    for case in cases() {
        match run_case(&exe, &case) {
            Ok((ok, answered, detail)) => {
                if !ok {
                    all_ok = false;
                }
                println!(
                    "NEURON_FA_CASE {} ok={ok} answered={answered} detail={detail}",
                    case.label
                );
            }
            Err(e) => {
                println!("NEURON_FA_CASE {} COULD_NOT_RUN {e}", case.label);
                // The probe cannot decide — it must not look green.
                std::process::exit(2);
            }
        }
    }

    if all_ok {
        println!("VERDICT: OK — every spelling of NEURON_USE_FLASH_ATTN behaved as documented");
        return Ok(());
    }
    println!("VERDICT: MISMATCH — at least one spelling behaved differently");
    std::process::exit(1);
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args().any(|a| a == "--child") {
        run_child().await
    } else {
        run_parent()
    }
}
