//! A/B probe for the mistralrs leg's MTP (multi-token prediction) wiring.
//!
//! **What it measures**: whether the built-in MTP draft head really
//! engages on a REAL checkpoint, and what it costs. The only trustworthy
//! signal is the engine's own Throughput line —
//! `Throughput (T/s) …, MTP accept 87.5% (len 2.30, depth 3.10)` —
//! because that line is printed by the backend only when speculative
//! tokens were actually proposed and verified.
//!
//! **Why two processes**: the accept rate is printed by mistral.rs's
//! own tracing subscriber (stdout), not by a return value, and the
//! neuron runtime cannot introspect an attached draft head. So the probe
//! runs itself as a child (`--child`), captures the child's stdout, and
//! reads the line out of it. The child goes through the PRODUCT path
//! (`MistralRsLlmRuntime::load` + `LlmRuntime::chat`) — not through
//! mistral.rs directly — so what is measured is what a user gets.
//!
//! **The checkpoint facts are printed before the load**, so a refusal
//! can be attributed: "the checkpoint does ship the head" and "the
//! platform refuses to attach it" are two different claims, and only
//! the second one is a blocker.
//!
//! **Exit codes** (a probe that cannot decide must not look green):
//!
//! | code | meaning |
//! |------|---------|
//! | 0 | MTP engaged — an accept rate was observed |
//! | 1 | MTP was requested but never engaged (no accept-rate line) |
//! | 2 | the probe could not run (no weights, child crashed) |
//! | 3 | blocked by a backend/platform prerequisite (refusal observed and named) |
//!
//! Environment:
//!
//! * `NEURON_MTP_MODEL_DIR` — HF-format snapshot (default
//!   `~/.neuron/models/llm/Qwen/Qwen3.5-4B`).
//! * `NEURON_MTP_PROMPT` — the question both legs answer.
//! * `NEURON_MTP_MAX_TOKENS` — generation cap (default 64).
//! * `NEURON_MTP_ISQ` — forwarded to the children as
//!   `NEURON_MISTRALRS_ISQ` (default `q4_0`; F32 on CPU does not fit).
//!
//! Run:
//!
//! ```bash
//! cargo run --release --example mtp_ab_probe --features mistralrs
//! ```

use neuron_lib::config::DeviceSelection;
use neuron_lib::llm::{ChatMessage, ChatRequest, MistralRsLlmRuntime};
use std::path::PathBuf;

/// Sentinels the parent parses out of the child's stdout.
const ANSWER_BEGIN: &str = "NEURON_MTP_ANSWER_BEGIN";
const ANSWER_END: &str = "NEURON_MTP_ANSWER_END";
const METRICS_PREFIX: &str = "NEURON_MTP_METRICS ";
const FACTS_PREFIX: &str = "NEURON_MTP_FACTS ";
const ACCEPT_MARKER: &str = "MTP accept ";

/// A question with a short, checkable answer. Greedy decoding (temperature
/// 0) keeps the two legs comparable: MTP changes HOW the tokens are
/// produced, not WHICH tokens the model ends up with.
const DEFAULT_PROMPT: &str =
    "What is the smallest prime number greater than 5? Reply with just the number.";

fn default_dir() -> String {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_default();
    format!("{home}/.neuron/models/llm/Qwen/Qwen3.5-4B")
}

/// What the checkpoint itself says — read straight off disk, before any
/// weight is loaded.
///
/// `mtp_tensors` counts the tensor names in the safetensors index that
/// belong to the draft head. A checkpoint with zero of them can never
/// engage MTP, no matter what the platform allows.
fn checkpoint_facts(dir: &PathBuf) -> String {
    let mut facts: Vec<String> = Vec::new();
    if let Ok(raw) = std::fs::read_to_string(dir.join("config.json")) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) {
            let model_type = v.get("model_type").and_then(|m| m.as_str()).unwrap_or("?");
            facts.push(format!("model_type={model_type}"));
            // Qwen3.5 / Qwen3.8 keep the draft depth under this key.
            let depth = v.get("mtp_num_hidden_layers").and_then(|m| m.as_u64());
            facts.push(format!(
                "mtp_num_hidden_layers={}",
                depth
                    .map(|d| d.to_string())
                    .unwrap_or_else(|| "absent".into())
            ));
        }
    }
    if let Ok(raw) = std::fs::read_to_string(dir.join("model.safetensors.index.json")) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) {
            let n = v
                .get("weight_map")
                .and_then(|m| m.as_object())
                .map(|m| m.keys().filter(|k| k.starts_with("mtp.")).count())
                .unwrap_or(0);
            facts.push(format!("mtp_tensors={n}"));
        }
    }
    facts.join(" ")
}

async fn run_child() -> Result<(), Box<dyn std::error::Error>> {
    let dir =
        PathBuf::from(std::env::var("NEURON_MTP_MODEL_DIR").unwrap_or_else(|_| default_dir()));
    let prompt = std::env::var("NEURON_MTP_PROMPT").unwrap_or_else(|_| DEFAULT_PROMPT.to_string());
    let max_tokens: u32 = std::env::var("NEURON_MTP_MAX_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);

    // Facts first: if the load is refused, the parent still needs to know
    // whether the checkpoint was the problem.
    println!("{FACTS_PREFIX}{}", checkpoint_facts(&dir));

    let load_started = std::time::Instant::now();
    let runtime = MistralRsLlmRuntime::load(&dir.to_string_lossy(), &DeviceSelection::Cpu).await?;
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
    let answer_started = std::time::Instant::now();
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

/// Every accept rate printed in a chunk of backend output.
///
/// The line is `… MTP accept 87.5% (len 2.30, depth 3.10)` — the rate
/// is a per-window figure, so a run can print several; all of them are
/// reported, and the LAST one is the run's most recent state.
fn parse_accept_rates(text: &str) -> Vec<f64> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(idx) = rest.find(ACCEPT_MARKER) {
        rest = &rest[idx + ACCEPT_MARKER.len()..];
        let digits: String = rest
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.')
            .collect();
        if let Ok(v) = digits.parse::<f64>() {
            out.push(v);
        }
    }
    out
}

fn extract_between(text: &str, begin: &str, end: &str) -> Option<String> {
    let start = text.find(begin)? + begin.len();
    let stop = text[start..].find(end)? + start;
    Some(text[start..stop].trim().to_string())
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
    accept_rates: Vec<f64>,
}

fn parse_leg(name: &str, out: std::process::Output) -> Leg {
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
    let answer = extract_between(&stdout, ANSWER_BEGIN, ANSWER_END);
    Leg {
        name: name.to_string(),
        accept_rates: parse_accept_rates(&stdout),
        stdout,
        stderr,
        code: out.status.code(),
        facts,
        answer,
        load_s,
        answer_s,
        completion_tokens,
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

fn report_leg(leg: &Leg) {
    println!("── leg: {} ──", leg.name);
    println!("  exit        : {:?}", leg.code);
    if !leg.facts.is_empty() {
        println!("  checkpoint  : {}", leg.facts);
    }
    if let Some(load_s) = leg.load_s {
        println!("  load_s      : {load_s:.1}");
    }
    if let Some(answer_s) = leg.answer_s {
        println!("  answer_s    : {answer_s:.1}");
    }
    if let Some(tokens) = leg.completion_tokens {
        println!("  completion  : {tokens} tokens");
    }
    match tok_s(leg) {
        Some(v) => println!("  tok/s       : {v:.2}"),
        None => println!("  tok/s       : (no reading)"),
    }
    if leg.accept_rates.is_empty() {
        println!("  MTP accept  : (no accept-rate line was printed)");
    } else {
        println!(
            "  MTP accept  : {:?} (last {})",
            leg.accept_rates,
            leg.accept_rates.last().copied().unwrap_or_default()
        );
    }
    match &leg.answer {
        Some(a) => println!("  answer      : {a:?}"),
        None => println!("  answer      : (none — the leg did not finish)"),
    }
}

fn print_tail(leg: &Leg) {
    let mut lines: Vec<&str> = leg
        .stderr
        .lines()
        .chain(leg.stdout.lines())
        .filter(|l| !l.trim().is_empty())
        .collect();
    let tail = if lines.len() > 12 {
        lines.split_off(lines.len() - 12)
    } else {
        lines
    };
    if !tail.is_empty() {
        println!("  ── output tail ──");
        for l in tail {
            println!("  | {l}");
        }
    }
}

fn run_parent() -> Result<(), Box<dyn std::error::Error>> {
    let exe = std::env::current_exe()?;
    let dir = std::env::var("NEURON_MTP_MODEL_DIR").unwrap_or_else(|_| default_dir());
    let isq = std::env::var("NEURON_MTP_ISQ").unwrap_or_else(|_| "q4_0".into());

    println!("MTP A/B probe");
    println!("  checkpoint : {dir}");
    println!("  isq        : {isq}");
    println!();

    // Which legs to run. A single leg is a valid choice: on a machine
    // where the MTP leg is expected to be refused, the baseline leg is
    // the expensive half (it is the one that loads AND decodes), and the
    // verdict does not need it.
    let legs_wanted: Vec<(&str, &str)> = match std::env::var("NEURON_MTP_LEGS") {
        Err(_) => vec![("off", "off"), ("on", "on")],
        Ok(raw) => {
            let mut out = Vec::new();
            for token in raw.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
                match token {
                    "off" => out.push(("off", "off")),
                    "on" => out.push(("on", "on")),
                    other => {
                        return Err(format!(
                            "NEURON_MTP_LEGS got an unknown leg {other:?} (use off,on)"
                        )
                        .into());
                    }
                }
            }
            if out.is_empty() {
                return Err("NEURON_MTP_LEGS selected no leg".into());
            }
            out
        }
    };

    let mut legs: Vec<Leg> = Vec::new();
    for (name, mtp) in &legs_wanted {
        println!("running leg '{name}' (NEURON_MISTRALRS_MTP={mtp}) …");
        let out = std::process::Command::new(&exe)
            .arg("--child")
            .env("NEURON_MISTRALRS_MTP", *mtp)
            .env("NEURON_MISTRALRS_THROUGHPUT_LOGGING", "on")
            .env("NEURON_MISTRALRS_ISQ", &isq)
            .env("NEURON_MTP_MODEL_DIR", &dir)
            .output()?;
        legs.push(parse_leg(name, out));
        println!("leg '{name}' finished\n");
    }

    for leg in &legs {
        report_leg(leg);
    }
    println!();

    let off = legs.iter().find(|l| l.name == "off");
    let on = legs.iter().find(|l| l.name == "on");

    // Gate 1: the BASELINE leg has to work. Without a baseline there is
    // no A/B, and a broken baseline must not be mistaken for "MTP is
    // fine because the numbers look close".
    if let Some(off) = off {
        if off.answer.is_none() {
            println!("VERDICT: cannot decide — the baseline leg did not produce an answer");
            print_tail(off);
            std::process::exit(2);
        }
    } else {
        println!("note: the baseline leg was not run (NEURON_MTP_LEGS) — the verdict below");
        println!("      is about MTP only, not about a speed comparison.");
    }

    let Some(on) = on else {
        println!("VERDICT: cannot decide — the MTP leg never ran");
        std::process::exit(2);
    };

    // Gate 2: did the MTP leg refuse to load? That is a REAL reading —
    // the request reached the backend and the backend said no — but it
    // is not a measurement of MTP, so it gets its own exit code.
    if on.answer.is_none() {
        let refusal = format!("{} {}", on.stderr, on.stdout);
        if refusal.contains("PagedAttention") {
            println!("VERDICT: BLOCKED — the backend refused to attach the draft head.");
            println!("  The checkpoint facts above say whether the head itself is present.");
            println!("  Upstream requires PagedAttention for MTP; mistral.rs gates paged");
            println!("  attention on CUDA-on-unix / Metal, so a Windows build cannot");
            println!("  attach a built-in MTP head at all.");
            print_tail(on);
            std::process::exit(3);
        }
        println!("VERDICT: FAIL — the MTP leg did not produce an answer");
        print_tail(on);
        std::process::exit(1);
    }

    // Gate 3: the accept-rate line is the ONLY evidence that speculative
    // decoding actually ran. A successful answer without that line means
    // MTP was requested and did nothing.
    if on.accept_rates.is_empty() {
        println!("VERDICT: FAIL — MTP was requested but the engine never printed an");
        println!("  accept rate, so no speculative token was ever proposed.");
        print_tail(on);
        std::process::exit(1);
    }

    // Gate 4: MTP changes HOW tokens are produced, not WHICH tokens.
    // A different answer means the verification path is not faithful.
    let same_answer = match off {
        Some(off) => off.answer == on.answer,
        None => true,
    };
    if let Some(off) = off {
        if let (Some(a), Some(b)) = (tok_s(off), tok_s(on)) {
            println!(
                "tok/s      : off {a:.2} → on {b:.2}  ({:+.1}%)",
                (b - a) / a * 100.0
            );
        }
        println!(
            "answers    : {}",
            if same_answer {
                "identical".to_string()
            } else {
                format!("DIFFERENT\n  off: {:?}\n  on : {:?}", off.answer, on.answer)
            }
        );
    } else {
        println!("answers    : (no baseline leg to compare against)");
    }
    println!(
        "accept rate: {:?} (last {})",
        on.accept_rates,
        on.accept_rates.last().copied().unwrap_or_default()
    );

    if !same_answer {
        println!("VERDICT: FAIL — the two legs disagree; the verification path is not faithful");
        std::process::exit(1);
    }
    println!("VERDICT: OK — MTP engaged (accept rate observed on both the request and the output)");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args().any(|a| a == "--child") {
        run_child().await
    } else {
        run_parent()
    }
}
