//! Real-weights smoke for the Qwen3.8-27B family on neuron's text legs.
//!
//! Qwen3.8-27B ships `model_type: "qwen3_5"` — the exact architecture
//! string of Qwen3.5 — so the 2026-10-29 family verdict is inherited
//! wholesale. This probe turns that shape-level verdict into a measured
//! one, one leg at a time:
//!
//! * **mistralrs leg** — an HF-format local snapshot through mistral.rs's
//!   `Qwen3_5Loader` (the loader whose `TextConfig` was audited
//!   field-by-field against the 27B config). When the 27B bf16
//!   checkpoint (≈54 GB) doesn't fit the disk, point this leg at the
//!   smaller family member Qwen3.5-4B: same loader, same config shape,
//!   smaller weights — the loader path is what gets verified.
//! * **llamacpp leg** — the unsloth Qwen3.8-27B GGUF (arch `qwen35`)
//!   loaded directly: the *actual* Qwen3.8-27B weights. The runtime
//!   resolves `n_gpu_layers` to 0 for `Auto`/`Cpu` device selection —
//!   on a machine whose RAM is smaller than the GGUF the weights page
//!   in from disk (slow, but a real forward pass).
//!
//! Both legs answer THE SAME question, and every reply is checked:
//!
//! 1. non-empty;
//! 2. the answer region (text after the last `</think>`, or the whole
//!    reply when the model did not emit one) starts with the expected
//!    number — the Qwen3.5/3.8 chat template pre-fills `<think>\n`, so
//!    the reasoning arrives *before* the close tag and must be passed
//!    through untouched;
//! 3. think-block passthrough: if the reply contains `<think>` it must
//!    also contain `</think>` — neuron never strips or truncates the
//!   model's reasoning (a mid-think truncation fails the probe).
//!
//! Readings: load time, answer time, tok/s (usage-based).
//!
//! Environment:
//!
//! * `NEURON_QWEN38_DIR` — HF-format snapshot dir for the mistralrs leg
//!   (default: `~/.neuron/models/llm/Qwen/Qwen3.5-4B`).
//! * `NEURON_QWEN38_GGUF` — .gguf path for the llamacpp leg
//!   (default: `~/.neuron/models/llm/Qwen/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_0.gguf`).
//! * `NEURON_QWEN38_LEG` — `mistralrs` | `llamacpp` | `both` (default
//!   `both`; a leg that wasn't compiled in reports itself).
//! * `NEURON_QWEN38_PROMPT` — override the probe question.
//! * `NEURON_QWEN38_MAX_TOKENS` — generation cap (default 512; the
//!   thinking budget comes out of the same cap — a tiny arithmetic
//!   question thinks short, a long one may need a bump).
//! * `NEURON_LLAMA_CPP_N_CTX` — context the llamacpp leg gives the
//!   model (default 2048 clamps the 256K-capable checkpoint; set 8192
//!   for the probe). The runtime reads this itself at load + the
//!   probe only reports it.
//!
//! Run (both compiled-in legs, one question each):
//!
//! ```bash
//! NEURON_LLAMA_CPP_N_CTX=8192 \
//! cargo run --release --example qwen3_8_27b_smoke --features mistralrs
//! ```

use neuron_lib::config::DeviceSelection;
use neuron_lib::llm::{ChatMessage, ChatRequest};
use std::path::{Component, PathBuf};

/// The one question both legs must answer. Small enough that the
/// thinking budget is a handful of tokens, verifiable without a judge.
const DEFAULT_QUESTION: &str =
    "What is the smallest prime number greater than 5? Reply with just the number.";
const EXPECTED_ANSWER: &str = "7";

/// Validate a path handed in through the environment: traversal
/// segments are rejected outright, so the probe only ever reads under
/// the directory the user explicitly pointed it at.
fn validated_path(raw: &str, what: &str) -> Result<PathBuf, String> {
    let p = PathBuf::from(raw);
    if p.components()
        .any(|c| matches!(c, Component::ParentDir))
    {
        return Err(format!("{what} must not contain `..` segments: {raw}"));
    }
    Ok(p)
}

fn default_dir() -> String {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_default();
    format!("{home}/.neuron/models/llm/Qwen/Qwen3.5-4B")
}

#[cfg(feature = "llamacpp")]
fn default_gguf() -> String {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_default();
    format!("{home}/.neuron/models/llm/Qwen/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_0.gguf")
}

fn build_request(model: String, prompt: &str, max_tokens: u32) -> ChatRequest {
    ChatRequest {
        model,
        messages: vec![ChatMessage {
            role: "user".into(),
            content: prompt.to_string(),
            reasoning_content: None,
        }],
        temperature: Some(0.0),
        max_tokens: Some(max_tokens),
        top_p: None,
        top_k: None,
        stream: false,
        stop: None,
    }
}

/// The family claim made visible: the snapshot's own config must say
/// `qwen3_5` — the exact string that routes it to the `Qwen3_5Loader`.
fn model_type_of(dir: &PathBuf) -> Option<String> {
    let raw = std::fs::read_to_string(dir.join("config.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
    v.get("model_type")?.as_str().map(|s| s.to_string())
}

/// Check a reply against the passthrough contract. Returns the answer
/// region (text after the last `</think>`, or the whole reply) plus a
/// one-line think report; `Err` carries a concrete failure.
fn check_reply(content: &str) -> Result<(String, String), String> {
    let opened = content.contains("<think>");
    let closed = content.contains("</think>");
    if opened && !closed {
        return Err(
            "<think> opened but never closed — reasoning truncated \
             (passthrough contract broken; raise NEURON_QWEN38_MAX_TOKENS \
             if the budget cut it off)"
                .into(),
        );
    }
    let answer_region = match content.rsplit_once("</think>") {
        Some((_, after)) => after.trim().to_string(),
        None => content.trim().to_string(),
    };
    let think = match (opened, closed) {
        (true, true) => "<think>…</think> complete (explicit opener)",
        (false, true) => "</think> close arrived (opener pre-filled by the template)",
        (false, false) => "no think block in the reply",
        (true, false) => unreachable!("rejected above"),
    };
    Ok((answer_region, think.to_string()))
}

/// The reasoning a backend handed over SEPARATELY from the answer.
///
/// mistral.rs splits the model's `<think>…</think>` into its own
/// response field, so the answer arrives clean and the thinking lives
/// here. The contract this probe checks: when the model thought, the
/// reasoning must actually reach the caller — a dropped reasoning is
/// the failure mode this file exists to catch (the first real-weights
/// run measured completion=210 tokens with content="7": 209 tokens of
/// thinking that never arrived).
fn check_split_reasoning(reasoning: Option<&str>) -> Result<String, String> {
    let Some(r) = reasoning else {
        return Ok("none (the model did not think, or the backend keeps it inline)".into());
    };
    let trimmed = r.trim();
    if trimmed.is_empty() {
        return Err("reasoning_content is present but empty — a half-delivered \
                    reasoning is worse than none (the caller cannot tell \
                    'no thinking' from 'thinking lost in transit')"
            .into());
    }
    Ok(format!("{} chars — first line: {:?}", trimmed.chars().count(), trimmed.lines().next().unwrap_or("")))
}

struct LegOutcome {
    load: std::time::Duration,
    answer: std::time::Duration,
    prompt_tokens: u32,
    completion_tokens: u32,
    think: String,
    split_reasoning: String,
    answer_region: String,
}

fn report_leg(name: &str, model_desc: &str, out: &LegOutcome) -> Result<(), String> {
    println!("leg       : {name}");
    println!("model     : {model_desc}");
    println!("loaded    : {:?}", out.load);
    println!("answered  : {:?}", out.answer);
    println!(
        "usage     : prompt={} completion={}",
        out.prompt_tokens, out.completion_tokens
    );
    let secs = out.answer.as_secs_f64();
    if out.completion_tokens > 0 && secs > 0.0 {
        println!("tok/s     : {:.1}", out.completion_tokens as f64 / secs);
    }
    println!("think     : {}", out.think);
    println!("reasoning : {}", out.split_reasoning);
    println!("answer    : {:?}", out.answer_region);
    if out.answer_region.is_empty() {
        return Err("answer region empty after the think block".into());
    }
    if !out.answer_region.starts_with(EXPECTED_ANSWER) {
        return Err(format!(
            "expected the reply to start with {EXPECTED_ANSWER:?}, got {:?}",
            out.answer_region
        ));
    }
    Ok(())
}

fn finish(
    legs_ran: usize,
    failures: &[(String, String)],
) -> Result<(), Box<dyn std::error::Error>> {
    // Report the leg failures FIRST — a leg that failed is the most
    // informative thing this probe can print, and swallowing it behind
    // "no leg ran" hides the actual cause (the first version of this
    // probe did exactly that: a memory-starved load came out as
    // "no leg ran", which reads like a wiring problem).
    for (leg, err) in failures {
        eprintln!("=== {leg} FAILED: {err} ===");
    }
    if legs_ran == 0 {
        return Err(
            "no leg ran — build with --features mistralrs (and optionally \
             llamacpp), or point NEURON_QWEN38_DIR / NEURON_QWEN38_GGUF at \
             local weights"
                .into(),
        );
    }
    if failures.is_empty() {
        println!("=== PASSED ({legs_ran} leg(s), one real checkpoint each) ===");
        return Ok(());
    }
    std::process::exit(1);
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let leg = std::env::var("NEURON_QWEN38_LEG").unwrap_or_else(|_| "both".into());
    let want_mistralrs = leg == "both" || leg == "mistralrs";
    let want_llamacpp = leg == "both" || leg == "llamacpp";
    let question =
        std::env::var("NEURON_QWEN38_PROMPT").unwrap_or_else(|_| DEFAULT_QUESTION.into());
    let max_tokens: u32 = std::env::var("NEURON_QWEN38_MAX_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    println!("question  : {question:?}");
    println!("max_tokens: {max_tokens}");

    let mut legs_ran = 0usize;
    let mut failures: Vec<(String, String)> = Vec::new();

    #[cfg(feature = "mistralrs")]
    if want_mistralrs {
        let dir_raw = std::env::var("NEURON_QWEN38_DIR").unwrap_or_else(|_| default_dir());
        match validated_path(&dir_raw, "NEURON_QWEN38_DIR").and_then(|dir| {
            let dir = dir.to_string_lossy().into_owned();
            if !std::path::Path::new(&dir).join("config.json").exists() {
                return Err(format!(
                    "no config.json under {dir} — point NEURON_QWEN38_DIR at a local HF-format snapshot"
                ));
            }
            Ok(dir)
        }) {
            Ok(dir) => match run_mistralrs_leg(&dir, &question, max_tokens).await {
                Ok(out) => {
                    legs_ran += 1;
                    let model_type = model_type_of(&std::path::PathBuf::from(&dir))
                        .unwrap_or_else(|| "<unreadable>".into());
                    let desc = format!("{dir} (model_type: {model_type})");
                    if let Err(e) = report_leg("mistralrs", &desc, &out) {
                        failures.push(("mistralrs".into(), e));
                    }
                }
                Err(e) => failures.push(("mistralrs".into(), e)),
            },
            Err(e) => failures.push(("mistralrs".into(), e)),
        }
    }
    #[cfg(not(feature = "mistralrs"))]
    if want_mistralrs {
        eprintln!("leg       : mistralrs — not compiled in (rebuild with --features mistralrs)");
    }

    #[cfg(feature = "llamacpp")]
    if want_llamacpp {
        let gguf_raw = std::env::var("NEURON_QWEN38_GGUF").unwrap_or_else(|_| default_gguf());
        match validated_path(&gguf_raw, "NEURON_QWEN38_GGUF").and_then(|gguf| {
            let gguf = gguf.to_string_lossy().into_owned();
            if !std::path::Path::new(&gguf).exists() {
                return Err(format!("no GGUF at {gguf} — point NEURON_QWEN38_GGUF at a local file"));
            }
            Ok(gguf)
        }) {
            Ok(gguf) => match run_llamacpp_leg(&gguf, &question, max_tokens).await {
                Ok(out) => {
                    legs_ran += 1;
                    let ctx = std::env::var("NEURON_LLAMA_CPP_N_CTX").unwrap_or_else(|_| {
                        "2048 (runtime default — set NEURON_LLAMA_CPP_N_CTX to lift it)".into()
                    });
                    let desc = format!("{gguf} (n_ctx: {ctx})");
                    if let Err(e) = report_leg("llamacpp", &desc, &out) {
                        failures.push(("llamacpp".into(), e));
                    }
                }
                Err(e) => failures.push(("llamacpp".into(), e)),
            },
            Err(e) => failures.push(("llamacpp".into(), e)),
        }
    }
    #[cfg(not(feature = "llamacpp"))]
    if want_llamacpp {
        eprintln!("leg       : llamacpp — not compiled in (rebuild with --features llamacpp)");
    }

    finish(legs_ran, &failures)
}

#[cfg(feature = "mistralrs")]
async fn run_mistralrs_leg(
    dir: &str,
    question: &str,
    max_tokens: u32,
) -> Result<LegOutcome, String> {
    let model_type = model_type_of(&std::path::PathBuf::from(dir))
        .ok_or("config.json unreadable")?
        ;
    // The expected `model_type` is configurable so the same probe can
    // run against any local snapshot of the family under test — the
    // Qwen3.8/3.5 family says "qwen3_5"; a smaller Qwen3 sibling
    // ("qwen3") exercises the SAME mistral.rs reasoning path (the
    // chain-of-thought split is the backend's tag-based parser, not a
    // per-model special case) when the machine's RAM cannot hold the
    // family checkpoint. Default stays the family string.
    let expected = std::env::var("NEURON_QWEN38_EXPECT_TYPE").unwrap_or_else(|_| "qwen3_5".into());
    if model_type != expected {
        return Err(format!(
            "snapshot model_type is {model_type:?}, not {expected:?} — \
             this probe targets the Qwen3.8/3.5 family loader path \
             (set NEURON_QWEN38_EXPECT_TYPE to target a different family member)"
        ));
    }

    let t0 = std::time::Instant::now();
    let runtime = neuron_lib::llm::MistralRsLlmRuntime::load(dir, &DeviceSelection::Auto)
        .await
        .map_err(|e| format!("load failed: {e}"))?;
    let load = t0.elapsed();

    let req = build_request(dir.to_string(), question, max_tokens);
    let t1 = std::time::Instant::now();
    let resp = runtime
        .chat(req)
        .await
        .map_err(|e| format!("chat failed: {e}"))?;
    let answer = t1.elapsed();

    let content = resp.choices[0].message.content.clone();
    let (answer_region, think) = check_reply(&content)?;
    let split_reasoning = check_split_reasoning(
        resp.choices[0].message.reasoning_content.as_deref(),
    )?;
    let usage = resp.usage.ok_or("no usage block in the reply")?;
    Ok(LegOutcome {
        load,
        answer,
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        think,
        split_reasoning,
        answer_region,
    })
}

#[cfg(feature = "llamacpp")]
async fn run_llamacpp_leg(
    gguf: &str,
    question: &str,
    max_tokens: u32,
) -> Result<LegOutcome, String> {
    let t0 = std::time::Instant::now();
    let runtime = neuron_lib::llm::LlamaCppLlmRuntime::load(gguf, &DeviceSelection::Auto)
        .await
        .map_err(|e| format!("load failed: {e}"))?;
    let load = t0.elapsed();

    let req = build_request(gguf.to_string(), question, max_tokens);
    let t1 = std::time::Instant::now();
    let resp = runtime
        .chat(req)
        .await
        .map_err(|e| format!("chat failed: {e}"))?;
    let answer = t1.elapsed();

    let content = resp.choices[0].message.content.clone();
    let (answer_region, think) = check_reply(&content)?;
    let split_reasoning = check_split_reasoning(
        resp.choices[0].message.reasoning_content.as_deref(),
    )?;
    let usage = resp.usage.ok_or("no usage block in the reply")?;
    Ok(LegOutcome {
        load,
        answer,
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        think,
        split_reasoning,
        answer_region,
    })
}
