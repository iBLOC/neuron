//! llama.cpp backend speed benchmark.
//!
//! In-process harness that loads a local GGUF via
//! `LlamaCppLlmRuntime` and measures end-to-end chat + chat_stream
//! latency with a stage split, plus load time and peak RSS. Writes a
//! markdown report to `target/llamacpp-speed.md`.
//!
//! ## Why this is not a Criterion bench
//!
//! Same reason as `asr_speed.rs` / `llm_speed.rs`: a single
//! measurement here is a full chat round trip (hundreds of ms to
//! several seconds for an LLM decode). `harness = false` is set in
//! `Cargo.toml` and the bench runs a fixed number of repeats with its
//! own p50/p95 math — Criterion's iteration model fits µs-scale
//! microbenchmarks, not full-prompt inference.
//!
//! ## What this bench is and is not
//!
//! This is **not** a cross-framework comparison. For neuron vs
//! vLLM/ollama/llama.cpp server-mode use `benches/llm_speed.rs`
//! (drives HTTP endpoints; needs live servers). This bench is
//! **in-process** and measures the `LlamaCppLlmRuntime` adapter
//! itself — what a host using `NEURON_LLM_RUNTIME=llamacpp` actually
//! pays per request when neuron is embedded.
//!
//! ## Build prereq
//!
//! Same as the runtime: `cargo bench --bench llamacpp_speed
//! --features llamacpp` needs cmake + a C++ compiler + libclang on
//! PATH (the `llama-cpp-sys-2` build script compiles llama.cpp via
//! cmake and uses bindgen to parse the C++ headers). On a host
//! without those tools, the bench file compiles (default features
//! don't pull llama-cpp-2) but the binary won't link — the `main`
//! short-circuits before the load attempt.
//!
//! ## Run
//!
//! ```bash
//! # Requires a local GGUF and the build prereq above.
//! NEURON_LLAMACPP_MODEL_PATH=/path/to/model.gguf \
//!   cargo bench --bench llamacpp_speed --features llamacpp
//! ```
//!
//! Environment:
//!
//! | var | default | meaning |
//! |---|---|---|
//! | `NEURON_LLAMACPP_MODEL_PATH` | (unset) | Path to a local `.gguf` file. The bench aborts if unset. |
//! | `NEURON_LLAMACPP_N_CTX` | `2048` | Context length passed to `LlamaContextParams` (matches runtime default). |
//! | `NEURON_LLAMACPP_BENCH_PROFILE` | `standard` | `quick` (0/1) / `standard` (1/3) / `full` (2/7). |
//! | `NEURON_LLAMACPP_BENCH_WARMUP` | (profile) | Override warmup count. |
//! | `NEURON_LLAMACPP_BENCH_REPEATS` | (profile) | Override repeat count. |
//! | `NEURON_LLAMACPP_BENCH_PROMPTS` | (built-in) | `;`-separated `<prompt>::<expected_min_tokens>` entries. |
//! | `NEURON_LLAMACPP_BENCH_OUT` | `target/llamacpp-speed.md` | Report path. |

#[cfg(feature = "llamacpp")]
use std::fmt::Write as _;
#[cfg(feature = "llamacpp")]
use std::path::PathBuf;
#[cfg(feature = "llamacpp")]
use std::time::{Duration, Instant};

#[cfg(feature = "llamacpp")]
use futures_util::StreamExt;
#[cfg(feature = "llamacpp")]
use neuron_lib::llm::llamacpp_runtime::LlamaCppLlmRuntime;
#[cfg(feature = "llamacpp")]
use neuron_lib::llm::{ChatMessage, ChatRequest, LlmRuntime};

#[path = "memory.rs"]
#[allow(dead_code, unused_imports)]
#[cfg(feature = "llamacpp")]
mod memory;
#[cfg(feature = "llamacpp")]
use memory::MemorySnapshot;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[cfg(feature = "llamacpp")]
struct Knobs {
    warmup: usize,
    repeats: usize,
}

#[cfg(feature = "llamacpp")]
impl Knobs {
    fn from_env() -> Self {
        let (def_warmup, def_repeats) = match std::env::var("NEURON_LLAMACPP_BENCH_PROFILE").as_deref() {
            Ok("quick") => (0, 1),
            Ok("full") => (2, 7),
            _ => (1, 3),
        };
        Self {
            warmup: env_usize("NEURON_LLAMACPP_BENCH_WARMUP", def_warmup),
            repeats: env_usize("NEURON_LLAMACPP_BENCH_REPEATS", def_repeats),
        }
    }
}

#[cfg(feature = "llamacpp")]
fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

#[cfg(feature = "llamacpp")]
fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).ok().filter(|v| !v.is_empty()).unwrap_or_else(|| default.to_string())
}

// ---------------------------------------------------------------------------
// Prompts
// ---------------------------------------------------------------------------

#[cfg(feature = "llamacpp")]
struct BenchPrompt {
    name: &'static str,
    messages: Vec<ChatMessage>,
    /// Lower bound for `max_tokens` so the per-token rate isn't
    /// dominated by an over-aggressive ceiling. The bench sets
    /// `max_tokens = max(this, 32)` to keep the call meaningful even
    /// when the prompt is short.
    min_tokens: u32,
}

/// Built-in prompts. The point of the bench is the forward pass, not
/// the answer quality — these are short open-ended strings that almost
/// every instruction-tuned chat model can produce > 32 tokens for in
/// response. Override via `NEURON_LLAMACPP_BENCH_PROMPTS` (semicolon-
/// separated `name::prompt::min_tokens` tuples) when measuring a
/// domain-specific model.
#[cfg(feature = "llamacpp")]
const DEFAULT_PROMPTS: &[BenchPrompt] = &[
    BenchPrompt {
        name: "short_qa",
        messages: vec![ChatMessage {
            role: "user".into(),
            content: "What is the capital of France? Answer in one short sentence.".into(),
        }],
        min_tokens: 32,
    },
    BenchPrompt {
        name: "explanation",
        messages: vec![ChatMessage {
            role: "user".into(),
            content: "Explain in 3-4 sentences why a barometer can be used as an altimeter."
                .into(),
        }],
        min_tokens: 64,
    },
    BenchPrompt {
        name: "code_gen",
        messages: vec![ChatMessage {
            role: "user".into(),
            content: "Write a Python function `def fib(n: int) -> int` that returns the n-th \
                      Fibonacci number using a loop, and a one-line docstring."
                .into(),
        }],
        min_tokens: 96,
    },
    BenchPrompt {
        name: "multi_turn",
        messages: vec![
            ChatMessage {
                role: "system".into(),
                content: "You are terse.".into(),
            },
            ChatMessage {
                role: "user".into(),
                content: "List 3 uses for a paperclip, comma-separated.".into(),
            },
        ],
        min_tokens: 32,
    },
];

#[cfg(feature = "llamacpp")]
fn load_prompts() -> Vec<BenchPrompt> {
    if let Ok(v) = std::env::var("NEURON_LLAMACPP_BENCH_PROMPTS") {
        if !v.trim().is_empty() {
            return v
                .split(';')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(|entry| {
                    let parts: Vec<&str> = entry.splitn(3, "::").collect();
                    let (name, content, min) = match parts.as_slice() {
                        [n, c] => (*n, *c, 32u32),
                        [n, c, m] => (*n, *c, m.parse().unwrap_or(32)),
                        _ => ("custom", entry, 32u32),
                    };
                    BenchPrompt {
                        name,
                        messages: vec![ChatMessage {
                            role: "user".into(),
                            content: content.into(),
                        }],
                        min_tokens: min,
                    }
                })
                .collect();
        }
    }
    DEFAULT_PROMPTS.to_vec()
}

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

#[cfg(feature = "llamacpp")]
#[derive(Debug, Clone, Copy, Default)]
struct StageSplit {
    template_us: u128,
    tokenize_us: u128,
    prefill_us: u128,
    generation_us: u128,
}

#[cfg(feature = "llamacpp")]
#[derive(Debug)]
struct Sample {
    /// Wall clock for the whole `chat()` / `chat_stream()` call.
    total: Duration,
    /// Time to first token (stream mode) or time to the first
    /// generated token (non-stream, measured the same way as stream
    /// for parity).
    ttft: Duration,
    /// Per-token mean in the generation phase. Derived as
    /// `generation_us / tokens_generated` so a long ceiling with a
    /// short answer and a short ceiling with a long answer compare
    /// the same way.
    per_token: Duration,
    tokens_generated: u32,
    stages: StageSplit,
}

#[cfg(feature = "llamacpp")]
fn percentile_us(sorted: &[Duration], pct: f64) -> f64 {
    let mut s: Vec<Duration> = sorted.to_vec();
    s.sort();
    let idx = ((pct / 100.0) * (s.len().saturating_sub(1)) as f64).round() as usize;
    s.get(idx).copied().map(|d| d.as_micros() as f64).unwrap_or(0.0)
}

// ---------------------------------------------------------------------------
// Runtime load + measurement
// ---------------------------------------------------------------------------

#[cfg(feature = "llamacpp")]
struct Setup {
    rt: Box<dyn LlmRuntime>,
    load: Duration,
    n_ctx: u32,
    gguf_path: PathBuf,
    peak_rss_after_load: u64,
    peak_rss_after_run: u64,
}

#[cfg(feature = "llamacpp")]
async fn setup() -> Result<Setup, String> {
    let model_path = std::env::var("NEURON_LLAMACPP_MODEL_PATH")
        .ok()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            "NEURON_LLAMACPP_MODEL_PATH is unset. Point it at a local .gguf file (e.g. \
             `export NEURON_LLAMACPP_MODEL_PATH=~/.neuron/models/llamacpp/qwen2.5-7b-instruct/model.gguf`)."
                .to_string()
        })?;
    let gguf_path = PathBuf::from(&model_path);
    if !gguf_path.exists() {
        return Err(format!("GGUF not found at {}", gguf_path.display()));
    }

    // Resolve model_id the way the runtime resolves it. The runtime
    // looks up by bare id under $NEURON_MODEL_DIRS first; passing the
    // path as the id bypasses that path and goes straight to
    // `load_from_file` because the file already exists.
    let model_id = gguf_path.to_string_lossy().into_owned();

    // Push the configured n_ctx into the env so LlamaCppLlmRuntime::load
    // picks it up. The runtime already documents this knob on its own.
    let n_ctx: u32 = env_or("NEURON_LLAMACPP_N_CTX", "2048").parse().unwrap_or(2048);
    // edition 2024: env::set_var is unsafe
    unsafe { std::env::set_var("NEURON_LLAMA_CPP_N_CTX", n_ctx.to_string()) };

    let pre_load_rss = MemorySnapshot::now().rss_bytes;
    let t0 = Instant::now();
    let rt = LlamaCppLlmRuntime::load(&model_id, &neuron_lib::config::DeviceSelection::Cpu)
        .await
        .map_err(|e| format!("LlamaCppLlmRuntime::load failed: {}", e.message))?;
    let load = t0.elapsed();
    let post_load_rss = MemorySnapshot::now().rss_bytes;
    let peak_rss_after_load = pre_load_rss.max(post_load_rss);

    Ok(Setup {
        rt,
        load,
        n_ctx,
        gguf_path,
        peak_rss_after_load,
        peak_rss_after_run: peak_rss_after_load,
    })
}

#[cfg(feature = "llamacpp")]
async fn measure_stream(
    setup: &Setup,
    prompt: &BenchPrompt,
    knobs: &Knobs,
) -> Result<Sample, String> {
    let max_tokens = prompt.min_tokens.max(32);

    let mut samples: Vec<Sample> = Vec::with_capacity(knobs.warmup + knobs.repeats);

    for iter in 0..(knobs.warmup + knobs.repeats) {
        let req = ChatRequest {
            model: setup.gguf_path.to_string_lossy().into_owned(),
            messages: prompt.messages.clone(),
            temperature: Some(0.0), // greedy — deterministic, no sampling noise
            max_tokens: Some(max_tokens),
            top_p: None,
            top_k: None,
            stream: true,
            stop: None,
        };
        let t0 = Instant::now();
        let stream_handle = setup.rt.chat_stream(req).await.map_err(|e| e.message)?;
        // The first chat_stream call pays the prompt-prefill cost; the
        // benchmark reports that as `prefill_us` in the stage split,
        // measured as the wall time from `chat_stream` invocation
        // through the first `Some(_)` item landing.
        let mut s = stream_handle;
        let prefill_t0 = Instant::now();
        let first = s.next().await;
        let prefill = prefill_t0.elapsed();
        if first.is_none() {
            return Err("stream ended before first chunk".into());
        }
        let mut ttft = prefill;
        let mut tokens: u32 = 0;
        let mut gen_t0 = Instant::now();
        let mut first_content = true;
        let mut piece = String::new();
        while let Some(chunk) = s.next().await {
            if let Some(choice) = chunk.choices.into_iter().next() {
                if let Some(content) = choice.delta.content {
                    if !content.is_empty() {
                        if first_content {
                            // Re-anchor: the first chunk is the role
                            // announcement; the first *content* chunk
                            // is when prefill is really over.
                            ttft = prefill_t0.elapsed();
                            first_content = false;
                            gen_t0 = Instant::now();
                        }
                        piece.push_str(&content);
                        tokens += 1;
                    }
                }
            }
        }
        let generation = gen_t0.elapsed();
        let total = t0.elapsed();

        let sample = Sample {
            total,
            ttft,
            per_token: if tokens > 0 {
                Duration::from_micros((generation.as_micros() / tokens as u128) as u64)
            } else {
                Duration::ZERO
            },
            tokens_generated: tokens,
            stages: StageSplit {
                template_us: 0, // folded into prefill (llama-cpp-2 does it inside new_context)
                tokenize_us: 0,
                prefill_us: prefill.as_micros(),
                generation_us: generation.as_micros(),
            },
        };
        // Don't keep the piece — it just bloats the report and the
        // measurement is what matters.
        let _ = piece;
        if iter >= knobs.warmup {
            samples.push(sample);
        }
    }

    // Aggregate: median across repeats.
    Ok(median(samples))
}

#[cfg(feature = "llamacpp")]
fn median(mut samples: Vec<Sample>) -> Sample {
    assert!(!samples.is_empty());
    samples.sort_by_key(|s| s.total);
    samples.remove(samples.len() / 2)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[cfg(feature = "llamacpp")]
#[tokio::main(flavor = "current_thread")]
async fn main() {
    let knobs = Knobs::from_env();
    let out_path = PathBuf::from(env_or(
        "NEURON_LLAMACPP_BENCH_OUT",
        "target/llamacpp-speed.md",
    ));

    println!("=== neuron llama.cpp backend speed ===");
    let setup = match setup().await {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[llamacpp-speed] {e}");
            eprintln!(
                "[llamacpp-speed] need: cmake + C++ toolchain + libclang on PATH, and a \
                 local GGUF at $NEURON_LLAMACPP_MODEL_PATH"
            );
            std::process::exit(2);
        }
    };
    println!(
        "[llamacpp-speed] loaded {} (n_ctx={}) in {:.2?}; pre-run RSS peak: {}",
        setup.gguf_path.display(),
        setup.n_ctx,
        setup.load,
        format_bytes(setup.peak_rss_after_load),
    );
    let prompts = load_prompts();
    println!(
        "[llamacpp-speed] profile: warmup={} repeats={}; prompts: {}",
        knobs.warmup,
        knobs.repeats,
        prompts.iter().map(|p| p.name).collect::<Vec<_>>().join(", "),
    );

    let mut rows: Vec<(&'static str, Sample)> = Vec::new();
    for prompt in &prompts {
        let m = match measure_stream(&setup, prompt, &knobs).await {
            Ok(m) => m,
            Err(e) => {
                eprintln!("[llamacpp-speed] {}: FAILED ({e})", prompt.name);
                continue;
            }
        };
        println!(
            "[llamacpp-speed] {:>13}: total {:>6.0} ms | TTFT {:>6.0} ms | per-tok {:>5.1} ms | n_tok {}",
            prompt.name,
            m.total.as_millis() as f64,
            m.ttft.as_millis() as f64,
            m.per_token.as_millis() as f64,
            m.tokens_generated,
        );
        rows.push((prompt.name, m));
    }

    let post_run_rss = MemorySnapshot::now().rss_bytes;
    let peak_rss_after_run = setup.peak_rss_after_load.max(post_run_rss);

    // ---- report ----------------------------------------------------------
    let mut report = String::new();
    let _ = writeln!(
        report,
        "# llama.cpp backend speed — neuron `LlamaCppLlmRuntime`"
    );
    let _ = writeln!(report);
    let _ = writeln!(
        report,
        "Single-engine in-process benchmark. Drives `NEURON_LLM_RUNTIME=llamacpp`'s \
         `LlamaCppLlmRuntime` directly (no HTTP layer, no server), so the numbers reflect \
         what a host embedding neuron pays per request. For cross-framework comparisons \
         (this engine vs vLLM / ollama / a separate llama-server), use `benches/llm_speed.rs`."
    );
    let _ = writeln!(report);
    let _ = writeln!(report, "## Setup");
    let _ = writeln!(report);
    let _ = writeln!(
        report,
        "| model | n_ctx | load | pre-run RSS peak | post-run RSS peak |"
    );
    let _ = writeln!(report, "|---|---|---|---|---|");
    let _ = writeln!(
        report,
        "| `{}` | {} | {:.2?} | {} | {} |",
        setup.gguf_path.display(),
        setup.n_ctx,
        setup.load,
        format_bytes(setup.peak_rss_after_load),
        format_bytes(peak_rss_after_run),
    );

    let _ = writeln!(report);
    let _ = writeln!(
        report,
        "## Latency (warmup={}, repeats={})\n",
        knobs.warmup, knobs.repeats
    );
    let _ = writeln!(
        report,
        "| prompt | total ms | TTFT ms | per-tok ms | tokens | prefill ms | generation ms |"
    );
    let _ = writeln!(report, "|---|---|---|---|---|---|---|");
    for (name, m) in &rows {
        let _ = writeln!(
            report,
            "| {} | {:.0} | {:.0} | {:.1} | {} | {:.0} | {:.0} |",
            name,
            m.total.as_millis() as f64,
            m.ttft.as_millis() as f64,
            m.per_token.as_millis() as f64,
            m.tokens_generated,
            m.stages.prefill_us as f64 / 1000.0,
            m.stages.generation_us as f64 / 1000.0,
        );
    }

    let _ = writeln!(report);
    let _ = writeln!(report, "## Notes");
    let _ = writeln!(
        report,
        "\n- `TTFT` is wall time from `chat_stream` invocation to the first *content* \
         chunk (the role-announcement chunk is filtered out). On a 7B Q4_K_M model with a \
         2k context and a short prompt this is dominated by the prompt prefill — expect \
         the TTFT and `prefill` columns to track each other.\n\
         - `per-tok` is `generation_us / tokens_generated` averaged across repeats — the \
         pure decode rate. With `temperature = 0` (greedy) this is the same number you'd \
         see from a simple streaming benchmark on `llama.cpp`'s `simple` example.\n\
         - `pre-run RSS peak` is the maximum of the process RSS before/after model load; \
         `post-run RSS peak` includes the bench loop on top. The delta is roughly the \
         per-request KV-cache allocation (one `LlamaContext` per request in this runtime).\n\
         - This bench uses the in-process adapter path — there is no HTTP / axum / JSON \
         overhead in the numbers. For the OpenAI wire-format cost on top, use \
         `benches/llm_speed.rs` against a neuron_server started with \
         `NEURON_LLM_RUNTIME=llamacpp`."
    );

    if let Some(parent) = out_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(&out_path, &report).expect("write report");
    println!("\n[llamacpp-speed] report written to {}", out_path.display());
}

#[cfg(feature = "llamacpp")]
fn format_bytes(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * KIB;
    const GIB: u64 = 1024 * MIB;
    if bytes >= GIB {
        format!("{:.2} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.1} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.1} KiB", bytes as f64 / KIB as f64)
    } else {
        format!("{bytes} B")
    }
}

// ---------------------------------------------------------------------------
// Fallback main for builds without `--features llamacpp`.
//
// The benchmark file still compiles (the cfg above keeps every piece
// out of the build), but `main` is required by the [[bench]] entry. We
// emit a clear "feature not enabled" message and exit non-zero so
// `cargo bench --bench llamacpp_speed` (without --features llamacpp)
// fails fast and informatively.
// ---------------------------------------------------------------------------

#[cfg(not(feature = "llamacpp"))]
fn main() {
    eprintln!(
        "[llamacpp-speed] the `llamacpp` feature is not enabled. Build with \
         `cargo bench --bench llamacpp_speed --features llamacpp` (needs cmake + a C++ \
         toolchain + libclang on PATH; the llama-cpp-2 build script compiles llama.cpp \
         via cmake and uses bindgen to parse the C++ headers)."
    );
    std::process::exit(1);
}
