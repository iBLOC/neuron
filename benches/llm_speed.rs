//! Cross-framework LLM speed benchmark: neuron vs vLLM / ollama / llama.cpp.
//!
//! Unlike `benches/perf.rs` (microbenchmarks of data-structure ops) and
//! `benches/compare.rs` (a scaffold whose external backends are all
//! noop stubs), this harness measures **real model inference over the
//! OpenAI HTTP wire format** and can therefore line neuron up against
//! any other OpenAI-compatible server on the same machine.
//!
//! It deliberately starts nothing: it is a pure HTTP client driving
//! whatever endpoints `NEURON_BENCH_ENDPOINTS` names. The driver script
//! (`scripts/run-llm-speed-compare.sh`) is responsible for starting and
//! stopping the servers, which keeps this binary usable against a
//! server you already have running, or a remote one.
//!
//! ## Why this is not a Criterion bench
//!
//! Criterion's model assumes a tight loop of cheap iterations. A single
//! measurement here is a multi-second HTTP round trip against a
//! multi-gigabyte model, so we run a small fixed number of repeats and
//! compute p50/p95 ourselves. `harness = false` is set in `Cargo.toml`.
//!
//! ## What is measured
//!
//! Per (framework x prompt size), on two request shapes:
//!
//! * **non-stream** — wall latency plus the server's own
//!   `usage.prompt_tokens` / `usage.completion_tokens`.
//! * **stream** — time to first token (TTFT) and the decode window.
//!
//! From those: `prefill tok/s = prompt_tokens / TTFT` and
//! `decode tok/s = (streamed_tokens - 1) / decode_window`. Decode
//! throughput is the headline number because it is the least sensitive
//! to the prompt-template differences called out in the report
//! appendix.
//!
//! ## Client-side timing
//!
//! Every timestamp is taken by this client. neuron's streaming path
//! stamps `created` at second resolution and drops the terminal
//! finish/usage chunk, so there is no server-side timing to trust.
//! Doing it client-side is also what a real user experiences, and it
//! makes the four frameworks comparable without patching any of them.
//!
//! ## Run
//!
//! ```bash
//! # driver script starts the servers first, then:
//! cargo bench --bench llm_speed
//!
//! # or point at servers you started yourself:
//! NEURON_BENCH_ENDPOINTS="neuron=http://127.0.0.1:18083,ollama=http://127.0.0.1:11434" \
//! NEURON_BENCH_MODELS="neuron=minicpm5-2b,ollama=minicpm5-2b" \
//! cargo bench --bench llm_speed
//! ```
//!
//! Full environment contract: set `NEURON_BENCH_HELP=1`.

use std::fmt::Write as _;
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use serde_json::{json, Value};

// `memory.rs` is shared with `compare.rs` / `memory_bench.rs`, so it
// exposes more surface than this target uses (MemoryTrace, now()).
#[path = "memory.rs"]
#[allow(dead_code, unused_imports)]
mod memory;
use memory::MemorySnapshot;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Tuning knobs. Every field can be overridden by an individual env
/// var; the profile only sets defaults so that a quick smoke run and
/// a full measurement run share one code path.
#[derive(Clone, Debug)]
struct Knobs {
    warmup: usize,
    repeats: usize,
    max_tokens: u32,
    prompt_words: Vec<usize>,
    concurrency: Vec<usize>,
}

impl Knobs {
    fn from_env() -> Self {
        let (def_warmup, def_repeats, def_max_tokens, def_words, def_conc) =
            match env_str("NEURON_BENCH_PROFILE").as_deref() {
                Some("quick") => (1, 1, 8, vec![64], vec![1]),
                Some("full") => (2, 5, 128, vec![96, 768, 3072], vec![1, 4, 8]),
                // "standard" and everything unrecognised.
                _ => (1, 3, 32, vec![64, 512], vec![1, 4]),
            };

        Self {
            warmup: env_usize("NEURON_BENCH_WARMUP", def_warmup),
            repeats: env_usize("NEURON_BENCH_REPEATS", def_repeats),
            max_tokens: env_u32("NEURON_BENCH_MAX_TOKENS", def_max_tokens),
            prompt_words: env_usize_list("NEURON_BENCH_PROMPT_WORDS")
                .unwrap_or(def_words),
            concurrency: env_usize_list("NEURON_BENCH_CONCURRENCY").unwrap_or(def_conc),
        }
    }
}

#[derive(Clone, Debug)]
struct Endpoint {
    name: String,
    base: String,
    model: String,
    /// Free-form label rendered in the report ("cpu", "cuda:0", ...).
    /// Set via `NEURON_BENCH_DEVICE_<NAME>`; never inferred, because a
    /// wrong guess here would make the comparison actively misleading.
    device: String,
    /// Weight storage format label ("f32 safetensors", "gguf q8_0", ...).
    /// Set via `NEURON_BENCH_PRECISION_<NAME>`.
    precision: String,
    /// PID whose RSS we sample. Set via `NEURON_BENCH_PID_<NAME>`.
    pid: Option<u32>,
}

/// One measured request. All timings are seconds as f64 so they can be
/// averaged and compared without unit churn.
#[derive(Clone, Debug, Default)]
struct Sample {
    ttft: Option<f64>,
    decode_window: Option<f64>,
    stream_tokens: usize,
    e2e: f64,
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    finish_reason: Option<String>,
}

impl Sample {
    /// Tokens per second during steady-state decode.
    fn decode_tps(&self) -> Option<f64> {
        if let Some(window) = self.decode_window {
            if window > 0.0 && self.stream_tokens >= 2 {
                return Some((self.stream_tokens - 1) as f64 / window);
            }
        }
        // Fall back to the non-stream numbers. We don't know the
        // prefill portion inside `e2e`, so we treat the whole response
        // as steady-state decode — an underestimate, but a usable
        // upper bound on the actual decode rate.
        if let (Some(c), e) = (self.completion_tokens, self.e2e) {
            if e > 0.0 && c > 0 {
                return Some(c as f64 / e);
            }
        }
        None
    }

    /// Tokens per second while processing the prompt.
    fn prefill_tps(&self) -> Option<f64> {
        let ttft = self.ttft?;
        if ttft <= 0.0 {
            return None;
        }
        Some(self.prompt_tokens.unwrap_or(0) as f64 / ttft)
    }
}

// ---------------------------------------------------------------------------
// Small env helpers
// ---------------------------------------------------------------------------

fn env_str(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|s| !s.trim().is_empty())
}

fn env_usize(key: &str, default: usize) -> usize {
    env_str(key)
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(default)
}

fn env_u32(key: &str, default: u32) -> u32 {
    env_str(key)
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(default)
}

fn env_usize_list(key: &str) -> Option<Vec<usize>> {
    env_str(key).map(|s| {
        s.split(',')
            .filter_map(|p| p.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
            .collect()
    })
}

/// Parse `"a=1,b=2"` into a pair list. Used for endpoints, model ids
/// and the per-framework report labels.
fn parse_pairs(raw: &str) -> Vec<(String, String)> {
    raw.split(',')
        .filter_map(|item| {
            let (k, v) = item.split_once('=')?;
            let k = k.trim().to_string();
            let v = v.trim().to_string();
            if k.is_empty() || v.is_empty() {
                None
            } else {
                Some((k, v))
            }
        })
        .collect()
}

fn lookup(pairs: &[(String, String)], key: &str, default: &str) -> String {
    pairs
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(key))
        .map(|(_, v)| v.clone())
        .unwrap_or_else(|| default.to_string())
}

/// Report labels are keyed by a `NAME` suffix in the env var, e.g.
/// `NEURON_BENCH_DEVICE_OLLAMA=cuda:0`. Framework names can contain
/// dots (`llama.cpp`), which are not valid in an env var name, so the
/// suffix is the name uppercased with non-alphanumerics stripped.
fn env_key_suffix(name: &str) -> String {
    name.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect::<String>()
        .to_uppercase()
}

// ---------------------------------------------------------------------------
// Prompt construction
// ---------------------------------------------------------------------------

/// Per-request nonce salt. A hashed counter rather than the raw number so
/// consecutive tags share no leading characters, and long enough that the
/// opening tokens of every request differ. Weak nonces ("rep 0" vs
/// "rep 1") still let llama.cpp's slot cache answer 91 of 92 prompt
/// tokens; with a full-width salt it re-prefills every request, which is
/// what a server with no cache (neuron) does unconditionally.
fn next_salt() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{:08x}", n.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32)
}

/// Deterministic filler text. Real words rather than a repeated token
/// so tokenizers of every flavour see something close to natural
/// language; the exact count is not important because each server
/// reports its own `prompt_tokens` and the table shows that.
fn filler_prompt(words: usize) -> String {
    const CORPUS: &[&str] = &[
        "the", "system", "processes", "requests", "through", "a", "network",
        "layer", "which", "routes", "each", "call", "to", "the", "model",
        "runtime", "responsible", "for", "turning", "a", "prompt", "into",
        "tokens", "and", "then", "into", "text", "again", "a", "cache",
        "holds", "the", "attention", "state", "so", "that", "each", "new",
        "token", "costs", "roughly", "the", "same", "no", "matter", "how",
        "long", "the", "conversation", "has", "already", "grown", "the",
        "benchmark", "runs", "the", "same", "work", "on", "every", "engine",
        "so", "that", "the", "numbers", "can", "be", "placed", "side", "by",
        "side", "without", "worrying", "about", "different", "inputs",
    ];
    let mut out = String::with_capacity(words * 6);
    for i in 0..words {
        if i > 0 {
            out.push(' ');
        }
        out.push_str(CORPUS[i % CORPUS.len()]);
    }
    out
}

// ---------------------------------------------------------------------------
// HTTP: one non-stream request
// ---------------------------------------------------------------------------

/// reqwest hides the actual cause under `source()`; the top-level `Display`
/// alone says only "error sending request", which cannot distinguish a
/// refused connection from a reset one or a timeout.
fn err_chain(e: &reqwest::Error) -> String {
    let mut out = e.to_string();
    let mut src = std::error::Error::source(e);
    while let Some(s) = src {
        out.push_str(&format!(" <- {s}"));
        src = s.source();
    }
    out
}

/// Send one chat request, retrying once on a transport-level failure.
///
/// Nothing here can turn a transport error into an HTTP status to react to,
/// and a server dropping a socket between two requests is a property of the
/// host, not of the framework's speed — one retry separates the two.
async fn send_chat(
    client: &reqwest::Client,
    url: &str,
    body: &Value,
) -> Result<reqwest::Response, String> {
    let mut last = String::new();
    for attempt in 0..2 {
        match client.post(url).json(body).send().await {
            Ok(r) => return Ok(r),
            Err(e) => last = format!("send: {}", err_chain(&e)),
        }
        if attempt == 0 {
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    }
    Err(last)
}

async fn chat_nonstream(
    client: &reqwest::Client,
    ep: &Endpoint,
    prompt: &str,
    max_tokens: u32,
) -> Result<Sample, String> {
    let base = ep.base.trim_end_matches('/');
    let url = if base.ends_with("/v1") {
        format!("{base}/chat/completions")
    } else {
        format!("{base}/v1/chat/completions")
    };
    let body = json!({
        "model": ep.model,
        "messages": [{ "role": "user", "content": prompt }],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": false,
    });

    let start = Instant::now();
    let resp = send_chat(client, &url, &body).await?;
    let status = resp.status();
    let text = resp
        .text()
        .await
        .map_err(|e| format!("read body: {e}"))?;
    let e2e = start.elapsed().as_secs_f64();

    if !status.is_success() {
        return Err(format!("HTTP {status}: {}", truncate(&text, 300)));
    }
    let v: Value = serde_json::from_str(&text).map_err(|e| format!("bad JSON: {e}"))?;

    let mut s = Sample {
        e2e,
        ..Default::default()
    };
    if let Some(u) = v.get("usage") {
        s.prompt_tokens = u.get("prompt_tokens").and_then(Value::as_u64).map(|x| x as u32);
        s.completion_tokens = u
            .get("completion_tokens")
            .and_then(Value::as_u64)
            .map(|x| x as u32);
    }
    s.finish_reason = v
        .pointer("/choices/0/finish_reason")
        .and_then(Value::as_str)
        .map(str::to_string);
    // Some servers omit `usage`; fall back to counting nothing rather
    // than inventing a number.
    Ok(s)
}

// ---------------------------------------------------------------------------
// HTTP: one streaming request
// ---------------------------------------------------------------------------

async fn chat_stream(
    client: &reqwest::Client,
    ep: &Endpoint,
    prompt: &str,
    max_tokens: u32,
) -> Result<Sample, String> {
    let base = ep.base.trim_end_matches('/');
    let url = if base.ends_with("/v1") {
        format!("{base}/chat/completions")
    } else {
        format!("{base}/v1/chat/completions")
    };
    let body = json!({
        "model": ep.model,
        "messages": [{ "role": "user", "content": prompt }],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": true,
    });

    let start = Instant::now();
    let resp = send_chat(client, &url, &body).await?;
    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("HTTP {status}: {}", truncate(&text, 300)));
    }

    let mut s = Sample {
        e2e: 0.0,
        ..Default::default()
    };
    let mut first_content: Option<f64> = None;
    let mut last_content: Option<f64> = None;
    let mut stream_tokens = 0usize;
    let mut pending: Vec<u8> = Vec::new();

    let mut stream = resp.bytes_stream();
    let overall_deadline = std::time::Duration::from_secs(60);
    while let Some(item) = match tokio::time::timeout(overall_deadline, stream.next()).await {
        Ok(opt) => opt,
        Err(_) => return Err(format!("stream stalled > {overall_deadline:?}")),
    } {
        let arrived = start.elapsed().as_secs_f64();
        let chunk = item.map_err(|e| format!("stream read: {e}"))?;
        pending.extend_from_slice(&chunk);

        // SSE events are separated by a blank line. Drain every
        // complete one; leave any partial tail for the next chunk.
        while let Some(pos) = find_blank_line(&pending) {
            let event: Vec<u8> = pending.drain(..pos).collect();
            let event = String::from_utf8_lossy(&event);
            for payload in data_payloads(&event) {
                if payload == "[DONE]" {
                    continue;
                }
                let Ok(v) = serde_json::from_str::<Value>(&payload) else {
                    continue;
                };
                // A content-bearing delta is one streamed token. Models with a
                // hybrid thinking mode put their thought trace in one of three
                // fields depending on the server: `reasoning_content` (vLLM,
                // neuron), `reasoning` (ollama), or back in `content` (llama.cpp
                // — which keeps thinking disabled by default). All three count.
                let content = v.pointer("/choices/0/delta/content").and_then(Value::as_str);
                let reasoning_content = v
                    .pointer("/choices/0/delta/reasoning_content")
                    .and_then(Value::as_str);
                let reasoning = v
                    .pointer("/choices/0/delta/reasoning")
                    .and_then(Value::as_str);
                let has_token = matches!(content, Some(t) if !t.is_empty())
                    || matches!(reasoning_content, Some(t) if !t.is_empty())
                    || matches!(reasoning, Some(t) if !t.is_empty());
                if has_token {
                    stream_tokens += 1;
                    first_content.get_or_insert(arrived);
                    last_content = Some(arrived);
                }
                if s.finish_reason.is_none() {
                    s.finish_reason = v
                        .pointer("/choices/0/finish_reason")
                        .and_then(Value::as_str)
                        .map(str::to_string);
                }
                if let Some(u) = v.get("usage") {
                    if s.prompt_tokens.is_none() {
                        s.prompt_tokens =
                            u.get("prompt_tokens").and_then(Value::as_u64).map(|x| x as u32);
                    }
                    if s.completion_tokens.is_none() {
                        s.completion_tokens = u
                            .get("completion_tokens")
                            .and_then(Value::as_u64)
                            .map(|x| x as u32);
                    }
                }
            }
        }
    }

    s.e2e = start.elapsed().as_secs_f64();
    s.stream_tokens = stream_tokens;
    s.ttft = first_content;
    s.decode_window = match (first_content, last_content) {
        (Some(a), Some(b)) if b > a => Some(b - a),
        _ => None,
    };
    if stream_tokens == 0 {
        return Err("stream produced no content tokens".to_string());
    }
    Ok(s)
}

/// Offset just past the first `\n\n` / `\r\n\r\n` in the buffer, if any.
fn find_blank_line(buf: &[u8]) -> Option<usize> {
    if let Some(p) = buf.windows(2).position(|w| w == b"\n\n") {
        return Some(p + 2);
    }
    if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
        return Some(p + 4);
    }
    None
}

/// Extract every `data: <payload>` line in one SSE event.
fn data_payloads(event: &str) -> Vec<String> {
    event
        .lines()
        .filter_map(|line| {
            let rest = line.strip_prefix("data:")?;
            Some(rest.trim().to_string())
        })
        .filter(|p| !p.is_empty())
        .collect()
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}…", &s[..n])
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
struct Series {
    values: Vec<f64>,
}

impl Series {
    fn push(&mut self, v: Option<f64>) {
        if let Some(x) = v {
            if x.is_finite() {
                self.values.push(x);
            }
        }
    }

    fn p(&self, q: f64) -> Option<f64> {
        if self.values.is_empty() {
            return None;
        }
        let mut sorted = self.values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((sorted.len() as f64 - 1.0) * q).round() as usize;
        Some(sorted[idx.min(sorted.len() - 1)])
    }

    fn median(&self) -> Option<f64> {
        self.p(0.5)
    }
}

// ---------------------------------------------------------------------------
// Benchmark body
// ---------------------------------------------------------------------------

/// Everything collected for one (framework, prompt size, concurrency) cell.
#[derive(Clone, Debug, Default)]
struct Cell {
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    ttft: Series,
    prefill: Series,
    decode: Series,
    e2e: Series,
    /// Aggregate completion tokens across the concurrency window,
    /// divided by the window's wall time. Equals the single-request
    /// decode rate at concurrency 1.
    aggregate_tps: Option<f64>,
    peak_rss_bytes: u64,
    notes: Vec<String>,
}

async fn run_cell(
    client: &reqwest::Client,
    ep: &Endpoint,
    knobs: &Knobs,
    prompt_words: usize,
    concurrency: usize,
) -> Result<Cell, String> {
    let base_prompt = filler_prompt(prompt_words);
    let mut cell = Cell::default();

    // A unique head on every prompt keeps prefix caching honest. llama.cpp
    // and ollama reuse the KV of an identical prefix, which would turn a
    // repeat's TTFT into a cache lookup instead of a prefill measurement —
    // llama.cpp reported 12 of 13 prompt tokens cached before this. Prefix
    // matching starts at the first token, so the nonce has to lead.
    let nonce = |tag: &str| format!("[{tag} {}] {base_prompt}", next_salt());

    // Warmup: also absorbs the first-request lazy model load that
    // every one of these runtimes does. Never timed.
    for i in 0..knobs.warmup.max(1) {
        let prompt = nonce(&format!("warmup {i}"));
        chat_nonstream(client, ep, &prompt, knobs.max_tokens)
            .await
            .map_err(|e| format!("warmup: {e}"))?;
        let _ = MemorySnapshot::of_pid(ep.pid.unwrap_or(0));
    }

    if concurrency <= 1 {
        for i in 0..knobs.repeats {
            // Separate nonces for the two modes within a repeat: sharing one
            // let the stream run read the KV the non-stream run had just
            // written, which showed up as a 108 ms TTFT beside a 4.8 s e2e.
            let ns_prompt = nonce(&format!("rep {i} ns"));
            let ss_prompt = nonce(&format!("rep {i} ss"));
            let rss = MemorySnapshot::of_pid(ep.pid.unwrap_or(0));
            if rss.peak_rss_bytes > cell.peak_rss_bytes {
                cell.peak_rss_bytes = rss.peak_rss_bytes;
            }

            let ns = chat_nonstream(client, ep, &ns_prompt, knobs.max_tokens)
                .await
                .map_err(|e| format!("nonstream: {e}"))?;
            // Some backends (notably mistralrs on CPU with in-situ
            // quantisation) don't emit token-by-token deltas in the SSE
            // shape the harness expects; treat a stream stall as
            // "non-stream only" rather than dropping the cell.
            let ss = match chat_stream(client, ep, &ss_prompt, knobs.max_tokens).await {
                Ok(s) => s,
                Err(e) => {
                    cell.notes.push(format!("stream unavailable: {e}"));
                    Sample {
                        e2e: ns.e2e,
                        prompt_tokens: ns.prompt_tokens,
                        completion_tokens: ns.completion_tokens,
                        ..Default::default()
                    }
                }
            };

            // The non-stream run is the authoritative token count
            // (neuron drops the terminal usage chunk on the stream
            // path); the stream run supplies the timing. They generate
            // greedily so the counts agree — flag it when they don't.
            cell.prompt_tokens = cell.prompt_tokens.or(ns.prompt_tokens);
            cell.completion_tokens = ns.completion_tokens.or(ss.completion_tokens);
            if let (Some(a), Some(b)) = (ns.completion_tokens, Some(ss.stream_tokens as u32)) {
                if a != b {
                    cell.notes.push(format!(
                        "token count mismatch: nonstream={a} stream={b}"
                    ));
                }
            }

            // TTFT is measured on the stream run but `prompt_tokens`
            // is only reliable from the non-stream one, so graft it on
            // before deriving the prefill rate.
            let mut st = ss;
            st.prompt_tokens = st.prompt_tokens.or(ns.prompt_tokens);

            cell.ttft.push(st.ttft);
            cell.prefill.push(st.prefill_tps());
            cell.decode.push(st.decode_tps());
            cell.e2e.push(Some(ns.e2e));
            cell.aggregate_tps = st.decode_tps();
        }
    } else {
        // Throughput shape: C simultaneous non-stream requests. The
        // window's wall time turns into an aggregate token rate, which
        // is the number that separates a batching engine from a
        // serialised one.
        for i in 0..knobs.repeats {
            let rss = MemorySnapshot::of_pid(ep.pid.unwrap_or(0));
            if rss.peak_rss_bytes > cell.peak_rss_bytes {
                cell.peak_rss_bytes = rss.peak_rss_bytes;
            }

            let window_start = Instant::now();
            let mut handles = Vec::with_capacity(concurrency);
            for j in 0..concurrency {
                let client = client.clone();
                let ep = ep.clone();
                let prompt = nonce(&format!("rep {i} req {j}"));
                let max_tokens = knobs.max_tokens;
                handles.push(tokio::spawn(async move {
                    chat_nonstream(&client, &ep, &prompt, max_tokens).await
                }));
            }

            let mut total_tokens = 0u32;
            let mut latencies = Vec::new();
            for h in handles {
                let sample = h
                    .await
                    .map_err(|e| format!("join: {e}"))?
                    .map_err(|e| format!("concurrent request: {e}"))?;
                total_tokens += sample.completion_tokens.unwrap_or(0);
                latencies.push(sample.e2e);
                cell.prompt_tokens = cell.prompt_tokens.or(sample.prompt_tokens);
                cell.completion_tokens = cell.completion_tokens.or(sample.completion_tokens);
            }
            let wall = window_start.elapsed().as_secs_f64();
            if wall > 0.0 && total_tokens > 0 {
                cell.aggregate_tps = Some(total_tokens as f64 / wall);
            }
            for l in latencies {
                cell.e2e.push(Some(l));
            }
            // TTFT is meaningless under concurrency for a serialised
            // engine; leave the series empty and let the report show
            // "n/a" rather than a number that invites misreading.
        }
    }

    // De-duplicate repeated mismatch notes.
    cell.notes.sort();
    cell.notes.dedup();
    Ok(cell)
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

async fn probe(client: &reqwest::Client, ep: &Endpoint) -> bool {
    // Tolerate either `/v1` or `/v1/...` in the base URL, so that
    // `ollama=http://host:11434/v1` and `ollama=http://host:11434` both
    // hit `/v1/models`.
    let base = ep.base.trim_end_matches('/');
    let url = if base.ends_with("/v1") {
        format!("{base}/models")
    } else {
        format!("{base}/v1/models")
    };
    match client.get(&url).send().await {
        Ok(r) => r.status().is_success(),
        Err(_) => false,
    }
}

fn default_endpoints() -> Vec<(String, String)> {
    vec![
        ("neuron".into(), "http://127.0.0.1:18083".into()),
        ("ollama".into(), "http://127.0.0.1:11434".into()),
        ("llama.cpp".into(), "http://127.0.0.1:18080".into()),
        ("vllm".into(), "http://127.0.0.1:18000".into()),
    ]
}

fn build_endpoints() -> Vec<Endpoint> {
    let pairs = env_str("NEURON_BENCH_ENDPOINTS")
        .map(|s| parse_pairs(&s))
        .filter(|v| !v.is_empty())
        .unwrap_or_else(default_endpoints);
    let models = env_str("NEURON_BENCH_MODELS")
        .map(|s| parse_pairs(&s))
        .unwrap_or_default();
    // Pids may be given either as a single pairs list or one var per
    // framework; the pairs list is what the driver script emits.
    let pids = env_str("NEURON_BENCH_PIDS")
        .map(|s| parse_pairs(&s))
        .unwrap_or_default();

    pairs
        .into_iter()
        .map(|(name, base)| {
            let suffix = env_key_suffix(&name);
            let pid_key = format!("NEURON_BENCH_PID_{suffix}");
            Endpoint {
                model: lookup(&models, &name, &name),
                device: env_str(&format!("NEURON_BENCH_DEVICE_{suffix}"))
                    .unwrap_or_else(|| "unspecified".to_string()),
                precision: env_str(&format!("NEURON_BENCH_PRECISION_{suffix}"))
                    .unwrap_or_else(|| "unspecified".to_string()),
                pid: env_str(&pid_key)
                    .and_then(|s| s.parse().ok())
                    .or_else(|| lookup(&pids, &name, "").parse().ok()),
                name,
                base,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

fn fmt_opt(v: Option<f64>, digits: usize) -> String {
    match v {
        Some(x) => format!("{x:.digits$}"),
        None => "n/a".to_string(),
    }
}

fn fmt_mb(bytes: u64) -> String {
    if bytes == 0 {
        "n/a".to_string()
    } else {
        format!("{:.0}", bytes as f64 / 1_048_576.0)
    }
}

fn host_line() -> String {
    let mut out = format!(
        "`{}/{}`, {} cores reported by std::thread::available_parallelism",
        std::env::consts::OS,
        std::env::consts::ARCH,
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
    );
    if let Some(info) = env_str("NEURON_BENCH_HOST_INFO") {
        let _ = write!(out, ", {info}");
    }
    out
}

fn git_rev() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

fn render_report(
    knobs: &Knobs,
    endpoints: &[Endpoint],
    cells: &[(String, usize, usize, Cell)], // (framework, prompt_words, concurrency, cell)
    unavailable: &[(String, String)],
) -> String {
    let mut md = String::new();
    let _ = writeln!(md, "# neuron vs vLLM / ollama / llama.cpp — LLM inference speed");
    let _ = writeln!(md);
    let _ = writeln!(md, "Generated by `benches/llm_speed.rs` on {}.", chrono::Utc::now().to_rfc3339());
    let _ = writeln!(md);
    let _ = writeln!(md, "- Host: {}", host_line());
    let _ = writeln!(md, "- neuron revision: `{}`", git_rev());
    let _ = writeln!(
        md,
        "- knobs: warmup={}, repeats={}, max_tokens={}, prompt_words={:?}, concurrency={:?}",
        knobs.warmup, knobs.repeats, knobs.max_tokens, knobs.prompt_words, knobs.concurrency
    );
    let _ = writeln!(md);

    if !unavailable.is_empty() {
        let _ = writeln!(md, "## Not measured this run");
        let _ = writeln!(md);
        for (name, why) in unavailable {
            let _ = writeln!(md, "- **{name}** — {why}");
        }
        let _ = writeln!(md);
    }

    // ---- Headline table: concurrency 1 ----
    let _ = writeln!(md, "## Decode throughput (concurrency 1)");
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "The headline number. `decode tok/s` is steady-state generation rate after the first token; `prefill tok/s` is `prompt_tokens / TTFT`."
    );
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "| framework | device | weights | prompt (words) | prompt tok | out tok | TTFT ms (p50) | prefill tok/s | decode tok/s (p50) | e2e ms (p50) | peak RSS MB |"
    );
    let _ = writeln!(
        md,
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    );

    for (name, words, conc, cell) in cells {
        if *conc != 1 {
            continue;
        }
        let ep = endpoints.iter().find(|e| &e.name == name);
        let (device, precision) = match ep {
            Some(e) => (e.device.clone(), e.precision.clone()),
            None => ("?".into(), "?".into()),
        };
        let _ = writeln!(
            md,
            "| {name} | {device} | {precision} | {words} | {} | {} | {} | {} | {} | {} | {} |",
            cell.prompt_tokens.map(|x| x.to_string()).unwrap_or_else(|| "n/a".into()),
            cell.completion_tokens.map(|x| x.to_string()).unwrap_or_else(|| "n/a".into()),
            fmt_opt(cell.ttft.median().map(|s| s * 1000.0), 1),
            fmt_opt(cell.prefill.median(), 1),
            fmt_opt(cell.decode.median(), 2),
            fmt_opt(cell.e2e.median().map(|s| s * 1000.0), 1),
            fmt_mb(cell.peak_rss_bytes),
        );
    }
    let _ = writeln!(md);

    // ---- Scaling table ----
    let concurrencies: Vec<usize> = {
        let mut v: Vec<usize> = cells.iter().map(|(_, _, c, _)| *c).collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    if concurrencies.len() > 1 {
        let _ = writeln!(md, "## Concurrency scaling");
        let _ = writeln!(md);
        let _ = writeln!(
            md,
            "`aggregate tok/s` is total completion tokens across the concurrent window divided by the window's wall time. A batching engine scales here; a serialised engine does not, and that is a capability difference rather than a measurement artefact."
        );
        let _ = writeln!(md);
        let _ = writeln!(
            md,
            "| framework | prompt (words) | concurrency | aggregate tok/s | e2e ms (p50) | e2e ms (p95) |"
        );
        let _ = writeln!(md, "|---|---:|---:|---:|---:|---:|");
        for (name, words, conc, cell) in cells {
            let _ = writeln!(
                md,
                "| {name} | {words} | {conc} | {} | {} | {} |",
                fmt_opt(cell.aggregate_tps, 2),
                fmt_opt(cell.e2e.median().map(|s| s * 1000.0), 1),
                fmt_opt(cell.e2e.p(0.95).map(|s| s * 1000.0), 1),
            );
        }
        let _ = writeln!(md);
    }

    // ---- Notes ----
    let mut any_note = false;
    for (name, words, conc, cell) in cells {
        for n in &cell.notes {
            if !any_note {
                let _ = writeln!(md, "## Run notes");
                let _ = writeln!(md);
                any_note = true;
            }
            let _ = writeln!(md, "- **{name}** (prompt {words} words, c={conc}): {n}");
        }
    }
    if any_note {
        let _ = writeln!(md);
    }

    // ---- Methodology ----
    let _ = writeln!(md, "## Methodology");
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "1. **Same checkpoint.** Every framework serves the same MiniCPM5-2B weights. neuron and vLLM read the HF `safetensors` directory directly; ollama and llama.cpp consume a GGUF converted from that same directory."
    );
    let _ = writeln!(
        md,
        "2. **Precision differs and is shown per row.** neuron's candle path defaults to materialising weights as F32, which doubles both memory and arithmetic against a BF16 engine; `NEURON_LLM_DTYPE=f16|bf16` halves that and the `weights` column records what each row actually ran. candle's CPU backend has no BF16 matmul, so `bf16` loads but cannot run there. F32 weights for a 2B model need ~10 GB resident, which will page on a 16 GB host and make the timings measure disk rather than compute — check free RAM before trusting a row."
    );
    let _ = writeln!(
        md,
        "3. **Prompt rendering differs.** neuron wraps the message in a hardcoded ChatML template, ollama and vLLM apply the checkpoint's `chat_template.jinja`. The resulting token counts differ slightly, which is why `prompt tok` is reported from each server's own `usage` and why prefill and decode are reported as separate numbers. `decode tok/s` is the metric least affected by this."
    );
    let _ = writeln!(
        md,
        "4. **All timing is client-side**, taken around the HTTP round trip on loopback. neuron's streaming path stamps `created` at second resolution and drops the terminal finish/usage chunk, so no server-side clock is trusted anywhere in this report."
    );
    let _ = writeln!(
        md,
        "5. **Warmup is discarded.** Every runtime lazy-loads weights on the first request; {} warmup request(s) per cell are thrown away before timing begins.",
        knobs.warmup
    );
    let _ = writeln!(
        md,
        "6. **Decode rate** = `(streamed tokens - 1) / (last content chunk arrival - first content chunk arrival)`. Tokens are counted as streamed content or `reasoning_content` deltas — hybrid-thinking models put their thought trace in the latter and it costs decode time all the same — cross-checked against the non-stream `usage.completion_tokens`; any disagreement is listed under Run notes."
    );
    let _ = writeln!(
        md,
        "7. **peak RSS** is the `PeakWorkingSetSize` of the server process, sampled externally by pid (`benches/memory.rs::MemorySnapshot::of_pid`). Frameworks without a configured pid show `n/a`."
    );
    let _ = writeln!(
        md,
        "8. **Greedy decoding throughout** (`temperature: 0`), so repeated runs generate identical work and the numbers are comparable across frameworks and across repeats."
    );
    let _ = writeln!(
        md,
        "9. **Every request gets a unique prompt head**, so no framework's prefix cache can answer a repeat. Without this, llama.cpp reported 12 of 13 prompt tokens cached and its prefill rate looked impossibly fast against a server that re-prefills from scratch every call. The nonce is a couple of tokens of constant size, so the workload stays comparable across repeats."
    );
    let _ = writeln!(md);
    md
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    if env_str("NEURON_BENCH_HELP").is_some() {
        print_help();
        return;
    }

    let knobs = Knobs::from_env();
    let endpoints = build_endpoints();

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        // A pooled socket a server has already closed surfaces as a bare
        // "error sending request" on the next call. On loopback the cost of
        // a fresh TCP handshake is nothing next to a multi-second generate.
        .pool_max_idle_per_host(0)
        // CPU decode of a 2.5B model can take minutes per request.
        .timeout(Duration::from_secs(3600))
        .build()
        .expect("reqwest client");

    println!("[llm-speed] probing {} endpoint(s)...", endpoints.len());
    let mut available = Vec::new();
    let mut unavailable: Vec<(String, String)> = Vec::new();
    for ep in &endpoints {
        let ok = rt.block_on(probe(&client, ep));
        if ok {
            println!(
                "[llm-speed]   {:<10} {} (model={}, device={}, {}) — reachable",
                ep.name, ep.base, ep.model, ep.device, ep.precision
            );
            available.push(ep.clone());
        } else {
            println!("[llm-speed]   {:<10} {} — not available", ep.name, ep.base);
            unavailable.push((ep.name.clone(), format!("`{}` did not answer `/v1/models`", ep.base)));
        }
    }

    if available.is_empty() {
        eprintln!(
            "[llm-speed] FATAL: no reachable OpenAI-compatible endpoint. \
             Start the servers first (see scripts/run-llm-speed-compare.sh) or \
             point NEURON_BENCH_ENDPOINTS at one."
        );
        std::process::exit(2);
    }

    let mut cells: Vec<(String, usize, usize, Cell)> = Vec::new();
    for ep in &available {
        for &words in &knobs.prompt_words {
            for &conc in &knobs.concurrency {
                let label = format!(
                    "[llm-speed] {} prompt={} words concurrency={}",
                    ep.name, words, conc
                );
                println!("{label} ...");
                let started = Instant::now();
                match rt.block_on(run_cell(&client, ep, &knobs, words, conc)) {
                    Ok(cell) => {
                        println!(
                            "            done in {:.1}s — decode {} tok/s, TTFT {} ms, e2e {} ms",
                            started.elapsed().as_secs_f64(),
                            fmt_opt(cell.decode.median(), 2),
                            fmt_opt(cell.ttft.median().map(|s| s * 1000.0), 1),
                            fmt_opt(cell.e2e.median().map(|s| s * 1000.0), 1),
                        );
                        for n in &cell.notes {
                            println!("            note: {n}");
                        }
                        cells.push((ep.name.clone(), words, conc, cell));
                    }
                    Err(e) => {
                        println!("            FAILED: {e}");
                        unavailable.push((ep.name.clone(), e));
                    }
                }
            }
        }
    }

    if cells.is_empty() {
        eprintln!("[llm-speed] FATAL: every measurement failed.");
        std::process::exit(2);
    }

    let report = render_report(&knobs, &endpoints, &cells, &unavailable);
    let out_path = env_str("NEURON_BENCH_OUT")
        .unwrap_or_else(|| "target/llm-speed-compare.md".to_string());
    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match std::fs::write(&out_path, &report) {
        Ok(()) => println!("[llm-speed] report written to {out_path}"),
        Err(e) => eprintln!("[llm-speed] could not write {out_path}: {e}"),
    }
    println!();
    print!("{report}");
}

fn print_help() {
    println!(
        "\
neuron LLM speed comparison harness (benches/llm_speed.rs)

Drives OpenAI-compatible /v1/chat/completions endpoints and reports
time-to-first-token, prefill tok/s, decode tok/s, end-to-end latency
and peak process RSS.

ENDPOINTS
  NEURON_BENCH_ENDPOINTS   \"name=base_url,...\"
                           default: neuron=127.0.0.1:18083,
                                    ollama=127.0.0.1:11434,
                                    llama.cpp=127.0.0.1:18080,
                                    vllm=127.0.0.1:18000
  NEURON_BENCH_MODELS      \"name=model_id,...\"  (defaults to the name)
  NEURON_BENCH_PID_<NAME>  server pid, enables the peak RSS column
  NEURON_BENCH_DEVICE_<NAME>      free-form device label, e.g. \"cuda:0\"
  NEURON_BENCH_PRECISION_<NAME>   free-form weights label, e.g. \"gguf q8_0\"
                           <NAME> is the framework name uppercased with
                           punctuation stripped: llama.cpp -> LLAMACPP

WORKLOAD
  NEURON_BENCH_PROFILE     quick | standard (default) | full
  NEURON_BENCH_WARMUP      warmup requests per cell, discarded (default 1)
  NEURON_BENCH_REPEATS     timed repeats per cell (default 3)
  NEURON_BENCH_MAX_TOKENS  completion budget (default 32)
  NEURON_BENCH_PROMPT_WORDS     \"64,512\" — filler words per prompt
  NEURON_BENCH_CONCURRENCY      \"1,4\"
  individual variables override whatever the profile set

OUTPUT
  NEURON_BENCH_OUT         report path (default target/llm-speed-compare.md)
  NEURON_BENCH_HOST_INFO   free-form string embedded in the report header

EXIT CODES
  0  report written
  2  nothing reachable, or every measurement failed
"
    );
}
