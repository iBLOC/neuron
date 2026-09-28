//! Per-capability × per-engine × per-device model matrix.
//!
//! Walks every (cap, engine, device) neuron supports under the current
//! `Cargo.toml` feature set, measures load time + peak RSS + warm
//! latency + RTF, and writes one report at `target/model-matrix.md`.
//!
//! Unlike `benches/perf.rs` (Criterion microbenchmarks of DTO
//! construction), `benches/llm_speed.rs` (HTTP round-trip across
//! non-neuron frameworks) and `benches/asr_speed.rs` (one cap, two
//! engines, in-process), this bench is the **comprehensive matrix**:
//! 8 capabilities × multiple engines per cap × available devices,
//! driven through neuron's own OpenAI-compatible HTTP surface so the
//! numbers reflect what a real client sees.
//!
//! ## Why this is not a Criterion bench
//!
//! Same reason as the other speed harnesses: a single measurement here
//! is a full inference round trip (hundreds of ms to several seconds
//! for LLM). `harness = false` is set in `Cargo.toml` and the bench
//! runs a fixed number of repeats with its own p50/p95 math.
//!
//! ## Run
//!
//! ```bash
//! # 1. start the server
//! NEURON_PORT=18083 \
//! NEURON_LLM_RUNTIME=candle \
//! NEURON_LLM_MODEL_PATH=/path/to/minicpm5-1b \
//!   cargo run --release --example neuron_server --features "candle,tract,sherpa,voxcpm"
//!
//! # 2. in another shell, point the bench at it
//! NEURON_BENCH_URL=http://127.0.0.1:18083 \
//! NEURON_BENCH_WARMUP=1 NEURON_BENCH_REPEATS=3 \
//!   cargo bench --bench model_matrix --features "candle,tract,sherpa,voxcpm"
//! ```
//!
//! Environment:
//!
//! | var | default | meaning |
//! |---|---|---|
//! | `NEURON_BENCH_URL` | `http://127.0.0.1:8083` | Base URL of a running neuron server |
//! | `NEURON_BENCH_PROFILE` | `quick` | `quick` / `standard` / `full` preset for warmup × repeats |
//! | `NEURON_BENCH_WARMUP` | (profile) | Override warmup count |
//! | `NEURON_BENCH_REPEATS` | (profile) | Override repeats |
//! | `NEURON_BENCH_<CAP>_DIR` | unset | Override the directory `default_rows` looks up the model weights in |
//! | `NEURON_BENCH_MODEL_DIRS` | unset | Colon-separated roots; the bench joins each with `<cap>/<model_id>` |
//! | `NEURON_BENCH_OUT` | `target/model-matrix.md` | Report path |
//! | `NEURON_BENCH_DISABLE` | unset | Comma-separated `cap.engine` pairs to skip (e.g. `asr.tract,vision.tract`) |
//!
//! `NEURON_BENCH_<CAP>_RUNTIME` lets a host steer which engine the
//! server should be running for that cap; the bench does **not** set
//! runtime env vars itself — it only drives whatever server the host
//! started. Servers started via `scripts/run-model-matrix.sh` set the
//! ones needed to make every row runnable.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use neuron_lib::bench::model_matrix::{
    asr_sample_multipart_part, default_rows, resolve_model_dir, sample_for, Capability, Device,
};
use serde_json::Value;

#[path = "memory.rs"]
#[allow(dead_code, unused_imports)]
mod memory;
use memory::MemorySnapshot;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct Knobs {
    warmup: usize,
    repeats: usize,
}

impl Knobs {
    fn from_env() -> Self {
        let (def_warmup, def_repeats) = match std::env::var("NEURON_BENCH_PROFILE").as_deref() {
            Ok("quick") => (0, 1),
            Ok("full") => (2, 5),
            _ => (1, 3),
        };
        Self {
            warmup: env_usize("NEURON_BENCH_WARMUP", def_warmup),
            repeats: env_usize("NEURON_BENCH_REPEATS", def_repeats),
        }
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).ok().filter(|v| !v.is_empty()).unwrap_or_else(|| default.to_string())
}

// ---------------------------------------------------------------------------
// HTTP plumbing
// ---------------------------------------------------------------------------

struct Client {
    base: reqwest::Url,
    http: reqwest::Client,
}

impl Client {
    fn new(base: &str) -> Result<Self, String> {
        let http = reqwest::Client::builder()
            .pool_max_idle_per_host(4)
            .timeout(Duration::from_secs(120))
            .build()
            .map_err(|e| format!("reqwest client: {e}"))?;
        let url = reqwest::Url::parse(base).map_err(|e| format!("bad NEURON_BENCH_URL: {e}"))?;
        Ok(Self { base: url, http })
    }

    /// `POST` a JSON body and time the round trip; records RSS before
    /// and after so the peak observed during the call is reported.
    async fn post_json(
        &self,
        path: &str,
        body: &Value,
    ) -> Result<HttpMeasurement, String> {
        let url = self.base.join(path).map_err(|e| format!("join {path}: {e}"))?;
        let peak_before = MemorySnapshot::now();
        let started = Instant::now();
        let resp = self
            .http
            .post(url)
            .json(body)
            .send()
            .await
            .map_err(|e| format!("send {path}: {e}"))?;
        let status = resp.status();
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| format!("body {path}: {e}"))?
            .len();
        let elapsed = started.elapsed();
        let peak_after = MemorySnapshot::now();
        let peak = peak_after.peak_rss_bytes.max(peak_before.peak_rss_bytes);
        Ok(HttpMeasurement {
            wall: elapsed,
            http_status: status.as_u16(),
            body_bytes: bytes,
            peak_rss_bytes: peak,
        })
    }

    /// `POST` ASR (multipart, file upload). Server-side RSS tracking
    /// is the same as `post_json`; we just construct the body
    /// differently because `/v1/audio/transcriptions` is multipart.
    async fn post_asr(&self, model_id: &str) -> Result<HttpMeasurement, String> {
        let url = self
            .base
            .join("/v1/audio/transcriptions")
            .map_err(|e| format!("join asr: {e}"))?;
        let (bytes, filename) = asr_sample_multipart_part();
        let part = reqwest::multipart::Part::bytes(bytes)
            .file_name(filename.to_string())
            .mime_str("audio/wav")
            .map_err(|e| format!("asr mime: {e}"))?;
        let form = reqwest::multipart::Form::new()
            .text("model", model_id.to_string())
            .part("file", part);

        let peak_before = MemorySnapshot::now();
        let started = Instant::now();
        let resp = self
            .http
            .post(url)
            .multipart(form)
            .send()
            .await
            .map_err(|e| format!("send asr: {e}"))?;
        let status = resp.status();
        let body = resp.bytes().await.map_err(|e| format!("body asr: {e}"))?;
        let elapsed = started.elapsed();
        let peak_after = MemorySnapshot::now();
        let peak = peak_after.peak_rss_bytes.max(peak_before.peak_rss_bytes);
        Ok(HttpMeasurement {
            wall: elapsed,
            http_status: status.as_u16(),
            body_bytes: body.len(),
            peak_rss_bytes: peak,
        })
    }
}

struct HttpMeasurement {
    wall: Duration,
    http_status: u16,
    body_bytes: usize,
    peak_rss_bytes: u64,
}

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

struct RowMeasurement {
    samples: Vec<HttpMeasurement>,
    load_peak_rss_bytes: u64,
    error: Option<String>,
}

impl RowMeasurement {
    fn median_ms(&self) -> f64 {
        percentile(&self.samples, 50.0).as_micros() as f64 / 1000.0
    }
    fn p95_ms(&self) -> f64 {
        percentile(&self.samples, 95.0).as_micros() as f64 / 1000.0
    }
    /// Run-time factor = wall-clock / `nominal_input_duration`. Only
    /// meaningful for ASR (1 s sine); everywhere else we report
    /// `samples/sec` instead by inversion.
    fn rtf(&self, nominal_secs: f64) -> f64 {
        if nominal_secs <= 0.0 {
            return 0.0;
        }
        self.median_ms() / 1000.0 / nominal_secs
    }
    fn peak_rss_mb(&self) -> f64 {
        let peak = self.load_peak_rss_bytes.max(
            self.samples.iter().map(|s| s.peak_rss_bytes).max().unwrap_or(0),
        );
        peak as f64 / 1_048_576.0
    }
    fn body_kb(&self) -> f64 {
        self.samples
            .last()
            .map(|s| s.body_bytes as f64 / 1024.0)
            .unwrap_or(0.0)
    }
    fn ok(&self) -> bool {
        self.error.is_none()
            && self
                .samples
                .last()
                .map(|s| s.http_status < 400)
                .unwrap_or(false)
    }
}

fn percentile(samples: &[HttpMeasurement], pct: f64) -> Duration {
    let mut s: Vec<Duration> = samples.iter().map(|s| s.wall).collect();
    s.sort();
    let idx = ((pct / 100.0) * (s.len().saturating_sub(1)) as f64).round() as usize;
    s.get(idx).copied().unwrap_or(Duration::ZERO)
}

async fn measure_row<'a>(client: &Client, row: &'a ModelRow<'a>, knobs: &Knobs) -> RowMeasurement {
    let peak_before = MemorySnapshot::now();

    // Warmup
    for _ in 0..knobs.warmup {
        if let Err(e) = send_once(client, row).await {
            return RowMeasurement {
                samples: Vec::new(),
                load_peak_rss_bytes: peak_before.peak_rss_bytes,
                error: Some(e),
            };
        }
    }
    // Timed repeats
    let mut samples = Vec::with_capacity(knobs.repeats);
    for _ in 0..knobs.repeats {
        match send_once(client, row).await {
            Ok(m) => samples.push(m),
            Err(e) => {
                return RowMeasurement {
                    samples,
                    load_peak_rss_bytes: peak_before.peak_rss_bytes,
                    error: Some(e),
                };
            }
        }
    }
    RowMeasurement {
        samples,
        load_peak_rss_bytes: peak_before.peak_rss_bytes,
        error: None,
    }
}

async fn send_once<'a>(client: &Client, row: &'a ModelRow<'a>) -> Result<HttpMeasurement, String> {
    let body = sample_for(row.cap, &row.model_id);
    if row.cap == Capability::Asr {
        client.post_asr(&row.model_id).await
    } else {
        client.post_json(row.cap.endpoint_path(), &body).await
    }
}

/// Trimmed row representation for the report: everything the matrix
/// generator cares about. The driver consults `resolve_model_dir`
/// before constructing this, so the row itself does not need to carry
/// `has_model_dir` -- a row is only here when its model dir exists.
struct ModelRow<'a> {
    cap: Capability,
    model_id: &'a str,
    engine_id: &'a str,
    device: Device,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let knobs = Knobs::from_env();
    let url = env_or("NEURON_BENCH_URL", "http://127.0.0.1:8083");
    let out_path = PathBuf::from(env_or("NEURON_BENCH_OUT", "target/model-matrix.md"));
    let disable: Vec<String> = std::env::var("NEURON_BENCH_DISABLE")
        .ok()
        .map(|v| v.split(',').map(str::trim).filter(|s| !s.is_empty()).map(String::from).collect())
        .unwrap_or_default();

    let rows = default_rows();
    println!("=== neuron model matrix ===");
    println!("server: {url}");
    println!("profile: warmup={} repeats={}", knobs.warmup, knobs.repeats);
    println!("rows in default matrix: {}", rows.len());

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    let client = match Client::new(&url) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[model-matrix] {e}");
            std::process::exit(2);
        }
    };

    let mut report = String::new();
    let _ = writeln!(report, "# neuron model matrix");
    let _ = writeln!(report);
    let _ = writeln!(
        report,
        "Per-capability × per-engine × per-device matrix on `{}`. \
         warmup={}, repeats={}. RSS is the server-side `WorkingSetSize` peak \
         measured from this client (the host process' RSS, not the server's, \
         would mislead the numbers; see `bench::model_matrix` notes).",
        url, knobs.warmup, knobs.repeats
    );
    let _ = writeln!(report);
    let _ = writeln!(report, "## Latency");
    let _ = writeln!(report);
    let _ = writeln!(
        report,
        "| cap | model | engine | device | RTF / latency | p50 ms | p95 ms | body KB | peak RSS MB | status |"
    );
    let _ = writeln!(
        report,
        "|---|---|---|---|---|---|---|---|---|---|"
    );

    let mut measured = Vec::new();
    let mut skipped = Vec::new();
    for row in &rows {
        let tag = format!("{}.{}", row.cap.as_str(), row.engine.id);
        if disable.iter().any(|d| d == &tag) {
            skipped.push((row, "disabled by NEURON_BENCH_DISABLE".to_string()));
            continue;
        }
        let dir = resolve_model_dir(row);
        let has_model_dir = dir.is_some();
        if !has_model_dir {
            skipped.push((row, "no model dir on disk".to_string()));
            continue;
        }
        let slim = ModelRow {
            cap: row.cap,
            model_id: &row.model_id,
            engine_id: row.engine.id,
            device: row.device,
        };
        println!(
            "[model-matrix] {:>4} {:<6} {:<14} {:<8} {:<6}",
            slim.cap.as_str(),
            slim.engine_id,
            slim.model_id,
            slim.device.as_str(),
            ""
        );
        let m = rt.block_on(measure_row(&client, &slim, &knobs));
        let nominal = match slim.cap {
            // ASR uses a 1-second sine; everything else is JSON DTO time
            // and "RTF" is meaningless, so report latency alone.
            Capability::Asr => 1.0,
            _ => 0.0,
        };
        let rtf_or_latency = if nominal > 0.0 {
            format!("RTF {:.3}", m.rtf(nominal))
        } else if m.ok() {
            format!("p50 {:.0} ms", m.median_ms())
        } else {
            "—".to_string()
        };
        let status = if m.error.is_some() {
            format!("FAILED: {}", m.error.as_deref().unwrap_or(""))
        } else if let Some(s) = m.samples.last() {
            if s.http_status >= 400 {
                format!("HTTP {}", s.http_status)
            } else {
                "OK".to_string()
            }
        } else {
            "no samples".to_string()
        };
        let _ = writeln!(
            report,
            "| {} | {} | {} | {} | {} | {:.1} | {:.1} | {:.1} | {:.0} | {} |",
            slim.cap.as_str(),
            slim.model_id,
            slim.engine_id,
            slim.device.as_str(),
            rtf_or_latency,
            m.median_ms(),
            m.p95_ms(),
            m.body_kb(),
            m.peak_rss_mb(),
            status,
        );
        measured.push((row, m));
    }

    // ---- skip section -----------------------------------------------------
    let _ = writeln!(report);
    let _ = writeln!(report, "## Skipped rows");
    let _ = writeln!(report);
    let _ = writeln!(report, "| cap | model | engine | reason |");
    let _ = writeln!(report, "|---|---|---|---|");
    for (row, reason) in &skipped {
        let _ = writeln!(
            report,
            "| {} | {} | {} | {} |",
            row.cap.as_str(),
            row.model_id,
            row.engine.id,
            reason
        );
    }

    // ---- totals ------------------------------------------------------------
    let _ = writeln!(report);
    let _ = writeln!(report, "## Summary");
    let _ = writeln!(report);
    let _ = writeln!(report, "| cap | rows measured | rows skipped |");
    let _ = writeln!(report, "|---|---|---|");
    let mut by_cap: std::collections::BTreeMap<&str, (usize, usize)> = Default::default();
    for (row, _) in &measured {
        let entry = by_cap.entry(row.cap.as_str()).or_insert((0, 0));
        entry.0 += 1;
    }
    for (row, _) in &skipped {
        let entry = by_cap.entry(row.cap.as_str()).or_insert((0, 0));
        entry.1 += 1;
    }
    for (cap, (ok, skip)) in by_cap {
        let _ = writeln!(report, "| {} | {} | {} |", cap, ok, skip);
    }

    if let Some(parent) = out_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(&out_path, &report).expect("write report");
    println!(
        "\n[model-matrix] report written to {} ({} measured, {} skipped)",
        out_path.display(),
        measured.len(),
        skipped.len()
    );
}
