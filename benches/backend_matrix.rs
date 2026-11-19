//! Same-model × backend matrix — one row per (model, backend, device).
//!
//! # What this is
//!
//! `benches/model_bench.rs` answers *"how fast is each model?"* — one
//! row per model in the capability catalog, each on the backend that
//! model is registered to. This harness answers the *other* question:
//! **"the same weights, on every backend that can serve them — how
//! fast is each?"**
//!
//! The two questions are not the same, and the gap is where decisions
//! get made wrong:
//!
//! * a model can be servable by **several** backends (Qwen3-0.6B by
//!   both `candle` and `mistralrs`), and the two differ by an order of
//!   magnitude on the same machine;
//! * a model can be servable by **none** of the local backends
//!   (Qwen3.5/3.8 by candle — its GDN stack has no upstream candle
//!   module) — that refusal is a fact worth *measuring*, not leaving
//!   to a footnote;
//! * a backend can be **unavailable** on this host (llama.cpp needs a
//!   C++ toolchain; mistralrs+cuda needs a candle-kernels/CUDA pair
//!   that may not compile) — the matrix must say so per cell instead
//!   of quietly dropping the row.
//!
//! Every cell therefore ends in exactly one of: a measured number, a
//! **skip with the real reason**, or a **failure with the real error**.
//!
//! # Methodology (identical to `model_bench.rs` so the numbers join
//! with the per-model sweep)
//!
//! One fixed prompt, greedy (temperature 0), 64 new tokens, streamed;
//! one warmup iteration discarded; decode tok/s = (chunks − 1) /
//! (total − first-chunk) seconds; TTFT = the first chunk's latency.
//!
//! # Running it
//!
//! ```bash
//! # CPU only (default backends: candle, mistralrs)
//! cargo bench --bench backend_matrix --features "candle,mistralrs"
//!
//! # CUDA rows need the MSVC environment (nvcc needs cl.exe):
//! scripts\cuda-build.bat cargo bench --bench backend_matrix --features "candle,cuda,mistralrs"
//! ```
//!
//! # Environment
//!
//! | var | default | meaning |
//! |---|---|---|
//! | `NEURON_BENCH_MATRIX_MODELS` | the local model library | comma-separated model **directories** (or names under the model library) |
//! | `NEURON_BENCH_MATRIX_BACKENDS` | `candle,mistralrs` | comma-separated backend ids (`candle` / `mistralrs` / `litert` / `llamacpp`) |
//! | `NEURON_BENCH_MATRIX_DEVICE` | `cuda` if the feature is on, else `cpu` | device the backends run on |
//! | `NEURON_BENCH_MATRIX_DEVICES` | the one device above | comma-separated devices to sweep (`cpu,cuda`) |
//! | `NEURON_BENCH_REPEATS` | `5` | measured iterations per cell (after warmup) |
//! | `NEURON_BENCH_WARMUP` | `1` | warmup iterations per cell |
//! | `NEURON_BENCH_MATRIX_OUT` | `target/backend-matrix-<device>.md` | markdown report path |
//!
//! # Output
//!
//! `target/backend-matrix-<device>.md` — one table, one row per
//! (model, backend), plus a coverage summary of which backend can
//! serve which model at all.

use std::path::{Path, PathBuf};
use std::time::Instant;

use futures_util::StreamExt;
use neuron_lib::config::{DeviceSelection, NeuronConfig};
use neuron_lib::llm::{ChatMessage, ChatRequest, LlmImpl, LlmRuntimeRegistry};

/// Same prompt as `model_bench.rs` — the two reports are meant to be
/// read together, so the workload must not drift.
const LLM_PROMPT: &str = "用一句话解释什么是神经网络。";
const MAX_TOKENS: u32 = 64;

/// One measured cell: a model, a backend, a device.
#[derive(Clone)]
struct Cell {
    model: String,
    model_dir: PathBuf,
    backend: String,
    device: String,
}

/// What came out of one cell.
enum Outcome {
    Measured {
        load_s: f64,
        ttft_ms: f64,
        tok_per_s: f64,
        total_ms: f64,
    },
    /// The backend could not serve this model (architecture refusal,
    /// missing weights, backend not compiled in). The reason is the
    /// measurement — "candle refuses Qwen3.5" is a row, not a gap.
    Skipped(String),
    Failed(String),
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx]
}

/// The local model library — where `~/.neuron/models/llm/<id>` lives.
fn model_library() -> PathBuf {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_default();
    PathBuf::from(home).join(".neuron").join("models").join("llm")
}

/// Resolve one entry of the model list: an absolute/relative directory
/// wins; otherwise it is looked up in the local model library.
///
/// MSYS/Git-Bash paths (`/c/Users/...`) are translated to their Windows
/// spelling (`C:/Users/...`) — a shell-authored path that "exists" in
/// the shell is not a path Rust can open, and the failure mode is a
/// silent "no weights" skip that reads like a missing model.
fn resolve_model_dir(entry: &str) -> Option<PathBuf> {
    let candidates = [
        PathBuf::from(entry),
        msys_to_windows(entry).map(PathBuf::from).unwrap_or_else(|| PathBuf::from(entry)),
        model_library().join(entry),
    ];
    for c in candidates {
        if c.join("config.json").is_file() {
            return Some(c);
        }
    }
    None
}

/// `/c/Users/x` → `C:/Users/x` (MSYS drive-letter form → Windows).
fn msys_to_windows(p: &str) -> Option<String> {
    let rest = p.strip_prefix('/')?;
    let (drive, tail) = rest.split_at(rest.find('/')?);
    if drive.len() != 1 {
        return None;
    }
    let drive = drive.chars().next()?.to_ascii_uppercase();
    Some(format!("{drive}:{tail}"))
}

/// Every model directory in the local library, one level deep
/// (`<id>` and `<vendor>/<id>` — the Qwen3.5/Qwen3.8 snapshots live
/// under `Qwen/`, and a sweep that misses them would silently drop the
/// most interesting row of the matrix).
fn default_models() -> Vec<String> {
    let lib = model_library();
    let mut out = Vec::new();
    let mut push_dir = |dir: &Path| {
        if dir.join("config.json").is_file() {
            out.push(dir.to_string_lossy().into_owned());
        }
    };
    if let Ok(entries) = std::fs::read_dir(&lib) {
        for e in entries.flatten() {
            let path = e.path();
            if !path.is_dir() {
                continue;
            }
            push_dir(&path);
            // One level deeper: vendor sub-folders.
            if let Ok(subs) = std::fs::read_dir(&path) {
                for s in subs.flatten() {
                    let sp = s.path();
                    if sp.is_dir() {
                        push_dir(&sp);
                    }
                }
            }
        }
    }
    out.sort();
    out
}

fn device_from(name: &str) -> DeviceSelection {
    match name {
        "cuda" | "gpu" | "nvidia" => DeviceSelection::Nvidia,
        "cpu" => DeviceSelection::Cpu,
        _ => DeviceSelection::Auto,
    }
}

fn device_label(d: DeviceSelection) -> &'static str {
    match d {
        DeviceSelection::Nvidia => "cuda",
        DeviceSelection::Cpu => "cpu",
        _ => "auto",
    }
}

fn chat_request(model: &str) -> ChatRequest {
    ChatRequest {
        model: model.to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: LLM_PROMPT.to_string(),
            reasoning_content: None,
        }],
        temperature: Some(0.0),
        max_tokens: Some(MAX_TOKENS),
        top_p: None,
        top_k: None,
        stream: true,
        stop: None,
    }
}

/// Run one (model, backend, device) cell.
///
/// The two text backends locate the weights differently, and that
/// difference is the whole point of the matrix, so it is handled
/// explicitly here:
///
/// * **candle** resolves the weights from `NEURON_LLM_MODEL_PATH`
///   (the directory), falling back to the model library by id;
/// * **mistralrs** takes the weight **directory itself** as the model
///   id (mistral.rs loads a local directory directly).
async fn run_cell(cell: &Cell, device: DeviceSelection, warmup: usize, repeats: usize) -> Outcome {
    let impl_id = match LlmImpl::parse(&cell.backend) {
        Ok(i) => i,
        Err(e) => return Outcome::Skipped(format!("backend not available in this build: {e}")),
    };

    // candle reads the weights from this env var; mistralrs gets the
    // directory as its model id. Setting the env var for both is
    // harmless (mistralrs ignores it) and keeps one code path.
    unsafe {
        std::env::set_var("NEURON_LLM_MODEL_PATH", &cell.model_dir);
    }

    let model_id = match cell.backend.as_str() {
        "candle" => cell.model.clone(),
        // A path is the model id for the path-based backends.
        _ => cell.model_dir.to_string_lossy().into_owned(),
    };

    let mut registry = LlmRuntimeRegistry::new(std::sync::Arc::new(NeuronConfig::default()));
    if let Err(e) = registry.register(model_id.clone(), impl_id, device, 0) {
        return Outcome::Skipped(format!("register refused: {e}"));
    }

    let t_load = Instant::now();
    let rt = match registry.runtime_for(&model_id).await {
        Ok(rt) => rt,
        Err(e) => {
            let msg = e.message.clone();
            let load_s = t_load.elapsed().as_secs_f64();
            return classify_load_error(&msg, load_s);
        }
    };
    let load_s = t_load.elapsed().as_secs_f64();

    let mut tok_samples: Vec<f64> = Vec::new();
    let mut ttft_samples: Vec<f64> = Vec::new();
    let mut total_samples: Vec<f64> = Vec::new();
    for i in 0..(warmup + repeats) {
        let req = chat_request(&model_id);
        let t0 = Instant::now();
        let mut stream = match rt.chat_stream(req).await {
            Ok(s) => s,
            Err(e) => return Outcome::Failed(format!("chat_stream failed: {e}")),
        };
        let mut first: Option<f64> = None;
        let mut chunks = 0usize;
        while let Some(chunk) = stream.next().await {
            if first.is_none() {
                first = Some(t0.elapsed().as_secs_f64() * 1000.0);
            }
            for _choice in chunk.choices.iter() {
                chunks += 1;
            }
        }
        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let first_ms = first.unwrap_or(total_ms);
        if i >= warmup {
            let decode_s = (total_ms - first_ms).max(0.0) / 1000.0;
            let tok_per_s = if decode_s > 0.0 {
                (chunks.saturating_sub(1)) as f64 / decode_s
            } else {
                0.0
            };
            tok_samples.push(tok_per_s);
            ttft_samples.push(first_ms);
            total_samples.push(total_ms);
        }
    }
    if tok_samples.is_empty() {
        return Outcome::Failed("no measured iteration".to_string());
    }
    let mut tok_sorted = tok_samples.clone();
    tok_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut ttft_sorted = ttft_samples.clone();
    ttft_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut total_sorted = total_samples.clone();
    total_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Outcome::Measured {
        load_s,
        ttft_ms: percentile(&ttft_sorted, 0.50),
        tok_per_s: percentile(&tok_sorted, 0.50),
        total_ms: percentile(&total_sorted, 0.50),
    }
}

/// Tell an architecture refusal ("this backend cannot serve this
/// model") apart from a genuine failure. A refusal is a *measured*
/// fact about the matrix — it belongs in the table, not in the
/// failures list.
fn classify_load_error(msg: &str, load_s: f64) -> Outcome {
    let lower = msg.to_ascii_lowercase();
    let refusal_markers = [
        "not supported by the candle backend",
        "not supported",
        "cannot serve",
        "does not fit on the devices",
        "no such file or directory",
        "not found",
        "no module for",
        "unsupported model type",
        "moe",
    ];
    if refusal_markers.iter().any(|m| lower.contains(m)) {
        return Outcome::Skipped(format!("{msg} (after {load_s:.1}s)"));
    }
    Outcome::Failed(format!("{msg} (after {load_s:.1}s)"))
}

fn fmt_outcome(o: &Outcome) -> String {
    match o {
        Outcome::Measured {
            load_s,
            ttft_ms,
            tok_per_s,
            total_ms,
        } => format!(
            "{tok_per_s:.1} | {ttft_ms:.0} | {total_ms:.0} | {load_s:.1}"
        ),
        Outcome::Skipped(_) => "— | — | — | —".to_string(),
        Outcome::Failed(_) => "— | — | — | —".to_string(),
    }
}

#[tokio::main]
async fn main() {
    let backends: Vec<String> = env_or("NEURON_BENCH_MATRIX_BACKENDS", "candle,mistralrs")
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    let model_entries: Vec<String> = match std::env::var("NEURON_BENCH_MATRIX_MODELS") {
        Ok(v) => v.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect(),
        Err(_) => default_models(),
    };

    let devices: Vec<String> = match std::env::var("NEURON_BENCH_MATRIX_DEVICES") {
        Ok(v) => v.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect(),
        Err(_) => {
            let default = if cfg!(feature = "cuda") { "cuda" } else { "cpu" };
            vec![default.to_string()]
        }
    };

    let warmup: usize = env_or("NEURON_BENCH_WARMUP", "1").parse().unwrap_or(1);
    let repeats: usize = env_or("NEURON_BENCH_REPEATS", "5").parse().unwrap_or(5);

    let mut cells: Vec<Cell> = Vec::new();
    for entry in &model_entries {
        let Some(dir) = resolve_model_dir(entry) else {
            eprintln!("[matrix] skip (no weights): {entry}");
            continue;
        };
        let model = dir
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| entry.clone());
        for backend in &backends {
            for device in &devices {
                cells.push(Cell {
                    model: model.clone(),
                    model_dir: dir.clone(),
                    backend: backend.clone(),
                    device: device.clone(),
                });
            }
        }
    }

    println!("[matrix] {} cells ({} models × {} backends × {} devices), warmup={warmup} repeats={repeats}",
        cells.len(),
        model_entries.len(),
        backends.len(),
        devices.len());

    let mut rows: Vec<(Cell, Outcome)> = Vec::new();
    for cell in &cells {
        let device = device_from(&cell.device);
        print!("[matrix] {}/{} on {} … ", cell.model, cell.backend, cell.device);
        let out = run_cell(cell, device, warmup, repeats).await;
        match &out {
            Outcome::Measured { tok_per_s, .. } => {
                println!("{tok_per_s:.1} tok/s");
            }
            Outcome::Skipped(why) => println!("skipped — {why}"),
            Outcome::Failed(why) => println!("FAILED — {why}"),
        }
        rows.push((cell.clone(), out));
    }

    // ---- markdown report -------------------------------------------------
    let device_tag = devices.join("-");
    let out_path = env_or(
        "NEURON_BENCH_MATRIX_OUT",
        &format!("target/backend-matrix-{device_tag}.md"),
    );
    let mut md = String::new();
    md.push_str("# Same model × backend matrix\n\n");
    md.push_str(&format!(
        "Host: {} — {} threads. Workload: one fixed prompt, greedy, {MAX_TOKENS} new tokens, \
         streamed; decode tok/s = (chunks − 1) / (total − first chunk); warmup {warmup} \
         discarded, {repeats} measured, p50.\n\n",
        std::env::consts::OS,
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
    ));
    md.push_str("| model | backend | device | decode tok/s | TTFT (ms) | total (ms) | load (s) | status |\n");
    md.push_str("|---|---|---|---:|---:|---:|---:|---|\n");
    for (cell, out) in &rows {
        let (status, numbers) = match out {
            Outcome::Measured { .. } => ("measured".to_string(), fmt_outcome(out)),
            Outcome::Skipped(why) => (format!("skipped — {why}"), fmt_outcome(out)),
            Outcome::Failed(why) => (format!("failed — {why}"), fmt_outcome(out)),
        };
        let mut parts = numbers.split(" | ");
        let tok = parts.next().unwrap_or("—");
        let ttft = parts.next().unwrap_or("—");
        let total = parts.next().unwrap_or("—");
        let load = parts.next().unwrap_or("—");
        md.push_str(&format!(
            "| {} | {} | {} | {tok} | {ttft} | {total} | {load} | {status} |\n",
            cell.model, cell.backend, cell.device,
        ));
    }

    // Coverage: which backend can serve which model at all. This is
    // the half of the question that has no speed number but decides
    // the other half.
    md.push_str("\n## Backend coverage (same weights, no timing)\n\n");
    md.push_str("| model | candle | mistralrs | other backends |\n|---|---|---|---|\n");
    let mut models: Vec<String> = rows.iter().map(|(c, _)| c.model.clone()).collect();
    models.dedup();
    for model in &models {
        md.push_str(&format!("| {model} |"));
        for backend in ["candle", "mistralrs"] {
            let verdict = rows
                .iter()
                .find(|(c, _)| &c.model == model && c.backend == backend)
                .map(|(_, o)| match o {
                    Outcome::Measured { .. } => "servable",
                    Outcome::Skipped(_) => "refused / unavailable",
                    Outcome::Failed(_) => "failed",
                })
                .unwrap_or("not attempted");
            md.push_str(&format!(" {verdict} |"));
        }
        let others: Vec<String> = rows
            .iter()
            .filter(|(c, _)| &c.model == model && c.backend != "candle" && c.backend != "mistralrs")
            .map(|(c, o)| {
                let v = match o {
                    Outcome::Measured { .. } => "servable",
                    Outcome::Skipped(_) => "refused / unavailable",
                    Outcome::Failed(_) => "failed",
                };
                format!("{}: {v}", c.backend)
            })
            .collect();
        md.push_str(&format!(
            " {} |\n",
            if others.is_empty() { "—".to_string() } else { others.join(", ") }
        ));
    }

    if let Some(parent) = Path::new(&out_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(&out_path, &md).expect("write report");
    println!("[matrix] report: {out_path}");

    let failed = rows.iter().filter(|(_, o)| matches!(o, Outcome::Failed(_))).count();
    let skipped = rows.iter().filter(|(_, o)| matches!(o, Outcome::Skipped(_))).count();
    let measured = rows.len() - failed - skipped;
    println!("[matrix] {measured} measured, {skipped} skipped, {failed} failed");
}
