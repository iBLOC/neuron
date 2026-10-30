//! Per-model benchmark harness — one case per model in the capability
//! catalog, measured on the local GPU.
//!
//! # What this is
//!
//! `benches/perf.rs` measures neuron's *data structures*; `benches/model_matrix.rs`
//! drives a running neuron server over HTTP. This harness sits between the
//! two: it loads each model **in-process** through the same public registry
//! API a host would use (`LlmRuntimeRegistry::register` → `runtime_for`), and
//! times the real forward path on the selected device. That makes the numbers
//! the library's own device timings — no HTTP, no server startup, no JSON.
//!
//! The case list is **generated from the capability catalog**
//! (`capabilities::catalog::default_capabilities()`), so every model neuron
//! claims to serve gets a row — including the ones whose weights are not on
//! this machine, which are reported as `skipped` with the reason.
//!
//! # Metrics per capability
//!
//! | capability | workload | headline metric |
//! |---|---|---|
//! | LLM | 1 prompt, greedy, 64 new tokens, streamed | TTFT (ms), decode tok/s |
//! | TTS | one fixed sentence | synth latency, audio s / wall s |
//! | ASR | one fixed WAV | latency, RTF (audio s / wall s) |
//! | Rerank | 1 query × 8 documents | latency, docs/s |
//! | Embedding | 8 short texts | latency, texts/s |
//! | Vision | one 640×640 image | latency, images/s |
//! | OCR | one 640×640 image | latency, images/s |
//! | Decision | 1 question | latency, questions/s |
//! | Chronos2 | one 64-point series | latency, forecasts/s |
//!
//! # Running it
//!
//! ```bash
//! # CUDA build needs the MSVC environment (nvcc needs cl.exe):
//! scripts\cuda-build.bat cargo bench --bench model_bench --features "candle,cuda"
//!
//! # CPU-only:
//! cargo bench --bench model_bench
//! ```
//!
//! # Environment
//!
//! | var | default | meaning |
//! |---|---|---|
//! | `NEURON_BENCH_DEVICE` | `cuda` if the feature is on, else `cpu` | `cuda` / `cpu` |
//! | `NEURON_BENCH_MODELS` | unset | comma-separated model ids to run (default: all) |
//! | `NEURON_BENCH_REPEATS` | `5` | measured iterations per case (after warmup) |
//! | `NEURON_BENCH_WARMUP` | `1` | warmup iterations per case |
//! | `NEURON_BENCH_OUT` | `target/model-bench-<device>.md` | report path |
//! | `NEURON_BENCH_MODEL_DIRS` | unset | colon-separated roots searched for weights |
//! | `NEURON_BENCH_LLM_DIR` | unset | explicit directory for the LLM case |
//! | `NEURON_BENCH_ASR_DIR` | unset | explicit directory for the ASR case |
//! | `NEURON_BENCH_RERANK_DIR` | unset | explicit directory for the rerank case |
//! | `NEURON_BENCH_EMBEDDING_DIR` | unset | explicit directory for the embedding case |
//! | `NEURON_BENCH_TTS_DIR` | unset | explicit directory for the TTS case |
//! | `NEURON_BENCH_VISION_DIR` | unset | explicit directory for the vision case |
//! | `NEURON_BENCH_OCR_DIR` | unset | explicit directory for the OCR case |
//! | `NEURON_BENCH_DECISION_DIR` | unset | explicit directory for the decision case |
//! | `NEURON_BENCH_CHRONOS2_DIR` | unset | explicit directory for the chronos2 case |
//! | `NEURON_BENCH_IMAGE` | unset | PNG/JPEG used for the vision + OCR cases |
//! | `NEURON_BENCH_AUDIO` | unset | WAV used for the ASR case |

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use futures_util::StreamExt;
use neuron_lib::config::DeviceSelection;

// ---------------------------------------------------------------------------
// Result model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum Status {
    Ok,
    Skipped,
    Failed,
}

impl Status {
    fn as_str(self) -> &'static str {
        match self {
            Status::Ok => "ok",
            Status::Skipped => "skipped",
            Status::Failed => "failed",
        }
    }
}

/// One measured row.
#[derive(Debug, Clone)]
struct Row {
    model: String,
    engine: String,
    capability: String,
    status: Status,
    /// The capability-specific headline number (tok/s, docs/s, RTF, …).
    metric: String,
    value: Option<f64>,
    /// Warmup wall time of the first iteration (ms).
    warmup_ms: Option<f64>,
    /// p50 / p95 over the measured iterations (ms).
    p50_ms: Option<f64>,
    p95_ms: Option<f64>,
    /// Time to first token, p50 over the measured iterations (LLM rows).
    ttft_ms: Option<f64>,
    /// The device this case actually ran on.
    device: String,
    /// Process VRAM delta across the case (MiB), when nvidia-smi is present.
    vram_mb: Option<f64>,
    /// Why the case was skipped / failed.
    note: String,
}

impl Row {
    fn skip(model: &str, engine: &str, capability: &str, note: impl Into<String>) -> Self {
        Row {
            model: model.to_string(),
            engine: engine.to_string(),
            capability: capability.to_string(),
            status: Status::Skipped,
            metric: String::new(),
            value: None,
            warmup_ms: None,
            p50_ms: None,
            p95_ms: None,
            ttft_ms: None,
            device: String::new(),
            vram_mb: None,
            note: note.into(),
        }
    }

    fn failed(model: &str, engine: &str, capability: &str, note: impl Into<String>) -> Self {
        Row {
            status: Status::Failed,
            ..Row::skip(model, engine, capability, note)
        }
    }
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

fn env(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.is_empty())
}

fn env_usize(key: &str, default: usize) -> usize {
    env(key).and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// The device a capability runs on.
///
/// `NEURON_BENCH_DEVICE` sets the default for every capability;
/// `NEURON_BENCH_<CAP>_DEVICE` overrides one of them. The override
/// exists because some engines are not GPU-buildable on every host —
/// e.g. the burn (VoxCPM) CUDA backend links against NCCL, which the
/// CUDA toolkit does not ship, so the TTS rows stay on the CPU unless
/// NCCL is installed.
fn device_for(capability: &str, default: DeviceSelection) -> DeviceSelection {
    let key = format!("NEURON_BENCH_{}_DEVICE", capability.to_uppercase());
    match env(&key).as_deref() {
        Some("cuda") | Some("nvidia") | Some("gpu") => DeviceSelection::Nvidia,
        Some("cpu") => DeviceSelection::Cpu,
        _ => default,
    }
}

/// Keep `NEURON_DEVICE` in sync with the device the registry was given:
/// the burn (VoxCPM) runtime reads `NEURON_DEVICE` itself.
fn sync_device_env(device: DeviceSelection) {
    let value = match device {
        DeviceSelection::Nvidia => "nvidia",
        _ => "cpu",
    };
    unsafe { std::env::set_var("NEURON_DEVICE", value) };
}

/// Roots searched for model weights.
fn model_roots() -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Some(v) = env("NEURON_BENCH_MODEL_DIRS") {
        roots.extend(v.split(':').filter(|s| !s.is_empty()).map(PathBuf::from));
    }
    if let Some(v) = env("NEURON_MODEL_DIRS") {
        roots.extend(v.split(':').filter(|s| !s.is_empty()).map(PathBuf::from));
    }
    // The model's own default root (~/.neuron/models) — neuron's puller
    // writes into <root>/<engine>/<id> and <root>/<capability>/<id>.
    roots.push(neuron_lib::config::NeuronConfig::resolved_model_dir());
    roots.dedup();
    roots
}

/// Resolve the weight directory for one case.
///
/// Search order:
/// 1. the capability-specific override (`NEURON_BENCH_<CAP>_DIR`)
/// 2. `<root>/<cap>/<id>` and `<root>/<id>` for every root
/// 3. `<root>/<engine>/<id>` (neuron's own pull layout)
fn resolve_dir(capability: &str, engine: &str, model_id: &str) -> Option<PathBuf> {
    let override_key = format!("NEURON_BENCH_{}_DIR", capability.to_uppercase());
    if let Some(dir) = env(&override_key) {
        let p = PathBuf::from(dir);
        if p.exists() {
            return Some(p);
        }
    }
    let mut candidates: Vec<PathBuf> = Vec::new();
    for root in model_roots() {
        candidates.push(root.join(capability).join(model_id));
        candidates.push(root.join(model_id));
        candidates.push(root.join(engine).join(model_id));
    }
    candidates.into_iter().find(|p| p.is_dir())
}

/// Percentile over a sorted slice.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx]
}

/// Sample the process's VRAM usage through nvidia-smi (MiB).
fn vram_used_mb() -> Option<f64> {
    // `--query-gpu=memory.used` works on WDDM (Windows); the
    // per-process variant (`--query-compute-apps`) does not report there.
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let total: f64 = text
        .lines()
        .filter_map(|l| l.trim().parse::<f64>().ok())
        .sum();
    Some(total)
}

/// GPU name + total VRAM, for the report header.
fn gpu_description() -> String {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total", "--format=csv,noheader"])
        .output();
    match out {
        Ok(o) if o.status.success() => {
            let text = String::from_utf8_lossy(&o.stdout);
            text.lines().next().unwrap_or("").trim().to_string()
        }
        _ => "no nvidia-smi".to_string(),
    }
}

/// Timing loop: one warmup + N measured iterations.
struct Timing {
    warmup_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    /// Whatever the workload wants to report (tok/s, docs/s, …).
    value: f64,
    metric: String,
}

impl Timing {
    fn from(samples: Vec<(f64, f64)>, metric: &str) -> Self {
        let mut times: Vec<f64> = samples.iter().map(|(ms, _)| *ms).collect();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let values: Vec<f64> = samples.iter().map(|(_, v)| *v).collect();
        let value = if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        };
        Timing {
            warmup_ms: samples.first().map(|(ms, _)| *ms).unwrap_or(0.0),
            p50_ms: percentile(&times, 0.50),
            p95_ms: percentile(&times, 0.95),
            value,
            metric: metric.to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const LLM_PROMPT: &str = "用一句话解释什么是神经网络。";

fn chat_request(model: &str) -> neuron_lib::llm::ChatRequest {
    neuron_lib::llm::ChatRequest {
        model: model.to_string(),
        messages: vec![neuron_lib::llm::ChatMessage {
            role: "user".to_string(),
            content: LLM_PROMPT.to_string(),
        }],
        temperature: Some(0.0),
        max_tokens: Some(64),
        top_p: None,
        top_k: None,
        stream: true,
        stop: None,
    }
}

fn rerank_documents() -> Vec<String> {
    (0..8)
        .map(|i| {
            format!(
                "Neuron 是一个用 Rust 编写的本地推理服务，支持 OpenAI 兼容接口。文档段落 {i}。"
            )
        })
        .collect()
}

fn embedding_texts() -> Vec<String> {
    (0..8)
        .map(|i| format!("neuron local inference benchmark sentence number {i}"))
        .collect()
}

/// A 640×640 RGB test image with a few high-contrast blocks.
fn synthetic_png() -> Vec<u8> {
    let mut img = image::RgbImage::new(640, 640);
    for (x, y, p) in img.enumerate_pixels_mut() {
        let band = ((x / 40) + (y / 40)) % 2 == 0;
        *p = if band {
            image::Rgb([235, 235, 235])
        } else {
            image::Rgb([40, 40, 40])
        };
    }
    let mut out = std::io::Cursor::new(Vec::new());
    // PNG encoding needs the `image` crate's png feature; fall back to a
    // minimal encoder path if it is unavailable.
    match image::DynamicImage::ImageRgb8(img).write_to(&mut out, image::ImageFormat::Png) {
        Ok(()) => out.into_inner(),
        Err(_) => Vec::new(),
    }
}

/// A 2-second 16 kHz mono WAV (sine tone) so the ASR path has real audio
/// to chew on even when no speech fixture is available.
fn synthetic_wav() -> Vec<u8> {
    let sample_rate = 16_000u32;
    let secs = 2.0f32;
    let n = (sample_rate as f32 * secs) as usize;
    let mut samples = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / sample_rate as f32;
        samples.push((440.0 * t * std::f32::consts::TAU).sin() * 0.2);
    }
    let data_len = samples.len() * 2;
    let mut wav = Vec::with_capacity(44 + data_len);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len as u32).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&(data_len as u32).to_le_bytes());
    for s in samples {
        wav.extend_from_slice(&((s * 32767.0) as i16).to_le_bytes());
    }
    wav
}

/// Fixture bytes for the ASR case (user file wins over the synthetic tone).
fn audio_fixture() -> Vec<u8> {
    if let Some(p) = env("NEURON_BENCH_AUDIO") {
        if let Ok(bytes) = std::fs::read(&p) {
            return bytes;
        }
    }
    synthetic_wav()
}

/// Fixture bytes for the vision + OCR cases.
fn image_fixture() -> Vec<u8> {
    if let Some(p) = env("NEURON_BENCH_IMAGE") {
        if let Ok(bytes) = std::fs::read(&p) {
            return bytes;
        }
    }
    synthetic_png()
}

// ---------------------------------------------------------------------------
// The cases
// ---------------------------------------------------------------------------

/// Run one capability case and time it.
///
/// `work` returns `(wall_ms, headline_value)` for a single iteration.
async fn time_case<F, Fut>(warmup: usize, repeats: usize, mut work: F) -> Result<Timing, String>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<(f64, f64), String>>,
{
    let mut samples: Vec<(f64, f64)> = Vec::new();
    for i in 0..(warmup + repeats) {
        let t0 = Instant::now();
        let (value, _) = work().await?;
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        if i >= warmup {
            samples.push((ms, value));
        } else {
            // Keep the warmup sample only for the warmup number.
            samples.push((ms, value));
            if i + 1 == warmup {
                // The warmup iteration is reported separately; drop it from
                // the measured set.
                samples.remove(0);
            }
        }
    }
    Ok(Timing::from(samples, ""))
}

/// Per-capability runners. Each returns `Ok(Timing)` or an error string.
async fn run_llm(
    model_id: &str,
    engine: &str,
    dir: Option<PathBuf>,
    device: DeviceSelection,
    warmup: usize,
    repeats: usize,
) -> Result<Timing, String> {
    let dir = dir.ok_or_else(|| "weights not found on this machine".to_string())?;
    unsafe { std::env::set_var("NEURON_LLM_MODEL_PATH", &dir) };

    let mut registry = neuron_lib::llm::registry::LlmRuntimeRegistry::new(
        std::sync::Arc::new(neuron_lib::config::NeuronConfig::default()),
    );
    let impl_id = neuron_lib::llm::LlmImpl::parse(engine).map_err(|e| e.to_string())?;
    registry
        .register(model_id.to_string(), impl_id, device, 0)
        .map_err(|e| e.to_string())?;
    let rt = registry
        .runtime_for(model_id)
        .await
        .map_err(|e| e.to_string())?;

    let mut samples: Vec<(f64, f64)> = Vec::new();
    let mut ttft_ms: Vec<f64> = Vec::new();
    for i in 0..(warmup + repeats) {
        let req = chat_request(model_id);
        let t0 = Instant::now();
        let mut stream = rt.chat_stream(req).await.map_err(|e| e.to_string())?;
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
            // Decode speed: the streamed chunks after the first one.
            let decode_s = (total_ms - first_ms).max(0.0) / 1000.0;
            let tok_per_s = if decode_s > 0.0 {
                (chunks.saturating_sub(1)) as f64 / decode_s
            } else {
                0.0
            };
            samples.push((total_ms, tok_per_s));
            ttft_ms.push(first_ms);
        }
    }
    if samples.is_empty() {
        return Err("no measured iteration".to_string());
    }
    let mut t = Timing::from(samples, "decode tok/s");
    // The TTFT p50 rides in the `warmup_ms` slot of `Timing` — the caller
    // moves it into the row's dedicated TTFT column.
    let mut sorted = ttft_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t.warmup_ms = percentile(&sorted, 0.50);
    Ok(t)
}

async fn run_asr(
    model_id: &str,
    engine: &str,
    dir: Option<PathBuf>,
    device: DeviceSelection,
    warmup: usize,
    repeats: usize,
) -> Result<Timing, String> {
    let dir = dir.ok_or_else(|| "weights not found on this machine".to_string())?;
    unsafe { std::env::set_var("NEURON_ASR_MODEL_PATH", &dir) };
    let audio = audio_fixture();

    let mut registry =
        neuron_lib::asr::registry::AsrRuntimeRegistry::new(std::sync::Arc::new(neuron_lib::config::NeuronConfig::default()));
    let impl_id = neuron_lib::asr::AsrImpl::parse(engine).map_err(|e| e.to_string())?;
    registry
        .register(model_id.to_string(), impl_id, device, 0)
        .map_err(|e| e.to_string())?;
    let rt = registry
        .runtime_for(model_id)
        .await
        .map_err(|e| e.to_string())?;

    let mut samples: Vec<(f64, f64)> = Vec::new();
    for i in 0..(warmup + repeats) {
        let t0 = Instant::now();
        rt.transcribe(&audio, None).await.map_err(|e| e.to_string())?;
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        if i >= warmup {
            // RTF: audio seconds (2 s fixture) per wall second.
            let audio_s = audio.len() as f64 / (16_000.0 * 2.0);
            let rtf = if ms > 0.0 {
                audio_s / (ms / 1000.0)
            } else {
                0.0
            };
            samples.push((ms, rtf));
        }
    }
    Ok(Timing::from(samples, "RTF"))
}

async fn run_rerank(
    model_id: &str,
    engine: &str,
    dir: Option<PathBuf>,
    device: DeviceSelection,
    warmup: usize,
    repeats: usize,
) -> Result<Timing, String> {
    let dir = dir.ok_or_else(|| "weights not found on this machine".to_string())?;
    unsafe { std::env::set_var("NEURON_RERANK_MODEL_PATH", &dir) };

    let mut registry = neuron_lib::rerank::registry::RerankRuntimeRegistry::new(
        std::sync::Arc::new(neuron_lib::config::NeuronConfig::default()),
    );
    let impl_id = neuron_lib::rerank::RerankImpl::parse(engine).map_err(|e| e.to_string())?;
    registry
        .register(model_id.to_string(), impl_id, device, 0)
        .map_err(|e| e.to_string())?;
    let rt = registry
        .runtime_for(model_id)
        .await
        .map_err(|e| e.to_string())?;

    let docs = rerank_documents();
    let mut samples: Vec<(f64, f64)> = Vec::new();
    for i in 0..(warmup + repeats) {
        let req = neuron_lib::rerank::RerankRequest::new(
            model_id,
            "What is neuron?",
            docs.clone(),
        );
        let t0 = Instant::now();
        rt.rerank(req).await.map_err(|e| e.to_string())?;
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        if i >= warmup {
            samples.push((ms, docs.len() as f64 / (ms / 1000.0).max(1e-9)));
        }
    }
    Ok(Timing::from(samples, "docs/s"))
}

async fn run_embedding(
    model_id: &str,
    engine: &str,
    dir: Option<PathBuf>,
    device: DeviceSelection,
    warmup: usize,
    repeats: usize,
) -> Result<Timing, String> {
    let dir = dir.ok_or_else(|| "weights not found on this machine".to_string())?;
    unsafe { std::env::set_var("NEURON_EMBEDDING_MODEL_PATH", &dir) };

    let mut registry = neuron_lib::embedding::registry::EmbeddingRuntimeRegistry::new(
        std::sync::Arc::new(neuron_lib::config::NeuronConfig::default()),
    );
    let impl_id = neuron_lib::embedding::EmbeddingImpl::parse(engine).map_err(|e| e.to_string())?;
    registry
        .register(model_id.to_string(), impl_id, device, 0)
        .map_err(|e| e.to_string())?;
    let rt = registry
        .runtime_for(model_id)
        .await
        .map_err(|e| e.to_string())?;

    let texts = embedding_texts();
    let mut samples: Vec<(f64, f64)> = Vec::new();
    for i in 0..(warmup + repeats) {
        let req = neuron_lib::embedding::EmbeddingRequest::new(
            model_id,
            neuron_lib::embedding::EmbeddingInput::Batch(texts.clone()),
        );
        let t0 = Instant::now();
        rt.embed(req).await.map_err(|e| e.to_string())?;
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        if i >= warmup {
            samples.push((ms, texts.len() as f64 / (ms / 1000.0).max(1e-9)));
        }
    }
    Ok(Timing::from(samples, "texts/s"))
}

async fn run_tts(
    model_id: &str,
    engine: &str,
    dir: Option<PathBuf>,
    device: DeviceSelection,
    warmup: usize,
    repeats: usize,
) -> Result<Timing, String> {
    let dir = dir.ok_or_else(|| "weights not found on this machine".to_string())?;
    unsafe { std::env::set_var("NEURON_TTS_MODEL_PATH", &dir) };

    let mut registry =
        neuron_lib::tts::registry::TtsRuntimeRegistry::new(std::sync::Arc::new(neuron_lib::config::NeuronConfig::default()));
    let impl_id = neuron_lib::tts::TtsImpl::parse(engine).map_err(|e| e.to_string())?;
    registry
        .register(model_id.to_string(), impl_id, device, 0)
        .map_err(|e| e.to_string())?;
    let rt = registry
        .runtime_for(model_id)
        .await
        .map_err(|e| e.to_string())?;

    let text = "神经元是一个本地优先的推理服务。";
    let mut samples: Vec<(f64, f64)> = Vec::new();
    for i in 0..(warmup + repeats) {
        let t0 = Instant::now();
        let audio = rt.synthesize(text, None).await.map_err(|e| e.to_string())?;
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        if i >= warmup {
            // audio seconds produced per wall second
            let audio_s = audio.audio.len() as f64 / (audio.sample_rate as f64 * 2.0);
            samples.push((ms, audio_s / (ms / 1000.0).max(1e-9)));
        }
    }
    Ok(Timing::from(samples, "audio s / wall s"))
}

async fn run_vision(
    model_id: &str,
    engine: &str,
    dir: Option<PathBuf>,
    device: DeviceSelection,
    warmup: usize,
    repeats: usize,
) -> Result<Timing, String> {
    let dir = dir.ok_or_else(|| "weights not found on this machine".to_string())?;
    unsafe { std::env::set_var("NEURON_VISION_MODEL_PATH", &dir) };

    let mut registry = neuron_lib::vision::VisionRuntimeRegistry::new(
        std::sync::Arc::new(neuron_lib::config::NeuronConfig::default()),
    );
    let impl_id = neuron_lib::vision::VisionImpl::parse(engine).map_err(|e| e.to_string())?;
    registry
        .register(model_id.to_string(), impl_id, device, 0)
        .map_err(|e| e.to_string())?;
    let rt = registry
        .runtime_for(model_id)
        .await
        .map_err(|e| e.to_string())?;

    let image = image_fixture();
    let b64 = base64_encode(&image);
    let mut samples: Vec<(f64, f64)> = Vec::new();
    for i in 0..(warmup + repeats) {
        let req = neuron_lib::vision::VisionRequest {
            model: model_id.to_string(),
            task: neuron_lib::vision::VisionTask::Detect,
            image: neuron_lib::vision::VisionImageSource::Base64 {
                media_type: "image/png".to_string(),
                data: b64.clone(),
            },
            confidence_threshold: Some(0.25),
            iou_threshold: Some(0.45),
            max_detections: Some(100),
            classes: None,
            lora_adapters: None,
        };
        let t0 = Instant::now();
        rt.run(req).await.map_err(|e| e.to_string())?;
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        if i >= warmup {
            samples.push((ms, 1000.0 / ms.max(1e-9)));
        }
    }
    Ok(Timing::from(samples, "images/s"))
}

async fn run_ocr(
    model_id: &str,
    engine: &str,
    dir: Option<PathBuf>,
    device: DeviceSelection,
    warmup: usize,
    repeats: usize,
) -> Result<Timing, String> {
    let dir = dir.ok_or_else(|| "weights not found on this machine".to_string())?;
    unsafe { std::env::set_var("NEURON_OCR_MODEL_PATH", &dir) };

    let mut registry =
        neuron_lib::ocr::registry::OcrRuntimeRegistry::new(std::sync::Arc::new(neuron_lib::config::NeuronConfig::default()));
    let impl_id = neuron_lib::ocr::OcrImpl::parse(engine).map_err(|e| e.to_string())?;
    registry
        .register(model_id.to_string(), impl_id, device, 0)
        .map_err(|e| e.to_string())?;
    let rt = registry
        .runtime_for(model_id)
        .await
        .map_err(|e| e.to_string())?;

    let image = image_fixture();
    let b64 = base64_encode(&image);
    let mut samples: Vec<(f64, f64)> = Vec::new();
    for i in 0..(warmup + repeats) {
        let req = neuron_lib::ocr::OcrRequest {
            model: model_id.to_string(),
            image: neuron_lib::vision::VisionImageSource::Base64 {
                media_type: "image/png".to_string(),
                data: b64.clone(),
            },
            languages: None,
            det_thresh: None,
            det_box_thresh: None,
            det_unclip_ratio: None,
            det_max_side: None,
            rec_batch_size: None,
        };
        let t0 = Instant::now();
        rt.run(req).await.map_err(|e| e.to_string())?;
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        if i >= warmup {
            samples.push((ms, 1000.0 / ms.max(1e-9)));
        }
    }
    Ok(Timing::from(samples, "images/s"))
}

async fn run_decision(
    model_id: &str,
    engine: &str,
    dir: Option<PathBuf>,
    device: DeviceSelection,
    warmup: usize,
    repeats: usize,
) -> Result<Timing, String> {
    let dir = dir.ok_or_else(|| "weights not found on this machine".to_string())?;
    unsafe { std::env::set_var("NEURON_DECISION_MODEL_PATH", &dir) };

    let mut registry = neuron_lib::decision::registry::DecisionRuntimeRegistry::new(
        std::sync::Arc::new(neuron_lib::config::NeuronConfig::default()),
    );
    let impl_id = neuron_lib::decision::DecisionImpl::parse(engine).map_err(|e| e.to_string())?;
    registry
        .register(
            model_id.to_string(),
            impl_id,
            device,
            0,
            dir.to_string_lossy().to_string(),
        )
        .map_err(|e| e.to_string())?;
    let rt = registry
        .runtime_for(model_id)
        .await
        .map_err(|e| e.to_string())?;

    let mut samples: Vec<(f64, f64)> = Vec::new();
    for i in 0..(warmup + repeats) {
        let req = neuron_lib::decision::DecisionRequest {
            model: model_id.to_string(),
            state: serde_json::json!(["the user asked for a status update"]),
            questions: vec![neuron_lib::decision::DecisionQuestion {
                qtype: neuron_lib::decision::QuestionType::Noul,
                instructions: serde_json::json!("Should the agent reply now?"),
                criteria: None,
                labels: None,
                options: None,
                option_order: None,
            }],
            max_len: None,
            head_max_len: None,
        };
        let t0 = Instant::now();
        rt.decide(req).await.map_err(|e| e.to_string())?;
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        if i >= warmup {
            samples.push((ms, 1000.0 / ms.max(1e-9)));
        }
    }
    Ok(Timing::from(samples, "questions/s"))
}

#[cfg(feature = "chronos2")]
async fn run_chronos2(
    model_id: &str,
    _engine: &str,
    dir: Option<PathBuf>,
    device: DeviceSelection,
    warmup: usize,
    repeats: usize,
) -> Result<Timing, String> {
    let dir = dir.ok_or_else(|| "weights not found on this machine".to_string())?;
    unsafe { std::env::set_var("NEURON_CHRONOS2_MODEL_PATH", &dir) };

    let mut registry = neuron_lib::chronos2::Chronos2Registry::new();
    registry
        .register(model_id.to_string(), device, 0)
        .map_err(|e| e.to_string())?;
    let rt = registry.runtime_for(model_id).await;

    let series: Vec<f64> = (0..64).map(|i| (i as f64 * 0.5).sin()).collect();
    let mut samples: Vec<(f64, f64)> = Vec::new();
    for i in 0..(warmup + repeats) {
        let req = neuron_lib::chronos2::ForecastRequest {
            model: model_id.to_string(),
            series: series.clone(),
            horizon: 12,
            method: neuron_lib::chronos2::ForecastMethod::Median,
        };
        let t0 = Instant::now();
        rt.forecast(req).await.map_err(|e| e.to_string())?;
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        if i >= warmup {
            samples.push((ms, 1000.0 / ms.max(1e-9)));
        }
    }
    Ok(Timing::from(samples, "forecasts/s"))
}

// ---------------------------------------------------------------------------
// Base64 (the bench target does not depend on the `base64` crate directly)
// ---------------------------------------------------------------------------

fn base64_encode(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(TABLE[((n >> 18) & 63) as usize] as char);
        out.push(TABLE[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            TABLE[((n >> 6) & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            TABLE[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

/// Format a measured value.
///
/// Small ratios (e.g. "audio seconds per wall second" for a multi-minute
/// CPU synthesis) need more precision than 2 digits to say anything at
/// all, so they get 4. `whole` prints an integer count (the VRAM column
/// is MiB, never a ratio).
fn fmt_opt(v: Option<f64>, digits: usize) -> String {
    match v {
        Some(x) if x.abs() < 1.0 => format!("{x:.4}"),
        Some(x) => format!("{x:.digits$}"),
        None => "—".to_string(),
    }
}

fn fmt_int(v: Option<f64>) -> String {
    match v {
        Some(x) => format!("{x:.0}"),
        None => "—".to_string(),
    }
}

fn write_report(rows: &[Row], device: &str, path: &Path) -> std::io::Result<()> {
    let mut out = String::new();
    out.push_str("# Per-model benchmark — neuron\n\n");
    out.push_str(&format!(
        "- **Device**: {device}\n- **GPU**: {}\n- **Generated**: {}\n",
        gpu_description(),
        chrono_stamp()
    ));
    // The knobs that change the numbers — recorded so a report can be
    // reproduced (or not) later.
    let mut cfg: Vec<String> = Vec::new();
    for key in [
        "NEURON_LLM_DTYPE",
        "NEURON_LLAMA_CPP_N_CTX",
        "NEURON_MISTRALRS_ISQ",
        "NEURON_LLAMA_CPP_GPU_LAYERS",
        "NEURON_USE_FLASH_ATTN",
        "NEURON_ASR_SHERPA_NUM_THREADS",
        "NEURON_OCR_EXECUTOR",
    ] {
        if let Some(v) = env(key) {
            cfg.push(format!("- `{key}` = `{v}`"));
        }
    }
    if !cfg.is_empty() {
        out.push_str("\n### Knobs\n\n");
        for line in cfg {
            out.push_str(&line);
            out.push('\n');
        }
    }
    out.push_str(&format!(
        "- **Cases**: {} (ok {} / skipped {} / failed {})\n\n",
        rows.len(),
        rows.iter().filter(|r| r.status == Status::Ok).count(),
        rows.iter().filter(|r| r.status == Status::Skipped).count(),
        rows.iter().filter(|r| r.status == Status::Failed).count(),
    ));
    out.push_str("Warmup iterations are excluded from p50/p95. `metric` is the capability-specific\n");
    out.push_str("headline number; `p50`/`p95` are wall-clock milliseconds per iteration.\n\n");

    // Group by capability.
    let mut by_cap: BTreeMap<String, Vec<&Row>> = BTreeMap::new();
    for r in rows {
        by_cap.entry(r.capability.clone()).or_default().push(r);
    }

    for (cap, mut group) in by_cap {
        group.sort_by(|a, b| a.model.cmp(&b.model));
        out.push_str(&format!("## {cap}\n\n"));
        out.push_str("| model | engine | status | metric | value | p50 ms | p95 ms | VRAM MiB | note |\n");
        out.push_str("|---|---|---|---|---|---|---|---|---|\n");
        for r in group {
            out.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                r.model,
                r.engine,
                r.status.as_str(),
                if r.metric.is_empty() {
                    "—".to_string()
                } else {
                    r.metric.clone()
                },
                fmt_opt(r.value, 2),
                fmt_opt(r.p50_ms, 1),
                fmt_opt(r.p95_ms, 1),
                fmt_int(r.vram_mb),
                r.note.replace('|', "\\|"),
            ));
        }
        out.push('\n');
    }

    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(path, out)
}

fn chrono_stamp() -> String {
    // No chrono dependency in the bench target — use the system clock.
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("unix-epoch {now}")
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let device_name = env("NEURON_BENCH_DEVICE").unwrap_or_else(|| {
        if cfg!(feature = "cuda") {
            "cuda".to_string()
        } else {
            "cpu".to_string()
        }
    });
    let device = match device_name.as_str() {
        "cuda" | "nvidia" | "gpu" => DeviceSelection::Nvidia,
        _ => DeviceSelection::Cpu,
    };

    let warmup = env_usize("NEURON_BENCH_WARMUP", 1);
    let repeats = env_usize("NEURON_BENCH_REPEATS", 5);
    let only: Vec<String> = env("NEURON_BENCH_MODELS")
        .map(|v| v.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_default();

    let out_path = PathBuf::from(env("NEURON_BENCH_OUT").unwrap_or_else(|| {
        format!("target/model-bench-{device_name}.md")
    }));

    println!(
        "[model-bench] device={device_name} warmup={warmup} repeats={repeats} gpu={}",
        gpu_description()
    );

    let caps = neuron_lib::capabilities::catalog::default_capabilities();
    let mut rows: Vec<Row> = Vec::new();

    for cap in caps {
        let model_id = cap.model_id.clone();
        if !only.is_empty() && !only.iter().any(|m| m == &model_id) {
            continue;
        }
        let engine = cap.engine_impl.clone();
        // The capability is derived from the modality set + engine so the
        // table groups sensibly.
        let capability = capability_of(&cap);

        let dir = resolve_dir(&capability, &engine, &model_id);
        let device = device_for(&capability, device);
        // Per-capability iteration count: a 2B-parameter TTS synthesis
        // takes minutes on the CPU, so the TTS rows get their own budget
        // (NEURON_BENCH_REPEATS_TTS, default 1).
        let repeats = {
            let key = format!("NEURON_BENCH_REPEATS_{}", capability.to_uppercase());
            env_usize(&key, repeats)
        };
        sync_device_env(device);
        let vram_before = vram_used_mb();

        // Run the case inside a spawned tokio task: a native panic deep in
        // an engine (candle's llama loader `unwrap()`s shape mismatches —
        // MiniCPM5-1B's q_proj is [2048,1536] vs the expected [1536,1536])
        // then surfaces as a JoinError for THIS case instead of killing
        // the whole sweep. Cases run sequentially, so the process-global
        // env-var setup inside the runners stays safe.
        let case_model = model_id.clone();
        let case_engine = engine.clone();
        let case_capability = capability.clone();
        let joined = tokio::spawn(async move {
            match case_capability.as_str() {
                "llm" => run_llm(&case_model, &case_engine, dir.clone(), device, warmup, repeats)
                    .await
                    .map(|t| (Status::Ok, Some(t), String::new()))
                    .map_err(|e| (Status::Skipped, None, e)),
                "asr" => run_asr(&case_model, &case_engine, dir.clone(), device, warmup, repeats)
                    .await
                    .map(|t| (Status::Ok, Some(t), String::new()))
                    .map_err(|e| (Status::Skipped, None, e)),
                "rerank" => {
                    run_rerank(&case_model, &case_engine, dir.clone(), device, warmup, repeats)
                        .await
                        .map(|t| (Status::Ok, Some(t), String::new()))
                        .map_err(|e| (Status::Skipped, None, e))
                }
                "embedding" => {
                    run_embedding(&case_model, &case_engine, dir.clone(), device, warmup, repeats)
                        .await
                        .map(|t| (Status::Ok, Some(t), String::new()))
                        .map_err(|e| (Status::Skipped, None, e))
                }
                "tts" => run_tts(&case_model, &case_engine, dir.clone(), device, warmup, repeats)
                    .await
                    .map(|t| (Status::Ok, Some(t), String::new()))
                    .map_err(|e| (Status::Skipped, None, e)),
                "vision" => {
                    run_vision(&case_model, &case_engine, dir.clone(), device, warmup, repeats)
                        .await
                        .map(|t| (Status::Ok, Some(t), String::new()))
                        .map_err(|e| (Status::Skipped, None, e))
                }
                "ocr" => run_ocr(&case_model, &case_engine, dir.clone(), device, warmup, repeats)
                    .await
                    .map(|t| (Status::Ok, Some(t), String::new()))
                    .map_err(|e| (Status::Skipped, None, e)),
                "decision" => {
                    run_decision(&case_model, &case_engine, dir.clone(), device, warmup, repeats)
                        .await
                        .map(|t| (Status::Ok, Some(t), String::new()))
                        .map_err(|e| (Status::Skipped, None, e))
                }
                #[cfg(feature = "chronos2")]
                "chronos2" => {
                    run_chronos2(&case_model, &case_engine, dir.clone(), device, warmup, repeats)
                        .await
                        .map(|t| (Status::Ok, Some(t), String::new()))
                        .map_err(|e| (Status::Skipped, None, e))
                }
                _ => Err((
                    Status::Skipped,
                    None,
                    "no in-process runner for this capability".to_string(),
                )),
            }
        });
        let (status, timing, note) = match joined.await {
            Ok(Ok(triple)) => triple,
            Ok(Err(triple)) => triple,
            Err(join_err) => (
                Status::Failed,
                None,
                match join_err.try_into_panic() {
                    Ok(panic) => {
                        let msg = panic
                            .downcast_ref::<&str>()
                            .map(|s| s.to_string())
                            .or_else(|| panic.downcast_ref::<String>().cloned())
                            .unwrap_or_else(|| "engine panicked".to_string());
                        format!("engine panicked (process survived): {msg}")
                    }
                    Err(_) => "case task cancelled".to_string(),
                },
            ),
        };

        let vram_after = vram_used_mb();
        let vram_delta = match (vram_before, vram_after) {
            (Some(a), Some(b)) => Some(b - a),
            _ => None,
        };

        let mut row = Row {
            model: model_id.clone(),
            engine: engine.clone(),
            capability: capability.clone(),
            status,
            metric: timing.as_ref().map(|t| t.metric.clone()).unwrap_or_default(),
            value: timing.as_ref().map(|t| t.value),
            warmup_ms: None,
            p50_ms: timing.as_ref().map(|t| t.p50_ms),
            p95_ms: timing.as_ref().map(|t| t.p95_ms),
            ttft_ms: None,
            device: match device {
                DeviceSelection::Nvidia => "cuda".to_string(),
                DeviceSelection::Cpu => "cpu".to_string(),
                _ => format!("{device:?}"),
            },
            vram_mb: vram_delta,
            note: note.clone(),
        };
        if capability == "llm" {
            // For the LLM rows the timing's warmup slot carries the TTFT p50.
            row.ttft_ms = timing.as_ref().map(|t| t.warmup_ms);
        }
        println!(
            "[model-bench] {:<28} {:<12} {:<8} {}",
            model_id,
            engine,
            status.as_str(),
            if let Some(t) = &timing {
                format!(
                    "{} = {:.2} | p50 {:.1} ms | p95 {:.1} ms",
                    t.metric, t.value, t.p50_ms, t.p95_ms
                )
            } else {
                note.clone()
            }
        );
        rows.push(row);
        // Write the report after every case: one engine that aborts the
        // process (a native ONNX Runtime exception, say) must not take
        // the whole sweep's numbers with it.
        if let Err(e) = write_report(&rows, &device_name, &out_path) {
            eprintln!("[model-bench] failed to write {}: {e}", out_path.display());
        }
    }

    // Chronos2 has no catalog entry by design (nothing downloadable loads
    // as-is — the checkpoint comes from the upstream Chronos2 release), so
    // the catalog loop never produces a case for it. Run one manually when
    // the feature is compiled in AND a checkpoint is reachable.
    #[cfg(feature = "chronos2")]
    {
        const CHRONOS2_ID: &str = "chronos-t5-small";
        let dir = resolve_dir("chronos2", "candle", CHRONOS2_ID);
        let chrono_device = device_for("chronos2", device);
        sync_device_env(chrono_device);
        let vram_before = vram_used_mb();
        let (status, timing, note) =
            match run_chronos2(CHRONOS2_ID, "candle", dir, chrono_device, warmup, repeats).await {
                Ok(t) => (Status::Ok, Some(t), String::new()),
                Err(e) => (Status::Skipped, None, e),
            };
        let vram_after = vram_used_mb();
        let vram_mb = match (vram_before, vram_after) {
            (Some(a), Some(b)) => Some(b - a),
            _ => None,
        };
        let row = Row {
            model: CHRONOS2_ID.to_string(),
            engine: "candle".to_string(),
            capability: "chronos2".to_string(),
            status,
            metric: timing.as_ref().map(|t| t.metric.clone()).unwrap_or_default(),
            value: timing.as_ref().map(|t| t.value),
            warmup_ms: None,
            p50_ms: timing.as_ref().map(|t| t.p50_ms),
            p95_ms: timing.as_ref().map(|t| t.p95_ms),
            ttft_ms: None,
            device: match chrono_device {
                DeviceSelection::Nvidia => "cuda".to_string(),
                DeviceSelection::Cpu => "cpu".to_string(),
                _ => format!("{chrono_device:?}"),
            },
            vram_mb,
            note,
        };
        println!(
            "[model-bench] {:<28} {:<12} {:<8} {}",
            CHRONOS2_ID,
            "candle",
            status.as_str(),
            if let Some(t) = &timing {
                format!("{} = {:.2} | p50 {:.1} ms", t.metric, t.value, t.p50_ms)
            } else {
                row.note.clone()
            }
        );
        rows.push(row);
        if let Err(e) = write_report(&rows, &device_name, &out_path) {
            eprintln!("[model-bench] failed to write {}: {e}", out_path.display());
        }
    }

    if let Err(e) = write_report(&rows, &device_name, &out_path) {
        eprintln!("[model-bench] failed to write {}: {e}", out_path.display());
        std::process::exit(1);
    }
    println!("[model-bench] report written to {}", out_path.display());
}

/// Map a capability entry to the benchmark group it belongs to.
fn capability_of(cap: &neuron_lib::capabilities::model_caps::ModelCapabilities) -> String {
    use neuron_lib::capabilities::Modality;
    let has = |m: Modality| cap.input_modalities.contains(&m);
    if has(Modality::AudioInput) {
        "asr".to_string()
    } else if has(Modality::ImageInput) {
        if cap.engine_impl == "paddle-ocr" {
            "ocr".to_string()
        } else {
            "vision".to_string()
        }
    } else if cap.output_modalities.contains(&Modality::EmbeddingOutput) {
        // The rerank/embedding split is by the API style the entry
        // declares, NOT by engine_impl — the three reranker families
        // run on three different engines (candle / mxbai / jina).
        if has(Modality::TextInput) && cap.api_styles.iter().any(|s| s.as_str() == "openai-rerank") {
            "rerank".to_string()
        } else {
            "embedding".to_string()
        }
    } else if cap.output_modalities.contains(&Modality::AudioOutput) {
        "tts".to_string()
    } else if cap.engine_impl == "tract" && cap.model_id.starts_with("laya") {
        "decision".to_string()
    } else if cap.engine_impl == "candle" {
        "llm".to_string()
    } else {
        "llm".to_string()
    }
}
