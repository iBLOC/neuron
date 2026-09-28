//! ASR speed benchmark: neuron's own tract ONNX path vs sherpa-onnx.
//!
//! Both engines run the **same exported SenseVoiceSmall graph** (fp32
//! `model.onnx` by default) on the same WAV files, in-process:
//!
//! * `tract`  -- `TractSenseVoiceRuntime`: in-repo fbank / LFR / CMVN,
//!   tract forward, in-repo greedy CTC (feature `tract`).
//! * `sherpa` -- `SherpaAsrRuntime`: the official sherpa-onnx Rust
//!   binding; feature extraction through decoding all live inside
//!   onnxruntime-backed sherpa-onnx (feature `sherpa`).
//!
//! The gap between the two columns is "the price of running the ONNX
//! ourselves with tract" -- what a host saves or pays by dropping the
//! sherpa-onnx C++ dependency (and its bundled onnxruntime) in favour of
//! the pure-Rust stack. The tract column also splits fbank / forward /
//! decode so the report can attribute where the time goes.
//!
//! ## Why this is not a Criterion bench
//!
//! Same reason as `llm_speed.rs`: one measurement is a full
//! utterance-sized forward pass (hundreds of ms), so we run a small
//! fixed number of repeats and compute p50/p95 ourselves. Both engines
//! get identical warmup/repeat treatment.
//!
//! ## Parity check
//!
//! Speed without correctness is noise, so after the timing loop the
//! report compares the two engines' transcriptions per WAV after
//! stripping whitespace and the SentencePiece ▁ marker. A mismatch
//! flags the row instead of silently pretending both engines agree.
//!
//! ## Run
//!
//! ```bash
//! cargo bench --bench asr_speed --features "tract,sherpa"
//! # or via the driver script (downloads the model first if missing):
//! ./scripts/run-asr-speed-compare.sh
//! ```
//!
//! Environment: `NEURON_ASR_MODEL_PATH` (model dir), `NEURON_ASR_BENCH_WAVS`
//! (comma-separated WAVs; default: all *.wav in the model dir),
//! `NEURON_ASR_ONNX_FILE` (graph file; default `model.onnx`),
//! `NEURON_ASR_BENCH_PROFILE` = quick | standard | full. The report lands
//! in `target/asr-speed-compare.md`.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use neuron_lib::asr::wav::decode_wav;
use neuron_lib::asr::AsrRuntime;

#[cfg(feature = "tract")]
use neuron_lib::asr::TractSenseVoiceRuntime;
#[cfg(feature = "sherpa")]
use neuron_lib::asr::SherpaAsrRuntime;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct Knobs {
    warmup: usize,
    repeats: usize,
}

impl Knobs {
    fn from_env() -> Self {
        let (def_warmup, def_repeats) = match std::env::var("NEURON_ASR_BENCH_PROFILE").as_deref() {
            Ok("quick") => (0, 1),
            Ok("full") => (2, 7),
            _ => (1, 3),
        };
        Self {
            warmup: env_usize("NEURON_ASR_BENCH_WARMUP", def_warmup),
            repeats: env_usize("NEURON_ASR_BENCH_REPEATS", def_repeats),
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
// Engines
// ---------------------------------------------------------------------------

/// Per-stage wall-clock split (tract only; sherpa's internals are opaque).
#[derive(Debug, Clone, Copy, Default)]
struct StageSplit {
    feature_us: u128,
    forward_us: u128,
    decode_us: u128,
}

enum Engine {
    #[cfg(feature = "tract")]
    Tract {
        rt: TractSenseVoiceRuntime,
        load: Duration,
    },
    #[cfg(feature = "sherpa")]
    Sherpa {
        rt: SherpaAsrRuntime,
        load: Duration,
    },
}

impl Engine {
    fn name(&self) -> &'static str {
        match self {
            #[cfg(feature = "tract")]
            Engine::Tract { .. } => "tract",
            #[cfg(feature = "sherpa")]
            Engine::Sherpa { .. } => "sherpa",
        }
    }

    fn load_time(&self) -> Duration {
        match self {
            #[cfg(feature = "tract")]
            Engine::Tract { load, .. } => *load,
            #[cfg(feature = "sherpa")]
            Engine::Sherpa { load, .. } => *load,
        }
    }

    /// One full audio -> text pass. Returns the wall time, the stage
    /// split (zeros for engines that cannot report one) and the text.
    async fn transcribe_once(&self, wav: &WavFile) -> Result<(Duration, StageSplit, String), String> {
        match self {
            #[cfg(feature = "tract")]
            Engine::Tract { rt, .. } => {
                let t0 = Instant::now();
                let (out, st) = rt
                    .transcribe_timed(&wav.samples, None)
                    .map_err(|e| e.message)?;
                let wall = t0.elapsed();
                Ok((
                    wall,
                    StageSplit { feature_us: st.feature_us, forward_us: st.forward_us, decode_us: st.decode_us },
                    out.text,
                ))
            }
            #[cfg(feature = "sherpa")]
            Engine::Sherpa { rt, .. } => {
                let t0 = Instant::now();
                let out = rt.transcribe(&wav.bytes, None).await.map_err(|e| e.message)?;
                Ok((t0.elapsed(), StageSplit::default(), out.text))
            }
        }
    }

    /// Build every engine compiled into this run. The concrete
    /// `from_dir` constructors are used so the benchmark keeps the
    /// concrete types (tract's stage timings are not on the trait).
    fn build_all(model_dir: &PathBuf) -> Vec<Engine> {
        // Both from_dir implementations read NEURON_ASR_MODEL_PATH, and
        // both honor NEURON_ASR_ONNX_FILE, so the run is pinned to one graph.
        // std::env::set_var is unsafe on edition 2024
        unsafe { std::env::set_var("NEURON_ASR_MODEL_PATH", model_dir) };
        let mut engines = Vec::new();

        #[cfg(feature = "tract")]
        {
            let t0 = Instant::now();
            unsafe { std::env::set_var("NEURON_ASR_ONNX_FILE", "model.prepped.onnx") };
            match TractSenseVoiceRuntime::from_dir("sensevoice-small", model_dir) {
                Ok(rt) => {
                    let load = t0.elapsed();
                    println!("[asr-speed] tract: loaded {} in {load:.1?}", rt.onnx_file.display());
                    engines.push(Engine::Tract { rt, load });
                }
                Err(e) => println!("[asr-speed] tract: NOT AVAILABLE ({})", e.message),
            }
        }
        #[cfg(not(feature = "tract"))]
        let _ = model_dir;

        #[cfg(feature = "sherpa")]
        {
            let t0 = Instant::now();
            unsafe { std::env::set_var("NEURON_ASR_ONNX_FILE", "model.onnx") };
            match SherpaAsrRuntime::from_dir(model_dir) {
                Ok(rt) => {
                    let load = t0.elapsed();
                    println!("[asr-speed] sherpa: loaded in {load:.1?}");
                    engines.push(Engine::Sherpa { rt, load });
                }
                Err(e) => println!("[asr-speed] sherpa: NOT AVAILABLE ({})", e.message),
            }
        }

        engines
    }
}

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Measurement {
    samples: Vec<Duration>,
    stages: Vec<StageSplit>,
    text: String,
}

impl Measurement {
    fn percentile_us(&self, pct: f64) -> f64 {
        percentile(&self.samples, pct).as_micros() as f64
    }
    fn median_stages(&self) -> StageSplit {
        let n = self.stages.len().max(1) as u128;
        StageSplit {
            feature_us: self.stages.iter().map(|s| s.feature_us).sum::<u128>() / n,
            forward_us: self.stages.iter().map(|s| s.forward_us).sum::<u128>() / n,
            decode_us: self.stages.iter().map(|s| s.decode_us).sum::<u128>() / n,
        }
    }
}

fn percentile(sorted: &[Duration], pct: f64) -> Duration {
    let mut s = sorted.to_vec();
    s.sort();
    let idx = ((pct / 100.0) * (s.len().saturating_sub(1)) as f64).round() as usize;
    s.get(idx).copied().unwrap_or(Duration::ZERO)
}

async fn measure(engine: &Engine, wav: &WavFile, knobs: &Knobs) -> Result<Measurement, String> {
    for _ in 0..knobs.warmup {
        engine.transcribe_once(wav).await?;
    }
    let mut samples = Vec::with_capacity(knobs.repeats);
    let mut stages = Vec::with_capacity(knobs.repeats);
    let mut text = String::new();
    for _ in 0..knobs.repeats {
        let (t, st, tx) = engine.transcribe_once(wav).await?;
        samples.push(t);
        stages.push(st);
        text = tx;
    }
    Ok(Measurement { samples, stages, text })
}

// ---------------------------------------------------------------------------
// WAV loading
// ---------------------------------------------------------------------------

struct WavFile {
    name: String,
    bytes: Vec<u8>,
    samples: Vec<f32>,
    duration_secs: f64,
}

fn load_wavs(model_dir: &PathBuf) -> Result<Vec<WavFile>, String> {
    let list = match std::env::var("NEURON_ASR_BENCH_WAVS") {
        Ok(v) if !v.is_empty() => v.split(',').map(|s| PathBuf::from(s.trim())).collect::<Vec<_>>(),
        _ => {
            let mut found: Vec<PathBuf> = std::fs::read_dir(model_dir)
                .map_err(|e| format!("read {}: {e}", model_dir.display()))?
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().map(|e| e == "wav").unwrap_or(false))
                .collect();
            found.sort();
            found
        }
    };
    let mut wavs = Vec::new();
    for path in list {
        let bytes = std::fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let (samples, sample_rate) = decode_wav(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        let duration_secs = samples.len() as f64 / sample_rate as f64;
        wavs.push(WavFile {
            name: path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
            bytes,
            samples,
            duration_secs,
        });
    }
    if wavs.is_empty() {
        return Err(format!(
            "no WAV files found in {} (or NEURON_ASR_BENCH_WAVS is empty)",
            model_dir.display()
        ));
    }
    Ok(wavs)
}

fn normalize_for_parity(text: &str) -> String {
    text.chars().filter(|c| !c.is_whitespace() && *c != '\u{2581}').collect()
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let knobs = Knobs::from_env();
    let model_dir = PathBuf::from(env_or("NEURON_ASR_MODEL_PATH", "models/sherpa-sensevoice"));
    let onnx_file = env_or("NEURON_ASR_ONNX_FILE", "model.onnx");

    println!("=== neuron ASR speed compare ===");
    println!("model dir: {}", model_dir.display());
    println!("graph: {onnx_file} (both engines pinned to the same file)");
    println!("profile: warmup={} repeats={}", knobs.warmup, knobs.repeats);

    let wavs = match load_wavs(&model_dir) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("[asr-speed] {e}");
            eprintln!("[asr-speed] run scripts/run-asr-speed-compare.sh first, or set NEURON_ASR_MODEL_PATH");
            std::process::exit(2);
        }
    };

    // Single-threaded runtime: the engines are CPU-bound and the comparison
    // is fairest when nothing else spins on the same cores.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    let engines = Engine::build_all(&model_dir);

    if engines.is_empty() {
        println!("[asr-speed] no engines available: build with --features \"tract,sherpa\"");
        std::process::exit(3);
    }

    let mut results: Vec<(&'static str, String, Measurement)> = Vec::new();

    for wav in &wavs {
        println!("[asr-speed] {} ({:.2}s audio)", wav.name, wav.duration_secs);
        for engine in &engines {
            let m = match rt.block_on(measure(engine, wav, &knobs)) {
                Ok(m) => m,
                Err(e) => {
                    println!("[asr-speed]   {}: FAILED ({e})", engine.name());
                    continue;
                }
            };
            println!(
                "[asr-speed]   {:>7}: p50 {:>8.1} ms | RTF {:.3} | text: {}",
                engine.name(),
                m.percentile_us(50.0) / 1000.0,
                (m.percentile_us(50.0) / 1_000_000.0) / wav.duration_secs,
                truncate(&m.text, 40),
            );
            results.push((engine.name(), wav.name.clone(), m));
        }
    }

    // ---- report ----------------------------------------------------------
    let mut report = String::new();
    let _ = writeln!(report, "# ASR speed compare — SenseVoiceSmall");
    let _ = writeln!(report);
    let _ = writeln!(report, "Engine-by-engine: `tract` runs `model.prepped.onnx` (dynamic PE moved out by `scripts/prep-sensevoice-onnx.py` into a `pe` graph input), `sherpa` runs the original `model.onnx`.");
    let _ = writeln!(report);
    let _ = writeln!(report, "## Load time");
    let _ = writeln!(report);
    let _ = writeln!(report, "| engine | load (graph read + optimize) |");
    let _ = writeln!(report, "|---|---|");
    for engine in &engines {
        let _ = writeln!(report, "| {} | {:.1?} |", engine.name(), engine.load_time());
    }

    let _ = writeln!(report);
    let _ = writeln!(
        report,
        "## Latency (warmup={}, repeats={})\n",
        knobs.warmup, knobs.repeats
    );
    let _ = writeln!(
        report,
        "| wav | dur s | engine | p50 ms | p95 ms | RTF p50 | fbank ms | forward ms | decode ms | text |"
    );
    let _ = writeln!(report, "|---|---|---|---|---|---|---|---|---|---|");

    for wav in &wavs {
        for engine in &engines {
            let Some(m) = results
                .iter()
                .find(|(e, w, _)| *e == engine.name() && *w == wav.name)
                .map(|(_, _, m)| m)
            else {
                let _ = writeln!(
                    report,
                    "| {} | {:.2} | {} | FAILED | | | | | | |",
                    wav.name, wav.duration_secs, engine.name()
                );
                continue;
            };
            let st = m.median_stages();
            let _ = writeln!(
                report,
                "| {} | {:.2} | {} | {:.1} | {:.1} | {:.3} | {:.1} | {:.1} | {:.1} | `{}` |",
                wav.name,
                wav.duration_secs,
                engine.name(),
                m.percentile_us(50.0) / 1000.0,
                m.percentile_us(95.0) / 1000.0,
                (m.percentile_us(50.0) / 1_000_000.0) / wav.duration_secs,
                st.feature_us as f64 / 1000.0,
                st.forward_us as f64 / 1000.0,
                st.decode_us as f64 / 1000.0,
                escape_table(&m.text),
            );
        }
    }

    // ---- parity check ----------------------------------------------------
    let _ = writeln!(report);
    let _ = writeln!(report, "## Transcription parity (whitespace- and ▁-insensitive)");
    let _ = writeln!(report);
    let _ = writeln!(report, "| wav | verdict | tract text | sherpa text |");
    let _ = writeln!(report, "|---|---|---|---|");
    for wav in &wavs {
        let get = |name: &str| {
            results
                .iter()
                .find(|(e, w, _)| *e == name && *w == wav.name)
                .map(|(_, _, m)| normalize_for_parity(&m.text))
        };
        let (a, b) = (get("tract"), get("sherpa"));
        match (a, b) {
            (Some(a), Some(b)) if a == b => {
                let _ = writeln!(report, "| {} | MATCH | `{}` | `{}` |", wav.name, truncate(&a, 60), truncate(&b, 60));
            }
            (Some(a), Some(b)) => {
                let _ = writeln!(report, "| {} | **MISMATCH** | `{}` | `{}` |", wav.name, truncate(&a, 60), truncate(&b, 60));
            }
            _ => {
                let _ = writeln!(report, "| {} | SKIPPED (an engine failed) | | |", wav.name);
            }
        }
    }

    let out_path = PathBuf::from("target/asr-speed-compare.md");
    if let Some(parent) = out_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(&out_path, &report).expect("write report");
    println!("\n[asr-speed] report written to {}", out_path.display());
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let t: String = s.chars().take(n).collect();
        format!("{t}…")
    }
}

fn escape_table(s: &str) -> String {
    s.replace('|', "\\|").replace('\n', " ")
}
