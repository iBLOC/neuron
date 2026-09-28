//! Library-side support for the `benches/model_matrix.rs` runner.
//!
//! This module is deliberately library-only -- the bench binary lives
//! in `benches/` and is what HTTP-drives a running neuron server. The
//! split lets the matrix table be defined in one place (and unit
//! tested) while keeping the runner thin.
//!
//! ## What it provides
//!
//! * [`Capability`] -- every neuron capability that has at least one
//!   loadable runtime (`Llm`, `Tts`, `Asr`, `Rerank`, `Embedding`,
//!   `Vision`, `Chronos2`, `Decision`).
//! * [`Engine`] -- per-capability engine identifier (candle /
//!   mistralrs / tract / sherpa / voxcpm / litert / ort).
//! * [`Device`] -- `Cpu` / `Nvidia` / `Metal`; `available()` filters
//!   down to what the current build actually has compiled.
//! * [`MatrixRow`] -- one (cap, model, engine, device) entry plus the
//!   `NEURON_MODEL_DIRS`-style hint the bench uses to locate the model
//!   weights on disk.
//! * [`default_rows`] -- the curated (engine, model) set for each cap
//!   on the current feature set. Rows whose engine requires a missing
//!   feature are filtered out so the bench binary is portable across
//!   feature combinations.
//! * [`sample_for`] -- a deterministic, build-stable sample payload
//!   per cap (LLM chat message, rerank query+docs, embedding input,
//!   TTS text, ASR 1-second of 1 kHz sine, etc.) so the matrix runs
//!   reproducibly across machines.
//! * [`resolve_model_dir`] -- shared "NEURON_MODEL_DIRS"-aware lookup
//!   so the bench driver and the example server agree on model
//!   locations.

use std::collections::HashSet;
use std::path::PathBuf;

use serde_json::{json, Value};

/// Capability axis of the matrix -- one entry per neuron capability
/// that has at least one loadable runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Capability {
    Llm,
    Tts,
    Asr,
    Rerank,
    Embedding,
    Vision,
    Chronos2,
    Decision,
}

impl Capability {
    /// OpenAI / Anthropic endpoint path this capability is reached at.
    /// Used by the bench driver to assemble per-row HTTP requests.
    pub fn endpoint_path(self) -> &'static str {
        match self {
            Capability::Llm => "/v1/chat/completions",
            Capability::Tts => "/v1/audio/speech",
            Capability::Asr => "/v1/audio/transcriptions",
            Capability::Rerank => "/v1/rerank",
            Capability::Embedding => "/v1/embeddings",
            Capability::Vision => "/v1/vision/detect",
            Capability::Chronos2 => "/v1/forecast",
            Capability::Decision => "/v1/decisions",
        }
    }

    /// Stable name used in report tables.
    pub fn as_str(self) -> &'static str {
        match self {
            Capability::Llm => "llm",
            Capability::Tts => "tts",
            Capability::Asr => "asr",
            Capability::Rerank => "rerank",
            Capability::Embedding => "embedding",
            Capability::Vision => "vision",
            Capability::Chronos2 => "chronos2",
            Capability::Decision => "decision",
        }
    }

    /// Env-var name (without the `NEURON_BENCH_` prefix) that names the
    /// model directory this cap's bench row should pull weights from.
    pub fn model_dir_env(self) -> &'static str {
        match self {
            Capability::Llm => "NEURON_BENCH_LLM_DIR",
            Capability::Tts => "NEURON_BENCH_TTS_DIR",
            Capability::Asr => "NEURON_BENCH_ASR_DIR",
            Capability::Rerank => "NEURON_BENCH_RERANK_DIR",
            Capability::Embedding => "NEURON_BENCH_EMBEDDING_DIR",
            Capability::Vision => "NEURON_BENCH_VISION_DIR",
            Capability::Chronos2 => "NEURON_BENCH_CHRONOS2_DIR",
            Capability::Decision => "NEURON_BENCH_DECISION_DIR",
        }
    }

    /// Env-var name (without the `NEURON_BENCH_` prefix) that selects
    /// which engine row to test. `None` means the cap has only one
    /// engine in the current build.
    pub fn runtime_env(self) -> Option<&'static str> {
        match self {
            Capability::Llm => Some("NEURON_BENCH_LLM_RUNTIME"),
            Capability::Tts => Some("NEURON_BENCH_TTS_RUNTIME"),
            Capability::Asr => Some("NEURON_BENCH_ASR_RUNTIME"),
            Capability::Rerank => Some("NEURON_BENCH_RERANK_RUNTIME"),
            Capability::Embedding => Some("NEURON_BENCH_EMBEDDING_RUNTIME"),
            Capability::Vision => Some("NEURON_BENCH_VISION_RUNTIME"),
            _ => None,
        }
    }

    /// Which engines this capability can test, in the canonical order
    /// the matrix reports them. Each entry is paired with the cargo
    /// feature that has to be enabled for the row to be runnable.
    pub fn engines(self) -> &'static [Engine] {
        match self {
            Capability::Llm => &[
                Engine { id: "candle", feature: "candle" },
                Engine { id: "mistralrs", feature: "mistralrs" },
                Engine { id: "litert", feature: "litert" },
            ],
            Capability::Tts => &[Engine { id: "voxcpm", feature: "voxcpm" }],
            Capability::Asr => &[
                Engine { id: "sherpa", feature: "sherpa" },
                Engine { id: "tract", feature: "tract" },
                Engine { id: "candle", feature: "candle" },
            ],
            Capability::Rerank => &[
                Engine { id: "candle-bge", feature: "candle" },
                Engine { id: "candle-mxbai", feature: "candle" },
                Engine { id: "candle-jina", feature: "candle" },
                Engine { id: "cohere", feature: "candle" },
            ],
            Capability::Embedding => &[Engine { id: "candle", feature: "candle" }],
        Capability::Vision => &[
            Engine { id: "candle-yolo", feature: "candle" },
            Engine { id: "tract", feature: "tract" },
        ],
            Capability::Chronos2 => &[Engine { id: "candle", feature: "chronos2" }],
            Capability::Decision => &[Engine { id: "tract", feature: "tract" }],
        }
    }
}

/// Engine identifier within a capability. Tied to the cargo feature
/// that has to be enabled for the row to be runnable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Engine {
    pub id: &'static str,
    pub feature: &'static str,
}

/// Device the inference will land on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    Nvidia,
    Metal,
}

impl Device {
    pub fn as_str(self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
            Device::Nvidia => "nvidia",
            Device::Metal => "metal",
        }
    }

    /// Devices the current binary actually has compiled support for.
    /// `Nvidia` requires `cuda`, `Metal` requires `metal`; `Cpu` is
    /// always available.
    pub fn available() -> Vec<Device> {
        let mut out = vec![Device::Cpu];
        #[cfg(feature = "cuda")]
        out.push(Device::Nvidia);
        #[cfg(feature = "metal")]
        out.push(Device::Metal);
        out
    }
}

/// One (cap, model, engine, device) row in the matrix. The driver
/// resolves `model_dir` from `model_dir_env` first, then falls back to
/// `$NEURON_BENCH_MODEL_DIRS/<cap>/<model_id>` (a single multi-cap
/// model root for hosts that dump everything into one tree).
#[derive(Debug, Clone)]
pub struct MatrixRow {
    pub cap: Capability,
    pub model_id: String,
    pub engine: Engine,
    pub device: Device,
    pub model_dir_env: String,
    pub runtime_env: Option<String>,
    /// Free-form note the bench driver writes into the report so a
    /// reader can tell at a glance why a row was skipped or measured.
    pub note: &'static str,
}

/// Curated (engine, model) set for each cap on the current feature
/// set. Rows whose `engine.feature` is not enabled are dropped here so
/// the runner never has to consult cargo features itself.
pub fn default_rows() -> Vec<MatrixRow> {
    let available: HashSet<&'static str> = {
        #[allow(unused_mut)]
        let mut s = HashSet::new();
        #[cfg(feature = "candle")]
        s.insert("candle");
        #[cfg(feature = "flash-attn")]
        {
            s.insert("flash-attn");
        }
        #[cfg(feature = "metal")]
        s.insert("metal");
        #[cfg(feature = "cuda")]
        s.insert("cuda");
        #[cfg(feature = "chronos2")]
        s.insert("chronos2");
        #[cfg(feature = "voxcpm")]
        s.insert("voxcpm");
        #[cfg(feature = "litert")]
        s.insert("litert");
        #[cfg(feature = "mistralrs")]
        s.insert("mistralrs");
        #[cfg(feature = "asr-audio")]
        s.insert("asr-audio");
        #[cfg(feature = "tract")]
        s.insert("tract");
        #[cfg(feature = "sherpa")]
        s.insert("sherpa");
        #[cfg(feature = "ort")]
        s.insert("ort");
        s
    };

    let mut rows: Vec<MatrixRow> = Vec::new();
    for cap in all_caps() {
        for engine in cap.engines() {
            if !available.contains(engine.feature) {
                continue;
            }
            for model_id in model_ids_for(cap, engine.id) {
                for device in Device::available() {
                    rows.push(MatrixRow {
                        cap,
                        model_id: model_id.to_string(),
                        engine: *engine,
                        device,
                        model_dir_env: cap.model_dir_env().to_string(),
                        runtime_env: cap.runtime_env().map(str::to_string),
                        note: "",
                    });
                }
            }
        }
    }
    rows
}

fn all_caps() -> Vec<Capability> {
    vec![
        Capability::Llm,
        Capability::Tts,
        Capability::Asr,
        Capability::Rerank,
        Capability::Embedding,
        Capability::Vision,
        Capability::Chronos2,
        Capability::Decision,
    ]
}

/// Default model id the matrix probes per (cap, engine). These match
/// the catalog entries in `src/capabilities/catalog.rs`; override with
/// `NEURON_BENCH_<CAP>_DIR` to point at a different local copy.
fn model_ids_for(cap: Capability, engine_id: &str) -> &'static [&'static str] {
    match (cap, engine_id) {
        (Capability::Llm, "candle") => &["minicpm5-1b"],
        (Capability::Llm, "mistralrs") => &["minicpm5-1b"],
        (Capability::Llm, "litert") => &["gemma-3n-e2b-it"],
        (Capability::Tts, _) => &["voxcpm-0.5b"],
        (Capability::Asr, "sherpa") | (Capability::Asr, "tract") => &["sensevoice-small"],
        (Capability::Asr, "candle") => &["whisper-base"],
        (Capability::Rerank, "candle-bge") => &["bge-reranker-base"],
        (Capability::Rerank, "candle-mxbai") => &["mxbai-rerank-base-v2"],
        (Capability::Rerank, "candle-jina") => &["jina-reranker-base-v2"],
        (Capability::Rerank, "cohere") => &["rerank-english-v3.0"],
        (Capability::Embedding, _) => &["all-minilm-l6-v2"],
        (Capability::Vision, "candle-yolo") => &["yolov8n"],
        (Capability::Vision, "ort") => &["yolov8n"],
        (Capability::Vision, "tract") => &["yolov8n"],
        (Capability::Chronos2, _) => &["chronos2-t5-base"],
        (Capability::Decision, _) => &["laya"],
        _ => &[],
    }
}

/// Resolve a model directory for a row. Order:
///
/// 1. `NEURON_BENCH_<CAP>_DIR` if set (override).
/// 2. `$NEURON_BENCH_MODEL_DIRS` (colon-separated, mirroring
///    `NEURON_MODEL_DIRS`) joined with `<cap>/<model_id>`.
/// 3. `~/.neuron/models/<cap>/<model_id>` as a last resort.
pub fn resolve_model_dir(row: &MatrixRow) -> Option<PathBuf> {
    if let Ok(p) = std::env::var(&row.model_dir_env) {
        return Some(PathBuf::from(p));
    }
    let roots = std::env::var("NEURON_BENCH_MODEL_DIRS")
        .ok()
        .or_else(|| std::env::var("NEURON_MODEL_DIRS").ok())
        .or_else(|| std::env::var("NEURON_MODEL_DIR").ok())
        .unwrap_or_else(|| {
            let home = std::env::var("USERPROFILE")
                .or_else(|_| std::env::var("HOME"))
                .unwrap_or_default();
            format!("{home}/.neuron/models")
        });
    for root in roots.split(':').filter(|s| !s.is_empty()) {
        let cap = row.cap.as_str();
        let candidate = PathBuf::from(root).join(cap).join(&row.model_id);
        if candidate.is_dir() {
            return Some(candidate);
        }
    }
    None
}

/// Build the per-capability payload the bench driver POSTs. The shape
/// matches the live endpoint contract (`POST /v1/chat/completions`,
/// `POST /v1/embeddings`, etc.); `cap.endpoint_path()` is the route.
///
/// Returned as `serde_json::Value` so the bench can serialize it
/// directly. For ASR the driver has to instead form a multipart
/// payload (see [`asr_sample_multipart_part`]) because the endpoint
/// takes a file upload, not JSON.
pub fn sample_for(cap: Capability, model_id: &str) -> Value {
    match cap {
        Capability::Llm => json!({
            "model": model_id,
            "messages": [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "In one sentence, what is the capital of France?"}
            ],
            "max_tokens": 32,
            "temperature": 0.0,
            "stream": false,
        }),
        Capability::Tts => json!({
            "model": model_id,
            "input": "The quick brown fox jumps over the lazy dog.",
            "voice": null,
            "response_format": "wav",
            "speed": 1.0,
        }),
        Capability::Rerank => json!({
            "model": model_id,
            "query": "What is the capital of France?",
            "documents": [
                "Paris is the capital and most populous city of France.",
                "Berlin is the capital of Germany.",
                "Tokyo is the capital of Japan.",
            ],
            "top_n": 3,
            "return_documents": true,
        }),
        Capability::Embedding => json!({
            "model": model_id,
            "input": [
                "Hello world.",
                "neuron inference benchmark.",
            ],
            "encoding_format": "float",
        }),
        Capability::Vision => json!({
            "model": model_id,
            "task": "detect",
            // 4x4 grayscale image is enough to drive an inference path;
            // the matrix does not grade accuracy, only latency.
            "image": {"width": 320, "height": 320, "channels": 3, "data_b64": "AAAA"},
            "confidence_threshold": 0.25,
        }),
        Capability::Chronos2 => json!({
            "model": model_id,
            // 24-point synthetic sine; reproducible across machines.
            "series": (0..24).map(|i| (i as f64 * 0.4).sin()).collect::<Vec<_>>(),
            "horizon": 8,
            "method": "mean",
        }),
        Capability::Decision => json!({
            "model": model_id,
            "state": {"user": "kim", "tags": ["a", "b"], "score": 3},
            "questions": [{
                "qtype": "choice",
                "options": ["Paris", "Berlin", "Madrid"],
                "criterion": "European capital",
            }],
        }),
        // ASR is multipart, not JSON; the bench driver builds it directly
        // from `asr_sample_multipart_part()` instead of calling here.
        Capability::Asr => json!({"_multipart": true, "model": model_id}),
    }
}

/// Produce a minimal in-memory WAV (16-bit PCM, mono, 16 kHz, 1 second
/// of 1 kHz sine). The bench posts this as the multipart `file` field
/// to `/v1/audio/transcriptions`. Same payload shape every run so
/// `RTF = latency / 1.0 s` is exact.
pub fn asr_sample_multipart_part() -> (Vec<u8>, &'static str) {
    let sample_rate = 16_000usize;
    let seconds = 1;
    let n = sample_rate * seconds;
    let mut samples = Vec::with_capacity(44 + 2 * n);
    let byte_rate = sample_rate * 2;
    let data_len = (2 * n) as u32;
    let chunk_len = 36 + data_len;

    samples.extend_from_slice(b"RIFF");
    samples.extend_from_slice(&chunk_len.to_le_bytes());
    samples.extend_from_slice(b"WAVE");
    samples.extend_from_slice(b"fmt ");
    samples.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    samples.extend_from_slice(&1u16.to_le_bytes()); // PCM
    samples.extend_from_slice(&1u16.to_le_bytes()); // mono
    samples.extend_from_slice(&(sample_rate as u32).to_le_bytes());
    samples.extend_from_slice(&(byte_rate as u32).to_le_bytes());
    samples.extend_from_slice(&2u16.to_le_bytes()); // block align
    samples.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    samples.extend_from_slice(b"data");
    samples.extend_from_slice(&data_len.to_le_bytes());

    for i in 0..n {
        let phase = std::f32::consts::TAU * 1000.0 * i as f32 / sample_rate as f32;
        let s = (phase.sin() * 0.5 * i16::MAX as f32) as i16;
        samples.extend_from_slice(&s.to_le_bytes());
    }
    (samples, "sine_1s.wav")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_rows_respects_feature_flags() {
        let rows = default_rows();
        // the cap->endpoint map is the wire contract -- assert it stays
        // consistent across every row
        for row in &rows {
            // LLM rows must report a runtime env so the bench knows which
            // engine to ask the server to use
            if row.cap == Capability::Llm {
                assert!(row.runtime_env.is_some());
            }
            // ASR's sherpa + tract engines need an onnx file; the bench
            // honours NEURON_ASR_ONNX_FILE separately.
        }
    }

    #[test]
    fn test_default_rows_skip_unsupported_engines() {
        let rows = default_rows();
        // when `sherpa` is disabled, no row should advertise a sherpa engine
        if !cfg!(feature = "sherpa") {
            for row in &rows {
                if row.cap == Capability::Asr {
                    assert_ne!(row.engine.id, "sherpa");
                }
            }
        }
        if !cfg!(feature = "tract") {
            for row in &rows {
                assert_ne!(row.engine.id, "tract");
            }
        }
    }

    #[test]
    fn test_sample_for_caps_is_openai_compatible() {
        for cap in all_caps() {
            if cap == Capability::Asr {
                continue;
            }
            let v = sample_for(cap, "test-model");
            // every non-ASR payload must carry a model id
            assert_eq!(
                v.get("model").and_then(|m| m.as_str()),
                Some("test-model"),
                "cap {} payload missing model id: {v}",
                cap.as_str()
            );
        }
    }

    #[test]
    fn test_asr_sample_wav_is_one_second() {
        let (bytes, name) = asr_sample_multipart_part();
        assert!(name.ends_with(".wav"));
        assert!(bytes.starts_with(b"RIFF"));
        // 44-byte header + 2 bytes per sample * 16000 samples = 32044 bytes
        assert_eq!(bytes.len(), 44 + 2 * 16_000);
    }
}
