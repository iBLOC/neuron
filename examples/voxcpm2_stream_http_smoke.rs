//! End-to-end HTTP smoke for the model-level streaming TTS endpoint
//! (`POST /v1/audio/speech/stream`).
//!
//! Boots an embedded neuron server with `NEURON_TTS_RUNTIME=voxcpm2` and
//! `NEURON_TTS_MODEL_PATH` pointing at a local VoxCPM2 checkpoint, then
//! issues ONE streaming request and measures:
//!
//! 1. **TTFB** — time until the first audio byte arrives.
//! 2. **Chunk cadence** — how the bytes arrive over time (the whole point
//!    of model-level streaming: bytes trickle in while the model is still
//!    generating).
//! 3. **Container sanity** — the response is a WAV (placeholder-sized
//!    header + PCM payload) or raw PCM, and the payload is non-trivial.
//!
//! It then replays the same input through the legacy `/v1/audio/speech`
//! endpoint for comparison, so the report shows both the wall-clock and
//! the "first byte" numbers side by side.
//!
//! Run:
//!
//! ```bash
//! NEURON_TTS_RUNTIME=voxcpm2 \
//! NEURON_TTS_MODEL_PATH=$HOME/.neuron/models/burn/voxcpm2 \
//! cargo run --release --example voxcpm2_stream_http_smoke --features voxcpm
//! ```
//!
//! CPU cost: the 2B model materializes as f32 (~9.2 GB) and each latent
//! patch takes ~1-2 min on a desktop CPU, so budget ~15 min and ~10 GB RAM.

use std::time::{Duration, Instant};

use futures_util::StreamExt;

use neuron_lib::{run_embedded, NeuronConfig, NeuronEmbedConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // `run_embedded` builds its own current-thread runtime — calling it from
    // inside a `#[tokio::main]` would re-enter the runtime and panic.
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| format!("build tokio runtime: {e}"))?;
    runtime.block_on(async_main())
}

async fn async_main() -> Result<(), Box<dyn std::error::Error>> {
    let port: u16 = std::env::var("NEURON_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let cfg = NeuronEmbedConfig {
        port,
        config: NeuronConfig::new(),
    };
    let handle = run_embedded(cfg).await?;
    let port = handle.addr.port();
    eprintln!("[smoke:voxcpm2-stream] server bound on 127.0.0.1:{port}");

    let input = std::env::var("NEURON_SMOKE_INPUT")
        .unwrap_or_else(|_| "今天天气真不错。".to_string());

    // ---- 1. the streaming endpoint ---------------------------------------
    let (ttfb, first_audio, total, chunks, bytes, header) = streaming_request(port, &input).await?;
    eprintln!(
        "[smoke:voxcpm2-stream] /v1/audio/speech/stream: ttfb(header)={:?} first_audio={:?} \
         total={:?} chunks={} bytes={}",
        ttfb, first_audio, total, chunks, bytes.len()
    );
    eprintln!("[smoke:voxcpm2-stream] response head: {header:?}");

    if bytes.is_empty() {
        eprintln!("[smoke:voxcpm2-stream] FATAL: empty audio body");
        handle.request_shutdown();
        handle.join().await;
        std::process::exit(1);
    }
    if bytes.len() < 1024 {
        eprintln!(
            "[smoke:voxcpm2-stream] FATAL: suspiciously small body ({} bytes)",
            bytes.len()
        );
        handle.request_shutdown();
        handle.join().await;
        std::process::exit(1);
    }

    // The WAV variant starts with a RIFF header; the PCM variant does not.
    if &bytes[..4] == b"RIFF" {
        eprintln!("[smoke:voxcpm2-stream] container: WAV (placeholder-sized header)");
    } else {
        eprintln!("[smoke:voxcpm2-stream] container: raw PCM");
    }

    // Save both bodies so a CPU run and a CUDA run can be compared
    // byte-for-byte.
    let out_dir = std::path::Path::new("target/voxcpm2-stream-http");
    let _ = std::fs::create_dir_all(out_dir);
    let tag = std::env::var("NEURON_SMOKE_TAG").unwrap_or_else(|_| "run".to_string());
    let stream_path = out_dir.join(format!("{tag}-stream.wav"));
    let _ = std::fs::write(&stream_path, &bytes);

    // ---- 2. the legacy endpoint, for comparison --------------------------
    let (legacy_ttfb, legacy_total, legacy_bytes) = legacy_request(port, &input).await?;
    eprintln!(
        "[smoke:voxcpm2-stream] /v1/audio/speech:  ttfb={:?} total={:?} bytes={}",
        legacy_ttfb, legacy_total, legacy_bytes.len()
    );

    let legacy_path = out_dir.join(format!("{tag}-legacy.wav"));
    let _ = std::fs::write(&legacy_path, &legacy_bytes);
    eprintln!(
        "[smoke:voxcpm2-stream] wrote {} and {}",
        stream_path.display(),
        legacy_path.display()
    );

    let speedup = legacy_ttfb.as_secs_f64() / first_audio.as_secs_f64().max(1e-9);
    eprintln!(
        "[smoke:voxcpm2-stream] first-audio-byte speedup vs the legacy endpoint: {:.1}x \
         ({} -> {})",
        speedup,
        format!("{:?}", legacy_ttfb),
        format!("{:?}", first_audio)
    );

    eprintln!("[smoke:voxcpm2-stream] shutting down server...");
    handle.request_shutdown();
    handle.join().await;
    Ok(())
}

/// Stream `/v1/audio/speech/stream`, timing the first byte and every chunk.
async fn streaming_request(
    port: u16,
    input: &str,
) -> Result<
    (Duration, Duration, Duration, usize, Vec<u8>, String),
    Box<dyn std::error::Error>,
> {
    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "model": "voxcpm2",
        "input": input,
        "response_format": "wav",
    });
    let started = Instant::now();
    let response = client
        .post(format!("http://127.0.0.1:{port}/v1/audio/speech/stream"))
        .json(&body)
        .send()
        .await?;
    let status = response.status();
    let header = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("<none>")
        .to_string();
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        return Err(format!("stream endpoint returned {status}: {text}").into());
    }

    let mut stream = response.bytes_stream();
    let mut ttfb: Option<Duration> = None;
    let mut first_audio: Option<Duration> = None;
    let mut bytes: Vec<u8> = Vec::new();
    let mut chunks = 0usize;
    let mut last_report = Instant::now();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if ttfb.is_none() {
            ttfb = Some(started.elapsed());
            eprintln!(
                "[smoke:voxcpm2-stream] first byte (container header) after {:?}",
                ttfb.unwrap()
            );
        }
        chunks += 1;
        bytes.extend_from_slice(&chunk);
        // The first chunk is the 44-byte WAV header; the first *audio* bytes
        // arrive with the second chunk.
        if first_audio.is_none() && bytes.len() > 44 {
            first_audio = Some(started.elapsed());
            eprintln!(
                "[smoke:voxcpm2-stream] first AUDIO byte after {:?}",
                first_audio.unwrap()
            );
        }
        eprintln!(
            "[smoke:voxcpm2-stream] +{:?} chunk {chunks} (+{} bytes, {} total)",
            started.elapsed(),
            chunk.len(),
            bytes.len()
        );
        if last_report.elapsed() > Duration::from_secs(5) {
            last_report = Instant::now();
        }
    }
    let total = started.elapsed();
    Ok((
        ttfb.unwrap_or(total),
        first_audio.unwrap_or(total),
        total,
        chunks,
        bytes,
        header,
    ))
}

/// The legacy endpoint: full synthesis, then a chunked-transfer body.
async fn legacy_request(
    port: u16,
    input: &str,
) -> Result<(Duration, Duration, Vec<u8>), Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "model": "voxcpm2",
        "input": input,
        "response_format": "wav",
    });
    let started = Instant::now();
    let response = client
        .post(format!("http://127.0.0.1:{port}/v1/audio/speech"))
        .json(&body)
        .send()
        .await?;
    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        return Err(format!("speech endpoint returned {status}: {text}").into());
    }
    let bytes = response.bytes().await?;
    let elapsed = started.elapsed();
    Ok((elapsed, elapsed, bytes.to_vec()))
}
