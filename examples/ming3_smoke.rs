//! Real-weights smoke for Ming-3.0-tts-0.5B (`--features ming3`).
//!
//! Answers the questions the tiny random-weight tests cannot:
//! 1. does the real `tokenizer.json` pass the trained-contract validation
//!    (pinned ids `<audioPatch>`=151705, `<audio>`=151706 …)?
//! 2. do all ~1180 checkpoint tensors remap onto the `Ming3Model` tree
//!    (the loader fails loudly on any missing tensor)?
//! 3. does the full pipeline (task → contract → AR loop → AudioVAE → WAV)
//!    produce a real 44.1 kHz waveform for a real sentence?
//!
//! Run (budget ~6 GB RAM for the f32-materialized weights; minutes on the
//! CPU backend — the AR loop is latency-bound, see voxcpm2.rs's `Auto` arm):
//!
//! ```bash
//! NEURON_TTS_MODEL_PATH=<checkpoint dir> \
//! cargo run --release --example ming3_smoke --features ming3
//! ```

use std::path::PathBuf;
use std::time::Instant;

use neuron_lib::tts::ming3::runtime::Ming3TtsRuntime;
use neuron_lib::tts::{TtsRuntime, wav_data_bytes};

/// RMS of the WAV payload (silence would be near zero; a real utterance is
/// clearly above it).
fn rms_of_wav(audio: &[u8]) -> Option<f32> {
    let payload = wav_data_bytes(audio)?;
    let samples: Vec<f32> = payload
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32767.0)
        .collect();
    if samples.is_empty() {
        return None;
    }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    Some((sum_sq / samples.len() as f32).sqrt())
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), String> {
    // Resolve the checkpoint dir: env, or the repo-local scratch dir. Every
    // later path is built from the canonicalized directory.
    let model_dir = match std::env::var("NEURON_TTS_MODEL_PATH") {
        Ok(p) => PathBuf::from(p),
        Err(_) => std::env::current_dir()
            .map_err(|e| format!("cwd: {e}"))?
            .join(".local")
            .join("ming3-ckpt"),
    };
    let dir = model_dir
        .canonicalize()
        .map_err(|e| format!("checkpoint dir not resolvable: {e}"))?;
    println!("[ming3-smoke] checkpoint dir: {}", dir.display());

    // The three input files, checked explicitly.
    let config_path = dir.join("config.json");
    let tokenizer_path = dir.join("tokenizer.json");
    let weights_path = dir.join("model.safetensors");
    let config_size = std::fs::metadata(&config_path).map_err(|e| e.to_string())?.len();
    let tokenizer_size = std::fs::metadata(&tokenizer_path).map_err(|e| e.to_string())?.len();
    let weights_size = std::fs::metadata(&weights_path).map_err(|e| e.to_string())?.len();
    println!("[ming3-smoke]   config: {config_size} bytes");
    println!("[ming3-smoke]   tokenizer: {tokenizer_size} bytes");
    println!("[ming3-smoke]   weights: {weights_size} bytes");

    // Stage 1 — load: contract validation + full checkpoint remap.
    let t0 = Instant::now();
    let runtime = Ming3TtsRuntime::load("ming3", &neuron_lib::config::DeviceSelection::Cpu)
        .await
        .map_err(|e| format!("load failed: {}", e.message))?;
    println!("[ming3-smoke] load OK in {:.1}s", t0.elapsed().as_secs_f32());

    // Stage 2 — synthesize a short sentence through the whole pipeline.
    let text = "你好，世界。今天天气真不错。";
    let t1 = Instant::now();
    let audio = runtime
        .synthesize(text, Some("zh"))
        .await
        .map_err(|e| format!("synthesize failed: {}", e.message))?;
    println!(
        "[ming3-smoke] synthesize OK in {:.1}s — text={text:?}",
        t1.elapsed().as_secs_f32()
    );
    println!(
        "[ming3-smoke]   format={} sample_rate={} channels={} bytes={} duration={:.2}s",
        audio.format, audio.sample_rate, audio.channels, audio.audio.len(), audio.duration_secs
    );
    let rms = rms_of_wav(&audio.audio).unwrap_or(0.0);
    println!("[ming3-smoke]   waveform RMS = {rms:.5} (silence would be near zero)");

    // Stage 3 — write the artifact inside the canonicalized checkpoint dir.
    let out = dir.join("out.wav");
    std::fs::write(&out, &audio.audio).map_err(|e| format!("write failed: {e}"))?;
    println!("[ming3-smoke] wrote {}", out.display());

    // Smoke verdict: real audio must be non-trivial and non-silent.
    if audio.duration_secs < 0.5 {
        return Err(format!("utterance too short: {:.2}s", audio.duration_secs));
    }
    if rms < 1e-4 {
        return Err(format!("waveform is silent (RMS {rms})"));
    }
    println!("[ming3-smoke] PASS");
    Ok(())
}
