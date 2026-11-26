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

/// Minimal PCM16 mono WAV reader that accepts any sample rate (the ASR
/// decoder is 16 kHz-only by contract). Returns `(sample_rate, samples)`.
fn decode_pcm16_mono(bytes: &[u8]) -> Result<(u32, Vec<f32>), String> {
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".to_string());
    }
    let mut pos = 12usize;
    let (mut channels, mut sample_rate, mut bits) = (0u16, 0u32, 0u16);
    let (mut data_start, mut data_len) = (0usize, 0usize);
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;
        match id {
            b"fmt " => {
                channels = u16::from_le_bytes([bytes[pos + 10], bytes[pos + 11]]);
                sample_rate = u32::from_le_bytes([
                    bytes[pos + 12],
                    bytes[pos + 13],
                    bytes[pos + 14],
                    bytes[pos + 15],
                ]);
                bits = u16::from_le_bytes([bytes[pos + 22], bytes[pos + 23]]);
            }
            b"data" => {
                data_start = pos + 8;
                data_len = size.min(bytes.len().saturating_sub(data_start));
            }
            _ => {}
        }
        pos = pos + 8 + size + (size & 1);
    }
    if channels != 1 || bits != 16 || data_start == 0 {
        return Err(format!(
            "expected PCM16 mono (got channels={channels} bits={bits})"
        ));
    }
    let samples: Vec<f32> = bytes[data_start..data_start + data_len]
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32767.0)
        .collect();
    Ok((sample_rate, samples))
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
    // The runtime resolves weights through NEURON_TTS_MODEL_PATH
    // (`env_var_result` prepends the NEURON_ prefix); hand it the
    // canonicalized directory (Windows `\\?\` form is accepted as-is).
    // Safe: this runs at the very start of a single-threaded main, before
    // any thread or env-reading runtime exists.
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe {
        std::env::set_var("NEURON_TTS_MODEL_PATH", &dir);
    }
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

    // Smoke verdict: real audio must be non-trivial and non-silent. The
    // duration floor is deliberately LOW (0.3 s): the port's stop head
    // swings run-to-run between ≈0.5 s and ≈1 s for a short sentence
    // (the duration-fidelity debt is recorded in the ledger U+37F3 ④❶) —
    // this smoke judges silence and wiring, not duration fidelity.
    if audio.duration_secs < 0.3 {
        return Err(format!("utterance too short: {:.2}s", audio.duration_secs));
    }
    if rms < 1e-4 {
        return Err(format!("waveform is silent (RMS {rms})"));
    }

    // Stage 4 — voice clone (behind `ming3-clone`): a real reference clip
    // (the upstream built-in voice preset, mono 48 kHz) drives the CAM++
    // speaker-embedding leg and the AudioVAE prompt latents.
    #[cfg(feature = "ming3-clone")]
    {
        let ref_path = dir.join("ref-lingguang.wav");
        let ref_bytes = std::fs::read(&ref_path)
            .map_err(|e| format!("reference wav {}: {e}", ref_path.display()))?;
        // The ASR WAV decoder is 16 kHz-only by contract (SenseVoice's
        // graphs), so the 48 kHz preset is decoded here: PCM16 mono,
        // layout verified against the file's own fmt chunk.
        let ref_samples = decode_pcm16_mono(&ref_bytes).map_err(|e| format!("ref decode: {e}"))?;
        println!(
            "[ming3-smoke] clone ref: {} samples @ {} Hz",
            ref_samples.1.len(),
            ref_samples.0
        );
        let ref_16k = neuron_lib::tts::ming3::spkemb::resample_linear(
            &ref_samples.1,
            ref_samples.0 as usize,
            16_000,
        );
        let transcript = "诶，你今天有空吗？我们能不能聊聊这件事啊？";
        let t2 = Instant::now();
        let cloned = runtime
            .synthesize_clone(
                "从山脚一路走上山顶，风把云都吹散了。",
                Some(&ref_16k),
                Some(transcript),
            )
            .await
            .map_err(|e| format!("clone failed: {}", e.message))?;
        println!(
            "[ming3-smoke] clone OK in {:.1}s — bytes={} duration={:.2}s",
            t2.elapsed().as_secs_f32(),
            cloned.audio.len(),
            cloned.duration_secs
        );
        let clone_rms = rms_of_wav(&cloned.audio).unwrap_or(0.0);
        println!("[ming3-smoke]   clone waveform RMS = {clone_rms:.5}");
        let clone_out = dir.join("out-clone.wav");
        std::fs::write(&clone_out, &cloned.audio).map_err(|e| format!("write failed: {e}"))?;
        println!("[ming3-smoke] wrote {}", clone_out.display());
        if cloned.duration_secs < 0.3 {
            return Err(format!("clone too short: {:.2}s", cloned.duration_secs));
        }
        if clone_rms < 1e-4 {
            return Err(format!("clone waveform is silent (RMS {clone_rms})"));
        }
    }

    println!("[ming3-smoke] PASS");
    Ok(())
}
