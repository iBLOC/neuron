//! Real-weights smoke for VoxCPM2 model-level streaming decode.
//!
//! Answers ONE question with real checkpoint weights: does
//! `StreamingVaeDecoder` (chunk-by-chunk causal decode, the piece the HTTP
//! `synthesize_stream` override would use) reproduce the full
//! `AudioVaeV2::decode` waveform on the same latent sequence?
//!
//! The unit tests in `audiovae_v2_stream.rs` only assert structural
//! fidelity because random-init toy weights amplify boundary-tap error
//! through the upsampling stages; this smoke runs the REAL `audiovae.pth`
//! decoder, where the cascade should stay bounded.
//!
//! Stage A (default): loads only the VAE (audiovae.pth, 377 MB), encodes a
//! deterministic synthetic 16 kHz signal into a real latent, then compares
//!   full = vae.decode(latent)
//!   streamed = StreamingVaeDecoder over 4-frame latent patches (the exact
//!              patch shape `generate_streaming` emits per callback)
//! sample-by-sample, and writes both WAVs under target/voxcpm2-stream-smoke/.
//!
//! Stage B (`--with-lm`): additionally loads the 2B LM (model.safetensors,
//! ~9 GB f32 resident), generates a short utterance's latent, and runs the
//! same comparison on an in-distribution latent. Slow on CPU (minutes).
//!
//! Run:
//!
//! ```bash
//! NEURON_TTS_MODEL_PATH=~/.neuron/models/burn/voxcpm2 \
//! cargo run --release --example voxcpm2_stream_smoke --features voxcpm
//!
//! # with the 2B LM stage (budget ~10 GB RAM):
//! NEURON_TTS_MODEL_PATH=~/.neuron/models/burn/voxcpm2 \
//! cargo run --release --example voxcpm2_stream_smoke --features voxcpm -- --with-lm
//! ```

use std::f32::consts::PI;
use std::path::PathBuf;
use std::time::Instant;

use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::tensor::{Shape, Tensor, TensorData};

use neuron_lib::tts::voxcpm_burn::audiovae_v2::AudioVaeV2;
use neuron_lib::tts::voxcpm_burn::audiovae_v2_stream::StreamingVaeDecoder;
use neuron_lib::tts::voxcpm_burn::voxcpm2_model::{
    ClonePrompt, GenerationOptions, VoxCpm2Tokenizer, VoxCPM2, VoxCPM2Config,
};
use neuron_lib::tts::voxcpm_burn::weights::load_checkpoint_into;

struct Metrics {
    len_full: usize,
    len_stream: usize,
    max_abs: f32,
    mean_abs: f32,
    pearson: f32,
    first_divergence: Option<usize>,
}

fn compare(full: &[f32], stream: &[f32]) -> Metrics {
    let n = full.len().min(stream.len());
    let mut max_abs = 0f32;
    let mut sum_abs = 0f32;
    let mut first_div: Option<usize> = None;
    let mut mean_s = 0f32;
    let mut mean_f = 0f32;
    for i in 0..n {
        let d = (full[i] - stream[i]).abs();
        if d > max_abs {
            max_abs = d;
        }
        sum_abs += d;
        if first_div.is_none() && d > 1e-5 {
            first_div = Some(i);
        }
        mean_s += stream[i];
        mean_f += full[i];
    }
    mean_s /= n as f32;
    mean_f /= n as f32;
    let mut cov = 0f32;
    let mut var_s = 0f32;
    let mut var_f = 0f32;
    for i in 0..n {
        let ds = stream[i] - mean_s;
        let df = full[i] - mean_f;
        cov += ds * df;
        var_s += ds * ds;
        var_f += df * df;
    }
    Metrics {
        len_full: full.len(),
        len_stream: stream.len(),
        max_abs,
        mean_abs: sum_abs / n as f32,
        pearson: cov / (var_s.sqrt() * var_f.sqrt() + 1e-9),
        first_divergence: first_div,
    }
}

fn print_metrics(tag: &str, m: &Metrics) {
    println!("[{tag}] full={} streamed={} samples", m.len_full, m.len_stream);
    println!(
        "[{tag}] max_abs_diff={:.3e} mean_abs_diff={:.3e} pearson_r={:.6}",
        m.max_abs, m.mean_abs, m.pearson
    );
    match m.first_divergence {
        Some(i) => println!(
            "[{tag}] first divergence (>1e-5) at sample {i} ({:.1}% in)",
            100.0 * i as f32 / m.len_full as f32
        ),
        None => println!("[{tag}] no sample differs by more than 1e-5 — bit-comparable"),
    }
}

fn verdict(tag: &str, m: &Metrics) -> bool {
    if m.max_abs <= 1e-4 {
        println!("[{tag}] VERDICT: PASS (bit-comparable, max_abs <= 1e-4)");
        true
    } else if m.pearson > 0.999 && m.max_abs < 1e-2 {
        println!("[{tag}] VERDICT: PASS (audibly identical: pearson > 0.999, max_abs < 1e-2)");
        true
    } else {
        println!("[{tag}] VERDICT: INVESTIGATE (deviation exceeds audible-identity bounds)");
        false
    }
}

fn write_wav(path: &std::path::Path, samples: &[f32], sample_rate: u32) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .unwrap_or_else(|e| panic!("create {}: {e}", path.display()));
    for &s in samples {
        let _ = writer.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16);
    }
    let _ = writer.finalize();
    println!("[wav] wrote {} ({} samples @ {sample_rate} Hz)", path.display(), samples.len());
}

/// Decode `latent [1, D, T]` both ways and compare. `data` is the latent's
/// f32 row-major buffer (dims [1, D, T]); returns (full, streamed, pass).
fn run_comparison(
    vae: &AudioVaeV2<NdArray>,
    latent: &Tensor<NdArray, 3>,
    data: &[f32],
    d: usize,
    t: usize,
    patch: usize,
    tag: &str,
    wav_dir: &std::path::Path,
) -> (Vec<f32>, Vec<f32>, bool) {
    let t0 = Instant::now();
    let full = vae.decode(latent.clone(), None);
    let full: Vec<f32> = full.to_data().to_vec().unwrap();
    println!("[{tag}] full decode: {:.2?}", t0.elapsed());

    // Streamed decode in `patch`-frame latent chunks — the exact shape the
    // generate_streaming callback emits.
    let mut streamed: Vec<f32> = Vec::with_capacity(full.len());
    let mut decoder = StreamingVaeDecoder::new(vae);
    let mut chunk_max: Vec<f32> = Vec::new();
    let t_stream = Instant::now();
    let mut first_chunk_secs = None;
    for start in (0..t).step_by(patch) {
        let end = (start + patch).min(t);
        let width = end - start;
        let mut buf = vec![0f32; d * width];
        for dd in 0..d {
            for tt in 0..width {
                buf[dd * width + tt] = data[dd * t + start + tt];
            }
        }
        let chunk = Tensor::<NdArray, 3>::from_data(
            TensorData::new(buf, Shape::from([1usize, d, width])),
            &NdArrayDevice::Cpu,
        );
        let tc = Instant::now();
        let out = decoder.decode_chunk(chunk);
        let out: Vec<f32> = out.to_data().to_vec().unwrap();
        if first_chunk_secs.is_none() {
            first_chunk_secs = Some(tc.elapsed());
        }
        chunk_max.push(out.iter().map(|v| v.abs()).fold(0f32, f32::max));
        streamed.extend_from_slice(&out);
    }
    println!(
        "[{tag}] streamed decode: {:.2?} total, first chunk {:.2?} ({} chunks)",
        t_stream.elapsed(),
        first_chunk_secs.unwrap(),
        chunk_max.len()
    );

    let m = compare(&full, &streamed);
    print_metrics(tag, &m);
    let pass = verdict(tag, &m);

    write_wav(&wav_dir.join(format!("{tag}-full.wav")), &full, vae.out_sample_rate as u32);
    write_wav(
        &wav_dir.join(format!("{tag}-streamed.wav")),
        &streamed,
        vae.out_sample_rate as u32,
    );

    (full, streamed, pass)
}

fn main() {
    let with_lm = std::env::args().any(|a| a == "--with-lm");

    let dir = std::env::var("NEURON_TTS_MODEL_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("USERPROFILE")
                .or_else(|_| std::env::var("HOME"))
                .expect("USERPROFILE/HOME not set");
            PathBuf::from(home).join(".neuron/models/burn/voxcpm2")
        });
    for file in ["config.json", "audiovae.pth"] {
        assert!(
            dir.join(file).exists(),
            "{} not found under {} — set NEURON_TTS_MODEL_PATH",
            file,
            dir.display()
        );
    }
    println!("[smoke] model dir: {}", dir.display());

    let wav_dir = std::path::Path::new("target/voxcpm2-stream-smoke");
    let device = NdArrayDevice::Cpu;

    let config = VoxCPM2Config::from_dir(&dir).expect("parse config.json");
    let patch = config.patch_size.max(1);
    println!("[smoke] patch_size = {patch} latent frames");

    // ---- load the real VAE ----
    let vae_config = config.audio_vae_config.clone().unwrap_or_default();
    let mut vae: AudioVaeV2<NdArray> = vae_config.init(&device);
    let t0 = Instant::now();
    let report = load_checkpoint_into(&mut vae, &dir.join("audiovae.pth"))
        .expect("load audiovae.pth");
    println!(
        "[smoke] audiovae.pth loaded in {:.2?} (applied={}, missing={}, unused={}, errors={})",
        t0.elapsed(),
        report.applied,
        report.missing.len(),
        report.unused.len(),
        report.errors.len()
    );
    assert!(report.missing.is_empty(), "missing: {:?}", report.missing);
    assert!(report.errors.is_empty(), "errors: {:?}", report.errors);

    let (d, hop) = (vae.latent_dim, vae.chunk_size);

    // ---- Stage A: real-encoder latent from a deterministic synthetic signal ----
    // 16 patches x patch frames x hop samples of 16 kHz input.
    let n_patches_a = 16;
    let n = n_patches_a * patch * hop;
    let mut x = vec![0f32; n];
    for (i, xi) in x.iter_mut().enumerate() {
        let t = i as f32 / 16000.0;
        let dur = n as f32 / 16000.0;
        let env = (1.0 - (-8.0 * t).exp()) * (1.0 - (-8.0 * (dur - t)).exp());
        let vib = 1.0 + 0.02 * (2.0 * PI * 5.0 * t).sin();
        *xi = 0.5
            * env
            * (0.4 * (2.0 * PI * 220.0 * vib * t).sin()
                + 0.25 * (2.0 * PI * 551.0 * t).sin()
                + 0.15 * (2.0 * PI * 880.0 * t).sin());
    }
    let audio = Tensor::<NdArray, 3>::from_data(
        TensorData::new(x, Shape::from([1usize, 1usize, n])),
        &device,
    );
    let latent = vae.encode(audio, Some(vae.sample_rate));
    let [_, ld, lt] = latent.dims();
    assert_eq!(ld, d, "encoder latent dim");
    assert_eq!(lt, n / hop, "encoder latent frames");
    let latent_data: Vec<f32> = latent.to_data().to_vec().unwrap();
    println!("[stage-a] encoded synthetic signal -> latent [1, {ld}, {lt}]");

    let (_, _, pass_a) = run_comparison(
        &vae,
        &latent,
        &latent_data,
        d,
        lt,
        patch,
        "stage-a",
        wav_dir,
    );

    // ---- Stage B (optional): in-distribution latent from the 2B LM ----
    let mut pass_b: Option<bool> = None;
    if with_lm {
        assert!(
            dir.join("model.safetensors").exists() && dir.join("tokenizer.json").exists(),
            "--with-lm needs model.safetensors + tokenizer.json under {}",
            dir.display()
        );
        let mut model: VoxCPM2<NdArray> = config.init(&device);
        let t0 = Instant::now();
        let report = load_checkpoint_into(&mut model, &dir.join("model.safetensors"))
            .expect("load model.safetensors");
        println!(
            "[smoke] model.safetensors loaded in {:.2?} (applied={}, missing={}, unused={}, errors={})",
            t0.elapsed(),
            report.applied,
            report.missing.len(),
            report.unused.len(),
            report.errors.len()
        );
        assert!(report.missing.is_empty(), "missing: {:?}", report.missing);

        let tokenizer = VoxCpm2Tokenizer::from_file(&dir.join("tokenizer.json"))
            .expect("load tokenizer.json");

        let opts = GenerationOptions {
            min_len: 2,
            max_len: 10,
            inference_timesteps: 10,
            cfg_value: 2.0,
        };
        println!("[stage-b] generating {} patches with the 2B LM (slow on CPU)...", opts.max_len);
        let t0 = Instant::now();
        let out = model
            .generate(
                "今天天气真不错，我们出去走走吧。",
                ClonePrompt::None,
                &tokenizer,
                &opts,
                &vae,
                &device,
            )
            .expect("generate");
        println!("[stage-b] generate: {:.1?} for {} patches", t0.elapsed(), out.num_patches);

        let [_, gd, gt] = out.latent.dims();
        assert_eq!(gd, d, "LM latent dim");
        let gen_data: Vec<f32> = out.latent.to_data().to_vec().unwrap();
        println!("[stage-b] generated latent [1, {gd}, {gt}]");

        let (_, _, pass) = run_comparison(
            &vae,
            &out.latent,
            &gen_data,
            gd,
            gt,
            patch,
            "stage-b",
            wav_dir,
        );
        pass_b = Some(pass);
    }

    println!(
        "\n[smoke] RESULT: stage-a={}, stage-b={}",
        if pass_a { "PASS" } else { "FAIL" },
        match pass_b {
            Some(true) => "PASS".to_string(),
            Some(false) => "FAIL".to_string(),
            None => "skipped (pass --with-lm)".to_string(),
        }
    );
    if pass_a && pass_b.unwrap_or(true) {
        println!(
            "[smoke] StreamingVaeDecoder reproduces the full decode on real weights — \
             the HTTP synthesize_stream override is unblocked."
        );
    }
}
