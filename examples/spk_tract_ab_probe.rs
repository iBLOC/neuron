//! **声纹后端 A/B 探针** —— tract（纯 Rust）vs ort（onnxruntime）。
//!
//! ## 为什么要有这个探针
//!
//! 「声纹是内置必须的功能」⇒ ⇒ **每个发行产物**（桌面 exe／APK）都得含它。
//! 而 `speaker-embedding` 今天蕴含 `ort`（load-dynamic）⇒ ⇒ 发行包**还要带
//! onnxruntime.dll** ⇒ ⇒ 纯 Rust 后端（tract）能替掉它，那这一格才可能
//! 「零配置」地进每个渠道。
//!
//! ⇒ ⇒ 所以要**量**：同一份 fbank、同一个权重，**tract 的向量与 ort 的向量
//! 差多少**（余弦）—— 「能跑」与「跑对」是两件事。
//!
//! ## 用法
//!
//! ```text
//! cargo run --offline --no-default-features --features speaker-embedding,tract \
//!   --example spk_tract_ab_probe -- <arm> <campplus.onnx> <a.wav> [b.wav]
//!
//! arm = tract | ort
//! ```
//!
//! 输出：每条臂一行 `ARM=… FILE=… DIM=… NORM=… VEC=…`（192 个数，逗号分隔）。
//!
//! ⚠️ **两条臂的 fbank 必须是同一份** ⇒ ⇒ 这里直接调产品路径那份实现
//! （`neuron_lib::spk::campplus::kaldi_fbank` ＋ `mean_normalize`），
//! **不重写第二份**（特征提取只有一份实现，不会漂）。

use std::io::Write;

/// CAM++ 输出维度。
const DIM: usize = 192;
/// fbank 的 mel bin 数。
const MEL: usize = 80;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// 产品路径那份 fbank（**唯一一份实现**）。
#[cfg(feature = "speaker-embedding")]
fn feats_of(samples: &[f32]) -> (Vec<f32>, usize) {
    let mut f = neuron_lib::spk::campplus::kaldi_fbank(samples, 16_000);
    let frames = f.len() / MEL;
    neuron_lib::spk::campplus::mean_normalize(&mut f, frames, MEL);
    (f, frames)
}

/// **tract 臂** —— 纯 Rust ONNX 执行后端（发行包不必带 onnxruntime.dll）。
#[cfg(feature = "tract")]
fn tract_embedding(
    model_path: &str,
    feats: &[f32],
    frames: usize,
) -> Result<Vec<f32>, String> {
    use tract_onnx::prelude::*;

    let mut model = tract_onnx::onnx()
        .model_for_path(model_path)
        .map_err(|e| format!("tract load {model_path}: {e}"))?;
    let n_in = model.inputs.len();
    if n_in == 0 {
        return Err("campplus graph has no inputs".to_string());
    }
    for idx in 0..n_in {
        model
            .set_input_fact(idx, f32::fact([1, frames, MEL]).into())
            .map_err(|e| format!("tract input fact: {e}"))?;
    }
    let model = model
        .into_optimized()
        .map_err(|e| format!("tract optimize: {e}"))?
        .into_runnable()
        .map_err(|e| format!("tract runnable: {e}"))?;
    let tensor = Tensor::from_shape(&[1, frames, MEL], feats)
        .map_err(|e| format!("tract tensor: {e}"))?;
    let out = model
        .run(tvec!(tensor.into()))
        .map_err(|e| format!("tract run: {e}"))?;
    let arr = out[0]
        .to_array_view::<f32>()
        .map_err(|e| format!("tract output: {e}"))?;
    Ok(arr.iter().copied().collect())
}

/// **ort 臂** —— onnxruntime（今天的实现）。
#[cfg(feature = "speaker-embedding")]
fn ort_embedding(model_path: &str, samples: &[f32]) -> Result<Vec<f32>, String> {
    let ex = neuron_lib::spk::campplus::CampplusExtractor::open(std::path::Path::new(model_path))?;
    ex.embedding(samples)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!("usage: spk_tract_ab_probe <arm:tract|ort> <model.onnx> <a.wav> [b.wav]");
        std::process::exit(2);
    }
    let arm = args[0].clone();
    let model_path = args[1].clone();

    // WAV → 16 kHz 单声道 f32（与产品路径同一个解码器）。
    let mut wavs: Vec<(String, Vec<f32>)> = Vec::new();
    for f in &args[2..] {
        let bytes = match std::fs::read(f) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("read {f}: {e}");
                std::process::exit(2);
            }
        };
        match neuron_lib::asr::wav::decode_wav(&bytes) {
            Ok((samples, sr)) => {
                eprintln!("wav {f}: sr={sr} samples={}", samples.len());
                wavs.push((f.clone(), samples));
            }
            Err(e) => {
                eprintln!("wav {f}: {e}");
                std::process::exit(2);
            }
        }
    }
    if wavs.is_empty() {
        eprintln!("no wav");
        std::process::exit(2);
    }

    let mut vectors: Vec<(String, Vec<f32>)> = Vec::new();

    match arm.as_str() {
        "tract" => {
            // 这一臂要两个 feature 同时在 —— fbank 那份实现住在
            // speaker-embedding 后面 => 只开 tract 时这一臂编不过
            // => 用「两个都要」的 cfg，不许留一个半开的组合。
            #[cfg(all(feature = "tract", feature = "speaker-embedding"))]
            {
                for (f, samples) in &wavs {
                    let (feats, frames) = feats_of(samples);
                    match tract_embedding(&model_path, &feats, frames) {
                        Ok(v) => {
                            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                            let csv = v
                                .iter()
                                .map(|x| format!("{x:.6}"))
                                .collect::<Vec<_>>()
                                .join(",");
                            println!("ARM=tract FILE={f} DIM={} NORM={norm:.6} VEC={csv}", v.len());
                            std::io::stdout().flush().ok();
                            vectors.push((f.clone(), v));
                        }
                        Err(e) => {
                            eprintln!("tract arm failed on {f}: {e}");
                            std::process::exit(3);
                        }
                    }
                }
            }
            #[cfg(not(all(feature = "tract", feature = "speaker-embedding")))]
            {
                eprintln!("the tract arm needs BOTH tract and speaker-embedding");
                std::process::exit(2);
            }
        }
        "ort" => {
            #[cfg(feature = "speaker-embedding")]
            {
                for (f, samples) in &wavs {
                    match ort_embedding(&model_path, samples) {
                        Ok(v) => {
                            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                            let csv = v
                                .iter()
                                .map(|x| format!("{x:.6}"))
                                .collect::<Vec<_>>()
                                .join(",");
                            println!("ARM=ort FILE={f} DIM={} NORM={norm:.6} VEC={csv}", v.len());
                            std::io::stdout().flush().ok();
                            vectors.push((f.clone(), v));
                        }
                        Err(e) => {
                            eprintln!("ort arm failed on {f}: {e}");
                            std::process::exit(3);
                        }
                    }
                }
            }
            #[cfg(not(feature = "speaker-embedding"))]
            {
                eprintln!("this build has no `speaker-embedding` feature");
                std::process::exit(2);
            }
        }
        other => {
            eprintln!("unknown arm {other}");
            std::process::exit(2);
        }
    }

    // 同一条臂内：不同文件的余弦（不同人 ⇒ 明显低于 1）。
    for i in 0..vectors.len() {
        for j in (i + 1)..vectors.len() {
            println!(
                "SELF-COSINE({arm}) {} vs {} = {:.6}",
                vectors[i].0,
                vectors[j].0,
                cosine(&vectors[i].1, &vectors[j].1)
            );
        }
    }
    std::io::stdout().flush().ok();
}
