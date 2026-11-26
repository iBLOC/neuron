//! **声纹探针** —— 真机读数：真权重 ＋ 真音频 ⇒ 一条真向量。
//!
//! 用法：
//! ```text
//! set NEURON_SPK_MODEL_PATH=<含 campplus.onnx 的目录>
//! set ORT_DYLIB_PATH=<onnxruntime.dll 所在目录>
//! cargo run --offline --features speaker-embedding --example speaker_embedding_probe -- a.wav b.wav
//! ```
//!
//! 输出：每个文件的维度／模长／前 5 个分量；两个文件之间的**余弦相似度**
//! （同一个人的两段录音 ⇒ 接近 1；不同的人 ⇒ 明显更低 ⇒ 这就是「认得出」
//! 与「认不错」的第一道真机对照）。

use neuron_lib::spk::{exe_dir, resolve_model_path, SpeakerEmbeddingRuntime};

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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let files: Vec<String> = std::env::args().skip(1).collect();
    if files.is_empty() {
        eprintln!("usage: speaker_embedding_probe <a.wav> [b.wav]");
        std::process::exit(2);
    }

    let env = neuron_lib::config::env_var("SPK_MODEL_PATH");
    let dir = exe_dir();
    let path = match resolve_model_path(env.as_deref(), dir.as_deref(), None) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("model resolution failed: {e}");
            std::process::exit(2);
        }
    };
    println!("model: {}", path.display());
    // ⚠️ **显式 flush** —— ort 的 load-dynamic 在进程退出时会 abort（见 AGENTS）
    // ⇒ ⇒ 不 flush 的话 stdout 缓冲区会跟着一起丢 ⇒ ⇒ 读数「看起来没发生」。
    use std::io::Write;
    std::io::stdout().flush().ok();

    #[cfg(feature = "speaker-embedding")]
    let rt = neuron_lib::spk::campplus::CampplusSpeakerEmbeddingRuntime::open(&path)?;
    #[cfg(not(feature = "speaker-embedding"))]
    {
        eprintln!("this build has no `speaker-embedding` feature");
        std::process::exit(2);
    }

    let mut vectors: Vec<(String, Vec<f32>)> = Vec::new();
    for f in &files {
        let bytes = std::fs::read(f)?;
        let spk = rt.embed(&bytes).await.map_err(|e| format!("{f}: {e}"))?;
        let norm: f32 = spk.embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
        let head: Vec<String> = spk
            .embedding
            .iter()
            .take(5)
            .map(|v| format!("{v:.4}"))
            .collect();
        println!(
            "{f}: dim={} norm={norm:.4} head=[{}]",
            spk.dim,
            head.join(", ")
        );
        vectors.push((f.clone(), spk.embedding));
        std::io::stdout().flush().ok();
    }

    if vectors.len() >= 2 {
        for i in 0..vectors.len() {
            for j in (i + 1)..vectors.len() {
                let sim = cosine(&vectors[i].1, &vectors[j].1);
                println!("cosine({}, {}) = {sim:.4}", vectors[i].0, vectors[j].0);
            }
        }
    }
    Ok(())
}
