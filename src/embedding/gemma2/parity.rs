//! Element-wise numeric parity against the upstream transformers
//! implementation, on the real checkpoint.
//!
//! The fixture `testdata/egemma2_parity.safetensors` is produced by
//! `scripts/gen-egemma2-parity.py`, which runs the upstream
//! `EmbeddingGemma2Model` itself (transformers >= 5.18, fp32, eager
//! attention) on the real `google/embeddinggemma-2` weights and dumps
//! its `last_hidden_state` — nothing upstream is re-typed, so the
//! reference is a real reference rather than a second transcription.
//! The input batch mixes sequence lengths so the bidirectional
//! padding-mask path is exercised.
//!
//! The Rust test needs both the fixture (checked in) and the real
//! checkpoint (1.49 GB, not in CI), so it is `#[ignore]`d:
//!
//! ```bash
//! cargo test --lib egemma2_parity -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use candle::{DType, Device, Tensor, D};

/// Where the checked-in fixture lives.
fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("embedding")
        .join("gemma2")
        .join("testdata")
        .join(name)
}

/// Max absolute difference and the reference magnitude, for the
/// relative-error report (same metric as the MOSS parity tests).
fn max_diff(a: &Tensor, b: &Tensor) -> (f64, f64) {
    let av = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let bv = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(av.len(), bv.len(), "shape mismatch");
    let mut max_abs = 0.0f64;
    let mut max_ref = 0.0f64;
    for (x, y) in av.iter().zip(bv.iter()) {
        let d = (x - y).abs() as f64;
        if d > max_abs {
            max_abs = d;
        }
        let m = y.abs() as f64;
        if m > max_ref {
            max_ref = m;
        }
    }
    (max_abs, max_ref)
}

/// Locate the checkpoint: `NEURON_EMBEDDING_MODEL_PATH`, else the
/// default `gemma2/embeddinggemma-2` subtree (same rule as the
/// runtime).
fn checkpoint_dir() -> Option<PathBuf> {
    if let Ok(p) = crate::config::env_var_result("EMBEDDING_MODEL_PATH") {
        let path = PathBuf::from(p);
        if path.join("model.safetensors").exists() {
            return Some(path);
        }
        return None;
    }
    let path = crate::config::NeuronConfig::resolved_model_dir()
        .join("gemma2")
        .join("embeddinggemma-2");
    if path.join("model.safetensors").exists() {
        Some(path)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::gemma2::config::TextConfig;
    use crate::embedding::gemma2::text_model::TextModel;

    /// The vendored tower must reproduce the upstream fp32 forward
    /// element-wise on the real weights.
    #[tokio::test]
    #[serial_test::serial]
    #[ignore]
    async fn egemma2_parity() {
        let fixture_file = fixture_path("egemma2_parity.safetensors");
        if !fixture_file.exists() {
            eprintln!(
                "SKIP: parity fixture {} missing — regenerate with \
                 scripts/gen-egemma2-parity.py (needs a venv with \
                 transformers >= 5.18)",
                fixture_file.display()
            );
            return;
        }
        let Some(model_dir) = checkpoint_dir() else {
            eprintln!(
                "SKIP: checkpoint missing — download google/embeddinggemma-2 \
                 to <model root>/gemma2/embeddinggemma-2/ or set \
                 NEURON_EMBEDDING_MODEL_PATH"
            );
            return;
        };

        // Fixture: upstream inputs + reference output.
        let bytes = std::fs::read(&fixture_file).expect("read fixture");
        let device = Device::Cpu;
        let fixture = candle::safetensors::load_buffer(&bytes, &device).expect("parse fixture");
        let ids = fixture["input_ids"]
            .to_dtype(DType::U32)
            .expect("ids dtype");
        let mask = fixture["attention_mask"]
            .to_dtype(DType::U32)
            .expect("mask dtype");
        let reference = fixture["reference"].clone();

        // Port: real checkpoint config + weights, fp32 CPU.
        let raw = std::fs::read(model_dir.join("config.json")).expect("read config.json");
        let json: serde_json::Value = serde_json::from_slice(&raw).expect("parse config.json");
        let cfg: TextConfig = serde_json::from_value(json["text_config"].clone()).expect("parse text_config");
        let vb = unsafe {
            crate::candle_compat::from_mmaped_safetensors(
                &[model_dir.join("model.safetensors")],
                DType::F32,
                &device,
            )
        }
        .expect("mmap weights");
        let model = TextModel::new(&cfg, vb.pp("language_model")).expect("build TextModel");

        let has_pad = mask
            .min(D::Minus1)
            .and_then(|t| t.min(D::Minus1))
            .and_then(|t| t.reshape(()))
            .and_then(|t| t.to_scalar::<u32>())
            .unwrap_or(1)
            == 0;
        let out = model
            .forward(&ids, if has_pad { Some(&mask) } else { None })
            .expect("forward");
        assert_eq!(out.dims(), reference.dims(), "output shape");

        let (max_abs, max_ref) = max_diff(&out, &reference);
        let rel = max_abs / max_ref.max(1e-12);
        println!("egemma2 parity: max_abs={max_abs:.3e}, max_ref={max_ref:.3e}, rel={rel:.3e}");
        assert!(
            rel < 2e-4,
            "parity drifted: max_abs={max_abs:.3e} / max_ref={max_ref:.3e} = {rel:.3e} (fp32 CPU vs upstream fp32 eager)"
        );
    }
}
