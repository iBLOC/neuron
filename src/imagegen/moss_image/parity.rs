//! Numerical parity against the upstream MOSS-Image implementation.
//!
//! The fixtures under `testdata/` were produced by
//! `scripts/gen-moss-parity.py`, which runs the **upstream code
//! itself** (the Gated DeltaNet kernel from the installed
//! `transformers` package, the DiT from the upstream
//! `pipeline_moss_image.py` in the model repo) and dumps the inputs
//! and the reference outputs. Nothing upstream is re-typed here, so
//! these tests compare neuron's candle port against the reference
//! implementation rather than against a second transcription of it.
//!
//! The fixtures are deterministic (synthesised from a fixed
//! trigonometric sequence, not drawn from an RNG), so the expected
//! values are reproducible bit for bit.
//!
//! Tolerances: the reference runs in fp32 PyTorch on the CPU; neuron
//! runs the same math in candle's fp32 CPU backend. The two differ
//! only by summation order inside the matmuls, so the tolerances are
//! relative-to-magnitude rather than absolute.

use std::path::{Path, PathBuf};

use candle::{DType, Device, Tensor};

use super::text_encoder::chunk_gated_delta_rule;

/// Where the checked-in fixtures live.
fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("imagegen")
        .join("moss_image")
        .join("testdata")
        .join(name)
}

/// Read a safetensors file into `(name → tensor)`.
fn read_fixture(name: &str) -> Vec<(String, Tensor)> {
    let path = fixture_path(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "missing parity fixture {} ({e}) — regenerate it with \
             `python scripts/gen-moss-parity.py`",
            path.display()
        )
    });
    let device = Device::Cpu;
    let mut tensors: std::collections::HashMap<String, Tensor> =
        candle::safetensors::load_buffer(&bytes, &device)
            .unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
    for (key, tensor) in tensors.iter_mut() {
        *tensor = tensor
            .to_dtype(DType::F32)
            .unwrap_or_else(|e| panic!("dtype {key}: {e}"));
    }
    let mut out: Vec<(String, Tensor)> = tensors.into_iter().collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn get(fixture: &[(String, Tensor)], name: &str) -> Tensor {
    fixture
        .iter()
        .find(|(k, _)| k == name)
        .map(|(_, t)| t.clone())
        .unwrap_or_else(|| panic!("fixture has no tensor {name}"))
}

/// Max absolute difference, and the magnitude of the reference, for a
/// relative-error report.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imagegen::moss_image::dit::{MossDit, MossDitConfig};

    /// The Gated DeltaNet chunk kernel must match the upstream
    /// PyTorch implementation element-wise.
    ///
    /// This is the kernel where the port had the most subtle bugs
    /// (the UT transform's rank, the `value` reassignment, the decay
    /// mask's order, the un-exponentiated `g_last`), so it carries a
    /// real numerical anchor rather than a shape check.
    #[test]
    fn test_gdn_kernel_matches_upstream() {
        let fixture = read_fixture("moss_gdn_parity.safetensors");
        let query = get(&fixture, "query");
        let key = get(&fixture, "key");
        let value = get(&fixture, "value");
        let g = get(&fixture, "g");
        let beta = get(&fixture, "beta");
        let expected = get(&fixture, "output");

        let out = chunk_gated_delta_rule(&query, &key, &value, &g, &beta)
            .expect("gdn kernel");
        assert_eq!(out.dims(), expected.dims());

        let (max_abs, max_ref) = max_diff(&out, &expected);
        let rel = if max_ref > 0.0 { max_abs / max_ref } else { max_abs };
        println!(
            "gdn parity: max abs diff {max_abs:.3e}, reference magnitude {max_ref:.3e}, \
             relative {rel:.3e}"
        );
        // The chunked scan accumulates in fp32 with a different
        // summation order than PyTorch's; 1e-4 relative is well
        // inside that noise and far below any semantic difference.
        assert!(
            rel < 1e-4,
            "gdn kernel diverges from upstream: max abs {max_abs:.3e}, relative {rel:.3e}"
        );
    }

    /// The whole diffusion transformer must match the upstream
    /// implementation element-wise — for the normal path and for the
    /// CFG negative branch that skips a block.
    #[test]
    fn test_dit_matches_upstream() {
        let fixture = read_fixture("moss_dit_parity.safetensors");
        let cfg = MossDitConfig {
            dim: 64,
            ffn_dim: 128,
            num_heads: 4,
            num_layers: 2,
            text_dim: 96,
            freq_dim: 32,
            latent_channels: 8,
            rope_base: 10_000.0,
        };

        // Load the upstream weights (their names are the checkpoint's
        // own — the same paths the runtime uses).
        let mut tensors: std::collections::HashMap<String, Tensor> = Default::default();
        for (name, tensor) in &fixture {
            if let Some(stripped) = name.strip_prefix("weight::") {
                tensors.insert(stripped.to_string(), tensor.clone());
            }
        }
        let device = Device::Cpu;
        let vb = crate::candle_compat::from_tensors(tensors, DType::F32, &device);
        let dit = MossDit::new(vb, cfg).expect("build the tiny dit");

        let latents = get(&fixture, "latents");
        let text = get(&fixture, "text");
        let expected = get(&fixture, "output");
        let expected_skip = get(&fixture, "output_skip0");

        let out = dit.forward(&latents, 500.0, &text, &[]).expect("forward");
        let (max_abs, max_ref) = max_diff(&out, &expected);
        let rel = if max_ref > 0.0 { max_abs / max_ref } else { max_abs };
        println!(
            "dit parity: max abs diff {max_abs:.3e}, reference magnitude {max_ref:.3e}, \
             relative {rel:.3e}"
        );
        assert!(
            rel < 1e-4,
            "the DiT diverges from upstream: max abs {max_abs:.3e}, relative {rel:.3e}"
        );

        // The CFG negative branch skips block 0 — the port must skip
        // the same block and produce the same output.
        let out_skip = dit
            .forward(&latents, 500.0, &text, &[0])
            .expect("forward with a skipped block");
        let (max_abs, max_ref) = max_diff(&out_skip, &expected_skip);
        let rel = if max_ref > 0.0 { max_abs / max_ref } else { max_abs };
        println!(
            "dit parity (block skipped): max abs diff {max_abs:.3e}, relative {rel:.3e}"
        );
        assert!(
            rel < 1e-4,
            "the DiT's block skipping diverges from upstream: max abs {max_abs:.3e}, \
             relative {rel:.3e}"
        );
    }
}
