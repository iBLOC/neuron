//! Real-weight parity: neuron's port vs the upstream implementation on
//! the **actual MOSS-Image checkpoint**.
//!
//! The synthetic fixtures in `parity.rs` verify the formulas; these
//! tests verify the real weights end to end, component by component
//! (each component is loaded on its own so it fits in a 24 GB card —
//! the whole pipeline at once does not).
//!
//! The references come from `scripts/gen-moss-real-ref.py`, which runs
//! the upstream code (transformers' `Qwen3_5Model`, the repo's own
//! `pipeline_moss_image.py`, diffusers' `AutoencoderKLFlux2`) on the
//! same checkpoint.
//!
//! Run:
//!
//! ```text
//! python scripts/gen-moss-real-ref.py --model-dir <dir>
//! cargo test --lib moss_real_weights -- --ignored --nocapture
//! ```
//!
//! The model directory is taken from `NEURON_IMAGEGEN_MODEL_PATH`
//! (or the standard `$NEURON_MODEL_DIRS/imagegen/<id>/` layout) and
//! the references from `target/moss-real/`.

use std::path::{Path, PathBuf};

use candle::{DType, Device, Tensor};

use super::config::{load_configs, shard_paths};
use super::dit::{MossDit, MossDitConfig};
use super::text_encoder::{Qwen35TextConfig, Qwen35TextModel};
use super::vae::{Flux2Vae, Flux2VaeConfig};

/// The model directory: the env var first, then the catalog layout.
fn model_dir() -> PathBuf {
    if let Ok(path) = crate::config::env_var_result("IMAGEGEN_MODEL_PATH") {
        return PathBuf::from(path);
    }
    let config = crate::config::NeuronConfig::from_env();
    for root in &config.model_dirs {
        let candidate = root.join("imagegen").join("moss-image");
        if candidate.join("transformer").is_dir() {
            return candidate;
        }
    }
    panic!(
        "no MOSS-Image model directory found — set NEURON_IMAGEGEN_MODEL_PATH or pull the \
         model into $NEURON_MODEL_DIRS/imagegen/moss-image/"
    );
}

/// The reference outputs written by `scripts/gen-moss-real-ref.py`.
fn reference_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("moss-real")
}

fn read_reference(name: &str) -> Option<Vec<(String, Tensor)>> {
    let path = reference_dir().join(name);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(_) => {
            println!(
                "skipping: no reference at {} — generate it with \
                 `python scripts/gen-moss-real-ref.py --only {}`",
                path.display(),
                name.trim_end_matches("_ref.safetensors")
            );
            return None;
        }
    };
    let device = Device::Cpu;
    let tensors: std::collections::HashMap<String, Tensor> =
        candle::safetensors::load_buffer(&bytes, &device)
            .unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
    let mut out: Vec<(String, Tensor)> = tensors.into_iter().collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Some(out)
}

fn get(fixture: &[(String, Tensor)], name: &str) -> Tensor {
    fixture
        .iter()
        .find(|(k, _)| k == name)
        .map(|(_, t)| t.clone())
        .unwrap_or_else(|| panic!("reference has no tensor {name}"))
}

/// Max absolute difference and the reference magnitude.
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

/// Report the relative error against the reference magnitude.
fn report(label: &str, got: &Tensor, expected: &Tensor, tolerance: f64) {
    let (max_abs, max_ref) = max_diff(got, expected);
    let rel = if max_ref > 0.0 { max_abs / max_ref } else { max_abs };
    println!(
        "{label}: max abs diff {max_abs:.3e}, reference magnitude {max_ref:.3e}, \
         relative {rel:.3e}"
    );
    assert!(
        rel < tolerance,
        "{label} diverges from the upstream reference: max abs {max_abs:.3e}, \
         relative {rel:.3e} (tolerance {tolerance:.1e})"
    );
}

/// The device the real-weight tests run on.
fn device() -> Device {
    match crate::config::env_var("DEVICE").as_deref() {
        Some("cuda") | Some("nvidia") => {
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(0).expect("cuda device")
            }
            #[cfg(not(feature = "cuda"))]
            {
                panic!("the cuda feature is not enabled")
            }
        }
        _ => Device::Cpu,
    }
}

/// The dtype the real weights are loaded in. `f16` matches the
/// reference script's default, so both sides round the bf16
/// checkpoint identically.
fn dtype() -> DType {
    match crate::config::env_var("IMAGEGEN_DTYPE").as_deref() {
        Some("f32") | Some("float32") => DType::F32,
        Some("bf16") | Some("bfloat16") => DType::BF16,
        _ => DType::F16,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The text encoder on the real checkpoint: the hidden states of a
    /// real prompt must match transformers' `Qwen3_5Model`.
    #[test]
    #[ignore = "needs the real checkpoint (see the module docs)"]
    fn test_text_encoder_matches_upstream_on_real_weights() {
        let dir = model_dir();
        let configs = load_configs(&dir).expect("configs");
        let Some(reference) = read_reference("text_encoder_ref.safetensors") else {
            return;
        };
        let expected = get(&reference, "hidden");
        let ids = get(&reference, "token_ids");

        let dev = device();
        let dt = dtype();
        let text_dir = dir.join("text_encoder");
        let paths =
            shard_paths(&text_dir, "model.safetensors.index.json").expect("shards");
        let vb = unsafe {
            crate::candle_compat::from_mmaped_safetensors(&paths, dt, &dev)
        }
        .expect("text encoder weights");
        let model = Qwen35TextModel::new(vb.pp("language_model"), configs.text_encoder)
            .expect("text encoder");

        let tokens = ids.to_dtype(DType::U32).expect("token ids");
        let hidden = model.forward_hidden(&tokens).expect("forward");
        report("text encoder (real weights)", &hidden, &expected, 5e-2);
    }

    /// The DiT on the real checkpoint, fed the text encoder's real
    /// hidden states.
    #[test]
    #[ignore = "needs the real checkpoint (see the module docs)"]
    fn test_dit_matches_upstream_on_real_weights() {
        let dir = model_dir();
        let configs = load_configs(&dir).expect("configs");
        let Some(reference) = read_reference("dit_ref.safetensors") else {
            return;
        };
        let latents = get(&reference, "latents");
        let encoder_states = get(&reference, "encoder_hidden_states");
        let expected = get(&reference, "output");

        let dev = device();
        let dt = dtype();
        let transformer_dir = dir.join("transformer");
        let paths = shard_paths(
            &transformer_dir,
            "diffusion_pytorch_model.safetensors.index.json",
        )
        .expect("shards");
        let vb = unsafe {
            crate::candle_compat::from_mmaped_safetensors(&paths, dt, &dev)
        }
        .expect("transformer weights");
        let dit = MossDit::new(vb, configs.dit).expect("dit");

        let out = dit
            .forward(&latents, 500.0, &encoder_states, &[])
            .expect("forward");
        report("dit (real weights)", &out, &expected, 5e-2);
    }

    /// The FLUX.2 decoder on the real checkpoint.
    #[test]
    #[ignore = "needs the real checkpoint (see the module docs)"]
    fn test_vae_matches_upstream_on_real_weights() {
        let dir = model_dir();
        let configs = load_configs(&dir).expect("configs");
        let Some(reference) = read_reference("vae_ref.safetensors") else {
            return;
        };
        let packed = get(&reference, "packed_latent");
        let expected = get(&reference, "decoded");

        let dev = device();
        let vae_path = dir
            .join("vae")
            .join("diffusion_pytorch_model.safetensors");
        let vb = unsafe {
            crate::candle_compat::from_mmaped_safetensors(&[vae_path], DType::F32, &dev)
        }
        .expect("vae weights");
        let vae = Flux2Vae::new(vb, configs.vae).expect("vae");

        let decoded = vae.decode(&packed).expect("decode");
        report("vae (real weights)", &decoded, &expected, 5e-2);
    }

    /// neuron's hard-coded prompt template must render exactly what
    /// the upstream chat template produces — and tokenize to the same
    /// ids.
    ///
    /// Only the tokenizer is needed, so this check does not depend on
    /// any weights being present.
    #[test]
    #[ignore = "needs the real checkpoint's tokenizer (see the module docs)"]
    fn test_prompt_template_matches_upstream_jinja() {
        let path = reference_dir().join("prompt_ref.safetensors");
        let Ok(bytes) = std::fs::read(&path) else {
            println!(
                "skipping: no reference at {} — generate it with \
                 `python scripts/gen-moss-real-ref.py --only prompt`",
                path.display()
            );
            return;
        };
        let metadata = read_safetensors_metadata(&bytes);
        let upstream = metadata
            .get("rendered_prompt")
            .expect("the reference must record the rendered prompt")
            .clone();
        let prompt = metadata
            .get("prompt")
            .expect("the reference must record the prompt")
            .clone();

        // 1. The rendered text must be identical.
        let ours = super::super::render_prompt(&prompt);
        assert_eq!(
            ours, upstream,
            "neuron's hard-coded prompt template does not render what the upstream chat \
             template renders"
        );

        // 2. And it must tokenize to the same ids.
        let tokenizer_path = model_dir().join("tokenizer").join("tokenizer.json");
        let tokenizer =
            tokenizers::Tokenizer::from_file(&tokenizer_path).expect("tokenizer");
        let ours_ids = tokenizer
            .encode(ours.as_str(), false)
            .expect("encode")
            .get_ids()
            .to_vec();
        let prompt_reference =
            read_reference("prompt_ref.safetensors").expect("the prompt reference");
        let reference_ids: Vec<u32> = get(&prompt_reference, "token_ids")
            .to_vec1::<i32>()
            .expect("ids")
            .into_iter()
            .map(|v| v as u32)
            .collect();
        assert_eq!(
            ours_ids, reference_ids,
            "neuron's prompt template does not tokenize like the upstream one"
        );
        println!(
            "prompt template: identical rendering and {} identical token ids",
            ours_ids.len()
        );
    }
}

/// The `__metadata__` block of a safetensors file.
fn read_safetensors_metadata(bytes: &[u8]) -> std::collections::HashMap<String, String> {
    if bytes.len() < 8 {
        panic!("short safetensors file");
    }
    let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let header: serde_json::Value =
        serde_json::from_slice(&bytes[8..8 + header_len]).expect("safetensors header");
    header
        .get("__metadata__")
        .and_then(|m| m.as_object())
        .map(|m| {
            m.iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                .collect()
        })
        .unwrap_or_default()
}
