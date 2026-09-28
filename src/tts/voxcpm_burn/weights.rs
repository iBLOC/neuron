//! Direct checkpoint → burn-module loading for VoxCPM2.
//!
//! Upstream ships `model.safetensors` (HF, bf16) and `audiovae.pth` (PyTorch
//! pickle, fp32). burn-store can read both formats natively, so neuron needs
//! no external weight-conversion step — but the raw tensors must be adapted
//! before they fit the burn module tree:
//!
//! 1. **dtype**: checkpoints ship bf16 (safetensors) / fp32 (pth); the ndarray
//!    backend this runtime targets has no bf16 matmul, so every tensor is
//!    converted to f32 on load.
//! 2. **Linear layout**: burn 0.21 stores `Linear.weight` as `[d_in, d_out]`
//!    while PyTorch ships `[d_out, d_in]` — 2-D `*.weight` tensors are
//!    transposed (embedding / VAE sr-condition weights keep PyTorch's
//!    `[rows, cols]` and are excluded by the `embed` name rule).
//! 3. **norm naming**: `MiniCPMRMSNorm` wraps burn's `RmsNorm`, whose param is
//!    `gamma` behind our `inner` field, while PyTorch names it `weight` —
//!    `*.input_layernorm.weight` / `*.post_attention_layernorm.weight` /
//!    `*.norm.weight` are rewritten to `*.inner.gamma`.
//! 4. **enum variants**: the audiovae/decoder layer sequences are burn enums;
//!    PyTorch paths carry no variant segment, so apply runs with
//!    `skip_enum_variants = true`.
//!
//! Conversion is lazy per tensor (`TensorSnapshot::from_closure`), so peak
//! memory is the f32 module plus one transient tensor rather than two full
//! copies of the checkpoint.

use std::path::Path;
use std::rc::Rc;

use burn::prelude::Backend;
use burn::tensor::{DType, Shape, TensorData};
use burn_store::{ModuleSnapshot, ModuleStore, PytorchStore, SafetensorsStore, TensorSnapshot};

/// Summary of one checkpoint application, mirrored out of burn-store's
/// `ApplyResult` so callers don't need the burn-store types.
#[derive(Debug, Default, Clone)]
pub struct LoadReport {
    pub applied: usize,
    pub missing: Vec<String>,
    pub unused: Vec<String>,
    pub errors: Vec<String>,
}

impl LoadReport {
    /// A load is only trustworthy when every module tensor found a checkpoint
    /// source and no shape/apply errors occurred. `unused` (checkpoint keys
    /// the module doesn't consume) is tolerated but reported.
    pub fn ensure_complete(&self, checkpoint: &str) -> Result<(), String> {
        if !self.errors.is_empty() {
            return Err(format!(
                "checkpoint '{checkpoint}': {} apply error(s): {}",
                self.errors.len(),
                self.errors.join("; ")
            ));
        }
        if !self.missing.is_empty() {
            return Err(format!(
                "checkpoint '{checkpoint}': {} module tensor(s) missing from checkpoint: {}",
                self.missing.len(),
                self.missing.join("; ")
            ));
        }
        if self.applied == 0 {
            return Err(format!(
                "checkpoint '{checkpoint}': no tensors were applied (naming mismatch?)"
            ));
        }
        Ok(())
    }
}

/// Load a PyTorch-family checkpoint (`.safetensors` / `.pth` / `.pt` / `.bin`)
/// into `module`, applying the dtype/layout/name adaptations listed in the
/// module docs.
pub fn load_checkpoint_into<B: Backend, M: ModuleSnapshot<B>>(
    module: &mut M,
    path: &Path,
) -> Result<LoadReport, String> {
    let snapshots = read_snapshots(path)?;
    let transformed = transform_snapshots(snapshots)?;

    let result = module.apply(transformed, None, None, true);

    let report = LoadReport {
        applied: result.applied.len(),
        missing: result.missing.iter().map(|(p, _)| p.clone()).collect(),
        unused: result.unused.iter().cloned().collect(),
        errors: result
            .errors
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>(),
    };
    report.ensure_complete(&path.display().to_string())?;
    Ok(report)
}

fn read_snapshots(path: &Path) -> Result<Vec<TensorSnapshot>, String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    match ext.as_str() {
        "safetensors" => {
            let mut store = SafetensorsStore::from_file(path);
            let snapshots = store
                .get_all_snapshots()
                .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
            Ok(snapshots.values().cloned().collect())
        }
        "pth" | "pt" | "bin" => {
            let mut store = PytorchStore::from_file(path);
            let snapshots = store
                .get_all_snapshots()
                .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
            Ok(snapshots.values().cloned().collect())
        }
        other => Err(format!(
            "unsupported checkpoint extension '{other}' for {} (expected .safetensors / .pth)",
            path.display()
        )),
    }
}

/// True for burn `Linear.weight` tensors: 2-D, suffix `.weight`, and not an
/// embedding (`embed_tokens`, `scale_embed`, `bias_embed`). Convs are 3-D and
/// already share PyTorch's `[out, in, k]` layout, so they pass through.
fn is_transposed_linear_weight(name: &str, rank: usize) -> bool {
    rank == 2 && name.ends_with(".weight") && !name.to_ascii_lowercase().contains("embed")
}

/// Rewrite burn-target paths for the RmsNorm wrapper naming.
fn remap_norm_path(name: &str) -> String {
    for stem in [
        "input_layernorm",
        "post_attention_layernorm",
        "norm",
    ] {
        let bare = format!("{stem}.weight");
        let dotted = format!(".{bare}");
        if name == bare.as_str() {
            return format!("{stem}.inner.gamma");
        }
        if let Some(base) = name.strip_suffix(&dotted) {
            return format!("{base}.{stem}.inner.gamma");
        }
    }
    name.to_string()
}

fn transpose_2d(data: &TensorData) -> Result<TensorData, String> {
    let rank = data.shape.rank();
    if rank != 2 {
        return Err(format!("transpose_2d expects rank 2, got rank {rank}"));
    }
    let dims = data.shape.to_vec();
    let (rows, cols) = (dims[0], dims[1]);
    let vec: Vec<f32> = data
        .to_vec()
        .map_err(|e| format!("failed to materialize tensor as f32: {e}"))?;
    let mut out = vec![0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = vec[i * cols + j];
        }
    }
    Ok(TensorData::new(out, Shape::from([cols, rows])))
}

fn transform_snapshots(snapshots: Vec<TensorSnapshot>) -> Result<Vec<TensorSnapshot>, String> {
    snapshots
        .into_iter()
        .map(|snapshot| {
            let name = snapshot.full_path();
            let name = remap_norm_path(&name);
            let path_stack: Vec<String> = name
                .split('.')
                .map(str::to_string)
                .collect();
            let container_stack = snapshot.container_stack.clone().unwrap_or_default();
            let tensor_id = snapshot.tensor_id.unwrap_or_default();

            let rank = snapshot.shape.rank();
            let dims = snapshot.shape.to_vec();
            let transposed = is_transposed_linear_weight(&name, rank);
            let out_dims: Vec<usize> = if transposed {
                vec![dims[1], dims[0]]
            } else {
                dims
            };

            let closure = Rc::new(move || {
                let mut data = snapshot
                    .to_data()
                    .map_err(|e| burn_store::TensorSnapshotError::DataError(e.to_string()))?;
                if data.dtype != DType::F32 {
                    data = data.convert::<f32>();
                }
                if transposed {
                    data = transpose_2d(&data)
                        .map_err(|e| burn_store::TensorSnapshotError::DataError(e))?;
                }
                Ok(data)
            });

            Ok(TensorSnapshot::from_closure(
                closure,
                DType::F32,
                Shape::from(out_dims),
                path_stack,
                container_stack,
                tensor_id,
            ))
        })
        .collect()
}

/// Test-only helpers shared by the voxcpm_burn checkpoint tests: build a
/// PyTorch-style fixture (transposed Linears, `weight`-named norms, flat
/// enum-less paths) from a burn module and serialize it to safetensors bytes.
#[cfg(all(test, feature = "voxcpm"))]
pub(crate) mod fixture {
    use burn::prelude::Backend;
    use burn::tensor::{TensorData};
    use burn_store::{ModuleSnapshot, TensorSnapshot};
    use std::collections::HashMap;

    use super::{is_transposed_linear_weight, transpose_2d};

    /// Serialize `{name: TensorData}` into safetensors bytes, mirroring what
    /// an upstream PyTorch `save_file` export looks like.
    pub fn serialize_safetensors(
        tensors: Vec<(String, TensorData)>,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        // TensorView borrows the payload, so the byte buffers must outlive
        // the map: stage them first, then reference them during serialize.
        let mut staged = Vec::new();
        for (name, data) in tensors {
            let f32s: Vec<f32> = data.to_vec()?;
            let mut bytes = Vec::with_capacity(f32s.len() * 4);
            for v in f32s {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            staged.push((name, data.shape.to_vec(), bytes));
        }
        let mut map = HashMap::new();
        for (name, shape, bytes) in &staged {
            let view = safetensors::tensor::TensorView::new(
                safetensors::tensor::Dtype::F32,
                shape.clone(),
                bytes,
            )
            .map_err(|e| format!("tensor view: {e}"))?;
            map.insert(name.clone(), view);
        }
        Ok(safetensors::serialize(&map, &None)?)
    }

    /// Build a PyTorch-style checkpoint from a burn module by inverting each
    /// adaptation that [`super::load_checkpoint_into`] performs.
    pub fn pytorch_style_checkpoint<B: Backend, M: ModuleSnapshot<B>>(
        module: &M,
    ) -> (Vec<u8>, Vec<(String, Vec<f32>)>) {
        let burn_snapshots = module.collect(None, None, true);
        let mut tensors = Vec::new();
        let mut expected = Vec::new();
        for snapshot in burn_snapshots {
            let burn_name = snapshot.full_path();
            let data = snapshot.to_data().unwrap();
            let f32s = data.to_vec::<f32>().unwrap();
            expected.push((burn_name.clone(), f32s));

            let (name, data) = if burn_name.ends_with(".inner.gamma") {
                (burn_name.replace(".inner.gamma", ".weight"), data)
            } else if is_transposed_linear_weight(&burn_name, data.shape.rank()) {
                (burn_name.clone(), transpose_2d(&data).unwrap())
            } else {
                (burn_name.clone(), data)
            };
            tensors.push((name, data));
        }
        (serialize_safetensors(tensors).unwrap(), expected)
    }
}

#[cfg(all(test, feature = "voxcpm"))]
mod tests {
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use burn::module::Module;
    use burn::tensor::{Int, Shape, Tensor, TensorData};
    use burn_store::TensorSnapshot;

    use super::fixture::pytorch_style_checkpoint;
    use super::*;

    #[test]
    fn test_transpose_2d_roundtrip() {
        let data = TensorData::new(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::from([2usize, 3]),
        );
        let t = transpose_2d(&data).unwrap();
        assert_eq!(t.shape.to_vec(), vec![3, 2]);
        let out: Vec<f32> = t.to_vec().unwrap();
        assert_eq!(out, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        // Transposing again restores the original layout.
        let back = transpose_2d(&t).unwrap();
        assert_eq!(back.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_transpose_2d_rejects_non_rank2() {
        let data = TensorData::new(vec![1.0f32; 8], Shape::from([2usize, 2, 2]));
        assert!(transpose_2d(&data).is_err());
    }

    #[test]
    fn test_is_transposed_linear_weight_rules() {
        // Linear weights → transpose.
        assert!(is_transposed_linear_weight("base_lm.layers.0.self_attn.q_proj.weight", 2));
        assert!(is_transposed_linear_weight("enc_to_lm_proj.weight", 2));
        assert!(is_transposed_linear_weight("stop_head.weight", 2));
        // Embeddings keep PyTorch layout.
        assert!(!is_transposed_linear_weight(
            "base_lm.embed_tokens.weight",
            2
        ));
        assert!(!is_transposed_linear_weight(
            "decoder.sr_cond_model.2.scale_embed.weight",
            2
        ));
        assert!(!is_transposed_linear_weight(
            "decoder.sr_cond_model.2.bias_embed.weight",
            2
        ));
        // Convs are 3-D.
        assert!(!is_transposed_linear_weight("encoder.block.0.weight_v", 3));
        // Norm tensors are renamed before this check in the real pipeline.
        assert!(!is_transposed_linear_weight("base_lm.norm.weight", 1));
    }

    #[test]
    fn test_remap_norm_path_rules() {
        assert_eq!(remap_norm_path("norm.weight"), "norm.inner.gamma");
        assert_eq!(
            remap_norm_path("base_lm.norm.weight"),
            "base_lm.norm.inner.gamma"
        );
        assert_eq!(
            remap_norm_path("base_lm.layers.0.input_layernorm.weight"),
            "base_lm.layers.0.input_layernorm.inner.gamma"
        );
        assert_eq!(
            remap_norm_path("feat_decoder.estimator.decoder.layers.3.post_attention_layernorm.weight"),
            "feat_decoder.estimator.decoder.layers.3.post_attention_layernorm.inner.gamma"
        );
        // Non-norm names pass through untouched.
        assert_eq!(
            remap_norm_path("base_lm.layers.0.self_attn.q_proj.weight"),
            "base_lm.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(remap_norm_path("fsq_layer.in_proj.bias"), "fsq_layer.in_proj.bias");
    }

    // -----------------------------------------------------------------------
    // End-to-end fixture: simulate the upstream PyTorch-style checkpoint
    // (PyTorch Linear layout, `weight`-named norms, flat enum-less paths),
    // then run it through `load_checkpoint_into` and verify every module
    // tensor is restored bit-exact. This exercises the whole adaptation
    // pipeline without shipping a multi-GB real checkpoint in CI.
    // -----------------------------------------------------------------------

    use burn::config::Config;

    #[test]
    fn test_load_checkpoint_into_restores_minicpm_weights() {
        use crate::tts::voxcpm_burn::minicpm4::{MiniCPMConfig, MiniCPMModel};

        let device = NdArrayDevice::Cpu;
        let rope = crate::tts::voxcpm_burn::minicpm4::RopeScalingConfig::new(
            "longrope".to_string(),
            vec![1.0, 1.0],
            vec![1.0, 1.0],
            8,
        );
        // burn's Config derive omits defaulted fields (kv_channels, no_rope)
        // from the generated `new`, so set them post-construction.
        let mut config = MiniCPMConfig::new(
            1,        // bos_token_id
            2,        // eos_token_id
            8,        // hidden_size (head_dim 4 × 2 heads)
            16,       // intermediate_size
            64,       // max_position_embeddings
            2,        // num_attention_heads
            2,        // num_hidden_layers
            1,        // num_key_value_heads
            1e-5,     // rms_norm_eps
            rope,     // rope_scaling
            16,       // vocab_size (embedding present)
            false,    // use_mup
            12.0,     // scale_emb
            1,        // dim_model_base
            1.4,      // scale_depth
            10000.0,  // rope_theta
        );
        config.no_rope = true; // exercises the Option-rope path
        let mut source: MiniCPMModel<NdArray> = config.init(Some((1, 64)), &device);

        let (bytes, _expected) = pytorch_style_checkpoint(&source);

        // Re-init a fresh module (different random init) and load through the
        // real checkpoint pipeline.
        let mut target: MiniCPMModel<NdArray> = config.init(Some((1, 64)), &device);

        let dir = std::env::temp_dir()
            .join(format!("neuron-voxcpm2-weights-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        std::fs::write(&path, &bytes).unwrap();

        let report = load_checkpoint_into(&mut target, &path);
        let _ = std::fs::remove_dir_all(&dir);

        let report = report.unwrap_or_else(|e| panic!("load failed: {e}"));
        assert!(report.missing.is_empty(), "missing: {:?}", report.missing);
        assert!(report.errors.is_empty(), "errors: {:?}", report.errors);
        assert!(report.applied > 0);
        assert!(report.unused.is_empty(), "unused: {:?}", report.unused);

        // The two modules must now agree tensor-for-tensor.
        let a = source.collect(None, None, false);
        let b = target.collect(None, None, false);
        assert_eq!(a.len(), b.len());
        for (sa, sb) in a.iter().zip(b.iter()) {
            assert_eq!(sa.full_path(), sb.full_path());
            let da = sa.to_data().unwrap();
            let db = sb.to_data().unwrap();
            assert_eq!(
                da.to_vec::<f32>().unwrap(),
                db.to_vec::<f32>().unwrap(),
                "tensor {} differs after load",
                sa.full_path()
            );
        }

        // And the forward pass still runs on the no_rope path.
        let input = Tensor::<NdArray, 2, Int>::from_data([[3i64, 5]], &device);
        let embeds = source
            .embed_tokens
            .as_ref()
            .unwrap()
            .forward(input);
        let _ = source.forward(embeds, true);
    }
}
