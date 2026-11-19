//! Direct checkpoint → burn-module loading for Ming-3.0-tts.
//!
//! Upstream ships ONE `model.safetensors` (bf16) holding the backbone, the
//! DiT flow head, the aggregator and the AudioVAE — all under the PyTorch
//! module tree of `BailingMMNativeForConditionalGeneration`. The shared
//! machinery in [`crate::tts::weights`] does the dtype/transpose/norm
//! adaptations; this module only owns the **path remap** from the upstream
//! tree to neuron's `Ming3Model` fields:
//!
//! | upstream key prefix                    | burn module path        |
//! |----------------------------------------|-------------------------|
//! | `model.model.`                          | `backbone.`             |
//! | `audio.encoder.encoder.`                | `encoder.backbone.`     |
//! | `audio.encoder.aggregator.`             | `encoder.aggregator.`   |
//! | `audio.encoder.`                        | `encoder.`              |
//! | `audio.decoder.decoder.`                | `decoder.backbone.`     |
//! | `audio.decoder.head.istft.`             | `decoder.head.`         |
//! | `audio.decoder.`                        | `decoder.`              |
//! | `flowloss.cfm.model.t_embedder.time_mlp.{0,2}.` | `flowloss.t_embedder.{fc_in,fc_out}.` |
//! | `flowloss.cfm.model.c_embedder.cond_embedder.`  | `flowloss.c_embedder.` |
//! | `flowloss.cfm.model.`                   | `flowloss.`             |
//! | (suffix) `.mlp.ff.0.0.`                 | `.mlp.project_in.`      |
//! | (suffix) `.mlp.ff.2.`                   | `.mlp.project_out.`     |
//!
//! Keys the burn tree does not consume are tolerated as `unused` and
//! reported: the dead `audio.encoder.norm` LayerNorm (defined but never
//! called upstream) and the precomputed RoPE `inv_freq` buffers.

use std::path::Path;

use burn::prelude::Backend;
use burn_store::ModuleSnapshot;

use crate::tts::weights::{transform_snapshots, LoadReport};

/// Rewrite one upstream checkpoint key into the burn module tree.
pub(crate) fn remap_ming_path(name: &str) -> String {
    // Component prefixes (longest first).
    let prefixes = [
        ("model.model.", "backbone."),
        ("audio.encoder.encoder.", "encoder.backbone."),
        ("audio.encoder.aggregator.", "encoder.aggregator."),
        ("audio.encoder.", "encoder."),
        ("audio.decoder.decoder.", "decoder.backbone."),
        ("audio.decoder.head.istft.", "decoder.head."),
        ("audio.decoder.", "decoder."),
        ("flowloss.cfm.model.", "flowloss."),
    ];
    let mut out = name.to_string();
    for (from, to) in prefixes {
        if let Some(rest) = out.strip_prefix(from) {
            out = format!("{to}{rest}");
            break;
        }
    }
    // Field renames inside the blocks (suffix-level: they apply to both the
    // DiT and the Aggregator trees).
    for (from, to) in [
        (".t_embedder.time_mlp.0.", ".t_embedder.fc_in."),
        (".t_embedder.time_mlp.2.", ".t_embedder.fc_out."),
        (".c_embedder.cond_embedder.", ".c_embedder."),
        (".mlp.ff.0.0.", ".mlp.project_in."),
        (".mlp.ff.2.", ".mlp.project_out."),
        // Upstream `to_out` is a ModuleList([Linear, Dropout]) — the burn
        // side is a bare Linear.
        (".attn.to_out.0.", ".attn.to_out."),
        // The DiT/Aggregator norms are burn's bare `RmsNorm` (param path
        // `.gamma`, no `.inner.` wrapper like the MiniCPMRMSNorm one).
        (".norm1.weight", ".norm1.gamma"),
        (".norm2.weight", ".norm2.gamma"),
        (".norm_final.weight", ".norm_final.gamma"),
    ] {
        if let Some(base) = out.strip_suffix(from) {
            return format!("{base}{to}");
        }
        if let Some(at) = out.find(from) {
            let mut replaced = String::with_capacity(out.len());
            replaced.push_str(&out[..at]);
            replaced.push_str(to);
            replaced.push_str(&out[at + from.len()..]);
            return replaced;
        }
    }
    out
}

/// Load the Ming-3.0-tts `model.safetensors` into a burn module (f32).
pub fn load_ming3_checkpoint_into<B: Backend, M: ModuleSnapshot<B>>(
    module: &mut M,
    path: &Path,
) -> Result<LoadReport, String> {
    let snapshots = crate::tts::weights::read_snapshots(path)?;
    let transformed = transform_snapshots(snapshots, burn::tensor::DType::F32, &remap_ming_path)?;

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

// (No hand-built reports: `load_ming3_checkpoint_into` mirrors the shared
// loader's report construction.)

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tts::weights::fixture::pytorch_style_checkpoint;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use burn_store::ModuleSnapshot;

    type TestBackend = NdArray;

    /// Round-trip: build a tiny Ming3Model, serialize it in the upstream
    /// PyTorch layout (transposed linears, `weight` norms, upstream paths),
    /// load it back through [`load_ming3_checkpoint_into`], and require
    /// every module tensor to find its checkpoint source.
    #[test]
    fn remap_loads_every_ming3_tensor() {
        let device = NdArrayDevice::default();
        let config: crate::tts::ming3::config::Ming3Config = serde_json::from_value(
            serde_json::json!({
                "model_type": "dense",
                "llm_config": {
                    "hidden_size": 32, "intermediate_size": 64,
                    "max_position_embeddings": 512, "num_attention_heads": 4,
                    "num_hidden_layers": 2, "num_key_value_heads": 2,
                    "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
                    "vocab_size": 64, "qkv_bias": true
                },
                "audio_tokenizer_config": {
                    "enc_kwargs": { "latent_dim": 8, "input_dim": 16, "hop_size": 16,
                        "backbone": { "hidden_size": 32, "intermediate_size": 64,
                            "max_position_embeddings": 512, "num_attention_heads": 4,
                            "num_hidden_layers": 2, "num_key_value_heads": 2,
                            "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
                            "vocab_size": 1, "use_sliding_window": true,
                            "sliding_window": 8 } },
                    "dec_kwargs": { "latent_dim": 8, "output_dim": 16,
                        "backbone": { "hidden_size": 32, "intermediate_size": 64,
                            "max_position_embeddings": 512, "num_attention_heads": 4,
                            "num_hidden_layers": 2, "num_key_value_heads": 2,
                            "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
                            "vocab_size": 1, "use_sliding_window": true,
                            "sliding_window": 8 } },
                    "sample_rate": 44100, "patch_size": 4
                },
                "ditar_config": { "depth": 1, "hidden_size": 32, "num_heads": 4,
                    "mlp_ratio": 4, "patch_size": 1, "history_patch_size": 1 },
                "aggregator_config": { "depth": 1, "hidden_size": 32, "num_heads": 4,
                    "mlp_ratio": 4 }
            }),
        )
        .expect("tiny config");

        let mut model =
            crate::tts::ming3::model::Ming3Model::<TestBackend>::init(&config, &device);

        // Invert the burn tree into the upstream PyTorch layout.
        let burn_snapshots = model.collect(None, None, true);
        let mut tensors = Vec::new();
        for snapshot in burn_snapshots {
            let burn_name = snapshot.full_path();
            let data = snapshot.to_data().unwrap();
            // burn → upstream: undo the norm rename and the transposes; the
            // fixture helper does both for generic trees.
            let (name, data) = if burn_name.ends_with(".inner.gamma") {
                (burn_name.replace(".inner.gamma", ".weight"), data)
            } else if crate::tts::weights::is_transposed_linear_weight(
                &burn_name,
                data.shape.rank(),
            ) {
                (burn_name.clone(), crate::tts::weights::transpose_2d(&data).unwrap())
            } else {
                (burn_name.clone(), data)
            };
            // Undo the path remap: burn field names → upstream prefixes.
            let upstream = invert_ming_path(&name);
            tensors.push((upstream, data));
        }
        let bytes = crate::tts::weights::fixture::serialize_safetensors(tensors).unwrap();

        // Load it back into a fresh model through the real remap chain.
        let dir = std::env::temp_dir().join("neuron-ming3-fixture-test");
        std::fs::create_dir_all(&dir).unwrap();
        let ckpt = dir.join("model.safetensors");
        std::fs::write(&ckpt, &bytes).unwrap();

        let mut loaded =
            crate::tts::ming3::model::Ming3Model::<TestBackend>::init(&config, &device);
        let report = load_ming3_checkpoint_into(&mut loaded, &ckpt).expect("load");
        assert!(
            report.missing.is_empty(),
            "missing after remap: {missing:?}",
            missing = report.missing
        );
        assert!(report.errors.is_empty(), "errors: {:?}", report.errors);
        // The fixture is the inverted burn tree, so every checkpoint key
        // must land on a module field — nothing may be left unused.
        assert!(
            report.unused.is_empty(),
            "unexpected unused keys: {:?}",
            report.unused
        );
    }

    /// burn field name → upstream checkpoint name (inverse of
    /// [`remap_ming_path`]), used only by the fixture.
    fn invert_ming_path(burn_name: &str) -> String {
        let suffix_pairs = [
            (".mlp.project_in.", ".mlp.ff.0.0."),
            (".mlp.project_out.", ".mlp.ff.2."),
            (".t_embedder.fc_in.", ".t_embedder.time_mlp.0."),
            (".t_embedder.fc_out.", ".t_embedder.time_mlp.2."),
            (".c_embedder.", ".c_embedder.cond_embedder."),
            (".attn.to_out.", ".attn.to_out.0."),
            (".norm1.gamma", ".norm1.weight"),
            (".norm2.gamma", ".norm2.weight"),
            (".norm_final.gamma", ".norm_final.weight"),
        ];
        let mut out = burn_name.to_string();
        for (burn_suffix, upstream_suffix) in suffix_pairs {
            if let Some(at) = out.find(burn_suffix) {
                out = format!(
                    "{}{}{}",
                    &out[..at],
                    upstream_suffix,
                    &out[at + burn_suffix.len()..]
                );
                break;
            }
        }
        let prefixes = [
            ("backbone.", "model.model."),
            ("encoder.backbone.", "audio.encoder.encoder."),
            ("encoder.aggregator.", "audio.encoder.aggregator."),
            ("encoder.", "audio.encoder."),
            ("decoder.backbone.", "audio.decoder.decoder."),
            ("decoder.head.", "audio.decoder.head.istft."),
            ("decoder.", "audio.decoder."),
            ("flowloss.", "flowloss.cfm.model."),
        ];
        for (burn_prefix, upstream_prefix) in prefixes {
            if let Some(rest) = out.strip_prefix(burn_prefix) {
                return format!("{upstream_prefix}{rest}");
            }
        }
        out
    }

    #[test]
    fn remap_rules_match_the_upstream_tree() {
        assert_eq!(remap_ming_path("model.model.norm.weight"), "backbone.norm.weight");
        assert_eq!(
            remap_ming_path("audio.encoder.encoder.layers.3.self_attn.q_proj.bias"),
            "encoder.backbone.layers.3.self_attn.q_proj.bias"
        );
        assert_eq!(
            remap_ming_path("audio.encoder.fc1.weight"),
            "encoder.fc1.weight"
        );
        assert_eq!(
            remap_ming_path("audio.decoder.decoder.layers.0.mlp.up_proj.weight"),
            "decoder.backbone.layers.0.mlp.up_proj.weight"
        );
        assert_eq!(
            remap_ming_path("audio.decoder.head.istft.window"),
            "decoder.head.window"
        );
        assert_eq!(
            remap_ming_path("audio.decoder.head.out.bias"),
            "decoder.head.out.bias"
        );
        assert_eq!(
            remap_ming_path(
                "flowloss.cfm.model.t_embedder.time_mlp.0.weight"
            ),
            "flowloss.t_embedder.fc_in.weight"
        );
        assert_eq!(
            remap_ming_path(
                "flowloss.cfm.model.c_embedder.cond_embedder.weight"
            ),
            "flowloss.c_embedder.weight"
        );
        assert_eq!(
            remap_ming_path("flowloss.cfm.model.blocks.0.attn.to_q.bias"),
            "flowloss.blocks.0.attn.to_q.bias"
        );
        assert_eq!(
            remap_ming_path(
                "linear_proj_audio.blocks.2.mlp.ff.0.0.weight"
            ),
            "linear_proj_audio.blocks.2.mlp.project_in.weight"
        );
        assert_eq!(
            remap_ming_path("stop_head.weight"),
            "stop_head.weight"
        );
        assert_eq!(
            remap_ming_path("audio.encoder.norm.weight"),
            "encoder.norm.weight"
        );
    }
}
