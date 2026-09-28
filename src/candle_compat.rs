//! Compatibility layer between neuron and the pinned `candle` revision.
//!
//! One wrapper lives here: a [`candle_nn::var_builder::SimpleBackend`]
//! implementation that tolerates the `layer_norm` bias probe introduced
//! upstream in candle `b96522e` (Sep 2026, "BERT safetensors weight/bias
//! gamma/beta compatibility"). That change probes `bias`/`beta` via
//! `contains_tensor` *before* consulting `config.affine`, so every
//! bias-free norm (RmsNorm — i.e. llama/qwen/gemma checkpoints) fails to
//! load with "Failed to find weight tensor". The shim reports a
//! synthetic `bias` for normalization paths only; for `affine = false`
//! the probe result is never loaded, and for `affine = true` checkpoints
//! the real bias exists so the inner backend answers first. Remove this
//! module once upstream gates the probe on `affine`.
//!
//! Every `VarBuilder::from_mmaped_safetensors` / `from_tensors` call site
//! in neuron goes through the constructors here so a single candle
//! revision serves all runtimes (LLM, embedding, rerank, ASR, vision,
//! chronos2, mage_vl).

use candle::DType;
use candle_nn::var_builder::SimpleBackend;
use candle_nn::{Init, VarBuilder};
use std::collections::HashMap;

/// Backend wrapper: forwards to any inner backend, synthesizing zero
/// biases for normalization paths when (and only when) the checkpoint
/// does not carry one.
struct NormBiasShim<B: SimpleBackend> {
    inner: B,
}

/// True for `…norm.bias` / `…norm.beta` probes on paths that look like a
/// normalization weight. Deliberately narrow: a genuinely missing Linear
/// bias must still surface as an error, not silently become zeros.
fn is_norm_bias_probe(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    (lower.ends_with(".bias") || lower.ends_with(".beta")) && lower.contains("norm")
}

impl<B: SimpleBackend> SimpleBackend for NormBiasShim<B> {
    fn get(
        &self,
        s: candle::Shape,
        name: &str,
        h: Init,
        dtype: DType,
        dev: &candle::Device,
    ) -> candle::Result<candle::Tensor> {
        match self.inner.get(s.clone(), name, h, dtype, dev) {
            Ok(t) => Ok(t),
            Err(e) => {
                if is_norm_bias_probe(name) {
                    // `affine = false` norms never reach this get; an
                    // affine norm with a missing checkpoint bias gets a
                    // zero bias, which is what its `Init::Const(0.)`
                    // default would have been anyway.
                    Ok(candle::Tensor::zeros(s, dtype, dev)?)
                } else {
                    Err(e)
                }
            }
        }
    }

    fn get_unchecked(
        &self,
        name: &str,
        dtype: DType,
        dev: &candle::Device,
    ) -> candle::Result<candle::Tensor> {
        self.inner.get_unchecked(name, dtype, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.inner.contains_tensor(name) || is_norm_bias_probe(name)
    }
}

/// Drop-in replacement for `VarBuilder::from_mmaped_safetensors` with the
/// norm-bias shim applied. Unsafe for the same reason as upstream: the
/// safety contract of memory-mapped file access.
pub unsafe fn from_mmaped_safetensors<P: AsRef<std::path::Path>>(
    paths: &[P],
    dtype: DType,
    dev: &candle::Device,
) -> candle::Result<VarBuilder<'static>> {
    // `MmapedSafetensors::multi` is unsafe (mmap); edition 2024 wants the
    // unsafe op spelled out even inside an unsafe fn.
    let tensors = unsafe { candle::safetensors::MmapedSafetensors::multi(paths)? };
    Ok(VarBuilder::from_backend(
        Box::new(NormBiasShim { inner: tensors }),
        dtype,
        dev.clone(),
    ))
}

/// Drop-in replacement for `VarBuilder::from_tensors` with the norm-bias
/// shim applied.
pub fn from_tensors(
    ts: HashMap<String, candle::Tensor>,
    dtype: DType,
    dev: &candle::Device,
) -> VarBuilder<'static> {
    VarBuilder::from_backend(
        Box::new(NormBiasShim { inner: ts }),
        dtype,
        dev.clone(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Tensor;

    /// Identify `layer.norm.bias` (or `.beta`) on the name that an
    /// upstream probe would surface as `contains_tensor`. False on
    /// `linear.bias` so a real Linear weight load still errors when the
    /// checkpoint is missing it.
    fn probe_is_norm_bias() {
        assert!(is_norm_bias_probe("model.layers.0.input_layernorm.bias"));
        assert!(is_norm_bias_probe("q_norm.bias"));
        assert!(is_norm_bias_probe("q_norm.beta"));
        assert!(!is_norm_bias_probe("model.layers.0.mlp.gate_proj.bias"));
    }

    fn mk(shape: candle::Shape, dev: &candle::Device) -> candle::Tensor {
        Tensor::zeros(shape, DType::F32, dev).unwrap()
    }

    fn shape(n: usize) -> candle::Shape {
        candle::Shape::from(vec![n])
    }

    #[test]
    fn contains_tensor_returns_true_for_norm_bias_even_when_missing() {
        // Inner backend has no `q_norm.bias` — the shim must still
        // claim it exists so the upstream `layer_norm` probe accepts
        // the path. Without this shim the RmsNorm line fails.
        let dev = candle::Device::Cpu;
        let vb = from_tensors(HashMap::new(), DType::F32, &dev);
        assert!(vb.contains_tensor("q_norm.bias"));
        assert!(vb.contains_tensor("q_norm.beta"));
        assert!(!vb.contains_tensor("q_norm.weight")); // not a probe — must be absent
    }

    #[test]
    fn contains_tensor_passes_through_for_real_weights() {
        // The weight comes from the inner backend; the shim must not
        // shadow it. The bias path is norm-shaped, so the shim *does*
        // claim it even though the inner backend knows nothing about it
        // — that is the entire point of the workaround.
        let dev = candle::Device::Cpu;
        let mut ts = HashMap::new();
        ts.insert("model.layers.0.input_layernorm.weight".into(), mk(shape(64), &dev));
        let vb = from_tensors(ts, DType::F32, &dev);
        assert!(vb.contains_tensor("model.layers.0.input_layernorm.weight"));
        // A *Linear*.bias is NOT a norm-bias probe; if the inner
        // backend does not have it, the shim does not lie about it —
        // `get` must still fail downstream.
        assert!(!vb.contains_tensor("model.layers.0.mlp.gate_proj.bias"));
    }

    #[test]
    fn get_synthesizes_zero_bias_only_for_norm_paths() {
        let dev = candle::Device::Cpu;
        // `q_norm.weight` is present; `q_norm.bias` is not (RmsNorm
        // affine=false model). Probe the bias — must yield zeros.
        let mut ts = HashMap::new();
        ts.insert("q_norm.weight".into(), mk(shape(64), &dev));
        let vb = from_tensors(ts, DType::F32, &dev);
        let bias = vb
            .get_with_hints((64,), "q_norm.bias", Init::Const(0.0))
            .expect("shim must synthesize the bias");
        assert_eq!(bias.dims(), &[64]);
        // All zeros — exactly what `Init::Const(0.)` would have produced.
        let flat = bias.to_vec1::<f32>().unwrap();
        assert!(flat.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn get_propagates_real_errors_outside_norm_bias() {
        let dev = candle::Device::Cpu;
        let ts = HashMap::new(); // empty
        let vb = from_tensors(ts, DType::F32, &dev);
        // `q.weight` (real missing weight) must error — the shim only
        // forgives the *bias* probe.
        let r = vb.get_with_hints((64,), "q.weight", Init::Const(1.0));
        assert!(r.is_err());
    }
}
