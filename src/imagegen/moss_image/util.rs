//! Small numeric helpers shared by the MOSS-Image modules: a
//! numerically safe `tanh` (candle has none), query-chunked attention
//! that bounds the score-matrix footprint, and deterministic seeded
//! normal noise.

use candle::{DType, Result, Tensor};

/// Query-axis block for the chunked attention. 512 × 24 heads ×
/// ~8k keys in fp32 ≈ 400 MB peak — big enough for throughput,
/// small enough to survive next to ~27 GB of resident weights.
const ATTN_QUERY_BLOCK: usize = 512;

/// `tanh` on an fp32 tensor, computed through the stable
/// `(e^{2x}−1)/(e^{2x}+1)` form. Inputs are clamped to ±40 first:
/// `tanh(40)` already rounds to 1.0 in fp32 while `e^80` stays
/// representable, so the clamp changes no representable result.
pub fn tanh_f32(x: &Tensor) -> Result<Tensor> {
    let x = x.to_dtype(DType::F32)?;
    let x = x.clamp(-40.0, 40.0)?;
    let e = x.affine(2.0, 0.0)?.exp()?;
    (&e - 1.0)? / (&e + 1.0)?
}

/// Scaled dot-product attention over `[B, H, T, hd]` tensors with the
/// softmax computed in fp32, chunked along the query axis so the
/// score matrix never exceeds `[B, H, BLOCK, T]`.
///
/// `mask` is an optional additive `[T, T]` mask in fp32 (causal
/// masks pass `-1e9` above the diagonal); it is sliced per query
/// block. Softmax is computed in fp32 and the result cast back to
/// `q`'s dtype, mirroring PyTorch SDPA's fp32 accumulation.
pub fn attention_chunked(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let t = q.dim(2)?;
    let mut outs = Vec::new();
    let mut start = 0;
    while start < t {
        let len = ATTN_QUERY_BLOCK.min(t - start);
        let qb = q.narrow(2, start, len)?;
        let scores = qb.matmul(&k.transpose(2, 3)?)?.affine(scale, 0.0)?;
        let scores = scores.to_dtype(DType::F32)?;
        let scores = match mask {
            Some(mask) => {
                let m = mask.narrow(0, start, len)?;
                scores.broadcast_add(&m)?
            }
            None => scores,
        };
        let probs = candle_nn::ops::softmax_last_dim(&scores)?.to_dtype(q.dtype())?;
        outs.push(probs.matmul(v)?);
        start += len;
    }
    Tensor::cat(&outs, 2)
}

/// Build an additive causal mask `[T, T]` fp32: `0` on and below the
/// diagonal, `-1e9` above (matches the standard `is_causal` SDPA mask
/// once fed through softmax in fp32).
pub fn causal_mask(t: usize, device: &candle::Device) -> Result<Tensor> {
    let mut mask = vec![0.0f32; t * t];
    for i in 0..t {
        for j in (i + 1)..t {
            mask[i * t + j] = -1e9;
        }
    }
    Tensor::from_vec(mask, (t, t), device)
}

/// Deterministic standard-normal samples from a 64-bit seed
/// (xorshift64* generator + Box-Muller). Generated on CPU and moved
/// with `to_device`, so the same seed produces the same noise on
/// every backend — including CUDA, whose native RNG is not seeded the
/// same way.
pub fn seeded_randn(
    n: usize,
    seed: u64,
    dtype: DType,
    device: &candle::Device,
) -> Result<Tensor> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(0x1234_5678_9ABC_DEF0) | 1;
    let mut next_u64 = move || {
        let mut x = state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    let mut normals: Vec<f32> = Vec::with_capacity(n);
    while normals.len() < n {
        let u1 = (next_u64() >> 11) as f64 / (1u64 << 53) as f64 + f64::EPSILON;
        let u2 = (next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        normals.push((r * theta.cos()) as f32);
        if normals.len() < n {
            normals.push((r * theta.sin()) as f32);
        }
    }
    let cpu = Tensor::from_vec(normals, (n,), &candle::Device::Cpu)?;
    let shaped = cpu.to_dtype(dtype)?;
    match device.location() {
        candle::DeviceLocation::Cpu => Ok(shaped),
        _ => shaped.to_device(device),
    }
}

/// Sample a fresh random seed when a request did not provide one.
pub fn fresh_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x5EED_0000_0000_0001)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        | 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tanh_matches_expected_values() {
        let dev = candle::Device::Cpu;
        let x = Tensor::from_vec(vec![-10.0f32, -1.0, 0.0, 1.0, 10.0], (5,), &dev).unwrap();
        let out = tanh_f32(&x).unwrap().to_vec1::<f32>().unwrap();
        for (got, want) in out.iter().zip([-1.0f32, -0.7615942, 0.0, 0.7615942, 1.0]) {
            assert!((got - want).abs() < 1e-5, "got {got} want {want}");
        }
    }

    #[test]
    fn test_tanh_saturates_instead_of_nan() {
        let dev = candle::Device::Cpu;
        let x = Tensor::from_vec(vec![-1000.0f32, 1000.0], (2,), &dev).unwrap();
        let out = tanh_f32(&x).unwrap().to_vec1::<f32>().unwrap();
        assert!(out[0].is_finite() && out[0] <= -0.999);
        assert!(out[1].is_finite() && out[1] >= 0.999);
    }

    #[test]
    fn test_attention_chunked_uniform_keys_gives_uniform_mix() {
        // With all-equal scores, attention output must equal the mean
        // of the values regardless of chunking.
        let dev = candle::Device::Cpu;
        let t = 1300; // forces 3 chunks (512 + 512 + 276)
        let q = Tensor::zeros((1, 2, t, 16), DType::F32, &dev).unwrap();
        let k = Tensor::zeros((1, 2, t, 16), DType::F32, &dev).unwrap();
        let vals: Vec<f32> = (0..2 * t * 16).map(|i| i as f32 * 0.01).collect();
        let v = Tensor::from_vec(vals, (1, 2, t, 16), &dev).unwrap();
        let out = attention_chunked(&q, &k, &v, 0.25, None).unwrap();
        assert_eq!(out.dims(), &[1, 2, t, 16]);
        // Every query attends uniformly → each output is the mean of
        // that head's value rows. v[h, tt, j] = (h·t·16 + tt·16 + j)·0.01,
        // so the expected mean is (h·t·16 + (t−1)/2·16 + j)·0.01.
        let out = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for h in 0..2 {
            let mut mean = vec![0.0f32; 16];
            for tt in 0..t {
                for (j, m) in mean.iter_mut().enumerate() {
                    *m += out[(h * t + tt) * 16 + j];
                }
            }
            for (j, m) in mean.iter().enumerate() {
                let expected =
                    (h as f32 * t as f32 * 16.0 + (t as f32 - 1.0) / 2.0 * 16.0 + j as f32)
                        * 0.01;
                assert!(
                    (m / t as f32 - expected).abs() < 0.02,
                    "head {h} ch {j}: {} vs {expected}",
                    m / t as f32
                );
            }
        }
    }

    #[test]
    fn test_causal_mask_shape_and_values() {
        let m = causal_mask(4, &candle::Device::Cpu).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(m.len(), 4);
        for (i, row) in m.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                if j > i {
                    assert!(v <= -1e8, "row {i} col {j} = {v}");
                } else {
                    assert_eq!(v, 0.0);
                }
            }
        }
    }

    #[test]
    fn test_seeded_randn_deterministic_and_reasonable() {
        let dev = candle::Device::Cpu;
        let a = seeded_randn(4096, 42, DType::F32, &dev).unwrap();
        let b = seeded_randn(4096, 42, DType::F32, &dev).unwrap();
        assert_eq!(
            a.to_vec1::<f32>().unwrap(),
            b.to_vec1::<f32>().unwrap(),
            "same seed must produce identical noise"
        );
        let c = seeded_randn(4096, 43, DType::F32, &dev).unwrap();
        assert_ne!(a.to_vec1::<f32>().unwrap(), c.to_vec1::<f32>().unwrap());
        let vals = a.to_vec1::<f32>().unwrap();
        let mean = vals.iter().sum::<f32>() / vals.len() as f32;
        let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / vals.len() as f32;
        assert!(mean.abs() < 0.05, "mean {mean}");
        assert!((var - 1.0).abs() < 0.1, "var {var}");
    }
}
