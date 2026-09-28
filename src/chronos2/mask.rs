//! Attention masks for the Chronos2 encoder/decoder.
//!
//! Chronos2 uses three types of masks:
//!
//! 1. **Causal mask**: prevents the decoder from attending to
//!    future positions. Upper-triangular with -inf above the
//!    diagonal, 0.0 on/below.
//!
//! 2. **Padding mask**: marks PAD tokens so they don't
//!    contribute to attention. Shape `[batch, seq_len]` where
//!    1 = real token, 0 = padding.
//!
//! 3. **Composite mask**: combines causal + padding masks for
//!    the decoder self-attention. The encoder uses only the
//!    padding mask (no causality constraint).

use candle::{Device, Result, Tensor};

/// Special token id for padding.
const PAD_TOKEN_ID: u32 = 2;

/// Build a causal mask for decoder self-attention.
///
/// Shape: `[1, 1, seq_len, seq_len]`
/// Values: 0.0 on/below diagonal, -inf above.
pub fn causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    let mut data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            data[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(data, (1, 1, seq_len, seq_len), device)
}

/// Build a padding mask from token ids.
///
/// Returns a mask tensor of shape `[batch, 1, 1, seq_len]`
/// where 0.0 attends and -inf masks out PAD tokens.
pub fn padding_mask(token_ids: &Tensor) -> Result<Tensor> {
    let (batch, seq_len) = token_ids.dims2()?;
    let device = token_ids.device();
    let data = token_ids.flatten_all()?.to_vec1::<u32>()?;

    let mut mask_data = Vec::with_capacity(batch * seq_len);
    for &id in &data {
        if id == PAD_TOKEN_ID {
            mask_data.push(f32::NEG_INFINITY);
        } else {
            mask_data.push(0.0f32);
        }
    }

    Tensor::from_vec(mask_data, (batch, 1, 1, seq_len), device)
}

/// Build a composite mask combining causal + padding masks.
///
/// For decoder self-attention: positions that are either
/// causally masked OR padding-masked get -inf.
pub fn composite_mask(token_ids: &Tensor) -> Result<Tensor> {
    let (batch, seq_len) = token_ids.dims2()?;
    let device = token_ids.device();
    let data = token_ids.flatten_all()?.to_vec1::<u32>()?;

    let mut mask_data = Vec::with_capacity(batch * seq_len * seq_len);
    for b in 0..batch {
        for i in 0..seq_len {
            for j in 0..seq_len {
                let is_pad = data[b * seq_len + j] == PAD_TOKEN_ID;
                let is_future = j > i;
                if is_pad || is_future {
                    mask_data.push(f32::NEG_INFINITY);
                } else {
                    mask_data.push(0.0f32);
                }
            }
        }
    }

    Tensor::from_vec(mask_data, (batch, 1, seq_len, seq_len), device)
}

/// Build an encoder padding mask (no causality).
///
/// Shape: `[batch, 1, 1, seq_len]`
pub fn encoder_mask(token_ids: &Tensor) -> Result<Tensor> {
    padding_mask(token_ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_mask_shape() {
        let dev = Device::Cpu;
        let m = causal_mask(4, &dev).unwrap();
        assert_eq!(m.dims4().unwrap(), (1, 1, 4, 4));
    }

    #[test]
    fn causal_mask_values() {
        let dev = Device::Cpu;
        let m = causal_mask(3, &dev).unwrap();
        let data: Vec<f32> = m.flatten_all().unwrap().to_vec1().unwrap();

        // Row 0: [0, -inf, -inf]
        assert_eq!(data[0], 0.0);
        assert_eq!(data[1], f32::NEG_INFINITY);
        assert_eq!(data[2], f32::NEG_INFINITY);

        // Row 1: [0, 0, -inf]
        assert_eq!(data[3], 0.0);
        assert_eq!(data[4], 0.0);
        assert_eq!(data[5], f32::NEG_INFINITY);

        // Row 2: [0, 0, 0]
        assert_eq!(data[6], 0.0);
        assert_eq!(data[7], 0.0);
        assert_eq!(data[8], 0.0);
    }

    #[test]
    fn padding_mask_no_padding() {
        let dev = Device::Cpu;
        let ids = Tensor::from_vec(vec![0u32, 1, 3, 4], (1, 4), &dev).unwrap();
        let m = padding_mask(&ids).unwrap();
        assert_eq!(m.dims4().unwrap(), (1, 1, 1, 4));
        let data: Vec<f32> = m.flatten_all().unwrap().to_vec1().unwrap();
        assert!(data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn padding_mask_with_padding() {
        let dev = Device::Cpu;
        let ids = Tensor::from_vec(vec![0u32, 3, 2, 2], (1, 4), &dev).unwrap();
        let m = padding_mask(&ids).unwrap();
        let data: Vec<f32> = m.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(data[0], 0.0); // BOS
        assert_eq!(data[1], 0.0); // real token
        assert_eq!(data[2], f32::NEG_INFINITY); // PAD
        assert_eq!(data[3], f32::NEG_INFINITY); // PAD
    }

    #[test]
    fn padding_mask_batch() {
        let dev = Device::Cpu;
        // Batch of 2 sequences.
        let ids = Tensor::from_vec(
            vec![0u32, 3, 2, 0, 2, 3],
            (2, 3),
            &dev,
        )
        .unwrap();
        let m = padding_mask(&ids).unwrap();
        assert_eq!(m.dims4().unwrap(), (2, 1, 1, 3));
    }

    #[test]
    fn composite_mask_shape() {
        let dev = Device::Cpu;
        let ids = Tensor::from_vec(vec![0u32, 3, 4, 2], (1, 4), &dev).unwrap();
        let m = composite_mask(&ids).unwrap();
        assert_eq!(m.dims4().unwrap(), (1, 1, 4, 4));
    }

    #[test]
    fn composite_mask_causal_and_padding() {
        let dev = Device::Cpu;
        // Sequence: BOS, token, PAD, token
        let ids = Tensor::from_vec(vec![0u32, 3, 2, 4], (1, 4), &dev).unwrap();
        let m = composite_mask(&ids).unwrap();
        let data: Vec<f32> = m.flatten_all().unwrap().to_vec1().unwrap();

        // Row 0 (pos 0): can attend to pos 0 only (pos 1,2,3 are future)
        assert_eq!(data[0], 0.0); // pos 0
        assert_eq!(data[1], f32::NEG_INFINITY); // pos 1 (future)
        assert_eq!(data[2], f32::NEG_INFINITY); // pos 2 (future + PAD)
        assert_eq!(data[3], f32::NEG_INFINITY); // pos 3 (future)

        // Row 1 (pos 1): can attend to pos 0,1 (pos 2,3 are future)
        assert_eq!(data[4], 0.0); // pos 0
        assert_eq!(data[5], 0.0); // pos 1
        assert_eq!(data[6], f32::NEG_INFINITY); // pos 2 (future + PAD)
        assert_eq!(data[7], f32::NEG_INFINITY); // pos 3 (future)

        // Row 2 (pos 2, PAD): can attend to pos 0,1 (pos 3 is future)
        assert_eq!(data[8], 0.0); // pos 0
        assert_eq!(data[9], 0.0); // pos 1
        assert_eq!(data[10], f32::NEG_INFINITY); // pos 2 (PAD)
        assert_eq!(data[11], f32::NEG_INFINITY); // pos 3 (future)

        // Row 3 (pos 3): can attend to pos 0,1,3 (pos 2 is PAD)
        assert_eq!(data[12], 0.0); // pos 0
        assert_eq!(data[13], 0.0); // pos 1
        assert_eq!(data[14], f32::NEG_INFINITY); // pos 2 (PAD)
        assert_eq!(data[15], 0.0); // pos 3
    }

    #[test]
    fn composite_mask_all_padding() {
        let dev = Device::Cpu;
        let ids = Tensor::from_vec(vec![2u32, 2, 2], (1, 3), &dev).unwrap();
        let m = composite_mask(&ids).unwrap();
        let data: Vec<f32> = m.flatten_all().unwrap().to_vec1().unwrap();
        // All positions are PAD - all should be masked.
        assert!(data.iter().all(|&v| v == f32::NEG_INFINITY));
    }

    #[test]
    fn composite_mask_no_padding() {
        let dev = Device::Cpu;
        let ids = Tensor::from_vec(vec![0u32, 3, 4, 5], (1, 4), &dev).unwrap();
        let m = composite_mask(&ids).unwrap();
        let data: Vec<f32> = m.flatten_all().unwrap().to_vec1().unwrap();

        // Should be same as causal mask.
        // Row 0: [0, -inf, -inf, -inf]
        assert_eq!(data[0], 0.0);
        assert_eq!(data[1], f32::NEG_INFINITY);
        assert_eq!(data[2], f32::NEG_INFINITY);
        assert_eq!(data[3], f32::NEG_INFINITY);
    }

    #[test]
    fn encoder_mask_same_as_padding_mask() {
        let dev = Device::Cpu;
        let ids = Tensor::from_vec(vec![0u32, 3, 2, 4], (1, 4), &dev).unwrap();
        let enc = encoder_mask(&ids).unwrap();
        let pad = padding_mask(&ids).unwrap();
        let enc_data: Vec<f32> = enc.flatten_all().unwrap().to_vec1().unwrap();
        let pad_data: Vec<f32> = pad.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(enc_data, pad_data);
    }
}
