//! Token + positional embeddings for the T5-style encoder /
//! decoder. Two separate structs live here because the encoder
//! uses a learned positional embedding (T5 convention) while
//! Chronos2's decoder uses sinusoidal. We model both as plain
//! Rust types so the architecture-completion commit can choose
//! which to construct per-side.

use candle::{DType, Tensor};

#[derive(Debug, Clone)]
pub struct T5EmbeddingsConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    /// Maximum sequence length the positional embedding table
    /// supports. Must be >= every input the model sees; T5
    /// small uses 512, Chronos2 uses 1024.
    pub max_seq_len: usize,
    /// Whether the embedding is learned (true) or sinusoidal
    /// (false). T5 uses learned; the Chronos2 decoder sticks
    /// with sinusoidal to support variable horizon without
    /// retraining the embedding table.
    pub learned: bool,
}

#[derive(Debug, Clone)]
pub struct T5Embeddings {
    pub config: T5EmbeddingsConfig,
    /// Token embedding table (`vocab_size × d_model`). Always
    /// learned; for the sinusoidal side this is the only
    /// learned component.
    pub token: Tensor,
    /// Positional embedding table (`max_seq_len × d_model`) for
    /// the learned side, or a sinusoidal helper for the
    /// sinusoidal side. The forward pass takes the right slice.
    pub position: Tensor,
}

/// Output of the embedding forward pass: the token-id
/// embedding (looked up) added to the positional embedding.
/// Shape: `[batch, seq_len, d_model]`.
#[derive(Debug, Clone)]
pub struct T5EmbeddingsOutput {
    pub embeddings: Tensor,
}

impl T5Embeddings {
    /// Build the sinusoidal position table for the
    /// non-learned side. T5 / Transformer convention:
    ///
    ///   PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    ///   PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    ///
    /// We allocate once at construction; the forward pass
    /// slices the prefix `seq_len` columns.
    pub fn sinusoidal_table(d_model: usize, max_seq_len: usize) -> Tensor {
        // Allocate as a CPU f32 tensor; the runtime is free to
        // .to_device(...) when moving to GPU. The vendor layer
        // stays device-agnostic on purpose.
        let mut data = vec![0f32; max_seq_len * d_model];
        for pos in 0..max_seq_len {
            for i in 0..(d_model / 2) {
                let freq = 1f32 / 10000f32.powf((2 * i) as f32 / d_model as f32);
                let angle = pos as f32 * freq;
                data[pos * d_model + 2 * i] = angle.sin();
                data[pos * d_model + 2 * i + 1] = angle.cos();
            }
        }
        Tensor::from_vec(data, (max_seq_len, d_model), &candle::Device::Cpu)
            .expect("sinusoidal_table: from_vec should not fail for valid shape")
    }

    /// Look up token embeddings for `[batch, seq_len]`, slice
    /// the position table to the same `seq_len`, and add them.
    /// Output shape: `[batch, seq_len, d_model]`.
    pub fn embed(&self, token_ids: &Tensor) -> Tensor {
        let (_batch, seq_len) = token_ids.dims2().expect("token_ids must be 2D");
        // index_select takes a flat 1D index set; flatten
        // token_ids ([batch, seq_len]) to [batch*seq_len] and
        // reshape the result back to [batch, seq_len, d_model].
        let flat = token_ids.flatten_all().expect("flatten token_ids");
        let flat_emb = self.token.index_select(&flat, 0).expect("index_select");
        let token = flat_emb
            .reshape((_batch, seq_len, self.config.d_model))
            .expect("reshape token emb");
        let pos = self.position.narrow(0, 0, seq_len).expect("narrow pos");
        let token_f32 = token.to_dtype(DType::F32).expect("token to f32");
        let pos_f32 = pos.to_dtype(DType::F32).expect("pos to f32");
        token_f32.broadcast_add(&pos_f32).expect("broadcast_add")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sinusoidal_table_shape_is_max_seq_len_by_d_model() {
        let t = T5Embeddings::sinusoidal_table(8, 16);
        assert_eq!(t.dims2().unwrap(), (16, 8));
    }

    #[test]
    fn sinusoidal_table_first_row_differs_from_second() {
        let t = T5Embeddings::sinusoidal_table(4, 4);
        // Different positions produce different sin/cos values.
        let data: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
        assert_ne!(&data[0..4], &data[4..8]);
    }
}
