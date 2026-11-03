//! Vendored EmbeddingGemma 2 (google/embeddinggemma-2) — the second
//! embedding family in neuron, and the first non-BERT one.
//!
//! Layout:
//! - [`config`] — serde mirror of the checkpoint's `text_config`
//!   plus the per-layer override resolution (layers 5/11/17/23 are
//!   `full_attention` with head_dim 512 / 1 KV head; the rest are
//!   `sliding_attention` with head_dim 256 / 2 KV heads).
//! - [`text_model`] — the candle port of the bidirectional text
//!   tower: per-layer embeddings (PLE) gating, dual RoPE, bare v-norm,
//!   and the 512→768 `embedding_projection` head.
//! - [`runtime`] — the [`EmbeddingRuntime`] impl: masked mean pool +
//!   L2 normalize + Matryoshka truncation via the OpenAI `dimensions`
//!   parameter.
//!
//! Upstream reference: transformers
//! `models/embedding_gemma2/modeling_embedding_gemma2.py` (the port
//! is element-wise faithful to it; numeric parity is pinned by the
//! `#[ignore]`d real-weight tests and `scripts/gen-egemma2-parity.py`).

#[cfg(feature = "candle")]
pub mod config;
#[cfg(feature = "candle")]
pub mod parity;
#[cfg(feature = "candle")]
pub mod runtime;
#[cfg(feature = "candle")]
pub mod text_model;
#[cfg(feature = "candle")]
pub mod weight_names;
