#[cfg(feature = "flash-attn")]
pub mod qwen2;
// Qwen3 has NO vendored module: the pinned candle-transformers (rev
// 66a8cf1) already ships a FlashAttention path for qwen3 — the fused CPU
// kernel without the `flash-attn` feature, the CUDA FA2 kernel with it.
// The vendored qwen3 was a leftover of the older 0.9.2 pin: it duplicated
// what upstream does now, and it never compiled against the pinned
// candle-flash-attn API.

/// Vendored llama with explicit `head_dim` support (needed by wide-
/// attention llama-architecture checkpoints such as MiniCPM5-1B; see the
/// module docs). Upstream-compatible when `head_dim` is absent, so every
/// checkpoint that loaded before keeps loading — neuron routes ALL llama
/// dispatches through this module.
pub mod llama_hd;
