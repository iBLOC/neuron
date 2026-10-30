#[cfg(feature = "flash-attn")]
pub mod qwen2;
#[cfg(feature = "flash-attn")]
pub mod qwen3;

/// Vendored llama with explicit `head_dim` support (needed by wide-
/// attention llama-architecture checkpoints such as MiniCPM5-1B; see the
/// module docs). Upstream-compatible when `head_dim` is absent, so every
/// checkpoint that loaded before keeps loading — neuron routes ALL llama
/// dispatches through this module.
pub mod llama_hd;
