//! Vendor tree for the Mage-VL port (`microsoft/Mage-VL`).
//!
//! Mage-VL is a codec-native streaming VLM (Mage-ViT encoder +
//! Qwen3-4B decoder). Neure consumes it in two stages:
//!
//! 1. **StreamMind gate** (`gate.rs`) — event-gated streaming: a
//!    lightweight Mamba+Qwen3 classifier scores each rolling codec
//!    window and emits `p_speak`; generation is only invoked when a
//!    response-worthy event completes. This stage is self-contained:
//!    its weights ship as `streammind_gate.safetensors` inside the HF
//!    checkpoint and it runs on top of *any* provider of Mage-ViT
//!    tokens.
//!
//! 2. **Vision encoder** (`vision.rs`) — Mage-ViT, the from-scratch
//!    codec ViT (24 layers, 3D RoPE over t/h/w on a 16×16 patch grid,
//!    I/P-frame patch sparsity). Its token stream feeds both the gate
//!    and the LLM projector. Landed incrementally; see module docs.
//!
//! Reference sources:
//! - GitHub: <https://github.com/microsoft/Mage> (`mage_vl/`)
//! - HF checkpoint code mirrored at
//!   `OpenSourceProjects/mage/hf-model-code/` for review parity.

pub mod config;
pub mod gate;
pub mod http;
pub mod model;
pub mod processor;
pub mod runtime;
pub(crate) mod text_tower;
pub mod vision;

pub use config::{MageVLConfig, MageVlVisionConfig, PlaceholderTokens, StreamMindGateConfig};
pub use gate::StreamMindGate;
pub use model::MageVLModel;
pub use processor::{placeholder_positions, scatter_merged, PlaceholderKind};
pub use runtime::MageVlRuntime;
