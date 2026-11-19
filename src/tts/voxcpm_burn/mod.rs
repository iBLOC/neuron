//! Vendored burn-based VoxCpm TTS model.
//! Source: https://github.com/madushan1000/voxcpm_rs (in-house; Apache-2.0)
//!
//! See voxcpm_model.rs for the main VoxCpm model, minicpm4.rs for the text encoder,
//! and audiovae.rs for the audio decoder.

pub mod audiovae;
pub mod audiovae_v2;
pub mod audiovae_v2_stream;
pub mod compat;
// minicpm4.rs / weights.rs live at the tts level (they are shared with the
// Ming-3.0-tts model, which does not gate on the voxcpm feature);
// re-exported here so the `super::minicpm4` / `super::weights` paths inside
// this directory keep resolving.
pub use crate::tts::{minicpm4, weights};
pub mod voxcpm2_model;
pub mod voxcpm_model;
