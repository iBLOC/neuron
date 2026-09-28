//! Vendored burn-based VoxCpm TTS model.
//! Source: https://github.com/madushan1000/voxcpm_rs (in-house; Apache-2.0)
//!
//! See voxcpm_model.rs for the main VoxCpm model, minicpm4.rs for the text encoder,
//! and audiovae.rs for the audio decoder.

pub mod audiovae;
pub mod audiovae_v2;
pub mod audiovae_v2_stream;
pub mod compat;
pub mod minicpm4;
pub mod voxcpm2_model;
pub mod voxcpm_model;
pub mod weights;
