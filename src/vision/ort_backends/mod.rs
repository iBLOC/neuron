//! Reference plug-in backends for the ONNX vision runtime.
//!
//! These modules contain **drop-in implementations** of the
//! `build_session` + `run_session` contract documented in
//! `src/vision/ort_runtime.rs`. They live here as reference code —
//! users copy them into their own crate (along with their chosen
//! ONNX executor dep) to activate real ONNX inference.
//!
//! ## Why these don't depend on `ort` or `tract` directly
//!
//! neuron itself stays executor-agnostic. Adding a hard dep on `ort`
//! upstream is blocked by [ort 2.0.0-rc.12's vitis.rs compile bug];
//! adding `tract` as a dep would force every neuron user to pay for it.
//!
//! Instead, this directory documents the plug-in contract and provides
//! ready-to-copy code. Users add the executor to *their* crate and
//! wire `build_session` + `run_session` to it.

pub mod tract_backend;
pub use tract_backend::TractSession;