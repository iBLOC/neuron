//! Library-side support for `benches/` runners.
//!
//! See [`model_matrix`] for the curated (capability × engine × device)
//! matrix the cross-runtime harness in `benches/model_matrix.rs` walks.
//! Other benches may add their own modules here later.

pub mod model_matrix;
