//! Real ONNX inference backend using tract (pure-Rust, no C++ dependencies).
//!
//! This module provides a concrete `OrtSessionRunner` implementation
//! backed by the `tract-onnx` crate. It loads ONNX models and runs
//! inference with `ndarray` tensors.
//!
//! ## Usage
//!
//! ```rust
//! use neure_lib::vision::ort_backends::TractSession;
//! use neure_lib::vision::ort_runtime::OrtSession;
//!
//! let session = TractSession::new("/path/to/model.onnx")?;
//! let ort_session = OrtSession::new(session);
//! let (output, shape) = ort_session.run("images", &input_tensor, &[1, 3, 640, 640], &["output0".into()])?;
//! ```
//!
//! ## Requirements
//!
//! - `tract-onnx` crate (auto-enabled with `--features tract`)
//! - `ndarray` crate (auto-enabled with `--features tract`)
//! - An ONNX model file (YOLOv8, RF-DETR, DETR, etc.)

use std::path::Path;

use crate::llm::NeureError;
use crate::vision::ort_runtime::OrtSessionRunner;

/// Concrete ONNX session backed by tract (pure-Rust).
/// Loads an ONNX model and runs inference with `ndarray` tensors.
pub struct TractSession {
    model: tract_onnx::prelude::TypedRunnableModel<
        tract_onnx::prelude::Graph<tract_onnx::prelude::TypedFact, Box<dyn tract_onnx::prelude::Op>>,
    >,
}

impl TractSession {
    /// Load an ONNX model from a file path.
    ///
    /// The model is optimized (float16 → float32, graph optimization)
    /// and ready for inference.
    pub fn new(onnx_path: &str) -> Result<Self, NeureError> {
        let path = Path::new(onnx_path);
        if !path.exists() {
            return Err(NeureError::invalid_input(format!(
                "ONNX model not found: {}",
                path.display()
            )));
        }

        let model = tract_onnx::onnx()
            .model_for_path(path)
            .map_err(|e| NeureError::invalid_input(format!("tract load: {e}")))?
            .into_typed()
            .map_err(|e| NeureError::invalid_input(format!("tract optimize: {e}")))?
            .into_optimized()
            .map_err(|e| NeureError::invalid_input(format!("tract finalize: {e}")))?
            .into_runnable()
            .map_err(|e| NeureError::invalid_input(format!("tract make runnable: {e}")))?;

        Ok(Self { model })
    }
}

impl OrtSessionRunner for TractSession {
    fn run_inference(
        &self,
        _input_name: &str,
        input: &[f32],
        input_shape: &[i64],
        _output_names: &[String],
    ) -> Result<(Vec<f32>, Vec<i64>), NeureError> {
        // Convert input to ndarray
        let shape: Vec<usize> = input_shape.iter().map(|d| *d as usize).collect();
        let input_array = ndarray::Array::from_shape_vec(shape, input.to_vec())
            .map_err(|e| NeureError::invalid_input(format!("tensor shape: {e}")))?;

        // Run inference
        let result = self
            .model
            .run(tract_onnx::prelude::tvec!(input_array.into()))
            .map_err(|e| NeureError::invalid_input(format!("tract run: {e}")))?;

        // Extract output tensor
        let output = result[0]
            .to_array_view::<f32>()
            .map_err(|e| NeureError::invalid_input(format!("tract extract: {e}")))?;

        let out_shape: Vec<i64> = output.shape().iter().map(|d| *d as i64).collect();
        let flat: Vec<f32> = output.iter().cloned().collect();

        Ok((flat, out_shape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tract_session_creation_fails_without_model() {
        let result = TractSession::new("/nonexistent/model.onnx");
        assert!(result.is_err());
    }
}
