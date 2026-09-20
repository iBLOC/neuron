//! Real ONNX inference backend using tract (pure-Rust, no C++ dependencies).
//!
//! Loads ONNX models and runs inference with `ndarray` tensors.
//! Activated via the `tract` Cargo feature.

use std::path::Path;
use std::sync::Arc;

use tract_onnx::prelude::*;

use crate::llm::NeuronError;
use crate::vision::ort_runtime::OrtSessionRunner;

pub type RunnableModelHandle = Arc<TypedRunnableModel<TypedModel>>;

pub struct TractSession {
    model: RunnableModelHandle,
}

impl TractSession {
    pub fn new(onnx_path: &str) -> Result<Self, NeuronError> {
        let path = Path::new(onnx_path);
        if !path.exists() {
            return Err(NeuronError::invalid_input(format!(
                "ONNX model not found: {}",
                path.display()
            )));
        }

        let model = tract_onnx::onnx()
            .model_for_path(path)
            .map_err(|e| NeuronError::invalid_input(format!("tract load: {e}")))?
            .into_optimized()
            .map_err(|e| NeuronError::invalid_input(format!("tract optimize: {e}")))?
            .into_runnable()
            .map_err(|e| NeuronError::invalid_input(format!("tract runnable: {e}")))?;

        Ok(Self { model: Arc::new(model) })
    }
}

impl OrtSessionRunner for TractSession {
    fn run_inference(
        &self,
        _input_name: &str,
        input: &[f32],
        input_shape: &[i64],
        _output_names: &[String],
    ) -> Result<(Vec<f32>, Vec<i64>), NeuronError> {
        let shape: Vec<usize> = input_shape.iter().map(|d| *d as usize).collect();
        let input_array = ndarray::Array::from_shape_vec(shape, input.to_vec())
            .map_err(|e| NeuronError::invalid_input(format!("tensor shape: {e}")))?;
        let input_tensor: Tensor = input_array.into();

        let result = self
            .model
            .run(tvec![input_tensor.into_tvalue()])
            .map_err(|e| NeuronError::invalid_input(format!("tract run: {e}")))?;

        let output = result[0]
            .to_array_view::<f32>()
            .map_err(|e| NeuronError::invalid_input(format!("tract extract: {e}")))?;

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