//! onnxruntime-backed session for the PP-OCR graphs, via the `ort` crate.
//!
//! Unlike tract — which bakes one compiled plan per concrete input shape and
//! needs the [`super::paddle_runtime::TractPlanCache`] — onnxruntime runs
//! dynamic shapes natively: one `Session` per graph serves every (h, w) the
//! export declares. That removes the per-shape compile cost entirely and is
//! why `ort` is the designated fast executor (parity + speed verified
//! against onnxruntime 1.28 / ort 2.0.0-rc.13 on 2026-10-09).

use std::path::Path;
use std::sync::Mutex;

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

use crate::llm::NeuronError;

use super::paddle_runtime::OcrSession;

pub struct OrtGraphSession {
    session: Mutex<Session>,
}

impl OrtGraphSession {
    pub fn open(model_path: &Path) -> Result<Self, NeuronError> {
        let session = Session::builder()
            .map_err(|e| NeuronError::invalid_input(format!("ort builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| NeuronError::invalid_input(format!("ort optimization level: {e}")))?
            .commit_from_file(model_path)
            .map_err(|e| {
                NeuronError::invalid_input(format!(
                    "ort load {} (is it prepped with scripts/prep-ppocr-onnx.py?): {e}",
                    model_path.display()
                ))
            })?;
        Ok(Self { session: Mutex::new(session) })
    }
}

impl OcrSession for OrtGraphSession {
    fn run(
        &self,
        chw: Vec<f32>,
        h: u32,
        w: u32,
    ) -> Result<(Vec<f32>, Vec<usize>), NeuronError> {
        // `run` takes &mut; the mutex serializes concurrent requests onto the
        // one session. onnxruntime is thread-safe internally, but the crate
        // models the borrow this way — a per-device session pool is the
        // follow-up if single-session throughput ever becomes the bottleneck.
        let mut session = self.session.lock().unwrap();
        let input = Tensor::from_array((
            vec![1i64, 3, h as i64, w as i64],
            chw,
        ))
        .map_err(|e| NeuronError::invalid_input(format!("ort input tensor: {e}")))?;
        let outputs = session
            .run(ort::inputs![input])
            .map_err(|e| NeuronError::new(format!("ort run: {e}")))?;
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| NeuronError::new(format!("ort extract: {e}")))?;
        Ok((
            data.to_vec(),
            shape.iter().map(|d| *d as usize).collect(),
        ))
    }

    fn name(&self) -> &'static str {
        "ort"
    }
}
