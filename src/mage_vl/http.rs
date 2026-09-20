//! HTTP surface for the Mage-VL vendor tree.
//!
//! `POST /v1/vision/gate/score` accepts one or more pre-encoded
//! segments (`[T_i, P, D]` Mage-ViT tokens) and returns the
//! StreamMind p_speak probability per segment. The endpoint is
//! self-contained: the gate weights are loaded lazily from the path
//! in `NEURON_MAGE_VL_GATE_PATH` (default
//! `$NEURON_MODEL_DIRS/mistralrs/microsoft/Mage-VL/streammind_gate.safetensors`)
//! and cached in-process.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use axum::{extract::State, http::StatusCode, Json};
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use tokio::sync::OnceCell;

use super::gate::StreamMindGate;
use crate::server::ServerError;

#[derive(Clone, Debug, Deserialize)]
pub struct SegmentRequest {
    /// Flat row-major `[T*P*D]` f32 tokens.
    pub tokens: Vec<f32>,
    pub t: usize,
    pub p: usize,
    pub d: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct GateScoreRequest {
    pub segments: Vec<SegmentRequest>,
    /// Optional override path to `streammind_gate.safetensors`.
    #[serde(default)]
    pub gate_path: Option<String>,
    /// Gate threshold for the `events` shortlist.
    #[serde(default = "default_threshold")]
    pub threshold: f32,
}

fn default_threshold() -> f32 {
    0.5
}

#[derive(Debug, Serialize)]
pub struct GateScoreResponse {
    pub probabilities: Vec<f32>,
    pub events: Vec<usize>,
    pub took_ms: u64,
    pub segment_lengths: Vec<usize>,
}

#[derive(Default)]
pub struct GateRuntime {
    cell: OnceCell<Arc<StreamMindGate>>,
}

impl std::fmt::Debug for GateRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GateRuntime")
            .field("loaded", &self.cell.get().is_some())
            .finish()
    }
}

impl GateRuntime {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn get(
        &self,
        gate_path: &std::path::Path,
        device: &Device,
    ) -> Result<Arc<StreamMindGate>, ServerError> {
        self.cell
            .get_or_try_init(|| async {
                let gate = StreamMindGate::load_safetensors(gate_path, device)
                    .map_err(|e| ServerError::Internal(format!("gate load failed: {e}")))?;
                Ok::<_, ServerError>(Arc::new(gate))
            })
            .await
            .cloned()
    }
}

fn default_gate_path() -> PathBuf {
    if let Ok(p) = crate::config::env_var_result("MAGE_VL_GATE_PATH") {
        return PathBuf::from(p);
    }
    let root = crate::config::env_var_result("MODEL_DIRS")
        .ok()
        .and_then(|v| v.split(':').next().map(|s| s.to_string()))
        .unwrap_or_else(|| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
            format!("{home}/.neuron/models")
        });
    PathBuf::from(root)
        .join("mistralrs")
        .join("microsoft/Mage-VL")
        .join("streammind_gate.safetensors")
}

fn bad_request(msg: String) -> (StatusCode, String) {
    (StatusCode::BAD_REQUEST, msg)
}

fn unavailable(msg: String) -> (StatusCode, String) {
    (StatusCode::SERVICE_UNAVAILABLE, msg)
}

fn internal(msg: String) -> (StatusCode, String) {
    (StatusCode::INTERNAL_SERVER_ERROR, msg)
}

/// `POST /v1/vision/gate/score` — score one or more contiguous
/// segments of Mage-ViT tokens.
pub async fn gate_score(
    State(state): State<crate::server::ServerState>,
    Json(req): Json<GateScoreRequest>,
) -> Result<Json<GateScoreResponse>, (StatusCode, String)> {
    if req.segments.is_empty() {
        return Err(bad_request("segments cannot be empty".into()));
    }
    if !(0.0..=1.0).contains(&req.threshold) {
        return Err(bad_request("threshold must be in [0, 1]".into()));
    }
    for (idx, seg) in req.segments.iter().enumerate() {
        if seg.t == 0 || seg.p == 0 || seg.d == 0 {
            return Err(bad_request(format!(
                "segment {idx}: t/p/d must all be > 0"
            )));
        }
        let expected = seg.t * seg.p * seg.d;
        if seg.tokens.len() != expected {
            return Err(bad_request(format!(
                "segment {idx}: tokens.len()={} but t*p*d={expected}",
                seg.tokens.len()
            )));
        }
    }

    let gate_path = req
        .gate_path
        .as_deref()
        .map(PathBuf::from)
        .unwrap_or_else(default_gate_path);
    if !gate_path.exists() {
        return Err(unavailable(format!(
            "streammind_gate.safetensors not found at {} — \
             set NEURON_MAGE_VL_GATE_PATH or pass gate_path in the request",
            gate_path.display()
        )));
    }

    let device = Device::Cpu;
    let gate = state
        .mage_vl_gate
        .get(&gate_path, &device)
        .await
        .map_err(|e| internal(format!("{e}")))?;

    let started = Instant::now();
    let mut tensors = Vec::with_capacity(req.segments.len());
    let mut lengths = Vec::with_capacity(req.segments.len());
    for seg in req.segments.iter() {
        let flat = Tensor::from_vec(seg.tokens.clone(), (seg.t, seg.p, seg.d), &device)
            .map_err(|e| internal(format!("{e}")))?;
        tensors.push(flat);
        lengths.push(seg.t);
    }

    let probabilities = gate
        .score_segments(&tensors)
        .map_err(|e| internal(format!("{e}")))?;
    let events = probabilities
        .iter()
        .enumerate()
        .filter_map(|(i, p)| (*p >= req.threshold).then_some(i))
        .collect();

    Ok(Json(GateScoreResponse {
        probabilities,
        events,
        took_ms: started.elapsed().as_millis() as u64,
        segment_lengths: lengths,
    }))
}