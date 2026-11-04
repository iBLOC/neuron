//! Chronos2 time-series forecasting — neuron backend.
//!
//! Public surface mirrors the existing per-capability pattern
//! (`LlmRuntime`, `TtsRuntime`, ...): a `Chronos2Runtime` trait, a
//! `Chronos2Registry` (lazy-load, resource-tracked), and a
//! `feature = "chronos2"` gate around the real candle-based
//! implementation.
//!
//! ## Current status (2026-10-31)
//!
//! The trait + DTOs, the registry, the `POST /v1/forecast` route and a
//! candle runtime (`CandleChronos2Runtime`, behind the `chronos2`
//! feature) are all in place. The runtime loads a safetensors
//! checkpoint, runs a forward pass and returns a quantile output head.
//!
//! **KNOWN GAP (found 2026-10-31 by `benches/model_bench.rs` on the
//! real upstream checkpoint): the vendored port does NOT match the
//! upstream Chronos-T5 architecture.** The upstream release
//! (`amazon/chronos-t5-small`) ships Google's **vanilla T5** module
//! tree — `shared.weight`, `encoder.block.N.layer.0.SelfAttention.{q,k,v,o}`,
//! `DenseReluDense.wi/wo`, `layer_norm` (LayerNorm), plus
//! `relative_attention_bias` — while the vendored port expects a
//! different, RmsNorm-based naming (`token_emb`, `self_q`/`self_k`,
//! `attn_norm`/`ffn_norm`, `ffn_up`/`ffn_down`). The checkpoint also
//! needs the **MeanScaleUniformBins** tokenizer
//! (`chronos_config.tokenizer_class` in config.json), which the port
//! does not implement.
//!
//! So the failure is NOT a key-remap away (unlike the jina reranker,
//! where a remap WAS enough): the port is a different architecture.
//! Serving the upstream release needs a real T5 port — encoder +
//! decoder + relative attention bias + the bin tokenizer. Until then
//! `load()` fails naming the missing tensor (`token_emb.weight`) and
//! the registry surfaces that error verbatim instead of the old canned
//! "port not vendored" note.
//!
//! The config parser now accepts BOTH the port's flattened key names
//! and the upstream T5 names (`n_positions` → `max_seq_len`,
//! `num_layers` → `n_enc`, `num_decoder_layers` → `n_dec`,
//! `layer_norm_epsilon` → `layer_norm_eps`).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::config::{DeviceSelection, ResourceTracker};

#[cfg(feature = "candle")]
pub mod mask;
pub mod metrics;
pub mod registry;
pub mod tokenizer;
pub use registry::Chronos2Registry;

#[cfg(feature = "chronos2")]
pub mod candle_runtime;
#[cfg(feature = "chronos2")]
pub mod output;
#[cfg(feature = "chronos2")]
pub mod vendor;
#[cfg(feature = "chronos2")]
pub use candle_runtime::{CandleChronos2Error, CandleChronos2Runtime, Chronos2Config};
#[cfg(feature = "chronos2")]
pub use output::{Chronos2OutputHead, InterpolationMethod, OutputHeadConfig};
#[cfg(feature = "chronos2")]
pub use vendor::{T5Block, T5BlockConfig, T5BlockOutput, T5Embeddings, T5EmbeddingsConfig};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastRequest {
    /// Model id. `NEURON_CHRONOS2_RUNTIME` + `NEURON_CHRONOS2_MODEL_PATH`
    /// decide the model at startup; the request carries a name for the
    /// HTTP layer's audit / per-request logging only.
    pub model: String,
    /// Time series to forecast, in chronological order. Minimum
    /// length 1; recommended >= 16 for the candle port to produce
    /// non-trivial output.
    pub series: Vec<f64>,
    /// Number of future steps to produce.
    pub horizon: u32,
    /// Forecast method hint. `mean` returns the expected value of
    /// the model's predictive distribution; `median` returns the
    /// 50% quantile; `quantile:N` returns the N-quantile
    /// (`0 <= N <= 1`). Defaults to `mean`.
    #[serde(default = "default_method")]
    pub method: ForecastMethod,
}

fn default_method() -> ForecastMethod {
    ForecastMethod::Mean
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ForecastMethod {
    Mean,
    Median,
    Quantile,
}

impl ForecastMethod {
    /// Parse the "quantile:N" syntax where N is a float in [0, 1].
    /// Returns `None` if the input is malformed or out of range.
    pub fn parse_quantile_arg(s: &str) -> Option<Self> {
        let rest = s.strip_prefix("quantile:")?;
        let q: f64 = rest.parse().ok()?;
        if !(0.0..=1.0).contains(&q) {
            return None;
        }
        Some(Self::Quantile)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastResponse {
    pub object: String,
    pub model: String,
    pub method: ForecastMethod,
    /// Forecasted values, in chronological order. Length == `horizon`.
    pub forecast: Vec<f64>,
    pub took_ms: u64,
}

/// The runtime trait every Chronos2 backend implements. Mirrors the
/// shape of `LlmRuntime` so the rest of the neuron wiring (registry,
/// ServerState shortcut, env-var discovery) stays homogeneous.
#[async_trait]
pub trait Chronos2Runtime: Send + Sync {
    /// Return the model id this runtime serves.
    fn model_id(&self) -> &str;
    /// Run a forecast synchronously. Returns the time-bucketed
    /// predictions plus a per-request timing field. The runtime is
    /// responsible for honoring `request.method` (mean / median /
    /// quantile) when collapsing the predictive distribution.
    async fn forecast(
        &self,
        request: ForecastRequest,
    ) -> Result<ForecastResponse, Chronos2Error>;
}

/// Error surface. Mirrors the existing LLM / TTS / Rerank patterns
/// so the HTTP layer can map variants onto 4xx / 5xx codes uniformly.
#[derive(Debug, Error)]
pub enum Chronos2Error {
    #[error("not implemented: {0}")]
    NotImplemented(String),
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("model not loaded: {0}")]
    NotLoaded(String),
    #[error("runtime error: {0}")]
    Runtime(String),
}

/// Stub runtime returned by the registry when no real backend is
/// configured. Lets the HTTP layer surface a clean 503 with a
/// helpful message instead of crashing on the missing
/// `feature = "chronos2"` build.
pub struct StubChronos2Runtime {
    model_id: String,
    /// Why the real runtime is unavailable (a failed checkpoint load, or
    /// the feature being off). Surfaced verbatim by `forecast` so callers
    /// see the ACTUAL reason instead of a canned "port not vendored"
    /// note — that canned text was actively misleading once the port
    /// landed: it told users to enable a feature that was already on
    /// (found 2026-10-31 by the per-model bench).
    load_error: Option<String>,
}

impl StubChronos2Runtime {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            load_error: None,
        }
    }

    pub fn with_load_error(model_id: impl Into<String>, load_error: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            load_error: Some(load_error.into()),
        }
    }
}

#[async_trait]
impl Chronos2Runtime for StubChronos2Runtime {
    fn model_id(&self) -> &str {
        &self.model_id
    }
    async fn forecast(
        &self,
        _request: ForecastRequest,
    ) -> Result<ForecastResponse, Chronos2Error> {
        Err(Chronos2Error::NotImplemented(match &self.load_error {
            Some(err) => format!(
                "chronos2 runtime unavailable for '{}': {err}",
                self.model_id
            ),
            None => "chronos2 candle port is not yet implemented; this is the Sprint 3 skeleton. \
             Enable the `chronos2` cargo feature once the architecture is vendored."
                .into(),
        }))
    }
}

/// Locate a Chronos2 model directory the way the LLM candle runtime
/// does. `NEURON_CHRONOS2_MODEL_PATH` overrides for tests + custom
/// deployments; the default resolution scans `$NEURON_MODEL_DIRS/chronos2/<id>/`
/// for the matching id.
pub fn resolve_model_path(model: &str) -> Option<std::path::PathBuf> {
    if let Ok(p) = crate::config::env_var_result("CHRONOS2_MODEL_PATH") {
        let pb = std::path::PathBuf::from(p);
        if pb.is_dir() {
            return Some(pb);
        }
    }
    for root in crate::config::env_var_result("MODEL_DIRS")
        .unwrap_or_else(|_| crate::config::NeuronConfig::default_model_dir().to_string_lossy().into_owned())
        .split(':')
    {
        let candidate = std::path::PathBuf::from(root).join("chronos2").join(model);
        if candidate.is_dir() {
            return Some(candidate);
        }
    }
    None
}

/// Resource-quota helper. Chronos2 is small (≤ 1B params for
/// amazon/chronos-bolt-small) so 1 GiB is enough.
pub fn required_memory_bytes() -> u64 {
    1_073_741_824
}

/// Helper for the HTTP layer to keep the env-var handling in one
/// place. Returns `(model_id, model_dir)` if a chronos2 model is
/// configured; `None` when the runtime is not enabled (caller maps
/// that to a 503).
pub fn env_discovery() -> Option<(String, std::path::PathBuf)> {
    let model_id = crate::config::env_var_result("CHRONOS2_DEFAULT_MODEL")
        .unwrap_or_else(|_| "chronos-bolt-small".to_string());
    let path = resolve_model_path(&model_id)?;
    Some((model_id, path))
}

/// Internal registry bookkeeping; the public API lives in
/// `mod.rs::Chronos2Registry`. The struct itself is exported so
/// the tests can construct one without a stub runtime's `default_runtime`
/// detour.
pub struct RegistryInner {
    pub registered: Vec<RegisteredChronos2>,
    pub loaded: Arc<tokio::sync::Mutex<HashMap<String, Arc<dyn Chronos2Runtime>>>>,
    pub resources: Mutex<ResourceTracker>,
}

#[derive(Debug, Clone)]
pub struct RegisteredChronos2 {
    pub model_id: String,
    pub device: DeviceSelection,
    pub required_memory_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_quantile_accepts_in_range_values() {
        assert_eq!(ForecastMethod::parse_quantile_arg("quantile:0.5"), Some(ForecastMethod::Quantile));
        assert_eq!(ForecastMethod::parse_quantile_arg("quantile:0"), Some(ForecastMethod::Quantile));
        assert_eq!(ForecastMethod::parse_quantile_arg("quantile:1"), Some(ForecastMethod::Quantile));
    }

    #[test]
    fn parse_quantile_rejects_out_of_range_or_malformed() {
        assert_eq!(ForecastMethod::parse_quantile_arg("quantile:-0.1"), None);
        assert_eq!(ForecastMethod::parse_quantile_arg("quantile:1.5"), None);
        assert_eq!(ForecastMethod::parse_quantile_arg("quantile:abc"), None);
        assert_eq!(ForecastMethod::parse_quantile_arg("mean"), None);
    }

    #[test]
    fn default_method_is_mean() {
        assert_eq!(default_method(), ForecastMethod::Mean);
    }

    #[test]
    fn required_memory_bytes_is_one_gib() {
        assert_eq!(required_memory_bytes(), 1_073_741_824);
    }

    #[tokio::test]
    async fn stub_runtime_returns_not_implemented() {
        let rt = StubChronos2Runtime::new("test");
        assert_eq!(rt.model_id(), "test");
        let resp = rt
            .forecast(ForecastRequest {
                model: "test".into(),
                series: vec![1.0, 2.0, 3.0],
                horizon: 2,
                method: ForecastMethod::Mean,
            })
            .await;
        assert!(matches!(resp.unwrap_err(), Chronos2Error::NotImplemented(_)));
    }
}
