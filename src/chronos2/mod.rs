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
pub mod bolt;
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

/// Forecast method hint.
///
/// ⚠️ **The level is part of the value.** The earlier shape carried only the
/// word "quantile" and threw the level away — so "quantile:0.9" and
/// "quantile:0.1" were the same thing, *and* the wire form could not even
/// parse a level at all (the serde derive only knew the bare word). The
/// level now survives end to end: request → runtime → response.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ForecastMethod {
    Mean,
    Median,
    /// A quantile level in [0, 1].
    Quantile(f64),
}

impl ForecastMethod {
    /// Wire form: "mean" | "median" | "quantile:N".
    pub fn to_wire(self) -> String {
        match self {
            ForecastMethod::Mean => "mean".into(),
            ForecastMethod::Median => "median".into(),
            ForecastMethod::Quantile(q) => format!("quantile:{q}"),
        }
    }

    /// Parse the wire form. A bare "quantile" means the median.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim() {
            "mean" => Some(ForecastMethod::Mean),
            "median" => Some(ForecastMethod::Median),
            "quantile" => Some(ForecastMethod::Quantile(0.5)),
            other => other
                .strip_prefix("quantile:")
                .and_then(|rest| rest.trim().parse::<f64>().ok())
                .filter(|q| (0.0..=1.0).contains(q))
                .map(ForecastMethod::Quantile),
        }
    }

    /// Parse the "quantile:N" syntax where N is a float in [0, 1].
    /// Returns `None` if the input is malformed or out of range.
    pub fn parse_quantile_arg(s: &str) -> Option<Self> {
        let rest = s.strip_prefix("quantile:")?;
        let q: f64 = rest.trim().parse().ok()?;
        if !(0.0..=1.0).contains(&q) {
            return None;
        }
        Some(Self::Quantile(q))
    }
}

impl Serialize for ForecastMethod {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_wire())
    }
}

impl<'de> Deserialize<'de> for ForecastMethod {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let raw = String::deserialize(d)?;
        ForecastMethod::parse(&raw).ok_or_else(|| {
            serde::de::Error::custom(format!(
                "unknown forecast method '{raw}' (expected mean | median | quantile:N)"
            ))
        })
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
    /// ⚠ **The named checkpoint is not available on this neuron.**
    ///
    /// The message always names the id the caller asked for — a caller who
    /// mistyped has to be able to see *which* id was refused — and says why
    /// (not installed / the installed copy is a different model / nothing to
    /// fall back to).
    ///
    /// ⚠ This variant exists because the old shape answered a miss with
    /// the **stub** runtime, whose canned note (*"chronos2 candle port is not
    /// yet implemented"*) named neither the model nor the reason ⇒ ⇒ a
    /// caller could not tell "you asked for a checkpoint that is not
    /// installed" from "neuron cannot do time series at all".
    #[error("{0}")]
    ModelNotAvailable(String),
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

/// The model id this neuron serves when the caller does not name one.
pub fn default_model_id() -> String {
    crate::config::env_var_result("CHRONOS2_DEFAULT_MODEL")
        .unwrap_or_else(|_| "chronos-bolt-small".to_string())
}

/// True when `dir` holds a checkpoint (a `config.json` **plus** at least one
/// `*.safetensors` file).
///
/// ⚠ Why this has to be told apart: `NEURON_CHRONOS2_MODEL_PATH` may point
/// either at a **per-id tree** (`<root>/<id>/…`) or at **one checkpoint dir**
/// (`<root>/config.json` + `<root>/*.safetensors`) ⇒ ⇒ the two shapes
/// look alike from the outside ⇒ ⇒ without this check the resolver
/// would answer "here is a checkpoint" for a tree that holds none.
fn is_checkpoint_dir(dir: &std::path::Path) -> bool {
    if !dir.join("config.json").is_file() {
        return false;
    }
    std::fs::read_dir(dir)
        .ok()
        .map(|entries| {
            entries.flatten().any(|entry| {
                entry
                    .path()
                    .extension()
                    .map(|ext| ext == "safetensors")
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false)
}

/// The identity a checkpoint declares for itself, if it declares one.
///
/// Upstream checkpoints carry `_name_or_path` in `config.json`
/// (e.g. `"autogluon/chronos-bolt-small"`) ⇒ ⇒ the last segment is the
/// model's own name ⇒ ⇒ **that is what makes "serve another checkpoint
/// under this name" detectable** instead of undetectable.
pub fn declared_checkpoint_identity(dir: &std::path::Path) -> Option<String> {
    let raw = std::fs::read_to_string(dir.join("config.json")).ok()?;
    let value: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let name = value
        .get("_name_or_path")
        .and_then(|n| n.as_str())
        .or_else(|| value.get("name").and_then(|n| n.as_str()))?;
    let last = name.rsplit('/').next().unwrap_or(name).trim();
    (!last.is_empty()).then(|| last.to_string())
}

/// The roots the catalog tree (`<root>/chronos2/<id>`) is searched under.
///
/// ⚠ `NEURON_MODEL_DIRS` is a **list**. The separator is `;` on Windows and
/// `:` elsewhere — **and** a Windows drive letter contains `:` ⇒ ⇒ a
/// naive `split(':')` would cut `C:\models` in half (`C` + `\models`) ⇒
/// ⇒ the search would silently miss every checkpoint. Split on `;` first;
/// when there is none, split on `:` while keeping a `X:` prefix glued to the
/// first piece.
fn model_dir_roots() -> Vec<std::path::PathBuf> {
    let raw = match crate::config::env_var_result("MODEL_DIRS") {
        Ok(v) => v,
        Err(_) => return vec![crate::config::NeuronConfig::default_model_dir()],
    };
    let mut roots = Vec::new();
    for part in raw.split(';') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let (drive, rest) = if cfg!(windows) {
            match part.find(':') {
                Some(i) if i == 1 && part.as_bytes()[0].is_ascii_alphabetic() => {
                    (&part[..=1], &part[2..])
                }
                _ => ("", part),
            }
        } else {
            ("", part)
        };
        if rest.is_empty() {
            roots.push(std::path::PathBuf::from(part));
            continue;
        }
        for piece in rest.split(':') {
            let piece = piece.trim();
            if piece.is_empty() {
                continue;
            }
            roots.push(std::path::PathBuf::from(format!("{drive}{piece}")));
        }
    }
    roots
}

/// Refuse to serve a checkpoint under a name it does not claim.
///
/// ⚠ **This is the "no silent swap" guard.** A checkpoint that declares its
/// own identity (`config.json`'s `_name_or_path`) must declare **this** id
/// ⇒ ⇒ "asked for X, got Y" is refused instead of served.
fn check_identity(
    path: std::path::PathBuf,
    model: &str,
) -> Result<std::path::PathBuf, String> {
    if let Some(declared) = declared_checkpoint_identity(&path)
        && declared != model
    {
        return Err(format!(
            "the checkpoint at {} declares the model '{declared}', but '{model}' was requested (serving one checkpoint under another name is not allowed)",
            path.display()
        ));
    }
    Ok(path)
}

/// Locate the checkpoint directory for the **named** model.
///
/// ⚠ **The requested id decides the path — never the default id.**
/// The old shape returned `NEURON_CHRONOS2_MODEL_PATH` for *every* id ⇒ ⇒ a
/// request for a checkpoint that is not installed could be answered with
/// whatever weights that env var happened to point at ⇒ ⇒ "asked for X,
/// got Y" (a silent swap).
///
/// Search order (first hit wins):
///   1. `NEURON_CHRONOS2_MODEL_PATH/<model>` — a per-id tree;
///   2. `NEURON_CHRONOS2_MODEL_PATH` itself — ONE checkpoint dir ⇒ ⇒ it may
///      only answer for the id the env declares as the default model;
///   3. `$NEURON_MODEL_DIRS/chronos2/<model>/` — the catalog tree.
pub fn resolve_model_path(model: &str) -> Option<std::path::PathBuf> {
    resolve_model_path_with_reason(model).ok()
}

/// Same resolution as [`resolve_model_path`], but **explains the miss**: the id
/// that was asked for and every path that was tried.
///
/// ⇒ ⇒ "why is this model unavailable" is answerable from the error alone
/// (the caller does not have to guess which env var to check).
pub fn resolve_model_path_with_reason(
    model: &str,
) -> Result<std::path::PathBuf, String> {
    let mut tried: Vec<String> = Vec::new();

    // 1) `<NEURON_CHRONOS2_MODEL_PATH>/<model>` — a per-id tree.
    if let Ok(root) = crate::config::env_var_result("CHRONOS2_MODEL_PATH") {
        let candidate = std::path::PathBuf::from(&root).join(model);
        if candidate.is_dir() {
            return check_identity(candidate, model);
        }
        tried.push(candidate.display().to_string());
    }

    // 2) `NEURON_CHRONOS2_MODEL_PATH` itself — ONE checkpoint dir. It may only
    //    answer for the id the env declares as the default model; otherwise a
    //    mis-typed id would be served someone else's weights.
    if let Ok(root) = crate::config::env_var_result("CHRONOS2_MODEL_PATH") {
        let dir = std::path::PathBuf::from(&root);
        if is_checkpoint_dir(&dir) {
            let default_id = default_model_id();
            if default_id == model {
                return check_identity(dir, model);
            }
            // ⚠ **不把这个路径写进 tried ——** 它装的是
            // **别的**模型 ⇒ ⇒ 把别的 checkpoint 的名字写进
            // 回包里 ⇒ ⇒ 就把「没静默回落到别的
            // checkpoint」那一条判据自己碰破了 ⇒ ⇒
            // 回包里只能出现**点名的那枚**。
        }
    }

    // 3) the catalog tree: `<NEURON_MODEL_DIRS>/chronos2/<model>`.
    for root in model_dir_roots() {
        let candidate = root.join("chronos2").join(model);
        if candidate.is_dir() {
            return check_identity(candidate, model);
        }
        tried.push(candidate.display().to_string());
    }

    Err(format!(
        "no chronos2 checkpoint installed for model '{model}' (looked in: {})",
        tried.join("; ")
    ))
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
    let model_id = default_model_id();
    let path = resolve_model_path(&model_id)?;
    Some((model_id, path))
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
        assert_eq!(
            ForecastMethod::parse_quantile_arg("quantile:0.5"),
            Some(ForecastMethod::Quantile(0.5))
        );
        assert_eq!(
            ForecastMethod::parse_quantile_arg("quantile:0"),
            Some(ForecastMethod::Quantile(0.0))
        );
        assert_eq!(
            ForecastMethod::parse_quantile_arg("quantile:1"),
            Some(ForecastMethod::Quantile(1.0))
        );
    }

    /// The level has to survive the wire — otherwise "quantile:0.9" and
    /// "quantile:0.1" are the same request, and "quantile:0.9" cannot even
    /// be parsed by the old derive.
    #[test]
    fn the_quantile_level_survives_the_wire() {
        let parsed: ForecastMethod =
            serde_json::from_str("\"quantile:0.9\"").expect("parse quantile:0.9");
        assert_eq!(parsed, ForecastMethod::Quantile(0.9));
        assert_eq!(
            serde_json::to_string(&parsed).unwrap(),
            "\"quantile:0.9\""
        );
        let m: ForecastMethod = serde_json::from_str("\"mean\"").unwrap();
        assert_eq!(m, ForecastMethod::Mean);
        assert_eq!(serde_json::to_string(&m).unwrap(), "\"mean\"");
        assert!(serde_json::from_str::<ForecastMethod>("\"quantile:1.5\"").is_err());
        assert!(serde_json::from_str::<ForecastMethod>("\"nonsense\"").is_err());
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
