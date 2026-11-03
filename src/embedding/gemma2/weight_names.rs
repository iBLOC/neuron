//! Weight-path + shape verification against the upstream checkpoint.
//!
//! `google/embeddinggemma-2` ships **one unsharded** `model.safetensors`
//! (~1.49 GB, 1376 tensors). A typo in one `pp(...)` segment in the
//! vendored tower only surfaces at load time as "cannot find tensor X"
//! — after a 1.5 GB download. This module turns that into a
//! seconds-long check that touches only the safetensors header (the
//! first 171 kB of the file).
//!
//! How it works (same recipe as
//! `src/imagegen/moss_image/weight_names.rs`):
//!
//! 1. The **recorder** backend answers every `VarBuilder` request
//!    with a one-element zero tensor and remembers the path **and the
//!    requested shape**. Building [`TextModel`] with it yields the
//!    exact set of paths + shapes the port asks for, without
//!    materializing 1.5 GB.
//! 2. The **fetch** helpers pull `config.json` and the JSON header of
//!    `model.safetensors` (two Range requests over the network).
//! 3. The **diff** is strict in both directions for the text tower:
//!    every path the port requests must exist with an equal shape,
//!    and every `language_model.*` tensor in the checkpoint must be
//!    requested — text-only inference reads the *complete* text
//!    sub-model, so an unread `language_model.*` tensor is a silent
//!    mistake (e.g. a `linear_no_bias` where the checkpoint carries a
//!    different parameterization). The vision / audio towers are
//!    never requested by design.
//!
//! The tests here are `#[ignore]`d — they need the internet and are
//! not part of CI:
//!
//! ```bash
//! cargo test --lib verify_egemma2_weight_paths -- --ignored
//! cargo test --lib verify_egemma2_weight_shapes -- --ignored
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use candle::{DType, Device, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::{Init, VarBuilder};

/// The upstream repo (embedded in URLs; the test treats it as a slug).
const REPO: &str = "google/embeddinggemma-2";

/// A `SimpleBackend` that records every requested path (and the shape
/// it was requested with) and answers with a one-element zero tensor.
#[derive(Clone, Default)]
pub struct NameRecorder {
    seen: Arc<Mutex<Vec<(String, Vec<usize>)>>>,
}

impl NameRecorder {
    pub fn new() -> Self {
        Self::default()
    }

    /// The recorded `(path, shape)` pairs, sorted by path.
    pub fn entries(&self) -> Vec<(String, Vec<usize>)> {
        let mut v = self.seen.lock().unwrap().clone();
        v.sort();
        v.dedup();
        v
    }
}

impl SimpleBackend for NameRecorder {
    fn get(
        &self,
        s: Shape,
        name: &str,
        _h: Init,
        dtype: DType,
        dev: &Device,
    ) -> candle::Result<Tensor> {
        self.seen
            .lock()
            .unwrap()
            .push((name.to_string(), s.dims().to_vec()));
        Tensor::zeros(1, dtype, dev)
    }

    fn get_unchecked(
        &self,
        name: &str,
        _dtype: DType,
        _dev: &Device,
    ) -> candle::Result<Tensor> {
        self.seen
            .lock()
            .unwrap()
            .push((name.to_string(), Vec::new()));
        Err(candle::Error::Msg(format!(
            "get_unchecked is not supported by the name recorder ({name})"
        )))
    }

    fn contains_tensor(&self, _name: &str) -> bool {
        true
    }
}

/// Build the vendored text tower against a recorder positioned at the
/// `language_model.` prefix, returning the recorded entries.
pub fn record_requested_paths(cfg: &super::config::TextConfig) -> Vec<(String, Vec<usize>)> {
    let device = Device::Cpu;
    let recorder = NameRecorder::new();
    let vb = VarBuilder::from_backend(Box::new(recorder.clone()), DType::F32, device);
    super::text_model::TextModel::new(cfg, vb.pp("language_model"))
        .expect("TextModel must build against the recorder");
    recorder.entries()
}

/// Fetch a byte range of a URL (https only, up to 3 redirects).
async fn fetch_bytes_range(url: &str, start: u64, end: u64) -> Result<Vec<u8>, String> {
    let parsed = reqwest::Url::parse(url).map_err(|e| format!("bad url {url}: {e}"))?;
    if parsed.scheme() != "https" {
        return Err(format!("only https URLs are allowed: {url}"));
    }
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::limited(3))
        .build()
        .map_err(|e| format!("client: {e}"))?;
    let resp = client
        .get(parsed)
        .header("User-Agent", "neuron-verify-egemma2-weights")
        .header("Range", format!("bytes={start}-{end}"))
        .send()
        .await
        .map_err(|e| format!("GET {url}: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("GET {url}: HTTP {}", resp.status()));
    }
    resp.bytes()
        .await
        .map(|b| b.to_vec())
        .map_err(|e| format!("body {url}: {e}"))
}

/// Parse a safetensors JSON header: `{name: {"dtype":..,"shape":[..]}}`
/// (with the 8-byte little-endian length prefix in front).
fn parse_safetensors_header(head: &[u8]) -> Result<BTreeMap<String, Vec<usize>>, String> {
    if head.len() < 8 {
        return Err("short read for the safetensors header".to_string());
    }
    let header_len = u64::from_le_bytes(head[0..8].try_into().unwrap()) as usize;
    if header_len + 8 > head.len() {
        return Err(format!(
            "safetensors header needs {} bytes, got {} — raise the range",
            header_len + 8,
            head.len()
        ));
    }
    let json: serde_json::Value = serde_json::from_slice(&head[8..8 + header_len])
        .map_err(|e| format!("header json: {e}"))?;
    let mut out = BTreeMap::new();
    for (name, entry) in json.as_object().ok_or("header is not an object")? {
        if name == "__metadata__" {
            continue;
        }
        let shape: Vec<usize> = entry["shape"]
            .as_array()
            .ok_or_else(|| format!("no shape for {name}"))?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect();
        out.insert(name.clone(), shape);
    }
    Ok(out)
}

/// The upstream `(name → shape)` map, and the text_config parsed from
/// the live `config.json`.
async fn upstream_state() -> Result<(BTreeMap<String, Vec<usize>>, super::config::TextConfig), String>
{
    let base = format!("https://huggingface.co/{REPO}/resolve/main");

    // config.json — parse text_config so the recorder builds the tower
    // with the checkpoint's own layer counts and head shapes.
    let body = fetch_bytes_range(&format!("{base}/config.json"), 0, 65_535).await?;
    let json: serde_json::Value =
        serde_json::from_slice(&body).map_err(|e| format!("config.json: {e}"))?;
    let text_config = json
        .get("text_config")
        .cloned()
        .ok_or("config.json has no text_config")?;
    let cfg: super::config::TextConfig = serde_json::from_value(text_config)
        .map_err(|e| format!("parse text_config: {e}"))?;

    // safetensors header — first 8 bytes give the JSON length.
    let weights = format!("{base}/model.safetensors");
    let first = fetch_bytes_range(&weights, 0, 7).await?;
    if first.len() < 8 {
        return Err("short read for the header length prefix".to_string());
    }
    let header_len = u64::from_le_bytes(first[0..8].try_into().unwrap()) as usize;
    let head = fetch_bytes_range(&weights, 0, (8 + header_len - 1) as u64).await?;
    let shapes = parse_safetensors_header(&head)?;
    Ok((shapes, cfg))
}

/// Diff the port's requested paths against the upstream header.
/// Strict both ways inside `language_model.*`. The recorded paths
/// already carry the `language_model.` prefix (the recorder sits at
/// that checkpoint prefix), so they are compared verbatim.
pub fn diff_paths(
    requested: &[(String, Vec<usize>)],
    upstream: &BTreeMap<String, Vec<usize>>,
) -> Result<(), String> {
    let mut problems = Vec::new();

    // Every requested path must exist with an equal shape.
    for (name, shape) in requested {
        match upstream.get(name) {
            None => problems.push(format!("missing upstream: {name}")),
            Some(up) if up != shape => {
                problems.push(format!("shape mismatch: {name} wants {shape:?}, upstream {up:?}"))
            }
            Some(_) => {}
        }
    }

    // Every `language_model.*` tensor must be requested (text-only
    // inference reads the complete text sub-model).
    let requested_full: BTreeSet<&str> = requested
        .iter()
        .map(|(n, _)| n.as_str())
        .collect();
    for name in upstream.keys() {
        if name.starts_with("language_model.") && !requested_full.contains(name.as_str()) {
            problems.push(format!("unused upstream tensor: {name}"));
        }
    }

    if problems.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{} weight-path problem(s):\n  - {}",
            problems.len(),
            problems.join("\n  - ")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `cargo test --lib verify_egemma2_weight_paths -- --ignored`
    /// Network: fetches config.json + the safetensors header (~200 kB)
    /// and diffs the port's paths + shapes against them.
    #[tokio::test]
    #[ignore]
    async fn verify_egemma2_weight_paths() {
        let (upstream, cfg) = upstream_state().await.expect("fetch upstream state");
        let requested = record_requested_paths(&cfg);
        assert!(
            requested.len() > 100,
            "the recorder saw only {} paths — the build must have short-circuited",
            requested.len()
        );
        match diff_paths(&requested, &upstream) {
            Ok(()) => println!(
                "egemma2 weight paths OK: {} requested, {} language_model.* upstream tensors, all matched",
                requested.len(),
                upstream.keys().filter(|k| k.starts_with("language_model.")).count()
            ),
            Err(e) => panic!("{e}"),
        }
    }

    /// A pure-offline sanity check of the header parser against a
    /// hand-built mini header (no network).
    #[test]
    fn test_parse_safetensors_header_mini_fixture() {
        let entry = serde_json::json!({
            "embed_tokens.weight": {"dtype": "BF16", "shape": [64, 16], "data_offsets": [0, 2048]}
        });
        let json = serde_json::to_string(&entry).unwrap();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(json.len() as u64).to_le_bytes());
        bytes.extend_from_slice(json.as_bytes());
        let parsed = parse_safetensors_header(&bytes).unwrap();
        assert_eq!(parsed.get("embed_tokens.weight"), Some(&vec![64usize, 16]));
    }

    /// The diff's unused-tensor rule: a checkpoint tensor the port
    /// never reads inside `language_model.*` is a hard failure.
    #[test]
    fn test_diff_flags_unused_language_model_tensor() {
        let mut upstream = BTreeMap::new();
        upstream.insert("language_model.norm.weight".to_string(), vec![8usize]);
        upstream.insert("language_model.sneaky_extra.weight".to_string(), vec![8usize]);
        // Recorded paths carry the language_model. prefix.
        let requested = vec![("language_model.norm.weight".to_string(), vec![8usize])];
        let err = diff_paths(&requested, &upstream).unwrap_err();
        assert!(err.contains("sneaky_extra"), "{err}");
    }
}
