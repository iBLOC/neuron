//! Weight-path verification against the upstream checkpoint.
//!
//! The upstream repo (`OpenMOSS-Team/MOSS-Image-Base`) ships one model
//! with three sub-models and 1,473 tensors. A typo in one `pp(...)`
//! segment in neuron's port only surfaces at load time as "cannot
//! find tensor X" — after a 27 GB download. This module turns that
//! into a seconds-long check that touches only the index files (a few
//! hundred kB).
//!
//! How it works:
//!
//! 1. The **recorder** backend ([`NameRecorder`]) answers every
//!    `VarBuilder` request with a zero tensor of the requested shape
//!    and remembers the path. Building the three modules with it
//!    yields the exact set of paths the port asks for. Names are
//!    dimension-independent, so the modules can be built with the
//!    real layer counts (42 DiT blocks / 32 text layers) and tiny
//!    dims.
//! 2. The **fetch** helpers read the upstream index files: the two
//!    `*.safetensors.index.json` manifests and the JSON header of the
//!    single-file VAE (only the first bytes are needed).
//! 3. The **diff** classifies the mismatch: a path neuron requests
//!    but the checkpoint lacks is a hard failure; a checkpoint tensor
//!    neuron never reads is expected for the vision tower, the VAE
//!    encoder and derived constants.
//!
//! The network test is `#[ignore]`d — it needs the internet and is
//! not part of CI.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use candle::{DType, Device, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::{Init, VarBuilder};

/// Checkpoint tensors the port deliberately never reads.
const EXPECTED_UNUSED_PREFIXES: &[&str] = &[
    "visual.",     // Qwen3.5 vision tower (text-only inference)
    "encoder.",    // VAE encoder (decode-only)
    "quant_conv.", // VAE encoder-side quant conv
];

/// Exact checkpoint tensors the port deliberately never reads.
const EXPECTED_UNUSED_EXACT: &[&str] = &[
    "rope.bands",             // derived constant — rebuilt from rope_base
    "bn.num_batches_tracked", // BatchNorm bookkeeping counter
];

/// A `SimpleBackend` that records every requested path (and the shape
/// it was requested with) and answers with a zero tensor of that
/// shape.
#[derive(Clone, Default)]
pub struct NameRecorder {
    seen: Arc<Mutex<Vec<(String, Vec<usize>)>>>,
}

impl NameRecorder {
    pub fn new() -> Self {
        Self::default()
    }

    /// The recorded paths, sorted and de-duplicated.
    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.seen.lock().unwrap().iter().map(|(n, _)| n.clone()).collect();
        v.sort();
        v.dedup();
        v
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
        // A one-element tensor: the module constructors only *store*
        // the loaded tensor (no shape arithmetic at construction), so
        // the recorded shape is the information we need — and this
        // keeps a real-config probe from allocating the ~10 GB the
        // full-size weights would take.
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

/// Build a `VarBuilder` over a [`NameRecorder`].
pub fn recording_vb(device: &Device, dtype: DType) -> (VarBuilder<'static>, NameRecorder) {
    let recorder = NameRecorder::new();
    let vb = VarBuilder::from_backend(Box::new(recorder.clone()), dtype, device.clone());
    (vb, recorder)
}

/// The outcome of diffing the port's paths against the checkpoint.
#[derive(Debug, Default)]
pub struct WeightPathDiff {
    /// Paths the port requests that the checkpoint does not have —
    /// each one would abort the load.
    pub missing: Vec<String>,
    /// Checkpoint tensors the port never reads.
    pub unused: Vec<String>,
    /// Of those, the ones that are NOT expected (a real gap).
    pub unexpected_unused: Vec<String>,
}

/// Classify the difference between the port's requested paths and the
/// checkpoint's tensor names.
pub fn diff(requested: &[String], checkpoint: &BTreeSet<String>) -> WeightPathDiff {
    let mut diff = WeightPathDiff::default();
    for name in requested {
        if !checkpoint.contains(name) {
            diff.missing.push(name.clone());
        }
    }
    for name in checkpoint {
        if requested.iter().any(|r| r == name) {
            continue;
        }
        let expected = EXPECTED_UNUSED_EXACT.contains(&name.as_str())
            || EXPECTED_UNUSED_PREFIXES
                .iter()
                .any(|p| name.starts_with(p));
        if expected {
            diff.unused.push(name.clone());
        } else {
            diff.unexpected_unused.push(name.clone());
        }
    }
    diff.missing.sort();
    diff.unused.sort();
    diff.unexpected_unused.sort();
    diff
}

/// Fetch a URL's body as bytes (https only).
async fn fetch_bytes(url: &str) -> Result<Vec<u8>, String> {
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
        .header("User-Agent", "neuron-verify-moss-weights")
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

/// The upstream tensor names of the three sub-models.
pub async fn upstream_tensor_names(repo: &str) -> Result<BTreeSet<String>, String> {
    // The repo id is embedded in URLs — keep it to a slug so it cannot
    // smuggle a path or query string into the request target.
    if !is_safe_repo_id(repo) {
        return Err(format!("unsafe repo id: {repo}"));
    }
    let base = format!("https://huggingface.co/{repo}/resolve/main");
    let mut names = BTreeSet::new();

    let transformer = format!("{base}/transformer/diffusion_pytorch_model.safetensors.index.json");
    let body = fetch_bytes(&transformer).await?;
    let manifest: serde_json::Value =
        serde_json::from_slice(&body).map_err(|e| format!("transformer index: {e}"))?;
    collect_weight_map(&manifest, &mut names)?;

    let text_encoder = format!("{base}/text_encoder/model.safetensors.index.json");
    let body = fetch_bytes(&text_encoder).await?;
    let manifest: serde_json::Value =
        serde_json::from_slice(&body).map_err(|e| format!("text_encoder index: {e}"))?;
    collect_weight_map(&manifest, &mut names)?;

    // The VAE is a single safetensors file: its JSON header (the
    // first bytes) lists the tensor names.
    let vae = format!("{base}/vae/diffusion_pytorch_model.safetensors");
    let head = fetch_bytes_range(&vae, 0, 1_048_575).await?;
    let header = parse_safetensors_header(&head)?;
    merge_header_names(&header, &mut names)?;
    Ok(names)
}

/// The upstream `(name → shape)` map of all three sub-models.
///
/// The transformer and the text encoder are sharded; every shard
/// carries its own header, so the shapes are collected from all of
/// them (the manifests say which shard holds which tensor). The VAE
/// is a single file.
pub async fn upstream_shapes(
    repo: &str,
) -> Result<BTreeMap<String, Vec<usize>>, String> {
    if !is_safe_repo_id(repo) {
        return Err(format!("unsafe repo id: {repo}"));
    }
    let mut shapes = BTreeMap::new();

    let transformer = upstream_shard_shapes(
        repo,
        "transformer",
        "diffusion_pytorch_model.safetensors.index.json",
    )
    .await?;
    shapes.extend(transformer);

    let text_encoder =
        upstream_shard_shapes(repo, "text_encoder", "model.safetensors.index.json").await?;
    shapes.extend(text_encoder);

    let base = format!("https://huggingface.co/{repo}/resolve/main");
    let vae = format!("{base}/vae/diffusion_pytorch_model.safetensors");
    let head = fetch_bytes_range(&vae, 0, 1_048_575).await?;
    let header = parse_safetensors_header(&head)?;
    merge_header_shapes(&header, &mut shapes)?;
    Ok(shapes)
}

/// Parse a safetensors JSON header: `{name: {"dtype":..,"shape":[..]}}`.
fn parse_safetensors_header(head: &[u8]) -> Result<serde_json::Value, String> {
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
    serde_json::from_slice(&head[8..8 + header_len]).map_err(|e| format!("header json: {e}"))
}

/// Merge the names of a safetensors header into the set.
fn merge_header_names(
    header: &serde_json::Value,
    out: &mut BTreeSet<String>,
) -> Result<(), String> {
    let map = header
        .as_object()
        .ok_or_else(|| "safetensors header is not an object".to_string())?;
    for key in map.keys() {
        if key != "__metadata__" {
            out.insert(key.clone());
        }
    }
    Ok(())
}

/// Merge the shapes of a safetensors header into the index.
fn merge_header_shapes(
    header: &serde_json::Value,
    out: &mut BTreeMap<String, Vec<usize>>,
) -> Result<(), String> {
    let map = header
        .as_object()
        .ok_or_else(|| "safetensors header is not an object".to_string())?;
    for (key, meta) in map {
        if key == "__metadata__" {
            continue;
        }
        let shape = meta
            .get("shape")
            .and_then(|s| s.as_array())
            .map(|dims| {
                dims.iter()
                    .filter_map(|d| d.as_u64().map(|v| v as usize))
                    .collect::<Vec<usize>>()
            })
            .ok_or_else(|| format!("tensor {key} has no shape in the header"))?;
        out.insert(key.clone(), shape);
    }
    Ok(())
}

/// Read the shapes of the tensors the port requests out of the
/// upstream shard headers.
///
/// The transformer and the text encoder are sharded; every shard
/// carries its own header, so the shapes are collected from all of
/// them (the manifests say which shard holds which tensor).
async fn upstream_shard_shapes(
    repo: &str,
    subdir: &str,
    manifest_file: &str,
) -> Result<BTreeMap<String, Vec<usize>>, String> {
    if !is_safe_repo_id(repo) {
        return Err(format!("unsafe repo id: {repo}"));
    }
    let base = format!("https://huggingface.co/{repo}/resolve/main/{subdir}");
    let body = fetch_bytes(&format!("{base}/{manifest_file}")).await?;
    let manifest: serde_json::Value =
        serde_json::from_slice(&body).map_err(|e| format!("{manifest_file}: {e}"))?;
    let weight_map = manifest
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| format!("{manifest_file}: no weight_map"))?;

    let mut shards: BTreeSet<String> = BTreeSet::new();
    for file in weight_map.values() {
        if let Some(f) = file.as_str() {
            shards.insert(f.to_string());
        }
    }
    let mut shapes = BTreeMap::new();
    for shard in shards {
        let head = fetch_bytes_range(&format!("{base}/{shard}"), 0, 1_048_575).await?;
        let header = parse_safetensors_header(&head)?;
        merge_header_shapes(&header, &mut shapes)?;
    }
    Ok(shapes)
}

/// `owner/name` with slug characters only.
fn is_safe_repo_id(repo: &str) -> bool {
    let mut parts = repo.split('/');
    let (Some(owner), Some(name)) = (parts.next(), parts.next()) else {
        return false;
    };
    if parts.next().is_some() {
        return false;
    }
    let slug_ok = |s: &str| {
        !s.is_empty()
            && s.chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
    };
    slug_ok(owner) && slug_ok(name)
}

fn collect_weight_map(
    manifest: &serde_json::Value,
    out: &mut BTreeSet<String>,
) -> Result<(), String> {
    let map = manifest
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| "index has no weight_map object".to_string())?;
    for key in map.keys() {
        out.insert(key.clone());
    }
    Ok(())
}

/// Byte-range GET (the VAE header is at the front of the file).
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
        .header("User-Agent", "neuron-verify-moss-weights")
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imagegen::moss_image::{
        dit::{MossDit, MossDitConfig},
        text_encoder::{Qwen35TextConfig, Qwen35TextModel},
        vae::{Flux2Vae, Flux2VaeConfig},
    };

    /// Build the three modules with a recording `VarBuilder` and
    /// return every path the port requests.
    ///
    /// The layer counts are the REAL ones (42 DiT blocks / 32 text
    /// layers) — names are dimension-independent, so the recorded set
    /// matches the upstream checkpoint exactly while the tensors stay
    /// tiny.
    fn requested_paths() -> Vec<String> {
        let dev = Device::Cpu;
        let dtype = DType::F32;
        let mut all: Vec<String> = Vec::new();

        let dit_cfg = MossDitConfig {
            dim: 64,
            ffn_dim: 128,
            num_heads: 4,
            num_layers: 42,
            text_dim: 96,
            freq_dim: 32,
            latent_channels: 8,
            rope_base: 10_000.0,
        };
        let (vb, rec) = recording_vb(&dev, dtype);
        MossDit::new(vb, dit_cfg).expect("dit builds");
        all.extend(rec.names());

        let mut layer_types = Vec::with_capacity(32);
        for i in 0..32 {
            layer_types.push(if i % 4 == 3 {
                "full_attention".to_string()
            } else {
                "linear_attention".to_string()
            });
        }
        let te_cfg = Qwen35TextConfig {
            hidden_size: 64,
            num_hidden_layers: 32,
            layer_types,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            intermediate_size: 128,
            rms_norm_eps: 1e-6,
            vocab_size: 512,
            rope_theta: 10_000_000.0,
            partial_rotary_factor: 0.25,
            linear_num_key_heads: 2,
            linear_num_value_heads: 4,
            linear_key_head_dim: 8,
            linear_value_head_dim: 8,
            linear_conv_kernel_dim: 4,
        };
        let (vb, rec) = recording_vb(&dev, dtype);
        Qwen35TextModel::new(vb.pp("language_model"), te_cfg).expect("text encoder builds");
        all.extend(rec.names());

        let (vb, rec) = recording_vb(&dev, dtype);
        Flux2Vae::new(vb, Flux2VaeConfig::default()).expect("vae builds");
        all.extend(rec.names());

        all.sort();
        all.dedup();
        all
    }

    /// The port's requested paths, written out for offline diffing.
    ///
    /// The output path is fixed (inside the crate's `target/`) rather
    /// than taken from the environment, so a stray variable cannot
    /// redirect the write outside the workspace.
    #[test]
    #[ignore = "diagnostic dumper — run explicitly"]
    fn dump_requested_weight_names() {
        let all = requested_paths();
        let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("moss-weight-names.txt");
        if let Some(parent) = out.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        std::fs::write(&out, all.join("\n")).expect("write names");
        println!("wrote {} weight names to {}", all.len(), out.display());
    }

    /// **The real check**: diff the port's requested paths against the
    /// upstream checkpoint's own index (1,473 tensors). Needs network
    /// access, hence `#[ignore]`.
    ///
    /// Run with:
    /// `cargo test --lib verify_weight_paths_against_upstream -- --ignored`
    #[tokio::test]
    #[ignore = "needs network access to huggingface.co"]
    async fn verify_weight_paths_against_upstream() {
        let requested = requested_paths();
        assert!(
            requested.len() > 900,
            "the recorder collected suspiciously few paths: {}",
            requested.len()
        );
        let checkpoint = upstream_tensor_names("OpenMOSS-Team/MOSS-Image-Base")
            .await
            .expect("fetch the upstream index");
        println!(
            "upstream tensors: {}, neuron requests: {}",
            checkpoint.len(),
            requested.len()
        );
        let diff = diff(&requested, &checkpoint);
        assert!(
            diff.missing.is_empty(),
            "neuron requests tensors the checkpoint does not have: {:?}",
            diff.missing
        );
        assert!(
            diff.unexpected_unused.is_empty(),
            "checkpoint tensors neuron never reads: {:?}",
            diff.unexpected_unused
        );
    }

    /// **Shape check**: every requested path must also have the shape
    /// the port expects. The recorder captures the requested shape;
    /// the upstream shard headers carry the real one.
    ///
    /// A shape mismatch (a transposed Linear, say) is what a
    /// weight-name check alone cannot see.
    #[tokio::test]
    #[ignore = "needs network access to huggingface.co"]
    async fn verify_weight_shapes_against_upstream() {
        let repo = "OpenMOSS-Team/MOSS-Image-Base";
        let upstream = upstream_shapes(repo).await.expect("fetch the upstream shapes");

        // Build the modules with the REAL config so the requested
        // shapes match the checkpoint's (the tiny-config probe would
        // report tiny shapes).
        let real_dit = crate::imagegen::moss_image::dit::MossDitConfig {
            dim: 3072,
            ffn_dim: 8192,
            num_heads: 24,
            num_layers: 42,
            text_dim: 4096,
            freq_dim: 256,
            latent_channels: 128,
            rope_base: 10_000.0,
        };
        let real_te = crate::imagegen::moss_image::text_encoder::Qwen35TextConfig {
            hidden_size: 4096,
            num_hidden_layers: 32,
            layer_types: {
                let mut v = Vec::with_capacity(32);
                for i in 0..32 {
                    v.push(if i % 4 == 3 {
                        "full_attention".to_string()
                    } else {
                        "linear_attention".to_string()
                    });
                }
                v
            },
            num_attention_heads: 16,
            num_key_value_heads: 4,
            head_dim: 256,
            intermediate_size: 12288,
            rms_norm_eps: 1e-6,
            vocab_size: 248_320,
            rope_theta: 10_000_000.0,
            partial_rotary_factor: 0.25,
            linear_num_key_heads: 16,
            linear_num_value_heads: 32,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
        };
        let real_vae = crate::imagegen::moss_image::vae::Flux2VaeConfig::default();

        let dev = Device::Cpu;
        let dtype = DType::F32;
        let mut entries: Vec<(String, Vec<usize>)> = Vec::new();

        let (vb, rec) = recording_vb(&dev, dtype);
        crate::imagegen::moss_image::dit::MossDit::new(vb, real_dit).expect("dit builds");
        entries.extend(rec.entries());

        let (vb, rec) = recording_vb(&dev, dtype);
        crate::imagegen::moss_image::text_encoder::Qwen35TextModel::new(
            vb.pp("language_model"),
            real_te,
        )
        .expect("text encoder builds");
        entries.extend(rec.entries());

        let (vb, rec) = recording_vb(&dev, dtype);
        crate::imagegen::moss_image::vae::Flux2Vae::new(vb, real_vae).expect("vae builds");
        entries.extend(rec.entries());

        let mut mismatches: Vec<(String, Vec<usize>, Vec<usize>)> = Vec::new();
        for (name, want) in &entries {
            let Some(got) = upstream.get(name) else {
                mismatches.push((name.clone(), want.clone(), Vec::new()));
                continue;
            };
            if got != want {
                mismatches.push((name.clone(), want.clone(), got.clone()));
            }
        }
        assert!(
            mismatches.is_empty(),
            "{} tensors have the wrong shape (name, neuron wants, upstream has): {:?}",
            mismatches.len(),
            &mismatches[..mismatches.len().min(10)]
        );
        println!("{} tensor shapes match the upstream checkpoint", entries.len());
    }

    #[test]
    fn test_repo_id_validation() {
        assert!(is_safe_repo_id("OpenMOSS-Team/MOSS-Image-Base"));
        assert!(is_safe_repo_id("owner/model.v2"));
        assert!(!is_safe_repo_id("owner"));
        assert!(!is_safe_repo_id("owner/name/extra"));
        assert!(!is_safe_repo_id("owner/na me"));
        assert!(!is_safe_repo_id("owner/../../etc/passwd"));
        assert!(!is_safe_repo_id("owner/name?x=1"));
        assert!(!is_safe_repo_id("/owner/name"));
    }

    #[test]
    fn test_diff_classifies_missing_and_unused() {
        let mut checkpoint = BTreeSet::new();
        checkpoint.insert("a.weight".to_string());
        checkpoint.insert("visual.blocks.0.x".to_string());
        checkpoint.insert("rope.bands".to_string());
        let requested = vec!["a.weight".to_string(), "b.weight".to_string()];
        let diff = diff(&requested, &checkpoint);
        assert_eq!(diff.missing, vec!["b.weight".to_string()]);
        assert_eq!(diff.unused.len(), 2, "visual + rope.bands are expected");
        assert!(diff.unexpected_unused.is_empty());
    }

    #[test]
    fn test_diff_flags_unexpected_unused() {
        let mut checkpoint = BTreeSet::new();
        checkpoint.insert("surprise.weight".to_string());
        let diff = diff(&[], &checkpoint);
        assert!(diff.missing.is_empty());
        assert!(diff.unused.is_empty());
        assert_eq!(diff.unexpected_unused, vec!["surprise.weight".to_string()]);
    }
}
