//! Memory profile store — per-model memory reference values plus the
//! runtime correction loop.
//!
//! Layering, lowest to highest precedence:
//!
//! 1. **Embedded reference table** (`assets/model_memory_profiles.json`,
//!   compiled in with `include_str!`) — platform reference values,
//!   derived from parameter count × storage dtype × overhead, and
//!   hand-corrected where real bench numbers exist (see the `source`
//!   field per entry). This is the "提前在不同平台 benchmark 提供的
//!   参考值" that ships with the project.
//! 2. **Deployment overrides** (`<model_dirs[0]>/scheduler/observed_profiles.json`,
//!     or `NEURON_SCHEDULER_PROFILE_PATH`) — persisted observed
//!     values from *this* machine.
//! 3. **Runtime observation** — each time the scheduler evicts a
//!     loaded model it measures the RSS delta and folds it into the
//!     estimate with an EWMA (alpha 0.3), then persists layer 2. The
//!     first load uses the shipped reference; every subsequent cycle
//!     is a little more accurate for the actual runtime environment.
//!
//! When a model id has no entry at all, the estimator falls back to a
//! parameter-count heuristic parsed from the id (e.g. `qwen3-0.6b` →
//! 0.75 B) and finally to a flat default. Scheduling quality degrades
//! gracefully; it never depends on a table being complete.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

/// Where an estimate came from — surfaced in `/v1/scheduler/status`
/// so operators can see which values are still shipped references
/// and which have been corrected by the runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProfileSource {
    /// Measured on real hardware (bench suite or prior observation).
    Bench,
    /// Param-count × dtype × overhead reference value.
    Derived,
    /// Blended with this deployment's observed RSS deltas.
    Observed,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProfileEntry {
    /// Estimated resident bytes per platform key (`cpu` / `cuda` /
    /// `metal`); `"default"` is used when the platform key is absent.
    pub bytes: HashMap<String, u64>,
    pub source: ProfileSource,
    pub params_billion: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProfileEntryFile {
    bytes: HashMap<String, u64>,
    #[serde(default)]
    source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    params_billion: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProfileFile {
    version: u8,
    #[serde(default)]
    platform_bytes_per_param: HashMap<String, f64>,
    #[serde(default)]
    overhead_factor: Option<f64>,
    #[serde(default)]
    fallback_model_bytes: Option<u64>,
    #[serde(default)]
    models: HashMap<String, ProfileEntryFile>,
}

const EMBEDDED_PROFILES: &str = include_str!("../../assets/model_memory_profiles.json");

/// Observation blending factor: new estimate =
/// (1-α)·old + α·observed.
const EWMA_ALPHA: f64 = 0.3;
/// RSS deltas below this are allocator noise, not model weights.
const OBSERVATION_NOISE_FLOOR_BYTES: u64 = 16 * 1024 * 1024;

pub struct MemoryProfileStore {
    inner: Mutex<Inner>,
    overrides_path: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct Inner {
    models: HashMap<String, ProfileEntry>,
    platform_bytes_per_param: HashMap<String, f64>,
    overhead_factor: f64,
    fallback_model_bytes: u64,
}

impl MemoryProfileStore {
    /// Store seeded from the compiled-in reference table.
    pub fn embedded() -> Self {
        let file: ProfileFile = serde_json::from_str(EMBEDDED_PROFILES)
            .expect("embedded model_memory_profiles.json is valid JSON");
        Self {
            inner: Mutex::new(Self::inner_from(&file)),
            overrides_path: None,
        }
    }

    /// Merge a deployment overrides file (layer 2) over the embedded
    /// table. Missing files are fine — first boot on a new machine.
    pub fn with_overrides_path(mut self, path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        if let Ok(content) = std::fs::read_to_string(&path)
            && let Ok(file) = serde_json::from_str::<ProfileFile>(&content)
        {
            let mut inner = self.inner.lock().unwrap();
            for (id, entry) in file.models {
                inner.models.insert(
                    id,
                    ProfileEntry {
                        bytes: entry.bytes,
                        source: parse_source(&entry.source),
                        params_billion: entry.params_billion,
                    },
                );
            }
        }
        self.overrides_path = Some(path);
        self
    }

    fn inner_from(file: &ProfileFile) -> Inner {
        Inner {
            models: file
                .models
                .iter()
                .map(|(id, e)| {
                    (
                        id.clone(),
                        ProfileEntry {
                            bytes: e.bytes.clone(),
                            source: parse_source(&e.source),
                            params_billion: e.params_billion,
                        },
                    )
                })
                .collect(),
            platform_bytes_per_param: file.platform_bytes_per_param.clone(),
            overhead_factor: file.overhead_factor.unwrap_or(1.3),
            fallback_model_bytes: file.fallback_model_bytes.unwrap_or(2_000_000_000),
        }
    }

    /// Estimated resident bytes for `model_id` on `platform`
    /// (e.g. `"cpu"`). Never fails — worst case is the flat default.
    pub fn estimate(&self, model_id: &str, platform: &str) -> u64 {
        let inner = self.inner.lock().unwrap();
        if let Some(entry) = inner.models.get(model_id) {
            if let Some(b) = entry.bytes.get(platform).or_else(|| entry.bytes.get("default")) {
                return *b;
            }
        }
        if let Some(params) = parse_params_billion(model_id) {
            let bpp = inner
                .platform_bytes_per_param
                .get(platform)
                .or_else(|| inner.platform_bytes_per_param.get("default"))
                .copied()
                .unwrap_or(4.0);
            return (params * 1e9 * bpp * inner.overhead_factor) as u64;
        }
        inner.fallback_model_bytes
    }

    /// Fold an observed unload delta into the estimate (EWMA) and
    /// persist the overrides file. `observed_bytes` of 0 means the
    /// probe was unavailable — the call is a no-op then.
    pub fn observe(&self, model_id: &str, platform: &str, observed_bytes: u64) {
        if observed_bytes < OBSERVATION_NOISE_FLOOR_BYTES {
            return;
        }
        let overrides = {
            let mut inner = self.inner.lock().unwrap();
            let entry = inner
                .models
                .entry(model_id.to_string())
                .or_insert_with(|| ProfileEntry {
                    bytes: HashMap::new(),
                    source: ProfileSource::Observed,
                    params_billion: parse_params_billion(model_id),
                });
            let current = entry
                .bytes
                .get(platform)
                .or_else(|| entry.bytes.get("default"))
                .copied();
            let blended = match current {
                Some(prev) => ((1.0 - EWMA_ALPHA) * prev as f64 + EWMA_ALPHA * observed_bytes as f64)
                    as u64,
                None => observed_bytes,
            };
            entry.bytes.insert(platform.to_string(), blended);
            entry.source = ProfileSource::Observed;
            self.overrides_snapshot_locked(&inner)
        };
        self.persist(&overrides);
    }

    /// Snapshot for `/v1/scheduler/status`.
    pub fn snapshot(&self) -> HashMap<String, ProfileEntry> {
        self.inner.lock().unwrap().models.clone()
    }

    /// Serialize only the entries that were corrected by observation,
    /// so the overrides file stays a thin deployment-local delta over
    /// the shipped table.
    fn overrides_snapshot_locked(&self, inner: &Inner) -> String {
        let models: HashMap<String, ProfileEntryFile> = inner
            .models
            .iter()
            .filter(|(_, e)| e.source == ProfileSource::Observed)
            .map(|(id, e)| {
                (
                    id.clone(),
                    ProfileEntryFile {
                        bytes: e.bytes.clone(),
                        source: "observed".to_string(),
                        params_billion: e.params_billion,
                        note: None,
                    },
                )
            })
            .collect();
        let file = ProfileFile {
            version: 1,
            platform_bytes_per_param: inner.platform_bytes_per_param.clone(),
            overhead_factor: Some(inner.overhead_factor),
            fallback_model_bytes: Some(inner.fallback_model_bytes),
            models,
        };
        serde_json::to_string(&file).unwrap_or_else(|_| "{}".to_string())
    }

    fn persist(&self, content: &str) {
        if let Some(path) = &self.overrides_path
            && let Some(parent) = path.parent()
        {
            let _ = std::fs::create_dir_all(parent);
            let _ = std::fs::write(path, content);
        }
    }
}

fn parse_source(s: &str) -> ProfileSource {
    match s {
        "observed" => ProfileSource::Observed,
        "bench" => ProfileSource::Bench,
        _ => ProfileSource::Derived,
    }
}

/// Parse a parameter count from a model id: the last `-`/`_`/`/`
/// separated token shaped like `<number>b` (e.g. `qwen3-0.6b` → 0.6,
/// `minicpm5-2b` → 2.0, `Qwen/Qwen3-0.6B` → 0.6). Returns None when
/// no token matches.
pub fn parse_params_billion(model_id: &str) -> Option<f64> {
    let lower = model_id.to_lowercase();
    lower
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '.')
        .filter_map(|t| {
            let stem = t.strip_suffix('b')?;
            if stem.is_empty() || !stem.chars().all(|c| c.is_ascii_digit() || c == '.') {
                return None;
            }
            stem.parse::<f64>().ok().filter(|v| *v > 0.0 && *v < 1000.0)
        })
        .last()
}

/// Default overrides location: `<first model dir>/scheduler/observed_profiles.json`.
pub fn default_overrides_path(model_dirs: &[PathBuf]) -> Option<PathBuf> {
    model_dirs
        .first()
        .map(|d| d.join("scheduler").join("observed_profiles.json"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedded_profiles_parse_and_cover_known_ids() {
        let store = MemoryProfileStore::embedded();
        // Core catalog ids must resolve to the explicit table entries.
        let est = store.estimate("minicpm5-1b", "cpu");
        assert_eq!(est, 5_200_000_000);
        let est = store.estimate("minicpm5-2b", "cuda");
        assert_eq!(est, 6_550_000_000);
        // voxcpm2's ~10 GB budget is load-bearing (see AGENTS.md).
        assert_eq!(store.estimate("voxcpm2", "cpu"), 10_500_000_000);
    }

    #[test]
    fn test_estimate_falls_back_to_param_heuristic() {
        let store = MemoryProfileStore::embedded();
        // No table entry → parsed from the id.
        let est = store.estimate("some-vendor/qwen2-1b", "cpu");
        let expected = (1.0 * 1e9 * 4.0 * 1.3) as u64;
        assert_eq!(est, expected);
        // No param pattern either → flat default.
        assert_eq!(store.estimate("mystery-model", "cpu"), 2_000_000_000);
    }

    #[test]
    fn test_parse_params_billion_token_shapes() {
        assert_eq!(parse_params_billion("qwen3-0.6b"), Some(0.6));
        assert_eq!(parse_params_billion("minicpm5-2b"), Some(2.0));
        assert_eq!(parse_params_billion("Qwen/Qwen3-0.6B"), Some(0.6));
        assert_eq!(parse_params_billion("llama-3.2-3b"), Some(3.0));
        assert_eq!(parse_params_billion("voxcpm2"), None);
        assert_eq!(parse_params_billion("chatglm-3-6b"), Some(6.0));
    }

    #[test]
    fn test_observe_blends_and_persists() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("scheduler").join("observed.json");
        let store =
            MemoryProfileStore::embedded().with_overrides_path(&path);

        // First observation seeds the estimate directly.
        store.observe("some-unknown-7b-model", "cpu", 30_000_000_000);
        assert_eq!(store.estimate("some-unknown-7b-model", "cpu"), 30_000_000_000);

        // Second observation blends: 0.7*30G + 0.3*20G = 27G.
        store.observe("some-unknown-7b-model", "cpu", 20_000_000_000);
        assert_eq!(store.estimate("some-unknown-7b-model", "cpu"), 27_000_000_000);

        // Sub-noise observations are ignored.
        store.observe("some-unknown-7b-model", "cpu", 1_000);
        assert_eq!(store.estimate("some-unknown-7b-model", "cpu"), 27_000_000_000);

        // Overrides file exists and carries the observed entry.
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("some-unknown-7b-model"));
        assert!(content.contains("\"observed\""));
    }

    #[test]
    fn test_overrides_file_layers_over_embedded() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("observed.json");
        std::fs::write(
            &path,
            r#"{"version":1,"models":{"minicpm5-1b":{"bytes":{"cpu":123456789},"source":"observed"}}}"#,
        )
        .unwrap();
        let store = MemoryProfileStore::embedded().with_overrides_path(&path);
        assert_eq!(store.estimate("minicpm5-1b", "cpu"), 123_456_789);
        // Untouched ids keep the shipped reference.
        assert_eq!(store.estimate("voxcpm2", "cpu"), 10_500_000_000);
    }

    #[test]
    fn test_default_overrides_path_under_first_model_dir() {
        let dirs = vec![PathBuf::from("/models")];
        let p = default_overrides_path(&dirs).unwrap();
        assert_eq!(
            p,
            PathBuf::from("/models").join("scheduler").join("observed_profiles.json")
        );
        assert!(default_overrides_path(&[]).is_none());
    }
}
