use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::models::EngineType;

/// A pre-registered model entry. `capability` selects the target
/// registry; `engine` is the *impl id* (e.g. `"candle"`, `"echo"`,
/// `"burn"`), parsed by that capability's `*Impl::parse`. Serialized
/// configs without `capability` default to `Llm` for backward compat.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRegistration {
    pub model_id: String,
    /// The impl id (`"candle"` / `"echo"` / `"burn"`), NOT the
    /// capability — see `capability` for the target registry.
    pub engine: String,
    /// Target registry (LLM / TTS / ASR / Rerank / Embedding).
    #[serde(default)]
    pub capability: EngineType,
    pub device: DeviceSelection,
    pub required_memory_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeureConfig {
    pub port: u16,
    pub host: String,
    pub llm_model_path: Option<String>,
    pub tts_model_path: Option<String>,
    pub asr_model_path: Option<String>,
    pub rerank_model_path: Option<String>,
    pub device: DeviceSelection,
    pub default_llm_model: Option<String>,
    pub default_tts_model: Option<String>,
    pub default_asr_model: Option<String>,
    pub default_rerank_model: Option<String>,
    /// One or more filesystem roots scanned for downloaded model
    /// weights. Models are searched in order; the first directory
    /// that contains a `{engine}/{id}/` subdirectory wins. Defaults
    /// to a single entry `$HOME/.neure/models/`.
    pub model_dirs: Vec<PathBuf>,
    /// Source id used when no per-engine or per-model override is
    /// set. Defaults to `"modelscope"`.
    pub default_source_id: String,
    /// Engine-level source override: pull requests for this engine
    /// use the registered source under this id instead of the
    /// default source.
    pub per_engine_source: HashMap<EngineType, String>,
    /// Model-level source override. Key is `"engine/id"` (e.g.
    /// `"llm/qwen2.5-0.5b"`); value is the source id.
    pub per_model_source: HashMap<String, String>,
    /// Per-source base URL / endpoint. Source implementations may
    /// use this to redirect downloads (e.g. HuggingFaceSource
    /// passes it through to `HF_ENDPOINT`).
    pub source_endpoints: HashMap<String, String>,
    /// Pre-registered models for the runtime registries.
    pub registrations: Vec<ModelRegistration>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum DeviceSelection {
    #[default]
    Auto,
    Cpu,
    Nvidia,
    Apple,
    Vulkan,
}

impl NeureConfig {
    pub fn default_port() -> u16 {
        8083
    }

    pub fn default_host() -> String {
        "0.0.0.0".to_string()
    }

    /// Default model directory: `$HOME/.neure/models/`.
    pub fn default_model_dir() -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(home).join(".neure").join("models")
    }

    /// Default candle device for inference. Returns CPU since
    /// the chronos2 WIP path uses this and candle's CPU device
    /// is always available; explicit CUDA / Metal selection
    /// happens at runtime once the model is loaded.
    pub fn default_device() -> candle_core::Device {
        candle_core::Device::Cpu
    }

    pub fn new() -> Self {
        Self {
            port: Self::default_port(),
            host: Self::default_host(),
            llm_model_path: None,
            tts_model_path: None,
            asr_model_path: None,
            rerank_model_path: None,
            device: DeviceSelection::Auto,
            default_llm_model: None,
            default_tts_model: None,
            default_asr_model: None,
            default_rerank_model: None,
            model_dirs: vec![Self::default_model_dir()],
            default_source_id: "modelscope".to_string(),
            per_engine_source: HashMap::new(),
            per_model_source: HashMap::new(),
            source_endpoints: HashMap::new(),
            registrations: Vec::new(),
        }
    }

    pub fn with_port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    pub fn with_host(mut self, host: impl Into<String>) -> Self {
        self.host = host.into();
        self
    }

    pub fn with_llm_model(mut self, model: impl Into<String>) -> Self {
        self.default_llm_model = Some(model.into());
        self
    }

    pub fn with_tts_model(mut self, model: impl Into<String>) -> Self {
        self.default_tts_model = Some(model.into());
        self
    }

    pub fn with_asr_model(mut self, model: impl Into<String>) -> Self {
        self.default_asr_model = Some(model.into());
        self
    }

    pub fn with_rerank_model(mut self, model: impl Into<String>) -> Self {
        self.default_rerank_model = Some(model.into());
        self
    }

    pub fn with_device(mut self, device: DeviceSelection) -> Self {
        self.device = device;
        self
    }

    /// Append a model directory to the search list. Multiple calls
    /// accumulate.
    pub fn with_model_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.model_dirs.push(path.into());
        self
    }

    /// Set the default source id (used when no per-engine or
    /// per-model override is set).
    pub fn with_default_source(mut self, source_id: impl Into<String>) -> Self {
        self.default_source_id = source_id.into();
        self
    }

    /// Override the source for all models of a given engine.
    pub fn with_engine_source(mut self, engine: EngineType, source_id: impl Into<String>) -> Self {
        self.per_engine_source.insert(engine, source_id.into());
        self
    }

    /// Override the source for a specific `(engine, model_id)` pair.
    /// The id is stored as a `"engine/id"` key.
    pub fn with_model_source(
        mut self,
        engine: EngineType,
        model_id: impl Into<String>,
        source_id: impl Into<String>,
    ) -> Self {
        let key = format!("{}/{}", engine.as_str(), model_id.into());
        self.per_model_source.insert(key, source_id.into());
        self
    }

    /// Set a base URL / endpoint for a registered source.
    pub fn with_source_endpoint(
        mut self,
        source_id: impl Into<String>,
        endpoint: impl Into<String>,
    ) -> Self {
        self.source_endpoints
            .insert(source_id.into(), endpoint.into());
        self
    }

    pub fn with_registration(
        self,
        model_id: impl Into<String>,
        engine: impl Into<String>,
        device: DeviceSelection,
        required_memory_bytes: u64,
    ) -> Self {
        self.with_registration_for_capability(
            EngineType::Llm,
            model_id,
            engine,
            device,
            required_memory_bytes,
        )
    }

    /// Register a model under a specific capability. Use this when
    /// registering TTS / ASR / Rerank / Embedding models (the
    /// convenience `with_registration` is LLM-only for backward
    /// compat).
    pub fn with_registration_for_capability(
        mut self,
        capability: EngineType,
        model_id: impl Into<String>,
        engine: impl Into<String>,
        device: DeviceSelection,
        required_memory_bytes: u64,
    ) -> Self {
        self.registrations.push(ModelRegistration {
            model_id: model_id.into(),
            engine: engine.into(),
            capability,
            device,
            required_memory_bytes,
        });
        self
    }

    pub fn with_llm_registration(
        self,
        model_id: impl Into<String>,
        engine: impl Into<String>,
        device: DeviceSelection,
    ) -> Self {
        self.with_registration_for_capability(
            EngineType::Llm,
            model_id,
            engine,
            device,
            2_000_000_000,
        )
    }

    pub fn with_tts_registration(
        self,
        model_id: impl Into<String>,
        engine: impl Into<String>,
        device: DeviceSelection,
    ) -> Self {
        self.with_registration_for_capability(
            EngineType::Tts,
            model_id,
            engine,
            device,
            2_000_000_000,
        )
    }

    pub fn with_asr_registration(
        self,
        model_id: impl Into<String>,
        engine: impl Into<String>,
        device: DeviceSelection,
    ) -> Self {
        self.with_registration_for_capability(
            EngineType::Asr,
            model_id,
            engine,
            device,
            1_500_000_000,
        )
    }

    pub fn with_rerank_registration(
        self,
        model_id: impl Into<String>,
        engine: impl Into<String>,
        device: DeviceSelection,
    ) -> Self {
        self.with_registration_for_capability(
            EngineType::Rerank,
            model_id,
            engine,
            device,
            500_000_000,
        )
    }

    pub fn with_embedding_registration(
        self,
        model_id: impl Into<String>,
        engine: impl Into<String>,
        device: DeviceSelection,
    ) -> Self {
        self.with_registration_for_capability(
            EngineType::Embedding,
            model_id,
            engine,
            device,
            200_000_000,
        )
    }

    /// Build a config from process environment. See the spec for the
    /// full env-var table.
    pub fn from_env() -> Self {
        Self::from_env_map(&std::env::vars().collect())
    }

    /// Build a config from a custom env map. Used by tests so they
    /// don't have to mutate real process env.
    pub fn from_env_map(env: &HashMap<String, String>) -> Self {
        let mut cfg = Self::new();
        apply_port_env(&mut cfg, env);
        apply_host_env(&mut cfg, env);
        apply_model_dirs_env(&mut cfg, env);
        apply_default_source_env(&mut cfg, env);
        apply_per_engine_sources_env(&mut cfg, env);
        apply_per_model_sources_env(&mut cfg, env);
        apply_source_endpoints_env(&mut cfg, env);
        cfg
    }

    /// Cross-field validation. Currently a no-op returning Ok; future
    /// checks (e.g. "all model_dirs exist on disk", "default_source_id
    /// is in SourceRegistry") plug in here without changing the
    /// `from_env_map` contract.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        Ok(())
    }
}

fn apply_port_env(cfg: &mut NeureConfig, env: &HashMap<String, String>) {
    // NEURE_PORT: parses as u16, invalid values fall back to the
    // default rather than panicking — env var typos shouldn't
    // crash startup.
    if let Some(v) = env.get("NEURE_PORT")
        && let Ok(port) = v.parse::<u16>() {
            cfg.port = port;
        }
}

fn apply_host_env(cfg: &mut NeureConfig, env: &HashMap<String, String>) {
    // NEURE_HOST: any non-empty string wins. Empty values fall
    // through to the default (0.0.0.0) so a stray `NEURE_HOST=`
    // shell export doesn't break bind.
    if let Some(v) = env.get("NEURE_HOST")
        && !v.is_empty() {
            cfg.host = v.clone();
        }
}

fn apply_model_dirs_env(cfg: &mut NeureConfig, env: &HashMap<String, String>) {
    // NEURE_MODEL_DIRS (multi-value, colon-separated) takes
    // priority over NEURE_MODEL_DIR (single-value legacy).
    if let Some(dirs) = env.get("NEURE_MODEL_DIRS") {
        cfg.model_dirs = dirs
            .split(':')
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
            .collect();
    } else if let Some(dir) = env.get("NEURE_MODEL_DIR") {
        cfg.model_dirs = vec![PathBuf::from(dir)];
    }
}

fn apply_default_source_env(cfg: &mut NeureConfig, env: &HashMap<String, String>) {
    if let Some(v) = env.get("NEURE_DEFAULT_SOURCE") {
        cfg.default_source_id = v.clone();
    }
}

fn apply_per_engine_sources_env(cfg: &mut NeureConfig, env: &HashMap<String, String>) {
    let engine_map: &[(EngineType, &str)] = &[
        (EngineType::Llm, "NEURE_LLM_SOURCE"),
        (EngineType::Tts, "NEURE_TTS_SOURCE"),
        (EngineType::Asr, "NEURE_ASR_SOURCE"),
        (EngineType::Rerank, "NEURE_RERANK_SOURCE"),
    ];
    for (engine, var) in engine_map {
        if let Some(v) = env.get(*var) {
            cfg.per_engine_source.insert(*engine, v.clone());
        }
    }
}

fn apply_per_model_sources_env(cfg: &mut NeureConfig, env: &HashMap<String, String>) {
    // Per-model: NEURE_MODEL_SOURCE_<ENGINE>_<ID_UPPER>.
    // For example, NEURE_MODEL_SOURCE_LLM_QWEN2_5_0_5B=qwen2-5-0-5b → "llm/qwen2-5-0-5b"
    // (We keep underscores as hyphens in the model_id portion.
    // A trailing dot-separated version like `qwen2.5-0.5b` is not
    // representable in env-var form; users can call `with_model_source`
    // directly for those.)
    for (k, v) in env {
        if let Some(rest) = k.strip_prefix("NEURE_MODEL_SOURCE_") {
            let lower = rest.to_lowercase();
            if let Some((engine, model_id)) = lower.split_once('_') {
                let engine_typed = match engine {
                    "llm" => Some(EngineType::Llm),
                    "tts" => Some(EngineType::Tts),
                    "asr" => Some(EngineType::Asr),
                    "rerank" => Some(EngineType::Rerank),
                    _ => None,
                };
                if let Some(e) = engine_typed {
                    let normalized = model_id.replace('_', "-");
                    cfg.per_model_source
                        .insert(format!("{}/{}", e.as_str(), normalized), v.clone());
                }
            }
        }
    }
}

fn apply_source_endpoints_env(cfg: &mut NeureConfig, env: &HashMap<String, String>) {
    // Source endpoints: NEURE_SOURCE_<ID>_ENDPOINT.
    for (k, v) in env {
        if let Some(rest) = k.strip_prefix("NEURE_SOURCE_")
            && let Some(source_id) = rest.strip_suffix("_ENDPOINT") {
                let id_lower = source_id.to_lowercase();
                cfg.source_endpoints.insert(id_lower, v.clone());
            }
    }
}

impl Default for NeureConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Validate that a path exists and is a directory. Used by all 5
/// model types' runtime loaders to dedupe the "is this path even
/// usable?" check that follows every `std::env::var(...).map(...)`.
///
/// Errors (the `label` is the env-var name or runtime description,
/// so the error reads naturally regardless of who calls this):
///   * path missing → `"{label} does not exist: …"`
///   * path is a file → `"{label} is not a directory: …"`
///
/// This deliberately does NOT read the env var. Each caller reads
/// its own env var (so it can produce a runtime-specific error when
/// the var is unset, e.g. "Download a Qwen model from HuggingFace")
/// and then calls `ensure_dir` on the resulting `PathBuf`.
pub fn ensure_dir(path: &std::path::Path, label: &str) -> Result<(), String> {
    if !path.exists() {
        return Err(format!("{label} does not exist: {}", path.display()));
    }
    if !path.is_dir() {
        return Err(format!("{label} is not a directory: {}", path.display()));
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct ResourceTracker {
    total_cpu_bytes: u64,
    total_gpu_bytes: u64,
    used_cpu_bytes: u64,
    used_gpu_bytes: u64,
}

impl Default for ResourceTracker {
    fn default() -> Self {
        Self {
            total_cpu_bytes: 16_000_000_000,
            total_gpu_bytes: 8_000_000_000,
            used_cpu_bytes: 0,
            used_gpu_bytes: 0,
        }
    }
}

impl ResourceTracker {
    pub fn new(total_cpu_bytes: u64, total_gpu_bytes: u64) -> Self {
        Self {
            total_cpu_bytes,
            total_gpu_bytes,
            used_cpu_bytes: 0,
            used_gpu_bytes: 0,
        }
    }

    pub fn can_load(&self, required: u64, device: &DeviceSelection) -> bool {
        match device {
            DeviceSelection::Cpu | DeviceSelection::Auto => {
                self.total_cpu_bytes.saturating_sub(self.used_cpu_bytes) >= required
            }
            DeviceSelection::Nvidia
            | DeviceSelection::Apple
            | DeviceSelection::Vulkan => {
                self.total_gpu_bytes.saturating_sub(self.used_gpu_bytes) >= required
            }
        }
    }

    pub fn allocate(&mut self, required: u64, device: &DeviceSelection) -> Result<(), String> {
        if !self.can_load(required, device) {
            let (avail, total) = match device {
                DeviceSelection::Cpu | DeviceSelection::Auto => {
                    (self.total_cpu_bytes - self.used_cpu_bytes, self.total_cpu_bytes)
                }
                _ => (self.total_gpu_bytes - self.used_gpu_bytes, self.total_gpu_bytes),
            };
            return Err(format!(
                "insufficient memory: required={required}, available={avail}, total={total}"
            ));
        }
        match device {
            DeviceSelection::Cpu | DeviceSelection::Auto => self.used_cpu_bytes += required,
            _ => self.used_gpu_bytes += required,
        }
        Ok(())
    }

    pub fn release(&mut self, required: u64, device: &DeviceSelection) {
        match device {
            DeviceSelection::Cpu | DeviceSelection::Auto => {
                self.used_cpu_bytes = self.used_cpu_bytes.saturating_sub(required);
            }
            _ => {
                self.used_gpu_bytes = self.used_gpu_bytes.saturating_sub(required);
            }
        }
    }

    pub fn available_cpu_bytes(&self) -> u64 {
        self.total_cpu_bytes.saturating_sub(self.used_cpu_bytes)
    }

    pub fn available_gpu_bytes(&self) -> u64 {
        self.total_gpu_bytes.saturating_sub(self.used_gpu_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_neure_config_default_port_8083() {
        let config = NeureConfig::default();
        assert_eq!(config.port, 8083);
    }

    #[test]
    fn test_neure_config_default_host_0_0_0_0() {
        let config = NeureConfig::default();
        assert_eq!(config.host, "0.0.0.0");
    }

    #[test]
    fn test_neure_config_default_device_auto() {
        let config = NeureConfig::default();
        assert_eq!(config.device, DeviceSelection::Auto);
    }

    #[test]
    fn test_neure_config_serialize_roundtrip() {
        let config = NeureConfig {
            port: 9000,
            host: "127.0.0.1".to_string(),
            llm_model_path: Some("/models/llm".to_string()),
            tts_model_path: None,
            asr_model_path: None,
            rerank_model_path: Some("/models/rerank".to_string()),
            device: DeviceSelection::Nvidia,
            default_llm_model: Some("qwen3".to_string()),
            default_tts_model: None,
            default_asr_model: None,
            default_rerank_model: Some("bge-reranker-base".to_string()),
            model_dirs: vec![PathBuf::from("/m1"), PathBuf::from("/m2")],
            default_source_id: "modelscope".to_string(),
            per_engine_source: HashMap::new(),
            per_model_source: HashMap::new(),
            source_endpoints: HashMap::new(),
            registrations: Vec::new(),
        };

        let json = serde_json::to_string(&config).unwrap();
        let parsed: NeureConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.port, 9000);
        assert_eq!(parsed.host, "127.0.0.1");
        assert_eq!(parsed.device, DeviceSelection::Nvidia);
        assert_eq!(parsed.rerank_model_path.as_deref(), Some("/models/rerank"));
        assert_eq!(
            parsed.default_rerank_model.as_deref(),
            Some("bge-reranker-base")
        );
        assert_eq!(parsed.model_dirs.len(), 2);
    }

    #[test]
    fn test_neure_config_builder_pattern() {
        let config = NeureConfig::new()
            .with_port(9090)
            .with_device(DeviceSelection::Apple)
            .with_llm_model("qwen3-0.6b");

        assert_eq!(config.port, 9090);
        assert_eq!(config.device, DeviceSelection::Apple);
        assert_eq!(config.default_llm_model.as_deref(), Some("qwen3-0.6b"));
    }

    #[test]
    fn test_neure_config_with_tts_asr_rerank_model_builders() {
        let config = NeureConfig::new()
            .with_tts_model("voxcpm-0.5b")
            .with_asr_model("whisper-base")
            .with_rerank_model("bge-reranker-base");

        assert_eq!(config.default_tts_model.as_deref(), Some("voxcpm-0.5b"));
        assert_eq!(config.default_asr_model.as_deref(), Some("whisper-base"));
        assert_eq!(
            config.default_rerank_model.as_deref(),
            Some("bge-reranker-base")
        );
        assert_eq!(config.default_llm_model, None);
    }

    #[test]
    fn test_neure_config_default_validate_returns_ok() {
        let c = NeureConfig::new();
        assert!(c.validate().is_ok());
    }

    #[test]
    fn test_neure_config_validate_after_from_env_map_returns_ok() {
        let mut env = HashMap::new();
        env.insert("NEURE_PORT".to_string(), "9999".to_string());
        env.insert("NEURE_HOST".to_string(), "127.0.0.1".to_string());
        let c = NeureConfig::from_env_map(&env);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn test_neure_config_with_model_builders_accept_string_and_str() {
        let owned = String::from("qwen3-0.6b");
        let config = NeureConfig::new()
            .with_llm_model("qwen3-0.6b")
            .with_tts_model(String::from("voxcpm-0.5b"))
            .with_asr_model(owned.as_str())
            .with_rerank_model("bge-reranker-base".to_string());
        assert_eq!(config.default_llm_model.as_deref(), Some("qwen3-0.6b"));
        assert_eq!(config.default_tts_model.as_deref(), Some("voxcpm-0.5b"));
        assert_eq!(config.default_asr_model.as_deref(), Some("qwen3-0.6b"));
        assert_eq!(
            config.default_rerank_model.as_deref(),
            Some("bge-reranker-base")
        );
    }

    // -- new: multi-dir + source override --

    #[test]
    fn test_with_model_dir_appends_to_vec() {
        let c = NeureConfig::new()
            .with_model_dir("/opt/models")
            .with_model_dir("/home/user/models");
        assert_eq!(c.model_dirs.len(), 3); // 1 default + 2 added
        assert_eq!(c.model_dirs[1].to_str().unwrap(), "/opt/models");
        assert_eq!(c.model_dirs[2].to_str().unwrap(), "/home/user/models");
    }

    #[test]
    fn test_with_default_source_overrides() {
        let c = NeureConfig::new().with_default_source("modelscope");
        assert_eq!(c.default_source_id, "modelscope");
    }

    #[test]
    fn test_with_engine_source_adds_to_map() {
        let c = NeureConfig::new()
            .with_engine_source(EngineType::Llm, "modelscope")
            .with_engine_source(EngineType::Tts, "huggingface");
        assert_eq!(
            c.per_engine_source.get(&EngineType::Llm).unwrap(),
            "modelscope"
        );
        assert_eq!(
            c.per_engine_source.get(&EngineType::Tts).unwrap(),
            "huggingface"
        );
        assert!(c.per_engine_source.get(&EngineType::Asr).is_none());
    }

    #[test]
    fn test_with_model_source_uses_engine_id_key() {
        let c =
            NeureConfig::new().with_model_source(EngineType::Llm, "qwen2.5-0.5b", "private-hub");
        assert_eq!(
            c.per_model_source.get("llm/qwen2.5-0.5b").unwrap(),
            "private-hub"
        );
    }

    #[test]
    fn test_with_source_endpoint_stores_in_map() {
        let c = NeureConfig::new().with_source_endpoint("huggingface", "https://hf-mirror.com");
        assert_eq!(
            c.source_endpoints.get("huggingface").unwrap(),
            "https://hf-mirror.com"
        );
    }

    #[test]
    fn test_from_env_parses_neure_model_dirs_multi_value() {
        let mut env = HashMap::new();
        env.insert("NEURE_MODEL_DIRS".to_string(), "/a:/b:/c".to_string());
        let c = NeureConfig::from_env_map(&env);
        assert_eq!(c.model_dirs.len(), 3);
        assert_eq!(c.model_dirs[0].to_str().unwrap(), "/a");
        assert_eq!(c.model_dirs[2].to_str().unwrap(), "/c");
    }

    #[test]
    fn test_from_env_parses_neure_model_dir_legacy_single() {
        let mut env = HashMap::new();
        env.insert("NEURE_MODEL_DIR".to_string(), "/single".to_string());
        let c = NeureConfig::from_env_map(&env);
        assert_eq!(c.model_dirs.len(), 1);
        assert_eq!(c.model_dirs[0].to_str().unwrap(), "/single");
    }

    #[test]
    fn test_from_env_model_dirs_takes_priority_over_legacy_model_dir() {
        let mut env = HashMap::new();
        env.insert("NEURE_MODEL_DIRS".to_string(), "/multi".to_string());
        env.insert("NEURE_MODEL_DIR".to_string(), "/single".to_string());
        let c = NeureConfig::from_env_map(&env);
        assert_eq!(c.model_dirs.len(), 1);
        assert_eq!(c.model_dirs[0].to_str().unwrap(), "/multi");
    }

    #[test]
    fn test_from_env_parses_per_engine_sources() {
        let mut env = HashMap::new();
        env.insert("NEURE_LLM_SOURCE".to_string(), "modelscope".to_string());
        env.insert("NEURE_TTS_SOURCE".to_string(), "private-tts".to_string());
        let c = NeureConfig::from_env_map(&env);
        assert_eq!(
            c.per_engine_source.get(&EngineType::Llm).unwrap(),
            "modelscope"
        );
        assert_eq!(
            c.per_engine_source.get(&EngineType::Tts).unwrap(),
            "private-tts"
        );
        assert!(c.per_engine_source.get(&EngineType::Asr).is_none());
    }

    #[test]
    fn test_from_env_parses_source_endpoint() {
        let mut env = HashMap::new();
        env.insert(
            "NEURE_SOURCE_HUGGINGFACE_ENDPOINT".to_string(),
            "https://hf-mirror.com".to_string(),
        );
        let c = NeureConfig::from_env_map(&env);
        assert_eq!(
            c.source_endpoints.get("huggingface").unwrap(),
            "https://hf-mirror.com"
        );
    }

    #[test]
    fn test_from_env_parses_neure_port() {
        let mut env = HashMap::new();
        env.insert("NEURE_PORT".to_string(), "9999".to_string());
        let c = NeureConfig::from_env_map(&env);
        assert_eq!(c.port, 9999);
    }

    #[test]
    fn test_from_env_parses_neure_port_invalid_value_keeps_default() {
        let mut env = HashMap::new();
        env.insert("NEURE_PORT".to_string(), "not-a-number".to_string());
        let c = NeureConfig::from_env_map(&env);
        assert_eq!(c.port, NeureConfig::default_port());
    }

    #[test]
    fn test_from_env_parses_neure_host() {
        let mut env = HashMap::new();
        env.insert("NEURE_HOST".to_string(), "127.0.0.1".to_string());
        let c = NeureConfig::from_env_map(&env);
        assert_eq!(c.host, "127.0.0.1");
    }

    #[test]
    fn test_from_env_parses_neure_host_empty_string_keeps_default() {
        let mut env = HashMap::new();
        env.insert("NEURE_HOST".to_string(), "".to_string());
        let c = NeureConfig::from_env_map(&env);
        assert_eq!(c.host, NeureConfig::default_host());
    }

    #[test]
    fn test_default_model_dir_ends_with_neure_models() {
        let c = NeureConfig::new();
        assert!(c.model_dirs[0].ends_with(".neure/models"));
    }

    #[test]
    fn test_ensure_dir_nonexistent_path_returns_does_not_exist_error() {
        let dir = std::env::temp_dir().join("neure-test-does-not-exist-12345");
        let result = ensure_dir(&dir, "NEURE_TEST_RESOLVE_VAR");
        let err = result.err().unwrap();
        assert!(err.contains("does not exist"));
        assert!(err.contains("NEURE_TEST_RESOLVE_VAR"));
    }

    #[test]
    fn test_ensure_dir_file_not_directory_returns_not_a_directory_error() {
        let path = std::env::temp_dir().join("neure-test-not-a-dir.txt");
        std::fs::write(&path, b"not a dir").unwrap();
        let result = ensure_dir(&path, "NEURE_TEST_RESOLVE_VAR");
        std::fs::remove_file(&path).ok();
        let err = result.err().unwrap();
        assert!(err.contains("is not a directory"));
    }

    #[test]
    fn test_ensure_dir_existing_dir_returns_ok() {
        let dir = std::env::temp_dir();
        let result = ensure_dir(&dir, "NEURE_TEST_RESOLVE_VAR");
        assert!(result.is_ok());
    }

    #[test]
    fn test_resource_tracker_defaults() {
        let rt = ResourceTracker::default();
        assert_eq!(rt.available_cpu_bytes(), 16_000_000_000);
        assert_eq!(rt.available_gpu_bytes(), 8_000_000_000);
    }

    #[test]
    fn test_resource_tracker_can_load_returns_true_when_enough_memory() {
        let rt = ResourceTracker::default();
        assert!(rt.can_load(1_000_000_000, &DeviceSelection::Cpu));
    }

    #[test]
    fn test_resource_tracker_can_load_returns_false_when_over_capacity() {
        let rt = ResourceTracker::new(100, 100);
        assert!(!rt.can_load(200, &DeviceSelection::Cpu));
    }

    #[test]
    fn test_resource_tracker_allocate_deducts_used_memory() {
        let mut rt = ResourceTracker::new(1000, 1000);
        rt.allocate(500, &DeviceSelection::Cpu).unwrap();
        assert_eq!(rt.available_cpu_bytes(), 500);
    }

    #[test]
    fn test_resource_tracker_allocate_fails_when_over_capacity() {
        let mut rt = ResourceTracker::new(100, 100);
        assert!(rt.allocate(200, &DeviceSelection::Cpu).is_err());
    }

    #[test]
    fn test_resource_tracker_release_returns_memory() {
        let mut rt = ResourceTracker::new(1000, 1000);
        rt.allocate(500, &DeviceSelection::Cpu).unwrap();
        rt.release(300, &DeviceSelection::Cpu);
        assert_eq!(rt.available_cpu_bytes(), 800);
    }
}
