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
pub struct NeuronConfig {
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
    /// to a single entry `$HOME/.neuron/models/`.
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
    /// Model scheduler knobs (queue, memory budget, idle recycling).
    /// Serde-defaulted so pre-scheduler serialized configs load
    /// unchanged.
    #[serde(default)]
    pub scheduler: crate::scheduler::SchedulerConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum DeviceSelection {
    #[default]
    Auto,
    Cpu,
    /// NVIDIA GPU. `NEURON_DEVICE` accepts both `nvidia` and the `cuda`
    /// alias (the scheduler's platform key for this device is `cuda`).
    #[serde(alias = "cuda")]
    Nvidia,
    Apple,
    Vulkan,
}

impl NeuronConfig {
    pub fn default_port() -> u16 {
        8083
    }

    pub fn default_host() -> String {
        "0.0.0.0".to_string()
    }

    /// Default model directory: `$HOME/.neuron/models/`.
    ///
    /// Deliberately does not touch the filesystem, so callers that want
    /// a stable path (tests, path arithmetic) get one. Use
    /// [`Self::resolved_model_dir`] when bootstrapping a config.
    ///
    /// `HOME` is not set for GUI-launched processes on Windows, so fall
    /// back to `USERPROFILE` before giving up and returning a
    /// cwd-relative path (which would silently strand the model library).
    pub fn default_model_dir() -> PathBuf {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_else(|_| ".".to_string());
        PathBuf::from(home).join(".neuron").join("models")
    }

    /// [`Self::default_model_dir`] plus a migration fallback: when the
    /// new path is absent but the pre-rename `$HOME/.neure/models/`
    /// exists, the old path is returned so an already-downloaded model
    /// library survives the rename.
    pub fn resolved_model_dir() -> PathBuf {
        let current = Self::default_model_dir();
        if current.is_dir() {
            return current;
        }
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_else(|_| ".".to_string());
        let legacy = PathBuf::from(home).join(".neure").join("models");
        if legacy.is_dir() {
            tracing::warn!(
                legacy = %legacy.display(),
                replacement = %current.display(),
                "using the pre-rename model directory; move it to the new path to silence this",
            );
            return legacy;
        }
        current
    }

    /// Default candle device for inference. Returns CPU since
    /// the chronos2 WIP path uses this and candle's CPU device
    /// is always available; explicit CUDA / Metal selection
    /// happens at runtime once the model is loaded.
    pub fn default_device() -> candle::Device {
        candle::Device::Cpu
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
            model_dirs: vec![Self::resolved_model_dir()],
            default_source_id: "modelscope".to_string(),
            per_engine_source: HashMap::new(),
            per_model_source: HashMap::new(),
            source_endpoints: HashMap::new(),
            registrations: Vec::new(),
            scheduler: crate::scheduler::SchedulerConfig::default(),
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
        apply_device_env(&mut cfg, env);
        apply_model_dirs_env(&mut cfg, env);
        apply_default_source_env(&mut cfg, env);
        apply_per_engine_sources_env(&mut cfg, env);
        apply_per_model_sources_env(&mut cfg, env);
        apply_source_endpoints_env(&mut cfg, env);
        apply_scheduler_env(&mut cfg, env);
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

/// Current env-var prefix. Every public knob is read as
/// `NEURON_<SUFFIX>`.
const ENV_PREFIX: &str = "NEURON_";

/// Prefix used before the neure → neuron rename. Still honored so
/// deployed configs keep working; every fallback hit logs a warning.
const LEGACY_ENV_PREFIX: &str = "NEURE_";

/// Read an env var by suffix, preferring `NEURON_<suffix>` and falling
/// back to the pre-rename `NEURE_<suffix>`.
///
/// `suffix` is the part after the prefix, e.g. `"LLM_MODEL_PATH"`.
pub fn env_var(suffix: &str) -> Option<String> {
    resolve_env(suffix, |key| std::env::var(key).ok())
}

/// [`env_var`] against a caller-supplied snapshot, such as the map
/// handed to [`NeuronConfig::from_env_map`].
pub fn env_map_get(env: &HashMap<String, String>, suffix: &str) -> Option<String> {
    resolve_env(suffix, |key| env.get(key).cloned())
}

/// Result-returning counterpart of [`env_var`], so call sites that
/// `match` or `if let Ok(..)` on `std::env::var` keep their shape. An
/// unset var and a non-UTF-8 var are both reported as `NotPresent`.
pub fn env_var_result(suffix: &str) -> Result<String, std::env::VarError> {
    match resolve_env(suffix, |key| std::env::var(key).ok()) {
        Some(v) => Ok(v),
        None => Err(std::env::VarError::NotPresent),
    }
}

fn resolve_env(suffix: &str, lookup: impl Fn(&str) -> Option<String>) -> Option<String> {
    let current = format!("{ENV_PREFIX}{suffix}");
    if let Some(v) = lookup(&current) {
        return Some(v);
    }
    let legacy = format!("{LEGACY_ENV_PREFIX}{suffix}");
    let v = lookup(&legacy)?;
    tracing::warn!(
        legacy = %legacy,
        replacement = %current,
        "env var uses the pre-rename name; update it, this fallback is temporary",
    );
    Some(v)
}

/// Strip either the current or the legacy prefix. Warns when the legacy
/// one matched, mirroring [`resolve_env`].
fn strip_env_prefix<'a>(key: &'a str, suffix: &str) -> Option<&'a str> {
    let current = format!("{ENV_PREFIX}{suffix}");
    if let Some(rest) = key.strip_prefix(&current) {
        return Some(rest);
    }
    let legacy = format!("{LEGACY_ENV_PREFIX}{suffix}");
    let rest = key.strip_prefix(&legacy)?;
    tracing::warn!(
        legacy = %key,
        replacement = %format!("{current}{rest}"),
        "env var uses the pre-rename name; update it, this fallback is temporary",
    );
    Some(rest)
}

fn apply_port_env(cfg: &mut NeuronConfig, env: &HashMap<String, String>) {
    // NEURON_PORT: parses as u16, invalid values fall back to the
    // default rather than panicking — env var typos shouldn't
    // crash startup.
    if let Some(v) = env_map_get(env, "PORT")
        && let Ok(port) = v.parse::<u16>()
    {
        cfg.port = port;
    }
}

fn apply_host_env(cfg: &mut NeuronConfig, env: &HashMap<String, String>) {
    // NEURON_HOST: any non-empty string wins. Empty values fall
    // through to the default (0.0.0.0) so a stray `NEURON_HOST=`
    // shell export doesn't break bind.
    if let Some(v) = env_map_get(env, "HOST")
        && !v.is_empty()
    {
        cfg.host = v;
    }
}

fn apply_device_env(cfg: &mut NeuronConfig, env: &HashMap<String, String>) {
    // NEURON_DEVICE: accepts auto | cpu | nvidia (alias: cuda) | apple |
    // vulkan. Unknown values fall through to the default (`Auto`) — an
    // env-var typo shouldn't crash startup; a startup-time warning is
    // logged in ServerState when the resolved value isn't recognised by
    // the candle backend.
    if let Some(v) = env_map_get(env, "DEVICE")
        && let Ok(d) = serde_json::from_str::<DeviceSelection>(&format!("\"{v}\""))
    {
        cfg.device = d;
    }
}

fn apply_model_dirs_env(cfg: &mut NeuronConfig, env: &HashMap<String, String>) {
    // NEURON_MODEL_DIRS (multi-value, colon-separated) takes
    // priority over NEURON_MODEL_DIR (single-value legacy).
    if let Some(dirs) = env_map_get(env, "MODEL_DIRS") {
        cfg.model_dirs = dirs
            .split(':')
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
            .collect();
    } else if let Some(dir) = env_map_get(env, "MODEL_DIR") {
        cfg.model_dirs = vec![PathBuf::from(dir)];
    }
}

fn apply_default_source_env(cfg: &mut NeuronConfig, env: &HashMap<String, String>) {
    if let Some(v) = env_map_get(env, "DEFAULT_SOURCE") {
        cfg.default_source_id = v;
    }
}

fn apply_per_engine_sources_env(cfg: &mut NeuronConfig, env: &HashMap<String, String>) {
    let engine_map: &[(EngineType, &str)] = &[
        (EngineType::Llm, "LLM_SOURCE"),
        (EngineType::Tts, "TTS_SOURCE"),
        (EngineType::Asr, "ASR_SOURCE"),
        (EngineType::Rerank, "RERANK_SOURCE"),
    ];
    for (engine, suffix) in engine_map {
        if let Some(v) = env_map_get(env, suffix) {
            cfg.per_engine_source.insert(*engine, v);
        }
    }
}

fn apply_per_model_sources_env(cfg: &mut NeuronConfig, env: &HashMap<String, String>) {
    // Per-model: NEURON_MODEL_SOURCE_<ENGINE>_<ID_UPPER>.
    // For example, NEURON_MODEL_SOURCE_LLM_QWEN2_5_0_5B=qwen2-5-0-5b → "llm/qwen2-5-0-5b"
    // (We keep underscores as hyphens in the model_id portion.
    // A trailing dot-separated version like `qwen2.5-0.5b` is not
    // representable in env-var form; users can call `with_model_source`
    // directly for those.)
    for (k, v) in env {
        if let Some(rest) = strip_env_prefix(k, "MODEL_SOURCE_") {
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

fn apply_source_endpoints_env(cfg: &mut NeuronConfig, env: &HashMap<String, String>) {
    // Source endpoints: NEURON_SOURCE_<ID>_ENDPOINT.
    for (k, v) in env {
        if let Some(rest) = strip_env_prefix(k, "SOURCE_")
            && let Some(source_id) = rest.strip_suffix("_ENDPOINT")
        {
            let id_lower = source_id.to_lowercase();
            cfg.source_endpoints.insert(id_lower, v.clone());
        }
    }
}

fn apply_scheduler_env(cfg: &mut NeuronConfig, env: &HashMap<String, String>) {
    // NEURON_SCHEDULER_ENABLED: true/1/false/0 (anything unparseable
    // keeps the default, mirroring the other env appliers).
    if let Some(v) = env_map_get(env, "SCHEDULER_ENABLED") {
        match v.to_lowercase().as_str() {
            "true" | "1" | "yes" => cfg.scheduler.enabled = true,
            "false" | "0" | "no" => cfg.scheduler.enabled = false,
            _ => {}
        }
    }
    // NEURON_SCHEDULER_BUDGET_MB: resident-weight budget in MiB; 0 /
    // unset = derive from total RAM.
    if let Some(v) = env_map_get(env, "SCHEDULER_BUDGET_MB")
        && let Ok(mb) = v.parse::<u64>()
    {
        cfg.scheduler.budget_bytes = mb.saturating_mul(1024 * 1024);
    }
    if let Some(v) = env_map_get(env, "SCHEDULER_IDLE_SECS")
        && let Ok(secs) = v.parse::<u64>()
    {
        cfg.scheduler.idle_evict_secs = secs;
    }
    if let Some(v) = env_map_get(env, "SCHEDULER_MAX_CONCURRENCY")
        && let Ok(n) = v.parse::<u32>()
    {
        cfg.scheduler.max_concurrent_per_model = n;
    }
    if let Some(v) = env_map_get(env, "SCHEDULER_MAX_QUEUE_WAIT_SECS")
        && let Ok(secs) = v.parse::<u64>()
    {
        cfg.scheduler.max_queue_wait_secs = secs;
    }
    if let Some(v) = env_map_get(env, "SCHEDULER_PLATFORM")
        && !v.is_empty()
    {
        cfg.scheduler.platform = v.to_lowercase();
    }
}

impl Default for NeuronConfig {
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
    fn test_neuron_config_default_port_8083() {
        let config = NeuronConfig::default();
        assert_eq!(config.port, 8083);
    }

    #[test]
    fn test_neuron_config_default_host_0_0_0_0() {
        let config = NeuronConfig::default();
        assert_eq!(config.host, "0.0.0.0");
    }

    #[test]
    fn test_neuron_config_default_device_auto() {
        let config = NeuronConfig::default();
        assert_eq!(config.device, DeviceSelection::Auto);
    }

    #[test]
    fn test_device_selection_accepts_nvidia_and_cuda_alias() {
        for value in ["nvidia", "cuda"] {
            let parsed: DeviceSelection =
                serde_json::from_str(&format!("\"{value}\"")).expect("parse device");
            assert_eq!(parsed, DeviceSelection::Nvidia, "value: {value}");
        }
        for value in ["auto", "cpu", "apple", "vulkan"] {
            let parsed: DeviceSelection =
                serde_json::from_str(&format!("\"{value}\"")).expect("parse device");
            assert_eq!(
                parsed,
                match value {
                    "auto" => DeviceSelection::Auto,
                    "cpu" => DeviceSelection::Cpu,
                    "apple" => DeviceSelection::Apple,
                    _ => DeviceSelection::Vulkan,
                },
                "value: {value}"
            );
        }
    }

    #[test]
    fn test_neuron_config_serialize_roundtrip() {
        let config = NeuronConfig {
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
            scheduler: crate::scheduler::SchedulerConfig::default(),
        };

        let json = serde_json::to_string(&config).unwrap();
        let parsed: NeuronConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.port, 9000);
        assert_eq!(parsed.host, "127.0.0.1");
        assert_eq!(parsed.device, DeviceSelection::Nvidia);
        assert_eq!(parsed.rerank_model_path.as_deref(), Some("/models/rerank"));
        assert_eq!(
            parsed.default_rerank_model.as_deref(),
            Some("bge-reranker-base")
        );
        assert_eq!(parsed.model_dirs.len(), 2);
        assert!(parsed.scheduler.enabled);
    }

    #[test]
    fn test_neuron_config_scheduler_defaults_roundtrip_without_field() {
        // Pre-scheduler serialized configs (no `scheduler` key) must
        // still load.
        let legacy = r#"{"port":8083,"host":"0.0.0.0","llm_model_path":null,"tts_model_path":null,"asr_model_path":null,"rerank_model_path":null,"device":"auto","default_llm_model":null,"default_tts_model":null,"default_asr_model":null,"default_rerank_model":null,"model_dirs":[],"default_source_id":"modelscope","per_engine_source":{},"per_model_source":{},"source_endpoints":{},"registrations":[]}"#;
        let parsed: NeuronConfig = serde_json::from_str(legacy).unwrap();
        assert!(parsed.scheduler.enabled);
        assert_eq!(parsed.scheduler.budget_bytes, 0);
    }

    #[test]
    fn test_from_env_parses_scheduler_knobs() {
        let mut env = HashMap::new();
        env.insert("NEURON_SCHEDULER_ENABLED".to_string(), "false".to_string());
        env.insert("NEURON_SCHEDULER_BUDGET_MB".to_string(), "4096".to_string());
        env.insert("NEURON_SCHEDULER_IDLE_SECS".to_string(), "30".to_string());
        env.insert("NEURON_SCHEDULER_MAX_CONCURRENCY".to_string(), "8".to_string());
        env.insert("NEURON_SCHEDULER_MAX_QUEUE_WAIT_SECS".to_string(), "60".to_string());
        env.insert("NEURON_SCHEDULER_PLATFORM".to_string(), "CUDA".to_string());
        let c = NeuronConfig::from_env_map(&env);
        assert!(!c.scheduler.enabled);
        assert_eq!(c.scheduler.budget_bytes, 4 * 1024 * 1024 * 1024);
        assert_eq!(c.scheduler.idle_evict_secs, 30);
        assert_eq!(c.scheduler.max_concurrent_per_model, 8);
        assert_eq!(c.scheduler.max_queue_wait_secs, 60);
        assert_eq!(c.scheduler.effective_platform(), "cuda");
        assert!(!c.scheduler.observes_rss());
    }

    #[test]
    fn test_neuron_config_builder_pattern() {
        let config = NeuronConfig::new()
            .with_port(9090)
            .with_device(DeviceSelection::Apple)
            .with_llm_model("qwen3-0.6b");

        assert_eq!(config.port, 9090);
        assert_eq!(config.device, DeviceSelection::Apple);
        assert_eq!(config.default_llm_model.as_deref(), Some("qwen3-0.6b"));
    }

    #[test]
    fn test_neuron_config_with_tts_asr_rerank_model_builders() {
        let config = NeuronConfig::new()
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
    fn test_neuron_config_default_validate_returns_ok() {
        let c = NeuronConfig::new();
        assert!(c.validate().is_ok());
    }

    #[test]
    fn test_neuron_config_validate_after_from_env_map_returns_ok() {
        let mut env = HashMap::new();
        env.insert("NEURON_PORT".to_string(), "9999".to_string());
        env.insert("NEURON_HOST".to_string(), "127.0.0.1".to_string());
        let c = NeuronConfig::from_env_map(&env);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn test_neuron_config_with_model_builders_accept_string_and_str() {
        let owned = String::from("qwen3-0.6b");
        let config = NeuronConfig::new()
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
        let c = NeuronConfig::new()
            .with_model_dir("/opt/models")
            .with_model_dir("/home/user/models");
        assert_eq!(c.model_dirs.len(), 3); // 1 default + 2 added
        assert_eq!(c.model_dirs[1].to_str().unwrap(), "/opt/models");
        assert_eq!(c.model_dirs[2].to_str().unwrap(), "/home/user/models");
    }

    #[test]
    fn test_with_default_source_overrides() {
        let c = NeuronConfig::new().with_default_source("modelscope");
        assert_eq!(c.default_source_id, "modelscope");
    }

    #[test]
    fn test_with_engine_source_adds_to_map() {
        let c = NeuronConfig::new()
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
            NeuronConfig::new().with_model_source(EngineType::Llm, "qwen2.5-0.5b", "private-hub");
        assert_eq!(
            c.per_model_source.get("llm/qwen2.5-0.5b").unwrap(),
            "private-hub"
        );
    }

    #[test]
    fn test_with_source_endpoint_stores_in_map() {
        let c = NeuronConfig::new().with_source_endpoint("huggingface", "https://hf-mirror.com");
        assert_eq!(
            c.source_endpoints.get("huggingface").unwrap(),
            "https://hf-mirror.com"
        );
    }

    #[test]
    fn test_from_env_parses_neuron_model_dirs_multi_value() {
        let mut env = HashMap::new();
        env.insert("NEURON_MODEL_DIRS".to_string(), "/a:/b:/c".to_string());
        let c = NeuronConfig::from_env_map(&env);
        assert_eq!(c.model_dirs.len(), 3);
        assert_eq!(c.model_dirs[0].to_str().unwrap(), "/a");
        assert_eq!(c.model_dirs[2].to_str().unwrap(), "/c");
    }

    #[test]
    fn test_from_env_parses_neuron_model_dir_legacy_single() {
        let mut env = HashMap::new();
        env.insert("NEURON_MODEL_DIR".to_string(), "/single".to_string());
        let c = NeuronConfig::from_env_map(&env);
        assert_eq!(c.model_dirs.len(), 1);
        assert_eq!(c.model_dirs[0].to_str().unwrap(), "/single");
    }

    #[test]
    fn test_from_env_model_dirs_takes_priority_over_legacy_model_dir() {
        let mut env = HashMap::new();
        env.insert("NEURON_MODEL_DIRS".to_string(), "/multi".to_string());
        env.insert("NEURON_MODEL_DIR".to_string(), "/single".to_string());
        let c = NeuronConfig::from_env_map(&env);
        assert_eq!(c.model_dirs.len(), 1);
        assert_eq!(c.model_dirs[0].to_str().unwrap(), "/multi");
    }

    #[test]
    fn test_from_env_parses_per_engine_sources() {
        let mut env = HashMap::new();
        env.insert("NEURON_LLM_SOURCE".to_string(), "modelscope".to_string());
        env.insert("NEURON_TTS_SOURCE".to_string(), "private-tts".to_string());
        let c = NeuronConfig::from_env_map(&env);
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
            "NEURON_SOURCE_HUGGINGFACE_ENDPOINT".to_string(),
            "https://hf-mirror.com".to_string(),
        );
        let c = NeuronConfig::from_env_map(&env);
        assert_eq!(
            c.source_endpoints.get("huggingface").unwrap(),
            "https://hf-mirror.com"
        );
    }

    #[test]
    fn test_from_env_parses_neuron_port() {
        let mut env = HashMap::new();
        env.insert("NEURON_PORT".to_string(), "9999".to_string());
        let c = NeuronConfig::from_env_map(&env);
        assert_eq!(c.port, 9999);
    }

    #[test]
    fn test_from_env_parses_neuron_port_invalid_value_keeps_default() {
        let mut env = HashMap::new();
        env.insert("NEURON_PORT".to_string(), "not-a-number".to_string());
        let c = NeuronConfig::from_env_map(&env);
        assert_eq!(c.port, NeuronConfig::default_port());
    }

    #[test]
    fn test_from_env_parses_neuron_host() {
        let mut env = HashMap::new();
        env.insert("NEURON_HOST".to_string(), "127.0.0.1".to_string());
        let c = NeuronConfig::from_env_map(&env);
        assert_eq!(c.host, "127.0.0.1");
    }

    #[test]
    fn test_from_env_parses_neuron_host_empty_string_keeps_default() {
        let mut env = HashMap::new();
        env.insert("NEURON_HOST".to_string(), "".to_string());
        let c = NeuronConfig::from_env_map(&env);
        assert_eq!(c.host, NeuronConfig::default_host());
    }

    #[test]
    fn test_default_model_dir_ends_with_neuron_models() {
        // Asserts the pure path helper; `NeuronConfig::new` layers the
        // legacy fallback on top and is covered separately.
        assert!(NeuronConfig::default_model_dir().ends_with(".neuron/models"));
    }

    #[test]
    fn test_env_map_get_prefers_current_name_over_legacy() {
        let mut env = HashMap::new();
        env.insert("NEURON_PORT".to_string(), "1111".to_string());
        env.insert("NEURE_PORT".to_string(), "2222".to_string());
        assert_eq!(env_map_get(&env, "PORT").unwrap(), "1111");
    }

    #[test]
    fn test_env_map_get_falls_back_to_legacy_name() {
        let mut env = HashMap::new();
        env.insert("NEURE_PORT".to_string(), "2222".to_string());
        assert_eq!(env_map_get(&env, "PORT").unwrap(), "2222");
    }

    #[test]
    fn test_env_map_get_returns_none_when_neither_name_is_set() {
        assert!(env_map_get(&HashMap::new(), "PORT").is_none());
    }

    #[test]
    fn test_from_env_map_accepts_legacy_port() {
        let mut env = HashMap::new();
        env.insert("NEURE_PORT".to_string(), "9999".to_string());
        assert_eq!(NeuronConfig::from_env_map(&env).port, 9999);
    }

    #[test]
    fn test_from_env_map_accepts_legacy_model_dirs() {
        let mut env = HashMap::new();
        env.insert("NEURE_MODEL_DIRS".to_string(), "/a:/b".to_string());
        let c = NeuronConfig::from_env_map(&env);
        assert_eq!(c.model_dirs.len(), 2);
        assert_eq!(c.model_dirs[0].to_str().unwrap(), "/a");
    }

    #[test]
    fn test_from_env_map_accepts_legacy_per_engine_source() {
        let mut env = HashMap::new();
        env.insert("NEURE_LLM_SOURCE".to_string(), "hf-mirror".to_string());
        let c = NeuronConfig::from_env_map(&env);
        assert_eq!(
            c.per_engine_source.get(&EngineType::Llm).unwrap(),
            "hf-mirror"
        );
    }

    #[test]
    fn test_from_env_map_accepts_legacy_source_endpoint() {
        let mut env = HashMap::new();
        env.insert(
            "NEURE_SOURCE_HUGGINGFACE_ENDPOINT".to_string(),
            "https://hf-mirror.com".to_string(),
        );
        let c = NeuronConfig::from_env_map(&env);
        assert_eq!(
            c.source_endpoints.get("huggingface").unwrap(),
            "https://hf-mirror.com"
        );
    }

    #[test]
    fn test_ensure_dir_nonexistent_path_returns_does_not_exist_error() {
        let dir = std::env::temp_dir().join("neuron-test-does-not-exist-12345");
        let result = ensure_dir(&dir, "NEURON_TEST_RESOLVE_VAR");
        let err = result.err().unwrap();
        assert!(err.contains("does not exist"));
        assert!(err.contains("NEURON_TEST_RESOLVE_VAR"));
    }

    #[test]
    fn test_ensure_dir_file_not_directory_returns_not_a_directory_error() {
        let path = std::env::temp_dir().join("neuron-test-not-a-dir.txt");
        std::fs::write(&path, b"not a dir").unwrap();
        let result = ensure_dir(&path, "NEURON_TEST_RESOLVE_VAR");
        std::fs::remove_file(&path).ok();
        let err = result.err().unwrap();
        assert!(err.contains("is not a directory"));
    }

    #[test]
    fn test_ensure_dir_existing_dir_returns_ok() {
        let dir = std::env::temp_dir();
        let result = ensure_dir(&dir, "NEURON_TEST_RESOLVE_VAR");
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
