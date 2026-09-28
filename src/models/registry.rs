use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum EngineType {
    #[default]
    Llm,
    Tts,
    Asr,
    Rerank,
    Embedding,
    Vision,
    Decision,
}

impl EngineType {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "llm" => Ok(Self::Llm),
            "tts" => Ok(Self::Tts),
            "asr" => Ok(Self::Asr),
            "rerank" => Ok(Self::Rerank),
            "embedding" => Ok(Self::Embedding),
            "vision" => Ok(Self::Vision),
            "decision" => Ok(Self::Decision),
            _ => Err(format!("unknown engine type: {s}")),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Llm => "llm",
            Self::Tts => "tts",
            Self::Asr => "asr",
            Self::Rerank => "rerank",
            Self::Embedding => "embedding",
            Self::Vision => "vision",
            Self::Decision => "decision",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadedModel {
    pub engine: EngineType,
    pub id: String,
    pub path: PathBuf,
    pub size_bytes: u64,
    pub file_count: u32,
    pub files: Vec<String>,
    pub modified_at: DateTime<Utc>,
    pub compatible_engines: Vec<String>,
}

pub struct Registry {
    roots: Vec<PathBuf>,
}

impl Registry {
    /// Construct a registry with a single root directory. Equivalent
    /// to `Registry::multi(vec![root])`.
    pub fn new(root: PathBuf) -> Self {
        Self { roots: vec![root] }
    }

    /// Construct a registry that scans multiple root directories.
    /// `list()` aggregates entries from all roots; on a
    /// `(engine, id)` collision, the first root wins. `get()` and
    /// `delete()` look at every root in order. `model_path()` and
    /// `root()` always refer to the first root (used for
    /// write/delete operations and for the legacy single-root view).
    pub fn multi(roots: Vec<PathBuf>) -> Self {
        Self { roots }
    }

    pub fn roots(&self) -> &[PathBuf] {
        &self.roots
    }

    /// Return the first root. Kept for backward compatibility with
    /// code that used `Registry` with a single root and accessed
    /// `.root()` directly (e.g. `pull_model` building `dest_dir`).
    pub fn root(&self) -> &Path {
        &self.roots[0]
    }

    pub fn list(&self) -> Vec<DownloadedModel> {
        use std::collections::HashMap;
        let mut by_key: HashMap<(EngineType, String), DownloadedModel> = HashMap::new();
        for root in &self.roots {
            if !root.is_dir() {
                continue;
            }
            for engine in [
                EngineType::Llm,
                EngineType::Tts,
                EngineType::Asr,
                EngineType::Rerank,
                EngineType::Embedding,
                EngineType::Decision,
            ] {
                let engine_dir = root.join(engine.as_str());
                if !engine_dir.is_dir() {
                    continue;
                }
                if let Ok(entries) = fs::read_dir(&engine_dir) {
                    for entry in entries.flatten() {
                        if let Some(model) = self.scan_model_dir(engine, &entry.path()) {
                            let key = (engine, model.id.clone());
                            by_key.entry(key).or_insert(model);
                        }
                    }
                }
            }
        }
        let mut out: Vec<DownloadedModel> = by_key.into_values().collect();
        out.sort_by(|a, b| {
            (a.engine.as_str(), a.id.as_str()).cmp(&(b.engine.as_str(), b.id.as_str()))
        });
        out
    }

    pub fn get(&self, engine: EngineType, id: &str) -> Option<DownloadedModel> {
        for root in &self.roots {
            let path = root.join(engine.as_str()).join(id);
            if path.is_dir()
                && let Some(m) = self.scan_model_dir(engine, &path) {
                    return Some(m);
                }
        }
        None
    }

    pub fn delete(&self, engine: EngineType, id: &str) -> Result<(), String> {
        validate_model_id(id)?;
        let path = self.model_path(engine, id);
        if !path.is_dir() {
            return Err(format!("model not found: {}/{}", engine.as_str(), id));
        }
        fs::remove_dir_all(&path).map_err(|e| format!("failed to delete {}: {}", path.display(), e))
    }

    pub fn model_path(&self, engine: EngineType, id: &str) -> PathBuf {
        self.root().join(engine.as_str()).join(id)
    }

    fn scan_model_dir(&self, engine: EngineType, path: &Path) -> Option<DownloadedModel> {
        let id = path.file_name()?.to_str()?.to_string();
        let mut size_bytes: u64 = 0;
        let mut files: Vec<String> = Vec::new();
        let mut latest_mtime: SystemTime = SystemTime::UNIX_EPOCH;

        for entry in fs::read_dir(path).ok()?.flatten() {
            let entry_path = entry.path();
            let metadata = entry.metadata().ok()?;
            if metadata.is_file() {
                size_bytes += metadata.len();
                if let Ok(rel) = entry_path.strip_prefix(path) {
                    files.push(rel.to_string_lossy().to_string());
                }
                if let Ok(mtime) = metadata.modified()
                    && mtime > latest_mtime {
                        latest_mtime = mtime;
                    }
            }
        }

        let modified_at: DateTime<Utc> = latest_mtime.into();

        Some(DownloadedModel {
            engine,
            id,
            path: path.to_path_buf(),
            size_bytes,
            file_count: files.len() as u32,
            files,
            modified_at,
            compatible_engines: detect_compatible_engines(path),
        })
    }
}

fn detect_compatible_engines(path: &Path) -> Vec<String> {
    let mut engines = Vec::new();
    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if name.ends_with(".tflite") {
                    engines.push("litert".to_string());
                } else if name.ends_with(".safetensors") || name == "config.json" {
                    engines.push("candle".to_string());
                } else if name.ends_with(".mpk") {
                    engines.push("burn".to_string());
                }
            }
        }
    }
    engines.sort();
    engines.dedup();
    engines
}

pub fn validate_model_id(id: &str) -> Result<(), String> {
    if id.is_empty() {
        return Err("model id cannot be empty".to_string());
    }
    if id.starts_with('/') {
        return Err("model id cannot be an absolute path".to_string());
    }
    if id.contains("..") {
        return Err("model id cannot contain '..'".to_string());
    }
    for c in id.chars() {
        if !c.is_ascii_alphanumeric() && !matches!(c, '-' | '_' | '.' | '/' | ':' | '@') {
            return Err(format!("invalid character in model id: {c:?}"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn make_fake_model(root: &Path, engine: EngineType, id: &str, files: &[&str]) {
        let dir = root.join(engine.as_str()).join(id);
        fs::create_dir_all(&dir).unwrap();
        for f in files {
            fs::write(dir.join(f), b"fake content").unwrap();
        }
    }

    #[test]
    fn test_engine_type_parse_valid() {
        assert_eq!(EngineType::parse("llm").unwrap(), EngineType::Llm);
        assert_eq!(EngineType::parse("LLM").unwrap(), EngineType::Llm);
        assert_eq!(EngineType::parse("rerank").unwrap(), EngineType::Rerank);
    }

    #[test]
    fn test_engine_type_parse_invalid() {
        assert!(EngineType::parse("foo").is_err());
    }

    #[test]
    fn test_engine_type_as_str_roundtrip() {
        for e in [
            EngineType::Llm,
            EngineType::Tts,
            EngineType::Asr,
            EngineType::Rerank,
            EngineType::Embedding,
        ] {
            assert_eq!(EngineType::parse(e.as_str()).unwrap(), e);
        }
    }

    #[test]
    fn test_engine_type_embedding_parses() {
        assert_eq!(
            EngineType::parse("embedding").unwrap(),
            EngineType::Embedding
        );
        assert_eq!(
            EngineType::parse("EMBEDDING").unwrap(),
            EngineType::Embedding
        );
        assert_eq!(EngineType::Embedding.as_str(), "embedding");
    }

    #[test]
    fn test_registry_list_finds_downloaded_models() {
        let tmp = TempDir::new().unwrap();
        make_fake_model(
            tmp.path(),
            EngineType::Llm,
            "qwen2.5-0.5b",
            &["config.json", "model.safetensors"],
        );
        make_fake_model(tmp.path(), EngineType::Rerank, "bge-base", &["config.json"]);

        let r = Registry::new(tmp.path().to_path_buf());
        let models = r.list();
        assert_eq!(models.len(), 2);

        let llm = models.iter().find(|m| m.engine == EngineType::Llm).unwrap();
        assert_eq!(llm.id, "qwen2.5-0.5b");
        assert_eq!(llm.file_count, 2);

        let rerank = models
            .iter()
            .find(|m| m.engine == EngineType::Rerank)
            .unwrap();
        assert_eq!(rerank.id, "bge-base");
    }

    #[test]
    fn test_registry_get_returns_details() {
        let tmp = TempDir::new().unwrap();
        make_fake_model(tmp.path(), EngineType::Llm, "test-model", &["config.json"]);

        let r = Registry::new(tmp.path().to_path_buf());
        let m = r.get(EngineType::Llm, "test-model").unwrap();
        assert_eq!(m.id, "test-model");
        assert_eq!(m.engine, EngineType::Llm);
    }

    #[test]
    fn test_registry_get_nonexistent_returns_none() {
        let tmp = TempDir::new().unwrap();
        let r = Registry::new(tmp.path().to_path_buf());
        assert!(r.get(EngineType::Llm, "nonexistent").is_none());
    }

    #[test]
    fn test_registry_delete_removes_dir() {
        let tmp = TempDir::new().unwrap();
        make_fake_model(tmp.path(), EngineType::Llm, "to-delete", &["config.json"]);

        let r = Registry::new(tmp.path().to_path_buf());
        assert!(r.delete(EngineType::Llm, "to-delete").is_ok());
        assert!(!tmp.path().join("llm/to-delete").exists());
    }

    #[test]
    fn test_registry_delete_nonexistent_returns_error() {
        let tmp = TempDir::new().unwrap();
        let r = Registry::new(tmp.path().to_path_buf());
        assert!(r.delete(EngineType::Llm, "nonexistent").is_err());
    }

    #[test]
    fn test_validate_id_accepts_valid() {
        assert!(validate_model_id("qwen2.5-0.5b").is_ok());
        assert!(validate_model_id("bge-reranker-base").is_ok());
        assert!(validate_model_id("model_with.dots").is_ok());
        assert!(validate_model_id("org/model").is_ok());
        assert!(validate_model_id("model:v1").is_ok());
        assert!(validate_model_id("model@v1.0").is_ok());
    }

    #[test]
    fn test_validate_id_rejects_path_traversal() {
        assert!(validate_model_id("../etc/passwd").is_err());
        assert!(validate_model_id("foo/../bar").is_err());
        assert!(validate_model_id("/abs/path").is_err());
        assert!(validate_model_id("").is_err());
        assert!(validate_model_id("foo$bar").is_err());
    }

    // -- multi-root --

    #[test]
    fn test_registry_multi_aggregates_across_roots() {
        let tmp1 = TempDir::new().unwrap();
        let tmp2 = TempDir::new().unwrap();
        make_fake_model(tmp1.path(), EngineType::Llm, "alpha", &["config.json"]);
        make_fake_model(tmp2.path(), EngineType::Llm, "beta", &["config.json"]);
        make_fake_model(tmp2.path(), EngineType::Rerank, "gamma", &["config.json"]);

        let r = Registry::multi(vec![tmp1.path().to_path_buf(), tmp2.path().to_path_buf()]);
        let models = r.list();
        assert_eq!(models.len(), 3);
        let ids: Vec<String> = models.iter().map(|m| m.id.clone()).collect();
        assert!(ids.contains(&"alpha".to_string()));
        assert!(ids.contains(&"beta".to_string()));
        assert!(ids.contains(&"gamma".to_string()));
    }

    #[test]
    fn test_registry_multi_first_root_wins_on_collision() {
        let tmp1 = TempDir::new().unwrap();
        let tmp2 = TempDir::new().unwrap();
        make_fake_model(tmp1.path(), EngineType::Llm, "shared", &["config.json"]);
        make_fake_model(tmp2.path(), EngineType::Llm, "shared", &["config.json"]);

        let r = Registry::multi(vec![tmp1.path().to_path_buf(), tmp2.path().to_path_buf()]);
        let models = r.list();
        let shared: Vec<&DownloadedModel> = models.iter().filter(|m| m.id == "shared").collect();
        assert_eq!(shared.len(), 1);
        assert!(shared[0].path.starts_with(tmp1.path()));
    }

    #[test]
    fn test_registry_multi_get_searches_all_roots() {
        let tmp1 = TempDir::new().unwrap();
        let tmp2 = TempDir::new().unwrap();
        make_fake_model(tmp1.path(), EngineType::Llm, "in-first", &["config.json"]);
        make_fake_model(tmp2.path(), EngineType::Llm, "in-second", &["config.json"]);

        let r = Registry::multi(vec![tmp1.path().to_path_buf(), tmp2.path().to_path_buf()]);
        assert!(r.get(EngineType::Llm, "in-first").is_some());
        assert!(r.get(EngineType::Llm, "in-second").is_some());
        assert!(r.get(EngineType::Llm, "missing").is_none());
    }

    #[test]
    fn test_registry_multi_skips_nonexistent_roots() {
        let tmp = TempDir::new().unwrap();
        make_fake_model(tmp.path(), EngineType::Llm, "real", &["config.json"]);

        let r = Registry::multi(vec![
            PathBuf::from("/nonexistent/path/12345"),
            tmp.path().to_path_buf(),
        ]);
        let models = r.list();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "real");
    }
}
