//! Config-shape adapters: HF checkpoint configs vs. what a backend expects.
//!
//! ## Why this module exists
//!
//! Checkpoints published with transformers v5 nest the RoPE settings inside
//! text_config.rope_parameters:
//!
//!     "text_config": {
//!       "rope_parameters": {
//!         "mrope_section": [24, 20, 20],
//!         "rope_theta": 5000000,
//!         "rope_type": "default"
//!       }
//!     }
//!
//! mistral.rs v0.8.22's Qwen3-VL loader expects the FLAT shape
//! (text_config.rope_theta plus text_config.rope_scaling.mrope_section) and
//! fails on the nested one with "missing field rope_theta" - the checkpoint
//! is fine, the reader is behind. Tencent's WeVisOne-VL (a Qwen3-VL backbone)
//! ships exactly the v5 shape, so without an adapter it is "listed but
//! unreachable".
//!
//! ## What the adapter does - and does not do
//!
//! The adapter COPIES the config into a shadow directory next to the weights
//! (every other file is hard-linked, so no extra disk) and points the backend
//! at that directory. The checkpoint's own config.json is NEVER modified - a
//! re-download or a different tool keeps seeing the original file.
//!
//! Only the two fields mistral.rs actually reads are flattened. Everything
//! else (including rope_parameters itself) is passed through untouched.

use std::path::Path;

/// What the adapter changed in the config.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigShapeFix {
    /// The normalized config text (pretty-printed JSON).
    pub text: String,
    /// The config paths that were added, e.g. "text_config.rope_theta".
    pub added_fields: Vec<String>,
}

/// Flatten the transformers-v5 rope_parameters nesting into the shape
/// mistral.rs's Qwen3-VL loader expects.
///
/// Returns Ok(None) when the config needs no change (it is already flat, or
/// it is not a Qwen3-VL-style config at all).
pub fn normalize_qwen3vl_text_config(raw: &str) -> Result<Option<ConfigShapeFix>, String> {
    let mut root: serde_json::Value = serde_json::from_str(raw)
        .map_err(|e| format!("config.json is not valid JSON: {e}"))?;

    let Some(text_cfg) = root.get_mut("text_config").and_then(|t| t.as_object_mut()) else {
        return Ok(None);
    };
    let Some(rope_params) = text_cfg.get("rope_parameters").and_then(|v| v.as_object()).cloned()
    else {
        return Ok(None);
    };

    let mut added_fields = Vec::new();

    // rope_theta - mistral.rs's TextConfig requires it as a flat f64.
    if !text_cfg.contains_key("rope_theta") {
        if let Some(theta) = rope_params.get("rope_theta") {
            text_cfg.insert("rope_theta".to_string(), theta.clone());
            added_fields.push("text_config.rope_theta".to_string());
        }
    }

    // rope_scaling - mistral.rs's TextConfig requires it as an object
    // holding mrope_section (Qwen3-VL's mRoPE split of the head dim).
    if !text_cfg.contains_key("rope_scaling") {
        if let Some(section) = rope_params.get("mrope_section") {
            let mut scaling = serde_json::Map::new();
            scaling.insert("mrope_section".to_string(), section.clone());
            text_cfg.insert("rope_scaling".to_string(), serde_json::Value::Object(scaling));
            added_fields.push("text_config.rope_scaling".to_string());
        }
    }

    if added_fields.is_empty() {
        return Ok(None);
    }

    let text = serde_json::to_string_pretty(&root)
        .map_err(|e| format!("re-serializing the normalized config failed: {e}"))?;
    Ok(Some(ConfigShapeFix { text, added_fields }))
}

/// The precision a checkpoint's config declares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeclaredDtype {
    BF16,
    F16,
    F32,
}

/// Read the precision a checkpoint declares, if it declares one.
///
/// Transformers v4 wrote it as `torch_dtype`, v5 as `dtype`, and it may
/// live at the top level or inside `text_config`. A checkpoint that says
/// `bfloat16` is a half-precision checkpoint: loading it as F32 doubles
/// the resident memory (a 4B checkpoint goes from ~8 GB to ~16 GB) for
/// no gain, because the backend has to cast the stored weights anyway.
///
/// Returns `None` when the config declares nothing — the caller then
/// keeps the backend's own default.
pub fn declared_dtype(config_text: &str) -> Option<DeclaredDtype> {
    let root: serde_json::Value = serde_json::from_str(config_text).ok()?;

    let mut candidates: Vec<&serde_json::Value> = Vec::new();
    for key in ["dtype", "torch_dtype"] {
        if let Some(v) = root.get(key) {
            candidates.push(v);
        }
    }
    if let Some(tc) = root.get("text_config") {
        for key in ["dtype", "torch_dtype"] {
            if let Some(v) = tc.get(key) {
                candidates.push(v);
            }
        }
    }

    for v in candidates {
        let Some(name) = v.as_str() else { continue };
        match name.to_ascii_lowercase().as_str() {
            "bfloat16" | "bf16" => return Some(DeclaredDtype::BF16),
            "float16" | "f16" | "half" => return Some(DeclaredDtype::F16),
            "float32" | "f32" | "float" => return Some(DeclaredDtype::F32),
            _ => continue,
        }
    }
    None
}

/// Point the mistralrs backend at a directory whose config.json the backend
/// can read.
///
/// * A HF repo id (not a local directory) is returned unchanged - the adapter
///   has nothing to do before the weights are fetched.
/// * A local directory whose config is already in the expected shape is
///   returned unchanged.
/// * Otherwise a shadow directory (<model_dir>/.neuron-mistralrs) is (re)built
///   with the normalized config plus hard links to every other file, and its
///   path is returned.
///
/// The shadow's links are refreshed on every call, so a re-downloaded
/// checkpoint is never served with stale weights.
pub fn prepare_mistralrs_model_dir(model_id: &str) -> Result<String, String> {
    let dir = Path::new(model_id);
    if !dir.is_dir() {
        return Ok(model_id.to_string());
    }

    let cfg_path = dir.join("config.json");
    let Ok(raw) = std::fs::read_to_string(&cfg_path) else {
        // No config to adapt (or unreadable) - let the backend report it.
        return Ok(model_id.to_string());
    };

    let Some(fix) = normalize_qwen3vl_text_config(&raw)? else {
        return Ok(model_id.to_string());
    };

    let shadow = dir.join(".neuron-mistralrs");
    std::fs::create_dir_all(&shadow)
        .map_err(|e| format!("creating {} failed: {e}", shadow.display()))?;

    // Hard-link every regular file except the config (which we rewrite).
    // Sub-directories are skipped - the shadow dir itself is one of them.
    for entry in std::fs::read_dir(dir).map_err(|e| format!("reading {} failed: {e}", dir.display()))? {
        let entry = entry.map_err(|e| format!("reading {} failed: {e}", dir.display()))?;
        let path = entry.path();
        if path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name() else { continue };
        if name == "config.json" {
            continue;
        }
        let target = shadow.join(name);
        // Drop the old link first: a hard link keeps the OLD inode, so a
        // re-downloaded checkpoint would otherwise be served stale weights.
        if target.exists() {
            let _ = std::fs::remove_file(&target);
        }
        if std::fs::hard_link(&path, &target).is_err() {
            // No hard-link support on this filesystem - copy instead.
            std::fs::copy(&path, &target).map_err(|e| {
                format!("linking {} into the shadow dir failed: {e}", path.display())
            })?;
        }
    }

    std::fs::write(shadow.join("config.json"), &fix.text)
        .map_err(|e| format!("writing the normalized config failed: {e}"))?;

    Ok(shadow.to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    const V5_CONFIG: &str = r#"{
      "architectures": ["Qwen3VLForConditionalGeneration"],
      "model_type": "qwen3_vl",
      "text_config": {
        "hidden_size": 2048,
        "rope_parameters": {
          "mrope_interleaved": true,
          "mrope_section": [24, 20, 20],
          "rope_theta": 5000000,
          "rope_type": "default"
        }
      },
      "vision_config": { "depth": 24, "patch_size": 16 }
    }"#;

    const FLAT_CONFIG: &str = r#"{
      "text_config": {
        "rope_theta": 5000000,
        "rope_scaling": { "mrope_section": [24, 20, 20] }
      }
    }"#;

    #[test]
    fn test_normalize_flattens_the_v5_rope_parameters() {
        let fix = normalize_qwen3vl_text_config(V5_CONFIG)
            .expect("parse")
            .expect("the v5 shape needs a fix");
        let v: serde_json::Value = serde_json::from_str(&fix.text).expect("valid json");
        assert_eq!(v["text_config"]["rope_theta"], 5000000);
        assert_eq!(
            v["text_config"]["rope_scaling"]["mrope_section"],
            serde_json::json!([24, 20, 20])
        );
        // The original nesting must survive - we add, we do not rewrite.
        assert_eq!(v["text_config"]["rope_parameters"]["rope_theta"], 5000000);
        assert_eq!(
            v["text_config"]["rope_parameters"]["mrope_section"],
            serde_json::json!([24, 20, 20])
        );
        assert!(fix.added_fields.contains(&"text_config.rope_theta".to_string()));
        assert!(fix.added_fields.contains(&"text_config.rope_scaling".to_string()));
    }

    #[test]
    fn test_normalize_leaves_an_already_flat_config_alone() {
        assert!(
            normalize_qwen3vl_text_config(FLAT_CONFIG)
                .expect("parse")
                .is_none(),
            "a flat config must not be rewritten"
        );
    }

    #[test]
    fn test_normalize_ignores_a_config_without_a_text_config() {
        let raw = r#"{"vision_config": {"depth": 24}}"#;
        assert!(normalize_qwen3vl_text_config(raw).expect("parse").is_none());
    }

    #[test]
    fn test_prepare_builds_a_shadow_dir_with_the_normalized_config() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path().join("model");
        std::fs::create_dir_all(&dir).expect("mkdir");
        std::fs::write(dir.join("config.json"), V5_CONFIG).expect("write config");
        std::fs::write(dir.join("model.safetensors"), b"weights").expect("write weights");
        std::fs::write(dir.join("tokenizer.json"), b"{}").expect("write tokenizer");

        let prepared = prepare_mistralrs_model_dir(dir.to_str().unwrap()).expect("prepare");
        assert_ne!(prepared, dir.to_str().unwrap(), "a shadow dir must be used");
        let shadow = Path::new(&prepared);
        assert!(shadow.join("model.safetensors").exists(), "the weights must be linked in");
        assert!(shadow.join("tokenizer.json").exists(), "the tokenizer must be linked in");

        let raw = std::fs::read_to_string(shadow.join("config.json")).expect("read normalized");
        let v: serde_json::Value = serde_json::from_str(&raw).expect("valid json");
        assert_eq!(v["text_config"]["rope_theta"], 5000000);

        // The checkpoint's own config must be untouched.
        let original = std::fs::read_to_string(dir.join("config.json")).expect("read original");
        let orig_v: serde_json::Value = serde_json::from_str(&original).expect("valid json");
        assert!(orig_v["text_config"].get("rope_theta").is_none());
        assert!(orig_v["text_config"].get("rope_scaling").is_none());
    }

    #[test]
    fn test_declared_dtype_is_read_from_the_v5_shape() {
        let raw = r#"{
          "dtype": "bfloat16",
          "text_config": { "dtype": "bfloat16" }
        }"#;
        assert_eq!(declared_dtype(raw), Some(DeclaredDtype::BF16));
    }

    #[test]
    fn test_declared_dtype_is_read_from_the_v4_torch_dtype_key() {
        let raw = r#"{ "torch_dtype": "float16" }"#;
        assert_eq!(declared_dtype(raw), Some(DeclaredDtype::F16));
    }

    #[test]
    fn test_declared_dtype_is_none_when_the_config_says_nothing() {
        assert_eq!(declared_dtype(r#"{ "text_config": {} }"#), None);
        assert_eq!(declared_dtype("not json"), None);
    }

    #[test]
    fn test_prepare_passes_through_a_flat_config() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path().join("model");
        std::fs::create_dir_all(&dir).expect("mkdir");
        std::fs::write(dir.join("config.json"), FLAT_CONFIG).expect("write config");
        let prepared = prepare_mistralrs_model_dir(dir.to_str().unwrap()).expect("prepare");
        assert_eq!(prepared, dir.to_str().unwrap(), "no shadow dir for a flat config");
    }
}