//! LoRA (Low-Rank Adaptation) for YOLO-style vision models.
//!
//! Allows dynamically loading adapters that extend a base detection model
//! (e.g. YOLOv8n trained on COCO 80 classes) with new classes (e.g.
//! "warehouse pallet", "forklift", "package") without retraining the base
//! model. Adapters are safetensors files containing rank-r LoRA matrices
//! for selected conv layers + a metadata block describing the new classes.
//!
//! ## Workflow
//!
//! 1. User trains LoRA adapter offline (Python + peft) on a custom dataset.
//! 2. Adapter saved to a directory with `metadata.json` + `lora.safetensors`.
//! 3. `POST /v1/vision/lora/register` with the path → server loads into memory.
//! 4. `POST /v1/vision/detect` with `lora_adapters: ["warehouse-v1"]` →
//!    server merges adapter weights into the base model before forward pass.
//! 5. `DELETE /v1/vision/lora/{id}` to remove.
//!
//! ## Class ID allocation
//!
//! Base COCO classes occupy IDs `[0, 80)`. LoRA-added classes are appended
//! in the order adapters are registered: e.g. first LoRA with 3 classes
//! gets IDs `[80, 83)`, second with 5 gets `[83, 88)`, etc. Allocation is
//! stable across the lifetime of a `LoraRegistry` — once an id is assigned,
//! it doesn't move, even if other adapters are removed.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::llm::{ChatResult, NeuronError};

/// Status of a registered LoRA adapter.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LoraAdapterStatus {
    Loaded,
    Merged,
    Unloaded,
    Error(String),
}

/// Metadata for one LoRA adapter. Persisted as `metadata.json` next to
/// the safetensors weight file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraAdapterMeta {
    pub id: String,
    pub name: String,
    pub base_model: String,
    pub rank: u32,
    pub alpha: f32,
    pub target_modules: Vec<String>,
    pub custom_classes: Vec<String>,
    /// Inclusive start of the class_id range assigned to this adapter.
    pub class_id_start: u32,
    /// Exclusive end of the class_id range.
    pub class_id_end: u32,
    pub size_bytes: u64,
    pub loaded_at: DateTime<Utc>,
    pub status: LoraAdapterStatus,
}

/// One LoRA tensor — the A or B matrix for a single target module.
///
/// Convention: shape is `[out_dim, in_dim]` (PyTorch convention).
/// For a conv layer, `in_dim = in_channels * kernel_size`, `out_dim = out_channels * kernel_size`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraTensor {
    pub name: String,
    pub in_dim: u32,
    pub out_dim: u32,
    pub rank: u32,
    pub data: Vec<f32>,
}

/// Full LoRA adapter (metadata + tensors).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraAdapter {
    #[serde(flatten)]
    pub meta: LoraAdapterMeta,
    pub tensors: Vec<LoraTensor>,
}

/// In-memory state of the LoRA registry.
///
/// Thread-safe: uses `RwLock` so reads (lookups during detect) are fast
/// and writes (register / unregister) are rare.
pub struct LoraRegistry {
    by_id: RwLock<HashMap<String, Arc<LoraAdapter>>>,
    /// Next available class_id (monotonically increasing).
    next_class_id: RwLock<u32>,
}

impl LoraRegistry {
    pub fn new() -> Self {
        Self {
            by_id: RwLock::new(HashMap::new()),
            // 80 = COCO class count; LoRA classes start at 80.
            next_class_id: RwLock::new(80),
        }
    }

    /// Register a new LoRA adapter. Returns the assigned class_id range.
    pub fn register(&self, adapter: LoraAdapter) -> Result<(u32, u32), NeuronError> {
        let mut by_id = self.by_id.write().unwrap();
        if by_id.contains_key(&adapter.meta.id) {
            return Err(NeuronError::invalid_input(format!(
                "LoraAdapterExists: id={}",
                adapter.meta.id
            )));
        }
        let mut next = self.next_class_id.write().unwrap();
        let n_classes = adapter.meta.custom_classes.len() as u32;
        let start = *next;
        let end = start + n_classes;
        *next = end;
        let mut meta = adapter.meta;
        meta.class_id_start = start;
        meta.class_id_end = end;
        meta.status = LoraAdapterStatus::Loaded;
        let adapter = LoraAdapter { meta, tensors: adapter.tensors };
        by_id.insert(adapter.meta.id.clone(), Arc::new(adapter));
        Ok((start, end))
    }

    /// Unregister a LoRA adapter by id. The class_id range is NOT reclaimed
    /// (stability guarantee).
    pub fn unregister(&self, id: &str) -> Option<Arc<LoraAdapter>> {
        self.by_id.write().unwrap().remove(id)
    }

    /// Look up a single adapter by id.
    pub fn get(&self, id: &str) -> Option<Arc<LoraAdapter>> {
        self.by_id.read().unwrap().get(id).cloned()
    }

    /// Look up multiple adapters by id. Returns an error if any id is missing.
    pub fn get_many(&self, ids: &[String]) -> Result<Vec<Arc<LoraAdapter>>, NeuronError> {
        let by_id = self.by_id.read().unwrap();
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            let adapter = by_id.get(id).ok_or_else(|| {
                NeuronError::invalid_input(format!("LoraAdapterNotFound: id={id}"))
            })?;
            out.push(adapter.clone());
        }
        Ok(out)
    }

    /// List all registered adapters.
    pub fn list(&self) -> Vec<Arc<LoraAdapter>> {
        self.by_id.read().unwrap().values().cloned().collect()
    }

    /// Total number of base + LoRA classes.
    pub fn num_classes(&self) -> u32 {
        *self.next_class_id.read().unwrap()
    }

    /// Look up the class name for an arbitrary class_id, searching across
    /// base COCO classes (0..80) and all registered LoRA adapters.
    ///
    /// Returns the class name and the LoRA id (None for base COCO classes).
    pub fn class_name(&self, class_id: u32) -> (String, Option<String>) {
        if class_id < 80 {
            return (crate::vision::coco_classes::class_name(class_id), None);
        }
        // Search registered adapters
        let by_id = self.by_id.read().unwrap();
        for adapter in by_id.values() {
            let start = adapter.meta.class_id_start;
            let end = adapter.meta.class_id_end;
            if class_id >= start && class_id < end {
                let idx = (class_id - start) as usize;
                if let Some(name) = adapter.meta.custom_classes.get(idx) {
                    return (name.clone(), Some(adapter.meta.id.clone()));
                }
            }
        }
        (format!("class_{class_id}"), None)
    }
}

impl Default for LoraRegistry {
    fn default() -> Self { Self::new() }
}

/// HTTP request DTO for `POST /v1/vision/lora/register`.
///
/// `weight_path` is a path on the server's filesystem. The server reads
/// the safetensors file, validates the metadata, and stores the adapter
/// in memory. The file is not modified.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraRegisterRequest {
    pub id: String,
    pub name: String,
    pub base_model: String,
    pub weight_path: String,
    /// Class names added by this adapter. Length determines the size of
    /// the classification head expansion.
    pub custom_classes: Vec<String>,
}

/// HTTP response DTO for `POST /v1/vision/lora/register`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraRegisterResponse {
    pub id: String,
    pub status: LoraAdapterStatus,
    pub class_id_start: u32,
    pub class_id_end: u32,
    pub size_bytes: u64,
}

/// HTTP response DTO for `GET /v1/vision/lora/list`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraListResponse {
    pub adapters: Vec<LoraAdapterMeta>,
}

/// Load a LoRA adapter from a directory on disk.
///
/// Expected layout:
/// ```text
/// <weight_path>/
///   metadata.json    (LoraAdapterMeta without class_id_start/end/loaded_at/status)
///   lora.safetensors (key format: lora_A.<module_path>, lora_B.<module_path>)
/// ```
pub fn load_lora_from_path<P: AsRef<Path>>(
    weight_path: P,
    request: &LoraRegisterRequest,
) -> ChatResult<LoraAdapter> {
    let path = weight_path.as_ref();
    let meta_path = path.join("metadata.json");
    let weights_path = path.join("lora.safetensors");

    if !meta_path.exists() {
        return Err(NeuronError::invalid_input(format!(
            "LoraMetadataNotFound: {}",
            meta_path.display()
        )));
    }
    if !weights_path.exists() {
        return Err(NeuronError::invalid_input(format!(
            "LoraWeightsNotFound: {}",
            weights_path.display()
        )));
    }

    // Read metadata.json
    let meta_bytes = std::fs::read(&meta_path).map_err(|e| {
        NeuronError::invalid_input(format!("failed to read metadata: {e}"))
    })?;
    let mut meta: LoraAdapterMeta = serde_json::from_slice(&meta_bytes).map_err(|e| {
        NeuronError::invalid_input(format!("invalid metadata.json: {e}"))
    })?;
    meta.id = request.id.clone();
    meta.name = request.name.clone();
    meta.base_model = request.base_model.clone();
    meta.custom_classes = request.custom_classes.clone();
    meta.size_bytes = std::fs::metadata(&weights_path)
        .map(|m| m.len())
        .unwrap_or(0);

    // Read safetensors weights — defer to a future commit when YOLOv8
    // forward pass is implemented. v1.0 stores metadata only.
    let tensors = parse_safetensors_lora(&weights_path)?;

    Ok(LoraAdapter { meta, tensors })
}

/// Parse a LoRA safetensors file into a Vec<LoraTensor>.
///
/// Tensor key convention:
/// - `lora_A.<module_path>` — A matrix, shape `[rank, in_dim]`
/// - `lora_B.<module_path>` — B matrix, shape `[out_dim, rank]`
///
/// Delegates to `lora_weights::parse_safetensors_lora` which does the
/// actual byte parsing + tensor decoding.
#[cfg(feature = "candle")]
pub fn parse_safetensors_lora(path: &Path) -> ChatResult<Vec<LoraTensor>> {
    super::lora_weights::parse_safetensors_lora(path)
        .map_err(|e| NeuronError::new(e.message))
}

/// LoRA safetensors parsing stub when the `candle` feature is off
/// (the `safetensors` crate isn't available). Returns an empty
/// tensor list; the rest of the LoRA infrastructure still works.
#[cfg(not(feature = "candle"))]
pub fn parse_safetensors_lora(_path: &Path) -> ChatResult<Vec<LoraTensor>> {
    Ok(Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_meta(id: &str, classes: Vec<&str>) -> LoraAdapterMeta {
        LoraAdapterMeta {
            id: id.into(),
            name: id.into(),
            base_model: "yolov8n".into(),
            rank: 8,
            alpha: 16.0,
            target_modules: vec!["backbone.c2f.0.cv1.conv".into()],
            custom_classes: classes.into_iter().map(String::from).collect(),
            class_id_start: 0,
            class_id_end: 0,
            size_bytes: 1024,
            loaded_at: Utc::now(),
            status: LoraAdapterStatus::Loaded,
        }
    }

    fn make_adapter(id: &str, classes: Vec<&str>) -> LoraAdapter {
        LoraAdapter {
            meta: make_meta(id, classes),
            tensors: vec![],
        }
    }

    #[test]
    fn test_lora_registry_register_allocates_class_ids() {
        let reg = LoraRegistry::new();
        let (start, end) = reg.register(make_adapter("v1", vec!["pallet", "forklift"]))
            .expect("register");
        assert_eq!(start, 80);
        assert_eq!(end, 82);
        assert_eq!(reg.num_classes(), 82);
    }

    #[test]
    fn test_lora_registry_class_ids_increment() {
        let reg = LoraRegistry::new();
        let (s1, e1) = reg.register(make_adapter("v1", vec!["a", "b"])).expect("register");
        let (s2, e2) = reg.register(make_adapter("v2", vec!["c", "d", "e"])).expect("register");
        assert_eq!((s1, e1), (80, 82));
        assert_eq!((s2, e2), (82, 85));
        assert_eq!(reg.num_classes(), 85);
    }

    #[test]
    fn test_lora_registry_duplicate_id_rejected() {
        let reg = LoraRegistry::new();
        reg.register(make_adapter("v1", vec!["a"])).expect("register");
        let result = reg.register(make_adapter("v1", vec!["b"]));
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("LoraAdapterExists"));
    }

    #[test]
    fn test_lora_registry_get_missing_returns_none() {
        let reg = LoraRegistry::new();
        assert!(reg.get("nope").is_none());
    }

    #[test]
    fn test_lora_registry_get_many_with_missing_errors() {
        let reg = LoraRegistry::new();
        reg.register(make_adapter("v1", vec!["a"])).expect("register");
        let result = reg.get_many(&["v1".into(), "missing".into()]);
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("LoraAdapterNotFound"));
    }

    #[test]
    fn test_lora_registry_unregister_keeps_class_id_allocation() {
        let reg = LoraRegistry::new();
        reg.register(make_adapter("v1", vec!["a", "b"])).expect("register");
        reg.register(make_adapter("v2", vec!["c"])).expect("register");
        reg.unregister("v1");
        // v1 is gone but its class IDs (80-82) are not reclaimed
        assert!(reg.get("v1").is_none());
        assert!(reg.get("v2").is_some());
        assert_eq!(reg.num_classes(), 83);
    }

    #[test]
    fn test_lora_registry_list_returns_all() {
        let reg = LoraRegistry::new();
        reg.register(make_adapter("a", vec!["x"])).expect("register");
        reg.register(make_adapter("b", vec!["y"])).expect("register");
        let list = reg.list();
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_lora_registry_default() {
        let reg = LoraRegistry::default();
        assert_eq!(reg.num_classes(), 80);
    }

    #[test]
    fn test_lora_registry_class_name_base_coco() {
        let reg = LoraRegistry::new();
        let (name, lora) = reg.class_name(0);
        assert_eq!(name, "person");
        assert!(lora.is_none());
        let (name, lora) = reg.class_name(79);
        assert_eq!(name, "toothbrush");
        assert!(lora.is_none());
    }

    #[test]
    fn test_lora_registry_class_name_lora_class() {
        let reg = LoraRegistry::new();
        reg.register(make_adapter("v1", vec!["pallet", "forklift"])).expect("register");
        let (name, lora) = reg.class_name(80);
        assert_eq!(name, "pallet");
        assert_eq!(lora, Some("v1".into()));
        let (name, lora) = reg.class_name(81);
        assert_eq!(name, "forklift");
        assert_eq!(lora, Some("v1".into()));
    }

    #[test]
    fn test_lora_registry_class_name_after_unregister_falls_back() {
        let reg = LoraRegistry::new();
        reg.register(make_adapter("v1", vec!["pallet"])).expect("register");
        // After unregister, class_id 80's class is gone but the id is not reclaimed
        reg.unregister("v1");
        let (name, lora) = reg.class_name(80);
        assert_eq!(name, "class_80");
        assert!(lora.is_none());
    }

    #[test]
    fn test_lora_registry_class_name_multiple_adapters() {
        let reg = LoraRegistry::new();
        reg.register(make_adapter("a", vec!["x", "y"])).expect("register");
        reg.register(make_adapter("b", vec!["z"])).expect("register");
        let (n0, l0) = reg.class_name(80);
        assert_eq!(n0, "x"); assert_eq!(l0, Some("a".into()));
        let (n1, l1) = reg.class_name(81);
        assert_eq!(n1, "y"); assert_eq!(l1, Some("a".into()));
        let (n2, l2) = reg.class_name(82);
        assert_eq!(n2, "z"); assert_eq!(l2, Some("b".into()));
    }
}
