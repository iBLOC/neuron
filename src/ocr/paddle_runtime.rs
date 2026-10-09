//! PP-OCR two-stage OCR pipeline (DB++ detection + SVTR-LCNet recognition)
//! served through tract ONNX sessions.
//!
//! Model directory layout (resolved via `NEURON_OCR_MODEL_PATH`, else
//! `<model_root>/ocr/<model_id>`):
//!
//! ```text
//!   det.onnx   (or det.prepped.onnx)  — detection graph, must be prepped
//!   rec.onnx   (or rec.prepped.onnx)  — recognition graph, must be prepped
//!   dict.txt   (or ppocrv5_dict.txt / ppocrv6_dict.txt)
//! ```
//!
//! "Prepped" = the export ran through `scripts/prep-ppocr-onnx.py`: tract's
//! TDim parser rejects Paddle2ONNX's dotted symbol names and `floor(...)`
//! shape hints. The rewrite is metadata-only; compute is untouched.
//!
//! tract compiles one plan per concrete input shape, and both PP-OCR graphs
//! have dynamic extents (det: H×W, rec: W). The runtime bounds the compile
//! cost with plan caches — det keyed by the (rounded-to-32) input size and
//! rec by 64px width buckets — so a workload over a handful of document
//! geometries pays each compile once.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;

#[cfg(feature = "tract")]
use tract_onnx::prelude::*;

use crate::config::DeviceSelection;
use crate::llm::{ChatResult, ModelInfo, NeuronError};
use crate::vision::{BBox, VisionImageSource};

use super::ctc::Vocab;
use super::db_postprocess::{extract_boxes, DbParams, DetBox};
use super::lang_route::{self, RecPack};
use super::preprocess::{
    decode_rgb, det_resize_dims, det_to_chw, rec_bucket_w, rec_needed_w, rec_to_chw,
};
use super::{OcrRequest, OcrResponse, OcrRuntime, OcrTextRegion};

#[cfg(feature = "tract")]
type RunnablePlan = Arc<TypedRunnableModel<TypedModel>>;

const REC_TARGET_H: u32 = 48;
/// Maximum plans (concrete input shapes) kept compiled per graph.
const PLAN_CACHE_CAP: usize = 8;
/// Default ort sessions per graph (`NEURON_OCR_SESSIONS` overrides).
const DEFAULT_SESSION_POOL: usize = 2;

fn executor_name(e: OcrExecutor) -> &'static str {
    match e {
        OcrExecutor::Tract => "tract",
        OcrExecutor::Ort => "ort",
    }
}

/// Build the executor-specific session for one graph. `pool_size` applies
/// to ort (its `run` takes `&mut`, so concurrent requests need one session
/// each); tract plans are `run(&self)` and share freely.
fn build_session_for(
    executor: OcrExecutor,
    path: &Path,
    pool_size: usize,
) -> Result<Arc<dyn OcrSession>, NeuronError> {
    match executor {
        #[cfg(feature = "tract")]
        OcrExecutor::Tract => Ok(Arc::new(TractPlanCache::new(path.to_path_buf()))),
        #[cfg(not(feature = "tract"))]
        OcrExecutor::Tract => Err(NeuronError::invalid_input(
            "tract executor selected but this build was compiled without the `tract` feature",
        )),
        #[cfg(feature = "ort")]
        OcrExecutor::Ort => Ok(Arc::new(super::ort_session::OrtGraphSession::open(
            path,
            pool_size,
        )?)),
        #[cfg(not(feature = "ort"))]
        OcrExecutor::Ort => Err(NeuronError::invalid_input(
            "ort executor selected but this build was compiled without the `ort` feature",
        )),
    }
}

/// Executor-agnostic forward for one PP-OCR graph: `[1, 3, h, w]` CHW f32
/// input in, flattened output + declared output shape out. Implemented by
/// [`TractPlanCache`] (per-shape compiled plans) and
/// [`super::ort_session::OrtGraphSession`] (native dynamic shapes).
pub trait OcrSession: Send + Sync {
    fn run(&self, chw: Vec<f32>, h: u32, w: u32)
        -> Result<(Vec<f32>, Vec<usize>), NeuronError>;
    fn name(&self) -> &'static str;
}

/// Which executor backs the two graphs. Selected at load time via
/// `NEURON_OCR_EXECUTOR` (`tract` | `ort`). Default is tract — pure Rust,
/// no external runtime binary — with `ort` as the explicit fast path
/// (onnxruntime runs these CNNs 5-12× faster than tract on CPU).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OcrExecutor {
    Tract,
    Ort,
}

/// Resolve the executor from an explicit request (the `NEURON_OCR_EXECUTOR`
/// env value, already read). Pure so tests can cover it without env mutation.
pub fn choose_executor(explicit: Option<&str>) -> Result<OcrExecutor, NeuronError> {
    match explicit.map(str::trim).filter(|s| !s.is_empty()) {
        Some("tract") => {
            #[cfg(feature = "tract")]
            {
                Ok(OcrExecutor::Tract)
            }
            #[cfg(not(feature = "tract"))]
            {
                Err(NeuronError::invalid_input(
                    "NEURON_OCR_EXECUTOR=tract but this build was compiled without the `tract` feature",
                ))
            }
        }
        Some("ort") => {
            #[cfg(feature = "ort")]
            {
                Ok(OcrExecutor::Ort)
            }
            #[cfg(not(feature = "ort"))]
            {
                Err(NeuronError::invalid_input(
                    "NEURON_OCR_EXECUTOR=ort but this build was compiled without the `ort` feature",
                ))
            }
        }
        Some(other) => Err(NeuronError::invalid_input(format!(
            "unknown NEURON_OCR_EXECUTOR '{other}' (want tract | ort)"
        ))),
        None => {
            #[cfg(feature = "tract")]
            {
                Ok(OcrExecutor::Tract)
            }
            #[cfg(all(not(feature = "tract"), feature = "ort"))]
            {
                Ok(OcrExecutor::Ort)
            }
            #[cfg(not(any(feature = "tract", feature = "ort")))]
            {
                Err(NeuronError::not_implemented(
                    "the PP-OCR pipeline requires the `tract` or `ort` feature",
                ))
            }
        }
    }
}

pub struct PaddleOcrRuntime {
    model_id: String,
    det: Arc<dyn OcrSession>,
    rec: Arc<dyn OcrSession>,
    executor: OcrExecutor,
    vocab: Arc<Vocab>,
    det_params: DbParams,
    det_max_side: u32,
    /// ort sessions per graph (tract ignores this).
    session_pool: usize,
    /// Optional `rec-<pack>/` language packs discovered in the model dir.
    packs: HashMap<String, RecPackAssets>,
    /// Pack (session, vocab) pairs, built lazily on the first routed request.
    loaded_packs: Mutex<HashMap<String, (Arc<dyn OcrSession>, Arc<Vocab>)>>,
}

/// One discovered language pack: a recognition graph + its matching vocab.
/// PP-OCR language packs are independent weights *and* dicts — never mix a
/// pack's graph with the default dict.
struct RecPackAssets {
    rec_path: PathBuf,
    dict_path: PathBuf,
}

/// Scan `<model_dir>/rec-<pack>/` subdirectories for optional language
/// packs. Each pack must hold a rec graph (`rec.onnx` / `rec.prepped.onnx`)
/// and a dict; incomplete directories are reported and skipped.
fn discover_rec_packs(dir: &Path) -> HashMap<String, RecPackAssets> {
    let mut packs = HashMap::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return packs;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(dir_name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let Some(pack) = dir_name.strip_prefix("rec-") else {
            continue;
        };
        if pack.is_empty() {
            continue;
        }
        let (rec_path, dict_path) = (
            find_file(&path, &["rec.prepped.onnx", "rec.onnx", "rec_infer.onnx"]),
            find_file(&path, &["dict.txt", "keys.txt"]),
        );
        match (rec_path, dict_path) {
            (Some(rec_path), Some(dict_path)) => {
                packs.insert(pack.to_string(), RecPackAssets { rec_path, dict_path });
            }
            _ => eprintln!(
                "[neuron] ocr: language pack dir {} lacks a rec graph or dict; ignored",
                path.display()
            ),
        }
    }
    packs
}

/// tract-backed session: compiles (and FIFO-caches) one optimized plan per
/// concrete (h, w) input, since both PP-OCR graphs declare dynamic extents.
#[cfg(feature = "tract")]
pub struct TractPlanCache {
    model_path: PathBuf,
    plans: Mutex<HashMap<(u32, u32), RunnablePlan>>,
    order: Mutex<VecDeque<(u32, u32)>>,
}

#[cfg(feature = "tract")]
impl TractPlanCache {
    pub fn new(model_path: PathBuf) -> Self {
        Self {
            model_path,
            plans: Mutex::new(HashMap::new()),
            order: Mutex::new(VecDeque::new()),
        }
    }

    /// Compile (or fetch) the plan for a concrete `[1, 3, h, w]` input.
    /// Compilation happens under the map lock — deliberate: it serializes
    /// concurrent first-hits onto the same shape, which is cheaper than
    /// duplicating the 0.3-2 s optimize pass. The lock is never held across
    /// an await (the runtime's forward passes are fully synchronous).
    fn plan_for(&self, h: u32, w: u32) -> Result<RunnablePlan, NeuronError> {
        let mut plans = self.plans.lock().unwrap();
        if let Some(plan) = plans.get(&(h, w)) {
            return Ok(plan.clone());
        }
        let model = tract_onnx::onnx()
            .model_for_path(&self.model_path)
            .map_err(|e| {
                NeuronError::invalid_input(format!(
                    "tract load {} (is it prepped with scripts/prep-ppocr-onnx.py?): {e}",
                    self.model_path.display()
                ))
            })?
            .with_input_fact(0, f32::fact([1usize, 3, h as usize, w as usize]).into())
            .map_err(|e| NeuronError::invalid_input(format!("tract input fact: {e}")))?
            .into_optimized()
            .map_err(|e| NeuronError::invalid_input(format!("tract optimize: {e}")))?
            .into_runnable()
            .map_err(|e| NeuronError::invalid_input(format!("tract runnable: {e}")))?;
        let plan: RunnablePlan = Arc::new(model);

        let mut order = self.order.lock().unwrap();
        if order.len() >= PLAN_CACHE_CAP {
            if let Some(evict) = order.pop_front() {
                plans.remove(&evict);
            }
        }
        order.push_back((h, w));
        plans.insert((h, w), plan.clone());
        Ok(plan)
    }
}

#[cfg(feature = "tract")]
impl OcrSession for TractPlanCache {
    fn run(
        &self,
        chw: Vec<f32>,
        h: u32,
        w: u32,
    ) -> Result<(Vec<f32>, Vec<usize>), NeuronError> {
        let plan = self.plan_for(h, w)?;
        let arr = ndarray::ArrayD::<f32>::from_shape_vec(
            ndarray::IxDyn(&[1usize, 3, h as usize, w as usize]),
            chw,
        )
        .map_err(|e| NeuronError::invalid_input(format!("det input shape: {e}")))?;
        let outputs = plan
            .run(tvec![arr.into_tvalue()])
            .map_err(|e| NeuronError::new(format!("tract run: {e}")))?;
        let view = outputs[0]
            .to_array_view::<f32>()
            .map_err(|e| NeuronError::new(format!("tract extract: {e}")))?;
        let shape: Vec<usize> = view.shape().to_vec();
        Ok((view.iter().copied().collect(), shape))
    }

    fn name(&self) -> &'static str {
        "tract"
    }
}

impl PaddleOcrRuntime {
    pub async fn load(model_id: &str, _device: &DeviceSelection) -> ChatResult<Box<dyn OcrRuntime>> {
        let dir = resolve_model_dir(model_id).ok_or_else(|| {
            NeuronError::not_initialized(format!(
                "OCR model '{model_id}' not found: set NEURON_OCR_MODEL_PATH to a directory \
                 containing det.onnx / rec.onnx / dict.txt (all prepped with \
                 scripts/prep-ppocr-onnx.py)"
            ))
        })?;
        let det_path = find_file(
            &dir,
            &["det.prepped.onnx", "det.onnx", "det_infer.onnx", "inference_det.onnx"],
        )
        .ok_or_else(|| {
            NeuronError::invalid_input(format!(
                "no detection graph in {} (want det.onnx / det.prepped.onnx)",
                dir.display()
            ))
        })?;
        let rec_path = find_file(
            &dir,
            &["rec.prepped.onnx", "rec.onnx", "rec_infer.onnx", "inference_rec.onnx"],
        )
        .ok_or_else(|| {
            NeuronError::invalid_input(format!(
                "no recognition graph in {} (want rec.onnx / rec.prepped.onnx)",
                dir.display()
            ))
        })?;
        let dict_path = find_file(
            &dir,
            &["dict.txt", "ppocrv5_dict.txt", "ppocrv6_dict.txt", "keys.txt"],
        )
        .ok_or_else(|| {
            NeuronError::invalid_input(format!(
                "no recognition vocab in {} (want dict.txt / ppocrv5_dict.txt)",
                dir.display()
            ))
        })?;

        let vocab = Arc::new(Vocab::load(&dict_path)?);
        let executor = choose_executor(crate::config::env_var("OCR_EXECUTOR").as_deref())?;
        let session_pool = crate::config::env_var("OCR_SESSIONS")
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|n| *n > 0)
            .unwrap_or(DEFAULT_SESSION_POOL);
        let packs = discover_rec_packs(&dir);
        if !packs.is_empty() {
            let mut names: Vec<&str> = packs.keys().map(|s| s.as_str()).collect();
            names.sort_unstable();
            tracing::info!(model_id, packs = ?names, "ocr: discovered recognition language packs");
        }
        let det = build_session_for(executor, &det_path, session_pool)?;
        let rec = build_session_for(executor, &rec_path, session_pool)?;
        tracing::info!(
            model_id,
            dir = %dir.display(),
            vocab = vocab.tokens.len(),
            executor = executor_name(executor),
            "ocr: loaded PP-OCR pipeline manifest"
        );
        Ok(Box::new(PaddleOcrRuntime {
            model_id: model_id.to_string(),
            det,
            rec,
            executor,
            vocab,
            det_params: DbParams::default(),
            det_max_side: 960,
            session_pool,
            packs,
            loaded_packs: Mutex::new(HashMap::new()),
        }))
    }

    /// Resolve the (session, vocab, pack-name) triple for a request. Routed
    /// packs build lazily and cache; unrouted requests use the default rec
    /// with no locking.
    fn resolve_rec(
        &self,
        languages: Option<&Vec<String>>,
    ) -> Result<(Arc<dyn OcrSession>, Arc<Vocab>, String), NeuronError> {
        let pack = match languages {
            Some(langs) => {
                let known: Vec<String> = self.packs.keys().cloned().collect();
                lang_route::select_pack(langs, &known)
            }
            None => RecPack::Default,
        };
        let RecPack::Named(name) = pack else {
            return Ok((self.rec.clone(), self.vocab.clone(), "default".to_string()));
        };
        if let Some((session, vocab)) = self.loaded_packs.lock().unwrap().get(&name) {
            return Ok((session.clone(), vocab.clone(), name));
        }
        let assets = self.packs.get(&name).ok_or_else(|| {
            NeuronError::invalid_input(format!(
                "languages routed to pack '{name}' but no rec-{name}/ directory exists in \
                 the model dir"
            ))
        })?;
        let session = build_session_for(self.executor, &assets.rec_path, self.session_pool)?;
        let vocab = Arc::new(Vocab::load(&assets.dict_path)?);
        tracing::info!(
            pack = %name,
            vocab = vocab.tokens.len(),
            "ocr: loaded recognition language pack"
        );
        let mut loaded = self.loaded_packs.lock().unwrap();
        let (session, vocab) = loaded
            .entry(name.clone())
            .or_insert_with(|| (session, vocab));
        Ok((session.clone(), vocab.clone(), name))
    }
}

#[async_trait]
impl OcrRuntime for PaddleOcrRuntime {
    async fn load(model: &str, device: &DeviceSelection) -> ChatResult<Box<dyn OcrRuntime>>
    where
        Self: Sized,
    {
        PaddleOcrRuntime::load(model, device).await
    }

    async fn run(&self, req: OcrRequest) -> ChatResult<OcrResponse> {
        let start = Instant::now();

        let bytes = fetch_image(&req.image).await?;
        let (rgb, src_w, src_h) = decode_rgb(&bytes)?;

        // ---- Stage 1: detection ------------------------------------------
        let params = DbParams {
            thresh: req.det_thresh.unwrap_or(self.det_params.thresh),
            box_thresh: req.det_box_thresh.unwrap_or(self.det_params.box_thresh),
            unclip_ratio: req.det_unclip_ratio.unwrap_or(self.det_params.unclip_ratio),
            min_size: self.det_params.min_size,
            max_candidates: self.det_params.max_candidates,
        };
        let max_side = req.det_max_side.unwrap_or(self.det_max_side);
        let (det_w, det_h) = det_resize_dims(src_w, src_h, max_side);
        let det_chw = det_to_chw(&rgb, src_w, src_h, det_w, det_h)?;
        let (prob, prob_shape) = self.det.run(det_chw, det_h, det_w)?;
        let (map_h, map_w) = match prob_shape.as_slice() {
            [_, _, h, w] => (*h, *w),
            other => {
                return Err(NeuronError::new(format!(
                    "unexpected detection output shape {other:?} (want [1,1,H,W])"
                )))
            }
        };
        let scale_x = map_w as f32 / src_w as f32;
        let scale_y = map_h as f32 / src_h as f32;
        let boxes = extract_boxes(&prob, map_w, map_h, scale_x, scale_y, &params);

        // ---- Stage 2: recognition, line by line --------------------------
        let (rec_session, rec_vocab, pack_used) = self.resolve_rec(req.languages.as_ref())?;
        let mut regions: Vec<OcrTextRegion> = Vec::with_capacity(boxes.len());
        for (i, det_box) in boxes.iter().enumerate() {
            let (line, line_w, line_h) = crop_rgb(&rgb, src_w, src_h, det_box);
            let needed = rec_needed_w(line_w, line_h, REC_TARGET_H);
            let bucket = rec_bucket_w(needed);
            let rec_chw = rec_to_chw(&line, line_w, line_h, bucket, REC_TARGET_H)?;
            let (probs, shape) = rec_session.run(rec_chw, REC_TARGET_H, bucket)?;
            let (t, vocab) = match shape.as_slice() {
                [_, t, v] => (*t, *v),
                other => {
                    return Err(NeuronError::new(format!(
                        "unexpected recognition output shape {other:?} (want [1,T,vocab])"
                    )))
                }
            };
            if vocab != rec_vocab.tokens.len() {
                return Err(NeuronError::invalid_input(format!(
                    "vocab size {} does not match the recognition graph's {} outputs — \
                     wrong dict for this rec export (pack '{pack_used}')",
                    rec_vocab.tokens.len(),
                    vocab
                )));
            }
            let (text, confidence) = rec_vocab.decode(&probs, t, vocab);
            if text.is_empty() {
                continue;
            }
            regions.push(OcrTextRegion {
                bbox: BBox {
                    x: det_box.x0.max(0.0),
                    y: det_box.y0.max(0.0),
                    w: (det_box.x1 - det_box.x0).max(1.0),
                    h: (det_box.y1 - det_box.y0).max(1.0),
                },
                text,
                confidence,
                line: i,
            });
        }

        // ---- Stage 3: reading order --------------------------------------
        sort_reading_order(&mut regions);

        let text = regions
            .iter()
            .map(|r| r.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        for (line, region) in regions.iter_mut().enumerate() {
            region.line = line;
        }

        Ok(OcrResponse {
            model: self.model_id.clone(),
            image_size: (src_w, src_h),
            inference_time_ms: start.elapsed().as_millis() as u32,
            text,
            regions,
            // The recognition pack actually used ("default" or e.g. "latin").
            languages: vec![pack_used],
        })
    }

    fn list_models(&self) -> Vec<ModelInfo> {
        vec![ModelInfo::new(self.model_id.clone(), "ocr")]
    }

    fn name(&self) -> &str {
        "paddle-ocr"
    }
}

/// Fetch image bytes from a request source: base64 / data URL / http(s) URL,
/// plus server-local paths (`file://` URI or a bare existing path) which
/// vision's fetcher rejects — handy for the smoke example and hosts that
/// share a filesystem with neuron.
async fn fetch_image(source: &VisionImageSource) -> Result<Vec<u8>, NeuronError> {
    match source {
        VisionImageSource::Base64 { data, .. } => {
            use base64::Engine as _;
            base64::engine::general_purpose::STANDARD
                .decode(data)
                .map_err(|e| NeuronError::invalid_input(format!("invalid base64: {e}")))
        }
        VisionImageSource::ImageUrl { image_url } => {
            let url = image_url.url.trim();
            if let Some(rest) = url.strip_prefix("data:") {
                let (_, payload) = rest.split_once(',').ok_or_else(|| {
                    NeuronError::invalid_input("invalid data URL (missing ',')".to_string())
                })?;
                use base64::Engine as _;
                return base64::engine::general_purpose::STANDARD
                    .decode(payload)
                    .map_err(|e| NeuronError::invalid_input(format!("invalid data URL: {e}")));
            }
            if let Some(path) = url.strip_prefix("file://").map(PathBuf::from) {
                return std::fs::read(&path).map_err(|e| {
                    NeuronError::invalid_input(format!("cannot read {}: {e}", path.display()))
                });
            }
            let as_path = PathBuf::from(url);
            if as_path.is_file() {
                return std::fs::read(&as_path).map_err(|e| {
                    NeuronError::invalid_input(format!("cannot read {}: {e}", url))
                });
            }
            let resp = reqwest::get(url)
                .await
                .map_err(|e| NeuronError::invalid_input(format!("download failed: {e}")))?;
            let bytes = resp
                .bytes()
                .await
                .map_err(|e| NeuronError::invalid_input(format!("read failed: {e}")))?;
            Ok(bytes.to_vec())
        }
        VisionImageSource::FileId { .. } => Err(NeuronError::not_implemented(
            "file_id requires host-supplied storage".to_string(),
        )),
    }
}

/// Crop an axis-aligned box out of an RGB8 buffer, clamped to the image and
/// guaranteed non-empty.
pub fn crop_rgb(rgb: &[u8], w: u32, h: u32, b: &DetBox) -> (Vec<u8>, u32, u32) {
    let x0 = b.x0.floor().clamp(0.0, (w - 1) as f32) as u32;
    let y0 = b.y0.floor().clamp(0.0, (h - 1) as f32) as u32;
    let x1 = (b.x1.ceil() as u32).clamp(x0 + 1, w);
    let y1 = (b.y1.ceil() as u32).clamp(y0 + 1, h);
    let cw = x1 - x0;
    let ch = y1 - y0;
    let mut out = Vec::with_capacity(cw as usize * ch as usize * 3);
    for yy in y0..y1 {
        let row = ((yy as usize * w as usize) + x0 as usize) * 3;
        out.extend_from_slice(&rgb[row..row + cw as usize * 3]);
    }
    (out, cw, ch)
}

/// Order text regions the way a person reads: rows top-to-bottom,
/// left-to-right within a row. Rows are formed greedily — a region starts a
/// new row when its top edge sits below every bottom edge seen in the
/// current row — then each row is re-sorted by x, so same-row regions with
/// jittered y still land in reading order. This is a geometric heuristic
/// (no layout model); it handles the receipts / forms / screenshots that
/// make up the v1 use cases.
pub fn sort_reading_order(regions: &mut [OcrTextRegion]) {
    let n = regions.len();
    let mut by_y: Vec<usize> = (0..n).collect();
    by_y.sort_by(|&a, &b| {
        regions[a]
            .bbox
            .y
            .total_cmp(&regions[b].bbox.y)
    });

    let mut rows: Vec<Vec<usize>> = Vec::new();
    let mut row_bottom = f32::NEG_INFINITY;
    for &i in &by_y {
        let y0 = regions[i].bbox.y;
        if rows.is_empty() || y0 > row_bottom {
            rows.push(Vec::new());
            row_bottom = f32::NEG_INFINITY;
        }
        row_bottom = row_bottom.max(y0 + regions[i].bbox.h);
        rows.last_mut().unwrap().push(i);
    }

    let mut order: Vec<usize> = Vec::with_capacity(n);
    for row in &mut rows {
        row.sort_by(|&a, &b| regions[a].bbox.x.total_cmp(&regions[b].bbox.x));
        order.extend(row.iter().copied());
    }

    let mut slots: Vec<Option<OcrTextRegion>> = regions.iter().cloned().map(Some).collect();
    for (out_pos, src_idx) in order.into_iter().enumerate() {
        regions[out_pos] = slots[src_idx].take().expect("each index visited once");
    }
}

/// `NEURON_OCR_MODEL_PATH` wins (file or directory); else scan the standard
/// model roots for `<root>/ocr/<model_id>`.
pub fn resolve_model_dir(model_id: &str) -> Option<PathBuf> {
    if let Ok(p) = std::env::var("NEURON_OCR_MODEL_PATH") {
        let path = PathBuf::from(&p);
        return Some(if path.is_dir() { path } else { path.parent()?.to_path_buf() });
    }
    let roots = std::env::var("NEURON_MODEL_DIRS")
        .or_else(|_| std::env::var("NEURON_MODEL_DIR"))
        .unwrap_or_else(|_| {
            let home = std::env::var("USERPROFILE")
                .or_else(|_| std::env::var("HOME"))
                .unwrap_or_default();
            format!("{home}/.neuron/models")
        });
    for root in roots.split(':').filter(|s| !s.is_empty()) {
        let candidate = Path::new(root).join("ocr").join(model_id);
        if candidate.is_dir() {
            return Some(candidate);
        }
    }
    None
}

fn find_file(dir: &Path, names: &[&str]) -> Option<PathBuf> {
    for name in names {
        let p = dir.join(name);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discover_rec_packs_finds_complete_dirs_only() {
        let dir = std::env::temp_dir().join(format!("neuron-ocr-packs-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let latin = dir.join("rec-latin");
        std::fs::create_dir_all(&latin).unwrap();
        std::fs::write(latin.join("rec.prepped.onnx"), b"x").unwrap();
        std::fs::write(latin.join("dict.txt"), "A\nB\n").unwrap();
        // incomplete pack: no dict
        let broken = dir.join("rec-korean");
        std::fs::create_dir_all(&broken).unwrap();
        std::fs::write(broken.join("rec.onnx"), b"x").unwrap();
        // non-pack dirs are ignored
        std::fs::create_dir_all(dir.join("det")).unwrap();

        let packs = discover_rec_packs(&dir);
        assert!(packs.contains_key("latin"), "{:?}", packs.keys());
        assert!(!packs.contains_key("korean"));
        assert_eq!(packs["latin"].dict_path.file_name().unwrap(), "dict.txt");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn choose_executor_resolves_default_and_validates_names() {
        // Default: tract when compiled, else ort.
        let default = choose_executor(None).unwrap();
        #[cfg(feature = "tract")]
        assert_eq!(default, OcrExecutor::Tract);
        #[cfg(all(not(feature = "tract"), feature = "ort"))]
        assert_eq!(default, OcrExecutor::Ort);

        // Empty / whitespace values fall through to the default.
        assert_eq!(choose_executor(Some("")).unwrap(), default);
        assert_eq!(choose_executor(Some("  ")).unwrap(), default);

        // Unknown names are a clear input error.
        assert!(choose_executor(Some("tesseract")).is_err());

        // Explicit names only resolve when their feature is compiled in.
        #[cfg(feature = "tract")]
        assert_eq!(choose_executor(Some("tract")).unwrap(), OcrExecutor::Tract);
        #[cfg(feature = "ort")]
        assert_eq!(choose_executor(Some("ort")).unwrap(), OcrExecutor::Ort);
        #[cfg(not(feature = "ort"))]
        {
            let err = choose_executor(Some("ort")).err().unwrap();
            assert!(err.message.contains("`ort` feature"), "{err:?}");
        }
    }

    fn region(x: f32, y: f32, w: f32, h: f32, text: &str) -> OcrTextRegion {
        OcrTextRegion {
            bbox: BBox { x, y, w, h },
            text: text.into(),
            confidence: 0.9,
            line: 0,
        }
    }

    #[test]
    fn reading_order_groups_rows_then_sorts_by_x() {
        // visually: [B A] on row 1, [D C] on row 2 (x out of order).
        // Row 2 reading order is D (x=10) then C (x=100).
        let mut regions = vec![
            region(10.0, 100.0, 40.0, 20.0, "D"),
            region(150.0, 0.0, 40.0, 20.0, "B"),
            region(10.0, 0.0, 40.0, 20.0, "A"),
            region(100.0, 100.0, 40.0, 20.0, "C"),
        ];
        sort_reading_order(&mut regions);
        let texts: Vec<&str> = regions.iter().map(|r| r.text.as_str()).collect();
        assert_eq!(texts, vec!["A", "B", "D", "C"]);
    }

    #[test]
    fn reading_order_survives_same_row_y_jitter() {
        // Same row (overlapping bands), x order opposite to y order: the
        // in-row x sort must win.
        let mut regions = vec![
            region(40.0, 12.0, 30.0, 20.0, "a"),
            region(0.0, 10.0, 30.0, 20.0, "b"),
        ];
        sort_reading_order(&mut regions);
        assert_eq!(regions[0].text, "b");
        assert_eq!(regions[1].text, "a");
    }

    #[test]
    fn crop_rgb_clamps_and_copies_rows() {
        // 4x2 image, 2 distinct row colors; box covering x 1..3, all rows.
        let mut rgb = vec![1u8; 4 * 2 * 3];
        for x in 0..4 {
            for c in 0..3 {
                rgb[(x * 3 + c) as usize] = 10; // row 0
                rgb[((4 + x) * 3 + c) as usize] = 20; // row 1
            }
        }
        let b = DetBox { x0: -5.0, y0: 0.0, x1: 3.0, y1: 99.0, score: 1.0 };
        let (crop, cw, ch) = crop_rgb(&rgb, 4, 2, &b);
        assert_eq!((cw, ch), (3, 2));
        assert_eq!(crop[0], 10);
        assert_eq!(&crop[9..12], &[20, 20, 20]);
    }
}
