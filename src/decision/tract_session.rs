//! Multi-input / multi-output tract session for the Laya decision graph.
//!
//! Unlike [`crate::vision::ort_backends::tract_backend::TractSession`] -- which
//! is a single f32 input, single f32 output, positionally fed -- a decision
//! graph takes five mixed-dtype inputs and returns two outputs, addressed by
//! name. That contract is different enough that extending the vision session
//! would make both worse, so this is a sibling rather than a reuse.
//!
//! Graph I/O, matching `laya/onnx_agent.py::ONNXAgent._infer`:
//!
//! | name | dtype | shape |
//! |---|---|---|
//! | `input_ids` | i64 | (n, L) |
//! | `attention_mask` | i64 | (n, L) |
//! | `marker_pos` | i64 | (n, kmax) |
//! | `marker_mask` | bool | (n, kmax) |
//! | `qtype` | i64 | (n,) |
//! | -> `logits` | f32 | (n, kmax) |
//! | -> `act_logits` | f32 | (n, 2) |
//!
//! The graph must have been prepped with `scripts/prep-laya-onnx.py` first:
//! tract cannot execute the symbolic shape/value_info subgraphs that
//! `onnx.shape_inference` leaves behind.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use ndarray::{Array1, Array2};
use tract_onnx::prelude::*;

use crate::config::DeviceSelection;
use crate::llm::{ChatResult, ModelInfo, NeuronError};

use super::postprocess::{self, QuestionType, Temperature, TemperatureByOptions};
use super::prompt::{self, EncodedQuestion, PromptConfig, SpecialIds};
use super::{DecisionRequest, DecisionResponse, DecisionRuntime, DecisionUsage};

/// One request's tensors, already collated.
pub struct DecisionBatch {
    pub input_ids: Array2<i64>,
    pub attention_mask: Array2<i64>,
    pub marker_pos: Array2<i64>,
    pub marker_mask: Array2<bool>,
    pub qtype: Array1<i64>,
    /// Per-row true option count, which is what the calibration indexes with.
    pub option_counts: Vec<usize>,
    /// Summed token count, reported as the request's prompt tokens.
    pub token_count: usize,
}

impl DecisionBatch {
    pub fn rows(&self) -> usize {
        self.qtype.len()
    }
}

/// Collate encoded questions the way `laya::common::collate_items` does: pad
/// `input_ids`/`attention_mask` to the longest row and `marker_pos` /
/// `marker_mask` to the widest option set.
///
/// `pad_id` must be the tokenizer's pad token, since the graph sees the padding.
pub fn collate(items: &[EncodedQuestion], pad_id: i64) -> Result<DecisionBatch, NeuronError> {
    collate_to(items, pad_id, None, None)
}

/// [`collate`], but pad to the extents the graph declares instead of to the
/// longest row in the batch.
///
/// This matters because a prepped graph has *grounded* dims: `--seq 64
/// --markers 5` produces `input_ids: i64[1, 64]` and `marker_pos: i64[1, 5]`,
/// and tract rejects a feed of any other width rather than padding it. Passing
/// the declared extents here makes a short question pad out to the graph
/// instead of failing at `run`.
///
/// A question that is *longer* than a declared extent is an error, not a
/// truncation: dropping tokens or options would silently mis-score it.
pub fn collate_to(
    items: &[EncodedQuestion],
    pad_id: i64,
    seq: Option<usize>,
    kmax: Option<usize>,
) -> Result<DecisionBatch, NeuronError> {
    if items.is_empty() {
        return Err(NeuronError::invalid_input("decision batch is empty"));
    }
    let n = items.len();
    let natural_l = items.iter().map(|i| i.seq_len).max().unwrap_or(0);
    let natural_k = items.iter().map(|i| i.marker_pos.len()).max().unwrap_or(0);
    let l = seq.unwrap_or(natural_l);
    let kmax = kmax.unwrap_or(natural_k);

    if natural_l > l {
        return Err(NeuronError::invalid_input(format!(
            "encoded question length {natural_l} exceeds the graph's sequence extent {l}"
        )));
    }
    if natural_k > kmax {
        return Err(NeuronError::invalid_input(format!(
            "encoded question has {natural_k} options, more than the graph's marker extent {kmax}"
        )));
    }

    let mut input_ids = Array2::from_elem((n, l), pad_id);
    let mut attention_mask = Array2::zeros((n, l));
    let mut marker_pos = Array2::zeros((n, kmax));
    let mut marker_mask = Array2::from_elem((n, kmax), false);
    let mut qtype = Array1::zeros(n);
    let mut option_counts = Vec::with_capacity(n);
    let mut token_count = 0usize;

    for (r, it) in items.iter().enumerate() {
        if it.seq_len > l || it.input_ids.len() != it.seq_len {
            return Err(NeuronError::invalid_input("encoded question length mismatch"));
        }
        for (c, &id) in it.input_ids.iter().enumerate() {
            input_ids[[r, c]] = id;
            attention_mask[[r, c]] = 1;
        }
        token_count += it.seq_len;
        let k = it.marker_pos.len();
        if k > kmax || it.marker_mask.len() != k {
            return Err(NeuronError::invalid_input("encoded question marker mismatch"));
        }
        for (c, &m) in it.marker_pos.iter().enumerate() {
            marker_pos[[r, c]] = m;
            marker_mask[[r, c]] = it.marker_mask[c];
        }
        qtype[r] = it.qtype;
        option_counts.push(it.option_count);
    }

    Ok(DecisionBatch {
        input_ids,
        attention_mask,
        marker_pos,
        marker_mask,
        qtype,
        option_counts,
        token_count,
    })
}

/// Resolve where a decision model's artifacts live.
///
/// Order:
/// 1. `NEURON_DECISION_MODEL_PATH` -- an explicit `.onnx` file or its directory.
/// 2. `$NEURON_MODEL_DIRS` roots (colon separated), then `decision/{model_id}/`.
///
/// The directory must contain exactly one `*.onnx` (prepped) and a
/// `tokenizer.json`, and may contain `rl_agent_config.json`.
pub fn resolve_model_dir(model_id: &str) -> Option<PathBuf> {
    if let Ok(p) = std::env::var("NEURON_DECISION_MODEL_PATH") {
        let path = PathBuf::from(p);
        let dir = if path.is_dir() { path } else { path.parent()?.to_path_buf() };
        return Some(dir);
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
        let candidate = Path::new(root).join("decision").join(model_id);
        if candidate.is_dir() {
            return Some(candidate);
        }
    }
    None
}

/// The graph's declared output tensor names, in `graph.output` order.
///
/// tract discards these names when it loads the graph -- an output boundary
/// node keeps the producing ONNX *node's* name -- so the contract is restated
/// here and mapped onto `raw.outputs` positionally. An export that declares
/// its outputs in a different order needs this list reordered to match.
pub const OUTPUT_NAMES: [&str; 2] = ["logits", "act_logits"];

fn find_onnx(dir: &Path) -> Result<PathBuf, NeuronError> {
    let mut found: Vec<PathBuf> = Vec::new();
    let entries = std::fs::read_dir(dir).map_err(|e| {
        NeuronError::invalid_input(format!("read model dir {}: {e}", dir.display()))
    })?;
    for entry in entries.flatten() {
        let p = entry.path();
        if p.extension().and_then(|e| e.to_str()) == Some("onnx") {
            found.push(p);
        }
    }
    match found.len() {
        1 => Ok(found.remove(0)),
        0 => Err(NeuronError::invalid_input(format!(
            "no .onnx graph in {} -- run scripts/prep-laya-onnx.py first",
            dir.display()
        ))),
        n => Err(NeuronError::invalid_input(format!(
            "{n} .onnx graphs in {}; leave exactly one prepped graph",
            dir.display()
        ))),
    }
}

/// Temperatures loaded from `rl_agent_config.json`, already clamped.
#[derive(Debug, Clone)]
pub struct Calibration {
    pub temperature: Temperature,
    pub by_options: TemperatureByOptions,
    /// Entries that got clamped or replaced, for the startup warning.
    pub rejected: Vec<String>,
}

impl Default for Calibration {
    fn default() -> Self {
        Self {
            temperature: [1.0, 1.0, 1.0],
            by_options: TemperatureByOptions::new(),
            rejected: Vec::new(),
        }
    }
}

impl Calibration {
    /// Parse the two temperature tables out of a `rl_agent_config.json` blob.
    ///
    /// Mirrors `ONNXAgent.__init__`: every entry goes through
    /// [`postprocess::clamp_temperature`], and anything whose applied value
    /// differs from the raw one is recorded so the host can warn that the
    /// affected buckets are uncalibrated.
    pub fn from_config(cfg: &serde_json::Value) -> Self {
        let raw = cfg
            .get("temperature")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let mut temperature = [1.0f64, 1.0, 1.0];
        let mut rejected = Vec::new();
        for (i, t) in raw.iter().take(3).enumerate() {
            let applied = postprocess::clamp_temperature(t.as_f64().unwrap_or(f64::NAN));
            temperature[i] = applied;
            if t.as_f64() != Some(applied) {
                rejected.push(format!("temperature[{i}]={t} -> {applied}"));
            }
        }

        let mut by_options = TemperatureByOptions::new();
        if let Some(map) = cfg.get("temperature_by_options").and_then(|v| v.as_object()) {
            for (k, v) in map {
                let applied = postprocess::clamp_temperature(v.as_f64().unwrap_or(f64::NAN));
                by_options.insert(k.clone(), applied);
                if v.as_f64() != Some(applied) {
                    rejected.push(format!("{k}={v} -> {applied}"));
                }
            }
        }

        Self { temperature, by_options, rejected }
    }
}

fn read_config(dir: &Path) -> serde_json::Value {
    std::fs::read_to_string(dir.join("rl_agent_config.json"))
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or(serde_json::json!({}))
}

fn load_calibration(dir: &Path) -> Calibration {
    let cfg = read_config(dir);
    let c = Calibration::from_config(&cfg);
    if c.rejected.is_empty() && cfg.as_object().map(|o| o.is_empty()).unwrap_or(true) {
        tracing::debug!("decision: no rl_agent_config.json in {}; using unit temperatures", dir.display());
    }
    c
}

fn load_prompt_config(dir: &Path) -> PromptConfig {
    let cfg = read_config(dir);
    PromptConfig {
        max_len: cfg
            .get("max_len")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or(prompt::DEFAULT_MAX_LEN),
        head_max_len: cfg
            .get("head_max_len")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or(prompt::DEFAULT_HEAD_MAX_LEN),
    }
}

/// Static extents the graph declares for its inputs. `None` where a dim is
/// left symbolic and the caller may choose.
///
/// `prep-laya-onnx.py` grounds every dim it is told to (`--seq N --markers K`),
/// and the export itself hardcodes `act_logits` to `f32[1, 2]`. A feed that
/// disagrees with a grounded dim is *rejected* by tract rather than padded, so
/// the runtime has to know these extents and shape the batch to them.
#[derive(Debug, Clone, Copy, Default)]
pub struct GraphShapes {
    /// First dim shared by every rank-2 input.
    pub batch: Option<usize>,
    /// Sequence length, i.e. the last dim of `input_ids`.
    pub seq: Option<usize>,
    /// Option slots, i.e. the last dim of `marker_pos`.
    pub kmax: Option<usize>,
}

/// tract-backed decision runtime.
pub struct TractDecisionRuntime {
    model: Arc<TypedRunnableModel<TypedModel>>,
    tok: Arc<tokenizers::Tokenizer>,
    specials: SpecialIds,
    pad_id: i64,
    calibration: Calibration,
    prompt_cfg: PromptConfig,
    /// Slot of each named graph input, in the graph's declared order.
    input_slots: HashMap<String, usize>,
    /// Slot of each named graph output, in the graph's declared order.
    output_slots: HashMap<String, usize>,
    /// Declared static extents, read off the graph before optimization.
    shapes: GraphShapes,
    model_id: String,
}

impl TractDecisionRuntime {
    /// Build from an already-resolved directory. Separated from `load` so tests
    /// can point at a fixture without touching env vars.
    pub fn from_dir(model_id: &str, dir: &Path) -> Result<Self, NeuronError> {
        let onnx_path = find_onnx(dir)?;
        let tok_path = dir.join("tokenizer.json");
        if !tok_path.exists() {
            return Err(NeuronError::invalid_input(format!(
                "tokenizer.json missing from {}",
                dir.display()
            )));
        }

        let raw = tract_onnx::onnx()
            .model_for_path(&onnx_path)
            .map_err(|e| NeuronError::invalid_input(format!("tract load: {e}")))?;

        // Capture the name -> slot maps before anything touches the graph:
        // `into_optimized` renames the surviving nodes and `into_runnable`
        // hands back a plan that is not inspectable at all.
        //
        // Graph *inputs* come out named after their ONNX tensors, so those map
        // by name. Graph *outputs* do not: tract names a boundary node after
        // the producing ONNX *node* (`node__to_copy_147`), and the tensor name
        // (`logits`) is discarded. The output *order* is a property of the
        // graph boundary and survives optimization, so outputs are mapped
        // positionally against [`OUTPUT_NAMES`], which mirrors the export's
        // declared `graph.output` order and `onnx_agent.py::_infer`'s
        // `sess.run(["logits", "act_logits"], ...)`.
        let mut input_slots = HashMap::new();
        for (idx, outlet) in raw.inputs.clone().into_iter().enumerate() {
            input_slots.insert(raw.node(outlet.node).name.clone(), idx);
        }

        // Declared static extents, read before `into_optimized` renames things.
        // Only `input_ids` and `marker_pos` matter: the other three inputs
        // share their dims with those two.
        //
        // A dim the export left symbolic comes back as `None` here, and the
        // caller then pads to the batch's natural extent instead.
        let mut shapes = GraphShapes::default();
        for outlet in raw.inputs.clone() {
            let name = raw.node(outlet.node).name.clone();
            let fact = raw.outlet_fact(outlet).map_err(|e| {
                NeuronError::invalid_input(format!("tract input fact for {name}: {e}"))
            })?;
            let dims = fact.shape.as_concrete_finite().ok().flatten();
            let concrete = |i: usize| dims.as_ref().and_then(|d| d.get(i)).copied();
            match name.as_str() {
                "input_ids" => {
                    shapes.batch = concrete(0);
                    shapes.seq = concrete(1);
                }
                "marker_pos" => {
                    shapes.batch = shapes.batch.or(concrete(0));
                    shapes.kmax = concrete(1);
                }
                _ => {}
            }
        }
        if raw.outputs.len() != OUTPUT_NAMES.len() {
            return Err(NeuronError::invalid_input(format!(
                "decision graph declares {} output(s), expected {}: {:?}",
                raw.outputs.len(),
                OUTPUT_NAMES.len(),
                OUTPUT_NAMES
            )));
        }
        let mut output_slots = HashMap::new();
        for (idx, name) in OUTPUT_NAMES.iter().enumerate() {
            output_slots.insert(name.to_string(), idx);
        }

        let model = raw
            .into_optimized()
            .map_err(|e| NeuronError::invalid_input(format!("tract optimize: {e}")))?
            .into_runnable()
            .map_err(|e| NeuronError::invalid_input(format!("tract runnable: {e}")))?;

        let tok = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| NeuronError::invalid_input(format!("tokenizer load: {e}")))?;
        let specials = SpecialIds::from_tokenizer(&tok).map_err(NeuronError::invalid_input)?;
        let pad_id = tok
            .token_to_id("[PAD]")
            .or_else(|| tok.token_to_id("<pad>"))
            .map(|i| i as i64)
            .unwrap_or(0);

        Ok(Self {
            model: Arc::new(model),
            tok: Arc::new(tok),
            specials,
            pad_id,
            calibration: load_calibration(dir),
            prompt_cfg: load_prompt_config(dir),
            input_slots,
            output_slots,
            shapes,
            model_id: model_id.to_string(),
        })
    }

    /// Static extents the graph declares. `None` where a dim is symbolic.
    pub fn shapes(&self) -> GraphShapes {
        self.shapes
    }

    /// Temperature tables as loaded, after clamping.
    pub fn calibration(&self) -> &Calibration {
        &self.calibration
    }

    /// Tokenizer pad id the batch is padded with.
    pub fn pad_id(&self) -> i64 {
        self.pad_id
    }

    /// Encode one request into a collated batch, without running it.
    ///
    /// Upstream raises `question %r options exceed head_max_len` when
    /// `build_sequence` drops markers past `max_len`; the same condition is
    /// checked here so a truncated question cannot be silently mis-scored.
    pub fn encode(
        &self,
        req: &DecisionRequest,
    ) -> Result<(DecisionBatch, Vec<QuestionType>), NeuronError> {
        if req.questions.is_empty() {
            return Err(NeuronError::invalid_input("decision request has no questions"));
        }
        let mut cfg = self.prompt_cfg;
        if let Some(n) = req.max_len {
            cfg.max_len = n;
        }
        if let Some(n) = req.head_max_len {
            cfg.head_max_len = n;
        }

        let state_ids = {
            let text = prompt::serialize_state(&req.state)
                .replace(self.specials.mask_text.as_str(), " ");
            prompt::encode_text(&self.tok, &text).map_err(NeuronError::invalid_input)?
        };

        let mut items = Vec::with_capacity(req.questions.len());
        let mut qtypes = Vec::with_capacity(req.questions.len());
        for (i, q) in req.questions.iter().enumerate() {
            let expected = q.rendered_options()?.len();
            let enc = q
                .encode(&self.tok, &self.specials, &req.state, Some(&state_ids), &cfg)
                .map_err(|e| NeuronError::invalid_input(format!("question[{i}]: {e}")))?;
            if enc.marker_pos.len() != expected {
                return Err(NeuronError::invalid_input(format!(
                    "question[{i}] options exceed head_max_len={}",
                    cfg.head_max_len
                )));
            }
            items.push(enc);
            qtypes.push(q.qtype);
        }
        Ok((collate_to(&items, self.pad_id, self.shapes.seq, self.shapes.kmax)?, qtypes))
    }

    fn output_slot(&self, name: &str) -> Result<usize, NeuronError> {
        self.output_slots
            .get(name)
            .copied()
            .ok_or_else(|| {
                let mut have: Vec<&str> = self.output_slots.keys().map(|s| s.as_str()).collect();
                have.sort_unstable();
                NeuronError::invalid_input(format!(
                    "decision graph has no output named {name}; it exposes {have:?}"
                ))
            })
    }

    /// Run the collated batch and return raw `(logits, act_logits)` per row.
    ///
    /// Splits into runs of the graph's declared batch extent whenever that
    /// extent is static and disagrees with the batch width. The Laya export
    /// hardcodes `act_logits` to `f32[1, 2]`, so a prepped graph is batch-1
    /// and an N-question request has to be N runs, not one.
    pub fn forward(
        &self,
        batch: &DecisionBatch,
    ) -> Result<(Vec<Vec<f64>>, Vec<[f64; 2]>), NeuronError> {
        let n = batch.rows();
        let run_width = match self.shapes.batch {
            Some(b) if b > 0 && b != n => b,
            _ => n,
        };
        let mut logits = Vec::with_capacity(n);
        let mut acts = Vec::with_capacity(n);
        let mut start = 0usize;
        while start < n {
            let end = (start + run_width).min(n);
            let (l, a) = self.forward_rows(batch, start, end, run_width)?;
            logits.extend(l);
            acts.extend(a);
            start = end;
        }
        Ok((logits, acts))
    }

    /// One `run` over rows `start..end`, laid out at `run_width`.
    ///
    /// A short tail is padded by repeating its last row, because the graph
    /// only accepts exactly `run_width` rows; the duplicate results are
    /// dropped before returning.
    fn forward_rows(
        &self,
        batch: &DecisionBatch,
        start: usize,
        end: usize,
        run_width: usize,
    ) -> Result<(Vec<Vec<f64>>, Vec<[f64; 2]>), NeuronError> {
        let keep = end - start;
        let rows: Vec<usize> = (start..end)
            .chain(std::iter::repeat(end - 1).take(run_width - keep))
            .collect();
        let sub = DecisionBatch {
            input_ids: batch.input_ids.select(ndarray::Axis(0), &rows),
            attention_mask: batch.attention_mask.select(ndarray::Axis(0), &rows),
            marker_pos: batch.marker_pos.select(ndarray::Axis(0), &rows),
            marker_mask: batch.marker_mask.select(ndarray::Axis(0), &rows),
            qtype: batch.qtype.select(ndarray::Axis(0), &rows),
            option_counts: batch.option_counts[start..end].to_vec(),
            token_count: 0,
        };
        let n = run_width;
        let kmax = sub.marker_pos.shape()[1];

        let mut feeds: Vec<Option<TValue>> = (0..self.input_slots.len()).map(|_| None).collect();
        let mut push = |name: &str, value: Tensor| -> Result<(), NeuronError> {
            let slot = self.input_slots.get(name).ok_or_else(|| {
                let mut have: Vec<&str> = self.input_slots.keys().map(|s| s.as_str()).collect();
                have.sort_unstable();
                NeuronError::invalid_input(format!(
                    "decision graph has no input named {name}; it takes {have:?}"
                ))
            })?;
            if feeds[*slot].is_some() {
                return Err(NeuronError::invalid_input(format!("duplicate feed for {name}")));
            }
            feeds[*slot] = Some(value.into_tvalue());
            Ok(())
        };
        push("input_ids", sub.input_ids.clone().into_dyn().into())?;
        push("attention_mask", sub.attention_mask.clone().into_dyn().into())?;
        push("marker_pos", sub.marker_pos.clone().into_dyn().into())?;
        push("marker_mask", sub.marker_mask.clone().into_dyn().into())?;
        push("qtype", sub.qtype.clone().into_dyn().into())?;
        drop(push);

        let missing: Vec<usize> = feeds
            .iter()
            .enumerate()
            .filter(|(_, f)| f.is_none())
            .map(|(i, _)| i)
            .collect();
        if !missing.is_empty() {
            return Err(NeuronError::invalid_input(format!(
                "decision graph declares {} inputs but the runtime feeds 5 (unfed slots {missing:?})",
                feeds.len()
            )));
        }
        let feeds: TVec<TValue> = feeds.into_iter().map(|f| f.unwrap()).collect();

        let outs = self
            .model
            .run(feeds)
            .map_err(|e| NeuronError::invalid_input(format!("tract run: {e}")))?;

        let logits_slot = self.output_slot("logits")?;
        let act_slot = self.output_slot("act_logits")?;
        let logits_v = outs
            .get(logits_slot)
            .ok_or_else(|| NeuronError::invalid_input("decision graph returned no logits slot"))?
            .to_array_view::<f32>()
            .map_err(|e| NeuronError::invalid_input(format!("logits dtype: {e}")))?;
        let act_v = outs
            .get(act_slot)
            .ok_or_else(|| NeuronError::invalid_input("decision graph returned no act_logits slot"))?
            .to_array_view::<f32>()
            .map_err(|e| NeuronError::invalid_input(format!("act_logits dtype: {e}")))?;

        let mut logits = Vec::with_capacity(n);
        for r in 0..n {
            let mut row = Vec::with_capacity(kmax);
            for c in 0..kmax {
                row.push(logits_v[[r, c]] as f64);
            }
            logits.push(row);
        }
        let mut acts = Vec::with_capacity(n);
        for r in 0..n {
            acts.push([act_v[[r, 0]] as f64, act_v[[r, 1]] as f64]);
        }
        // Drop the duplicated rows the short tail was padded out with.
        logits.truncate(keep);
        acts.truncate(keep);
        Ok((logits, acts))
    }
}

#[async_trait]
impl DecisionRuntime for TractDecisionRuntime {
    async fn load(model: &str, _device: &DeviceSelection) -> ChatResult<Box<dyn DecisionRuntime>> {
        let dir = resolve_model_dir(model).ok_or_else(|| {
            NeuronError::not_initialized(format!(
                "Decision model dir not found for {model}; set NEURON_DECISION_MODEL_PATH"
            ))
        })?;
        let rt = Self::from_dir(model, &dir)?;
        for entry in &rt.calibration.rejected {
            tracing::warn!(
                "decision: checkpoint ships temperatures outside [{}, {}] which would distort \
                 confidence; clamping {entry}. Treat confidence from the affected buckets as uncalibrated.",
                postprocess::TEMP_MIN,
                postprocess::TEMP_MAX
            );
        }
        Ok(Box::new(rt))
    }

    async fn decide(&self, req: DecisionRequest) -> ChatResult<DecisionResponse> {
        let (batch, qtypes) = self.encode(&req)?;
        let (logits, acts) = self.forward(&batch)?;
        let usage = DecisionUsage::from_tokens(batch.token_count as u32);

        let mut data = Vec::with_capacity(req.questions.len());
        for (r, q) in req.questions.iter().enumerate() {
            let k = batch.option_counts[r];
            let mut result = super::score_question(
                qtypes[r],
                k,
                &logits[r][..k],
                Some(&acts[r]),
                &self.calibration.temperature,
                &self.calibration.by_options,
            );
            result.index = r;
            result.unpermute(q.option_order.as_deref())?;
            result.apply_answer(q)?;
            data.push(result);
        }
        Ok(DecisionResponse::new(self.model_id.clone(), data, usage))
    }

    fn list_models(&self) -> Vec<ModelInfo> {
        vec![ModelInfo::new(self.model_id.clone(), "decision")]
    }

    fn name(&self) -> &str {
        "tract"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision::prompt::EncodedQuestion;

    fn enc(ids: &[i64], markers: &[i64], qtype: i64) -> EncodedQuestion {
        EncodedQuestion {
            input_ids: ids.to_vec(),
            attention_mask: vec![1; ids.len()],
            marker_pos: markers.to_vec(),
            marker_mask: vec![true; markers.len()],
            qtype,
            option_count: markers.len(),
            seq_len: ids.len(),
        }
    }

    #[test]
    fn test_collate_pads_ids_and_markers_independently() {
        let b = collate(&[enc(&[1, 2, 3], &[1, 2], 0), enc(&[4, 5], &[1], 1)], 0).unwrap();
        assert_eq!(b.input_ids.shape(), &[2, 3]);
        assert_eq!(b.marker_pos.shape(), &[2, 2]);
        // short row is padded with pad_id and masked out
        assert_eq!(b.input_ids[[1, 2]], 0);
        assert_eq!(b.attention_mask[[1, 2]], 0);
        // the 1-option row's second marker slot is zeroed and unmasked
        assert_eq!(b.marker_pos[[1, 1]], 0);
        assert!(!b.marker_mask[[1, 1]]);
        assert!(b.marker_mask[[1, 0]]);
        assert_eq!(b.option_counts, vec![2, 1]);
        assert_eq!(b.token_count, 5);
    }

    #[test]
    fn test_collate_uses_pad_id_for_short_rows() {
        let b = collate(&[enc(&[7], &[], 2), enc(&[7, 8, 9], &[1], 2)], 99).unwrap();
        assert_eq!(b.input_ids[[0, 1]], 99);
        assert_eq!(b.input_ids[[0, 2]], 99);
    }

    #[test]
    fn test_collate_rejects_empty_batch() {
        assert!(collate(&[], 0).is_err());
    }

    #[test]
    fn test_collate_to_pads_out_to_declared_extents() {
        // Natural extents are 3 x 2; the graph declares 5 x 4.
        let b = collate_to(&[enc(&[1, 2, 3], &[1, 2], 0)], 0, Some(5), Some(4)).unwrap();
        assert_eq!(b.input_ids.shape(), &[1, 5]);
        assert_eq!(b.marker_pos.shape(), &[1, 4]);
        // The tail is pad_id and masked out, exactly like a short batch row.
        assert_eq!(b.input_ids[[0, 3]], 0);
        assert_eq!(b.input_ids[[0, 4]], 0);
        assert_eq!(b.attention_mask[[0, 3]], 0);
        assert!(!b.marker_mask[[0, 2]]);
        assert!(!b.marker_mask[[0, 3]]);
        assert!(b.marker_mask[[0, 1]]);
        // The reported option count is the question's, not the padded width.
        assert_eq!(b.option_counts, vec![2]);
    }

    #[test]
    fn test_collate_to_rejects_sequence_longer_than_the_graph() {
        let err = collate_to(&[enc(&[1, 2, 3, 4], &[1], 0)], 0, Some(3), None)
            .err()
            .expect("an over-long question must not be truncated");
        assert!(
            err.message.contains("exceeds the graph's sequence extent 3"),
            "{}",
            err.message
        );
    }

    #[test]
    fn test_collate_to_rejects_more_options_than_the_graph() {
        let err = collate_to(&[enc(&[1, 2], &[1, 2, 3], 0)], 0, None, Some(2))
            .err()
            .expect("dropping options would silently mis-score the question");
        assert!(
            err.message.contains("more than the graph's marker extent 2"),
            "{}",
            err.message
        );
    }

    #[test]
    fn test_collate_to_with_no_extents_matches_collate() {
        let items = [enc(&[1, 2, 3], &[1, 2], 0), enc(&[4, 5], &[1], 1)];
        let a = collate(&items, 7).unwrap();
        let b = collate_to(&items, 7, None, None).unwrap();
        assert_eq!(a.input_ids, b.input_ids);
        assert_eq!(a.marker_pos, b.marker_pos);
        assert_eq!(a.qtype, b.qtype);
        assert_eq!(a.option_counts, b.option_counts);
    }

    #[test]
    fn test_calibration_defaults_to_unit_temperatures() {
        let c = Calibration::from_config(&serde_json::json!({}));
        assert_eq!(c.temperature, [1.0, 1.0, 1.0]);
        assert!(c.by_options.is_empty());
    }

    #[test]
    fn test_calibration_reads_both_tables_and_clamps() {
        let c = Calibration::from_config(&serde_json::json!({
            "temperature": [0.8, 1.35, 2.25],
            "temperature_by_options": {"choice:2": 0.1006, "choice:11+": 9.0}
        }));
        assert_eq!(c.temperature, [0.8, 1.35, 2.25]);
        // 0.1006 clamps up to TEMP_MIN, 9.0 clamps down to TEMP_MAX
        assert_eq!(c.by_options["choice:2"], 0.5);
        assert_eq!(c.by_options["choice:11+"], 5.0);
        assert_eq!(c.rejected.len(), 2);
    }

    #[test]
    fn test_calibration_records_non_numeric_as_rejected() {
        let c = Calibration::from_config(&serde_json::json!({"temperature": ["x", 1.0, 1.0]}));
        assert_eq!(c.temperature[0], 1.0);
        assert_eq!(c.rejected.len(), 1);
    }

    #[test]
    #[serial_test::serial]
    fn test_resolve_model_dir_reads_explicit_file_path() {
        let tmp = tempfile::tempdir().unwrap();
        // std::env::set_var is unsafe on edition 2024
        unsafe {
            std::env::set_var("NEURON_DECISION_MODEL_PATH", tmp.path().join("model.onnx"));
        }
        let got = resolve_model_dir("ignored").unwrap();
        unsafe {
            std::env::remove_var("NEURON_DECISION_MODEL_PATH");
        }
        assert_eq!(got, tmp.path().to_path_buf());
    }

    #[test]
    fn test_find_onnx_requires_exactly_one_graph() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(find_onnx(tmp.path()).is_err());
        std::fs::write(tmp.path().join("a.onnx"), b"x").unwrap();
        assert!(find_onnx(tmp.path()).is_ok());
        std::fs::write(tmp.path().join("b.onnx"), b"x").unwrap();
        assert!(find_onnx(tmp.path()).is_err());
    }

    #[test]
    fn test_from_dir_requires_tokenizer() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("model.onnx"), b"not really onnx").unwrap();
        let err = TractDecisionRuntime::from_dir("m", tmp.path())
            .err()
            .expect("must fail without a tokenizer");
        assert!(err.message.contains("tokenizer.json"), "{}", err.message);
    }
}
