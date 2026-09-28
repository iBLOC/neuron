//! SenseVoice ASR executed by neuron's own ONNX stack (tract).
//!
//! This is the "run the ONNX ourselves" counterpart to
//! [`super::sherpa::SherpaAsrRuntime`]: the same sherpa-onnx exported
//! SenseVoiceSmall graph, the same audio, but every pipeline stage --
//! fbank, LFR stacking, CMVN, the tract forward, greedy CTC decode --
//! lives in this repo instead of inside sherpa-onnx / onnxruntime.
//! Its original purpose is the `benches/asr_speed.rs` comparison, but it
//! is a full runtime: `NEURON_ASR_RUNTIME=tract` serves
//! `/v1/audio/transcriptions` with no sherpa dependency at all.
//!
//! Preprocessing contract (read from the graph's ONNX `metadata_props`
//! at load time, mirroring what sherpa-onnx does):
//!
//! | stage | rule |
//! |---|---|
//! | input scaling | samples in [-1,1]; multiplied by 32768 when `normalize_samples = 0` |
//! | fbank | `super::fbank` (hamming, snip_edges, 80 bins, 20 Hz..=Nyquist) |
//! | LFR | stack `lfr_window_size` frames, stride `lfr_window_shift` -> 560-dim |
//! | CMVN | `(x + neg_mean) * inv_stddev`, stats from metadata, 560-dim |
//! | position encoding | computed in Rust (`positional_encoding`): the graph's dynamic-`Range` PE is replaced by a `pe` graph input by `scripts/prep-sensevoice-onnx.py` |
//! | graph | `x` f32[1,T,560] + `x_length`/`language`/`text_norm` i32[1] + `pe` f32[T+4,560] -> `logits` f32[1,T+4,25055] |
//! | decode | greedy CTC (argmax, skip blank, collapse repeats); rows 0..4 carry lang/emo/event/itn meta tokens |
//!
//! The fp32 export (`model.onnx`) is the default graph. The int8 export
//! loads too but runs 281 `DynamicQuantizeLinear` nodes, where tract's
//! quantization rounding (half-away-from-zero vs onnxruntime's
//! half-to-even) amplifies into percent-level logit divergence -- see
//! `scripts/prep-laya-onnx.py` step 4 for the same finding on the Laya
//! graphs. Point `NEURON_ASR_ONNX_FILE` at `model.int8.onnx` only if the
//! extra memory is worth the numeric drift.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use ndarray::{Array2, Array3};
use tract_onnx::prelude::*;

use crate::config::DeviceSelection;
use crate::llm::{ChatResult, NeuronError};

use super::fbank::fbank;
use super::onnx_meta::OnnxMetadata;
use super::wav::decode_wav;
use super::{AsrRuntime, Transcription};

/// SentencePiece "word boundary" marker, replaced with a real space in
/// the user-facing text.
const SP_SPACE: char = '\u{2581}';

/// Sinusoidal position encoding, replicating what the export's deleted
/// `/encoder/embed` chain computed (see `scripts/prep-sensevoice-onnx.py`,
/// which asserts the chain's constants before touching the graph):
///
/// ```text
///   positions p = 1..=t_enc            (Range(1, T_enc + 1, 1))
///   freq[i]     = 10000^(-i/(dim/2-1)) for i in 0..dim/2
///   pe[p]       = [ sin(p * freq) | cos(p * freq) ]   (concat on axis -1)
/// ```
///
/// The graph prepends 4 prefix embeddings, so callers pass
/// `t_enc = lfr_frames + 4`.
pub fn positional_encoding(t_enc: usize, dim: usize) -> Array2<f32> {
    let half = dim / 2;
    let scale = (10000.0f32).ln() / (half - 1) as f32;
    let freq: Vec<f32> = (0..half).map(|i| (-(i as f32) * scale).exp()).collect();
    let mut pe = Array2::<f32>::zeros((t_enc, dim));
    for (p, mut row) in pe.rows_mut().into_iter().enumerate() {
        let pos = (p + 1) as f32;
        for (i, &f) in freq.iter().enumerate() {
            let a = pos * f;
            row[i] = a.sin();
            row[half + i] = a.cos();
        }
    }
    pe
}

/// The graph's declared input names. `pe` is the position encoding, moved
/// out of the graph by `scripts/prep-sensevoice-onnx.py` because tract
/// cannot translate the export's dynamic `Range`-based PE computation.
const INPUT_NAMES: [&str; 5] = ["x", "x_length", "language", "text_norm", "pe"];

/// Preprocessing + decoding values recovered from the ONNX metadata map.
#[derive(Debug, Clone)]
pub struct SenseVoiceMeta {
    pub lfr_window: usize,
    pub lfr_shift: usize,
    pub neg_mean: Vec<f32>,
    pub inv_stddev: Vec<f32>,
    pub vocab_size: usize,
    pub blank_id: u64,
    /// Whether the model was trained on samples in [-1, 1] (true) or on
    /// int16-range samples (false -> this runtime scales by 32768 first).
    pub normalize_samples: bool,
    /// `(name, embedding index)` pairs, e.g. `("zh", 3)`; from `lang_*` keys.
    pub lang_ids: Vec<(String, i32)>,
    pub with_itn: i32,
    pub without_itn: i32,
    /// fbank dim, derived as `neg_mean.len() / lfr_window` (560 / 7 = 80).
    pub num_mel_bins: usize,
}

impl SenseVoiceMeta {
    pub fn from_metadata(meta: &OnnxMetadata) -> Result<Self, NeuronError> {
        let required_u64 = |key: &str| -> Result<u64, NeuronError> {
            let raw = meta.get(key).ok_or_else(|| {
                NeuronError::invalid_input(format!(
                    "ONNX metadata key {key:?} missing -- this does not look like a \
                     sherpa-onnx SenseVoice export"
                ))
            })?;
            raw.trim().parse::<u64>().map_err(|e| {
                NeuronError::invalid_input(format!("metadata {key}={raw:?}: {e}"))
            })
        };

        let lfr_window = required_u64("lfr_window_size")? as usize;
        let lfr_shift = required_u64("lfr_window_shift")?;
        let neg_mean = meta
            .get_f32_vec("neg_mean")
            .transpose()
            .map_err(NeuronError::invalid_input)?
            .ok_or_else(|| NeuronError::invalid_input("ONNX metadata key neg_mean missing"))?;
        let inv_stddev = meta
            .get_f32_vec("inv_stddev")
            .transpose()
            .map_err(NeuronError::invalid_input)?
            .ok_or_else(|| NeuronError::invalid_input("ONNX metadata key inv_stddev missing"))?;
        if neg_mean.is_empty() || neg_mean.len() != inv_stddev.len() {
            return Err(NeuronError::invalid_input(format!(
                "neg_mean ({} entries) and inv_stddev ({} entries) disagree",
                neg_mean.len(),
                inv_stddev.len()
            )));
        }
        if neg_mean.len() % lfr_window != 0 {
            return Err(NeuronError::invalid_input(format!(
                "CMVN length {} is not a multiple of lfr_window_size {lfr_window}",
                neg_mean.len()
            )));
        }

        // Language ids: optional except `lang_auto`, which is the default
        // request language and must exist on every SenseVoice export.
        let mut lang_ids: Vec<(String, i32)> = Vec::new();
        for key in ["lang_auto", "lang_zh", "lang_en", "lang_yue", "lang_ja", "lang_ko", "lang_nospeech"] {
            if let Some(v) = meta.get(key) {
                let id: i32 = v.trim().parse().map_err(|e| {
                    NeuronError::invalid_input(format!("metadata {key}={v:?}: {e}"))
                })?;
                lang_ids.push((key.trim_start_matches("lang_").to_string(), id));
            }
        }
        if !lang_ids.iter().any(|(n, _)| n == "auto") {
            return Err(NeuronError::invalid_input(
                "ONNX metadata has no lang_auto entry; cannot default the language input",
            ));
        }

        Ok(Self {
            lfr_window,
            lfr_shift: lfr_shift as usize,
            num_mel_bins: neg_mean.len() / lfr_window,
            neg_mean,
            inv_stddev,
            vocab_size: required_u64("vocab_size")? as usize,
            blank_id: meta
                .get_u64("blank_id")
                .transpose()
                .map_err(NeuronError::invalid_input)?
                .unwrap_or(0),
            normalize_samples: required_u64("normalize_samples").unwrap_or(1) == 1,
            lang_ids,
            with_itn: required_u64("with_itn")? as i32,
            without_itn: required_u64("without_itn")? as i32,
        })
    }

    /// Resolve a request language (`"auto"`, `"zh"`, ...) to the embedding
    /// index the graph's `language` input expects.
    pub fn language_id(&self, lang: Option<&str>) -> Result<i32, NeuronError> {
        let name = lang.unwrap_or("auto");
        self.lang_ids
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, id)| *id)
            .ok_or_else(|| {
                let have: Vec<&str> = self.lang_ids.iter().map(|(n, _)| n.as_str()).collect();
                NeuronError::invalid_input(format!(
                    "unknown SenseVoice language {name:?}; this export supports {have:?}"
                ))
            })
    }
}

/// Per-stage wall-clock splits of one transcription, for the benchmark.
#[derive(Debug, Clone, Copy, Default)]
pub struct SenseVoiceTimings {
    /// fbank + LFR + CMVN (feature extraction, CPU, Rust).
    pub feature_us: u128,
    /// The tract forward pass.
    pub forward_us: u128,
    /// Greedy CTC argmax decode over the logits.
    pub decode_us: u128,
}

pub struct TractSenseVoiceRuntime {
    model: Arc<TypedRunnableModel<TypedModel>>,
    sym_table: Arc<HashMap<u32, String>>,
    meta: SenseVoiceMeta,
    /// Slot of each named graph input, in the graph's declared order.
    input_slots: HashMap<String, usize>,
    model_id: String,
    /// Which graph file was loaded (fp32 vs int8 lineage), for diagnostics.
    pub onnx_file: PathBuf,
}

impl TractSenseVoiceRuntime {
    /// Build from an already-resolved model directory. Separated from
    /// `load` so tests and the benchmark can point at a directory without
    /// touching env vars.
    pub fn from_dir(model_id: &str, dir: &Path) -> Result<Self, NeuronError> {
        let onnx_file = resolve_onnx_file(dir)?;
        let sym_table = load_sym_table(&dir.join("tokens.txt"))?;

        let raw = tract_onnx::onnx()
            .model_for_path(&onnx_file)
            .map_err(|e| NeuronError::invalid_input(format!("tract load: {e}")))?;

        // Capture the name -> slot map before optimization renames nodes
        // (same reason as the decision session).
        let mut input_slots = HashMap::new();
        for (idx, outlet) in raw.inputs.clone().into_iter().enumerate() {
            input_slots.insert(raw.node(outlet.node).name.clone(), idx);
        }
        for name in INPUT_NAMES {
            if !input_slots.contains_key(name) {
                let have: Vec<&str> = input_slots.keys().map(|s| s.as_str()).collect();
                return Err(NeuronError::invalid_input(format!(
                    "SenseVoice graph exposes inputs {have:?}, expected {INPUT_NAMES:?} -- \
                     is {} really a SenseVoice export?",
                    onnx_file.display()
                )));
            }
        }
        if raw.outputs.len() != 1 {
            return Err(NeuronError::invalid_input(format!(
                "SenseVoice graph declares {} outputs, expected 1 (logits)",
                raw.outputs.len()
            )));
        }

        let meta = {
            // Read the metadata straight from the file: tract discards it.
            // Parsed before the load so a non-SenseVoice graph fails fast.
            let bytes = std::fs::read(&onnx_file).map_err(|e| {
                NeuronError::invalid_input(format!("read {}: {e}", onnx_file.display()))
            })?;
            let parsed = OnnxMetadata::parse(&bytes).map_err(NeuronError::invalid_input)?;
            SenseVoiceMeta::from_metadata(&parsed)?
        };

        let model = raw
            .into_optimized()
            .map_err(|e| NeuronError::invalid_input(format!("tract optimize: {e}")))?
            .into_runnable()
            .map_err(|e| NeuronError::invalid_input(format!("tract runnable: {e}")))?;

        Ok(Self {
            model: Arc::new(model),
            sym_table: Arc::new(sym_table),
            meta,
            input_slots,
            model_id: model_id.to_string(),
            onnx_file,
        })
    }

    pub fn meta(&self) -> &SenseVoiceMeta {
        &self.meta
    }

    /// fbank -> LFR stack -> CMVN -> position encoding. Returns the graph
    /// input tensor `[1, T, 560]`, the `pe` table `[1, T+4, 560]` and the
    /// number of LFR frames. The prepped graph (see `scripts/prep-sensevoice-onnx.py`)
    /// consumes `pe` with a leading batch dim of 1 so its shape is
    /// tract-codegen-friendly.
    pub fn features(&self, samples: &[f32]) -> Result<(Array3<f32>, Array3<f32>, usize), NeuronError> {
        let scaled: Vec<f32> = if self.meta.normalize_samples {
            samples.to_vec()
        } else {
            samples.iter().map(|s| s * 32768.0).collect()
        };
        let mel = fbank(&scaled, 16000, self.meta.num_mel_bins);
        let (window, shift, dim) = (self.meta.lfr_window, self.meta.lfr_shift, self.meta.num_mel_bins);
        if mel.num_frames < window {
            return Err(NeuronError::invalid_input(format!(
                "audio too short: {} fbank frames, the LFR window needs {window}",
                mel.num_frames
            )));
        }
        let lfr_frames = 1 + (mel.num_frames - window) / shift;

        let mut x = Array3::<f32>::zeros((1, lfr_frames, dim * window));
        for f in 0..lfr_frames {
            for w in 0..window {
                let src = (f * shift + w) * dim;
                x.slice_mut(ndarray::s![0, f, w * dim..(w + 1) * dim])
                    .assign(&ndarray::arr1(&mel.data[src..src + dim]));
            }
        }
        // CMVN: (x + neg_mean) * inv_stddev, stats are per stacked dim.
        for (d, (m, s)) in x
            .iter_mut()
            .zip(self.meta.neg_mean.iter().zip(self.meta.inv_stddev.iter()))
        {
            *d = (*d + m) * s;
        }
        // The graph prepends 4 prefix embeddings (lang/emo/event/itn) to the
        // LFR frames, so the encoder runs over lfr_frames + 4 positions.
        // A leading batch dim of 1 matches the rest of the graph's [1, T, dim].
        let pe = positional_encoding(lfr_frames + 4, self.meta.num_mel_bins * self.meta.lfr_window)
            .insert_axis(ndarray::Axis(0));
        Ok((x, pe, lfr_frames))
    }

    /// Run the graph over one utterance's features, returning the logits
    /// `[T', vocab]` (batch dim squeezed).
    pub fn forward(
        &self,
        x: &Array3<f32>,
        pe: &Array3<f32>,
        lang_id: i32,
        itn_id: i32,
    ) -> Result<Array2<f32>, NeuronError> {
        let t = x.shape()[1];
        let mut feeds: Vec<Option<TValue>> = (0..self.input_slots.len()).map(|_| None).collect();
        let mut push = |name: &str, value: Tensor| -> Result<(), NeuronError> {
            let slot = self.input_slots.get(name).copied().ok_or_else(|| {
                NeuronError::invalid_input(format!("graph has no input named {name}"))
            })?;
            feeds[slot] = Some(value.into_tvalue());
            Ok(())
        };
        push("x", x.clone().into_dyn().into())?;
        push("x_length", ndarray::arr1(&[t as i32]).into_dyn().into())?;
        push("language", ndarray::arr1(&[lang_id]).into_dyn().into())?;
        push("text_norm", ndarray::arr1(&[itn_id]).into_dyn().into())?;
        push("pe", pe.clone().into_dyn().into())?;
        drop(push);

        let feeds: TVec<TValue> = feeds
            .into_iter()
            .enumerate()
            .map(|(i, f)| f.ok_or_else(|| NeuronError::invalid_input(format!("unfed graph input slot {i}"))))
            .collect::<Result<_, _>>()?;

        let outs = self
            .model
            .run(feeds)
            .map_err(|e| NeuronError::invalid_input(format!("tract run: {e}")))?;
        let view = outs[0]
            .to_array_view::<f32>()
            .map_err(|e| NeuronError::invalid_input(format!("logits dtype: {e}")))?;
        let shape = view.shape().to_vec();
        let logits = view
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|e| {
                NeuronError::invalid_input(format!(
                    "logits shape {shape:?} is not [batch, frames, vocab]: {e}"
                ))
            })?
            .slice(ndarray::s![0, .., ..])
            .to_owned();
        Ok(logits)
    }

    /// Full pipeline with per-stage timings. `transcribe` delegates here;
    /// the benchmark uses the splits to attribute cost per stage.
    pub fn transcribe_timed(
        &self,
        samples: &[f32],
        lang: Option<&str>,
    ) -> Result<(Transcription, SenseVoiceTimings), NeuronError> {
        let mut timings = SenseVoiceTimings::default();

        let t0 = Instant::now();
        let (x, pe, lfr_frames) = self.features(samples)?;
        let lang_id = self.meta.language_id(lang)?;
        let itn_id = self.meta.with_itn;
        timings.feature_us = t0.elapsed().as_micros();

        let t1 = Instant::now();
        let logits = self.forward(&x, &pe, lang_id, itn_id)?;
        timings.forward_us = t1.elapsed().as_micros();

        let t2 = Instant::now();
        let ids = ctc_greedy(logits.view(), self.meta.blank_id);
        if ids.len() != lfr_frames + 4 {
            // The graph prepends 4 meta-token rows (lang/emo/event/itn) to
            // the LFR frames. Any other arithmetic means an export whose
            // output geometry we did not anticipate -- decode what came
            // back, but leave a trace.
            tracing::debug!(
                "sensevoice: graph returned {} logits rows for {lfr_frames} LFR frames \
                 (expected {}); decoding what came back",
                ids.len(),
                lfr_frames + 4
            );
        }
        let (text, language) = self.render(&ids);
        timings.decode_us = t2.elapsed().as_micros();

        Ok((
            Transcription {
                text,
                language,
                duration_secs: Some(samples.len() as f32 / 16000.0),
            },
            timings,
        ))
    }

    /// Map CTC token ids to text. The first four rows are meta tokens:
    /// `[language, emotion, event, itn]`; the rest is the transcription.
    /// Returns the text (SentencePiece `▁` -> space) and the detected
    /// language when the first token decodes to something like `<|zh|>`.
    fn render(&self, ids: &[u64]) -> (String, Option<String>) {
        if ids.len() < 4 {
            return (String::new(), None);
        }
        let sym = |id: u64| -> String {
            self.sym_table
                .get(&(id as u32))
                .cloned()
                .unwrap_or_else(|| format!("<id_{id}>"))
        };
        let lang_raw = sym(ids[0]);
        let language = lang_raw
            .trim_start_matches("<|")
            .trim_end_matches("|>")
            .to_string();
        let language = if language.is_empty() || language == lang_raw { None } else { Some(language) };

        let mut text = String::new();
        for &id in &ids[4..] {
            text.push_str(&sym(id));
        }
        (text.replace(SP_SPACE, " ").trim().to_string(), language)
    }
}

/// Greedy CTC decode over `[frames, vocab]` logits: per-frame argmax,
/// skip the blank id, collapse consecutive repeats. Same algorithm as
/// sherpa-onnx `OfflineCtcGreedySearchDecoder`.
pub fn ctc_greedy(logits: ndarray::ArrayView2<f32>, blank_id: u64) -> Vec<u64> {
    let mut ids = Vec::with_capacity(logits.shape()[0]);
    let mut prev: Option<u64> = None;
    for row in logits.rows() {
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in row.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        let id = best as u64;
        if id != blank_id && prev != Some(id) {
            ids.push(id);
        }
        prev = Some(id);
    }
    ids
}

/// Model file selection inside a SenseVoice directory:
/// `NEURON_ASR_ONNX_FILE` names one explicitly; otherwise the fp32
/// `model.onnx` wins over `model.int8.onnx` (see the module docs for why).
pub fn resolve_onnx_file(dir: &Path) -> Result<PathBuf, NeuronError> {
    if let Ok(name) = std::env::var("NEURON_ASR_ONNX_FILE") {
        let p = dir.join(&name);
        if !p.is_file() {
            return Err(NeuronError::invalid_input(format!(
                "NEURON_ASR_ONNX_FILE={name} but {} does not exist",
                p.display()
            )));
        }
        return Ok(p);
    }
    for name in ["model.onnx", "model.int8.onnx"] {
        let p = dir.join(name);
        if p.is_file() {
            return Ok(p);
        }
    }
    Err(NeuronError::invalid_input(format!(
        "no model.onnx / model.int8.onnx in {}",
        dir.display()
    )))
}

fn load_sym_table(path: &Path) -> Result<HashMap<u32, String>, NeuronError> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| NeuronError::invalid_input(format!("read {}: {e}", path.display())))?;
    let mut table = HashMap::new();
    for line in raw.lines() {
        if line.is_empty() {
            continue;
        }
        // tokens never contain a literal space, so the id is after the
        // final space and the symbol is everything before it
        let Some((sym, id)) = line.rsplit_once(' ') else {
            continue;
        };
        let id: u32 = id.trim().parse().map_err(|e| {
            NeuronError::invalid_input(format!("{}: bad token id in {line:?}: {e}", path.display()))
        })?;
        table.insert(id, sym.to_string());
    }
    if table.is_empty() {
        return Err(NeuronError::invalid_input(format!(
            "{}: no `symbol id` lines found",
            path.display()
        )));
    }
    Ok(table)
}

/// Where a SenseVoice model directory lives: `NEURON_ASR_MODEL_PATH` first,
/// then `$NEURON_MODEL_DIRS`-style roots as `asr/{model_id}`.
pub fn resolve_model_dir(model_id: &str) -> Result<PathBuf, NeuronError> {
    if let Ok(p) = std::env::var("NEURON_ASR_MODEL_PATH") {
        let path = PathBuf::from(&p);
        if !path.is_dir() {
            return Err(NeuronError::invalid_input(format!(
                "NEURON_ASR_MODEL_PATH={p} is not a directory"
            )));
        }
        return Ok(path);
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
        let candidate = Path::new(root).join("asr").join(model_id);
        if candidate.is_dir() {
            return Ok(candidate);
        }
    }
    Err(NeuronError::invalid_input(format!(
        "SenseVoice model dir not found for {model_id}; set NEURON_ASR_MODEL_PATH to a \
         directory containing model.onnx + tokens.txt"
    )))
}

#[async_trait]
impl AsrRuntime for TractSenseVoiceRuntime {
    async fn load(model: &str, _device: &DeviceSelection) -> ChatResult<Box<dyn AsrRuntime>>
    where
        Self: Sized,
    {
        let dir = resolve_model_dir(model)?;
        let rt = Self::from_dir(model, &dir)?;
        Ok(Box::new(rt))
    }

    async fn transcribe(&self, audio: &[u8], lang: Option<&str>) -> ChatResult<Transcription> {
        if audio.is_empty() {
            return Err(NeuronError::invalid_input("empty audio"));
        }
        let (samples, _sr) =
            decode_wav(audio).map_err(|e| NeuronError::invalid_input(format!("WAV decode: {e}")))?;
        let (text, _timings) = self.transcribe_timed(&samples, lang)?;
        Ok(text)
    }

    fn name(&self) -> &str {
        "sensevoice"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta() -> SenseVoiceMeta {
        SenseVoiceMeta {
            lfr_window: 7,
            lfr_shift: 6,
            neg_mean: vec![0.5; 560],
            inv_stddev: vec![0.1; 560],
            vocab_size: 25055,
            blank_id: 0,
            normalize_samples: false,
            lang_ids: vec![
                ("auto".into(), 0),
                ("zh".into(), 3),
                ("en".into(), 4),
            ],
            with_itn: 14,
            without_itn: 15,
            num_mel_bins: 80,
        }
    }

    #[test]
    fn test_language_id_resolves_and_defaults() {
        let m = meta();
        assert_eq!(m.language_id(None).unwrap(), 0); // auto
        assert_eq!(m.language_id(Some("zh")).unwrap(), 3);
        assert_eq!(m.language_id(Some("en")).unwrap(), 4);
        assert!(m.language_id(Some("fr")).is_err());
    }

    #[test]
    fn test_ctc_greedy_skips_blank_and_repeats() {
        // 7 frames x 5 vocab; frame argmaxes: blank, 2, 2, 1, 3, 3, 4 --
        // consecutive runs collapse and the blank is skipped
        let logits: Vec<f32> = vec![
            [5.0, 0.0, 0.0, 0.0, 0.0], // blank
            [0.0, 0.0, 5.0, 0.0, 0.0], // 2
            [0.0, 0.0, 4.0, 0.0, 0.0], // 2 again (same run)
            [0.0, 5.0, 0.0, 0.0, 0.0], // 1
            [0.0, 0.0, 0.0, 5.0, 0.0], // 3
            [0.0, 0.0, 0.0, 4.0, 0.0], // 3 again
            [0.0, 0.0, 0.0, 0.0, 5.0], // 4
        ]
        .into_iter()
        .flatten()
        .collect();
        let arr = Array3::from_shape_vec((1, 7, 5), logits.clone()).unwrap();
        let ids = ctc_greedy(arr.slice(ndarray::s![0, .., ..]), 0);
        assert_eq!(ids, vec![2, 1, 3, 4]);

        // a different blank id changes the outcome: frames whose argmax is
        // 2 are now dropped as blanks
        let mut logits2 = logits;
        for f in 0..3 {
            logits2[f * 5] = 0.0;
            logits2[f * 5 + 2] = 5.0;
        }
        let arr = Array3::from_shape_vec((1, 7, 5), logits2).unwrap();
        let ids = ctc_greedy(arr.slice(ndarray::s![0, .., ..]), 2);
        assert_eq!(ids, vec![1, 3, 4]);
    }

    #[test]
    fn test_sym_table_parses_symbol_id_pairs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokens.txt");
        std::fs::write(&path, "<unk> 0\n▁the 3\n<|zh|> 24884\n\n").unwrap();
        let table = load_sym_table(&path).unwrap();
        assert_eq!(table[&0], "<unk>");
        assert_eq!(table[&3], "▁the");
        assert_eq!(table[&24884], "<|zh|>");
        // an empty file is an error, not an empty table
        std::fs::write(&path, "").unwrap();
        assert!(load_sym_table(&path).is_err());
    }

    #[test]
    fn test_resolve_onnx_file_prefers_fp32() {
        let dir = tempfile::tempdir().unwrap();
        // only int8 present -> int8 is picked
        std::fs::write(dir.path().join("model.int8.onnx"), b"x").unwrap();
        let got = resolve_onnx_file(dir.path()).unwrap();
        assert!(got.ends_with("model.int8.onnx"));
        // fp32 present -> wins
        std::fs::write(dir.path().join("model.onnx"), b"x").unwrap();
        let got = resolve_onnx_file(dir.path()).unwrap();
        assert!(got.ends_with("model.onnx"));
        // explicit override wins over both
        unsafe { std::env::set_var("NEURON_ASR_ONNX_FILE", "model.int8.onnx") };
        let got = resolve_onnx_file(dir.path()).unwrap();
        unsafe { std::env::remove_var("NEURON_ASR_ONNX_FILE") };
        assert!(got.ends_with("model.int8.onnx"));
    }

    #[test]
    #[serial_test::serial]
    fn test_resolve_model_dir_requires_dir() {
        unsafe { std::env::set_var("NEURON_ASR_MODEL_PATH", "definitely/not/a/dir") };
        let err = resolve_model_dir("m").err().unwrap();
        unsafe { std::env::remove_var("NEURON_ASR_MODEL_PATH") };
        assert!(err.message.contains("not a directory"), "{}", err.message);
    }
}
