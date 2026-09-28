//! Decision capability: non-autoregressive discriminative decision scoring.
//!
//! Laya-style models are encoder-only scorers, not generative LLMs. Given a
//! typed question (`choice` / `score` / `noul`) plus a set of options and a
//! state, one forward pass emits a raw logit per option and a two-class
//! act/escalate pair. Nothing generates text, so this cannot ride the
//! [`crate::llm::ChatLlmRuntime`] trait: there is no token stream, no KV cache
//! and no autoregressive `offset`. It is therefore its own capability.
//!
//! The calibrated probabilities are the product, and the calibration lives
//! entirely on the host -- see [`postprocess`] for why that matters.
//!
//! Layout:
//! - [`postprocess`] -- temperature calibration + softmax (pure, golden-tested)
//! - [`registry`] -- model registry + lazy load, same shape as rerank
//! - [`tract_session`] -- multi-input / multi-output tract session (`tract`)

pub mod postprocess;
pub mod prompt;
pub mod registry;

#[cfg(feature = "tract")]
pub mod tract_session;

pub use registry::DecisionRuntimeRegistry;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::config::DeviceSelection;
use crate::llm::{ChatResult, ModelInfo, NeuronError};

pub use postprocess::QuestionType;

/// Which engine backs a registered decision model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DecisionImpl {
    /// Pure-Rust ONNX via tract. Requires a graph prepped with
    /// `scripts/prep-laya-onnx.py`.
    #[cfg(feature = "tract")]
    Tract,
}

impl DecisionImpl {
    pub fn as_str(&self) -> &'static str {
        // Matches on `*self`, not `self`: without the `tract` feature the enum
        // is uninhabited, and `match` on a reference to it is non-exhaustive
        // because references are always considered inhabited.
        match *self {
            #[cfg(feature = "tract")]
            Self::Tract => "tract",
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            #[cfg(feature = "tract")]
            "tract" | "onnx" => Ok(Self::Tract),
            other => Err(format!("unknown DecisionImpl: {other}")),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RegisteredDecision {
    pub model_id: String,
    pub impl_id: DecisionImpl,
    pub device: DeviceSelection,
    pub required_memory_bytes: u64,
    /// Path to the prepped ONNX graph, from `NEURON_DECISION_MODEL_PATH`.
    pub model_path: String,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct DecisionRegistryKey {
    pub model_id: String,
    pub impl_id: DecisionImpl,
    pub device: DeviceSelection,
}

/// JSON object that keeps the *input* order of its keys.
///
/// `serde_json::Map` is a `BTreeMap`, so round-tripping a `choice` rubric
/// through it sorts the labels alphabetically. Upstream derives the answer keys
/// as `list(q["crit"].keys())` and zips them against the probabilities by
/// index, so a reordering silently relabels the model's answer. Deserializing
/// straight off `MapAccess` preserves the order the JSON text had.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OrderedMap(pub Vec<(String, serde_json::Value)>);

impl OrderedMap {
    pub fn get(&self, key: &str) -> Option<&serde_json::Value> {
        self.0.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn get_ci(&self, key: &str) -> Option<&serde_json::Value> {
        let lower = key.to_lowercase();
        self.0
            .iter()
            .find(|(k, _)| k.to_lowercase() == lower)
            .map(|(_, v)| v)
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(|(k, _)| k.as_str())
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Serialize for OrderedMap {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut m = s.serialize_map(Some(self.0.len()))?;
        for (k, v) in &self.0 {
            m.serialize_entry(k, v)?;
        }
        m.end()
    }
}

impl<'de> Deserialize<'de> for OrderedMap {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = OrderedMap;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a JSON object")
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> Result<OrderedMap, A::Error> {
                let mut out = Vec::new();
                while let Some((k, v)) = map.next_entry::<String, serde_json::Value>()? {
                    out.push((k, v));
                }
                Ok(OrderedMap(out))
            }
        }
        d.deserialize_map(V)
    }
}

/// The `criteria` field, whose shape depends on the question type.
///
/// `choice` takes either a rubric object or a plain list of labels; `score`
/// takes a list of per-level criteria; `noul` takes an object with `false` /
/// `true` keys. A JSON array cannot be told apart statically, so both land in
/// [`CriteriaValue::Levels`] and [`prompt::render_options`] interprets it.
#[derive(Debug, Clone, PartialEq)]
pub enum CriteriaValue {
    Rubric(OrderedMap),
    Levels(Vec<serde_json::Value>),
}

impl Serialize for CriteriaValue {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Rubric(m) => m.serialize(s),
            Self::Levels(v) => v.serialize(s),
        }
    }
}

impl<'de> Deserialize<'de> for CriteriaValue {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = CriteriaValue;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a JSON object or array")
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(self, map: A) -> Result<Self::Value, A::Error> {
                Ok(CriteriaValue::Rubric(OrderedMap::deserialize(
                    serde::de::value::MapAccessDeserializer::new(map),
                )?))
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut out = Vec::new();
                while let Some(v) = seq.next_element::<serde_json::Value>()? {
                    out.push(v);
                }
                Ok(CriteriaValue::Levels(out))
            }
        }
        // accept either container
        d.deserialize_any(V)
    }
}

/// One typed question to score.
///
/// The field names follow `laya/onnx_agent.py::ONNXAgent._to_internal`'s input
/// (`qdef`), so a schema written for the Python agent ports over unchanged.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionQuestion {
    /// `choice` (pick one of N), `score` (rate N levels) or `noul` (no/yes).
    #[serde(rename = "type")]
    pub qtype: QuestionType,
    /// The instruction, e.g. "Which action should the agent take?".
    /// Structured values are JSON-serialized, matching upstream `_to_internal`.
    pub instructions: serde_json::Value,
    /// Rubric / level descriptions. See [`CriteriaValue`]; order is significant
    /// for `choice`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub criteria: Option<CriteriaValue>,
    /// Display labels for `noul` only. Must map exactly `false` and `true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub labels: Option<std::collections::BTreeMap<String, String>>,
    /// Pre-rendered option texts. Bypasses `criteria` rendering entirely.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub options: Option<Vec<String>>,
    /// Reorder the rendered options before encoding. Does not reorder the
    /// returned probabilities, which stay in rendered order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub option_order: Option<Vec<usize>>,
}

impl DecisionQuestion {
    /// Instruction as the prompt grammar sees it: strings pass through,
    /// anything structured becomes JSON text.
    pub fn instruction_text(&self) -> String {
        prompt::serialize_state(&self.instructions)
    }

    /// The option texts, either pre-rendered or built from `criteria`.
    pub fn rendered_options(&self) -> Result<Vec<String>, NeuronError> {
        if let Some(o) = &self.options {
            return Ok(o.clone());
        }
        prompt::render_options(self.qtype, self.criteria.as_ref(), self.labels.as_ref())
            .map_err(NeuronError::invalid_input)
    }

    /// Encode against a loaded tokenizer. `state` is shared across questions
    /// and `state_ids` lets the caller tokenize it once.
    pub fn encode(
        &self,
        tok: &tokenizers::Tokenizer,
        specials: &prompt::SpecialIds,
        state: &serde_json::Value,
        state_ids: Option<&[i64]>,
        cfg: &prompt::PromptConfig,
    ) -> Result<prompt::EncodedQuestion, NeuronError> {
        let rendered = self.rendered_options()?;
        let truncate_left = state.is_array();
        prompt::build_sequence(
            tok,
            specials,
            self.qtype,
            &self.instruction_text(),
            self.criteria.as_ref(),
            self.labels.as_ref(),
            Some(&rendered),
            self.option_order.as_deref(),
            state,
            state_ids,
            cfg,
            truncate_left,
        )
        .map_err(NeuronError::invalid_input)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionRequest {
    pub model: String,
    /// The state / context every question is scored in. A JSON array is taken
    /// to mean "recent first", so truncation drops from the front -- the rule
    /// upstream applies via `truncate_left=isinstance(state, list)`.
    pub state: serde_json::Value,
    pub questions: Vec<DecisionQuestion>,
    /// Override the checkpoint's sequence budget.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_len: Option<usize>,
    /// Override the checkpoint's head-section budget.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head_max_len: Option<usize>,
}

impl DecisionRequest {
    pub fn new(
        model: impl Into<String>,
        state: serde_json::Value,
        questions: Vec<DecisionQuestion>,
    ) -> Self {
        Self { model: model.into(), state, questions, max_len: None, head_max_len: None }
    }
}

/// act (index 0) vs escalate (index 1), after softmax.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ActProbabilities {
    pub act: f64,
    pub escalate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionResult {
    /// Position in the request's `questions` array.
    pub index: usize,
    pub qtype: QuestionType,
    pub option_count: usize,
    /// Calibrated per-option probabilities, summing to 1, in **rendered option
    /// order** -- `probabilities[i]` is `P(rendered_options()[i])` even when
    /// `option_order` shuffled what the model saw.
    pub probabilities: Vec<f64>,
    /// `argmax(probabilities)`, indexing the rendered options.
    pub chosen_index: usize,
    /// `choice` only: the winning label, `rendered_options()[chosen_index]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chosen: Option<String>,
    /// `score` only: the expected level, `sum(i * p[i])`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    /// `noul` only: `P(true)`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub noul: Option<f64>,
    /// Confidence in [0, 1]. Normalized entropy for `choice`/`score`; for
    /// `noul` it is the majority probability `max(P(true), P(false))`, matching
    /// upstream's per-type answers.
    pub confidence: f64,
    /// The temperature actually applied, after clamping to [0.5, 5.0].
    pub temperature_scale: f64,
    /// The bucket key that selected the scale, e.g. `"choice:3-5"`.
    pub temperature_bucket: String,
    pub act: Option<ActProbabilities>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

impl DecisionUsage {
    /// Built from the collated batch's attention-mask sum, matching upstream
    /// `usage.input_tokens = int(b["attention_mask"].sum())`.
    pub fn from_tokens(prompt_tokens: u32) -> Self {
        Self { prompt_tokens, total_tokens: prompt_tokens }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionResponse {
    pub object: String,
    pub model: String,
    pub data: Vec<DecisionResult>,
    pub usage: DecisionUsage,
}

impl DecisionResponse {
    pub fn new(
        model: impl Into<String>,
        data: Vec<DecisionResult>,
        usage: DecisionUsage,
    ) -> Self {
        Self { object: "list".to_string(), model: model.into(), data, usage }
    }
}

#[async_trait]
pub trait DecisionRuntime: Send + Sync {
    async fn load(model: &str, device: &DeviceSelection) -> ChatResult<Box<dyn DecisionRuntime>>
    where
        Self: Sized;

    async fn decide(&self, req: DecisionRequest) -> ChatResult<DecisionResponse>;

    fn list_models(&self) -> Vec<ModelInfo>;

    fn name(&self) -> &str;
}

/// Convenience for runtimes: score one question's raw logits end to end.
pub fn score_question(
    qtype: QuestionType,
    option_count: usize,
    logits: &[f64],
    act_logits: Option<&[f64]>,
    temperature: &postprocess::Temperature,
    by_options: &postprocess::TemperatureByOptions,
) -> DecisionResult {
    let bucket = postprocess::temp_bucket(qtype, option_count);
    let scale = postprocess::temperature_scale(by_options, temperature, qtype, option_count);
    let probabilities = postprocess::calibrated_probs(logits, option_count, scale);
    let chosen_index = probabilities
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let confidence = postprocess::confidence_from_probs(&probabilities, option_count);
    DecisionResult {
        index: 0,
        qtype,
        option_count,
        probabilities,
        chosen_index,
        chosen: None,
        score: None,
        noul: None,
        confidence,
        temperature_scale: scale,
        temperature_bucket: bucket,
        act: act_logits.map(|a| {
            let p = postprocess::act_probs(a);
            ActProbabilities { act: p[0], escalate: p[1] }
        }),
    }
}

impl DecisionResult {
    /// Undo `option_order`: move each marker-ordered probability back to the
    /// slot of the rendered option it came from.
    ///
    /// `build_sequence` lays the `[MASK]` markers out in `option_order`, so
    /// `logits[j]` scores `rendered[order[j]]`. Everything downstream labels by
    /// the rendered index (`keys = list(q["crit"].keys())` upstream), so the
    /// permutation has to be inverted before scoring or the winning probability
    /// gets attached to the wrong label. Identity order -- the only case the
    /// upstream agent exercises -- is a no-op.
    pub fn unpermute(&mut self, option_order: Option<&[usize]>) -> Result<(), NeuronError> {
        let Some(order) = option_order else { return Ok(()) };
        if order.len() != self.probabilities.len() {
            return Err(NeuronError::invalid_input(format!(
                "option_order has {} entries for {} options",
                order.len(),
                self.probabilities.len()
            )));
        }
        let mut seen = vec![false; order.len()];
        let mut out = vec![0.0; self.probabilities.len()];
        for (j, &i) in order.iter().enumerate() {
            if i >= out.len() || seen[i] {
                return Err(NeuronError::invalid_input(format!(
                    "option_order entry {i} is out of range or repeated"
                )));
            }
            seen[i] = true;
            out[i] = self.probabilities[j];
        }
        self.probabilities = out;
        self.chosen_index = self
            .probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
            .unwrap_or(0);
        Ok(())
    }

    /// Fill the type-specific answer fields, mirroring `ONNXAgent._infer`'s
    /// `answers[qid]` construction.
    pub fn apply_answer(&mut self, q: &DecisionQuestion) -> Result<(), NeuronError> {
        let opts = q.rendered_options()?;
        match self.qtype {
            QuestionType::Choice => {
                self.chosen = opts.get(self.chosen_index).cloned();
                self.score = None;
                self.noul = None;
            }
            QuestionType::Score => {
                let expected: f64 = self
                    .probabilities
                    .iter()
                    .enumerate()
                    .map(|(i, &p)| i as f64 * p)
                    .sum();
                self.score = Some(expected);
                self.chosen = None;
                self.noul = None;
            }
            QuestionType::Noul => {
                let p_true = self.probabilities.get(1).copied().unwrap_or(0.0);
                self.noul = Some(p_true);
                // upstream reports majority probability for noul, not entropy
                self.confidence = p_true.max(1.0 - p_true);
                self.chosen = None;
                self.score = None;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // `DecisionImpl`'s only variant is compiled behind `tract`, so `parse`
    // rejects everything in a default build.
    #[test]
    #[cfg(feature = "tract")]
    fn test_decision_impl_parse_tract() {
        assert!(DecisionImpl::parse("tract").is_ok());
    }

    #[test]
    #[cfg(feature = "tract")]
    fn test_decision_impl_parse_accepts_onnx_alias() {
        assert_eq!(
            DecisionImpl::parse("onnx").unwrap().as_str(),
            DecisionImpl::parse("tract").unwrap().as_str()
        );
    }

    #[test]
    fn test_decision_impl_parse_unknown_errors() {
        assert!(DecisionImpl::parse("mistralrs").is_err());
    }

    #[test]
    fn test_decision_request_deserializes_the_laya_qdef_shape() {
        // Field names follow `laya/onnx_agent.py::_to_internal`'s input.
        let req: DecisionRequest = serde_json::from_str(
            r#"{
                "model": "laya",
                "state": "The user is frustrated.",
                "questions": [{
                    "type": "choice",
                    "instructions": "Which action?",
                    "criteria": {"refund": "give money back", "cancel": null}
                }]
            }"#,
        )
        .unwrap();
        assert_eq!(req.model, "laya");
        let q = &req.questions[0];
        assert_eq!(q.qtype, QuestionType::Choice);
        assert_eq!(q.instruction_text(), "Which action?");
        let opts = q.rendered_options().unwrap();
        assert_eq!(opts, vec!["refund: give money back", "cancel"]);
    }

    #[test]
    fn test_decision_question_instructions_json_dumps_structured_values() {
        let q: DecisionQuestion = serde_json::from_str(
            r#"{"type":"noul","instructions":{"goal":"ship"}}"#,
        )
        .unwrap();
        assert_eq!(q.instruction_text(), "{\"goal\": \"ship\"}");
    }

    #[test]
    fn test_decision_question_options_override_bypasses_criteria() {
        let q: DecisionQuestion = serde_json::from_str(
            r#"{"type":"choice","instructions":"q","options":["x","y"]}"#,
        )
        .unwrap();
        assert_eq!(q.rendered_options().unwrap(), vec!["x", "y"]);
    }

    #[test]
    fn test_decision_question_choice_requires_something_to_render() {
        let q: DecisionQuestion =
            serde_json::from_str(r#"{"type":"choice","instructions":"q"}"#).unwrap();
        assert!(q.rendered_options().is_err());
    }

    #[test]
    fn test_ordered_map_keeps_json_input_order() {
        let m: OrderedMap =
            serde_json::from_str(r#"{"zebra":null,"apple":null,"mango":null}"#).unwrap();
        assert_eq!(m.keys().collect::<Vec<_>>(), vec!["zebra", "apple", "mango"]);
    }

    #[test]
    fn test_criteria_value_accepts_object_and_array() {
        let rubric: CriteriaValue = serde_json::from_str(r#"{"a":"b"}"#).unwrap();
        assert!(matches!(rubric, CriteriaValue::Rubric(_)));
        let levels: CriteriaValue = serde_json::from_str(r#"["a","b"]"#).unwrap();
        assert!(matches!(levels, CriteriaValue::Levels(_)));
    }

    #[test]
    fn test_decision_usage_counts_prompt_tokens() {
        let u = DecisionUsage::from_tokens(7);
        assert_eq!(u.prompt_tokens, 7);
        assert_eq!(u.total_tokens, 7);
    }

    #[test]
    fn test_question_type_serde_is_lowercase() {
        let json = serde_json::to_string(&QuestionType::Noul).unwrap();
        assert_eq!(json, "\"noul\"");
    }

    #[test]
    fn test_decision_response_openai_shape() {
        let resp = DecisionResponse::new(
            "laya",
            vec![DecisionResult {
                index: 0,
                qtype: QuestionType::Choice,
                option_count: 2,
                probabilities: vec![0.25, 0.75],
                chosen_index: 1,
                chosen: Some("b".to_string()),
                score: None,
                noul: None,
                confidence: 0.5,
                temperature_scale: 0.5,
                temperature_bucket: "choice:2".to_string(),
                act: Some(ActProbabilities { act: 0.9, escalate: 0.1 }),
            }],
            DecisionUsage { prompt_tokens: 4, total_tokens: 4 },
        );
        let value = serde_json::to_value(&resp).unwrap();
        assert_eq!(value["object"], "list");
        assert_eq!(value["model"], "laya");
        assert!(value["data"].is_array());
        assert_eq!(value["data"][0]["chosen_index"], 1);
        assert_eq!(value["data"][0]["temperature_bucket"], "choice:2");
        assert!(value["usage"]["prompt_tokens"].is_number());
    }

    #[test]
    fn test_score_question_picks_argmax_and_records_bucket() {
        let temperature = [0.5, 5.0, 0.5];
        let by_options = std::collections::BTreeMap::new();
        let r = score_question(
            QuestionType::Choice,
            3,
            &[1.0, 5.0, 2.0],
            None,
            &temperature,
            &by_options,
        );
        assert_eq!(r.chosen_index, 1);
        assert_eq!(r.temperature_bucket, "choice:3-5");
        assert_eq!(r.temperature_scale, 0.5);
        assert_eq!(r.probabilities.len(), 3);
        assert!(r.act.is_none());
        // confidence is normalized entropy over 3 options
        assert!(r.confidence > 0.0 && r.confidence < 1.0, "{}", r.confidence);
    }

    #[test]
    fn test_score_question_k1_is_fully_confident() {
        let r = score_question(
            QuestionType::Choice,
            1,
            &[3.0],
            None,
            &[1.0, 1.0, 1.0],
            &Default::default(),
        );
        assert_eq!(r.confidence, 1.0);
        assert_eq!(r.probabilities, vec![1.0]);
    }

    #[test]
    fn test_unpermute_moves_probs_back_to_rendered_order() {
        let mut r = score_question(
            QuestionType::Choice,
            3,
            &[5.0, 1.0, 2.0],
            None,
            &[1.0, 1.0, 1.0],
            &Default::default(),
        );
        assert_eq!(r.chosen_index, 0);
        // the model saw [c, a, b]; give back [a, b, c]
        r.unpermute(Some(&[2, 0, 1])).unwrap();
        assert_eq!(r.chosen_index, 2);
        assert!((r.probabilities[2] - 0.0).abs() > 0.5, "{:?}", r.probabilities);
    }

    #[test]
    fn test_unpermute_rejects_duplicate_or_out_of_range() {
        let mut r = score_question(
            QuestionType::Choice,
            2,
            &[1.0, 2.0],
            None,
            &[1.0, 1.0, 1.0],
            &Default::default(),
        );
        assert!(r.unpermute(Some(&[0, 0])).is_err());
        assert!(r.unpermute(Some(&[0, 5])).is_err());
        assert!(r.unpermute(Some(&[0])).is_err());
    }

    #[test]
    fn test_apply_answer_choice_reports_the_winning_label() {
        let q: DecisionQuestion = serde_json::from_str(
            r#"{"type":"choice","instructions":"q","criteria":{"zebra":null,"apple":null}}"#,
        )
        .unwrap();
        let mut r = score_question(
            QuestionType::Choice,
            2,
            &[0.0, 4.0],
            None,
            &[1.0, 1.0, 1.0],
            &Default::default(),
        );
        r.apply_answer(&q).unwrap();
        // probabilities are in rendered (input) order, so index 1 is "apple"
        assert_eq!(r.chosen.as_deref(), Some("apple"));
    }

    #[test]
    fn test_apply_answer_score_reports_expected_level() {
        let q: DecisionQuestion = serde_json::from_str(
            r#"{"type":"score","instructions":"q","criteria":["bad","ok","good"]}"#,
        )
        .unwrap();
        let mut r = score_question(
            QuestionType::Score,
            3,
            &[-1000.0, 0.0, -1000.0],
            None,
            &[1.0, 1.0, 1.0],
            &Default::default(),
        );
        r.apply_answer(&q).unwrap();
        assert!((r.score.unwrap() - 1.0).abs() < 1e-6, "{:?}", r.score);
    }

    #[test]
    fn test_apply_answer_noul_reports_majority_probability_confidence() {
        let q: DecisionQuestion =
            serde_json::from_str(r#"{"type":"noul","instructions":"q"}"#).unwrap();
        let mut r = score_question(
            QuestionType::Noul,
            2,
            &[0.0, 2.0],
            None,
            &[1.0, 1.0, 1.0],
            &Default::default(),
        );
        r.apply_answer(&q).unwrap();
        let p_true = r.noul.unwrap();
        assert!(p_true > 0.8, "{p_true}");
        // noul confidence is the majority probability, not entropy
        assert!((r.confidence - p_true).abs() < 1e-12, "{}", r.confidence);
    }

    #[test]
    fn test_score_question_attaches_act_probabilities() {
        let temperature = [1.0, 1.0, 1.0];
        let by_options = std::collections::BTreeMap::new();
        let r = score_question(
            QuestionType::Noul,
            2,
            &[0.0, 0.0],
            Some(&[1000.0, -1000.0]),
            &temperature,
            &by_options,
        );
        let act = r.act.expect("act present");
        assert!(act.act > 0.999);
        assert!((act.act + act.escalate - 1.0).abs() < 1e-9);
    }
}
