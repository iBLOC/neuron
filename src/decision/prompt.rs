//! Prompt grammar for the Laya decision head.
//!
//! This is a faithful port of `laya/common.py::build_sequence` + `render_options`
//! + `serialize_state` + `render_criterion`. The input layout is load bearing --
//! the model was trained on exactly this template, and any deviation silently
//! degrades the calibrated probabilities:
//!
//! ```text
//! [CLS] <type> question: <ins> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
//! ```
//!
//! Details that are easy to get wrong and are therefore pinned by tests:
//! - options are *rendered* from criteria by [`render_options`], not passed as
//!   raw strings. `noul` in particular renders as
//!   `"false: no, the statement does not hold"` / `"true: yes, the statement
//!   holds"` -- the bare words "false"/"true" are wrong.
//! - each option is tokenized with a **leading space** (`" " + opt`), and capped
//!   at [`MAX_OPTION_TOKENS`] tokens by the tokenizer, not after the fact.
//! - the budget works backwards: options take what they need first, and the
//!   instruction gets `max(8, head_max_len - sum(option lens))`. If that leaves
//!   less than [`MIN_INSTRUCTION_BUDGET`], the options are shrunk instead.
//! - a `[MASK]` inside the instruction or an option is replaced by a space.
//! - markers that fall outside `max_len` are DROPPED, not an error.
//! - the state is `json.dumps(state, ensure_ascii=False)` when it is not already
//!   a string.

use std::collections::BTreeMap;

use serde_json::Value;
use tokenizers::Tokenizer;

use super::postprocess::QuestionType;
use super::CriteriaValue;

/// Per-option truncation budget, in tokens (upstream `max_length=48`).
pub const MAX_OPTION_TOKENS: usize = 48;

/// Floor on the instruction's token budget (`head_ids[: max(8, opt_budget)]`).
pub const MIN_INSTRUCTION_BUDGET: usize = 8;

/// Below this leftover budget the options are shrunk to make room.
pub const MIN_OPTION_BUDGET: usize = 16;

/// Per-option budget floor when the options have to be shrunk (`max(4, ...)`).
pub const MIN_PER_OPTION: usize = 4;

/// Default total sequence budget (upstream `build_sequence(max_len=512)`).
pub const DEFAULT_MAX_LEN: usize = 512;

/// Default head-section budget (upstream `head_max_len=192`).
pub const DEFAULT_HEAD_MAX_LEN: usize = 192;

/// Sequence-length budgeting for one model.
#[derive(Debug, Clone, Copy)]
pub struct PromptConfig {
    pub max_len: usize,
    pub head_max_len: usize,
}

impl Default for PromptConfig {
    fn default() -> Self {
        Self { max_len: DEFAULT_MAX_LEN, head_max_len: DEFAULT_HEAD_MAX_LEN }
    }
}

/// Special token ids the template is built from.
#[derive(Debug, Clone)]
pub struct SpecialIds {
    pub cls: u32,
    pub sep: u32,
    pub mask: u32,
    /// Literal mask token text, stripped from instruction/option/state text.
    pub mask_text: String,
}

impl SpecialIds {
    /// Look the three template tokens up in a loaded tokenizer.
    pub fn from_tokenizer(tok: &Tokenizer) -> Result<Self, String> {
        let get = |name: &str| -> Result<u32, String> {
            tok.token_to_id(name)
                .ok_or_else(|| format!("tokenizer has no {name} token"))
        };
        Ok(Self {
            cls: get("[CLS]")?,
            sep: get("[SEP]")?,
            mask: get("[MASK]")?,
            mask_text: "[MASK]".to_string(),
        })
    }
}

/// A single encoded question, ready to feed the graph (batch of 1).
#[derive(Debug, Clone)]
pub struct EncodedQuestion {
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    /// Index of each option's `[MASK]` token. Entries past `max_len` are gone.
    pub marker_pos: Vec<i64>,
    /// `true` for the first `marker_pos.len()` slots.
    pub marker_mask: Vec<bool>,
    pub qtype: i64,
    pub option_count: usize,
    pub seq_len: usize,
}

// --------------------------------------------------------------------------
// option rendering
// --------------------------------------------------------------------------

/// JSON-dump a value the way Python's
/// `json.dumps(v, ensure_ascii=False, separators=(", ", ": "), default=str)` does.
///
/// serde_json's own `to_string` uses compact separators, which produce
/// different text and therefore different tokens, so this is a separate path.
///
/// DIVERGENCE: Python dicts preserve insertion order while `serde_json::Object`
/// sorts keys, so a *nested* object criterion renders its keys alphabetically
/// here. Top-level criteria are unaffected -- those go through
/// [`super::OrderedMap`], which keeps input order. String, number, bool, null
/// and list criteria match exactly.
fn py_dumps(v: &Value) -> String {
    match v {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => format!("\"{}\"", py_escape(s)),
        Value::Array(items) => {
            let body: Vec<String> = items.iter().map(py_dumps).collect();
            format!("[{}]", body.join(", "))
        }
        Value::Object(map) => {
            let body: Vec<String> = map
                .iter()
                .map(|(k, val)| format!("\"{}\": {}", py_escape(k), py_dumps(val)))
                .collect();
            format!("{{{}}}", body.join(", "))
        }
    }
}

fn py_escape(s: &str) -> String {
    // ensure_ascii=False: non-ASCII passes through unescaped.
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Render one criterion value as text: strings pass through, anything
/// structured becomes JSON (upstream `render_criterion`).
pub fn render_criterion(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => py_dumps(other),
    }
}

fn is_blank(v: &Value) -> bool {
    match v {
        Value::Null => true,
        Value::String(s) => s.is_empty(),
        _ => false,
    }
}

/// Default `noul` labels: `{"false": "false", "true": "true"}`.
pub fn resolve_noul_labels(
    labels: Option<&BTreeMap<String, String>>,
) -> Result<(String, String), String> {
    const BAD: &str =
        "noul labels must map exactly 'false' and 'true' to distinct non-empty strings";
    let (false_label, true_label) = match labels {
        None => ("false".to_string(), "true".to_string()),
        Some(map) => {
            if map.len() != 2 {
                return Err(BAD.to_string());
            }
            let f = map.get("false").ok_or_else(|| BAD.to_string())?;
            let t = map.get("true").ok_or_else(|| BAD.to_string())?;
            (f.clone(), t.clone())
        }
    };
    let false_label = false_label.trim().to_string();
    let true_label = true_label.trim().to_string();
    if false_label.is_empty() || true_label.is_empty() || false_label == true_label {
        return Err(BAD.to_string());
    }
    Ok((false_label, true_label))
}

/// Render option texts in label-index order.
///
/// `noul` semantic order is always `[false, true]`.
///
/// `criteria` mirrors upstream's `q["crit"]`: a rubric object for `choice`, a
/// list for `score`, an optional object for `noul`. When [`build_sequence`] is
/// given `pre_rendered` options this function is bypassed entirely.
pub fn render_options(
    qtype: QuestionType,
    criteria: Option<&CriteriaValue>,
    labels: Option<&BTreeMap<String, String>>,
) -> Result<Vec<String>, String> {
    if qtype != QuestionType::Noul && labels.is_some() {
        return Err("labels is only supported for noul questions".to_string());
    }
    match qtype {
        QuestionType::Choice => match criteria {
            None => Err("choice questions require criteria".to_string()),
            Some(CriteriaValue::Rubric(m)) => {
                if m.is_empty() {
                    return Err("choice criteria must not be empty".to_string());
                }
                Ok(m.0
                    .iter()
                    .map(|(k, v)| {
                        // only None/"" mean "no description"; 0 and false are
                        // legitimate criterion values
                        if is_blank(v) {
                            k.clone()
                        } else {
                            format!("{k}: {}", render_criterion(v))
                        }
                    })
                    .collect())
            }
            // `choice` + a list means bare labels -- upstream `_to_internal`
            // rewrites it to `{c: None for c in crit}` before rendering.
            Some(CriteriaValue::Levels(list)) => {
                if list.is_empty() {
                    return Err("choice criteria must not be empty".to_string());
                }
                list.iter()
                    .map(|v| match v {
                        Value::String(s) => Ok(s.clone()),
                        other => Err(format!(
                            "choice label must be a string, got {}",
                            py_dumps(other)
                        )),
                    })
                    .collect()
            }
        },
        QuestionType::Score => match criteria {
            Some(CriteriaValue::Levels(list)) => Ok(list
                .iter()
                .enumerate()
                .map(|(i, c)| format!("level {i}: {}", render_criterion(c)))
                .collect()),
            Some(CriteriaValue::Rubric(_)) => Err("score criteria must be a list".to_string()),
            None => Err("score questions require criteria".to_string()),
        },
        QuestionType::Noul => {
            let (false_label, true_label) = resolve_noul_labels(labels)?;
            let empty = Value::Null;
            let (false_crit, true_crit) = match criteria {
                None => (&empty, &empty),
                // upstream lowercases the criteria keys before use
                Some(CriteriaValue::Rubric(m)) => {
                    (m.get_ci("false").unwrap_or(&empty), m.get_ci("true").unwrap_or(&empty))
                }
                Some(CriteriaValue::Levels(_)) => {
                    return Err("noul criteria must be an object".to_string())
                }
            };
            let false_body = if is_blank(false_crit) {
                "no, the statement does not hold".to_string()
            } else {
                render_criterion(false_crit)
            };
            let true_body = if is_blank(true_crit) {
                "yes, the statement holds".to_string()
            } else {
                render_criterion(true_crit)
            };
            Ok(vec![
                format!("{false_label}: {false_body}"),
                format!("{true_label}: {true_body}"),
            ])
        }
    }
}

/// Non-string states become compact-JSON-free JSON text
/// (`json.dumps(state, ensure_ascii=False)`).
pub fn serialize_state(state: &Value) -> String {
    match state {
        Value::String(s) => s.clone(),
        // Python's json.dumps default separators are (", ", ": ").
        other => py_dumps(other),
    }
}

// --------------------------------------------------------------------------
// sequence construction
// --------------------------------------------------------------------------

/// Tokenize a text span with no added special tokens.
pub fn encode_text(tok: &Tokenizer, text: &str) -> Result<Vec<i64>, String> {
    let enc = tok.encode(text, false).map_err(|e| format!("tokenize: {e}"))?;
    Ok(enc.get_ids().iter().map(|&i| i as i64).collect())
}

/// Encode one typed question.
///
/// `pre_rendered` bypasses [`render_options`] when the caller already has the
/// final option texts. `state_ids` lets a caller tokenize the shared state once
/// and reuse it across every question. `truncate_left` picks which end of the
/// state is dropped when the total exceeds `max_len`.
#[allow(clippy::too_many_arguments)]
pub fn build_sequence(
    tok: &Tokenizer,
    specials: &SpecialIds,
    qtype: QuestionType,
    instruction: &str,
    criteria: Option<&CriteriaValue>,
    labels: Option<&BTreeMap<String, String>>,
    pre_rendered: Option<&[String]>,
    option_order: Option<&[usize]>,
    state: &Value,
    state_ids: Option<&[i64]>,
    cfg: &PromptConfig,
    truncate_left: bool,
) -> Result<EncodedQuestion, String> {
    let mask_tok = &specials.mask_text;

    let opts: Vec<String> = match pre_rendered {
        Some(o) => o.to_vec(),
        None => render_options(qtype, criteria, labels)?,
    };
    if opts.is_empty() {
        return Err("decision question has no options".to_string());
    }

    let order: Vec<usize> = match option_order {
        Some(o) => {
            for &i in o {
                if i >= opts.len() {
                    return Err(format!("option_order index {i} out of range {}", opts.len()));
                }
            }
            o.to_vec()
        }
        None => (0..opts.len()).collect(),
    };

    // `ins` strips any [MASK] the caller put in the instruction.
    let ins = instruction.replace(mask_tok.as_str(), " ");
    let preamble = format!("{} question: {}", qtype.as_str(), ins);
    let mut head_ids = encode_text(tok, &preamble)?;

    let mut opt_ids: Vec<Vec<i64>> = Vec::with_capacity(order.len());
    for &i in &order {
        // Cap at the tokenizer, not after the fact. Note the LEADING SPACE:
        // `tok(" " + opt)` tokenizes differently from `tok(opt)`.
        let opt_text = format!(" {}", opts[i].replace(mask_tok.as_str(), " "));
        let enc = tok
            .encode(opt_text.as_str(), false)
            .map_err(|e| format!("tokenize option: {e}"))?;
        let mut ids: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();
        ids.truncate(MAX_OPTION_TOKENS);
        let mut with_marker = Vec::with_capacity(ids.len() + 1);
        with_marker.push(specials.mask as i64);
        with_marker.extend(ids);
        opt_ids.push(with_marker);
    }

    let opt_len: usize = opt_ids.iter().map(|o| o.len()).sum();
    let mut opt_budget = cfg.head_max_len.saturating_sub(opt_len);
    if opt_budget < MIN_OPTION_BUDGET {
        let per = std::cmp::max(
            MIN_PER_OPTION,
            cfg.head_max_len.saturating_sub(MIN_OPTION_BUDGET) / std::cmp::max(1, opt_ids.len()),
        );
        for o in &mut opt_ids {
            o.truncate(per);
        }
        let opt_len: usize = opt_ids.iter().map(|o| o.len()).sum();
        opt_budget = cfg.head_max_len.saturating_sub(opt_len);
    }
    head_ids.truncate(std::cmp::max(MIN_INSTRUCTION_BUDGET, opt_budget));

    let mut ids: Vec<i64> = Vec::with_capacity(cfg.max_len);
    ids.push(specials.cls as i64);
    ids.extend(&head_ids);
    ids.push(specials.sep as i64);

    let mut markers: Vec<i64> = Vec::with_capacity(opt_ids.len());
    for o in &opt_ids {
        markers.push(ids.len() as i64);
        ids.extend(o);
    }
    ids.push(specials.sep as i64);

    let room = cfg.max_len.saturating_sub(ids.len() + 1);
    let state_ids_owned: Vec<i64>;
    let state_ids: &[i64] = match state_ids {
        Some(s) => s,
        None => {
            let text = serialize_state(state).replace(mask_tok.as_str(), " ");
            state_ids_owned = encode_text(tok, &text)?;
            &state_ids_owned
        }
    };
    // `state_ids[-room:]` with room == 0 would be the WHOLE state, so index it
    // explicitly rather than relying on a negative slice.
    let st: &[i64] = if truncate_left {
        &state_ids[state_ids.len().saturating_sub(room)..]
    } else {
        &state_ids[..room.min(state_ids.len())]
    };
    ids.extend_from_slice(st);
    ids.push(specials.sep as i64);
    ids.truncate(cfg.max_len);

    let markers: Vec<i64> = markers.into_iter().filter(|&m| (m as usize) < cfg.max_len).collect();
    let seq_len = ids.len();
    let option_count = markers.len();

    Ok(EncodedQuestion {
        input_ids: ids,
        attention_mask: vec![1i64; seq_len],
        marker_pos: markers,
        marker_mask: vec![true; option_count],
        qtype: qtype.index() as i64,
        option_count,
        seq_len,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn json(s: &str) -> Value {
        serde_json::from_str(s).unwrap()
    }

    fn crit(s: &str) -> CriteriaValue {
        serde_json::from_str(s).unwrap()
    }

    // ---- render_criterion ------------------------------------------------

    #[test]
    fn test_render_criterion_string_passes_through() {
        assert_eq!(render_criterion(&Value::String("hello".into())), "hello");
    }

    #[test]
    fn test_render_criterion_number_becomes_json() {
        assert_eq!(render_criterion(&json("0")), "0");
        assert_eq!(render_criterion(&json("false")), "false");
    }

    #[test]
    fn test_render_criterion_uses_python_separators() {
        // json.dumps(..., separators=(", ", ": ")) -- NOT serde_json's compact form.
        assert_eq!(render_criterion(&json(r#"["a","b"]"#)), "[\"a\", \"b\"]");
    }

    #[test]
    fn test_render_criterion_nested_object_uses_python_separators() {
        assert_eq!(
            render_criterion(&json(r#"{"desc":"x"}"#)),
            "{\"desc\": \"x\"}"
        );
    }

    #[test]
    fn test_render_criterion_keeps_non_ascii_unescaped() {
        assert_eq!(render_criterion(&Value::String("高棉".into())), "高棉");
    }

    // ---- render_options --------------------------------------------------

    #[test]
    fn test_render_options_choice_blank_criterion_is_label_only() {
        let opts = render_options(
            QuestionType::Choice,
            Some(&crit(r#"{"refund":null,"cancel":""}"#)),
            None,
        )
        .unwrap();
        // Only None/"" mean "no description".
        assert_eq!(opts, vec!["refund", "cancel"]);
    }

    #[test]
    fn test_render_options_choice_keeps_json_input_order() {
        // Regression: serde_json::Map is a BTreeMap and would sort these. The
        // model's probabilities are position-indexed against `list(crit.keys())`,
        // so sorting relabels the answer.
        let opts = render_options(
            QuestionType::Choice,
            Some(&crit(r#"{"zebra":null,"apple":null,"mango":null}"#)),
            None,
        )
        .unwrap();
        assert_eq!(opts, vec!["zebra", "apple", "mango"]);
    }

    #[test]
    fn test_render_options_choice_list_criteria_are_bare_labels() {
        // `choice` + a JSON array means bare labels (upstream `_to_internal`).
        let opts = render_options(
            QuestionType::Choice,
            Some(&crit(r#"["refund","cancel"]"#)),
            None,
        )
        .unwrap();
        assert_eq!(opts, vec!["refund", "cancel"]);
    }

    #[test]
    fn test_render_options_choice_list_criteria_rejects_non_strings() {
        assert!(render_options(QuestionType::Choice, Some(&crit("[1,2]")), None).is_err());
    }

    #[test]
    fn test_render_options_choice_keeps_zero_and_false_criteria() {
        let opts = render_options(
            QuestionType::Choice,
            Some(&crit(r#"{"a":0,"b":false}"#)),
            None,
        )
        .unwrap();
        assert_eq!(opts, vec!["a: 0", "b: false"], "{opts:?}");
    }

    #[test]
    fn test_render_options_choice_described_criterion() {
        let opts = render_options(
            QuestionType::Choice,
            Some(&crit(r#"{"refund":"give money back"}"#)),
            None,
        )
        .unwrap();
        assert_eq!(opts, vec!["refund: give money back"]);
    }

    #[test]
    fn test_render_options_score_prefixes_level_index() {
        let opts = render_options(
            QuestionType::Score,
            Some(&crit(r#"["bad","ok","good"]"#)),
            None,
        )
        .unwrap();
        assert_eq!(
            opts,
            vec!["level 0: bad", "level 1: ok", "level 2: good"]
        );
    }

    #[test]
    fn test_render_options_noul_defaults_are_full_sentences() {
        // Regression: bare "false"/"true" is WRONG -- upstream renders sentences.
        let opts = render_options(QuestionType::Noul, None, None).unwrap();
        assert_eq!(
            opts,
            vec![
                "false: no, the statement does not hold",
                "true: yes, the statement holds",
            ]
        );
    }

    #[test]
    fn test_render_options_noul_false_then_true_order() {
        // Semantic order is always [false, true] regardless of object order.
        let opts = render_options(
            QuestionType::Noul,
            Some(&crit(r#"{"true":"T","false":"F"}"#)),
            None,
        )
        .unwrap();
        assert_eq!(opts, vec!["false: F", "true: T"]);
    }

    #[test]
    fn test_render_options_noul_accepts_mixed_case_crit_keys() {
        // upstream lowercases crit keys before use
        let opts =
            render_options(QuestionType::Noul, Some(&crit(r#"{"True":"T","False":"F"}"#)), None)
                .unwrap();
        assert_eq!(opts, vec!["false: F", "true: T"]);
    }

    #[test]
    fn test_render_options_noul_custom_labels() {
        let labels: BTreeMap<String, String> = [
            ("false".to_string(), "nope".to_string()),
            ("true".to_string(), "yep".to_string()),
        ]
        .into();
        let opts = render_options(QuestionType::Noul, None, Some(&labels)).unwrap();
        assert_eq!(
            opts,
            vec![
                "nope: no, the statement does not hold",
                "yep: yes, the statement holds",
            ]
        );
    }

    #[test]
    fn test_render_options_noul_blank_crit_falls_back_to_sentences() {
        let opts =
            render_options(QuestionType::Noul, Some(&crit(r#"{"false":"","true":null}"#)), None)
                .unwrap();
        assert_eq!(
            opts,
            vec![
                "false: no, the statement does not hold",
                "true: yes, the statement holds",
            ]
        );
    }

    #[test]
    fn test_render_options_labels_only_for_noul() {
        let labels: BTreeMap<String, String> =
            [("false".into(), "n".into()), ("true".into(), "y".into())].into();
        assert!(render_options(QuestionType::Choice, Some(&crit(r#"{"a":"b"}"#)), Some(&labels))
            .is_err());
    }

    #[test]
    fn test_render_options_choice_requires_criteria() {
        assert!(render_options(QuestionType::Choice, None, None).is_err());
    }

    // ---- noul labels -----------------------------------------------------

    #[test]
    fn test_resolve_noul_labels_rejects_distinctness_violation() {
        let labels: BTreeMap<String, String> =
            [("false".into(), "x".into()), ("true".into(), "x".into())].into();
        assert!(resolve_noul_labels(Some(&labels)).is_err());
    }

    #[test]
    fn test_resolve_noul_labels_rejects_empty() {
        let labels: BTreeMap<String, String> =
            [("false".into(), " ".into()), ("true".into(), "y".into())].into();
        assert!(resolve_noul_labels(Some(&labels)).is_err());
    }

    #[test]
    fn test_resolve_noul_labels_trims() {
        let labels: BTreeMap<String, String> =
            [("false".into(), " n ".into()), ("true".into(), "y".into())].into();
        assert_eq!(resolve_noul_labels(Some(&labels)).unwrap(), ("n".into(), "y".into()));
    }

    #[test]
    fn test_resolve_noul_labels_default_is_false_true() {
        assert_eq!(
            resolve_noul_labels(None).unwrap(),
            ("false".to_string(), "true".to_string())
        );
    }

    // ---- serialize_state -------------------------------------------------

    #[test]
    fn test_serialize_state_string_passes_through() {
        assert_eq!(serialize_state(&Value::String("hello".into())), "hello");
    }

    #[test]
    fn test_serialize_state_dict_becomes_json() {
        assert_eq!(serialize_state(&json(r#"{"a":1}"#)), "{\"a\": 1}");
    }

    #[test]
    fn test_serialize_state_list_becomes_json() {
        assert_eq!(serialize_state(&json("[1,2]")), "[1, 2]");
    }

    // ---- constants -------------------------------------------------------

    #[test]
    fn test_prompt_defaults_match_upstream_signature() {
        let cfg = PromptConfig::default();
        assert_eq!(cfg.max_len, DEFAULT_MAX_LEN);
        assert_eq!(cfg.head_max_len, DEFAULT_HEAD_MAX_LEN);
    }

    #[test]
    fn test_budget_constants_match_upstream_literals() {
        assert_eq!(MAX_OPTION_TOKENS, 48);
        assert_eq!(MIN_INSTRUCTION_BUDGET, 8);
        assert_eq!(MIN_OPTION_BUDGET, 16);
        assert_eq!(MIN_PER_OPTION, 4);
    }

    #[test]
    fn test_ordered_map_round_trips_input_order() {
        let m: super::super::OrderedMap =
            serde_json::from_str(r#"{"z":1,"a":2,"m":3}"#).unwrap();
        let keys: Vec<&str> = m.keys().collect();
        assert_eq!(keys, vec!["z", "a", "m"]);
        // serializing keeps that order too
        assert_eq!(serde_json::to_string(&m).unwrap(), r#"{"z":1,"a":2,"m":3}"#);
    }

    #[test]
    fn test_ordered_map_lookup_is_case_insensitive_on_request() {
        let m: super::super::OrderedMap = serde_json::from_str(r#"{"False":1,"TRUE":2}"#).unwrap();
        assert_eq!(m.get("false"), None);
        assert_eq!(m.get("False"), Some(&json("1")));
        assert_eq!(m.get_ci("false"), Some(&json("1")));
        assert_eq!(m.get_ci("true"), Some(&json("2")));
    }
}

/// Replay of the golden vectors produced by `scripts/gen-laya-prompt-golden.py`
/// from the *installed* `laya` package.
///
/// The generator calls `laya.common.build_sequence` / `render_options` /
/// `serialize_state` and records what came back. Anything here that differs
/// from those vectors means the port is not bit-exact and the model would be
/// fed a template it was not trained on.
#[cfg(test)]
mod golden_tests {
    use super::*;

    const GOLDEN: &str = include_str!("../../tests/fixtures/laya_prompt_golden.json");
    const TOKENIZER_JSON: &str = include_str!("../../tests/fixtures/laya_test_tokenizer.json");

    #[derive(serde::Deserialize)]
    struct Golden {
        cases: Vec<Case>,
    }

    #[derive(serde::Deserialize)]
    struct Case {
        name: String,
        qtype: String,
        /// Un-normalised caller input, so the Rust path covers its own handling
        /// of `choice`-as-list and case-insensitive `noul` keys.
        q_raw: Value,
        truncate_left: bool,
        max_len: usize,
        head_max_len: usize,
        rendered_options: Vec<String>,
        state_text: String,
        state: Value,
        option_order: Option<Vec<usize>>,
        input_ids: Vec<i64>,
        marker_pos: Vec<i64>,
    }

    fn fixture_tokenizer() -> Tokenizer {
        TOKENIZER_JSON.parse::<Tokenizer>().expect("committed fixture tokenizer parses")
    }

    impl Case {
        fn criteria(&self) -> Option<CriteriaValue> {
            self.q_raw
                .get("crit")
                .filter(|c| !c.is_null())
                .map(|c| serde_json::from_value(c.clone()).expect("crit parses"))
        }

        fn labels(&self) -> Option<BTreeMap<String, String>> {
            self.q_raw
                .get("labels")
                .filter(|l| !l.is_null())
                .map(|l| serde_json::from_value(l.clone()).expect("labels parse"))
        }

        fn instruction(&self) -> String {
            match self.q_raw.get("ins").expect("q_raw has ins") {
                Value::String(s) => s.clone(),
                other => serialize_state(other),
            }
        }
    }

    #[test]
    fn test_fixture_tokenizer_is_space_sensitive() {
        // If this stops holding the leading-space behaviour is no longer pinned
        // and the golden test silently weakens.
        let tok = fixture_tokenizer();
        let bare = encode_text(&tok, "foo").unwrap().len();
        let spaced = encode_text(&tok, " foo").unwrap().len();
        assert_eq!(bare, 3, "one token per byte");
        assert_eq!(spaced, 4, "the leading space must be its own token");
    }

    #[test]
    fn test_golden_serialize_state_matches_reference() {
        let golden: Golden = serde_json::from_str(GOLDEN).unwrap();
        assert!(!golden.cases.is_empty());
        for case in &golden.cases {
            assert_eq!(serialize_state(&case.state), case.state_text, "case {}", case.name);
        }
    }

    #[test]
    fn test_golden_render_options_matches_reference() {
        let golden: Golden = serde_json::from_str(GOLDEN).unwrap();
        for case in &golden.cases {
            let qt = QuestionType::parse(&case.qtype).unwrap();
            let criteria = case.criteria();
            let labels = case.labels();
            let got = render_options(qt, criteria.as_ref(), labels.as_ref())
                .unwrap_or_else(|e| panic!("{}: render_options failed: {e}", case.name));
            assert_eq!(got, case.rendered_options, "case {}", case.name);
        }
    }

    #[test]
    fn test_golden_build_sequence_matches_reference() {
        let tok = fixture_tokenizer();
        let specials = SpecialIds::from_tokenizer(&tok).expect("fixture has the template tokens");
        let golden: Golden = serde_json::from_str(GOLDEN).unwrap();

        let mut saw_dropped_markers = false;
        for case in &golden.cases {
            let qt = QuestionType::parse(&case.qtype).unwrap();
            let criteria = case.criteria();
            let labels = case.labels();
            let cfg = PromptConfig { max_len: case.max_len, head_max_len: case.head_max_len };

            let enc = build_sequence(
                &tok,
                &specials,
                qt,
                &case.instruction(),
                criteria.as_ref(),
                labels.as_ref(),
                None,
                case.option_order.as_deref(),
                &case.state,
                None,
                &cfg,
                case.truncate_left,
            )
            .unwrap_or_else(|e| panic!("{}: build_sequence failed: {e}", case.name));

            assert_eq!(enc.input_ids, case.input_ids, "case {} input_ids", case.name);
            assert_eq!(enc.marker_pos, case.marker_pos, "case {} marker_pos", case.name);
            assert!(enc.seq_len <= case.max_len, "case {} overran max_len", case.name);
            assert_eq!(enc.marker_mask, vec![true; enc.marker_pos.len()]);
            if enc.marker_pos.len() < case.rendered_options.len() {
                saw_dropped_markers = true;
            }
        }
        assert!(
            saw_dropped_markers,
            "fixture must include a case where markers fall past max_len"
        );
    }

    #[test]
    fn test_golden_fixture_covers_all_three_qtypes() {
        let golden: Golden = serde_json::from_str(GOLDEN).unwrap();
        for wanted in ["choice", "score", "noul"] {
            assert!(
                golden.cases.iter().any(|c| c.qtype == wanted),
                "fixture has no {wanted} case"
            );
        }
    }

    #[test]
    fn test_build_sequence_rejects_out_of_range_option_order() {
        let tok = fixture_tokenizer();
        let specials = SpecialIds::from_tokenizer(&tok).unwrap();
        let err = build_sequence(
            &tok,
            &specials,
            QuestionType::Choice,
            "pick",
            Some(&CriteriaValue::Rubric(Default::default())),
            None,
            Some(&["a".to_string(), "b".to_string()]),
            Some(&[0, 5]),
            &Value::String("s".into()),
            None,
            &PromptConfig::default(),
            false,
        );
        assert!(err.is_err());
    }
}
