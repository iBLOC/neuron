//! Host-side temperature calibration for the Laya decision head.
//!
//! The decision head emits one raw logit per `[MASK]` option marker. Nothing in
//! the ONNX graph turns those logits into usable probabilities -- the
//! calibration lives entirely on the host, and getting it wrong silently turns
//! the model's output into noise. This module is the bit-exact port of
//! `laya/common.py` (`clamp_temperature`, `temp_bucket`) plus the scoring loop
//! in `laya/onnx_agent.py`:
//!
//! ```python
//! self.temperature = [clamp_temperature(t) for t in self.temperature_raw]
//! t_scale = self.temperature_by_options.get(temp_bucket(qt, k), self.temperature[qt])
//! z = logits[r, :k] / t_scale
//! p = np.exp(z - z.max()); p = p / p.sum()
//! ```
//!
//! Verified against `tests/test_temperature_loading.py`, which asserts that
//! the raw triple `[0.1, 10, -1]` clamps to `[0.5, 5.0, 0.5]`.
//!
//! The `golden_postprocess_matches_reference` test replays the vectors produced
//! by `scripts/gen-laya-golden.py`, whose reference implementation is a direct
//! transcription of the upstream Python.

use std::collections::BTreeMap;

/// Lower clamp bound for any fitted temperature.
pub const TEMP_MIN: f64 = 0.5;
/// Upper clamp bound for any fitted temperature.
pub const TEMP_MAX: f64 = 5.0;

/// The three typed question kinds Laya supports.
///
/// The numeric values are the `qtype` input fed to the ONNX graph and the index
/// into the per-type `temperature` triple.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuestionType {
    Choice,
    Score,
    Noul,
}

impl QuestionType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Choice => "choice",
            Self::Score => "score",
            Self::Noul => "noul",
        }
    }

    /// Value fed to the model's `qtype` input and index into `temperature`.
    pub fn index(&self) -> usize {
        match self {
            Self::Choice => 0,
            Self::Score => 1,
            Self::Noul => 2,
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "choice" => Ok(Self::Choice),
            "score" => Ok(Self::Score),
            "noul" => Ok(Self::Noul),
            other => Err(format!("unknown QuestionType: {other}")),
        }
    }
}

/// Fitted temperatures, one per [`QuestionType`], already clamped.
pub type Temperature = [f64; 3];

/// Per-bucket temperature overrides, keyed by [`temp_bucket`] output.
pub type TemperatureByOptions = BTreeMap<String, f64>;

/// Clamp a fitted temperature into the usable range.
///
/// Non-finite values fall back to 1.0 ("uncalibrated") rather than poisoning
/// the softmax, matching upstream `clamp_temperature`.
pub fn clamp_temperature(t: f64) -> f64 {
    if !t.is_finite() {
        return 1.0;
    }
    t.clamp(TEMP_MIN, TEMP_MAX)
}

/// Clamp a whole fitted table, as upstream does at load time.
pub fn clamp_temperature_table(raw: &[f64; 3]) -> Temperature {
    [
        clamp_temperature(raw[0]),
        clamp_temperature(raw[1]),
        clamp_temperature(raw[2]),
    ]
}

/// Clamp every per-bucket override.
pub fn clamp_temperature_by_options(raw: &BTreeMap<String, f64>) -> TemperatureByOptions {
    raw.iter()
        .map(|(k, &v)| (k.clone(), clamp_temperature(v)))
        .collect()
}

/// Bucket key for a question, of the form `"<qtype>:<size>"`.
///
/// The size band is a function of the option count only: `2` for `k <= 2`,
/// `3-5` for `k <= 5`, `6-10` for `k <= 10`, else `11+`. Note that a single
/// option (`k == 1`) lands in the `2` band.
pub fn temp_bucket(qtype: QuestionType, option_count: usize) -> String {
    let size = if option_count <= 2 {
        "2"
    } else if option_count <= 5 {
        "3-5"
    } else if option_count <= 10 {
        "6-10"
    } else {
        "11+"
    };
    format!("{}:{}", qtype.as_str(), size)
}

/// Look up the temperature to divide the logits by.
///
/// The per-bucket override wins when present; otherwise fall back to the
/// per-type value. Both are expected to already be clamped.
pub fn temperature_scale(
    by_options: &TemperatureByOptions,
    temperature: &Temperature,
    qtype: QuestionType,
    option_count: usize,
) -> f64 {
    by_options
        .get(&temp_bucket(qtype, option_count))
        .copied()
        .unwrap_or(temperature[qtype.index()])
}

/// Temperature-scaled softmax over the first `option_count` logits.
///
/// `logits` is the raw model output including the `-1e4` padding slots; only
/// the first `option_count` entries are real options and only those are scored.
pub fn calibrated_probs(logits: &[f64], option_count: usize, scale: f64) -> Vec<f64> {
    let k = option_count.min(logits.len());
    if k == 0 {
        return Vec::new();
    }
    let z: Vec<f64> = logits[..k].iter().map(|l| l / scale).collect();
    softmax(&z)
}

/// Normalized-entropy confidence: 1.0 for a one-option question or a point
/// mass, 0.0 for a uniform distribution over `option_count` options.
///
/// Port of `laya/common.py::confidence_from_probs`. Note the guard is
/// `k < 2` -- a degenerate single-option question is *fully* confident by
/// definition, not partially.
pub fn confidence_from_probs(p: &[f64], option_count: usize) -> f64 {
    if option_count < 2 {
        return 1.0;
    }
    let k = option_count.min(p.len());
    if k == 0 {
        return 1.0;
    }
    let ent: f64 = p[..k]
        .iter()
        .map(|&x| {
            let clipped = x.clamp(1e-12, 1.0);
            -x * clipped.ln()
        })
        .sum();
    (1.0 - ent / (k as f64).ln()).clamp(0.0, 1.0)
}

/// Softmax over the two-class `act_logits` (act vs escalate).
pub fn act_probs(act_logits: &[f64]) -> [f64; 2] {
    let p = softmax(act_logits);
    [p.first().copied().unwrap_or(0.5), p.get(1).copied().unwrap_or(0.5)]
}

/// Numerically stable softmax: subtract the max before exponentiating.
fn softmax(z: &[f64]) -> Vec<f64> {
    let max = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = z.iter().map(|v| (v - max).exp()).collect();
    let sum: f64 = e.iter().sum();
    if sum == 0.0 {
        let n = z.len() as f64;
        return vec![1.0 / n; z.len()];
    }
    e.into_iter().map(|v| v / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- clamp -----------------------------------------------------------

    #[test]
    fn test_clamp_temperature_low_clamps_up_to_min() {
        assert_eq!(clamp_temperature(0.1), TEMP_MIN);
    }

    #[test]
    fn test_clamp_temperature_high_clamps_down_to_max() {
        assert_eq!(clamp_temperature(10.0), TEMP_MAX);
    }

    #[test]
    fn test_clamp_temperature_negative_clamps_up_to_min() {
        // Regression: upstream test asserts [0.1, 10, -1] -> [0.5, 5.0, 0.5].
        assert_eq!(clamp_temperature(-1.0), TEMP_MIN);
    }

    #[test]
    fn test_clamp_temperature_in_range_is_identity() {
        assert_eq!(clamp_temperature(1.35), 1.35);
    }

    #[test]
    fn test_clamp_temperature_non_finite_falls_back_to_one() {
        assert_eq!(clamp_temperature(f64::NAN), 1.0);
        assert_eq!(clamp_temperature(f64::INFINITY), 1.0);
        assert_eq!(clamp_temperature(f64::NEG_INFINITY), 1.0);
    }

    #[test]
    fn test_clamp_temperature_table_matches_upstream_triple() {
        assert_eq!(
            clamp_temperature_table(&[0.1, 10.0, -1.0]),
            [0.5, 5.0, 0.5]
        );
    }

    // ---- bucket ----------------------------------------------------------

    #[test]
    fn test_temp_bucket_one_option_lands_in_two_band() {
        assert_eq!(temp_bucket(QuestionType::Choice, 1), "choice:2");
    }

    #[test]
    fn test_temp_bucket_two_options_is_two_band() {
        assert_eq!(temp_bucket(QuestionType::Choice, 2), "choice:2");
        assert_eq!(temp_bucket(QuestionType::Noul, 2), "noul:2");
    }

    #[test]
    fn test_temp_bucket_boundaries_pick_expected_bands() {
        assert_eq!(temp_bucket(QuestionType::Choice, 3), "choice:3-5");
        assert_eq!(temp_bucket(QuestionType::Choice, 5), "choice:3-5");
        assert_eq!(temp_bucket(QuestionType::Choice, 6), "choice:6-10");
        assert_eq!(temp_bucket(QuestionType::Choice, 10), "choice:6-10");
        assert_eq!(temp_bucket(QuestionType::Choice, 11), "choice:11+");
    }

    #[test]
    fn test_temp_bucket_uses_qtype_name() {
        assert_eq!(temp_bucket(QuestionType::Score, 4), "score:3-5");
    }

    // ---- scale lookup ----------------------------------------------------

    fn table(pairs: &[(&str, f64)]) -> TemperatureByOptions {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn test_temperature_scale_bucket_overrides_type_value() {
        let by = table(&[("choice:2", 0.5)]);
        let t = [0.8, 1.35, 2.25];
        assert_eq!(temperature_scale(&by, &t, QuestionType::Choice, 2), 0.5);
    }

    #[test]
    fn test_temperature_scale_missing_bucket_falls_back_to_type_value() {
        let by = table(&[]);
        let t = [0.8, 1.35, 2.25];
        assert_eq!(temperature_scale(&by, &t, QuestionType::Choice, 6), 0.8);
        assert_eq!(temperature_scale(&by, &t, QuestionType::Score, 4), 1.35);
        assert_eq!(temperature_scale(&by, &t, QuestionType::Noul, 2), 2.25);
    }

    #[test]
    fn test_temperature_scale_noul_index_is_two() {
        let by = table(&[]);
        let t = [0.8, 1.35, 2.25];
        // The Noul bucket is missing, so this must read t[2], not t[1].
        assert_eq!(temperature_scale(&by, &t, QuestionType::Noul, 2), 2.25);
    }

    // ---- softmax ---------------------------------------------------------

    #[test]
    fn test_calibrated_probs_sums_to_one() {
        let p = calibrated_probs(&[1.0, 2.0, 3.0], 3, 1.0);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_calibrated_probs_ignores_padding_slots() {
        // The ONNX graph pads logits at -1e4. Scoring all slots would be wrong.
        let with_pad = calibrated_probs(&[1.0, 2.0, -1e4, -1e4], 2, 1.0);
        let without = calibrated_probs(&[1.0, 2.0], 2, 1.0);
        for (a, b) in with_pad.iter().zip(without.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
        assert_eq!(with_pad.len(), 2);
    }

    #[test]
    fn test_calibrated_probs_higher_logit_wins() {
        let p = calibrated_probs(&[0.0, 5.0], 2, 1.0);
        assert!(p[1] > p[0]);
    }

    #[test]
    fn test_calibrated_probs_small_scale_sharpens() {
        let flat = calibrated_probs(&[1.0, 1.5], 2, 5.0);
        let sharp = calibrated_probs(&[1.0, 1.5], 2, 0.5);
        assert!(sharp[1] > flat[1], "smaller temperature must sharpen");
    }

    #[test]
    fn test_calibrated_probs_single_option_is_certain() {
        let p = calibrated_probs(&[7.0], 1, 0.5);
        assert_eq!(p, vec![1.0]);
    }

    #[test]
    fn test_calibrated_probs_empty_option_count_returns_empty() {
        assert!(calibrated_probs(&[1.0, 2.0], 0, 1.0).is_empty());
    }

    #[test]
    fn test_act_probs_sums_to_one() {
        let a = act_probs(&[333.5, -394.5]);
        assert!((a[0] + a[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_act_probs_is_stable_for_extreme_logits() {
        // Real act_logits reach ~1e3; naive exp() would overflow.
        let a = act_probs(&[1000.0, -1000.0]);
        assert!(a[0].is_finite() && a[1].is_finite());
        assert!((a[0] + a[1] - 1.0).abs() < 1e-12);
        assert!(a[0] > 0.999);
    }

    #[test]
    fn test_softmax_equal_inputs_give_uniform() {
        let p = softmax(&[2.0, 2.0, 2.0]);
        for v in p {
            assert!((v - 1.0 / 3.0).abs() < 1e-12);
        }
    }

    // ---- question type ---------------------------------------------------

    #[test]
    fn test_question_type_parse_roundtrip() {
        for (s, qt) in [("choice", QuestionType::Choice), ("score", QuestionType::Score), ("noul", QuestionType::Noul)] {
            assert_eq!(QuestionType::parse(s).unwrap(), qt);
            assert_eq!(qt.as_str(), s);
        }
    }

    #[test]
    fn test_question_type_parse_unknown_errors() {
        assert!(QuestionType::parse("classify").is_err());
    }

    #[test]
    fn test_question_type_index_matches_temperature_slot() {
        assert_eq!(QuestionType::Choice.index(), 0);
        assert_eq!(QuestionType::Score.index(), 1);
        assert_eq!(QuestionType::Noul.index(), 2);
    }

    // ---- golden vectors --------------------------------------------------
    //
    // Replays the fixture produced by scripts/gen-laya-golden.py. The reference
    // probabilities there come from a direct transcription of the upstream
    // Python, so a mismatch means this port drifted.

    #[derive(serde::Deserialize)]
    struct GoldenFile {
        postprocess: Vec<GoldenCase>,
    }

    #[derive(serde::Deserialize)]
    struct GoldenCase {
        name: String,
        qtype: String,
        option_count: usize,
        logits: Vec<f64>,
        act_logits: Vec<f64>,
        temperature_raw: Vec<f64>,
        temperature_by_options_raw: BTreeMap<String, f64>,
        expected_scale: f64,
        expected_probs: Vec<f64>,
        expected_act_probs: Vec<f64>,
    }

    #[test]
    fn test_golden_postprocess_matches_reference() {
        let raw = include_str!("../../tests/fixtures/laya_decision_golden.json");
        let file: GoldenFile = serde_json::from_str(raw).expect("fixture parses");
        assert!(!file.postprocess.is_empty(), "fixture has no postprocess cases");

        for case in &file.postprocess {
            let qtype = QuestionType::parse(&case.qtype).unwrap();
            let raw_t: [f64; 3] = [
                case.temperature_raw[0],
                case.temperature_raw[1],
                case.temperature_raw[2],
            ];
            let temperature = clamp_temperature_table(&raw_t);
            let by_options =
                clamp_temperature_by_options(&case.temperature_by_options_raw);

            let scale = temperature_scale(&by_options, &temperature, qtype, case.option_count);
            assert_eq!(
                scale, case.expected_scale,
                "{}: scale mismatch (bucket {})",
                case.name,
                temp_bucket(qtype, case.option_count)
            );

            let probs = calibrated_probs(&case.logits, case.option_count, scale);
            assert_eq!(probs.len(), case.expected_probs.len(), "{}: length", case.name);
            for (i, (got, want)) in probs.iter().zip(case.expected_probs.iter()).enumerate() {
                assert!(
                    (got - want).abs() < 1e-9,
                    "{}: prob[{i}] got {got} want {want}",
                    case.name
                );
            }

            let act = act_probs(&case.act_logits);
            for (i, (got, want)) in act.iter().zip(case.expected_act_probs.iter()).enumerate() {
                assert!(
                    (got - want).abs() < 1e-9,
                    "{}: act[{i}] got {got} want {want}",
                    case.name
                );
            }
        }
    }
}
