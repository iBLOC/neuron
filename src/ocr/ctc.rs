//! CTC greedy decoding + PP-OCR vocabulary handling.

use std::path::Path;

use crate::llm::NeuronError;

/// PP-OCR recognition vocabulary.
///
/// Index layout mirrors PaddleOCR's `CTCLabelDecode`: token 0 is the CTC
/// blank, tokens `1..=len(dict)` are the dict lines in file order, and the
/// last token is a literal space. For `ppocrv5_dict.txt` that is
/// 1 + 18383 + 1 = 18385 — exactly the export's output vocab, verified
/// 2026-10-09 (v6: 1 + 18708 + 1 = 18710).
#[derive(Debug, Clone)]
pub struct Vocab {
    pub tokens: Vec<String>,
}

impl Vocab {
    pub fn from_dict_lines(lines: Vec<String>) -> Self {
        let mut tokens = Vec::with_capacity(lines.len() + 2);
        tokens.push("blank".to_string());
        tokens.extend(lines);
        tokens.push(" ".to_string());
        Self { tokens }
    }

    /// Load a `ppocrv5_dict.txt` / `ppocrv6_dict.txt`-shaped file: one token
    /// per line, order preserved. `\r` is stripped for Windows-edited dicts;
    /// lines are otherwise verbatim (a dict entry may legitimately be any
    /// UTF-8 string, including punctuation-only).
    pub fn load(path: &Path) -> Result<Self, NeuronError> {
        let text = std::fs::read_to_string(path).map_err(|e| {
            NeuronError::invalid_input(format!(
                "cannot read OCR vocab {}: {e}",
                path.display()
            ))
        })?;
        let lines: Vec<String> = text
            .split('\n')
            .map(|l| l.trim_end_matches('\r').to_string())
            .collect();
        // A trailing newline on the last dict line is a file artifact, not a
        // token; an *interior* empty line would shift every index, which the
        // vocab-size check in the runtime catches.
        let mut lines = lines;
        if lines.last().is_some_and(|l| l.is_empty()) {
            lines.pop();
        }
        Ok(Self::from_dict_lines(lines))
    }

    /// CTC greedy decode over a `[T, vocab]` probability tensor (row-major).
    ///
    /// Collapses consecutive repeats and drops the blank index; confidence
    /// is the mean probability of the emitted (non-blank, non-repeat) frames.
    /// The PP-OCR rec graphs ship with a final Softmax, so rows are
    /// probabilities, not logits.
    pub fn decode(&self, probs: &[f32], t: usize, vocab_size: usize) -> (String, f32) {
        assert_eq!(probs.len(), t * vocab_size, "probs must be t*vocab");
        let mut last = 0usize;
        let mut out = String::new();
        let mut conf_sum = 0f32;
        let mut conf_n = 0u32;
        for frame in 0..t {
            let row = &probs[frame * vocab_size..(frame + 1) * vocab_size];
            // Manual argmax with strict-greater so exact ties pick the
            // LOWEST index — index 0 (blank) must win a uniform row, matching
            // numpy's argmax (Iterator::max_by would return the last tie).
            let mut idx = 0usize;
            let mut best = row[0];
            for (i, &p) in row.iter().enumerate().skip(1) {
                if p > best {
                    best = p;
                    idx = i;
                }
            }
            let p = best;
            if idx != 0 && idx != last {
                if let Some(token) = self.tokens.get(idx) {
                    out.push_str(token);
                }
                conf_sum += p;
                conf_n += 1;
            }
            last = idx;
        }
        let conf = if conf_n > 0 { conf_sum / conf_n as f32 } else { 0.0 };
        (out, conf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab() -> Vocab {
        // index: 0=blank, 1=H, 2=I, 3=空格(space via dict), ... last=" "
        Vocab::from_dict_lines(vec!["H".into(), "I".into(), "你".into()])
    }

    #[test]
    fn vocab_layout_matches_ppocr_ctc_decode() {
        let v = vocab();
        assert_eq!(v.tokens.len(), 5); // blank + 3 dict + trailing space
        assert_eq!(v.tokens[0], "blank");
        assert_eq!(v.tokens[1], "H");
        assert_eq!(v.tokens[3], "你");
        assert_eq!(v.tokens[4], " ");
    }

    #[test]
    fn greedy_collapses_repeats_and_skips_blank() {
        let v = vocab();
        // T=6, vocab=5. argmax per frame: 1,1,0,3,3,4
        // → H (repeat dropped), blank dropped, 你 (repeat dropped), space
        let mut probs = vec![0f32; 6 * 5];
        for (frame, idx) in [1usize, 1, 0, 3, 3, 4].into_iter().enumerate() {
            probs[frame * 5 + idx] = 0.9;
        }
        let (text, conf) = v.decode(&probs, 6, 5);
        assert_eq!(text, "H你 ");
        assert!((conf - 0.9).abs() < 1e-6, "{conf}");
    }

    #[test]
    fn all_blank_decodes_empty_with_zero_confidence() {
        let v = vocab();
        let probs = vec![0f32; 4 * 5];
        // every row uniform → argmax index 0 = blank
        let (text, conf) = v.decode(&probs, 4, 5);
        assert_eq!(text, "");
        assert_eq!(conf, 0.0);
    }

    #[test]
    fn confidence_is_mean_of_emitted_frames() {
        let v = vocab();
        let mut probs = vec![0f32; 3 * 5];
        probs[0 * 5 + 1] = 0.7; // H @ 0.7
        probs[1 * 5 + 0] = 1.0; // blank (not counted)
        probs[2 * 5 + 2] = 0.5; // I @ 0.5
        let (text, conf) = v.decode(&probs, 3, 5);
        assert_eq!(text, "HI");
        assert!((conf - 0.6).abs() < 1e-6, "{conf}");
    }

    #[test]
    fn load_strips_cr_and_trailing_newline() {
        let dir = std::env::temp_dir().join("neuron-ocr-vocab-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dict.txt");
        std::fs::write(&path, "H\r\nI\r\n你\r\n").unwrap();
        let v = Vocab::load(&path).unwrap();
        assert_eq!(v.tokens, vec!["blank", "H", "I", "你", " "]);
    }
}
