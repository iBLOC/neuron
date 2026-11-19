//! Token-level input contract for Ming-3.0-tts (dense 0.5B).
//!
//! Line-level port of upstream `prompt_contract.py`. The released
//! checkpoint learned one exact token layout; rebuilding it by hand in
//! every caller is how silent quality regressions happen, so the sequence
//! is assembled here and the tokenizer is validated against it at load
//! time rather than producing plausible-sounding but wrong audio.

use super::ContractTokenizer;

/// A test-only tokenizer stub shared by the ming3 tests: the pinned special
/// tokens resolve to their trained ids; everything else encodes
/// deterministically to fresh memoized ids (so composed strings equal the
/// concatenation of their parts, like a real BPE tokenizer with added
/// tokens).
#[cfg(test)]
pub(crate) mod tests_stub {
    use super::DENSE_TOKEN_IDS;
    use crate::tts::ming3::ContractTokenizer;
    use std::collections::HashMap;
    use std::sync::Mutex;

    pub struct StubTokenizer {
        specials: HashMap<&'static str, u32>,
        next_id: Mutex<u32>,
        /// When set, fresh ids cycle within `[vocab/2, vocab-2]` so they
        /// never leave a tiny model's embedding table.
        vocab_cap: Option<u32>,
        memo: Mutex<HashMap<String, Vec<u32>>>,
    }

    impl StubTokenizer {
        pub fn new() -> Self {
            let mut specials = HashMap::new();
            for (token, id) in DENSE_TOKEN_IDS {
                specials.insert(*token, *id);
            }
            Self {
                specials,
                next_id: Mutex::new(1_000_000),
                vocab_cap: None,
                memo: Mutex::new(HashMap::new()),
            }
        }

        /// A variant for tiny in-memory models: every id (specials
        /// included) lands below `vocab` so embedding gathers stay in
        /// range. The ids keep their relative order, so contract tests on
        /// positions/structure behave identically.
        pub fn with_small_vocab(vocab: u32) -> Self {
            let mut stub = Self::new();
            for (index, (token, _id)) in DENSE_TOKEN_IDS.iter().enumerate() {
                let id = (index as u32 + 1) % vocab.max(1);
                stub.specials.insert(*token, id);
            }
            *stub.next_id.lock().unwrap() = 0;
            stub.vocab_cap = Some(vocab);
            stub
        }

        fn fresh_id(&self) -> u32 {
            let mut next = self.next_id.lock().unwrap();
            let counter = *next;
            *next += 1;
            match self.vocab_cap {
                Some(v) => {
                    let base = v / 2;
                    let span = (v - base - 1).max(1);
                    base + (counter % span)
                }
                None => counter,
            }
        }
    }

    impl Default for StubTokenizer {
        fn default() -> Self {
            Self::new()
        }
    }

    impl ContractTokenizer for StubTokenizer {
        fn encode_text(&self, value: &str) -> Vec<u32> {
            let mut memo = self.memo.lock().unwrap();
            if let Some(ids) = memo.get(value) {
                return ids.clone();
            }
            // Split the string on the pinned special tokens, encode each
            // remainder to fresh ids.
            let mut ids: Vec<u32> = Vec::new();
            let mut rest = value;
            loop {
                let hit = self
                    .specials
                    .keys()
                    .filter_map(|token| rest.find(token).map(|at| (at, *token)))
                    .min_by_key(|(at, _)| *at);
                match hit {
                    Some((at, token)) => {
                        let (before, after) = rest.split_at(at);
                        if !before.is_empty() {
                            ids.push(self.fresh_id());
                        }
                        ids.push(self.specials[&token]);
                        rest = &after[token.len()..];
                    }
                    None => break,
                }
            }
            if !rest.is_empty() {
                // Real CJK BPE emits (roughly) one token per character.
                for _ in rest.chars() {
                    ids.push(self.fresh_id());
                }
            }
            memo.insert(value.to_string(), ids.clone());
            ids
        }

        fn token_id(&self, token: &str) -> Option<u32> {
            self.specials.get(token).copied()
        }
    }
}

// --- Dense (Qwen2-based 0.5b) -------------------------------------------

pub const DENSE_SYSTEM_PREFIX: &str = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
pub const DENSE_USER_PREFIX: &str = "<|im_start|>user\n";
pub const DENSE_USER_SUFFIX: &str = "<|im_end|>\n";
pub const DENSE_ASSISTANT_PREFIX: &str = "<|im_start|>assistant\n";
pub const DENSE_SPEAKER_START: &str = "<|vision_start|>";
pub const DENSE_SPEAKER_PAD: &str = "<|vision_pad|>";
pub const DENSE_SPEAKER_END: &str = "<|vision_end|>\n";
/// The checkpoint was trained with the assistant-side caption terminated by
/// `<|im_end|>`, not `<|endoftext|>`.
pub const DENSE_INSTRUCTION_EOS: &str = "<|im_end|>";

/// Token IDs the released 0.5b checkpoint was trained with. Names alone are
/// not enough: a stock Qwen2 tokenizer with the same extra tokens added in
/// a different order gives every name the right spelling and the wrong
/// embedding row, which degrades audio without raising anything.
pub const DENSE_TOKEN_IDS: &[(&str, u32)] = &[
    ("<|endoftext|>", 151643),
    ("<|im_start|>", 151644),
    ("<|im_end|>", 151645),
    ("<|vision_start|>", 151652),
    ("<|vision_end|>", 151653),
    ("<|vision_pad|>", 151654),
    ("<audioPatch>", 151705),
    ("<audio>", 151706),
];

// --- Shared --------------------------------------------------------------

pub const TEXT_INPUT_PREFIX: &str = " Text input:\n";
pub const AUDIO_START: &str = "<audio>";
pub const AUDIO_PATCH: &str = "<audioPatch>";

/// A BGM caption carries its own field list and must not get the text prefix.
const BGM_MARKERS: [&str; 5] = ["Genre: ", "Mood: ", "Instrument: ", "Theme: ", "Duration: "];

/// The tokens the tokenizer must be able to express without splitting.
fn required_tokens() -> [&'static str; 5] {
    [
        AUDIO_START,
        AUDIO_PATCH,
        DENSE_SPEAKER_START,
        DENSE_SPEAKER_PAD,
        DENSE_INSTRUCTION_EOS,
    ]
}

/// Fail loudly when the tokenizer cannot express the trained layout.
///
/// A tokenizer that silently splits `<audioPatch>` into sub-words still
/// produces audio, just conditioned on a sequence the model never saw.
pub fn validate_tokenizer<T: ContractTokenizer + ?Sized>(tokenizer: &T) -> Result<(), String> {
    let mut broken: Vec<(String, String)> = Vec::new();
    for token in required_tokens() {
        let Some(token_id) = tokenizer.token_id(token) else {
            broken.push((token.to_string(), "None".to_string()));
            continue;
        };
        let encoded = tokenizer.encode_text(token);
        if encoded != [token_id] {
            broken.push((token.to_string(), format!("{encoded:?}")));
        }
    }
    if !broken.is_empty() {
        let details = broken
            .iter()
            .map(|(name, got)| format!("{name:?} -> {got:?}"))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "Tokenizer does not match the Ming-3.0-tts dense contract; these tokens are \
             missing or split: {details}. Load the tokenizer shipped with the checkpoint."
        ));
    }

    let mut mismatched: Vec<(String, u32, u32)> = Vec::new();
    for (token, want) in DENSE_TOKEN_IDS {
        let got = tokenizer.token_id(token).unwrap_or(u32::MAX);
        if got != *want {
            mismatched.push(((*token).to_string(), *want, got));
        }
    }
    if !mismatched.is_empty() {
        let details = mismatched
            .iter()
            .map(|(token, want, got)| format!("{token:?}: expected {want}, got {got}"))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "Tokenizer IDs do not match Ming-3.0-tts training: {details}. These IDs index \
             embedding rows, so a name-only match is not enough. Load the tokenizer shipped \
             with the checkpoint."
        ));
    }
    Ok(())
}

/// The assembled input sequence plus the rows to overwrite with features.
///
/// `speaker_positions` are the index of each `<|vision_pad|>` row (its
/// projected speaker embedding lands there); `latent_position` is the index
/// of the first prompt-latent row (`None` when there is no acoustic prompt).
pub struct BuiltInput {
    pub input_ids: Vec<u32>,
    pub speaker_positions: Vec<usize>,
    pub latent_position: Option<usize>,
}

/// Assemble the input sequence (`build_input_ids` upstream).
///
/// The caller embeds `input_ids` and then overwrites the recorded rows:
/// `speaker_positions[i]` receives the i-th projected speaker embedding and
/// `latent_position..latent_position + prompt_latent_steps` receives the
/// projected prompt latents (one row per patch).
#[allow(clippy::too_many_arguments)]
pub fn build_input_ids<T: ContractTokenizer + ?Sized>(
    tokenizer: &T,
    prompt: &str,
    text: &str,
    num_speakers: usize,
    instruction: Option<&str>,
    prompt_text: Option<&str>,
    prompt_latent_steps: usize,
) -> Result<BuiltInput, String> {
    let mut speaker_block: Vec<u32> = Vec::new();
    for index in 0..num_speakers {
        speaker_block.extend(tokenizer.encode_text(&format!("  speaker_{}:", index + 1)));
        speaker_block.extend(tokenizer.encode_text(DENSE_SPEAKER_START));
        speaker_block.extend(tokenizer.encode_text(DENSE_SPEAKER_PAD));
        speaker_block.extend(tokenizer.encode_text(DENSE_SPEAKER_END));
    }

    let instruction_block: Vec<u32> = match instruction {
        Some(instruction) => {
            let mut block = tokenizer.encode_text(instruction);
            block.extend(tokenizer.encode_text(DENSE_INSTRUCTION_EOS));
            block
        }
        None => Vec::new(),
    };

    let mut prompt_text_block: Vec<u32> = Vec::new();
    let mut latent_block: Vec<u32> = Vec::new();
    if prompt_latent_steps > 0 {
        if let Some(prompt_text) = prompt_text {
            prompt_text_block = tokenizer.encode_text(prompt_text);
            latent_block = std::iter::repeat_n(
                tokenizer.encode_text(AUDIO_PATCH),
                prompt_latent_steps,
            )
            .flatten()
            .collect();
        }
    }

    let text_prefix: Vec<u32> = if BGM_MARKERS.iter().all(|m| text.contains(m)) {
        Vec::new()
    } else {
        tokenizer.encode_text(TEXT_INPUT_PREFIX)
    };

    let mut head = tokenizer.encode_text(DENSE_SYSTEM_PREFIX);
    head.extend(tokenizer.encode_text(DENSE_USER_PREFIX));
    head.extend(tokenizer.encode_text(prompt));

    let mut turn_end = tokenizer.encode_text(DENSE_USER_SUFFIX);
    turn_end.extend(tokenizer.encode_text(DENSE_ASSISTANT_PREFIX));

    let mut input_ids = head;
    input_ids.extend(speaker_block);
    input_ids.extend(text_prefix);
    input_ids.extend(prompt_text_block);
    input_ids.extend(tokenizer.encode_text(text));
    input_ids.extend(turn_end);
    input_ids.extend(instruction_block);
    input_ids.extend(tokenizer.encode_text(AUDIO_START));

    let speaker_start_id = tokenizer
        .token_id(DENSE_SPEAKER_START)
        .ok_or("tokenizer is missing <|vision_start|>")?;
    let speaker_pad_id = tokenizer
        .token_id(DENSE_SPEAKER_PAD)
        .ok_or("tokenizer is missing <|vision_pad|>")?;
    let speaker_positions: Vec<usize> = input_ids
        .iter()
        .enumerate()
        .filter(|(_, id)| **id == speaker_start_id)
        .map(|(position, _)| position + 1)
        .collect();
    if speaker_positions.len() != num_speakers {
        return Err(format!(
            "expected {num_speakers} speaker slots, found {}",
            speaker_positions.len()
        ));
    }
    for position in &speaker_positions {
        if input_ids[*position] != speaker_pad_id {
            return Err("speaker slot is not followed by its pad token".to_string());
        }
    }

    let latent_position = if latent_block.is_empty() {
        None
    } else {
        Some(input_ids.len())
    };
    input_ids.extend(latent_block);
    Ok(BuiltInput {
        input_ids,
        speaker_positions,
        latent_position,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::tests_stub::StubTokenizer;

    #[test]
    fn validate_tokenizer_accepts_a_trained_shape() {
        let tokenizer = StubTokenizer::new();
        validate_tokenizer(&tokenizer).expect("stub matches the trained ids");
    }

    #[test]
    fn validate_tokenizer_rejects_a_tokenizer_that_splits_markers() {
        let tokenizer = StubTokenizer::new();
        // Drop <audioPatch> from the id table: token_to_id misses → broken.
        let partial = UnawareTokenizer(&tokenizer);
        let err = validate_tokenizer(&partial).unwrap_err();
        assert!(err.contains("missing or split"), "got: {err}");
    }

    struct UnawareTokenizer<'a>(&'a StubTokenizer);

    impl ContractTokenizer for UnawareTokenizer<'_> {
        fn encode_text(&self, value: &str) -> Vec<u32> {
            // Splits <audioPatch> into sub-word pieces — the exact failure
            // mode the contract refuses.
            self.0
                .encode_text(value)
                .into_iter()
                .flat_map(|id| vec![id, id])
                .collect()
        }

        fn token_id(&self, token: &str) -> Option<u32> {
            self.0.token_id(token)
        }
    }

    #[test]
    fn validate_tokenizer_rejects_a_name_only_match() {
        // Same token names, but the audio ids are shifted AND the encoder
        // agrees with the shifted ids: both per-token checks pass and the
        // failure is caught only by the pinned-id table — the
        // "right spelling, wrong embedding row" case.
        struct Shifted(StubTokenizer);
        impl ContractTokenizer for Shifted {
            fn encode_text(&self, value: &str) -> Vec<u32> {
                // Consistent with the shifted id table: each special token
                // encodes to its own (shifted) single id.
                if value == "<audioPatch>" {
                    return vec![151704];
                }
                if value == "<audio>" {
                    return vec![151705];
                }
                self.0.encode_text(value)
            }
            fn token_id(&self, token: &str) -> Option<u32> {
                match token {
                    "<audioPatch>" => Some(151704),
                    "<audio>" => Some(151705),
                    other => self.0.token_id(other),
                }
            }
        }
        let shifted = Shifted(StubTokenizer::new());
        let err = validate_tokenizer(&shifted).unwrap_err();
        assert!(err.contains("expected"), "got: {err}");
        assert!(err.contains("151705"), "got: {err}");
        assert!(err.contains("151706"), "got: {err}");
    }

    #[test]
    fn build_input_ids_no_speakers_no_prompt() {
        let tokenizer = StubTokenizer::new();
        let built = build_input_ids(
            &tokenizer,
            "Please generate speech based on the following description.\n",
            "你好。",
            0,
            None,
            None,
            0,
        )
        .expect("built");

        // head(system+user+prompt) + text_prefix + text + turn_end + <audio>
        let expected_len = tokenizer.encode_text(DENSE_SYSTEM_PREFIX).len()
            + tokenizer.encode_text(DENSE_USER_PREFIX).len()
            + tokenizer.encode_text("Please generate speech based on the following description.\n").len()
            + tokenizer.encode_text(TEXT_INPUT_PREFIX).len()
            + tokenizer.encode_text("你好。").len()
            + tokenizer.encode_text(DENSE_USER_SUFFIX).len()
            + tokenizer.encode_text(DENSE_ASSISTANT_PREFIX).len()
            + tokenizer.encode_text(AUDIO_START).len();
        assert_eq!(built.input_ids.len(), expected_len);
        assert!(built.speaker_positions.is_empty());
        assert_eq!(built.latent_position, None);
        // The sequence ends on the audio start marker.
        let audio_id = tokenizer.token_id(AUDIO_START).unwrap();
        assert_eq!(*built.input_ids.last().unwrap(), audio_id);
    }

    #[test]
    fn build_input_ids_splices_speaker_and_latent_rows() {
        let tokenizer = StubTokenizer::new();
        let prompt_latent_steps = 7;
        let built = build_input_ids(
            &tokenizer,
            "prompt",
            "hello",
            1,
            Some("audio caption"),
            Some("transcript"),
            prompt_latent_steps,
        )
        .expect("built");

        // Exactly one speaker slot, and it is the row right after
        // <|vision_start|>.
        assert_eq!(built.speaker_positions.len(), 1);
        let pad_id = tokenizer.token_id(DENSE_SPEAKER_PAD).unwrap();
        assert_eq!(built.input_ids[built.speaker_positions[0]], pad_id);

        // The latent block sits at the tail: <audioPatch> × steps, directly
        // after <audio>.
        let patch_id = tokenizer.token_id(AUDIO_PATCH).unwrap();
        let latent_start = built.latent_position.expect("latent block present");
        assert_eq!(built.input_ids.len() - latent_start, prompt_latent_steps);
        assert!(built.input_ids[latent_start..]
            .iter()
            .all(|id| *id == patch_id));
        let audio_id = tokenizer.token_id(AUDIO_START).unwrap();
        assert_eq!(built.input_ids[latent_start - 1], audio_id);
    }

    #[test]
    fn build_input_ids_bgm_caption_skips_the_text_prefix() {
        let tokenizer = StubTokenizer::new();
        let bgm = "Genre: pop. Mood: calm. Instrument: piano. Theme: rain. Duration: 3s.";
        let built_bgm = build_input_ids(&tokenizer, "prompt", bgm, 0, None, None, 0).expect("bgm");
        // Same prompt and plain text: identical except the prefix block.
        let plain = "plain words here";
        let built_plain =
            build_input_ids(&tokenizer, "prompt", plain, 0, None, None, 0).expect("plain");
        let prefix_ids = tokenizer.encode_text(TEXT_INPUT_PREFIX);
        let plain_ids = tokenizer.encode_text(plain);
        let bgm_ids = tokenizer.encode_text(bgm);
        let expected_len = built_plain.input_ids.len() - plain_ids.len() + bgm_ids.len()
            - prefix_ids.len();
        assert_eq!(built_bgm.input_ids.len(), expected_len);
        // And nowhere in the BGM sequence does the prefix appear.
        assert!(built_bgm
            .input_ids
            .windows(prefix_ids.len())
            .all(|w| w != prefix_ids.as_slice()));
    }

    #[test]
    fn build_input_ids_without_prompt_text_has_no_latent_block() {
        // latent steps > 0 but no transcript: no latent block, no positions.
        let tokenizer = StubTokenizer::new();
        let built = build_input_ids(&tokenizer, "prompt", "hi", 0, None, None, 5).expect("built");
        assert_eq!(built.latent_position, None);
    }
}
