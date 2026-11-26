//! Ming-3.0-tts (inclusionAI, dense 0.5B) support.
//!
//! Upstream: `inclusionAI/Ming-3.0-tts` (MIT), checkpoint
//! `inclusionAI/Ming-3.0-tts-0.5B` — a Qwen2.5-0.5B backbone + DiT flow
//! head + AudioVAE (44.1 kHz) + CAM++ speaker encoder, everything in one
//! `model.safetensors`.
//!
//! This module is a line-level port of the upstream Python reference. The
//! string constants, JSON caption key orders and token ids in
//! [`contract`] / [`tasks`] are part of the model input protocol: the
//! checkpoint was trained against them exactly, and a paraphrased prompt
//! silently degrades audio instead of raising.
//!
//! Layout of the port (matches the upstream file tree):
//! - [`contract`] ← `prompt_contract.py` (token-level input contract)
//! - [`tasks`] ← `tts_tasks.py` (task/prompt contracts)
//! - [`config`] ← `configuration_bailingmm.py` + `config.json` shapes
//!
//! Not ported yet (documented gaps): text normalization FST (pynini), the
//! text fragment splitter, streaming decode, voice-clone reference audio.

pub mod audiovae;
pub mod cfm;
pub mod config;
pub mod contract;
pub mod dit;
pub mod model;
pub mod runtime;
#[cfg(feature = "ming3-clone")]
pub mod spkemb;
pub mod tasks;
pub mod weights;

/// A tokenizer viewed through the interface the input contract needs.
///
/// The Python reference calls `tokenizer.encode(value, add_special_tokens=False)`
/// and `tokenizer.convert_tokens_to_ids(token)`. Both are pure lookups, so
/// the contract layer only depends on this trait and can be unit-tested
/// against a stub; the production impl wraps the checkpoint's
/// `tokenizers::Tokenizer` (the ids are pinned in [`contract::DENSE_TOKEN_IDS`]
/// and validated at load time, so a wrong tokenizer fails loudly).
pub trait ContractTokenizer: Send + Sync {
    /// `tokenizer.encode(value, add_special_tokens=False)` upstream.
    fn encode_text(&self, value: &str) -> Vec<u32>;
    /// `tokenizer.convert_tokens_to_ids(token)` upstream.
    fn token_id(&self, token: &str) -> Option<u32>;
}

impl ContractTokenizer for tokenizers::Tokenizer {
    fn encode_text(&self, value: &str) -> Vec<u32> {
        // `add_special_tokens=False` — the contract builds every marker
        // itself; a prepended BOS would shift every splice position.
        match self.encode(value, false) {
            Ok(encoding) => encoding.get_ids().to_vec(),
            Err(_) => Vec::new(),
        }
    }

    fn token_id(&self, token: &str) -> Option<u32> {
        self.token_to_id(token)
    }
}
