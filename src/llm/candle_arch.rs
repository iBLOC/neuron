//! Causal LM architecture detection for the Candle backend.
//!
//! This module isolates the "what kind of model is on disk?" logic
//! from the rest of the runtime. It exposes:
//!
//! - [`CausalArch`] — the un-built config + model_type string tuple
//!   returned by [`detect_arch`]. The runtime later matches on this
//!   to construct the appropriate `CausalModel`.
//! - [`detect_arch`] — parses `config.json` and dispatches on the
//!   `model_type` field across the 6 supported families
//!   (Qwen 2 / Qwen 3 / Llama / Phi-3 / Mistral / ChatGLM).
//! - [`detect_model_type`] — same config.json read but only
//!   extracts the `model_type` string. Used by tests and
//!   diagnostics when the full config is not needed.
//!
//! The dispatch follows the same per-family loader pattern used by
//! [`EricLBuehler/candle-vllm`](https://github.com/EricLBuehler/candle-vllm)
//! (see its `src/openai/models/` directory): one independent
//! function per architecture, each with its own `load_config` and
//! optional family-specific JSON-shape handling. New families can
//! be added by writing a new `load_X` function and adding one
//! match arm to [`detect_arch`] — no edits to existing families.

use candle_transformers::models::{chatglm, llama, mistral, phi3, qwen2, qwen3};

/// Architecture variant for a loaded causal LM model.
///
/// Each variant carries the candle-side Config type so that
/// `detect_arch` can hold an un-built config alongside the
/// model_type string. Qwen 2.5 and Qwen 3.5 are arch-compatible
/// with their .0 versions (same candle module), so we only keep
/// one variant per major version.
#[derive(Debug)]
pub enum CausalArch {
    Qwen2(qwen2::Config),
    Qwen3(qwen3::Config),
    Llama(llama::LlamaConfig),
    Phi3(phi3::Config),
    Mistral(mistral::Config),
    ChatGlm(chatglm::Config),
}

// -- per-family loaders --------------------------------------------------
//
// Each `load_X` reads the same `config_data` blob and returns either
// the parsed Config wrapped in the corresponding CausalArch variant,
// or a parse-error string. detect_arch's big match becomes a one-line
// dispatch to one of these.

fn load_qwen2(config_data: &str, model_type: String) -> Result<CausalArch, String> {
    let cfg: qwen2::Config =
        serde_json::from_str(config_data).map_err(|e| format!("parse as qwen2 config: {e}"))?;
    Ok(CausalArch::Qwen2(cfg).with_model_type(model_type))
}

fn load_qwen3(config_data: &str, model_type: String) -> Result<CausalArch, String> {
    let cfg: qwen3::Config =
        serde_json::from_str(config_data).map_err(|e| format!("parse as qwen3 config: {e}"))?;
    Ok(CausalArch::Qwen3(cfg).with_model_type(model_type))
}

fn load_llama(config_data: &str, model_type: String) -> Result<CausalArch, String> {
    let cfg: llama::LlamaConfig =
        serde_json::from_str(config_data).map_err(|e| format!("parse as llama config: {e}"))?;
    Ok(CausalArch::Llama(cfg).with_model_type(model_type))
}

fn load_phi3(config_data: &str, model_type: String) -> Result<CausalArch, String> {
    let cfg: phi3::Config =
        serde_json::from_str(config_data).map_err(|e| format!("parse as phi3 config: {e}"))?;
    Ok(CausalArch::Phi3(cfg).with_model_type(model_type))
}

fn load_mistral(config_data: &str, model_type: String) -> Result<CausalArch, String> {
    let cfg: mistral::Config =
        serde_json::from_str(config_data).map_err(|e| format!("parse as mistral config: {e}"))?;
    Ok(CausalArch::Mistral(cfg).with_model_type(model_type))
}

fn load_chatglm(config_data: &str, model_type: String) -> Result<CausalArch, String> {
    // chatglm::Config doesn't derive Deserialize in
    // candle-transformers 0.9.2, so we parse the required
    // fields from the raw JSON and construct the struct
    // manually via `parse_chatglm_config`.
    let cfg = parse_chatglm_config(config_data)?;
    Ok(CausalArch::ChatGlm(cfg).with_model_type(model_type))
}

/// Parse a ChatGLM3-style `config.json` blob into a
/// `candle_transformers::models::chatglm::Config`.
///
/// `chatglm::Config` does not derive `Deserialize` in
/// candle-transformers 0.9.2, so we read the required integer
/// fields manually and let the rest fall back to documented
/// defaults. Kept as a free function so `load_chatglm` is just
/// a thin wrapper like the other family loaders.
fn parse_chatglm_config(config_data: &str) -> Result<chatglm::Config, String> {
    let v: serde_json::Value =
        serde_json::from_str(config_data).map_err(|e| format!("parse as chatglm config: {e}"))?;
    let f = |k: &str| -> Result<u64, String> {
        v.get(k)
            .and_then(|x| x.as_u64())
            .ok_or_else(|| format!("chatglm config missing field {k:?}"))
    };
    Ok(chatglm::Config {
        num_layers: f("num_layers")? as usize,
        padded_vocab_size: f("padded_vocab_size")? as usize,
        hidden_size: f("hidden_size")? as usize,
        ffn_hidden_size: f("ffn_hidden_size")? as usize,
        kv_channels: f("kv_channels")? as usize,
        num_attention_heads: f("num_attention_heads")? as usize,
        seq_length: f("seq_length")? as usize,
        layernorm_epsilon: v
            .get("layernorm_epsilon")
            .and_then(|x| x.as_f64())
            .unwrap_or(1e-5),
        rmsnorm: v.get("rmsnorm").and_then(|x| x.as_bool()).unwrap_or(true),
        apply_residual_connection_post_layernorm: v
            .get("apply_residual_connection_post_layernorm")
            .and_then(|x| x.as_bool())
            .unwrap_or(false),
        post_layer_norm: v
            .get("post_layer_norm")
            .and_then(|x| x.as_bool())
            .unwrap_or(false),
        add_bias_linear: v
            .get("add_bias_linear")
            .and_then(|x| x.as_bool())
            .unwrap_or(false),
        add_qkv_bias: v
            .get("add_qkv_bias")
            .and_then(|x| x.as_bool())
            .unwrap_or(true),
        bias_dropout_fusion: v
            .get("bias_dropout_fusion")
            .and_then(|x| x.as_bool())
            .unwrap_or(true),
        multi_query_attention: v
            .get("multi_query_attention")
            .and_then(|x| x.as_bool())
            .unwrap_or(false),
        multi_query_group_num: v
            .get("multi_query_group_num")
            .and_then(|x| x.as_u64())
            .map(|n| n as usize)
            .unwrap_or(0),
        apply_query_key_layer_scaling: v
            .get("apply_query_key_layer_scaling")
            .and_then(|x| x.as_bool())
            .unwrap_or(false),
        attention_softmax_in_fp32: v
            .get("attention_softmax_in_fp32")
            .and_then(|x| x.as_bool())
            .unwrap_or(false),
        fp32_residual_connection: v
            .get("fp32_residual_connection")
            .and_then(|x| x.as_bool())
            .unwrap_or(false),
    })
}

/// Tiny helper: pair a `CausalArch` variant with its model_type
/// string. The runtime used to return `(CausalArch, String)`
/// tuples; the model_type is mostly diagnostic (logged at load
/// time) so we keep it on the variant instead via a side channel.
///
/// Today this is a no-op identity helper that exists to make the
/// family loaders read as a clean `Ok(LOAD.with_model_type(...))`
/// chain. The model_type is currently dropped (the runtime only
/// logs it once and then never reads it again); we keep the
/// receiver for future use (e.g. eager load-time warnings about
/// a specific model_type the user is loading).
impl CausalArch {
    fn with_model_type(self, _model_type: String) -> Self {
        // Intentionally ignore the model_type for now — the
        // runtime's load() already eprintln!s the type before
        // calling detect_arch. If we ever need to keep the type
        // around (e.g. for a richer "which exact model is this?"
        // health field), wrap CausalArch in a struct that also
        // holds the String.
        self
    }
}

/// Parse a model's `config.json` and return the architecture
/// variant + the `model_type` string for diagnostics.
///
/// Supports the 6 families wired into the runtime: Qwen 2 / 2.5 /
/// 3 / 3.5, Llama 2 / 3, Phi-3, Mistral, ChatGLM3. Each
/// model_type arm reads the same config.json blob and deserializes
/// into the appropriate `candle_transformers::models::*` Config
/// type via a per-family loader. ChatGLM3 has no `Deserialize`
/// derive in candle-transformers 0.9.2 and is parsed field-by-
/// field in [`parse_chatglm_config`].
///
/// MoE variants of these families (`qwen3moe`, `qwen2moe`,
/// `mixtral`, `deepseek`) are explicitly rejected with a clear
/// error rather than silently dispatched to the dense loader —
/// the dense forward pass produces garbage on MoE weights.
pub fn detect_arch(model_dir: &std::path::Path) -> Result<(CausalArch, String), String> {
    let config_path = model_dir.join("config.json");
    let config_data =
        std::fs::read_to_string(&config_path).map_err(|e| format!("read config.json: {e}"))?;
    let config_json: serde_json::Value =
        serde_json::from_str(&config_data).map_err(|e| format!("parse config.json: {e}"))?;

    let model_type = config_json
        .get("model_type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "config.json missing 'model_type' field".to_string())?
        .to_string();

    let arch = match model_type.as_str() {
        // Qwen 2 family (Qwen 2.0 / 2.5 are arch-compatible; both
        // ship config.json with model_type="qwen2"). Some HF
        // repos use the dotted/underscored variants of the name
        // — accept all three spellings.
        "qwen2" | "qwen2_5" | "qwen2.5" => load_qwen2(&config_data, model_type.clone()),

        // Qwen 3 family (Qwen 3.0 / 3.5). qwen3moe is rejected
        // below — see the MoE block.
        "qwen3" | "qwen3_5" | "qwen3.5" => load_qwen3(&config_data, model_type.clone()),

        // Llama 2 / 3 / 3.1 / 3.2 — all share the same
        // candle-transformers LlamaConfig. The "llama2" spelling
        // covers older HF repos that predate the rename to
        // "llama".
        "llama" | "llama2" => load_llama(&config_data, model_type.clone()),

        // Phi-3 (mini / small / medium) only. Phi-2 uses a
        // different config layout and isn't supported here.
        "phi3" => load_phi3(&config_data, model_type.clone()),

        // Mistral 7B. mixtral (MoE) is rejected below.
        "mistral" => load_mistral(&config_data, model_type.clone()),

        // ChatGLM3 (THUDM). chatglm2 / chatglm4 use a different
        // arch and aren't supported.
        "chatglm" => load_chatglm(&config_data, model_type.clone()),

        // -- MoE families: explicitly rejected.
        //
        // These have the same model_type stem as a supported
        // dense family but a different weight layout. Silently
        // falling through to the dense loader would parse fine
        // (qwen2::Config ignores unknown fields by default)
        // but produce garbage output at forward time. Better to
        // error out at load time with a clear message.
        "qwen2moe" | "qwen3moe" | "qwen2_moe" | "qwen3_moe" | "mixtral" | "deepseek"
        | "deepseek_v2" | "deepseek_v3" => {
            return Err(format!(
                "MoE model_type {model_type:?} is not supported. \
                 neuron only loads dense causal LMs. For MoE inference, \
                 use candle-vllm or another MoE-aware runtime."
            ));
        }

        other => Err(format!(
            "unsupported model_type: {other:?} (supported: \
             qwen2, qwen2_5, qwen2.5, qwen3, qwen3_5, qwen3.5, \
             llama, llama2, phi3, mistral, chatglm)"
        )),
    }?;
    Ok((arch, model_type))
}

/// Detect only the `model_type` string from `config.json` (for
/// tests and diagnostic use). Unlike `detect_arch`, this does NOT
/// require the full Config to be parseable — it works on any
/// valid `config.json` even when the model_type is one we don't
/// support.
pub fn detect_model_type(model_dir: &std::path::Path) -> Result<String, String> {
    let config_path = model_dir.join("config.json");
    let config_data =
        std::fs::read_to_string(&config_path).map_err(|e| format!("read config.json: {e}"))?;
    let config_json: serde_json::Value =
        serde_json::from_str(&config_data).map_err(|e| format!("parse config.json: {e}"))?;

    config_json
        .get("model_type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "config.json missing 'model_type' field".to_string())
        .map(String::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a synthetic MiniCPM5-2B `config.json` mirroring the
    /// real OpenBMB checkpoint (Standard `LlamaForCausalLM`,
    /// 2026-09-07 release, 2.52B params, 42 layers, GQA 16Q/2KV,
    /// 131K context, vocab 130560, rope_theta 5_000_000,
    /// `tie_word_embeddings = false`) to a tempdir and confirm
    /// `detect_arch` dispatches it through the `CausalArch::Llama`
    /// variant without losing any structural field. The companion
    /// catalog-side test in `src/capabilities/catalog.rs` only
    /// verifies the catalog entry; this test is the candle-path
    /// guarantee. Field values are mirrored verbatim from the HF
    /// config.json at `huggingface:openbmb/MiniCPM5-2B` — keep
    /// them in sync if OpenBMB revises the checkpoint.
    #[test]
    fn test_detect_arch_minicpm5_2b_dispatches_to_llama() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("config.json"),
            r#"{
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": 2048,
                "intermediate_size": 6144,
                "num_attention_heads": 16,
                "num_hidden_layers": 42,
                "num_key_value_heads": 2,
                "head_dim": 128,
                "max_position_embeddings": 131072,
                "rms_norm_eps": 1e-06,
                "rope_theta": 5000000,
                "rope_scaling": null,
                "tie_word_embeddings": false,
                "torch_dtype": "bfloat16",
                "vocab_size": 130560,
                "bos_token_id": 0,
                "eos_token_id": [1, 130073],
                "use_cache": true
            }"#,
        )
        .expect("write synthetic minicpm5-2b config.json");
        let (arch, model_type) =
            detect_arch(dir.path()).expect("detect_arch should accept Standard LlamaForCausalLM");
        assert_eq!(
            model_type, "llama",
            "MiniCPM5-2B must dispatch through the Llama model_type"
        );
        match arch {
            CausalArch::Llama(cfg) => {
                assert_eq!(cfg.hidden_size, 2048);
                assert_eq!(cfg.intermediate_size, 6144);
                assert_eq!(cfg.num_attention_heads, 16);
                // `num_key_value_heads` is Option<usize> in
                // candle-transformers 0.10.2; GQA=2 means Some(2).
                assert_eq!(cfg.num_key_value_heads, Some(2));
                assert_eq!(cfg.num_hidden_layers, 42);
                assert_eq!(cfg.max_position_embeddings, 131072);
                assert_eq!(cfg.vocab_size, 130560);
                assert_eq!(
                    cfg.rope_theta, 5_000_000.0,
                    "MiniCPM5-2B uses OpenBMB's rope_theta=5e6, not the Llama default 1e4"
                );
                assert_eq!(
                    cfg.tie_word_embeddings,
                    Some(false),
                    "MiniCPM5-2B ships with tie_word_embeddings=false \
                     (separate embed/output projection); candle-transformers \
                     exposes this as Option<bool>"
                );
            }
            other => panic!(
                "expected CausalArch::Llama for minicpm5-2b, got {other:?}"
            ),
        }
    }
}
