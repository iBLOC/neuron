use crate::llm::candle_runtime::*;
use crate::llm::{ChatMessage, ChatRequest};
use std::fs;

#[test]
fn test_candle_list_models_includes_qwen3_and_excludes_qwen3_5() {
    let runtime = CandleLlmRuntime::new();
    let models = runtime.list_models();
    let ids: Vec<&str> = models.iter().map(|m| m.id.as_str()).collect();
    assert!(ids.contains(&"qwen3-0.6b"));
    assert!(ids.contains(&"qwen2.5-0.5b"));
    // Qwen3.5 cannot be parsed by the candle qwen3 config loader (nested
    // `text_config` + hybrid linear attention) — advertising it here would
    // promise a model that fails at load time.
    assert!(!ids.contains(&"qwen3.5-0.6b"));
    assert!(!ids.contains(&"qwen3.5-4b"));
}

#[test]
fn test_decode_token_for_stream_replaces_sentencepiece_word_marker() {
    assert_eq!(
        CandleLlmRuntime::decode_token_for_stream("\u{2581}Hello"),
        " Hello"
    );
    assert_eq!(CandleLlmRuntime::decode_token_for_stream("Hello"), "Hello");
    assert_eq!(CandleLlmRuntime::decode_token_for_stream(""), "");
    assert_eq!(
        CandleLlmRuntime::decode_token_for_stream("a\u{2581}b\u{2581}c"),
        "a b c"
    );
    assert_eq!(
        CandleLlmRuntime::decode_token_for_stream("\u{2581}\u{2581}multi"),
        "  multi"
    );
}

#[test]
fn test_sample_token_from_logits_empty_returns_zero() {
    let logits: Vec<f32> = vec![];
    assert_eq!(
        CandleLlmRuntime::sample_token_from_logits(&logits, 0.0, 1.0, 0),
        0
    );
    assert_eq!(
        CandleLlmRuntime::sample_token_from_logits(&logits, 1.0, 1.0, 0),
        0
    );
    assert_eq!(
        CandleLlmRuntime::sample_token_from_logits(&logits, 0.5, 1.0, 0),
        0
    );
}

#[test]
fn test_sample_token_from_logits_zero_temperature_is_argmax() {
    let logits = vec![1.0_f32, 2.0, 3.0];
    assert_eq!(
        CandleLlmRuntime::sample_token_from_logits(&logits, 0.0, 1.0, 0),
        2
    );
    let logits = vec![3.0_f32, 1.0, 2.0];
    assert_eq!(
        CandleLlmRuntime::sample_token_from_logits(&logits, 0.0, 1.0, 0),
        0
    );
    assert_eq!(
        CandleLlmRuntime::sample_token_from_logits(&logits, -0.5, 1.0, 0),
        0
    );
}

#[test]
fn test_sample_token_from_logits_dominant_logit_always_wins() {
    let logits = vec![20.0_f32, 0.0, 0.0, 0.0, 0.0];
    for _ in 0..100 {
        assert_eq!(
            CandleLlmRuntime::sample_token_from_logits(&logits, 1.0, 1.0, 0),
            0
        );
    }
}

#[test]
fn test_sample_token_from_logits_handles_nan_without_panic() {
    let logits = vec![f32::NAN, 0.0, 0.0];
    let result = CandleLlmRuntime::sample_token_from_logits(&logits, 1.0, 1.0, 0);
    assert_eq!(result, 0);
}

#[test]
fn test_sample_token_from_logits_top_p_truncates_to_dominant() {
    let logits = vec![1.609_f32, 0.788, 0.223];
    for _ in 0..100 {
        assert_eq!(
            CandleLlmRuntime::sample_token_from_logits(&logits, 1.0, 0.5, 0),
            0
        );
    }
}

#[test]
fn test_sample_token_from_logits_top_p_excludes_tail() {
    let logits = vec![1.609_f32, 0.788, 0.223];
    for _ in 0..100 {
        let result = CandleLlmRuntime::sample_token_from_logits(&logits, 1.0, 0.8, 0);
        assert!(result < 2, "top_p=0.8 must exclude index 2, got {result}");
    }
}

#[test]
fn test_sample_token_from_logits_top_p_one_is_no_truncation() {
    let logits = vec![2.0_f32, 1.0, 0.0, 0.0];
    for _ in 0..50 {
        let result = CandleLlmRuntime::sample_token_from_logits(&logits, 0.5, 1.0, 0);
        assert!((result as usize) < logits.len());
    }
}

#[test]
fn test_sample_token_from_logits_top_p_zero_falls_back_to_argmax() {
    let logits = vec![1.0_f32, 5.0, 2.0];
    assert_eq!(
        CandleLlmRuntime::sample_token_from_logits(&logits, 0.0, 0.0, 0),
        1
    );
}

#[test]
fn test_sample_token_from_logits_uniform_input_returns_valid_index() {
    let logits = vec![1.0_f32; 10];
    for _ in 0..20 {
        let result = CandleLlmRuntime::sample_token_from_logits(&logits, 0.5, 1.0, 0);
        assert!((result as usize) < logits.len());
    }
}

#[test]
fn test_sample_token_from_logits_top_k_truncates_to_top_k() {
    let logits = vec![2.0_f32, 1.0, 0.5, 0.25, 0.125];
    for _ in 0..100 {
        let result = CandleLlmRuntime::sample_token_from_logits(&logits, 1.0, 1.0, 1);
        assert_eq!(result, 0, "top_k=1 must only return index 0, got {result}");
    }
}

#[test]
fn test_sample_token_from_logits_top_k_two_keeps_two() {
    let logits = vec![2.0_f32, 1.0, 0.5, 0.25];
    for _ in 0..100 {
        let result = CandleLlmRuntime::sample_token_from_logits(&logits, 1.0, 1.0, 2);
        assert!(
            result < 2,
            "top_k=2 must exclude index 2 and 3, got {result}"
        );
    }
}

#[test]
fn test_sample_token_from_logits_top_k_zero_is_no_truncation() {
    let logits = vec![2.0_f32, 1.0, 0.5, 0.25];
    for _ in 0..50 {
        let result = CandleLlmRuntime::sample_token_from_logits(&logits, 1.0, 1.0, 0);
        assert!((result as usize) < logits.len());
    }
}

#[test]
fn test_sample_token_from_logits_top_k_larger_than_vocab_is_no_truncation() {
    let logits = vec![1.0_f32, 2.0, 3.0];
    for _ in 0..20 {
        let result = CandleLlmRuntime::sample_token_from_logits(&logits, 1.0, 1.0, 100);
        assert!((result as usize) < logits.len());
    }
}

#[test]
fn test_sample_token_from_logits_top_k_composes_with_top_p() {
    let logits = vec![1.609_f32, 0.788, 0.223];
    for _ in 0..100 {
        let result = CandleLlmRuntime::sample_token_from_logits(&logits, 1.0, 0.5, 2);
        assert!(
            result < 2,
            "top_k=2 + top_p=0.5 must exclude 2, got {result}"
        );
    }
}

#[test]
fn test_check_stop_suffix_no_match_returns_none() {
    let stops: Vec<String> = vec!["END".to_string()];
    assert_eq!(
        CandleLlmRuntime::check_stop_suffix("hello world", &stops),
        None
    );
    assert_eq!(CandleLlmRuntime::check_stop_suffix("", &stops), None);
    assert_eq!(CandleLlmRuntime::check_stop_suffix("ENDing", &stops), None);
}

#[test]
fn test_check_stop_suffix_full_match_returns_zero() {
    let stops: Vec<String> = vec!["END".to_string()];
    assert_eq!(CandleLlmRuntime::check_stop_suffix("END", &stops), Some(0));
}

#[test]
fn test_check_stop_suffix_partial_match_returns_prefix_length() {
    let stops: Vec<String> = vec!["stop".to_string()];
    assert_eq!(
        CandleLlmRuntime::check_stop_suffix("Hello world stop", &stops),
        Some(12)
    );
}

#[test]
fn test_check_stop_suffix_first_match_wins() {
    let stops: Vec<String> = vec!["END".to_string(), "DONE".to_string()];
    assert_eq!(
        CandleLlmRuntime::check_stop_suffix("Hello DONE", &stops),
        Some(6)
    );
}

#[test]
fn test_check_stop_suffix_empty_stop_string_ignored() {
    let stops: Vec<String> = vec!["".to_string(), "END".to_string()];
    assert_eq!(
        CandleLlmRuntime::check_stop_suffix("the END", &stops),
        Some(4)
    );
}

#[test]
fn test_check_stop_suffix_empty_stops_returns_none() {
    let stops: Vec<String> = vec![];
    assert_eq!(
        CandleLlmRuntime::check_stop_suffix("anything", &stops),
        None
    );
}

#[test]
fn test_check_stop_suffix_multibyte_utf8() {
    let stops: Vec<String> = vec!["\u{3002}".to_string()];
    assert_eq!(
        CandleLlmRuntime::check_stop_suffix("\u{4f60}\u{597d}\u{4e16}\u{754c}\u{3002}", &stops),
        Some("\u{4f60}\u{597d}\u{4e16}\u{754c}".len())
    );
}

#[test]
#[serial_test::serial]
fn test_candle_load_without_env_returns_useful_error() {
    unsafe { std::env::remove_var("NEURON_LLM_MODEL_PATH") };
    // Isolate from the developer's real model library: point HOME and
    // USERPROFILE at an empty temp dir so the model-library fallback in
    // resolve_model_path finds nothing for the requested id.
    let temp = std::env::temp_dir().join(format!("neuron-load-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp);
    std::fs::create_dir_all(&temp).unwrap();
    unsafe {
        std::env::set_var("HOME", &temp);
        std::env::set_var("USERPROFILE", &temp);
    }
    let resolved = CandleLlmRuntime::resolve_model_path("qwen3-0.6b");
    let _ = std::fs::remove_dir_all(&temp);
    assert!(resolved.is_err());
    let err_msg = resolved.err().unwrap();
    assert!(
        err_msg.contains("not found under") || err_msg.contains("NEURON_LLM_MODEL_PATH"),
        "got: {err_msg}"
    );
}

#[test]
#[serial_test::serial]
fn test_candle_load_resolves_model_from_library_by_id() {
    unsafe { std::env::remove_var("NEURON_LLM_MODEL_PATH") };
    // Fake a pulled model in the library: <home>/.neuron/models/llm/<id>/
    // with the three required artefacts.
    let temp = std::env::temp_dir().join(format!("neuron-lib-test-{}", std::process::id()));
    let model_dir = temp
        .join(".neuron")
        .join("models")
        .join("llm")
        .join("qwen3-0.6b");
    std::fs::create_dir_all(&model_dir).unwrap();
    std::fs::write(model_dir.join("config.json"), b"{}").unwrap();
    std::fs::write(model_dir.join("tokenizer.json"), b"{}").unwrap();
    std::fs::write(model_dir.join("model.safetensors"), b"stub").unwrap();
    unsafe {
        std::env::set_var("HOME", &temp);
        std::env::set_var("USERPROFILE", &temp);
    }
    let resolved = CandleLlmRuntime::resolve_model_path("qwen3-0.6b");
    let _ = std::fs::remove_dir_all(&temp);
    let path = resolved.unwrap_or_else(|e| panic!("resolve failed: {e}"));
    let last = path
        .components()
        .next_back()
        .map(|c| c.as_os_str().to_string_lossy().to_string())
        .unwrap_or_default();
    assert_eq!(last, "qwen3-0.6b");
}

#[tokio::test]
async fn test_candle_chat_without_load_returns_not_initialized() {
    let runtime = CandleLlmRuntime::new();
    let req = ChatRequest {
        model: "qwen3-0.6b".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
            reasoning_content: None,
        }],
        temperature: None,
        max_tokens: None,
        top_p: None,
        top_k: None,
        stream: false,
        stop: None,
    };

    let result = runtime.chat(req).await;
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert_eq!(err.error_type, "not_initialized");
    assert!(err.message.contains("not loaded"), "got: {}", err.message);
}

#[tokio::test]
async fn test_candle_chat_stream_without_load_returns_not_initialized() {
    let runtime = CandleLlmRuntime::new();
    let req = ChatRequest {
        model: "qwen3-0.6b".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
            reasoning_content: None,
        }],
        temperature: None,
        max_tokens: None,
        top_p: None,
        top_k: None,
        stream: true,
        stop: None,
    };

    let result = runtime.chat_stream(req).await;
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert_eq!(err.error_type, "not_initialized");
    assert!(err.message.contains("not loaded"), "got: {}", err.message);
}

#[test]
fn test_chatml_prompt_format_system_user_assistant() {
    let messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: "You are helpful".to_string(),
            reasoning_content: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "Hi".to_string(),
            reasoning_content: None,
        },
    ];

    let prompt = CandleLlmRuntime::render_chatml_prompt(&messages);
    assert!(prompt.contains("<|im_start|>system"));
    assert!(prompt.contains("You are helpful"));
    assert!(prompt.contains("<|im_start|>user"));
    assert!(prompt.contains("<|im_start|>assistant"));
    assert!(prompt.ends_with("<|im_start|>assistant\n"));
}

#[test]
fn test_chatml_prompt_single_user() {
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "hello".to_string(),
        reasoning_content: None,
    }];
    let prompt = CandleLlmRuntime::render_chatml_prompt(&messages);
    assert_eq!(
        prompt,
        "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
    );
}

#[test]
fn test_chatml_prompt_empty_messages() {
    let messages: Vec<ChatMessage> = vec![];
    let prompt = CandleLlmRuntime::render_chatml_prompt(&messages);
    assert_eq!(prompt, "<|im_start|>assistant\n");
}

#[test]
fn test_detect_arch_qwen2_from_config() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = r#"{"model_type": "qwen2", "hidden_size": 896}"#;
    fs::write(tmp.path().join("config.json"), cfg).unwrap();

    let model_type = CandleLlmRuntime::detect_model_type(tmp.path()).unwrap();
    assert_eq!(model_type, "qwen2");
}

#[test]
fn test_detect_arch_qwen2_5_from_config() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = r#"{"model_type": "qwen2_5", "hidden_size": 1536}"#;
    fs::write(tmp.path().join("config.json"), cfg).unwrap();

    let model_type = CandleLlmRuntime::detect_model_type(tmp.path()).unwrap();
    assert_eq!(model_type, "qwen2_5");
}

#[test]
fn test_detect_arch_qwen3_from_config() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = r#"{"model_type": "qwen3", "hidden_size": 1024}"#;
    fs::write(tmp.path().join("config.json"), cfg).unwrap();

    let model_type = CandleLlmRuntime::detect_model_type(tmp.path()).unwrap();
    assert_eq!(model_type, "qwen3");
}

/// Verbatim shape of the upstream `Qwen/Qwen3.5-4B` `config.json`
/// (2026-10): the text tower is nested under `text_config` and mixes
/// linear-attention and full-attention layers, so the flat
/// `qwen3::Config` the candle loader uses cannot parse it.
const REAL_QWEN3_5_CONFIG: &str = r#"{
  "architectures": ["Qwen3_5ForConditionalGeneration"],
  "image_token_id": 248056,
  "model_type": "qwen3_5",
  "text_config": {
    "attention_bias": false,
    "attn_output_gate": true,
    "dtype": "bfloat16",
    "eos_token_id": 248044,
    "full_attention_interval": 4,
    "head_dim": 256,
    "hidden_act": "silu",
    "hidden_size": 2560,
    "intermediate_size": 9216,
    "layer_types": [
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention"
    ],
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_value_head_dim": 128,
    "max_position_embeddings": 262144,
    "mlp_only_layers": [],
    "model_type": "qwen3_5_text",
    "mtp_num_hidden_layers": 1,
    "num_attention_heads": 16,
    "num_hidden_layers": 32,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-06,
    "tie_word_embeddings": true,
    "vocab_size": 248320,
    "mamba_ssm_dtype": "float32",
    "rope_parameters": {
      "mrope_interleaved": true,
      "mrope_section": [11, 11, 10],
      "rope_type": "default",
      "rope_theta": 10000000,
      "partial_rotary_factor": 0.25
    }
  },
  "tie_word_embeddings": true,
  "video_token_id": 248057,
  "vision_config": {
    "depth": 24,
    "hidden_act": "gelu_pytorch_tanh",
    "hidden_size": 1024,
    "in_channels": 3,
    "intermediate_size": 4096,
    "model_type": "qwen3_5",
    "num_heads": 16,
    "num_position_embeddings": 2304,
    "out_hidden_size": 2560,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2
  }
}"#;

/// The `model_type` string is recognized (so diagnostics can name it),
/// but the arch loader must FAIL on a real Qwen3.5 checkpoint instead of
/// silently dispatching to the dense Qwen3 loader. Regression guard for
/// the "string recognized ⇒ model loadable" assumption.
#[test]
fn test_qwen3_5_real_config_is_rejected_at_arch_load() {
    let tmp = tempfile::tempdir().unwrap();
    fs::write(tmp.path().join("config.json"), REAL_QWEN3_5_CONFIG).unwrap();

    let model_type = CandleLlmRuntime::detect_model_type(tmp.path()).unwrap();
    assert_eq!(model_type, "qwen3_5");

    let err = CandleLlmRuntime::detect_arch(tmp.path())
        .err()
        .expect("Qwen3.5 must not load on the candle qwen3 path");
    assert!(
        err.contains("Qwen 3.5") && err.contains("mistralrs"),
        "expected the explicit Qwen 3.5 rejection naming the working backends, got: {err}"
    );
}

#[test]
fn test_detect_arch_qwen3_5_moe_is_rejected_with_moe_error() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = r#"{"model_type": "qwen3_5_moe", "text_config": {"model_type": "qwen3_5_moe_text"}}"#;
    fs::write(tmp.path().join("config.json"), cfg).unwrap();

    let err = CandleLlmRuntime::detect_arch(tmp.path())
        .err()
        .expect("Qwen3.5-MoE must not load on the dense candle path");
    assert!(
        err.contains("MoE") && err.contains("qwen3_5_moe"),
        "expected the explicit MoE rejection, got: {err}"
    );
}

#[test]
fn test_detect_arch_llama_3_from_config() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = r#"{
        "model_type": "llama",
        "vocab_size": 128256,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 32,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "max_position_embeddings": 131072
    }"#;
    fs::write(tmp.path().join("config.json"), cfg).unwrap();

    let model_type = CandleLlmRuntime::detect_model_type(tmp.path()).unwrap();
    assert_eq!(model_type, "llama");

    let result = CandleLlmRuntime::detect_arch(tmp.path());
    assert!(result.is_ok(), "expected Ok, got: {result:?}");
}

#[test]
fn test_detect_arch_phi3_from_config() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = r#"{
        "model_type": "phi3",
        "vocab_size": 32064,
        "hidden_act": "silu",
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "num_hidden_layers": 32,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "max_position_embeddings": 4096
    }"#;
    fs::write(tmp.path().join("config.json"), cfg).unwrap();

    let result = CandleLlmRuntime::detect_arch(tmp.path());
    assert!(result.is_ok(), "expected Ok, got: {result:?}");
}

#[test]
fn test_detect_arch_mistral_from_config() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = r#"{
        "model_type": "mistral",
        "vocab_size": 32000,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 32,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "max_position_embeddings": 32768
    }"#;
    fs::write(tmp.path().join("config.json"), cfg).unwrap();

    let result = CandleLlmRuntime::detect_arch(tmp.path());
    assert!(result.is_ok(), "expected Ok, got: {result:?}");
}

#[test]
fn test_detect_arch_chatglm_from_config() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = r#"{
        "model_type": "chatglm",
        "num_layers": 28,
        "padded_vocab_size": 65024,
        "hidden_size": 4096,
        "ffn_hidden_size": 13696,
        "kv_channels": 128,
        "num_attention_heads": 32,
        "seq_length": 8192
    }"#;
    fs::write(tmp.path().join("config.json"), cfg).unwrap();

    let result = CandleLlmRuntime::detect_arch(tmp.path());
    assert!(result.is_ok(), "expected Ok, got: {result:?}");
}

#[test]
fn test_detect_arch_unknown_returns_error() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = r#"{"model_type": "gpt2"}"#;
    fs::write(tmp.path().join("config.json"), cfg).unwrap();

    let model_type = CandleLlmRuntime::detect_model_type(tmp.path()).unwrap();
    assert_eq!(model_type, "gpt2");

    let result = CandleLlmRuntime::detect_arch(tmp.path());
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert!(err.contains("unsupported model_type"), "got: {err}");
    assert!(err.contains("gpt2"), "got: {err}");
}

// -- MoE rejection ------------------------------------------------------
//
// MoE model types (qwen3moe / qwen2moe / mixtral / deepseek) share
// their model_type stem with a supported dense family but use a
// different weight layout. The dense forward pass produces garbage
// on MoE weights, so detect_arch must reject these with a clear
// error containing both the literal substring "MoE" and the
// rejected model_type — the assertions below pin both parts of
// the contract.

#[test]
fn test_detect_arch_rejects_qwen3moe_with_clear_error() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = r#"{"model_type": "qwen3moe"}"#;
    fs::write(tmp.path().join("config.json"), cfg).unwrap();

    let result = CandleLlmRuntime::detect_arch(tmp.path());
    assert!(
        result.is_err(),
        "qwen3moe must not silently fall through to dense loader"
    );
    let err = result.err().unwrap();
    assert!(err.contains("MoE"), "error should mention MoE: got {err}");
    assert!(
        err.contains("qwen3moe"),
        "error should name the rejected model_type: got {err}"
    );
}

#[test]
fn test_detect_arch_rejects_qwen2moe_with_clear_error() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = r#"{"model_type": "qwen2moe"}"#;
    fs::write(tmp.path().join("config.json"), cfg).unwrap();

    let result = CandleLlmRuntime::detect_arch(tmp.path());
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert!(err.contains("MoE"), "got: {err}");
    assert!(err.contains("qwen2moe"), "got: {err}");
}

#[test]
fn test_detect_arch_rejects_mixtral_with_clear_error() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = r#"{"model_type": "mixtral"}"#;
    fs::write(tmp.path().join("config.json"), cfg).unwrap();

    let result = CandleLlmRuntime::detect_arch(tmp.path());
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert!(err.contains("MoE"), "got: {err}");
    assert!(err.contains("mixtral"), "got: {err}");
}

#[test]
fn test_detect_arch_rejects_deepseek_with_clear_error() {
    // deepseek is a unified MoE arch — no dense variant exists.
    let tmp = tempfile::tempdir().unwrap();
    let cfg = r#"{"model_type": "deepseek_v3"}"#;
    fs::write(tmp.path().join("config.json"), cfg).unwrap();

    let result = CandleLlmRuntime::detect_arch(tmp.path());
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert!(err.contains("MoE"), "got: {err}");
    assert!(err.contains("deepseek_v3"), "got: {err}");
}

#[test]
fn test_detect_model_type_returns_moe_string_without_arch_failure() {
    // detect_model_type is intentionally permissive: it just reads
    // the model_type string and never tries to parse the full
    // Config. Even for MoE (which detect_arch rejects), it should
    // return the raw string. This is what diagnostic tooling
    // relies on.
    let tmp = tempfile::tempdir().unwrap();
    let cfg = r#"{"model_type": "qwen3moe"}"#;
    fs::write(tmp.path().join("config.json"), cfg).unwrap();

    let result = CandleLlmRuntime::detect_model_type(tmp.path()).unwrap();
    assert_eq!(result, "qwen3moe");
}

#[test]
fn test_detect_arch_missing_model_type_returns_error() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = r#"{"hidden_size": 1024}"#;
    fs::write(tmp.path().join("config.json"), cfg).unwrap();

    let result = CandleLlmRuntime::detect_model_type(tmp.path());
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert!(err.contains("model_type"), "got: {err}");
}

#[test]
fn test_detect_arch_invalid_json_returns_error() {
    let tmp = tempfile::tempdir().unwrap();
    fs::write(tmp.path().join("config.json"), "not valid json{").unwrap();

    let result = CandleLlmRuntime::detect_model_type(tmp.path());
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert!(err.contains("parse config.json"), "got: {err}");
}

#[test]
fn test_detect_arch_missing_config_returns_error() {
    let tmp = tempfile::tempdir().unwrap();
    let result = CandleLlmRuntime::detect_model_type(tmp.path());
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert!(err.contains("read config.json"), "got: {err}");
}

#[test]
fn test_model_type_str_mapping() {
    let runtime = CandleLlmRuntime::new();
    let models = runtime.list_models();
    let ids: Vec<&str> = models.iter().map(|m| m.id.as_str()).collect();
    assert!(ids.contains(&"qwen2.5-0.5b"));
    assert!(ids.contains(&"qwen2.5-1.5b"));
    assert!(ids.contains(&"qwen3-0.6b"));
}

// ---- parse_use_flash_attn env-var resolution ----
//
// `parse_use_flash_attn` is `pub(crate)` in `candle_runtime.rs`
// and is brought in by the file-level `use ...::*;` glob at the
// top of this file. Each test sets / unsets
// `NEURON_USE_FLASH_ATTN` directly; they all carry
// `#[serial_test::serial]` because env mutation is a
// shared-global side effect and we run with the default test
// thread count (>= 2).

#[test]
#[serial_test::serial]
fn test_parse_use_flash_attn_unset_is_not_set() {
    unsafe { std::env::remove_var("NEURON_USE_FLASH_ATTN") };
    assert_eq!(parse_use_flash_attn(), FlashAttnRequest::NotSet);
}

#[test]
#[serial_test::serial]
fn test_parse_use_flash_attn_truthy_values_request_flash_attn() {
    for v in ["1", "true", "TRUE", "yes", "on", "ON"] {
        unsafe { std::env::set_var("NEURON_USE_FLASH_ATTN", v) };
        assert_eq!(
            parse_use_flash_attn(),
            FlashAttnRequest::Enabled,
            "value `{v}` should request FlashAttention-2"
        );
    }
    unsafe { std::env::remove_var("NEURON_USE_FLASH_ATTN") };
}

#[test]
#[serial_test::serial]
fn test_parse_use_flash_attn_falsy_values_disable_flash_attn() {
    for v in ["0", "false", "FALSE", "no", "off", "OFF"] {
        unsafe { std::env::set_var("NEURON_USE_FLASH_ATTN", v) };
        assert_eq!(
            parse_use_flash_attn(),
            FlashAttnRequest::Disabled,
            "value `{v}` should turn FlashAttention-2 off"
        );
    }
    unsafe { std::env::remove_var("NEURON_USE_FLASH_ATTN") };
}

/// The unknown spelling is KEPT, not folded into "off": the operator
/// has to see the value they actually typed. (This test used to assert
/// the opposite — that an unknown value silently became `None` — which
/// is exactly the "I asked for FA and got sdpa, silently" failure the
/// 2026-11-24 knife killed.)
#[test]
#[serial_test::serial]
fn test_parse_use_flash_attn_unknown_values_are_kept_verbatim() {
    for v in ["", "2", "enabled", "onn"] {
        unsafe { std::env::set_var("NEURON_USE_FLASH_ATTN", v) };
        assert_eq!(
            parse_use_flash_attn(),
            FlashAttnRequest::Unknown(v.to_string()),
            "value `{v}` should be kept verbatim"
        );
    }
    unsafe { std::env::remove_var("NEURON_USE_FLASH_ATTN") };
}

#[test]
#[serial_test::serial]
fn test_flash_attn_env_var_affects_mistral_config() {
    unsafe { std::env::set_var("NEURON_USE_FLASH_ATTN", "1") };
    let use_flash_attn = matches!(parse_use_flash_attn(), FlashAttnRequest::Enabled);
    assert!(
        use_flash_attn,
        "NEURON_USE_FLASH_ATTN=1 should parse as true"
    );

    let mut cfg = candle_transformers::models::mistral::Config::config_7b_v0_1(false);
    assert!(!cfg.use_flash_attn, "default should be false");

    if use_flash_attn {
        cfg.use_flash_attn = true;
    }
    assert!(cfg.use_flash_attn, "after env var override, should be true");

    unsafe { std::env::remove_var("NEURON_USE_FLASH_ATTN") };
}

#[test]
#[serial_test::serial]
fn test_flash_attn_qwen2_no_support() {
    unsafe { std::env::set_var("NEURON_USE_FLASH_ATTN", "1") };
    let use_flash_attn = matches!(parse_use_flash_attn(), FlashAttnRequest::Enabled);
    assert!(use_flash_attn);

    let _cfg = candle_transformers::models::qwen2::Config {
        vocab_size: 151936,
        hidden_size: 896,
        intermediate_size: 4864,
        num_hidden_layers: 24,
        num_attention_heads: 14,
        num_key_value_heads: 2,
        max_position_embeddings: 32768,
        sliding_window: 32768,
        max_window_layers: 24,
        tie_word_embeddings: false,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-6,
        use_sliding_window: false,
        hidden_act: candle_nn::Activation::Silu,
    };

    unsafe { std::env::remove_var("NEURON_USE_FLASH_ATTN") };
}

#[test]
#[serial_test::serial]
fn test_flash_attn_llama_still_works() {
    unsafe { std::env::set_var("NEURON_USE_FLASH_ATTN", "1") };
    let use_flash_attn = matches!(parse_use_flash_attn(), FlashAttnRequest::Enabled);
    assert!(use_flash_attn);

    unsafe { std::env::remove_var("NEURON_USE_FLASH_ATTN") };
}
// ---- FlashAttention resolver + operator-facing messages ----
//
// The knob is not the truth: `NEURON_USE_FLASH_ATTN=1` is a REQUEST.
// What the candle leg actually does is decided by **four** prerequisites —
// the `flash-attn` build feature, an architecture that has an FA path,
// a CUDA device, and a **dtype the kernels accept** (f16/bf16 — *not*
// f32) — and every missing one has to be NAMED. These judgments pin the
// resolver's branch order and the messages' content, including the
// negative control that a degraded state must not blame a prerequisite
// that is actually present.
//
// The fourth one was found the hard way (real machine, 2026-11-25):
// with f32 weights the FA path was entered and then died inside the
// forward pass with `flash-attn is only supported for f16/bf16 (F32)` —
// after the weights were loaded and the VRAM was taken. A prerequisite
// that is only discovered inside the kernel has to be checked *before*
// the model is built.

/// Build a prerequisite set (feature, arch, device, dtype).
fn prereqs(feature: bool, arch: bool, device: bool, dtype: bool) -> FlashAttnPrereqs {
    FlashAttnPrereqs {
        feature_compiled: feature,
        arch_covered: arch,
        device_is_cuda: device,
        dtype_supported: dtype,
    }
}

#[test]
fn test_resolve_flash_attn_branch_order() {
    // A missing feature makes every other question moot.
    assert_eq!(
        resolve_flash_attn(
            &FlashAttnRequest::Enabled,
            prereqs(false, false, false, false)
        ),
        FlashAttnStatus::FeatureNotCompiled
    );
    // Feature present, architecture missing: the device is irrelevant.
    assert_eq!(
        resolve_flash_attn(
            &FlashAttnRequest::Enabled,
            prereqs(true, false, false, false)
        ),
        FlashAttnStatus::ArchNotCovered
    );
    // Feature + architecture present, device missing.
    assert_eq!(
        resolve_flash_attn(
            &FlashAttnRequest::Enabled,
            prereqs(true, true, false, false)
        ),
        FlashAttnStatus::DeviceNotCuda
    );
    // Everything but the dtype ⇒ the kernel would die inside the
    // forward pass ⇒ ⇒ the dtype has to be a *named* prerequisite.
    assert_eq!(
        resolve_flash_attn(&FlashAttnRequest::Enabled, prereqs(true, true, true, false)),
        FlashAttnStatus::DtypeNotSupported
    );
    // Everything present ⇒ really on.
    assert_eq!(
        resolve_flash_attn(&FlashAttnRequest::Enabled, prereqs(true, true, true, true)),
        FlashAttnStatus::Enabled
    );
}

#[test]
fn test_resolve_flash_attn_non_requests_never_degrade() {
    // Not asking for FA is not a degraded state — it is the sdpa path.
    assert_eq!(
        resolve_flash_attn(
            &FlashAttnRequest::NotSet,
            prereqs(false, false, false, false)
        ),
        FlashAttnStatus::NotRequested
    );
    assert_eq!(
        resolve_flash_attn(
            &FlashAttnRequest::Disabled,
            prereqs(false, false, false, false)
        ),
        FlashAttnStatus::DisabledByEnv
    );
    // An unknown spelling is its own state — it must never be reported
    // as "off", which is the silence this knife exists to kill.
    assert_eq!(
        resolve_flash_attn(
            &FlashAttnRequest::Unknown("onn".to_string()),
            prereqs(false, false, false, false),
        ),
        FlashAttnStatus::UnknownValue
    );
}

#[test]
fn test_flash_attn_effective_flag_is_only_true_when_enabled() {
    // The flag handed to the model builder must be the EFFECTIVE one.
    // If it were the requested one, a CPU run in a build that has the
    // feature would build a FlashAttention model and then die inside
    // the attention kernel.
    for status in [
        FlashAttnStatus::NotRequested,
        FlashAttnStatus::DisabledByEnv,
        FlashAttnStatus::UnknownValue,
        FlashAttnStatus::FeatureNotCompiled,
        FlashAttnStatus::ArchNotCovered,
        FlashAttnStatus::DeviceNotCuda,
        FlashAttnStatus::DtypeNotSupported,
    ] {
        assert!(
            !status.is_enabled(),
            "{status:?} must not enable FlashAttention"
        );
    }
    assert!(FlashAttnStatus::Enabled.is_enabled());
}

#[test]
fn test_flash_attn_degraded_messages_name_the_missing_prerequisite() {
    // Each degraded state must name what is missing — and ONLY what is
    // missing, so the operator is not sent to fix the wrong thing.
    let feature_msg = flash_attn_status_message(
        FlashAttnStatus::FeatureNotCompiled,
        &FlashAttnRequest::Enabled,
        FlashAttnArch::Qwen2,
    )
    .expect("feature-not-compiled must say something");
    assert!(feature_msg.contains("flash-attn"), "got: {feature_msg}");
    assert!(
        !feature_msg.contains("device is not CUDA"),
        "must not blame the device: {feature_msg}"
    );

    let arch_msg = flash_attn_status_message(
        FlashAttnStatus::ArchNotCovered,
        &FlashAttnRequest::Enabled,
        FlashAttnArch::Phi3,
    )
    .expect("arch-not-covered must say something");
    assert!(
        arch_msg.contains("phi3"),
        "must name the architecture: {arch_msg}"
    );
    assert!(
        !arch_msg.contains("device is not CUDA"),
        "must not blame the device: {arch_msg}"
    );

    let device_msg = flash_attn_status_message(
        FlashAttnStatus::DeviceNotCuda,
        &FlashAttnRequest::Enabled,
        FlashAttnArch::Qwen2,
    )
    .expect("device-not-cuda must say something");
    assert!(
        device_msg.contains("CUDA"),
        "must name the device: {device_msg}"
    );
    assert!(
        !device_msg.contains("build with"),
        "must not blame the build: {device_msg}"
    );
}

#[test]
fn test_flash_attn_unknown_value_message_quotes_the_raw_value() {
    let msg = flash_attn_status_message(
        FlashAttnStatus::UnknownValue,
        &FlashAttnRequest::Unknown("onn".to_string()),
        FlashAttnArch::Qwen2,
    )
    .expect("an unknown spelling must be reported");
    assert!(
        msg.contains("onn"),
        "must quote what the operator typed: {msg}"
    );
    assert!(
        msg.contains("NEURON_USE_FLASH_ATTN"),
        "must name the knob: {msg}"
    );
}

#[test]
fn test_flash_attn_silent_states_produce_no_message() {
    // Never asked for / explicitly off / genuinely on: no noise. A knob
    // that prints on every startup trains operators to ignore it.
    assert!(
        flash_attn_status_message(
            FlashAttnStatus::NotRequested,
            &FlashAttnRequest::NotSet,
            FlashAttnArch::Qwen2
        )
        .is_none()
    );
    assert!(
        flash_attn_status_message(
            FlashAttnStatus::DisabledByEnv,
            &FlashAttnRequest::Disabled,
            FlashAttnArch::Qwen2
        )
        .is_none()
    );
    assert!(
        flash_attn_status_message(
            FlashAttnStatus::Enabled,
            &FlashAttnRequest::Enabled,
            FlashAttnArch::Qwen2
        )
        .is_none()
    );
}

#[test]
fn test_flash_attn_arch_coverage_matches_the_implementations() {
    // The four covered architectures have a real FA path: qwen2/qwen3
    // via src/llm/vendor/, llama via vendor/llama_hd.rs, mistral via
    // candle-transformers' own `use_flash_attn` field.
    assert!(FlashAttnArch::Qwen2.supports_flash_attn());
    assert!(FlashAttnArch::Qwen3.supports_flash_attn());
    assert!(FlashAttnArch::Llama.supports_flash_attn());
    assert!(FlashAttnArch::Mistral.supports_flash_attn());
    // Phi3 and ChatGlm have NO such field — the old code printed
    // "use_flash_attn=true" for them while never passing the flag
    // anywhere.
    assert!(!FlashAttnArch::Phi3.supports_flash_attn());
    assert!(!FlashAttnArch::ChatGlm.supports_flash_attn());
}

#[test]
fn test_flash_attn_needs_a_dtype_the_kernel_accepts() {
    // ⚠ **真机撞出来的那一条**（2026-11-25）：FA 内核只认 f16/bf16 ⇒ ⇒
    // 默认的 f32 进不去 ⇒ ⇒ 旧口径会把 f32 的权重**照送进内核**，然后
    // 在 forward pass 里炸（那时模型已加载、显存已占）⇒ ⇒ 现在它是
    // **前提**，不是意外。
    assert_eq!(
        resolve_flash_attn(&FlashAttnRequest::Enabled, prereqs(true, true, true, false)),
        FlashAttnStatus::DtypeNotSupported
    );
    assert!(!FlashAttnStatus::DtypeNotSupported.is_enabled());
    // The dtype is the *last* prerequisite checked — an earlier missing
    // one is the cheaper thing to explain.
    assert_eq!(
        resolve_flash_attn(
            &FlashAttnRequest::Enabled,
            prereqs(true, true, false, false)
        ),
        FlashAttnStatus::DeviceNotCuda
    );
}

#[test]
fn test_dtype_complaint_names_the_knob() {
    // 「降级必须响亮**且点名缺的那一个前提**」⇒ ⇒ 抱怨里必须出现
    // `NEURON_LLM_DTYPE=bf16` ⇒ ⇒ 不许只说「FA 用不了」。
    let msg = flash_attn_status_message(
        FlashAttnStatus::DtypeNotSupported,
        &FlashAttnRequest::Enabled,
        FlashAttnArch::Qwen2,
    )
    .expect("a requested-but-not-enabled FA must produce a line");
    assert!(msg.contains("NEURON_LLM_DTYPE=bf16"), "got: {msg}");
    assert!(msg.contains("f16/bf16"), "got: {msg}");
    assert!(msg.contains("NOT in use"), "got: {msg}");
}

#[test]
#[serial_test::serial]
fn test_flash_attn_env_to_message_chain() {
    // The full chain the operator experiences: env value → request →
    // status → message. A build without the `flash-attn` feature has
    // to say so out loud when FA was requested.
    unsafe { std::env::set_var("NEURON_USE_FLASH_ATTN", "1") };
    let request = parse_use_flash_attn();
    assert_eq!(request, FlashAttnRequest::Enabled);
    let status = resolve_flash_attn(
        &request,
        prereqs(
            cfg!(feature = "flash-attn"),
            FlashAttnArch::Qwen2.supports_flash_attn(),
            false,
            true,
        ),
    );
    assert!(!status.is_enabled());
    let msg = flash_attn_status_message(status, &request, FlashAttnArch::Qwen2)
        .expect("a requested-but-not-enabled FA must produce a line");
    if cfg!(feature = "flash-attn") {
        // With the feature compiled, the missing prerequisite is the
        // device (this judgment runs on the CPU).
        assert_eq!(status, FlashAttnStatus::DeviceNotCuda);
        assert!(msg.contains("device is not CUDA"), "got: {msg}");
    } else {
        assert_eq!(status, FlashAttnStatus::FeatureNotCompiled);
        assert!(msg.contains("flash-attn"), "got: {msg}");
    }
    unsafe { std::env::remove_var("NEURON_USE_FLASH_ATTN") };
}

// ---- 刀 12：条目带来的 dtype 真的压过全局 env ----
//
// 「自动降级」要能救一个模型，光在**策略层**把 dtype 算对是不够的 ——
// 决策必须**到达执行者**：物化权重时用的必须是条目里那一档。
// 以前 dtype 只有一个全局 env 旋钮 ⇒ ⇒ 策略层算出的 bf16 到运行时会被
// env 的 f32 盖掉 ⇒ ⇒ 「自动降级」是**假动作**（账本说 bf16、真实占用
// 是 f32 的两倍，而账本正是用来算「装不装得下」的那份）。
#[test]
#[serial_test::serial]
fn test_entry_dtype_overrides_global_env() {
    unsafe { std::env::set_var("NEURON_LLM_DTYPE", "f32") };
    assert_eq!(
        resolve_weight_dtype(Some(crate::accel::DTypeChoice::BF16)),
        DType::BF16,
        "the entry's dtype must win over NEURON_LLM_DTYPE"
    );
    assert_eq!(
        resolve_weight_dtype(Some(crate::accel::DTypeChoice::F32)),
        DType::F32,
        "an explicit f32 entry must not be silently promoted to bf16"
    );
    assert_eq!(
        resolve_weight_dtype(Some(crate::accel::DTypeChoice::F16)),
        DType::F16
    );
    assert_eq!(resolve_weight_dtype(None), DType::F32);
    unsafe { std::env::set_var("NEURON_LLM_DTYPE", "bf16") };
    assert_eq!(
        resolve_weight_dtype(None),
        DType::BF16,
        "without an entry the global knob still applies"
    );
    unsafe { std::env::remove_var("NEURON_LLM_DTYPE") };
}

// ===========================================================================
// 刀 16 续（2026-11-26）：**模型库落点的两个真相源**
//
// 真机逐字（产品形态：App 内起 neuron，端脑 HTTP 面真打一次）：
//   engine_execution_error: llm load failed for 'qwen3-0.6b': neuron_error:
//   CandleLlmRuntime load failed: not_implemented: CandleLlmRuntime:
//   model 'qwen3-0.6b' not found under ./.neuron/models
// ⇒ ⇒ 引擎**注册上了**（注册面走 `config.model_dirs` ＝ 宿主给的
// `<data_dir>/models`），加载面却走 `resolved_model_dir()`（Android
// 没有 `$HOME` ⇒ ⇒ 退化成 `./.neuron/models`）⇒ ⇒ **两个真相源**
// ⇒ ⇒ 「注册上了、找不到权重」⇒ ⇒ 失败长得像「模型没下载」⇒ ⇒ 而盘上**明明有**。
// ===========================================================================

/// **纯函数**：LLM 权重的布局是 `<root>/llm/<id>`（模型库）或
/// `<root>/<id>`（直接指路）⇒ ⇒ 判据住在纯函数里 ⇒ ⇒ 零环境依赖。
#[test]
fn pure_layout_of_llm_model_dir() {
    let root = std::env::temp_dir().join(format!("neuron-llm-layout-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let lib = root.join("llm").join("qwen3-0.6b");
    std::fs::create_dir_all(&lib).unwrap();
    assert_eq!(
        CandleLlmRuntime::find_llm_model_dir(&[root.clone()], "qwen3-0.6b"),
        Some(lib),
        "the library layout is <root>/llm/<id>"
    );
    let direct = root.join("qwen2.5-0.5b");
    std::fs::create_dir_all(&direct).unwrap();
    assert_eq!(
        CandleLlmRuntime::find_llm_model_dir(&[root.clone()], "qwen2.5-0.5b"),
        Some(direct),
        "the direct layout is <root>/<id>"
    );
    assert_eq!(
        CandleLlmRuntime::find_llm_model_dir(&[root.clone()], "not-there"),
        None,
        "a miss must not be guessed"
    );
    let _ = std::fs::remove_dir_all(&root);
}

/// **注册面与加载面必须找到【同一个目录】** —— 第五个洞的**正面**判据。
#[test]
fn the_registration_face_and_the_loading_face_find_the_same_dir() {
    let root = std::env::temp_dir().join(format!("neuron-llm-same-dir-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let model_dir = root.join("llm").join("qwen3-0.6b");
    std::fs::create_dir_all(&model_dir).unwrap();
    std::fs::write(model_dir.join("config.json"), b"{}").unwrap();
    std::fs::write(model_dir.join("tokenizer.json"), b"{}").unwrap();
    std::fs::write(model_dir.join("model.safetensors"), b"stub").unwrap();
    // 注册面（`LlmRuntimeRegistry::try_register_downloaded`）的判据：
    // `<dir>/llm/<id>/model.safetensors` 存在。
    assert!(
        model_dir.join("model.safetensors").exists(),
        "the registration face must see the weights"
    );
    // 加载面用的是**同一份根** ⇒ ⇒ 必须落在**同一个目录**上。
    let loading_face = CandleLlmRuntime::resolve_model_path_in("qwen3-0.6b", &[root.clone()])
        .unwrap_or_else(|e| panic!("the loading face must find it: {e}"));
    assert_eq!(
        loading_face, model_dir,
        "the loading face must land on the very directory the registration face found"
    );
    let _ = std::fs::remove_dir_all(&root);
}

/// **嵌脑那一格**：`NEURON_MODELS_DIR` 也是 LLM 那条腿的根 ⇒ ⇒
/// 引擎注册上了就**找得到权重**（此前只认 `$HOME/.neuron/models` ⇒ ⇒
/// 手机上恒 `./.neuron/models`）。
///
/// ⚠️ **不碰进程环境**（注入取值函数）⇒ ⇒ 不与别的判据抢环境变量
/// （共享工作树上并行测试互抢 env 是**已量过**的事）。
#[test]
fn the_embedded_slot_is_a_root_for_the_llm_leg() {
    let root = std::env::temp_dir().join(format!("neuron-llm-embedded-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let model_dir = root.join("llm").join("qwen3-0.6b");
    std::fs::create_dir_all(&model_dir).unwrap();
    std::fs::write(model_dir.join("config.json"), b"{}").unwrap();
    std::fs::write(model_dir.join("tokenizer.json"), b"{}").unwrap();
    std::fs::write(model_dir.join("model.safetensors"), b"stub").unwrap();
    // 端脑启动时把宿主给的 <data_dir>/models 指到 NEURON_MODELS_DIR ⇒ ⇒
    // 那一格必须是根 ⇒ ⇒ 加载面就找得到。
    let roots = crate::config::NeuronConfig::model_roots_from(
        |var| {
            if var == "NEURON_MODELS_DIR" {
                Some(root.display().to_string())
            } else {
                None
            }
        },
        std::env::temp_dir().join("neuron-llm-embedded-default"),
    );
    let path = CandleLlmRuntime::resolve_model_path_in("qwen3-0.6b", &roots)
        .unwrap_or_else(|e| panic!("the embedded slot must be a root: {e}"));
    assert_eq!(path, model_dir);
    let _ = std::fs::remove_dir_all(&root);
}

/// **找不到 ⇒ ⇒ 错误里说清试过哪些目录**（不许只说一句「not found」）。
#[test]
fn the_miss_error_names_the_tried_dirs() {
    let root = std::env::temp_dir().join(format!("neuron-llm-miss-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let err = CandleLlmRuntime::resolve_model_path_in("llm-miss-probe-model", &[root.clone()])
        .unwrap_err();
    assert!(
        err.contains("llm-miss-probe-model"),
        "err must name the model: {err}"
    );
    assert!(
        err.contains(&root.display().to_string()),
        "err must name the tried dir: {err}"
    );
    let _ = std::fs::remove_dir_all(&root);
}
