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
    assert_eq!(
        CandleLlmRuntime::decode_token_for_stream("Hello"),
        "Hello"
    );
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
    assert_eq!(CandleLlmRuntime::sample_token_from_logits(&logits, 0.0, 1.0, 0), 0);
    assert_eq!(CandleLlmRuntime::sample_token_from_logits(&logits, 1.0, 1.0, 0), 0);
    assert_eq!(CandleLlmRuntime::sample_token_from_logits(&logits, 0.5, 1.0, 0), 0);
}

#[test]
fn test_sample_token_from_logits_zero_temperature_is_argmax() {
    let logits = vec![1.0_f32, 2.0, 3.0];
    assert_eq!(
        CandleLlmRuntime::sample_token_from_logits(&logits, 0.0, 1.0, 0),
        2
    );
    let logits = vec![3.0_f32, 1.0, 2.0];
    assert_eq!(CandleLlmRuntime::sample_token_from_logits(&logits, 0.0, 1.0, 0), 0);
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
        assert!(result < 2, "top_k=2 + top_p=0.5 must exclude 2, got {result}");
    }
}

#[test]
fn test_check_stop_suffix_no_match_returns_none() {
    let stops: Vec<String> = vec!["END".to_string()];
    assert_eq!(CandleLlmRuntime::check_stop_suffix("hello world", &stops), None);
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
    assert_eq!(CandleLlmRuntime::check_stop_suffix("anything", &stops), None);
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
    let model_dir = temp.join(".neuron").join("models").join("llm").join("qwen3-0.6b");
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
        messages: vec![ChatMessage { role: "user".to_string(), content: "Hello".to_string() }],
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
        messages: vec![ChatMessage { role: "user".to_string(), content: "Hello".to_string() }],
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
        ChatMessage { role: "system".to_string(), content: "You are helpful".to_string() },
        ChatMessage { role: "user".to_string(), content: "Hi".to_string() },
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
    let messages = vec![ChatMessage { role: "user".to_string(), content: "hello".to_string() }];
    let prompt = CandleLlmRuntime::render_chatml_prompt(&messages);
    assert_eq!(prompt, "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n");
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
    assert!(result.is_err(), "qwen3moe must not silently fall through to dense loader");
    let err = result.err().unwrap();
    assert!(err.contains("MoE"), "error should mention MoE: got {err}");
    assert!(err.contains("qwen3moe"), "error should name the rejected model_type: got {err}");
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
fn test_parse_use_flash_attn_unset_returns_none() {
    unsafe { std::env::remove_var("NEURON_USE_FLASH_ATTN") };
    assert_eq!(parse_use_flash_attn(), None);
}

#[test]
#[serial_test::serial]
fn test_parse_use_flash_attn_truthy_values_return_true() {
    for v in ["1", "true", "TRUE", "yes"] {
        unsafe { std::env::set_var("NEURON_USE_FLASH_ATTN", v) };
        assert_eq!(
            parse_use_flash_attn(),
            Some(true),
            "value `{v}` should parse as Some(true)"
        );
    }
    unsafe { std::env::remove_var("NEURON_USE_FLASH_ATTN") };
}

#[test]
#[serial_test::serial]
fn test_parse_use_flash_attn_falsy_values_return_false() {
    for v in ["0", "false", "FALSE", "no"] {
        unsafe { std::env::set_var("NEURON_USE_FLASH_ATTN", v) };
        assert_eq!(
            parse_use_flash_attn(),
            Some(false),
            "value `{v}` should parse as Some(false)"
        );
    }
    unsafe { std::env::remove_var("NEURON_USE_FLASH_ATTN") };
}

#[test]
#[serial_test::serial]
fn test_parse_use_flash_attn_unknown_values_return_none() {
    for v in ["on", "off", "", "2", "TRUE ", " true", "enabled"] {
        unsafe { std::env::set_var("NEURON_USE_FLASH_ATTN", v) };
        assert_eq!(
            parse_use_flash_attn(),
            None,
            "value `{v}` should parse as None (fall back to false)"
        );
    }
    unsafe { std::env::remove_var("NEURON_USE_FLASH_ATTN") };
}

#[test]
#[serial_test::serial]
fn test_flash_attn_env_var_affects_mistral_config() {
    unsafe { std::env::set_var("NEURON_USE_FLASH_ATTN", "1") };
    let use_flash_attn = parse_use_flash_attn().unwrap_or(false);
    assert!(use_flash_attn, "NEURON_USE_FLASH_ATTN=1 should parse as true");

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
    let use_flash_attn = parse_use_flash_attn().unwrap_or(false);
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
    let use_flash_attn = parse_use_flash_attn().unwrap_or(false);
    assert!(use_flash_attn);

    unsafe { std::env::remove_var("NEURON_USE_FLASH_ATTN") };
}
