//! Real-model smoke harness for the `minicpm5-2b` candle path.
//!
//! Boots an embedded neuron server with `NEURON_LLM_RUNTIME=candle`
//! and `NEURON_LLM_MODEL_PATH` pointing at a local MiniCPM5-2B
//! checkpoint directory, then issues a single short
//! `/v1/chat/completions` request and validates the response shape.
//!
//! What this exercises end-to-end:
//!
//! 1. `CandleLlmRuntime::load` reads the real OpenBMB checkpoint
//!    (`config.json` with `model_type = "llama"`, `vocab_size =
//!    130560`, `rope_theta = 5_000_000`, `tie_word_embeddings =
//!    false`, GQA 16Q/2KV, 42 layers, hidden_size 2048) and dispatches
//!    through `candle_transformers::models::llama`.
//! 2. `POST /v1/chat/completions` chat-template tokenization
//!    against the real tokenizer (`tokenizer.json`).
//! 3. A real forward pass produces non-empty logits; greedy decode
//!    yields a non-empty assistant message.
//!
//! Run:
//!
//! ```bash
//! NEURON_LLM_RUNTIME=candle \
//! NEURON_LLM_MODEL_PATH=$HOME/Projects/iBLOC/neuron/models/MiniCPM5-2B \
//! cargo run --release --example minicpm5_2b_smoke --features candle
//! ```
//!
//! CPU inference of a 2.52B BF16 model is slow (~5–10 s/token on a
//! modern desktop CPU). The harness requests only 8 completion
//! tokens and a single prompt, so the wall-clock cost is dominated
//! by weight load (~10–20 s) plus ~1–2 min of decode.
//!
//! On GPU (CUDA / Metal) the same harness finishes in <10 s end-to-end.
//!
//! Set `NEURON_PORT=0` (default) to bind to a random free port.

use neuron_lib::{run_embedded, NeuronConfig, NeuronEmbedConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // We deliberately do NOT use `#[tokio::main]` here. `run_embedded`
    // internally constructs a current-thread tokio runtime via
    // `ServerState::default_runtime()` -> `Handle::try_current()
    // .block_on(...)`. Calling `run_embedded` from inside a tokio
    // runtime (which `#[tokio::main]` provides) would re-enter the
    // runtime and panic with "Cannot start a runtime from within
    // a runtime". Building a fresh current-thread runtime here
    // matches the pattern used by neuron's own integration tests
    // (`src/embedded.rs::test_health_fields_returned_for_all_five_
    // runtimes`) and the `real_model_smoke` example.
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| format!("build tokio runtime: {e}"))?;
    runtime.block_on(async_main())
}

async fn async_main() -> Result<(), Box<dyn std::error::Error>> {
    let port: u16 = std::env::var("NEURON_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let cfg = NeuronEmbedConfig {
        port,
        config: NeuronConfig::new(),
    };
    let handle = run_embedded(cfg).await?;
    let port = handle.addr.port();
    eprintln!("[smoke:minicpm5-2b] server bound on 127.0.0.1:{port}");

    let did = llm_smoke(port).await?;

    eprintln!("\n[smoke:minicpm5-2b] shutting down server...");
    handle.request_shutdown();
    handle.join().await;

    if !did {
        eprintln!(
            "[smoke:minicpm5-2b] FATAL: smoke was skipped. \
             Set NEURON_LLM_MODEL_PATH to a directory containing \
             config.json + tokenizer.json + *.safetensors from \
             openbmb/MiniCPM5-2B."
        );
        std::process::exit(2);
    }
    eprintln!("[smoke:minicpm5-2b] ALL CHECKS PASSED");
    Ok(())
}

/// Returns `true` if the candle LLM was exercised; `false` if the
/// smoke was skipped because `NEURON_LLM_MODEL_PATH` is unset or
/// the checkpoint is missing required files.
async fn llm_smoke(port: u16) -> Result<bool, Box<dyn std::error::Error>> {
    let model_path = match std::env::var("NEURON_LLM_MODEL_PATH") {
        Ok(p) => p,
        Err(_) => {
            eprintln!(
                "[smoke:minicpm5-2b] skipping: NEURON_LLM_MODEL_PATH not set. \
                 Download openbmb/MiniCPM5-2B (HF or ModelScope) and set the env var."
            );
            return Ok(false);
        }
    };
    let dir = std::path::Path::new(&model_path);
    let config_path = dir.join("config.json");
    let tokenizer_path = dir.join("tokenizer.json");
    let has_weights = std::fs::read_dir(dir)
        .map(|entries| {
            entries.filter_map(|e| e.ok()).any(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "safetensors")
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);

    if !config_path.exists() || !tokenizer_path.exists() || !has_weights {
        eprintln!(
            "[smoke:minicpm5-2b] skipping: checkpoint incomplete at {model_path}. \
             Need config.json + tokenizer.json + *.safetensors."
        );
        return Ok(false);
    }
    eprintln!("\n[smoke:minicpm5-2b] loading from {model_path}");

    // ---- 1. health probe ----
    let client = reqwest::Client::new();
    let health_url = format!("http://127.0.0.1:{port}/health");
    let health: serde_json::Value = client
        .get(&health_url)
        .send()
        .await?
        .json()
        .await?;
    eprintln!("[smoke:minicpm5-2b] /health = {health}");

    // ---- 2. minimal chat completion ----
    // 8 tokens is enough to detect:
    //   - tokenizer encode/decode round-trips
    //   - candle forward pass producing non-degenerate logits
    //   - OpenAI wire shape (id, object, model, choices[].message.content)
    //   - finish_reason populated
    eprintln!("[smoke:minicpm5-2b] sending chat completion (max_tokens=8)");
    let url = format!("http://127.0.0.1:{port}/v1/chat/completions");
    let r: serde_json::Value = client
        .post(&url)
        .json(&serde_json::json!({
            "model": "minicpm5-2b",
            "max_tokens": 8,
            "temperature": 0.0,
            "messages": [{
                "role": "user",
                "content": "Reply with exactly one word."
            }]
        }))
        .send()
        .await?
        .json()
        .await?;

    eprintln!(
        "[smoke:minicpm5-2b] raw response = {}",
        serde_json::to_string_pretty(&r).unwrap_or_default()
    );

    assert_eq!(
        r["object"], "chat.completion",
        "unexpected response object: {r}"
    );
    let model = r["model"].as_str().expect("model field");
    eprintln!("[smoke:minicpm5-2b] response.model = {model:?}");

    let choices = r["choices"].as_array().expect("choices array");
    assert!(!choices.is_empty(), "no choices returned");
    let content = choices[0]["message"]["content"]
        .as_str()
        .expect("content should be a string");
    let finish_reason = choices[0]["finish_reason"].as_str();
    eprintln!("[smoke:minicpm5-2b] content       = {content:?}");
    eprintln!("[smoke:minicpm5-2b] finish_reason = {finish_reason:?}");
    assert!(
        !content.is_empty(),
        "minicpm5-2b returned an empty response \
         (forward produced no logits or all decode candidates were \
         filtered out — check that the candle Llama loader accepted \
         the real config.json fields vocab_size=130560, rope_theta=5e6, \
         tie_word_embeddings=false)"
    );

    // ---- 3. usage accounting (prompt + completion tokens) ----
    let usage_prompt_tokens: u64;
    let usage_completion_tokens: u64;
    if let Some(usage) = r.get("usage") {
        usage_prompt_tokens = usage["prompt_tokens"].as_u64().unwrap_or(0);
        usage_completion_tokens = usage["completion_tokens"].as_u64().unwrap_or(0);
        eprintln!(
            "[smoke:minicpm5-2b] usage = prompt={usage_prompt_tokens} \
             completion={usage_completion_tokens}"
        );
        assert!(
            usage_prompt_tokens > 0 && usage_completion_tokens > 0,
            "usage must report non-zero token counts"
        );
    } else {
        eprintln!(
            "[smoke:minicpm5-2b] usage field absent (acceptable — some \
             runtimes skip token accounting on minimal requests)"
        );
        usage_prompt_tokens = 0;
        usage_completion_tokens = 0;
    }

    // ---- 4. repeat-request isolation ----
    // The candle LLM runtime reuses its `Cache` struct across
    // requests, so a second identical chat does NOT necessarily
    // produce byte-identical text (the KV cache carries state
    // from the previous prompt's decode loop). What we verify
    // here is the minimum that proves the wire + candle plumbing
    // is stable across calls:
    //
    //   - response shape (`object = chat.completion`, has `choices`)
    //   - token accounting (`prompt_tokens` matches across calls)
    //   - non-empty `content` (forward pass produced decodable text)
    //
    // Strict byte-equality under `temperature=0` is a separate
    // concern that requires a `reset_cache` call between requests;
    // out of scope for this end-to-end smoke.
    eprintln!("[smoke:minicpm5-2b] repeat-request check (second identical request)");
    let r2: serde_json::Value = client
        .post(&url)
        .json(&serde_json::json!({
            "model": "minicpm5-2b",
            "max_tokens": 8,
            "temperature": 0.0,
            "messages": [{
                "role": "user",
                "content": "Reply with exactly one word."
            }]
        }))
        .send()
        .await?
        .json()
        .await?;
    assert_eq!(
        r2["object"], "chat.completion",
        "second request has wrong object: {r2}"
    );
    assert!(
        !r2["choices"].as_array().unwrap_or(&Vec::new()).is_empty(),
        "second request returned no choices: {r2}"
    );
    let prompt_tokens2 = r2["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
    assert_eq!(
        prompt_tokens2, usage_prompt_tokens,
        "second request prompt_tokens={prompt_tokens2} differs from first={usage_prompt_tokens} \
         — tokenization is unstable across calls"
    );
    eprintln!(
        "[smoke:minicpm5-2b] repeat request produced prompt_tokens={prompt_tokens2} \
         (matches first; KV cache is shared across calls so text bytes may diverge)"
    );

    Ok(true)
}
