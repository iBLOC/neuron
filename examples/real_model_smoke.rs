//! Real-model smoke harness for the neure runtime.
//!
//! Boots a single embedded server and runs whichever engine smokes
//! are configured by env vars:
//!
//! - **Embedding smoke**: three checks against `/v1/embeddings`
//!   using `all-MiniLM-L6-v2`. Requires `NEURE_EMBEDDING_MODEL_PATH`
//!   (default `/tmp/neure-smoke/models/embedding/all-MiniLM-L6-v2`)
//!   to exist on disk.
//!
//! - **mage-vl smoke**: a multimodal `/v1/chat/completions` request
//!   with a real inline PNG against `microsoft/Mage-VL`. Requires
//!   `NEURE_LLM_RUNTIME=mage_vl` and `NEURE_MAGE_VL_SHARD_DIR`
//!   (default points at the OpenSourceProjects checkout). Skipped
//!   silently when the engine isn't registered.
//!
//! Run with embedding alone (default):
//! ```bash
//! cargo run --example real_model_smoke --features candle
//! ```
//!
//! Run with mage-vl too (heavy — ~18 GB F32 on CPU):
//! ```bash
//! NEURE_LLM_RUNTIME=mage_vl \
//! NEURE_MAGE_VL_SHARD_DIR=$HOME/Projects/OpenSourceProjects/mage/mage-vl-checkpoint \
//! cargo run --release --example real_model_smoke --features candle
//! ```
//!
//! Set `NEURE_PORT=0` (default) to bind to a random free port.

use neure_lib::{run_embedded, NeureConfig, NeureEmbedConfig};

const RED_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEElEQVR4nGP4z8AARAwQCgAf7gP9i18U1AAAAABJRU5ErkJggg==";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let port: u16 = std::env::var("NEURE_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let cfg = NeureEmbedConfig {
        port,
        config: NeureConfig::new(),
    };
    let handle = run_embedded(cfg).await?;
    let port = handle.addr.port();
    eprintln!("[smoke] server bound on 127.0.0.1:{port}");

    // What engines got registered? Each smoke section reports itself.
    let did_embedding = embedding_smoke(port).await?;
    let did_mage_vl = mage_vl_smoke(port).await?;

    // ---- cleanup ----
    eprintln!("\n[smoke] shutting down server...");
    handle.request_shutdown();
    handle.join().await;

    match (did_embedding, did_mage_vl) {
        (false, false) => {
            eprintln!(
                "[smoke] FATAL: no engines were exercised. Set \
                 NEURE_EMBEDDING_MODEL_PATH (embedding) or \
                 NEURE_LLM_RUNTIME=mage_vl + NEURE_MAGE_VL_SHARD_DIR (mage-vl)."
            );
            std::process::exit(2);
        }
        _ => {
            eprintln!(
                "[smoke] ALL CHECKS PASSED (embedding={}, mage_vl={})",
                did_embedding, did_mage_vl
            );
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// Embedding smoke (unchanged from the original example)
// ---------------------------------------------------------------------------

async fn embedding_smoke(port: u16) -> Result<bool, Box<dyn std::error::Error>> {
    let model_path = std::env::var("NEURE_EMBEDDING_MODEL_PATH").unwrap_or_else(|_| {
        "/tmp/neure-smoke/models/embedding/all-MiniLM-L6-v2".to_string()
    });
    if !std::path::Path::new(&model_path)
        .join("model.safetensors")
        .exists()
    {
        eprintln!(
            "[smoke:embedding] skipping: model.safetensors not found in {model_path}. \
             Download all-MiniLM-L6-v2 first."
        );
        return Ok(false);
    }
    eprintln!("\n[smoke:embedding] loading from {model_path}");

    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{port}/v1/embeddings");

    // ---- 1. float encoding ----
    eprintln!("[smoke:embedding] 1. float encoding");
    let r: serde_json::Value = client
        .post(&url)
        .json(&serde_json::json!({
            "model": "all-minilm-l6-v2",
            "input": "Hello world, this is a real model smoke test."
        }))
        .send()
        .await?
        .json()
        .await?;
    let float_vec = r["data"][0]["embedding"].as_array().expect("array");
    let float_count = float_vec.len();
    let float_var = variance(float_vec.iter().filter_map(|v| v.as_f64()).collect());
    eprintln!("[smoke:embedding]   embedding dim = {float_count}");
    eprintln!("[smoke:embedding]   variance     = {float_var:.6}");
    assert_eq!(float_count, 384, "expected 384-dim embedding");
    assert!(float_var > 1e-6, "embedding has near-zero variance — model not loaded?");

    // ---- 2. base64 encoding ----
    eprintln!("[smoke:embedding] 2. base64 encoding");
    let r: serde_json::Value = client
        .post(&url)
        .json(&serde_json::json!({
            "model": "all-minilm-l6-v2",
            "input": "Hello world, this is a real model smoke test.",
            "encoding_format": "base64"
        }))
        .send()
        .await?
        .json()
        .await?;
    let b64 = r["data"][0]["embedding"].as_str().expect("base64 string");
    let bytes = base64_decode(b64).expect("valid base64");
    assert_eq!(bytes.len(), 384 * 4, "expected 1536 bytes (384 f32 LE)");
    let base64_vec: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    eprintln!("[smoke:embedding]   decoded bytes = {}", bytes.len());
    eprintln!("[smoke:embedding]   first 3 values = {:?}", &base64_vec[..3]);

    // Verify base64 roundtrip matches float encoding for the same input
    let base64_first: Vec<f32> = base64_vec[..3].to_vec();
    let float_first: Vec<f32> = float_vec
        .iter()
        .filter_map(|v| v.as_f64().map(|f| f as f32))
        .take(3)
        .collect();
    for (i, (a, b)) in float_first.iter().zip(base64_first.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(diff < 1e-5, "base64[{i}] {a} != float[{i}] {b} (diff {diff})");
    }
    eprintln!("[smoke:embedding]   base64/float roundtrip OK (max diff < 1e-5)");

    // ---- 3. batched forward + determinism ----
    eprintln!("[smoke:embedding] 3. batched forward (4 inputs)");
    let r: serde_json::Value = client
        .post(&url)
        .json(&serde_json::json!({
            "model": "all-minilm-l6-v2",
            "input": [
                "first document",
                "second document",
                "third document",
                "fourth document"
            ]
        }))
        .send()
        .await?
        .json()
        .await?;
    let data = r["data"].as_array().expect("data array");
    assert_eq!(data.len(), 4, "expected 4 rows back");
    for (i, row) in data.iter().enumerate() {
        let v = row["embedding"].as_array().expect("embedding");
        assert_eq!(v.len(), 384, "row {i} not 384-dim");
    }
    let r2: serde_json::Value = client
        .post(&url)
        .json(&serde_json::json!({
            "model": "all-minilm-l6-v2",
            "input": ["second document", "second document"]
        }))
        .send()
        .await?
        .json()
        .await?;
    let first = r["data"][1]["embedding"].as_array().unwrap();
    let repeat_first = r2["data"][0]["embedding"].as_array().unwrap();
    let mut max_diff = 0.0_f64;
    for (a, b) in first.iter().zip(repeat_first.iter()) {
        let av = a.as_f64().unwrap_or(0.0);
        let bv = b.as_f64().unwrap_or(0.0);
        max_diff = max_diff.max((av - bv).abs());
    }
    eprintln!("[smoke:embedding]   determinism max diff = {max_diff:.2e}");
    assert!(max_diff < 1e-5, "determinism broken: max diff {max_diff}");
    eprintln!("[smoke:embedding]   batched forward + determinism OK");

    Ok(true)
}

fn variance(values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
}

/// Minimal RFC 4648 base64 decoder (no external dep).
fn base64_decode(s: &str) -> Option<Vec<u8>> {
    const TABLE: &[u8; 128] = &{
        let mut t = [255u8; 128];
        let chars = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut i = 0;
        while i < 64 {
            t[chars[i] as usize] = i as u8;
            i += 1;
        }
        t
    };
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    let mut buf = [0u8; 4];
    let mut buf_len = 0;
    for &b in bytes {
        if b == b'=' {
            break;
        }
        let v = *TABLE.get(b as usize).unwrap_or(&255);
        if v == 255 {
            return None;
        }
        buf[buf_len] = v;
        buf_len += 1;
        if buf_len == 4 {
            out.push((buf[0] << 2) | (buf[1] >> 4));
            out.push((buf[1] << 4) | (buf[2] >> 2));
            out.push((buf[2] << 6) | buf[3]);
            buf_len = 0;
        }
    }
    if buf_len == 3 {
        out.push((buf[0] << 2) | (buf[1] >> 4));
        out.push((buf[1] << 4) | (buf[2] >> 2));
    } else if buf_len == 2 {
        out.push((buf[0] << 2) | (buf[1] >> 4));
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// mage-vl smoke (opt-in; requires NEURE_LLM_RUNTIME=mage_vl + shards)
// ---------------------------------------------------------------------------

/// Returns `true` if the engine was exercised; `false` if it was
/// skipped because the runtime wasn't configured or the shards
/// weren't on disk. The host embedding smoke already printed an
/// explanatory message; this one stays terse.
async fn mage_vl_smoke(port: u16) -> Result<bool, Box<dyn std::error::Error>> {
    let runtime = std::env::var("NEURE_LLM_RUNTIME").ok();
    if runtime.as_deref() != Some("mage_vl") {
        eprintln!(
            "[smoke:mage_vl] skipping: NEURE_LLM_RUNTIME != \"mage_vl\" \
             (got {runtime:?})"
        );
        return Ok(false);
    }

    let shard_dir = std::env::var("NEURE_MAGE_VL_SHARD_DIR").unwrap_or_else(|_| {
        "/home/ibloc/Projects/OpenSourceProjects/mage/mage-vl-checkpoint".to_string()
    });
    if !std::path::Path::new(&shard_dir)
        .join("model-00001-of-00002.safetensors")
        .exists()
    {
        eprintln!(
            "[smoke:mage_vl] skipping: shards not found at {shard_dir}. \
             Set NEURE_MAGE_VL_SHARD_DIR to the directory containing the two \
             model-0000X-of-00002.safetensors shards."
        );
        return Ok(false);
    }
    eprintln!("\n[smoke:mage_vl] loading from {shard_dir}");

    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{port}/v1/chat/completions");

    // ---- 1. multimodal chat completion ----
    eprintln!("[smoke:mage_vl] 1. multimodal chat completion (image + text)");
    let r: serde_json::Value = client
        .post(&url)
        .json(&serde_json::json!({
            "model": "microsoft/Mage-VL",
            "max_tokens": 8,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color is this image?"},
                    {"type": "image_url", "image_url": {
                        "url": format!("data:image/png;base64,{RED_PNG_B64}")
                    }}
                ]
            }]
        }))
        .send()
        .await?
        .json()
        .await?;
    assert_eq!(
        r["object"], "chat.completion",
        "unexpected response shape: {r}"
    );
    let content = r["choices"][0]["message"]["content"]
        .as_str()
        .expect("content should be a string");
    assert!(
        !content.is_empty(),
        "mage-vl returned an empty response (forward produced no logits?)"
    );
    eprintln!("[smoke:mage_vl]   reply = {content:?}");

    // ---- 2. cross-request isolation (reset_cache) ----
    // Two identical requests after the first must produce identical
    // content because MageVlRuntime resets the text-tower KV cache
    // before every execute. The OpenAI API doesn't surface raw logits
    // so we compare the decoded text — the text being byte-identical
    // is the strongest correctness signal available at the wire.
    eprintln!("[smoke:mage_vl] 2. cross-request isolation (reset_cache)");
    let body = serde_json::json!({
        "model": "microsoft/Mage-VL",
        "max_tokens": 8,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What color is this image?"},
                {"type": "image_url", "image_url": {
                    "url": format!("data:image/png;base64,{RED_PNG_B64}")
                }}
            ]
        }]
    });
    let r2: serde_json::Value = client
        .post(&url)
        .json(&body)
        .send()
        .await?
        .json()
        .await?;
    let content2 = r2["choices"][0]["message"]["content"]
        .as_str()
        .expect("content should be a string");
    assert_eq!(
        content, content2,
        "second request must be byte-identical to the first — \
         reset_cache failed (KV cross-request contamination); \
         got {content2:?} after {content:?}"
    );
    eprintln!("[smoke:mage_vl]   both requests produced identical text");

    Ok(true)
}
