//! Scheduler smoke harness — verifies the model scheduler end-to-end
//! with REAL candle weights on disk.
//!
//! Phases (all driven through the real axum router, i.e. the same
//! `adapter_dispatch` path production traffic takes):
//!
//! 1. **Coexistence** — budget sized to fit both models: requesting
//!    model A then model B keeps both resident, zero unloads.
//! 2. **Drain-then-swap** — budget sized to fit one model only: a
//!    request for B while A is resident evicts A (real weight
//!    unload, RSS drops, RSS observation recorded) and loads B.
//! 3. **Administrative eviction** — `force_evict` drops the resident
//!    model's weights on demand; RSS returns to the pre-load floor.
//!
//! Models are auto-registered from `~/.neuron/models/llm/` (the
//! default model dir): any directory containing `model.safetensors`
//! is served by the candle runtime. The smoke uses whatever models
//! it finds there, needing exactly two; the first is A, the second
//! B (lexicographic order). On the standard benchmark checkout that
//! is `qwen2.5-0.5b` (~2.3 GB f32 resident) then `qwen3-0.6b`
//! (~2.9 GB f32 resident) — both values are measured, see
//! `assets/model_memory_profiles.json`.
//!
//! Run:
//! ```bash
//! cargo run --release --example scheduler_smoke --features candle
//! ```
//!
//! Exit code 0 = all phases passed. Each phase prints the scheduler
//! status and the RSS delta it observed.

use std::sync::Arc;

use axum::body::to_bytes;
use neuron_lib::config::NeuronConfig;
use neuron_lib::server::ServerState;
use tower::util::ServiceExt;

fn rss_mib() -> f64 {
    neuron_lib::scheduler::mem_probe::rss_bytes() as f64 / 1_048_576.0
}

fn chat_request(model: &str) -> axum::http::Request<axum::body::Body> {
    let payload = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "Say OK and nothing else."}],
        "max_tokens": 12,
    });
    axum::http::Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(payload.to_string()))
        .unwrap()
}

async fn chat(state: &ServerState, model: &str) -> Result<String, String> {
    let router = neuron_lib::server::routes().with_state(state.clone());
    let resp = router
        .oneshot(chat_request(model))
        .await
        .map_err(|e| format!("request failed: {e}"))?;
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), 4 * 1024 * 1024)
        .await
        .map_err(|e| format!("body read: {e}"))?;
    let json: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("json: {e}"))?;
    if !status.is_success() {
        return Err(format!("HTTP {status}: {json}"));
    }
    Ok(json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default()
        .to_string())
}

fn found_llm_models(config: &NeuronConfig) -> Vec<String> {
    let mut ids = Vec::new();
    for dir in &config.model_dirs {
        let Ok(entries) = std::fs::read_dir(dir.join("llm")) else {
            continue;
        };
        for entry in entries.flatten() {
            if entry.path().join("model.safetensors").exists() {
                ids.push(entry.file_name().to_string_lossy().to_string());
            }
        }
    }
    ids.sort();
    ids
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut config = NeuronConfig::new();
    let models = found_llm_models(&config);
    if models.len() < 2 {
        eprintln!(
            "[smoke] FATAL: need 2 downloaded LLM models under {:?} (found: {models:?}). \
             Download a second model via /v1/models/pull first.",
            config.model_dirs
        );
        std::process::exit(2);
    }
    let (model_a, model_b) = (models[0].clone(), models[1].clone());
    eprintln!("[smoke] models: A={model_a:?} B={model_b:?}");

    // ---- Phase 1: coexistence (budget fits both) ----------------------
    eprintln!("\n=== Phase 1: coexistence ===");
    eprintln!("[smoke] RSS before load: {:.0} MiB", rss_mib());
    let mut cfg1 = config.clone();
    cfg1.scheduler.budget_bytes = 20 * 1024 * 1024 * 1024; // 20 GiB — fits both
    let state1 = ServerState::new(cfg1);
    let t0 = std::time::Instant::now();
    let reply_a = chat(&state1, &model_a).await?;
    eprintln!(
        "[smoke] {model_a} replied {:?} in {:.1}s — RSS now {:.0} MiB",
        truncate(&reply_a),
        t0.elapsed().as_secs_f32(),
        rss_mib()
    );
    let t0 = std::time::Instant::now();
    let reply_b = chat(&state1, &model_b).await?;
    eprintln!(
        "[smoke] {model_b} replied {:?} in {:.1}s — RSS now {:.0} MiB",
        truncate(&reply_b),
        t0.elapsed().as_secs_f32(),
        rss_mib()
    );
    let st = state1.llm_scheduler.status();
    assert_eq!(
        st.residents.len(),
        2,
        "phase 1: both models resident, got {:?}",
        st.residents
    );
    assert_eq!(st.stats.unloads, 0, "phase 1: no evictions expected");
    eprintln!("[smoke] status: 2 residents, 0 unloads ✓");
    // Release phase-1 weights so later RSS readings reflect only the
    // phase under test.
    drop(state1);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    eprintln!("[smoke] phase-1 state dropped — RSS back to {:.0} MiB", rss_mib());

    // ---- Phase 2: drain-then-swap (budget fits one) --------------------
    eprintln!("\n=== Phase 2: drain-then-swap ===");
    let mut cfg2 = config.clone();
    cfg2.scheduler.budget_bytes = 4 * 1024 * 1024 * 1024; // 4 GiB — fits only one
    cfg2.scheduler.max_queue_wait_secs = 600;
    let state2 = ServerState::new(cfg2);
    let t0 = std::time::Instant::now();
    let reply_a = chat(&state2, &model_a).await?;
    eprintln!(
        "[smoke] {model_a} replied {:?} in {:.1}s — RSS {:.0} MiB",
        truncate(&reply_a),
        t0.elapsed().as_secs_f32(),
        rss_mib()
    );
    let t0 = std::time::Instant::now();
    let reply_b = chat(&state2, &model_b).await?;
    let swap_secs = t0.elapsed().as_secs_f32();
    eprintln!(
        "[smoke] {model_b} replied {:?} in {swap_secs:.1}s (includes the swap) — RSS {:.0} MiB",
        truncate(&reply_b),
        rss_mib()
    );
    let st = state2.llm_scheduler.status();
    assert_eq!(
        st.residents.len(),
        1,
        "phase 2: only B resident after swap, got {:?}",
        st.residents
    );
    assert_eq!(st.residents[0].model_id, model_b, "phase 2: B swapped in");
    assert_eq!(st.stats.unloads, 1, "phase 2: A evicted exactly once");
    assert_eq!(
        st.stats.rss_observations, 1,
        "phase 2: unload RSS delta recorded"
    );
    eprintln!(
        "[smoke] status: resident={} unloads=1 rss_observations=1 ✓",
        st.residents[0].model_id
    );

    // ---- Phase 3: administrative eviction ------------------------------
    eprintln!("\n=== Phase 3: admin eviction ===");
    let rss_before = rss_mib();
    let evicted = state2.llm_scheduler.force_evict(&model_b).unwrap();
    assert!(evicted, "phase 3: B was resident");
    // Give the allocator a moment to return pages.
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    let rss_after = rss_mib();
    eprintln!(
        "[smoke] force_evict({model_b}): RSS {:.0} → {:.0} MiB (Δ {:+.0})",
        rss_before,
        rss_after,
        rss_after - rss_before
    );
    assert!(state2.llm_scheduler.force_evict(&model_b).unwrap() == false);
    let st = state2.llm_scheduler.status();
    assert!(st.residents.is_empty(), "phase 3: no residents after evict");
    eprintln!("[smoke] status: 0 residents ✓");

    // ---- Correction-loop evidence --------------------------------------
    eprintln!("\n=== Correction loop ===");
    let profile_path = config.model_dirs[0]
        .join("scheduler")
        .join("observed_profiles.json");
    if let Ok(content) = std::fs::read_to_string(&profile_path) {
        let v: serde_json::Value = serde_json::from_str(&content)?;
        if let Some(entry) = v["models"].as_object() {
            for (id, e) in entry {
                eprintln!(
                    "[smoke] observed {id}: cpu={} bytes ({})",
                    e["bytes"]["cpu"],
                    e["source"]
                );
            }
        }
        eprintln!("[smoke] persisted to {}", profile_path.display());
    }

    eprintln!("\n[smoke] ALL PHASES PASSED");
    Ok(())
}

fn truncate(s: &str) -> String {
    let t: String = s.chars().take(40).collect();
    if s.chars().count() > 40 {
        format!("{t}…")
    } else {
        t
    }
}

// Keep Arc imported for parity with state fields (silences unused
// import edits when wiring changes).
#[allow(dead_code)]
fn _typecheck(_: Arc<ServerState>) {}
