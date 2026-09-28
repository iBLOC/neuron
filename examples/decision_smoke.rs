//! End-to-end smoke for `POST /v1/decisions`.
//!
//! `tests/decision_parity.rs` drives `TractDecisionRuntime::forward` directly, so
//! it never touches the HTTP layer. This example closes that gap: it boots a real
//! embedded server and exercises the request through the handler, the registry,
//! the tract session and the postprocessing.
//!
//! Requires `--features tract`, a **prepped** model directory (run
//! `scripts/prep-laya-onnx.py` first — the raw export is not loadable), and:
//!
//! ```bash
//! NEURON_DECISION_RUNTIME=tract \
//! NEURON_DECISION_MODEL_PATH=/path/to/prepped-model-dir \
//! cargo run --release --example decision_smoke --features tract
//! ```
//!
//! Skips with a clear message when the env vars are absent, so it is safe to run
//! in CI without model weights. Set `NEURON_PORT=0` (the default) to bind to a
//! random free port.
//!
//! The prompt budget defaults to the `parity64dql` fixture (seq 64 /5 markers);
//! point `NEURON_DECISION_SMOKE_MAX_LEN` / `NEURON_DECISION_SMOKE_HEAD_MAX_LEN`
//! at a larger graph's extents when running against one.

use neuron_lib::{run_embedded, NeuronConfig, NeuronEmbedConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = match std::env::var("NEURON_DECISION_MODEL_PATH") {
        Ok(p) if !p.trim().is_empty() => p,
        _ => {
            eprintln!(
                "[smoke:decision] skipping: set NEURON_DECISION_MODEL_PATH to a prepped \
                 decision model dir (see scripts/prep-laya-onnx.py). \
                 Also set NEURON_DECISION_RUNTIME=tract."
            );
            return Ok(());
        }
    };
    if !std::path::Path::new(&model_path).exists() {
        eprintln!("[smoke:decision] skipping: {model_path} does not exist");
        return Ok(());
    }
    eprintln!("[smoke:decision] model dir: {model_path}");

    let port: u16 = std::env::var("NEURON_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let handle = run_embedded(NeuronEmbedConfig {
        port,
        config: NeuronConfig::new(),
    })
    .await?;
    let port = handle.addr.port();
    eprintln!("[smoke:decision] server bound on 127.0.0.1:{port}");

    let result = decision_smoke(port).await;

    eprintln!("\n[smoke:decision] shutting down server...");
    handle.request_shutdown();
    handle.join().await;

    match result {
        Ok(true) => {
            eprintln!("[smoke:decision] ALL CHECKS PASSED");
            Ok(())
        }
        Ok(false) => {
            eprintln!("[smoke:decision] FATAL: the decision engine was not registered");
            std::process::exit(2);
        }
        Err(e) => {
            eprintln!("[smoke:decision] FAILED: {e}");
            std::process::exit(1);
        }
    }
}

async fn decision_smoke(port: u16) -> Result<bool, Box<dyn std::error::Error>> {
    let base = format!("http://127.0.0.1:{port}");
    let client = reqwest::Client::new();

    // ---- 1. liveness ----
    let health: serde_json::Value = client
        .get(format!("{base}/health"))
        .send()
        .await?
        .json()
        .await?;
    assert_eq!(health["status"], "healthy", "unhealthy server: {health}");
    eprintln!("[smoke:decision]   /health ok");

    // ---- 2. the engine is actually registered ----
    // Without NEURON_DECISION_RUNTIME nothing registers, and every check below
    // would 4xx for the wrong reason. Bail out loudly instead.
    //
    // `/v1/info`'s `models` is built from the runtime registry, so it reports
    // what this process registered regardless of where the weights live.
    // `/v1/models` is the *download* catalog instead: it lists a decision
    // model only when the weights sit under `$NEURON_MODEL_DIRS/decision/<id>/`,
    // because the upstream Laya release ships safetensors and neuron's tract
    // path consumes a prepped ONNX -- see the spec for the export + prep chain.
    let info: serde_json::Value = client
        .get(format!("{base}/v1/info"))
        .send()
        .await?
        .json()
        .await?;
    let registered = info["models"]
        .as_array()
        .map(|v| {
            v.iter().any(|m| {
                m["id"]
                    .as_str()
                    .map_or(false, |id| id.starts_with("decision/"))
            })
        })
        .unwrap_or(false);
    if !registered {
        eprintln!(
            "[smoke:decision]   no decision model is registered — \
             set NEURON_DECISION_RUNTIME=tract and restart"
        );
        eprintln!("[smoke:decision]   /v1/info said: {info}");
        return Ok(false);
    }
    eprintln!("[smoke:decision]   /v1/info lists the registered decision model");

    let models: serde_json::Value = client
        .get(format!("{base}/v1/models"))
        .send()
        .await?
        .json()
        .await?;
    let advertised = models["data"]
        .as_array()
        .map(|v| {
            v.iter().any(|m| {
                m["capabilities"]
                    .as_array()
                    .map_or(false, |c| c.iter().any(|x| x == "decision"))
            })
        })
        .unwrap_or(false);
    eprintln!(
        "[smoke:decision]   /v1/models catalog lists it: {advertised} \
         (expected only when the weights are under $NEURON_MODEL_DIRS/decision/<id>/)"
    );

    let url = format!("{base}/v1/decisions");

    // ---- 3. one question of each type ----
    let choice = ask(&client, &url, vec![question("choice", 3)]).await?;
    check_result(&choice["data"][0], 3, "choice")?;
    eprintln!("[smoke:decision]   choice: {}", describe(&choice["data"][0]));

    let score = ask(&client, &url, vec![question("score", 4)]).await?;
    check_result(&score["data"][0], 4, "score")?;
    eprintln!("[smoke:decision]   score:  {}", describe(&score["data"][0]));

    let noul = ask(&client, &url, vec![question("noul", 2)]).await?;
    check_result(&noul["data"][0], 2, "noul")?;
    eprintln!("[smoke:decision]   noul:   {}", describe(&noul["data"][0]));

    // ---- 4. a multi-question request must match the single-question runs ----
    // The graph is grounded at batch 1, so this only works if `forward` splits
    // the batch. Comparing against the singles is what catches a chunking
    // off-by-one: a wrong row would still pass the shape checks above.
    let singles = vec![
        ask(&client, &url, vec![question("choice", 3)]).await?["data"][0].clone(),
        ask(&client, &url, vec![question("score", 4)]).await?["data"][0].clone(),
        ask(&client, &url, vec![question("noul", 2)]).await?["data"][0].clone(),
    ];
    let batched = ask(
        &client,
        &url,
        vec![question("choice", 3), question("score", 4), question("noul", 2)],
    )
    .await?;
    let rows = batched["data"].as_array().expect("data is an array");
    assert_eq!(rows.len(), 3, "one result per question");
    for (i, (got, want)) in rows.iter().zip(singles.iter()).enumerate() {
        assert_eq!(
            got["probabilities"], want["probabilities"],
            "question {i} probabilities differ between the batched and single runs"
        );
        assert_eq!(
            got["chosen_index"], want["chosen_index"],
            "question {i} decision differs between the batched and single runs"
        );
    }
    eprintln!("[smoke:decision]   3-question batch reproduced the single runs exactly");

    // ---- 5. usage is reported ----
    let prompt_tokens = batched["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
    assert!(prompt_tokens > 0, "usage.prompt_tokens must be > 0");
    eprintln!("[smoke:decision]   usage.prompt_tokens = {prompt_tokens}");

    // ---- 6. a malformed request is a 4xx, not a silent mis-score ----
    let bad = client
        .post(&url)
        .json(&serde_json::json!({"model": "laya", "state": {}, "questions": []}))
        .send()
        .await?;
    let status = bad.status();
    assert!(
        status.is_client_error(),
        "an empty `questions` array must be a 4xx, got {status}"
    );
    eprintln!("[smoke:decision]   empty questions rejected with {status}");

    Ok(true)
}

/// Prompt budget this smoke targets.
///
/// A prepped graph has its sequence extent grounded, and `collate_to` refuses
/// to truncate past it (silently shortening a decision prompt can change the
/// decision), so the request has to ask for a budget that fits. The default
/// matches the `parity64dql` fixture used by `tests/decision_parity.rs`
/// (seq 64 /5 markers). Raise these when pointing the smoke at a larger
/// prepped graph.
fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(default)
}

/// One question of the requested type with `n` options.
fn question(qtype: &str, n: usize) -> serde_json::Value {
    // `noul` is always binary and takes its criteria as a `{"false", "true"}`
    // object (a list is rejected) plus display labels from `labels`.
    if qtype == "noul" {
        return serde_json::json!({
            "type": "noul",
            "instructions": "Does the statement hold?",
            "criteria": {
                "false": "the statement does not hold",
                "true": "the statement holds",
            },
            "labels": {"false": "no", "true": "yes"},
        });
    }
    let labels: Vec<String> = (0..n).map(|i| format!("Option {i}")).collect();
    serde_json::json!({
        "type": qtype,
        "instructions": "What should happen next?",
        "criteria": labels,
    })
}

async fn ask(
    client: &reqwest::Client,
    url: &str,
    questions: Vec<serde_json::Value>,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let body = serde_json::json!({
        "model": "laya",
        "state": {"turn": 3, "last_error": "timeout"},
        "questions": questions,
        "max_len": env_usize("NEURON_DECISION_SMOKE_MAX_LEN", 64),
        "head_max_len": env_usize("NEURON_DECISION_SMOKE_HEAD_MAX_LEN", 32),
    });
    let resp = client.post(url).json(&body).send().await?;
    let status = resp.status();
    let text = resp.text().await?;
    if !status.is_success() {
        return Err(format!("POST /v1/decisions -> {status}: {text}").into());
    }
    Ok(serde_json::from_str(&text)?)
}

/// Structural sanity on one result. Panics on anything the postprocessing
/// contract forbids, since a wrong probability is worse than a crash.
fn check_result(
    r: &serde_json::Value,
    option_count: usize,
    qtype: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let probs = r["probabilities"].as_array().expect("probabilities is an array");
    assert_eq!(probs.len(), option_count, "one probability per option");
    let mut sum = 0.0f64;
    for p in probs {
        let v = p.as_f64().expect("probability is a number");
        assert!((0.0..=1.0).contains(&v), "probability {v} out of [0, 1]");
        sum += v;
    }
    assert!((sum - 1.0).abs() < 1e-6, "probabilities sum to {sum}, not 1");

    let chosen = r["chosen_index"].as_u64().expect("chosen_index") as usize;
    assert!(chosen < option_count, "chosen_index {chosen} >= {option_count}");
    let best = probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.as_f64().unwrap().partial_cmp(&b.1.as_f64().unwrap()).unwrap())
        .unwrap()
        .0;
    assert_eq!(chosen, best, "chosen_index is not the argmax of probabilities");

    let confidence = r["confidence"].as_f64().expect("confidence");
    assert!((0.0..=1.0).contains(&confidence), "confidence {confidence} out of [0, 1]");

    // The temperature table is clamped to this range by the runtime.
    let temp = r["temperature_scale"].as_f64().expect("temperature_scale");
    assert!(
        (0.5..=5.0).contains(&temp),
        "temperature_scale {temp} outside the clamped range"
    );

    assert_eq!(
        r["qtype"].as_str().expect("qtype"),
        qtype,
        "the response echoes the wrong question type"
    );

    if let Some(act) = r["act"].as_object() {
        let a = act["act"].as_f64().expect("act.act");
        let e = act["escalate"].as_f64().expect("act.escalate");
        assert!((0.0..=1.0).contains(&a) && (0.0..=1.0).contains(&e), "act probs out of range");
        assert!((a + e - 1.0).abs() < 1e-6, "act probabilities sum to {}", a + e);
    }

    match qtype {
        "choice" => assert!(r["chosen"].as_str().is_some(), "choice carries a label"),
        "score" => assert!(r["score"].as_f64().is_some(), "score carries an expected level"),
        "noul" => assert!(r["noul"].as_f64().is_some(), "noul carries P(true)"),
        _ => {}
    }
    Ok(())
}

fn describe(r: &serde_json::Value) -> String {
    format!(
        "chosen_index={} confidence={:.3} act={:.3}/{:.3} p={:?}",
        r["chosen_index"].as_u64().unwrap_or(99),
        r["confidence"].as_f64().unwrap_or(f64::NAN),
        r["act"]["act"].as_f64().unwrap_or(f64::NAN),
        r["act"]["escalate"].as_f64().unwrap_or(f64::NAN),
        r["probabilities"].as_array().map(|v| v.iter()
            .map(|p| p.as_f64().unwrap_or(f64::NAN))
            .collect::<Vec<_>>())
    )
}
