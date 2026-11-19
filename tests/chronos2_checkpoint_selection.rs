//! 真 HTTP 判据：**点名的那枚 checkpoint 不可用** ⇒ ⇒ 必须**明确报错**。
//!
//! 三条（承重的是第三条）：
//!   1. 点名一枚**没装**的 ⇒ ⇒ **错误**（不是 200）且
//!      错误体里**有那枚 id**；
//!   2. 点名一枚**装了但坏了**的 ⇒ ⇒ **错误**＋说清为什么（加载失败）；
//!   3. **不静默回落**：点名不存在的 ⇒ ⇒ 回包里
//!      **不许**出现**别的 checkpoint 的名字**。
//!
//! 形状照 `tests/chronos2_http_e2e.rs`（真 router ＋真 HTTP＋真磁盘）。
//!
//! ⚠ **一个 test 函数**（不是多个）—— 因为 env var 是
//! **进程全局**的 ⇒ ⇒ 多个 test 并行跑会互相踩
//! （这正是「按名字选 checkpoint」那一格的测试面
//! 最容易翻车的地方）。

#![cfg(feature = "chronos2")]

use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

/// Write a checkpoint that is **installed but broken**: a `config.json`
/// (declaring its identity) ＋ a garbage `*.safetensors` file.
///
/// ⇒ ⇒ the load fails ⇒ ⇒ the reply has to say so (and name the id).
fn write_broken_checkpoint(dir: &std::path::Path, declared_identity: &str) {
    std::fs::create_dir_all(dir).expect("create checkpoint dir");
    let config = serde_json::json!({
        "_name_or_path": format!("autogluon/{declared_identity}"),
        "d_model": 8,
        "d_ff": 16,
        "num_layers": 1,
        "num_decoder_layers": 1,
        "num_heads": 2,
        "vocab_size": 2,
    });
    std::fs::write(dir.join("config.json"), config.to_string()).expect("write config.json");
    std::fs::write(
        dir.join("model.safetensors"),
        b"this is not a safetensors file",
    )
    .expect("write weights");
}

/// Post one forecast request ⇒ ⇒ (status, body).
async fn post_forecast(
    app: axum::Router,
    model: &str,
) -> (StatusCode, serde_json::Value) {
    let payload = serde_json::json!({
        "model": model,
        "series": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "horizon": 4,
        "method": "median",
    });
    let request = Request::builder()
        .method("POST")
        .uri("/v1/forecast")
        .header("content-type", "application/json")
        .body(Body::from(payload.to_string()))
        .expect("request");
    let response = app.oneshot(request).await.expect("router call");
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| serde_json::Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    (status, value)
}

#[tokio::test(flavor = "multi_thread")]
async fn the_forecast_face_refuses_a_checkpoint_it_cannot_serve() {
    // ── setup: a per-id checkpoint tree holding ONE broken checkpoint ──
    let temp = std::env::temp_dir().join("neuron-chronos2-checkpoint-selection");
    let _ = std::fs::remove_dir_all(&temp);
    let tree = temp.join("chronos2");
    let broken_id = "chronos-bolt-broken";
    write_broken_checkpoint(&tree.join(broken_id), broken_id);
    // ⚠ A checkpoint that declares **another** model's identity ⇒ ⇒ neuron must
    //    refuse to serve it under this name (that is the silent swap).
    let mislabeled_id = "chronos-bolt-mislabeled";
    write_broken_checkpoint(&tree.join(mislabeled_id), "chronos-bolt-small");

    // The tree shape: `NEURON_CHRONOS2_MODEL_PATH/<id>`.
    unsafe {
        std::env::set_var("NEURON_CHRONOS2_MODEL_PATH", &tree);
    }

    let app = neuron_lib::server::create_router(neuron_lib::config::NeuronConfig::default());

    // ── ① 点名一枚**没装**的 ⇒ ⇒ 错误 ＋ 错误体里有那枚 id ──
    let missing_id = "chronos-bolt-not-installed-xyz";
    let (status, body) = post_forecast(app.clone(), missing_id).await;
    assert!(
        !status.is_success(),
        "a checkpoint that is not installed must not answer 200 (got {status})"
    );
    let text = body.to_string();
    assert!(
        text.contains(missing_id),
        "the error must name the checkpoint that was asked for: {text}"
    );

    // ── ③ **承重：不静默回落** ⇒ ⇒ 回包里不许出现别的模型名 ──
    assert!(
        !text.contains("chronos-bolt-small"),
        "the reply must not name another checkpoint: {text}"
    );
    assert!(
        !text.contains(broken_id),
        "the reply must not name the broken checkpoint: {text}"
    );
    assert!(
        body.get("forecast").is_none(),
        "an error must not carry a forecast: {text}"
    );

    // ── ② 点名一枚**装了但坏了**的 ⇒ ⇒ 错误 ＋说清为什么 ──
    let (status, body) = post_forecast(app.clone(), broken_id).await;
    assert!(
        !status.is_success(),
        "a broken checkpoint must not answer 200 (got {status})"
    );
    let text = body.to_string();
    assert!(
        text.contains(broken_id),
        "the error must name the broken checkpoint: {text}"
    );
    assert!(
        text.contains("failed to load"),
        "the error must say why it is unavailable: {text}"
    );

    // ── ④ 装的是**另一枚**的 checkpoint ⇒ ⇒ 拒绝（不得一枚空壳当那枚） ──
    let (status, body) = post_forecast(app.clone(), mislabeled_id).await;
    assert!(
        !status.is_success(),
        "a mislabeled checkpoint must not answer 200 (got {status})"
    );
    let text = body.to_string();
    assert!(
        text.contains(mislabeled_id),
        "the error must name the checkpoint that was asked for: {text}"
    );
    assert!(
        text.contains("declares"),
        "the error must say the checkpoint is another model: {text}"
    );
}
