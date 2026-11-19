//! End-to-end over the **real HTTP surface**: `POST /v1/forecast` → the real
//! axum router → the registry → the upstream-compatible Chronos-Bolt runtime
//! → the **released** `amazon/chronos-bolt-small` checkpoint.
//!
//! Why this test exists: the shape of the wire was already proven by unit
//! tests ⇒ ⇒ what was **not** proven is that the numbers on that wire are
//! the time-series foundation model's numbers. This test checks them against
//! the **upstream Python implementation's** own output
//! (`chronos-forecasting` 2.3.2, real checkpoint) ⇒ ⇒ a drift in *meaning*
//! fails here even though every shape still matches.
//!
//! Skipped — honestly, with a printed note — when the checkpoint is not on
//! disk (a 190 MB download is not a test fixture).

#![cfg(feature = "chronos2")]

use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

fn checkpoint_dir() -> Option<std::path::PathBuf> {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .ok()?;
    let dir = std::path::PathBuf::from(home)
        .join(".neuron")
        .join("models")
        .join("chronos2")
        .join("chronos-bolt-small");
    dir.is_dir().then_some(dir)
}

fn reference() -> serde_json::Value {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("chronos2")
        .join("testdata")
        .join("bolt_reference.json");
    let raw = std::fs::read_to_string(path).expect("reference fixture");
    serde_json::from_str(&raw).expect("reference json")
}

#[tokio::test(flavor = "multi_thread")]
async fn the_http_surface_serves_the_real_time_series_model() {
    let dir = checkpoint_dir();
    if dir.is_none() {
        eprintln!("skipped: chronos-bolt-small checkpoint not on disk");
        return;
    }

    // The registry resolves the model lazily per request ⇒ ⇒ the env var can
    // be set here, inside the test.
    unsafe {
        std::env::set_var("NEURON_CHRONOS2_DEFAULT_MODEL", "chronos-bolt-small");
    }

    let app = neuron_lib::server::create_router(neuron_lib::config::NeuronConfig::default());
    let reference = reference();
    let cases = reference["cases"].as_array().expect("cases");
    assert!(!cases.is_empty(), "reference fixture has no cases");

    for case in cases {
        let series = case["series"].as_array().expect("series");
        let payload = serde_json::json!({
            "model": "chronos-bolt-small",
            "series": series,
            "horizon": 16,
            "method": "median",
        });
        let request = Request::builder()
            .method("POST")
            .uri("/v1/forecast")
            .header("content-type", "application/json")
            .body(Body::from(payload.to_string()))
            .expect("request");

        let response = app.clone().oneshot(request).await.expect("router call");
        assert_eq!(
            response.status(),
            StatusCode::OK,
            "case {}: /v1/forecast did not return 200",
            case["name"]
        );
        let bytes = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        let value: serde_json::Value = serde_json::from_slice(&bytes).expect("json body");

        assert_eq!(value["model"], "chronos-bolt-small");
        assert_eq!(value["method"], "median");

        let forecast: Vec<f64> = value["forecast"]
            .as_array()
            .expect("forecast array")
            .iter()
            .map(|v| v.as_f64().expect("number"))
            .collect();
        let expected: Vec<f64> = case["quantiles"]
            .as_array()
            .expect("expected array")
            .iter()
            .map(|v| v.as_f64().expect("number"))
            .collect();
        assert_eq!(forecast.len(), expected.len());

        let worst = forecast
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            worst < 1e-2,
            "case {}: HTTP-surface median drift {worst} exceeds 1e-2 (got {:?}, expected {:?})",
            case["name"],
            &forecast[..],
            &expected[..]
        );
    }
}
