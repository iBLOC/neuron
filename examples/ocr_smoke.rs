//! End-to-end smoke for `POST /v1/ocr/recognize`.
//!
//! Boots a real embedded server, registers the PP-OCR pipeline
//! (`paddle-ocr` engine) from `NEURON_OCR_MODEL_PATH`, and drives the HTTP
//! handler → `OcrRuntimeRegistry` → tract sessions → DB++ post-processing →
//! CTC decode path with a real image.
//!
//! Requires `--features tract` and a model directory holding det/rec ONNX
//! (prepped with `scripts/prep-ppocr-onnx.py`) plus the dict file:
//!
//! ```bash
//! NEURON_OCR_MODEL_PATH=/path/to/ppocrv6-dir \
//! cargo run --release --example ocr_smoke --features tract -- /path/to/image.png
//! ```
//!
//! Skips with a clear message when the env var or the image argument is
//! absent, so it is safe in CI without model weights.

use neuron_lib::{run_embedded, EngineType, NeuronConfig, NeuronEmbedConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = match std::env::var("NEURON_OCR_MODEL_PATH") {
        Ok(p) if !p.trim().is_empty() => p,
        _ => {
            eprintln!(
                "[smoke:ocr] skipping: set NEURON_OCR_MODEL_PATH to a directory with \
                 det.onnx / rec.onnx / dict.txt (prepped with scripts/prep-ppocr-onnx.py)"
            );
            return Ok(());
        }
    };
    let image_path = match std::env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("[smoke:ocr] skipping: pass an image path as the first argument");
            return Ok(());
        }
    };
    // Optional second arg: comma-separated languages routed to a rec pack
    // (e.g. "fr" → rec-latin, "ko" → rec-korean).
    let languages: Vec<String> = std::env::args()
        .nth(2)
        .map(|s| s.split(',').map(|l| l.trim().to_string()).collect())
        .unwrap_or_default();
    if !std::path::Path::new(&image_path).is_file() {
        eprintln!("[smoke:ocr] skipping: image {image_path} does not exist");
        return Ok(());
    }
    eprintln!("[smoke:ocr] model dir: {model_path}");
    eprintln!("[smoke:ocr] image: {image_path}");
    if !languages.is_empty() {
        eprintln!("[smoke:ocr] languages: {languages:?}");
    }

    let model_id = "ppocr-smoke";
    let config = NeuronConfig::new().with_registration_for_capability(
        EngineType::Ocr,
        model_id,
        "paddle-ocr",
        neuron_lib::DeviceSelection::Cpu,
        500_000_000, // det+rec+fudge, well under the default CPU budget
    );

    let port: u16 = std::env::var("NEURON_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let handle = run_embedded(NeuronEmbedConfig { port, config }).await?;
    let port = handle.addr.port();
    eprintln!("[smoke:ocr] server bound on 127.0.0.1:{port}");

    let image_url = format!("file://{image_path}");
    let mut body = serde_json::json!({
        "model": model_id,
        "image": { "type": "image_url", "image_url": { "url": image_url } }
    });
    if !languages.is_empty() {
        body["languages"] = serde_json::json!(languages);
    }

    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{port}/v1/ocr/recognize");
    eprintln!("[smoke:ocr] POST {url}");

    let resp = client.post(&url).json(&body).send().await?;
    let status = resp.status();
    let payload: serde_json::Value = resp.json().await?;

    if !status.is_success() {
        eprintln!("[smoke:ocr] FAILED: {status}: {payload}");
        handle.request_shutdown();
        handle.join().await;
        std::process::exit(1);
    }
    if std::env::var("NEURON_OCR_DEBUG").as_deref() == Ok("1") {
        eprintln!("[smoke:ocr] payload: {payload}");
    }

    let text = payload["text"].as_str().unwrap_or_default();
    let regions = payload["regions"].as_array().map(|a| a.len()).unwrap_or(0);
    let ms = payload["inference_time_ms"].as_u64().unwrap_or(0);
    eprintln!(
        "[smoke:ocr] OK: {regions} regions in {ms} ms\n--- text ---\n{text}\n------------"
    );
    if let Some(layout) = payload["layout"].as_array() {
        let kinds: Vec<&str> = layout
            .iter()
            .filter_map(|r| r["kind"].as_str())
            .collect();
        eprintln!("[smoke:ocr] layout: {} regions {kinds:?}", layout.len());
    }

    handle.request_shutdown();
    handle.join().await;
    Ok(())
}
