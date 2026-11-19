//! Real-weights smoke for Tencent WeVisOne-VL on neuron's mistralrs leg.
//!
//! ONE question, REAL checkpoint weights: can neuron's mistralrs leg carry
//! an image into a Qwen3-VL backbone and get the model's structured
//! detection output back?
//!
//! The unit tests in `mistralrs_runtime.rs` pin the *wiring* (marker ->
//! attachment pairing, decode failures, the bridge forwarding) without
//! weights. They cannot answer "does the model id resolve, does the repo
//! layout fit, does the chat template place the image where the model
//! expects it" — only a real run can.
//!
//! By default the picture is synthesized in-process (white canvas, one red
//! square, one green disc) so the run has no external asset dependency and
//! the answer is checkable by eye.
//!
//! Environment:
//!
//! * `NEURON_WEVISONE_DIR` — local snapshot of tencent/WeVisOne-VL-2B
//!   (mistral.rs loads a local directory directly — no HF cache needed).
//! * `NEURON_WEVISONE_IMAGE` — ask about this local image instead of the
//!   synthetic scene (the official cookbook's own picture can be used).
//! * `NEURON_WEVISONE_PROMPT` — the task prompt, in the checkpoint's own
//!   family (`Tencent/WeVisOne-VL` `wevisone_vl/prompts.py`).
//! * `NEURON_WEVISONE_MAX_TOKENS` — generation cap (default 512).
//!
//! Run:
//!
//! ```bash
//! cargo run --release --example wevisone_vl_smoke --features mistralrs
//! ```

use neuron_lib::canonical::{MediaAttachment, mtmd_marker};
use neuron_lib::config::DeviceSelection;
use neuron_lib::llm::{ChatMessage, ChatRequest};

/// A deterministic test scene: white canvas, a red square, a green disc.
fn scene() -> Vec<u8> {
    let (w, h) = (640u32, 480u32);
    let mut img = image::RgbImage::from_pixel(w, h, image::Rgb([255, 255, 255]));

    // Red square: x 200..360, y 220..340
    for y in 220..340 {
        for x in 200..360 {
            img.put_pixel(x, y, image::Rgb([220, 20, 20]));
        }
    }
    // Green disc: centre (480, 120), radius 70
    let (cx, cy, r) = (480i32, 120i32, 70i32);
    for y in 40..200 {
        for x in 410..550 {
            let dx = x as i32 - cx;
            let dy = y as i32 - cy;
            if dx * dx + dy * dy <= r * r {
                img.put_pixel(x, y, image::Rgb([20, 160, 60]));
            }
        }
    }

    let mut buf = Vec::new();
    image::DynamicImage::ImageRgb8(img)
        .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
        .expect("encode png");
    buf
}

fn default_dir() -> String {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_default();
    format!("{home}/.neuron/models/llm/tencent/WeVisOne-VL-2B")
}

/// The picture to ask about, and its mime type.
fn picture() -> (Vec<u8>, String) {
    if let Ok(path) = std::env::var("NEURON_WEVISONE_IMAGE") {
        let bytes =
            std::fs::read(&path).unwrap_or_else(|e| panic!("reading {path} failed: {e}"));
        let mime = match path.to_ascii_lowercase().rsplit('.').next().unwrap_or("") {
            "jpg" | "jpeg" => "image/jpeg",
            "webp" => "image/webp",
            "bmp" => "image/bmp",
            _ => "image/png",
        };
        println!("image     : {path} ({} bytes)", bytes.len());
        return (bytes, mime.to_string());
    }
    let png = scene();
    println!("image     : {} bytes of synthetic PNG", png.len());
    (png, "image/png".to_string())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::var("NEURON_WEVISONE_DIR").unwrap_or_else(|_| default_dir());
    println!("model dir : {dir}");
    if !std::path::Path::new(&dir).join("config.json").exists() {
        return Err(
            format!("no config.json under {dir} — point NEURON_WEVISONE_DIR at a local snapshot")
                .into(),
        );
    }

    let (bytes, mime) = picture();

    let t0 = std::time::Instant::now();
    let runtime = neuron_lib::llm::MistralRsLlmRuntime::load(&dir, &DeviceSelection::Auto).await?;
    println!("loaded    : {:?} (weights + vision tower)", t0.elapsed());

    // The prompt family is the checkpoint's own (Tencent/WeVisOne-VL
    // `wevisone_vl/prompts.py`): detection asks for the labels to locate,
    // and the model answers with
    // `<ref>label</ref><box><x1><y1><x2><y2></box>` in a 0..1000 space.
    //
    // Ground truth of the synthetic scene (640x480):
    //   red square   x 200..360, y 220..340 -> <312><458><562><708>
    //   green disc   x 410..550, y  50..190 -> <640><104><859><396>
    let task = std::env::var("NEURON_WEVISONE_PROMPT").unwrap_or_else(|_| {
        "Locate all the instances that match the following description: red square. green circle."
            .to_string()
    });
    let prompt = format!("{task} {}", mtmd_marker());
    let max_tokens: u32 = std::env::var("NEURON_WEVISONE_MAX_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);

    let req = ChatRequest {
        model: dir.clone(),
        messages: vec![
            // The official engine sends this exact system prompt.
            ChatMessage {
                role: "system".into(),
                content: "You are a helpful assistant.".into(),
                reasoning_content: None,
            },
            ChatMessage {
                role: "user".into(),
                content: prompt,
                reasoning_content: None,
            },
        ],
        temperature: Some(0.0),
        max_tokens: Some(max_tokens),
        top_p: None,
        top_k: None,
        stream: false,
        stop: None,
    };
    let media = vec![MediaAttachment {
        bytes,
        mime_type: mime,
        detail: Some("high".into()),
    }];

    let t1 = std::time::Instant::now();
    let resp = runtime.chat_with_media(req, media).await?;
    println!("answered  : {:?}", t1.elapsed());
    println!("=== response ===");
    println!("{}", resp.choices[0].message.content);
    if let Some(usage) = resp.usage {
        println!(
            "=== usage === prompt={} completion={} total={}",
            usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
        );
    }
    Ok(())
}
