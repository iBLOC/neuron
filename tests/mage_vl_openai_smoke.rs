//! Adapter-level smoke: confirms the OpenAI Chat Completions wire
//! shape carries an inline `image_url` content block into a
//! canonical `ContentBlock::Image`. Always runs — no weights
//! needed. Separate from the gated full-wire test to keep that
//! binary focused on the wire-shape contract.

#![cfg(feature = "candle")]

const RED_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEElEQVR4nGP4z8AARAwQCgAf7gP9i18U1AAAAABJRU5ErkJggg==";

fn multimodal_body() -> serde_json::Value {
    serde_json::json!({
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
    })
}

#[test]
fn openai_parse_carries_image_into_canonical_request() {
    use neure_lib::adapter::ApiAdapter;
    let body = serde_json::to_vec(&multimodal_body()).unwrap();
    let headers = axum::http::HeaderMap::new();
    let canonical = neure_lib::adapter::openai_chat::OpenAiChatAdapter
        .parse(&body.into(), &headers)
        .expect("parse");
    let neure_lib::canonical::CanonicalRequest::Llm(llm) = canonical else {
        panic!("expected Llm canonical request");
    };
    let has_image = llm.messages.iter().any(|m| {
        m.content.iter().any(|b| {
            matches!(
                b,
                neure_lib::canonical::ContentBlock::Image(neure_lib::canonical::ImageBlock {
                    source: neure_lib::canonical::ImageSource::Base64 { .. },
                    ..
                })
            )
        })
    });
    assert!(has_image, "data-URL image must survive adapter parsing");
}
