//! Adapter-level smoke for the Anthropic Messages wire format —
//! runs unconditionally, no weights on disk.

#![cfg(feature = "candle")]

const RED_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEElEQVR4nGP4z8AARAwQCgAf7gP9i18U1AAAAABJRU5ErkJggg==";

fn anthropic_multimodal_body() -> serde_json::Value {
    serde_json::json!({
        "model": "microsoft/Mage-VL",
        "max_tokens": 8,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": RED_PNG_B64
                }},
                {"type": "text", "text": "What color is this image?"}
            ]
        }]
    })
}

/// Confirms the Anthropic Messages wire carries an inline `image`
/// content block into a canonical `ContentBlock::Image`, and that
/// source ordering (image before text) is preserved.
#[test]
fn anthropic_parse_carries_image_into_canonical_request() {
    use neuron_lib::adapter::ApiAdapter;
    let body = serde_json::to_vec(&anthropic_multimodal_body()).unwrap();
    let headers = axum::http::HeaderMap::new();
    let canonical = neuron_lib::adapter::anthropic_messages::AnthropicMessagesAdapter
        .parse(&body.into(), &headers)
        .expect("parse");
    let neuron_lib::canonical::CanonicalRequest::Llm(llm) = canonical else {
        panic!("expected Llm canonical request");
    };
    let user = llm
        .messages
        .iter()
        .find(|m| m.content.len() > 1)
        .expect("anthropic multimodal message should have 2+ content blocks");
    let has_image = user.content.iter().any(|b| {
        matches!(
            b,
            neuron_lib::canonical::ContentBlock::Image(neuron_lib::canonical::ImageBlock {
                source: neuron_lib::canonical::ImageSource::Base64 { .. },
                ..
            })
        )
    });
    assert!(has_image, "Anthropic inline image must survive adapter parsing");
    // Anthropic sends image + text in source order; the converter
    // preserves that ordering into the canonical `content` vector.
    let image_first = user
        .content
        .first()
        .map(|b| matches!(b, neuron_lib::canonical::ContentBlock::Image(_)))
        .unwrap_or(false);
    assert!(
        image_first,
        "Anthropic ordering must put the image before the text"
    );
}
