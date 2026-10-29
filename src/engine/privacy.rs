//! Outbound PII redaction for the remote (cloud) forwarder.
//!
//! The remote engine is the desktop's real external-model path: the
//! stack keeps its OpenAI face, but the bytes on the wire go to a
//! cloud provider. The mechanism lives in `synapse-privacy` (shared
//! with synapse / prefrontal so every path redacts identically); this
//! module adapts it to neuron's canonical request/response shapes.
//!
//! Contract, matching the prefrontal wiring:
//!
//! - Outbound: system text, message text blocks, tool-call arguments
//!   and tool-result text are pseudonymized before the wire format is
//!   built.
//! - Inbound: the canonical response is restored (text blocks and
//!   tool-use inputs) before it re-enters the stack, so the caller
//!   never sees a placeholder.
//!
//! The pseudonym map is per-engine, in memory only — never persisted.

use serde_json::Value;
use synapse_privacy::RedactionEngine;

use crate::canonical::{
    CanonicalLlmRequest, CanonicalLlmResponse, CanonicalMessage, ContentBlock, SystemBlock,
};

/// Redacts a canonical request in place and restores a canonical
/// response in place. Holds the pseudonym map for the engine's
/// lifetime, so the same raw value keeps the same token across calls
/// (agent loops resend full history every turn).
pub struct PrivacyFilter {
    engine: RedactionEngine,
}

impl PrivacyFilter {
    pub fn new() -> Self {
        Self {
            engine: RedactionEngine::with_regex_detector(),
        }
    }

    /// Number of distinct raw values currently mapped — a growth guard
    /// for long-lived engines.
    pub fn mapped_values(&self) -> usize {
        self.engine.mapped_values()
    }

    pub fn redact_request(&self, req: &mut CanonicalLlmRequest) {
        for block in req.system.iter_mut() {
            redact_system(block, &self.engine);
        }
        for msg in req.messages.iter_mut() {
            redact_message(msg, &self.engine);
        }
    }

    pub fn restore_response(&self, resp: &mut CanonicalLlmResponse) {
        for block in resp.content.iter_mut() {
            restore_block(block, &self.engine);
        }
    }
}

impl Default for PrivacyFilter {
    fn default() -> Self {
        Self::new()
    }
}

fn redact_system(block: &mut SystemBlock, engine: &RedactionEngine) {
    block.text = engine.redact(&block.text).text;
}

fn redact_message(msg: &mut CanonicalMessage, engine: &RedactionEngine) {
    for block in msg.content.iter_mut() {
        redact_block(block, engine);
    }
    for call in msg.tool_calls.iter_mut() {
        call.arguments = redact_json(&call.arguments, engine);
    }
    if let Some(reasoning) = msg.reasoning_content.as_mut() {
        *reasoning = engine.redact(reasoning).text;
    }
}

fn redact_block(block: &mut ContentBlock, engine: &RedactionEngine) {
    match block {
        ContentBlock::Text(t) => t.text = engine.redact(&t.text).text,
        ContentBlock::ToolUse(t) => t.input = redact_json(&t.input, engine),
        ContentBlock::ToolResult(t) => {
            for inner in t.content.iter_mut() {
                redact_block(inner, engine);
            }
        }
        ContentBlock::Reasoning(r) => r.text = engine.redact(&r.text).text,
        _ => {}
    }
}

fn restore_block(block: &mut ContentBlock, engine: &RedactionEngine) {
    match block {
        ContentBlock::Text(t) => t.text = engine.restore(&t.text),
        ContentBlock::ToolUse(t) => t.input = restore_json(&t.input, engine),
        ContentBlock::Reasoning(r) => r.text = engine.restore(&r.text),
        _ => {}
    }
}

/// Redact every string leaf of a JSON value. Keys are left alone —
/// they are schema, not data.
fn redact_json(value: &Value, engine: &RedactionEngine) -> Value {
    match value {
        Value::String(s) => Value::String(engine.redact(s).text),
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|v| redact_json(v, engine))
                .collect::<Vec<_>>(),
        ),
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(k, v)| (k.clone(), redact_json(v, engine)))
                .collect(),
        ),
        other => other.clone(),
    }
}

fn restore_json(value: &Value, engine: &RedactionEngine) -> Value {
    match value {
        Value::String(s) => Value::String(engine.restore(s)),
        Value::Array(items) => {
            Value::Array(items.iter().map(|v| restore_json(v, engine)).collect::<Vec<_>>())
        }
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(k, v)| (k.clone(), restore_json(v, engine)))
                .collect(),
        ),
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical::{
        ContentBlock, MessageRole, TextBlock, ToolCall, ToolUseBlock,
    };

    fn text_msg(text: &str) -> CanonicalMessage {
        CanonicalMessage {
            role: MessageRole::User,
            content: vec![ContentBlock::Text(TextBlock {
                text: text.to_string(),
            })],
            tool_calls: vec![],
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    fn request(messages: Vec<CanonicalMessage>) -> CanonicalLlmRequest {
        CanonicalLlmRequest {
            model: "deepseek-chat".into(),
            system: vec![SystemBlock {
                text: "你是助手，客户电话 13800138000".into(),
                cache_control: None,
            }],
            messages,
            tools: vec![],
            sampling: Default::default(),
            stop_sequences: vec![],
            stream: false,
            cache_control: None,
            metadata: Default::default(),
            extensions: Default::default(),
        }
    }

    #[test]
    fn request_text_and_system_redacted() {
        let filter = PrivacyFilter::new();
        let mut req = request(vec![text_msg("请回拨 13900139000")]);
        filter.redact_request(&mut req);

        assert!(req.system[0].text.contains("[PHONE_1]"), "{:?}", req.system[0]);
        assert!(!req.system[0].text.contains("13800138000"));

        let ContentBlock::Text(t) = &req.messages[0].content[0] else {
            panic!("expected text block");
        };
        assert_eq!(t.text, "请回拨 [PHONE_2]");
        assert!(!t.text.contains("13900139000"));
    }

    #[test]
    fn tool_call_arguments_redacted_and_restored() {
        let filter = PrivacyFilter::new();
        let mut msg = text_msg("打给 13800138000");
        msg.tool_calls = vec![ToolCall {
            id: "c1".into(),
            name: "dial".into(),
            arguments: serde_json::json!({ "to": "13800138000", "note": "客户" }),
        }];
        let mut req = request(vec![msg]);
        filter.redact_request(&mut req);

        let args = &req.messages[0].tool_calls[0].arguments;
        assert!(!args["to"].as_str().unwrap().contains("13800138000"));
        assert!(args["to"].as_str().unwrap().contains("[PHONE_"));

        // Response restore must bring the value back.
        let mut resp = CanonicalLlmResponse {
            id: "r1".into(),
            model: "deepseek-chat".into(),
            stop_reason: crate::canonical::StopReason::ToolUse,
            content: vec![ContentBlock::ToolUse(ToolUseBlock {
                id: "c1".into(),
                name: "dial".into(),
                input: serde_json::json!({ "to": "[PHONE_1]" }),
            })],
            usage: Default::default(),
            extensions: Default::default(),
        };
        filter.restore_response(&mut resp);
        let ContentBlock::ToolUse(t) = &resp.content[0] else {
            panic!("expected tool use block");
        };
        assert_eq!(t.input["to"], "13800138000");
    }

    #[test]
    fn clean_request_untouched() {
        let filter = PrivacyFilter::new();
        let mut req = request(vec![text_msg("今天天气如何")]);
        // Clean system prompt too — nothing here should be mapped.
        req.system[0].text = "你是助手".into();
        let before = match &req.messages[0].content[0] {
            ContentBlock::Text(t) => t.text.clone(),
            other => panic!("expected text block, got {other:?}"),
        };
        filter.redact_request(&mut req);
        let after = match &req.messages[0].content[0] {
            ContentBlock::Text(t) => t.text.clone(),
            other => panic!("expected text block, got {other:?}"),
        };
        assert_eq!(after, before);
        assert_eq!(req.system[0].text, "你是助手");
        assert_eq!(filter.mapped_values(), 0);
    }

    #[test]
    fn tool_result_nested_content_redacted() {
        let filter = PrivacyFilter::new();
        let mut msg = text_msg("查一下");
        msg.content = vec![ContentBlock::ToolResult(crate::canonical::ToolResultBlock {
            tool_use_id: "c1".into(),
            content: vec![ContentBlock::Text(TextBlock {
                text: "结果 13800138000".into(),
            })],
            is_error: None,
        })];
        let mut req = request(vec![msg]);
        filter.redact_request(&mut req);

        let ContentBlock::ToolResult(t) = &req.messages[0].content[0] else {
            panic!("expected tool result block");
        };
        let ContentBlock::Text(inner) = &t.content[0] else {
            panic!("expected nested text");
        };
        assert!(inner.text.contains("[PHONE_1]"), "{}", inner.text);
    }
}
