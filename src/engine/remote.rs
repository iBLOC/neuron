//! Remote-forwarding LLM engine — serves a model id by proxying the
//! canonical request to an OpenAI-compatible endpoint and mapping the
//! response back.
//!
//! Why: neuron's local runtimes (candle / litert / mistralrs) serve
//! weights from disk; the stack behind them had no way to expose a
//! cloud model, which made benchmarking the production stack (端脑 +
//! 丘脑 + 海马体 + neuron) against cloud-quality models impossible.
//! This engine closes that gap: the stack keeps its OpenAI face and
//! tool-calling round-trips verbatim (OpenAI wire on both sides).
//!
//! Config (read in `server::state::ServerState::new`):
//! * `NEURON_REMOTE_BASE_URL` — e.g. `https://api.deepseek.com/v1`
//! * `NEURON_REMOTE_API_KEY`  — bearer token
//! * `NEURON_REMOTE_MODELS`   — comma-separated model ids to register
//! * `NEURON_REMOTE_REDACT_PII` — `1`/`true` to pseudonymize outbound
//!   PII (and restore it on the way back) before the request leaves
//!   the device. Off by default.
//!
//! Streaming: upstream is always called non-streaming; `execute_stream`
//! re-emits the full response as a single delta chunk. The benchmark
//! adapters and the OpenAI face both work fine over that shape.

use async_trait::async_trait;
use futures_util::stream::BoxStream;
use serde_json::Value;
use std::time::Duration;

use crate::adapter::openai_chat::{
    canonical_request_to_openai_wire, openai_wire_response_to_canonical,
};
use crate::canonical::{CanonicalLlmRequest, CanonicalLlmResponse, CanonicalLlmStreamEvent};
use crate::engine::privacy::PrivacyFilter;
use crate::llm::ChatResult;
use crate::capabilities::ModelCapabilities;

pub struct RemoteLlmEngine {
    /// OpenAI-compatible base URL INCLUDING `/v1`.
    base_url: String,
    api_key: String,
    /// Upstream model id — kept distinct from the locally registered
    /// alias so the stack can expose `deepseek-chat` while proxying to
    /// whatever the remote actually names it.
    model: String,
    client: reqwest::Client,
    /// Outbound PII pseudonymization (None = disabled).
    privacy: Option<PrivacyFilter>,
}

impl RemoteLlmEngine {
    pub fn new(base_url: String, api_key: String, model: String) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            model,
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(600))
                .build()
                .unwrap_or_default(),
            privacy: None,
        }
    }

    /// Enable outbound PII pseudonymization for this engine.
    pub fn with_privacy(mut self) -> Self {
        self.privacy = Some(PrivacyFilter::new());
        self
    }
}

#[async_trait]
impl crate::engine::LlmRuntime for RemoteLlmEngine {
    async fn execute(
        &self,
        req: CanonicalLlmRequest,
    ) -> ChatResult<CanonicalLlmResponse> {
        // Redact on the way out, restore on the way back — the caller
        // never sees a pseudonym.
        let (req, restore) = match &self.privacy {
            Some(filter) => {
                let mut req = req;
                filter.redact_request(&mut req);
                (req, Some(filter))
            }
            None => (req, None),
        };
        let wire = canonical_request_to_openai_wire(&req);
        let url = format!("{}/chat/completions", self.base_url);
        let mut r = self.client.post(&url).json(&wire);
        if !self.api_key.is_empty() {
            r = r.bearer_auth(&self.api_key);
        }
        let resp = r
            .send()
            .await
            .map_err(|e| crate::llm::NeuronError::new(format!("remote request: {e}")))?;
        let status = resp.status();
        let body: Value = resp
            .json()
            .await
            .map_err(|e| crate::llm::NeuronError::new(format!("remote body at {status}: {e}")))?;
        if !status.is_success() {
            let msg = body["error"]["message"]
                .as_str()
                .unwrap_or("unknown remote error");
            return Err(crate::llm::NeuronError::new(format!(
                "remote HTTP {status}: {msg}"
            )));
        }
        let mut canonical =
            openai_wire_response_to_canonical(&body, &req.model).map_err(crate::llm::NeuronError::new)?;
        if let Some(filter) = restore {
            filter.restore_response(&mut canonical);
        }
        Ok(canonical)
    }

    async fn execute_stream(
        &self,
        req: CanonicalLlmRequest,
    ) -> ChatResult<BoxStream<'static, CanonicalLlmStreamEvent>> {
        let resp = self.execute(req).await?;
        let model = resp.model.clone();
        let stop = Some(resp.stop_reason.clone());
        let text: String = resp
            .content
            .iter()
            .filter_map(|b| match b {
                crate::canonical::ContentBlock::Text(t) => Some(t.text.clone()),
                _ => None,
            })
            .collect();
        let usage = Some(resp.usage.clone());
        let stream = async_stream::stream! {
            if !text.is_empty() {
                yield CanonicalLlmStreamEvent::ContentBlockDelta {
                    index: 0,
                    delta: crate::canonical::ContentDelta::TextDelta(text),
                };
            }
            yield CanonicalLlmStreamEvent::MessageDelta { stop_reason: stop, usage };
            yield CanonicalLlmStreamEvent::MessageStop;
            let _ = model;
        };
        Ok(Box::pin(stream))
    }

    fn capabilities(&self) -> &ModelCapabilities {
        static CAPS: std::sync::OnceLock<ModelCapabilities> = std::sync::OnceLock::new();
        CAPS.get_or_init(|| ModelCapabilities {
            engine_impl: "remote".into(),
            model_id: String::new(),
            supports_streaming: true,
            supports_tools: true,
            ..Default::default()
        })
    }

    fn name(&self) -> &str {
        "remote"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The forwarder must round-trip a full agentic conversation:
    /// system + user + assistant(tool_calls) + tool(result) + tools,
    /// producing OpenAI wire that a remote would accept verbatim.
    #[test]
    fn canonical_request_wire_round_trips_tools() {
        use crate::canonical::*;
        let req = CanonicalLlmRequest {
            model: "deepseek-chat".into(),
            system: vec![SystemBlock { text: "be brief".into(), cache_control: None }],
            messages: vec![
                CanonicalMessage {
                    role: MessageRole::User,
                    content: vec![ContentBlock::Text(TextBlock { text: "list files".into() })],
                    tool_calls: vec![],
                    tool_call_id: None,
                    reasoning_content: None,
                },
                CanonicalMessage {
                    role: MessageRole::Assistant,
                    content: vec![ContentBlock::Text(TextBlock { text: String::new() })],
                    tool_calls: vec![ToolCall {
                        id: "call_1".into(),
                        name: "bash".into(),
                        arguments: serde_json::json!({"command": "ls"}),
                    }],
                    tool_call_id: None,
                    reasoning_content: None,
                },
                CanonicalMessage {
                    role: MessageRole::Tool,
                    content: vec![ContentBlock::Text(TextBlock { text: "a.txt\nb.txt".into() })],
                    tool_calls: vec![],
                    tool_call_id: Some("call_1".into()),
                    reasoning_content: None,
                },
            ],
            tools: vec![ToolDefinition {
                name: "bash".into(),
                description: Some("run a command".into()),
                input_schema: serde_json::json!({"type": "object"}),
            }],
            sampling: Default::default(),
            stop_sequences: vec![],
            stream: false,
            cache_control: None,
            metadata: Default::default(),
            extensions: Default::default(),
        };

        let wire = canonical_request_to_openai_wire(&req);
        assert_eq!(wire["messages"][0]["role"], "system");
        // tool_calls message: arguments serialized as a JSON string
        assert_eq!(wire["messages"][2]["tool_calls"][0]["id"], "call_1");
        assert_eq!(
            wire["messages"][2]["tool_calls"][0]["function"]["name"],
            "bash"
        );
        assert!(wire["messages"][2]["tool_calls"][0]["function"]["arguments"]
            .as_str()
            .unwrap()
            .contains("\"command\""));
        // tool result: role tool + tool_call_id, content is the text
        assert_eq!(wire["messages"][3]["role"], "tool");
        assert_eq!(wire["messages"][3]["tool_call_id"], "call_1");
        assert_eq!(wire["messages"][3]["content"], "a.txt\nb.txt");
        // tools flatten to the OpenAI function envelope
        assert_eq!(wire["tools"][0]["type"], "function");
        assert_eq!(wire["tools"][0]["function"]["name"], "bash");
        assert_eq!(
            wire["tools"][0]["function"]["parameters"]["type"],
            "object"
        );
    }

    /// A remote response carrying tool_calls parses back into ToolUse
    /// content blocks with StopReason::ToolUse — the shape the agentic
    /// benchmark loop needs.
    #[test]
    fn remote_wire_response_parses_tool_calls() {
        let body = serde_json::json!({
            "id": "cmpl-1",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_9",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": "{\"path\": \"a.txt\", \"content\": \"hi\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        });
        let canonical = openai_wire_response_to_canonical(&body, "deepseek-chat").unwrap();
        assert!(
            matches!(canonical.stop_reason, crate::canonical::StopReason::ToolUse),
            "expected StopReason::ToolUse, got {:?}",
            canonical.stop_reason
        );
        assert_eq!(canonical.content.len(), 1);
        match &canonical.content[0] {
            crate::canonical::ContentBlock::ToolUse(t) => {
                assert_eq!(t.id, "call_9");
                assert_eq!(t.name, "write_file");
                assert_eq!(t.input["path"], "a.txt");
            }
            other => panic!("expected ToolUse, got {other:?}"),
        }
        assert_eq!(canonical.usage.input_tokens, 10);
    }
}
