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
use crate::canonical::{
    CanonicalLlmRequest, CanonicalLlmResponse, CanonicalLlmStreamEvent, ContentBlock,
};
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

    /// ⚠ **声纹是生物特征** —— 开了离场脱敏（`NEURON_REMOTE_REDACT_PII=1`）
    /// 就意味着「PII 不许离开这台机器」⇒ ⇒ 音频**不许**悄悄外发。
    ///
    /// ⇒ ⇒ 这里**明确报错**：静默丢音频会让模型「听不见」却照样答话
    /// （失败长得像成功）。
    fn guard_biometric_media(&self, req: &CanonicalLlmRequest) -> ChatResult<()> {
        if self.privacy.is_none() {
            return Ok(());
        }
        let has_audio = req
            .messages
            .iter()
            .any(|m| m.content.iter().any(|b| matches!(b, ContentBlock::Audio(_))));
        if has_audio {
            return Err(crate::llm::NeuronError::new(
                "audio is biometric data and this engine redacts PII on the way out — \
                 refusing to forward it (turn NEURON_REMOTE_REDACT_PII off to allow \
                 audio forwarding)"
                    .to_string(),
            ));
        }
        Ok(())
    }

}

#[async_trait]
impl crate::engine::LlmRuntime for RemoteLlmEngine {
    async fn execute(
        &self,
        req: CanonicalLlmRequest,
    ) -> ChatResult<CanonicalLlmResponse> {
        self.guard_biometric_media(&req)?;
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
        let wire = canonical_request_to_openai_wire(&req)
            .map_err(|e| crate::llm::NeuronError::new(format!("remote wire: {e}")))?;
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

    /// Multimodal execute — the upstream is OpenAI-compatible, so the
    /// media travels **in the request itself** (image_url / input_audio
    /// parts) ⇒ ⇒ the same forwarding path as `execute`.
    ///
    /// ⚠ **覆写它不是可选的**：the dispatcher hands every request that
    /// carries media to `execute_with_media` ⇒ ⇒ without this override
    /// the trait default answers `not_implemented` ⇒ ⇒ this leg could
    /// never be reached with an image or an audio clip.
    ///
    /// ⚠ The decoded attachments handed over by the dispatcher are NOT
    /// used here: the canonical request carries the very same media, and
    /// using both would feed every attachment to the model twice.
    async fn execute_with_media(
        &self,
        req: CanonicalLlmRequest,
        _media: Vec<crate::canonical::MediaAttachment>,
    ) -> ChatResult<CanonicalLlmResponse> {
        self.execute(req).await
    }

    /// Streaming twin of [`execute_with_media`] — the upstream is
    /// always called non-streaming and the answer is re-emitted as one
    /// delta (same as `execute_stream`).
    async fn execute_with_media_stream(
        &self,
        req: CanonicalLlmRequest,
        _media: Vec<crate::canonical::MediaAttachment>,
    ) -> ChatResult<BoxStream<'static, CanonicalLlmStreamEvent>> {
        self.execute_stream(req).await
    }

    fn capabilities(&self) -> &ModelCapabilities {
        static CAPS: std::sync::OnceLock<ModelCapabilities> = std::sync::OnceLock::new();
        CAPS.get_or_init(|| ModelCapabilities {
            engine_impl: "remote".into(),
            model_id: String::new(),
            // ⚠️ **文本恒在** —— 远端腿是 OpenAI 兼容的 chat 转发 ⇒ ⇒
            // 它**一定**收文本 ⇒ ⇒ 旧版这里报的是**空集合**（连
            // TextInput 都没有）⇒ ⇒ 「什么都收」比「只收文本」更骗人。
            input_modalities: [crate::capabilities::Modality::TextInput]
                .into_iter()
                .collect(),
            // ⚠️ **音频不瞎报** —— 这条腿**能转发**音频（第三刀已开），
            // 但「上游那枚模型能不能吃音频」**不是本进程的事实**
            // ⇒ ⇒ 报上去就成了谎话 ⇒ ⇒ 读面把它记成「判不了」。
            output_modalities: [crate::capabilities::Modality::TextOutput]
                .into_iter()
                .collect(),
            supports_streaming: true,
            supports_tools: true,
            ..Default::default()
        })
    }

    fn name(&self) -> &str {
        "remote"
    }

    /// 远端引擎 ⇒ **不占本机内存** ⇒ ⇒ 模型调度器不该把它当成本地权重
    /// 收编成 resident（否则闲置回收会把它逐出 —— 见 `LlmRuntime::is_remote`）。
    fn is_remote(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // The multimodal entry points are trait methods ⇒ ⇒ the trait must be
    // in scope for `engine.execute_with_media(...)` to resolve.
    use crate::engine::LlmRuntime;

    /// One canonical request carrying a single audio clip ("hello" -> aGVsbG8=).
    fn canonical_request_with_audio() -> CanonicalLlmRequest {
        use crate::canonical::*;
        CanonicalLlmRequest {
            model: "m".into(),
            system: vec![],
            messages: vec![CanonicalMessage {
                role: MessageRole::User,
                content: vec![ContentBlock::Audio(AudioBlock {
                    source: AudioSource::Base64 {
                        media_type: "audio/wav".into(),
                        data: "aGVsbG8=".into(),
                    },
                })],
                tool_calls: vec![],
                tool_call_id: None,
                reasoning_content: None,
            }],
            tools: vec![],
            sampling: Default::default(),
            stop_sequences: vec![],
            stream: false,
            cache_control: None,
            metadata: Default::default(),
            extensions: Default::default(),
        }
    }

    /// 接缝级：**假上游真接住** —— the audio must appear **verbatim** in
    /// the request body the upstream actually received. "It translated
    /// locally" is not a reading.
    #[tokio::test]
    async fn the_remote_leg_forwards_the_audio_part_verbatim() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        let seen = std::sync::Arc::new(std::sync::Mutex::new(None::<String>));
        let seen2 = seen.clone();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept");
            let mut acc: Vec<u8> = Vec::new();
            let mut chunk = [0u8; 8192];
            loop {
                let n = match socket.read(&mut chunk).await {
                    Ok(0) | Err(_) => break,
                    Ok(n) => n,
                };
                acc.extend_from_slice(&chunk[..n]);
                let text = String::from_utf8_lossy(&acc).to_string();
                if let Some(idx) = text.find("\r\n\r\n") {
                    let head = text[..idx].to_ascii_lowercase();
                    let cl = head
                        .lines()
                        .find_map(|l| l.strip_prefix("content-length:"))
                        .and_then(|v| v.trim().parse::<usize>().ok())
                        .unwrap_or(0);
                    if acc.len() >= idx + 4 + cl {
                        break;
                    }
                }
            }
            *seen2.lock().expect("lock") = Some(String::from_utf8_lossy(&acc).to_string());
            let body = r#"{"id":"c1","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"heard it"},"finish_reason":"stop"}]}"#;
            let resp = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = socket.write_all(resp.as_bytes()).await;
            let _ = socket.shutdown().await;
        });

        let engine = RemoteLlmEngine::new(format!("http://{addr}/v1"), "k".into(), "m".into());
        let resp = engine
            .execute_with_media(canonical_request_with_audio(), vec![])
            .await
            .expect("execute_with_media");
        let answer = resp
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text(t) => Some(t.text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");
        assert_eq!(answer, "heard it");

        let raw = seen
            .lock()
            .expect("lock")
            .clone()
            .expect("the upstream must have been called");
        let body = raw.split_once("\r\n\r\n").expect("a request body").1;
        let v: serde_json::Value = serde_json::from_str(body).expect("json body");
        assert_eq!(v["messages"][0]["content"][0]["type"], "input_audio");
        assert_eq!(v["messages"][0]["content"][0]["input_audio"]["data"], "aGVsbG8=");
        assert_eq!(v["messages"][0]["content"][0]["input_audio"]["format"], "wav");
    }

    /// A voiceprint is **biometric data** ⇒ ⇒ a redacting engine must not
    /// forward it ⇒ ⇒ **say so loudly** (silently dropping the audio would
    /// let the model answer without ever hearing anything).
    #[tokio::test]
    async fn the_privacy_filter_refuses_to_forward_audio() {
        let engine = RemoteLlmEngine::new("http://127.0.0.1:1/v1".into(), "k".into(), "m".into())
            .with_privacy();
        let err = engine
            .execute_with_media(canonical_request_with_audio(), vec![])
            .await
            .expect_err("audio must not leave a redacting engine");
        assert!(
            format!("{err:?}").contains("audio"),
            "the error must name the culprit, got {err:?}"
        );
    }

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

        let wire = canonical_request_to_openai_wire(&req).expect("wire");
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
