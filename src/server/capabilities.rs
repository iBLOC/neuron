//! `GET /v1/capabilities/llm` —— 「这枚 chat 模型能不能吃音频」的**可读面**。
//!
//! # 这一面为什么必须存在
//!
//! 「颞叶评估 LLM 能否输入音频」这个裁决（用户原话）⇒ ⇒ 颞叶需要一条
//! **能读到的事实** ⇒ ⇒ 而今天 neuron's 能力面**从不报** `AudioInput`
//! ⇒ ⇒ 颞叶就算想评估，也**读不到任何东西** ⇒ ⇒ 那一格只能是猜的。
//!
//! # 三条纪律
//!
//! ❶ **「判不了」与「不能」是**两件事** —— 远端引擎的上游能不能吃音频，
//! 不是本进程的事实 ⇒ ⇒ 诚实地写 `unknown` ＋ 原因 ⇒ ⇒ 消费方据此
//! 回落（ASR），而不是把「不知道」当成「不支持」或「支持」。
//!
//! ❷ **读面不许为了回答问题而加载权重** —— 「这枚模型能不能吃音频」在
//! 权重没加载时**判不了**（那取决于**旁边那个 mmproj** 的
//! `support_audio()`）⇒ ⇒ 诚实的答案是 `unknown` ⇒ ⇒ 绝不猜一个好看的。
//!
//! ❸ **事实要标出处** —— 每一格都带 `source`（这一格是「运行时事实」
//! 还是「按实现推的」）⇒ ⇒ 读的人知道哪些是确知、哪些是待确认。

use axum::Json;
use axum::extract::State;
use serde::Serialize;
use std::collections::BTreeSet;

use crate::capabilities::Modality;
use crate::capabilities::audio_input::{
    AudioInputState, audio_input_state, engine_impl_audio_state,
};
use crate::server::state::ServerState;

/// 一枚 chat 引擎的**能力事实**。
#[derive(Debug, Clone, Serialize)]
pub struct LlmCapabilityEntry {
    /// 模型 id。
    pub model_id: String,
    /// 引擎实现（candle / litert / mistralrs / llamacpp / remote / mage_vl …）。
    pub engine_impl: String,
    /// 这一格的事实**来自哪儿**（读的人要知道这是不是确切事实）。
    pub source: &'static str,
    /// 输入模态（逐个字面值）。
    pub input_modalities: Vec<String>,
    /// 音频那一格的**三态**。
    pub audio_input: AudioInputFact,
}

/// 音频那一格 —— 「判不了」与「不能」是**两件事**。
#[derive(Debug, Clone, Serialize)]
pub struct AudioInputFact {
    /// `supported` / `unsupported` / `unknown`。
    pub state: String,
    /// 这一格**为什么**是这个状态（判不出来也要说清）。
    pub reason: String,
}

fn modalities_to_strings(m: &BTreeSet<Modality>) -> Vec<String> {
    m.iter().map(|x| x.as_str().to_string()).collect()
}

/// 这一格**为什么**是这个状态。
fn reason_for(state: AudioInputState, engine_impl: &str, is_remote: bool) -> String {
    match state {
        AudioInputState::Supported => {
            "这个引擎自己报了 audio_input ⇒ ⇒ 它真能吃音频（本进程的事实）".to_string()
        }
        AudioInputState::Unsupported => match engine_impl {
            "candle" | "litert" => "纯文本后端 ⇒ ⇒ 确知不能吃音频".to_string(),
            _ => "这个引擎没报 audio_input ⇒ ⇒ 确知不能吃音频".to_string(),
        },
        AudioInputState::Unknown => {
            if is_remote {
                "上游模型能不能吃音频不是本进程的事实 ⇒ ⇒ 判不了".to_string()
            } else if engine_impl == "llamacpp" {
                "能不能吃音频取决于**旁边那个 mmproj** 的 support_audio() ⇒ ⇒ \
                 权重没加载 ⇒ ⇒ 判不了"
                    .to_string()
            } else {
                "这枚模型的模态事实要等权重加载后才知道（读面不许为了回答问题而加载权重）"
                    .to_string()
            }
        }
    }
}

/// **协议层**对五种载荷的口径（顶层那一节 —— 与引擎层**分开**）。
///
/// ⚠️ **两层不许混** —— 这一节说的是「neuron's chat 接口**收得下**哪些
/// 载荷」（收成块／附件，不折 JSON、不静默丢）；逐引擎那一节说的是
/// 「这个引擎**真能吃**哪些」⇒ ⇒ 今天**视频与文档都没有任何 chat 引擎能吃**
/// （llama.cpp 的 mtmd 只解图／音；目录里声明 `DocumentInput` 的只有 OCR 的
/// paddle-ocr ⇒ ⇒ 那是**另一条腿**）⇒ ⇒ 引擎层如实报，协议层照样 accepted。
fn payload_support_section() -> Vec<serde_json::Value> {
    use crate::capabilities::payloads;
    payloads::PAYLOADS
        .iter()
        .map(|p| {
            let state = payloads::protocol_support(p).unwrap_or(payloads::PayloadSupport::Refused);
            serde_json::json!({
                "payload": p,
                "state": state.as_str(),
                "reason": payloads::protocol_reason(p),
                "modality": payloads::payload_modality(p).map(|m| m.as_str()),
            })
        })
        .collect()
}

/// 读面 —— 本进程**此刻确知**的事实。
pub async fn llm_capabilities(State(state): State<ServerState>) -> Json<serde_json::Value> {
    let engines = collect(&state);
    Json(serde_json::json!({
        "object": "llm_capabilities",
        "payload_support": payload_support_section(),
        "engines": engines,
    }))
}

/// 收集两处注册表里的能力事实。
fn collect(state: &ServerState) -> Vec<LlmCapabilityEntry> {
    let mut out: Vec<LlmCapabilityEntry> = Vec::new();

    // ❶ 新式引擎注册表（远端腿 / mage-vl / register_model 注册的那些）
    //    ⇒ ⇒ 它们的 `capabilities()` 就是**运行时事实** ⇒ ⇒ 确切。
    for model_id in state.engines.llm.list() {
        let Some(engine) = state.engines.llm.lookup(&model_id) else {
            continue;
        };
        let caps = engine.capabilities();
        let modalities = caps.input_modalities.clone();
        let is_remote = engine.is_remote();
        let kind = audio_input_state(&modalities, is_remote);
        let engine_impl = if caps.engine_impl.is_empty() {
            engine.name().to_string()
        } else {
            caps.engine_impl.clone()
        };
        out.push(LlmCapabilityEntry {
            model_id: model_id.clone(),
            engine_impl: engine_impl.clone(),
            source: "engine_registry",
            input_modalities: modalities_to_strings(&modalities),
            audio_input: AudioInputFact {
                state: kind.as_str().to_string(),
                reason: reason_for(kind, &engine_impl, is_remote),
            },
        });
    }

    // ❷ 旧式注册表（candle / litert / mistralrs / llamacpp 那一批）
    //    ⇒ ⇒ **已加载**的 ⇒ ⇒ 拿运行时的**确切**能力（不再加载一次）；
    //    ⇒ ⇒ **没加载**的 ⇒ ⇒ 模态事实判不了（要等权重与旁边的 mmproj）。
    let registered = state.llm_registry.registered.lock().unwrap().clone();
    for reg in registered {
        let engine_impl = reg.impl_id.as_str().to_string();
        let (modalities, kind) = match state.llm_registry.loaded_runtime(&reg.model_id) {
            Some(rt) => {
                let m = crate::engine::bridge::input_modalities_of(&*rt);
                let s = audio_input_state(&m, false);
                (m, s)
            }
            None => {
                let s = engine_impl_audio_state(&engine_impl);
                (BTreeSet::from([Modality::TextInput]), s)
            }
        };
        out.push(LlmCapabilityEntry {
            model_id: reg.model_id.clone(),
            engine_impl: engine_impl.clone(),
            source: "model_registry",
            input_modalities: modalities_to_strings(&modalities),
            audio_input: AudioInputFact {
                state: kind.as_str().to_string(),
                reason: reason_for(kind, &engine_impl, false),
            },
        });
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::NeuronConfig;

    fn state() -> ServerState {
        ServerState::new(NeuronConfig::default())
    }

    #[tokio::test]
    async fn test_capabilities_face_has_a_stable_shape() {
        let s = state();
        let Json(v) = llm_capabilities(State(s)).await;
        assert_eq!(
            v.get("object").and_then(|x| x.as_str()),
            Some("llm_capabilities")
        );
        assert!(
            v.get("engines").and_then(|x| x.as_array()).is_some(),
            "engines must be an array even when nothing is registered"
        );
    }

    /// **四种载荷**（文本／图片／音频／视频）必须**逐格**出现在读面里
    /// —— 这是「保证支持这四种载荷」那句话的**可读**形态。
    #[tokio::test]
    async fn test_the_read_face_lists_all_five_payloads() {
        let s = state();
        let Json(v) = llm_capabilities(State(s)).await;
        let payloads = v
            .get("payload_support")
            .and_then(|x| x.as_array())
            .cloned()
            .expect("payload_support must be an array");
        let names: Vec<String> = payloads
            .iter()
            .filter_map(|p| {
                p.get("payload")
                    .and_then(|x| x.as_str())
                    .map(|s| s.to_string())
            })
            .collect();
        for want in ["text", "image", "audio", "video", "document"] {
            assert!(
                names.iter().any(|n| n == want),
                "{want} must be listed in payload_support, got {names:?}"
            );
        }
        for p in &payloads {
            let st = p.get("state").and_then(|x| x.as_str()).unwrap_or("");
            assert!(
                ["accepted", "refused"].contains(&st),
                "payload state must be one of the two states, got {st}"
            );
            let reason = p.get("reason").and_then(|x| x.as_str()).unwrap_or("");
            assert!(!reason.is_empty(), "every payload needs a reason");
            assert!(
                p.get("modality").and_then(|x| x.as_str()).is_some(),
                "every payload names its modality"
            );
        }
    }

    #[tokio::test]
    async fn test_every_entry_carries_an_audio_state_and_a_reason() {
        let s = state();
        let Json(v) = llm_capabilities(State(s)).await;
        let engines = v
            .get("engines")
            .and_then(|x| x.as_array())
            .cloned()
            .unwrap();
        for e in engines {
            let model_id = e.get("model_id").and_then(|x| x.as_str()).unwrap_or("");
            assert!(!model_id.is_empty(), "every entry needs a model id");
            let audio = e.get("audio_input").expect("audio_input");
            let st = audio.get("state").and_then(|x| x.as_str()).unwrap_or("");
            assert!(
                ["supported", "unsupported", "unknown"].contains(&st),
                "audio state must be one of the three states, got {st}"
            );
            let reason = audio.get("reason").and_then(|x| x.as_str()).unwrap_or("");
            assert!(!reason.is_empty(), "『判不了』也要说清为什么（{model_id}）");
            assert!(
                e.get("input_modalities")
                    .and_then(|x| x.as_array())
                    .is_some(),
                "every entry lists its input modalities"
            );
            assert!(
                !e.get("source")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .is_empty(),
                "every entry says where its fact came from"
            );
        }
    }

    #[tokio::test]
    async fn test_an_empty_registry_does_not_fabricate_models() {
        let s = state();
        let Json(v) = llm_capabilities(State(s.clone())).await;
        let engines = v
            .get("engines")
            .and_then(|x| x.as_array())
            .cloned()
            .unwrap();
        // 每一格都必须**对得上**某个注册条目（不许凭空多出模型）。
        let registered: Vec<String> = s.llm_registry.list_registered();
        for e in &engines {
            let model_id = e.get("model_id").and_then(|x| x.as_str()).unwrap_or("");
            let in_engines = s.engines.llm.lookup(model_id).is_some();
            assert!(
                in_engines || registered.iter().any(|r| r == model_id),
                "entry {model_id} is not registered anywhere"
            );
        }
    }
}
