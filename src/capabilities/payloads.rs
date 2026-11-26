//! 五种载荷 —— **文本／图片／音频／视频／文档** —— 的规范名与**协议层**口径。
//!
//! # 两层不许混（这是本模块存在的头号理由）
//!
//! ❶ **协议层**（本模块）—— neuron's chat 接口**收得下**哪些载荷 ⇒ ⇒
//!    每一种都必须**收成块／附件** ⇒ ⇒ **绝不**折成 JSON 混进 prompt，
//!    **绝不**静默丢掉 ⇒ ⇒ 「失败长得像成功」比失败更坏。
//!
//! ❷ **引擎层**（`super::audio_input` 与各运行时的 `capabilities()`）——
//!    某个引擎**真能吃**哪些 ⇒ ⇒ 今天**视频没有任何引擎能吃**
//!    （llama.cpp 的 mtmd 只解图／音），**文档也没有 chat 引擎能吃**
//!    （目录里声明 `DocumentInput` 的只有 OCR 的 paddle-ocr ⇒ ⇒ 那是**另一条
//!    腿**，不是 chat）⇒ ⇒ 引擎层如实报 unsupported，
//!    **协议层照样 accepted** ⇒ ⇒ 两个层次各说各的，**不许互相冒充**。
//!
//! # 为什么判据住在这儿
//!
//! ⚠️ 判据必须住在**默认档能编到的地方**（`llamacpp` 那一档要 libclang
//! ⇒ ⇒ 它的单元测试在默认 build 里根本不编）⇒ ⇒ 本模块无 feature 门。

use super::Modality;

/// 协议层保证收得下的五种载荷（**规范名** —— 别处都从这里取，不许各写各的）。
///
/// ⚠️ **第五种是 `document`（2026-11-25 补上）** —— 上一版把它当「别的
/// 东西」排除在外，**那个判断是错的**：端脑那侧早就在发 `file` part
/// （`{"type":"file","file":{"filename","file_data"}}`）⇒ ⇒ 而且能力面里
/// `pdf` 本来就是**一档模态** ⇒ ⇒ canonical 层也早有 `DocumentBlock` ⇒ ⇒
/// 它是**有生产者、有模态档、只缺消费端**的那一种 ⇒ ⇒ 不许再当它不存在。
///
/// ⚠️ **工具（`tool`）仍不在这里** —— 工具是**另一条线**（协议层的
/// tool_calls／tool 结果），不是**载荷**。
pub const PAYLOADS: [&str; 5] = ["text", "image", "audio", "video", "document"];

/// 一格载荷在**协议层**的状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PayloadSupport {
    /// **收得下**（收成块／附件，不折 JSON、不静默丢）。
    Accepted,
    /// **不收**（明确报错，不折、不丢）。
    Refused,
}

impl PayloadSupport {
    /// 载荷里那一格的字面值。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::Refused => "refused",
        }
    }
}

/// 载荷 → 模态（**唯一**的映射）。
///
/// ⚠️ 不认识的载荷 ⇒ ⇒ `None` ⇒ ⇒ **不猜**（猜错 ⇒ ⇒ 能力面会撒谎）。
pub fn payload_modality(payload: &str) -> Option<Modality> {
    match payload {
        "text" => Some(Modality::TextInput),
        "image" => Some(Modality::ImageInput),
        "audio" => Some(Modality::AudioInput),
        "video" => Some(Modality::VideoInput),
        // ⚠️ **文档 ⇒ DocumentInput** —— 模态枚举**早就有**这一格 ⇒ ⇒
        // 载荷清单此前漏了它（不是模态没有，是清单没跟上）。
        "document" => Some(Modality::DocumentInput),
        _ => None,
    }
}

/// 协议层对这一格载荷的口径（**今天五种都收**）。
pub fn protocol_support(payload: &str) -> Option<PayloadSupport> {
    match payload {
        "text" | "image" | "audio" | "video" | "document" => Some(PayloadSupport::Accepted),
        _ => None,
    }
}

/// 这一格载荷**为什么**是这个状态（读面要说清，不许只给一个字面值）。
pub fn protocol_reason(payload: &str) -> &'static str {
    match payload {
        "text" => "纯文本载荷是 chat 的原生形状",
        "image" => "图片载荷收成 ContentBlock::Image（附件，不折 JSON）",
        "audio" => "音频载荷收成 ContentBlock::Audio（附件，不折 JSON）",
        "video" => {
            "视频载荷收成 ContentBlock::Video（附件，不折 JSON；\
             今天还没有能解它的引擎 ⇒ ⇒ 引擎层如实报 unsupported）"
        }
        "document" => {
            "文档载荷收成 ContentBlock::Document（附件，不折 JSON；\
             线形是 OpenAI 的 file part ⇒ ⇒ 端脑那侧已经在发这个形状）；\
             但今天**没有任何 chat 引擎声明能读文档** ⇒ ⇒ 引擎那条腿如实拒绝"
        }
        _ => "这不是协议层认的载荷",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **五种载荷**：文本／图片／音频／视频／**文档**。
    ///
    /// ⚠️ **文档是第五种**（2026-11-25 补上）—— 上一版把它排除在外，**那个
    /// 判断是错的**：端脑早就在发 `file` part、`pdf` 本来就是一档模态、
    /// canonical 层早有 `DocumentBlock` ⇒ ⇒ 它是**有生产者、有模态档、
    /// 只缺消费端**的那一种。
    #[test]
    fn test_the_five_payloads_are_exactly_text_image_audio_video_document() {
        assert_eq!(PAYLOADS.len(), 5);
        for p in ["text", "image", "audio", "video", "document"] {
            assert!(
                PAYLOADS.contains(&p),
                "{p} must be one of the five payloads"
            );
        }
        // ⚠️ **工具仍不在这里** —— 工具是另一条线（tool_calls），不是载荷。
        assert!(!PAYLOADS.contains(&"tool"));
    }

    #[test]
    fn test_every_payload_maps_to_its_own_modality() {
        assert_eq!(payload_modality("text"), Some(Modality::TextInput));
        assert_eq!(payload_modality("image"), Some(Modality::ImageInput));
        assert_eq!(payload_modality("audio"), Some(Modality::AudioInput));
        assert_eq!(payload_modality("video"), Some(Modality::VideoInput));
        // ⚠️ **文档 ⇒ DocumentInput**（模态枚举早就有这一格）。
        assert_eq!(payload_modality("document"), Some(Modality::DocumentInput));
    }

    #[test]
    fn test_the_five_modalities_are_five_different_facts() {
        let mut seen = std::collections::BTreeSet::new();
        for p in PAYLOADS {
            let m = payload_modality(p).expect("known payload");
            assert!(
                seen.insert(m),
                "{p} must not share a modality with another payload"
            );
        }
        assert_eq!(seen.len(), 5);
    }

    #[test]
    fn test_an_unknown_payload_is_not_guessed() {
        // ⚠ **工具不是载荷** —— 工具是**另一条线**（tool_calls／tool 结果）。
        assert_eq!(payload_modality("tool"), None);
        assert_eq!(protocol_support("tool"), None);
        // ⚠ **`pdf` 是端脑能力面那一侧的**模态词**，不是本仓的载荷规范名**
        // ⇒ ⇒ 两个词汇表不许混 ⇒ ⇒ 载荷规范名只有 `document` 一个。
        assert_eq!(payload_modality("pdf"), None);
        assert_eq!(protocol_support("pdf"), None);
        assert_eq!(payload_modality(""), None);
    }

    #[test]
    fn test_the_protocol_layer_accepts_all_five() {
        for p in PAYLOADS {
            assert_eq!(
                protocol_support(p),
                Some(PayloadSupport::Accepted),
                "{p} must be accepted by the protocol layer"
            );
            assert!(!protocol_reason(p).is_empty(), "{p} needs a reason");
        }
    }

    /// **两层不许混** —— 文档那一格的理由必须**同时**说清两件事：协议层
    /// 收得下（附件，不折 JSON），而引擎层今天**没有**能读文档的 chat 引擎。
    #[test]
    fn the_document_reason_says_the_engine_truth_too() {
        let r = protocol_reason("document");
        assert!(
            r.contains("ContentBlock::Document"),
            "the protocol half: {r}"
        );
        assert!(
            r.contains("引擎"),
            "the reason must not hide the engine truth: {r}"
        );
    }

    #[test]
    fn test_payload_support_as_str() {
        assert_eq!(PayloadSupport::Accepted.as_str(), "accepted");
        assert_eq!(PayloadSupport::Refused.as_str(), "refused");
    }
}
