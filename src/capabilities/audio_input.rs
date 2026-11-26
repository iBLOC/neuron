//! 「LLM 能不能输入音频」—— 一条**可判**的事实。
//!
//! # 为什么单独一个模块（而不是写在 llama.cpp 那条腿里）
//!
//! ⚠️ **判据必须住在默认档能编到的地方** —— neuron's `llamacpp` 档要
//! bindgen ⇒ ⇒ libclang ⇒ ⇒ 那条腿的单元测试在默认 build 里**根本不编**
//! ⇒ ⇒ 判据写在那一档里 ⇒ ⇒ **CI 永远跑不到它** ⇒ ⇒ 「写了判据」≠
//! 「判据会被跑到」。⇒ ⇒ 本模块是**纯函数**（无 feature 门）⇒ ⇒ 判据住这儿。
//!
//! # 三条纪律
//!
//! ❶ **「判不了」不是「没有」** —— 远端引擎的上游能不能吃音频，不是本
//! 进程的事实 ⇒ ⇒ 如实说 `Unknown` ⇒ ⇒ 消费方据此回落（ASR），而不是
//! 假装「不支持」。
//!
//! ❷ **图与音分开判** —— 旧判据只有一个 `supports_media()` ⇒ ⇒ 「有
//! mmproj」被当成「能吃图」⇒ ⇒ **音频专用的 mmproj 会同时被误报成能吃图**。
//!
//! ❸ **「没有 vision」≠「没有模态」** —— 音频专用投影器（如 Qwen2-Audio）
//! 的 `support_vision()` 是 false、`support_audio()` 是 true ⇒ ⇒
//! 旧判据「没有 vision ⇒ 当纯文本」会把它**整个丢掉** ⇒ ⇒ 那不是「不支持」，
//! 是**「能用的东西被误杀」**。

use super::Modality;
use std::collections::BTreeSet;

/// 音频输入的**三态** —— 「判不了」与「不能」是两件事。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioInputState {
    /// **能**吃音频（本进程确知的事实）。
    Supported,
    /// **不能**吃音频（本进程确知的事实）。
    Unsupported,
    /// **判不了** —— 上游的模态不是本进程的事实（远端引擎／未加载的权重）。
    Unknown,
}

impl AudioInputState {
    /// 载荷里那一格的字面值。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Supported => "supported",
            Self::Unsupported => "unsupported",
            Self::Unknown => "unknown",
        }
    }
}

/// **投影器（mmproj）能不能用** —— 图与音**分别**看它自己的 flag。
///
/// ⚠️ 旧判据是「没有 vision ⇒ 当纯文本」⇒ ⇒ **音频专用投影器被误杀**
/// ⇒ ⇒ 这里改成「**图或音**有一项能用就留」。
pub fn mmproj_usable(support_vision: bool, support_audio: bool) -> bool {
    support_vision || support_audio
}

/// 一枚投影器给出的**输入模态**（文本恒在；图/音各自看 flag）。
pub fn mmproj_input_modalities(support_vision: bool, support_audio: bool) -> BTreeSet<Modality> {
    let mut out = BTreeSet::new();
    out.insert(Modality::TextInput);
    if support_vision {
        out.insert(Modality::ImageInput);
    }
    if support_audio {
        out.insert(Modality::AudioInput);
    }
    out
}

/// 一个**后端实现**的输入模态（纯文本后端 ⇒ 只有文本）。
///
/// ⚠️ 「能吃图」是**逐模型**的事（mistralrs 那条腿要看能力目录），
/// 「能吃音」是**逐投影器**的事（llama.cpp 那条腿要看旁边的 mmproj）
/// ⇒ ⇒ 这里只给**逐实现**能确定的那部分（纯文本后端），
/// 其余交给**运行时事实**覆盖。
pub fn backend_input_modalities(supports_image: bool, supports_audio: bool) -> BTreeSet<Modality> {
    let mut out = BTreeSet::new();
    out.insert(Modality::TextInput);
    if supports_image {
        out.insert(Modality::ImageInput);
    }
    if supports_audio {
        out.insert(Modality::AudioInput);
    }
    out
}

/// 从「引擎报了什么」推出**音频三态**。
///
/// - 引擎报了 `AudioInput` ⇒ **能**（本进程的事实）；
/// - 引擎是**远端**的、没报 ⇒ **判不了**（上游的事，本进程不知道）；
/// - 否则 ⇒ **不能**。
pub fn audio_input_state(
    input_modalities: &BTreeSet<Modality>,
    is_remote: bool,
) -> AudioInputState {
    if input_modalities.contains(&Modality::AudioInput) {
        AudioInputState::Supported
    } else if is_remote {
        AudioInputState::Unknown
    } else {
        AudioInputState::Unsupported
    }
}

/// 一枚**引擎实现**在「还没加载权重」这件事上**能确定**的音频状态。
///
/// ⚠️ **纯文本后端**（candle／litert）⇒ ⇒ 确知**不能**吃音频；
/// 其余（llama.cpp 要看旁边的 mmproj／mistral.rs 要看能力目录／
/// remote 是上游的事）⇒ ⇒ 诚实的答案是**判不了**。
pub fn engine_impl_audio_state(engine_impl: &str) -> AudioInputState {
    match engine_impl {
        "candle" | "litert" => AudioInputState::Unsupported,
        _ => AudioInputState::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_an_audio_only_projector_is_not_killed() {
        // ⚠️ 这条是本模块存在的**头号理由**：音频专用投影器没有 vision，
        // 旧判据会把它整个丢掉 ⇒ ⇒ 「能用的东西被误杀」。
        assert!(
            mmproj_usable(false, true),
            "an audio-only mmproj must stay usable"
        );
    }

    #[test]
    fn test_a_projector_with_neither_flag_is_not_usable() {
        assert!(!mmproj_usable(false, false));
    }

    #[test]
    fn test_a_vision_projector_stays_usable() {
        assert!(mmproj_usable(true, false));
    }

    #[test]
    fn test_audio_only_projector_reports_audio_but_not_image() {
        let m = mmproj_input_modalities(false, true);
        assert!(m.contains(&Modality::AudioInput));
        assert!(
            !m.contains(&Modality::ImageInput),
            "an audio-only projector must not be advertised as taking pictures"
        );
        assert!(m.contains(&Modality::TextInput));
    }

    #[test]
    fn test_multimodal_projector_reports_both() {
        let m = mmproj_input_modalities(true, true);
        assert!(m.contains(&Modality::ImageInput));
        assert!(m.contains(&Modality::AudioInput));
    }

    #[test]
    fn test_text_only_projector_reports_text_only() {
        let m = mmproj_input_modalities(false, false);
        assert_eq!(m.len(), 1);
        assert!(m.contains(&Modality::TextInput));
    }

    #[test]
    fn test_remote_without_audio_is_unknown_not_unsupported() {
        // ⚠️ 「判不了」≠「没有」—— 远端引擎的上游能不能吃音频，
        // 本进程不知道 ⇒ ⇒ 诚实的答案是 unknown。
        let m = BTreeSet::from([Modality::TextInput]);
        assert_eq!(
            audio_input_state(&m, true),
            AudioInputState::Unknown,
            "a remote upstream's modality is not this process's fact"
        );
    }

    #[test]
    fn test_local_without_audio_is_unsupported() {
        let m = BTreeSet::from([Modality::TextInput]);
        assert_eq!(audio_input_state(&m, false), AudioInputState::Unsupported);
    }

    #[test]
    fn test_reported_audio_wins_over_remote() {
        let m = BTreeSet::from([Modality::TextInput, Modality::AudioInput]);
        assert_eq!(audio_input_state(&m, true), AudioInputState::Supported);
        assert_eq!(audio_input_state(&m, false), AudioInputState::Supported);
    }

    #[test]
    fn test_text_only_backends_are_certainly_unsupported() {
        assert_eq!(
            engine_impl_audio_state("candle"),
            AudioInputState::Unsupported
        );
        assert_eq!(
            engine_impl_audio_state("litert"),
            AudioInputState::Unsupported
        );
        // 其余 ⇒ ⇒ 判不了（要等运行时事实）。
        assert_eq!(
            engine_impl_audio_state("llamacpp"),
            AudioInputState::Unknown
        );
        assert_eq!(engine_impl_audio_state("remote"), AudioInputState::Unknown);
    }

    #[test]
    fn test_audio_input_state_as_str() {
        assert_eq!(AudioInputState::Supported.as_str(), "supported");
        assert_eq!(AudioInputState::Unsupported.as_str(), "unsupported");
        assert_eq!(AudioInputState::Unknown.as_str(), "unknown");
    }
}
