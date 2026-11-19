//! Task and prompt contracts for Ming-3.0-tts (dense 0.5B).
//!
//! Line-level port of upstream `tts_tasks.py`. The strings and JSON key
//! order in this module are part of the model input protocol: the released
//! checkpoint was trained against these exact captions. Build requests
//! through [`build_task_generation_kwargs`] instead of re-assembling
//! prompts by hand.

/// The user-side prompt; the target language line is opt-in.
pub const DEFAULT_TTS_PROMPT: &str = "Please generate speech based on the following description.\n";

/// Language IDs accepted by `language_id`. Short codes and full names both
/// resolve to the name used in the `Target language:` line.
const TARGET_LANGUAGE_NAMES: &[(&str, &str)] = &[
    ("zh", "Chinese"),
    ("zh-cn", "Chinese"),
    ("chinese", "Chinese"),
    ("en", "English"),
    ("english", "English"),
    ("ja", "Japanese"),
    ("japanese", "Japanese"),
    ("ko", "Korean"),
    ("korean", "Korean"),
    ("de", "German"),
    ("german", "German"),
    ("fr", "French"),
    ("french", "French"),
    ("es", "Spanish"),
    ("spanish", "Spanish"),
    // Cantonese is a Chinese dialect, not a separate target language.
    ("yue", "Chinese"),
    ("zh-yue", "Chinese"),
    ("cantonese", "Chinese"),
];

/// The five languages added in Ming-3.0 on top of Chinese/English. These
/// require an explicit `language_id`.
pub const MULTILINGUAL_LANGUAGE_IDS: &[(&str, &str)] = &[
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("de", "German"),
    ("fr", "French"),
    ("es", "Spanish"),
];

pub const BASIC_ATTRIBUTE_SPEED: [&str; 3] = ["慢速", "中速", "快速"];
pub const BASIC_ATTRIBUTE_PITCH: [&str; 3] = ["低", "中", "高"];
pub const BASIC_ATTRIBUTE_VOLUME: [&str; 3] = ["低", "中", "高"];

pub const EMOTION_VALUES: [&str; 7] = ["高兴", "悲伤", "愤怒", "恐惧", "厌恶", "惊讶", "中性"];

pub const DIALECT_VALUES: [&str; 2] = ["chuanyu dialect", "cantonese"];

/// Recommended English built-in voices. Chinese IP voices are listed in
/// the upstream cookbooks.
pub const EN_IP_VOICES: [&str; 3] = ["US_Emma", "UK_George", "UK_Charlotte"];

const ZH_FIELDS: [&str; 8] = ["序号", "说话人", "方言", "风格", "语速", "基频", "音量", "情感"];
const EN_FIELDS: [&str; 8] = ["id", "speaker", "dialect", "style", "speed", "pitch", "volume", "emotion"];
const BGM_FIELDS: [&str; 6] = ["Genre", "Mood", "Instrument", "Theme", "ENV", "SNR"];

/// Task aliases: request name → runtime task.
///
/// `TTS` / `SEED` / `PLAIN` / `MULTILINGUAL` all run the plain pipeline;
/// `VOICE_DESIGN` / `STYLE` / `FREEFORM` share the `STYLE` runtime task.
pub const TASK_ALIASES: &[(&str, &str)] = &[
    ("TTS", "TTS"),
    ("SEED", "TTS"),
    ("PLAIN", "TTS"),
    ("MULTILINGUAL", "TTS"),
    ("MULTILINGUAL_TTS", "TTS"),
    ("BASIC", "BASIC"),
    ("EMOTION", "EMOTION"),
    ("DIALECT", "DIALECT"),
    ("IP", "IP"),
    ("VOICE_DESIGN", "STYLE"),
    ("STYLE", "STYLE"),
    ("FREEFORM", "STYLE"),
];

const STRUCTURED_TASKS: [&str; 6] = ["BASIC", "EMOTION", "DIALECT", "IP", "VOICE_DESIGN", "STYLE"];

/// The controls a task accepts; anything else is refused before it can
/// leak into the caption.
const ALLOWED_CONTROLS: &[(&str, &[&str])] = &[
    ("BASIC", &["speed", "pitch", "volume"]),
    ("DIALECT", &["dialect"]),
    ("EMOTION", &["emotion"]),
    ("VOICE_DESIGN", &["style"]),
    ("STYLE", &["style"]),
    ("IP", &["ip"]),
];

fn target_language_name(language: &str) -> &str {
    let lowered = language.trim().to_ascii_lowercase();
    TARGET_LANGUAGE_NAMES
        .iter()
        .find(|(code, _)| *code == lowered)
        .map(|(_, name)| *name)
        .unwrap_or(language)
}

/// True when target text must bypass Chinese/English text normalization.
///
/// The normalization frontend treats any CJK ideograph as Chinese, so
/// Japanese containing kanji would lose its kana. Explicit non-zh/en
/// languages are therefore passed through unchanged.
pub fn preserve_multilingual_text(language_id: Option<&str>) -> bool {
    match language_id {
        None => false,
        Some(language) => {
            let lowered = language.trim().to_ascii_lowercase().replace('_', "-");
            !matches!(lowered.as_str(), "zh" | "zh-cn" | "chinese" | "en" | "english")
        }
    }
}

/// Serialize a JSON string the way Python's `json.dumps(...,
/// ensure_ascii=False)` does — double quotes, backslash and control
/// characters escaped, everything else (CJK included) left raw.
///
/// The caption string is model input; serde_json's compact separators
/// would produce a *different* string than the trained captions, so the
/// formatting here intentionally matches Python's defaults (`", "` between
/// items, `": "` after keys).
fn py_json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Build the ordered assistant-side caption consumed during SFT.
///
/// The field order is trained input — do not sort or reorder it.
pub fn build_structured_instruction(
    dialect: Option<&str>,
    style: Option<&str>,
    speed: Option<&str>,
    pitch: Option<&str>,
    volume: Option<&str>,
    emotion: Option<&str>,
    ip: Option<&str>,
    schema: &str,
) -> Result<String, String> {
    let schema = schema.trim().to_ascii_lowercase();
    let (fields, ip_key) = match schema.as_str() {
        "zh" | "chinese" => (ZH_FIELDS.as_slice(), "IP"),
        "en" | "english" => (EN_FIELDS.as_slice(), "ip"),
        other => return Err(format!("Unsupported structured instruction schema: {other:?}")),
    };

    let values: [Option<String>; 8] = [
        // Upstream: `values = (1, "speaker_1", ...)` — the ordinal is the
        // JSON *number* 1, not the string "1".
        Some("1".to_string()),
        Some("speaker_1".to_string()),
        dialect.map(str::to_string),
        style.map(str::to_string),
        speed.map(str::to_string),
        pitch.map(str::to_string),
        volume.map(str::to_string),
        emotion.map(str::to_string),
    ];

    let mut out = String::from("{\"audio_sequence\": [{");
    for (index, field) in fields.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        out.push_str(&py_json_string(field));
        out.push_str(": ");
        if index == 0 {
            // The ordinal slot: bare JSON number.
            out.push_str("1");
        } else {
            match &values[index] {
                Some(value) => out.push_str(&py_json_string(value)),
                None => out.push_str("null"),
            }
        }
    }
    // BGM block with every field null.
    out.push_str(", \"BGM\": {");
    for (index, field) in BGM_FIELDS.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        out.push_str(&py_json_string(field));
        out.push_str(": null");
    }
    out.push('}');
    out.push_str(", ");
    out.push_str(&py_json_string(ip_key));
    out.push_str(": ");
    match ip {
        Some(value) => out.push_str(&py_json_string(value)),
        None => out.push_str("null"),
    }
    // Close the inner item object, the audio_sequence array, and the outer
    // object — three brackets, one per level (Python: `...null}]}`).
    out.push_str("}]}");
    Ok(out)
}

/// The controls a task request carries; unrelated fields are refused.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Controls {
    pub dialect: Option<String>,
    pub style: Option<String>,
    pub speed: Option<String>,
    pub pitch: Option<String>,
    pub volume: Option<String>,
    pub emotion: Option<String>,
    pub ip: Option<String>,
}

impl Controls {
    fn present(&self) -> Vec<(&'static str, &str)> {
        let mut out = Vec::new();
        if let Some(v) = &self.dialect {
            out.push(("dialect", v.as_str()));
        }
        if let Some(v) = &self.style {
            out.push(("style", v.as_str()));
        }
        if let Some(v) = &self.speed {
            out.push(("speed", v.as_str()));
        }
        if let Some(v) = &self.pitch {
            out.push(("pitch", v.as_str()));
        }
        if let Some(v) = &self.volume {
            out.push(("volume", v.as_str()));
        }
        if let Some(v) = &self.emotion {
            out.push(("emotion", v.as_str()));
        }
        if let Some(v) = &self.ip {
            out.push(("ip", v.as_str()));
        }
        out
    }

    fn is_empty(&self) -> bool {
        self.present().is_empty()
    }
}

/// Build a task caption while preventing unrelated controls from leaking.
pub fn build_task_instruction(
    task: &str,
    controls: &Controls,
    schema: Option<&str>,
) -> Result<String, String> {
    let task = task.trim().to_ascii_uppercase();
    let Some((_, allowed)) = ALLOWED_CONTROLS.iter().find(|(name, _)| *name == task) else {
        return Err(format!("Unsupported TTS control task: {task:?}"));
    };
    let present = controls.present();
    let unknown: Vec<&str> = present
        .iter()
        .map(|(name, _)| *name)
        .filter(|name| !allowed.contains(name))
        .collect();
    if !unknown.is_empty() {
        return Err(format!("{task} does not accept control fields: {unknown:?}"));
    }

    let resolved_schema =
        schema.unwrap_or(if task == "DIALECT" { "en" } else { "zh" });
    build_structured_instruction(
        controls.dialect.as_deref(),
        controls.style.as_deref(),
        controls.speed.as_deref(),
        controls.pitch.as_deref(),
        controls.volume.as_deref(),
        controls.emotion.as_deref(),
        controls.ip.as_deref(),
        resolved_schema,
    )
}

/// A resolved task request: exactly the fields the generation loop consumes.
#[derive(Debug, Clone, PartialEq)]
pub struct TaskRequest {
    /// The user-side prompt (`build_tts_prompt`).
    pub prompt: String,
    /// The (trimmed) text to synthesize.
    pub text: String,
    pub use_spk_emb: bool,
    pub use_zero_spk_emb: bool,
    /// The assistant-side caption for structured tasks / freeform
    /// instructions; `None` for plain TTS.
    pub instruction: Option<String>,
    /// Reference-audio path / speaker-embedding path, passed through for
    /// the runtime layer to resolve.
    pub prompt_wav_path: Option<String>,
    pub prompt_text: Option<String>,
    pub spk_emb_path: Option<String>,
    /// The runtime task name (`TTS` | `BASIC` | `EMOTION` | `DIALECT` |
    /// `IP` | `STYLE`).
    pub taskname: String,
    /// The `language_id` as given (for text-frontend decisions).
    pub text_language: Option<String>,
}

/// Resolve a task request into the fields consumed by the model
/// (`build_task_generation_kwargs` upstream).
#[allow(clippy::too_many_arguments)]
pub fn build_task_generation_kwargs(
    task: &str,
    text: &str,
    language_id: Option<&str>,
    target_language: Option<&str>,
    instruction: Option<&str>,
    controls: &Controls,
    instruction_schema: Option<&str>,
    prompt_wav_path: Option<&str>,
    prompt_text: Option<&str>,
    spk_emb_path: Option<&str>,
    use_spk_emb: bool,
    use_zero_spk_emb: bool,
) -> Result<TaskRequest, String> {
    let task = task.trim().to_ascii_uppercase().replace('-', "_");
    let Some((_, runtime_task)) = TASK_ALIASES.iter().find(|(name, _)| *name == task) else {
        let valid: Vec<&str> = TASK_ALIASES.iter().map(|(name, _)| *name).collect();
        return Err(format!(
            "Unsupported Ming TTS task {task:?}; expected one of {valid:?}"
        ));
    };
    let runtime_task = (*runtime_task).to_string();

    let mut language_id = language_id.map(str::to_string);
    let mut target_language = target_language.map(str::to_string);
    if task == "DIALECT" {
        // DIALECT does not use a language ID (upstream warns and drops it).
        let stray = [&language_id, &target_language]
            .iter()
            .any(|value| value.as_deref().is_some_and(|v| !v.trim().is_empty()));
        if stray {
            language_id = None;
            target_language = None;
        }
    }
    if let (Some(language), Some(target)) = (&language_id, &target_language) {
        if language.trim().to_ascii_lowercase() != target.trim().to_ascii_lowercase() {
            return Err("language_id and target_language must not conflict".to_string());
        }
    }
    if language_id.is_none() {
        language_id = target_language;
    }
    if matches!(task.as_str(), "MULTILINGUAL" | "MULTILINGUAL_TTS")
        && language_id.as_deref().is_none_or(|v| v.trim().is_empty())
    {
        return Err("MULTILINGUAL requires language_id".to_string());
    }

    let text = text.trim().to_string();
    if text.is_empty() {
        return Err("text must not be empty".to_string());
    }
    if use_spk_emb && use_zero_spk_emb {
        return Err("use_spk_emb and use_zero_spk_emb cannot both be true".to_string());
    }
    if use_spk_emb
        && prompt_wav_path.is_none()
        && spk_emb_path.is_none()
    {
        return Err("use_spk_emb requires a prompt wav or a speaker embedding".to_string());
    }

    let task_instruction: Option<String>;
    if task == "FREEFORM" {
        if !controls.is_empty() {
            return Err("FREEFORM uses instruction, not controls".to_string());
        }
        match instruction.map(str::trim).filter(|v| !v.is_empty()) {
            Some(value) => task_instruction = Some(value.to_string()),
            None => return Err("FREEFORM requires a non-empty instruction".to_string()),
        }
    } else if STRUCTURED_TASKS.contains(&task.as_str()) {
        if instruction.is_some() {
            return Err(format!(
                "{task} builds its instruction from controls; do not pass instruction"
            ));
        }
        if controls.is_empty() {
            return Err(format!("{task} requires task controls"));
        }
        let prompt_task = if task == "STYLE" { "VOICE_DESIGN" } else { task.as_str() };
        task_instruction = Some(build_task_instruction(
            prompt_task,
            controls,
            instruction_schema,
        )?);
    } else {
        if !controls.is_empty() {
            return Err(format!("{task} does not accept controls"));
        }
        task_instruction = instruction
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .map(str::to_string);
    }

    Ok(TaskRequest {
        prompt: build_tts_prompt(language_id.as_deref()),
        text,
        use_spk_emb,
        use_zero_spk_emb,
        instruction: task_instruction,
        prompt_wav_path: prompt_wav_path.map(str::to_string),
        prompt_text: prompt_text.map(str::to_string),
        spk_emb_path: spk_emb_path.map(str::to_string),
        taskname: runtime_task,
        text_language: language_id,
    })
}

/// Return the user-side prompt; the target language line is opt-in.
pub fn build_tts_prompt(target_language: Option<&str>) -> String {
    match target_language.map(str::trim).filter(|v| !v.is_empty()) {
        None => DEFAULT_TTS_PROMPT.to_string(),
        Some(language) => {
            format!("{DEFAULT_TTS_PROMPT}Target language: {}.\n", target_language_name(language))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_prompt_has_no_language_line() {
        assert_eq!(build_tts_prompt(None), DEFAULT_TTS_PROMPT);
        assert_eq!(build_tts_prompt(Some("  ")), DEFAULT_TTS_PROMPT);
    }

    #[test]
    fn language_line_uses_the_canonical_name() {
        assert_eq!(
            build_tts_prompt(Some("ja")),
            format!("{DEFAULT_TTS_PROMPT}Target language: Japanese.\n")
        );
        assert_eq!(
            build_tts_prompt(Some("Japanese")),
            format!("{DEFAULT_TTS_PROMPT}Target language: Japanese.\n")
        );
        // Cantonese resolves to Chinese (a dialect, not a language).
        assert_eq!(
            build_tts_prompt(Some("cantonese")),
            format!("{DEFAULT_TTS_PROMPT}Target language: Chinese.\n")
        );
    }

    #[test]
    fn structured_caption_is_byte_exact() {
        // The reference Python string for a BASIC speed control, zh schema:
        // json.dumps({"audio_sequence": [{"序号": 1, "说话人": "speaker_1",
        //   "方言": None, "风格": None, "语速": "快速", "基频": None,
        //   "音量": None, "情感": None, "BGM": {"Genre": None, "Mood": None,
        //   "Instrument": None, "Theme": None, "ENV": None, "SNR": None},
        //   "IP": None}]}, ensure_ascii=False)
        let caption = build_structured_instruction(
            None,
            None,
            Some("快速"),
            None,
            None,
            None,
            None,
            "zh",
        )
        .expect("caption");
        let expected = "{\"audio_sequence\": [{\"序号\": 1, \"说话人\": \"speaker_1\", \
            \"方言\": null, \"风格\": null, \"语速\": \"快速\", \"基频\": null, \
            \"音量\": null, \"情感\": null, \"BGM\": {\"Genre\": null, \"Mood\": null, \
            \"Instrument\": null, \"Theme\": null, \"ENV\": null, \"SNR\": null}, \
            \"IP\": null}]}";
        if caption != expected {
            for (index, (a, b)) in caption.chars().zip(expected.chars()).enumerate() {
                if a != b {
                    panic!("caption differs at char {index}: actual={a:?} expected={b:?}");
                }
            }
            panic!(
                "caption length differs: actual={} expected={}\n  actual={caption:?}\n  expected={expected:?}",
                caption.chars().count(),
                expected.chars().count()
            );
        }
    }

    #[test]
    fn dialect_caption_uses_the_en_schema_by_default() {
        let instruction =
            build_task_instruction("DIALECT", &Controls { dialect: Some("cantonese".into()), ..Default::default() }, None)
                .expect("caption");
        assert!(instruction.contains("\"id\": 1"), "en fields: {instruction}");
        assert!(instruction.contains("\"dialect\": \"cantonese\""), "got: {instruction}");
        assert!(instruction.contains("\"ip\": null"), "lower-case ip key: {instruction}");
        assert!(!instruction.contains("序号"), "no zh fields: {instruction}");
    }

    #[test]
    fn unrelated_controls_are_refused() {
        let err = build_task_instruction(
            "BASIC",
            &Controls { emotion: Some("高兴".into()), ..Default::default() },
            None,
        )
        .unwrap_err();
        assert!(err.contains("does not accept"), "got: {err}");
    }

    #[test]
    fn task_aliases_resolve_like_the_reference() {
        let request = |task| {
            build_task_generation_kwargs(task, " hello ", None, None, None, &Controls::default(), None, None, None, None, false, false)
        };
        assert_eq!(request("tts").expect("tts").taskname, "TTS");
        assert_eq!(request("SEED").expect("seed").taskname, "TTS");
        assert_eq!(request("PLAIN").expect("plain").taskname, "TTS");
        // Structured tasks need controls; FREEFORM needs an instruction —
        // the aliases resolve but the resolver refuses empty requests.
        let err = request("voice-design").unwrap_err();
        assert!(err.contains("requires task controls"), "design: {err}");
        let err = request("FREEFORM").unwrap_err();
        assert!(err.contains("non-empty instruction"), "freeform: {err}");
        let request_with = |task: &str, controls: Controls| {
            build_task_generation_kwargs(
                task,
                " hello ",
                None,
                None,
                None,
                &controls,
                None,
                None,
                None,
                None,
                false,
                true,
            )
        };
        let ip = request_with("IP", Controls { ip: Some("UK_George".into()), ..Default::default() });
        assert_eq!(ip.expect("ip").taskname, "IP");
        let style = request_with(
            "STYLE",
            Controls { style: Some("成熟沉稳的男声".into()), ..Default::default() },
        );
        assert_eq!(style.expect("style").taskname, "STYLE");
        let err = request("PODCAST").unwrap_err();
        assert!(err.contains("Unsupported Ming TTS task"), "got: {err}");
        // Both multilingual aliases demand a language_id even though they
        // alias to plain TTS (upstream checks before dispatching).
        for task in ["MULTILINGUAL", "MULTILINGUAL_TTS"] {
            let err = request(task).unwrap_err();
            assert!(err.contains("MULTILINGUAL requires language_id"), "{task}: {err}");
        }
    }

    #[test]
    fn multilingual_requires_a_language_id() {
        let err = build_task_generation_kwargs(
            "MULTILINGUAL",
            "こんにちは",
            None,
            None,
            None,
            &Controls::default(),
            None,
            None,
            None,
            None,
            false,
            false,
        )
        .unwrap_err();
        assert!(err.contains("MULTILINGUAL requires language_id"), "got: {err}");
    }

    #[test]
    fn freeform_requires_an_instruction_and_no_controls() {
        let err = build_task_generation_kwargs(
            "FREEFORM",
            "hi",
            None,
            None,
            None,
            &Controls::default(),
            None,
            None,
            None,
            None,
            false,
            false,
        )
        .unwrap_err();
        assert!(err.contains("non-empty instruction"), "got: {err}");

        let err = build_task_generation_kwargs(
            "FREEFORM",
            "hi",
            None,
            None,
            Some("warm voice"),
            &Controls { ip: Some("UK_George".into()), ..Default::default() },
            None,
            None,
            None,
            None,
            false,
            false,
        )
        .unwrap_err();
        assert!(err.contains("uses instruction, not controls"), "got: {err}");
    }

    #[test]
    fn speaker_flags_are_mutually_exclusive() {
        let err = build_task_generation_kwargs(
            "TTS",
            "hi",
            None,
            None,
            None,
            &Controls::default(),
            None,
            Some("ref.wav"),
            Some("text"),
            None,
            true,
            true,
        )
        .unwrap_err();
        assert!(err.contains("cannot both be true"), "got: {err}");

        let err = build_task_generation_kwargs(
            "TTS",
            "hi",
            None,
            None,
            None,
            &Controls::default(),
            None,
            None,
            None,
            None,
            true,
            false,
        )
        .unwrap_err();
        assert!(err.contains("requires a prompt wav or a speaker embedding"), "got: {err}");
    }

    #[test]
    fn dialect_drops_the_language_id_with_a_warning_upstream() {
        let request = build_task_generation_kwargs(
            "DIALECT",
            "今天天气几好",
            Some("Japanese"),
            None,
            None,
            &Controls { dialect: Some("cantonese".into()), ..Default::default() },
            None,
            None,
            None,
            None,
            false,
            false,
        )
        .expect("request");
        // The language line is dropped entirely: the prompt stays plain.
        assert_eq!(request.prompt, DEFAULT_TTS_PROMPT);
        assert_eq!(request.text_language, None);
    }

    #[test]
    fn structured_task_builds_the_caption_into_the_request() {
        let request = build_task_generation_kwargs(
            "IP",
            "The garden will remain open until sunset.",
            None,
            None,
            None,
            &Controls { ip: Some("UK_George".into()), ..Default::default() },
            None,
            None,
            None,
            None,
            false,
            true,
        )
        .expect("request");
        assert_eq!(request.taskname, "IP");
        assert!(request.use_zero_spk_emb);
        let caption = request.instruction.expect("caption");
        assert!(caption.contains("\"IP\": \"UK_George\""), "got: {caption}");
        assert!(caption.starts_with("{\"audio_sequence\": [{"), "got: {caption}");
    }

    #[test]
    fn preserve_multilingual_text_bypasses_zh_en_only() {
        assert!(!preserve_multilingual_text(None));
        assert!(!preserve_multilingual_text(Some("zh")));
        assert!(!preserve_multilingual_text(Some("english")));
        assert!(preserve_multilingual_text(Some("ja")));
        assert!(preserve_multilingual_text(Some("fr")));
    }
}
