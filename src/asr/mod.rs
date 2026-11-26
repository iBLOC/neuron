//! ASR (Automatic Speech Recognition) runtime implementations.
//!
//! Ships three runtimes behind feature flags:
//!
//! * `WhisperAsrRuntime` (feature `candle`) -- OpenAI Whisper via candle.
//! * `TractSenseVoiceRuntime` (feature `tract`) -- sherpa-onnx exported
//!   SenseVoiceSmall graphs executed on neuron's own ONNX stack, with the
//!   full preprocessing pipeline (fbank / LFR / CMVN / CTC) in-repo.
//! * `SherpaAsrRuntime` (feature `sherpa`) -- the same SenseVoice graphs
//!   through the official sherpa-onnx Rust binding (onnxruntime-backed).
//!   The last two exist side by side so `benches/asr_speed.rs` can run
//!   one graph on both engines and report the gap.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::config::DeviceSelection;
use crate::llm::ChatResult;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AsrImpl {
    #[cfg(feature = "candle")]
    Candle,
    #[cfg(feature = "tract")]
    Tract,
    #[cfg(feature = "sherpa")]
    Sherpa,
}

/// ⚠️⚠️ **从盘上事实推引擎**（2026-11-26）—— **不猜**。
///
/// ⚠️ **为什么**：ASR 有三条腿（candle 的 whisper／sherpa 的 onnx／tract
/// 的预处理图）⇒ ⇒ 客户端递的 `engine_impl` **可能没递**（或递错）⇒ ⇒
/// 「注册面」与「加载面」之间必须有一格**能从盘上自己看出来**的判定
/// ⇒ ⇒ 否则「引擎注册上了、权重找不到」又回来了。
///
/// ⚠️ **口径**（**只看盘上有什么**，不看目录名）：
///   · `config.json` ＋ `tokenizer.json` ＋ 至少一个 `*.safetensors`
///     ⇒ ⇒ candle 的 whisper 布局；
///   · 至少一个 `*.onnx` ＋ `tokens.txt` ⇒ ⇒ onnxruntime 那一族
///     （sherpa；tract 那族要的是**预处理的图**，不是原始 onnx ⇒ ⇒
///     不在这一格里推）。
///   ⇒ ⇒ 推不出来 ⇒ ⇒ `None` ⇒ ⇒ **不注册**（「判不了 ≠ 有」不许破）。
pub fn infer_asr_impl_from_dir(dir: &std::path::Path) -> Option<AsrImpl> {
    let has = |name: &str| dir.join(name).exists();
    let has_ext = |ext: &str| {
        std::fs::read_dir(dir)
            .map(|entries| {
                entries.filter_map(|e| e.ok()).any(|e| {
                    e.path()
                        .extension()
                        .map(|x| x.eq_ignore_ascii_case(ext))
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false)
    };
    #[cfg(feature = "candle")]
    {
        if has("config.json") && has("tokenizer.json") && has_ext("safetensors") {
            return Some(AsrImpl::Candle);
        }
    }
    #[cfg(feature = "sherpa")]
    {
        if has_ext("onnx") && has("tokens.txt") {
            return Some(AsrImpl::Sherpa);
        }
    }
    let _ = (&has, &has_ext);
    None
}

impl AsrImpl {
    pub fn as_str(&self) -> &'static str {
        match self {
            #[cfg(feature = "candle")]
            Self::Candle => "candle",
            #[cfg(feature = "tract")]
            Self::Tract => "tract",
            #[cfg(feature = "sherpa")]
            Self::Sherpa => "sherpa",
            #[cfg(not(any(feature = "candle", feature = "tract", feature = "sherpa")))]
            _ => "unavailable",
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            #[cfg(feature = "candle")]
            "candle" | "whisper" => Ok(Self::Candle),
            #[cfg(feature = "tract")]
            "tract" => Ok(Self::Tract),
            #[cfg(feature = "sherpa")]
            "sherpa" => Ok(Self::Sherpa),
            #[allow(unused_variables)]
            other => Err(format!("unknown AsrImpl: {other}")),
        }
    }
}

// ---------------------------------------------------------------------------
// ASR 权重目录解析（三条腿共用这一个）
// ---------------------------------------------------------------------------

/// **ASR 权重目录的解析器**（candle／sherpa／tract 三条腿共用这一个）。
///
/// ⚠️ **为什么必须共用**（2026-11-26）：三条腿各写各的解析器 ⇒ ⇒ 同一个模型
/// 在三条腿里被判成三个不同的结论 ⇒ ⇒ 真机上表现为「有的腿找得到、有的腿
/// 找不到」⇒ ⇒ **失败长得像配置问题**，而真相是「解析器写岔了」。
///
/// ⚠️ **布局是 `<root>/asr/<model_id>`** —— 与 puller 落盘口径逐字一致
/// （`Registry::model_path` ＝ `<root>/<engine>/<id>`，而
/// `EngineType::Asr.as_str()` ＝ `"asr"`）⇒ ⇒ **下载落在哪，加载就去哪找**。
///
/// ⚠️ **只认这一种布局**（不像 LLM 那条腿还兜一层 `<root>/<id>`）：ASR 从来
/// 没有「直接放在根下」的落法，而那一层兜底会让 `model_id == "asr"` 这种
/// 引擎目录名被当成模型目录 ⇒ ⇒ **布局不许含糊**。
pub fn find_asr_model_dir(
    roots: &[std::path::PathBuf],
    model_id: &str,
) -> Option<std::path::PathBuf> {
    for root in roots {
        let candidate = root.join("asr").join(model_id);
        if candidate.is_dir() {
            return Some(candidate);
        }
    }
    None
}

/// **纯函数**：按**给定的根**解析 ASR 权重目录；找不到 ⇒ ⇒ 错误里说清试过哪些。
pub fn resolve_asr_model_dir_in(
    roots: &[std::path::PathBuf],
    model_id: &str,
) -> Result<std::path::PathBuf, String> {
    if let Some(found) = find_asr_model_dir(roots, model_id) {
        return Ok(found);
    }
    let tried: Vec<String> = roots
        .iter()
        .map(|r| r.join("asr").join(model_id).display().to_string())
        .collect();
    Err(format!(
        "ASR model dir not found for '{model_id}'; tried {}",
        if tried.is_empty() {
            "(no model root configured)".to_string()
        } else {
            tried.join(", ")
        }
    ))
}

/// **模型库根**（纯函数那一半）—— `NEURON_MODEL_DIRS`（多根，平台分隔符）
/// → `NEURON_MODEL_DIR`（单根）→ **`NEURON_MODELS_DIR`**（**嵌脑那一格**：
/// 端脑启动时把宿主给的 `<data_dir>/models` 指到那儿）→ 默认档**排最后**。
///
/// ⚠️ **为什么自带一套**（2026-11-26）：neuron 的 `config.rs` 里那一格
/// (`NeuronConfig::model_roots`) **此刻还是别的会话未提交的** ⇒ ⇒ 这一刀
/// 若依赖它 ⇒ ⇒ 单独提交就是**编不过**的提交 ⇒ ⇒ **不借未提交的腿** ⇒ ⇒
/// 本刀自包含。
///
/// ⚠️ **多根分隔符走 `std::env::split_paths`**（Windows `;`／Unix `:`）
/// ⇒ ⇒ **Windows 的盘符 `C:` 不会被当成分隔符**（`split(':')` 会 ⇒ ⇒
/// 根永远对不上）⇒ ⇒ 单根那两格**不拆**。
pub fn asr_model_roots_from(
    env: &[(String, String)],
    default_root: &std::path::Path,
) -> Vec<std::path::PathBuf> {
    let mut roots = Vec::new();
    if let Some((_, v)) = env.iter().find(|(k, _)| k == "NEURON_MODEL_DIRS") {
        roots.extend(std::env::split_paths(v).filter(|p| !p.as_os_str().is_empty()));
    }
    for var in ["NEURON_MODEL_DIR", "NEURON_MODELS_DIR"] {
        if let Some((_, v)) = env.iter().find(|(k, _)| k == var) {
            if !v.is_empty() {
                roots.push(std::path::PathBuf::from(v));
            }
        }
    }
    if !roots.iter().any(|r| r == default_root) {
        roots.push(default_root.to_path_buf());
    }
    roots
}

/// **模型库根**（env 那一层 —— 薄壳，判据在纯函数里）。
///
/// ⚠️ 只在**没有递进来的根**时用（`load()` 那条老路）⇒ ⇒ 注册面递下来的
/// 那份（`config.model_dirs`）**优先** ⇒ ⇒ 两份清单不打架。
pub fn asr_model_roots() -> Vec<std::path::PathBuf> {
    let env: Vec<(String, String)> = ["NEURON_MODEL_DIRS", "NEURON_MODEL_DIR", "NEURON_MODELS_DIR"]
        .iter()
        .filter_map(|k| std::env::var(k).ok().map(|v| (k.to_string(), v)))
        .collect();
    let default_root = crate::config::NeuronConfig::resolved_model_dir();
    asr_model_roots_from(&env, &default_root)
}

/// **显式 env 覆盖那一格**（`NEURON_ASR_MODEL_PATH`）—— 设了就用它。
///
/// ⚠️ 这是**唯一**读进程环境的一格 ⇒ ⇒ 判据住在纯函数那边 ⇒ ⇒ **不与别的
/// 判据抢环境变量**（本仓已量过：并行测试互抢 env 是真的，上一刀的三条 ASR
/// 判据就是这么红的）。
pub fn explicit_asr_model_dir() -> Result<Option<std::path::PathBuf>, String> {
    let Ok(p) = crate::config::env_var_result("ASR_MODEL_PATH") else {
        return Ok(None);
    };
    let path = std::path::PathBuf::from(&p);
    if !path.is_dir() {
        return Err(format!("NEURON_ASR_MODEL_PATH={p} is not a directory"));
    }
    Ok(Some(path))
}

/// **解析入口**（显式 env 覆盖 → 给定的根）。
pub fn resolve_asr_model_dir_with_override_in(
    roots: &[std::path::PathBuf],
    model_id: &str,
) -> Result<std::path::PathBuf, String> {
    if let Some(dir) = explicit_asr_model_dir()? {
        return Ok(dir);
    }
    resolve_asr_model_dir_in(roots, model_id)
}

#[derive(Debug, Clone)]
pub struct RegisteredAsr {
    pub model_id: String,
    pub impl_id: AsrImpl,
    pub device: crate::config::DeviceSelection,
    pub required_memory_bytes: u64,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct AsrRegistryKey {
    pub model_id: String,
    pub impl_id: AsrImpl,
    pub device: crate::config::DeviceSelection,
}

#[cfg(feature = "candle")]
pub mod whisper;

#[cfg(feature = "candle")]
pub use whisper::WhisperAsrRuntime;

#[cfg(feature = "tract")]
pub mod fbank;
#[cfg(feature = "tract")]
pub mod onnx_meta;
#[cfg(feature = "tract")]
pub mod sensevoice;

#[cfg(feature = "tract")]
pub use sensevoice::{TractSenseVoiceRuntime, ctc_greedy as asr_ctc_greedy};

#[cfg(feature = "sherpa")]
pub mod sherpa;

#[cfg(feature = "sherpa")]
pub use sherpa::SherpaAsrRuntime;

#[cfg(feature = "asr-audio")]
pub mod audio;
pub mod wav;

pub mod registry;
pub use registry::AsrRuntimeRegistry;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transcription {
    pub text: String,
    pub language: Option<String>,
    pub duration_secs: Option<f32>,
}

impl Transcription {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            language: None,
            duration_secs: None,
        }
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        let text = String::from_utf8_lossy(bytes).to_string();
        Self::new(text)
    }
}

#[async_trait]
pub trait AsrRuntime: Send + Sync {
    async fn load(model: &str, device: &DeviceSelection) -> ChatResult<Box<dyn AsrRuntime>>
    where
        Self: Sized;

    /// **roots —— 模型库根**（2026-11-26 刀 16 续同源）：**注册面**（下载／
    /// 注册走 `config.model_dirs`，宿主交进来的那份）与**加载面**必须读
    /// **同一份** ⇒ ⇒ 否则嵌脑形态里宿主给的目录只被用来下载，加载时却去
    /// 别处找 ⇒ ⇒ **引擎注册上了、权重找不到**（失败长得像「模型没下载」
    /// ⇒ ⇒ 而盘上**明明有**那份权重）。
    ///
    /// ⚠️ **默认实现**＝走 `load(model, device)` ⇒ ⇒ 不读模型库的腿不必改
    /// ⇒ ⇒ **不破坏**任何既有实现（跨仓也安全：`AsrRuntime` 的实现者只在
    /// neuron 内）。
    async fn load_with_roots(
        model: &str,
        device: &DeviceSelection,
        _roots: &[std::path::PathBuf],
    ) -> ChatResult<Box<dyn AsrRuntime>>
    where
        Self: Sized,
    {
        Self::load(model, device).await
    }

    async fn transcribe(&self, audio: &[u8], lang: Option<&str>) -> ChatResult<Transcription>;

    fn name(&self) -> &str;
}

#[cfg(all(test, feature = "candle"))]
mod whisper_tests {
    use super::whisper::WhisperAsrRuntime;
    use super::*;

    #[tokio::test]
    async fn test_whisper_transcribe_empty_audio_rejected() {
        let runtime = WhisperAsrRuntime::new();
        let result = runtime.transcribe(&[], None).await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.message.contains("empty audio"));
    }

    #[tokio::test]
    async fn test_whisper_transcribe_garbage_bytes_rejected() {
        let runtime = WhisperAsrRuntime::new();
        let result = runtime.transcribe(b"not audio data", None).await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.message.contains("RIFF") || err.message.contains("ID3"));
    }
}

// ---------------------------------------------------------------------------
// 判据：ASR 权重目录的解析（纯函数，**零 env 读取** ⇒ ⇒ 不与别的判据抢 env）
// ---------------------------------------------------------------------------
#[cfg(test)]
mod asr_model_dir_tests {
    use super::*;

    fn tmpdir(tag: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("neuron-asr-dir-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).expect("tmp dir");
        p
    }

    /// **布局**＝`<root>/asr/<model_id>`（与 puller 落盘逐字同口径）。
    #[test]
    fn asr_model_dir_layout_is_root_asr_model_id() {
        let root = tmpdir("layout");
        let dir = root.join("asr").join("whisper-tiny");
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(
            find_asr_model_dir(&[root.clone()], "whisper-tiny"),
            Some(dir)
        );
        // ⚠️ **引擎目录不是模型目录** ⇒ ⇒ `model_id == "asr"` 不许命中
        // `<root>/asr` ⇒ ⇒ 布局不许含糊。
        assert_eq!(find_asr_model_dir(&[root.clone()], "asr"), None);
        assert_eq!(find_asr_model_dir(&[root.clone()], "not-there"), None);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// **多个根按序找** ⇒ ⇒ 命中第二个。
    #[test]
    fn asr_model_dir_searches_the_given_roots_in_order() {
        let r1 = tmpdir("multi1");
        let r2 = tmpdir("multi2");
        let dir = r2.join("asr").join("whisper-base");
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(
            find_asr_model_dir(&[r1.clone(), r2.clone()], "whisper-base"),
            Some(dir)
        );
        let _ = std::fs::remove_dir_all(&r1);
        let _ = std::fs::remove_dir_all(&r2);
    }

    /// ⚠️ **只搜给定的根**（优先级被钉死的那一格）—— 落在**第二个**根里的
    /// 模型，用**第一个**根去搜**不许找到** ⇒ ⇒ 加载面不许偷偷回落去别的
    /// 清单（env 根／默认档）⇒ ⇒ 否则同一个模型在同一个决策里被判成两个
    /// 相反的结论（LLM 那条腿「刀 16 续」修的就是这个）。
    #[test]
    fn asr_model_dir_only_searches_the_given_roots() {
        let r1 = tmpdir("only1");
        let r2 = tmpdir("only2");
        let dir = r2.join("asr").join("whisper-small");
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(find_asr_model_dir(&[r1.clone()], "whisper-small"), None);
        assert!(resolve_asr_model_dir_in(&[r1.clone()], "whisper-small").is_err());
        let _ = std::fs::remove_dir_all(&r1);
        let _ = std::fs::remove_dir_all(&r2);
    }

    /// **找不到 ⇒ ⇒ 错误里说清试过哪些目录**（不许只说一句「找不到」）。
    #[test]
    fn asr_model_dir_error_lists_every_tried_dir() {
        let r1 = tmpdir("tried1");
        let r2 = tmpdir("tried2");
        let err = resolve_asr_model_dir_in(&[r1.clone(), r2.clone()], "m").unwrap_err();
        let a = r1.join("asr").join("m").display().to_string();
        let b = r2.join("asr").join("m").display().to_string();
        assert!(err.contains("m"), "err must name the model: {err}");
        assert!(err.contains(&a), "err must show the tried dir: {err}");
        assert!(err.contains(&b), "err must show every tried dir: {err}");
        // ⚠️ **不许偷偷多搜**：清单里必须**只有**给定的根 ⇒ ⇒ 若加载面偷偷
        // 回落去 env 根／默认档，这一格当场红。
        for extra in super::asr_model_roots() {
            let p = extra.join("asr").join("m").display().to_string();
            assert!(
                !err.contains(&p),
                "err must not list a root we did not ask for: {p} in {err}"
            );
        }
        let _ = std::fs::remove_dir_all(&r1);
        let _ = std::fs::remove_dir_all(&r2);
    }

    /// **env 三格都要认**（纯函数）—— 多根 / 单根 / 嵌脑那一格。
    #[test]
    fn asr_model_roots_from_reads_the_three_env_slots() {
        let default_root = std::path::PathBuf::from("/default/models");
        let env = vec![
            ("NEURON_MODEL_DIRS".to_string(), "/multi".to_string()),
            ("NEURON_MODEL_DIR".to_string(), "/single".to_string()),
            ("NEURON_MODELS_DIR".to_string(), "/embedded".to_string()),
        ];
        let roots = asr_model_roots_from(&env, &default_root);
        assert!(
            roots
                .iter()
                .any(|r| r == &std::path::PathBuf::from("/multi"))
        );
        assert!(
            roots
                .iter()
                .any(|r| r == &std::path::PathBuf::from("/single"))
        );
        assert!(
            roots
                .iter()
                .any(|r| r == &std::path::PathBuf::from("/embedded"))
        );
    }

    /// ⚠️ **盘符不许被切开**（`split(':')` 那一族）—— Windows 上的路径自带
    /// 盘符 ⇒ ⇒ 切开就变成两个目录 ⇒ ⇒ 根永远对不上。
    ///
    /// ⚠️ **夹具必须走【多根那一格】**（`NEURON_MODEL_DIRS`）—— 单根那两格
    /// **按设计不拆** ⇒ ⇒ 拿单根槽当夹具 ⇒ ⇒ 这条判据对「多根被错拆」那个
    /// 真 bug 是**结构性免疫**的 ⇒ ⇒ **反向验证当场教过这一条**（把
    /// `split_paths` 换成 `split(':')`，它照样绿）⇒ ⇒ 判据要钉在**真会被
    /// 拆的那一格**上。
    #[test]
    fn asr_model_roots_from_survives_a_windows_drive_letter() {
        let default_root = std::path::PathBuf::from("/default/models");
        let win = r"C:\Users\probe\models";
        let env = vec![("NEURON_MODEL_DIRS".to_string(), win.to_string())];
        let roots = asr_model_roots_from(&env, &default_root);
        if cfg!(windows) {
            assert_eq!(
                roots.len(),
                2,
                "a Windows path must survive as ONE root: {roots:?}"
            );
            assert_eq!(roots[0], std::path::PathBuf::from(win));
        } else {
            assert_eq!(roots.len(), 3, "on Unix ':' separates roots: {roots:?}");
        }
        // 单根那两格**不拆** ⇒ ⇒ 一个带盘符的路径也必须是**一根**。
        let env = vec![("NEURON_MODELS_DIR".to_string(), win.to_string())];
        let roots = asr_model_roots_from(&env, &default_root);
        assert_eq!(
            roots.len(),
            2,
            "the single-root slot must never be split: {roots:?}"
        );
        assert_eq!(roots[0], std::path::PathBuf::from(win));
    }

    /// **默认档永远排最后**（不是「env 有就不看默认」）⇒ ⇒ 指了 env 也不会
    /// 丢掉默认库里已下载的模型。
    #[test]
    fn asr_model_roots_from_keeps_the_default_root_last() {
        let default_root = std::path::PathBuf::from("/default/models");
        let env = vec![("NEURON_MODELS_DIR".to_string(), "/embedded".to_string())];
        let roots = asr_model_roots_from(&env, &default_root);
        assert_eq!(
            roots.last(),
            Some(&default_root),
            "default must be last: {roots:?}"
        );
    }

    /// **多根按平台自己的分隔符**（Windows `;`／Unix `:`）⇒ ⇒ 用
    /// `std::env::join_paths` 造夹具 ⇒ ⇒ 判据本身不写死分隔符。
    #[test]
    fn asr_model_roots_from_splits_multi_root_with_the_platform_separator() {
        let default_root = std::path::PathBuf::from("/default/models");
        let joined = std::env::join_paths(["/a", "/b"]).expect("join");
        let env = vec![(
            "NEURON_MODEL_DIRS".to_string(),
            joined.to_string_lossy().to_string(),
        )];
        let roots = asr_model_roots_from(&env, &default_root);
        assert_eq!(roots.len(), 3, "two roots + the default: {roots:?}");
        assert_eq!(roots[0], std::path::PathBuf::from("/a"));
        assert_eq!(roots[1], std::path::PathBuf::from("/b"));
    }
}
