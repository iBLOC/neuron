//! **说话人嵌入（声纹）** —— 把一段人声变成一条 192 维向量。
//!
//! ## 为什么是「内置模型」（用户 2026-11-25 拍板）
//!
//! > 「声纹识别作为内置模型直接打包进 apk 和 exe 里面，不再需要用户配置」
//!
//! ⇒ ⇒ 权重解析**内置优先**：发行包把 `campplus.onnx` 放在可执行文件同级的
//! `models/` 下 ⇒ ⇒ 用户零配置；环境变量只是开发／CI 的覆盖口。
//!
//! ## 解析顺序
//!
//! 1. `NEURON_SPK_MODEL_PATH` —— 指向 onnx 文件，或含 `campplus.onnx` 的目录；
//! 2. **内置**：`<exe dir>/models/campplus.onnx`；
//! 3. 都没有 ⇒ **明确报错**（绝不拿零向量／随机向量冒充声纹）。
//!
//! ## 与音色克隆那条腿的关系
//!
//! CAM++ 的实现原是 Ming-3.0 音色克隆的内部零件（`tts/ming3/spkemb.rs`），
//! 现在搬到 [`campplus`] 并**两边共用** —— 特征提取只有一份实现，不会漂。

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

/// CAM++ 输出的维度（上游 `SPEAKER_EMBEDDING_DIM`）。
pub const SPEAKER_EMBEDDING_DIM: usize = 192;

/// 内置权重在发行包里的**文件名**。
pub const BUILTIN_MODEL_FILE: &str = "campplus.onnx";

/// 内置权重在发行包里的**相对目录**（相对可执行文件）。
pub const BUILTIN_MODEL_DIR: &str = "models";

/// CAM++ 的运行时实现（feature `speaker-embedding`）。
#[cfg(feature = "speaker-embedding")]
pub mod campplus;

/// 一条声纹向量。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerEmbedding {
    /// 声纹向量（CAM++ ⇒ 192 维）。
    pub embedding: Vec<f32>,
    /// 维度（冗余一份，客户端不用自己数）。
    pub dim: usize,
}

/// **声纹运行时** —— 音频进，声纹向量出。
#[async_trait::async_trait]
pub trait SpeakerEmbeddingRuntime: Send + Sync {
    /// 16 kHz 单声道 WAV（字节）→ 声纹向量。
    async fn embed(&self, audio: &[u8]) -> Result<SpeakerEmbedding, String>;

    /// 运行时名字（注册表／日志用）。
    fn name(&self) -> &str;
}

/// **诚实链的最后一环**：没有内置模型（或 build 没编 `speaker-embedding`）时，
/// 端点**明确说不可用** —— 而不是返回零向量／随机向量冒充声纹。
pub struct UnavailableSpeakerEmbeddingRuntime {
    /// 为什么不可用（**原话**透给调用方）。
    pub reason: String,
}

#[async_trait::async_trait]
impl SpeakerEmbeddingRuntime for UnavailableSpeakerEmbeddingRuntime {
    async fn embed(&self, _audio: &[u8]) -> Result<SpeakerEmbedding, String> {
        Err(self.reason.clone())
    }

    fn name(&self) -> &str {
        "unavailable"
    }
}

/// **懒加载**的声纹运行时：第一次调用时才开 ONNX 会话。
///
/// 为什么要懒：内置权重是 28 MB 的 onnx —— 每个测试、每次启动都 eagerly
/// 加载它 ⇒ ⇒ 启动变慢、测试变慢。懒加载把这份代价挪到**真的要用声纹**的那一刻。
pub struct LazySpeakerEmbeddingRuntime {
    /// 模型名（目前只有 `campplus`）。
    model: String,
    cell: tokio::sync::OnceCell<std::result::Result<Arc<dyn SpeakerEmbeddingRuntime>, String>>,
}

impl LazySpeakerEmbeddingRuntime {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            cell: tokio::sync::OnceCell::new(),
        }
    }

    async fn get(&self) -> Result<&Arc<dyn SpeakerEmbeddingRuntime>, String> {
        let cell = &self.cell;
        let slot = cell.get_or_init(|| async {
            #[cfg(feature = "speaker-embedding")]
            {
                let env = crate::config::env_var("SPK_MODEL_PATH");
                let dir = crate::spk::exe_dir();
                // ⚠️ **模型根目录覆盖口**（`NEURON_MODELS_DIR`）——
                // 移动端那条路（APK assets → filesDir/models）靠它指路。
                let models_dir = crate::config::env_var(crate::spk::MODELS_DIR_ENV)
                    .map(std::path::PathBuf::from);
                match crate::spk::resolve_model_path(
                    env.as_deref(),
                    dir.as_deref(),
                    models_dir.as_deref(),
                ) {
                    Ok(path) => match campplus::CampplusSpeakerEmbeddingRuntime::open(&path) {
                        Ok(rt) => Ok(Arc::new(rt) as Arc<dyn SpeakerEmbeddingRuntime>),
                        Err(e) => Err(format!("speaker embedding model failed to load: {e}")),
                    },
                    Err(e) => Err(e),
                }
            }
            #[cfg(not(feature = "speaker-embedding"))]
            {
                let _ = &self.model;
                Err("speaker embedding is not compiled in (build with --features speaker-embedding)".to_string())
            }
        })
        .await;
        match slot {
            Ok(rt) => Ok(rt),
            Err(e) => Err(e.clone()),
        }
    }
}

#[async_trait::async_trait]
impl SpeakerEmbeddingRuntime for LazySpeakerEmbeddingRuntime {
    async fn embed(&self, audio: &[u8]) -> Result<SpeakerEmbedding, String> {
        let rt = self.get().await?;
        rt.embed(audio).await
    }

    fn name(&self) -> &str {
        "lazy"
    }
}

/// 当前进程的可执行文件所在目录（拿不到就是 `None`）。
pub fn exe_dir() -> Option<PathBuf> {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
}

/// **模型根目录**的 env 后缀（读作 `NEURON_MODELS_DIR`）—— 内置权重的
/// **总目录**覆盖口。
///
/// ⚠️ **为什么必须有它**：桌面端把权重放在 exe 同级的 `models/` 下就行，
/// 但 **Android 上没有「exe 同级」这个位置**（`libencephalon.so` 在 APK 的
/// `lib/` 里，APK 的 assets 是压缩包里的条目，不能直接当文件读）⇒ ⇒
/// App 必须把 assets **拷到自己的 `filesDir/models`** ⇒ ⇒ 再由这一格
/// 把那个目录告诉 neuron。
pub const MODELS_DIR_ENV: &str = "MODELS_DIR";

/// 内置权重的候选位置（**纯函数**）—— 发行包把模型放在 exe 同级的 `models/` 下。
///
/// `models_dir`（`NEURON_MODELS_DIR`）**排在前面**：显式指路优先于
/// 「exe 同级」这个约定 —— 移动端那条路只有它能走通。
pub fn builtin_model_candidates(exe_dir: Option<&Path>, models_dir: Option<&Path>) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    if let Some(dir) = models_dir {
        out.push(dir.join(BUILTIN_MODEL_FILE));
    }
    if let Some(dir) = exe_dir {
        out.push(dir.join(BUILTIN_MODEL_DIR).join(BUILTIN_MODEL_FILE));
    }
    out
}

/// 解析声纹模型路径（**纯函数**）：环境变量 → 内置 → 报错。
///
/// `env` 指向 onnx 文件**或**含 `campplus.onnx` 的目录（两种都认）。
/// 报错里**列出看过哪些地方** —— 「没找到」必须说清「找过哪里」。
pub fn resolve_model_path(
    env: Option<&str>,
    exe_dir: Option<&Path>,
    models_dir: Option<&Path>,
) -> Result<PathBuf, String> {
    let mut tried: Vec<String> = Vec::new();
    if let Some(raw) = env.map(str::trim).filter(|s| !s.is_empty()) {
        let p = PathBuf::from(raw);
        let candidate = if p.is_dir() {
            p.join(BUILTIN_MODEL_FILE)
        } else {
            p
        };
        if candidate.is_file() {
            return Ok(candidate);
        }
        tried.push(format!("NEURON_SPK_MODEL_PATH={}", candidate.display()));
    }
    for cand in builtin_model_candidates(exe_dir, models_dir) {
        if cand.is_file() {
            return Ok(cand);
        }
        tried.push(cand.display().to_string());
    }
    Err(format!(
        "speaker embedding model not found (looked at: {})",
        if tried.is_empty() {
            "no candidate location".to_string()
        } else {
            tried.join(", ")
        }
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("neuron-spk-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("tmp dir");
        dir
    }

    #[test]
    fn the_builtin_candidate_is_models_campplus_next_to_the_exe() {
        let cands = builtin_model_candidates(Some(Path::new("/opt/shadow/bin")), None);
        assert_eq!(cands.len(), 1);
        assert!(cands[0].ends_with("models/campplus.onnx"), "{:?}", cands[0]);
    }

    #[test]
    fn no_exe_dir_means_no_builtin_candidate() {
        assert!(builtin_model_candidates(None, None).is_empty());
    }

    #[test]
    fn the_env_override_wins_over_the_builtin_location() {
        let dir = tmpdir("override");
        let file = dir.join("my-voiceprint.onnx");
        std::fs::write(&file, b"onnx").expect("write");
        let resolved = resolve_model_path(
            Some(file.to_str().expect("utf8")),
            Some(Path::new("/nonexistent/exe/dir")),
            None,
        )
        .expect("resolved");
        assert_eq!(resolved, file);
        std::fs::remove_file(&file).ok();
    }

    #[test]
    fn the_env_override_may_point_at_a_directory_holding_the_model() {
        let dir = tmpdir("dir-override");
        let file = dir.join(BUILTIN_MODEL_FILE);
        std::fs::write(&file, b"onnx").expect("write");
        let resolved =
            resolve_model_path(Some(dir.to_str().expect("utf8")), None, None).expect("resolved");
        assert_eq!(resolved, file);
        std::fs::remove_file(&file).ok();
    }

    #[test]
    fn the_builtin_location_is_used_when_the_env_is_absent() {
        let dir = tmpdir("builtin");
        let models = dir.join(BUILTIN_MODEL_DIR);
        std::fs::create_dir_all(&models).expect("models dir");
        let file = models.join(BUILTIN_MODEL_FILE);
        std::fs::write(&file, b"onnx").expect("write");
        let resolved = resolve_model_path(None, Some(dir.as_path()), None).expect("resolved");
        assert_eq!(resolved, file);
        std::fs::remove_file(&file).ok();
    }

    #[test]
    fn a_missing_model_names_every_place_it_looked() {
        let err = resolve_model_path(
            Some("/nope/missing.onnx"),
            Some(Path::new("/also/nope")),
            None,
        )
        .expect_err("must fail");
        assert!(err.contains("speaker embedding model not found"), "{err}");
        assert!(err.contains("/nope/missing.onnx"), "{err}");
        // ⚠️ **路径要按平台拼** —— Windows 上是 `\also\nope\models\campplus.onnx`
        // ⇒ ⇒ 写死 `"models/campplus.onnx"` 在 Windows 上必红（判据自己错了，
        // 不是产品错了 —— 这条判据是**别的会话**的红读数抓出来的）。
        let expected = Path::new("/also/nope")
            .join(BUILTIN_MODEL_DIR)
            .join(BUILTIN_MODEL_FILE)
            .display()
            .to_string();
        assert!(err.contains(&expected), "{err} (expected {expected})");
    }

    #[test]
    fn the_models_dir_override_wins_over_the_exe_sibling_layout() {
        let dir = tmpdir("models-dir");
        let models = dir.join("models");
        std::fs::create_dir_all(&models).expect("models dir");
        let file = models.join(BUILTIN_MODEL_FILE);
        std::fs::write(&file, b"onnx").expect("write");
        // exe 同级**没有**这个文件 ⇒ ⇒ 只有模型根目录这一条路能命中。
        let resolved = resolve_model_path(None, Some(dir.as_path()), Some(models.as_path()))
            .expect("resolved");
        assert_eq!(resolved, file);
        std::fs::remove_file(&file).ok();
    }

    #[test]
    fn a_missing_model_names_the_models_dir_it_looked_in() {
        let err =
            resolve_model_path(None, None, Some(Path::new("/models/here"))).expect_err("must fail");
        assert!(err.contains("speaker embedding model not found"), "{err}");
        let expected = Path::new("/models/here")
            .join(BUILTIN_MODEL_FILE)
            .display()
            .to_string();
        assert!(err.contains(&expected), "{err} (expected {expected})");
    }
}
