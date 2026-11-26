//! **人脸识别** —— 让「这是谁」有**事实**依据（而不是哈希出来的假向量）。
//!
//! ## 为什么是「内置模型」（用户 2026-11-25 拍板）
//!
//! > 「人脸模型也需要内置在枕叶中无需用户配置打包进安装包中」
//!
//! ⇒ ⇒ 权重解析**内置优先**：发行包把模型放在可执行文件同级的 `models/face/` 下
//! ⇒ ⇒ **用户零配置**；环境变量只是开发／CI 的覆盖口。
//!
//! ## 两个模型（都是 Apache-2.0）
//!
//! - **YuNet**（OpenCV Zoo，232 KB）—— 检脸 ＋ 5 个关键点；
//! - **MobileFaceNet**（Qualcomm AI Hub，378 KB ＋ 外部权重 3.9 MB）—— 128 维嵌入。
//!
//! ## 执行后端：tract（纯 Rust）
//!
//! ⇒ ⇒ **发行包不必带 `onnxruntime.dll`** —— 这是「打包进安装包」能落地的关键
//! ⇒ ⇒ （实测：SCRFD 全系列在 tract 上卡死在 FPN 的 `Resize`，YuNet 没有 Resize ⇒ 能跑）。

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

pub mod align;

#[cfg(feature = "face-embedding")]
pub mod mfn;

#[cfg(feature = "face-embedding")]
pub mod runtime;

#[cfg(feature = "face-embedding")]
pub mod yunet;

#[cfg(feature = "face-embedding")]
pub use runtime::FaceRuntime;

/// 人脸嵌入维度（MobileFaceNet ⇒ 128 维）。
pub const FACE_EMBEDDING_DIM: usize = 128;

/// 内置权重目录（**相对可执行文件**）：`models/face/`。
pub const BUILTIN_MODEL_DIR: &str = "models/face";

/// 内置权重文件名。
pub const DETECTOR_FILE: &str = "yunet.onnx";
pub const EMBEDDER_FILE: &str = "mobile_facenet.onnx";
/// 嵌入模型的**外部权重**（ONNX external data）—— 必须与模型同目录。
pub const EMBEDDER_DATA_FILE: &str = "mobile_facenet.data";

/// 一张脸在**原图**里的框。
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FaceBox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

/// 一张脸的嵌入（**事实**：这是从图像里量出来的，不是声称的）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceEmbedding {
    /// 脸框（原图坐标）。
    pub bbox: FaceBox,
    /// 检测分数。
    pub score: f32,
    /// 5 个关键点（原图坐标，arcface 顺序：左眼、右眼、鼻、左嘴角、右嘴角）。
    pub landmarks: Vec<[f32; 2]>,
    /// 128 维嵌入（L2 归一化 ⇒ 余弦相似度＝点积）。
    pub embedding: Vec<f32>,
}

/// 一张图里所有脸的嵌入。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceEmbeddingResult {
    pub faces: Vec<FaceEmbedding>,
    /// 维度（冗余一份，客户端不用自己数）。
    pub dim: usize,
}

/// **人脸运行时** —— 图片进，人脸嵌入出。
#[async_trait::async_trait]
pub trait FaceEmbeddingRuntime: Send + Sync {
    /// 一张图（JPEG/PNG 字节）→ 每张脸的嵌入。
    async fn embed_faces(&self, image: &[u8]) -> Result<FaceEmbeddingResult, String>;

    /// 运行时名字（注册表／日志用）。
    fn name(&self) -> &str;
}

/// **诚实链的最后一环**：没有内置模型（或 build 没编 `face-embedding`）时，
/// 端点**明确说不可用** —— 而不是返回零向量／随机向量冒充人脸。
pub struct UnavailableFaceEmbeddingRuntime {
    /// 为什么不可用（**原话**透给调用方）。
    pub reason: String,
}

#[async_trait::async_trait]
impl FaceEmbeddingRuntime for UnavailableFaceEmbeddingRuntime {
    async fn embed_faces(&self, _image: &[u8]) -> Result<FaceEmbeddingResult, String> {
        Err(self.reason.clone())
    }

    fn name(&self) -> &str {
        "unavailable"
    }
}

/// **懒加载**的人脸运行时：第一次调用时才开 ONNX 会话。
pub struct LazyFaceEmbeddingRuntime {
    cell: tokio::sync::OnceCell<std::result::Result<Arc<dyn FaceEmbeddingRuntime>, String>>,
}

impl Default for LazyFaceEmbeddingRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl LazyFaceEmbeddingRuntime {
    pub fn new() -> Self {
        Self {
            cell: tokio::sync::OnceCell::new(),
        }
    }

    async fn get(&self) -> Result<&Arc<dyn FaceEmbeddingRuntime>, String> {
        let cell = &self.cell;
        let slot = cell
            .get_or_init(|| async {
                #[cfg(feature = "face-embedding")]
                {
                    let env = crate::config::env_var("FACE_MODEL_DIR");
                    let exe = exe_dir();
                    let user = user_model_dir();
                    match resolve_face_models(env.as_deref(), exe.as_deref(), user.as_deref()) {
                        Ok(paths) => match FaceRuntime::open(&paths) {
                            Ok(rt) => Ok(Arc::new(rt) as Arc<dyn FaceEmbeddingRuntime>),
                            Err(e) => Err(format!("face model failed to load: {e}")),
                        },
                        Err(e) => Err(e),
                    }
                }
                #[cfg(not(feature = "face-embedding"))]
                {
                    Err("face recognition is not compiled in (build with --features face-embedding)".to_string())
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
impl FaceEmbeddingRuntime for LazyFaceEmbeddingRuntime {
    async fn embed_faces(&self, image: &[u8]) -> Result<FaceEmbeddingResult, String> {
        let rt = self.get().await?;
        rt.embed_faces(image).await
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

/// 开发期用的用户目录（`~/.neuron/models/face`）—— 发行包**不依赖**它。
pub fn user_model_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)?;
    Some(home.join(".neuron").join("models").join("face"))
}

/// 内置权重候选目录（**纯函数**）—— 发行包把模型放在 exe 同级的 `models/face/` 下。
///
/// ⚠️ **不止一处**：不同发行形态的「资源目录」位置不同 ⇒ ⇒ 候选按优先级列全，
/// 免得「权重明明在包里却找不到」：
///
/// - `<exe>/models/face`              —— Windows（资源与 exe 同级）；
/// - `<exe>/resources/models/face`    —— 某些安装器把资源塞进 `resources/`；
/// - `<exe>/../Resources/models/face` —— Linux deb（Tauri 的 `resource_dir()`）；
/// - `<exe>/../models/face`           —— macOS `Contents/` 布局。
pub fn builtin_model_dirs(exe_dir: Option<&Path>) -> Vec<PathBuf> {
    let Some(dir) = exe_dir else {
        return Vec::new();
    };
    let mut out = vec![dir.join(BUILTIN_MODEL_DIR)];
    out.push(dir.join("resources").join(BUILTIN_MODEL_DIR));
    if let Some(parent) = dir.parent() {
        if !parent.as_os_str().is_empty() {
            out.push(parent.join("Resources").join(BUILTIN_MODEL_DIR));
            out.push(parent.join(BUILTIN_MODEL_DIR));
        }
    }
    out
}

/// 解析出来的一组权重路径。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FaceModelPaths {
    /// 检脸模型（YuNet）。
    pub detector: PathBuf,
    /// 嵌入模型（MobileFaceNet）。
    pub embedder: PathBuf,
    /// 嵌入模型的外部权重（ONNX external data）—— 有就必须和模型同目录。
    pub embedder_data: Option<PathBuf>,
}

/// 解析人脸模型（**纯函数**）：env 覆盖 → 内置 → 用户目录 → 报错。
///
/// 报错里**列出看过哪些地方** —— 「没找到」必须说清「找过哪里」。
pub fn resolve_face_models(
    env_dir: Option<&str>,
    exe_dir: Option<&Path>,
    user_dir: Option<&Path>,
) -> Result<FaceModelPaths, String> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(raw) = env_dir.map(str::trim).filter(|s| !s.is_empty()) {
        candidates.push(PathBuf::from(raw));
    }
    candidates.extend(builtin_model_dirs(exe_dir));
    if let Some(u) = user_dir {
        candidates.push(u.to_path_buf());
    }

    let mut tried: Vec<String> = Vec::new();
    for dir in candidates {
        let detector = dir.join(DETECTOR_FILE);
        let embedder = dir.join(EMBEDDER_FILE);
        if detector.is_file() && embedder.is_file() {
            let data = dir.join(EMBEDDER_DATA_FILE);
            if references_external_data(&embedder) && !data.is_file() {
                return Err(format!(
                    "face embedder model {} uses external data but {} is missing",
                    embedder.display(),
                    data.display()
                ));
            }
            let embedder_data = if data.is_file() { Some(data) } else { None };
            return Ok(FaceModelPaths {
                detector,
                embedder,
                embedder_data,
            });
        }
        tried.push(format!(
            "{} (need {} + {})",
            dir.display(),
            DETECTOR_FILE,
            EMBEDDER_FILE
        ));
    }
    Err(format!(
        "face model not found (looked at: {})",
        if tried.is_empty() {
            "no candidate location".to_string()
        } else {
            tried.join(", ")
        }
    ))
}

/// 这个 ONNX 是不是「外部权重」布局（`*.data` 与模型同目录）。
fn references_external_data(onnx: &Path) -> bool {
    match std::fs::read(onnx) {
        Ok(bytes) => bytes.windows(5).any(|w| w == b".data"),
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("neuron-face-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("tmp dir");
        dir
    }

    fn fake_models(dir: &Path) {
        std::fs::write(dir.join(DETECTOR_FILE), b"onnx").expect("detector");
        std::fs::write(dir.join(EMBEDDER_FILE), b"onnx").expect("embedder");
    }

    #[test]
    fn the_builtin_candidate_is_models_face_next_to_the_exe() {
        let cands = builtin_model_dirs(Some(Path::new("/opt/shadow/bin")));
        assert!(!cands.is_empty());
        let expected = Path::new("/opt/shadow/bin")
            .join(BUILTIN_MODEL_DIR)
            .display()
            .to_string();
        assert_eq!(cands[0].display().to_string(), expected);
    }

    #[test]
    fn every_bundle_layout_is_a_candidate() {
        // 发行形态不止一种 ⇒ ⇒ 资源目录位置也不止一种 ⇒ ⇒ 候选必须列全，
        // 否则会出现「权重明明在包里，运行时却说没有」。
        //
        // ⚠️ 期望路径**必须用 join 构造**，不许硬编码 `bin/models/face` 这种
        // POSIX 后缀 —— Windows 上 join 用 `\` ⇒ ⇒ 硬编码后缀**恒不匹配**
        // ⇒ ⇒ 这条判据第一次是在 Windows 上红的（红读数由别的会话抓出来）。
        let exe = Path::new("/opt/shadow/bin");
        let parent = exe.parent().expect("exe has a parent");
        let expected = [
            exe.join(BUILTIN_MODEL_DIR),
            exe.join("resources").join(BUILTIN_MODEL_DIR),
            parent.join("Resources").join(BUILTIN_MODEL_DIR),
            parent.join(BUILTIN_MODEL_DIR),
        ];
        let cands = builtin_model_dirs(Some(exe));
        for want in &expected {
            assert!(cands.contains(want), "missing {want:?} in {cands:?}");
        }
    }

    #[test]
    fn no_exe_dir_means_no_builtin_candidate() {
        assert!(builtin_model_dirs(None).is_empty());
    }

    #[test]
    fn the_env_override_wins_over_the_builtin_location() {
        let dir = tmpdir("override");
        fake_models(&dir);
        let builtin = tmpdir("override-builtin");
        fake_models(&builtin);
        let resolved = resolve_face_models(
            Some(dir.to_str().expect("utf8")),
            Some(builtin.as_path()),
            None,
        )
        .expect("resolved");
        assert_eq!(resolved.detector, dir.join(DETECTOR_FILE));
    }

    #[test]
    fn the_builtin_location_is_used_when_the_env_is_absent() {
        let dir = tmpdir("builtin");
        let models = dir.join(BUILTIN_MODEL_DIR);
        std::fs::create_dir_all(&models).expect("models dir");
        fake_models(&models);
        let resolved = resolve_face_models(None, Some(dir.as_path()), None).expect("resolved");
        assert_eq!(resolved.detector, models.join(DETECTOR_FILE));
        assert_eq!(resolved.embedder, models.join(EMBEDDER_FILE));
    }

    #[test]
    fn a_missing_model_names_every_place_it_looked() {
        let err = resolve_face_models(
            Some("/nope/missing"),
            Some(Path::new("/also/nope")),
            Some(Path::new("/third/nope")),
        )
        .expect_err("must fail");
        assert!(err.contains("face model not found"), "{err}");
        assert!(err.contains("/nope/missing"), "{err}");
        // ⚠️ **路径按平台拼** —— Windows 上是反斜杠。
        let expected = Path::new("/also/nope")
            .join(BUILTIN_MODEL_DIR)
            .display()
            .to_string();
        assert!(err.contains(&expected), "{err} (expected {expected})");
    }

    #[test]
    fn an_embedder_with_external_data_needs_its_data_file() {
        let dir = tmpdir("external");
        std::fs::create_dir_all(&dir).expect("dir");
        std::fs::write(dir.join(DETECTOR_FILE), b"onnx").expect("detector");
        // 模型里出现 ".data" ⇒ 外部权重布局。
        std::fs::write(dir.join(EMBEDDER_FILE), b"onnx external .data here").expect("embedder");
        let err = resolve_face_models(Some(dir.to_str().expect("utf8")), None, None)
            .expect_err("must fail");
        assert!(err.contains("external data"), "{err}");
        assert!(err.contains(EMBEDDER_DATA_FILE), "{err}");
    }

    #[test]
    fn external_data_present_is_reported_as_the_data_path() {
        let dir = tmpdir("external-ok");
        std::fs::create_dir_all(&dir).expect("dir");
        std::fs::write(dir.join(DETECTOR_FILE), b"onnx").expect("detector");
        std::fs::write(dir.join(EMBEDDER_FILE), b"onnx external .data here").expect("embedder");
        std::fs::write(dir.join(EMBEDDER_DATA_FILE), b"weights").expect("data");
        let resolved =
            resolve_face_models(Some(dir.to_str().expect("utf8")), None, None).expect("resolved");
        assert_eq!(resolved.embedder_data, Some(dir.join(EMBEDDER_DATA_FILE)));
    }
}
