//! **声纹内置那条链的判据**（集成测试）。
//!
//! ⚠️ **为什么放集成测试而不是单元测试**：单元测试住在 lib 的 `cfg(test)`
//! 里 ⇒ ⇒ **别的会话正在飞的面**（此刻是 `src/server/capabilities.rs` 的
//! 一个 move 之后的借用）会让**整个 lib 的测试 target** 编译不过 ⇒ ⇒
//! 这一格必须能在**不碰别人在飞面**的前提下跑出读数 ⇒ ⇒ 走集成测试
//! （它只依赖 lib 的正常构建，不编 lib 的 `cfg(test)` 代码）。
//!
//! 覆盖：模型根目录覆盖口（`NEURON_MODELS_DIR`）／ env 覆盖的优先级 ／
//! 「没找到」必须说清**找过哪里** ／ 太短音频必须拒 ／ 坏权重必须点名。

use std::path::{Path, PathBuf};

use neuron_lib::spk::resolve_model_path;

fn tmpdir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("neuron-spk-it-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("tmp dir");
    dir
}

#[test]
fn the_models_dir_override_is_honored_before_the_exe_sibling_layout() {
    let dir = tmpdir("models-dir");
    let models = dir.join("models");
    std::fs::create_dir_all(&models).expect("models dir");
    let file = models.join("campplus.onnx");
    std::fs::write(&file, b"onnx").expect("write");
    // exe 同级**没有**这个文件 ⇒ ⇒ 只有模型根目录这一条路能命中。
    let resolved =
        resolve_model_path(None, Some(dir.as_path()), Some(models.as_path())).expect("resolved");
    assert_eq!(resolved, file);
    std::fs::remove_file(&file).ok();
}

#[test]
fn the_env_override_still_wins_over_the_models_dir() {
    let dir = tmpdir("env-wins");
    let models = dir.join("models");
    std::fs::create_dir_all(&models).expect("models dir");
    let in_models = models.join("campplus.onnx");
    std::fs::write(&in_models, b"onnx").expect("write");
    let explicit = dir.join("explicit.onnx");
    std::fs::write(&explicit, b"onnx").expect("write");
    let resolved = resolve_model_path(
        Some(explicit.to_str().expect("utf8")),
        Some(dir.as_path()),
        Some(models.as_path()),
    )
    .expect("resolved");
    assert_eq!(resolved, explicit);
    std::fs::remove_file(&in_models).ok();
    std::fs::remove_file(&explicit).ok();
}

#[test]
fn a_missing_model_names_every_place_it_looked_including_the_models_dir() {
    let err =
        resolve_model_path(None, None, Some(Path::new("/models/here"))).expect_err("must fail");
    assert!(err.contains("speaker embedding model not found"), "{err}");
    let expected = Path::new("/models/here")
        .join("campplus.onnx")
        .display()
        .to_string();
    assert!(err.contains(&expected), "{err} (expected {expected})");
}

#[cfg(feature = "speaker-embedding")]
#[test]
fn too_short_audio_is_refused_before_any_inference() {
    let err =
        neuron_lib::spk::campplus::ensure_usable_length(&[0.0f32; 100]).expect_err("must refuse");
    assert!(err.contains("too short"), "{err}");
}

#[cfg(feature = "speaker-embedding")]
#[test]
fn a_non_onnx_weight_fails_to_load_and_names_the_path() {
    let dir = tmpdir("bad-weight");
    let file = dir.join("not-a-model.onnx");
    std::fs::write(&file, b"definitely not onnx").expect("write");
    let err = neuron_lib::spk::campplus::CampplusExtractor::open(&file)
        .err()
        .expect("must fail");
    assert!(err.contains("campplus load"), "{err}");
    assert!(err.contains("not-a-model.onnx"), "{err}");
    std::fs::remove_file(&file).ok();
}
