//! 内置权重**布局**真机探针 —— 「权重放在发行布局的哪个位置，neuron 真能找到吗」。
//!
//! 目的：resolve_face_models 的候选列表只证明了「候选存在」，
//! 没证明「**每个候选位真能用**」⇒ ⇒ 这一格要拿**真权重**逐个布局试。
//!
//! 做法：把真权重复制到 4 种发行布局各自的目录 ⇒ ⇒ 调 resolve_face_models
//! ⇒ ⇒ 打印它选中的那个候选 ⇒ ⇒ 验完清掉（下一个布局）。
//!
//! 布局（对应 neuron 的 builtin_model_dirs()）：
//!   1. <exe>/models/face               —— Windows（资源与 exe 同级）
//!   2. <exe>/resources/models/face     —— 某些安装器把资源塞进 resources/
//!   3. <exe>/../Resources/models/face  —— Linux deb（Tauri 的 resource_dir()）
//!   4. <exe>/../models/face            —— macOS Contents 布局

use std::path::{Path, PathBuf};

fn copy_weights(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).expect("create dir");
    for f in ["yunet.onnx", "mobile_facenet.onnx", "mobile_facenet.data"] {
        std::fs::copy(src.join(f), dst.join(f)).expect("copy weight");
    }
}

fn main() {
    let ck = PathBuf::from(std::env::var("FACE_CKPT_DIR").unwrap_or_else(|_| {
        "C:\\Projects\\iBLOC\\neuron\\.local\\face-ckpt".to_string()
    }));
    let weights = ck.join("builtin");
    let root = ck.join("layout-probe");
    let exe_dir = root.join("bin");

    // 先清干净（上一轮可能留了东西）。
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&exe_dir).expect("create exe dir");

    let layouts: Vec<(&str, PathBuf)> = vec![
        ("1. <exe>/models/face", exe_dir.join("models").join("face")),
        (
            "2. <exe>/resources/models/face",
            exe_dir.join("resources").join("models").join("face"),
        ),
        (
            "3. <exe>/../Resources/models/face",
            root.join("Resources").join("models").join("face"),
        ),
        ("4. <exe>/../models/face", root.join("models").join("face")),
    ];

    for (name, dir) in &layouts {
        copy_weights(&weights, dir);
        match neuron_lib::face::resolve_face_models(None, Some(&exe_dir), None) {
            Ok(paths) => {
                let picked = paths.detector.parent().expect("parent");
                println!(
                    "OK   {name} => picked {}",
                    picked.display()
                );
                println!("     detector={}", paths.detector.display());
                println!(
                    "     embedder_data={:?}",
                    paths.embedder_data.as_ref().map(|p| p.display().to_string())
                );
            }
            Err(e) => println!("FAIL {name} => {e}"),
        }
        let _ = std::fs::remove_dir_all(dir);
    }
    let _ = std::fs::remove_dir_all(&root);
    println!();
    println!("（4 种发行布局各放一次真权重 => 每次都清掉 => 布局之间不互相顶替）");
}