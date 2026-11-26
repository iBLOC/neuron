//! 人脸检脸框**全量 dump** 探针（研究用）—— 只解码，不做嵌入。
//!
//! 目的：把「Rust 侧检出的每一张脸（640 空间的浮点框 + 分数）」原样打出来，
//! 好与 Python 正控（cv2 / onnxruntime）逐框对拍，定位 NMS 口径差异。
//!
//! 用法：cargo run --example face_boxes_dump -- <图片路径...>

use std::path::PathBuf;

fn main() {
    let ck = PathBuf::from(std::env::var("FACE_CKPT_DIR").unwrap_or_else(|_| {
        "C:\\Projects\\iBLOC\\neuron\\.local\\face-ckpt".to_string()
    }));
    let model_dir = std::env::var("FACE_MODEL_DIR").unwrap_or_else(|_| {
        ck.join("builtin").display().to_string()
    });
    let paths =
        neuron_lib::face::resolve_face_models(Some(&model_dir), None, None).expect("resolve models");

    let detector = neuron_lib::face::yunet::YunetDetector::open(&paths.detector).expect("open yunet");

    let args: Vec<String> = std::env::args().skip(1).collect();
    let files: Vec<PathBuf> = if args.is_empty() {
        vec![ck.join("largest_selfie.jpg")]
    } else {
        args.iter().map(PathBuf::from).collect()
    };

    for path in files {
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        let bytes = std::fs::read(&path).expect("read image");
        let img = image::load_from_memory(&bytes).expect("decode");
        let rgb = img.to_rgb8();
        let (w, h) = rgb.dimensions();
        let faces = detector.detect(rgb.as_raw(), w, h).expect("detect");
        println!("IMG|{name}|{w}x{h}|detect_faces={}", faces.len());
        for (i, f) in faces.iter().enumerate() {
            println!(
                "BOX|{name}|{i}|{:.9}|{:.6},{:.6},{:.6},{:.6}",
                f.score, f.x1, f.y1, f.w, f.h
            );
        }
    }
}
