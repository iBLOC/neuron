//! 人脸 **真机读数** 探针 —— 与 Python 参考实现（onnxruntime + cv2）逐项对照。
//!
//! 输出：
//! - `IMG|file|WxH|detect_faces=N|embed_faces=M|ms=T`
//! - `BOX|file|idx|score|x,y,w,h|lm0x,lm0y`   ← **640 空间**（与 cv2 同一坐标）
//! - `EMB|file|idx|v0,v1,v2,v3,v4,v5`        ← 128 维里的前 6 维
//! - 对每张图把**对齐后的 112×112 裁剪**写成 PNG（供 Python 逐像素对拍）

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
    println!("detector = {}", paths.detector.display());
    println!("embedder = {}", paths.embedder.display());

    let detector =
        neuron_lib::face::yunet::YunetDetector::open(&paths.detector).expect("open yunet");
    let rt = neuron_lib::face::FaceRuntime::open(&paths).expect("open runtime");

    let mut files: Vec<(String, PathBuf)> = Vec::new();
    for f in ["obama.jpg", "einstein.jpg", "largest_selfie.jpg"] {
        files.push((f.to_string(), ck.join(f)));
    }
    let lfw = ck.join("lfw");
    if lfw.is_dir() {
        let mut ps: Vec<PathBuf> = std::fs::read_dir(&lfw)
            .expect("read lfw dir")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .collect();
        ps.sort();
        for p in ps.into_iter().take(40) {
            if let Some(n) = p.file_name().and_then(|s| s.to_str()) {
                files.push((n.to_string(), p));
            }
        }
    }

    let out_dir = ck.join("rust_out");
    let _ = std::fs::create_dir_all(&out_dir);
    let mut dump: Vec<(String, Vec<serde_json::Value>)> = Vec::new();

    for (name, path) in files {
        let bytes = std::fs::read(&path).expect("read image");
        let img = image::load_from_memory(&bytes).expect("decode");
        let rgb = img.to_rgb8();
        let (w, h) = rgb.dimensions();
        let t0 = std::time::Instant::now();
        let faces = detector.detect(rgb.as_raw(), w, h).expect("detect");
        let dt = t0.elapsed().as_millis();
        // 第二遍：把「首帧一次性开销」与「稳态单张耗时」分开。
        let t1 = std::time::Instant::now();
        let _ = detector.detect(rgb.as_raw(), w, h).expect("detect2");
        let dt2 = t1.elapsed().as_millis();
        let res = rt.embed_faces_sync(&bytes).expect("embed");
        println!(
            "IMG|{}|{}x{}|detect_faces={}|embed_faces={}|ms={}|ms2={}",
            name, w, h, faces.len(), res.faces.len(), dt, dt2
        );
        for (i, f) in faces.iter().enumerate() {
            println!(
                "BOX|{}|{}|{:.6}|{:.4},{:.4},{:.4},{:.4}|{:.4},{:.4}",
                name, i, f.score, f.x1, f.y1, f.x2 - f.x1, f.y2 - f.y1, f.landmarks[0][0], f.landmarks[0][1]
            );
        }
        for (i, e) in res.faces.iter().enumerate() {
            let head: Vec<String> = e
                .embedding
                .iter()
                .take(6)
                .map(|v| format!("{:.6}", v))
                .collect();
            println!("EMB|{}|{}|{}", name, i, head.join(","));
        }

        // 落一份完整 JSON（供 Python 逐项对拍）。
        let mut arr: Vec<serde_json::Value> = Vec::new();
        for (i, e) in res.faces.iter().enumerate() {
            let raw = faces.get(i);
            arr.push(serde_json::json!({
                "score": e.score,
                "bbox_orig": [e.bbox.x, e.bbox.y, e.bbox.w, e.bbox.h],
                "landmarks_orig": e.landmarks,
                "box640": raw.map(|r| [r.x1, r.y1, r.x2 - r.x1, r.y2 - r.y1]),
                "lm640": raw.map(|r| r.landmarks),
                "emb": e.embedding,
            }));
        }
        dump.push((name.clone(), arr));

        // 把第一张脸的**对齐后 112×112 裁剪**落成 PNG（Python 侧逐像素对拍）。
        if let Some(f) = faces.first() {
            let sx = w as f32 / 640.0;
            let sy = h as f32 / 640.0;
            let pts = neuron_lib::face::runtime::reorder_landmarks(f.landmarks)
                .map(|p| [p[0] * sx, p[1] * sy]);
            if let Some(tf) =
                neuron_lib::face::align::similarity_transform(&pts, &neuron_lib::face::align::TEMPLATE_112)
            {
                println!(
                    "TF|{}|{:.6},{:.6},{:.6},{:.6},{:.4},{:.4}|lm={}",
                    name,
                    tf.a,
                    tf.b,
                    tf.c,
                    tf.d,
                    tf.tx,
                    tf.ty,
                    pts.iter()
                        .map(|p| format!("{:.3},{:.3}", p[0], p[1]))
                        .collect::<Vec<_>>()
                        .join(";")
                );
                if let Some(crop) =
                    neuron_lib::face::align::warp_crop(rgb.as_raw(), w, h, &tf, 112)
                {
                    let im = image::RgbImage::from_raw(112, 112, crop).expect("crop image");
                    let dest = out_dir.join(format!("{}.png", name));
                    im.save(&dest).expect("save crop");
                    println!("CROP|{}|{}", name, dest.display());
                }
            }
        }
    }
    let json = serde_json::to_string_pretty(&dump).expect("json");
    std::fs::write(out_dir.join("rust_face.json"), json).expect("write json");
    println!("PROBE DONE");
}