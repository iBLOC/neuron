//! 人脸模型 **真机探针** —— 用真权重 + 真照片给出读数。
//!
//! 读三件事：
//! 1. **检得出吗** —— YuNet 在真人照片上的框与关键点；
//! 2. **分得开吗** —— 三个**不同人**的嵌入余弦；
//! 3. **稳吗** —— 同一张脸换裁剪/尺度后，余弦应当仍然很高。

use std::path::{Path, PathBuf};

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len().min(b.len()) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na <= 0.0 || nb <= 0.0 {
        return f32::NAN;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn main() {
    let ck = PathBuf::from(std::env::var("FACE_CKPT_DIR").unwrap_or_else(|_| {
        "C:\\Projects\\iBLOC\\neuron\\.local\\face-ckpt".to_string()
    }));
    let model_dir = std::env::var("FACE_MODEL_DIR").unwrap_or_else(|_| {
        ck.join("builtin").display().to_string()
    });

    let paths = neuron_lib::face::resolve_face_models(Some(&model_dir), None, None).expect("resolve models");
    println!("detector = {}", paths.detector.display());
    println!("embedder = {}", paths.embedder.display());
    println!("ext-data  = {:?}", paths.embedder_data.as_ref().map(|p| p.display().to_string()));

    let rt = neuron_lib::face::FaceRuntime::open(&paths).expect("open runtime");

    let mut all: Vec<(String, Vec<f32>)> = Vec::new();
    for (label, file) in [
        ("selfie", "largest_selfie.jpg"),
        ("obama", "obama.jpg"),
        ("einstein", "einstein.jpg"),
    ] {
        let bytes = std::fs::read(ck.join(file)).expect("read image");
        let t0 = std::time::Instant::now();
        let res = rt.embed_faces_sync(&bytes).expect("embed");
        let ms = t0.elapsed().as_millis();
        println!("== {label} ({file}) faces={} elapsed={ms}ms", res.faces.len());
        for f in &res.faces {
            let norm = f.embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
            println!("   score={:.3} bbox=({:.1},{:.1},{:.1},{:.1}) lm0={:?} dim={} l2={:.4} head={:?}",
                f.score, f.bbox.x, f.bbox.y, f.bbox.w, f.bbox.h, &f.landmarks[0], f.embedding.len(), norm,
                &f.embedding[..5.min(f.embedding.len())]);
            all.push((label.to_string(), f.embedding.clone()));
        }
    }

    println!("-- 不同人之间的余弦（应当**低**）--");
    for i in 0..all.len() {
        for j in (i + 1)..all.len() {
            println!("   {} vs {} = {:.4}", all[i].0, all[j].0, cosine(&all[i].1, &all[j].1));
        }
    }

    println!("-- 同一张脸换尺度（中心裁 60%）后与原图的余弦（应当**高**）--");
    let bytes = std::fs::read(ck.join("largest_selfie.jpg")).expect("read");
    let img = image::load_from_memory(&bytes).expect("decode");
    let (w, h) = (img.width(), img.height());
    let cw = (w as f32 * 0.6) as u32;
    let ch = (h as f32 * 0.6) as u32;
    // 手动裁（不赌 image 0.25 的 crop API 形状）。
    let rgb = img.to_rgb8();
    let (ox, oy) = ((w - cw) / 2, (h - ch) / 2);
    let mut crop_buf = vec![0u8; (cw * ch * 3) as usize];
    for y in 0..ch {
        for x in 0..cw {
            let p = rgb.get_pixel(x + ox, y + oy);
            let di = ((y * cw + x) * 3) as usize;
            crop_buf[di] = p[0];
            crop_buf[di + 1] = p[1];
            crop_buf[di + 2] = p[2];
        }
    }
    let sub = image::RgbImage::from_raw(cw, ch, crop_buf).expect("crop image");
    let mut buf = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(sub)
        .write_to(&mut buf, image::ImageFormat::Png)
        .expect("png");
    let res = rt.embed_faces_sync(&buf.into_inner()).expect("embed crop");
    println!("   cropped faces={}", res.faces.len());
    for f in &res.faces {
        if let Some((_, orig)) = all.iter().find(|(l, _)| l == "selfie") {
            println!("   selfie(full) vs selfie(crop) = {:.4}", cosine(orig, &f.embedding));
        }
    }
    println!("PROBE DONE");
}
