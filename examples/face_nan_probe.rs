//! YuNet 输出体检 —— NaN/inf 从哪来？

use std::path::PathBuf;

use tract_onnx::prelude::*;

fn stats(name: &str, v: &[f32]) {
    let nan = v.iter().filter(|x| x.is_nan()).count();
    let inf = v.iter().filter(|x| x.is_infinite()).count();
    let mn = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let mx = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let neg = v.iter().filter(|x| **x < 0.0).count();
    let head: Vec<String> = v.iter().take(4).map(|x| format!("{:.6}", x)).collect();
    println!(
        "{name}: n={} nan={} inf={} neg={} min={:.6} max={:.6} head=[{}]",
        v.len(), nan, inf, neg, mn, mx, head.join(", ")
    );
}

fn main() {
    let ck = PathBuf::from(std::env::var("FACE_CKPT_DIR").unwrap_or_else(|_| {
        "C:\\Projects\\iBLOC\\neuron\\.local\\face-ckpt".to_string()
    }));
    let path = ck.join("builtin").join("yunet.onnx");

    // A) 优化后的模型
    let model = tract_onnx::onnx()
        .model_for_path(&path)
        .expect("load")
        .into_optimized()
        .expect("optimize")
        .into_runnable()
        .expect("runnable");
    let bytes = std::fs::read(ck.join("obama.jpg")).expect("read");
    let img = image::load_from_memory(&bytes).expect("decode");
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let data = preprocess(rgb.as_raw(), w, h);
    let t = Tensor::from_shape(&[1, 3, 640, 640], &data).expect("tensor");
    let out = model.run(tvec!(t.into())).expect("run");
    for (i, o) in out.iter().enumerate() {
        let v: Vec<f32> = o.to_array_view::<f32>().expect("view").iter().copied().collect();
        stats(&format!("opt out[{i}]"), &v);
    }
}

fn preprocess(rgb: &[u8], w: u32, h: u32) -> Vec<f32> {
    let n = 640usize;
    let sx = w as f32 / n as f32;
    let sy = h as f32 / n as f32;
    let mut out = vec![0.0f32; 3 * n * n];
    for oy in 0..n {
        let py = (((oy as f32 + 0.5) * sy) - 0.5).floor().max(0.0) as u32;
        let py = py.min(h - 1);
        for ox in 0..n {
            let px = (((ox as f32 + 0.5) * sx) - 0.5).floor().max(0.0) as u32;
            let px = px.min(w - 1);
            let base = ((py * w + px) * 3) as usize;
            let o = ox * n + oy;
            out[o] = rgb[base + 2] as f32;
            out[n * n + o] = rgb[base + 1] as f32;
            out[2 * n * n + o] = rgb[base] as f32;
        }
    }
    out
}