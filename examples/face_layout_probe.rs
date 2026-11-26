//! YuNet 输入布局矩阵 —— NCHW/NHWC × RGB/BGR × 值域，看哪种给出**有空间结构**的分数图。

use std::path::PathBuf;

use tract_onnx::prelude::*;

macro_rules! flat {
    ($o:expr) => {
        $o.to_array_view::<f32>().expect("view").iter().copied().collect::<Vec<f32>>()
    };
}

fn build(raw: &[u8], w: u32, h: u32, nhwc: bool, bgr: bool, scaled: bool) -> Vec<f32> {
    let mut data = vec![0.0f32; 3 * 640 * 640];
    let px = |x: u32, y: u32, ch: usize| -> f32 {
        let base = ((y * w + x) * 3) as usize;
        let v = if bgr { raw[base + 2 - ch] } else { raw[base + ch] } as f32;
        if scaled { (v - 127.5) / 128.0 } else { v / 255.0 }
    };
    for oy in 0..640usize {
        let sy = (((oy as f32 + 0.5) * h as f32 / 640.0).floor().max(0.0) as u32).min(h - 1);
        for ox in 0..640usize {
            let sx = (((ox as f32 + 0.5) * w as f32 / 640.0).floor().max(0.0) as u32).min(w - 1);
            for ch in 0..3usize {
                let v = px(sx, sy, ch);
                let o = if nhwc {
                    (oy * 640 + ox) * 3 + ch
                } else {
                    ch * 640 * 640 + ox * 640 + oy
                };
                data[o] = v;
            }
        }
    }
    data
}

fn main() {
    let dir = PathBuf::from(std::env::var("FACE_CKPT_DIR").unwrap_or_else(|_| {
        "C:\\Projects\\iBLOC\\neuron\\.local\\face-ckpt".to_string()
    }));
    let model = tract_onnx::onnx()
        .model_for_path(dir.join("builtin").join("yunet.onnx"))
        .expect("load")
        .into_optimized()
        .expect("optimize")
        .into_runnable()
        .expect("runnable");

    let bytes = std::fs::read(dir.join("obama.jpg")).expect("read");
    let img = image::load_from_memory(&bytes).expect("decode");
    let (w, h) = (img.width(), img.height());
    let raw = img.to_rgb8().into_raw();

    for nhwc in [false, true] {
        for bgr in [false, true] {
            for scaled in [false, true] {
                let data = build(&raw, w, h, nhwc, bgr, scaled);
                let shape: Vec<usize> = if nhwc {
                    vec![1, 640, 640, 3]
                } else {
                    vec![1, 3, 640, 640]
                };
                let t = match Tensor::from_shape(&shape, &data) {
                    Ok(t) => t,
                    Err(e) => {
                        println!("layout nhwc={nhwc} bgr={bgr} scaled={scaled} :: tensor FAIL {e}");
                        continue;
                    }
                };
                let out = match model.run(tvec!(t.into())) {
                    Ok(o) => o,
                    Err(e) => {
                        println!("layout nhwc={nhwc} bgr={bgr} scaled={scaled} :: run FAIL {e}");
                        continue;
                    }
                };
                let g1 = flat!(out[0]);
                let g2 = flat!(out[3]);
                let mut best = 0usize;
                let mut best_v = f32::MIN;
                let mut sorted = g1.clone();
                sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                for i in 0..g1.len() {
                    if g1[i] > best_v {
                        best_v = g1[i];
                        best = i;
                    }
                }
                let median = sorted[sorted.len() / 2];
                let g2max = g2.iter().copied().fold(f32::MIN, f32::max);
                println!("nhwc={nhwc} bgr={bgr} scaled={scaled}: g1 max={best_v:.4} @row={} col={} | median={median:.4} | g2max={g2max:.4} | top3={:?}",
                    best / 80, best % 80,
                    (0..3).map(|k| format!("{:.3}", sorted[k])).collect::<Vec<_>>());
            }
        }
    }
}
