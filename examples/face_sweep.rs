//! YuNet 输出语义诊断 —— 在真照片上把四组张量的峰值 ＋ 对应框打出来。

use std::path::PathBuf;

use tract_onnx::prelude::*;

macro_rules! flat {
    ($o:expr) => {
        $o.to_array_view::<f32>().expect("view").iter().copied().collect::<Vec<f32>>()
    };
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

    for file in ["largest_selfie.jpg", "obama.jpg", "einstein.jpg"] {
        let bytes = std::fs::read(dir.join(file)).expect("read");
        let img = image::load_from_memory(&bytes).expect("decode");
        let (w, h) = (img.width(), img.height());
        let raw = img.to_rgb8().into_raw();
        println!("== {file} {w}x{h}");
        let mut data = vec![0.0f32; 3 * 640 * 640];
        for oy in 0..640u32 {
            let sy = (((oy as f32 + 0.5) * h as f32 / 640.0).floor().max(0.0) as u32).min(h - 1);
            for ox in 0..640u32 {
                let sx = (((ox as f32 + 0.5) * w as f32 / 640.0).floor().max(0.0) as u32).min(w - 1);
                let base = ((sy * w + sx) * 3) as usize;
                let o = (ox * 640 + oy) as usize;
                data[o] = raw[base] as f32 / 255.0;
                data[640 * 640 + o] = raw[base + 1] as f32 / 255.0;
                data[2 * 640 * 640 + o] = raw[base + 2] as f32 / 255.0;
            }
        }
        let t = Tensor::from_shape(&[1, 3, 640, 640], &data).expect("tensor");
        let out = model.run(tvec!(t.into())).expect("run");
        let g1 = flat!(out[0]);
        let g2 = flat!(out[3]);
        let bbox = flat!(out[6]);
        let kps = flat!(out[9]);
        let n = g1.len();
        let mut best = 0usize;
        let mut best_v = f32::MIN;
        for i in 0..n {
            if g1[i] > best_v {
                best_v = g1[i];
                best = i;
            }
        }
        let g2max = g2.iter().copied().fold(f32::MIN, f32::max);
        println!("   g1.max={best_v:.4} @ anchor {best}   g2@same={:.4}  g2.max={g2max:.4}", g2[best]);
        println!("   bbox@anchor = {:?}", &bbox[best * 4..best * 4 + 4]);
        println!("   kps@anchor  = {:?}", &kps[best * 10..best * 10 + 10]);
        let g1b = flat!(out[1]);
        let bb = flat!(out[7]);
        let kb = flat!(out[10]);
        let mut bestb = 0usize;
        let mut bestbv = f32::MIN;
        for i in 0..g1b.len() {
            if g1b[i] > bestbv {
                bestbv = g1b[i];
                bestb = i;
            }
        }
        println!("   L1 g1.max={bestbv:.4} @ {bestb}  bbox={:?} kps={:?}", &bb[bestb * 4..bestb * 4 + 4],
            &kb[bestb * 10..bestb * 10 + 10]);
    }
}
