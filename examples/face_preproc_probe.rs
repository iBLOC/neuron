//! YuNet 预处理对照 —— 黑图 / 白图 / RGB 真图 / BGR 真图，看输出怎么变。

use std::path::PathBuf;

use tract_onnx::prelude::*;

macro_rules! flat {
    ($o:expr) => {
        $o.to_array_view::<f32>().expect("view").iter().copied().collect::<Vec<f32>>()
    };
}

fn build(raw: &[u8], w: u32, h: u32, bgr: bool) -> Vec<f32> {
    let mut data = vec![0.0f32; 3 * 640 * 640];
    for oy in 0..640u32 {
        let sy = (((oy as f32 + 0.5) * h as f32 / 640.0).floor().max(0.0) as u32).min(h - 1);
        for ox in 0..640u32 {
            let sx = (((ox as f32 + 0.5) * w as f32 / 640.0).floor().max(0.0) as u32).min(w - 1);
            let base = ((sy * w + sx) * 3) as usize;
            let o = (ox * 640 + oy) as usize;
            if bgr {
                data[o] = raw[base + 2] as f32 / 255.0;
                data[640 * 640 + o] = raw[base + 1] as f32 / 255.0;
                data[2 * 640 * 640 + o] = raw[base] as f32 / 255.0;
            } else {
                data[o] = raw[base] as f32 / 255.0;
                data[640 * 640 + o] = raw[base + 1] as f32 / 255.0;
                data[2 * 640 * 640 + o] = raw[base + 2] as f32 / 255.0;
            }
        }
    }
    data
}

fn report(tag: &str, model: &TypedRunnableModel<TypedModel>, data: Vec<f32>) {
    let t = Tensor::from_shape(&[1, 3, 640, 640], &data).expect("tensor");
    let out = model.run(tvec!(t.into())).expect("run");
    let g1 = flat!(out[0]);
    let bbox = flat!(out[6]);
    let mut best = 0usize;
    let mut best_v = f32::MIN;
    for i in 0..g1.len() {
        if g1[i] > best_v {
            best_v = g1[i];
            best = i;
        }
    }
    let mut idx: Vec<usize> = (0..g1.len()).collect();
    idx.sort_by(|a, b| g1[*b].partial_cmp(&g1[*a]).unwrap_or(std::cmp::Ordering::Equal));
    let top: Vec<String> = idx.iter().take(3).map(|i| format!("{i}:{:.3}", g1[*i])).collect();
    println!("{tag}: best={best_v:.4} @{best}  top=[{}]", top.join(", "));
    println!("      bbox@best={:?}", &bbox[best * 4..best * 4 + 4]);
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

    report("black", &model, vec![0.0f32; 3 * 640 * 640]);
    report("white", &model, vec![1.0f32; 3 * 640 * 640]);

    let bytes = std::fs::read(dir.join("obama.jpg")).expect("read");
    let img = image::load_from_memory(&bytes).expect("decode");
    let (w, h) = (img.width(), img.height());
    let raw = img.to_rgb8().into_raw();
    report("obama RGB", &model, build(&raw, w, h, false));
    report("obama BGR", &model, build(&raw, w, h, true));
}
