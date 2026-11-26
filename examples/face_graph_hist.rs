//! tract 性能体检：算子分布 + YuNet/MobileFaceNet 耗时。
use std::path::PathBuf;
use std::collections::BTreeMap;
use tract_onnx::prelude::*;

fn main() {
    let ck = PathBuf::from(std::env::var("FACE_CKPT_DIR").unwrap_or_else(|_| {
        "C:\\Projects\\iBLOC\\neuron\\.local\\face-ckpt".to_string()
    }));
    let bk = ck.join("builtin");

    for name in ["yunet.onnx", "mobile_facenet.onnx"] {
        let path = bk.join(name);
        let raw = tract_onnx::onnx().model_for_path(&path).expect("load");
        let raw_hist: BTreeMap<String, usize> = raw
            .nodes()
            .iter()
            .map(|(_, n)| format!("{:?}", n.op))
            .fold(BTreeMap::new(), |mut m, k| {
                *m.entry(k).or_insert(0) += 1;
                m
            });
        let model = raw.into_optimized().expect("optimize");
        let hist: BTreeMap<String, usize> = model
            .nodes()
            .iter()
            .map(|(_, n)| format!("{:?}", n.op))
            .fold(BTreeMap::new(), |mut m, k| {
                *m.entry(k).or_insert(0) += 1;
                m
            });
        println!("== {} raw_nodes={} opt_nodes={}", name, raw_hist.values().sum::<usize>(), hist.values().sum::<usize>());
        println!("   raw: {:?}", raw_hist);
        println!("   opt: {:?}", hist);
    }
}