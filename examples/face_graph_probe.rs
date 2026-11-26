//! YuNet 输出溯源 —— 每个输出的生产者吃的是「常量」还是「卷积栈」。

use std::path::PathBuf;

use tract_onnx::prelude::*;

fn main() {
    let dir = PathBuf::from(std::env::var("FACE_CKPT_DIR").unwrap_or_else(|_| {
        "C:\\Projects\\iBLOC\\neuron\\.local\\face-ckpt".to_string()
    }));
    let raw = tract_onnx::onnx()
        .model_for_path(dir.join("builtin").join("yunet.onnx"))
        .expect("load");
    for (oi, o) in raw.outputs.iter().enumerate() {
        let n = raw.node(o.node);
        let srcs: Vec<String> = n
            .inputs
            .iter()
            .map(|i| format!("#{} {:?}", i.node, raw.node(i.node).name))
            .collect();
        println!("out[{oi}] <- node #{} {:?} op={:?} inputs={:?}", o.node, n.name, n.op, srcs);
    }
    println!("-- 反查：哪些节点吃 Conv_0(#112) 的输出 --");
    let mut consumers = Vec::new();
    for (i, n) in raw.nodes.iter().enumerate() {
        if n.inputs.iter().any(|i| i.node == 112) {
            consumers.push((i, n.name.clone(), format!("{:?}", n.op)));
        }
    }
    for c in consumers.iter().take(12) {
        println!("   {c:?}");
    }
    println!("-- 尾部节点（最后 20 个）--");
    let total = raw.nodes.len();
    for i in (total.saturating_sub(20))..total {
        let n = raw.node(i);
        let srcs: Vec<String> = n
            .inputs
            .iter()
            .map(|i| format!("#{}", i.node))
            .collect();
        println!("   #{i} {:?} op={:?} inputs={:?}", n.name, n.op, srcs);
    }
}
