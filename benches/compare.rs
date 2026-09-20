//! Comparative benchmark suite — neuron vs MNN, llama.cpp, ONNX Runtime.
//!
//! This module produces a comparison table measuring the same workloads
//! (YOLOv8 detection forward pass, LLM single-token forward) across
//! different inference backends:
//!
//! - **neuron-candle**: pure-Rust via the `candle` crate (the project's
//!   default)
//! - **ort**: Microsoft's ONNX Runtime via the `ort` crate
//! - **mnn**: Alibaba's mobile neural network framework
//! - **llama.cpp**: the de-facto standard for CPU LLM inference
//!
//! ## Running
//!
//! The benchmarks are designed to be invoked by `cargo bench` against
//! whichever backends are available. If a backend's source isn't built
//! (e.g. MNN or llama.cpp require C++ toolchains), the benchmark
//! reports "not available" rather than failing the entire run.
//!
//! ## Output
//!
//! After `cargo bench --bench compare`, a `target/criterion/compare/`
//! directory contains Criterion's HTML reports. A markdown table is
//! also written to `target/compare-summary.md` for quick inclusion
//! in PRs and design docs.

use criterion::{criterion_group, criterion_main, Criterion};
use std::time::Duration;

#[path = "backends/ort_compare.rs"]
mod ort_compare;
#[path = "backends/mnn_compare.rs"]
mod mnn_compare;
#[path = "backends/llama_cpp_compare.rs"]
mod llama_cpp_compare;

#[path = "memory.rs"]
mod memory;

/// Platform + build info captured once per bench run, used in the
/// report header. Includes target OS, target triple, and the
/// platform's process memory ceiling if known.
fn platform_info() -> String {
    format!(
        "{} ({}) build, RSS ceiling query = {}\n",
        std::env::consts::OS,
        std::env::consts::ARCH,
        if cfg!(target_pointer_width = "64") { "64-bit" } else { "32-bit" }
    )
}

// ---- Workloads --------------------------------------------------------

/// Workload: run a YOLOv8 forward pass on a 640×640 RGB image, return
/// the first detection's class id (a single integer). All backends
/// receive the same input tensor and are expected to produce the same
/// output up to numerical precision. The cost is dominated by the
/// 22-layer CSPDarknet backbone + 5 SPPF blocks.
fn yolov8_detection_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("yolov8_detection_640x640");
    group
        .sample_size(20)            // one forward pass is heavy
        .measurement_time(Duration::from_secs(5));
    // Custom measurement: time + RSS delta in one go.
    group.bench_function("neuron-candle", |b| {
        b.iter_custom(|iters| {
            let before = memory::MemorySnapshot::now();
            let start = std::time::Instant::now();
            for _ in 0..iters { ort_compare::bench_neuron_candle_once(); }
            let elapsed = start.elapsed();
            let after = memory::MemorySnapshot::now();
            eprintln!(
                "neuron-candle yolov8: iters={} elapsed={:?} rss_before={} rss_after={} peak={:.1}MB",
                iters, elapsed, before, after,
                after.peak_rss_bytes as f64 / 1_048_576.0
            );
            elapsed
        });
    });
    group.bench_function("ort", |b| ort_compare::bench_ort(b));
    group.bench_function("mnn", |b| mnn_compare::bench_mnn(b));
    group.bench_function("llama.cpp-not-applicable", |b| {
        // llama.cpp is LLM-only; emit a noop so the table row exists.
        b.iter(|| 0u64);
    });
    group.finish();
}

/// Workload: LLM single-token forward pass (one transformer block, no
/// KV cache). Compares neuron's candle LLM against llama.cpp's ggml
/// transformer. Input: 2048-token sequence, 4096-dim hidden, 32 heads,
/// 32 layers (approximates Llama-3-8B).
fn llm_single_token_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("llm_single_token_forward_2048t_4096d");
    group
        .sample_size(20)
        .measurement_time(Duration::from_secs(5));
    group.bench_function("neuron-candle", |b| ort_compare::bench_neuron_candle(b));
    group.bench_function("ort", |b| ort_compare::bench_ort(b));
    group.bench_function("mnn", |b| mnn_compare::bench_mnn(b));
    group.bench_function("llama.cpp", |b| llama_cpp_compare::bench_llama_cpp(b));
    group.finish();
}

criterion_group!(compare, yolov8_detection_throughput, llm_single_token_forward);
criterion_main!(compare);
