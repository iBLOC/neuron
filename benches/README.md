# Benchmarking neuron

This directory contains Criterion benchmarks for measuring performance of neuron's data-structure operations and DTO construction/sorting. Real-engine benchmarks (LLM forward passes, TTS synthesis, rerank scoring) require a separate benchmarking harness with model weights — see `examples/real_model_smoke.rs` and `examples/real_model_rerank_smoke.rs` for end-to-end smoke tests.

## Running Benchmarks

Full benchmark (sample_size=50, measurement_time=5s per benchmark):
```bash
cargo bench --bench perf
```

Quick benchmark (1 sample, 1 iteration) for CI:
```bash
cargo bench --bench perf -- --quick
```

With specific features:
```bash
cargo bench --bench perf --features candle
```

## Benchmark Categories

| Benchmark | Purpose |
|-----------|---------|
| `base64_encode_micro` | Base64 encode 384 f32 values |
| `rerank_response_[10,100,1000]` | Response construction + sorting |
| `model_info_[10,100,1000]` | JSON serialization of ModelInfo |

## Removed in 2026-06-30 (echo stub removal)

These benchmarks were removed when the `Echo*Runtime` stubs were deleted:

- `embedding_echo_size_[1,8,32,128]` — embedding batch scaling via `EchoEmbeddingRuntime`
- `embedding_float` / `embedding_base64` — encoding format overhead via `EchoEmbeddingRuntime`
- `llm_non_streaming` / `llm_streaming` — LLM streaming overhead via `EchoLlmRuntime`
- `rerank_docs_[1,10,50,100]` — rerank document count scaling via `EchoRerankRuntime`
- `tts_len_[10,100,1000]` — TTS input length via `EchoTtsRuntime`

To re-introduce comparable benchmarks, point them at a real engine implementation behind a feature gate (e.g. `#[cfg(feature = "candle")]` + `CandleEmbeddingRuntime::embed(...)`).

## CI

Recommended CI command (fast smoke test):
```bash
cargo bench --bench perf -- --quick
```

This runs each benchmark with 1 sample and 1 iteration, completing in ~30 seconds.

## Memory benchmarks (cross-platform)

`benches/memory.rs` + `benches/memory_bench.rs` measure process-level memory (RSS, peak RSS, virtual, heap) across various operations. Platform support:

| Platform | Mechanism | Function used |
|---|---|---|
| Linux | `/proc/self/status` (VmRSS, VmHWM, VmSize) | `MemorySnapshot::now()` |
| macOS | `task_info` mach syscall | `MemorySnapshot::now()` |
| Windows | `GetProcessMemoryInfo` (via `windows-sys`) | `MemorySnapshot::now()` |
| Other | noop (returns 0) | — |

```bash
cargo bench --bench memory_bench
# → target/criterion/memory_bench/{memory_rss_read_1mb,alloc_1mb,alloc_100mb,after_free}/report/
```

The `MemoryTrace` helper records RSS + peak RSS for a single benchmark iteration, useful for the comparative bench's report generator.

## Comparative benchmarks (vs MNN, llama.cpp, ONNX Runtime)

The `compare.rs` benchmark + `backends/*.rs` stubs compare neuron's inference latency against the mainstream frameworks:

```bash
cargo bench --bench compare
cargo run --release --bin compare_report
# produces target/compare-summary.md with a comparison table
```

| Backend | v1.0 status | How to enable real forward pass |
|---|---|---|
| **neuron-candle** | ✅ Real (candle YOLOv8 forward) | enabled by default |
| **ort** (ONNX Runtime) | ⚠️ Stub (noop) | add `dep:ort` to `Cargo.toml` + `ort` feature, set `NEURON_YOLOV8_ONNX` |
| **mnn** (Alibaba) | ⚠️ Stub (noop) | build MNN from `https://github.com/alibaba/MNN`, add FFI binding in `crates/mnn-sys` |
| **llama.cpp** | ⚠️ Stub (noop for non-LLM) | build llama.cpp, add Rust FFI (e.g. `llama-cpp-sys-2`), load a GGUF model |

The `compare_report` binary reads Criterion's `target/criterion/compare/*/base/estimates.json` files and produces a markdown table comparing the same workload across all backends. "not available" entries indicate the backend wasn't built for this comparison — the table still shows the framework is in place.