# MNN Backend for Mobile / Edge Vision — Design Spec

**Date**: 2026-09-01
**Status**: 📋 Design phase (Phase 3 of Sprint 4 mobile/edge optimization)
**Predecessors**:
- [2026-09-01-mobile-phase-1-tensor-pool.md](./2026-09-01-mobile-phase-1-tensor-pool.md) — TensorPool landed
- [2026-09-01-mobile-phase-2-tract-onnx.md](./2026-09-01-mobile-phase-2-tract-onnx.md) — tract ONNX landed
- The ONNX plug-in contract — implemented by [`build_session` in `src/vision/ort_runtime.rs`](../../../src/vision/ort_runtime.rs)

## Motivation

The tract backend (Phase 2) covers the mid-range: pure-Rust ONNX inference on ARM, x86_64, and Apple Silicon. But tract's footprint (~30 MB binary, ~150 MB peak memory for YOLOv8n) is too large for low-end mobile devices (2-core ARM, 4 GB RAM) where:

- App size budgets are 30-50 MB total
- Available memory after OS is often < 1.5 GB
- Cold-start latency matters more than throughput

[MNN](https://github.com/alibaba/MNN) is Alibaba's mobile-first inference engine:

| Property | Value |
|---|---|
| Binary size | ~10 MB (MNN core) + ~2 MB per backend (OpenCL, Metal, NNAPI) |
| YOLOv8n memory peak | ~50 MB |
| YOLOv8n latency (2-core ARM) | ~10 ms |
| Quantization | INT8 / FP16 native |
| Platforms | Android, iOS, Linux, macOS, Windows |

For low-end mobile, MNN delivers ~5× smaller binary and ~3× faster inference vs tract.

## Goals

1. **Wire MNN as a drop-in third OrtSessionRunner** alongside the existing candle and tract backends
2. **No hard dep on MNN at neure's top level** — same feature-flag pattern as tract (`mnn-f` Cargo feature)
3. **Reuse the existing letterbox / decode / NMS pipeline** — only the session construction + run differ
4. **Document the FFI binding layer** so users know how to maintain it across MNN versions
5. **Benchmark MNN vs tract on the same YOLOv8n workload** to validate the latency claim

## Non-Goals (v3.x)

- **No custom MNN build orchestration**: neure doesn't compile MNN from source. Users vendor a pre-built `libMNN.so` / `libMNN.a` and point `MNN_LIB_DIR` at it.
- **No automatic GPU dispatch**: MNN has OpenCL / Metal / NNAPI backends; users opt in via env var (`MNN_BACKEND=opencl`).
- **No MNN → candle weight conversion**: we keep MNN models in MNN's `.mnn` format (it's the native format, ~30% smaller than ONNX).
- **No LLM via MNN**: MNN's transformer support is patchy. LLM stays on candle / mistralrs.

## Architecture

### Same plug-in contract as tract

MNN slots into the same `OrtSessionRunner` trait that `TractSession` implements:

```
┌────────────────────────────────────────────────┐
│ OrtVisionRuntime (always compiled)              │
│   ├─ image fetch + decode                      │
│   ├─ letterbox preprocessing                   │
│   ├─ CHW f32 tensor construction                │
│   ├─ session.run(input_tensor)                 │
│   └─ decode_yolov8_output + NMS                │
├────────────────────────────────────────────────┤
│ build_session(onnx_path) → OrtSession          │
│                                                  │
│   #[cfg(feature = "candle")]  → CandleSession  │
│   #[cfg(feature = "tract")]    → TractSession   │
│   #[cfg(feature = "mnn-f")]    → MnnSession     │  ← NEW
└────────────────────────────────────────────────┘
```

### FFI binding layer

The `mnn-sys` crate wraps MNN's C API:

```rust
// crates/mnn-sys/src/lib.rs
extern "C" {
    fn MNNCreateInterpreter(
        model_path: *const c_char,
    ) -> *mut MNNInterpreter;
    fn MNNSessionCreateFromFile(
        model_path: *const c_char,
        backend: *const c_char,  // "CPU" | "OpenCL" | "Metal" | "NNAPI"
        precision: *const c_char, // "Normal" | "High" | "Low"
    ) -> *mut MNNSession;
    fn MNNGetSessionInput(session: *mut MNNSession, name: *const c_char) -> *mut MNNTensor;
    fn MNNGetSessionOutput(session: *mut MNNSession, name: *const c_char) -> *mut MNNTensor;
    fn MNNSessionRun(session: *mut MNNSession) -> i32;
    // ...
}
```

`MnnSession` (in `src/vision/ort_backends/mnn_backend.rs`) wraps the raw pointers behind `Arc<Mutex<…>>` (MNN is not thread-safe per-session).

### Conversion path: ONNX → MNN

MNN ships a converter (`MNNConvert`) that produces `.mnn` files from ONNX. For each vision model in the catalog:

```bash
# YOLOv8
MNNConvert -f ONNX --modelFile yolov8n.onnx --MNNModel yolov8n.mnn -i input=1,3,640,640

# RF-DETR / RT-DETR / DETR: same pattern
MNNConvert -f ONNX --modelFile rtdetr-r50.onnx --MNNModel rtdetr-r50.mnn

# Florence-2: needs custom script (see MNN/examples)
```

The runtime loads `.mnn` files via `MNNSessionCreateFromFile`. The HTTP API (`/v1/models/pull`) gets a new `format` field to request `.mnn` instead of `.onnx`:

```bash
curl -X POST http://localhost:8083/v1/models/pull \
  -d '{"reference": "ultralytics/yolov8n", "engine": "vision", "id": "yolov8n", "format": "mnn"}'
```

### Memory profile target

For YOLOv8n (640×640 input) on a 2-core ARM device with 4 GB RAM:

| Metric | neure + tract | neure + MNN | Reduction |
|---|---|---|---|
| Binary size | 30 MB | **12 MB** | 2.5× |
| Cold-start (load + first inference) | 800 ms | **200 ms** | 4× |
| Steady-state memory peak | 150 MB | **50 MB** | 3× |
| Steady-state latency | 30 ms | **10 ms** | 3× |

(Combined with the TensorPool from Phase 1, MNN steady-state memory drops to ~30 MB — a 5× reduction over tract alone.)

## Implementation plan

### Phase 3a (1 week): FFI binding + session construction

- Add `crates/mnn-sys/` crate with `bindgen`-generated bindings to `MNNExpress.h` + `MNNSession.h`
- Add `mnn-f` Cargo feature pulling `mnn-sys` + `ndarray`
- Implement `MnnSession::new(path)` in `src/vision/ort_backends/mnn_backend.rs`
- Wire `build_session` in `ort_runtime.rs` to dispatch to `MnnSession` under `cfg(feature = "mnn-f")`
- 2 unit tests: load nonexistent path (should fail), load real `.mnn` if available

### Phase 3b (1 week): Inference + benchmarking

- Implement `OrtSessionRunner::run_inference` for `MnnSession` (input tensor copy + `MNNSessionRun` + output copy)
- Convert an existing YOLOv8n ONNX → MNN, run end-to-end through the existing HTTP endpoint
- Add benchmarks in `benches/mobile_bench.rs` for MNN vs tract on the same workload
- Document the binary-size delta in README

### Phase 3c (1 week): Production polish

- Add `MNNConvert` invoker to `/v1/models/pull` so users can pull and convert in one step
- Add OpenCL / NNAPI backend selection via `NEURE_MNN_BACKEND` env var
- Document the build flags (`MNN_LIB_DIR`, `NEURE_MNN_BACKEND`) in README
- Update AGENTS.md to mark Phase 3 complete

## Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| MNN's API breaks between minor versions (pre-2.0) | High | Pin to a specific MNN commit in `Cargo.toml`; add CI matrix for MNN 2.0+ / 2.1+ / 2.2+ |
| MNN's ARM NEON kernels are slower than custom kernels | Medium | Document the OpenCL fallback path; show perf in README |
| MNN licensing — Apache-2.0 with LLVM exceptions | Low | Compatible with neure's Apache-2.0 (no copyleft) |
| FFI safety — use-after-free if MNN's interpreter outlives the session | Medium | Wrap pointers in `Arc<MnnInterpreter>` with explicit `Drop`; add miri test |
| Cross-compilation to `aarch64-linux-android` | High | Document the toolchain (NDK r25+), add CI for the target |

## Open questions

1. **Should MNN be the default on mobile builds?** — Yes for the `mobile` Cargo feature; no for the default desktop build (tract is fine).
2. **Should we vendor MNN's converter or shell out?** — Shell out (MNNConvert is ~30 MB and only needed at pull time).
3. **What about MNN's quantization formats?** — Support `int8` only in v3.1; `fp16` is mostly automatic via MNN's backend precision.

## References

- [MNN repo](https://github.com/alibaba/MNN)
- [MNN benchmarks vs ncnn / TFLite](https://github.com/alibaba/MNN#benchmark)
- [MNNConvert usage](https://www.yuque.com/mnn/en/convert_tools)
- [MNNExpress API](https://www.yuque.com/mnn/en/correct_the_sample_code)
- [ONNX → MNN converter](https://www.yuque.com/mnn/en/onnx_2_mnn)