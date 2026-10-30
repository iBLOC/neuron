# Benchmarking neuron

This directory contains Criterion benchmarks for measuring performance of neuron's data-structure operations and DTO construction/sorting. Real-engine benchmarks (LLM forward passes, TTS synthesis, rerank scoring) require a separate benchmarking harness with model weights — see `examples/real_model_smoke.rs` and `examples/real_model_rerank_smoke.rs` for end-to-end smoke tests.

## Real-engine speed comparison harnesses

### `model_bench` — one benchmark per model in the capability catalog

The per-model harness: every entry of `capabilities::catalog::default_capabilities()`
becomes a case, loaded **in-process** through the public registry API
(`LlmRuntimeRegistry::register` → `runtime_for`) and timed on the local
device. No HTTP, no server startup — the numbers are the library's own
device timings.

| capability | workload | headline metric |
|---|---|---|
| LLM | 1 prompt, greedy, 64 new tokens, streamed | TTFT (ms), decode tok/s |
| TTS | one fixed sentence | audio s / wall s |
| ASR | one fixed WAV | RTF |
| Rerank | 1 query × 8 documents | docs/s |
| Embedding | 8 short texts | texts/s |
| Vision / OCR | one 640×640 image | images/s |
| Decision | 1 question | questions/s |
| Chronos2 | one 64-point series | forecasts/s |

Models whose weights are not on this machine are reported as `skipped`
with the reason, so the report doubles as a coverage matrix.

```bash
# CUDA build needs the MSVC environment (nvcc needs cl.exe):
scripts\cuda-build.bat cargo bench --bench model_bench --features "candle,cuda,tract,ocr,sherpa,voxcpm,chronos2,litert"

# CPU-only:
cargo bench --bench model_bench --features "candle,tract"

# → target/model-bench-<device>.md
```

Weights: `scripts/fetch-bench-models.sh [llm|asr|rerank|embedding|ocr|chronos2|decision|all]`
pulls the checkpoints into `$NEURON_MODEL_DIRS/<capability>/<model_id>/`.
The vision rows need an ONNX export first:
`python scripts/export-yolo-onnx.py` (Ultralytics `.pt` → `model.onnx` + `config.json`).

Environment: `NEURON_BENCH_DEVICE` (cuda/cpu), `NEURON_BENCH_MODELS` (subset),
`NEURON_BENCH_REPEATS`, `NEURON_BENCH_WARMUP`, `NEURON_BENCH_OUT`,
`NEURON_BENCH_MODEL_DIRS`, `NEURON_BENCH_<CAP>_DIR` (explicit weight dir per
capability), `NEURON_BENCH_IMAGE` / `NEURON_BENCH_AUDIO` (fixtures).
The report header records the tuning knobs that were in effect (dtype,
ISQ, context size, …).

### `llm_speed` — neuron vs vLLM / ollama / llama.cpp

HTTP round-trip harness, drives whatever endpoints `NEURON_BENCH_ENDPOINTS` names; see top-of-file doc comment for the env contract. Driver script: `scripts/run-llm-speed-compare.sh`.

### `asr_speed` — neuron's tract ONNX path vs sherpa-onnx

In-process ASR comparison on the **same SenseVoiceSmall graph**. Both engines are pinned to one WAV list, warmup + repeats give p50/p95; the tract column additionally splits fbank / forward / decode; a parity table compares the two engines' transcriptions after stripping whitespace + SentencePiece `▁`.

```bash
cargo bench --bench asr_speed --features "tract,sherpa"
# or via the driver (downloads the model + test WAVs first):
./scripts/run-asr-speed-compare.sh
```

Environment: `NEURON_ASR_MODEL_PATH`, `NEURON_ASR_BENCH_WAVS`, `NEURON_ASR_BENCH_PROFILE` (`quick`/`standard`/`full`), `NEURON_ASR_SHERPA_NUM_THREADS`. The report lands in `target/asr-speed-compare.md`. The graph the two engines run is *not* the same file: sherpa runs the original `model.onnx`, tract runs `model.prepped.onnx` produced by `scripts/prep-sensevoice-onnx.py`. Parity is verified inside the prep script (max |diff| < `1e-3`, argmax equal across T = 37..250).

### `model_matrix` — every (cap × engine × device) row in one report

Walks the full capability matrix under the current Cargo feature set and drives a running neuron server over its OpenAI-compatible HTTP surface. Each row reports warmup-after p50 / p95 wall time, peak RSS, body bytes, RTF (for ASR), and status. Skipped rows are reported separately with a reason, so a reader can tell "engine not compiled in" from "model dir missing on disk" at a glance.

```bash
# Quick path (one feature set, one server start):
./scripts/run-model-matrix.sh
```

The driver script:
1. Builds `examples/neuron_server` with the requested feature set.
2. Starts it on `127.0.0.1:$NEURON_PORT` (default 18083) and waits for `/health`.
3. Builds and runs `benches/model_matrix.rs` against that server.
4. Tears the server down on exit and writes `target/model-matrix.md`.

Environment:

| var | default | meaning |
|---|---|---|
| `NEURON_BENCH_URL` | `http://127.0.0.1:8083` | neuron server base URL |
| `NEURON_BENCH_PROFILE` | `quick` | `quick` / `standard` / `full` preset for warmup × repeats |
| `NEURON_BENCH_<CAP>_DIR` | unset | Override the directory `default_rows` looks the model weights up in (e.g. `NEURON_BENCH_ASR_DIR=$HOME/models/sherpa-sensevoice`) |
| `NEURON_BENCH_MODEL_DIRS` | unset | Colon-separated roots; the bench joins each with `<cap>/<model_id>` |
| `NEURON_BENCH_OUT` | `target/model-matrix.md` | Report path |
| `NEURON_BENCH_DISABLE` | unset | Comma-separated `cap.engine` pairs to skip |

`NEURON_BENCH_<CAP>_RUNTIME` lets the host steer which engine the *server* should be running; the bench does not set runtime env vars itself. Servers started via `scripts/run-model-matrix.sh` use the feature set passed in via `NEURON_FEATURES`. Matrix size is bounded by `Capability::engines()` × `Device::available()` per cap, so the full run on a multi-engine build is 12–20 rows.

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

## Comparative benchmarks (vs llama.cpp / ollama / vLLM) — `llm_speed`

The real cross-framework comparison lives in `llm_speed.rs`: an HTTP client
harness that measures TTFT, prefill / decode tok/s, e2e p50/p95 and peak RSS
( sampled by pid) against any OpenAI-compatible server, all on the same
checkpoint. It is not a Criterion bench — seconds-long round trips need their
own p50/p95 math — so it has `harness = false` and a `fn main()`.

```bash
./scripts/run-llm-speed-compare.sh          # starts servers it finds, runs the harness
cargo bench --bench llm_speed               # or run it yourself against servers you started
# → target/llm-speed-compare.md
```

Fairness rules built into the harness (see the report's Methodology appendix):
same checkpoint everywhere (safetensors for neuron/vLLM, a GGUF converted from
it for llama.cpp/ollama), precision declared per row, unique nonce on every
request so prefix caches can't answer a repeat, client-side timing only,
`reasoning_content` deltas counted as generated tokens, greedy decoding.

### Notes on neuron backends

The harness is backend-agnostic — it talks HTTP — so a second `neuron_server`
started with `NEURON_LLM_RUNTIME=mistralrs` slots in next to a `candle`
instance as a "neuron-mistralrs-cpu" row. Three caveats:

- **mistralrs CPU ISQ is load-heavy.** mistral.rs reads the BF16
  safetensors into F32 (~10 GB for a 2B model), then quantises in
  place to the `NEURON_MISTRALRS_ISQ` target (Q4_K here, ~1.8 GB).
  Cold start is ~30 s on the test bench and the working set during
  the first request peaks at ~2 GB — measurable on a 16 GB host but
  not dangerous; running mistralrs alongside another ~3-4 GB engine
  will OOM it.
- **mistralrs stream endpoint buffers chunks.** mistral.rs emits the
  first chunk only after the whole forward pass finishes; the harness's
  60 s per-token stall budget treats that as "stream unavailable",
  which surfaces in the report as
  `stream unavailable: stream stalled > 60s` plus the non-stream-derived
  e2e. The decode column still reflects the actual per-token wall
  time.
- **litert is Gemma-only.** The `litert` backend refuses non-Gemma
  checkpoints (`LitertLlmRuntime::supported_model_ids`), so it has no
  row against MiniCPM5-2B; the harness reports "no model registered"
  rather than running a bogus measurement.

## Comparative microbenchmark scaffold — `compare.rs`

> **Status: the non-neuron backends are noop stubs.** This suite does *not*
> currently measure real cross-framework latency — use `llm_speed` (above)
> for that. The rows below document how a real binding would be wired in.

```bash
cargo bench --bench compare
cargo run --release --bin compare_report
# produces target/compare-summary.md with a comparison table
```

| Backend | v1.0 status | How to enable real forward pass |
|---|---|---|
| **neuron-candle** | ✅ Real (candle YOLOv8 forward — note: a vision model, not an LLM) | enabled by default |
| **ort** (ONNX Runtime) | ⚠️ Stub (noop) | add `dep:ort` to `Cargo.toml` + `ort` feature, set `NEURON_YOLOV8_ONNX` |
| **mnn** (Alibaba) | ⚠️ Stub (noop) | build MNN from `https://github.com/alibaba/MNN`, add FFI binding in `crates/mnn-sys` |
| **llama.cpp** | ⚠️ Stub (noop for non-LLM) | build llama.cpp, add Rust FFI (e.g. `llama-cpp-sys-2`), load a GGUF model |

The `compare_report` binary reads Criterion's `target/criterion/compare/*/base/estimates.json` files and produces a markdown table comparing the same workload across all backends. "not available" entries indicate the backend wasn't built for this comparison — the table still shows the framework is in place.