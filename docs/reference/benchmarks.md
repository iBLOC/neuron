---
title: Benchmarks
---

# Benchmarks

Measured performance and memory numbers for neuron's inference paths. Every
table states the machine, revision and workload — numbers from different
hardware are not comparable. Reproduction commands are at the bottom.

## LLM decode throughput (MiniCPM5-2B vs reference engines)

Source: `benches/llm_speed.rs` report, generated 2026-09-27 on
Windows 11 / 12 cores / 16 GB RAM / RTX 3090, neuron rev `732f5c8`
(candle `aebc405`). Same MiniCPM5-2B checkpoint everywhere; greedy
decoding, 12 output tokens, unique prompt heads (no prefix-cache assist).

| framework | device | weights | TTFT ms (p50) | decode tok/s (p50) | peak RSS MB |
|---|---|---|---:|---:|---:|
| neuron (candle) | cpu | f16 | 3336 | 2.58 | 7451 |
| neuron (candle) | cuda | bf16 | 103 | 26.45 | 2432 |
| neuron (candle) | cpu | f16 | 8688 | 1.63 | 7451 |

prompt 64 / 256 words (see source report for both rows).

Known control points from the same machine (see
`AGENTS.md` → candle-upgrade A/B): llama.cpp CUDA q8_0 ≈ 131-142 tok/s on
the same class of model — neuron's candle CPU/CUDA paths trade raw speed
for in-process embedding and wire compatibility.

**Memory**: f16 weights for a 2B model ≈ 7.5 GB peak RSS; bf16 on CUDA
≈ 2.4 GB. candle's CPU backend has no bf16 matmul; F32 doubles memory
(~10 GB, pages on a 16 GB host — set `NEURON_LLM_DTYPE`).

## ASR latency (SenseVoiceSmall)

Source: `benches/asr_speed.rs` report (sherpa path; 2026-09-27, same host).

| wav | duration | engine | p50 ms | RTF |
|---|---:|---|---:|---:|
| en.wav | 7.15 s | sherpa | 482.9 | 0.068 |
| zh.wav | 5.59 s | sherpa | 361.4 | 0.065 |

RTF < 0.1 = 10× realtime. The tract path produces transcription-parity
output (verified 2026-09-26, see `scripts/prep-sensevoice-onnx.py` doc).

## OCR: tract vs ort executor (PP-OCRv6 medium)

Source: 2026-10-09 A/B on Windows (same host class), official
`PaddlePaddle/*_onnx` exports, release builds:

| workload | tract | ort (onnxruntime) | speedup |
|---|---:|---:|---:|
| det 736² forward | 3674 ms | 545 ms | 6.7× |
| rec 48×320 forward | 120-233 ms | 10-35 ms | 6-12× |
| **end-to-end page** (800×400, det + 5 lines) | 5418 ms | **526 ms** | **10.3×** |

End-to-end includes first-use executor setup. Text output is
byte-identical between executors. Memory: det 62 MB + rec 77 MB weights;
ort additionally loads onnxruntime at runtime.

## Vision TensorPool (allocation)

Source: `benches/mobile_bench.rs` (YOLOv8n 640×640 preallocation):

| path | cost |
|---|---|
| fresh `Tensor::zeros(1,3,640,640)` | ~460 µs |
| pooled get+release | ~292 ns (~1574×) |

Poolable intermediates ≈ 32 MB vs ~200 MB peak without pooling (~6×).

## Model file / memory footprints

Static weights sizes (disk) and documented resident-memory figures:

| model | weights | resident memory |
|---|---|---|
| minicpm5-1b | ~2 GB (bf16) | ~3 GB VRAM (bf16); CPU f16 higher |
| minicpm5-2b | ~5 GB (bf16) | ~6-7 GB VRAM (bf16); f32 ~10 GB RAM |
| voxcpm2 (TTS) | 4.58 GB bf16 + 377 MB fp32 | ~10 GB RAM (f32 materialized) |
| laya (decision) | ~500 MB int8 ONNX | int8 |
| PP-OCRv6 medium (OCR) | 62 + 77 MB | small; +130 MB layout, +368 MB table |
| all-minilm-l6-v2 | ~88 MB | |
| bge-reranker-base | ~1.08 GB | |
| yolov8n | ~6 MB | poolable intermediates ~32 MB |

Per-model decode tok/s for the rest of the candle tier, litert Gemmas and
vision latency are **not yet published** — the harnesses below produce
them on demand.

## Reproducing

```bash
# LLM TTFT / decode tok/s / peak RSS vs vLLM / ollama / llama.cpp
# (starts whatever servers are installed; report -> target/llm-speed-compare.md)
./scripts/run-llm-speed-compare.sh

# Per-capability × per-engine matrix against a running neuron_server :18083
# (report -> target/model-matrix.md)
cargo bench --bench model_matrix --features "candle,tract,sherpa,voxcpm"

# ASR tract-vs-sherpa (report -> target/asr-speed-compare.md)
cargo bench --bench asr_speed --features tract,sherpa

# Allocation / pooling microbenchmarks
cargo bench --bench mobile_bench
cargo bench --bench perf -- --quick
```

Methodology notes (prompt-nonce anti-caching, client-side timing, warmup
discard, greedy decode) live in each generated report's header — see
`benches/llm_speed.rs` and `AGENTS.md` for the full contract.

## Next steps

- [/reference/models](/reference/models) — supported model × backend matrix
- [/concepts/scheduler](/concepts/scheduler) — how the memory budget trades against these numbers
