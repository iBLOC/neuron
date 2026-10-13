---
title: Benchmarks
---

# Benchmarks

Measured performance and memory numbers for neuron's inference paths.
Within each capability, all rows come from the **same host and same
workload** so the models are directly comparable against each other.
Rows marked with a footnote come from an earlier run on the same host
class — compare within a table, not across tables. Every table states
machine, revision and workload; reproduction commands are at the bottom.

Host for all tables below: Windows 11, 12 cores, 16 GB RAM, RTX 3090
(CUDA rows need `--features cuda`, not part of default builds).

## LLM decode throughput (candle engine, greedy decode)

| model | params | device | dtype | TTFT p50 (ms) | decode tok/s (p50) | peak RSS (MB) |
|---|---|---|---|---:|---:|---:|
| qwen2.5-0.5b | 0.5B | cpu | f32 | 432 | 5.57 | 1 940 |
| qwen3-0.6b | 0.6B | cpu | f32 | 645 | 4.77 | 3 198 |
| minicpm5-2b ¹ | 2.52B | cpu | f16 | 3 336 | 2.58 | 7 451 |
| minicpm5-2b ¹ | 2.52B | cuda | bf16 | 103 | 26.45 | 2 432 |

¹ measured 2026-09-27 (rev `732f5c8`, 12-token decode, 64/256-word
prompts); other rows measured 2026-10-11 (rev `952a251`, 32-token
decode). Same host, same methodology (client-side timing, warmup
discarded, unique prompt heads) — treat the minicpm5 rows as
directionally comparable, the qwen rows as the current-code baseline.

Reading the table: decode rate is the steady-state generation speed;
CUDA bf16 needs `--features cuda` (not in default builds). candle's CPU
backend has no bf16 matmul and — as of the current build — rejects f16
logits on the CPU path, so CPU rows are f32 (highest memory). Larger
catalog models (llama-3-8b, mistral-7b, chatglm-3-6b, Gemma tier) can be
measured with the same harness; their numbers are not yet published.

## OCR end-to-end latency (PP-OCR tiers × executors)

| model | det+rec weights | tract (cold e2e, ms) | ort (cold e2e, ms) | ort speedup | regions |
|---|---|---:|---:|---:|---:|
| ppocrv5-mobile | 21 MB | 2 922 | 588 | 5.0× | 5 |
| ppocrv6-medium | 139 MB | 7 924 | 1 056 | 7.5× | 5 |
| ppocrv5-server | 173 MB | 13 160 | 1 362 | 9.7× | 5 |

All 12 runs on the same 800×400 zh/en page, one-shot process (numbers
include first-use executor setup: tract compiles its plans, ort loads
onnxruntime). Text output is identical across executors and tiers.
Latency scales with model tier; the ort executor claws back most of the
gap between mobile and server tiers (588 → 1 362 ms, vs 2 922 → 13 160
on tract).

## ASR latency (SenseVoiceSmall, sherpa engine)

| wav | duration | p50 (ms) | RTF |
|---|---:|---:|---:|
| en.wav | 7.15 s | 482.9 | 0.068 |
| zh.wav | 5.59 s | 361.4 | 0.065 |

RTF < 0.1 = 10× realtime. The tract path produces transcription-parity
output (verified 2026-09-26); its per-wav latency was not captured in
this run.

## Vision TensorPool (allocation)

| path | cost |
|---|---|
| fresh `Tensor::zeros(1,3,640,640)` | ~460 µs |
| pooled get+release | ~292 ns (~1574×) |

Poolable intermediates ≈ 32 MB vs ~200 MB peak without pooling (~6×).

## Static weights / memory footprints

| model | weights | resident memory |
|---|---|---|
| minicpm5-1b | ~2 GB (bf16) | ~3 GB VRAM (bf16); CPU f32 rows above |
| minicpm5-2b | ~5 GB (bf16) | ~6-7 GB VRAM (bf16); f32 ~10 GB RAM |
| voxcpm2 (TTS) | 4.58 GB bf16 + 377 MB fp32 | ~10 GB RAM (f32) |
| laya (decision) | ~500 MB int8 ONNX | int8 |
| PP-OCRv5 mobile (OCR) | 21 MB | |
| PP-OCRv5 server (OCR) | 173 MB | |
| PP-OCRv6 medium (OCR) | 139 MB | +130 MB layout, +368 MB table (optional) |
| all-minilm-l6-v2 | ~88 MB | |
| bge-reranker-base | ~1.08 GB | |
| yolov8n | ~6 MB | poolable intermediates ~32 MB |

## Reproducing

```bash
# LLM TTFT / decode tok/s / peak RSS vs vLLM / ollama / llama.cpp
# (starts installed servers; report -> target/llm-speed-compare.md)
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
