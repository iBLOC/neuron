---
title: Benchmarks
---

# Benchmarks

Measured speed, latency, memory and accuracy numbers for neuron's
inference paths — one row per **model × backend** combination, all from
the same host so numbers are directly comparable within a table.

::: tip Methodology
Every table states its machine, revision and workload. Client-side
timing, warmup discarded, greedy decode, unique prompt heads
(prompt-nonce anti-caching). Rows from different runs are marked with a
footnote — compare within a table, not across tables. Reproduction
commands at the bottom.
:::

Host for all tables: Windows 11, 12 cores, 16 GB RAM, RTX 3090 24 GB
(CUDA rows need `--features cuda`; not part of default builds).

## Per-model sweep — every catalog model, one row each (2026-10-29)

`benches/model_bench.rs` generates **one case per model in the capability
catalog** and runs it **in-process** through the public registry API
(`register` → `runtime_for`), on the local device. No HTTP, no server
startup — these are the library's own device timings. Models whose
weights are absent from the host are reported as `skipped` with the
reason, so the report doubles as a coverage matrix.

Full report: [`target/model-bench-cuda.md`](https://github.com/iBLOC/neuron/blob/main/target/model-bench-cuda.md)
(regenerate with `scripts\cuda-build.bat cargo bench --bench model_bench
--features "candle,cuda,tract,ocr,sherpa,voxcpm,chronos2,litert"`).

Measured: **41 cases — 14 measured, 27 skipped (weights absent), 0 failed.**

### LLM (candle, CUDA, greedy, 64 new tokens)

| model | params | decode tok/s | p50 (ms) | TTFT p50 (ms) | notes |
|---|---:|---:|---:|---:|---|
| qwen2.5-0.5b | 0.5B | 52.3 | 563 | — | f32 weights |
| qwen3-0.6b | 0.6B | 54.7 | 1 260 | — | f32 weights |

### Embedding (candle, CUDA, 8 short texts)

| model | texts/s | p50 (ms) |
|---|---:|---:|
| all-minilm-l6-v2 | 507.6 | 16.0 |
| bge-base-en-v1.5 | 322.1 | 24.1 |
| e5-base-v2 | 331.8 | 25.3 |
| gte-base | 397.8 | 19.1 |

### ASR (RTF = audio seconds per wall second; higher is better)

| model | engine | RTF | p50 (ms) | notes |
|---|---|---:|---:|---|
| sensevoice-small | sherpa | 17.2 | 117 | int8 sherpa-onnx graph |
| whisper-small | candle | 9.2 | 205 | CUDA |
| whisper-tiny | candle | 14.1 | 141 | CUDA |

### Vision / OCR (ONNX session path, tract executor, 640×640)

| model | engine | images/s | p50 (ms) | notes |
|---|---|---:|---:|---|
| yolov11n | ort | 2.18 | 466 | |
| yolov8n | ort | 2.03 | 491 | |
| yolov8s | ort | 0.95 | 1 049 | |
| ppocrv5-mobile | paddle-ocr | 1.50 | 681 | det+rec |

### TTS (burn)

| model | engine | p50 (ms) | notes |
|---|---|---:|---|
| voxcpm2 | burn | 316 222 | **CPU** — the burn CUDA backend links `nccl.lib`, which the CUDA toolkit does not ship; also measured ~6× slower than the CPU on this GPU (per-patch decode is latency-bound) |

::: warning What this sweep found
Building the harness surfaced **three real bugs**, all fixed:

1. **CUDA context was bound to the loading thread.** `chat_stream` runs
   the generation on a freshly spawned thread, and cudarc's primary
   context is bound per thread — every CUDA forward on that thread
   failed with `CUDA_ERROR_INVALID_CONTEXT` and the SSE body came back
   empty while the non-streaming path worked. qwen3-0.6b measured
   **22 tok/s** broken vs **54.7 tok/s** after the fix.
2. **Whisper's `mel_filters.safetensors` reader expected a fictional
   format** (a raw f32 blob behind an `ET\0\0` magic) instead of the
   standard safetensors layout, so the documented download
   (`spaces/lmz/candle-whisper`) was rejected outright.
3. **The rerank catalog pointed mxbai and jina at the BGE engine.**
   The three reranker checkpoints are three different architectures
   (XLM-RoBERTa / Qwen2-with-a-custom-head / XLMRoberta), so the two
   non-BGE rows failed to load. They now declare their own engines.

The sweep also exposed that the **candle YOLOv8 backend cannot consume
Ultralytics exports** (it approximates the architecture and has no
PAN-FPN neck), so the vision rows use the ONNX path
(`scripts/export-yolo-onnx.py` does the export).
:::

## LLM decode throughput (candle engine)

| model | params | device | dtype | TTFT p50 (ms) | decode tok/s (p50) | peak RSS (MB) |
|---|---|---|---|---:|---:|---:|
| qwen2.5-0.5b | 0.5B | cpu | f32 | 432 | 5.57 | 1 940 |
| qwen3-0.6b | 0.6B | cpu | f32 | 645 | 4.77 | 3 198 |
| minicpm5-2b ¹ | 2.52B | cpu | f16 | 3 336 | 2.58 | 7 451 |
| minicpm5-2b ¹ | 2.52B | cuda | bf16 | 103 | 26.45 | 2 432 |

¹ measured 2026-09-27 (rev `732f5c8`, 12-token decode); other rows
2026-10-11 (rev `952a251`, 32-token decode). Same host + methodology.

Reading the table: decode rate is steady-state generation speed. candle's
CPU backend has no bf16 matmul, so CPU rows are f32 (highest memory).
Larger catalog models (llama-3-8b, mistral-7b, chatglm-3-6b) run on the
same harness; numbers not yet published.

## LLM across engines (same machine, cross-framework)

| stack | model / quant | device | decode tok/s |
|---|---|---|---:|
| llama.cpp ² (reference control) | q8_0 | cuda | 131–143 |
| neuron / candle | 1B bf16 | cuda | 26–38 |
| neuron / mistralrs | 1B Q4_K (ISQ) | cpu | ~6.7 |
| neuron / candle | 1B f16 | cpu | 1.6–2.6 |

² llama.cpp is **not** part of neuron — it is the control row from the
[candle 0.11 upgrade A/B](https://github.com/huggingface/candle) that
sizes the same GPU's ceiling. The candle CUDA upgrade was perf-neutral
on single-stream decode (26.5/26.0 vs 30.6/38.0 before — within
run-to-run noise); the upgrade bought features, not speed.

## BitNet spike (microsoft/bitnet.cpp — not shipped, NO-GO 2026-10-13)

A 6-gate spike against the BitNet T-MAC inference stack ran on the
spike box (Zen 3 / RTX 3090) and is **not merged** — see
[target/bitnet-spike/REPORT.md](https://github.com/iBLOC/neuron/blob/main/target/bitnet-spike/REPORT.md)
for the full report. The TL-OFF i2_s path compiled cleanly under MSVC
+ 4 scratch patches; the T-MAC fast path is upstream dead code on the
fork the wrapper pins, so the table below is the LLAMAFILE sgemm
fallback. The official `ggml-model-i2_s.gguf` ships without
`tokenizer.ggml.pre` and the spike ships a 1-KV `patch_pre.py` to inject
`pre=llama-bpe` before model load.

| stack | model / quant | device | decode tok/s | peak RSS (MB) | notes |
|---|---|---|---:|---:|---|
| neuron / candle | 1B f16 | cpu | 1.6–2.6 | n/a | baseline (row above) |
| neuron / candle | 2.5B f16 | cpu | 2.58 | 7 451 | baseline (row above) |
| **bitnet.cpp spike ³** | 2B i2_s (1.58-bit) | cpu AVX2 | **16.8–18.6** | ~2 000 | 4 patches, no T-MAC; Zen 3 (no AVX-512, no VNNI); 127-token run at `--temp 0, -t 4` |

³ Not a neuron backend. Numbers are from the standalone `llama-completion.exe`
built under `target/bitnet-spike/BitNet/build-static/` and run on the
spike box (Zen 3 / RTX 3090, same as the rows above). The 6-7× CPU
ratio over candle's f32/f16 is real but conditional on (a) a working
T-MAC build, (b) the patched gguf artifact, and (c) chat-template
assembly staying in the OpenAI adapter (fork `-cnv` crashes on the
custom `bitnet-b1.58` arch). Re-evaluation gates are listed in the
spike report's §4.

## VLM accuracy & throughput (Mage-VL, first pass 2026-10-13)

`microsoft/Mage-VL` (MageViT + Qwen3-4B) on the `mage_vl` engine,
CUDA bf16, greedy, sampled subsets of the official test sets:

| benchmark | subset | metric | score | unparsed / 0-token |
|---|---|---|---|---|
| POPE | 450 q (random/popular/adversarial) | F1 / acc | 0.599 / 22.4% | 48.7% |
| MME | 280 q (14 sub-tasks) | perception / 1400 | 72.8 | 58.2% |
| MMStar | 144 q (18 categories) | accuracy | 13.9% | 48.6% |

Throughput: **874 questions in 5.7 min ≈ 2.6 q/s** end-to-end (image
base64 transport + prefill + decode; 0.2–0.7 s per question post-warmup).

::: warning Read before quoting
The unparsed / 0-token column dominates: on real images the first
sampled token is very often `<|im_end|>`, so the model returns an empty
body rather than a wrong answer. These numbers measure the **current
state of the Rust port**, not the upstream checkpoint — the [full
diagnosis](https://github.com/iBLOC/neuron/blob/main/docs/superpowers/specs/2026-10-13-mage-vl-vlm-bench.md)
lists the five correctness bugs already fixed (CUDA dtype, 36 GiB KV
prealloc, worker thread, chat template, pad expansion) and the residual
first-token-degenerate issue that is the next fix. Reproduction harness:
`scripts/vlm-bench/`.
:::

## OCR end-to-end latency (PP-OCR tiers × executors)

| model | det+rec weights | tract (cold e2e, ms) | ort (cold e2e, ms) | ort speedup | regions |
|---|---|---:|---:|---:|---:|
| ppocrv5-mobile | 21 MB | 2 922 | 588 | 5.0× | 5 |
| ppocrv6-medium | 139 MB | 7 924 | 1 056 | 7.5× | 5 |
| ppocrv5-server | 173 MB | 13 160 | 1 362 | 9.7× | 5 |

One-shot process on the same 800×400 zh/en page (numbers include
first-use executor setup: tract compiles plans, ort loads onnxruntime).
Text output identical across executors and tiers. The `ort` executor
claws back most of the mobile→server tier gap (588 → 1 362 ms, vs
2 922 → 13 160 on tract).

## ASR latency (SenseVoiceSmall, sherpa engine)

| wav | duration | p50 (ms) | RTF |
|---|---:|---:|---:|
| en.wav | 7.15 s | 482.9 | 0.068 |
| zh.wav | 5.59 s | 361.4 | 0.065 |

RTF < 0.1 = 10× realtime. The tract path produces transcription-parity
output (verified 2026-09-26); its per-wav latency was not captured in
this run.

## Vision memory pooling (TensorPool)

| path | cost |
|---|---|
| fresh `Tensor::zeros(1,3,640,640)` | ~460 µs |
| pooled get+release | ~292 ns (~1574×) |

Poolable YOLOv8n intermediates ≈ 32 MB vs ~200 MB peak without pooling
(~6× reduction).

## Static weights / memory footprints

| model | weights | resident memory |
|---|---|---|
| minicpm5-1b | ~2 GB (bf16) | ~3 GB VRAM (bf16); CPU f32 rows above |
| minicpm5-2b | ~5 GB (bf16) | ~6–7 GB VRAM (bf16); f32 ~10 GB RAM |
| mage-vl (VLM) | 9.5 GB bf16 + 1.07 GB gate | ~11 GB VRAM (bf16, CUDA) |
| voxcpm2 (TTS) | 4.58 GB bf16 + 377 MB fp32 | ~10 GB RAM (f32) |
| laya (decision) | ~500 MB int8 ONNX | int8 |
| PP-OCRv5 mobile (OCR) | 21 MB | |
| PP-OCRv5 server (OCR) | 173 MB | |
| PP-OCRv6 medium (OCR) | 139 MB | +130 MB layout, +368 MB table (optional) |
| all-minilm-l6-v2 (embedding) | ~88 MB | |
| bge-reranker-base (rerank) | ~1.08 GB | |
| yolov8n (vision) | ~6 MB | poolable intermediates ~32 MB |

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

# VLM accuracy: POPE / MME / MMStar against a running mage_vl server
# (reports -> scripts/vlm-bench/results/)
cd scripts/vlm-bench
python prep_datasets.py --bench pope  > data/pope.jsonl
python eval_vlm.py  --base-url http://127.0.0.1:18093 --bench pope

# Allocation / pooling microbenchmarks
cargo bench --bench mobile_bench
cargo bench --bench perf -- --quick

# Per-model sweep (one case per catalog model, in-process, on the local device)
# -> target/model-bench-<device>.md
scripts/fetch-bench-models.sh                 # weights -> ~/.neuron/models/<cap>/<id>/
python scripts/export-yolo-onnx.py             # Ultralytics .pt -> model.onnx + config.json
scripts\cuda-build.bat cargo bench --bench model_bench --features "candle,cuda,tract,ocr,sherpa,voxcpm,chronos2,litert"

# BitNet spike (numbers above, not part of neuron)
# All artifacts under target/bitnet-spike/ are scratch from a one-off
# NO-GO assessment; see target/bitnet-spike/REPORT.md for context.
```

Methodology notes (prompt-nonce anti-caching, client-side timing,
warmup discard, greedy decode) live in each generated report's header.

## See also

- [/reference/models](/reference/models) — supported model × backend matrix
- [/concepts/scheduler](/concepts/scheduler) — how the memory budget trades against these numbers
- [/concepts/engines](/concepts/engines) — when to pick which engine
