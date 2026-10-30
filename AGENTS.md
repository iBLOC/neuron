# neuron (神经元) — AGENTS guide

**Status:** READ-ONLY contributor reference. Updated 2026-10-30 (**MOSS-Image text-to-image capability landed** — the first image-generation capability: `imagegen` + `POST /v1/images/generations`, serving OpenMOSS MOSS-Image natively on candle with the DiT, the Qwen3.5 hybrid-GDN text encoder and the FLUX.2 decoder all vendored; **743 lib tests** default (up from 701); prior 2026-10-26: **VoxCPM2 GPU backend landed** — `--features voxcpm-cuda` + `NEURON_DEVICE=nvidia` runs the 2B model on an NVIDIA GPU through burn's CubeCL CUDA backend in f16; five latent dtype assumptions the f32 path hid were fixed; the CPU stays the default because VoxCPM2's per-patch decode measured ~6× *slower* on an RTX 3090 than on a 12-core CPU; **701 lib tests** default / **734 with `--features voxcpm`** / **735 with `--features voxcpm-cuda`** / **761 with `--features tract`** / **741 with `--features ort`** (all re-measured 2026-10-27 after the RustSec dependency patches); prior 2026-10-25: **TTS model-level streaming landed** — `POST /v1/audio/speech/stream`, VoxCPM2 decodes each latent patch as the model emits it; verified bit-comparable against the full decode on the real `audiovae.pth`; four real-checkpoint load bugs fixed — pth wrapper, VAE module-tree paths, burn-store contiguous index remapping, RoPE head dim — plus the TTS default-shortcut startup panic; prior 2026-10-15: **llama.cpp backend landed** — `NEURON_LLM_RUNTIME=llamacpp`, gated by the `llamacpp` Cargo feature, see the "Llama.cpp (loadable)" section; **681 lib tests** default as of 2026-10-15; prior 2026-10-09 baseline: post-OCR capability (PP-OCR), **679 lib tests** default / **739 with `--features tract`** / **715 with `--features ort`** / **750 with `tract,ort`**; prior 2026-09-27 baseline: post-candle 0.11 upgrade per `huggingface/candle.git#aebc405`; post-`f643c1c` LLM auto-registration + USERPROFILE fallback on Windows GUI processes; **VoxCPM2 (2B/48 kHz) local inference landed** — `NEURON_TTS_RUNTIME=voxcpm2`, direct checkpoint load, see the "VoxCPM2 (loadable)" section; **model scheduler landed** — bounded-memory LLM swap queue behind `/v1/chat/completions` + `/v1/messages`, see the "Model scheduler" bullet and `docs/superpowers/specs/2026-09-29-model-scheduler.md`; GitHub Pages CI now base-URL-configured + adds a `lychee` link-check job; prior 2026-09-25 baseline: post-rename `neure → neuron` per commit `e3d2c41` (2026-09-21); `AppState::register_engine` capability per `bccef7c`; earlier 2026-07-19 baseline: license switch + Shadow decoupling).

## What neuron is

neuron is a **library-only** Rust crate (no standalone binary) that host processes embed via `run_embedded()` to serve OpenAI-compatible inference:

- Loads model weights (LLM, TTS, ASR, Rerank, Embedding, Vision, Decision)
- Runs inference using one of 3 ML frameworks (candle / burn / litert)
- Serves the result on **10+ OpenAI-compatible HTTP endpoints** + 2 liveness endpoints
- **TTS chunked-transfer streaming** (`stream: true` on `/v1/audio/speech` → `Body::from_stream`)
- **TTS model-level streaming** (`POST /v1/audio/speech/stream`) — VoxCPM2 decodes each latent patch into audio the moment the model emits it, so the first bytes arrive after one patch instead of after the whole utterance. The legacy `stream: true` on `/v1/audio/speech` is unchanged (full synthesis, then chunked transfer).
- **ASR multi-format decoding** (WAV, MP3, FLAC, OGG/Vorbis via symphonia, gated by `asr-audio` feature)
- **SenseVoiceSmall via sherpa-onnx** (`NEURON_ASR_RUNTIME=sherpa`, feature `sherpa`) — the official `sherpa-onnx` Rust binding (https://github.com/k2-fsa/sherpa-onnx, 1.13.x). Recognizer construction, fbank, LFR, CMVN and greedy CTC all live inside onnxruntime-backed C++ (`src/asr/sherpa.rs`).
- **SenseVoiceSmall via neuron's own tract ONNX stack** (`NEURON_ASR_RUNTIME=tract`, feature `tract`) — kaldi-style fbank (`src/asr/fbank.rs`), LFR stacking + CMVN stats parsed from the graph's `metadata_props` (`src/asr/onnx_meta.rs`), tract forward, in-repo greedy CTC (`src/asr/sensevoice.rs::ctc_greedy`). The export's dynamic PE chain must be moved out via `scripts/prep-sensevoice-onnx.py` first; the script asserts the export's constants and verifies against the original onnxruntime forward (max diff < `1e-3`, argmax equal across T = 37..250). See `docs/superpowers/specs/2026-09-26-sherpa-onnx-asr-backend.md`.
- **Embedding batched forward** (`encode_batch` + `BatchLongest` padding, single BERT forward)
- **Embedding base64 encoding** (`encoding_format: "base64"` → f32 LE bytes, base64 string)
- **Multi-reranker families** (BGE / mxbai / jina, gated by `NEURON_RERANK_RUNTIME`) — BGE: XLM-RoBERTa cross-encoder, real candle forward. **mxbai: WORKING since 2026-10-30** — mxbai-rerank-base-v2 is a *generative* Qwen2 (~0.5B, tied embeddings, no classifier head despite the model card) that answers `1`/`0`; `MxbaiRerankRuntime` builds the checkpoint's chat-template prompt, runs `ModelForCausalLM::forward`, and scores `sigmoid(logit["1"] − logit["0"])` (verdict ids from `1_LogitScore/config.json`, 16/15). Direction verified on real weights (relevant doc → "1" ≈ 1.00, irrelevant → "0" ≈ 0.00). Two load-bearing details: pass the UNPREFIXED VarBuilder to `ModelForCausalLM::new` (it adds `model.` itself), and use `ModelForCausalLM` not `Qwen2Model` — the bare model returns hidden states, not logits (the first cut scored hidden dims 15/16; found by the generation probe `examples/mxbai_score_check.rs`). **jina: WORKING since 2026-10-30** — jina-reranker-v2-base-multilingual's released checkpoint uses the flash-attention module tree (`roberta.encoder.layers.N.mixer.Wqkv` fused [Q;K;V], `norm1`/`norm2`, `mlp.fc1/fc2`, `emb_ln`), which vanilla candle `xlm_roberta` cannot load by name. `JinaRerankRuntime::load_remapped_safetensors` loads the safetensors into a HashMap and REMAPS the keys to vanilla names — splitting each fused `Wqkv` [3·hidden, hidden] linear into separate query/key/value linears by rows, renaming `out_proj`→`attention.output.dense`, `norm1`→`attention.output.LayerNorm`, `norm2`→`output.LayerNorm`, `mlp.fc1/fc2`→`intermediate.dense`/`output.dense`, `emb_ln`→`embeddings.LayerNorm`, and `encoder.layers.N`→`encoder.layer.N` — then builds the standard `XLMRobertaForSequenceClassification`. No architecture port needed: the math is vanilla, only the names differed. Direction verified on real weights (Paris doc 0.83, irrelevant ≈ 0.03). Gotcha learned: the remap must translate the SUBMODULE names too (`out_proj`→`dense`), not just split the fused tensor — the first cut produced `.dense.out_proj.weight` and the loader named the exact missing tensor. The three families each declare their OWN engine (`candle` / `mxbai` / `jina`).
- **Cohere rerank adapter** (`NEURON_RERANK_RUNTIME=cohere`) — proxies requests to Cohere's hosted `/v1/rerank` API; no local weights, requires `NEURON_COHERE_API_KEY`
- **Model scheduler** (2026-09-29; `src/scheduler/`, spec `docs/superpowers/specs/2026-09-29-model-scheduler.md`) — LLM requests on `/v1/chat/completions` + `/v1/messages` (OpenAI & Anthropic, SSE + non-streaming) queue against a memory budget: same-model runs immediately, a different model loads alongside when it fits, otherwise queued resident-model work drains first and idle residents are evicted LRU-first before the swap; idle models recycle after `NEURON_SCHEDULER_IDLE_SECS`. Memory reference values ship in `assets/model_memory_profiles.json` and are corrected at runtime by observed unload RSS (EWMA → `~/.neuron/models/scheduler/observed_profiles.json`). Disable with `NEURON_SCHEDULER_ENABLED=false` for the legacy always-load path. `LlmRuntimeRegistry`'s loaded cache is now a **std** mutex (never held across `await`) with a synchronous `unload()` — the scheduler evicts from lease-drop contexts; `EngineRegistry::unregister` covers the canonical table.
- **Laya decision scoring** (`POST /v1/decisions`) — non-autoregressive discriminative scorers (int8 ONNX, encoder-only, one forward per question) served natively via `tract`; scores a question's options plus an act-vs-escalate head with calibrated confidence. Convai ships **safetensors only, no ONNX** (`convaiinnovations/laya{,-multilingual,-typed-decisions}`), so the graph is upstream `scripts/export_onnx.py` → this repo's `scripts/prep-laya-onnx.py`; `laya` is deliberately not in `SUPPORTED_CATALOG` since nothing downloadable is loadable as-is. See `docs/superpowers/specs/2026-09-24-laya-decision.md`.
- **FlashAttention** for Llama 2/3 + Mistral (upstream candle-transformers) + Qwen 2/2.5/3 (via `src/llm/vendor/` with `use_flash_attn` + `From<upstream::Config>` + flash-attn forward branch); Phi-3/ChatGLM still warn + fall back to SDPA. Requires `--features flash-attn` + CUDA. **Qwen 3.5 is NOT covered** — its architecture is not the Qwen3 dense one (see "Qwen 3.5" below).
- **Qwen 3.5 is NOT served by the candle backend** (verified 2026-10-29 against the upstream `Qwen/Qwen3.5-*` configs). The real Qwen 3.5 `config.json` uses `model_type: "qwen3_5"` and nests the text tower under `text_config`, whose layers mix `linear_attention` and `full_attention` (`full_attention_interval: 4`, `attn_output_gate: true`, `partial_rotary_factor: 0.25`, mrope, `mtp_num_hidden_layers: 1`) — a gated-delta-net stack that upstream candle-transformers has **no module for**. `detect_arch` therefore rejects the config with an explicit "Qwen 3.5 is not supported by the candle backend" error (it must not fall through to the dense Qwen3 loader). `qwen3_5_moe` is in the MoE rejection list. Qwen 3.5 is servable through the `mistralrs` backend (mistral.rs v0.8.22 ships `MultimodalLoaderType::Qwen3_5` / `Qwen3_5Moe` and the auto-pipeline routes `Qwen3_5ForConditionalGeneration` to it) or through `llamacpp` from a GGUF (llama-cpp-sys-2 0.1.159's bundled llama.cpp implements `LLM_ARCH_QWEN35` / `LLM_ARCH_QWEN35MOE`). Every upstream size (0.8B / 2B / 4B / 9B / 27B dense + 35B-A3B / 122B-A10B / 397B-A17B MoE) is a **vision-language** model.
- **Criterion benchmark suite** under `benches/perf.rs` (14 `bench_function` calls across all 5 model types; previously reported as 22 — corrected 2026-07-19)
- **Per-model benchmark harness** under `benches/model_bench.rs` (added 2026-10-29): one case per model in the capability catalog, run **in-process** through the public registry API (`register` → `runtime_for`) on the local device. `harness = false` (real forward passes take seconds, so the harness does its own warmup + p50/p95 and writes `target/model-bench-<device>.md`). Models whose weights are absent are reported as `skipped` with the reason, so the report doubles as a coverage matrix. Weights: `scripts/fetch-bench-models.sh`; the vision rows need `python scripts/export-yolo-onnx.py` (Ultralytics `.pt` → `model.onnx` + `config.json`). **CUDA builds need the MSVC environment** — nvcc cannot find `cl.exe` from a plain shell, so use `scripts\cuda-build.bat cargo bench --bench model_bench --features "candle,cuda,…"`.
- Runs only as an in-process library — the host binds the port, owns the process lifecycle, and calls `request_shutdown()` + `join().await` for graceful drain
- **Multi-source model registry** (`huggingface` + `hf-mirror` + `modelscope`): the puller accepts `<source>:<repo>` references and the platform default source is `modelscope` (set via `NEURON_DEFAULT_SOURCE`). `HF_TOKEN` and `MODELSCOPE_TOKEN` env vars authenticate against gated/private repos on their respective platforms.
- **Catalog dedup by `(engine, id)`**: a single `CatalogEntry` may carry multiple `SourceRef` entries (one per repo) so the UI can offer "download from HF / mirror / ModelScope" without re-listing the model three times.
- **Puller management API**: `GET /v1/models/pull` lists active + completed jobs, `DELETE /v1/models/pull/{job_id}` cancels in-progress downloads (writes `JobStatus::Cancelled`). Failed downloads auto-clean the partial on-disk directory.
- **Platform default model**: `minicpm5-1b` (OpenBMB MiniCPM5 1B, Apache 2.0, standard `LlamaForCausalLM`, 131K context, hybrid `<think>` reasoning). Loaded via the `candle` runtime — auto-detected from `config.json` `architectures = ["LlamaForCausalLM"]`. **FIXED 2026-10-30 (gap found 2026-10-29 by `benches/model_bench.rs` on real weights)**: candle's `llama::Config` had no `head_dim` field, so `q_proj` was sized `[hidden, hidden]`, but MiniCPM5-1B's attention is 16 heads × head_dim 128 = 2048 ≠ hidden 1536 — the loader panicked with a shape mismatch, and candle's `Block::load` even `unwrap()`s it (process abort). `src/llm/vendor/llama_hd.rs` now vendors llama with explicit `head_dim` (upstream-compatible when the field is absent — every llama dispatch routes through it), fixes the q/kv projection sizing AND the attention-output reshape (`heads × head_dim` width before `o_proj`, not `hidden_size`), and propagates load errors instead of unwrapping. The 1B measured **62.4 tok/s on an RTX 3090** — the fastest LLM row. The config-dispatch unit test never caught this because it never loads weights; the per-model bench did.

## When to edit what

| You want to... | Edit file | Caveat |
|---|---|---|
| Add a new LLM architecture | `src/llm/candle_runtime.rs` | Update `Qwen3::new` / use correct model struct from `candle_transformers::models` |
| Add a new TTS model | `src/tts/voxcpm_burn/` | 0.5B: vendored from `madushan1000/voxcpm_rs`, pre-converted burn `.mpk` weights. VoxCPM2: `voxcpm2_model.rs` + `audiovae_v2.rs`, loads upstream safetensors/pth directly via `weights.rs` |
| Add a new model type (e.g. embedding) | Create new `src/<model_type>/mod.rs` | Follow the 10-step pattern (see the "Adding a new model type" checklist below and prior specs under docs/superpowers/specs/) |
| Add a new vision task (classify/segment/pose) | Extend `src/vision/lora.rs` + `candle_yolo.rs` | v1.0 implements full task dispatch + response shape + LoRA; detect head has preprocessing/NMS wired, classify/segment/pose are stubs pending heads |
| Add a new vision model family (RT-DETR / DETR / RF-DETR / Florence-2) | Create new `<family>.rs` in `src/vision/`, add a new `VisionImpl` variant, wire the loader | Wire into `default_capabilities()` in `src/capabilities/catalog.rs` |
| Add a new output layout to the ONNX backend | Extend `OrtOutputLayout` in `src/vision/ort_runtime.rs` and add a new decoder function (see `decode_yolov8_output` / `decode_rfdetr_output` for templates) | The unified `decode_detection_output` dispatcher routes by layout; add a new match arm |
| Add a new ONNX executor to the `ort` backend | Replace `build_session` in `src/vision/ort_runtime.rs` | The `OrtSession` type alias lets you wire in `ort` / `tract` / `onnxruntime` without touching the rest of the runtime |
| Add a new engine backend | `src/llm/` (new file) + wire into registry `runtime_for()` dispatch + add Cargo feature | Match the `LlmRuntime` trait shape |
| Add a new model type (register it) | `state.rs` + `config.rs` (ModelRegistration) | Add registration processing loop in `ServerState::new` for the new capability |
| Wire registrations from config | `state.rs` (process `config.registrations` per capability) | Each cap needs its own `Impl::parse` loop |
| Add a new model source (e.g. ModelScope, hf-mirror) | `src/models/<name>.rs` + `src/models/source.rs` + `src/server/state.rs` | Implement `Source` trait; register in `SourceRegistry::with_defaults()` AND in `ServerState::new`'s source registry block. If the source speaks HF-protocol use `HuggingFaceSource::with_identity()` so it gets a unique id. |
| Add a new HTTP endpoint | `src/server/mod.rs` (handler) + `routes` block | Follow OpenAI shape if it's a vendor endpoint |
| Add a new env var | `src/config.rs` (default field) + `src/server/mod.rs` (env-var dispatch in `ServerState::new`) | Document in docs/reference/env-vars.md (both locales) and docs/concepts/engines.md |
| Add a new Tauri command (when embedded) | `src/embedded.rs` (the public API) + host project's `src-tauri` integration | Don't break the `NeuronHandle` contract |
| Add a new reranker family (e.g. Cohere) | `src/rerank/<name>.rs` + register `RerankImpl` variant in `src/rerank/mod.rs` | Match `RerankRuntime` trait shape; add env-var arm in `state.rs` |
| Add a new TTS stream format (mp3 streaming, opus) | extend `synthesize_stream` impl or override per-runtime | The trait's default impl uses 16 KiB chunks; override if the runtime can yield in a more efficient shape |
| Add a new ASR audio format (e.g. AAC) | `src/asr/audio.rs` | Add a `detect_format` arm + decode path via symphonia |
| Add a new SenseVoice engine | `src/asr/{sherpa,sensevoice}.rs` + `scripts/prep-sensevoice-onnx.py` | sherpa runs the export unchanged (`NEURON_ASR_RUNTIME=sherpa`); the tract path needs the prep script first because tract 0.21 cannot translate the export's dynamic `Range` PE chain |
| Add a new decision model / scorer | `src/decision/tract_session.rs` (`from_dir`) + `scripts/prep-laya-onnx.py` | The graph must be prepped first (grounded dims + `DynamicQuantizeLinear` decomposition) or tract cannot load it. Keep `OUTPUT_NAMES` in sync with the export's `graph.output` order — tract discards output tensor names. |
| Add a new PP-OCR model variant (e.g. v5 mobile tiers, multilingual rec) | `src/capabilities/catalog.rs` (catalog entry) + the model dir layout | A model dir = `det.prepped.onnx` + `rec.prepped.onnx` + `dict.txt`, optional `rec-<pack>/` language-pack subdirs (all prepped with `scripts/prep-ppocr-onnx.py` — metadata-only: dotted TDim symbols + `floor(...)` value_info hints break tract). Register with `capability = "ocr"`, `engine = "paddle-ocr"`. Pack dicts are `ppocr/utils/dict/ppocrv5_<script>_dict.txt`, not the v3-era `<script>_dict.txt` |
| Add a new OCR engine (e.g. `ort` executor, PP-Structure layout) | `src/ocr/paddle_runtime.rs` (`PlanCache` session) or a new `OcrImpl` variant in `src/ocr/mod.rs` + registry arm | The det/rec graphs take dynamic H×W / W, so tract plans are compiled per concrete shape — keep the det (H,W) plan cache and the 64 px rec width-bucketing, or compile time dominates |
| Extend the model scheduler to another capability | handler: `acquire` → execute with the lease (streaming: `HoldingStream`); add an `Arc<ModelScheduler>` + unload handle in `ServerState::new` | The scheduler is capability-generic (`Capability` keyed); v1 wires LLM only. The unload handle must reach every cache holding that capability's weights. |

### OCR (PP-OCR, loadable, 2026-10-09)

`POST /v1/ocr/recognize` serves a two-stage PP-OCR pipeline (DB++ text
detection + SVTR-LCNet recognition), gated by the `ocr` meta feature —
enabled by either executor feature, `tract` (default, pure Rust) or `ort`
(fast, host-supplied onnxruntime).
Weights are the official ONNX exports from the HF `PaddlePaddle/PP-OCRv5_*_onnx`
/ `PP-OCRv6_*_onnx` repos (Apache-2.0), prepped with
`scripts/prep-ppocr-onnx.py` — a metadata-only rewrite (Paddle2ONNX emits
`DynamicDimension.0` dotted symbol names and `floor(...)` value_info shape
hints that tract's TDim parser rejects; the export graphs themselves are pure
CNN + MatMul/Softmax with no `Range`/custom ops). Verified 2026-10-09 against
onnxruntime on all six v5-mobile/server + v6-medium det/rec exports
(max abs diff ≤ 2e-6 det maps, ≤ 2e-4 rec softmax) — parity spikes under
`target/ocr-spike/` + `target/ort-spike/` are scratch, not tracked.

Load-bearing points:

1. **`OcrRuntimeRegistry::default_runtime` never loads** — it always returns
   `UnavailableOcrRuntime`. Blocking on `runtime_for` from `ServerState::new`
   panics with "Cannot start a runtime from within a runtime" (every
   `#[tokio::main]` host); the lazy load happens on the handler's
   `runtime_for` async path. Same shape as the decision registry.
2. **Plan caches bound tract's per-shape compile cost.** Both graphs have
   dynamic extents; `PlanCache` compiles one plan per concrete (h, w), capped
   at 8 entries with FIFO eviction. rec widths are bucketed to multiples of
   64 (`rec_bucket_w`) so line crops don't multiply cache misses.
3. **DB++ post-processing is pure Rust** (`src/ocr/db_postprocess.rs`):
   binarize → 8-connected components → convex hull → min-area rect (rotating
   calipers) → box-thresh filter → unclip expansion
   (`offset = area·ratio/perimeter`). Output boxes are axis-aligned; rotated
   text crops keep their background (acceptable v1 trade-off vs OpenCV's
   polygon clipping).
4. **CTC layout mirrors `CTCLabelDecode`**: token 0 = blank, last = space,
   dict lines between. `ppocrv5_dict.txt` (18383) + 2 = the 18385-vocab
   export; v6 = 18710. The runtime errors (`invalid_input`) on vocab/graph
   mismatch instead of decoding garbage. Rec outputs are batch-major
   `[B, T, vocab]` in these exports (not Paddle's native `[T, B, V]`).
5. **Preprocessing replicates PaddleOCR verbatim, including BGR**: det uses
   ImageNet means applied to B, G, R in that order (channel 0 ← rgb[2]);
   rec uses `(x/255 − 0.5)/0.5` with zero-pad in normalized space, right-padded
   to the 64 px bucket. `DetResizeForTest` caps the longest side at
   `det_max_side` (default 960; v6 medium release default is 736) and rounds
   both sides to the *nearest* multiple of 32.
6. **Executors: tract is the default, `ort` the fast path** (`NEURON_OCR_EXECUTOR=ort`).
   Both implement the `OcrSession` trait in `src/ocr/paddle_runtime.rs`.
   tract CPU runs these CNNs 5-12× slower than onnxruntime (measured
   2026-10-09: v6 medium det 3.7 s vs 0.5 s at 736²; rec 120-233 ms vs
   10-35 ms per line) and pays a per-shape plan compile. The `ort` executor
   (`src/ocr/ort_session.rs`, feature `ort`) runs dynamic shapes natively —
   no plan cache — and measured 10.3× faster end-to-end on the smoke
   workload (526 ms vs 5.4 s including first compiles). The `ort` feature
   uses **load-dynamic** (host supplies `onnxruntime.dll` via
   `ORT_DYLIB_PATH` or `PATH`; rc.12's vitis.rs gating bug #547 is fixed in
   rc.13) — do NOT switch to static `download-binaries`: its /MD CRT
   objects conflict with the /MT objects cc-based deps (`esaxx_rs`) emit on
   Windows and fail linking with LNK2038. ort sessions live in a fixed pool
   (`NEURON_OCR_SESSIONS`, default 2) because `run` takes `&mut self`;
   tract plans are shared through `Arc` and need no pool.
7. **Multilingual rec packs** (`src/ocr/lang_route.rs`): optional
   `rec-<pack>/` subdirectories of the model dir hold per-script rec graph
   + dict pairs (HF `PaddlePaddle/<script>_PP-OCRv5_mobile_rec_onnx`). A
   request's `languages` field routes the request to ONE rec model: BCP-47
   codes through the static table (`fr`→`rec-latin`, `ko`→`rec-korean`,
   `ru`→`rec-cyrillic`, `ar`→`rec-arabic`, `hi`→`rec-devanagari`, plus
   `th`/`el`/`te`/`ta`; `zh`/`en`/`ja` and unrouted codes stay on the
   default rec), or an exact pack name (`"latin"`) as the escape hatch.
   Packs build lazily on first routed request and cache. **Dict gotcha**:
   the v5 language packs pair with `ppocr/utils/dict/ppocrv5_<script>_dict.txt`
   (latin: 836 lines → the 838-vocab export), NOT the older v3-era
   `<script>_dict.txt` files — the runtime's vocab/graph size check rejects
   the wrong pairing with a clear error. Response `languages` reports the
   pack actually used (`"default"` / `"latin"` / …).
8. **Optional layout model** (`src/ocr/layout.rs`, `ort` executor only):
   `layout.onnx` (PP-DocLayoutV3, HF `PaddlePaddle/PP-DocLayoutV3_onnx`,
   ~130 MB) in the model dir adds document structure — `layout[]` regions
   (25 classes: text/table/image/formula/…) in the response plus a per-line
   `kind` stamped from the smallest containing region. The yml lies about
   preprocessing; the verified contract is: RGB stretched to 800×800,
   `pixel/255.0` (raw 0-255 yields degenerate ~0.05 scores),
   `im_shape`=(h,w), `scale_factor`=(800/h, 800/w), output boxes
   `×(800/W, 800/H)` to reach source pixels. The op set (GridSample,
   ScatterND, dynamic Range) exceeds tract 0.21 — tract-only builds ignore
   the file with a warning. (No runtime kill-switch — omitting the file is
   the way to skip it.)
9. **Table structure (SLANeXt wired, `ort` only)**: `table.onnx`
   (`PaddlePaddle/SLANeXt_wired_onnx`, ~368 MB) in the model dir + layout
   running turns every `table` layout region into assembled `<table>` HTML
   on `LayoutRegion::html`: the region is re-cropped, SLANeXt predicts one
   cell quad per structure step (`loc [1,T,8]`, xyxyxyxy normalized to the
   512 canvas; `struct [1,T,50]` gates which steps are cells), and the
   already-recognized lines are assigned to cells by center containment.
   Decoding hard-won facts (all verified 2026-10-09): the alphabet is
   CTC-layout (0=blank, 1..48 = yml dict order, 49=space — `id 5`=`<tr>`,
   `id 6`=`</tr>` on a 4×4 grid); the wired export marks cells with
   `<td`-variant AND attribute tokens (`rowspan`/`colspan` carry the cell
   quads), so cell = non-structural id (`STRUCTURAL_IDS` =
   0,1,2,3,4,5,6,9,10,49) with a valid quad; preprocessing is RGB (the
   table ymls lie about BGR like layout's does), keep-ratio into 512²,
   top-left paste, `/255` + ImageNet. Known v1 limits: overlapping quads
   can duplicate/merge cells on synthetic tables (dedupe needs real-world
   tuning); colspan/rowspan spans are not reconstructed. Omitting
   `table.onnx` skips it (no runtime kill-switch).
10. **Table structure: SLANet_plus export is broken, use SLANeXt** (spiked
   2026-10-09, scratch under `target/ocr-table-spike/`): the official
   `PaddlePaddle/SLANet_plus_onnx` ONNX runs but its Loop-based
   autoregressive decoder is decoupled from the image — white noise, a
   blank canvas and a real table all decode to the same degenerate
   `<tr> rowspan="20" </tr>` sequence (RGB vs BGR identical too). The
   working PP-StructureV3 stack is **SLANeXt** (non-autoregressive, no
   Loop — that's why it exports cleanly): HF `PaddlePaddle/SLANeXt_wired_onnx`
   / `SLANeXt_wireless_onnx` (~368 MB each, wired = bordered tables),
   `PP-LCNet_x1_0_table_cls_onnx` (6.8 MB, wired/wireless router) and
   `RT-DETR-L_wireless_table_cell_det_onnx` (129 MB). Reference decode:
   ppocr-rs (`github.com/dariofinardi/PaddleOCR-OCR-rs`, Apache-2.0) —
   512 square canvas, ImageNet norm on **RGB** (the table ymls also lie
   about BGR), `<td`-like tokens carry the loc regression; the SLANeXt
   vocab is `table_structure_dict.txt`-shaped with merged `<td></td>`
   tokens. Not wired yet — next increment.

### MOSS-Image (text-to-image capability, 2026-10-30)

neuron gained its first **text-to-image capability**: `imagegen`, served
at `POST /v1/images/generations` (OpenAI Images shape, `b64_json`).
The model is OpenMOSS **MOSS-Image** (Apache-2.0, released 2026-10-16):
a 5B single-stream DiT + a Qwen3.5 hybrid-GDN text encoder + a FLUX.2
small decoder VAE, 13.6B params / ~27.2 GB of BF16 safetensors in one
HF repo. Activation: `NEURON_IMAGEGEN_RUNTIME=moss-image` (+ the
default `candle` feature); weights load from the upstream repo layout
**directly** — no conversion. Full rationale + porting notes:
`docs/superpowers/specs/2026-10-30-moss-image-text-to-image.md`.

Everything is vendored into `src/imagegen/moss_image/` — candle 0.11
has no module for the gated-delta-net text stack, and the DiT is a
single-stream joint-sequence transformer (shared time modulation,
4-axis RoPE), not a Flux double-stream. Load-bearing points:

1. **The text encoder is the expensive part** (~8.4B of the 13.6B).
   `model_type: qwen3_5` is 32 text layers = 24 Gated DeltaNet + 8 full
   attention (`head_dim` 256, partial rotary 0.25, interleaved mrope,
   `attn_output_gate`). `src/llm/candle_arch.rs` maps `qwen3_5` onto
   the *dense* Qwen3 loader — that path cannot read this checkpoint, so
   the hybrid stack is ported here (prefill-only; the `visual.*` tower
   is never touched).
2. **Three silent-NaN traps in the port** (all now guarded by unit
   tests): an RMSNorm must *divide* by the RMS (`broadcast_mul` with
   `var^-0.5`, NOT `broadcast_div` — the inversion grows activations by
   ~10⁶ and makes attention scores `inf`); the GDN tril mask must be
   applied **before** the `exp` (the reference does `.tril().exp()`,
   and `exp(+hundreds)` → `inf` → `inf·0 = NaN`); and the chunk
   recurrence must keep `g_last` **un-exponentiated** when computing
   `(g[..., -1] − g[..., :]).exp()`.
3. **candle's `+`/`-` do not broadcast** and its `matmul` is 4-D only —
   the GDN chunk kernel folds its NC axis into the batch dim (or
   squeezes per chunk) before every matmul, and the UT transform's
   `attn[..., i, :i]` must squeeze the fixed row index like torch does.
   Note also that the reference **reassigns** `value = attn @ v_beta`
   before the chunk loop reads it — a port that keeps the raw `v`
   computes a different recurrence (it drops the `beta` factor). A
   `beta = 1` unit test cannot catch that; the analytic anchor uses
   `beta = 0.5` for exactly that reason.
4. **The FLUX.2 decoder's mid block runs at the largest channel count**
   (384), while `conv_norm_out`/`conv_out` run at the smallest (96).
   The latent path is BN de-normalization → 2×2 pixel unshuffle →
   `post_quant_conv` → decoder, all fp32.
5. **The scheduler's sigma grid is snapped to the training grid**, not
   continuous — a 50-step run has exactly 50 distinct sigmas
   (`1.0, 0.996, 0.992, 0.987, 0.983, …, 0.024`, verified against the
   upstream Python). The CFG negative branch **skips DiT block 22**
   (two forwards per step, not three).
6. **CUDA-only in practice.** f16 weights ≈ 27 GB resident (f32 ≈ 54 GB,
   registration budget 28 GB); CPU is a multi-PFLOP-per-image workload
   and candle's CPU backend has no BF16 matmul. A 24 GB card needs the
   text encoder unloaded before the DiT runs — that staged residency is
   **not implemented yet** (both components load eagerly), and the
   capability is **not wired into the model scheduler** (LLM-only), so
   a 27 GB image model cannot coexist cooperatively with an LLM yet.
7. **Numerical parity against the upstream implementation**
   (`src/imagegen/moss_image/parity.rs` + the fixtures in
   `testdata/`). `scripts/gen-moss-parity.py` runs the **upstream code
   itself** — the GDN kernel from the installed `transformers`
   package (`transformers.models.qwen3_5.modeling_qwen3_5`), the DiT
   from the upstream `pipeline_moss_image.py` — and dumps
   deterministic inputs plus the reference outputs; nothing upstream
   is re-typed, so it is a real reference rather than a second
   transcription. The Rust test compares element-wise:
   `cargo test --lib moss_image::parity -- --nocapture`. Measured
   2026-10-30 (fp32, CPU): GDN kernel 1.1e-6, DiT 2.0e-6, DiT with a
   skipped block 1.3e-6 (float32 summation order, nothing semantic).
   **This caught the last two silent bugs**, which shape-level tests
   cannot see: the modulation chunks were read in the wrong order
   (upstream unpacks `shift_a, scale_a, gate_a, shift_f, scale_f,
   gate_f` — the port had scale and shift swapped), and the attention
   gate was applied to the wrong side of the output projection
   (upstream: `x + proj['o'](attn) * gate_a`; the port:
   `proj['o'](x + attn * gate_a)`). Before the fix the DiT's relative
   error was 0.38.
8. **Real-weight verification** (`src/imagegen/moss_image/real_weights.rs`
   + `scripts/gen-moss-real-ref.py`): the reference side runs the
   upstream implementations on the **real checkpoint** (transformers'
   `Qwen3_5Model`, the repo's own `pipeline_moss_image.py`,
   diffusers' `AutoencoderKLFlux2`) and dumps their outputs; the
   Rust test runs neuron's port on the same weights and compares
   element-wise: `cargo test --lib real_weights -- --ignored --nocapture`.
   The **VAE matches on the real weights** (relative error 1.5e-5
   against diffusers, measured 2026-10-30). The text encoder and the
   DiT need a host with a CUDA-enabled PyTorch and more RAM than a
   16 GB box (the text encoder is 8.4B parameters — a fp32 reference
   does not fit), so they are covered by the formula-level parity
   above instead. The tests **skip** when their reference is missing,
   so the suite stays green without the checkpoint.
   **This caught the prompt-template bug**: the upstream chat template
   renders `<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n<think>\n`
   and neuron's hard-coded template had dropped the four turn
   markers — the model would have seen a malformed conversation on
   every request. The check compares the rendered text **and** the
   token ids (32 for the reference prompt). Note that the turn
   markers are invisible in most terminal output (they look like
   `<|…|>` and get filtered), so the comparison is done on the raw
   bytes, not on what a human reads.
9. **Staged residency** — the pipeline is ~27 GB at f16, which does
   not fit a 24 GB card, so the components load **per stage**: the
   text encoder encodes the prompt and is dropped, the DiT samples
   and is dropped, the VAE decodes. Peak residency is one component
   (~17 GB). `NEURON_IMAGEGEN_RESIDENCY=cached` keeps them all
   resident instead (faster, needs the memory).
10. **No real-weight run of the full pipeline yet.** The port's weight
   paths and shapes are verified against the upstream checkpoint
   (below), and the forward pass matches upstream numerically, but no
   image has been generated from the real 27 GB checkpoint — image
   quality is unverified.
11. **Weight-path + shape verification**
   (`src/imagegen/moss_image/weight_names.rs`): a recording
   `VarBuilder` backend answers every request with a one-element
   tensor and remembers the path **and the requested shape**, so
   building the three modules yields exactly what the port asks for
   (without materializing 27 GB). Two `#[ignore]`d tests fetch the
   upstream index files + shard headers (a few MB) and diff:
   `cargo test --lib verify_weight_paths_against_upstream -- --ignored`
   and `cargo test --lib verify_weight_shapes_against_upstream -- --ignored`.
   Verified 2026-10-30 against `OpenMOSS-Team/MOSS-Image-Base`:
   transformer 462/463 read (only `rope.bands` — a derived constant
   neuron rebuilds), text_encoder 426/759 (the 333 unread are all
   `visual.*` — text-only inference skips the vision tower), VAE
   142/251 (the rest is the encoder half of a decode-only VAE). Zero
   missing paths, zero spurious requests, and **all 1,030 requested
   shapes match** the upstream headers (so the Linear orientations and
   conv layouts are right). The **unused** direction is the one that
   catches a silent mistake: `linear_no_bias` where the checkpoint
   carries a bias leaves that bias unread and the model quietly wrong.

### VoxCPM2 (loadable, 2026-09-27)

The OpenBMB 2026-04 release of VoxCPM (2B params, 48 kHz AudioVAE V2, 30 languages, three cloning modes — Voice Design, Controllable Cloning, Ultimate Cloning) is now **loadable** via `NEURON_TTS_RUNTIME=voxcpm2` (feature `voxcpm`). The former P0 list is closed except model-level streaming:

1. ~~**AudioVAE V2 port**~~ — landed as `src/tts/voxcpm_burn/audiovae_v2.rs`: same encoder as V1 (reused), decoder gains a sixth upsampling stage (`decoder_rates = [8, 6, 5, 2, 2, 2]`, `decoder_dim = 2048`, total ×1920 → 48 kHz) plus per-block **sample-rate conditioning** (`SampleRateConditionLayer`, `scale_bias` embeddings indexed by `torch.bucketize(sr, [20000, 30000, 40000])`; 48 kHz → bucket 3).
2. ~~**2B-parameter model config**~~ — landed as `src/tts/voxcpm_burn/voxcpm2_model.rs` (`VoxCPM2Config` serde-mirrors the upstream `config.json`; `VoxCPM2<B>` model). V2 deltas vs the V1 vendored tree: new `fusion_concat_proj` (`Linear(2h → h)`) fusing encoder output + masked feature embed into the residual LM's input; `residual_lm_no_rope: true` (RALM drops RoPE — `MiniCPMConfig::no_rope`, `rope_emb` is now `Option`); LocDiT V2 feeds the LM condition as **two raw prefix tokens** (`cat(lm_to_dit, res_to_dit)` along features, timestep as a separate token, slice offset `prefix + 2 + 1`); patch_size 4; FSQ latent 512; 28L/2048h TSLM (vocab 73448); 12L LocEnc + LocDiT.
3. ~~**Voice cloning wire format**~~ — `SpeechRequest` gains optional `reference_audio` (base64 / `data:` URI / server-local file path, decoded to mono 16 kHz via `src/tts/clone_audio.rs`, 30 s cap) and `prompt_text` (transcript). Reference-isolation prefix uses sentinel tokens 103/104. **Voice Design needs no new field**: upstream convention is a "(description)" prefix on `input`. Cloning ignores `stream: true` (returns full WAV). Continuation + combined prompt modes exist at the model layer (`ClonePrompt`) but only reference-isolation is wired to HTTP so far.
4. ~~**Multi-language (30)**~~ — model-side; no `lang` field needed (upstream auto-detects from text; the multichar-Chinese tokenizer wrapper `VoxCpm2Tokenizer` splits pure-CJK multi-char vocab entries per char, mirroring upstream `mask_multichar_chinese_tokens`).
5. ~~**Streaming override — pipeline landed, HTTP wiring blocked on fidelity**~~ — **wired and verified (2026-10-25)**. The old blocker was an artifact of the unit test's randomly-initialized toy weights: the per-chunk boundary taps get amplified through the upsampling stages, so that test could not assert fidelity. On the shipped `audiovae.pth` the streamed decode reproduces the full decode essentially bit-for-bit (measured with `examples/voxcpm2_stream_smoke.rs`): **stage A** (real-encoder latent) max_abs_diff 6.3e-6 / 9.4e-6, pearson r = 0.999999, identical sample count; **stage B** (2B-LM-generated latent) max_abs_diff 7.6e-7, pearson r = 1.000000. The endpoint is `POST /v1/audio/speech/stream` (`src/server/handlers.rs::audio_speech_stream`); the trait method is `TtsRuntime::synthesize_stream_incremental` (`src/tts/mod.rs`), overridden by `VoxCpm2TtsRuntime` (`src/tts/voxcpm2.rs`). `response_format` picks the container: `wav` (default — 44-byte header with `0xFFFFFFFF` placeholder sizes, the standard streamed-WAV convention) or `pcm` (raw 16-bit LE mono). The legacy `/v1/audio/speech` `stream: true` path is **unchanged** (full synthesis, then 16 KiB chunked transfer) so existing hosts see byte-identical output.

**End-to-end HTTP numbers** (`examples/voxcpm2_stream_http_smoke.rs`, real checkpoint, 12-core desktop CPU): first byte after **51.6 s** on the streaming endpoint vs **651.0 s** on `/v1/audio/speech` — a **12.6×** first-byte speedup — and both responses carried **exactly 92 204 bytes** (8 streamed chunks over 430 s), an end-to-end confirmation that the streamed audio is the same audio. Absolute latency is CPU-bound: the 2B model generates ~111 s per latent patch on an ndarray CPU backend, so the streaming win is "hear the first syllable 12× sooner", not "realtime" — a CUDA backend is what turns that into actual realtime.

**Weights load directly — no conversion**: `src/tts/voxcpm_burn/weights.rs` reads `model.safetensors` (SafetensorsStore) / `audiovae.pth` (PytorchStore) via burn-store's snapshot API and adapts per tensor: bf16/fp32 → f32 (ndarray has no bf16 matmul), PyTorch `[out,in]` → burn 0.21 `[in,out]` Linear transpose (2-D `*.weight` not containing `embed`), norm `*.weight` → `*.inner.gamma` rewrite, and `apply(..., skip_enum_variants=true)` to bridge PyTorch's flat `decoder.model.N.…` paths to burn's enum-wrapped module tree. Required model dir: `config.json` + `tokenizer.json` + `model.safetensors` (4.58 GB bf16) + `audiovae.pth` (377 MB fp32) — exactly the upstream download. **Budget ~10 GB RAM** (2.29B params as f32). The registry dispatches `model_id == "voxcpm2"` → `VoxCpm2TtsRuntime` under the shared `TtsImpl::Burn`; `NEURON_TTS_RUNTIME=voxcpm2` registers the catalog id.

**Real-checkpoint load fixes (2026-10-25)** — the "loads upstream checkpoints directly" claim was never exercised against the real files until the streaming smoke pulled `openbmb/VoxCPM2` from ModelScope. Three loader bugs and one model bug surfaced; all are fixed:

1. **`.pth` wrapper** — upstream ships `audiovae.pth` as a Lightning-style `{metadata: {...}, state_dict: {...}}`, not a flat state dict. The loader now tries `with_top_level_key("state_dict")` first and falls back to the flat read (burn-store errors with `KeyNotFound` rather than falling back itself).
2. **VAE module tree** — upstream nests the decoder under `decoder.model.N` and the sample-rate conditioning under `decoder.sr_cond_model.N`; neuron's burn tree keeps them as siblings (`decoder` / `sr_cond_model`). `remap_vae_path` rewrites the prefixes. The LM checkpoint (`model.safetensors`) has no `decoder.*` keys at all, so the rewrite is a no-op there (verified: 577 tensors applied, 0 missing, 0 unused).
3. **Contiguous index remapping** — burn-store's `PytorchStore` defaults to `map_indices_contiguous(true)`, which compacts the gaps in upstream's VAE lists and shifts every sample-rate-conditioning tensor onto the wrong decoder stage. The loader now calls `.map_indices_contiguous(false)`.
4. **RoPE head dim** (`src/tts/voxcpm_burn/minicpm4.rs`) — `MiniCPMConfig::init` built the RoPE cache from `hidden_size / num_attention_heads` while the attention projections use `kv_channels` as the head dim. For the VoxCPM2 encoder/DiT stacks (hidden 1024 / 16 heads = 64 vs `kv_channels` = 128) the `inv_freq` length (32) then mismatched the `short_factor`/`long_factor` length (64) and the cache build panicked with a broadcast error. The rope config now receives `kv_channels`.

**TTS default shortcut** — `TtsRuntimeRegistry::default_runtime` used `Handle::block_on` to load the first registered model, which panics with "Cannot start a runtime from within a runtime" inside `ServerState::new` (i.e. in every host that embeds neuron in a tokio runtime). The registry now exposes `lazy_default_runtime()`, handing out a proxy that resolves the model on the first async call — the same lazy-load intent, no startup panic.

**GPU backend (burn CUDA / CubeCL)** — the VoxCPM2 runtime is backend-generic: `LoadedVoxCpm2` is an enum with one arm per burn backend, and the inference bodies live in generic helpers (`synthesize_generic`, `stream_generic`). Build with `--features voxcpm-cuda` (adds `burn/cuda`) and set `NEURON_DEVICE=nvidia` (alias `cuda`) to run the 2B model on the GPU. **The default stays on the CPU** (`NEURON_DEVICE=auto` → CPU): measured on an RTX 3090 against a 12-core desktop CPU, the first audio byte of the same utterance arrived after **708 s on the GPU versus 112 s on the CPU**. VoxCPM2's autoregressive decode advances one latent patch (4 frames) per step, so the work is latency-bound rather than throughput-bound and the GPU's per-op launch overhead dominates. The GPU path also hit `CUDA_ERROR_ILLEGAL_ADDRESS` in a CubeCL kernel on the full 2B model (the tiny model runs clean), so it is opt-in, not a default.

The GPU path runs the model in **f16** (`burn::backend::cuda::Cuda<f16, i32>`): the f32 weights need ~9.2 GB of VRAM, which does not fit next to a desktop compositor's allocation on a 24 GB card, while f16 needs ~4.6 GB and roughly doubles the matmul throughput. The upstream checkpoint ships bf16, so f16 is not a precision regression for the weights.

Notes for the CUDA path:

- **No build-time CUDA toolkit** — `cubecl-cuda` uses cudarc with `driver` + `nvrtc` + `fallback-dynamic-loading`, so the driver is enough to build. At runtime the CUDA toolkit's `bin` directory must be on `PATH` so NVRTC can be loaded (same rule as the candle CUDA path).
- **The device's float dtype is set once, before anything touches the device** — burn's `set_default_dtypes` only accepts the first call per device (see `configure_cuda_device`).
- **f16 exposed dtype assumptions the f32 path never had.** Five fixes were needed, all of them latent bugs that the f32 CPU path hid:
  1. A reduction may accumulate in a wider dtype than the weights (the CubeCL backend sums f16 in f32), so the weight normalization in `WNCausalConv1d` / `WNCausalTransposeConv1d` casts the norm back to the weight dtype — otherwise the product is rejected as a dtype mismatch.
  2. Integer scalars mixed with float tensors (`self.scale as u32` in the FSQ layer, the RoPE position scale) are int/float ops; they must be float scalars.
  3. `Tensor::from_floats` always produces f32 data — the RoPE cache (`inv_freq`, the LongRoPE factors) has to be cast to the device's float dtype or the rotary multiply mismatches.
  4. `apply_rotary_pos_emb` forced the queries to f32 while the cache stayed f16; the rotary math now runs in the tensors' own dtype (identical behaviour on the f32 path).
  5. The Int dtype is backend-dependent (i64 on ndarray, i32 on CUDA) and burn's data reader does **not** convert, so the stop head's `argmax` result is cast to i64 before it is read back.
- **One CUDA session per process** — the CubeCL runtime does not survive a second session on the same device in one process, so the GPU test lives in a single test (`voxcpm2_model::cuda_f16_tests`).
- The `cuda` feature (candle's) and `voxcpm-cuda` (burn's) are independent: combine them as `--features cuda,voxcpm-cuda` to put both the LLM and the TTS on the GPU.

**Test strategy (no 4.58 GB weights in CI)**: `weights.rs` builds a PyTorch-style fixture from a random-init burn module (invert each adaptation, serialize via the `safetensors` dev-dep) and asserts `load_checkpoint_into` restores every tensor bit-exact; `voxcpm2_model.rs` runs **end-to-end tiny generate** (structural config, ~30 s of ndarray CPU) through zero-shot and reference-prompt paths down to the V2 VAE decode.

### Llama.cpp (loadable, 2026-10-15)

`NEURON_LLM_RUNTIME=llamacpp` registers a [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) backend via the safe Rust binding [`llama-cpp-2`](https://github.com/utilityai/llama-cpp-rs) (v0.1.159, MIT/Apache-2.0). One GGUF file is the whole story — 40+ model families load directly (Qwen 2/3, Llama 2/3, Mistral, DeepSeek, GLM, Phi 4, Gemma 2/3, Yi, Mixtral, DBRX, Command-R, ...) with no safetensors→GGUF conversion. Auto-applies the GGUF's embedded chat template when present (so Qwen2.5-Instruct gets `<|im_start|>` framing, Llama-3-Instruct gets `<|start_header_id|>`, Mistral-Instruct gets `[INST]`, etc.); falls back to ChatML for the pre-2024 GGUFs that omitted `tokenizer.chat_template` metadata.

**Build prereq** (gated behind the `llamacpp` Cargo feature — default builds are unaffected):

| Tool | Windows | Linux | macOS |
|---|---|---|---|
| `cmake` | Visual Studio 2022 Build Tools + cmake on PATH | `apt install cmake` / `dnf install cmake` | `brew install cmake` |
| C++ compiler | MSVC `cl.exe` (from VS Build Tools) | `gcc` / `clang` | Xcode CLT |
| `libclang` | `winget install LLVM` (for bindgen) | `apt install libclang-dev` | Xcode CLT (ships `libclang.dylib`) |

`cargo build --features llamacpp` on a host missing any of these fails inside `llama-cpp-sys-2`'s build script with a clear "cmake not found" / "no C++ compiler" / "couldn't find libclang" error pointing at the missing tool. **Default `cargo build` is unaffected** — the dep is `optional = true` and the entire `src/llm/llamacpp_runtime.rs` module is `#[cfg(feature = "llamacpp")]`-gated.

**Activation**:

```bash
# 1. Build (one-time, needs the toolchain above):
cargo build --release --features llamacpp

# 2. Pre-stage a GGUF (the runtime does NOT download weights):
mkdir -p ~/.neuron/models/llamacpp/qwen2.5-7b-instruct
# download Qwen2.5-7B-Instruct-Q4_K_M.gguf into that directory as model.gguf

# 3. Run with the env-var:
NEURON_LLM_RUNTIME=llamacpp \
  NEURON_LLM_DEFAULT_MODEL=qwen2.5-7b-instruct \
  NEURON_MODEL_DIRS=~/.neuron/models \
  ./target/release/your-host-binary
```

**Model id resolution** (in `LlamaCppLlmRuntime::resolve_gguf_path`):

1. Absolute path that exists on disk → use it.
2. Relative path that exists on disk → use it (canonicalized).
3. Bare id → search `$NEURON_MODEL_DIRS/llamacpp/<id>/model.gguf`, then `$NEURON_MODEL_DIRS/candle/<id>/model.gguf` (legacy conv for the candle-era path layout). The first hit wins.

**Sampling**: greedy by default; non-greedy requests (`temperature > 0`) get a chain of `penalties → top-K → top-P → temperature → dist` mirroring `examples/simple.cpp` in upstream llama.cpp. The same request with the same `stop` list always lands on the same seed (deterministic per request shape).

**GPU offload**:

| `DeviceSelection` | `n_gpu_layers` |
|---|---|
| `Auto` | 0 (CPU; matches candle's `Auto` default) |
| `Cpu` | 0 |
| `Cuda` / `Apple` / `Vulkan` | `u32::MAX` (full offload) |
| any | overridden by `NEURON_LLAMA_CPP_GPU_LAYERS=<n>` for partial offload |

**Context size**: `NEURON_LLAMA_CPP_N_CTX` (default `2048`). Each `chat()` / `chat_stream()` call creates a fresh `LlamaContext` (the llama-cpp-2 design is one context per stream of inference; the model + backend are shared via `Arc`); KV-cache memory per in-flight request = `n_ctx × n_layers × 2 × head_dim × sizeof(half)`. For a 7B f16 model at 2048 ctx that's ~1 GiB — drop the count on small-VRAM hosts.

**Wired endpoints**: `/v1/chat/completions` (chat + stream), `/v1/messages` via `adapter_dispatch` (Anthropic compat), `/v1/models` (one `ModelInfo` per registered id, `owned_by = "llamacpp"`). `/v1/embeddings`, `/v1/rerank`, vision endpoints are not served by this backend (use the candle / sherpa / vision runtimes for those).

**Catalog registration**: each id in `SUPPORTED_CATALOG` can ship a `llamacpp` source via the `CatalogEntry.sources` vector; the catalog itself doesn't add LlamaCpp entries by default — `NEURON_LLM_RUNTIME=llamacpp + NEURON_LLM_DEFAULT_MODEL=<id>` is the documented registration path (mirrors how `mistralrs` is wired).

**Not yet wired**: tool use (capabilities() reports `supports_tools = false`; ChatML fallback coerces unknown roles to user), prompt caching (capabilities() reports `supports_prompt_caching = false`; every chat call pays prompt prefill), extended thinking, audio input. Adding these is the obvious follow-up — the llama.cpp side has the `mtmd` audio path and the `llguidance` grammar (tool / JSON) feature available behind optional `llama-cpp-2` features. **Multimodal image input is wired** — see "MiniCPM-V 4.7 via llamacpp mtmd (loadable, 2026-10-15)" below.

### MiniCPM-V 4.7 via llamacpp mtmd (loadable, 2026-10-15)

The llama.cpp backend now serves multimodal image input through llama-cpp-2's `mtmd` feature — `NEURON_LLAMA_CPP_MMPROJ` (or auto-discovery) loads a vision-projector GGUF, and `POST /v1/chat/completions` with OpenAI's `image_url` content blocks flows through `MtmdBitmap::from_buffer` → `MtmdContext::tokenize` → `MtmdInputChunks::eval_chunks` → the existing generation loop. `MiniCPM-V 4.7` (OpenBMB, 2026-10-12, 35B-A3B + 1B variants) is the first upstream model that works end-to-end through this path (llama.cpp's `PROJECTOR_TYPE_MINICPMV4_7` was merged before the model release).

**Build**: the `llamacpp` feature now always pulls `llama-cpp-2` with the `mtmd` feature flag enabled, so the same toolchain prereq as the base llama.cpp backend (cmake + C++ compiler + libclang on PATH) gets you multimodal too — no extra flag to remember.

**Activation**:

```bash
# 1. Build (one-time):
cargo build --release --features llamacpp

# 2. Stage the GGUF + mmproj side by side:
mkdir -p ~/.neuron/models/llamacpp/minicpm-v-4.7-1b
#   text GGUF     -> ~/.neuron/models/llamacpp/minicpm-v-4.7-1b/model.gguf
#   mmproj GGUF   -> ~/.neuron/models/llamacpp/minicpm-v-4.7-1b/mmproj-MiniCPM-V-4.7-f16.gguf

# 3. Run (server detects the mmproj at load; capabilities() reports ImageInput):
NEURON_LLM_RUNTIME=llamacpp \
  NEURON_LLM_DEFAULT_MODEL=minicpm-v-4.7-1b \
  NEURON_MODEL_DIRS=~/.neuron/models \
  ./target/release/your-host-binary

# 4. Send a multimodal request (OpenAI shape):
#   POST /v1/chat/completions
#   {
#     "model": "minicpm-v-4.7-1b",
#     "messages": [{
#       "role": "user",
#       "content": [
#         {"type": "text", "text": "What's in this image?"},
#         {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}}
#       ]
#     }]
#   }
```

**mmproj auto-discovery** (in `LlamaCppLlmRuntime::load`): in priority order,

1. `$NEURON_LLAMA_CPP_MMPROJ` (absolute or relative; relative resolved against CWD, not the text-GGUF dir).
2. `<text-gguf-dir>/mmproj-*.gguf` (HF llama.cpp convention — most converted VLM GGUFs ship under this prefix).
3. `<text-gguf-dir>/model.mmproj.gguf` (alternative).
4. `<text-gguf-dir>/mmproj.gguf` (bare convention).

If none of these resolve, the runtime loads as a text-only LLM (capabilities() reports `ImageInput = false`) and `chat_with_media` returns a clear "no mmproj found" error. A second `model.gguf` (text + a quantized variant) is not mistaken for an mmproj — the `mmproj-` prefix is the marker.

**Trait shape** (load-bearing extension; see `docs/superpowers/specs/2026-10-15-minicpm-v-47-via-llamacpp-mtmd.md`):

```rust
// src/llm/mod.rs — default impl returns not_implemented; only
// LlamaCppLlmRuntime overrides.
async fn chat_with_media(
    &self,
    req: ChatRequest,
    media: Vec<MediaAttachment>,
) -> ChatResult<ChatResponse>;

// src/engine/mod.rs — same default-fallback pattern.
async fn execute_with_media(
    &self,
    req: CanonicalLlmRequest,
    media: Vec<MediaAttachment>,
) -> ChatResult<CanonicalLlmResponse>;
```

**Wire format on the request side** (the adapter already has this for Anthropic compat; the multimodal path reuses the same `ContentBlock::Image(ImageBlock)` shape):

```jsonc
{
  "role": "user",
  "content": [
    {"type": "text",      "text": "..."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,...", "detail": "high"}}
  ]
}
```

`dispatch.rs::extract_media` walks every user message, base64-decodes `ImageSource::Base64` to bytes, and routes through `execute_with_media` / `execute_with_media_stream`. v1 supports `ImageSource::Base64` only; `ImageSource::Url` and `ImageSource::FileId` are rejected at the translator with a clear "set `NEURON_MEDIA_FETCH_HOST_ALLOWLIST` for http(s) fetch" or "pre-resolve to Base64" error rather than silently dropping the image.

**Not yet wired (v1 multimodal scope)**:

- `ImageSource::Url` (http(s) fetch) — needs an http client + Mimosa-style host allowlist. The env-var knob is named (`NEURON_MEDIA_FETCH_HOST_ALLOWLIST`) but the fetch path is not implemented in this commit; pre-decode to a `data:` URI in the caller.
- `ImageSource::FileId` — host-private file abstraction; pre-resolve in the adapter.
- Audio input (`MtmdBitmap::from_audio_data` is plumbed through mtmd; capabilities() reports `AudioInput = false` so the adapter short-circuits).
- KV-cache prefix reuse across `eval_chunks` — every multimodal call pays full prefill; no `supports_prompt_caching` flag flip yet.
- Tool calls — `llama-cpp-2`'s `llguidance` feature is the path; not enabled.

**Multi-image / multi-turn**: OpenAI allows multiple `image_url` blocks per user message; `extract_media` preserves order and the mtmd marker substitution places one `<__media__>` per image in the text stream. KV cache is fresh per request (no prefix reuse), so multi-turn conversations still pay the full image+text prefill on every chat call.

## Architectural rules (DO NOT BREAK)

1. **Wire format is OpenAI byte-for-byte compatible** for `/v1/chat/completions`, `/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/rerank`, `/v1/embeddings`, AND Anthropic Messages v1 compatible for `/v1/messages`. Any client that works with OpenAI / vLLM / ollama / llama.cpp server / Anthropic SDK should work with neuron unchanged.

2. **All model types share the same trait shape**: `load`, `infer`, `list_models`, `name`. See the "Adding a new model type" checklist in existing specs and the trait shapes under src/.

3. **`run_embedded()` returns a `NeuronHandle` that owns the server lifetime**. Hosts MUST call `request_shutdown()` + `join().await` for graceful drain. Dropping the handle without shutdown leaks the bind port.

4. **Feature flags are the only way to gate ML framework pulls**:
   - `candle` (default) — pulls `candle-core` (imported as crate `candle` since upstream 0.11), `candle-nn`, `candle-transformers`, `tokenizers` (all from git rev `aebc405`, version 0.11.0 — see the "candle upgrade" note below)
   - `voxcpm` — pulls `burn`, `burn-store`
   - `litert` — pulls `litert-lm`
   - `mistralrs` — pulls `mistralrs` (mistral.rs v0.8.22, MIT); implicitly activates `candle`. mistral.rs pins `regex-automata ^0.4.13` while candle 0.11's `fancy-regex` needs `^0.4.16` — a fresh `Cargo.lock` (delete and regenerate) resolves both at 0.4.18; a lockfile pinned to 0.4.14 fails resolution.

5. **No echo stubs — real engines required (since 2026-06-30)**. Prior versions shipped `EchoLlmRuntime` / `EchoTtsRuntime` / `EchoAsrRuntime` / `EchoRerankRuntime` / `EchoEmbeddingRuntime` as always-available stubs for integration testing and wire-format validation. Those have been removed. Each `*RuntimeRegistry` now exposes `default_runtime()` which returns an `Unavailable*Runtime` error stub — every inference call against an unregistered or unloadable model returns `NeuronError::not_initialized(...)` so the HTTP layer can surface a clear 4xx response. To exercise the full HTTP stack end-to-end, integration tests must select a real engine via env vars (`NEURON_*_RUNTIME=candle|burn|mxbai|jina|cohere|litert|mistralrs`) and provide model weights.

6. **Models live under `$NEURON_MODEL_DIRS` (default `~/.neuron/models/`)** in `{engine}/{id}/` subdirectories. `NEURON_MODEL_DIRS` is a colon-separated list of roots; the first root containing a given model wins on collision. The legacy single-root `NEURON_MODEL_DIR` is still honored for backward compatibility. Each engine knows how to find its own files inside the subtree.

7. **The HTTP endpoints are stable**. Adding a new endpoint means adding a route + handler + (for OpenAI/Anthropic-compat ones) preserving the byte-for-byte shape. `/v1/chat/completions` and `/v1/messages` are routed through the `ApiAdapter` (`adapter_dispatch`) system; all other endpoints still go through their dedicated handlers. Future work will move the remaining OpenAI endpoints through `adapter_dispatch` too.

   Current routes (in `src/server/mod.rs`):
   - `GET /health`, `GET /v1/info` — liveness
   - `GET /v1/models?engine=&source=` — model catalog (with filters)
   - `GET /v1/catalog/sources` — list registered download sources (id, name, endpoint)
   - `POST /v1/models/pull` — start a download; `GET` lists active/completed jobs
   - `GET /v1/models/pull/{job_id}`, `DELETE /v1/models/pull/{job_id}` — query / cancel
   - `GET /v1/models/{engine}/{id}`, `DELETE /v1/models/{engine}/{id}` — local model file introspection / deletion
   - `POST /v1/chat/completions`, `POST /v1/audio/speech`, `POST /v1/audio/transcriptions`, `POST /v1/rerank`, `POST /v1/embeddings` — OpenAI-compat inference
   - `POST /v1/audio/speech/stream` — model-level streaming TTS (VoxCPM2): audio bytes arrive as the model produces them, not after the whole utterance
   - `POST /v1/vision/detect`, `/classify`, `/segment`, `/pose` — Vision tasks
   - `POST /v1/vision/lora/register`, `GET /v1/vision/lora/list` — LoRA management
   - `POST /v1/ocr/recognize` — PP-OCR text detection + recognition (lines + boxes + reading order)
   - `POST /v1/decisions` — Laya discriminative decision scoring (options + act/escalate)
   - `POST /v1/forecast` — Chronos2 zero-shot time-series forecast
   - `POST /v1/vision/gate/score` — Mage-VL StreamMind gate score
   - `POST /v1/images/generations` — MOSS-Image text-to-image (OpenAI Images shape, `b64_json`; neuron extensions: `negative_prompt`, `steps`, `guidance_scale`, `seed`)
   - `GET /v1/scheduler/status` — model scheduler snapshot (residents, queue, budget, correction stats)
   - `DELETE /v1/scheduler/models/{id}` — administratively evict a resident model (409 while in use)
   - `POST /v1/messages` — Anthropic Messages v1 (via `adapter_dispatch`)

8. **Each capability uses a `*RuntimeRegistry`** (e.g. `LlmRuntimeRegistry`) that holds registered model entries, a loaded-instance cache, and a `ResourceTracker`. Handlers route via `runtime_for(model_id)` which lazy-loads on first request. `ServerState` keeps both the registry (primary) and shortcut fields (backward compat).

9. **TTS trait has a default `synthesize_stream` impl** that yields 16 KiB chunks of the synthesized audio. Runtimes only need to override it if they can stream from the model itself. The `/v1/audio/speech` handler respects the request's `stream: bool` field; `false` (default) returns the full audio as a single response body, `true` returns axum `Body::from_stream` with chunked transfer encoding. **Model-level streaming is a separate trait method** — `TtsRuntime::synthesize_stream_incremental(text, TtsStreamOptions)` — because it needs a container choice (`TtsStreamFormat::Wav` | `Pcm`) and an optional voice-clone reference; the default impl falls back to the full-synthesis path so runtimes without model-level streaming keep working. It is served by `POST /v1/audio/speech/stream`.

10. **ASR audio decoding is feature-gated** behind `asr-audio` (opt-in). Without it, only the manual WAV decoder is compiled in. The `detect_format` function in `src/asr/audio.rs` sniffs magic bytes; `decode_to_pcm16k_mono` is the single entry point for all formats.

11. **Rerank enum variant `RerankImpl::Candle` is a deprecated alias** for `RerankImpl::Bge`. Parsing the string `"candle"` is still accepted (with a `tracing::warn!`) for one release cycle, then should be removed.

### candle upgraded to upstream main (2026-09-27)

neuron now pins `huggingface/candle.git#aebc405` (0.11.0, upstream main of
2026-09-25), up from `a9667ca` (0.10.2, Jan 2026). This brings new model
families, the cuTILE + fused-MoE kernel integration, paged flash-attn
kernels, and the CUDA launch-parameter cache. Performance on neuron's
single-stream decode is unchanged within noise (A/B below) — the upgrade
is for features, not speed. Four adaptation points, each load-bearing:

1. **Upstream `layer_norm` regression worked around in
   `src/candle_compat.rs`.** `candle-nn@b96522e` (Sep 2) probes
   `bias`/`beta` via `contains_tensor` *before* checking
   `config.affine`, so every bias-free norm (RmsNorm — llama/qwen/gemma/
   chatglm checkpoints) fails with "Failed to find weight tensor".
   Upstream CI misses this because unit tests use `VarMap`/zeros. The
   `NormBiasShim<B: SimpleBackend>` here reports a synthetic bias for
   `…norm.bias` / `…norm.beta` paths only; affine norms with a real
   checkpoint bias are unaffected, and `affine = false` never loads the
   probe result. ALL `VarBuilder::from_mmaped_safetensors` and
   `from_tensors` call sites go through `candle_compat::{from_mmaped_safetensors,
   from_tensors}` — remove the shim once upstream gates the probe on
   `affine`.
2. **Crate rename.** From 0.11, `candle-core` is re-exported as the crate
   name `candle` (`package = "candle-core"` in Cargo.toml), so sources
   use `use candle::…`; `candle-nn` keeps its name. Feature flags become
   `candle/cuda`, `candle/metal`.
3. **Strict `Tensor::repeat` semantics.** candle ≥ 0.11 follows PyTorch:
   a `0` repeat narrows the dim to zero (old candle treated it as a
   no-op). `src/mage_vl/gate.rs` used `repeat((0, rep, 1, 1))` for GQA
   head expansion — fixed to `repeat((1, rep, 1, 1))`, correct on both
   versions.
4. **CUDA-on-Windows build & runtime.** `candle-kernels/build.rs` (in the
   cargo git checkout, not this repo) needs `-Xcompiler /Zc:preprocessor`
   for the CUDA 13 CCCL preprocessor hard-error under MSVC; cargo does
   NOT re-run a build script when you edit files inside a git checkout —
   `rm -rf target/release/build/candle-kernels-*` after patching. At
   runtime, `Device::new_cuda` fails (and before 2026-09-27 *silently*
   fell back to CPU) unless cudarc's DLLs resolve — launching the host
   with the CUDA toolkit `bin` directory on `PATH` is the reliable fix.
   The `Auto` device fallback now logs the CUDA init failure instead of
   swallowing it.

**A/B benchmark** (same machine, greedy 12-token decode, llama.cpp CUDA as
the machine-mood control row, p50 of 3):

| row | a9667ca (0.10.2) | aebc405 (0.11.0) |
|---|---|---|
| neuron-candle-cuda (bf16) | 30.6 / 38.0 tok/s | 26.5 / 26.0 tok/s |
| neuron-candle-cpu (f16) | 2.15 / 2.12 tok/s | 2.58 / 1.63 tok/s |
| llama-cuda q8_0 (control) | 142.9 / 141.5 | 131-135 (separate run) |

Within run-to-run noise on both devices — the perf commits in the range
target batch serving and quantized paths, not single-stream f16/bf16
GEMM-bound decode. `neuron-mistralrs-cpu` (Q4_K ISQ) re-tested under
the upgrade hits **~6.7 tok/s** steady-state, matching the pre-upgrade
~7 tok/s — all three neuron backends (candle CPU/GPU + mistralrs CPU)
work end-to-end on 0.11.

## Model scheduler invariants (2026-09-29)

`src/scheduler/` + the `adapter_dispatch` integration. Full design in
`docs/superpowers/specs/2026-09-29-model-scheduler.md`; load-bearing
points if you touch it:

1. **Grant channels flush after the state lock drops.** A failed
   oneshot send drops a `ModelLease`, whose `Drop` calls `release`
   which re-locks — sending under the lock self-deadlocks. Same
   reason `run_evictions`/`do_unload` execute outside the lock with
   two-phase `evicting` marking (`mark_for_eviction` → unload →
   remove under lock → re-scan).
2. **`LlmRuntimeRegistry::unload` is synchronous by design.** The
   loaded cache uses a std mutex that is never held across `await`
   (the load runs outside it and re-checks on insert). Do not revert
   to `tokio::sync::Mutex` — the scheduler evicts from lease-drop
   contexts where blocking on a tokio mutex is not allowed.
3. **The lease must outlive execution, not the handler.** Streaming
   responses wrap the wire stream in `HoldingStream` so the model
   stays resident until the SSE body drops (completion *or* client
   disconnect). Dropping the lease early lets a swap unload weights
   under a live stream.
4. **Engine-resolution failures must call `mark_load_failed`.** The
   scheduler reserves the model's estimate at admission, before the
   lazy weights load; without the hook a failed load would pin a
   phantom reservation until the idle reaper.
5. **RSS observation is `cpu`-only** (`SchedulerConfig::observes_rss`):
   GPU-resident weights are invisible to process RSS, so `cuda` /
   `metal` platforms keep the shipped reference estimates.
6. `NEURON_SCHEDULER_ENABLED=false` must keep the legacy path
   byte-identical (no queue, no unload, silent default-model
   fallback) — hosts rely on it as an escape hatch.

## Conformance tests

```bash
cargo test --lib                                  # 748 unit tests (src/**/mod.rs) — 0 failures (2026-10-30, after the MOSS-Image parity fixtures landed; +5 parity/weight-name tests on the 2026-10-27 baseline of 743)
cargo test --lib --features voxcpm                # 734 unit tests — 0 failures (VoxCPM2 adds the weights/model/runtime/clone-audio/streaming suites; two pre-existing V1 audiovae roundtrip tests are slow, ~5 min, on ndarray)
cargo test --lib --features voxcpm-cuda           # 735 unit tests — 0 failures (adds the f16 GPU stack test; needs an NVIDIA GPU + the CUDA toolkit's bin on PATH)
cargo test --lib --features tract                 # 761 unit tests — 0 failures (tract adds the decision + vision-ONNX + sensevoice paths + the full PP-OCR pipeline suites: DB++ postprocess, CTC decode, preprocess, paddle runtime)
cargo test --lib --features ort                   # 741 unit tests — 0 failures (the PP-OCR pipeline suites on the ort executor + the layout suites; needs ORT_DYLIB_PATH only for the runtime, not the tests)
cargo test --lib --features tract,sherpa          # 695 unit tests — 0 failures (sherpa adds SherpaAsrRuntime + from_dir + the wav/onnx_meta/fbank/sensevoice coverage)
cargo build --bench asr_speed --features tract,sherpa  # in-process tract-vs-sherpa benchmark compiles
cargo bench  --bench asr_speed --features tract,sherpa  # writes target/asr-speed-compare.md
cargo build --bench llamacpp_speed --features llamacpp  # in-process llama.cpp backend benchmark compiles (needs cmake + C++ toolchain + libclang on PATH)
cargo bench  --bench llamacpp_speed --features llamacpp  # writes target/llamacpp-speed.md (set NEURON_LLAMACPP_MODEL_PATH=/path/to/model.gguf)
cargo build --bench model_matrix --features "candle,tract,sherpa,voxcpm"  # per-cap × per-engine matrix harness compiles
cargo bench  --bench model_matrix --features "candle,tract,sherpa,voxcpm"  # writes target/model-matrix.md (requires neuron_server on :18083)
cargo run --release --example scheduler_smoke --features candle  # optional real-weights scheduler verification: coexistence + drain-then-swap + admin evict + RSS observation (needs 2 downloaded LLM models)
cargo test --lib verify_weight_paths_against_upstream -- --ignored  # MOSS-Image: diff the port's requested tensor paths against the upstream checkpoint index (needs network; ~10 s)
cargo test --lib verify_weight_shapes_against_upstream -- --ignored  # MOSS-Image: same, for the tensor shapes (needs network; ~40 s)
cargo test --lib moss_image::parity -- --nocapture  # MOSS-Image: element-wise parity against the upstream PyTorch implementation (GDN kernel + DiT) — prints the measured relative error
cargo run --release --example voxcpm2_stream_smoke --features voxcpm  # real-weights streaming-vs-full decode comparison (needs the openbmb/VoxCPM2 download; --with-lm adds the 2B LM stage)
cargo run --release --example voxcpm2_stream_http_smoke --features voxcpm  # end-to-end HTTP check of POST /v1/audio/speech/stream: first-byte latency vs the legacy endpoint
cargo build --lib                                 # library-only build (no standalone binary)
cargo bench --no-run --bench perf                 # 14 Criterion benchmarks compile
cargo bench --bench perf -- --quick               # fast CI smoke
cargo bench --no-run --bench llm_speed            # cross-framework HTTP harness compiles (see below)
```

The **cross-framework LLM speed harness** (`benches/llm_speed.rs`, `harness = false`) needs live servers, so it is not part of CI — only its compilation is. To actually run it: `./scripts/run-llm-speed-compare.sh` starts whatever servers are installed (neuron via `examples/neuron_server`, llama.cpp, ollama, vLLM) and skips the rest; the report lands in `target/llm-speed-compare.md`. `benches/compare.rs` by contrast is a microbenchmark scaffold whose non-neuron backends are noop stubs — it does not measure real cross-framework latency.

Run all three before submitting a PR. **All must pass**.

> **Note**: The `tests/integration.rs` end-to-end suite referenced in earlier versions of this doc was retired — the integration scenarios it used to cover (HTTP round-trip with a real model) now live under `examples/real_model_smoke.rs` and require model weights on disk plus `NEURON_*_RUNTIME` env vars. They are not part of CI.

## ServerState fields

| Field | Type | Purpose |
|---|---|---|
| `llm_registry` | `Arc<LlmRuntimeRegistry>` | Primary LLM model registry |
| `tts_registry` | `Arc<TtsRuntimeRegistry>` | Primary TTS model registry |
| `asr_registry` | `Arc<AsrRuntimeRegistry>` | Primary ASR model registry |
| `rerank_registry` | `Arc<RerankRuntimeRegistry>` | Primary Rerank model registry |
| `embedding_registry` | `Arc<EmbeddingRuntimeRegistry>` | Primary Embedding model registry |
| `vision_registry` | `Arc<VisionRuntimeRegistry>` | Primary Vision model registry |
| `ocr_registry` | `Arc<OcrRuntimeRegistry>` | Primary OCR (PP-OCR) model registry |
| `decision_registry` | `Arc<DecisionRuntimeRegistry>` | Primary Decision (Laya) model registry |
| `imagegen_registry` | `Arc<ImageGenRuntimeRegistry>` | Primary image-generation (MOSS-Image) model registry |
| `chronos2_registry` | `Arc<Chronos2Registry>` | Chronos2 forecasting registry (backing `/v1/forecast`) |
| `vision_lora_registry` | `Arc<vision::LoraRegistry>` | Dynamically registered Vision LoRA adapters |
| `llm_scheduler` | `Arc<scheduler::ModelScheduler>` | Bounded-memory LLM scheduler (leases, queue, evictions) |
| `mage_vl_gate` | `Arc<mage_vl::http::GateRuntime>` | Mage-VL StreamMind event-gate runtime (`/v1/vision/gate/score`) |
| `engines` | `Arc<CapabilityRegistries>` | New (post-plugin-architecture): per-capability engine registry with `Arc<dyn AnyCapabilityEngine>` entries. Populated via `register_engine(...)` on `NeuronHandle`. Used by `adapter_dispatch` for `/v1/chat/completions` + `/v1/messages` and any future adapter-routed endpoint. |
| `adapters` | `Arc<AdapterRegistry>` | New (post-plugin-architecture): per-path `Arc<dyn ApiAdapter>` entries. Built-in adapters: `OpenAiChatAdapter` (`/v1/chat/completions`), `AnthropicMessagesAdapter` (`/v1/messages`). Hosts can register additional adapters via `register_adapter(...)`. |
| `llm` | `Arc<dyn LlmRuntime>` | Shortcut: first registered LLM (backward compat) |
| `tts` | `Arc<dyn TtsRuntime>` | Shortcut: first registered TTS (backward compat) |
| `asr` | `Arc<dyn AsrRuntime>` | Shortcut: first registered ASR (backward compat) |
| `rerank` | `Arc<dyn RerankRuntime>` | Shortcut: first registered Rerank (backward compat) |
| `embedding` | `Arc<dyn EmbeddingRuntime>` | Shortcut: first registered Embedding (backward compat) |
| `vision` | `Arc<dyn VisionRuntime>` | Shortcut: first registered Vision (backward compat) |
| `ocr` | `Arc<dyn OcrRuntime>` | Shortcut: first registered OCR (backward compat) |
| `decision` | `Arc<dyn DecisionRuntime>` | Shortcut: first registered Decision (backward compat) |
| `imagegen` | `Arc<dyn ImageGenRuntime>` | Shortcut: first registered image-generation model (backward compat) |
| `config` | `NeuronConfig` | Effective config |
| `puller` | `Arc<Puller>` | Download job manager |
| `catalog` | `Arc<Catalog>` | Model registry (supported + available models) |

Handlers use `registry.runtime_for(&model).await.unwrap_or_else(|_| state.shortcut.clone())`
to resolve a model string from an HTTP request. This means:
- If the model is registered → its runtime is returned (lazy-loaded on first access)
- If the model is NOT registered → falls back to the shortcut (first registered model)

## Integration

neuron is a library, so its "integration" is via the OpenAI-compatible HTTP surface it exposes when embedded. Concrete host patterns:

- **Tauri 2 desktop shell** (or any Rust desktop app): declare `neuron = { path = "../neuron" }` in `src-tauri/Cargo.toml`. Call `neuron::run_embedded(NeuronEmbedConfig { port: 8085, ... })` in the `setup()` hook. Expose a thin Tauri command (e.g. `neuron_health`) so the front-end can introspect the model state.
- **Server-side process** (any HTTP gateway or orchestrator agent): spawn the binary, point an HTTP client at `/v1/chat/completions` / `/v1/audio/speech` / etc. on the embedded port. No code coupling — just the OpenAI wire format.
- **Mobile** (Android / iOS): Rust hosts can `cargo-ndk` build neuron into a shared library and call `run_embedded()` from a Kotlin/Swift shim. Mobile platforms that prefer their own native inference (MNN / LiteRT) can run a thin Kotlin `OpenAIServer`-shaped wrapper instead — neuron is not required on-device.
- **Embedded hosts overriding the default port**: 8083 is the default; many hosts override to 8085 (or any free port) via `NeuronEmbedConfig.port`.

neuron has **no direct dependency** on any particular host — the integration contract is the OpenAI HTTP wire format, which is what makes the library drop-in portable.

## Configuration knobs (env vars)

| Env var | Default | Purpose |
|---|---|---|
| `NEURON_PORT` | 8083 | HTTP bind port (embedded hosts commonly override to 8085) |
| `NEURON_HOST` | `0.0.0.0` | HTTP bind host (read by `from_env_map()` since the 2026-06-26 follow-up; `embedded.rs::run_embedded` uses `config.host` for `TcpListener::bind`) |
| `NEURON_MODEL_DIRS` | `~/.neuron/models/` | Colon-separated list of roots scanned for downloaded model weights (use `NEURON_MODEL_DIR` for single legacy value) |
| `NEURON_DEFAULT_SOURCE` | `modelscope` | Default source id used by `POST /v1/models/pull` and `parse_reference` when the reference has no `<source>:` prefix |
| `NEURON_LLM_SOURCE` / `NEURON_TTS_SOURCE` / `NEURON_ASR_SOURCE` / `NEURON_RERANK_SOURCE` | (unset) | Per-engine source id override |
| `NEURON_MODEL_SOURCE_<ENGINE>_<ID>` | (unset) | Per-model source id override (underscores map to hyphens in the id) |
| `NEURON_SOURCE_<ID>_ENDPOINT` | (unset) | Base URL for the named source (e.g. `NEURON_SOURCE_HUGGINGFACE_ENDPOINT=https://hf-mirror.com`) |
| `NEURON_HUGGINGFACE_CLI` | (auto, on PATH) | Path to `huggingface-cli` binary for `HuggingFaceSource` (also used by `hf-mirror`) |
| `NEURON_MODELSCOPE_CLI` | (auto, on PATH) | Path to `modelscope-cli` binary for `ModelScopeSource` |
| `HF_TOKEN` | (unset) | Bearer token sent on every HuggingFace / hf-mirror HTTP download. Empty / unset is fine — `reqwest` simply omits the header. Used to un-gate private repos. |
| `MODELSCOPE_TOKEN` | (unset) | Bearer token sent on every ModelScope HTTP download. Same semantics as `HF_TOKEN`. |
| `NEURON_LLM_RUNTIME` | (unset) | `candle` \| `litert` \| `mistralrs` \| `llamacpp` — which LLM engine to load. `llamacpp` requires the `llamacpp` Cargo feature (`cmake` + C++ toolchain + `libclang` on PATH) and a local GGUF file at the resolved path; see the "Llama.cpp (loadable)" section. |
| `NEURON_LLM_DTYPE` | `f32` | `f32` \| `f16` \| `bf16` — storage dtype the candle runtime materialises weights (and the Llama KV cache) in. `f16`/`bf16` halve resident weight memory. candle's CPU backend has **no BF16 matmul** (loads but fails forward), so `f16` is the memory-lean choice on CPU; F32 for a 2B model is ~10 GB resident and pages on a 16 GB host. |
| `NEURON_USE_FLASH_ATTN` | (unset) | `1` / `true` opts into FlashAttention-2 in the candle runtime. Requires the `flash-attn` Cargo feature **and** a CUDA device to be useful |
| `NEURON_REMOTE_BASE_URL` | (unset) | Base URL of an OpenAI-compatible endpoint registered as a remote LLM engine — each id in `NEURON_REMOTE_MODELS` is proxied verbatim (tools included). Lets a host serve cloud-quality models without local weights |
| `NEURON_REMOTE_MODELS` | (unset) | Comma-separated model ids registered against `NEURON_REMOTE_BASE_URL`. The first id becomes the default engine |
| `NEURON_REMOTE_API_KEY` | (unset) | Bearer token sent to the remote endpoint |
| `NEURON_MISTRALRS_ISQ` | `q4_0` | `q4_0` \| `q8_0` \| `none` — in-situ quantization applied by mistral.rs at load time |
| `NEURON_LLAMA_CPP_N_CTX` | `2048` | Llama.cpp context length (tokens) for KV-cache sizing. Each `chat()` / `chat_stream()` call allocates a fresh context sized by this. Set lower on small-VRAM hosts (a 7B f16 model needs ~1 GiB at 2048). |
| `NEURON_LLAMA_CPP_GPU_LAYERS` | (auto from `DeviceSelection`) | Override GPU layer offload count for the llama.cpp backend. `0` = full CPU; `u32::MAX` = full GPU offload; integer in between = partial offload. Ignored unless `NEURON_LLM_RUNTIME=llamacpp`. |
| `NEURON_LLAMA_CPP_FORCE_CHATML` | (unset) | When `1` / `true` / `TRUE`, skip the GGUF's embedded chat template and render every request through the ChatML fallback (`<|im_start|>` framing). Useful for testing model behaviour under a known prompt format. |
| `NEURON_LLAMA_CPP_MMPROJ` | (unset) | Override path to a vision-projector GGUF (mmproj) for multimodal requests. Default auto-discovery searches `<text-gguf-dir>/mmproj-*.gguf` then `<dir>/model.mmproj.gguf` then `<dir>/mmproj.gguf`. When unset and no co-located mmproj is found, the runtime loads as text-only. |
| `NEURON_MEDIA_FETCH_HOST_ALLOWLIST` | (unset) | Comma-separated host allowlist for `ImageSource::Url` http(s) fetches in the multimodal path. **Not yet implemented in v1** — declared for future use; v1 only accepts `data:` URIs via `ImageSource::Base64`. |
| `NEURON_EMBEDDING_RUNTIME` | (unset) | `candle` — which Embedding engine to load |
| `NEURON_EMBEDDING_MODEL_PATH` | (unset) | Path to the Embedding model directory (must contain `config.json` + `tokenizer.json` + `model.safetensors`) |
| `NEURON_LLM_MODEL_PATH` | (unset) | Path to the LLM model directory (must contain `config.json` + `tokenizer.json` + `*.safetensors`) |
| `NEURON_TTS_RUNTIME` | (unset) | `voxcpm` / `burn` (0.5B, requires `--features voxcpm`) \| `voxcpm2` (2B/48 kHz, same feature; loads upstream checkpoints directly, ~10 GB RAM) |
| `NEURON_DEVICE` | `auto` | Device for the LLM and TTS engines: `auto` \| `cpu` \| `nvidia` (alias `cuda`) \| `apple` \| `vulkan`. `auto` picks the GPU for the candle LLM path; the **burn** TTS backend stays on the CPU because VoxCPM2's per-patch decode is latency-bound and measured ~6× slower on an RTX 3090 (see the "GPU backend" note in the VoxCPM2 section). `nvidia` puts VoxCPM2 on the GPU when built with `--features voxcpm-cuda`. |
| `NEURON_TTS_MODEL_PATH` | (unset) | Path to the TTS model directory. 0.5B: pre-converted burn `.mpk` weights (`config.json` + `tokenizer.json` + `model.mpk` + `audiovae.mpk`). VoxCPM2: the upstream `openbmb/VoxCPM2` download (`config.json` + `tokenizer.json` + `model.safetensors` + `audiovae.pth`) |
| `NEURON_ASR_RUNTIME` | (unset) | `candle` (Whisper) \| `tract` (SenseVoiceSmall via neuron's own tract ONNX stack) \| `sherpa` (SenseVoiceSmall via the official sherpa-onnx Rust binding) |
| `NEURON_ASR_MODEL_PATH` | (unset) | Path to the ASR model directory (SenseVoiceSmall needs `model.onnx` + `tokens.txt`; tract path also wants the `model.prepped.onnx` produced by `scripts/prep-sensevoice-onnx.py`) |
| `NEURON_ASR_ONNX_FILE` | (unset) | Override the SenseVoice graph file (e.g. `model.int8.onnx`); ignored by the Whisper runtime |
| `NEURON_ASR_SHERPA_NUM_THREADS` | (unset, default 2) | Thread count for the sherpa-onnx recognizer |
| `NEURON_RERANK_RUNTIME` | (unset) | `candle` (BGE-reranker, deprecated alias for `bge`) \| `bge` \| `mxbai` \| `jina` \| `cohere` |
| `NEURON_RERANK_MODEL_PATH` | (unset) | Path to the Rerank model directory (BGE/mxbai/jina only — cohere uses the API) |
| `NEURON_COHERE_API_KEY` | (unset) | Cohere API key (required when `NEURON_RERANK_RUNTIME=cohere`) |
| `NEURON_COHERE_ENDPOINT` | `https://api.cohere.com` | Cohere API base URL |
| `NEURON_COHERE_MODEL` | `rerank-english-v3.0` | Cohere model id (e.g. `rerank-multilingual-v3.0`) |
| `NEURON_VISION_RUNTIME` | (unset) | `candle` \| `candle-yolo` \| `ort` — which Vision engine to register (read since 2026-10-29; before that it was documented but never read). `candle` is an alias for `candle-yolo`. The candle RT-DETR / DETR ids were **removed** on 2026-10-29 — the architectures were never implemented, and a registration that answers `not_implemented` on every request is a promise the server cannot keep; serve those families through `ort`. `ort` is the **ONNX session path** — its executor is tract (pure Rust, `--features tract`), NOT onnxruntime: the `ort` Cargo feature does not touch the vision path. |
| `NEURON_VISION_MODEL_PATH` | (unset) | Path to the Vision model directory. `ort` reads `model.onnx` + `config.json` from it; `candle-yolo` reads `*.safetensors` from it (and refuses to run without a checkpoint that covers every architecture variable). |
| `NEURON_VISION_MODEL_ID` | `vision` | The model id vision requests refer to when the runtime is registered through `NEURON_VISION_RUNTIME` (embedded hosts can instead register through `config.registrations`). |
| `NEURON_DECISION_RUNTIME` | (unset) | `tract` — which Decision engine to load (requires `--features tract`) |
| `NEURON_DECISION_MODEL_ID` | `laya` | Model id registered for the Decision capability |
| `NEURON_DECISION_MODEL_PATH` | (unset) | Path to the prepped Decision model dir — a single `*.onnx` (plus `.onnx.data` if external) + `tokenizer.json`, optional `rl_agent_config.json` for the temperature tables |
| `NEURON_SCHEDULER_ENABLED` | `true` | `false` disables the model scheduler (requests never queue, weights never unload) |
| `NEURON_SCHEDULER_BUDGET_MB` | auto (80% RAM) | Resident-weights memory budget in MiB (`0` = auto) |
| `NEURON_SCHEDULER_IDLE_SECS` | `120` | Idle residents (0 leases, no queued demand) recycled after this |
| `NEURON_SCHEDULER_MAX_CONCURRENCY` | `4` | Concurrent leases per resident model |
| `NEURON_SCHEDULER_MAX_QUEUE_WAIT_SECS` | `300` | Queue dwell time before `503 queue_timeout` (`0` = forever) |
| `NEURON_SCHEDULER_PLATFORM` | from `NEURON_DEVICE` | Platform key into the memory profile (`cpu` \| `cuda` \| `metal`); RSS observation runs on `cpu` only |
| `NEURON_SCHEDULER_PROFILE_PATH` | (unset) | Override where observed memory-profile overrides persist (default `<first model dir>/scheduler/observed_profiles.json`) |
| `NEURON_OCR_MODEL_PATH` | (unset) | Path to the PP-OCR pipeline directory (`det.prepped.onnx` + `rec.prepped.onnx` + `dict.txt`, all prepped with `scripts/prep-ppocr-onnx.py`); falls back to scanning `<model_root>/ocr/<model_id>`. Requires the `ocr` feature (`tract` or `ort`) |
| `NEURON_OCR_EXECUTOR` | `tract` | `tract` \| `ort` — PP-OCR session executor. `ort` is ~10× faster on CPU but loads onnxruntime at runtime |
| `NEURON_OCR_SESSIONS` | `2` | ort sessions per graph (its `run` takes `&mut`, so concurrent requests need one session each; tract plans are `run(&self)` and ignore this) |
| `ORT_DYLIB_PATH` | (unset) | Path to `onnxruntime.dll` for the `ort` crate's load-dynamic linking (needed by the OCR `ort` executor; falls back to `PATH` lookup) |
| `NEURON_IMAGEGEN_RUNTIME` | (unset) | `moss-image` — which text-to-image engine to register (MOSS-Image on candle, `POST /v1/images/generations`). Requires the `candle` feature (default on) |
| `NEURON_IMAGEGEN_MODEL_PATH` | (unset) | Path to the MOSS-Image model dir — the upstream HF repo layout (`transformer/` + `text_encoder/` + `vae/` + `tokenizer/` + `scheduler/`). Unset falls back to `$NEURON_MODEL_DIRS/imagegen/<model_id>/` |
| `NEURON_IMAGEGEN_DTYPE` | `f16` | `f16` \| `f32` \| `bf16` — storage dtype for the DiT + text encoder weights (the VAE always decodes fp32). `f16` ≈ 27 GB resident, `f32` ≈ 54 GB; candle's CPU backend has no BF16 matmul, so `bf16` needs CUDA |
| `NEURON_IMAGEGEN_RESIDENCY` | `staged` | `staged` \| `cached` — how the three components share device memory. `staged` loads one at a time (the text encoder is dropped before the DiT loads), so the peak residency is ~17 GB at f16 instead of ~27 GB — the only way a 24 GB card can serve the model. `cached` keeps everything resident (faster; needs the memory) |

## Where the design specs live

- `docs/superpowers/specs/` — current design docs (8). These are **maintainer-facing**: spec rationales, design pivots, and post-mortems for major features. End users do not need them.

When you add a new model type or engine, write a one-page spec under `docs/superpowers/specs/` and link it from the docs (docs/concepts/engines.md) and AGENTS.md.

## Source registry (multi-source puller)

`SourceRegistry` ships with three registered sources out of the box:

| Source id | Backend | Default endpoint | CLI fallback | Notes |
|---|---|---|---|---|
| `huggingface` | HF REST + `huggingface-cli` | `https://huggingface.co` | `huggingface-cli` | Primary HF. `HF_TOKEN` env var for gated repos. |
| `hf-mirror` | HF REST + `huggingface-cli` | `https://hf-mirror.com` | `huggingface-cli` | China-friendly CDN; same wire protocol as HF so the HF download logic works unchanged. |
| `modelscope` | MS REST + `modelscope-cli` | `https://www.modelscope.cn` | `modelscope-cli` | Alibaba's open model hub. `MODELSCOPE_TOKEN` env var for gated repos. **Platform default source.** |

References on `POST /v1/models/pull` and in catalog `default_repo` use the `<source>:<repo>` syntax. When the prefix is omitted the platform falls back to `NEURON_DEFAULT_SOURCE` (default `modelscope`). The puller also strips a leading `<other-source>:` prefix if the user has overridden the source via `NEURON_LLM_SOURCE` / `NEURON_MODEL_SOURCE_*` — so requesting `modelscope:openbmb/MiniCPM5-1B` from a process whose per-engine source is `hf-mirror` correctly routes to `openbmb/MiniCPM5-1B` on the mirror rather than 404-ing.

Both HF-family sources fall back to a streaming HTTP path (reqwest `bytes_stream`) when the CLI binary is not on disk. The HTTP fallback:

- Reads `HF_TOKEN` and attaches it as `Authorization: Bearer <token>` (HF family only).
- Reads `HTTP_PROXY` / `HTTPS_PROXY` env vars and builds a `reqwest::Proxy` (intentionally skips `ALL_PROXY` — it may be SOCKS which reqwest can't handle without the `socks` feature).
- Streams to disk in chunks; publishes `DownloadProgress` every 1 MiB so the UI progress bar moves without burning CPU on the consumer side.

Catalog dedup: the same `(engine, id)` may appear multiple times in `SUPPORTED_CATALOG` (once per source). `Catalog::entries()` merges them into a single `CatalogEntry` whose `sources: Vec<SourceRef>` carries one entry per repo, so `/v1/models` shows each model exactly once but exposes every available download source.

## Platform default model

`DEFAULT_MODEL_ID` in `src/capabilities/catalog.rs` is what `/v1/models` tags with `"is_default": true`. Currently set to `minicpm5-1b`:

- **Model**: MiniCPM5-1B from OpenBMB — 1B `LlamaForCausalLM` (24 layers, 16 Q-heads GQA, 131K context), Apache 2.0.
- **Why this one**: 1B-class open-source SOTA on agentic tool-use, code generation, and difficult reasoning (per OpenBMB's public leaderboard); fits on 2-core 4 GB edge devices via candle (Llama path); built-in hybrid `<think>` reasoning without a separate model.
- **Where it lives**: `modelscope:OpenBMB/MiniCPM5-1B` (`source_id = "modelscope"` in the catalog). HF fallback at `huggingface:openbmb/MiniCPM5-1B` if the user has switched the default source via `NEURON_DEFAULT_SOURCE=huggingface`.
- **Runtime**: `candle` — auto-detected from `config.json` `architectures = ["LlamaForCausalLM"]` by `src/llm/candle_arch.rs::select_architecture()` → `load_llama()`.

### MiniCPM5-2B (registered alongside the default, not the default itself)

`minicpm5-2b` is registered in the same default catalog as a higher-fidelity alternative for hosts that have the VRAM headroom. It is NOT the platform default — `DEFAULT_MODEL_ID` remains `minicpm5-1b` to preserve the edge-tier branding.

- **Model**: MiniCPM5-2B from OpenBMB (released 2026-09-07) — 2.52B `LlamaForCausalLM` (42 layers, hidden_size 2048, intermediate_size 6144, GQA 16Q/2KV, head_dim 128, 131K native context, vocab 130560, `rope_theta=5_000_000`, `tie_word_embeddings=false`, bos=0, eos=[1, 130073]), Apache 2.0. Self-positioned by OpenBMB as "2B-class SOTA, competitive with 4B-class models"; same Standard `LlamaForCausalLM` architecture as MiniCPM5-1B.
- **Why alongside (not instead of) the 1B**: 2.52B BF16 needs ~6–7 GB VRAM (vs 1B's ~3 GB), so the 2B is impractical on the 2-core 4 GB edge tier the platform targets. Hosts that want higher-quality chat serve 2B explicitly via `with_llm_model("minicpm5-2b")` or the equivalent `NEURON_LLM_MODEL_PATH` pointing at OpenBMB/MiniCPM5-2B.
- **Where it lives**: `huggingface:openbmb/MiniCPM5-2B` (also on `modelscope:OpenBMB/MiniCPM5-2B` — same dedup-by-`(engine, id)` catalog path used for MiniCPM5-1B).
- **Runtime**: `candle` — same `model_type = "llama"` dispatch as MiniCPM5-1B. `candle_arch::detect_arch` returns `CausalArch::Llama` and the structural fields (42 layers, GQA 2, hidden_size 2048, intermediate_size 6144, vocab_size 130560, `rope_theta=5_000_000`, `tie_word_embeddings=false`) are preserved end-to-end. Verified by `src/llm/candle_arch.rs::test_detect_arch_minicpm5_2b_dispatches_to_llama` against a synthetic config blob mirroring OpenBMB's published checkpoint spec (see `src/capabilities/catalog.rs` top-of-file doc-comment for the side-by-side 1B/2B field table).
- **Hybrid `<think>`**: same `enable_thinking` chat-template flag as MiniCPM5-1B — no per-model code change needed.

Other catalog families — loadable by capability, by backend. ✓ = the model is served by that backend today; the source of truth is `default_capabilities()` in `src/capabilities/catalog.rs` (see docs/reference/models.md for the full table including defaults + build flags):

| Capability | candle | burn | litert | mistralrs | mage_vl | sherpa | tract | ort | Cohere |
|---|---|---|---|---|---|---|---|---|---|---|
| LLM | ✓ (Qwen2/3, Llama2/3, Phi-3, Mistral, ChatGLM, MiniCPM5) | | ✓ (`gemma3-1b`, `phi-4-mini`, `qwen2.5-1.5b`) | ✓ (40+ families) | ✓ (`microsoft/Mage-VL`) | | | | | |
| TTS | | ✓ (VoxCPM 0.5B + VoxCPM2) | | | | | | | | |
| ASR | ✓ (Whisper) | | | | | ✓ | ✓ | | | |
| Rerank | ✓ (BGE / mxbai / jina) | | | | | | | | | ✓ (hosted) |
| Embedding | ✓ (MiniLM) | | | | | | | | | |
| Vision | ✓ (ONNX session path: YOLOv8 / YOLO11 / RT-DETR / DETR / RF-DETR) | | | | | | | ✓ (tract executor) | | |
| Chronos2 | ✓ (needs `--features chronos2` + a checkpoint) | | | | | | | | | |
| Decision | | | | | | | ✓ | ✓ (bit-exact path) | | |

**`mage_vl` is a dedicated column even though it sits on candle.** `MageVlRuntime` (`src/mage_vl/runtime.rs`) implements `engine::LlmRuntime` on top of candle directly — its own MageViT vision tower + Mage-specific multimodal token sequencer + per-request KV-cache reset. candle-transformers ships none of those, so the generic candle LLM runtime cannot serve `microsoft/Mage-VL`. Listing it as a candle row would be misleading; the dedicated column expresses "this model id has its own engine, even though both share the candle primitives".

## Open TODOs (in source)

_The runtime registry refactor (2026-06-12) added `LlmRuntimeRegistry`, `TtsRuntimeRegistry`, `AsrRuntimeRegistry`, `RerankRuntimeRegistry`, `EmbeddingRuntimeRegistry` — each with lazy load + resource checking. The previous singleton-runtime era entries (streaming, weight-conversion) were completed in `d9e131d` and `11fe410`. All three post-refactor gaps (capability-aware registration processing, shortcut reflecting first registered model, `embedding_loaded` in `NeuronHealth`) were closed in the 2026-06-16 follow-up commit._

### Known gaps
_None — see commit log for the 2026-06-16 gap-closure follow-up, the 2026-06-25 plugin-architecture follow-up (Chunks 1-5 implemented: Canonical types, pluggable engines, pluggable adapters, Anthropic Messages v1 adapter, `/v1/messages` route wired), and the 2026-07-13 ModelScope-default + MiniCPM5-1B pass (3-source registry, MODELSCOPE_TOKEN, cancel API, dedup catalog, default-model switch)._

### Sprint 3 — Chronos2 forecasting (runtime landed, 2026-10-29 status)

The T5-style port **has landed**: `src/chronos2/vendor/` carries the encoder/decoder (`architecture.rs`, `config.rs`, `embeddings.rs`, `transformer.rs`) and `src/chronos2/candle_runtime.rs` implements `CandleChronos2Runtime` (loads a safetensors checkpoint, runs `forecast_batch`, quantile output head). The `chronos2` Cargo feature gates it — it is **not** in the default feature set, so a default build keeps the `StubChronos2Runtime` path (which answers `NotImplemented` → 503). `NEURON_CHRONOS2_MODEL_PATH` selects the checkpoint; `NEURON_CHRONOS2_RUNTIME` is documented but not read.

**Still not done**:
- A `SUPPORTED_CATALOG` entry (nothing downloadable loads as-is — the checkpoint comes from the upstream Chronos2 release).
- Parietal `forecast` A2A skill.

### Sprint 4 — Mobile / edge optimization (Phase 1 landed 2026-09-01)

`TensorPool` (in `src/vision/memory_pool.rs`) reuses pre-allocated tensors across forward passes to reduce allocation overhead and peak memory on mobile/edge devices. Thread-safe, bucketed by shape, diagnostics API (`size()`, `memory_bytes()`, `clear()`).

Wired into `CandleYoloRuntime` as a private field with three public diagnostic methods (`pool_size()`, `pool_memory_bytes()`, `pool_clear()`). Returned tensors are NOT zeroed — callers overwrite values before use, matching the typical forward-pass pattern where every tensor slot is filled by computation or `copy_from`.

Benchmark (`benches/mobile_bench.rs`, registered in `Cargo.toml`):
- `alloc_without_pool_640x640`: **~460 µs** per fresh `Tensor::zeros(1,3,640,640)` allocation
- `alloc_with_pool_640x640`: **~292 ns** per get+release — **~1574× speedup**
- `pool_warmup/warm_steady_state`: hot path stays below 200 ns after 5-call warmup
- `pool_size_scaling/[1,10,100,1000]`: throughput ~3.7 Melem/s regardless of pool size

For YOLOv8n (640×640 input), poolable intermediates (input tensor, P3/P4/P5 feature maps, head outputs) total ~32 MB vs ~200 MB peak without pooling — a **6× reduction** in poolable footprint.

**Phase 2 (pending)**: ONNX Runtime activation via tract backend (pure Rust ONNX, no C++ toolchain), targeting sub-30 ms YOLOv8n latency on 2-core ARM.
**Phase 3 (pending)**: MNN backend integration via FFI for low-end Android/iOS devices with 2-core ARM + 4 GB RAM.

523 lib tests pass as of 2026-09-01. No regression.

### Sprint 4 — Mobile / edge optimization (Phase 2 landed 2026-09-01)

The `tract` Cargo feature activates pure-Rust ONNX inference. When the feature is on, `build_session()` in `src/vision/ort_runtime.rs` dispatches to the existing `TractSession` in `src/vision/ort_backends/tract_backend.rs` (rewritten for tract 0.21 API: `Arc<TypedRunnableModel<TypedModel>>`, `tvec![…]`, `into_tvalue()`). When the feature is off, the previous `not_implemented` stub is still returned.

**What this unlocks**:
- YOLOv8 / RT-DETR / DETR / Florence-2 / RF-DETR ONNX exports run on ARM / x86_64 / Apple Silicon without any C++ toolchain, system ONNX Runtime, or GPU.
- The runtime's full preprocessing (letterbox 640×640), output decoding (YOLOv8 anchors / DETR queries), task dispatch (detect/classify/segment/pose), and NMS already work — only the session construction was missing.
- The TensorPool wired in Phase 1 buffers forward-pass intermediates, so tract's allocation-heavy decode benefits from the same pool.

**Build**:
```bash
cargo build --lib --features tract            # ~30s incremental
cargo test  --lib --features tract            # 524 lib tests pass (+1 tract test)
cargo build --lib                              # default slim build, 523 lib tests pass
```

**Verified** (commit `5b2ab73`):
- Default build: 523 lib tests, no regression
- With `tract`: 524 lib tests, no regression
- `build_session("/nonexistent/path.onnx")` returns a clear `invalid_input` error
- `build_session` with a real ONNX file loads via `tract_onnx::onnx().model_for_path(...)` and runs via `model.into_optimized().into_runnable()`

**Phase 3 (pending)**: MNN backend integration via FFI for low-end Android/iOS devices with 2-core ARM + 4 GB RAM.
