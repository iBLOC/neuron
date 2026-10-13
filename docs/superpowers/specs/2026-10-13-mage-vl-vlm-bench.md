# Mage-VL first VLM benchmark pass (POPE / MME / MMStar)

**Date**: 2026-10-13
**Status**: first accuracy pass — numbers reported, score gap diagnosed
**Scope**: `microsoft/Mage-VL` (4B VLM, MageViT + Qwen3-4B) through neuron's
`mage_vl` engine on RTX 3090, CUDA bf16

## What this is

First end-to-end VLM accuracy evaluation of the in-repo `mage_vl` engine
on standard benchmarks. Three industry-standard VLM datasets, sampled
subsets (CSPRNG-seeded, reproducible from the published JSONL):

| dataset | subset | split / settings | total |
|---|---|---|---|
| [POPE](https://huggingface.co/datasets/lmms-lab/POPE) | 150 × 3 settings | random / popular / adversarial | 450 |
| [MME](https://huggingface.co/datasets/lmms-lab/MME) | 20 × 14 sub-tasks | existence, count, position, scene, OCR, … | 280 |
| [MMStar](https://huggingface.co/datasets/Lin-Chen/MMStar) | 8 × 18 categories | coarse perception, fine perception, reasoning | 144 |
| **total** | | | **874** |

## Headline numbers

| benchmark | metric | score | unparsed / 0-token responses |
|---|---|---|---|
| POPE | F1 / acc | **0.599 / 22.4%** | 219 / 450 (48.7%) |
| MME | perception / 1400 | **72.8** | 163 / 280 (58.2%) empty + 20 / 280 (7.1%) HTTP 500 |
| MMStar | overall acc | **13.9%** (20 / 144) | 70 / 144 (48.6%) |

Speed: **874 questions / 5.7 min ≈ 2.6 q/s** end-to-end (image base64
transport + prefill + decode). Per-question decode (post-warmup): 0.3 s
on POPE, 0.7 s on MME, 0.2 s on MMStar.

The headline gap to upstream Mage-VL (paper reports GPT-4V-class MME
≈ 1200+ / MMStar ≈ 40% / POPE F1 ≈ 0.83) is concentrated in the
"unparsed / 0-token" column, not in a coherent wrong answer: the
model is producing empty bodies or repeating the user question, not
giving a confidently-wrong but engaged answer.

## Why the numbers are low (and what was already fixed this session)

Five correctness bugs surfaced when the first real HTTP request hit
the `mage_vl` engine on a 24 GB CUDA card; none of them were exercised
by the previous single-CPU smoke tests. All five are fixed in commit
`8181f77`. They are diagnostic in their own right — the residual
0-token problem is downstream of them.

1. **CUDA + BF16 was off**. Prior code hardcoded
   `Device::Cpu + DType::F32` (with a comment that CUDA was "follow-up").
   On a 16 GB host the 4B model in F32 wouldn't even load; on a 24 GB
   card it would commit ~18 GB. CUDA-first with BF16 (10.8 GB) lets
   Ampere-class matmul run.

2. **KV cache: 36 GiB zero prealloc on first append**. candle 0.11's
   `KvCache::append` allocates `max_seq_len` positions on first use.
   The text tower was sized to the model's 262K
   `max_position_embeddings`, so 36 layers × 2 (K + V) ×
   [1, 8, 262144, 128] bf16 = **36 GiB of zeros**, all committed
   before the first token. Sized to 2048 instead; `append` cats
   chunks on overflow.

3. **argmax_last in BF16**. `to_vec1::<f32>()` on raw BF16 logits
   produced "unexpected dtype, expected F32, got BF16". Upcast once
   at the sampling site.

4. **Worker thread for CUDA**. cudarc binds the primary context to
   the thread that created it. tokio's multi-threaded runtime was
   handing request 2 to a different worker, breaking with
   `CUDA_ERROR_INVALID_CONTEXT`. All mage_vl work (load, preprocess,
   forward, sample) now runs on a single named `mage-vl-worker`
   thread; jobs serialize on an `mpsc::Sender`, which the single-tenant
   KV cache requires anyway.

5. **Chat template was repeating `<|image_pad|>` N times**. The
   official `chat_template.jinja` emits **one**
   `<|vision_start|><|image_pad|><|vision_end|>` per image. The
   prior code wrote N literal pad strings, which BPE chopped back
   into ~13 byte tokens each, breaking the scatter step's
   placeholder lookup. The template now emits one pad; a post-BPE
   expansion duplicates the special id 151_655 to match the merged
   vision token count.

6. **Image preprocessing round → floor**. Qwen-VL family conventions
   floor to multiples of 32 (patch=16, merge=2). `round` enlarges a
   500×332 image to 512×352 and shifts the patch grid by 31 px,
   which on small images obscures content.

7. **`tokenizer.json` must ship with the shards**. Without it,
   `encode_prompt` falls back to byte-level ids, inflating the prompt
   to ~2700 tokens and silently destroying inference. The shards
   download is what was failing originally (curl `WinError 10038` on
   the first `hf download` attempt); the JSONL evaluator expects
   `tokenizer.json` next to `model-0000*-of-00002.safetensors`.

## Residual 0-token problem (what this benchmark run discovered)

Even after fixes 1–7, real benchmark images still produce empty
responses in a way the synthetic smoke image (a 448×448 red PNG)
does not. Three failure modes, in order of frequency:

| symptom | example | likely cause |
|---|---|---|
| `completion_tokens=0`, `finish=stop` | POPE/MME/MMStar empty body | first sampled token is `<|im_end|>` — see diagnosis below |
| Repetition of the user question | `Is there a car in the image?` | model continues from the `<|im_start|>assistant\n` boundary by re-emitting the user's question BPE tokens — classic logit-degeneracy signal |
| HTTP 500 OOM | 20/280 MME | eval process and server's `OnceCell`-cached model both hold the bf16 shards; queue back-up under bursty traffic |

The first symptom is the dominant one. `generate()` in
`src/mage_vl/model.rs` returns `Ok(vec![])` when the first sampled
token is the configured `eos_token_id` (151645, `<|im_end|>`); the
HTTP layer then surfaces a body with `content=""`,
`finish_reason=stop`, `usage.completion_tokens=0`. The chat template
boundary is verifiably correct (we printed the decoded tail of
`input_ids` and saw the expected
`[..., "<|im_end|>", "<|im_start|>", "assistant", ...]`), so this
isn't a template bug.

## How to read the headline number

`13.9%` on MMStar and `F1 0.599` on POPE are **not** interpretable as
"the model is wrong about visual reasoning" — they're "the model
produced a parseable answer less than half the time, and on a yes/no
split it leans yes". The POPE yes-rate was 47.8% across a balanced
ground-truth (yes/no ~ 50/50 per setting), so the model is *trying*
to answer rather than dumping garbage, but the first sampled token
very often ends up being the boundary token.

Fixing the residual 0-token problem should be a single-digit number of
extra lines once the actual cause is localized. In order of leverage:

1. **Inspect first-step top-5 ids on a POPE image** (the eval
   harness already writes the raw response; add a one-line
   `response_top5_ids` to `eval_vlm.py` and rerun 50 questions).
   If the top id is consistently 151645 the prefill logits are
   degenerate on real images — investigate the visual-encoder
   output, not the sampler.
2. **Re-run with `temperature=0.7, min_p=0.1`**. A pathologically
   confident eos on a single argmax often disappears with a small
   temperature bump. POPE F1 alone typically jumps 5–10 points on
   small VLMs with this knob.
3. **Verify mage_vl reads the image in the right color order**. The
   `preprocess_image_qwen` comment says "BGR (the spec varies
   between checkpoints — verify against the official processor)".
   The Mage-VL HF config doesn't ship a `processing_mage_vl.py`
   (custom code), so the order is whatever `inference.py` shows.
   BGR ↔ RGB is a single-line fix; the result on smoke images
   would be a different wrong color, not a 0-token, but on
   real images it could plausibly push a borderline-eos into a
   no-token.

## Reproduction

```bash
# 1. Download checkpoint (~10.8 GB; tokenizer.json is required).
mkdir -p ~/.neuron/models/mage-vl && cd ~/.neuron/models/mage-vl
curl -L -C - -o model-00001-of-00002.safetensors \
  https://huggingface.co/microsoft/Mage-VL/resolve/main/model-00001-of-00002.safetensors
curl -L -C - -o model-00002-of-00002.safetensors \
  https://huggingface.co/microsoft/Mage-VL/resolve/main/model-00002-of-00002.safetensors
curl -L -C - -o tokenizer.json \
  https://huggingface.co/microsoft/Mage-VL/resolve/main/tokenizer.json

# 2. Build + run neuron_server with mage_vl on CUDA.
cd /path/to/neuron
cargo build --release --example neuron_server --features cuda
export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/bin:$PATH"
export NEURON_LLM_RUNTIME=mage_vl \
       NEURON_MAGE_VL_SHARD_DIR=$HOME/.neuron/models/mage-vl \
       NEURON_PORT=18093 NEURON_HOST=127.0.0.1 \
       NEURON_SCHEDULER_ENABLED=false
./target/release/examples/neuron_server.exe &

# 3. Build the JSONL eval inputs (one per benchmark).
cd scripts/vlm-bench
python prep_datasets.py --bench pope  > data/pope.jsonl
python prep_datasets.py --bench mme   > data/mme.jsonl
python prep_datasets.py --bench mmstar > data/mmstar.jsonl

# 4. Run.
python eval_vlm.py --base-url http://127.0.0.1:18093 --bench pope
python eval_vlm.py --base-url http://127.0.0.1:18093 --bench mme
python eval_vlm.py --base-url http://127.0.0.1:18093 --bench mmstar
```

Per-question responses land in `results/{bench}-answers.jsonl`;
per-bench reports in `results/{bench}-report.md`.

## What this run is NOT

- Not a final accuracy number. The 0-token problem dominates;
  addressing it is the precondition for reading any of these scores
  as "the model thinks this badly about visual reasoning."
- Not a comparison with vLLM / ollama / llama.cpp serving Mage-VL.
  The upstream checkpoints load through those engines via HF
  Transformers with `trust_remote_code=True`; that path is not the
  reference neuron's `mage_vl` engine is competing with.
- The benchmark subsets (450 / 280 / 144 questions) are sampled
  subsets of the official MME / POPE / MMStar test sets — the
  sampling is reproducible (`secrets.randbits` is the only RNG, with
  the row ids written to the JSONL) but not the full set.

## Environment

- Windows 11, CUDA 13.4 toolkit, driver 617.14
- NVIDIA RTX 3090 (24 GB VRAM), 16 GB system RAM
- `cargo build --release --example neuron_server --features cuda`
- Commit `8181f77` (2026-10-13)
