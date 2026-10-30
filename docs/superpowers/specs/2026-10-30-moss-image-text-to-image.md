# MOSS-Image text-to-image capability (2026-10-30)

## What landed

neuron gained a **`imagegen` capability** — the first text-to-image
capability in the crate — serving OpenMOSS's **MOSS-Image** natively on
candle at `POST /v1/images/generations`.

- Model: [OpenMOSS-Team/MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base)
  (and `-SFT`), Apache-2.0, released 2026-10-16.
- Activation: `NEURON_IMAGEGEN_RUNTIME=moss-image` (+ the `candle`
  feature, which is on by default).
- Weights: the upstream HF repo layout, loaded **directly** — no
  conversion step (the same promise the VoxCPM2 path makes).
- Surface: OpenAI Images API shape — `{model, prompt, n, size,
  response_format}` → `{created, data:[{b64_json, size, seed}]}`, plus
  neuron extensions (`negative_prompt`, `steps`, `guidance_scale`,
  `seed`, `max_sequence_length`).

## Why native and not a proxy

neuron is a local-weights inference service. An API-forwarding backend
is out of scope by product decision, so the only question was: can the
model run in-process? It can — but only as a **GPU-practical** path.

## The model, component by component

The upstream repo ships one self-contained `pipeline_moss_image.py`
(~440 lines of plain PyTorch, no hidden components), which made the
port tractable. Three components, 13.6B parameters, ~27.2 GB of
BF16 safetensors:

| Component | Weights | Port |
|---|---|---|
| DiT (5.08B) | 10.2 GB | `src/imagegen/moss_image/dit.rs` — vendored |
| Text encoder (Qwen3.5 hybrid, ~8.4B) | 16.8 GB | `src/imagegen/moss_image/text_encoder.rs` — vendored |
| VAE (FLUX.2 small decoder) | 0.25 GB | `src/imagegen/moss_image/vae.rs` — assembled from candle's public SD-VAE blocks |
| Scheduler | — | `src/imagegen/moss_image/scheduler.rs` — vendored |

**The DiT is not a Flux clone.** It is a *single-stream* joint-sequence
transformer: text tokens are projected into DiT space, concatenated
with the image tokens, and 42 blocks attend over the union with one
**shared** time-conditioned modulation (`dim → 6·dim`) instead of
per-block adaLN. RoPE is 4-axis `(r, f, h, w)`. Per-head QK-norm, a
`_situ` soft-clamped gated FFN. This is *simpler* than Flux — no
double-stream, no MM-DiT — so the vendored file is small.

**The text encoder is the expensive part.** `model_type: qwen3_5` is a
hybrid stack: 32 text layers = 24 Gated DeltaNet (FLA-style chunked
kernel) + 8 full attention, `head_dim` 256, partial rotary 0.25,
interleaved mrope, `attn_output_gate`, vocab 248K. **candle 0.11 has
no module for it** and neuron's `detect_arch` maps `qwen3_5` onto the
*dense* Qwen3 loader, which cannot read this checkpoint. So the text
tower is vendored here too, prefill-only (the pipeline encodes each
prompt in one forward and takes `last_hidden_state`). The vision tower
(`visual.*`) is never touched.

## Porting notes that cost real debugging

These are the traps worth remembering — each one produced silently
wrong numbers or NaNs, not compile errors:

1. **`broadcast_div(variance^-0.5)` is the wrong direction.**
   `x / rms(x)` must *shrink* the activation. Writing
   `broadcast_div(&(var + eps).powf(-0.5))` divides by `1/rms`, i.e.
   multiplies by `rms` — which *grows* the activation by ~10⁶ for a
   typical weight scale and produced `inf` attention scores. Three
   sites (DiT norm, Qwen3.5 `(1+w)` norm, the GDN gated norm) had the
   same inversion; the DiT one is now guarded by
   `test_rms_norm_divides_by_rms_not_multiplies`.

2. **The GDN tril mask must be applied *before* the exp.** The
   reference does `.tril().exp().tril()`; candle has no `tril`, so the
   mask is multiplicative. Masking *after* the exp evaluates
   `exp(+hundreds)` in the upper triangle → `inf`, and `inf · 0 = NaN`.
   The mask now goes in first.

3. **`g_last` must stay un-exponentiated.** In the chunk recurrence
   the reference computes `(g[..., i, -1] - g[..., i, :]).exp()` with
   the *raw* cumulative decay. Pre-exponentiating `g_last` and then
   subtracting the raw cumsum flips the sign and explodes `exp` to
   `inf`.

4. **candle's `+` / `-` do not broadcast.** `tensor + [1,1,dim]` fails
   with a shape error; `broadcast_add` / `broadcast_sub` are needed
   wherever a per-token tensor meets a per-feature one. The GDN decay
   mask in particular is a broadcast subtract.

5. **candle's matmul is 4-D only.** The GDN chunk kernel needs
   `[B, H, NC, C, D]` batched matmuls; the NC axis is folded into the
   batch dimension (or squeezed per chunk) before every `matmul`.

6. **The UT transform drops a dimension.** torch's `attn[..., i, :i]`
   squeezes the fixed row index — a rank-preserving port produces a
   5-D tensor where the reference has 4-D, and the correction term
   then broadcasts against the wrong axis.

7. **`value` is REASSIGNED by the UT transform.** The reference does
   `value = attn @ v_beta` and the per-chunk loop then reads
   `value[:, :, i]` — the *transformed* values. A port that keeps the
   raw `v` computes a different recurrence (it drops the `beta`
   factor). This one is easy to miss because a `beta = 1` unit test
   cannot distinguish the two; the analytic anchor now uses
   `beta = 0.5` so it can.

8. **The FLUX.2 decoder's mid block runs at the *largest* channel
   count** (384 = the last entry of `decoder_block_out_channels`),
   while `conv_norm_out`/`conv_out` run at the smallest (96). Using the
   reversed-list tail for both gives a GroupNorm channel mismatch.

9. **VAE latent layout**: 128-channel latent → BN de-normalization
   (tracked `running_mean/var`, no affine) → 2×2 pixel unshuffle →
   `post_quant_conv` → decoder. Decoding is fp32 regardless of the DiT
   dtype.

10. **The scheduler's sigma grid is snapped, not continuous.** The
    inference sigmas come from the shifted grid
    `σ = shift·v/(1+(shift−1)·v)`, then each is snapped to the nearest
    point of the 1000-point *training* grid and consecutive duplicates
    collapse. A 50-step run therefore has exactly 50 distinct sigmas
    (verified against the upstream Python: `1.0, 0.996, 0.992, 0.987,
    0.983, …, 0.024`).

11. **The CFG negative branch skips DiT block 22** (`uncond_skip_layers
    = (22,)`) — two forwards per step, not three. `guidance_scale = 1.0`
    drops the negative branch entirely.

## Sampling pipeline (per request)

1. Prompt → Qwen3.5 chat template (single user turn,
   `enable_thinking=True`, so the rendered prompt ends with
   `assistant\n<think>\n`) → tokenized (truncated at 4096) → hybrid GDN
   encoder → `last_hidden_state` `[1, S, 4096]` fp32.
2. Seeded fp32 Gaussian noise `[1, 128, H/16, W/16]`. The RNG is
   neuron's own xorshift64* + Box-Muller on CPU, so the same seed
   gives the same noise on every backend (CUDA's native RNG is not
   seeded the same way).
3. Flow-matching Euler loop (50 steps): conditional forward +
   negative forward (block 22 skipped), combined as
   `uncond + cfg·(cond − uncond)` **in fp32**.
4. FLUX.2 decode → PNG (`[-1,1] → [0,255]`).

## Memory & speed reality

- f16 default (`NEURON_IMAGEGEN_DTYPE=f16|f32|bf16`): ~27 GB resident
  for the weights, ~54 GB at f32. The registration budget is 28 GB.
- **CPU is not practical**: 5B DiT × 4096 latent tokens × 50 steps is
  a ~2 PFLOP workload, plus candle's CPU backend has no BF16 matmul.
- **CUDA is the practical path.** The 27 GB of f16 weights do not fit
  on a 24 GB card together with activations, but the text encoder only
  needs one prefill pass — encode → free → load the DiT keeps the peak
  near 17 GB. That staged residency is *not* implemented yet (the
  runtime loads both components eagerly), so a 24 GB card needs the
  text encoder unloaded before the DiT runs. **This is the next
  increment**, and it mirrors what the upstream Gradio app does with
  its three memory modes.
- The image-generation registry is **not** wired into the model
  scheduler (`/v1/scheduler/*` is LLM-only), so a 27 GB image model
  and an LLM cannot be swapped cooperatively yet.

## What is NOT done

- **The text encoder and the DiT have not been compared on the real
  weights.** The VAE has (see below); the other two need a host with
  a CUDA-enabled PyTorch (the reference side) and more RAM than a
  16 GB box — the text encoder alone is 8.4B parameters, and a fp32
  reference does not fit. Their *formulas* are verified numerically
  against the upstream implementation (the parity section above) and
  their *weight paths and shapes* against the real checkpoint (below),
  but no run against the real 27 GB checkpoint has happened — image
  quality is unverified.
- No `n > 1` batching (the loop is sequential per image).
- `response_format` accepts only `b64_json` (neuron serves local
  bytes; there is no URL to hand out).
- The SFT variant is loadable through the same path (it is the same
  architecture), but no LoRA adapter support.
- The `imagegen` capability is **not wired into the model scheduler**
  (`/v1/scheduler/*` is LLM-only), so a 27 GB image model cannot be
  swapped cooperatively with an LLM yet.

## Real-weight verification (the VAE)

`scripts/gen-moss-real-ref.py` runs the upstream implementations on
the **real checkpoint** and dumps their outputs;
`src/imagegen/moss_image/real_weights.rs` then runs neuron's port on
the same weights and compares element-wise:

```
python scripts/gen-moss-real-ref.py --only vae --device cpu
cargo test --lib real_weights -- --ignored --nocapture
```

Measured 2026-10-30 on the real `OpenMOSS-Team/MOSS-Image-Base` VAE
(251 tensors, fp32): **relative error 1.5e-5** (max abs diff 1.1e-5
against a reference magnitude of 0.72). The FLUX.2 decoder — the
batch-norm de-normalization, the 2×2 pixel unshuffle and the decoder
stack itself — matches diffusers' `AutoencoderKLFlux2` on the real
weights.

**This caught the last bug: the prompt template was missing the turn
markers.** The upstream chat template renders

```
<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n
```

and neuron's hard-coded template had dropped the four
`<|im_start|>` / `<|im_end|>` markers — the model would have seen a
malformed conversation on every request. The check compares the
rendered text **and** the token ids (32 for the reference prompt), so
a template that renders the same text but tokenizes differently is
caught too.

The tests **skip** when their reference has not been generated, so
the suite stays green on a host without the checkpoint; each prints
the command that would produce the missing reference.

Note on the reference side: the reference runs the upstream code in
the *installed* packages (`transformers` for the text encoder,
`diffusers` for the VAE) and the repo's own `pipeline_moss_image.py`
for the DiT — nothing is re-typed.

## Staged residency (24 GB cards)

The pipeline needs 13.6B parameters — ~27 GB at f16 — which does not
fit a 24 GB card with the text encoder and the DiT resident at the
same time. The runtime therefore loads the components **per stage**:

1. the text encoder encodes the prompt (and the negative prompt),
   then is dropped;
2. the DiT runs the sampling loop, then is dropped;
3. the VAE decodes the latent to a PNG.

The peak residency is one component (~17 GB at f16) instead of all
27 GB. `NEURON_IMAGEGEN_RESIDENCY=cached` keeps everything resident
instead — faster (no reload) on a host with the memory for it;
`staged` is the default.

The cost of the staged mode is re-reading the weights from disk for
every request, which the memory-mapped loader does lazily.


## Numerical parity against the upstream implementation

Unit tests that assert *shapes* cannot catch a wrong formula. The
parity fixtures under `src/imagegen/moss_image/testdata/` close that
gap: `scripts/gen-moss-parity.py` runs the **upstream code itself** —
the Gated DeltaNet kernel from the installed `transformers` package
(`transformers.models.qwen3_5.modeling_qwen3_5`) and the DiT from the
upstream `pipeline_moss_image.py` in the model repo — and dumps the
inputs plus the reference outputs. Nothing upstream is re-typed in
the script, so the fixture is a genuine reference rather than a
second transcription. The inputs are deterministic (synthesised from
a fixed trigonometric sequence, not drawn from an RNG), so the
fixtures regenerate bit for bit.

`src/imagegen/moss_image/parity.rs` then runs neuron's port on the
same inputs and compares element-wise:

```
cargo test --lib moss_image::parity -- --nocapture
```

Measured 2026-10-30 (fp32, CPU):

| Component | relative error vs upstream |
|---|---|
| GDN chunk kernel | 1.1e-6 |
| DiT forward | 2.0e-6 |
| DiT forward with a skipped block (CFG negative branch) | 1.3e-6 |

The residual is float32 summation order, nothing semantic.

**This is what caught the last two bugs**, both of which were silent:

1. **The modulation chunks were read in the wrong order.** Upstream
   unpacks `shift_a, scale_a, gate_a, shift_f, scale_f, gate_f`; the
   port had scale and shift swapped in both the attention and the
   FFN branch. A shape-level test cannot see this — the tensors all
   have the right shape.
2. **The attention gate was applied to the wrong side of the output
   projection.** Upstream computes `x + proj['o'](attn) * gate_a`;
   the port computed `proj['o'](x + attn * gate_a)`.

Before the fix the DiT's relative error was **0.38**; after it,
**2.0e-6**.


## Weight-path verification (offline, seconds)

`src/imagegen/moss_image/weight_names.rs` closes the "typo in a
`pp(...)` segment" risk — the failure mode that would otherwise only
surface after a 27 GB download, as "cannot find tensor X".

A recording `VarBuilder` backend answers every request with a
one-element tensor and remembers the **path and the requested shape**.
Building the three modules with it yields the exact set of paths the
port asks for (names and shapes are dimension-independent, so the
probe can run with tiny dims — and with the real config for the shape
check, since the recorder never materializes the weights).

The test then fetches the upstream index files (the two
`*.safetensors.index.json` manifests and the JSON headers of the
shards — a few MB in total) and diffs them:

```
cargo test --lib verify_weight_paths_against_upstream -- --ignored
cargo test --lib verify_weight_shapes_against_upstream -- --ignored
```

Result against `OpenMOSS-Team/MOSS-Image-Base` (2026-10-30):

| Component | upstream tensors | requested | unread (all intentional) |
|---|---|---|---|
| transformer | 463 | **462** | `rope.bands` (derived — rebuilt from `rope_base`) |
| text_encoder | 759 | **426** | 333 × `visual.*` (text-only inference skips the vision tower) |
| vae | 251 | **142** | `encoder.*` + `quant_conv.*` (decode-only) + `bn.num_batches_tracked` |

- Zero missing paths, zero spurious requests.
- **All 1,030 requested shapes match the upstream headers** — so the
  Linear orientations, the conv kernel layouts and the per-head norm
  dims are right.
- The "unused" check is the one that catches a **silent** mistake:
  using `linear_no_bias` where the checkpoint carries a bias would
  leave that bias tensor unread and the model quietly wrong — the
  diff would flag it as an unexpected unused tensor.

What this does **not** cover: the forward pass' numerics (does the
image match the PyTorch reference?). That needs the real weights.


## Files

| Path | Role |
|---|---|
| `src/imagegen/mod.rs` | capability trait, DTOs, `ImageGenImpl` |
| `src/imagegen/registry.rs` | registry + lazy default runtime |
| `src/imagegen/moss_image/mod.rs` | runtime: load, prompt encode, sampling loop, PNG |
| `src/imagegen/moss_image/dit.rs` | DiT port |
| `src/imagegen/moss_image/text_encoder.rs` | Qwen3.5 hybrid GDN port |
| `src/imagegen/moss_image/vae.rs` | FLUX.2 decoder port |
| `src/imagegen/moss_image/scheduler.rs` | sigma grid + Euler step |
| `src/imagegen/moss_image/config.rs` | upstream config.json parsing |
| `src/imagegen/moss_image/util.rs` | tanh, chunked attention, seeded noise |

## References

- github.com/OpenMOSS/MOSS-Image (Apache-2.0) — pipeline + ComfyUI plugin
- huggingface.co/OpenMOSS-Team/MOSS-Image-Base — weights + model card
- Technical report: arXiv 2610.23374
