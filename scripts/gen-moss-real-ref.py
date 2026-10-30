#!/usr/bin/env python3
"""Produce reference outputs from the REAL MOSS-Image checkpoint.

The parity fixtures in ``src/imagegen/moss_image/testdata/`` use a
tiny synthetic model: they verify the *formulas*, but not the real
weights.  This script closes part of that gap by running the
**upstream implementations on the real checkpoint** and dumping their
outputs, so neuron's port can be compared against them on the same
weights.

Three components, each loaded on its own so it fits in a 24 GB card
(the whole pipeline at once does not):

* **text encoder** — ``transformers``' ``Qwen3_5Model`` on the
  bundled ``text_encoder/``; the reference is the
  ``last_hidden_state`` of a real prompt.
* **DiT** — the upstream ``MOSSImageTransformer`` from the repo's own
  ``pipeline_moss_image.py``, fed the text encoder's real hidden
  states.
* **VAE** — ``diffusers``' ``AutoencoderKLFlux2`` on the bundled
  ``vae/``.

The script also dumps the prompt string the upstream chat template
renders, so neuron's hard-coded template can be checked against it.

Usage:

    python scripts/gen-moss-real-ref.py \\
        --model-dir ~/.neuron/models/imagegen/moss-image \\
        --out target/moss-real \\
        --only vae

The outputs land in ``--out``; the Rust tests read them from there
(see ``src/imagegen/moss_image/real_weights.rs``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os

import torch

# The upstream file that ships in the model repo.
PIPELINE_FILE = "pipeline_moss_image.py"

# The prompt used for the text-encoder reference — Chinese + English,
# because MOSS-Image's headline feature is Chinese text rendering.
PROMPT = "一只戴着红色围巾的猫坐在窗台上，窗外是雪景。 A cat wearing a red scarf."


def load_upstream_pipeline(model_dir: str):
    """Import the upstream ``pipeline_moss_image.py`` from the model dir."""
    path = os.path.join(model_dir, PIPELINE_FILE)
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found — is the model dir complete?")
    spec = importlib.util.spec_from_file_location("upstream_pipeline_moss_image", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path: str, tensors: dict, meta: dict) -> None:
    from safetensors.torch import save_file

    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file(
        {k: v.contiguous() for k, v in tensors.items()},
        path,
        metadata={k: (v if isinstance(v, str) else json.dumps(v)) for k, v in meta.items()},
    )
    print(f"wrote {path}")


def deterministic(shape, base: float, amplitude: float = 1.0) -> torch.Tensor:
    """A deterministic, well-spread tensor of the requested shape."""
    n = 1
    for dim in shape:
        n *= dim
    idx = torch.arange(n, dtype=torch.float32)
    values = amplitude * (
        0.6 * torch.sin(idx * base + 0.37) + 0.4 * torch.cos(idx * base * 0.618 + 1.13)
    )
    return values.reshape(tuple(shape))


def gen_prompt_reference(model_dir: str, out_dir: str) -> None:
    """The upstream chat template's rendering of the reference prompt.

    Only the tokenizer is needed, so this works without any weights —
    it is what neuron's hard-coded template is checked against.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    print(f"rendered prompt: {rendered!r}", flush=True)
    token_ids = tokenizer(rendered, return_tensors="pt")["input_ids"][0]
    print(f"prompt tokens: {token_ids.shape[0]}", flush=True)
    save(
        os.path.join(out_dir, "prompt_ref.safetensors"),
        {"token_ids": token_ids.to(torch.int32)},
        {
            "component": "prompt",
            "reference": "transformers.AutoTokenizer.apply_chat_template",
            "prompt": PROMPT,
            "rendered_prompt": rendered,
        },
    )


def gen_vae(model_dir: str, out_dir: str, device: str) -> None:
    """The FLUX.2 decoder on the real checkpoint."""
    from diffusers import AutoencoderKLFlux2

    vae = (
        AutoencoderKLFlux2.from_pretrained(
            os.path.join(model_dir, "vae"), torch_dtype=torch.float32
        )
        .to(device)
        .eval()
    )
    # The MOSS latent layout: 128 channels, BN de-normalization, then a
    # 2x2 pixel unshuffle into the VAE's 32-channel latent.
    packed = deterministic((1, 128, 8, 8), 0.19)
    packed = packed.to(device)
    with torch.no_grad():
        z = packed
        bn = vae.bn
        eps = vae.config.batch_norm_eps
        z = (
            z * (bn.running_var + eps).sqrt().to(z).view(1, -1, 1, 1)
            + bn.running_mean.to(z).view(1, -1, 1, 1)
        )
        b, c, hh, ww = z.shape
        z = (
            z.reshape(b, c // 4, 2, 2, hh, ww)
            .permute(0, 1, 4, 2, 5, 3)
            .reshape(b, c // 4, 2 * hh, 2 * ww)
        )
        decoded = vae.decode(z, return_dict=False)[0].clamp(-1, 1)
    decoded_cpu = decoded.float().cpu()
    print(f"vae output: {tuple(decoded_cpu.shape)}", flush=True)
    save(
        os.path.join(out_dir, "vae_ref.safetensors"),
        {
            "packed_latent": packed.float().cpu(),
            "decoded": decoded_cpu,
        },
        {
            "component": "vae",
            "reference": "diffusers.AutoencoderKLFlux2",
            "dtype": "float32",
            "device": device,
        },
    )


def gen_text_encoder(model_dir: str, out_dir: str, device: str, dtype: torch.dtype) -> None:
    """The Qwen3.5 hybrid text encoder on the real checkpoint."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    print(f"rendered prompt: {rendered!r}", flush=True)
    token_ids = tokenizer(rendered, return_tensors="pt")["input_ids"][0]
    print(f"prompt tokens: {token_ids.shape[0]}", flush=True)

    print("loading the text encoder...", flush=True)
    encoder = (
        AutoModel.from_pretrained(
            os.path.join(model_dir, "text_encoder"), dtype=dtype
        )
        .to(device)
        .eval()
    )
    ids = token_ids.unsqueeze(0).to(device)
    with torch.no_grad():
        hidden = encoder(ids, use_cache=False, return_dict=True).last_hidden_state
    hidden_cpu = hidden.float().cpu()
    print(f"hidden states: {tuple(hidden_cpu.shape)}", flush=True)
    save(
        os.path.join(out_dir, "text_encoder_ref.safetensors"),
        {
            "hidden": hidden_cpu,
            "token_ids": token_ids.to(torch.int32),
        },
        {
            "component": "text_encoder",
            "reference": "transformers.models.qwen3_5.Qwen3_5Model",
            "prompt": PROMPT,
            "rendered_prompt": rendered,
            "dtype": str(dtype),
            "device": device,
        },
    )


def gen_dit(model_dir: str, out_dir: str, device: str, dtype: torch.dtype) -> None:
    """The DiT on the real checkpoint, fed the text encoder's outputs."""
    module = load_upstream_pipeline(model_dir)
    print("loading the transformer...", flush=True)
    dit = (
        module.MOSSImageTransformer.from_pretrained(
            os.path.join(model_dir, "transformer"), torch_dtype=dtype
        )
        .to(device)
        .eval()
    )
    # A deterministic latent (no RNG: the fixture must be reproducible).
    h = w = 32
    latent_channels = dit.config.latent_channels
    latents = deterministic((1, latent_channels, h, w), 0.37).to(
        device=device, dtype=dtype
    )
    # The text encoder's real hidden states (the same file the
    # text-encoder reference writes).
    ref_path = os.path.join(out_dir, "text_encoder_ref.safetensors")
    if not os.path.exists(ref_path):
        raise SystemExit(f"{ref_path} is missing — run the text encoder first")
    from safetensors.torch import load_file

    hidden = load_file(ref_path)["hidden"].to(device=device, dtype=dtype)
    mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=device)
    timestep = 500.0
    with torch.no_grad():
        out = dit(latents, timestep, hidden, mask).sample
    out_cpu = out.float().cpu()
    print(f"dit output: {tuple(out_cpu.shape)}", flush=True)
    save(
        os.path.join(out_dir, "dit_ref.safetensors"),
        {
            "latents": latents.float().cpu(),
            "encoder_hidden_states": hidden.float().cpu(),
            "output": out_cpu,
        },
        {
            "component": "dit",
            "reference": f"{PIPELINE_FILE} :: MOSSImageTransformer",
            "timestep": str(timestep),
            "dtype": str(dtype),
            "device": device,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default=os.path.expanduser("~/.neuron/models/imagegen/moss-image"),
        help="the downloaded MOSS-Image directory (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        default="target/moss-real",
        help="where the reference outputs go (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device (default: %(default)s)",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="compute dtype for the reference (default: %(default)s)",
    )
    parser.add_argument(
        "--only",
        choices=["prompt", "vae", "text_encoder", "dit"],
        help="generate just one reference",
    )
    args = parser.parse_args()

    model_dir = os.path.expanduser(args.model_dir)
    if not os.path.isdir(model_dir):
        raise SystemExit(f"model dir not found: {model_dir}")
    dtype = getattr(torch, args.dtype)
    device = args.device
    print(f"model dir: {model_dir}", flush=True)
    print(f"device: {device}, dtype: {args.dtype}", flush=True)

    if args.only in (None, "prompt"):
        print("== prompt reference ==", flush=True)
        gen_prompt_reference(model_dir, args.out)
    if args.only in (None, "vae"):
        print("== VAE reference ==", flush=True)
        gen_vae(model_dir, args.out, device)
    if args.only in (None, "text_encoder"):
        print("== text encoder reference ==", flush=True)
        gen_text_encoder(model_dir, args.out, device, dtype)
    if args.only in (None, "dit"):
        print("== DiT reference ==", flush=True)
        gen_dit(model_dir, args.out, device, dtype)

    print()
    print("reference outputs written to", os.path.abspath(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
