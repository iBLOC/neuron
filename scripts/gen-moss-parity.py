#!/usr/bin/env python3
"""Generate MOSS-Image parity fixtures from the upstream implementation.

neuron's MOSS-Image port is a candle re-implementation of OpenMOSS's
reference code.  This script produces the *reference* side of a
numerical comparison by running the **upstream implementations
themselves**:

* the Gated DeltaNet chunk kernel comes from the installed
  ``transformers`` package
  (``transformers.models.qwen3_5.modeling_qwen3_5``);
* the diffusion transformer comes from the upstream
  ``pipeline_moss_image.py``, which ships inside the model repo and is
  loaded as a module from its file path.

Both are upstream code — nothing is re-typed here — so the fixtures
are a genuine reference rather than a second transcription.

Two components are covered:

* ``gdn`` — ``torch_chunk_gated_delta_rule``: the most intricate
  piece of the port (chunked scan + in-chunk UT transform + a
  decaying state recurrence).
* ``dit`` — ``MOSSImageTransformer``: the joint-sequence attention,
  the 4-axis RoPE, the shared time modulation and the ``_situ``
  gated FFN in one go.

The inputs are **deterministic** (synthesised from a fixed
trigonometric sequence, not drawn from an RNG), so regenerating the
fixtures reproduces them bit for bit.

Usage:

    python scripts/gen-moss-parity.py
    python scripts/gen-moss-parity.py --repo OpenMOSS-Team/MOSS-Image-Base

The fixtures land in ``src/imagegen/moss_image/testdata/`` and are
checked in, so the Rust test needs neither Python nor torch.
"""

from __future__ import annotations

import argparse
import importlib.util
import ipaddress
import json
import os
import re
import socket
import tempfile
import urllib.error
import urllib.parse
import urllib.request

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "src", "imagegen", "moss_image", "testdata")
HF_ENDPOINT = "https://huggingface.co"

# The upstream file this script reads from the model repo.
PIPELINE_FILE = "pipeline_moss_image.py"

# `owner/name` with slug characters only.
REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")

# RFC 2544 benchmarking range (198.18.0.0/15). Python's `is_private`
# counts it as private, but transparent proxies (Clash TUN and
# friends) route ordinary hosts through it — it is not an
# internal-network target, so it stays reachable while loopback,
# RFC1918, link-local (the cloud metadata address lives there) and
# multicast stay blocked.
BENCHMARK_NET = ipaddress.ip_network("198.18.0.0/15")


def out_path(*parts: str) -> str:
    """Join a path inside the fixture directory, refusing to escape it."""
    root = os.path.realpath(OUT_DIR)
    target = os.path.realpath(os.path.join(root, *parts))
    if os.path.commonpath([root, target]) != root:
        raise SystemExit(f"refusing to write outside {root}: {target}")
    return target


def validate_endpoint(endpoint: str) -> str:
    """Accept only an https endpoint whose host resolves to a public IP.

    The endpoint is a request target: a private or loopback address
    would let this script read local services.
    """
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme != "https":
        raise SystemExit(f"endpoint must use https, got {parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise SystemExit("endpoint has no host")
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise SystemExit(f"cannot resolve endpoint host {host!r}: {exc}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip in BENCHMARK_NET:
            continue
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise SystemExit(
                f"endpoint host {host!r} resolves to the non-public address {ip}"
            )
    return endpoint.rstrip("/")


def validate_repo(repo: str) -> str:
    if not REPO_ID_RE.match(repo):
        raise SystemExit(f"invalid repo id: {repo!r} — expected 'owner/name'")
    return repo


def fetch_file(endpoint: str, repo: str, path: str) -> bytes:
    if not REPO_ID_RE.match(repo):
        raise SystemExit(f"invalid repo id: {repo!r}")
    url = f"{endpoint}/{repo}/resolve/main/{path}"
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise SystemExit("only https is allowed")
    req = urllib.request.Request(url, headers={"User-Agent": "neuron-parity"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"HTTP {exc.code} for {url}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"network error for {url}: {exc.reason}") from exc


def deterministic(shape, base: float, amplitude: float = 1.0) -> torch.Tensor:
    """A deterministic, well-spread tensor of the requested shape.

    Built from two incommensurate sinusoids over the flattened index,
    so the values are reproducible across runs and machines without
    depending on any RNG.
    """
    n = 1
    for dim in shape:
        n *= dim
    idx = torch.arange(n, dtype=torch.float32)
    values = amplitude * (
        0.6 * torch.sin(idx * base + 0.37) + 0.4 * torch.cos(idx * base * 0.618 + 1.13)
    )
    return values.reshape(tuple(shape))


def load_upstream_pipeline(endpoint: str, repo: str):
    """Import the upstream ``pipeline_moss_image.py`` as a module.

    The file lives in the model repo (it is the ``trust_remote_code``
    payload diffusers loads at inference time), so it is fetched and
    imported from its file path with the standard import machinery.
    The file is written to an OS-generated temporary path, so no path
    is ever built from downloaded content.
    """
    blob = fetch_file(endpoint, repo, PIPELINE_FILE)
    handle, path = tempfile.mkstemp(prefix="upstream_pipeline_", suffix=".py")
    try:
        with os.fdopen(handle, "wb") as sink:
            sink.write(blob)
        spec = importlib.util.spec_from_file_location(
            "upstream_pipeline_moss_image", path
        )
        if spec is None or spec.loader is None:
            raise SystemExit("cannot load the upstream pipeline module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def save_fixture(path: str, tensors: dict[str, torch.Tensor], meta: dict) -> None:
    from safetensors.torch import save_file

    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file(
        {k: v.contiguous() for k, v in tensors.items()},
        path,
        metadata={k: (v if isinstance(v, str) else json.dumps(v)) for k, v in meta.items()},
    )
    print(f"wrote {path} ({os.path.getsize(path)} bytes)")


def gen_gdn_fixture(repo: str) -> None:
    # The upstream kernel, imported from the installed transformers.
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        torch_chunk_gated_delta_rule,
    )

    # [B, S, H, D] — S spans a chunk boundary (64) plus a tail, so the
    # padding path is exercised.
    b, s, h, dk, dv = 2, 100, 4, 16, 16
    query = deterministic((b, s, h, dk), 0.37)
    key = deterministic((b, s, h, dk), 0.71)
    value = deterministic((b, s, h, dv), 1.13)
    beta = deterministic((b, s, h), 0.29) * 0.45 + 0.5
    # g is a log-decay (≤ 0), built the way the model builds it.
    a = deterministic((b, s, h), 0.53)
    dt_bias = deterministic((h,), 0.17)
    a_log = deterministic((h,), 0.23) * 0.5 + 0.5
    g = -a_log.exp() * torch.nn.functional.softplus(a + dt_bias)

    # The upstream kernel returns (output, final_state); the port's
    # Rust kernel returns the output alone.
    out, _final_state = torch_chunk_gated_delta_rule(
        query, key, value, g=g, beta=beta, use_qk_l2norm_in_kernel=True
    )
    tensors = {
        "query": query,
        "key": key,
        "value": value,
        "g": g,
        "beta": beta,
        "output": out,
    }
    meta = {
        "component": "gdn",
        "reference": "transformers.models.qwen3_5.modeling_qwen3_5::torch_chunk_gated_delta_rule",
        "repo": repo,
        "shape_b": str(b),
        "shape_s": str(s),
        "shape_h": str(h),
        "shape_dk": str(dk),
        "shape_dv": str(dv),
    }
    save_fixture(out_path("moss_gdn_parity.safetensors"), tensors, meta)


def gen_dit_fixture(endpoint: str, repo: str) -> None:
    module = load_upstream_pipeline(endpoint, repo)
    cls = module.MOSSImageTransformer

    # The tiny structural config — identical to the one the Rust test
    # builds, so the two sides run the same module tree.
    cfg = dict(
        dim=64,
        ffn_dim=128,
        num_heads=4,
        num_layers=2,
        text_dim=96,
        freq_dim=32,
        latent_channels=8,
        rope_base=10000.0,
    )
    model = cls(**cfg).eval()

    # Deterministic weights (the upstream default init draws from an
    # RNG; replacing it keeps the fixture reproducible).
    with torch.no_grad():
        for index, (_, parameter) in enumerate(model.named_parameters()):
            parameter.copy_(
                deterministic(tuple(parameter.shape), 0.019 * (index + 1), amplitude=0.35)
            )

    latents = deterministic((1, cfg["latent_channels"], 8, 8), 0.41)
    text = deterministic((1, 12, cfg["text_dim"]), 0.83)
    mask = torch.ones(1, 12, dtype=torch.bool)
    timestep = 500.0

    with torch.no_grad():
        out = model(latents, timestep, text, mask).sample
        # Also exercise the CFG negative-branch block skipping.
        out_skipped = model(latents, timestep, text, mask, skip_layers=[0]).sample

    tensors = {
        "latents": latents,
        "text": text,
        "output": out,
        "output_skip0": out_skipped,
    }
    for key, value in model.state_dict().items():
        tensors[f"weight::{key}"] = value.contiguous()

    meta = {
        "component": "dit",
        "reference": f"{repo} :: {PIPELINE_FILE} :: MOSSImageTransformer",
        "config": json.dumps(cfg),
        "timestep": str(timestep),
    }
    save_fixture(out_path("moss_dit_parity.safetensors"), tensors, meta)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default="OpenMOSS-Team/MOSS-Image-Base",
        help="upstream HF repo id (default: %(default)s)",
    )
    parser.add_argument(
        "--endpoint",
        default=HF_ENDPOINT,
        help="HuggingFace-compatible endpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--only",
        choices=["gdn", "dit"],
        help="generate just one fixture",
    )
    args = parser.parse_args()

    endpoint = validate_endpoint(args.endpoint)
    repo = validate_repo(args.repo)
    print(f"upstream repo: {repo}", flush=True)
    print(f"endpoint: {endpoint}", flush=True)

    if args.only in (None, "gdn"):
        print("== GDN kernel parity fixture ==", flush=True)
        gen_gdn_fixture(repo)
    if args.only in (None, "dit"):
        print("== DiT parity fixture ==", flush=True)
        gen_dit_fixture(endpoint, repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
