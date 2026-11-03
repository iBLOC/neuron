#!/usr/bin/env python
"""Generate the EmbeddingGemma 2 parity fixture by running the UPSTREAM
transformers implementation on the REAL checkpoint.

The Rust test (`cargo test --lib egemma2_parity -- --ignored --nocapture`)
compares neuron's vendored candle tower element-wise against the
`reference` tensor this script dumps — the same recipe as
`scripts/gen-moss-parity.py`: nothing upstream is re-typed here, so the
reference is a real reference rather than a second transcription.

Requirements (the released model needs transformers >= 5.18; the system
Python usually ships an older one, so use a dedicated venv — any
transformers >= 5.18 from PyPI works, e.g. 5.22.1):

    python -m venv target/embeddinggemma2-spike/venv-tf
    target/embeddinggemma2-spike/venv-tf/Scripts/python -m pip install \
        torch --index-url https://download.pytorch.org/whl/cpu
    target/embeddinggemma2-spike/venv-tf/Scripts/python -m pip install \
        "transformers>=5.18" safetensors

    # then, from the repo root:
    target/embeddinggemma2-spike/venv-tf/Scripts/python \
        scripts/gen-egemma2-parity.py [MODEL_DIR]

MODEL_DIR defaults to ~/.neuron/models/gemma2/embeddinggemma-2 (the
checkpoint the runtime loads — download google/embeddinggemma-2 there).

The script writes `src/embedding/gemma2/testdata/egemma2_parity.safetensors`
containing `input_ids` [B, T] i64, `attention_mask` [B, T] i64 and
`reference` [B, T, 768] f32 (upstream `last_hidden_state`, fp32, eager
attention). The input deliberately mixes sequence lengths so the
padding-mask path is covered.
"""

import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


def main() -> None:
    default_dir = Path.home() / ".neuron" / "models" / "gemma2" / "embeddinggemma-2"
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else default_dir
    for name in ("config.json", "model.safetensors", "tokenizer.json"):
        if not (model_dir / name).exists():
            raise SystemExit(f"missing {name} under {model_dir} — download the checkpoint first")

    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    if tokenizer.pad_token is None:
        # The checkpoint config carries pad_token_id 0 ("<pad>") but the
        # tokenizer doesn't expose it as the pad token by default.
        tokenizer.pad_token = "<pad>"
    model = AutoModel.from_pretrained(
        str(model_dir),
        dtype=torch.float32,          # fp32 reference — the port also computes fp32 on CPU
        attn_implementation="eager",  # the port mirrors the eager math path
    )
    model.eval()

    # Three fixed texts of different lengths — the batch forces
    # BatchLongest right padding, exercising the bidirectional padding
    # mask and the masked mean pool the Rust side implements.
    texts = [
        "The cat sat on the mat.",
        "A much longer sentence that forces right padding onto the batch sibling, "
        "because BatchLongest pads every member to the longest input in the batch.",
        "Embedding models map text into vectors used for retrieval.",
    ]
    enc = tokenizer(texts, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    print(f"input_ids: {tuple(input_ids.shape)}, pad token id {tokenizer.pad_token_id}")

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    reference = out.last_hidden_state  # [B, T, 768] — norm + projection, pre-pool
    print(f"reference: {tuple(reference.shape)}, finite={torch.isfinite(reference).all().item()}")

    out_dir = Path(__file__).resolve().parent.parent / "src" / "embedding" / "gemma2" / "testdata"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "egemma2_parity.safetensors"
    save_file(
        {
            "input_ids": input_ids.to(torch.int64),
            "attention_mask": attention_mask.to(torch.int64),
            "reference": reference.to(torch.float32).contiguous(),
        },
        str(out_path),
    )
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")

    # Side-car with the human-readable input for future regeneration diffs.
    meta = {
        "texts": texts,
        "model_dir": str(model_dir),
        "transformers_version": __import__("transformers").__version__,
        "dtype": "float32",
        "attn_implementation": "eager",
    }
    (out_dir / "egemma2_parity_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out_dir / 'egemma2_parity_meta.json'}")


if __name__ == "__main__":
    main()
