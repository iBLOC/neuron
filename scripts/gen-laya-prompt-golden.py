#!/usr/bin/env python3
"""Generate golden vectors for the Laya prompt grammar.

The reference is the *installed* `laya` package -- `laya.common.build_sequence`,
`render_options`, `serialize_state`, `render_criterion` -- not a transcription of
it. `src/decision/prompt.rs` must reproduce these outputs exactly.

Two golden files come out:

  tests/fixtures/laya_prompt_golden.json    -- cases, with `ids` + `markers`
  tests/fixtures/laya_test_tokenizer.json   -- the tokenizer the cases were made with

The tokenizer is deliberately synthetic and *committed alongside* the vectors so
the Rust replay test is unconditional and hermetic. It is a character-level
ByteLevel tokenizer (`use_regex=False`, empty merges), which gives three
properties the real checkpoints' tokenizers also have and the grammar depends on:

  * " foo" and "foo" tokenize differently -- pins the LEADING SPACE that
    `build_sequence` adds before every option text;
  * token count grows with text length -- exercises the head-budget squeeze and
    the left/right state truncation;
  * ids are stable across machines and library versions.

What it cannot pin is subword merge behaviour, which lives in the tokenizer and
not in the grammar. Pass `--tokenizer <hf-or-local>` to generate vectors against
a real checkpoint tokenizer instead (e.g. `convaiinnovations/laya-multilingual`)
and run the Rust test with `NEURON_TEST_TOKENIZER` pointing at that file.

Usage:
  gen-laya-prompt-golden.py [--tokenizer PATH_OR_HF_ID] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from laya.common import build_sequence, render_options, serialize_state

FIXTURE = "laya_prompt_golden.json"
TOKENIZER = "laya_test_tokenizer.json"

CLS, SEP, MASK, PAD = "[CLS]", "[SEP]", "[MASK]", "[PAD]"


class HfShim:
    """Make a `tokenizers.Tokenizer` look like the HF tokenizer `build_sequence` calls."""

    def __init__(self, tok):
        self._t = tok
        self.mask_token = MASK
        self.mask_token_id = tok.token_to_id(MASK)
        self.cls_token_id = tok.token_to_id(CLS)
        self.sep_token_id = tok.token_to_id(SEP)
        self.pad_token_id = tok.token_to_id(PAD)

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = self._t.encode(text, add_special_tokens=add_special_tokens).ids
        # HF truncation keeps the leading `max_length` tokens -- the same thing
        # `ids[:max_length]` produced before it was passed to the tokenizer.
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}


def synthetic_tokenizer():
    """Character-level ByteLevel tokenizer: one token per byte, space-sensitive."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {t: i for i, t in enumerate([PAD, "[UNK]", CLS, SEP, MASK])}
    for ch in alphabet:
        vocab[ch] = len(vocab)
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tok.decoder = decoders.ByteLevel()
    return tok


CASES = [
    # --- rendering: criteria shapes and option order -------------------------
    {
        "name": "choice_rubric_keeps_json_input_order",
        "state": "The customer wants a refund.",
        "q": {
            "t": "choice",
            "ins": "Which action should the agent take?",
            "crit": {"zebra": None, "apple": "give money back", "mango": 0, "kiwi": False},
        },
    },
    {
        "name": "choice_list_criteria_are_bare_labels",
        "state": "ok",
        "q": {"t": "choice", "ins": "Pick", "crit": ["refund", "cancel", "escalate"]},
    },
    {
        "name": "choice_structured_criterion_renders_as_json",
        "state": "ok",
        "q": {"t": "choice", "ins": "Pick", "crit": {"a": {"weight": 3, "desc": "x"}}},
    },
    {
        "name": "score_levels",
        "state": "The reply was curt.",
        "q": {"t": "score", "ins": "Rate the tone", "crit": ["rude", "neutral", "warm"]},
    },
    {
        "name": "noul_default_labels",
        "state": "The user has verified their email.",
        "q": {"t": "noul", "ins": "The user is verified", "crit": None},
    },
    {
        "name": "noul_criteria_and_mixed_case_keys",
        "state": "ok",
        "q": {
            "t": "noul",
            "ins": "Ship it?",
            "crit": {"True": "the tests pass", "False": "the tests fail"},
        },
    },
    {
        "name": "noul_custom_labels",
        "state": "ok",
        "q": {
            "t": "noul",
            "ins": "Escalate?",
            "crit": None,
            "labels": {"false": "nope", "true": "yep"},
        },
    },
    # --- grammar details ----------------------------------------------------
    {
        "name": "mask_tokens_are_stripped_from_ins_and_options",
        "state": "state with a [MASK] in it",
        "q": {"t": "choice", "ins": "has [MASK] inside", "crit": {"a[MASK]b": None, "c": None}},
    },
    {
        "name": "option_order_permutes_the_markers",
        "state": "ok",
        "q": {"t": "choice", "ins": "Pick", "crit": {"a": None, "b": None, "c": None}},
        "option_order": [2, 0, 1],
    },
    {
        "name": "structured_state_serializes_to_json",
        "state": {"user": "kim", "score": 3, "tags": ["a", "b"]},
        "q": {"t": "choice", "ins": "Pick", "crit": {"a": None, "b": None}},
    },
    {
        "name": "structured_instructions_serialized_by_to_internal",
        "state": "ok",
        "q": {"t": "choice", "ins": {"goal": "ship", "tone": "warm"}, "crit": {"a": None, "b": None}},
    },
    {
        "name": "list_state_truncates_left",
        "state": ["e5", "e4", "e3", "e2", "e1"],
        "q": {"t": "choice", "ins": "Pick", "crit": {"a": None, "b": None}},
        "truncate_left": True,
    },
    {
        "name": "list_state_truncates_right_when_asked",
        "state": ["e5", "e4", "e3", "e2", "e1"],
        "q": {"t": "choice", "ins": "Pick", "crit": {"a": None, "b": None}},
        "truncate_left": False,
    },
    # --- budget paths -------------------------------------------------------
    {
        "name": "long_options_squeeze_the_instruction_budget",
        "state": "s",
        "q": {
            "t": "score",
            "ins": "a long instruction that has to give up budget to the options below it",
            "crit": [
                "level criterion with a very long description " + ("filler " * 20),
                "another long level criterion " + ("padding " * 20),
                "third " + ("more " * 20),
            ],
        },
    },
    {
        "name": "long_state_is_truncated_from_the_left",
        "state": " ".join(f"word{i}" for i in range(400)),
        "q": {"t": "choice", "ins": "Pick", "crit": {"a": None, "b": None}},
        "truncate_left": True,
        "max_len": 96,
        "head_max_len": 48,
    },
    {
        "name": "markers_beyond_max_len_are_dropped",
        "state": " ".join(f"w{i}" for i in range(400)),
        "q": {
            "t": "choice",
            "ins": "Pick one of many",
            "crit": {f"k{i}": None for i in range(12)},
        },
        "max_len": 32,
        "head_max_len": 64,
    },
    {
        "name": "short_sequence_is_not_padded",
        "state": "hi",
        "q": {"t": "noul", "ins": "ok", "crit": None},
    },
]


def normalise(qdef):
    """Mirror `laya.onnx_agent.ONNXAgent._to_internal`.

    `build_sequence` / `render_options` never see the raw question dict; the
    agent normalises it first. `choice` criteria may arrive as a plain list of
    labels and `noul` criteria keys are lowercased.
    """
    q = dict(qdef)
    crit = q.get("crit")
    if q["t"] == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    elif q["t"] == "noul" and isinstance(crit, dict):
        crit = {str(k).lower(): v for k, v in crit.items()}
    q["crit"] = crit
    if not isinstance(q["ins"], str):
        q["ins"] = json.dumps(q["ins"], ensure_ascii=False)
    return q


def record(shim, case):
    state = case["state"]
    q = normalise(dict(case["q"]))
    max_len = case.get("max_len", 512)
    head_max_len = case.get("head_max_len", 192)
    ids, markers = build_sequence(
        shim,
        state,
        q,
        max_len,
        head_max_len,
        option_order=case.get("option_order"),
        truncate_left=case.get("truncate_left", False),
    )
    return {
        "name": case["name"],
        "qtype": q["t"],
        # `q` is what the agent hands build_sequence; `q_raw` is what a caller
        # sends. The Rust replay test renders from `q_raw` so its own
        # normalisation (choice-as-list, case-insensitive noul keys) is covered.
        "q": q,
        "q_raw": case["q"],
        "state": state,
        "option_order": case.get("option_order"),
        "truncate_left": case.get("truncate_left", False),
        "max_len": max_len,
        "head_max_len": head_max_len,
        "rendered_options": render_options(q),
        "state_text": serialize_state(state),
        "input_ids": list(ids),
        "marker_pos": list(markers),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer.json path or HF id; default builds the synthetic one")
    ap.add_argument("--out-dir", default=os.path.join("tests", "fixtures"))
    args = ap.parse_args()

    if args.tokenizer:
        from tokenizers import Tokenizer

        src = args.tokenizer
        if os.path.exists(src):
            tok = Tokenizer.from_file(src)
        else:
            tok = Tokenizer.from_pretrained(src)
        write_tok = None
    else:
        tok = synthetic_tokenizer()
        write_tok = TOKENIZER

    shim = HfShim(tok)
    for probe, expect in (("foo", 3), (" foo", 4)):
        got = len(shim(probe)["input_ids"])
        if got == expect:
            continue
        print(f"FATAL: tokenizer is not space-sensitive ({probe!r} -> {got} tokens, "
              f"expected {expect}); the leading-space behaviour cannot be pinned.", file=sys.stderr)
        return 2

    records = [record(shim, c) for c in CASES]

    os.makedirs(args.out_dir, exist_ok=True)
    out = {
        "generator": "scripts/gen-laya-prompt-golden.py",
        "reference": "laya.common.build_sequence",
        "tokenizer": write_tok or args.tokenizer,
        "cases": records,
    }
    path = os.path.join(args.out_dir, FIXTURE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")

    if write_tok:
        tok.save(os.path.join(args.out_dir, write_tok))

    dropped = sum(1 for r in records if len(r["marker_pos"]) != len(r["rendered_options"]))
    print(f"wrote {path} ({len(records)} cases, {os.path.getsize(path)} bytes)")
    if write_tok:
        print(f"wrote {os.path.join(args.out_dir, write_tok)}")
    print(f"cases with dropped markers: {dropped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
