#!/usr/bin/env python3
"""Generate golden vectors for the Laya decision capability.

Produces a JSON fixture that `tests/decision_parity.rs` compares against. Two
independent test groups, because they validate different halves of the
pipeline and can fail separately:

  numerics     Runs the Laya ONNX under onnxruntime and records the raw
               `logits` / `act_logits`. The Rust side runs the prepped graph
               under tract and must match. This covers both "did the graph prep
               change any numbers" and "does tract execute the graph correctly"
               in one comparison.

               The graph is rewritten in memory before it is run: every
               `MatMulInteger` becomes an explicit integer Cast/Sub/MatMul
               decomposition. onnxruntime's `MatMulInteger` CPU kernel does NOT
               compute the exact integer product -- measured 5-25% relative
               error against exact int64 arithmetic for reduction lengths
               8..16384 (onnxruntime 1.30.0; only trivial shapes are exact) --
               so recording its output would bake that arithmetic error into
               the golden vector. tract's `MatMulInteger` IS exact, so without
               the rewrite the fixture and the Rust side disagree purely
               because onnxruntime was wrong. The decomposition reuses the
               original output node name and the INT32 dtype, so every
               downstream consumer is untouched.

  postprocess  Runs a Python reference of the host-side temperature calibration
               (the part that is NOT in any ONNX graph) and records the expected
               probabilities. The Rust side feeds the same raw logits through
               its own postprocessing and must match. The reference here is the
               formula from `laya/common.py` + `laya/onnx_agent.py`:

                   scale   = temperature_by_options.get(temp_bucket(qt, k),
                                                        temperature[qt])
                   z       = logits[:k] / scale
                   p       = exp(z - max(z)); p /= sum(p)
                   act     = softmax(act_logits)

               with temp_bucket(qt, k) -> f"{qtype}:{size}" where size is "2" for
               k <= 2, "3-5" for k <= 5, "6-10" for k <= 10, else "11+"; and
               clamp_temperature clamping to [0.5, 5.0] with NaN/inf -> 1.0 and
               unparseable -> 1.0. Confirmed against tests/test_temperature_loading.py
               (asserts [0.1, 10, -1] clamps to [0.5, 5.0, 0.5]).

Usage:
  scripts/gen-laya-golden.py MODEL.onnx OUT.json [options]

Options:
  --seq N          Sequence length for the generated numerics inputs (default 64)
  --markers N      Option count for the generated numerics inputs (default 4)
  --cases N        Number of numerics cases (default 3)
  --seed N         RNG seed (default 20260924)
  --tolerance F    Relative tolerance recorded in the fixture (default 0.25).
                   This is NOT a precision claim about the model: it is the
                   measured tract-vs-onnxruntime floor (worst 1.65e-1) with
                   headroom. Quantization is a step function, so the few-ULP
                   gaps tract and ORT have on `Div`/`ReduceMean`/`Erf` land as
                   whole-code differences and come out percent-level. See
                   tests/decision_parity.rs for the measurements.
  --raw-gemm       Skip the MatMulInteger rewrite and record onnxruntime's
                   lossy integer GEMM as-is. Diagnostic only -- the resulting
                   fixture will NOT match tract.

The numerics inputs are deterministic pseudo-random token ids, not real prompts.
That is fine for parity -- the point is to exercise the graph with varied
activations and compare exact numbers -- but these are NOT a quality benchmark.

Requires: onnxruntime, numpy, onnx.

Exit codes:
  0   success
  2   bad arguments / missing dependency
  3   onnxruntime refused to run the model
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

try:
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper
except ImportError:
    print("error: onnx + onnxruntime are required (pip install onnx onnxruntime)",
          file=sys.stderr)
    sys.exit(2)


# ---- exact integer GEMM ---------------------------------------------------
# onnxruntime's `MatMulInteger` CPU kernel does not compute the integer product
# it is defined to compute. Measured against exact int64 arithmetic on
# `uint8[1,N] @ int8[N,K]` with a non-zero activation zero point:
#
#     N      K     ORT rel. error
#     2      4     0            (only trivial shapes are exact)
#     8     64     1.2e-01
#    64      4     2.5e-01
#  1024     64     1.4e-01
# 16384     64     1.1e-01
#
# The zero-point-free form is wrong too (766226 vs an exact 824128), so the bug
# is in the integer product itself and not in the zero-point correction. tract's
# `MatMulInteger` is exact, and onnxruntime's plain integer `MatMul` is exact at
# every shape tested. Rewriting `MatMulInteger` into that exact form therefore
# gives a graph both engines evaluate identically -- which is what a golden
# vector has to be measured on.

def exact_integer_gemm(model: "onnx.ModelProto") -> tuple["onnx.ModelProto", int]:
    """Replace every `MatMulInteger` with an exact integer decomposition.

        MatMulInteger(A, B, a_zp, b_zp) -> int32

    becomes

        D  = Sub(Cast(A, int32),  Cast(a_zp, int32))
        Bc = Sub(Cast(B, int32),  Cast(b_zp, int32))
        OUT = MatMul(D, Bc)                       -> int32

    The output node name and the INT32 dtype are preserved, so every downstream
    consumer sees the same tensor it saw before and needs no change. Missing
    zero points are treated as zero.
    """
    graph = model.graph
    names = {o.name for o in graph.output} | {i.name for i in graph.input}
    for init in graph.initializer:
        names.add(init.name)
    for node in graph.node:
        names.update(n for n in list(node.input) + list(node.output) if n)

    counter = [0]

    def fresh(base: str) -> str:
        counter[0] += 1
        stem = "".join(c if c.isalnum() or c == "_" else "_" for c in base) or "mmi"
        cand = f"exact_{stem}_{counter[0]}"
        while cand in names:
            counter[0] += 1
            cand = f"exact_{stem}_{counter[0]}"
        names.add(cand)
        return cand

    zero_i32 = None
    zero_i64 = None

    def zero_for(dtype: int) -> str:
        """A scalar zero initializer with the given ONNX element type."""
        nonlocal zero_i32, zero_i64
        if dtype == TensorProto.INT32:
            if zero_i32 is None:
                zero_i32 = fresh("zero_i32")
                graph.initializer.append(
                    numpy_helper.from_array(np.array(0, dtype=np.int32), zero_i32)
                )
            return zero_i32
        if zero_i64 is None:
            zero_i64 = fresh("zero_i64")
            graph.initializer.append(
                numpy_helper.from_array(np.array(0, dtype=np.int64), zero_i64)
            )
        return zero_i64

    def dtype_of(name: str) -> int:
        for init in graph.initializer:
            if init.name == name:
                return init.data_type
        raise SystemExit(
            f"error: cannot determine the element type of {name!r}; "
            "only constant weights and graph inputs are supported"
        )

    out_nodes = []
    n_rewritten = 0
    for node in graph.node:
        if node.op_type != "MatMulInteger":
            out_nodes.append(node)
            continue

        a = node.input[0]
        b = node.input[1]
        a_zp = node.input[2] if len(node.input) > 2 and node.input[2] else ""
        b_zp = node.input[3] if len(node.input) > 3 and node.input[3] else ""
        if len(node.output) != 1:
            raise SystemExit(f"error: {node.name or 'MatMulInteger'} has "
                             f"{len(node.output)} outputs")
        out = node.output[0]

        # D = A - a_zp, Bc = B - b_zp, both in int32, then one integer MatMul.
        a32, za32, d32 = fresh("a"), fresh("za"), fresh("d")
        b32, zb32, bc32 = fresh("b"), fresh("zb"), fresh("bc")

        a_zp_in = a_zp or zero_for(dtype_of(a))
        b_zp_in = b_zp or zero_for(dtype_of(b))

        out_nodes.extend([
            helper.make_node("Cast", [a], [a32], to=TensorProto.INT32),
            helper.make_node("Cast", [a_zp_in], [za32], to=TensorProto.INT32),
            helper.make_node("Sub", [a32, za32], [d32]),
            helper.make_node("Cast", [b], [b32], to=TensorProto.INT32),
            helper.make_node("Cast", [b_zp_in], [zb32], to=TensorProto.INT32),
            helper.make_node("Sub", [b32, zb32], [bc32]),
            helper.make_node("MatMul", [d32, bc32], [out]),
        ])
        n_rewritten += 1

    if n_rewritten:
        del graph.node[:]
        graph.node.extend(out_nodes)
    return model, n_rewritten

# ---- reference implementation of the host-side calibration ----------------
# Mirrors laya/common.py + laya/onnx_agent.py. Kept deliberately small and
# dependency-free so it can be read against the upstream source.

TEMP_MIN = 0.5
TEMP_MAX = 5.0
QTYPES = {"choice": 0, "score": 1, "noul": 2}


def clamp_temperature(t) -> float:
    try:
        v = float(t)
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(v):
        return 1.0
    return float(min(max(v, TEMP_MIN), TEMP_MAX))


def temp_bucket(qtype: str, k: int) -> str:
    if k <= 2:
        size = "2"
    elif k <= 5:
        size = "3-5"
    elif k <= 10:
        size = "6-10"
    else:
        size = "11+"
    return f"{qtype}:{size}"


def reference_probs(logits, qtype: str, option_count: int,
                    temperature_raw, temperature_by_options_raw):
    """Return (scale, probs) for one question."""
    qt = QTYPES[qtype]
    temperature = [clamp_temperature(t) for t in temperature_raw]
    by_options = {
        k_: clamp_temperature(v) for k_, v in temperature_by_options_raw.items()
    }
    scale = by_options.get(temp_bucket(qtype, option_count), temperature[qt])
    z = np.asarray(logits[:option_count], dtype=np.float64) / scale
    e = np.exp(z - z.max())
    return scale, (e / e.sum())


def reference_act_probs(act_logits):
    a = np.asarray(act_logits, dtype=np.float64)
    e = np.exp(a - a.max())
    return e / e.sum()


# ---- numerics input synthesis --------------------------------------------

ORT_TO_NP = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(bool)": np.bool_,
}


def declared_shape(m, fallback):
    """The model's declared shape if it is fully concrete, else `fallback`.

    ONNX exports declare these dims symbolically (`['batch', 'sequence']`), so a
    naive truthiness check treats `[None, None]` as a usable shape.
    """
    dims = list(m.shape)
    if dims and all(isinstance(d, int) and d > 0 for d in dims):
        return tuple(dims)
    return tuple(fallback)


def synth_inputs(meta, rng, seq, markers, vocab_size):
    """Build one deterministic, structurally valid input set for the graph."""
    feed = {}
    for m in meta:
        name, typ = m.name, m.type
        np_dt = ORT_TO_NP[typ]

        if typ == "tensor(bool)":
            # marker_mask: first `markers` true, rest false
            arr = np.zeros(declared_shape(m, (1, markers)), dtype=np.bool_)
            if arr.ndim >= 2:
                arr[..., : min(markers, arr.shape[-1])] = True
            feed[name] = arr
        elif name in ("marker_pos", "marker_positions"):
            # strictly increasing positions inside the sequence
            shape = declared_shape(m, (1, markers))
            n = shape[-1]
            lo = 1 if seq > 2 else 0
            pos = np.sort(rng.choice(np.arange(lo, max(seq - 1, lo + 1)), size=n, replace=False))
            feed[name] = pos.reshape(shape).astype(np_dt)
        elif name == "qtype":
            feed[name] = np.asarray([rng.integers(0, 3)], dtype=np_dt)
        elif name == "attention_mask":
            feed[name] = np.ones(declared_shape(m, (1, seq)), dtype=np_dt)
        elif name in ("input_ids", "hidden", "input", "token_ids"):
            shape = declared_shape(m, (1, seq) if np_dt == np.int64 else (1, seq, 1024))
            if np_dt in (np.float32, np.float16, np.float64):
                arr = rng.normal(0.0, 0.5, size=shape)
            else:
                arr = rng.integers(1, max(vocab_size, 2), size=shape)
            feed[name] = arr.astype(np_dt)
        else:
            raise SystemExit(f"error: do not know how to synthesise input {name!r} ({typ})")
    return feed


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("model")
    ap.add_argument("output")
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--markers", type=int, default=4)
    ap.add_argument("--cases", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--tolerance", type=float, default=0.25)
    ap.add_argument("--raw-gemm", action="store_true",
                    help="keep onnxruntime's lossy MatMulInteger (diagnostic)")
    ap.add_argument("--vocab-size", type=int, default=256000)
    args = ap.parse_args()

    try:
        model = onnx.load(args.model)
    except Exception as exc:
        print(f"error: onnx could not load {args.model}: {exc}", file=sys.stderr)
        return 3

    gemm = "onnxruntime-matmulinteger"
    if not args.raw_gemm:
        model, n_rewritten = exact_integer_gemm(model)
        if n_rewritten:
            gemm = f"exact-decomposition ({n_rewritten} MatMulInteger)"
        print(f"rewrote {n_rewritten} MatMulInteger node(s) into exact integer MatMul")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    try:
        sess = ort.InferenceSession(
            model.SerializeToString(), opts, providers=["CPUExecutionProvider"]
        )
    except Exception as exc:
        print(f"error: onnxruntime could not load {args.model}: {exc}", file=sys.stderr)
        return 3

    in_meta = sess.get_inputs()
    out_names = [o.name for o in sess.get_outputs()]
    print(f"model: {args.model}\n  inputs : {[(m.name, m.type, m.shape) for m in in_meta]}")
    print(f"  outputs: {out_names}")

    rng = np.random.default_rng(args.seed)

    # ---- group 1: numerics (ORT on the ORIGINAL graph) --------------------
    numerics = []
    for c in range(args.cases):
        feed = synth_inputs(in_meta, rng, args.seq, args.markers, args.vocab_size)
        outs = sess.run(None, {k: v for k, v in feed.items()})
        case = {
            "name": f"case_{c:03d}",
            "seq": args.seq,
            "markers": args.markers,
            "inputs": {k: np.asarray(v).tolist() for k, v in feed.items()},
            "outputs": {n: np.asarray(o).tolist() for n, o in zip(out_names, outs)},
        }
        numerics.append(case)
        print(f"  numerics[{c}] outputs: " +
              ", ".join(f"{n}{list(np.asarray(o).shape)}" for n, o in zip(out_names, outs)))

    # ---- group 2: postprocess (Python reference of the calibration) -------
    # Temperature tables are synthetic on purpose: the test validates the
    # bucket/clamp/softmax FORMULA. Real fitted values are model data loaded at
    # runtime, and the shipped `choice:11+` value (0.1006) is clamped to 0.5
    # anyway, so using fitted numbers here would test nothing extra.
    temperature_raw = [0.8, 1.35, 2.25]                       # choice, score, noul
    temperature_by_options_raw = {
        "choice:2": 0.1006,   # shipped value: clamps UP to 0.5
        "choice:3-5": 1.75,
        "choice:11+": 9.0,    # clamps DOWN to 5.0
        "noul:2": 0.2,        # clamps UP to 0.5
    }

    postprocess = []
    specs = [
        ("choice_k2_hits_bucket_and_clamps_up", "choice", 2),
        ("choice_k3_hits_3-5_bucket", "choice", 3),
        ("choice_k5_hits_3-5_bucket", "choice", 5),
        ("choice_k6_hits_6-10_fallback", "choice", 6),
        ("choice_k11_hits_11+_and_clamps_down", "choice", 11),
        ("choice_k4_missing_bucket_falls_back", "choice", 4),
        ("score_k3_hits_3-5_bucket", "score", 3),
        ("score_k4_missing_bucket_falls_back", "score", 4),
        ("noul_k2_hits_bucket_and_clamps_up", "noul", 2),
        ("k1_single_option", "choice", 1),
    ]
    for name, qtype, k in specs:
        logits = rng.normal(0.0, 3.0, size=k).astype(np.float64)
        act_logits = rng.normal(0.0, 2.0, size=2).astype(np.float64)
        scale, probs = reference_probs(
            logits, qtype, k, temperature_raw, temperature_by_options_raw
        )
        act = reference_act_probs(act_logits)
        postprocess.append({
            "name": name,
            "qtype": qtype,
            "qtype_index": QTYPES[qtype],
            "option_count": k,
            "logits": logits.tolist(),
            "act_logits": act_logits.tolist(),
            "temperature_raw": temperature_raw,
            "temperature_by_options_raw": temperature_by_options_raw,
            "expected_scale": scale,
            "expected_probs": probs.tolist(),
            "expected_act_probs": act.tolist(),
        })
        print(f"  postprocess[{name}] k={k} scale={scale:.4f} probs={np.round(probs, 5).tolist()}")

    fixture = {
        "schema": 1,
        "kind": "laya-decision-golden",
        "generated_by": "scripts/gen-laya-golden.py",
        "source_model": args.model,
        "reference_gemm": gemm,
        "seed": args.seed,
        "tolerance": args.tolerance,
        "numerics": numerics,
        "postprocess": postprocess,
    }
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(fixture, fh, indent=1)
    print(f"\n{args.output}: {len(numerics)} numerics case(s), "
          f"{len(postprocess)} postprocess case(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
