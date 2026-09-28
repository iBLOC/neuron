#!/usr/bin/env python3
"""Prove the DynamicQuantizeLinear decomposition matches onnxruntime exactly.

`prep-laya-onnx.py` step 4 rewrites `DynamicQuantizeLinear` into `Round`-based
primitives so that tract's non-conformant rounding never runs. That is only
safe if the rewrite reproduces onnxruntime bit-for-bit, which this script
establishes: it builds a graph of `DynamicQuantizeLinear` nodes over adversarial
and random inputs, records onnxruntime's answer, applies the rewrite, and
records the answer again. The two must be identical element-for-element.

The inputs deliberately hit every case the formula is sensitive to:
  * exact `.5` ties in `x / scale`, with odd and even zero points
  * all-nonnegative tensors, where the zero point goes negative and saturates
  * zero-point arguments landing exactly on `.5`
  * degenerate ranges (constant tensors) and subnormal-scale ranges
  * random tensors at several ranks and sizes

Usage:
  scripts/verify-laya-dql.py [--seed N] [--random-cases N] [--dump OUT.onnx]

Exit codes:
  0   bit-exact on every case
  2   bad arguments
  3   a mismatch -- the decomposition is wrong and must not be used
"""

from __future__ import annotations

import argparse
import importlib.util
import os
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


HERE = os.path.dirname(os.path.abspath(__file__))
PREP = os.path.join(HERE, "prep-laya-onnx.py")


def load_prep():
    spec = importlib.util.spec_from_file_location("prep_laya_onnx", PREP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cases(seed: int, n_random: int):
    """Yield (name, ndarray) test tensors."""
    fixed = [
        # min/max pin scale=1.0 and an odd zero point -> ties must round to even
        ("tie_odd_zp", np.asarray([-1.0, 254.0, 0.5, 1.5, -0.5, 2.5, 3.5, 4.5],
                                 dtype=np.float32)),
        # zero point at a .5 tie
        ("zp_tie", np.asarray([-0.5, 254.5, 0.25, 1.75, -0.25], dtype=np.float32)),
        ("zp_tie_2", np.asarray([-1.5, 253.5, 0.5, 1.5, 2.5], dtype=np.float32)),
        # all-nonnegative: raw zero point is negative and must saturate to 0
        ("nonneg", np.asarray([5.0, 260.0, 0.5, 1.5, 128.5], dtype=np.float32)),
        ("nonneg_small", np.asarray([0.0, 1.0, 0.5, 0.25], dtype=np.float32)),
        ("nonneg_tie", np.asarray([2.5, 257.5, 0.5, 1.5, 3.5], dtype=np.float32)),
        # ties across the whole positive and negative half-lines
        ("ties_wide", np.concatenate([
            np.arange(-64, 64, dtype=np.float32) + np.float32(0.5),
            np.asarray([-100.0, 100.0], dtype=np.float32),
        ])),
        # constant tensor: the range is anchored at 0, so scale is 7/255
        ("constant", np.asarray([7.0] * 16, dtype=np.float32)),
        # all-zero tensor: range collapses to 0 and onnxruntime falls back to 1
        ("all_zero", np.asarray([0.0] * 16, dtype=np.float32)),
        # entirely negative: hi must be pulled up to 0, not left at max(x)
        ("all_neg", np.asarray([-300.0, -1.0, -50.5], dtype=np.float32)),
        ("all_neg_tie", np.asarray([-255.0, -1.0, -127.5], dtype=np.float32)),
        # subnormal-scale range
        ("tiny_range", np.asarray([-1e-3, 1e-3, -5e-4, 5e-4, 0.0], dtype=np.float32)),
        # exactly representable halves at large magnitude
        ("big_ties", np.asarray([-100.5, 100.5, 2047.5, -2047.5, 0.5], dtype=np.float32)),
        # extreme asymmetry: the zero point saturates at 255
        ("straddle", np.asarray([-10000.0, 1.0, -5000.5], dtype=np.float32)),
    ]
    for name, arr in fixed:
        yield name, arr

    rng = np.random.default_rng(seed)
    for i in range(n_random):
        rank = int(rng.integers(1, 4))
        shape = tuple(int(rng.integers(1, 7)) for _ in range(rank))
        # A mix of raw floats and values snapped to .5, so ties are common.
        base = rng.normal(scale=float(rng.choice([0.1, 1.0, 10.0, 100.0])), size=shape)
        snapped = np.round(base * 2.0) / 2.0
        mask = rng.random(shape) < 0.5
        arr = np.where(mask, snapped, base).astype(np.float32)
        yield f"rand{i}", arr


def run(sess, names):
    outs = sess.run(names, {})
    return [np.asarray(v) for v in outs]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--random-cases", type=int, default=64)
    ap.add_argument("--dump", help="write the decomposed model here (diagnostic)")
    args = ap.parse_args()

    prep = load_prep()
    items = list(cases(args.seed, args.random_cases))

    nodes, inits, gouts = [], [], []
    for i, (name, arr) in enumerate(items):
        inits.append(numpy_helper.from_array(arr, f"x{i}"))
        nodes.append(helper.make_node("DynamicQuantizeLinear", [f"x{i}"],
                                      [f"y{i}", f"s{i}", f"z{i}"]))
        gouts += [
            helper.make_tensor_value_info(f"y{i}", TensorProto.UINT8, None),
            helper.make_tensor_value_info(f"s{i}", TensorProto.FLOAT, None),
            helper.make_tensor_value_info(f"z{i}", TensorProto.UINT8, None),
        ]

    graph = helper.make_graph(nodes, "verify_dql", [], gouts, inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9

    names = []
    for i, _ in enumerate(items):
        names += [f"y{i}", f"s{i}", f"z{i}"]

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    ref_sess = ort.InferenceSession(model.SerializeToString(), opts,
                                    providers=["CPUExecutionProvider"])
    ref = run(ref_sess, names)

    n = prep.decompose_dynamic_quantize_linear(model)
    if n != len(items):
        print(f"error: rewrote {n} nodes, expected {len(items)}", file=sys.stderr)
        return 3

    if args.dump:
        onnx.save(model, args.dump)
        print(f"wrote {args.dump}")

    got_sess = ort.InferenceSession(model.SerializeToString(), opts,
                                    providers=["CPUExecutionProvider"])
    got = run(got_sess, names)

    bad = 0
    for i, (name, arr) in enumerate(items):
        for j, part in enumerate(("y", "scale", "zero_point")):
            a, b = ref[3 * i + j], got[3 * i + j]
            if a.shape != b.shape:
                print(f"  FAIL {name:<14} {part:<10} shape {a.shape} != {b.shape}")
                bad += 1
                continue
            # NaN == NaN counts as agreement: the degenerate scale=0 case can
            # legitimately produce NaN in both engines. `np.isnan` rejects
            # integer dtypes (the zero point is uint8), so guard on the dtype.
            av, bv = np.asarray(a), np.asarray(b)
            both_nan = np.zeros(av.shape, dtype=bool)
            if np.issubdtype(av.dtype, np.floating):
                both_nan = np.isnan(av) & np.isnan(bv)
            diff = np.asarray((av != bv) & ~both_nan).ravel()
            mism = int(diff.sum())
            if mism:
                where = np.flatnonzero(diff)[:3]
                ar, br = av.ravel(), bv.ravel()
                sample = [(int(k), ar[k], br[k]) for k in where]
                print(f"  FAIL {name:<14} {part:<10} {mism} mismatching element(s), "
                      f"e.g. {sample}")
                bad += 1

    print(f"\n{len(items)} case(s) x 3 output(s): "
          f"{'BIT-EXACT vs onnxruntime' if bad == 0 else f'{bad} MISMATCH(ES)'}")
    return 0 if bad == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
