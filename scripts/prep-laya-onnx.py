#!/usr/bin/env python3
"""Prepare a Laya decision-model ONNX graph so tract 0.21 can load and run it.

Community Laya ONNX exports load fine in onnxruntime but fail tract 0.21 in
four distinct ways -- three are type-inference failures, one is a numerics
divergence. This script applies the four mechanical patches that close each
gap. It is pure protobuf surgery -- no PyTorch, no re-export, no weight
modification.

  1. GROUND SYMBOLIC DIMS. The exports are self-contradictory about batch:
     most branches declare `dim_param="b"` while the act_head branch hardcodes
     batch=1 (`act_logits f32[1,2]`). tract refuses to unify `Sym(b)` with
     `Val(1)`. Rewriting `dim_param` -> `dim_value` resolves it. Value_info
     entries that stay partially symbolic must be dropped whole so tract
     re-infers them.

  2. CONSTANT-FOLD SHAPE SUBGRAPHS. `Shape` -> `Range` (ModernBERT's RoPE
     arange) -> `Gather`/`Squeeze` chains are typed by tract as its internal
     `TDim`, which cannot unify with the ONNX-declared `I64`. Folding them into
     `Constant` nodes means `TDim` never appears. `Range` is the fatal one;
     leftover `Shape` nodes are fine. Note `Shape` only needs its input's
     *shape*, not its value, so it folds even when its input is a graph
     input -- a fold gate of the form "all inputs already evaluated" will
     wrongly skip it.

  3. `Clip` -> `Identity`. The three clamps (`marker_pos >= 0`, option count
     `>= 2`, `prob >= 1e-9`) fail tract's `Clip` type rule (`inputs[0].datum_type
     == inputs[1].datum_type`) as `TDim` vs `I64`. The clamps are trivial and
     move host-side into Rust. Inserting `Cast` does NOT help: tract's cast
     converts `TDim` -> i64 in `eval` but type inference still reports `TDim`.

  4. `DynamicQuantizeLinear` -> `Round`-based primitives. tract 0.21 rounds the
     quantized code half-away-from-zero; the ONNX spec and onnxruntime round
     half-to-even, so every `.5` tie lands one code too high in tract and the
     int8 matmuls amplify that single wrong code into percent-level error on
     the logits. tract's ONNX `Round` op *is* half-to-even, so expressing the
     same arithmetic with `Round` removes the divergence. Must run after step 3
     -- the saturation below is a real `Clip` and would be nulled by it.

Dim_param names vary per exporter (`b`/`L`/`K`/`16*b`/`b*L` from torch.export
vs `batch`/`sequence`/`markers` plus hundreds of `unk__*` from shape_inference),
so resolution is semantic (by position within known input tensors), never by
name matching.

Usage:
  scripts/prep-laya-onnx.py INPUT.onnx OUTPUT.onnx [options]

Options:
  --batch N        Ground the batch dimension to N (default 1)
  --seq N          Ground the sequence length to N (omit to leave dynamic)
  --markers N      Ground the marker/option count to N (omit to leave dynamic)
  --set NAME=VAL   Ground an arbitrary dim_param by name (repeatable). Use this
                   only for dims the semantic pass cannot place.
  --no-fold        Skip step 2 (diagnostic)
  --no-clip-swap   Skip step 3 (diagnostic)
  --no-dql-decompose  Skip step 4 (diagnostic; leaves tract's non-conformant
                   DynamicQuantizeLinear rounding in place)
  --quiet          Only print the summary line

Dynamic shape notes:
  * Batch is always grounded -- the act_head branch hardcodes it to 1 anyway.
  * Leaving --seq/--markers out keeps L and K symbolic. That is verified to
    work on the standalone decision head (tested at (L,K) = 16x4, 37x3, 512x16,
    1x1). The full encoder needs --seq: its RoPE `Range` nodes will not fold
    without a concrete sequence length, and an unfolded `Range` is fatal.

Exit codes:
  0   success
  2   bad arguments
  3   input model failed onnx.checker after patching (patch made it invalid)
  4   batch dimension left symbolic on a graph input
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

# Ops whose outputs we can compute offline from already-known tensors (plus
# `Shape`, handled separately because it needs only an input's shape).
FOLDABLE = {
    "Shape",
    "Range",
    "Gather",
    "GatherElements",
    "Squeeze",
    "Unsqueeze",
    "Cast",
    "Concat",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Neg",
    "Abs",
    "ConstantOfShape",
    "Equal",
    "Where",
}

MAX_FOLD_PASSES = 16


# --------------------------------------------------------------------------
# step 1: ground symbolic dims
# --------------------------------------------------------------------------

def _dims_of(vi):
    """Yield the Dim protos of a ValueInfoProto, or () if it has no shape."""
    if not (vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape")):
        return ()
    return vi.type.tensor_type.shape.dim


def collect_dim_params(model):
    """Map every dim_param name to the list of (where, index) it occurs at."""
    found: dict[str, list[tuple[str, int, int]]] = {}

    def note(vi, where, vi_index):
        for d_i, d in enumerate(_dims_of(vi)):
            if d.HasField("dim_param") and d.dim_param:
                found.setdefault(d.dim_param, []).append((where, vi_index, d_i))

    for i, vi in enumerate(model.graph.input):
        note(vi, "input", i)
    for i, vi in enumerate(model.graph.output):
        note(vi, "output", i)
    for i, vi in enumerate(model.graph.value_info):
        note(vi, "value_info", i)
    return found


def resolve_semantic_dims(model, batch, seq, markers, explicit):
    """Decide dim_param -> int without ever matching on the param's own name.

    Semantics come from where the dim sits inside a recognised input tensor:
    dim 0 of any input is the batch; dim 1 of the token/attention tensors is the
    sequence length; dim 1 of the marker tensors is the option count.
    """
    resolved: dict[str, int] = {}
    if batch is not None:
        resolved["__batch__"] = batch  # sentinel; every dim-0 param maps to it

    seq_like = ("input_ids", "hidden", "attention_mask", "input", "token_ids")
    marker_like = ("marker_pos", "marker_mask", "marker_positions", "marker")

    for vi in model.graph.input:
        name = vi.name
        dims = list(_dims_of(vi))
        for d_i, d in enumerate(dims):
            if not (d.HasField("dim_param") and d.dim_param):
                continue
            param = d.dim_param
            if d_i == 0 and batch is not None:
                resolved[param] = batch
            elif d_i == 1 and seq is not None and name in seq_like:
                resolved[param] = seq
            elif d_i == 1 and markers is not None and any(
                name.startswith(p) for p in marker_like
            ):
                resolved[param] = markers

    # Explicit --set always wins; it is the escape hatch for odd exports.
    resolved.update(explicit)
    resolved.pop("__batch__", None)
    return resolved


def ground_value_info(vi, mapping):
    """Ground what we can. Returns (n_grounded, n_symbolic_left)."""
    n, left = 0, 0
    for d in _dims_of(vi):
        if d.HasField("dim_param") and d.dim_param:
            if d.dim_param in mapping:
                d.dim_value = mapping[d.dim_param]
                d.ClearField("dim_param")
                n += 1
            else:
                left += 1
    return n, left


def ground_dims(model, mapping):
    n_grounded = 0
    for coll in ("input", "output", "value_info"):
        getattr(model.graph, coll)  # touch for clarity; same call shape below
    for vi in model.graph.input:
        n_grounded += ground_value_info(vi, mapping)[0]
    for vi in model.graph.output:
        n_grounded += ground_value_info(vi, mapping)[0]
    for vi in model.graph.value_info:
        n_grounded += ground_value_info(vi, mapping)[0]
    return n_grounded


def drop_symbolic_value_info(model):
    """Remove value_info entries that still carry any symbolic dim.

    tract re-infers these correctly from the grounded inputs; leaving them in
    makes tract trust a stale symbolic fact and fail later.
    """
    before = len(model.graph.value_info)
    keep = []
    for vi in model.graph.value_info:
        _, left = ground_value_info(vi, {})
        if left == 0:
            keep.append(vi)
    del model.graph.value_info[:]
    model.graph.value_info.extend(keep)
    return before - len(keep)


def check_batch_concrete(model):
    """Batch (dim 0 of every input) must be concrete; other dims may stay free.

    The act_head branch hardcodes batch=1, so a symbolic batch makes tract fail
    to unify regardless of what we feed. Sequence length and marker count have
    no such contradiction and are legitimately dynamic on the decision head.
    """
    bad = []
    for vi in model.graph.input:
        dims = list(_dims_of(vi))
        if dims and dims[0].HasField("dim_param") and dims[0].dim_param:
            bad.append(f"input:{vi.name}:{dims[0].dim_param}")
    return bad


def remaining_symbolic(model):
    """Name every dim_param still live on graph inputs/outputs."""
    out = set()
    for coll in ("input", "output"):
        for vi in getattr(model.graph, coll):
            for d in _dims_of(vi):
                if d.HasField("dim_param") and d.dim_param:
                    out.add(d.dim_param)
    return sorted(out)


def remaining_ops(model, ops):
    """Names of the given op types still present in the graph."""
    return sorted({n.op_type for n in model.graph.node if n.op_type in ops})


# --------------------------------------------------------------------------
# step 2: constant-fold shape subgraphs
# --------------------------------------------------------------------------

def known_shapes(model):
    """Map tensor name -> concrete shape tuple, from inputs/outputs/value_info."""
    out: dict[str, tuple[int, ...]] = {}

    def take(vi):
        dims = list(_dims_of(vi))
        if dims and all(d.HasField("dim_value") for d in dims):
            out[vi.name] = tuple(int(d.dim_value) for d in dims)

    for vi in model.graph.input:
        take(vi)
    for vi in model.graph.output:
        take(vi)
    for vi in model.graph.value_info:
        take(vi)
    return out


def _attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            return helper.get_attribute_value(a)
    return default


def try_eval(node, avail, shapes):
    """Compute a foldable node's output as an ndarray, or None if not foldable."""
    op = node.op_type

    # Shape needs only its input's shape -- it folds even for graph inputs.
    if op == "Shape":
        s = shapes.get(node.input[0])
        if s is None:
            return None
        arr = np.asarray(s, dtype=np.int64)
        start = _attr(node, "start", 0) or 0
        end = _attr(node, "end", 0) or len(s)
        return arr[start:end]

    if not all(i in avail for i in node.input if i):
        return None

    ins = [avail[i] for i in node.input if i]
    try:
        if op == "Range":
            st, li, de = (int(np.asarray(x).reshape(())) for x in ins)
            return np.arange(st, li, de, dtype=np.int64)
        if op == "Gather":
            axis = _attr(node, "axis", 0)
            return np.take(ins[0], ins[1].astype(np.int64), axis=axis)
        if op == "GatherElements":
            axis = _attr(node, "axis", 0)
            return np.take_along_axis(ins[0], ins[1].astype(np.int64), axis=axis)
        if op == "Squeeze":
            axes = ins[1] if len(ins) > 1 else _attr(node, "axes", None)
            return np.squeeze(ins[0], axis=tuple(int(a) for a in np.asarray(axes)))
        if op == "Unsqueeze":
            axes = ins[1] if len(ins) > 1 else _attr(node, "axes", None)
            return np.expand_dims(ins[0], axis=tuple(int(a) for a in np.asarray(axes)))
        if op == "Cast":
            to = _attr(node, "to")
            tgt = helper.tensor_dtype_to_np_dtype(to)
            return ins[0].astype(tgt)
        if op == "Concat":
            return np.concatenate(ins, axis=_attr(node, "axis", 0))
        if op == "Add":
            return ins[0] + ins[1]
        if op == "Sub":
            return ins[0] - ins[1]
        if op == "Mul":
            return ins[0] * ins[1]
        if op == "Div":
            return np.divide(ins[0], ins[1])
        if op == "Neg":
            return -ins[0]
        if op == "Abs":
            return np.abs(ins[0])
        if op == "Equal":
            return ins[0] == ins[1]
        if op == "Where":
            return np.where(ins[0], ins[1], ins[2])
        if op == "ConstantOfShape":
            val = _attr(node, "value")
            fill = numpy_helper.to_array(val).reshape(()) if val is not None else 0
            return np.full(tuple(int(x) for x in np.asarray(ins[0])), fill)
    except Exception:
        return None
    return None


def fold_shape_subgraphs(model):
    """Replace foldable shape-semantic nodes with Constants. Returns count."""
    folded_total = 0
    for _ in range(MAX_FOLD_PASSES):
        # Re-derive known shapes each pass: folding one Shape makes downstream
        # shapes decidable on the next pass.
        g = model.graph
        for vi in g.value_info:
            dims = list(_dims_of(vi))
            if dims and all(d.HasField("dim_value") for d in dims):
                pass  # known_shapes() rebuilds the map below

        avail: dict[str, np.ndarray] = {}
        for init in g.initializer:
            avail[init.name] = numpy_helper.to_array(init)

        shapes = known_shapes(model)

        this_pass = []
        for node in g.node:
            if node.op_type not in FOLDABLE:
                continue
            if not node.output:
                continue
            val = try_eval(node, avail, shapes)
            if val is None:
                continue
            out_name = node.output[0]
            this_pass.append((node, out_name, np.asarray(val)))

        if not this_pass:
            break

        for node, out_name, val in this_pass:
            del node.input[:]
            del node.attribute[:]
            del node.output[:]
            node.output.append(out_name)
            node.op_type = "Constant"
            node.domain = ""
            node.attribute.extend([helper.make_attribute("value", numpy_helper.from_array(val))])
            avail[out_name] = val
            folded_total += 1

    return folded_total


# --------------------------------------------------------------------------
# step 3: Clip -> Identity
# --------------------------------------------------------------------------

def swap_clip_for_identity(model):
    """Drop Clip's min/max inputs and turn it into Identity. Returns count."""
    n = 0
    for node in model.graph.node:
        if node.op_type != "Clip":
            continue
        if len(node.input) != 1:
            # Clip with computed min/max is exactly the failing case; the Laya
            # clamps are all of this shape (scalar min/max produced upstream).
            pass
        del node.attribute[:]
        keep = node.input[0]
        del node.input[:]
        node.input.append(keep)
        node.op_type = "Identity"
        node.domain = ""
        n += 1
    return n


# --------------------------------------------------------------------------
# step 4: DynamicQuantizeLinear -> Round-based primitives
# --------------------------------------------------------------------------

def tensor_names(graph):
    """Every tensor name already live in the graph."""
    out = {i.name for i in graph.initializer}
    out.update(i.name for i in graph.input)
    out.update(o.name for o in graph.output)
    for node in graph.node:
        out.update(x for x in list(node.input) + list(node.output) if x)
    return out


def decompose_dynamic_quantize_linear(model):
    """Rewrite every `DynamicQuantizeLinear` into `Round`-based primitives.

    tract 0.21 computes the quantized code as `saturate(f32::round(x / scale) + zp)`
    where `f32::round` is round-half-away-from-zero; the ONNX spec and
    onnxruntime use round-half-to-even. tract's ONNX `Round` op *is*
    half-to-even (verified against onnxruntime 1.30.0), so rewriting the same
    arithmetic with `Round` makes the two engines agree.

    The formula reproduced is onnxruntime's, established by probing it rather
    than read off the spec -- the spec's `(max - min) / 255` is NOT what
    onnxruntime computes. Every detail below is load-bearing:

        lo    = min(min(x), 0)          # the range is anchored at 0 on both ends
        hi    = max(max(x), 0)
        scale = (hi - lo) / 255, or 1 when that is 0   # all-zero tensor
        zp    = saturate_uint8(round(-lo / scale))
        y     = saturate_uint8(round(x / scale) + zp)

    The `+ zp` is outside the `round`, not inside it: with an odd zero point the
    two orderings disagree on a `.5` tie, and onnxruntime matches
    `round(x / scale) + zp`. All rounding is half-to-even, which is what
    tract's `Round` op does and what tract's `DynamicQuantizeLinear` does not.

    `y_scale` and `y_zero_point` keep their declared rank-0 shapes. The rewrite
    is purely additive around the existing names, so downstream consumers
    (notably `gen-laya-golden.py`'s `exact_integer_gemm`, which resolves the
    zero points by name) need no change. Returns the number of nodes rewritten.
    """
    graph = model.graph
    hits = [i for i, n in enumerate(graph.node) if n.op_type == "DynamicQuantizeLinear"]
    if not hits:
        return 0

    names = tensor_names(graph)
    counter = [0]

    def fresh(stem):
        counter[0] += 1
        cand = f"{stem}_{counter[0]}"
        while cand in names:
            counter[0] += 1
            cand = f"{stem}_{counter[0]}"
        names.add(cand)
        return cand

    zero, one, qmax = fresh("__dql_zero"), fresh("__dql_one"), fresh("__dql_qmax")
    graph.initializer.extend([
        numpy_helper.from_array(np.asarray(0.0, dtype=np.float32), zero),
        numpy_helper.from_array(np.asarray(1.0, dtype=np.float32), one),
        numpy_helper.from_array(np.asarray(255.0, dtype=np.float32), qmax),
    ])

    out_nodes = []
    for idx, node in enumerate(graph.node):
        if node.op_type != "DynamicQuantizeLinear":
            out_nodes.append(node)
            continue

        x = node.input[0]
        padded = list(node.output) + ["", "", ""]
        y, y_scale, y_zp = padded[0], padded[1], padded[2]

        xmin, xmax = fresh("__dql_xmin"), fresh("__dql_xmax")
        lo, hi = fresh("__dql_lo"), fresh("__dql_hi")
        rng = fresh("__dql_range")
        raw = fresh("__dql_raw")
        no_scale, scale = fresh("__dql_noscale"), (y_scale or fresh("__dql_scale"))
        neglo = fresh("__dql_neglo")
        zp_raw, zp_rnd, zp = fresh("__dql_zpraw"), fresh("__dql_zprnd"), fresh("__dql_zp")
        t, r, s = fresh("__dql_t"), fresh("__dql_r"), fresh("__dql_s")
        yf = fresh("__dql_yf")

        tag = node.name or f"dql{idx}"
        seq = [
            ("ReduceMin", [x], [xmin], {"keepdims": 0}),
            ("ReduceMax", [x], [xmax], {"keepdims": 0}),
            ("Min", [xmin, zero], [lo], {}),
            ("Max", [xmax, zero], [hi], {}),
            ("Sub", [hi, lo], [rng], {}),
            ("Div", [rng, qmax], [raw], {}),
            ("Equal", [raw, zero], [no_scale], {}),
            ("Where", [no_scale, one, raw], [scale], {}),
            ("Sub", [zero, lo], [neglo], {}),
            ("Div", [neglo, scale], [zp_raw], {}),
            ("Round", [zp_raw], [zp_rnd], {}),
            ("Clip", [zp_rnd, zero, qmax], [zp], {}),
            ("Div", [x, scale], [t], {}),
            ("Round", [t], [r], {}),
            ("Add", [r, zp], [s], {}),
            ("Clip", [s, zero, qmax], [yf], {}),
        ]
        if y:
            seq.append(("Cast", [yf], [y], {"to": TensorProto.UINT8}))
        if y_zp:
            seq.append(("Cast", [zp], [y_zp], {"to": TensorProto.UINT8}))

        for j, (op, ins, outs, attrs) in enumerate(seq):
            out_nodes.append(helper.make_node(op, ins, outs, name=f"{tag}__dql{j}", **attrs))

    del graph.node[:]
    graph.node.extend(out_nodes)
    return len(hits)


# --------------------------------------------------------------------------

def infer(model):
    # `infer_shapes` returns a NEW model proto -- rebind, or you patch the old
    # one and save the new ungrounded one. This bit us twice.
    return shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=None)
    ap.add_argument("--markers", type=int, default=None)
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VAL")
    ap.add_argument("--no-fold", action="store_true")
    ap.add_argument("--no-clip-swap", action="store_true")
    ap.add_argument("--no-dql-decompose", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    explicit = {}
    for item in args.set:
        if "=" not in item:
            print(f"error: --set expects NAME=VAL, got {item!r}", file=sys.stderr)
            return 2
        k, v = item.split("=", 1)
        explicit[k] = int(v)

    def say(msg):
        if not args.quiet:
            print(msg)

    model = onnx.load(args.input)
    mapping = resolve_semantic_dims(model, args.batch, args.seq, args.markers, explicit)
    say(f"dim resolution: {len(mapping)} dim_param(s) -> {mapping}")

    n_grounded = ground_dims(model, mapping)
    say(f"grounded {n_grounded} symbolic dim(s)")

    model = infer(model)
    n_dropped = drop_symbolic_value_info(model)

    leftover = check_batch_concrete(model)
    if leftover:
        print(
            f"error: batch dimension still symbolic: {leftover}\n"
            f"       act_head hardcodes batch=1, so tract cannot unify it. "
            f"Pass --batch N.",
            file=sys.stderr,
        )
        return 4

    free = remaining_symbolic(model)
    if free:
        say(f"left dynamic: {free} (L/K dynamic is verified on the decision head)")

    if not args.no_fold:
        n_folded = fold_shape_subgraphs(model)
        say(f"constant-folded {n_folded} shape-semantic node(s)")
    else:
        n_folded = 0

    # Informational only. An unfolded `Range` is what makes tract fail on
    # `TDim` vs `I64`, but only ModernBERT's RoPE arange is fatal -- leftover
    # `Shape`/`Range` nodes in other parts of the graph have been observed to
    # load fine, so this must not gate the exit code. The authoritative check
    # is the tract load test (tests/decision_parity.rs).
    leftover_ops = remaining_ops(model, ("Shape", "Range"))
    if leftover_ops:
        names = [n.name for n in model.graph.node if n.op_type in leftover_ops]
        say(
            f"un-folded shape ops remain ({leftover_ops}, e.g. {names[:3]}); "
            f"benign unless tract fails at a `Range` -- then ground --seq"
        )

    if not args.no_clip_swap:
        n_clip = swap_clip_for_identity(model)
        say(f"replaced {n_clip} Clip node(s) with Identity")
    else:
        n_clip = 0

    # After the Clip swap: this pass introduces Clips of its own (the uint8
    # saturations) and they must survive.
    if not args.no_dql_decompose:
        n_dql = decompose_dynamic_quantize_linear(model)
        say(f"decomposed {n_dql} DynamicQuantizeLinear node(s)")
    else:
        n_dql = 0

    model = infer(model)

    # infer_shapes re-creates symbolic `unk__*` value_info entries. Those
    # declared facts fight tract's GatherNd *inference* rule (distinct from its
    # correct `compute_shape`) and abort the analysis with "1,1,1,64 has no 4-th
    # dimension". Dropping them and letting tract re-derive the facts is what
    # makes the graph load, so this must run after the last infer_shapes.
    n_dropped += drop_symbolic_value_info(model)

    try:
        onnx.checker.check_model(model, full_check=True)
    except Exception as exc:
        print(f"error: patched model failed onnx.checker: {exc}", file=sys.stderr)
        return 3

    # onnx.save raises FileExistsError if the external-data sidecar already
    # exists, so clear both before writing. `location` is resolved relative to
    # the model file's directory, so it must be a bare filename -- passing
    # "dir/model.onnx.data" makes onnx look for "dir/dir/model.onnx.data".
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    sidecar = args.output + ".data"
    for p in (args.output, sidecar):
        if os.path.exists(p):
            os.remove(p)
    onnx.save(
        model,
        args.output,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(sidecar),
        size_threshold=1024,
    )

    print(
        f"{args.output}: grounded={n_grounded} dropped_vi={n_dropped} "
        f"folded={n_folded} clip_swapped={n_clip} dql_decomposed={n_dql}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
