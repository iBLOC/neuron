#!/usr/bin/env python3
"""Prepare a sherpa-onnx SenseVoice ONNX export so tract 0.21 can load it.

The SenseVoice export computes its sinusoidal position encoding at runtime
with a `Range` -> `Shape` -> `Gather` -> `Div`/`Mul` -> `Sin`/`Cos` chain
whose length is derived from `Shape(x)`. tract types Shape-derived values
as its internal `TDim`, and its typed-stage translator refuses `Range`
nodes fed by TDim -- the same failure documented in
`scripts/prep-laya-onnx.py` step 2 ("Range is the fatal one"). Unlike the
Laya case the length here is *dynamic*, so folding to a Constant is not
an option: the PE rows only depend on the position index, so the whole
chain is replaced with

    pe_table  Constant [T_MAX, 560]   (baked by evaluating the chain once)
    pe_sliced Slice(pe_table, starts=[0], ends=x_length+4, axes=[0])

`x_length + 4` is the encoder time extent (the graph prepends 4 prefix
frames for language/emo/event/itn embeddings and already materialises the
sum at node `Add(x_length, 4)`), and the Slice's dynamic `ends` is a
runtime op tract handles fine. Verified by re-running the original and
patched graphs on the same inputs and comparing logits.

Usage:
    python scripts/prep-sensevoice-onnx.py <in.onnx> <out.onnx> [--t-max 2000]

Requires: onnx, onnxruntime, numpy (only the script needs them; the Rust
runtime does not).
"""

import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# The /encoder/embed chain: Shape-derived T enters at `Range`, leaves as the
# position encoding `Cast_6_output_0` which node `Add_1` adds to the scaled
# input embedding. Node names are stable across the 2024-07-17 export; the
# script asserts every assumption before touching the graph.
CHAIN_FIRST = "/encoder/embed/Range"
CHAIN_LAST = "/encoder/embed/Cast_6_output_0"
PE_CONSUMER = "/encoder/embed/Add_1"
PE_INPUT_EMBEDDING = "/encoder/Mul_output_0"
LENGTH_SUM_OUTPUT = "/Add_output_0"  # Add(x_length, 4) -- encoder T extent
CHAIN_RUNTIME_INPUTS = {
    # end of the position Range, i64 scalar: T_enc + 1
    "/encoder/embed/Cast_output_0": "pe_end",
    # Gather(Shape(x), 2) -- the model width (d_model, 560), a CONSTANT dim
    # misread as dynamic by the export; baked with its real value
    "/encoder/embed/Gather_1_output_0": "pe_T",
}


def say(msg: str) -> None:
    print(f"[prep-sensevoice] {msg}")


def collect_chain(graph):
    """Dataflow closure upward from the producer of CHAIN_LAST: every node
    on the path back to the `Range` root, plus the Constants feeding it.
    Tensors with no producer in the graph must be the declared runtime
    inputs. Returns (nodes, boundary_inputs)."""
    by_output = {}
    for node in graph.node:
        for out in node.output:
            by_output[out] = node

    graph_inputs = {i.name for i in graph.input}
    chain, boundary, seen = [], set(), set()
    stack = [by_output[CHAIN_LAST]]
    while stack:
        node = stack.pop()
        if node.name in seen:
            continue
        seen.add(node.name)
        chain.append(node)
        for inp in node.input:
            if inp in CHAIN_RUNTIME_INPUTS:
                # declared cut point: becomes a mini-model input below
                boundary.add(inp)
                continue
            producer = by_output.get(inp)
            if producer is not None:
                if producer.op_type == "Constant":
                    if producer.name not in seen:
                        chain.append(producer)
                        seen.add(producer.name)
                else:
                    stack.append(producer)
            elif inp in graph_inputs:
                boundary.add(inp)
            else:
                raise SystemExit(f"chain node {node.name} consumes unknown tensor {inp!r}")

    if not any(n.name == CHAIN_FIRST for n in chain):
        raise SystemExit(f"chain does not reach the root {CHAIN_FIRST!r}")

    consumers = {}
    for node in graph.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node.name)
    for node in chain:
        for out in node.output:
            for consumer in consumers.get(out, []):
                if consumer not in seen and out != CHAIN_LAST:
                    raise SystemExit(
                        f"chain output {out!r} leaks into {consumer!r} -- aborting"
                    )
    return chain, sorted(set(boundary))


def assert_known_constants(graph, chain):
    """The Rust runtime hardcodes the PE formula parameters; assert the
    export's constants still match so a different export fails loudly
    here instead of producing silently wrong audio."""
    by_output = {}
    for node in graph.node:
        for out in node.output:
            by_output[out] = node
    def const_of(tensor):
        node = by_output[tensor]
        for a in node.attribute:
            if a.name == "value":
                return numpy_helper.to_array(a.t).flatten().tolist()
        raise SystemExit(f"{tensor} is not a plain Constant")
    expect = {
        "/encoder/embed/Constant_7_output_0": [2.0],       # d_model / 2
        "/encoder/embed/Constant_8_output_0": [1.0],       # - 1
        "/encoder/embed/Constant_9_output_0": [9.2103405], # ln(10000)
        "/encoder/embed/Constant_10_output_0": [0.0],      # freq range start
        "/encoder/embed/Constant_11_output_0": [1.0],      # freq range step
    }
    for tensor, want in expect.items():
        got = const_of(tensor)
        if len(got) != len(want) or any(abs(g - w) > 1e-5 * max(1.0, abs(w)) for g, w in zip(got, want)):
            raise SystemExit(f"PE constant {tensor} = {got}, expected {want}")


def positional_encoding(t_enc: int, dim: int) -> "np.ndarray":
    """The PE the deleted chain computed: positions 1..=T_enc crossed with
    10000^(-i/(dim/2 - 1)), concatenated [sin | cos] along the last axis."""
    half = dim // 2
    scale = np.float32(np.log(np.float32(10000.0))) / np.float32(half - 1)
    i = np.arange(half, dtype=np.float32)
    freq = np.exp(i * -scale, dtype=np.float32)
    pos = np.arange(1, t_enc + 1, dtype=np.float32)
    ang = pos[:, None] * freq[None, :]
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=-1).astype(np.float32)


def patch(graph, dim):
    chain, boundary = collect_chain(graph)
    say(f"PE chain: {len(chain)} nodes from {CHAIN_FIRST!r}, runtime inputs {boundary}")
    if boundary != sorted(CHAIN_RUNTIME_INPUTS):
        raise SystemExit(f"unexpected runtime inputs {boundary}")
    assert_known_constants(graph, chain)

    chain_names = {n.name for n in chain}
    keep = [n for n in graph.node if n.name not in chain_names]

    # the replacement: `pe` arrives as a graph input, shape [1, T, dim].
    # A leading batch dim keeps the graph-shape inference tract-friendly:
    # every downstream op already expects rank-3 activations, so [1, P,
    # dim] aligns with the (1, T, dim) Mul_output_0 it is added to. T_enc
    # is still dynamic via x_length+4 in Rust, but the *axis lengths* of
    # the graph input are symbolic -- this is the minimum amount of
    # dynamism tract's codegen pass accepts.
    pe_vi = helper.make_tensor_value_info("pe", TensorProto.FLOAT, [1, "P", dim])
    graph.input.append(pe_vi)

    consumer = next(n for n in keep if n.name == PE_CONSUMER)
    inputs = list(consumer.input)
    assert PE_INPUT_EMBEDDING in inputs, f"unexpected {PE_CONSUMER} inputs {inputs}"
    idx = inputs.index(CHAIN_LAST)
    consumer.input[idx] = "pe"
    say(f"rewired {PE_CONSUMER} input[{idx}] -> graph input 'pe' [1, P, {dim}]")

    del graph.node[:]
    graph.node.extend(keep)


def verify(src: str, patched_model, dim: int, t: int = 64):
    """Re-run original vs patched on identical inputs; the PE for the
    patched graph comes from positional_encoding()."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    a = ort.InferenceSession(src, so, providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(patched_model.SerializeToString(), so,
                             providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((1, t, dim)) * 0.5).astype(np.float32)
    feed = {"x": x, "x_length": np.array([t], np.int32),
            "language": np.array([0], np.int32), "text_norm": np.array([14], np.int32)}
    l1 = a.run(["logits"], feed)[0]
    pe = positional_encoding(t + 4, dim)[None, :, :]  # [1, T+4, dim] to match graph input
    feed["pe"] = pe
    l2 = b.run(["logits"], feed)[0]
    diff = float(np.abs(l1 - l2).max())
    same = bool(np.array_equal(l1.argmax(-1), l2.argmax(-1)))
    say(f"parity at T={t}: logits {l1.shape}, max |diff| = {diff:.3e}, argmax equal = {same}")
    if not same or diff > 1e-3:
        raise SystemExit("patched graph diverges from the original -- aborting")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(2)
    src, dst = sys.argv[1], sys.argv[2]
    dim = 560
    if "--dim" in sys.argv:
        dim = int(sys.argv[sys.argv.index("--dim") + 1])

    model = onnx.load(src, load_external_data=True)
    graph = model.graph
    patch(graph, dim)
    onnx.checker.check_model(model)
    verify(src, model, dim)
    onnx.save(model, dst)
    say(f"wrote {dst}")


if __name__ == "__main__":
    main()
