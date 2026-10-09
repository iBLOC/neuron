#!/usr/bin/env python3
"""Prep a PP-OCR (Paddle2ONNX) export so tract 0.21 can load it.

Two cosmetic fixes, no compute semantics touched. Verified 2026-10-09 on the
official `PaddlePaddle/PP-OCRv5_{mobile,server}_{det,rec}_onnx` and
`PaddlePaddle/PP-OCRv6_medium_{det,rec}_onnx` exports: all six load, optimize
and run through tract 0.21.10 after this prep, with onnxruntime parity
(max abs diff <= 2e-6 det, <= 2e-4 rec softmax).

1. Rename `.` in *input* dim params (`DynamicDimension.0` ->
   `DynamicDimension_0`). tract's TDim parser rejects identifiers with dots
   ("Failed to parse 'DynamicDimension.0'").
2. Delete `value_info` and clear output dim params. Paddle2ONNX decorates
   intermediates with symbolic shape *hints* like
   `floor(DynamicDimension.1/2 - 1/2) + 1`; tract's TDim parser has no
   function syntax and refuses them. They are metadata, not computation --
   tract propagates its own symbolic shapes through the graph during
   inference -> typed conversion, so dropping the hints is safe.

Unlike `prep-sensevoice-onnx.py` / `prep-laya-onnx.py` no compute-graph
surgery is needed: PP-OCR graphs are pure CNN (Conv / ConvTranspose / Resize
/ HardSigmoid / MatMul+Softmax attention blocks), with no `Range` chains and
no custom ops; DB post-processing and CTC decoding run outside the graph.

Usage:
    python scripts/prep-ppocr-onnx.py <in.onnx> <out.onnx>

Requires: onnx (only the script needs it; the Rust runtime does not).
"""
import sys

import onnx


def prep(graph):
    renamed = []
    for tensor in graph.input:
        for d in tensor.type.tensor_type.shape.dim:
            if d.dim_param and "." in d.dim_param:
                renamed.append((d.dim_param, d.dim_param.replace(".", "_")))
                d.dim_param = d.dim_param.replace(".", "_")
    del graph.value_info[:]
    stripped = 0
    for tensor in graph.output:
        for d in tensor.type.tensor_type.shape.dim:
            if d.dim_param:
                d.dim_param = ""
                stripped += 1
    return renamed, stripped


def main():
    src, dst = sys.argv[1], sys.argv[2]
    m = onnx.load(src, load_external_data=False)
    renamed, stripped = prep(m.graph)
    onnx.save(m, dst)
    print(f"[prep-ppocr] {src} -> {dst}: renamed {len(renamed)} input dims, "
          f"stripped value_info + {stripped} output dims")


if __name__ == "__main__":
    main()
