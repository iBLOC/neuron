#!/usr/bin/env python
"""Export the Ultralytics YOLO checkpoints to ONNX for neuron's vision
benchmark.

The neuron's ONNX vision path expects a *prepared* model directory:

    model.onnx     # static 1x3x640x640 graph
    config.json    # optional: input/output names, num_classes, output_layout

The upstream Ultralytics repos ship PyTorch `.pt` weights, so the export is
the missing step. `ultralytics` does it:

    yolo export model=yolov8n.pt format=onnx imgsz=640 opset=17

Usage:
    python scripts/export-yolo-onnx.py --models yolov8n yolov8s yolo11n
    python scripts/export-yolo-onnx.py --root ~/.neuron/models/vision

Requirements: `pip install ultralytics` (the PyPI mirror
https://pypi.tuna.tsinghua.edu.cn/simple works when pypi.org is blocked).
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# model id -> (upstream repo, checkpoint file inside it)
CHECKPOINTS = {
    "yolov8n": ("Ultralytics/YOLOv8", "yolov8n.pt"),
    "yolov8s": ("Ultralytics/YOLOv8", "yolov8s.pt"),
    "yolo11n": ("Ultralytics/YOLO11", "yolo11n.pt"),
}


def hf_download(repo: str, filename: str, dest_dir: Path) -> Path:
    dest = dest_dir / filename
    if dest.exists():
        print(f"[export] {dest} already present")
        return dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    print(f"[export] downloading {url}")
    # Prefer the `hf` CLI: the plain curl path trips over TLS shutdown
    # errors on some Windows schannel builds.
    rc = subprocess.call(
        ["hf", "download", repo, "--include", filename, "--local-dir", str(dest_dir)]
    )
    if rc != 0:
        print("[export] hf download failed, retrying with curl")
        rc = subprocess.call(
            ["curl", "-sSL", "--fail", "--retry", "3", "-o", str(dest), url]
        )
        if rc != 0:
            raise SystemExit(f"[export] download failed: {url}")
    return dest


def export(ckpt: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[export] {ckpt.name} -> {out_dir}/model.onnx")
    rc = subprocess.call(
        [
            "yolo",
            "export",
            f"model={ckpt}",
            "format=onnx",
            "imgsz=640",
            "opset=17",
            "simplify=False",
            f"project={out_dir.parent}",
            f"name={out_dir.name}",
        ]
    )
    if rc != 0:
        raise SystemExit(f"[export] ultralytics export failed for {ckpt}")
    produced = out_dir / "model.onnx"
    if not produced.exists():
        # ultralytics may write next to the checkpoint instead.
        produced = ckpt.with_suffix(".onnx")
        if produced.exists():
            os.replace(produced, out_dir / "model.onnx")
    if not (out_dir / "model.onnx").exists():
        raise SystemExit(f"[export] no model.onnx produced for {ckpt}")
    # The neuron's ort runtime reads config.json next to model.onnx.
    cfg = {
        "input_name": "images",
        "output_names": ["output0"],
        "input_shape": [1, 3, 640, 640],
        "num_classes": 80,
        "task": "detect",
        "output_layout": "yolov8",
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"[export] ok: {out_dir}/model.onnx ({(out_dir / 'model.onnx').stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["yolov8n", "yolov8s", "yolo11n"])
    ap.add_argument(
        "--root",
        default=os.path.expanduser("~/.neuron/models/vision"),
        help="destination root (one directory per model)",
    )
    args = ap.parse_args()
    root = Path(args.root).expanduser()

    for model in args.models:
        if model not in CHECKPOINTS:
            print(f"[export] unknown model {model}, skipping")
            continue
        repo, filename = CHECKPOINTS[model]
        out_dir = root / model
        ckpt = hf_download(repo, filename, out_dir)
        export(ckpt, out_dir)


if __name__ == "__main__":
    main()
