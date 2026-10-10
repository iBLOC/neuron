---
title: OCR (text detection + recognition)
---

# OCR (text detection + recognition)

neuron serves a two-stage PP-OCR pipeline (DB++ text detection + SVTR-LCNet
recognition) on `POST /v1/ocr/recognize`, with optional document layout
analysis and table structure recognition. This how-to covers installation,
model preparation, and the full request/response surface.

## 1. Pick a build feature

```toml
[dependencies]
neuron = { git = "https://github.com/iBLOC/neuron", default-features = false, features = ["tract"] }
# or, for the fast executor:
neuron = { git = "https://github.com/iBLOC/neuron", default-features = false, features = ["ort"] }
```

| Feature | Executor | Notes |
|---|---|---|
| `tract` | pure-Rust ONNX (default) | Zero system dependencies. Runs the CNNs 5-12× slower than onnxruntime on CPU. |
| `ort` | onnxruntime via the `ort` crate | ~10× faster end-to-end. Uses **load-dynamic**: the host supplies `onnxruntime.dll`/`.so` via `ORT_DYLIB_PATH` (or `PATH`). |

Both executors share the same wire format and produce byte-identical text.

## 2. Prepare the model directory

Download the official ONNX exports (Apache-2.0) from HuggingFace and run
neuron's prep script (from a neuron checkout) — a metadata-only rewrite
(Paddle2ONNX emits dotted symbol names and `floor(...)` shape hints that
tract's shape parser rejects):

```bash
pip install onnx
curl -LO https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det_onnx/resolve/main/inference.onnx
curl -LO https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx/resolve/main/inference.onnx
curl -LO https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/dict/ppocrv6_dict.txt
python scripts/prep-ppocr-onnx.py inference.onnx det.prepped.onnx   # from the det download
python scripts/prep-ppocr-onnx.py inference.onnx rec.prepped.onnx   # from the rec download
```

(The `ort` executor can technically load the un-prepped export, but always
prep — one artifact for both executors.)

Arrange the directory like this (only the first three files are required):

```
ppocrv6/
├── det.prepped.onnx        # detection graph (prepped)
├── rec.prepped.onnx        # recognition graph (prepped)
├── dict.txt                # recognition vocab (ppocrv5_dict.txt / ppocrv6_dict.txt)
├── rec-latin/              # optional: language packs (see below)
│   ├── rec.prepped.onnx
│   └── dict.txt            # ppocrv5_latin_dict.txt — NOT the old latin_dict.txt
├── layout.onnx             # optional: PP-DocLayoutV3 document layout (ort only)
└── table.onnx              # optional: SLANeXt wired table structure (ort only)
```

Model sources (all `PaddlePaddle/*` on HuggingFace, Apache-2.0):

| File | Repo | Size |
|---|---|---|
| det / rec | `PP-OCRv5_{mobile,server}_{det,rec}_onnx`, `PP-OCRv6_medium_{det,rec}_onnx` | 5-88 MB |
| layout | `PP-DocLayoutV3_onnx` | ~130 MB |
| table | `SLANeXt_wired_onnx` (bordered tables) | ~368 MB |

::: warning Do not use SLANet_plus
The widely mirrored `SLANet_plus_onnx` export is broken — its autoregressive
decoder ignores the image (white noise and real tables decode identically).
SLANeXt is the working table-structure path.
:::

## 3. Register the model

OCR registration is config-driven (capability `ocr`, engine `paddle-ocr`):

```rust
use neuron::{run_embedded, EngineType, NeuronConfig, NeuronEmbedConfig, DeviceSelection};

let config = NeuronConfig::new().with_registration_for_capability(
    EngineType::Ocr,
    "ppocrv6-medium",       // model id used in requests
    "paddle-ocr",           // engine
    DeviceSelection::Cpu,
    500_000_000,            // memory budget hint
);
let handle = run_embedded(NeuronEmbedConfig { port: 8085, config }).await?;
```

Point the runtime at the directory via the `NEURON_OCR_MODEL_PATH` env var
(or drop it under `<model_root>/ocr/ppocrv6-medium/`).

## 4. Recognize text

```bash
curl -X POST http://localhost:8085/v1/ocr/recognize \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ppocrv6-medium",
    "image": { "type": "image_url", "image_url": { "url": "file:///tmp/invoice.png" } }
  }'
```

The `image` field accepts the same shapes as the vision endpoints
(`image_url` with `http(s)`/`data:`/`file://` URLs or a local path, or
base64). Response:

```json
{
  "model": "ppocrv6-medium",
  "image_size": [1920, 1080],
  "inference_time_ms": 526,
  "text": "Quarterly Report\n净利润: ¥31.4M",
  "regions": [
    { "bbox": { "x": 120, "y": 88, "w": 420, "h": 44 },
      "text": "Quarterly Report", "confidence": 0.98, "line": 0 }
  ],
  "languages": ["default"]
}
```

`regions` is in reading order (rows top-to-bottom, left-to-right within a
row); `text` is the lines joined by newlines.

## 5. Other languages

Optional `rec-<pack>/` subdirectories hold per-script recognition models.
A request's `languages` field routes to one pack:

| Request value | Pack | Model |
|---|---|---|
| `zh` / `en` / `ja` (default) | default rec | `PP-OCRv5/v6_*_rec_onnx` |
| `fr`, `de`, `es`, … | `rec-latin` | `latin_PP-OCRv5_mobile_rec_onnx` |
| `ko` | `rec-korean` | `korean_PP-OCRv5_mobile_rec_onnx` |
| `ru`, `uk`, … | `rec-cyrillic` | `cyrillic_PP-OCRv5_mobile_rec_onnx` |
| `ar`, `fa`, … | `rec-arabic` | `arabic_PP-OCRv5_mobile_rec_onnx` |
| `hi`, `mr`, `ne` | `rec-devanagari` | `devanagari_PP-OCRv5_mobile_rec_onnx` |

An exact pack name (`"latin"`) also works as an escape hatch. Pack dicts
are `ppocr/utils/dict/ppocrv5_<script>_dict.txt` from the PaddleOCR repo —
the older `<script>_dict.txt` files do not match the v5 exports (the
runtime rejects the wrong pairing with a clear vocab-size error).

```bash
curl -X POST http://localhost:8085/v1/ocr/recognize \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ppocrv6-medium",
    "languages": ["fr"],
    "image": { "type": "image_url", "image_url": { "url": "file:///tmp/facture.png" } }
  }'
```

The response's `languages` field reports the pack actually used.

## 6. Layout and tables (optional)

With `layout.onnx` present (ort executor), the response gains a `layout[]`
array — 25 region classes (`text`, `table`, `image`, `display_formula`,
`doc_title`, …) with boxes and scores — and every text line carries a
`kind` naming its containing region. With `table.onnx` additionally
present, `table` regions also carry an assembled `html` field with the
recognized text placed into cells.

```json
{
  "layout": [
    { "kind": "table", "bbox": { "x": 78, "y": 377, "w": 524, "h": 205 }, "score": 0.96,
      "html": "<table><tr><td>Region</td><td>Q1</td></tr>…</table>" }
  ]
}
```

## Configuration knobs

| Env var | Default | Purpose |
|---|---|---|
| `NEURON_OCR_MODEL_PATH` | — | Model directory (falls back to `<model_root>/ocr/<id>`) |
| `NEURON_OCR_EXECUTOR` | `tract` | `tract` \| `ort` |
| `NEURON_OCR_SESSIONS` | `2` | ort sessions per graph (concurrency) |
| `NEURON_OCR_LAYOUT` | `true` | `false` skips `layout.onnx` |
| `NEURON_OCR_TABLES` | `true` | `false` skips `table.onnx` |
| `ORT_DYLIB_PATH` | — | onnxruntime library for the `ort` executor |

## Next steps

- [/reference/api](/reference/api) — full request/response schema
- [/concepts/scheduler](/concepts/scheduler) — memory-bounded model swapping
