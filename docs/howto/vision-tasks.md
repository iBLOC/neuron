---
title: Vision Tasks (detect / classify / segment / pose)
---

# Vision Tasks (detect / classify / segment / pose)

neuron exposes 4 vision task endpoints under `/v1/vision/*`, dispatched to the appropriate model family. This how-to covers the full request/response shape, the model selection rules, and the LoRA extension mechanism.

## The 4 tasks

| Endpoint | Task | Output |
|---|---|---|
| `POST /v1/vision/detect` | Object detection | Bounding boxes + class + score |
| `POST /v1/vision/classify` | Image classification | Class probabilities (top-K) |
| `POST /v1/vision/segment` | Semantic segmentation | Per-pixel class map |
| `POST /v1/vision/pose` | Human keypoint estimation | Keypoints + skeleton |

All 4 endpoints accept the same `multipart/form-data` request format with an image upload.

## Request format

```bash
curl -X POST http://localhost:8085/v1/vision/detect \
  -F "file=@./photo.jpg" \
  -F "model=yolov8n" \
  -F "confidence=0.5" \
  -F "iou=0.45" \
  -F "lora_adapter=warehouse-v1"   # optional
```

| Form field | Required | Type | Default | Description |
|---|---|---|---|---|
| `file` | ✅ | binary | — | Image bytes (PNG, JPEG, WebP) |
| `model` | ✅ | string | — | One of the registered model IDs (e.g. `yolov8n`, `rtdetr-r50`) |
| `confidence` | — | float | `0.25` | Minimum confidence threshold (0.0-1.0) |
| `iou` | — | float | `0.45` | IoU threshold for NMS (detection only) |
| `lora_adapter` | — | string | — | Name of a registered LoRA adapter (see below) |
| `max_detections` | — | integer | `100` | Cap on detections returned (detection only) |

## StreamMind gate (Mage-VL)

```
POST /v1/vision/gate/score
```

The Mage-VL StreamMind event gate scores a rolling codec window and emits `p_speak ∈ [0, 1]`: the likelihood that the window contains a response-worthy event. Clients poll the gate and only invoke full VLM generation when `p_speak` crosses a threshold — this is how Mage-VL achieves sub-second response time on streamed video without invoking Qwen3-4B on every frame.

### Request

```bash
curl -X POST http://localhost:8085/v1/vision/gate/score \
  -H "Content-Type: application/json" \
  -d '{
    "segments": [
      {
        "start_ms": 0,
        "end_ms": 500,
        "perception_tokens": [[0.12, 0.45, ...], ...]   // MageViT tokens for the window
      },
      {
        "start_ms": 500,
        "end_ms": 1000,
        "perception_tokens": [[0.32, 0.78, ...], ...]
      }
    ],
    "threshold": 0.5
  }'
```

| Field | Required | Type | Description |
|---|---|---|---|
| `segments` | ✅ | array | Ordered list of codec windows, each with `start_ms` / `end_ms` (window boundaries) + `perception_tokens` (MageViT output for the window). |
| `threshold` | — | float (0–1) | Optional cutoff; segments above are flagged in the response as "fire generation". |

### Response

```json
{
  "scores": [
    {"start_ms": 0,    "end_ms": 500,  "p_speak": 0.12, "fires": false},
    {"start_ms": 500,  "end_ms": 1000, "p_speak": 0.74, "fires": true}
  ],
  "model": "streammind_gate",
  "device": "cpu"
}
```

The gate ships as `streammind_gate.safetensors` inside the `microsoft/Mage-VL` HF checkpoint. `MageVlRuntime` lazy-loads it on first gate request. Use `MageVlRuntime::reset_cache()` between concurrent streams to avoid KV-cache cross-talk (see the Mage-VL runtime docs in `src/mage_vl/runtime.rs`).

Full Mage-VL text generation then goes through the standard routes — `POST /v1/chat/completions` with `model: "microsoft/Mage-VL"` and OpenAI-shaped `image_url` content, or `POST /v1/messages` for Anthropic Messages v1 shape. Both are backed by the same `MageVlRuntime` registered in `state.engines.llm` when `NEURON_LLM_RUNTIME=mage_vl`.

## Response: detection

```json
{
  "model": "yolov8n",
  "task": "detect",
  "image": {"width": 1920, "height": 1080},
  "detections": [
    {
      "class": "person",
      "score": 0.92,
      "bbox": [120, 240, 180, 360]
    },
    {
      "class": "bicycle",
      "score": 0.71,
      "bbox": [50, 60, 200, 180]
    }
  ]
}
```

`bbox` is `[x, y, w, h]` in pixel coordinates of the original image (after letterbox unstretch).

## Response: classification

```json
{
  "model": "vit-base-patch16-224",
  "task": "classify",
  "predictions": [
    {"class": "tabby_cat", "score": 0.87},
    {"class": "tiger_cat", "score": 0.09},
    {"class": "Egyptian_cat", "score": 0.02}
  ]
}
```

Top-K predictions are sorted by score descending. The default K is 5; configurable via `?top_k=N` query param.

## Response: segmentation

```json
{
  "model": "yolov8n-seg",
  "task": "segment",
  "image": {"width": 1920, "height": 1080},
  "classes": ["person", "bicycle", "car", ...],
  "mask_rle": "...",        // Run-length-encoded class index map
  "mask_shape": [1080, 1920]
}
```

The `mask_rle` is a run-length-encoded uint8 array of class indices. Decode it with `pycocotools.mask.decode(rle)` or any standard RLE library.

## Response: pose

```json
{
  "model": "yolov8n-pose",
  "task": "pose",
  "image": {"width": 1920, "height": 1080},
  "people": [
    {
      "score": 0.85,
      "keypoints": [
        {"name": "nose", "x": 100, "y": 50, "score": 0.95},
        {"name": "left_eye", "x": 95, "y": 45, "score": 0.92},
        ...
      ]
    }
  ]
}
```

COCO 17-keypoint format. `score < 0.3` keypoints are typically filtered client-side.

## Model selection

The dispatch rules in `src/vision/registry.rs`:

| Model family | Backend | Tasks |
|---|---|---|
| `yolov8n` / `yolov8s` / `yolov11n` | candle | detect |
| `rtdetr-r50` / `detr-resnet50` | candle | detect |
| `rf-detr-base` / `rf-detr-large` | ONNX | detect |
| `florence-2-base` | ONNX | detect (vision-language, text-prompted) |
| `vit-base-patch16-224` / `swin-tiny` | ONNX | classify |
| `yolov8n-seg` / `mask-rcnn-r50` | ONNX | segment |
| `yolov8n-pose` | ONNX | pose |

## LoRA adapter extension

Register a LoRA adapter to extend the base model's class list without retraining. This is the pattern for adding custom classes (e.g. "warehouse_pallet", "forklift", "package") on top of a base YOLOv8n trained on COCO 80.

### Step 1: Prepare the LoRA file

The LoRA file is a standard safetensors file with:
- `lora_A.<module_path>` tensors of shape `[rank, in_dim]`
- `lora_B.<module_path>` tensors of shape `[out_dim, rank]`
- A `metadata.json` with the source modules + custom class list

Example `metadata.json`:
```json
{
  "base_model": "yolov8n",
  "rank": 16,
  "alpha": 32.0,
  "modules": ["head.cls.cv3.conv"],
  "custom_classes": ["warehouse_pallet", "forklift", "package"],
  "merge_strategy": "linear"
}
```

### Step 2: Register the adapter

```bash
curl -X POST http://localhost:8085/v1/vision/lora/register \
  -F "name=warehouse-v1" \
  -F "base_model=yolov8n" \
  -F "lora_file=@./warehouse_lora.safetensors" \
  -F "metadata=./metadata.json;type=application/json"
```

Response:
```json
{
  "name": "warehouse-v1",
  "base_model": "yolov8n",
  "status": "Registered",
  "merged_class_count": 83
}
```

### Step 3: Use the adapter

```bash
curl -X POST http://localhost:8085/v1/vision/detect \
  -F "file=@./photo.jpg" \
  -F "model=yolov8n" \
  -F "lora_adapter=warehouse-v1"
```

The detection now returns the original 80 COCO classes plus your 3 custom classes.

### Step 4: List / unregister

```bash
curl http://localhost:8085/v1/vision/lora/list
# → {"adapters": [{"name": "warehouse-v1", "base_model": "yolov8n", "merged_class_count": 83}]}

curl -X POST http://localhost:8085/v1/vision/lora/unregister \
  -H "Content-Type: application/json" \
  -d '{"name": "warehouse-v1"}'
```

## Pre-loading vision models

Vision models are large (YOLOv8n is ~6MB, RF-DETR-large is ~80MB). The first detection request after a model load can take 1-3 seconds. To pre-load, hit the `/v1/vision/detect` endpoint once with a placeholder image — this triggers the model load and warms any caches.

## Common pitfalls

- **Wrong image format**: only PNG / JPEG / WebP are supported. Other formats (TIFF, BMP, HEIC) return 400.
- **Too many detections**: if you don't set `max_detections`, you may get 100+ boxes for cluttered scenes. Filter by score client-side.
- **LoRA on the wrong base model**: if your LoRA was trained on `yolov8s` but you request `yolov8n`, the merge will silently produce nonsense. Make sure `metadata.json::base_model` matches the request.
- **No CUDA for ONNX**: ONNX runtime falls back to CPU if CUDA isn't available, which is ~10x slower. If you need GPU, set `ORT_CUDA=1` before launching neuron.

## Next steps

- [LoRA Adapters](/howto/lora-adapters) — full LoRA workflow with Python training examples
- [Capabilities](/concepts/capabilities) — vision capability surface
- [Architecture](/concepts/architecture) — vision registry internals
