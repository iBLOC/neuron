---
title: LoRA Adapter Registration
---

# LoRA Adapter Registration

LoRA (Low-Rank Adaptation) lets you extend a base detection model's class list without retraining the base weights. This is the right pattern for adding custom classes (e.g. "warehouse_pallet", "forklift", "package") on top of a base YOLOv8n trained on COCO 80.

This how-to covers the full lifecycle: training a LoRA in Python, exporting the safetensors, registering with neuron, and using the merged model.

## Math refresher

Standard LoRA (Hu et al. 2021):

```
W' = W + (alpha / rank) * B @ A
```

where:
- `W` is the frozen base weight, shape `[out_dim, in_dim]`
- `A` is `[rank, in_dim]` (typically rank=16 or 32)
- `B` is `[out_dim, rank]`
- `alpha` is a scaling factor (typically `2 * rank`)
- `W'` is the merged weight with the same shape as `W`

For YOLOv8, LoRA is typically applied to the classification head's final linear layer (where `W` is `[num_classes, hidden_dim]`). The merged head has the same shape as the base head, so it can be a drop-in replacement.

## Training a LoRA in Python

The standard training pipeline is:

```python
import torch
from peft import LoraConfig, get_peft_model
from ultralytics import YOLO

# 1. Load the base YOLOv8 model
base_model = YOLO("yolov8n.pt")

# 2. Add LoRA adapter to the classification head
#    YOLOv8's classification head is in model.model.model[22] (for yolov8n)
#    The final conv is model.model.model[22].cv3.conv
head = base_model.model.model[22]
lora_config = LoraConfig(
    r=16,                     # rank
    lora_alpha=32,            # scaling factor
    target_modules=["cv3.conv"],
    lora_dropout=0.1,
    bias="none",
)
peft_model = get_peft_model(head, lora_config)

# 3. Train on your custom data
#    (use your standard YOLOv8 training loop with the LoRA-augmented head)
peft_model.train()
# ... training loop on custom dataset ...
# ... save the LoRA weights as a standard safetensors file
```

## Exporting the LoRA

The output is a single safetensors file with the LoRA factors plus a `metadata.json`:

```python
# After training, extract the LoRA factors
lora_state_dict = {}
for name, param in peft_model.named_parameters():
    if "lora_A" in name or "lora_B" in name:
        lora_state_dict[name] = param.data

# Save as safetensors
from safetensors.torch import save_file
save_file(lora_state_dict, "warehouse_lora.safetensors")

# Save metadata
import json
metadata = {
    "base_model": "yolov8n",
    "rank": 16,
    "alpha": 32.0,
    "modules": ["model.22.cv3.conv"],   # which YOLOv8 layer was LoRA-adapted
    "custom_classes": ["warehouse_pallet", "forklift", "package"],
    "merge_strategy": "linear"
}
with open("metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
```

The `metadata.json` keys that neuron reads:
- `base_model` (required) — must match the model name you request
- `rank` (required) — informational; neuron infers from `A` tensor shape
- `alpha` (required) — scaling factor for the merge
- `modules` (required) — list of layer paths that were adapted
- `custom_classes` (optional) — names of new classes; informational
- `merge_strategy` (optional, default `linear`) — `linear` (B@A * alpha/rank) or other

## Registering with neuron

```bash
curl -X POST http://localhost:8085/v1/vision/lora/register \
  -F "name=warehouse-v1" \
  -F "base_model=yolov8n" \
  -F "lora_file=@./warehouse_lora.safetensors" \
  -F "metadata=@./metadata.json;type=application/json"
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

`merged_class_count` is the base model's class count plus your `custom_classes` count. For YOLOv8n (COCO 80) + 3 custom classes, this is 83.

## Using the merged model

Once registered, the LoRA adapter is automatically applied to any detection request that:
- Targets the `base_model` (e.g. `yolov8n`)
- Specifies the adapter name in the `lora_adapter` form field

```bash
curl -X POST http://localhost:8085/v1/vision/detect \
  -F "file=@./photo.jpg" \
  -F "model=yolov8n" \
  -F "lora_adapter=warehouse-v1" \
  -F "confidence=0.4"
```

The response includes both COCO classes and your custom classes:

```json
{
  "model": "yolov8n",
  "task": "detect",
  "image": {"width": 1920, "height": 1080},
  "detections": [
    {"class": "person", "score": 0.91, "bbox": [120, 240, 180, 360]},
    {"class": "warehouse_pallet", "score": 0.78, "bbox": [400, 500, 600, 750]},
    {"class": "forklift", "score": 0.62, "bbox": [800, 400, 950, 700]}
  ]
}
```

The merge happens lazily on first request — registering a LoRA does not immediately materialize a new model in memory. The base + LoRA are merged and cached on first use.

## Listing and unregistering

```bash
# List all registered LoRA adapters
curl http://localhost:8085/v1/vision/lora/list
# → {"adapters": [{"name": "warehouse-v1", "base_model": "yolov8n", "merged_class_count": 83, "registered_at": "..."}]}

# Unregister
curl -X POST http://localhost:8085/v1/vision/lora/unregister \
  -H "Content-Type: application/json" \
  -d '{"name": "warehouse-v1"}'
```

Unregistering drops the cached merged weight. The base model stays loaded (it's used by other requests too).

## Performance considerations

| Operation | Latency | Notes |
|---|---|---|
| First detection with LoRA | +50-200ms (one-time merge) | Merge happens lazily on first use, cached after |
| Subsequent detections with LoRA | Same as base | Cached merged weight is reused |
| Register new LoRA | ~10-50ms | Just metadata + tensor validation, no model load |
| Unregister LoRA | <1ms | Drop the cache entry |

If you have many LoRAs (e.g. one per customer), each maintains its own cached merge. The cache is LRU-evicted if memory pressure is detected.

## Multiple LoRAs on the same base

You can register multiple LoRAs against the same base model. They don't interfere:

```bash
# Register 3 LoRAs for the same YOLOv8n base
curl ... /v1/vision/lora/register -F "name=warehouse-v1"  -F "base_model=yolov8n" -F "lora_file=@./warehouse.safetensors"
curl ... /v1/vision/lora/register -F "name=retail-v1"     -F "base_model=yolov8n" -F "lora_file=@./retail.safetensors"
curl ... /v1/vision/lora/register -F "name=construction-v1" -F "base_model=yolov8n" -F "lora_file=@./construction.safetensors"

# Switch between them per request via lora_adapter=...
```

## Common pitfalls

- **Wrong base_model**: if your LoRA was trained on `yolov8s` but you register against `yolov8n`, the merge will silently produce nonsense. Always verify the `base_model` field in `metadata.json` matches what you request.
- **Wrong layer names**: if the `modules` field in `metadata.json` doesn't match the actual safetensors tensor names, the merge will skip the LoRA factor (silently fall back to base weight). Check the keys in your safetensors file.
- **LoRA rank mismatch**: the LoRA rank in `metadata.json` should match the actual `A` tensor shape. If you say rank=16 but the tensor is rank=32, the merge produces wrong outputs.
- **Loading from a different machine**: LoRA files are architecture-specific. A LoRA trained on YOLOv8n won't work with YOLOv8s. The `modules` field in `metadata.json` should list the exact YOLOv8 layer paths.

## Next steps

- [Vision Tasks](/howto/vision-tasks) — full request/response shape for detect/classify/segment/pose
- [Capabilities](/concepts/capabilities) — vision capability surface
- [Embed neuron into a Rust Host](/howto/embed-into-host) — full Tauri 2 walkthrough
