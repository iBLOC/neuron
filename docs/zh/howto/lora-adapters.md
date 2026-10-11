---
title: LoRA 适配器注册
---

# LoRA 适配器注册

LoRA（Low-Rank Adaptation）让你无需重训基础权重即可扩展基础检测模型的类别列表。这是在 COCO 80 训练的 YOLOv8n 之上添加自定义类别（例如 "warehouse_pallet"、"forklift"、"package"）的正确模式。

本指南覆盖完整的生命周期：在 Python 中训练 LoRA、导出 safetensors、在 neuron 中注册，以及使用合并后的模型。

## 数学回顾

标准 LoRA（Hu et al. 2021）：

```
W' = W + (alpha / rank) * B @ A
```

其中：
- `W` 是冻结的基础权重，形状 `[out_dim, in_dim]`
- `A` 是 `[rank, in_dim]`（rank 通常为 16 或 32）
- `B` 是 `[out_dim, rank]`
- `alpha` 是缩放因子（通常为 `2 * rank`）
- `W'` 是合并后的权重，形状与 `W` 相同

对 YOLOv8 而言，LoRA 通常作用于分类头最后一个线性层（`W` 为 `[num_classes, hidden_dim]`）。合并后的头与基础头形状相同，因此可直接替换（drop-in）。

## 在 Python 中训练 LoRA

标准训练流水线由宿主驱动且纯用 PyTorch — neuron 不会派生 Python 子进程，也不关心你使用哪个训练框架。唯一的约定是：LoRA 因子须保存为标准的 safetensors 文件供 neuron 加载。

```python
import torch
from peft import LoraConfig, get_peft_model
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# 1. Load the base detector (any PyTorch detector works — torchvision,
#    timm, your own architecture, etc.). Pick whatever fits the host's
#    training pipeline.
base_model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
head = base_model.roi_heads.box_predictor  # the classification + box head

lora_config = LoraConfig(
    r=16,                     # rank
    lora_alpha=32,            # scaling factor
    target_modules=["cls_score", "bbox_pred"],
    lora_dropout=0.1,
    bias="none",
)
peft_model = get_peft_model(head, lora_config)

# 2. Train on your custom data
peft_model.train()
# ... standard PyTorch training loop on custom dataset ...
# ... save the LoRA factors as a standard safetensors file ...
```

## 导出 LoRA

输出是一个包含 LoRA 因子的 safetensors 文件，外加一个 `metadata.json`：

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

neuron 读取的 `metadata.json` 键：
- `base_model`（必填）— 必须与你请求的模型名一致
- `rank`（必填）— 仅供参考；neuron 会从 `A` 张量形状推断
- `alpha`（必填）— 合并时的缩放因子
- `modules`（必填）— 被适配的层路径列表
- `custom_classes`（可选）— 新类别的名称；仅供参考
- `merge_strategy`（可选，默认 `linear`）— `linear`（B@A * alpha/rank）或其他

## 在 neuron 中注册

```bash
curl -X POST http://localhost:8085/v1/vision/lora/register \
  -F "name=warehouse-v1" \
  -F "base_model=yolov8n" \
  -F "lora_file=@./warehouse_lora.safetensors" \
  -F "metadata=@./metadata.json;type=application/json"
```

响应：
```json
{
  "name": "warehouse-v1",
  "base_model": "yolov8n",
  "status": "Registered",
  "merged_class_count": 83
}
```

`merged_class_count` 是基础模型的类别数加上 `custom_classes` 的数量。对 YOLOv8n（COCO 80）+ 3 个自定义类别，即为 83。

## 使用合并后的模型

注册之后，LoRA 适配器会自动应用于满足以下条件的检测请求：
- 目标为 `base_model`（例如 `yolov8n`）
- 在 `lora_adapter` 表单字段中指定了适配器名

```bash
curl -X POST http://localhost:8085/v1/vision/detect \
  -F "file=@./photo.jpg" \
  -F "model=yolov8n" \
  -F "lora_adapter=warehouse-v1" \
  -F "confidence=0.4"
```

响应同时包含 COCO 类别和你的自定义类别：

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

合并在首个请求时惰性进行 — 注册 LoRA 并不会立即在内存中物化一个新模型。基础模型 + LoRA 在首次使用时合并并缓存。

## 列出与取消注册

```bash
# List all registered LoRA adapters
curl http://localhost:8085/v1/vision/lora/list
# → {"adapters": [{"name": "warehouse-v1", "base_model": "yolov8n", "merged_class_count": 83, "registered_at": "..."}]}

# Unregister
curl -X POST http://localhost:8085/v1/vision/lora/unregister \
  -H "Content-Type: application/json" \
  -d '{"name": "warehouse-v1"}'
```

取消注册会丢弃缓存的合并权重。基础模型保持加载状态（其他请求也在使用它）。

## 性能考量

| 操作 | 延迟 | 说明 |
|---|---|---|
| 首次带 LoRA 的检测 | +50-200ms（一次性合并） | 合并在首次使用时惰性进行，之后走缓存 |
| 后续带 LoRA 的检测 | 与基础模型相同 | 复用缓存的合并权重 |
| 注册新 LoRA | ~10-50ms | 仅元数据 + 张量校验，不加载模型 |
| 取消注册 LoRA | <1ms | 丢弃缓存条目 |

如果你有很多 LoRA（例如每个客户一个），每个都维护自己的缓存合并。检测到内存压力时，缓存按 LRU 淘汰。

## 同一基础模型上的多个 LoRA

你可以针对同一个基础模型注册多个 LoRA，它们互不干扰：

```bash
# Register 3 LoRAs for the same YOLOv8n base
curl ... /v1/vision/lora/register -F "name=warehouse-v1"  -F "base_model=yolov8n" -F "lora_file=@./warehouse.safetensors"
curl ... /v1/vision/lora/register -F "name=retail-v1"     -F "base_model=yolov8n" -F "lora_file=@./retail.safetensors"
curl ... /v1/vision/lora/register -F "name=construction-v1" -F "base_model=yolov8n" -F "lora_file=@./construction.safetensors"

# Switch between them per request via lora_adapter=...
```

## 常见陷阱

- **base_model 错误**：如果你的 LoRA 是在 `yolov8s` 上训练的，却注册到 `yolov8n`，合并会静默产生无意义结果。务必验证 `metadata.json` 中的 `base_model` 字段与你请求的模型一致。
- **层名错误**：如果 `metadata.json` 的 `modules` 字段与 safetensors 实际张量名不匹配，合并会跳过该 LoRA 因子（静默回退到基础权重）。请检查 safetensors 文件中的键。
- **LoRA rank 不匹配**：`metadata.json` 中的 LoRA rank 应与实际 `A` 张量形状一致。如果声明 rank=16 但张量实际是 rank=32，合并会产生错误输出。
- **从不同机器加载**：LoRA 文件与具体架构绑定。在 YOLOv8n 上训练的 LoRA 无法用于 YOLOv8s。`metadata.json` 的 `modules` 字段应列出确切的 YOLOv8 层路径。

## 后续步骤

- [视觉任务](/zh/howto/vision-tasks) — detect/classify/segment/pose 的完整请求/响应结构
- [能力](/zh/concepts/capabilities) — 视觉能力面
- [将 neuron 嵌入 Rust 宿主](/zh/howto/embed-into-host) — 完整的 Tauri 2 演练
