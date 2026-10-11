---
title: 视觉任务（detect / classify / segment / pose）
---

# 视觉任务（detect / classify / segment / pose）

neuron 在 `/v1/vision/*` 下暴露 4 个视觉任务端点，并分发到相应的模型家族。本指南覆盖完整的请求/响应结构、模型选择规则以及 LoRA 扩展机制。

## 4 种任务

| 端点 | 任务 | 输出 |
|---|---|---|
| `POST /v1/vision/detect` | 目标检测 | 边界框 + 类别 + 分数 |
| `POST /v1/vision/classify` | 图像分类 | 类别概率（top-K） |
| `POST /v1/vision/segment` | 语义分割 | 逐像素类别图 |
| `POST /v1/vision/pose` | 人体关键点估计 | 关键点 + 骨架 |

4 个端点都接受相同的 `multipart/form-data` 请求格式，附带图像上传。

## 请求格式

```bash
curl -X POST http://localhost:8085/v1/vision/detect \
  -F "file=@./photo.jpg" \
  -F "model=yolov8n" \
  -F "confidence=0.5" \
  -F "iou=0.45" \
  -F "lora_adapter=warehouse-v1"   # optional
```

| 表单字段 | 必填 | 类型 | 默认值 | 说明 |
|---|---|---|---|---|
| `file` | ✅ | binary | — | 图像字节（PNG、JPEG、WebP） |
| `model` | ✅ | string | — | 已注册的模型 ID 之一（例如 `yolov8n`、`rtdetr-r50`） |
| `confidence` | — | float | `0.25` | 最低置信度阈值（0.0-1.0） |
| `iou` | — | float | `0.45` | NMS 的 IoU 阈值（仅检测） |
| `lora_adapter` | — | string | — | 已注册 LoRA 适配器的名称（见下文） |
| `max_detections` | — | integer | `100` | 返回检测数的上限（仅检测） |

## StreamMind 门控（Mage-VL）

```
POST /v1/vision/gate/score
```

Mage-VL StreamMind 事件门控对滚动编解码窗口打分并输出 `p_speak ∈ [0, 1]`：窗口中包含值得回应事件的可能性。客户端轮询门控，仅在 `p_speak` 越过阈值时才调用完整的 VLM 生成 — 这就是 Mage-VL 在流式视频上实现亚秒级响应、而无需对每一帧都调用 Qwen3-4B 的方式。

### 请求

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

| 字段 | 必填 | 类型 | 说明 |
|---|---|---|---|
| `segments` | ✅ | array | 有序的编解码窗口列表，每个窗口含 `start_ms` / `end_ms`（窗口边界）+ `perception_tokens`（该窗口的 MageViT 输出）。 |
| `threshold` | — | float (0–1) | 可选的截断阈值；高于该值的段会在响应中标记为 "fire generation"（触发生成）。 |

### 响应

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

门控以 `streammind_gate.safetensors` 的形式随 `microsoft/Mage-VL` HF checkpoint 一起发布。`MageVlRuntime` 在首个门控请求时惰性加载它。在并发流之间请使用 `MageVlRuntime::reset_cache()` 以避免 KV 缓存串扰（见 `src/mage_vl/runtime.rs` 中的 Mage-VL 运行时文档）。

完整的 Mage-VL 文本生成随后走标准路由 — `POST /v1/chat/completions`，`model: "microsoft/Mage-VL"` 且使用 OpenAI 形状的 `image_url` 内容；或 `POST /v1/messages`，采用 Anthropic Messages v1 形状。当 `NEURON_LLM_RUNTIME=mage_vl` 时，两者都由注册在 `state.engines.llm` 中的同一个 `MageVlRuntime` 支持。

## 响应：检测

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

`bbox` 是原始图像像素坐标下的 `[x, y, w, h]`（letterbox 拉伸已还原）。

## 响应：分类

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

Top-K 预测按分数降序排列。默认 K 为 5；可通过 `?top_k=N` 查询参数配置。

## 响应：分割

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

`mask_rle` 是类别索引的行程编码（RLE）uint8 数组。可用 `pycocotools.mask.decode(rle)` 或任何标准 RLE 库解码。

## 响应：姿态

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

COCO 17 关键点格式。`score < 0.3` 的关键点通常由客户端过滤。

## 模型选择

`src/vision/registry.rs` 中的分发规则：

| 模型家族 | 后端 | 任务 |
|---|---|---|
| `yolov8n` / `yolov8s` / `yolov11n` | candle | 检测 |
| `rtdetr-r50` / `detr-resnet50` | candle | 检测 |
| `rf-detr-base` / `rf-detr-large` | ONNX | 检测 |
| `florence-2-base` | ONNX | 检测（视觉语言，文本提示） |
| `vit-base-patch16-224` / `swin-tiny` | ONNX | 分类 |
| `yolov8n-seg` / `mask-rcnn-r50` | ONNX | 分割 |
| `yolov8n-pose` | ONNX | 姿态 |

## LoRA 适配器扩展

注册 LoRA 适配器可以在不重训的情况下扩展基础模型的类别列表。这是在 COCO 80 训练的 YOLOv8n 之上添加自定义类别（例如 "warehouse_pallet"、"forklift"、"package"）的正确模式。

### 第 1 步：准备 LoRA 文件

LoRA 文件是标准的 safetensors 文件，包含：
- 形状为 `[rank, in_dim]` 的 `lora_A.<module_path>` 张量
- 形状为 `[out_dim, rank]` 的 `lora_B.<module_path>` 张量
- 一个带来源模块 + 自定义类别列表的 `metadata.json`

`metadata.json` 示例：
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

### 第 2 步：注册适配器

```bash
curl -X POST http://localhost:8085/v1/vision/lora/register \
  -F "name=warehouse-v1" \
  -F "base_model=yolov8n" \
  -F "lora_file=@./warehouse_lora.safetensors" \
  -F "metadata=./metadata.json;type=application/json"
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

### 第 3 步：使用适配器

```bash
curl -X POST http://localhost:8085/v1/vision/detect \
  -F "file=@./photo.jpg" \
  -F "model=yolov8n" \
  -F "lora_adapter=warehouse-v1"
```

检测现在会返回原有的 80 个 COCO 类别加上你的 3 个自定义类别。

### 第 4 步：列出 / 取消注册

```bash
curl http://localhost:8085/v1/vision/lora/list
# → {"adapters": [{"name": "warehouse-v1", "base_model": "yolov8n", "merged_class_count": 83}]}

curl -X POST http://localhost:8085/v1/vision/lora/unregister \
  -H "Content-Type: application/json" \
  -d '{"name": "warehouse-v1"}'
```

## 预加载视觉模型

视觉模型体积较大（YOLOv8n 约 6MB，RF-DETR-large 约 80MB）。模型加载后的首个检测请求可能耗时 1-3 秒。要预加载，可携带一张占位图像请求一次 `/v1/vision/detect` — 这会触发模型加载并预热各类缓存。

## 常见陷阱

- **图像格式错误**：仅支持 PNG / JPEG / WebP。其他格式（TIFF、BMP、HEIC）返回 400。
- **检测数过多**：如果不设置 `max_detections`，杂乱场景可能返回 100+ 个框。可在客户端按分数过滤。
- **LoRA 用在错误的基础模型上**：如果你的 LoRA 是在 `yolov8s` 上训练的，却请求 `yolov8n`，合并会静默产生无意义结果。务必确保 `metadata.json::base_model` 与请求一致。
- **ONNX 没有 CUDA**：CUDA 不可用时，ONNX 运行时会回退到 CPU，慢约 10 倍。如果需要 GPU，在启动 neuron 前设置 `ORT_CUDA=1`。

## 后续步骤

- [LoRA 适配器](/zh/howto/lora-adapters) — 完整的 LoRA 工作流（含 Python 训练示例）
- [能力](/zh/concepts/capabilities) — 视觉能力面
- [架构](/zh/concepts/architecture) — 视觉注册表内部机制
