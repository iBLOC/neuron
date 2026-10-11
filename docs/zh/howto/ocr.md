---
title: OCR（文本检测 + 识别）
---

# OCR（文本检测 + 识别）

neuron 在 `POST /v1/ocr/recognize` 上提供两阶段 PP-OCR 流水线（DB++ 文本检测 + SVTR-LCNet 识别），并支持可选的文档版面分析与表格结构识别。本指南涵盖安装、模型准备以及完整的请求/响应面。

## 1. 选择构建 feature

```toml
[dependencies]
neuron = { git = "https://github.com/iBLOC/neuron", default-features = false, features = ["tract"] }
# or, for the fast executor:
neuron = { git = "https://github.com/iBLOC/neuron", default-features = false, features = ["ort"] }
```

| Feature | 执行器 | 说明 |
|---|---|---|
| `tract` | 纯 Rust ONNX（默认） | 零系统依赖。CPU 上比 onnxruntime 慢 5-12×。 |
| `ort` | 经 `ort` crate 调用 onnxruntime | 端到端快约 10×。使用 **load-dynamic**：宿主通过 `ORT_DYLIB_PATH`（或 `PATH`）提供 `onnxruntime.dll`/`.so`。 |

两种执行器共享相同的线上格式（wire format），产生的文本逐字节一致。

## 2. 准备模型目录

从 HuggingFace 下载官方 ONNX 导出（Apache-2.0），并运行 neuron 的 prep 脚本（在 neuron 检出目录中执行）— 这是一次仅改元数据的重写（Paddle2ONNX 生成的点分符号名与 `floor(...)` 形状提示会被 tract 的形状解析器拒绝）：

```bash
pip install onnx
curl -LO https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det_onnx/resolve/main/inference.onnx
curl -LO https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx/resolve/main/inference.onnx
curl -LO https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/dict/ppocrv6_dict.txt
python scripts/prep-ppocr-onnx.py inference.onnx det.prepped.onnx   # from the det download
python scripts/prep-ppocr-onnx.py inference.onnx rec.prepped.onnx   # from the rec download
```

（`ort` 执行器在技术上可以加载未经 prep 的导出，但请始终执行 prep — 让两种执行器共用同一份产物。）

按如下方式组织目录（只有前三个文件是必需的）：

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

模型来源（均为 HuggingFace 上的 `PaddlePaddle/*`，Apache-2.0）：

| 文件 | 仓库 | 大小 |
|---|---|---|
| det / rec | `PP-OCRv5_{mobile,server}_{det,rec}_onnx`、`PP-OCRv6_medium_{det,rec}_onnx` | 5-88 MB |
| layout | `PP-DocLayoutV3_onnx` | ~130 MB |
| table | `SLANeXt_wired_onnx`（有边框的表格） | ~368 MB |

::: warning 不要使用 SLANet_plus
广为流传的 `SLANet_plus_onnx` 导出是坏的 — 其自回归解码器不依赖图像（白噪声与真实表格解码出相同的结果）。表格结构的可行路径是 SLANeXt。
:::

## 3. 注册模型

OCR 注册由配置驱动（capability `ocr`，engine `paddle-ocr`）：

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

通过 `NEURON_OCR_MODEL_PATH` 环境变量把运行时指向该目录（或将其放到 `<model_root>/ocr/ppocrv6-medium/` 下）。

## 4. 识别文本

```bash
curl -X POST http://localhost:8085/v1/ocr/recognize \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ppocrv6-medium",
    "image": { "type": "image_url", "image_url": { "url": "file:///tmp/invoice.png" } }
  }'
```

`image` 字段接受与视觉端点相同的形状（`http(s)`/`data:`/`file://` URL 形式的 `image_url` 或本地路径，或 base64）。响应：

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

`regions` 按阅读顺序排列（行自上而下，行内自左向右）；`text` 是以换行符连接的各行文本。

## 5. 其他语言

可选的 `rec-<pack>/` 子目录存放按文字系统划分的识别模型。请求的 `languages` 字段会路由到其中一个语种包：

| 请求值 | 语种包 | 模型 |
|---|---|---|
| `zh` / `en` / `ja`（默认） | 默认 rec | `PP-OCRv5/v6_*_rec_onnx` |
| `fr`、`de`、`es` 等 | `rec-latin` | `latin_PP-OCRv5_mobile_rec_onnx` |
| `ko` | `rec-korean` | `korean_PP-OCRv5_mobile_rec_onnx` |
| `ru`、`uk` 等 | `rec-cyrillic` | `cyrillic_PP-OCRv5_mobile_rec_onnx` |
| `ar`、`fa` 等 | `rec-arabic` | `arabic_PP-OCRv5_mobile_rec_onnx` |
| `hi`、`mr`、`ne` | `rec-devanagari` | `devanagari_PP-OCRv5_mobile_rec_onnx` |

直接给出精确的语种包名（`"latin"`）也可作为逃生通道。语种包词典应使用 PaddleOCR 仓库中的 `ppocr/utils/dict/ppocrv5_<script>_dict.txt` — 较旧的 `<script>_dict.txt` 文件与 v5 导出不匹配（运行时会以明确的词表大小错误拒绝错误配对）。

```bash
curl -X POST http://localhost:8085/v1/ocr/recognize \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ppocrv6-medium",
    "languages": ["fr"],
    "image": { "type": "image_url", "image_url": { "url": "file:///tmp/facture.png" } }
  }'
```

响应中的 `languages` 字段报告实际使用的语种包。

## 6. 版面与表格（可选）

存在 `layout.onnx` 时（ort 执行器），响应会增加一个 `layout[]` 数组 — 25 个区域类别（`text`、`table`、`image`、`display_formula`、`doc_title` 等）及各自的框与分数 — 并且每个文本行都带有一个 `kind`，标明其所属区域。若额外存在 `table.onnx`，`table` 区域还会携带组装好的 `html` 字段，识别出的文本会填入各单元格。

```json
{
  "layout": [
    { "kind": "table", "bbox": { "x": 78, "y": 377, "w": 524, "h": 205 }, "score": 0.96,
      "html": "<table><tr><td>Region</td><td>Q1</td></tr>…</table>" }
  ]
}
```

## 配置项

| 环境变量 | 默认值 | 用途 |
|---|---|---|
| `NEURON_OCR_MODEL_PATH` | — | 模型目录（回退到 `<model_root>/ocr/<id>`） |
| `NEURON_OCR_EXECUTOR` | `tract` | `tract` \| `ort` |
| `NEURON_OCR_SESSIONS` | `2` | 每个图的 ort 会话数（并发） |
| `NEURON_OCR_LAYOUT` | `true` | `false` 跳过 `layout.onnx` |
| `NEURON_OCR_TABLES` | `true` | `false` 跳过 `table.onnx` |
| `ORT_DYLIB_PATH` | — | `ort` 执行器使用的 onnxruntime 库 |

## 后续步骤

- [/reference/api](/zh/reference/api) — 完整的请求/响应 schema
- [/concepts/scheduler](/zh/concepts/scheduler) — 内存受限的模型换入换出
