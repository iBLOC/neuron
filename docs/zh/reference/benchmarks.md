---
title: 基准测试
---

# 基准测试

neuron 各推理路径的实测性能与内存数据。每张表都标注机器、代码版本与
工作负载——不同硬件的数字不可直接比较。复现命令见文末。

## LLM 解码吞吐(MiniCPM5-2B,对比参考引擎)

数据来源:`benches/llm_speed.rs` 报告,2026-09-27 生成于 Windows 11 /
12 核 / 16 GB 内存 / RTX 3090,neuron rev `732f5c8`(candle `aebc405`)。
全部使用同一 MiniCPM5-2B 权重;贪心解码,输出 12 token,唯一化提示头
(防前缀缓存干扰)。

| 框架 | 设备 | 权重 | TTFT ms (p50) | 解码 tok/s (p50) | 峰值 RSS MB |
|---|---|---|---:|---:|---:|
| neuron(candle) | cpu | f16 | 3336 | 2.58 | 7451 |
| neuron(candle) | cuda | bf16 | 103 | 26.45 | 2432 |
| neuron(candle) | cpu | f16 | 8688 | 1.63 | 7451 |

(提示词 64 / 256 词两行,详见源报告。)

同机参照(见 `AGENTS.md` → candle 升级 A/B):llama.cpp CUDA q8_0 在同类
模型上约 131-142 tok/s——neuron 的 candle CPU/CUDA 路径以换取向进程内嵌
入与线格式兼容。

**内存**:2B 模型 f16 权重峰值 RSS 约 7.5 GB;CUDA bf16 约 2.4 GB。
candle 的 CPU 后端无 bf16 matmul;F32 会让内存翻倍(约 10 GB,16 GB
主机上会换页——设置 `NEURON_LLM_DTYPE`)。

## ASR 时延(SenseVoiceSmall)

数据来源:`benches/asr_speed.rs` 报告(sherpa 路径;2026-09-27,同机)。

| 音频 | 时长 | 引擎 | p50 ms | RTF |
|---|---:|---|---:|---:|
| en.wav | 7.15 s | sherpa | 482.9 | 0.068 |
| zh.wav | 5.59 s | sherpa | 361.4 | 0.065 |

RTF < 0.1 = 10 倍实时。tract 路径输出与 sherpa 转写一致(2026-09-26
验证,见 `scripts/prep-sensevoice-onnx.py` 文档)。

## OCR:tract 与 ort 执行器对比(PP-OCRv6 medium)

数据来源:2026-10-09 同级 Windows 主机 A/B,官方 `PaddlePaddle/*_onnx`
导出,release 构建:

| 工作负载 | tract | ort(onnxruntime) | 加速比 |
|---|---:|---:|---:|
| 检测 736² 前向 | 3674 ms | 545 ms | 6.7× |
| 识别 48×320 前向 | 120-233 ms | 10-35 ms | 6-12× |
| **端到端整页**(800×400,检测 + 5 行) | 5418 ms | **526 ms** | **10.3×** |

端到端含执行器首次初始化。两种执行器的文本输出逐字节一致。内存:检测
62 MB + 识别 77 MB 权重;ort 运行时额外加载 onnxruntime。

## Vision TensorPool(内存分配)

数据来源:`benches/mobile_bench.rs`(YOLOv8n 640×640 预分配):

| 路径 | 开销 |
|---|---|
| 直接 `Tensor::zeros(1,3,640,640)` | 约 460 µs |
| 池化 get+release | 约 292 ns(约 1574×) |

可池化中间张量约 32 MB,对比无池化峰值约 200 MB(约 6×)。

## 模型文件 / 内存占用

静态权重(磁盘)与已记录的常驻内存:

| 模型 | 权重 | 常驻内存 |
|---|---|---|
| minicpm5-1b | 约 2 GB(bf16) | bf16 约 3 GB 显存;CPU f16 更高 |
| minicpm5-2b | 约 5 GB(bf16) | bf16 约 6-7 GB 显存;f32 约 10 GB 内存 |
| voxcpm2(TTS) | 4.58 GB bf16 + 377 MB fp32 | 约 10 GB 内存(f32 具象化) |
| laya(决策) | 约 500 MB int8 ONNX | int8 |
| PP-OCRv6 medium(OCR) | 62 + 77 MB | 较小;+130 MB 版面、+368 MB 表格 |
| all-minilm-l6-v2 | 约 88 MB | |
| bge-reranker-base | 约 1.08 GB | |
| yolov8n | 约 6 MB | 可池化中间张量约 32 MB |

candle 其余 LLM 档位、litert Gemma 系列与 vision 时延的逐模型
tok/s 数据**暂未发布**——下面的基准工具可按需产出。

## 复现方法

```bash
# LLM TTFT / 解码 tok/s / 峰值 RSS,对比 vLLM / ollama / llama.cpp
# (自动启动已安装的服务;报告 -> target/llm-speed-compare.md)
./scripts/run-llm-speed-compare.sh

# 能力 × 引擎矩阵,对运行中的 neuron_server :18083
# (报告 -> target/model-matrix.md)
cargo bench --bench model_matrix --features "candle,tract,sherpa,voxcpm"

# ASR tract 与 sherpa 对比(报告 -> target/asr-speed-compare.md)
cargo bench --bench asr_speed --features tract,sherpa

# 内存分配 / 池化微基准
cargo bench --bench mobile_bench
cargo bench --bench perf -- --quick
```

方法论说明(提示词防缓存、客户端计时、预热丢弃、贪心解码)在各报告
头部与 `benches/llm_speed.rs`、`AGENTS.md` 中有完整约定。

## 下一步

- [/zh/reference/models](/zh/reference/models) — 支持的模型 × 后端矩阵
- [/zh/concepts/scheduler](/zh/concepts/scheduler) — 内存预算与这些数字的权衡
