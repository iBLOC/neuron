---
title: 基准测试
---

# 基准测试

neuron 各推理路径的实测性能与内存数据。**同一能力内的所有行来自同一台
主机、同一负载**,模型之间可直接横向对比;带脚注的行来自较早一轮同级别
主机实测——表内可比,跨表不可比。每张表都标注机器、代码版本与工作负载,
复现命令见文末。

以下所有表的主机:Windows 11,12 核,16 GB 内存,RTX 3090(CUDA 行需要
`--features cuda`,不在默认构建内)。

## LLM 解码吞吐(candle 引擎,贪心解码)

| 模型 | 参数量 | 设备 | 精度 | TTFT p50 (ms) | 解码 tok/s (p50) | 峰值 RSS (MB) |
|---|---|---|---|---:|---:|---:|
| qwen2.5-0.5b | 0.5B | cpu | f32 | 432 | 5.57 | 1 940 |
| qwen3-0.6b | 0.6B | cpu | f32 | 645 | 4.77 | 3 198 |
| minicpm5-2b ¹ | 2.52B | cpu | f16 | 3 336 | 2.58 | 7 451 |
| minicpm5-2b ¹ | 2.52B | cuda | bf16 | 103 | 26.45 | 2 432 |

¹ 2026-09-27 实测(rev `732f5c8`,12 token 解码,64/256 词提示);其余行
2026-10-11 实测(rev `952a251`,32 token 解码)。同一主机、同一方法论
(客户端计时、预热丢弃、唯一提示头)——minicpm5 两行作为方向性参考,
qwen 两行是当前代码的基线。

解读:解码速率是稳态生成速度;CUDA bf16 需要 `--features cuda`(不在
默认构建内)。candle 的 CPU 后端无 bf16 matmul,且当前构建的 CPU 路径
拒绝 f16 logits,因此 CPU 行均为 f32(内存最高)。目录中更大的模型
(llama-3-8b、mistral-7b、chatglm-3-6b、Gemma 系)可用同一工具实测,
数据暂未发布。

## OCR 端到端时延(PP-OCR 档位 × 执行器)

| 模型 | 检测+识别权重 | tract(冷启动端到端,ms) | ort(冷启动端到端,ms) | ort 加速比 | 区域数 |
|---|---|---:|---:|---:|---:|
| ppocrv5-mobile | 21 MB | 2 922 | 588 | 5.0× | 5 |
| ppocrv6-medium | 139 MB | 7 924 | 1 056 | 7.5× | 5 |
| ppocrv5-server | 173 MB | 13 160 | 1 362 | 9.7× | 5 |

12 次运行全部使用同一张 800×400 中英混排页,单次进程(数字包含执行器
首次初始化:tract 编译 plan,ort 加载 onnxruntime)。文本输出跨执行器、
跨档位完全一致。时延随模型档位上升;ort 执行器能收回 mobile 与 server
档位之间的大部分差距(588 → 1 362 ms,而 tract 上是 2 922 → 13 160 ms)。

## ASR 时延(SenseVoiceSmall,sherpa 引擎)

| 音频 | 时长 | p50 (ms) | RTF |
|---|---:|---:|---:|
| en.wav | 7.15 s | 482.9 | 0.068 |
| zh.wav | 5.59 s | 361.4 | 0.065 |

RTF < 0.1 = 10 倍实时。tract 路径输出与 sherpa 转写一致(2026-09-26
验证);本轮未采集其逐条时延。

## Vision TensorPool(内存分配)

| 路径 | 开销 |
|---|---|
| 直接 `Tensor::zeros(1,3,640,640)` | 约 460 µs |
| 池化 get+release | 约 292 ns(约 1574×) |

可池化中间张量约 32 MB,对比无池化峰值约 200 MB(约 6×)。

## 静态权重 / 内存占用

| 模型 | 权重 | 常驻内存 |
|---|---|---|
| minicpm5-1b | 约 2 GB(bf16) | bf16 约 3 GB 显存;CPU f32 见上表 |
| minicpm5-2b | 约 5 GB(bf16) | bf16 约 6-7 GB 显存;f32 约 10 GB 内存 |
| voxcpm2(TTS) | 4.58 GB bf16 + 377 MB fp32 | 约 10 GB 内存(f32 具象化) |
| laya(决策) | 约 500 MB int8 ONNX | int8 |
| PP-OCRv5 mobile(OCR) | 21 MB | |
| PP-OCRv5 server(OCR) | 173 MB | |
| PP-OCRv6 medium(OCR) | 139 MB | +130 MB 版面、+368 MB 表格(可选) |
| all-minilm-l6-v2 | 约 88 MB | |
| bge-reranker-base | 约 1.08 GB | |
| yolov8n | 约 6 MB | 可池化中间张量约 32 MB |

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
