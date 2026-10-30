---
title: 基准测试
---

# 基准测试

neuron 各推理路径的实测速度、延迟、内存与精度数据 —— 每行一个
**模型 × 后端** 组合,同一主机采集,表内数字可直接横向对比。

::: tip 测量方法
每张表都标明机器、修订版本和工作负载。客户端计时、丢弃预热、贪婪解码、
唯一 prompt 前缀(防缓存)。跨批次数据以脚注标注 —— 只在表内对比,
不要跨表对比。文末附复现命令。
:::

所有表格的主机:Windows 11,12 核,16 GB 内存,RTX 3090 24 GB
(CUDA 行需要 `--features cuda`,默认构建不含)。

## 按模型全量扫描 —— 目录里每个模型一行(2026-10-29)

`benches/model_bench.rs` 按能力目录**为每个模型生成一个用例**,并通过
公开的注册表 API(`register` → `runtime_for`)**进程内**跑在本地设备上。
不经过 HTTP、不启动服务 —— 这些就是库自己的设备计时。主机上缺权重的
模型会标成 `skipped` 并给出原因,所以这份报告同时是一张覆盖矩阵。

完整报告:[`target/model-bench-cuda.md`](https://github.com/iBLOC/neuron/blob/main/target/model-bench-cuda.md)
(用 `scripts\cuda-build.bat cargo bench --bench model_bench
--features "candle,cuda,tract,ocr,sherpa,voxcpm,chronos2,litert"` 重新生成)。

实测:**41 个用例 —— 17 个测出数据,24 个跳过(缺权重),0 个失败。**
每个用例跑在独立的 tokio task 里,引擎 panic 会变成一行 `failed` 并附上
panic 内容,不会杀掉整个扫描。

### LLM(candle,CUDA,贪婪解码,生成 64 个新 token,f32 权重)

| 模型 | 参数量 | decode tok/s | p50 (ms) | 备注 |
|---|---:|---:|---:|---|
| minicpm5-1b | 1B | **62.4** | 1 099 | 平台默认;本组最快 |
| qwen3-0.6b | 0.6B | 60.7 | 1 131 | |
| qwen2.5-0.5b | 0.5B | 54.5 | 538 | |
| minicpm5-2b | 2.5B | 35.2 | 1 881 | |

### Embedding(candle,CUDA,8 条短文本)

| 模型 | texts/s | p50 (ms) |
|---|---:|---:|
| all-minilm-l6-v2 | 664.9 | 12.8 |
| bge-base-en-v1.5 | 213.3 | 38.3 |
| e5-base-v2 | 719.2 | 11.0 |
| gte-base | 825.3 | 9.5 |

### ASR(RTF = 音频秒数 / 墙钟秒数,越大越好)

| 模型 | 引擎 | RTF | p50 (ms) | 备注 |
|---|---|---:|---:|---|
| sensevoice-small | sherpa | 19.1 | 106 | int8 sherpa-onnx 图 |
| whisper-small | candle | 9.6 | 199 | CUDA |
| whisper-tiny | candle | 15.8 | 126 | CUDA |

### Vision / OCR(ONNX 会话路径,tract 执行器,640×640)

| 模型 | 引擎 | images/s | p50 (ms) | 备注 |
|---|---|---:|---:|---|
| yolov11n | ort | 2.50 | 399 | |
| yolov8n | ort | 2.44 | 412 | |
| yolov8s | ort | 1.17 | 844 | |
| ppocrv5-mobile | paddle-ocr | 1.82 | 533 | det+rec |

### Rerank(1 query × 8 documents)

| 模型 | 引擎 | docs/s | p50 (ms) | 备注 |
|---|---|---:|---:|---|
| mxbai-rerank-base-v2 | mxbai | **37.0** | 215 | 生成式 yes/no 打分器;4 篇文档方向性验证通过(相关 → "1" ≈ 1.00,不相关 → "0" ≈ 0.00) |
| jina-reranker-base-v2 | jina | 15.7 | 509 | flash checkpoint 经键重映射加载;方向已验证(相关 0.83,不相关 ≈ 0.03) |
| bge-reranker-base | candle | 16.2 | 495 | XLM-RoBERTa 交叉编码器 |

### TTS(burn,CPU)

| 模型 | 引擎 | p50 (ms) | 备注 |
|---|---|---:|---|
| voxcpm2 | burn | 251 191 | **CPU** —— burn 的 CUDA 后端要链 `nccl.lib`,而 CUDA toolkit 不带;而且实测在这张卡上比 CPU 还慢约 6 倍(逐 patch 解码是延迟瓶颈) |

::: warning 这轮扫描查出三个真 bug,均已修复
1. **CUDA context 绑定在加载线程上。** `chat_stream` 把生成放在新起的
   线程里,而 cudarc 的 primary context 是按线程绑定的 —— 该线程上每次
   CUDA 前向都报 `CUDA_ERROR_INVALID_CONTEXT`,SSE 体是空的,而非流式
   路径正常。qwen3-0.6b 修前实测 **22 tok/s**,修后 **54.7 tok/s**。
2. **Whisper 的 `mel_filters.safetensors` 读成了虚构格式**(一个
   `ET\0\0` 魔数后面接裸 f32),而不是标准 safetensors 布局,于是官方
   下载源(`spaces/lmz/candle-whisper`)的文件直接被拒。
3. **rerank 目录把 mxbai 和 jina 指到了 BGE 引擎。** 三个 reranker
   checkpoint 是三种不同架构(XLM-RoBERTa / 带自定义头的 Qwen2 /
   XLMRoberta),两个非 BGE 行因此加载失败。现在它们各自声明自己的引擎。

扫描同时确认:**candle 的 YOLOv8 后端无法消费 Ultralytics 导出文件**
(架构是近似实现、没有 PAN-FPN neck),所以视觉行走 ONNX 路径
(导出脚本 `scripts/export-yolo-onnx.py`)。

最大的一条:**平台默认模型 minicpm5-1b 在 candle llama 路径上无法加载**
—— candle 的 `llama::Config` 没有 `head_dim` 字段,q_proj 永远按方阵建,
而 MiniCPM5-1B 的注意力比 hidden 宽(16 头 × 128 = 2048 vs hidden 1536)。
**已于 2026-10-30 修复**:`src/llm/vendor/llama_hd.rs` vendor 了带显式
`head_dim` 的 llama(字段缺省时与上游行为一致),并补上宽注意力需要的
注意力输出 reshape;平台默认模型现在在本卡跑出 **62.4 tok/s** —— 全组
最快。
:::

## LLM 解码吞吐(candle 引擎)

| 模型 | 参数量 | 设备 | 精度 | TTFT p50 (ms) | 解码 tok/s (p50) | 峰值内存 (MB) |
|---|---|---|---|---:|---:|---:|
| qwen2.5-0.5b | 0.5B | cpu | f32 | 432 | 5.57 | 1 940 |
| qwen3-0.6b | 0.6B | cpu | f32 | 645 | 4.77 | 3 198 |
| minicpm5-2b ¹ | 2.52B | cpu | f16 | 3 336 | 2.58 | 7 451 |
| minicpm5-2b ¹ | 2.52B | cuda | bf16 | 103 | 26.45 | 2 432 |

¹ 2026-09-27 测得(rev `732f5c8`,12 token 解码);其余行 2026-10-11
(rev `952a251`,32 token 解码)。同主机同方法。

解读:解码速率是稳态生成速度。candle 的 CPU 后端没有 bf16 matmul,
所以 CPU 行是 f32(内存最高)。更大的目录模型(llama-3-8b、mistral-7b、
chatglm-3-6b)可用同一 harness 测量,数字尚未发布。

## LLM 跨引擎对比(同机、跨框架)

| 技术栈 | 模型 / 量化 | 设备 | 解码 tok/s |
|---|---|---|---:|
| llama.cpp ²(对照) | q8_0 | cuda | 131–143 |
| neuron / candle | 1B bf16 | cuda | 26–38 |
| neuron / mistralrs | 1B Q4_K (ISQ) | cpu | ~6.7 |
| neuron / candle | 1B f16 | cpu | 1.6–2.6 |

² llama.cpp **不是** neuron 的一部分 —— 它是 candle 0.11 升级 A/B
测试里的对照行,用来标定同一块 GPU 的上限。candle CUDA 升级对
单流解码性能中性(26.5/26.0 vs 升级前 30.6/38.0 —— 在噪声范围内),
升级买到的是特性而非速度。

## BitNet spike(microsoft/bitnet.cpp —— 未合入,NO-GO 2026-10-13)

针对 BitNet T-MAC 推理栈跑了一轮 6-gate spike,跑在 spike 主机
(Zen 3 / RTX 3090),**未合入** —— 完整报告见
[target/bitnet-spike/REPORT.md](https://github.com/iBLOC/neuron/blob/main/target/bitnet-spike/REPORT.md)。
TL-OFF i2_s 路径在 MSVC + 4 个 scratch patch 下能编,但 T-MAC 快路径
在 wrapper 钉死的 fork 快照上是死代码,下面表是 LLAMAFILE sgemm
回退路径的数据。官方 `ggml-model-i2_s.gguf` 缺 `tokenizer.ggml.pre`,
spike 配了一个 1-KV 的 `patch_pre.py` 在模型加载前注入
`pre=llama-bpe`。

| 技术栈 | 模型 / 量化 | 设备 | 解码 tok/s | 峰值 RSS (MB) | 备注 |
|---|---|---|---:|---:|---|
| neuron / candle | 1B f16 | cpu | 1.6–2.6 | n/a | 基线(上表) |
| neuron / candle | 2.5B f16 | cpu | 2.58 | 7 451 | 基线(上表) |
| **bitnet.cpp spike ³** | 2B i2_s (1.58-bit) | cpu AVX2 | **16.8–18.6** | ~2 000 | 4 个 patch,无 T-MAC;Zen 3(无 AVX-512、无 VNNI);`--temp 0, -t 4` 下 127 token |

³ 不是 neuron 后端。数据来自 spike 目录里
`target/bitnet-spike/BitNet/build-static/` 自建的 `llama-completion.exe`,
在 spike 主机(Zen 3 / RTX 3090,与上表同机)上跑。CPU 上对 candle f32/f16
的 6-7× 优势是真的,但有三个前提:(a) T-MAC 路径能编译过;(b) 用了补丁
过的 gguf;(c) chat-template 装配仍然放在 OpenAI adapter 里(fork 的
`-cnv` 在自定义 `bitnet-b1.58` arch 下崩溃)。重新评估的门见 spike
报告 §4。

## VLM 精度与吞吐(Mage-VL 首轮,2026-10-13)

`microsoft/Mage-VL`(MageViT + Qwen3-4B),`mage_vl` 引擎,CUDA bf16,
贪婪解码,官方测试集抽样子集:

| 数据集 | 子集 | 指标 | 得分 | 未解析 / 0-token |
|---|---|---|---|---|
| POPE | 450 题(random/popular/adversarial) | F1 / acc | 0.599 / 22.4% | 48.7% |
| MME | 280 题(14 个子任务) | perception / 1400 | 72.8 | 58.2% |
| MMStar | 144 题(18 个类目) | 准确率 | 13.9% | 48.6% |

吞吐:**874 题 / 5.7 分钟 ≈ 2.6 题/秒** 端到端(含图像 base64 传输 +
prefill + 解码;预热后每题 0.2–0.7 秒)。

::: warning 引用前先读
"未解析 / 0-token" 一列占主导:在真实图像上,首个采样 token 经常是
`<|im_end|>`,模型直接返回空响应而不是错误答案。这些数字反映的是
**Rust 移植的当前状态**,不是上游 checkpoint 的能力 ——
[完整诊断](https://github.com/iBLOC/neuron/blob/main/docs/superpowers/specs/2026-10-13-mage-vl-vlm-bench.md)
列出了已修复的五个正确性 bug(CUDA 精度、36 GiB KV 预分配、worker
线程、chat 模板、pad 扩展)和下一个要修的"首 token 退化"问题。
复现脚本:`scripts/vlm-bench/`。
:::

## OCR 端到端延迟(PP-OCR 档位 × 执行器)

| 模型 | det+rec 权重 | tract(冷启动 e2e, ms) | ort(冷启动 e2e, ms) | ort 提速 | 区域数 |
|---|---|---:|---:|---:|---:|
| ppocrv5-mobile | 21 MB | 2 922 | 588 | 5.0× | 5 |
| ppocrv6-medium | 139 MB | 7 924 | 1 056 | 7.5× | 5 |
| ppocrv5-server | 173 MB | 13 160 | 1 362 | 9.7× | 5 |

同一张 800×400 中英文页面、单次进程(数字含执行器首次初始化:tract
编译 plan,ort 加载 onnxruntime)。文本输出在执行器和档位间完全一致。
`ort` 执行器收复了大部分 mobile→server 档位差距(588 → 1 362 ms,
而 tract 是 2 922 → 13 160)。

## ASR 延迟(SenseVoiceSmall,sherpa 引擎)

| wav | 时长 | p50 (ms) | RTF |
|---|---:|---:|---:|
| en.wav | 7.15 s | 482.9 | 0.068 |
| zh.wav | 5.59 s | 361.4 | 0.065 |

RTF < 0.1 = 10 倍实时。tract 路径输出与 sherpa 逐字一致
(2026-09-26 验证);本次未采集其逐条延迟。

## 视觉内存池(TensorPool)

| 路径 | 开销 |
|---|---|
| 全新 `Tensor::zeros(1,3,640,640)` | ~460 µs |
| 池化 get+release | ~292 ns(~1574×) |

YOLOv8n 可池化中间量 ≈ 32 MB,不池化峰值 ~200 MB(约 6× 削减)。

## 静态权重 / 内存占用

| 模型 | 权重 | 常驻内存 |
|---|---|---|
| minicpm5-1b | ~2 GB (bf16) | ~3 GB 显存 (bf16);CPU f32 见上表 |
| minicpm5-2b | ~5 GB (bf16) | ~6–7 GB 显存 (bf16);f32 ~10 GB 内存 |
| mage-vl (VLM) | 9.5 GB bf16 + 1.07 GB 门控 | ~11 GB 显存 (bf16, CUDA) |
| voxcpm2 (TTS) | 4.58 GB bf16 + 377 MB fp32 | ~10 GB 内存 (f32) |
| laya (decision) | ~500 MB int8 ONNX | int8 |
| PP-OCRv5 mobile (OCR) | 21 MB | |
| PP-OCRv5 server (OCR) | 173 MB | |
| PP-OCRv6 medium (OCR) | 139 MB | +130 MB 版面, +368 MB 表格(可选) |
| all-minilm-l6-v2 (embedding) | ~88 MB | |
| bge-reranker-base (rerank) | ~1.08 GB | |
| yolov8n (vision) | ~6 MB | 可池化中间量 ~32 MB |

## 复现

```bash
# LLM TTFT / 解码 tok/s / 峰值内存 vs vLLM / ollama / llama.cpp
# (自动启动已安装的服务;报告 -> target/llm-speed-compare.md)
./scripts/run-llm-speed-compare.sh

# 能力 × 引擎矩阵(需要 neuron_server 跑在 :18083)
# (报告 -> target/model-matrix.md)
cargo bench --bench model_matrix --features "candle,tract,sherpa,voxcpm"

# ASR tract-vs-sherpa(报告 -> target/asr-speed-compare.md)
cargo bench --bench asr_speed --features tract,sherpa

# VLM 精度:POPE / MME / MMStar(需要 mage_vl server)
# (报告 -> scripts/vlm-bench/results/)
cd scripts/vlm-bench
python prep_datasets.py --bench pope  > data/pope.jsonl
python eval_vlm.py  --base-url http://127.0.0.1:18093 --bench pope

# 分配 / 内存池微基准
cargo bench --bench mobile_bench
cargo bench --bench perf -- --quick

# 按模型全量扫描(目录里每个模型一个用例,进程内,跑在本地设备)
# -> target/model-bench-<device>.md
scripts/fetch-bench-models.sh                 # 权重 -> ~/.neuron/models/<cap>/<id>/
python scripts/export-yolo-onnx.py             # Ultralytics .pt -> model.onnx + config.json
scripts\cuda-build.bat cargo bench --bench model_bench --features "candle,cuda,tract,ocr,sherpa,voxcpm,chronos2,litert"
```

方法学细节(prompt 防缓存、客户端计时、预热丢弃、贪婪解码)写在
每份生成报告的头部。

## 另见

- [/zh/reference/models](/zh/reference/models) — 支持的模型 × 后端矩阵
- [/zh/concepts/scheduler](/zh/concepts/scheduler) — 内存预算如何与这些数字做交换
- [/zh/concepts/engines](/zh/concepts/engines) — 何时选哪个引擎
