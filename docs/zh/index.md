---
title: neuron 是什么?
description: Apache-2.0 Rust 推理运行时——LLM / TTS / ASR / Rerank / Embedding / Vision / OCR
layout: home

hero:
  name: neuron
  text: 神经网络推理运行时
  tagline: OpenAI 与 Anthropic 线格式兼容。纯库设计,嵌入任何 Rust 宿主进程。
  actions:
    - theme: brand
      text: 快速开始
      link: /zh/guide/quick-start
    - theme: alt
      text: GitHub
      link: https://github.com/iBLOC/neuron
    - theme: alt
      text: API 参考
      link: /zh/reference/api

features:
  - title: 🤖 9 大能力面
    details: |
      LLM(Qwen / Llama / Phi / Mistral / ChatGLM / Gemma / MiniCPM5 / mistralrs 40+ 模型) ·
      TTS(VoxCPM 0.5B + VoxCPM2 2B/48kHz) · ASR(Whisper + SenseVoice) ·
      Rerank(BGE / mxbai / jina / Cohere) · Embedding(MiniLM BERT) ·
      Vision(YOLOv8 / RT-DETR / DETR / RF-DETR / Florence-2) ·
      OCR(PP-OCRv5/v6 + 版面分析 + 表格结构) · 决策打分(Laya) · 时序预测(Chronos2)
  - title: 🔌 OpenAI + Anthropic 线格式兼容
    details: |
      10+ OpenAI 兼容路由(/v1/chat/completions、/v1/audio/speech、/v1/embeddings、
      /v1/rerank、/v1/vision/*、/v1/ocr/recognize、/v1/models/pull 等)+
      /v1/messages 实现 Anthropic Messages v1(vision、tool_use、cache_control、
      extended thinking、SSE 流式)。
  - title: 🛠 3 套 ML 框架按需选择
    details: |
      candle(默认,LLM/ASR/Rerank/Embedding/Vision)· burn(VoxCPM 语音合成)·
      litert_lm(端侧 LLM 子进程)——外加 tract / ort 两套 ONNX 执行器
      (SenseVoice 语音识别、Laya 决策打分、PP-OCR 文字识别)。Feature flag 按需裁剪。
  - title: 🧠 有界内存模型调度器
    details: |
      内存装不下的模型也能服务:请求排队等常驻任务排空,权重按 LRU 换出,
      空闲模型自动回收。基于 RSS 实测的内存画像在运行时自校正。可一键关闭。
  - title: 📦 纯库设计,进程内嵌入
    details: |
      没有独立守护进程。宿主调用 neuron::run_embedded(...) 拿到 NeuronHandle,
      由它管理 axum 服务生命周期。Tauri 2 / Flutter(cargo-ndk)/ 服务端进程 /
      任意 Rust 二进制,即插即用。
  - title: 🌍 多源模型仓库
    details: |
      插件化下载源:HuggingFace、hf-mirror(国内 CDN)、ModelScope。
      支持按引擎 / 按模型覆盖下载源;<source>:<repo> 语法 +
      默认源回退;目录按源去重。
  - title: 📜 Apache-2.0
    details: |
      以 Apache License 2.0 发布。VoxCPM 自研移植(src/tts/voxcpm_burn/)
      与 YOLOv8 清洁室架构实现(src/vision/yolov8_arch.rs)。无 AGPL-3.0,
      无传染性协议。
---
