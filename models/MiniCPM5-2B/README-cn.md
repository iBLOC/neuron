---
license: apache-2.0
language:
  - zh
  - en
library_name: transformers
pipeline_tag: text-generation
tags:
  - minicpm
  - minicpm5
  - llama
  - text-generation
  - long-context
  - tool-calling
  - on-device
  - edge-ai
datasets:
  - openbmb/Ultra-FineWeb
  - openbmb/UltraX-Preview
  - openbmb/Ultra-FineWeb-L3
  - openbmb/UltraData-Math
  - openbmb/UltraData-Code
  - openbmb/UltraData-SFT-2605
  - openbmb/UltraData-SFT-Agent-2609
  - openbmb/UltraData-RL-2609
---

![](https://raw.githubusercontent.com/OpenBMB/MiniCPM/main/assets/minicpm_logo.png)

[MiniCPM 技术报告](https://arxiv.org/pdf/2506.07900) | [MiniCPM 知识库](https://modelbest.feishu.cn/wiki/UtWxwcERfiRIpIkBOjuc3h9tn1D) | [GitHub 仓库](https://github.com/OpenBMB/MiniCPM) | [UltraData](https://ultradata.openbmb.cn/) | [在线 Demo](https://huggingface.co/spaces/openbmb/MiniCPM5-2B-Demo)

[English](https://huggingface.co/openbmb/MiniCPM5-2B/blob/main/README.md) | 中文

## 亮点

我们正式发布 **MiniCPM5-2B**，这是继 [MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) 之后 **MiniCPM5** 系列的第二个模型。它是一款面向端侧、本地部署和资源受限场景的 2B 稠密 Transformer，能够达到同尺寸开源模型 SOTA 水平。

🏆 **同尺寸开源模型 SOTA**：与同尺寸优秀开源模型相比，MiniCPM5-2B 在该对比范围内达到 SOTA 水平，整体表现可与 4B 级模型竞争，并在代码、数学、长文本、工具调用和 Agent 任务上展现出明显优势。

<div id="capability-comparison-radar" class="radar-visual" role="img" aria-label="能力雷达图：对比 MiniCPM5-2B 与 4B 级模型；每个维度分别将最高分归一化为 100%。">
  <style>
    #capability-comparison-radar {
      --foreground: #171717;
      display: block; width: 100%; max-width: 620px; margin: 0 auto; background: #fff;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
    }
    #capability-comparison-radar svg { display: block; width: 100%; height: auto; background: #fff; }
    #capability-comparison-radar .title { fill: var(--foreground); font-size: 16px; font-weight: 600; letter-spacing: 0; }
    #capability-comparison-radar .axis-label { fill: #333; font-size: 12px; font-weight: 600; }
    #capability-comparison-radar .ring { fill: none; stroke: rgba(128,128,128,.18); stroke-width: .8; }
    #capability-comparison-radar .spoke { stroke: rgba(128,128,128,.25); stroke-width: .8; }
    #capability-comparison-radar .ring-label { fill: #8a8a8a; font-size: 9px; }
    #capability-comparison-radar .series { stroke-linejoin: round; }
    #capability-comparison-radar .series.primary { stroke-width: 2.4; }
    #capability-comparison-radar .legend-label { fill: #171717; font-size: 12px; font-weight: 600; }
    #capability-comparison-radar .legend-average { fill: #666; font-size: 11px; }
    #capability-comparison-radar .legend-swatch { rx: 3; }
    @media (max-width: 900px) { #capability-comparison-radar { overflow-x: auto; } #capability-comparison-radar svg { min-width: 620px; } }
  </style>
  <svg viewBox="0 0 680 560" aria-hidden="true">
    <g transform="translate(-20 0)">
      <text x="40" y="28" class="title" text-anchor="start">各能力维度雷达图</text>
      <polygon points="340.0,235.0 362.5,243.2 374.5,263.9 370.3,287.5 352.0,302.9 328.0,302.9 309.7,287.5 305.5,263.9 317.5,243.2" class="ring" stroke="rgba(128,128,128,.18)" stroke-width=".8"/>
      <text x="344.0" y="238.0" class="ring-label">20%</text>
      <polygon points="340.0,200.0 385.0,216.4 408.9,257.8 400.6,305.0 363.9,335.8 316.1,335.8 279.4,305.0 271.1,257.8 295.0,216.4" class="ring" stroke="rgba(128,128,128,.18)" stroke-width=".8"/>
      <text x="344.0" y="203.0" class="ring-label">40%</text>
      <polygon points="340.0,165.0 407.5,189.6 443.4,251.8 430.9,322.5 375.9,368.7 304.1,368.7 249.1,322.5 236.6,251.8 272.5,189.6" class="ring" stroke="rgba(128,128,128,.18)" stroke-width=".8"/>
      <text x="344.0" y="168.0" class="ring-label">60%</text>
      <polygon points="340.0,130.0 430.0,162.8 477.9,245.7 461.2,340.0 387.9,401.6 292.1,401.6 218.8,340.0 202.1,245.7 250.0,162.8" class="ring" stroke="rgba(128,128,128,.18)" stroke-width=".8"/>
      <text x="344.0" y="133.0" class="ring-label">80%</text>
      <polygon points="340.0,95.0 452.5,135.9 512.3,239.6 491.6,357.5 399.9,434.4 280.1,434.4 188.4,357.5 167.7,239.6 227.5,135.9" class="ring" stroke="rgba(29,111,208,.28)" stroke-width="1"/>
      <text x="344.0" y="98.0" class="ring-label">100%</text>
      <line x1="340" y1="270" x2="340.0" y2="95.0" class="spoke"/>
      <text x="340.0" y="63.5" class="axis-label" text-anchor="middle">代码推理</text>
      <line x1="340" y1="270" x2="452.5" y2="135.9" class="spoke"/>
      <text x="472.7" y="111.8" class="axis-label" text-anchor="start">数学推理</text>
      <line x1="340" y1="270" x2="512.3" y2="239.6" class="spoke"/>
      <text x="543.4" y="234.1" class="axis-label" text-anchor="start">指令遵循</text>
      <line x1="340" y1="270" x2="491.6" y2="357.5" class="spoke"/>
      <text x="518.8" y="373.2" class="axis-label" text-anchor="start">综合知识</text>
      <line x1="340" y1="270" x2="399.9" y2="434.4" class="spoke"/>
      <text x="410.6" y="464.0" class="axis-label" text-anchor="start">长文本</text>
      <line x1="340" y1="270" x2="280.1" y2="434.4" class="spoke"/>
      <text x="269.4" y="464.0" class="axis-label" text-anchor="end">工具调用</text>
      <line x1="340" y1="270" x2="188.4" y2="357.5" class="spoke"/>
      <text x="161.2" y="373.3" class="axis-label" text-anchor="end">代码智能体</text>
      <line x1="340" y1="270" x2="167.7" y2="239.6" class="spoke"/>
      <text x="136.6" y="234.1" class="axis-label" text-anchor="end">搜索智能体</text>
      <line x1="340" y1="270" x2="227.5" y2="135.9" class="spoke"/>
      <text x="207.3" y="111.8" class="axis-label" text-anchor="end">通用智能体</text>
      <polygon points="340.0,136.7 450.1,138.8 498.4,242.1 491.6,357.5 398.4,430.3 289.5,408.8 188.4,357.5 188.2,243.2 249.3,161.9" class="series" fill="#E85D4C" fill-opacity=".10" stroke="#E85D4C" stroke-width="1.5"/>
      <circle cx="340.0" cy="136.7" r="3" fill="#E85D4C"/>
      <circle cx="450.1" cy="138.8" r="3" fill="#E85D4C"/>
      <circle cx="498.4" cy="242.1" r="3" fill="#E85D4C"/>
      <circle cx="491.6" cy="357.5" r="3" fill="#E85D4C"/>
      <circle cx="398.4" cy="430.3" r="3" fill="#E85D4C"/>
      <circle cx="289.5" cy="408.8" r="3" fill="#E85D4C"/>
      <circle cx="188.4" cy="357.5" r="3" fill="#E85D4C"/>
      <circle cx="188.2" cy="243.2" r="3" fill="#E85D4C"/>
      <circle cx="249.3" cy="161.9" r="3" fill="#E85D4C"/>
      <polygon points="340.0,134.4 448.9,140.2 512.3,239.6 462.2,340.5 367.8,346.5 308.0,358.0 231.0,332.9 242.3,252.8 250.1,162.9" class="series" fill="#2A9D8F" fill-opacity=".10" stroke="#2A9D8F" stroke-width="1.5"/>
      <circle cx="340.0" cy="134.4" r="3" fill="#2A9D8F"/>
      <circle cx="448.9" cy="140.2" r="3" fill="#2A9D8F"/>
      <circle cx="512.3" cy="239.6" r="3" fill="#2A9D8F"/>
      <circle cx="462.2" cy="340.5" r="3" fill="#2A9D8F"/>
      <circle cx="367.8" cy="346.5" r="3" fill="#2A9D8F"/>
      <circle cx="308.0" cy="358.0" r="3" fill="#2A9D8F"/>
      <circle cx="231.0" cy="332.9" r="3" fill="#2A9D8F"/>
      <circle cx="242.3" cy="252.8" r="3" fill="#2A9D8F"/>
      <circle cx="250.1" cy="162.9" r="3" fill="#2A9D8F"/>
      <polygon points="340.0,189.3 411.4,184.9 502.9,241.3 455.3,336.6 356.7,315.9 288.5,411.4 320.8,281.1 266.8,257.1 298.8,220.9" class="series" fill="#E09F3E" fill-opacity=".10" stroke="#E09F3E" stroke-width="1.5"/>
      <circle cx="340.0" cy="189.3" r="3" fill="#E09F3E"/>
      <circle cx="411.4" cy="184.9" r="3" fill="#E09F3E"/>
      <circle cx="502.9" cy="241.3" r="3" fill="#E09F3E"/>
      <circle cx="455.3" cy="336.6" r="3" fill="#E09F3E"/>
      <circle cx="356.7" cy="315.9" r="3" fill="#E09F3E"/>
      <circle cx="288.5" cy="411.4" r="3" fill="#E09F3E"/>
      <circle cx="320.8" cy="281.1" r="3" fill="#E09F3E"/>
      <circle cx="266.8" cy="257.1" r="3" fill="#E09F3E"/>
      <circle cx="298.8" cy="220.9" r="3" fill="#E09F3E"/>
      <polygon points="340.0,95.0 452.5,135.9 499.8,241.8 476.2,348.7 399.9,434.4 280.1,434.4 220.0,339.3 167.7,239.6 227.5,135.9" class="series primary" fill="#1D6FD0" fill-opacity=".22" stroke="#1D6FD0" stroke-width="2.4"/>
      <circle cx="340.0" cy="95.0" r="3" fill="#1D6FD0"/>
      <circle cx="452.5" cy="135.9" r="3" fill="#1D6FD0"/>
      <circle cx="499.8" cy="241.8" r="3" fill="#1D6FD0"/>
      <circle cx="476.2" cy="348.7" r="3" fill="#1D6FD0"/>
      <circle cx="399.9" cy="434.4" r="3" fill="#1D6FD0"/>
      <circle cx="280.1" cy="434.4" r="3" fill="#1D6FD0"/>
      <circle cx="220.0" cy="339.3" r="3" fill="#1D6FD0"/>
      <circle cx="167.7" cy="239.6" r="3" fill="#1D6FD0"/>
      <circle cx="227.5" cy="135.9" r="3" fill="#1D6FD0"/>
      <rect x="40" y="490" width="18" height="18" class="legend-swatch" fill="#1D6FD0"/>
      <text x="66" y="504" class="legend-label">MiniCPM5-2B</text>
      <text x="66" y="520" class="legend-average">平均分 53.9</text>
      <rect x="195" y="490" width="18" height="18" class="legend-swatch" fill="#E85D4C"/>
      <text x="221" y="504" class="legend-label">Qwen3.5-4B</text>
      <text x="221" y="520" class="legend-average">平均分 51.1</text>
      <rect x="350" y="490" width="18" height="18" class="legend-swatch" fill="#2A9D8F"/>
      <text x="376" y="504" class="legend-label">granite-4.2-3B</text>
      <text x="376" y="520" class="legend-average">平均分 42.7</text>
      <rect x="505" y="490" width="18" height="18" class="legend-swatch" fill="#E09F3E"/>
      <text x="531" y="504" class="legend-label">LFM2.5-2.6B</text>
      <text x="531" y="520" class="legend-average">平均分 33.2</text>
      <text x="340" y="548" class="legend-average" text-anchor="middle">每条轴：该维最高分 = 100%</text>
    </g>
  </svg>
</div>

📂 **开放高质量数据**：与模型一同开源其背后的高质量训练数据，均属于 [UltraData](https://ultradata.openbmb.cn/) 数据体系：[UltraX](https://huggingface.co/datasets/openbmb/UltraX-Preview)，高质量网页预训练数据集；[UltraData-Code](https://huggingface.co/datasets/openbmb/UltraData-Code)，L0–L3 分级代码治理，推动代码能力显著跃升；[UltraData-SFT-Agent-2609](https://huggingface.co/datasets/openbmb/UltraData-SFT-Agent-2609)，50 万 Agent 训练样本，赋能端侧 Agent 综合能力提升；[UltraData-RL-2609](https://huggingface.co/datasets/openbmb/UltraData-RL-2609)，超 8 万条高质量 RL 训练样本，覆盖数学、代码、通用知识与长文本推理。

## 模型列表

你可以按运行环境选择对应模型格式：

**MiniCPM5-2B**

- **[MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B)** · [ModelScope](https://www.modelscope.cn/models/OpenBMB/MiniCPM5-2B) · BF16 正式版（经 RL + OPD 后训练） **👈 当前页面**
- **[MiniCPM5-2B-SFT](https://huggingface.co/openbmb/MiniCPM5-2B-SFT)** · [ModelScope](https://www.modelscope.cn/models/OpenBMB/MiniCPM5-2B-SFT) · BF16 SFT 单独 checkpoint（RL / OPD 之前）
- **[MiniCPM5-2B-Midtrain](https://huggingface.co/openbmb/MiniCPM5-2B-Midtrain)** · [ModelScope](https://www.modelscope.cn/models/OpenBMB/MiniCPM5-2B-Midtrain) · BF16 mid-training checkpoint（SFT 之前）
- **[MiniCPM5-2B-Base](https://huggingface.co/openbmb/MiniCPM5-2B-Base)** · [ModelScope](https://www.modelscope.cn/models/OpenBMB/MiniCPM5-2B-Base) · BF16 base checkpoint（仅预训练）
- **[MiniCPM5-2B-GGUF](https://huggingface.co/openbmb/MiniCPM5-2B-GGUF)** · [ModelScope](https://www.modelscope.cn/models/OpenBMB/MiniCPM5-2B-GGUF) · GGUF，适用于 llama.cpp / Ollama / LM Studio
- **[MiniCPM5-2B-MLX](https://huggingface.co/openbmb/MiniCPM5-2B-MLX)** · [ModelScope](https://www.modelscope.cn/models/OpenBMB/MiniCPM5-2B-MLX) · MLX / 4bit，适用于 Apple Silicon
- **[MiniCPM5-2B-GPTQ](https://huggingface.co/openbmb/MiniCPM5-2B-GPTQ)** · [ModelScope](https://www.modelscope.cn/models/OpenBMB/MiniCPM5-2B-GPTQ) · GPTQ / 4bit 量化模型
- **[MiniCPM5-2B-DSpark](https://huggingface.co/openbmb/MiniCPM5-2B-DSpark)** · [ModelScope](https://www.modelscope.cn/models/OpenBMB/MiniCPM5-2B-DSpark) · DSpark 草稿模型，用于推理加速
- **[MiniCPM5-2B-DSpark-GGUF](https://huggingface.co/openbmb/MiniCPM5-2B-DSpark-GGUF)** · [ModelScope](https://www.modelscope.cn/models/OpenBMB/MiniCPM5-2B-DSpark-GGUF) · GGUF 版本的 DSpark 草稿模型
- **[MiniCPM5-2B-LiteRT](https://huggingface.co/litert-community/MiniCPM5-2B)** · [ModelScope](https://www.modelscope.cn/models/litert-community/MiniCPM5-2B) · MiniCPM5-2B 的 LiteRT-LM 版本

**MiniCPM5-1B**

- **[MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B)** · [ModelScope](https://www.modelscope.cn/models/OpenBMB/MiniCPM5-1B) · BF16 正式版（经 RL + OPD 后训练）
- **[MiniCPM5-1B-SFT](https://huggingface.co/openbmb/MiniCPM5-1B-SFT)** · [ModelScope](https://www.modelscope.cn/models/OpenBMB/MiniCPM5-1B-SFT) · BF16 SFT 单独 checkpoint（RL / OPD 之前）
- **[MiniCPM5-1B-Base](https://huggingface.co/openbmb/MiniCPM5-1B-Base)** · [ModelScope](https://www.modelscope.cn/models/OpenBMB/MiniCPM5-1B-Base) · BF16 base checkpoint（仅预训练）
- **[MiniCPM5-1B-GGUF](https://huggingface.co/openbmb/MiniCPM5-1B-GGUF)** · [ModelScope](https://www.modelscope.cn/models/OpenBMB/MiniCPM5-1B-GGUF) · GGUF，适用于 llama.cpp / Ollama / LM Studio
- **[MiniCPM5-1B-MLX](https://huggingface.co/openbmb/MiniCPM5-1B-MLX)** · [ModelScope](https://www.modelscope.cn/models/OpenBMB/MiniCPM5-1B-MLX) · MLX / 4bit，适用于 Apple Silicon

## 模型信息

MiniCPM5-2B 具有以下特性：

- **类型**：Causal Language Model
- **架构**：标准 `LlamaForCausalLM`
- **参数数量**：2,516,756,480
- **非嵌入参数数量**：1,981,982,720
- **层数**：42
- **注意力头（GQA）**：16 个 Q heads / 2 个 KV heads
- **上下文长度**：131,072

## 简介

MiniCPM5-2B 是 MiniCPM5 系列的第二个模型，面向本地助手、coding agent、工具调用流程以及需要紧凑模型的推理场景。它在较小部署成本下提供原生长上下文能力。

## 评测结果

我们选取 **LFM2.5-2.6B**、**Qwen3.5-2B**、**Gemma-4-E2B-it** 等同尺寸开源模型进行横向比较，并同时列出 **Qwen3.5-4B**、**granite-4.2-3B**、**Nemotron-3-Nano-4B**、**Gemma-4-E4B-it**、**LFM2.5-8B-A1B** 等更大规模模型作为参考。

在这组对比中，MiniCPM5-2B 达到同尺寸开源模型 SOTA 水平（平均分 53.9 ），也超过了参与对比的全部更大规模模型（最高 51.1）。其优势主要体现在代码推理、数学推理、长文本、工具调用与多个智能体任务上。

<div style="width:100%;max-width:1080px;margin:0 auto;padding:16px 0;background:#fff;
font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'PingFang SC',
'Hiragino Sans GB','Microsoft YaHei',sans-serif;color:#171717">
  <h1 style="margin:0 0 14px;font-size:22px;font-weight:700;color:#1D6FD0;
  letter-spacing:0.02em">MiniCPM5-2B 与基线模型评测结果</h1>
  <table class="vl-table" style="width:100%;margin:0;table-layout:fixed;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums"><thead><tr><th rowspan="2" style="padding:7px 5px;text-align:left;font-weight:600;border-bottom:2px solid #1D6FD0;color:#1D6FD0;width:18%"></th><th rowspan="2" style="padding:7px 4px;text-align:center;font-weight:600;border-bottom:2px solid #1D6FD0;color:#1D6FD0;font-size:12px;width:9.111%;background:rgba(29, 111, 208, 0.08);vertical-align:middle;word-break:normal;">MiniCPM5-2B</th><th colspan="3" style="padding:6px 4px;text-align:center;font-weight:600;color:#1D6FD0;font-size:13px;border-bottom:1px solid rgba(29, 111, 208, 0.2);border-left:1px solid rgba(29, 111, 208, 0.25);">2B 级模型</th><th colspan="5" style="padding:6px 4px;text-align:center;font-weight:600;color:#1D6FD0;font-size:13px;border-bottom:1px solid rgba(29, 111, 208, 0.2);border-left:1px solid rgba(29, 111, 208, 0.25);">4B 级模型</th></tr><tr><th style="padding:6px 4px;text-align:center;font-weight:500;border-bottom:2px solid #1D6FD0;color:#1D6FD0;font-size:11px;width:9.111%;border-left:1px solid rgba(29, 111, 208, 0.25);word-break:normal;vertical-align:middle;">LFM2.5-2.6B</th><th style="padding:6px 4px;text-align:center;font-weight:500;border-bottom:2px solid #1D6FD0;color:#1D6FD0;font-size:11px;width:9.111%;word-break:normal;vertical-align:middle;">Qwen3.5-2B</th><th style="padding:6px 4px;text-align:center;font-weight:500;border-bottom:2px solid #1D6FD0;color:#1D6FD0;font-size:11px;width:9.111%;word-break:normal;vertical-align:middle;">Gemma-4-E2B-it</th><th style="padding:6px 4px;text-align:center;font-weight:500;border-bottom:2px solid #1D6FD0;color:#1D6FD0;font-size:11px;width:9.111%;border-left:1px solid rgba(29, 111, 208, 0.25);word-break:normal;vertical-align:middle;">Qwen3.5-4B</th><th style="padding:6px 4px;text-align:center;font-weight:500;border-bottom:2px solid #1D6FD0;color:#1D6FD0;font-size:11px;width:9.111%;word-break:normal;vertical-align:middle;">granite-4.2-3B</th><th style="padding:6px 4px;text-align:center;font-weight:500;border-bottom:2px solid #1D6FD0;color:#1D6FD0;font-size:11px;width:9.111%;word-break:normal;vertical-align:middle;">Nemotron-3-Nano-4B</th><th style="padding:6px 4px;text-align:center;font-weight:500;border-bottom:2px solid #1D6FD0;color:#1D6FD0;font-size:11px;width:9.111%;word-break:normal;vertical-align:middle;">Gemma-4-E4B-it</th><th style="padding:6px 4px;text-align:center;font-weight:500;border-bottom:2px solid #1D6FD0;color:#1D6FD0;font-size:11px;width:9.111%;word-break:normal;vertical-align:middle;">LFM2.5-8B-A1B</th></tr></thead><tbody>
<tr style="background:rgba(29, 111, 208, 0.03)"><td class="benchmark-cell" style="padding:5px 5px;padding-left:14px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:14px;font-weight:600;line-height:1.22;color:#171717">平均分</div></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(29, 111, 208, 0.08);vertical-align:middle;font-size:14px;line-height:1.2;font-weight:600;"><strong style="color:#1D6FD0">53.9</strong></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;font-weight:600;">33.2</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;font-weight:600;">28.0</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;font-weight:600;">24.6</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;font-weight:600;">51.1</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;font-weight:600;">42.7</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;font-weight:600;">32.6</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;font-weight:600;">31.2</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;font-weight:600;">28.4</td></tr>
<tr><td colspan="10" style="padding:5px 10px;font-weight:600;color:#1D6FD0;border-bottom:1px solid rgba(29, 111, 208, 0.2);background:rgba(29, 111, 208, 0.14)">代码推理</td></tr>
<tr><td class="benchmark-cell" style="padding:5px 5px;padding-left:14px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:14px;font-weight:600;line-height:1.22;color:#171717">LiveCodeBench v6</div></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(29, 111, 208, 0.08);vertical-align:middle;font-size:14px;line-height:1.2;"><strong style="color:#1D6FD0">69.1</strong></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">42.1</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">20.2</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">42.9</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">56.4</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">58.9</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">50.7</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">53.9</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">39.8</td></tr>
<tr><td class="benchmark-cell" style="padding:5px 5px;padding-left:14px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:14px;font-weight:600;line-height:1.22;color:#171717">LCB-Pro 25Q2 (Easy)</div></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(29, 111, 208, 0.08);vertical-align:middle;font-size:14px;line-height:1.2;"><strong style="color:#1D6FD0">68.0</strong></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">30.9</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">10.3</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">27.1</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">58.3</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">54.6</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">51.6</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">45.8</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">27.8</td></tr>
<tr><td class="benchmark-cell" style="padding:5px 5px;padding-left:14px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:14px;font-weight:600;line-height:1.22;color:#171717">LCB-Pro 25Q2 (Medium)</div></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(29, 111, 208, 0.08);vertical-align:middle;font-size:14px;line-height:1.2;"><strong style="color:#1D6FD0">17.5</strong></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">0.0</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">0.0</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">0.0</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">7.0</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">5.3</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">5.3</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">1.8</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">0.0</td></tr>
<tr><td class="benchmark-cell" style="padding:5px 5px;padding-left:14px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:14px;font-weight:600;line-height:1.22;color:#171717">OJBench</div></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(29, 111, 208, 0.08);vertical-align:middle;font-size:14px;line-height:1.2;"><strong style="color:#1D6FD0">32.5</strong></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">11.2</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">2.6</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">11.6</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">24.8</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">21.8</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">20.0</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">19.0</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">8.2</td></tr>
<tr><td class="benchmark-cell" style="padding:5px 5px;padding-left:14px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:14px;font-weight:600;line-height:1.22;color:#171717">SciCode (wbg)</div></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(29, 111, 208, 0.08);vertical-align:middle;font-size:14px;line-height:1.2;"><strong style="color:#1D6FD0">26.3</strong><sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">14.2<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">2.8<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">20.9<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">16.1<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">24.9<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">16.4<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">24.4<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">7.8<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td></tr>
<tr><td colspan="10" style="padding:5px 10px;font-weight:600;color:#1D6FD0;border-bottom:1px solid rgba(29, 111, 208, 0.2);background:rgba(29, 111, 208, 0.14)">数学推理</td></tr>
<tr><td class="benchmark-cell" style="padding:5px 5px;padding-left:14px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:14px;font-weight:600;line-height:1.22;color:#171717">AIME 2025</div></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(29, 111, 208, 0.08);vertical-align:middle;font-size:14px;line-height:1.2;"><strong style="color:#1D6FD0">86.5</strong></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">41.9</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">29.6</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">31.7</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">78.8</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">79.4</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">56.3</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">37.1</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">46.0</td></tr>
<tr><td class="benchmark-cell" style="padding:5px 5px;padding-left:14px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:14px;font-weight:600;line-height:1.22;color:#171717">AIME 2026</div></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(29, 111, 208, 0.08);vertical-align:middle;font-size:14px;line-height:1.2;"><strong style="color:#1D6FD0">86.5</strong></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">45.2</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">29.0</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">39.8</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">82.7</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">83.5</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">62.1</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">45.0</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">56.7</td></tr>
<tr><td class="benchmark-cell" style="padding:5px 5px;padding-left:14px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:14px;font-weight:600;line-height:1.22;color:#171717">HMMT Feb 2026</div></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(29, 111, 208, 0.08);vertical-align:middle;font-size:14px;line-height:1.2;"><strong>63.8</strong></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">33.7</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">20.5</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">17.8</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;"><strong style="color:#1D6FD0">64.0</strong></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">60.8</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">51.3</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">30.1</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">38.5</td></tr>
<tr><td class="benchmark-cell" style="padding:5px 5px;padding-left:14px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:14px;font-weight:600;line-height:1.22;color:#171717">MATH-500</div></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(29, 111, 208, 0.08);vertical-align:middle;font-size:14px;line-height:1.2;"><strong>94.6</strong></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">89.6</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">85.8</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">85.4</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;"><strong style="color:#1D6FD0">99.0</strong></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">97.0</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">91.6</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">88.2</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">93.2</td></tr>
<tr><td colspan="10" style="padding:5px 10px;font-weight:600;color:#1D6FD0;border-bottom:1px solid rgba(29, 111, 208, 0.2);background:rgba(29, 111, 208, 0.14)">指令遵循</td></tr>
<tr><td class="benchmark-cell" style="padding:5px 5px;padding-left:14px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:14px;font-weight:600;line-height:1.22;color:#171717">IFBench</div></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(29, 111, 208, 0.08);vertical-align:middle;font-size:14px;line-height:1.2;"><strong>66.3</strong></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">59.0</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">46.0</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">25.7</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">59.0</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;"><strong style="color:#1D6FD0">73.0</strong></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">58.3</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">28.3</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">51.0</td></tr>
<tr><td class="benchmark-cell" style="padding:5px 5px;padding-left:14px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:14px;font-weight:600;line-height:1.22;color:#171717">IFEval</div></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(29, 111, 208, 0.08);vertical-align:middle;font-size:14px;line-height:1.2;">86.7</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;"><strong>93.4</strong></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">77.5</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">31.4</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">90.2</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;"><strong style="color:#1D6FD0">93.7</strong></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">88.0</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">44.4</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">90.8</td></tr>
<tr><td class="benchmark-cell" style="padding:5px 5px;padding-left:14px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:14px;font-weight:600;line-height:1.22;color:#171717">Multi-IF</div></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(29, 111, 208, 0.08);vertical-align:middle;font-size:14px;line-height:1.2;">71.8</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;"><strong style="color:#1D6FD0">76.8</strong></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">57.1</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">40.3</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">73.6</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">75.9</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">65.9</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">45.9</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">71.4</td></tr>
<tr><td colspan="10" style="padding:5px 10px;font-weight:600;color:#1D6FD0;border-bottom:1px solid rgba(29, 111, 208, 0.2);background:rgba(29, 111, 208, 0.14)">综合知识</td></tr>
<tr><td class="benchmark-cell" style="padding:5px 5px;padding-left:14px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:14px;font-weight:600;line-height:1.22;color:#171717">MMLU-Pro</div></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(29, 111, 208, 0.08);vertical-align:middle;font-size:14px;line-height:1.2;"><strong>70.8</strong></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">65.2</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">64.3</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">56.0</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;"><strong style="color:#1D6FD0">78.0</strong></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">65.8</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">65.7</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">68.3</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">63.1</td></tr>
<tr><td class="benchmark-cell" style="padding:5px 5px;padding-left:14px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:14px;font-weight:600;line-height:1.22;color:#171717">MMLU-Redux</div></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(29, 111, 208, 0.08);vertical-align:middle;font-size:14px;line-height:1.2;"><strong>84.7</strong></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">80.0</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">80.0</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">71.8</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;"><strong style="color:#1D6FD0">88.7</strong></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">78.9</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">79.8</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">83.7</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">80.0</td></tr>
<tr><td class="benchmark-cell" style="padding:5px 5px;padding-left:14px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:14px;font-weight:600;line-height:1.22;color:#171717">HLE</div></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(29, 111, 208, 0.08);vertical-align:middle;font-size:14px;line-height:1.2;"><strong>8.9</strong><sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">6.2<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">2.6<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">4.8<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;"><strong style="color:#1D6FD0">9.9</strong><sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">6.6<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">4.9<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">3.8<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">6.9<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td></tr>
<tr><td class="benchmark-cell" style="padding:5px 5px;padding-left:14px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:14px;font-weight:600;line-height:1.22;color:#171717">GPQA-Diamond</div></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(29, 111, 208, 0.08);vertical-align:middle;font-size:14px;line-height:1.2;"><strong>70.2</strong><sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">55.8<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">45.6<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">43.3<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;"><strong style="color:#1D6FD0">77.1</strong><sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">55.9<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">51.3<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">57.6<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">51.3<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td></tr>
<tr><td class="benchmark-cell" style="padding:5px 5px;padding-left:14px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:14px;font-weight:600;line-height:1.22;color:#171717">SuperGPQA</div></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(29, 111, 208, 0.08);vertical-align:middle;font-size:14px;line-height:1.2;"><strong>40.8</strong></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">26.2</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">38.6</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">30.3</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;"><strong style="color:#1D6FD0">52.8</strong></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">39.9</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">37.8</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">38.7</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">34.5</td></tr>
<tr><td colspan="10" style="padding:5px 10px;font-weight:600;color:#1D6FD0;border-bottom:1px solid rgba(29, 111, 208, 0.2);background:rgba(29, 111, 208, 0.14)">长文本</td></tr>
<tr><td class="benchmark-cell" style="padding:5px 5px;padding-left:14px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:14px;font-weight:600;line-height:1.22;color:#171717">AA-LCR</div></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(29, 111, 208, 0.08);vertical-align:middle;font-size:14px;line-height:1.2;"><strong>59.0</strong><sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">5.3<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">28.7<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">17.0<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;"><strong style="color:#1D6FD0">61.0</strong><sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">24.3<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">17.3<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">33.0<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">0.0<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td></tr>
<tr><td class="benchmark-cell" style="padding:5px 5px;padding-left:14px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:14px;font-weight:600;line-height:1.22;color:#171717">NoLiMa</div></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(29, 111, 208, 0.08);vertical-align:middle;font-size:14px;line-height:1.2;"><strong style="color:#1D6FD0">68.1</strong></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">0.7</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">17.1</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">3.9</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">43.5</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">5.1</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">1.1</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">2.3</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">0.5</td></tr>
<tr><td class="benchmark-cell" style="padding:5px 5px;padding-left:14px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:14px;font-weight:600;line-height:1.22;color:#171717">LongBenchPro</div></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(29, 111, 208, 0.08);vertical-align:middle;font-size:14px;line-height:1.2;"><strong>44.8</strong></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">23.7</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">8.2</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">42.2</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;"><strong style="color:#1D6FD0">58.4</strong></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">34.8</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">27.9</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">53.5</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">19.6</td></tr>
<tr><td class="benchmark-cell" style="padding:5px 5px;padding-left:14px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:14px;font-weight:600;line-height:1.22;color:#171717">LongBench v2</div></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(29, 111, 208, 0.08);vertical-align:middle;font-size:14px;line-height:1.2;"><strong>43.7</strong></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">30.3</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">24.9</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">33.2</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;"><strong style="color:#1D6FD0">47.3</strong></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">36.0</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">32.0</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">42.7</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">30.4</td></tr>
<tr><td colspan="10" style="padding:5px 10px;font-weight:600;color:#1D6FD0;border-bottom:1px solid rgba(29, 111, 208, 0.2);background:rgba(29, 111, 208, 0.14)">工具调用</td></tr>
<tr><td class="benchmark-cell" style="padding:5px 5px;padding-left:14px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:14px;font-weight:600;line-height:1.22;color:#171717">τ³-Bench Banking</div></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(29, 111, 208, 0.08);vertical-align:middle;font-size:14px;line-height:1.2;"><strong style="color:#1D6FD0">20.8</strong><sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">7.2<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">2.1</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">3.9</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">6.8<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">5.6<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">1.2</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">4.1</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">3.4</td></tr>
<tr><td class="benchmark-cell" style="padding:5px 5px;padding-left:14px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:14px;font-weight:600;line-height:1.22;color:#171717">τ²-Bench Telecom</div></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(29, 111, 208, 0.08);vertical-align:middle;font-size:14px;line-height:1.2;"><strong style="color:#1D6FD0">97.1</strong></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">90.4</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">69.0<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">20.8<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">92.1<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">40.9</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">28.1<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">20.8<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">16.1<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td></tr>
<tr><td class="benchmark-cell" style="padding:5px 5px;padding-left:14px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:14px;font-weight:600;line-height:1.22;color:#171717">BFCL v4</div></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(29, 111, 208, 0.08);vertical-align:middle;font-size:14px;line-height:1.2;"><strong style="color:#1D6FD0">66.6</strong></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">61.1</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">43.6</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">36.6</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">56.8</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">52.2</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">43.7</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">47.0</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">49.2</td></tr>
<tr><td colspan="10" style="padding:5px 10px;font-weight:600;color:#1D6FD0;border-bottom:1px solid rgba(29, 111, 208, 0.2);background:rgba(29, 111, 208, 0.14)">代码智能体</td></tr>
<tr><td class="benchmark-cell" style="padding:5px 5px;padding-left:14px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:14px;font-weight:600;line-height:1.22;color:#171717">SWE-bench Verified</div></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(29, 111, 208, 0.08);vertical-align:middle;font-size:14px;line-height:1.2;"><strong style="color:#1D6FD0">46.4</strong></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">6.0</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">5.0</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">2.0</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">33.6</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">36.8</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">3.0</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">15.0</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">0.4</td></tr>
<tr><td class="benchmark-cell" style="padding:5px 5px;padding-left:14px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:14px;font-weight:600;line-height:1.22;color:#171717">SWE-bench Pro</div></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(29, 111, 208, 0.08);vertical-align:middle;font-size:14px;line-height:1.2;"><strong>14.4</strong></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">0.6</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">0.8</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">0.0</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;"><strong style="color:#1D6FD0">28.2</strong></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">12.3</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">0.1</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">3.3</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">0.4</td></tr>
<tr><td class="benchmark-cell" style="padding:5px 5px;padding-left:14px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:14px;font-weight:600;line-height:1.22;color:#171717">Terminal-Bench v2.1</div></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(29, 111, 208, 0.08);vertical-align:middle;font-size:14px;line-height:1.2;"><strong>8.6</strong><sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">4.5<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">3.0<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">0.4<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;"><strong style="color:#1D6FD0">25.8</strong><sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">13.9<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">3.8<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">1.9<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">1.9</td></tr>
<tr><td colspan="10" style="padding:5px 10px;font-weight:600;color:#1D6FD0;border-bottom:1px solid rgba(29, 111, 208, 0.2);background:rgba(29, 111, 208, 0.14)">搜索智能体</td></tr>
<tr><td class="benchmark-cell" style="padding:5px 5px;padding-left:14px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:14px;font-weight:600;line-height:1.22;color:#171717">BrowseComp-ZH</div></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(29, 111, 208, 0.08);vertical-align:middle;font-size:14px;line-height:1.2;"><strong style="color:#1D6FD0">43.5</strong></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">9.8</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">18.2</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">4.7</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">39.6</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">21.1</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">3.3</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">7.0</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">13.2</td></tr>
<tr><td class="benchmark-cell" style="padding:5px 5px;padding-left:14px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:14px;font-weight:600;line-height:1.22;color:#171717">BrowseComp Top100</div></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(29, 111, 208, 0.08);vertical-align:middle;font-size:14px;line-height:1.2;"><strong style="color:#1D6FD0">39.7</strong></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">13.7</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">19.3</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">6.0</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">33.3</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">19.0</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">4.7</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">6.3</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">9.7</td></tr>
<tr><td class="benchmark-cell" style="padding:5px 5px;padding-left:14px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:14px;font-weight:600;line-height:1.22;color:#171717">GAIA Text-103</div></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(29, 111, 208, 0.08);vertical-align:middle;font-size:14px;line-height:1.2;"><strong style="color:#1D6FD0">88.7</strong></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">49.5</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">47.9</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">30.1</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">78.6</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">57.3</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">26.5</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">39.5</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">41.1</td></tr>
<tr><td colspan="10" style="padding:5px 10px;font-weight:600;color:#1D6FD0;border-bottom:1px solid rgba(29, 111, 208, 0.2);background:rgba(29, 111, 208, 0.14)">通用智能体</td></tr>
<tr><td class="benchmark-cell" style="padding:5px 5px;padding-left:14px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:14px;font-weight:600;line-height:1.22;color:#171717">GDPval-AA v2</div></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(29, 111, 208, 0.08);vertical-align:middle;font-size:14px;line-height:1.2;"><strong style="color:#1D6FD0">19.6</strong><sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">4.5</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">0.0</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">0.0</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">11.7</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">0.0<sup style="font-size:0.72em;opacity:0.7">&dagger;</sup></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">0.0</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">0.0</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">0.0</td></tr>
<tr><td class="benchmark-cell" style="padding:5px 5px;padding-left:14px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:14px;font-weight:600;line-height:1.22;color:#171717">Claw-Gym</div></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(29, 111, 208, 0.08);vertical-align:middle;font-size:14px;line-height:1.2;"><strong>59.2</strong></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">19.3</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">25.5</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">31.3</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">51.6</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;"><strong style="color:#1D6FD0">60.0</strong></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">33.7</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">37.9</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">2.7</td></tr>
<tr><td class="benchmark-cell" style="padding:5px 5px;padding-left:14px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:14px;font-weight:600;line-height:1.22;color:#171717">WildClaw</div></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(29, 111, 208, 0.08);vertical-align:middle;font-size:14px;line-height:1.2;"><strong style="color:#1D6FD0">23.9</strong></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">10.2</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">9.2</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">8.9</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">17.0</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">20.0</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">8.9</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">14.3</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">4.5</td></tr>
<tr><td class="benchmark-cell" style="padding:5px 5px;padding-left:14px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:14px;font-weight:600;line-height:1.22;color:#171717">QwenClaw</div></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(29, 111, 208, 0.08);vertical-align:middle;font-size:14px;line-height:1.2;"><strong style="color:#1D6FD0">42.9</strong></td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">19.3</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">18.2</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">14.5</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);border-left:1px solid rgba(29, 111, 208, 0.25);vertical-align:middle;font-size:14px;line-height:1.2;">37.1</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">36.4</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">16.8</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">16.7</td><td style="padding:5px 3px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:14px;line-height:1.2;">4.5</td></tr>
</tbody></table>
  <p style="margin:6px 0 0;font-size:11px;line-height:1.55;opacity:0.75">1. <strong style="color:#1D6FD0">蓝色加粗</strong>为该行全场最优结果（含 4B 级模型）；<strong>黑色加粗</strong>为 2B 级模型中的最优结果。<br>2. 带 <sup style="font-size:0.72em;opacity:0.7">&dagger;</sup> 的分数取自 Artificial Analysis 官方公布值，其余为内部复现结果。</p>
</div>

## 训练流程

MiniCPM5-2B 的训练过程是 **[UltraData 分级数据管理体系](https://arxiv.org/pdf/2602.09003)** 的一次完整实践，覆盖 base training、mid-training 与后训练三个阶段。

**Base training** 采用逐级推进的训练配方，包含 stable training 与 decay training，用于建立基础语言能力与训练稳定性。随后进入 **mid-training**，进一步强化目标能力并适配数据分布。训练语料来自我们同步开源的 [Ultra-FineWeb](https://huggingface.co/datasets/openbmb/Ultra-FineWeb)、[Ultra-FineWeb-L3](https://huggingface.co/datasets/openbmb/Ultra-FineWeb-L3)、[UltraX](https://huggingface.co/datasets/openbmb/UltraX-Preview)、[UltraData-Code](https://huggingface.co/datasets/openbmb/UltraData-Code) 与 [UltraData-Math](https://huggingface.co/datasets/openbmb/UltraData-Math)。

**后训练阶段**分为 **SFT**、**RL** 与 **OPD** 三步。我们先使用 **400B tokens deep-thinking SFT** 建立深度思考和通用对话能力，相关 SFT 数据已同步开源为 [UltraData-SFT-2605](https://huggingface.co/datasets/openbmb/UltraData-SFT-2605)与[UltraData-SFT-Agent-2609](https://huggingface.co/datasets/openbmb/UltraData-SFT-Agent-2609)。随后针对数学、代码、Agent 和写作等方向训练专用 **RL teacher**（相关数据已同步开源为[UltraData-RL-2609](https://huggingface.co/datasets/openbmb/UltraData-RL-2609)），并通过 **On-Policy Distillation (OPD)** 将这些 teacher 的能力蒸馏回同一个发布模型。

![MiniCPM5-2B 训练流程](https://raw.githubusercontent.com/OpenBMB/MiniCPM/main/assets/minicpm5/minicpm5_2b_training_recipe.jpg)

### RL + OPD 带来了什么？

**RL + OPD** 是 MiniCPM5-2B 后训练中的关键环节。**RL** 阶段，使用了 [JustRL II](https://panhaoxuan.notion.site/justrl-ii-small-llms-to-128k-reasoning-with-a-critic-cn) 阐述的 critic-based 算法，大幅提升训练稳定性，并在多个领域取得了显著的收益。在下面列出的基准中，RL + OPD 在推理与通用能力上平均提升 **↑ 10.96 分**，Agent 能力平均提升 **↑ 6.96 分**。

**OPD** 阶段对 16 个 RL 训练所得到的专家模型（含 5 个 agentic 专家模型）实现了能力合并。训练方式上，我们在 response 序列的每个位置分别对学生模型和教师模型 logits 计算全词表的反向 KL 散度作为优势估计值，替代原有的 verification-based advantage；训练数据上，我们的 OPD 直接复用各 RL teacher 训练时 prompt 作为蒸馏数据，无需额外构造语料。

![MiniCPM5-2B RL + OPD 增益](https://raw.githubusercontent.com/OpenBMB/MiniCPM/main/assets/minicpm5/minicpm5_2b_rl_opd_score_gains.png)

## 快速上手

> [!Tip]
> 我们建议在生成时使用以下采样参数组合：`temperature=1.0, top_p=0.95, min_p=0.0`。
>
> 如果遇到重复输出，请尝试：`temperature=1.0, top_p=0.95, min_p=0.0, repetition_penalty=1.05`。
>
> 请注意，不同推理框架对采样参数的支持程度有所差异。

### vLLM

```bash
pip install "vllm>=0.21"
vllm serve openbmb/MiniCPM5-2B --port 8000
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openbmb/MiniCPM5-2B",
    "messages": [{"role": "user", "content": "你是谁？可以简单介绍一下自己吗？"}],
    "max_tokens": 128,
    "temperature": 1.0
  }'
```

### SGLang

```bash
pip install "sglang[srt]>=0.5.16"
python -m sglang.launch_server --model-path openbmb/MiniCPM5-2B --port 30000
```

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openbmb/MiniCPM5-2B",
    "messages": [{"role": "user", "content": "你是谁？可以简单介绍一下自己吗？"}],
    "max_tokens": 128,
    "temperature": 1.0
  }'
```

**投机采样（DSpark）**：我们同步开源了为 MiniCPM5-2B 训练的 DSpark 草稿模型 [MiniCPM5-2B-DSpark](https://huggingface.co/openbmb/MiniCPM5-2B-DSpark)。在 SGLang 中启用后可加速解码，且不改变目标模型的输出：

```bash
python -m sglang.launch_server \
  --model-path openbmb/MiniCPM5-2B \
  --trust-remote-code \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path openbmb/MiniCPM5-2B-DSpark \
  --speculative-dspark-block-size 7 \
  --port 30000
```
### Llama.cpp

```bash
llama-server -m MiniCPM5-2B-F16.gguf -a MiniCPM5-2B --port 8080 -ngl 99 -c 8192 --jinja
```

在这个例子里 `-c 8192` 设置了上下文长度为 8192，可以根据需要修改上下文长度。

```bash
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniCPM5-2B",
        "messages": [{"role": "user", "content": "1+1=?"}],
        "temperature": 1.0, "top_p": 0.95, "min_p": 0.0, "max_tokens": 256
    }'
```

llama.cpp 默认设置 `min_p=0.05`，会过滤掉概率低于最高概率 token 5% 的 token。这种过滤反而可能引发重复——它恰恰过滤掉了模型摆脱重复循环所需的那些 token。我们明确将 `min_p` 设为 `0.0` 以禁用该过滤。

### Transformers

```bash
pip install -U "transformers>=5.6" accelerate torch
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = "openbmb/MiniCPM5-2B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)
messages = [{"role": "user", "content": "你是谁？可以简单介绍一下自己吗？"}]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

## 工具调用

工具调用**推荐使用 SGLang**。MiniCPM5-2B 以 XML 格式产出工具调用，SGLang 内置的 `minicpm5` parser 会自动将其转换为 OpenAI 兼容的 `tool_calls` 字段。

```bash
python -m sglang.launch_server --model-path openbmb/MiniCPM5-2B --port 30000 \
    --tool-call-parser minicpm5      # 或：--tool-call-parser auto
```

## GitHub Cookbooks 与 Agent Skills

MiniCPM5-2B 使用**标准** `LlamaForCausalLM` **架构**，主流推理引擎可直接加载，**无需自定义算子，也无模型代码 fork**。逐步部署和微调说明请参考下方 GitHub cookbooks；Agent Skills 作为 GitHub 资源提供给使用 Cursor / Claude Code 类 coding agent 的用户。

### 部署

| 后端         | 模型格式 / 适用场景                              | Cookbook                                                                                        | Agent Skill                                                                                                               |
| ------------ | ------------------------------------------------ | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Transformers | BF16 / FP16，本地 Python 推理，GPU + CPU         | [transformers.md](https://github.com/OpenBMB/MiniCPM/blob/main/docs/deployment/transformers.md) | [minicpm5-deploy-transformers](https://github.com/OpenBMB/MiniCPM/blob/main/skills/minicpm5-deploy-transformers/SKILL.md) |
| vLLM         | BF16 / FP16 OpenAI server                        | [vllm.md](https://github.com/OpenBMB/MiniCPM/blob/main/docs/deployment/vllm.md)                 | [minicpm5-deploy-vllm](https://github.com/OpenBMB/MiniCPM/blob/main/skills/minicpm5-deploy-vllm/SKILL.md)                 |
| SGLang       | BF16 / FP16 OpenAI server，推荐用于 tool calling | [sglang.md](https://github.com/OpenBMB/MiniCPM/blob/main/docs/deployment/sglang.md)             | [minicpm5-deploy-sglang](https://github.com/OpenBMB/MiniCPM/blob/main/skills/minicpm5-deploy-sglang/SKILL.md)             |
| llama.cpp    | GGUF，CPU/GPU 本地推理                           | [llama_cpp.md](https://github.com/OpenBMB/MiniCPM/blob/main/docs/deployment/llama_cpp.md)       | [minicpm5-deploy-llama-cpp](https://github.com/OpenBMB/MiniCPM/blob/main/skills/minicpm5-deploy-llama-cpp/SKILL.md)       |
| Ollama       | GGUF，本地端侧运行                               | [ollama.md](https://github.com/OpenBMB/MiniCPM/blob/main/docs/deployment/ollama.md)             | [minicpm5-deploy-ollama](https://github.com/OpenBMB/MiniCPM/blob/main/skills/minicpm5-deploy-ollama/SKILL.md)             |
| LM Studio    | GGUF，Mac 桌面应用与 OpenAI server               | [lmstudio.md](https://github.com/OpenBMB/MiniCPM/blob/main/docs/deployment/lmstudio.md)         | [minicpm5-deploy-lmstudio](https://github.com/OpenBMB/MiniCPM/blob/main/skills/minicpm5-deploy-lmstudio/SKILL.md)         |
| MLX          | MLX / 4bit，Apple Silicon 本地推理               | [mlx.md](https://github.com/OpenBMB/MiniCPM/blob/main/docs/deployment/mlx.md)                   | [minicpm5-deploy-mlx](https://github.com/OpenBMB/MiniCPM/blob/main/skills/minicpm5-deploy-mlx/SKILL.md)                   |
| ArcLight     | GGUF 本地端侧 / CPU / 桌面 / 服务器              | [arclight.md](https://github.com/OpenBMB/MiniCPM/blob/main/docs/deployment/arclight.md)         | [minicpm5-deploy-arclight](https://github.com/OpenBMB/MiniCPM/blob/main/skills/minicpm5-deploy-arclight/SKILL.md)         |
| vLLM Ascend  | BF16 / FP16 OpenAI server                        | [vllm_ascend.md](https://github.com/OpenBMB/MiniCPM/blob/main/docs/deployment/vllm_ascend.md)   | [minicpm5-deploy-vllm-ascend](https://github.com/OpenBMB/MiniCPM/blob/main/skills/minicpm5-deploy-vllm-ascend/SKILL.md)   |
| LiteRT-LM    | `.litertlm` 端侧运行时：Android / iOS / 桌面 / IoT，CPU + GPU | [litert.md](https://github.com/OpenBMB/MiniCPM/blob/main/docs/deployment/litert.md)             | [minicpm5-deploy-litert](https://github.com/OpenBMB/MiniCPM/blob/main/skills/minicpm5-deploy-litert/SKILL.md) 

### 微调

| 框架          | 适用场景        | Cookbook                                                                                      | Agent Skill                                                                                                                   |
| ------------- | --------------- | --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| TRL + PEFT    | LoRA / SFT 微调 | [trl.md](https://github.com/OpenBMB/MiniCPM/blob/main/docs/finetune/trl.md)                   | [minicpm5-finetune-trl](https://github.com/OpenBMB/MiniCPM/blob/main/skills/minicpm5-finetune-trl/SKILL.md)                   |
| LLaMA-Factory | 微调            | [llamafactory.md](https://github.com/OpenBMB/MiniCPM/blob/main/docs/finetune/llamafactory.md) | [minicpm5-finetune-llamafactory](https://github.com/OpenBMB/MiniCPM/blob/main/skills/minicpm5-finetune-llamafactory/SKILL.md) |
| ms-swift      | 微调            | [ms_swift.md](https://github.com/OpenBMB/MiniCPM/blob/main/docs/finetune/ms_swift.md)         | [minicpm5-finetune-ms-swift](https://github.com/OpenBMB/MiniCPM/blob/main/skills/minicpm5-finetune-ms-swift/SKILL.md)         |
| unsloth       | 微调            | [unsloth.md](https://github.com/OpenBMB/MiniCPM/blob/main/docs/finetune/unsloth.md)           | [minicpm5-finetune-unsloth](https://github.com/OpenBMB/MiniCPM/blob/main/skills/minicpm5-finetune-unsloth/SKILL.md)           |

### 其他支持的框架

除上文列出的部署与微调框架外，MiniCPM5-2B 也支持通过 FlagOS 进行多芯片部署。

#### FlagOS 介绍

为解决不同 AI 芯片大规模落地应用，北京智源研究院联合众多科研机构、芯片企业、系统厂商、算法和软件相关单位等国内外机构共同发起并创立了 FlagOS 开源社区。

FlagOS 社区致力于打造面向多种 AI 芯片的统一、开源的系统软件栈，包括大型算子库、统一AI编译器、并行训推框架、统一通信库等核心开源项目，构建「模型-系统-芯片」三层贯通的开放技术生态，通过“一次开发跨芯迁移”释放硬件计算潜力，打破不同芯片软件栈之间生态隔离，有效降低开发者的迁移成本。FlagOS 社区构建人工智能软硬件生态，突破单一闭源垄断，推动AI硬件技术大范围落地发展，立足中国、拥抱全球合作。

官网速递：[https://flagos.io](https://flagos.io/)

<details>
<summary>FlagOS 多 AI 芯片支持与使用方式</summary>

#### FlagOS 多 AI 芯片支持

基于 FlagOS 极短时间内适配 MiniCPM5-2B 到 9 种不同的 AI 芯片，得益于众智 FlagOS 的多芯片统一 AI 系统软件栈的能力。目前，在 FlagOS 团队构建的面向多架构人工智能芯片的大模型自动迁移、适配与发布平台 FlagRelease 上，已发布 MiniCPM5-2B 的多芯片版本。细节如下：

| Vendor    | ModelScope                                                                                                | Huggingface                                                                                     |
| --------- | --------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Nvidia    | [MiniCPM5-2B-nvidia-FlagOS](https://www.modelscope.cn/models/FlagRelease/MiniCPM5-2B-nvidia-FlagOS)       | [MiniCPM5-2B-nvidia-FlagOS](https://huggingface.co/FlagRelease/MiniCPM5-2B-nvidia-FlagOS)       |
| Hygon     | [MiniCPM5-2B-hygon-FlagOS](https://www.modelscope.cn/models/FlagRelease/MiniCPM5-2B-hygon-FlagOS)         | [MiniCPM5-2B-hygon-FlagOS](https://huggingface.co/FlagRelease/MiniCPM5-2B-hygon-FlagOS)         |
| Metax     | [MiniCPM5-2B-metax-FlagOS](https://www.modelscope.cn/models/FlagRelease/MiniCPM5-2B-metax-FlagOS)         | [MiniCPM5-2B-metax-FlagOS](https://huggingface.co/FlagRelease/MiniCPM5-2B-metax-FlagOS)         |
| Iluvatar  | [MiniCPM5-2B-iluvatar-FlagOS](https://www.modelscope.cn/models/FlagRelease/MiniCPM5-2B-iluvatar-FlagOS)   | [MiniCPM5-2B-iluvatar-FlagOS](https://huggingface.co/FlagRelease/MiniCPM5-2B-iluvatar-FlagOS)   |
| Zhenwu    | [MiniCPM5-2B-zhenwu-FlagOS](https://www.modelscope.cn/models/FlagRelease/MiniCPM5-2B-zhenwu-FlagOS)       | [MiniCPM5-2B-zhenwu-FlagOS](https://huggingface.co/FlagRelease/MiniCPM5-2B-zhenwu-FlagOS)       |
| Mthreads  | [MiniCPM5-2B-mthreads-FlagOS](https://www.modelscope.cn/models/FlagRelease/MiniCPM5-2B-mthreads-FlagOS)   | [MiniCPM5-2B-mthreads-FlagOS](https://huggingface.co/FlagRelease/MiniCPM5-2B-mthreads-FlagOS)   |
| Kunlunxin | [MiniCPM5-2B-kunlunxin-FlagOS](https://www.modelscope.cn/models/FlagRelease/MiniCPM5-2B-kunlunxin-FlagOS) | [MiniCPM5-2B-kunlunxin-FlagOS](https://huggingface.co/FlagRelease/MiniCPM5-2B-kunlunxin-FlagOS) |
| Ascend    | [MiniCPM5-2B-ascend-FlagOS](https://modelscope.cn/models/FlagRelease/MiniCPM5-2B-ascend-FlagOS)           | [MiniCPM5-2B-ascend-FlagOS](https://huggingface.co/FlagRelease/MiniCPM5-2B-ascend-FlagOS)       |
| ARM-v9    | [MiniCPM5-2B-Armv9-FlagOS](https://modelscope.cn/models/FlagRelease/MiniCPM5-2B-Armv9-FlagOS)             | [MiniCPM5-2B-Armv9-FlagOS](https://huggingface.co/FlagRelease/MiniCPM5-2B-Armv9-FlagOS)         |

#### FlagOS 使用方式

##### 使用 FlagOS 在 Nvidia 体验性能加速

###### From FlagRelease（**推荐**）

FlagRelease是FlagOS团队构建的一套面向多架构人工智能芯片的大模型自动迁移、适配与发布平台，已发布MiniCPM5-2B的多芯片版本。FlagRelease 已内置相关软件包，无需用户安装。

###### FlagRelease 镜像关键版本信息

###### FlagRelease 使用速递

| Vendor    | ModelScope                                                                                                | Huggingface                                                                                     |
| --------- | --------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Nvidia    | [MiniCPM5-2B-nvidia-FlagOS](https://www.modelscope.cn/models/FlagRelease/MiniCPM5-2B-nvidia-FlagOS)       | [MiniCPM5-2B-nvidia-FlagOS](https://huggingface.co/FlagRelease/MiniCPM5-2B-nvidia-FlagOS)       |
| Hygon     | [MiniCPM5-2B-hygon-FlagOS](https://www.modelscope.cn/models/FlagRelease/MiniCPM5-2B-hygon-FlagOS)         | [MiniCPM5-2B-hygon-FlagOS](https://huggingface.co/FlagRelease/MiniCPM5-2B-hygon-FlagOS)         |
| Metax     | [MiniCPM5-2B-metax-FlagOS](https://www.modelscope.cn/models/FlagRelease/MiniCPM5-2B-metax-FlagOS)         | [MiniCPM5-2B-metax-FlagOS](https://huggingface.co/FlagRelease/MiniCPM5-2B-metax-FlagOS)         |
| Iluvatar  | [MiniCPM5-2B-iluvatar-FlagOS](https://www.modelscope.cn/models/FlagRelease/MiniCPM5-2B-iluvatar-FlagOS)   | [MiniCPM5-2B-iluvatar-FlagOS](https://huggingface.co/FlagRelease/MiniCPM5-2B-iluvatar-FlagOS)   |
| Zhenwu    | [MiniCPM5-2B-zhenwu-FlagOS](https://www.modelscope.cn/models/FlagRelease/MiniCPM5-2B-zhenwu-FlagOS)       | [MiniCPM5-2B-zhenwu-FlagOS](https://huggingface.co/FlagRelease/MiniCPM5-2B-zhenwu-FlagOS)       |
| Mthreads  | [MiniCPM5-2B-mthreads-FlagOS](https://www.modelscope.cn/models/FlagRelease/MiniCPM5-2B-mthreads-FlagOS)   | [MiniCPM5-2B-mthreads-FlagOS](https://huggingface.co/FlagRelease/MiniCPM5-2B-mthreads-FlagOS)   |
| Kunlunxin | [MiniCPM5-2B-kunlunxin-FlagOS](https://www.modelscope.cn/models/FlagRelease/MiniCPM5-2B-kunlunxin-FlagOS) | [MiniCPM5-2B-kunlunxin-FlagOS](https://huggingface.co/FlagRelease/MiniCPM5-2B-kunlunxin-FlagOS) |
| Ascend    | [MiniCPM5-2B-ascend-FlagOS](https://modelscope.cn/models/FlagRelease/MiniCPM5-2B-ascend-FlagOS)           | [MiniCPM5-2B-ascend-FlagOS](https://huggingface.co/FlagRelease/MiniCPM5-2B-ascend-FlagOS)       |
| ARM-v9    | [MiniCPM5-2B-Armv9-FlagOS](https://modelscope.cn/models/FlagRelease/MiniCPM5-2B-Armv9-FlagOS)             | [MiniCPM5-2B-Armv9-FlagOS](https://huggingface.co/FlagRelease/MiniCPM5-2B-Armv9-FlagOS)         |

###### 从零开始

- 依赖Python3.12, GLIBC_2.39, GLIBCXX_3.4.33, CXXABI_1.3.15 环境

###### Vllm 版本

###### 安装 FlagOS 算子库

官方仓库：[https://github.com/flagos-ai/FlagGems](https://github.com/flagos-ai/FlagGems)

```PowerShell
pip install flag-gems==4.2.1rc0
pip install triton==3.5.1
```

###### 开启加速

通过在vllm执行推理的源码中增加flagGems的导入即可开启flagGems加速

```Bash
import flag_gems
flag_gems.enable(record=True, once=True, path="/root/gems.txt")
```

```Bash
vllm serve ${model_path} \
--trust-remote-code \
--dtype bfloat16 \
--enforce-eager \
--port ${Port} \
--served-model-name ${model_name} \
--gpu-memory-utilization 0.85
```

##### 使用 FlagOS 统一多芯片后端插件

**[vllm-plugin-FL](https://github.com/flagos-ai/vllm-plugin-FL)** 是一个为 **vLLM** 推理/服务框架构建的插件，它基于 **FlagOS 的统一多芯片后端**开发，旨在扩展 vLLM 在多种硬件环境下的功能和性能表现。

###### vllm-plugin-FL 使用

| 厂商   | 从零开始                                                                                                       | 从 FlagRelease 开始                                                                              |                                                                                           |
| ------ | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| 英伟达 | [vllm-plugin-FL/MiniCPM5-2B](https://github.com/flagos-ai/vllm-plugin-FL/blob/main/examples/minicpm/README.md) | [MiniCPM5-2B-ModelScope](https://www.modelscope.cn/models/FlagRelease/MiniCPM5-2B-nvidia-FlagOS) | [MiniCPM5-2B-nvidia-FlagOS](https://huggingface.co/FlagRelease/MiniCPM5-2B-nvidia-FlagOS) |

</details>

## 局限性与免责声明

本模型不具备自主意识或法律主体资格，其输出仅为基于统计模式的文本生成结果，可能不准确、有偏见或具冒犯性，也可能被精心设计的提示词（“越狱”）操纵而产生不符合预期的内容。对政治、健康、金融、法律等敏感话题的回答未经专家审核，不应视为专业建议。

本模型按“**现状**”提供，不附带任何明示或默示担保，开发者不对使用本模型产生的任何损害承担责任。使用者应仅将模型用于合法、合规且符合伦理的目的，自行配置必要的安全措施，并按当地要求标识 AI 生成内容；不得故意越狱、注入攻击或诱导模型产生有害内容，若进行此类测试，风险自担。

## 开源协议

MiniCPM 模型权重与相关代码依照 [Apache-2.0](https://github.com/OpenBMB/MiniCPM/blob/main/LICENSE) 协议发布。

## 引用

如果觉得我们的工作有帮助，请引用：

```bibtex
@article{minicpm4,
  title={Minicpm4: Ultra-efficient llms on end devices},
  author={MiniCPM, Team},
  journal={arXiv preprint arXiv:2506.07900},
  year={2025}
}
```
