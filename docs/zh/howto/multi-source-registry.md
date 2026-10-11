---
title: 多来源模型注册表
---

# 多来源模型注册表

neuron 内置**多来源模型注册表**，可以从 HuggingFace、hf-mirror（中国 CDN）、ModelScope 或任何自定义来源拉取模型权重。按引擎与按模型的来源覆盖让你能精细控制每个模型的来源。

## 快速开始：下载模型

下载模型最简单的方式是通过 HTTP API：

```bash
# Pull Qwen2.5-0.5B from HuggingFace
curl -X POST http://localhost:8085/v1/models/pull \
  -H "Content-Type: application/json" \
  -d '{"source": "huggingface", "repo": "Qwen/Qwen2.5-0.5B-Instruct"}'

# Same model, from ModelScope (China CDN)
curl -X POST http://localhost:8085/v1/models/pull \
  -H "Content-Type: application/json" \
  -d '{"source": "modelscope", "repo": "Qwen/Qwen2.5-0.5B-Instruct"}'
```

响应中包含一个可供轮询的 `job_id`：

```bash
curl http://localhost:8085/v1/models/pull
# → [{"id": "abc123", "source": "huggingface", "repo": "...", "status": "Running", ...}]

curl http://localhost:8085/v1/models/pull/abc123
# → {"id": "abc123", "status": "Running", "bytes_downloaded": 52428800, ...}

# Cancel if needed
curl -X DELETE http://localhost:8085/v1/models/pull/abc123
```

## 默认来源

默认情况下，neuron 使用 **ModelScope** 作为来源（通过 `NEURON_DEFAULT_SOURCE=modelscope` 设置）。这对中国及亚太地区的用户是正确的默认值；其他地区的用户应通过环境变量切换到 `huggingface`。

```bash
# Switch default source globally
export NEURON_DEFAULT_SOURCE=huggingface
```

## 来源覆盖语法

发起拉取请求时，可以使用 `<source>:<repo>` 简写：

```bash
# Full form
curl -X POST http://localhost:8085/v1/models/pull \
  -d '{"source": "huggingface", "repo": "Qwen/Qwen2.5-0.5B-Instruct"}'

# Shorthand
curl -X POST http://localhost:8085/v1/models/pull \
  -d '{"source_repo": "huggingface:Qwen/Qwen2.5-0.5B-Instruct"}'
```

如果用户已通过 `NEURON_LLM_SOURCE` / `NEURON_MODEL_SOURCE_*` 覆盖了来源，puller 还会剥离开头的 `<other-source>:` 前缀 — 因此，在一个按引擎来源为 `hf-mirror` 的进程中请求 `modelscope:openbmb/MiniCPM5-1B`，会正确路由到镜像而不是 404。

## 按引擎与按模型的来源覆盖

```bash
# All LLM models come from HuggingFace (overrides the default modelscope)
export NEURON_LLM_SOURCE=huggingface

# All TTS models come from ModelScope
export NEURON_TTS_SOURCE=modelscope

# This specific LLM model comes from a custom mirror
export NEURON_MODEL_SOURCE_LLM_QWEN2_5_0_5B="hf-mirror"

# All downloads from a custom source go through this endpoint
export NEURON_SOURCE_HUGGINGFACE_ENDPOINT="https://my-internal-mirror.example.com"
```

来源 ID 用下划线代替空格（例如 `huggingface`、`hf-mirror`、`modelscope`）。模型 ID 用下划线代替连字符（例如 `qwen2.5-0.5b`）。

## 自定义来源

可以在运行时通过 `Source` trait 添加自定义下载来源：

```rust
use neuron::models::source::{Source, SourceMetadata};

struct MyOrgSource {
    endpoint: String,
    auth_token: Option<String>,
}

#[async_trait]
impl Source for MyOrgSource {
    fn id(&self) -> &str { "myorg" }
    fn metadata(&self) -> SourceMetadata { SourceMetadata { name: "MyOrg", base_url: self.endpoint.clone() } }
    async fn download(&self, repo: &str, dest_dir: &Path, progress: impl Fn(DownloadProgress)) -> Result<()> {
        // custom download logic
    }
}

// Register on the catalog
state.catalog.register_source(Arc::new(MyOrgSource { ... }));
```

## 按来源配置端点

```bash
# Use a custom HuggingFace endpoint (e.g. enterprise proxy)
export NEURON_SOURCE_HUGGINGFACE_ENDPOINT="https://my-proxy.example.com"
export NEURON_SOURCE_HUGGINGFACE_TOKEN="..."

# Use a custom ModelScope endpoint
export NEURON_SOURCE_MODELSCOPE_ENDPOINT="https://my-modelscope-mirror"
```

## 目录去重

同一个 `(engine, id)` 可能在 `SUPPORTED_CATALOG` 中出现多次（每个来源一次）。`Catalog::entries()` 会把它们合并成单个 `CatalogEntry`，其 `sources: Vec<SourceRef>` 为每个仓库携带一个条目，因此 `GET /v1/models` 中每个模型只出现一次，同时暴露所有可用的下载来源。

```json
GET /v1/models
{
  "data": [
    {
      "id": "qwen2.5-0.5b",
      "sources": [
        {"id": "huggingface", "repo": "Qwen/Qwen2.5-0.5B-Instruct"},
        {"id": "modelscope", "repo": "Qwen/Qwen2.5-0.5B-Instruct"},
        {"id": "hf-mirror", "repo": "Qwen/Qwen2.5-0.5B-Instruct"}
      ]
    }
  ]
}
```

## 无 CLI 来源的 HTTP 回退

当 CLI 二进制不在磁盘上时，`huggingface` 与 `hf-mirror` 都会回退到流式 HTTP 路径（reqwest `bytes_stream`）。该 HTTP 回退会：

- 读取 `HF_TOKEN` 并附加为 `Authorization: Bearer <token>`（仅 HF 系来源）
- 读取 `HTTP_PROXY` / `HTTPS_PROXY` 环境变量并构建 `reqwest::Proxy`
- 分块流式写入磁盘；每 1 MiB 发布一次 `DownloadProgress`

ModelScope 来源仅使用 `modelscope-cli`（没有 HTTP 回退）。

## 失败的下载

如果下载中途失败，puller 会：
1. 将任务标记为 `Failed` 并附上错误信息
2. 自动清理磁盘上不完整的目录
3. 在目录（catalog）中递增该来源的失败计数器

之后你可以重试拉取 — puller 会从头开始（v0 不支持断点续传）。

## 为离线 / 隔离部署捆绑模型

对于桌面应用或隔离（air-gapped）服务器，可以把模型预捆绑进你的发行包：

```bash
# Build the model directory manually:
export NEURON_MODEL_DIRS=/opt/myapp/models
mkdir -p /opt/myapp/models
# (manually download weights via huggingface-cli or scp from another machine)

# neuron discovers models on startup by scanning NEURON_MODEL_DIRS
curl http://localhost:8085/v1/models
```

注册表会自动发现预捆绑的模型，无需任何来源拉取。对于不希望用户在首次启动时等待下载的生产级桌面应用，这是正确的模式。

## 后续步骤

- [将 neuron 嵌入 Rust 宿主](/zh/howto/embed-into-host) — 完整的 Tauri 2 演练
- [环境变量](/zh/reference/env-vars) — 完整的 `NEURON_*` 变量表
- [架构](/zh/concepts/architecture) — 注册表内部机制
