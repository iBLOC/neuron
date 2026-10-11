---
title: 模型调度器
---

# 模型调度器

neuron 宿主常常受内存约束（边缘盒子、桌面应用）。模型调度器让一份有界的内存预算同时服务*超过其容纳量的模型*：LLM 请求排队而不是 OOM，当另一个模型需要腾地方时，常驻模型按最近最少使用（LRU）换出，空闲模型自行回收。

## 心智模型

- 权重驻留在内存中的模型是一个**常驻模型（resident）**。每个在途请求持有其常驻模型的一个**租约（lease）**；每个模型的租约数以 `NEURON_SCHEDULER_MAX_CONCURRENCY` 为上限。
- **同模型请求立即执行**（取一个租约）。对*另一个*模型的请求，预算装得下就并排加载；装不下则排队等待常驻模型的工作排空，然后按 LRU 优先换出空闲常驻模型，直到换装放得下（**排空后换装**，drain-then-swap）。
- 零租约且无排队需求的常驻模型在 `NEURON_SCHEDULER_IDLE_SECS`（默认 120）后回收。
- 预算默认为系统内存的 80%；每个模型的参考体积随包附带于 `assets/model_memory_profiles.json`，并在运行时按观测到的卸载 RSS 修正（EWMA，持久化到 `<model_dir>/scheduler/observed_profiles.json`）——调度器越跑越会猜你的模型。

## 覆盖范围

调度器设计上是能力面无关的，但 v1 只接了 **LLM**（`/v1/chat/completions` + `/v1/messages`，流式与非流式）。流式响应持有租约直到 SSE body 完成或客户端断开，因此模型换装绝不会从一条活跃流底下把权重卸走。

## 观测调度器

```bash
curl http://localhost:8085/v1/scheduler/status
```

```json
{
  "capability": "llm",
  "enabled": true,
  "budget_mb": 12800,
  "residents": [
    { "model": "minicpm5-1b", "leases": 2, "idle_secs": 14 }
  ],
  "queue": []
}
```

管理员可以强制换出一个常驻模型（仍有租约或排队工作时报 409）：

```bash
curl -X DELETE http://localhost:8085/v1/scheduler/models/minicpm5-1b
```

## 可调参数

| 环境变量 | 默认值 | 用途 |
|---|---|---|
| `NEURON_SCHEDULER_ENABLED` | `true` | `false` 恢复传统的总是加载路径（无队列、无换出）——给自行管理内存的宿主留的逃生舱 |
| `NEURON_SCHEDULER_BUDGET_MB` | 80% 内存 | 常驻权重的预算 |
| `NEURON_SCHEDULER_IDLE_SECS` | `120` | 空闲回收时长 |
| `NEURON_SCHEDULER_MAX_CONCURRENCY` | `4` | 每个常驻模型的租约数 |
| `NEURON_SCHEDULER_MAX_QUEUE_WAIT_SECS` | `300` | 排队停留超过此时长即报 `503 queue_timeout` |
| `NEURON_SCHEDULER_PLATFORM` | 取自 `NEURON_DEVICE` | 内存档案键（`cpu` / `cuda` / `metal`） |

GPU 常驻权重对进程 RSS 不可见，因此 `cuda`/`metal` 平台直接使用随包的参考估计；RSS 驱动的修正只在 `cpu` 上运行。

## 下一步

- [/howto/ocr](/zh/howto/ocr) — OCR 管线（不受调度器管理；其模型很小且为显式注册）
- [/reference/env-vars](/zh/reference/env-vars) — 完整的可调参数清单
