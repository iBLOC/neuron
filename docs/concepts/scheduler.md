---
title: The Model Scheduler
---

# The Model Scheduler

neuron hosts are often memory-constrained (edge boxes, desktop apps). The
model scheduler makes a bounded memory budget serve *more models than fit*:
LLM requests queue instead of OOMing, resident weights are evicted
least-recently-used when a different model needs the room, and idle models
recycle themselves.

## The mental model

- A model whose weights are resident is a **resident**. Each in-flight
  request holds a **lease** on its resident; leases cap at
  `NEURON_SCHEDULER_MAX_CONCURRENCY` per model.
- **Same-model requests run immediately** (a lease is taken). A request for
  a *different* model loads alongside if the budget fits; otherwise it
  queues while resident-model work drains, then idle residents are evicted
  LRU-first until the swap fits (**drain-then-swap**).
- Residents with zero leases and no queued demand recycle after
  `NEURON_SCHEDULER_IDLE_SECS` (default 120).
- The budget defaults to 80% of system RAM; reference per-model sizes ship
  in `assets/model_memory_profiles.json` and are corrected at runtime by
  observed unload RSS (an EWMA persisted to
  `<model_dir>/scheduler/observed_profiles.json`) — the scheduler gets
  better at guessing your models as it runs.

## What it covers

The scheduler is capability-generic by design, but v1 wires **LLM only**
(`/v1/chat/completions` + `/v1/messages`, streaming and non-streaming).
Streaming responses hold their lease until the SSE body completes or the
client disconnects, so a model swap never unloads weights out from under a
live stream.

## Observing it

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

Administrators can force-evict a resident (409 while leases or queued work
remain):

```bash
curl -X DELETE http://localhost:8085/v1/scheduler/models/minicpm5-1b
```

## Knobs

| Env var | Default | Purpose |
|---|---|---|
| `NEURON_SCHEDULER_ENABLED` | `true` | `false` restores the legacy always-load path (no queue, no eviction) — the escape hatch for hosts that manage memory themselves |
| `NEURON_SCHEDULER_BUDGET_MB` | 80% RAM | Resident-weights budget |
| `NEURON_SCHEDULER_IDLE_SECS` | `120` | Idle recycle age |
| `NEURON_SCHEDULER_MAX_CONCURRENCY` | `4` | Leases per resident |
| `NEURON_SCHEDULER_MAX_QUEUE_WAIT_SECS` | `300` | Queue dwell before `503 queue_timeout` |
| `NEURON_SCHEDULER_PLATFORM` | from `NEURON_DEVICE` | Memory-profile key (`cpu` / `cuda` / `metal`) |

GPU-resident weights are invisible to process RSS, so on `cuda`/`metal`
platforms the shipped reference estimates are used as-is; RSS-driven
correction runs on `cpu` only.

## Next steps

- [/howto/ocr](/howto/ocr) — the OCR pipeline (not scheduler-managed; its
  models are small and explicitly registered)
- [/reference/env-vars](/reference/env-vars) — the full knob list
