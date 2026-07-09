# HEARTH — API Contract

**Status:** Draft. The API *is* the product — this contract should be stable while
backends and models churn beneath it. All examples are illustrative.

Base URL (default): `http://127.0.0.1:8080`
Auth: `Authorization: Bearer <token>` (token generated on first run, stored `0600`).
Loopback-only by default.

---

## Design rules

1. **OpenAI-compatible core.** `/v1/chat/completions`, `/v1/embeddings`, `/v1/models`
   match OpenAI's shapes so any SDK works by swapping `base_url`. No surprises.
2. **Extensions are additive & namespaced** under `/v1/hearth/`. A client that only speaks
   OpenAI never needs them.
3. **Routing hints are optional.** Clients may pass an `intent` hint to skip classification;
   omitting it is always valid.
4. **Escalation is transparent.** Responses report which backend/model served the request.

---

## Core (OpenAI-compatible)

### `POST /v1/chat/completions`

Standard OpenAI request. HEARTH adds **optional** fields (ignored by OpenAI clients):

```jsonc
{
  "model": "auto",                     // "auto" = let the router decide; or a specific id
  "messages": [{"role": "user", "content": "Summarize this diff"}],
  "stream": true,
  "hearth": {                          // optional extension block
    "intent": "summarize",             // routing hint; skips classification
    "allow_escalation": false,         // hard-pin to local for this call
    "adapter": "commit-msgs-v3"        // request a specific fine-tuned adapter
  }
}
```

Response adds a `hearth` block in the final chunk / object:

```jsonc
{
  "id": "chatcmpl-...",
  "choices": [{ "message": {"role": "assistant", "content": "..."} }],
  "usage": { "prompt_tokens": 812, "completion_tokens": 96 },
  "hearth": {
    "served_by": "local",              // "local" | "remote"
    "backend": "mlx",
    "model": "qwen2.5-coder:7b-mlx",
    "adapter": "commit-msgs-v3",
    "escalated": false,
    "estimated_frontier_tokens_saved": 908
  }
}
```

### `POST /v1/embeddings`

Standard OpenAI embeddings shape. `model: "auto"` selects the configured local embedder.

### `GET /v1/models`

Lists servable models and adapters from the registry (id, backend, context, capabilities).

---

## Extensions (`/v1/hearth/`)

### `POST /v1/hearth/route`

Ask the router what it *would* do, without executing — useful for debugging policy.

```jsonc
// req
{ "messages": [...], "intent": null }
// resp
{ "class": "code", "backend": "local", "model": "qwen2.5-coder:7b-mlx",
  "would_escalate": false, "reason": "class policy: code→local unless low confidence" }
```

### `POST /v1/hearth/classify`

Structured classification/extraction as a first-class op (returns typed JSON, not prose).

```jsonc
// req
{ "text": "...", "labels": ["bug", "feature", "question"] }
// resp
{ "label": "bug", "confidence": 0.91 }
```

### `POST /v1/hearth/summarize`

Convenience wrapper: summarize text/file with length + style controls. Always local unless
`allow_escalation: true`.

### `POST /v1/hearth/rag/ingest` · `POST /v1/hearth/rag/query`  *(Phase 3)*

```jsonc
// ingest
{ "collection": "cambot", "paths": ["Sources/"], "chunk": {"size": 800, "overlap": 100} }
// query
{ "collection": "cambot", "query": "how does astrisctl auth?", "k": 6, "answer": false }
// query resp (answer:false → just chunks)
{ "chunks": [{ "text": "...", "source": "Sources/.../Auth.swift", "score": 0.83 }] }
```

### `POST /v1/hearth/train/*` · `GET /v1/hearth/train/{run_id}`  *(Phase 4)*

Kick off / inspect LoRA runs. Long-running → returns a `run_id`; poll for status + eval.

```jsonc
// start
{ "base_model": "qwen2.5-coder:7b-mlx", "dataset": "commit-msgs.jsonl",
  "method": "qlora", "epochs": 3 }
// resp
{ "run_id": "train_...", "status": "queued" }
```

### Admin (`/v1/hearth/admin/`)

- `GET /admin/metrics` — token-savings rollups, escalation rate, backend mix, latency.
- `GET /admin/health` · `GET /admin/ready` — liveness/readiness (warm models loaded).
- `POST /admin/models/{id}/load|unload` — memory management.
- `POST /admin/adapters/{id}/promote|retire` — adapter lifecycle.

---

## Error model

Standard OpenAI-style error envelope, plus a `hearth.code` for HEARTH-specific cases:

```jsonc
{ "error": { "message": "remote budget exhausted; escalation denied",
             "type": "budget_exhausted", "code": "hearth.budget.exhausted" } }
```

Notable HEARTH error types: `budget_exhausted`, `escalation_denied`, `model_not_loaded`,
`adapter_not_found`, `backend_unavailable`. Clients should treat a local-only failure as
retryable-with-escalation only if their policy allows it.

---

## Streaming

SSE, OpenAI-compatible (`data: {...}\n\n`, terminating `data: [DONE]`). The final data event
before `[DONE]` carries the `hearth` telemetry block.
