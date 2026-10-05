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
  "model": "auto",                     // "auto"/"" = the router's per-class ladder, else the
                                       // registry default; or an id from GET /v1/models
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

**`model` is honoured, never substituted.** A specific id is served by exactly that model's
weights: with the `mlx` backend the gateway holds one provider per model id
(`hearth.serving.ModelPool`, LRU under `HEARTH_RAM_CEILING_GB`, default 24), loading on first
use and evicting the least-recently-used model when the next one would not fit. The `hearth`
block's `model` (and the stream's final chunk) is filled in by the provider instance that
generated, not copied from the request. An id the server cannot serve is refused **before**
anything is generated or a stream is opened — same for `stream: true` and for
`POST /v1/hearth/agent`:

```jsonc
// HTTP 404
{ "error": { "message": "The model 'bogus/model' does not exist in the HEARTH model registry (config/models.yaml). Servable models: [...]",
             "type": "invalid_request_error", "param": "model", "code": "model_not_found" } }
```

404 covers: an id not in `config/models.yaml`; a registered id without the `chat` capability
(e.g. the bge embed model); an id for another backend (e.g. `echo` while serving `mlx`). The
message lists what is servable. A registered chat model whose weights are **not on disk** is
not a 404: the load fails and the request gets `503` `provider_unavailable` whose message
names the `hearth models pull <id>` to run (HEARTH never downloads on load). The CLI applies
the same check: `hearth run --model <unknown>` / `hearth agent --model <unknown>` exit 2.

**`hearth.adapter` reports what served.** The response's `hearth.adapter` is the adapter
whose weights actually generated the answer: the requested id, or — when none was
requested — the task class's promoted adapter (the default path, unchanged). It is `null`
when base weights answered: no adapter selected, the request escalated to a remote, the
adapter failed to load and the request was retried on base weights, or the backend ignores
adapters (`echo`). An explicitly requested adapter that is not registered (or is retired) is
refused before anything runs, exactly like an unknown model:

```jsonc
// HTTP 404
{ "error": { "message": "adapter 'no-such-adapter' cannot be served: unknown adapter: 'no-such-adapter'",
             "type": "invalid_request_error", "param": "hearth.adapter", "code": "adapter_not_found" } }
```

### `POST /v1/embeddings`

Standard OpenAI embeddings shape. `model: "auto"` selects the configured local embedder.

### `GET /v1/models`

Lists exactly the models a chat request can name and be served by (id, backend, context,
capabilities): the registry's `chat` models for the active backend. With `mlx` that excludes
`echo` and embed-only entries. OpenAI clients — and the `/chat` page — build their model
picker from this list, so an entry that could only 404 would be a trap; and
`/v1/embeddings` ignores `model` (it always uses the configured embedder), so listing an embed
model would serve nobody. The list and the 404 above share one rule
(`hearth.serving.servable_for` / `check_model`), so they cannot disagree. Weights-on-disk is
not checked here: a listed model that was never pulled answers 503 with the pull command
(`hearth doctor --offline` reports which reachable models are on disk).

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
  A request that ended in an error (the local provider failed, or a remote stream died
  mid-answer) is recorded too: it counts in `requests` and in `failed` / `failure_rate`
  (added keys; nothing renamed), and — after a failed escalation — in `escalations_failed`,
  since the remote may already have received the prompt. `backend_mix` and `latency_ms`
  count only requests that were served an answer.
- `GET /admin/health` — liveness (unauthenticated): the process is up. Says nothing about
  weights.
- `GET /admin/ready` — readiness (unauthenticated). **Ready means the default model can
  serve a request now**: a load of it has completed with weights in memory at least once
  (warmup or any request; residency granted without a load never counts), its most recent
  load did not fail, and — for a backend that can check without loading (`mlx`) — its
  weights still resolve on disk. Residency is reported separately, so a default that was
  LRU-evicted to make room for another model stays `200` (it reloads on demand). With
  `HEARTH_WARMUP=false` nothing loads until the first request, so a default whose weights
  resolve on disk is `200` before that request; a backend that cannot check the disk stays
  `503 loading` until its first load.

  | code | `status` | `reason` / `detail` (examples) | meaning |
  |---|---|---|---|
  | 200 | `ready` | — | the default holds weights now (`loaded: true`) |
  | 200 | `ready` | detail `'<id>' loaded before and is not resident now (evicted to make room); it reloads on demand` | `loaded: false`, weights on disk |
  | 200 | `ready` | detail `warmup disabled (HEARTH_WARMUP=false); '<id>' is on disk and loads on the first request` | `loaded: false`, never loaded yet |
  | 503 | `loading` | `warmup in progress` | the startup warmup thread is loading the default weights |
  | 503 | `loading` | `weights for '<id>' are not loaded (HEARTH_WARMUP is off) and the '<backend>' backend cannot verify them without loading` | a plugin/test backend without a disk probe, before its first load |
  | 503 | `failed` | `weights for '<id>' do not resolve on disk: ModelNotOnDiskError: …` | never pulled, or deleted after loading |
  | 503 | `failed` | `last load of '<id>' failed: …` / `warmup of '<id>' failed: …` | the load raised (corrupt checkpoint, mlx missing, over the RAM ceiling) |
  | 503 | `failed` | `HEARTH_DEFAULT_MODEL='<id>' is not in the model registry …` | the configured default names no registered model (`auto` would silently be served by the catalog default) |
  | 503 | `failed` | `default model is not servable: …` | the default is registered but not a chat model of this backend |

  Every body also carries `backend`, `model` (the default id), `loaded` (the default holds
  weights right now) and `resident` (ids in memory). The `echo` backend is ready when chosen
  explicitly (`HEARTH_BACKEND=echo`). When `HEARTH_BACKEND=auto` (the default) falls back to
  echo because `mlx_lm` is not importable — usually a venv pruned by a bare `uv run` — `/ready`
  is `503` with `status: "stub"`, `backend: "echo"` and a `reason` naming the repair command,
  `/health` adds a `backend_fallback` field with the same text, and a WARNING is logged at
  startup. The server still starts (admin/metrics and non-inference CLI paths keep working);
  it just cannot be mistaken for real inference.
  Measured on 2026-10-05 with real weights (warmup on): `503 loading` at 0.05 s after start,
  `200 ready` at ~1.05 s (7B from page cache).
- `GET /admin/models` — what is resident right now, read off the provider instances
  themselves (auth required):

  ```jsonc
  { "backend": "mlx", "default": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    "ram_ceiling_gb": 24.0, "resident_ram_gb": 6.5,
    "resident": [ { "model": "mlx-community/Qwen2.5-3B-Instruct-4bit", "ram_gb": 2.0,
                    "provider_model": "mlx-community/Qwen2.5-3B-Instruct-4bit",
                    "loaded": true,
                    "loaded_path": "/Users/…/.hearth/models/models--mlx-community--Qwen2.5-3B-Instruct-4bit/snapshots/…",
                    "generations": 5 } ] }
  ```

  `loaded_path` is the directory `mlx_lm.load` actually read and `generations` counts the
  generations that instance ran — evidence of which weights answered, independent of anything
  a response says about itself. `ram_gb` is the registry's estimate used for the ceiling, not
  a measurement. Order is LRU (least recently used first).
- `POST /admin/models/{id}/load|unload` — memory management. *(planned; not implemented)*
- `POST /admin/adapters/{id}/promote|retire` — adapter lifecycle.

---

## Operator UI

### `GET /chat`

A chat page for the operator, served by the gateway itself. Open
`http://127.0.0.1:8080/chat` after `hearth serve`. Same origin as `/v1/chat/completions`,
so **no CORS middleware exists and none is needed**.

- **Self-contained.** One inline HTML document — no CDN script, no web font, no external
  stylesheet or image. The page issues exactly two requests, both relative: `/v1/models`
  and `/v1/chat/completions`. `tests/test_gateway_chat_ui.py` greps the served bytes and
  fails on any off-origin reference, because a single font request from a page discussing
  bank statements would be an egress channel.
- **Unauthenticated route, no credential in the document.** A browser navigation cannot
  send an `Authorization` header, so `/chat` is not behind `require_token`. The token is
  therefore *not* injected server-side: that would turn `GET /chat` into an
  unauthenticated token-disclosure endpoint, downgrading a `0600` file to no permission
  at all. Instead the operator pastes `~/.hearth/token` into the page once; it lives in
  `localStorage` on the loopback origin and is sent as a bearer header per request.
  **All `/v1/*` routes keep their auth dependency unchanged.**
- **Streaming.** Uses SSE (`stream: true`), rendering deltas as they arrive and honouring
  the `[DONE]` sentinel. `401`/`422`/`5xx` and mid-stream `error` events render as text
  rather than hanging.
- **Provenance.** Each reply shows the `hearth` telemetry — `served_by` (“served
  on-device” / “served remotely”), `model`, `backend`, `adapter` — and a visible
  **Truncated** warning when `finish_reason == "length"`.

The route is static markup; it exposes no data and grants no capability a caller without
a token did not already have. See `src/hearth/gateway/chat_ui.py` for the full rationale.

---

## Error model

Standard OpenAI-style error envelope, plus a `hearth.code` for HEARTH-specific cases:

```jsonc
{ "error": { "message": "remote budget exhausted; escalation denied",
             "type": "budget_exhausted", "code": "hearth.budget.exhausted" } }
```

A request naming a model the server cannot serve gets OpenAI's own shape: HTTP 404,
`type: invalid_request_error`, `param: model`, `code: model_not_found` (see
`POST /v1/chat/completions`).

Notable HEARTH error types: `budget_exhausted`, `escalation_denied`, `model_not_loaded`,
`adapter_not_found`, `backend_unavailable`. Clients should treat a local-only failure as
retryable-with-escalation only if their policy allows it.

---

## Streaming

SSE, OpenAI-compatible (`data: {...}\n\n`, terminating `data: [DONE]`). The final data event
before `[DONE]` carries the `hearth` telemetry block.

Every stream ends with `[DONE]`, on every path. A failure is an in-band `error` event just
before it: `hearth.provider.unavailable` (the provider failed), `model_not_found`,
`hearth.budget.exhausted`, `hearth.response_format.invalid_json`,
`hearth.metrics.unavailable` (the answer and its final telemetry chunk were delivered, but
the request could not be recorded), or `hearth.stream.internal_error` (anything else).
