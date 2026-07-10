# RUNBOOK — Wiring CAMBOT + Claude Code to a live HEARTH and reading token savings (G2/G8)

This runbook lands the "live consumer wiring" follow-up: point real consumers (CAMBOT over
HTTP, Claude Code over MCP) at a running HEARTH daemon, drive a workload, and read the
token-savings numbers HEARTH records. Every command/endpoint below exists in the codebase.

> **What needs real resources.** The wiring, endpoints, and measurement here all work with
> the shipped `echo` backend (offline, no GPU) — you can rehearse the whole flow today.
> Producing *credible* savings numbers on a *real* workload needs (a) the MLX backend for
> real local answers and (b) the actual CAMBOT app / a live Claude Code session. Those two
> are the parts that can't be manufactured in-repo.

---

## 1. Start the HEARTH daemon

```sh
uv run hearth serve                    # binds 127.0.0.1:8080 (loopback) by default
```

- Backend: defaults to `auto` (MLX when importable, else `echo`). Force it with
  `HEARTH_BACKEND=mlx` (real answers) or `HEARTH_BACKEND=echo` (offline rehearsal). See
  `src/hearth/config.py`.
- On first start HEARTH writes a bearer token to `~/.hearth/token` (mode 0600). The HTTP
  API requires it on all routes except the liveness/readiness probes.

```sh
export HEARTH_URL="http://127.0.0.1:8080"
export HEARTH_TOKEN="$(cat ~/.hearth/token)"
```

- Readiness (deployment gates traffic on this; 200 only once the warm model is resident,
  503 otherwise — echo is always ready):

  ```sh
  curl -s "$HEARTH_URL/v1/hearth/admin/ready"
  curl -s "$HEARTH_URL/v1/hearth/admin/health"     # liveness + version + backend + model
  ```

---

## 2. Wire CAMBOT (HTTP consumer)

CAMBOT talks to HEARTH over HTTP through a client SDK. Both SDKs mirror the same endpoints.

### Python (`hearth.client.HearthClient`)

Runnable example: `examples/cambot_offload.py`.

```sh
# dry run (offline, safe): prints what it would send
uv run python examples/cambot_offload.py

# against the live daemon:
uv run python examples/cambot_offload.py --live
```

The example offloads a `summarize` and a `classify` subtask, both with
`allow_escalation=False`, so HEARTH answers locally and never makes a remote call on
CAMBOT's behalf (`src/hearth/client.py`).

### Swift (`HearthClient`)

Snippet: `examples/cambot_offload.swift`, using the package at
`swift/Sources/Hearth/HearthClient.swift`. It calls `summarize`/`classify`, which set
`HearthOptions(allowEscalation: false)` internally. Add the `Hearth` package as a
dependency (see `swift/README.md`) and call `offloadExample()` from a CAMBOT call site.

Both clients hit the same routes: `POST /v1/chat/completions`, `POST /v1/embeddings`,
`POST /v1/hearth/rag/query`.

---

## 3. Wire Claude Code (MCP consumer)

Register HEARTH's stdio MCP server so Claude Code can delegate subtasks to the local model.
Full instructions + a config JSON block: **`examples/claude_code_mcp.md`**. In short:

```sh
uv sync --extra mcp
claude mcp add hearth -- uv run hearth mcp     # run from the HEARTH repo root
```

The MCP tools (`hearth_summarize`, `hearth_classify`, `hearth_extract`, `hearth_draft`,
`hearth_rag_query`) run on HEARTH's router **in-process** with escalation disabled — no
HTTP, no bearer token, and no frontier tokens (`src/hearth/mcp/tools.py`,
`src/hearth/mcp/server.py`).

---

## 4. Drive a workload

Do real work through the consumers: have CAMBOT summarize/extract/classify captured text,
and have Claude Code call the `hearth_*` tools for subtasks it would otherwise burn frontier
tokens on. Each offloaded task is one that did not hit a frontier model — that is the
savings HEARTH accounts for (Phase 2 observability, `src/hearth/observability/`).

---

## 5. Read the token-savings numbers (G2/G8)

HEARTH estimates the frontier tokens it saved and its escalation rate. There are two
readouts:

### A. Admin metrics endpoint (authoritative for the running daemon)

```sh
curl -s -H "Authorization: Bearer $HEARTH_TOKEN" \
     "$HEARTH_URL/v1/hearth/admin/metrics?since=24h" | python -m json.tool
```

Served by `GET /v1/hearth/admin/metrics` (`src/hearth/gateway/app.py`), auth-gated, and
returns the same rollup shape as `hearth stats`. Key fields:

- `requests` — total handled.
- `estimated_frontier_tokens_saved` — the headline savings number.
- `escalations` and `escalation_rate` — how often HEARTH chose to escalate.
- `backend_mix`, `class_mix`, `latency_ms` (`p50`/`p95`).

`since` accepts a window like `7d` / `24h` / `30m` (parsed by `_parse_since`); omit it for
all-time.

### B. `hearth stats` CLI

```sh
uv run hearth stats --since 24h
```

> **Important caveat (Phase 2).** Metrics are held **in-memory per process**. A fresh
> `hearth stats` invocation is a *different process* from the daemon and therefore reports
> an **empty** store — it accumulates only within its own process. To read the *running
> daemon's* numbers, use the **admin metrics endpoint (A)** above, which queries the live
> daemon. `hearth stats` is documented this way in `src/hearth/cli.py`; persistence across
> processes is a listed future follow-up, not a bug to work around here.

---

## What can't be verified in-repo

- **Real savings magnitude.** Meaningful `estimated_frontier_tokens_saved` needs a real
  workload driven by the actual CAMBOT app and a live Claude Code session against the MLX
  backend. The plumbing (clients, MCP server, metrics endpoint) is fully in place and can
  be rehearsed on `echo`, but the numbers themselves come from real usage on real hardware.
- **CAMBOT app itself.** This repo ships the client SDKs CAMBOT depends on, not CAMBOT.

## Prerequisites not yet in the codebase

- There is **no** cross-process/persisted metrics store yet, so `hearth stats` cannot roll
  up a running daemon's numbers from a separate shell — use the admin metrics endpoint.
  Persisting records to JSONL is a listed follow-up (`src/hearth/cli.py:stats` docstring).
