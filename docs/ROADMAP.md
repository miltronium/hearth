# HEARTH — Roadmap

**Status:** Draft. Companion to [PROPOSAL.md](PROPOSAL.md) and [ARCHITECTURE.md](ARCHITECTURE.md).

Phases are ordered so that **each one ships something usable on its own**. You get
token savings at the end of Phase 1; everything after compounds it. Don't build ahead —
resist the urge to start Phase 4 (training) before Phase 2 (measurement) can prove it helps.

Guiding rule: **build the walking skeleton first, then thicken it.** A thin end-to-end
path (client → gateway → one model → response) exists at the end of Phase 0 and is never
broken again.

---

## Phase 0 — Scaffold & walking skeleton

**Goal:** a repo you can build in, and one request that goes end-to-end through one model.

- Standalone git repo, `pyproject.toml`, Python 3.12, `uv`/`pipx` install.
- `hearth serve` starts FastAPI on loopback; `hearth doctor` checks environment (Apple
  Silicon, RAM, MLX importable, model dir writable).
- `MLXProvider` loads exactly one hardcoded model and answers `/v1/chat/completions`
  (non-streaming is fine).
- Bench 7B vs 14B coder models on this machine → record numbers → resolve Open Question #1.

**Acceptance:** `curl` a chat completion and get a coherent local response. Latency numbers
recorded for candidate models.

---

## Phase 1 — Gateway + MLX + registry + CLI → **Offload (G1)**

**Goal:** cheap tasks run locally through a real API, from any OpenAI client.

- Full OpenAI-compat: `/v1/chat/completions` (+ streaming), `/v1/embeddings`, `/v1/models`.
- `MLXProvider` productionized: streaming, adapter slot (unused yet), `footprint()`.
- Registry with `models.yaml` (pinned + checksummed); `hearth models pull|list|rm`.
- `hearth run` CLI (one-shot; `--file`, `--intent`).
- Local bearer-token auth.

**Acceptance:** point an OpenAI SDK at HEARTH and run summarize/extract/draft tasks with no
frontier calls. `hearth run "summarize" --file X` works.

---

## Phase 2 — Router/policy + escalation + budget + observability → **Smart escalation + proof (G2, G8)**

**Goal:** HEARTH decides local-vs-escalate, stays in budget, and *proves* the savings.

- Task classifier (rules + tiny model + `intent` hint short-circuit).
- `routing.yaml` policy engine; `RemoteProvider` for escalation (endpoint/auth via config).
- Confidence gating for escalation-eligible classes.
- Token-budget accountant (per-day remote budget; prefer local when scarce).
- Observability: per-request records, `hearth stats`, `/v1/hearth/admin/metrics`,
  estimated-frontier-tokens-saved.

**Acceptance:** mixed workload routes correctly; escalations are logged with reasons;
`hearth stats` shows a credible weekly token-savings number and escalation rate.

---

## Phase 3 — Embeddings + local RAG/memory → **Grounded context (G3)**

**Goal:** agents retrieve cheap local context instead of stuffing frontier prompts.

- Embedding provider (MLX embeddings or `bge`/`nomic` via a provider).
- `VectorStore` interface + embedded impl (SQLite+`sqlite-vec` or LanceDB).
- Per-project collections; `hearth rag ingest <path>`; `/v1/hearth/rag/query`.
- Optional retrieve-then-answer with a local model.

**Acceptance:** ingest a repo, query it, get relevant chunks in <1 s; a client uses
retrieved context to answer without sending whole files to a frontier model.

---

## Phase 4 — Fine-tuning + adapter registry + eval harness → **Improvement loop (G4)**

**Goal:** the model gets measurably better at *your* work, safely.

- Dataset builder (JSONL + provenance) from repo artifacts / accepted outputs.
- `hearth train` → LoRA/QLoRA via `mlx_lm.lora`.
- Eval harness with golden sets per class; **adapter must beat incumbent to be promotable.**
- Adapter registry lifecycle: `candidate → promoted → retired`; A/B behind a flag.
- Adapters hot-swappable per request in `MLXProvider`.

**Acceptance:** train an adapter on a domain task, eval shows a lift over base, promote it,
and routed requests use it. A regression on the golden set blocks promotion.

---

## Phase 5 — Swift SDK + CAMBOT integration + MCP server → **Client-agnostic reuse (G5)**

**Goal:** the actual consumers wire in — CAMBOT, Claude Code, generic apps.

- Swift package client (async, streaming) → CAMBOT calls HEARTH over HTTP/UDS.
- HEARTH **MCP server** so Claude Code can delegate subtasks to the local model
  (this directly saves *your* Claude tokens — see [INTEGRATION.md](INTEGRATION.md)).
- Python client convenience wrapper.
- **Client-agnostic conformance test suite** that runs with no CAMBOT present.

**Acceptance:** CAMBOT offloads a real task to HEARTH; Claude Code uses the HEARTH MCP tool
to summarize/extract locally; conformance suite is green without CAMBOT.

---

## Phase 6 — Embedded Swift path (Foundation Models / Core ML) → **Offline on-device (G6)**

**Goal:** fully offline inference inside a Swift app, no daemon.

- `FoundationModelsProvider` via Swift (on-device ~3B, no downloads).
- Optional Core ML export path for small models / ANE acceleration.
- Swift SDK exposes an in-process mode mirroring the HTTP API's shape.
- Resolve Open Question #3 (adapter portability to embedded mode).

**Acceptance:** CAMBOT performs an on-device inference with networking disabled, no daemon
running.

---

## Phase 7 — Plugin API + multi-model serving + quant pipeline → **Long-term extensibility (G7)**

**Goal:** grow without touching the core.

- Documented plugin API for new `ModelProvider`s, routes, and vector stores (entry-points).
- Multi-model concurrent serving with memory-aware scheduling.
- Quantization/conversion pipeline (`hearth models convert`) for new checkpoints.
- Hardening: graceful degradation, model warmup, health/readiness endpoints.

**Acceptance:** a third-party backend loads as a plugin with zero core edits; two models
serve concurrently within the RAM ceiling.

---

## Sequencing notes

- **1 → 2 is the critical spine.** Everything downstream assumes measurement (Phase 2).
- **3, 4, 6 are somewhat independent** and can reorder based on what hurts most: reorder
  4 before 3 if domain quality is the bigger pain than context cost.
- **5 can start partially after Phase 1** (a basic Swift/HTTP client) but the MCP server and
  conformance suite belong after routing exists.
- Keep the walking skeleton green at every phase boundary.
