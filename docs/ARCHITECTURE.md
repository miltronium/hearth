# HEARTH — Architecture

**Status:** Draft. Companion to [PROPOSAL.md](PROPOSAL.md).

This document describes the target system design. It is deliberately interface-first:
the contracts matter more than any single implementation, because backends and models
will churn while the contracts should not.

---

## 1. Layered overview

HEARTH is six cooperating layers plus cross-cutting observability:

```
┌────────────────────────────────────────────────────────────┐
│ 1. Gateway / API      OpenAI-compatible HTTP + extensions    │
│ 2. Router / Policy    task classification, select, escalate  │
│ 3. Provider layer     ModelProvider interface + backends     │
│ 4. Registry           models, quant, capabilities, adapters  │
│ 5. Memory / RAG       embeddings + local vector store        │
│ 6. Training           dataset curation, LoRA/QLoRA, eval      │
│ ── Observability ──   metrics, traces, token accounting       │
└────────────────────────────────────────────────────────────┘
```

Dependency direction is strictly downward (Gateway → Router → Provider → Registry).
Higher layers never reach around lower ones; lower layers know nothing of clients.

---

## 2. Gateway / API layer

- **Transport:** FastAPI (Python), HTTP on `127.0.0.1` by default (loopback only unless
  explicitly configured). Optional Unix domain socket for local clients.
- **Compatibility:** implements the subset of the OpenAI API that clients actually use —
  `/v1/chat/completions` (incl. streaming), `/v1/embeddings`, `/v1/models`. This means any
  OpenAI SDK works by changing `base_url`.
- **Extensions (additive, namespaced under `/v1/hearth/`):** `/route`, `/classify`,
  `/summarize`, `/rag/query`, `/train/*`, `/admin/*`. See [API.md](API.md).
- **Auth:** local bearer token (generated on first run, stored `0600`). Loopback-only by
  default makes this low-stakes, but the token gates the extension/admin routes.

The gateway is thin. It validates, authenticates, and hands off to the router. No model
logic lives here.

---

## 3. Router / Policy layer

The heart of the token-savings story. For each request:

1. **Classify** the task into a class: `summarize | extract | classify | rank | draft |
   code | reason | chat | embed`. Classification is cheap (rules + a tiny local classifier
   model; the request's declared `intent` hint short-circuits it when provided).
2. **Select** a backend + model for that class from policy (see routing table below).
3. **Gate on confidence** (optional, for escalation-eligible classes): run local, have the
   model or a lightweight judge score confidence; if below threshold, escalate.
4. **Gate on budget:** consult the token-budget accountant. Prefer local while remote
   budget is scarce; allow escalation while budget remains.
5. **Execute** via the chosen `ModelProvider`, streaming through to the client.
6. **Record** backend, latency, tokens, and estimated-frontier-tokens-saved.

Routing policy is **data, not code** — a declarative config (`routing.yaml`) so it can be
tuned without redeploying:

```yaml
# routing.yaml (illustrative)
defaults:
  local_model: qwen2.5-coder:7b-mlx
  remote_model: claude            # provider-configured; may be internal endpoint
  remote_budget_tokens_per_day: 200000
classes:
  summarize: { backend: local,  escalate: never }
  extract:   { backend: local,  escalate: never }
  classify:  { backend: local,  escalate: never }
  rank:      { backend: local,  escalate: never }
  draft:     { backend: local,  escalate: on_low_confidence, threshold: 0.6 }
  code:      { backend: local,  escalate: on_low_confidence, threshold: 0.7 }
  reason:    { backend: remote, escalate: always }
  chat:      { backend: local,  escalate: on_low_confidence, threshold: 0.65 }
```

**Escalation is a first-class, auditable event**, not a silent fallback. Every escalation
records *why* (low confidence / class policy / explicit client request) so the policy can
be tuned against real data.

---

## 4. Provider layer — the key abstraction

Every backend implements one interface. This is what keeps backend churn from becoming
rewrites.

```python
# Illustrative — Python
class ModelProvider(Protocol):
    name: str
    def capabilities(self) -> Capabilities: ...        # chat? embed? stream? adapters?
    async def generate(self, req: GenRequest) -> AsyncIterator[GenChunk]: ...
    async def embed(self, texts: list[str]) -> list[Vector]: ...
    def load(self, model_id: str, adapter: str | None = None) -> None: ...
    def unload(self, model_id: str) -> None: ...
    def footprint(self, model_id: str) -> ResourceEstimate: ...   # RAM/VRAM
```

Concrete providers:

| Provider | Backend | Role |
| --- | --- | --- |
| `MLXProvider` | `mlx-lm` on Apple Silicon | **Default.** Fast unified-memory inference; supports LoRA adapters natively. |
| `OllamaProvider` | Ollama / llama.cpp (GGUF) | Broad model catalog, easy pulls, alternate quant formats. |
| `CoreMLProvider` | Core ML | Path toward embedded/offline + ANE acceleration. |
| `FoundationModelsProvider` | Apple Foundation Models (Swift bridge) | On-device ~3B, fully offline, no downloads. Primarily via the Swift SDK. |
| `RemoteProvider` | Frontier/internal HTTP | Escalation target. Endpoint + auth are config; HEARTH stays agnostic. |

Adding a backend = one new class + a registry entry. No router or gateway change.

---

## 5. Registry

A declarative catalog of models and adapters — the source of truth for what can be served.

- **Model entries:** id, backend, quant, context length, capability flags, RAM footprint,
  source URI, checksum. Pinned and checksummed for reproducibility.
- **Adapter entries:** id, base model, task/domain, training run id, eval scores, status
  (`candidate | promoted | retired`). Adapters are hot-swappable per request.
- **Lifecycle:** `hearth models pull|list|rm`, `hearth adapters list|promote|retire`.
- **Memory policy:** default one resident base model + N small adapters; lazy load/unload
  driven by `footprint()` against a configured RAM ceiling.

---

## 6. Memory / RAG layer

Gives agents cheap, grounded context so they stop stuffing raw files into frontier prompts.

- **Embeddings** via a local model (MLX embeddings or `bge`/`nomic-embed` through a provider).
- **Vector store:** start with an embedded store (SQLite + `sqlite-vec`, or LanceDB) — no
  external service, file-based, portable. Pluggable behind a `VectorStore` interface.
- **Collections:** per-project namespaces (e.g. one per repo). Ingest = chunk → embed → store.
- **Query API:** `/v1/hearth/rag/query` returns top-k grounded chunks a client can inject,
  or HEARTH can answer directly with a local model + retrieved context.

RAG is optional and additive — Phase 3. Nothing else depends on it.

---

## 7. Training subsystem

Parameter-efficient fine-tuning, local, gated by eval. Explicitly *not* a general training
platform.

Pipeline:

```
curate dataset → format (chat/instruction) → LoRA/QLoRA train (mlx_lm.lora)
   → eval vs base on golden set → gate → register adapter (candidate)
   → human promote → serve
```

- **Datasets:** curated from your own code, docs, past sessions, and accepted outputs.
  Stored as versioned JSONL with provenance. A dataset builder helps assemble instruction
  pairs from repo artifacts.
- **Training:** `mlx_lm.lora` (LoRA/QLoRA) on Apple Silicon. Small base models (3B–14B).
- **Eval harness:** golden sets per task class; an adapter must **beat the incumbent** to be
  promotable. Metrics: task-specific (exact-match/F1 for extract/classify; win-rate via
  judge for draft/code). Results stored with the adapter entry.
- **Promotion:** a deliberate, logged action. Candidates are servable behind a flag for
  A/B before promotion.

---

## 8. Observability & token accounting

Cross-cutting, present from Phase 2.

- **Per-request record:** class, chosen backend/model/adapter, tokens in/out, latency,
  escalated?(+why), estimated-frontier-tokens-saved.
- **Rollups:** daily token-savings, escalation rate, backend mix, p50/p95 latency.
- **Surface:** `hearth stats` CLI + a `/v1/hearth/admin/metrics` endpoint. Structured logs
  (JSON lines) for later analysis.
- **Savings estimate:** for each locally-served request, estimate what the frontier call
  would have cost (tokens × class-typical multiplier) — this is the number that justifies
  the project.

---

## 9. Deployment models

HEARTH runs three ways; a client picks whichever fits.

1. **Local daemon (default).** A `launchd` LaunchAgent keeps HEARTH warm (models resident)
   and serves HTTP/UDS. Best latency; shared by all local clients.
2. **CLI / on-demand.** `hearth run ...` spawns, serves one request, exits (or attaches to a
   running daemon). Good for scripts and CI.
3. **Embedded Swift library.** For CAMBOT and other Swift apps that want *no* daemon: a
   Swift package wrapping Apple Foundation Models / Core ML for fully offline on-device
   inference. Same conceptual API, in-process. (Phase 6.)

---

## 10. Tech stack & rationale

| Concern | Choice | Why |
| --- | --- | --- |
| Gateway + training | Python 3.12, FastAPI | `mlx-lm`, training tooling, and OpenAI-compat ecosystem are Python-native |
| Default inference | MLX / `mlx-lm` | Fastest local inference on Apple Silicon; unified memory; native LoRA |
| Alt inference | Ollama (GGUF) | Easiest broad model catalog + quant variety |
| Embedded / offline | Core ML + Foundation Models | ANE acceleration; on-device model with no downloads; Swift-native for CAMBOT |
| Vector store | SQLite+`sqlite-vec` / LanceDB | File-based, embeddable, no service to run |
| Config | YAML (`routing.yaml`, `models.yaml`) | Routing/catalog are *data*; tune without redeploy |
| Client SDKs | Swift package + Python client + raw HTTP | Meets each consumer where it lives |
| Cross-client delegation | MCP server | Lets Claude Code itself offload subtasks to HEARTH |

See [DECISIONS.md](DECISIONS.md) for the ADRs behind these.

---

## 11. Directory layout (target)

```
hearth/
├── README.md
├── docs/                    # these design docs
├── pyproject.toml
├── src/hearth/
│   ├── gateway/             # FastAPI app, OpenAI-compat + extension routes
│   ├── router/              # classification, policy, budget, escalation
│   ├── providers/           # mlx.py, ollama.py, coreml.py, remote.py, base.py
│   ├── registry/            # model + adapter catalog
│   ├── memory/              # embeddings + vector store (RAG)
│   ├── training/            # dataset builder, lora runner, eval harness
│   ├── observability/       # metrics, token accounting, logging
│   └── cli/                 # `hearth` entrypoint
├── mcp/                     # HEARTH MCP server (for Claude Code et al.)
├── swift/                   # Swift SDK + embedded Foundation Models path
├── config/                  # routing.yaml, models.yaml (defaults)
├── evals/                   # golden sets per task class
└── tests/                   # incl. client-agnostic conformance suite
```
