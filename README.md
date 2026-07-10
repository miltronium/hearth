# HEARTH

> **Working title.** HEARTH = an on-device intelligence layer. Rename freely.
> Tagline: *the always-on local fire every agent draws from.*

HEARTH is a **standalone, local-first model service for Apple Silicon**. It exposes
local LLMs, embeddings, and fine-tuned adapters behind a stable, OpenAI-compatible
API so that **any agent or client** — CAMBOT, Claude Code, shell scripts, other bots —
can offload work to on-device models, escalate to a frontier model only when it's
actually worth it, and get *better over time* through local fine-tuning.

HEARTH is **not** part of CAMBOT. CAMBOT is simply its first consumer. The whole point
is a reusable resource that outlives and out-scopes any single client.

---

## Why this exists

Frontier-model tokens (e.g. Claude Code) are the scarce resource. A large fraction of
day-to-day agent work is *not* frontier-hard: summarizing a file before it enters
context, ranking search hits, drafting a commit message, classifying an intent,
extracting fields, first-draft boilerplate. Paying premium tokens for that work is
what burns limits.

HEARTH's thesis: **route the cheap, high-volume work to a capable local model, and
reserve frontier tokens for genuine reasoning.** Everything else in this project —
routing, fine-tuning, embedded inference — exists to make that split safe, measurable,
and improvable.

## What it does

- **Offload** — cheap tasks run locally (MLX on Apple Silicon), never touching your frontier budget.
- **Escalate** — a policy layer decides when a task is too hard and hands off to a frontier/remote model.
- **Embed** — local embeddings + a small vector store give agents cheap, grounded context (RAG).
- **Train** — LoRA/QLoRA fine-tuning on your own code and docs, on-device, with an eval gate before anything ships.
- **Serve any client** — OpenAI-compatible HTTP, a `hearth` CLI, a Swift SDK, and an MCP server (so Claude Code itself can delegate subtasks locally).
- **Run fully offline** — an embedded Swift path (Apple Foundation Models / Core ML) for on-device inference with no daemon.

## Who consumes it

| Client | How it talks to HEARTH |
| --- | --- |
| **CAMBOT** (Swift core + Python MCP) | Swift SDK and/or HTTP |
| **Claude Code** | HEARTH MCP server → Claude delegates subtasks to local model |
| **Shell / CI** | `hearth` CLI |
| **Any OpenAI-SDK app** | point `base_url` at HEARTH |

## Status

🟢 **Phases 0–7 all shipped and green** — the planned build is complete. Every target
capability below is implemented and tested: OpenAI-compatible gateway (streaming +
embeddings), a router/policy layer with escalation + token budgeting + observability,
local RAG, LoRA/QLoRA fine-tuning with an eval gate, a Swift SDK + Python client + MCP
server, an offline embedded Swift path (Foundation Models, with a Core ML seam), a plugin
API, and multi-model serving + a quantization/export pipeline.

**202 Python tests + a Swift package, all green** on the `echo` backend with no model
downloaded. The two validations that needed real hardware / a live consumer are **now done
on Apple Silicon** (Apple M3 Pro / 36 GB): an end-to-end LoRA training run on real 7B weights
(train → eval gate both directions → promote → live serving) and live CAMBOT / Claude Code /
Swift wiring showing **2,210 estimated frontier tokens saved** over an all-local session. Full
evidence: [docs/RESULTS.md](docs/RESULTS.md). See [docs/ROADMAP.md](docs/ROADMAP.md) for the
phase-by-phase result log.

## Run it now

```bash
uv sync --extra dev            # install (core + test deps; no MLX needed)
uv run pytest -q               # 202 passing (1 skip: sqlite-vec extension absent)
uv run hearth doctor           # environment preflight
uv run hearth run "hello"      # one-shot (echo backend until MLX is installed)
uv run hearth serve            # OpenAI-compatible server on http://127.0.0.1:8080
uv run hearth stats            # token-savings + escalation rollups
uv run hearth mcp              # MCP server (stdio) so Claude Code can offload subtasks

# real Apple Silicon inference:
uv sync --extra mlx            # pulls mlx + mlx-lm
uv run hearth models pull mlx-community/Qwen2.5-Coder-7B-Instruct-4bit
HEARTH_BACKEND=mlx uv run hearth serve
```

**Optional extras:** `mlx` (real inference), `remote` (Anthropic escalation),
`embeddings` (MLX RAG embeddings), `mcp` (MCP server), `vec` (sqlite-vec vector store),
`coreml` (Core ML export), `dev` (tests). Everything installs and runs without them —
the core install uses the offline `echo` backend and a dependency-free vector store.

## CLI surface

`hearth doctor · serve · run · mcp · stats · train · eval · models (list/pull/rm/convert/export-coreml) · rag (ingest/query) · adapters (list/promote/retire) · plugins`

## Documentation map

| Doc | What's in it |
| --- | --- |
| [docs/PROPOSAL.md](docs/PROPOSAL.md) | The pitch: problem, vision, goals/non-goals, principles, success metrics, risks. **Start here.** |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design: layers, component interfaces, backends, data flow, deployment models, tech stack. |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Phased build plan (Phase 0–7), deliverables, acceptance criteria. |
| [docs/API.md](docs/API.md) | The gateway API contract: OpenAI-compatible endpoints + HEARTH extensions. |
| [docs/INTEGRATION.md](docs/INTEGRATION.md) | How CAMBOT, Claude Code (MCP), and generic clients consume HEARTH. |
| [docs/PLUGINS.md](docs/PLUGINS.md) | Writing third-party providers / vector stores against the plugin entry points. |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Architecture Decision Records — the *why* behind the big choices. |
| [docs/RUNBOOK_training.md](docs/RUNBOOK_training.md) | Validate the LoRA path end-to-end on real weights (Apple Silicon). |
| [docs/RUNBOOK_consumer_wiring.md](docs/RUNBOOK_consumer_wiring.md) | Wire CAMBOT + Claude Code to a live HEARTH and read the token-savings numbers. |
| [docs/HANDOFF.md](docs/HANDOFF.md) | For a Claude Code running on real hardware: how to pick up the two hardware-blocked follow-ups and partner back. |
| [docs/RESULTS.md](docs/RESULTS.md) | Real-hardware validation results (Apple M3 Pro): the LoRA train→gate→promote→serve run and live consumer token-savings numbers. |
| [examples/](examples/) | Runnable consumer examples: CAMBOT offload (Python + Swift), Claude Code MCP registration. |

## Stack (see ARCHITECTURE for rationale)

- **Gateway + training:** Python 3.12, FastAPI, [`mlx-lm`](https://github.com/ml-explore/mlx-examples), MLX embeddings.
- **Alternate backends:** Ollama/llama.cpp (GGUF), Core ML, Apple Foundation Models (Swift).
- **Client SDKs:** Swift package (for CAMBOT), Python client, plain HTTP.
- **Hardware baseline:** Apple Silicon, 32 GB+ unified memory.

## Quickstart

```bash
# install (editable, from the repo)
uv sync --extra mlx            # core + real Apple Silicon inference

# pull a local coder model and start the daemon
uv run hearth models pull mlx-community/Qwen2.5-Coder-7B-Instruct-4bit
HEARTH_BACKEND=mlx uv run hearth serve   # OpenAI-compatible server on http://127.0.0.1:8080

# one-shot from the CLI
uv run hearth run "summarize this file" --file src/foo.swift

# any OpenAI client just points at it
export OPENAI_BASE_URL=http://127.0.0.1:8080/v1
```
