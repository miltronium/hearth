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

## What it does (target capabilities)

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

🟢 **Phase 0 complete — walking skeleton runs.** A request goes end-to-end
(client → gateway → provider → response) on the `echo` fallback backend, with no model
downloaded. MLX is wired as the real backend behind an optional extra. Next: Phase 1
(productionize MLX + registry + streaming). See [docs/ROADMAP.md](docs/ROADMAP.md).

## Run it now

```bash
uv sync --extra dev            # install (core + test deps; no MLX needed)
uv run pytest -q               # 9 passing
uv run hearth doctor           # environment preflight
uv run hearth run "hello"      # one-shot (echo backend until MLX is installed)
uv run hearth serve            # OpenAI-compatible server on http://127.0.0.1:8080

# real Apple Silicon inference:
uv sync --extra mlx            # pulls mlx + mlx-lm
HEARTH_BACKEND=mlx uv run hearth serve
```

## Documentation map

| Doc | What's in it |
| --- | --- |
| [docs/PROPOSAL.md](docs/PROPOSAL.md) | The pitch: problem, vision, goals/non-goals, principles, success metrics, risks. **Start here.** |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design: layers, component interfaces, backends, data flow, deployment models, tech stack. |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Phased build plan (Phase 0–7), deliverables, acceptance criteria. |
| [docs/API.md](docs/API.md) | The gateway API contract: OpenAI-compatible endpoints + HEARTH extensions. |
| [docs/INTEGRATION.md](docs/INTEGRATION.md) | How CAMBOT, Claude Code (MCP), and generic clients consume HEARTH. |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Architecture Decision Records — the *why* behind the big choices. |

## Target stack (see ARCHITECTURE for rationale)

- **Gateway + training:** Python 3.12, FastAPI, [`mlx-lm`](https://github.com/ml-explore/mlx-examples), MLX embeddings.
- **Alternate backends:** Ollama/llama.cpp (GGUF), Core ML, Apple Foundation Models (Swift).
- **Client SDKs:** Swift package (for CAMBOT), Python client, plain HTTP.
- **Hardware baseline:** Apple Silicon, 32 GB+ unified memory.

## Quickstart (target — not yet built)

```bash
# install
pipx install hearth            # or: uv tool install hearth

# pull a local coder model and start the daemon
hearth models pull qwen2.5-coder:7b-mlx
hearth serve                   # OpenAI-compatible server on http://127.0.0.1:8080

# one-shot from the CLI
hearth run "summarize this file" --file src/foo.swift

# any OpenAI client just points at it
export OPENAI_BASE_URL=http://127.0.0.1:8080/v1
```
