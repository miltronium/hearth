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
- **Escalate** — a policy layer decides when a task is too hard and hands off to a frontier/remote model. Opt-in: the shipped `config/routing.yaml` is no-egress; select `HEARTH_ROUTING_YAML=config/routing.remote.yaml` to enable escalation (see [docs/PRIVACY.md](docs/PRIVACY.md)).
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
server, an offline embedded Swift path (Foundation Models + a **working Core ML generation
loop**), a plugin API, and multi-model serving + a quantization/export pipeline.

**219 Python tests + a Swift package (two products, 20 tests), all green** on the `echo`
backend with no model downloaded, at the end of the phase build (a historical count: for what
is true now, run `uv run --no-sync pytest -q` and `uv run --no-sync python
scripts/hearth_status.py`). Three real-hardware validations are **done on Apple Silicon**:
an end-to-end LoRA training run on real 7B weights (train → eval gate both directions → promote →
live serving); live CAMBOT / Claude Code / Swift wiring showing **2,210 estimated frontier tokens
saved** over an all-local session; and (ADR-011) a **fully-offline Core ML generation loop** —
`hearth models export-coreml` → Swift `CoreMLProvider.generate` ran a real Qwen2.5 on the ANE with
no daemon and no network, greedy-matching the source model. Full evidence:
[docs/RESULTS.md](docs/RESULTS.md). See [docs/ROADMAP.md](docs/ROADMAP.md) for the
phase-by-phase result log.

## Quick start

New here? Follow **[docs/GUIDE.md §0, "Learn HEARTH in 15 minutes"](docs/GUIDE.md#0-learn-hearth-in-15-minutes)**.
The short version, from the repo root:

```bash
# install: every extra in ONE command (a partial sync uninstalls the others)
uv sync --extra mlx --extra mcp --extra dev --extra files
uv run --no-sync python -c "import mlx_lm, mcp, openpyxl, pypdf; print('ok')"   # verify

uv run --no-sync hearth doctor --offline   # safe to use offline right now? exit 0 = SAFE, 1 = UNSAFE
uv run --no-sync hearth models list        # (default) marks the model that serves
uv run --no-sync hearth run "hello"        # one local completion; stderr says which model served it
uv run --no-sync hearth serve              # OpenAI-compatible API + chat page at http://127.0.0.1:8080/chat

# any OpenAI client: base_url http://127.0.0.1:8080/v1, api key = the token in ~/.hearth/token
export OPENAI_BASE_URL=http://127.0.0.1:8080/v1 OPENAI_API_KEY="$(cat ~/.hearth/token)"
```

- **Always `uv run --no-sync`.** A bare `uv run` re-syncs the venv to the default set,
  uninstalls `mlx`, and HEARTH silently falls back to the `echo` stub.
- **Models load from disk only.** `hearth models pull <id>` is the one command that downloads;
  `hearth doctor --offline` shows where each model resolved from.
- **Help is built in.** `hearth --help` lists the commands in learning order;
  `hearth COMMAND --help` gives examples, the `HEARTH_*` variables read and exit codes; the full
  reference is the man page: `man ./man/hearth.1` (GUIDE §2.6 shows how to make `man hearth` work).

**Optional extras:** `mlx` (real inference), `remote` (Anthropic escalation),
`embeddings` (MLX RAG embeddings), `mcp` (MCP server), `vec` (sqlite-vec vector store),
`coreml` (Core ML export), `files` (PDF/XLSX parsers), `dev` (tests). Add one by appending it
to the full sync command. The core runs without them, using the offline `echo` backend and a
dependency-free vector store.

## CLI surface

`hearth doctor [--offline] · models (list/pull/rm/convert/export-coreml) · serve · run · agent · mcp · stats · rag (ingest/query) · train · eval · prereg (init/check) · adapters (list/promote/retire) · plugins · version`

## Documentation map

| Doc | What's in it |
| --- | --- |
| [docs/GUIDE.md](docs/GUIDE.md) | **The user guide.** Learn HEARTH in 15 minutes, then install, models, serve, `/chat`, API, agent, RAG, MCP, routing, finance, training, env vars, troubleshooting. |
| [man/hearth.1](man/hearth.1) | The reference manual, generated from the CLI (`man ./man/hearth.1`). |
| [docs/README.md](docs/README.md) | Index of every doc with its status (current / partly stale / historical). |
| [docs/PROPOSAL.md](docs/PROPOSAL.md) | The pitch: problem, vision, goals/non-goals, principles, success metrics, risks. |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design: layers, component interfaces, backends, data flow, deployment models, tech stack. |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Phased build plan (Phase 0–7), deliverables, acceptance criteria. |
| [docs/API.md](docs/API.md) | The gateway API contract: OpenAI-compatible endpoints + HEARTH extensions. |
| [docs/INTEGRATION.md](docs/INTEGRATION.md) | How CAMBOT, Claude Code (MCP), and generic clients consume HEARTH. |
| [docs/PLUGINS.md](docs/PLUGINS.md) | Writing third-party providers / vector stores against the plugin entry points. |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Architecture Decision Records — the *why* behind the big choices. |
| [docs/PRIVACY.md](docs/PRIVACY.md) | Confidential-work model: what stays local, the only egress vectors, sealed no-egress mode (`scripts/hearth_private.sh`), and how to verify. |
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
