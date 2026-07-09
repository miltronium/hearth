# HEARTH — Proposal

**Status:** Draft / pre-implementation
**Working title:** HEARTH (rename freely)
**Audience:** the developer(s) who will build this as a standalone project

---

## 1. Problem

Frontier-model usage (Claude Code and similar) is bounded by token limits. In practice,
those limits are hit not because the *hard* work is expensive, but because a large volume
of **routine, low-judgment work** is paid for at frontier prices:

- summarizing/condensing files before they enter an agent's context,
- ranking or filtering search results,
- drafting commit messages, changelog entries, docstrings,
- classifying intent, extracting structured fields from text,
- first-pass boilerplate and scaffolding,
- reformatting, translating between formats.

Each of these is well within the reach of a small local model running on Apple Silicon.
Paying premium tokens for them is the leak.

Meanwhile, three adjacent capabilities are unavailable today and would compound the win:

1. **No domain memory.** The model re-derives project-specific facts every session.
2. **No local grounding.** Retrieval over your own code/docs would cut context cost, but
   there's no cheap local embedding + search path.
3. **No improvement loop.** Nothing gets fine-tuned on your stack, so quality never
   ratchets up for *your* work specifically.

## 2. Vision

A **standalone, local-first intelligence layer** — HEARTH — that any agent or client can
use as a shared resource. It runs capable models on-device, decides locally-vs-escalate
per request, grounds answers in a local vector store, and improves over time through
on-device fine-tuning. It is client-agnostic: CAMBOT is the first consumer, not the owner.

The north star: **an agent's default should be "ask HEARTH first."** HEARTH answers what
it can cheaply and locally, and escalates to a frontier model only when the task genuinely
warrants it — with the escalation decision measured, tunable, and auditable.

## 3. Goals

- **G1 — Offload.** Serve cheap, high-volume tasks from local models with zero frontier cost.
- **G2 — Smart escalation.** A policy layer that classifies each request and routes local-vs-remote, with a token budget and confidence gating.
- **G3 — Grounding.** Local embeddings + vector store so agents retrieve context cheaply instead of stuffing it into frontier prompts.
- **G4 — Improvement loop.** LoRA/QLoRA fine-tuning on your own code and docs, with an eval gate that must pass before an adapter is promoted.
- **G5 — Client-agnostic reuse.** Stable OpenAI-compatible API + CLI + Swift SDK + MCP server. No client is privileged.
- **G6 — On-device / offline.** An embedded Swift path (Foundation Models / Core ML) for fully offline inference with no running daemon.
- **G7 — Extensible for the long haul.** Pluggable backends and routes so new models, quant formats, and capabilities drop in without rewrites.
- **G8 — Measurable.** Every request records which backend served it, latency, and estimated frontier-tokens-saved. You can *prove* the savings.

## 4. Non-goals

- **Not** a frontier-model replacement. Local 7B–14B models are a *filter and first-drafter*, not a substitute for Claude on hard reasoning. Escalation is a feature, not a failure.
- **Not** a training-from-scratch or large-scale-training platform. HEARTH does parameter-efficient fine-tuning (LoRA/QLoRA) on small models, locally. Big training runs belong on ACAI/AppleML.
- **Not** a hosted/multi-tenant service. HEARTH is single-user, local, on *your* machine (with an optional embedded library mode). Fleet hosting is explicitly out of scope for v1.
- **Not** CAMBOT-coupled. No CAMBOT types leak into HEARTH. If it only works for CAMBOT, it's wrong.

## 5. Design principles

1. **Local-first, escalate-by-exception.** The default answer path is on-device. Remote is a deliberate, budgeted decision.
2. **Client-agnostic.** The API is the product. CAMBOT is a consumer like any other.
3. **OpenAI-compatible surface.** Instant compatibility with existing SDKs and tools; drop-in `base_url` swap. HEARTH-specific power lives in *additive* extension endpoints.
4. **Pluggable backends.** MLX, Ollama/GGUF, Core ML, Foundation Models, and remote all sit behind one `ModelProvider` interface.
5. **Measure everything.** No routing decision, no adapter promotion, no model swap happens without metrics and an eval to back it.
6. **Quality gates, not vibes.** A fine-tuned adapter or new default model ships only after it beats the incumbent on a golden eval set.
7. **Privacy by default.** Nothing leaves the device unless an explicit escalation policy sends it, and escalation targets are configurable (incl. internal-only endpoints).
8. **Grow without rewrites.** New capability = new plugin/route, not surgery on the core.

## 6. High-level architecture

```
                    ┌─────────────────────────────────────────────┐
   clients          │                   HEARTH                     │
 ───────────        │                                              │
  CAMBOT  ─┐        │   ┌───────────┐   ┌──────────────────────┐   │
  Claude   ├──HTTP──┼──▶│  Gateway  │──▶│   Router / Policy     │   │
  Code(MCP)│  /CLI  │   │ (OpenAI-  │   │  classify → select    │   │
  scripts ─┘        │   │ compat +  │   │  budget · confidence  │   │
                    │   │ extensions)│  └───────┬──────────────┘   │
                    │   └───────────┘           │                  │
                    │                  ┌────────▼─────────┐        │
                    │                  │ ModelProvider API │        │
                    │        ┌─────────┼─────────┬─────────┼──────┐ │
                    │        ▼         ▼         ▼         ▼      ▼ │
                    │      MLX     Ollama/    Core ML   Foundation Remote
                    │    (default) GGUF                 Models    (frontier/
                    │                                             internal) │
                    │                                              │
                    │   ┌────────────┐  ┌───────────┐  ┌────────┐  │
                    │   │  Registry  │  │  Memory /  │  │ Train  │  │
                    │   │ models +   │  │  RAG store │  │ LoRA + │  │
                    │   │ adapters   │  │ (embeddings)│ │  eval  │  │
                    │   └────────────┘  └───────────┘  └────────┘  │
                    │           Observability · token accounting   │
                    └─────────────────────────────────────────────┘
```

Full component detail in [ARCHITECTURE.md](ARCHITECTURE.md).

## 7. Phased delivery (summary)

Detail and acceptance criteria in [ROADMAP.md](ROADMAP.md).

| Phase | Theme | Unlocks |
| --- | --- | --- |
| **0** | Scaffold + walking skeleton | Repo, one MLX model behind one HTTP endpoint |
| **1** | Gateway + MLX + registry + CLI | **Offload cheap work** (G1) |
| **2** | Router/policy + escalation + budget + observability | **Smart escalation, measurable savings** (G2, G8) |
| **3** | Embeddings + local RAG/memory | **Cheap grounded context** (G3) |
| **4** | Fine-tuning (LoRA/QLoRA) + adapter registry + eval harness | **Improvement loop** (G4) |
| **5** | Swift SDK + CAMBOT integration + MCP server | **Client-agnostic reuse** (G5) |
| **6** | Embedded Swift path (Foundation Models / Core ML) | **Offline on-device** (G6) |
| **7** | Plugin API + multi-model serving + quant pipeline | **Long-term extensibility** (G7) |

Each phase ships something usable. Phase 1 alone already saves tokens.

## 8. Success metrics

- **Token savings rate** — % of requests served locally, and estimated frontier tokens saved per week. (Primary metric.)
- **Escalation precision** — of tasks escalated, what fraction genuinely needed it (spot-audited)? Too many escalations = policy too timid; too few = quality complaints.
- **Local quality** — local model win-rate vs frontier on the golden eval set, per task class.
- **Latency** — p50/p95 for local requests (target: local summarize < 2 s on a 7B 4-bit model).
- **Adapter lift** — measured eval improvement from each promoted fine-tuned adapter vs base.

## 9. Risks & mitigations

| Risk | Mitigation |
| --- | --- |
| Local model quality too low → users distrust it | Escalation + confidence gating; conservative default routing; eval gates; make escalation cheap and invisible |
| Scope creep into a training platform | Hard non-goal: PEFT-only, small models, local. Big training → ACAI/AppleML |
| Backend churn (MLX/Ollama APIs move) | `ModelProvider` abstraction isolates every backend behind one interface |
| CAMBOT coupling creeps in | API-first; a conformance test suite that runs with *no* CAMBOT present |
| Memory pressure on 32 GB machines | Model registry tracks footprint; lazy load/unload; one resident model + adapters by default |
| "It works on my machine" fragility | Deterministic model catalog with pinned quant + checksums; `hearth doctor` preflight |

## 10. Open questions (resolve during Phase 0)

1. **Default coder model** — Qwen2.5-Coder-7B vs 14B at 32 GB? (Bench both in Phase 0.)
2. **Escalation target** — public frontier API, or route to an internal endpoint? (HEARTH stays agnostic; config decides.)
3. **Adapter format portability** — do fine-tuned MLX adapters need a Core ML export path for the embedded Swift mode, or is embedded mode base-model-only in v1?
4. **Process model** — always-on `launchd` LaunchAgent vs on-demand spawn? (Lean LaunchAgent for warm models.)
5. **Repo home** — new standalone git repo under `apps/HEARTH` (recommended) vs monorepo.
