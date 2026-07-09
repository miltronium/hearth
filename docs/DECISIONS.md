# HEARTH — Architecture Decision Records

**Status:** Draft. Each ADR captures a decision, the context, and the *why*, so future work
doesn't relitigate settled ground — or, if it must, knows exactly what to reconsider.

Format: **Context → Decision → Consequences → Revisit-if.**

---

## ADR-001 — HEARTH is standalone, not part of CAMBOT

**Context.** The prompt for HEARTH came from CAMBOT hitting frontier token limits. The easy
path is to bolt a local-model helper into CAMBOT.

**Decision.** Build HEARTH as an independent project with its own repo. CAMBOT is consumer #1,
not the owner. No CAMBOT types appear in HEARTH; a conformance suite runs with no CAMBOT present.

**Consequences.** Reusable by Claude Code, scripts, and future agents. Slightly more upfront
work (an API instead of function calls). Forces clean boundaries.

**Revisit if.** HEARTH only ever has one consumer for 6+ months — then the abstraction tax
isn't paying off.

---

## ADR-002 — OpenAI-compatible API as the primary surface

**Context.** Every client either speaks OpenAI already or has an SDK for it.

**Decision.** Implement the OpenAI subset clients actually use (`/v1/chat/completions`,
`/v1/embeddings`, `/v1/models`). HEARTH-specific features live in additive, namespaced
`/v1/hearth/` routes and an optional `hearth` request block.

**Consequences.** Zero-friction adoption (swap `base_url`). We inherit some of OpenAI's shape
quirks. Extensions must stay optional so pure-OpenAI clients never break.

**Revisit if.** A large fraction of usage needs capabilities that don't map onto the OpenAI
shape at all.

---

## ADR-003 — MLX as the default backend on Apple Silicon

**Context.** Target hardware is Apple Silicon, 32 GB+. Options: MLX, llama.cpp/Ollama, Core ML.

**Decision.** MLX (`mlx-lm`) is the default inference backend; Ollama/GGUF is a first-class
alternate; Core ML / Foundation Models serve the embedded/offline path.

**Why.** MLX is fastest on Apple unified memory, is Apple's own framework, and supports LoRA
adapters natively — which the training subsystem (ADR-006) depends on.

**Consequences.** Best local performance and a clean fine-tuning story. Ties the default path
to Apple Silicon (acceptable — that's the target). MLX API churn is a risk, isolated by ADR-004.

**Revisit if.** MLX stalls or a backend clearly beats it on Apple Silicon for both inference
and LoRA.

---

## ADR-004 — Single `ModelProvider` interface behind every backend

**Context.** Backends (MLX, Ollama, Core ML, Foundation Models, remote) have very different
APIs and will churn.

**Decision.** All backends implement one `ModelProvider` interface. The router and gateway
only ever see that interface. Adding a backend = one new class + a registry entry.

**Consequences.** Backend churn stays contained; new backends are cheap. A little indirection.
The interface must be chosen carefully to not leak backend specifics (contract tests enforce this).

**Revisit if.** The interface accumulates backend-specific escape hatches — a sign it's wrong.

---

## ADR-005 — Routing policy is declarative data, not code

**Context.** Local-vs-escalate decisions need frequent tuning as models and needs change.

**Decision.** Routing lives in `routing.yaml` (task class → backend, escalation rule, budget).
The engine executes the policy; changing behavior means editing config, not code.

**Consequences.** Fast iteration, no redeploy to retune, easy to A/B. Config validation is now
essential (a bad `routing.yaml` shouldn't take the service down → validate + fall back to
safe defaults).

**Revisit if.** Policy needs arbitrary logic YAML can't express — then move to a small rules
DSL, still data-driven.

---

## ADR-006 — Training is PEFT-only (LoRA/QLoRA), local, eval-gated

**Context.** "Train on my domain" could mean anything from prompt-tuning to full fine-tunes.
Full training on-device is impractical and out of scope.

**Decision.** Support only parameter-efficient fine-tuning (LoRA/QLoRA via `mlx_lm.lora`) on
small base models, locally. Every adapter must beat the incumbent on a golden eval set before
it can be promoted. Big training runs are explicitly delegated to ACAI/AppleML.

**Consequences.** Feasible on 32 GB, fast iteration, hot-swappable adapters, safe promotion.
Quality ceiling is the base model's — accepted (HEARTH is a filter/first-drafter, not a
frontier replacement).

**Revisit if.** PEFT plateaus below usefulness for the domain tasks that matter most.

---

## ADR-007 — Escalation is explicit and measured, never a silent fallback

**Context.** The token-savings claim only holds if we can *see* when and why we spent frontier
tokens.

**Decision.** Every escalation is a first-class, logged event with a reason (low confidence /
class policy / explicit request / local failure). A budget accountant tracks remote spend and
estimated-tokens-saved. `hearth stats` surfaces both.

**Consequences.** The core value prop is measurable and tunable. Slightly more bookkeeping per
request. Prevents the failure mode where "local-first" quietly escalates everything.

**Revisit if.** Telemetry overhead ever shows up in latency (it shouldn't at this scale).

---

## ADR-008 — Embedded vector store, no external service

**Context.** RAG needs a vector store, but HEARTH must stay a single local install with no
infra to run.

**Decision.** Use an embedded, file-based store (SQLite + `sqlite-vec`, or LanceDB) behind a
`VectorStore` interface.

**Consequences.** Zero-ops, portable, backup-able as files. Won't scale to millions of vectors
— fine for per-project code/doc collections. Pluggable if that changes.

**Revisit if.** Collections routinely exceed what an embedded store handles well.

---

## ADR-009 — Three deployment models (daemon / CLI / embedded)

**Context.** Different consumers want different lifecycles: a warm shared service, a one-shot
script, or a no-daemon offline library.

**Decision.** Support all three: a `launchd` LaunchAgent daemon (default, warm models), a
`hearth run` CLI (one-shot or attach), and an embedded Swift library (in-process, offline).

**Consequences.** Meets every consumer where it is. More surface to maintain — mitigated by all
three sharing the same core (gateway/router/providers) except the embedded path, which
necessarily reimplements a thin slice in Swift.

**Revisit if.** One mode goes unused, or the embedded path diverges enough to become its own
project.

---

## ADR-010 — Ship an MCP server so frontier agents can offload to HEARTH

**Context.** The originating pain is Claude Code token limits. Claude Code speaks MCP.

**Decision.** Provide a `hearth mcp` server exposing summarize/classify/extract/rag/draft tools,
so a frontier agent can delegate routine subtasks to the local model mid-task.

**Consequences.** Directly attacks the originating problem — the orchestrator reasons, HEARTH
does the volume. Depends on the router (Phase 2) existing first. Keeps HEARTH useful to agents
we don't control.

**Revisit if.** MCP is superseded as the agent-tool protocol — then swap the adapter, keep the
core.
