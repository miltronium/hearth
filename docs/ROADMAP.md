# HEARTH — Roadmap

**Status:** Phases 0–7 all shipped and green (211 Python tests + Swift package). The planned
build is complete **and both hardware-blocked follow-ups are now validated on real Apple
Silicon** — see [RESULTS.md](RESULTS.md) and "Remaining follow-ups" at the end of this file.
Companion to [PROPOSAL.md](PROPOSAL.md) and [ARCHITECTURE.md](ARCHITECTURE.md).

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

**Result (done).** Bench on a 36 GB M-series: Qwen2.5-Coder **7B** 4-bit = 26.1 tok/s
(~4.5 GB resident), **14B** 4-bit = 12.4 tok/s (~9 GB). **7B chosen as `default_model`**;
14B kept in the registry for a higher-quality tier. `MLXProvider` validated end-to-end
against the real 7B weights.

> **Locked-down networks:** the public model host (`huggingface.co`) may be proxy-blocked.
> Weights are cached under `~/.cache/huggingface` after the first pull, so set
> `HF_HUB_OFFLINE=1` (and `TRANSFORMERS_OFFLINE=1`) to load from cache with no network.
> Pre-warm the cache once from an unrestricted terminal, then run offline. An internal
> `HF_ENDPOINT` mirror also works with no code change. (Phase 1 registry formalizes this.)

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

**Result (done).** Full OpenAI-compat gateway: `/v1/chat/completions` (streaming + non-stream),
`/v1/embeddings`, `/v1/models`. `MLXProvider` productionized with streaming, an adapter slot, and
`footprint()`. Registry backed by `config/models.yaml` (pinned + checksummed) with
`hearth models list|pull|rm`. `hearth run` one-shot CLI (`--file`, `--intent`). Local bearer-token
auth with constant-time comparison (`secrets.compare_digest`). Offline-safe via the `echo` backend;
real inference behind `HEARTH_BACKEND=mlx`.

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

**Result (done).** Task classifier (`router/classify.py`) combining rules + an `intent` hint
short-circuit. Policy engine driven by `config/routing.yaml` (`router/policy.py`) with a
`RemoteProvider` (`providers/remote.py`) for escalation (endpoint/auth via config) and confidence
gating for escalation-eligible classes. Token-budget accountant (`observability/budget.py`,
per-day remote budget; prefers local when scarce). Observability: per-request records +
`observability/metrics.py`, `hearth stats` rollups, and `/v1/hearth/admin/metrics` reporting
estimated frontier-tokens-saved. Router wired end-to-end through the gateway
(`test_router_gateway.py`).

---

## Phase 3 — Embeddings + local RAG/memory → **Grounded context (G3)**

**Goal:** agents retrieve cheap local context instead of stuffing frontier prompts.

- Embedding provider (MLX embeddings or `bge`/`nomic` via a provider).
- `VectorStore` interface + embedded impl (SQLite+`sqlite-vec` or LanceDB).
- Per-project collections; `hearth rag ingest <path>`; `/v1/hearth/rag/query`.
- Optional retrieve-then-answer with a local model.

**Acceptance:** ingest a repo, query it, get relevant chunks in <1 s; a client uses
retrieved context to answer without sending whole files to a frontier model.

**Result (done).** Embedding provider protocol with a dependency-free `HashEmbedder`
(default, offline, deterministic) and an optional `MLXEmbedder` behind a new `[embeddings]`
extra (`HEARTH_EMBEDDER=hash|mlx`). `VectorStore` protocol + embedded `SQLiteVectorStore`
(one file per collection under `~/.hearth/rag/`, brute-force cosine, numpy as an optional
speedup). RAG layer: line-aware chunker, `ingest` (walks text files, skips binaries/vendor
dirs), `query` with optional local-only `answer`. `/v1/embeddings` is now real
(OpenAI-compatible); `/v1/hearth/rag/{ingest,query}` and `hearth rag {ingest,query}` shipped.
Default path needs no extras and no network. Follow-up: a `sqlite-vec`/LanceDB backend can
drop in behind `VectorStore`.

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

**Result (done).** Dataset builder (versioned JSONL + provenance), `hearth train` orchestrator
wiring `mlx_lm.lora` behind `[mlx]` with an injectable runner, eval harness (exact-match/token-F1
+ judge hook) with a beats-incumbent promotion gate, adapter registry lifecycle
(`candidate→promoted→retired`, gate-enforced promote, A/B flag), and per-request adapter
hot-swap in `MLXProvider` (`GenRequest.adapter`, cached loads; router resolves id→path, degrades
to base on failure). CLI: `hearth train`, `hearth adapters list|promote|retire`. Offline-safe
(fakes; no real training run in tests). Real training needs `uv sync --extra mlx` + a cached
base model + `HF_HUB_OFFLINE=1`.

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

**Result (done).** Swift package client (`swift/Sources/Hearth/HearthClient.swift`, async +
streaming) talking to HEARTH over HTTP. HEARTH MCP server (`hearth mcp`, stdio —
`src/hearth/mcp/`) exposing offload tools so Claude Code can delegate subtasks to the local
model. Python convenience client (`src/hearth/client.py`). Client-agnostic conformance suite
(`tests/test_conformance.py`) green with no CAMBOT present. Live CAMBOT wiring is left to the
consumer; the surface it targets is shipped and tested.

---

## Phase 6 — Embedded Swift path (Foundation Models / Core ML) → **Offline on-device (G6)**

**Goal:** fully offline inference inside a Swift app, no daemon.

- `FoundationModelsProvider` via Swift (on-device ~3B, no downloads).
- Optional Core ML export path for small models / ANE acceleration.
- Swift SDK exposes an in-process mode mirroring the HTTP API's shape.
- Resolve Open Question #3 (adapter portability to embedded mode).

**Acceptance:** CAMBOT performs an on-device inference with networking disabled, no daemon
running.

**Result (done).** `HearthInference` protocol unifies HTTP (`HearthClient`) and on-device
transports behind one interface. `FoundationModelsProvider` does fully-offline, no-daemon
on-device inference via Apple FoundationModels (`SystemLanguageModel`/`LanguageModelSession`),
`#if canImport` + `@available` guarded with a compiling fallback stub and `isAvailable`/
`unavailableReason` probes (`HearthError.onDeviceUnavailable` otherwise); streaming diffs
cumulative snapshots into deltas. Live-verified on macOS 26. **Open Question #3 resolved:**
embedded mode is base-model-only in v1 — FoundationModels can't load MLX LoRA adapters (use
the HTTP daemon when an adapter is needed); Core ML left as a marked extension point. See
`swift/OFFLINE.md`.

---

## Phase 7 — Plugin API + multi-model serving + quant pipeline → **Long-term extensibility (G7)**

**Goal:** grow without touching the core.

- Documented plugin API for new `ModelProvider`s, routes, and vector stores (entry-points).
- Multi-model concurrent serving with memory-aware scheduling.
- Quantization/conversion pipeline (`hearth models convert`) for new checkpoints.
- Hardening: graceful degradation, model warmup, health/readiness endpoints.

**Acceptance:** a third-party backend loads as a plugin with zero core edits; two models
serve concurrently within the RAM ceiling.

**Result (done).** Entry-point plugin API (`src/hearth/plugins.py`) discovers backends in
three groups — `hearth.providers`, `hearth.vector_stores`, `hearth.embedders` — imports and
Protocol-validates them, and registers them by name so `select_provider` /
`select_vector_store` / `select_embedder` resolve `HEARTH_BACKEND=<plugin>` (etc.) with zero
core edits; a broken plugin logs a warning and is skipped, never crashing startup.
`hearth plugins list` shows discovered plugins + group + status. In-repo example
(`examples/plugin/`, own `pyproject.toml` + trivial `ModelProvider`) verified live — installs,
lists `ok`, serves via `HEARTH_BACKEND=hello` unchanged — plus `docs/PLUGINS.md`. Multi-model
serving: `ModelManager` (`src/hearth/serving/manager.py`) holds resident models within
`HEARTH_RAM_CEILING_GB` using each provider's `footprint().ram_gb`, lazy-loads on demand, and
thread-safely LRU-evicts to make room (two models serve concurrently within the ceiling).
Quant/conversion pipeline: `hearth models convert` → `mlx_lm.convert` behind `[mlx]` with an
injectable runner (fake in tests). Hardening: `GET /v1/hearth/admin/ready` (200 only once the
warm model is resident, else 503), default-model warmup on `hearth serve` (config flag, on for
mlx / no-op for echo; a failed warmup logs and continues in degraded mode), and graceful
provider degradation (bad adapter → base-weights retry; a raising provider → clean 503
`hearth.provider.unavailable`, never a 500). Offline-safe (fakes/simulated entry points; no
real models or conversions in tests). 138 → 175 Python tests, all green; echo skeleton intact.

---

## Sequencing notes

- **1 → 2 is the critical spine.** Everything downstream assumes measurement (Phase 2).
- **3, 4, 6 are somewhat independent** and can reorder based on what hurts most: reorder
  4 before 3 if domain quality is the bigger pain than context cost.
- **5 can start partially after Phase 1** (a basic Swift/HTTP client) but the MCP server and
  conformance suite belong after routing exists.
- Keep the walking skeleton green at every phase boundary.

---

## Remaining follow-ups (post-Phase 7)

The phased build is done. The two code-side follow-ups are shipped and green; the two
hardware-blocked items are **now validated on real Apple Silicon** (Apple M3 Pro / 36 GB) —
full evidence in [RESULTS.md](RESULTS.md).

- **`sqlite-vec` VectorStore backend (done).** `SqliteVecVectorStore` (`memory/store.py`) drops in
  behind the `VectorStore` protocol via `HEARTH_VECTOR_STORE=sqlite-vec` (KNN over a `vec0` virtual
  table; L2-distance→cosine score conversion documented). Default stays the dependency-free
  brute-force `SQLiteVectorStore`; `sqlite-vec` is lazy-imported behind `uv sync --extra vec`.
  Extension-gated tests skip cleanly when the native lib is absent.
- **Core ML / ANE path — generation loop done, validated end-to-end (ADR-011).** The offline
  Swift generation loop is wired and **validated on real weights** (Task C): `hearth models
  export-coreml` produces a Core ML `.mlpackage` + a sidecar contract (`hearth-coreml.json` +
  tokenizer), and Swift `CoreMLProvider.generate` runs fully offline via swift-transformers'
  tokenizer + `LanguageModel` — a real Qwen2.5-0.5B answered `"In one word, what color is the
  sky?"` → `Blue` on the ANE, greedy-matching the source PyTorch model (fp16 precision divergence
  only). The dependency is quarantined in an opt-in `HearthCoreML` product so the core `Hearth`
  stays zero-dep / macOS 13. **Approach A (non-stateful padded-prefill) is the shipped/validated
  path.** Approach B (stateful KV-cache, O(1)/token) is now a **math-validated recipe with an
  isolated runtime blocker** (2026-07-20): greedy parity + `torch.export` + coremltools `States`
  convert/save all succeed, but CoreML `predict()` on the saved fp16 stateful model SIGBUSes
  (`-14`/`ANECCompile FAILED`) on this stack (macOS 26 *Internal* + torch 2.7.1, coremltools-untested).
  Recipe: `scripts/coreml_stateful_reference.py`; full findings + next steps: [RESULTS.md](RESULTS.md)
  → Task C-2. (The winning contract is single-token + fixed-width mask + `writePos`, so it needs a
  small custom Swift decode loop — revising ADR-011's "no Swift change" assumption.)
- **Real training run (done — validated on real weights).** `scripts/train_lora_real.sh` +
  `docs/RUNBOOK_training.md` drove a real `hearth train` → eval gate → `hearth adapters promote`
  lifecycle on `Qwen2.5-Coder-7B-4bit` (Apple M3 Pro). The eval gate was exercised **both
  directions with real scores** (refused a 0.20-vs-1.0 candidate; promoted a genuine 1.0-vs-0.2
  winner that learned an arbitrary routing convention the base can't), and the promoted adapter
  **serves live** through the gateway (verified with an on-daemon adapter-vs-base A/B). See
  [RESULTS.md](RESULTS.md) → Task A.
- **Live consumer wiring (done — real G2/G8 numbers).** `examples/cambot_offload.{py,swift}`,
  `examples/claude_code_mcp.md`, and `docs/RUNBOOK_consumer_wiring.md` were run against a live
  `HEARTH_BACKEND=mlx` daemon: CAMBOT (Python) offloaded real subtasks locally, the Swift SDK
  called it live, and Claude Code offloaded a real subtask via the MCP server (stdio, escalation
  provably off). `/v1/hearth/admin/metrics` reported **`estimated_frontier_tokens_saved: 2210`**
  over 19 all-local requests (class_mix across classify/summarize/extract/draft). See
  [RESULTS.md](RESULTS.md) → Task B.

> **One shipped-code fix came out of this run:** `MLXProvider._strip_terminators` now truncates
> at the *first* chat terminator (a LoRA-tuned model can emit a literal `<|im_end|>` mid-stream
> and ramble); without it a promoted adapter served garbage. Suite stays green (211 passed, 1
> skipped). Details in [RESULTS.md](RESULTS.md) → Finding 2.
