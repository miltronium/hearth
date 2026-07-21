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

---

## ADR-011 — Core ML generation loop: stateful KV-cache export + swift-transformers tokenizer

**Context.** Phase 6 shipped the `CoreMLProvider` seam (model load + gating + protocol
conformance) but left the token-generation loop deferred: the exported `.mlpackage` is a plain
traced forward with no tokenizer contract and no KV cache, so `generate`/`generateStream` throw
`onDeviceUnavailable`. Completing it is the last real feature gap in the fully-offline,
no-daemon path (ADR-009 deployment model #3). Two forks drive the work: how decode is shaped
(re-run a padded window vs. a stateful KV cache) and how Swift tokenizes (own it vs. depend on
a library). The Swift package deliberately carries **zero SwiftPM dependencies** to date.

**Decision.**
1. **Decode = stateful KV cache (Approach B).** Rewrite `hearth models export-coreml`
   (`src/hearth/coreml.py`) to emit a *stateful* Core ML model using coremltools' `States` API
   — single-token `input_ids` + position input, per-layer KV read/write states, `logits`
   output pinned by name — so decode is O(1)/token instead of re-running a padded 512-window
   each step (O(n²)).
2. **Tokenizer = swift-transformers.** Add `huggingface/swift-transformers` (resolves to 1.3.3)
   as a dependency; use its `Tokenizers` module to read the exported `tokenizer.json` (byte-level
   BPE, merges, added/special tokens) and lean on its `Generation`/`Models` Core ML layer as the
   stateful decode reference. We spend the zero-dep principle specifically to make the single most
   error-prone, hardest-to-verify-offline piece (tokenization) boring and correct.
3. **Quarantine the dependency in an opt-in `HearthCoreML` product; keep the core zero-dep at
   macOS 13.** swift-transformers drags a heavy transitive tree (swift-nio, swift-crypto,
   swift-asn1, swift-collections, swift-atomics, swift-system, yyjson, EventSource, jinja,
   huggingface). Rather than burden every HTTP-only consumer, the Swift package now ships **two
   products**: `Hearth` (HTTP client + `HearthInference` + FoundationModels — zero external deps,
   macOS 13) and `HearthCoreML` (the offline loop; depends on `Hearth` + swift-transformers).
   `CoreMLProvider` moved to the `HearthCoreML` target. The package platform **stays macOS 13 /
   `swift-tools-version: 5.9`** — we do *not* raise it (that would need tools 6.0 and break the
   HTTP path on macOS 13/14 and older toolchains); the KV-cache code is `@available(macOS 15,
   iOS 18, *)`-gated instead. (One transitive pin: swift-collections held `<1.6.0`, a workaround
   for a pre-release Swift 6.3 that rejects its `@inline(always)`; remove once the toolchain
   stabilizes.)
4. **Model contract via sidecar.** Export writes `<stem>.hearth-coreml.json` (eos token ids —
   including the Finding-2b terminator set — bos, vocab size, max_seq_len, input/output names,
   chat template id) + copies of `tokenizer.json` / `tokenizer_config.json` next to the
   `.mlpackage` (stem-prefixed siblings; see `coreml.sidecar_paths`). `CoreMLProvider` loads
   these at init and falls back to `onDeviceUnavailable` when the sidecar is absent.

**Consequences.** Fully-offline generation finally works end-to-end, and it's fast (KV-cached)
rather than a toy — *without* imposing swift-transformers on daemon-only consumers (they import
`Hearth` and inherit nothing). Costs: the export rewrite is model-specific (layer/head plumbing)
and is the tall pole; `HearthCoreML` needs macOS 15 at runtime and a Swift 6 toolchain to build;
the SwiftPM surface gains a second product. The sidecar/manifest wiring and sampling/stop are
pure, offline-testable functions (Python suite 219 → green; Swift package builds + tests green
on real Apple Silicon); the stateful loop itself is hardware-gated and validated via
`HANDOFF.md` Task C (greedy-parity vs. the mlx daemon). ChatML/Qwen framing ships first (the
model already validated in RESULTS), manifest-driven so other templates extend cleanly.

**Revisit if.** swift-transformers stalls or its API churns painfully → vendor a minimal
tokenizer. Or Apple ships a higher-level on-device LLM generation API that subsumes the
hand-built stateful loop → adopt it and keep the `HearthInference` seam. Or the two-product
split proves unnecessary (core consumers all want Core ML anyway) → collapse back to one target.

**Update (2026-07-20) — Approach B VALIDATED end-to-end on CPU; contract revised; ANE deferred.**
The stateful export now runs fully offline in the CoreML runtime and **greedy-matches** the stock
PyTorch model token-for-token (`Qwen2.5-0.5B`: `"The capital of France is"` → `" Paris. It is the
largest city in Europe and the third"`). Reference: `scripts/coreml_stateful_reference.py`; full
writeup RESULTS.md → Task C-2. Three fixes got it from a hard `predict()` SIGBUS to parity, none of
them OS-related (an earlier "Internal macOS build" guess was wrong — the blocker had a clean size
threshold and reproduced across torch versions, the signature of a compiler/graph issue):
- **Per-layer separate state buffers** (`keyCache{i}`/`valueCache{i}`), NOT one 5-D `keyCache[N,…]`
  sliced per layer. Slicing a 5-D state along the layer dim hard-SIGBUSes CoreML's execution-plan
  builder above ~128 seq.
- **Convert `compute_units=CPU_ONLY`.** The `-14` / execution-plan failure is the **ANE compiler**
  (`ANECCompile FAILED`); CPU-only conversion removes it.
- **`compute_precision=FLOAT32`** (states stay fp16 — coremltools mandates fp16 states). fp16
  compute degenerated the decode into repetition; fp32 gives exact parity.

Contract + ADR revisions: the winning contract is **fully static, single-token** (`inputIds [1,1]`,
fixed-width `causalMask`, explicit `writePos`; one-hot **blend** write) — so ADR point 1 (multi-token
decode) and point 2's "no Swift change" both no longer hold: driving it needs a small custom decode
loop in `CoreMLGeneration.swift` (swift-transformers stays for tokenization only).

Remaining (separately-filable, not OS-build): the **ANE compiler** can't plan this stateful fp16
graph above ~128 seq (repro: `scripts/coreml_stateful_repro.py`), so Approach B runs on **CPU** for
now (fine for the 0.5B). Approach A stays the shipped default (it uses the ANE); Approach B is the
CPU O(1)/token upgrade, to be folded into `_coreml_export_runner` behind `--stateful` together with
the Swift decode loop.
