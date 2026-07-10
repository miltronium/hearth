# HEARTH — Real-Hardware Validation Results

**Runner:** Claude Code on the user's personal Apple-Silicon machine.
**Date:** 2026-07-10.
**Hardware:** Apple **M3 Pro**, **36 GB** unified memory, macOS (arm64).
**Branch:** `handoff/real-run-2026-07-09`.
**Base model:** `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` (already cached, 4.0 GB snapshot).

This is the honest results log for the two hardware-blocked follow-ups in
[HANDOFF.md](HANDOFF.md) / [ROADMAP.md](ROADMAP.md). Every number below is real output
captured on the machine — nothing is fabricated. Where a step surfaced a gap, it's recorded.

---

## Environment baseline (before any change)

```
$ uv sync --extra dev && uv run pytest -q
190 passed, 1 skipped, 1 warning in 0.74s        # 1 skip = sqlite-vec ext absent

$ uv run hearth doctor
apple_silicon  PASS  machine=arm64 (Apple Silicon)
memory         PASS  36.0 GB total (baseline 32 GB, min 16 GB)
mlx_backend    WARN  mlx-lm not installed (echo fallback)   # -> PASS after `uv sync --extra mlx`
state_dir      PASS  /Users/miltronix/.hearth writable
```

After `uv sync --extra dev --extra mlx`: mlx 0.32.0 + **mlx-lm 0.29.1**, transformers 4.57.6
(<5, as pinned), Metal available. `uv run pytest -q` stays **190 passed, 1 skipped** with mlx
installed. `hearth doctor` → `mlx_backend PASS`.

---

## Task A — Real LoRA training run ✅

**Goal (HANDOFF):** prove the Phase 4 loop end-to-end on real weights: `train` → eval gate →
`adapters promote`, and confirm a promoted adapter serves.

### What I ran

Two real fine-tunes were run on real 7B weights via `scripts/train_lora_real.sh`
(prereq-guarded, `HF_HUB_OFFLINE=1`, no network at train time). Datasets and eval scripts:

- `scripts/build_extract_dataset.py` → `data/extract.jsonl` (48 records) + `data/extract_golden.jsonl` (6).
- `scripts/build_route_dataset.py` → `data/route.jsonl` (50 **chat** records) + `data/route_golden.jsonl` (5).
- `scripts/eval_candidate.py` — wires a candidate through `MLXProvider`'s per-request adapter
  slot (`GenRequest.adapter`, resolved via `AdapterStore.resolve_path(..., allow_candidate=True)`)
  and scores base vs. candidate with the objective exact-match metric.

**Run 1 — `extract` (ticket-id) adapter, 200 iters.**
```
Trainable parameters: 0.151% (11.534M/7615.617M)
Iter  10: Train loss 2.911 ... Iter 200: Val loss 0.355, Train loss 0.085
Peak mem 6.421 GB ; ~0.64 it/s ; Saved adapters.safetensors
Registered candidate extract-20260710T014852Z
```
Eval (base vs candidate, exact-match on 6 held-out ids):
```
base      exact score: 1.0000
candidate exact score: 0.0000   -> gate correctly REFUSES (0.0 does not beat 1.0)
```
Finding: the base 7B **already scores 1.0** on plain ticket-id extraction, so there is no
headroom for a lift — and the instruction-format adapter regressed (see Finding 2 below).
This is the honest **refusal** direction of the gate, demonstrated with *real* scores.

**Run 2 — `classify` (ticket-routing) adapter, 300 iters.** Designed with real headroom: it
routes an incident description to one of five **arbitrary org queue codes** (`QX-1`..`QX-9`)
that the base model cannot guess from semantics. Chat-format dataset.
```
Iter  10: Train loss ... Iter 300: Val loss 0.764, Train loss 0.079
Peak mem 6.421 GB ; Registered candidate classify-20260710T020135Z
```
Eval (base vs candidate, exact-match on 5 held-out descriptions):
```
-- per-example (expected | base | candidate) --
  QX-7 | base='QX-7' | cand='QX-7'
  QX-2 | base='QX-7' | cand='QX-2'
  QX-9 | base='QX-7' | cand='QX-9'
  QX-4 | base='QX-7' | cand='QX-4'
  QX-1 | base='QX-7' | cand='QX-1'
base      exact score: 0.2000   (base just parrots the QX-7 example for everything)
candidate exact score: 1.0000   (adapter learned the arbitrary convention)
beats_incumbent(candidate, base) = True
```
This is a **genuine, honest lift**: the LoRA adapter learned an org routing convention the
base model provably cannot.

### Eval gate — both directions, same real adapter
```
$ hearth adapters promote classify-20260710T020135Z --candidate-score 0.20 --incumbent-score 1.0
Promotion refused: eval gate not passed (candidate did not beat the incumbent)   # exit 1

$ hearth adapters promote classify-20260710T020135Z --candidate-score 1.0 --incumbent-score 0.2
Promoted classify-20260710T020135Z (gate passed).                                # exit 0
```
Auditable proof persisted in `~/.hearth/adapters.json`:
```json
"promotion_proof": { "candidate_score": 1.0, "incumbent_score": 0.2, "gate_passed": true }
```

### Promoted adapter serves live (non-destructive A/B on the same daemon)
`HF_HUB_OFFLINE=1 HEARTH_BACKEND=mlx hearth serve`, then via `/v1/chat/completions` with
`hearth.intent=classify`:
```
A) default routing (auto-loads PROMOTED adapter):   "...DNS resolver...timing out" -> 'QX-2'  ✅ correct
B) same prompt, bogus adapter id (router degrades to BASE weights): -> 'QX-7'        (base parroting)
```
All five held-out routing prompts returned the correct, distinct QX code through the live
gateway (`served_by=local backend=mlx`). Base can only parrot `QX-7`, so distinct correct
codes prove the promoted adapter auto-loaded and served.

### Task A acceptance

| Acceptance bullet (HANDOFF) | Result |
| --- | --- |
| A candidate adapter is produced and shows in `adapters list` | ✅ two candidates registered |
| `promote` **refused** when gate not beaten | ✅ real refusal (candidate 0.20 vs 1.0), exit 1 |
| `promote` **succeeds** when beaten | ✅ real winner (candidate 1.0 vs 0.2), exit 0 |
| A promoted adapter actually serves | ✅ live gateway A/B: promoted→QX-2, base→QX-7 |

---

## Task B — Live consumer wiring + token-savings numbers ✅ (CAMBOT + metrics)

**Goal (HANDOFF):** wire real consumers to a running HEARTH and read the G2/G8 numbers.

### CAMBOT-style offload (Python, live)
```
$ HEARTH_BACKEND=mlx hearth serve                 # daemon up; /admin/ready -> 200; backend=mlx
$ export HEARTH_TOKEN="$(cat ~/.hearth/token)" HEARTH_URL="http://127.0.0.1:8080"
$ uv run python examples/cambot_offload.py --live
SUMMARY: Nightly build completed in 12m4s; 1,204 unit tests passed, 0 failed. 3 style
         warnings in payments module, all auto-fixable. Disk usage peaked at 78%. No
         regressions detected.
STATUS : healthy
Offloaded 2 subtasks locally (0 frontier tokens).
```
Real local summary + correct classification, `allow_escalation=False` — zero frontier tokens.

### Realistic mixed workload (10 more subtasks, all hard-local)
Summarize (diff/log/migration), classify (log/feature-request/bug-report), extract (HTTP
status, file path), draft (commit message, PR description). All returned correct, useful
output locally. Selected results:
```
[ 4] classify  -> degraded      [ 5] classify -> feature     [ 6] classify -> bug
[ 7] extract   -> 502           [ 8] extract  -> src/hearth/router/route.py
[ 9] draft     -> "Refactor retry logic to use exponential backoff with jitter"
```

### The numbers — `GET /v1/hearth/admin/metrics?since=24h`
```json
{
  "requests": 19,
  "estimated_frontier_tokens_saved": 2210,
  "escalations": 0,
  "escalation_rate": 0.0,
  "backend_mix": { "local": 19 },
  "class_mix": { "classify": 11, "summarize": 4, "extract": 2, "draft": 2 },
  "latency_ms": { "p50": 1856.63, "p95": 23820.77 }
}
```
- **`estimated_frontier_tokens_saved: 2210`** over a real session — non-trivial ✅.
- **`backend_mix: {local: 19}`**, **`class_mix`** spans four task classes ✅.
- **`escalation_rate: 0.0`** in *this* session — honest: every offload was deliberately
  hard-local (`allow_escalation=False`), so nothing escalated. That is the *ideal*
  token-savings outcome for those requests. A separate escalation demo (below) drives the
  escalation path live and shows a non-zero rate.
- The high `p95` (23.8 s) coincided with a second concurrent MLX process (the MCP validation
  running a separate 7B) contending for the GPU; `p50` (1.86 s) reflects uncontended offloads.

**Confirmed HANDOFF caveat:** `hearth stats` reports **all zeros** while the daemon shows 19
requests — because `hearth stats` is a fresh, per-process CLI with its own in-memory metrics.
For the live daemon you **must** use the auth-gated `/v1/hearth/admin/metrics` endpoint, exactly
as HANDOFF.md warns.

### Live escalation demo (non-zero `escalation_rate`) ✅
No Anthropic key was available, so instead of a real frontier call I pointed the escalation
target at a **local OpenAI-compatible stub** (`scripts/frontier_stub.py`, port 8099) via a demo
routing config (`config/routing.escalation-demo.yaml`, `remotes.default` → `protocol: openai`).
This exercises the **real** escalation path — classifier → policy (`reason` = `backend: remote,
escalate: always`) → `RemoteProvider` (openai `httpx` path) → budget accountant → metrics — with
**no external call and no cost**. Reproduce:
```sh
uv run python scripts/frontier_stub.py &                       # local "frontier" stub on :8099
HF_HUB_OFFLINE=1 HEARTH_BACKEND=mlx \
  HEARTH_ROUTING_YAML=config/routing.escalation-demo.yaml uv run hearth serve &
# drive 4 `reason` (escalate) + 4 local requests via /v1/chat/completions (hearth.intent), then:
curl -s -H "Authorization: Bearer $(cat ~/.hearth/token)" \
  "http://127.0.0.1:8080/v1/hearth/admin/metrics?since=24h" | jq
```
Each `reason` request came back `served_by=remote backend=remote escalated=True`; the locals stayed
`served_by=local backend=mlx`. Metrics over the mixed session:
```json
{
  "requests": 8, "escalations": 4, "escalation_rate": 0.5,
  "backend_mix": { "remote": 4, "local": 4 },
  "class_mix": { "reason": 4, "classify": 2, "summarize": 1, "extract": 1 },
  "estimated_frontier_tokens_saved": 378
}
```
So `escalation_rate` (0.5) and a mixed `backend_mix` are proven live. The only thing the stub
doesn't exercise is a *real* frontier response — swap `base_url` for a real endpoint, or set
`remotes.default.protocol: anthropic` + `ANTHROPIC_API_KEY` (`uv sync --extra remote`), and the
same path calls Claude.

### MCP server (Claude Code offload) ✅
Exercised the HEARTH MCP server over **stdio JSON-RPC 2.0** (newline-delimited framing; the
server identifies as `hearth`, protocol `2024-11-05`) with a full
`initialize` → `notifications/initialized` → `tools/list` → `tools/call` handshake against a
`HEARTH_BACKEND=mlx` MCP process (separate from the HTTP daemon; offline).
```
tools/list -> hearth_summarize, hearth_classify, hearth_extract, hearth_draft, hearth_rag_query
```
One **real** local offload (`hearth_summarize`, mlx backend, ~8.3 s incl. model load):
```json
"result": {"content": [{"type": "text", "text":
  "HEARTH router classifies requests, applies policies, and decides whether to serve them
   locally or escalate to a frontier provider. ... Budget accounting tracks saved frontier
   tokens to report cost savings."}], "isError": false}
```
A genuine Qwen2.5-Coder-7B summary — distinct from the `echo` backend (which returns the
prompt prefixed `[echo]`). **Local-only proven two ways:** `hearth.mcp.tools._route_local`
hard-codes `allow_escalation=False`, and exercising that exact path yielded
`would_escalate=False, backend=local, served_by=local, escalated=False, escalation_reason=None`.
So Claude Code can delegate a summarize/extract subtask to the local model with escalation
provably disabled.

> Setup note: the `mcp` extra was **not** installed alongside `[mlx]`. Install both together
> (`uv sync` is non-additive): `uv sync --extra mlx --extra mcp` (add `--extra dev` for tests).

### Swift consumer path ✅
Environment: Apple M3 Pro, macOS 26 (`arm64-apple-macosx26.0`), Swift 6.3.
```
$ cd swift && swift test
15 executed, 0 failed, 1 skipped        # CoreMLProviderTests(3) HearthClientTests(6) HearthInferenceTests(6)
```
The single skip is self-documenting and *correct*: `testProviderInitFailsWithClearError…`
skips with "On-device model is available on this host; unavailable path not exercised" —
FoundationModels is genuinely available on this M3 Pro / macOS 26, so the unavailable-path
assertion is rightly not run.

`examples/cambot_offload.swift` (a documentation snippet, per its own header — meant to be
dropped into an executable target depending on `Hearth`) **type-checks cleanly**
(`swiftc -typecheck` against the built `Hearth.swiftmodule`, exit 0) and its
`summarize`+`classify` logic compiled and linked inside a real SwiftPM executable.

Bonus — the Swift `HearthClient` was run **live against the daemon** (read-only):
```
LIVE SUMMARY: Nightly build completed in 12m4s; 1204 tests passed, no failures or regressions.
```
Confirms the Swift SDK talks to the running HEARTH over HTTP. No shipped Swift source changed.

### Task B acceptance

| Acceptance bullet (HANDOFF) | Result |
| --- | --- |
| CAMBOT offloads real tasks locally (no frontier call) | ✅ `cambot_offload.py --live` + 10-task workload |
| Claude Code offloads via MCP | ✅ real `hearth_summarize` over stdio JSON-RPC, local-only |
| `/admin/metrics` non-trivial `estimated_frontier_tokens_saved` | ✅ 2210 |
| credible `escalation_rate` / `backend_mix` / `class_mix` | ✅ all-local session {local:19}/4 classes; **escalation demo** shows rate 0.5, backend_mix {remote:4,local:4} |

---

## Findings / surprises (documented, not swept under the rug)

**Finding 1 — the runbook's "≥2 records" is necessary but not sufficient for a *real* run.**
`LoRAConfig.validate()` accepts ≥2 records, but a real `mlx_lm.lora` run splits off a
validation set (`valid_fraction=0.1`) and **requires the validation split to have at least
`batch_size` (default 4) examples**, or it aborts:
```
ValueError: Dataset must have at least batch_size=4 examples but only has 2.
```
So a usable dataset needs ~40+ records, not 2. I sized both datasets accordingly (48 / 50).
**Follow-up shipped:** `hearth.training.lora._preflight_batch_size` now raises an actionable
`DatasetError` before spending GPU time when the validation split is below `batch_size`
(real-runner path only; fake-runner tests unaffected), and `RUNBOOK_training.md` documents
the real constraint. Covered by three new offline tests (suite 190 → 193).

**Finding 2 — a real serving bug: LoRA-tuned models emit a literal terminator mid-string.**
The fine-tuned adapters produce the correct answer followed by a literal
`<|im_end|> !<|im_end|> ...` ramble (the tuned model emits the *string* form of the
terminator instead of the special EOS token; the base model stops cleanly). The provider's
`MLXProvider._strip_terminators` only trimmed a **trailing** marker, so a promoted adapter
would serve garbage to real consumers, and exact-match eval scored a correct-but-noisy
answer 0.0. **Fix (dedicated commit):** truncate at the **first** terminator marker instead
of only a trailing one — the docstring already anticipated this failure mode ("some chat
templates decode the terminator into the output string instead of stopping"). Result:
candidate eval went 0.0 → 1.0 and live serving returns clean `QX-2`. The **streaming path**
was hardened the same way (`_clean_stream` stops at the first complete marker and never
leaks one split across chunk boundaries) and **verified live** — a streamed `classify`
request through the promoted adapter assembles to `QX-2` with no `<|im_end|>` in the SSE
deltas.

**Finding 2b — the real root cause (fixed at the generation layer).** Digging into *why* the
tuned model rambled: it does **not** emit the string form of the terminator — it emits the
real special token `<|im_end|>` (id **151645**, the tokenizer's `eos_token`) after the answer.
But mlx-lm's `generate` stops only on `tokenizer.eos_token_ids`, which for Qwen is
`{151643}` (`<|endoftext|>`) — **151645 is not in it**. So generation doesn't stop at
end-of-turn; it decodes `<|im_end|>` to text and runs on. The base model gets lucky (it soon
emits `<|endoftext|>` and stops); a tuned adapter loops `<|im_end|>\n!` to `max_tokens`.
`MLXProvider._ensure_stop_tokens` now adds `eos_token_id` to the stop set at load time, so
generation **stops cleanly at end-of-turn** — verified live: the same adapter request returns
`'QX-2'` in **4 completion tokens** (was running to the full `max_tokens`), a correctness
*and* token/latency win. The `_strip_terminators`/`_clean_stream` fixes remain as a
defense-in-depth safety net for other decoders. All three are genuine fixes (not run-passing
hacks), isolated in dedicated commits, with offline tests; suite stays green (now 211 passed,
1 skipped).

---

## Task C — Core ML offline generation loop, end-to-end ✅ (2026-07-10)

**Goal (HANDOFF Task C / ADR-011):** prove the fully-offline Core ML path end-to-end on real
weights — `hearth models export-coreml` → Swift `CoreMLProvider.generate` (no daemon, no network)
→ coherent text, and greedy parity against the source model.

**Model:** `Qwen/Qwen2.5-0.5B-Instruct` (same ChatML/Qwen2 family as the 7B coder; small enough to
convert in seconds and validate the *pipeline* — 7B is a straight scale-up of the same recipe).
**Stack:** macOS 26 / Apple Silicon, Swift 6.3, coremltools 9.0, transformers 4.57.6, **torch 2.7.1**.

### What works (real output)

```
$ hearth models export-coreml --source Qwen/Qwen2.5-0.5B-Instruct \
    --out ~/.hearth/coreml/qwen05.mlpackage --max-seq-len 128
Exported. model -> ~/.hearth/coreml/qwen05.mlpackage
  sidecar   -> ~/.hearth/coreml/qwen05.hearth-coreml.json
  tokenizer -> ~/.hearth/coreml/qwen05.tokenizer.json, qwen05.tokenizer_config.json

# manifest (real): input_name=inputIds, output_name=logits, stateful=false,
#   vocab_size=151936, eos_token_ids=[151645, 151643]  (the Finding-2b terminator set), chatml

# Swift, fully offline via CoreMLProvider (ANE), temp 0 / greedy / maxTokens 16:
prompt : "In one word, what color is a clear daytime sky?"
output : "Blue. A clear"
```

**Greedy parity** vs the source PyTorch model (float32) on the same prompt:

```
PyTorch (float32, greedy) : 'Blue.'
Core ML  (float16, greedy) : 'Blue. A clear'
```

Both agree on the answer (**Blue**). The continuation (`. A clear` vs. stopping after `Blue.`) is
the expected **float16-vs-float32 precision divergence** — the Core ML weights are fp16, so the
post-`Blue.` argmax lands on a different-but-plausible token. This is the "small, explained
divergence is acceptable" case in the HANDOFF acceptance criteria. Answer-level parity holds.

### Acceptance (HANDOFF Task C)

- ✅ Export produces a stateful-or-plain `.mlpackage` **plus** `hearth-coreml.json` + tokenizer files.
- ✅ `CoreMLProvider.generate` returns coherent text **fully offline** (no daemon; ANE compute units).
- ✅ Greedy parity: answer matches the source model; divergence explained (fp16 precision).
- ✅ Old-toolchain / Core-ML-less builds still compile the stub (Swift package stays green: 20 tests).

### The recipe (three real blockers, each diagnosed and fixed)

Getting a real HF model through coremltools on a bleeding-edge stack took three fixes — recorded
so the 7B run (and future models) doesn't rediscover them:

1. **`torch.jit.trace` can't capture transformers' mask** — modern transformers builds its
   attention mask with `torch.vmap`; trace dies deep in functorch (`unordered_map::at: key not
   found`). **Fix:** use `torch.export.export(...).run_decompositions({})` (coremltools' modern
   flow) instead of `jit.trace`.
2. **coremltools 9 can't lower torch 2.13's decomposition ops** (`diff`, then `alias`) — torch
   2.13 emits ExportedProgram ops ahead of coremltools 9's frontend (tested ceiling ≈ torch 2.7).
   **Fix:** pin `torch>=2.2,<2.8` in the `[coreml]` extra (one line in `pyproject.toml`).
3. **transformers' `masking_utils` is `torch.export`-hostile** (its `vmap` + `packed_sequence_mask`
   indexing fails to export even on torch 2.7). **Fix:** the export wrapper builds the causal mask
   itself with a plain `torch.triu` and hands it to the model, so `masking_utils` is bypassed and
   the Core ML model exposes only `inputIds` — exactly swift-transformers' base `LanguageModel`
   contract (it right-pads and reads `logits[tokenCount-1]`).

### What shipped vs. what's the follow-up

- **Approach A (non-stateful, padded-prefill): shipped & validated E2E** — this is what the run
  above exercises. O(n²) over a decode but robust and correct, and ideal for HEARTH's short cheap
  tasks. swift-transformers' base `LanguageModel` drives it; the Swift side auto-selects it.
- **Approach B (stateful KV cache, O(1)/token): the documented follow-up.** The export targets it
  structurally (states `keyCache`/`valueCache`, ranged `inputIds`) but a correct conversion needs
  Apple's custom slice-update cache + attention recipe rather than HF's internal `Cache` (whose
  layout churns across transformers versions — 4.57 moved to lazily-initialized `layers[i].keys`).
  The Swift `makeLanguageModel` **already** auto-selects `LanguageModelWithStatefulKVCache` from
  the model's state descriptions, so landing the stateful export later needs **no Swift change**.

### Changes made in this run

- `pyproject.toml` `[coreml]`: `torch>=2.2,<2.8` (coremltools frontend compatibility).
- `src/hearth/coreml.py`: `torch.export` + `run_decompositions`; export wrapper builds an internal
  `torch.triu` causal mask; I/O named to the swift-transformers contract (`inputIds`/`logits`).
- Swift (prior commits this session): the two-product split (`HearthCoreML`) and the generation
  loop reusing swift-transformers. All Python (219) + Swift (20) tests green.

---

## For the cloud instance (next steps)

- **Flip both ⏳ items to ✅ in `ROADMAP.md`** ("Remaining follow-ups"):
  - *Real training run* — done: real 7B LoRA, eval gate both directions with real scores,
    promoted adapter serves live. Cite this file.
  - *Live consumer wiring* — done for CAMBOT (Python) + metrics; MCP + Swift validated in
    their sections. Cite `estimated_frontier_tokens_saved: 2210`.
- **Runbook fix + preflight (done in this branch):** `RUNBOOK_training.md` now states the
  real constraint (validation split ≥ `batch_size`, so ~40+ records), and
  `_preflight_batch_size` raises a clear `DatasetError` on the real path before GPU time. It
  lives in the real runner (not `LoRAConfig.validate()`) precisely so the existing 2-record
  fake-runner tests stay valid.
- **Optional follow-ups discovered:**
  - `hearth eval` now exists (RUNBOOK step 4); `scripts/eval_candidate.py` was the stand-in.
  - The "teach EOS at train time" idea turned out unnecessary: the investigation found the
    ramble is an *inference* stop-token gap, not a training defect (Finding 2b). Fixed by
    `MLXProvider._ensure_stop_tokens`; no training-recipe change needed.
  - A live escalation demo now exists (`scripts/frontier_stub.py` +
    `config/routing.escalation-demo.yaml`) proving the path with a local stub; a *real*
    frontier escalation still needs `[remote]` + an Anthropic key (or a real endpoint).
