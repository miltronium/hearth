# HEARTH — Handoff to a Real-Hardware Claude Code

**Who this is for.** A Claude Code instance running on the user's **personal Apple-Silicon
machine** — one that *has* the resources a sandboxed/cloud instance does not: a real GPU,
the ability to download model weights from Hugging Face, a live CAMBOT app, and a live
Claude Code MCP session. You are the partner who can *execute and verify* the two
follow-ups that the cloud instance built but could not run.

**Who built the rest.** A cloud Claude Code instance shipped Phases 0–7 and both code-side
follow-ups (sqlite-vec store, Core ML path). Everything it could verify without hardware is
**already green**: 190 Python tests + a Swift package, ruff-clean. It could *not* run a real
training job or drive a live consumer — that's your half. See
[ROADMAP.md](ROADMAP.md) → "Remaining follow-ups" for the authoritative status.

---

## The division of labor

| Item | State | Owner |
| --- | --- | --- |
| Phases 0–7 | ✅ shipped + tested | cloud instance |
| sqlite-vec VectorStore backend | ✅ shipped (ext-gated tests skip w/o native lib) | cloud instance |
| Core ML export + Swift `CoreMLProvider` | ✅ shipped (generation loop deferred) | cloud instance |
| **Real LoRA training run** | ⏳ harness + runbook ready — **needs you to execute** | **you** |
| **Live CAMBOT / Claude Code wiring + real token-savings numbers** | ✅ done (Task B, see RESULTS.md) | **you** |
| **Core ML offline generation loop** (ADR-011) | ✅ done (Task C) — validated E2E on real weights (Approach A); stateful KV-cache is the follow-up | cloud + **you** |

Nothing below requires you to write new features. Your job is to **run the shipped
harnesses on real hardware, capture honest results, and hand them back**. If you find a
gap in the harness while running it, fix the harness — but keep the surface identical to
what the runbooks describe.

---

## Prerequisites on your machine

```bash
# 1. Clone / pull the repo, then install with real inference:
uv sync --extra mlx            # mlx + mlx-lm + transformers<5
# optional, depending on the task:
uv sync --extra remote         # Anthropic escalation (if testing escalation)
uv sync --extra embeddings     # MLX RAG embeddings
uv sync --extra vec            # sqlite-vec backend (validates that follow-up on real lib)

# 2. Confirm the baseline is green before you change anything:
uv run pytest -q               # expect 190 passed, 1 skipped (or 191 passed if [vec] installed)
uv run hearth doctor           # environment preflight — must be clean

# 3. Pre-cache the base model (network needed HERE and only here):
uv run hearth models pull mlx-community/Qwen2.5-Coder-7B-Instruct-4bit
```

The harnesses **refuse to download** at run time (they set `HF_HUB_OFFLINE=1`), so the
`models pull` above is the one online step. If the model isn't cached, the training script
fails fast with a clear message rather than hitting the network.

---

## Task A — Real LoRA training run

**Goal:** prove the Phase 4 loop end-to-end on real weights: `train` → eval gate →
`adapters promote`, exactly as the fake-runner tests exercise it.

**Runbook:** [RUNBOOK_training.md](RUNBOOK_training.md) — read it fully first.
**Harness:** `scripts/train_lora_real.sh` (prereq-guarded, download-refusing).

```bash
# minimal path (the script wraps these and checks prereqs):
scripts/train_lora_real.sh --help
scripts/train_lora_real.sh \
  --base mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \
  --task extract \
  --data <your-training.jsonl>          # ≥2 records; see hearth.training.dataset

# what it drives, if you'd rather run by hand:
HF_HUB_OFFLINE=1 uv run hearth train --task extract \
  --base mlx-community/Qwen2.5-Coder-7B-Instruct-4bit --data <data.jsonl>
uv run hearth adapters list             # confirm a *candidate* was registered
uv run hearth adapters promote <id> --candidate-score <s> --incumbent-score <s>
```

**Acceptance (record each):**
- A candidate adapter is produced and shows in `hearth adapters list`.
- `hearth adapters promote` is **refused** when the eval gate isn't beaten, and **succeeds**
  when it is (the gate is the whole point — verify both directions).
- A promoted adapter actually serves (route a request through the gateway and confirm it
  loads the adapter).

## Task B — Live consumer wiring + token-savings numbers

**Goal:** wire real consumers to a running HEARTH and read the G2/G8 numbers that the
cloud instance could only stub on the `echo` backend.

**Runbook:** [RUNBOOK_consumer_wiring.md](RUNBOOK_consumer_wiring.md).
**Examples:** `examples/cambot_offload.py`, `examples/cambot_offload.swift`,
`examples/claude_code_mcp.md`.

```bash
# start the daemon with the real backend:
HEARTH_BACKEND=mlx uv run hearth serve

# CAMBOT-style offload (Python) against the live daemon:
uv run python examples/cambot_offload.py --live

# register HEARTH's MCP server with your Claude Code (see examples/claude_code_mcp.md),
# then drive a real workload through it.

# read the numbers — NOTE hearth stats is per-process, so for the live daemon use the
# auth-gated admin endpoint:
TOKEN=$(cat ~/.hearth/token)
curl -s -H "Authorization: Bearer $TOKEN" \
  http://127.0.0.1:8080/v1/hearth/admin/metrics | jq
```

**Acceptance (record each):**
- CAMBOT and Claude Code both offload real tasks to HEARTH (local, no frontier call).
- `/v1/hearth/admin/metrics` reports a non-trivial `estimated_frontier_tokens_saved` and a
  credible `escalation_rate` / `backend_mix` / `class_mix` over a real session.

## Task C — Core ML stateful generation loop (real export + parity)

**Goal:** validate the ADR-011 offline Core ML path end-to-end on real weights — export a
stateful `.mlpackage`, run `CoreMLProvider.generate` fully offline (no daemon, no network), and
confirm **greedy parity** against the mlx daemon for a fixed prompt.

**Decision record:** [DECISIONS.md](DECISIONS.md) → ADR-011.
**Export:** `hearth models export-coreml` (`src/hearth/coreml.py`, stateful KV-cache export).
**Provider:** `swift/Sources/HearthCoreML/CoreMLProvider.swift` + the generation loop
(swift-transformers tokenizer, opt-in `HearthCoreML` product, `@available(macOS 15, iOS 18, *)`).

```sh
# 1. Export a stateful Core ML model + sidecar (needs the [coreml] extra + cached source):
uv sync --extra coreml
HF_HUB_OFFLINE=1 uv run hearth models export-coreml \
  --source mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \
  --out ~/.hearth/coreml/qwen-coder.mlpackage \
  --compute-units cpuAndNeuralEngine --precision float16

# confirm the sidecar contract landed next to the .mlpackage:
#   ~/.hearth/coreml/qwen-coder.mlpackage  +  hearth-coreml.json  +  tokenizer.json

# 2. Drive CoreMLProvider offline from Swift (a small runner in swift/, or a test target):
#    engine = try CoreMLProvider(modelURL: url); print(try await engine.generate(...))

# 3. Greedy parity: same prompt, temperature 0, against the mlx daemon:
HEARTH_BACKEND=mlx uv run hearth run "classify: <fixed prompt>" --temperature 0
```

**Acceptance (record each):**
- The export produces a stateful `.mlpackage` **plus** `hearth-coreml.json` + `tokenizer.json`.
- `CoreMLProvider.generate` returns coherent text **fully offline** (no daemon, network off).
- `generateStream` yields incremental deltas and stops cleanly on the Finding-2b terminator set
  (no `<|im_end|>` leakage — verify the same way Task B did for the daemon).
- **Greedy parity:** Core ML (temp 0) and the mlx daemon (temp 0) agree on a fixed short prompt
  (or the divergence is explained — quantization/precision differences are acceptable if small).
- Old-toolchain / Core-ML-less builds still compile and report the unavailable contract (the
  stub path stays green).

> If the stateful export needs model-specific plumbing the harness doesn't cover yet, fix the
> harness and call it out in a dedicated commit (same rule as Tasks A/B) — don't hack a run to
> pass.

---

## How to hand results back (the partnering protocol)

The cloud instance picks up **from git** — that's our shared channel. When you finish (or
get blocked), do this so it can continue seamlessly:

1. **Write results, don't just report them.** Create `docs/RESULTS.md` capturing, per task:
   the exact commands you ran, the real numbers/outputs, pass/fail against each acceptance
   bullet, and anything that surprised you. Paste real output — **never fabricate a number**.
   If a step failed, record the failure verbatim; a documented failure is a valid result.
2. **Branch + commit.** Work on `handoff/real-run-<yyyy-mm-dd>`. Use conventional commits,
   e.g. `test(training): real LoRA run results on M-series` /
   `docs(results): live consumer token-savings numbers`. Keep harness fixes in their own
   commit separate from the results log.
3. **Push and open a PR** against `main` titled `Handoff: real-hardware validation`. In the
   PR body, check off which acceptance bullets passed and link `docs/RESULTS.md`.
4. **Leave a next-step note.** End `docs/RESULTS.md` with a short "For the cloud instance:"
   section — what to mark done in `ROADMAP.md`, and any follow-up code work you discovered
   (e.g. wiring the Core ML generation loop once you've exported a model).

The cloud instance will then: fold your real numbers into `ROADMAP.md` (flipping the two
⏳ items to ✅ with citations), update the README status, and open the next work items you
surfaced. That's the loop — you run reality, it keeps the repo's story straight.

## Ground rules (same for both of us)

- **Don't overclaim.** "Verified on real weights" only after you actually ran it. If you
  couldn't, say so and why.
- **Keep the baseline green.** `uv run pytest -q` and `cd swift && swift test` must stay
  passing on any change you make. Run them before you push.
- **Don't edit the cloud instance's shipped code** to make a run pass without saying so —
  if the harness needs a fix, fix it and call it out in a dedicated commit.
- **Loopback + local only.** No exposing the daemon off-box, no sending repo contents to
  external services beyond what the runbooks already do (HF model pull, your own Claude).
