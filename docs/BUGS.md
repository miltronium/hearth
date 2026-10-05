# HEARTH — known bugs and gaps (backlog)

This is the backlog of known defects and gaps that are **not fixed yet**. It is not a roadmap
and not a status report. Each item was checked against the code at `d803f1a`
(`cmux/integration`, 2026-10-02) by reading the cited lines and, where a probe is named, by
running it. A claim nobody could check is labelled as such.

**Numbering.** `B-NNN`, assigned once, never reused. A fixed item moves to
[Fixed recently](#fixed-recently--do-not-re-open) and keeps its ID. New items take the next
free number, whatever their priority.

**Priority.** P0 offline-safety / privacy · P1 correctness · P2 quality / UX · P3 hygiene.
**Status.** `open` · `in-progress-on-branch` (someone is working on it; do not start a
second fix) · `blocked` (needs something outside the repo).

**Re-verify, don't remember.** Line numbers rot and so does this file. Before you work an
item, re-check its evidence. For the measurable parts (weights on disk, egress posture,
golden-set sizes, adapter proofs, doc staleness) run
`uv run --no-sync python scripts/hearth_status.py`. It measures; this file only remembers.
Every acceptance test below checks an **outcome**, not a configuration (CLAUDE.md §3): it
must fail when the fix is reverted.

---

## Contents

**P0: offline-safety / privacy**
- ~~[B-001](#b-001) `hearth train` / `models convert` / `models export-coreml` can download~~ — fixed in `13105ef`
- [B-002](#b-002) cmux C7: a sealed workspace does not contain its panes' children
- [B-003](#b-003) Escalation fails, then local fails: no record that the prompt may have left

**P1: correctness**
- [B-004](#b-004) MLXProvider ignores `GenRequest.model`; the model ladder silently serves the default (in progress)
- [B-005](#b-005) `/ready` returns 200 with no weights loaded (in progress)
- [B-006](#b-006) `HEARTH_BACKEND=auto` silently becomes the echo stub, and the stub reports ready
- [B-007](#b-007) An exception after the stream relay ends the SSE stream with no `[DONE]`
- ~~[B-008](#b-008) A relative `HEARTH_ROUTING_YAML` resolves from the current working directory~~ — fixed in `2550766`
- [B-009](#b-009) Router confidence is a prompt-length stub; short messages escalate under `routing.remote.yaml`
- [B-010](#b-010) A promoted adapter with no significance proof is served by default; golden sets are below `min_n`

**P2: quality / UX**
- [B-011](#b-011) The MLX embedder cannot work: default id 404s and mlx-lm has no BERT
- [B-012](#b-012) Gateway agent route: no finance tools, and `rag_search` is not pinned to a collection
- [B-013](#b-013) Agent: no search tool, an unmeasured step cap, no tool-calling eval set
- [B-014](#b-014) Untested end to end: a cmux open-tier launch; the `/chat` agent toggle in a browser

**P3: hygiene**
- [B-015](#b-015) `scripts/train_lora_real.sh --promote` always fails (removed flags)
- [B-016](#b-016) `docs/RESULTS.md` and `docs/HANDOFF.md` are stale; HANDOFF isn't monitored
- [B-017](#b-017) `docs/LEARNING_plan.md` cites stale line numbers and a wrong class count
- [B-018](#b-018) ruff: 82 findings outside `src/`
- [B-019](#b-019) Unregistered weights on disk: `Qwen/Qwen2.5-0.5B-Instruct`
- [B-020](#b-020) Starlette deprecation: `httpx` with `starlette.testclient`
- [B-021](#b-021) `audit_resolution` swaps `socket.socket.connect` process-wide
- [B-022](#b-022) `REASON_LOCAL_FAILURE` is declared and never used
- [B-023](#b-023) Test-infra trap: empty `NO_PROXY` hides client disconnects from loopback tests

**Added 2026-10-03 (found while closing B-001)**
- ~~[B-024](#b-024) `train_lora_real.sh` refuses models fetched with `hearth models pull` (P2)~~ — fixed in `e00e5e1`
- ~~[B-025](#b-025) Status probe expands `~` in `HEARTH_ROUTING_YAML`; the router does not (P2)~~ — fixed in `2550766`
- ~~[B-026](#b-026) A failed `hearth train` leaves an empty run directory (P3)~~ — fixed in `3fae564`
- ~~[B-027](#b-027) A failed training subprocess ends `hearth train` in a traceback (P2)~~ — fixed in `0f1bbe2`
- [B-028](#b-028) `export-coreml` reports missing coremltools before a missing model (P3)
- ~~[B-029](#b-029) An unregistered `HEARTH_DEFAULT_MODEL` is silently ignored; `doctor --offline` does not warn (P2)~~ — fixed in `d931473`
- [B-030](#b-030) Test-infra: `sandbox-exec` is unavailable to agents and inside harness worktrees (P3)

**Added 2026-10-05 (found while fixing B-008, B-024..B-029)**
- [B-031](#b-031) `hearth doctor --offline` renders a non-fatal WARN row as FAIL (P3)
- [B-032](#b-032) `train_lora_real.sh` runs `uv run --no-sync` from the caller's cwd (P3)

[Fixed recently, do not re-open](#fixed-recently--do-not-re-open)

---

## P0: offline-safety / privacy

### B-001
**`hearth train`, `hearth models convert`, `hearth models export-coreml` can download weights**

- **Priority:** P0 · **Status:** **FIXED** — merged in `13105ef` (commits `42ebf1f`,
  `2868160`, `1d78c4d`; see "Fixed recently"). Kept here for the record. · **Effort:** M
- **Evidence:** All three hand a repo id straight to a loader that goes online on a cache
  miss, bypassing `providers/mlx.py:resolve_local_model`:
  - `src/hearth/training/lora.py:145` puts `config.base_model` into `--model`, and
    `lora.py:200` runs `python -m mlx_lm.lora` with it.
  - `src/hearth/convert.py:108` calls `mlx_lm.convert(config.source, ...)`.
  - `src/hearth/coreml.py:417-420` and `coreml.py:534-542` call
    `AutoConfig/AutoTokenizer/AutoModelForCausalLM.from_pretrained(config.source)`.
  - `scripts/hearth_status.py` says so: `download_egress: downloads allowed
    (HF_HUB_OFFLINE=unset, TRANSFORMERS_OFFLINE=unset) — governs ... hearth train,
    hearth models convert / export-coreml`. Commit `41a42c8` lists this under "Not covered".
- **Impact:** If the operator mistypes a model id, or the cache lacks it, these commands
  silently reach Hugging Face. HEARTH's default is disk-only and these three are the
  exceptions. Today the only defence is remembering `HF_HUB_OFFLINE=1` (docs/PRIVACY.md).
- **Fix outline:** Resolve every source through `resolve_local_model` (honouring
  `HEARTH_ALLOW_DOWNLOADS`), then pass the resolved local path to mlx-lm / transformers.
  Pass `local_files_only=True` to `from_pretrained`. Run the `mlx_lm.lora` subprocess with
  `HF_HUB_OFFLINE=1` in its env unless downloads are allowed.
- **Acceptance test:** For each command, with sockets refused and counted (the
  `test_offline_model_resolution.py` pattern), use an id that is not on disk. The command
  must fail with `ModelNotOnDiskError` and **zero** connect attempts. Then plant a hub-layout
  cache and check the command loads from it, again with zero connects. Revert the
  resolution and the zero-connect assertion must fail. `download_egress` in the status
  script should then report disk-only for these paths.

### B-002
**cmux C7: sealing a workspace does not contain its panes' child processes**

- **Priority:** P0 · **Status:** open (blocks cmux C6 graduation) · **Effort:** L
- **Evidence:** `docs/cmux/TODO.md:9` ("OPEN — C7 workspace containment (the sealed tier does
  not contain panes)"), `docs/cmux/FINDING_pane_egress.md` §8, `docs/cmux/HANDOFF.md:132`,
  ADR-C009, `docs/cmux/RESULTS.md` §1.8. The pf/LuLu seal covers the cmux binary, not the
  processes a pane spawns, and a sealed pane reached the public internet with every gate
  green. This is row 1 of the CLAUDE.md §3 table. *Not re-measured here.* That needs
  cmux.app and a live probe; the evidence is the recorded negative result.
- **Impact:** Anything run inside a "sealed" pane can send data off the machine. Until this
  closes, nothing confidential should go through cmux (`docs/cmux/HANDOFF.md:29`).
- **Fix outline:** Scoped in `FINDING_pane_egress.md` §8. Contain at the workspace
  (process-tree / per-user / sandbox profile), not at the app binary. Decide the intended
  data class first (`FINDING_pane_egress.md:269`).
- **Acceptance test:** The in-pane probe from RESULTS §1.8, run from a child process of a
  sealed pane, fails closed for all three probes (`docs/cmux/RESULTS.md:155`). Rerun it
  against the current seal and it must still reach the internet, which proves the probe can
  tell the two apart.

### B-003
**When an escalation fails and the local fallback then fails, no RequestRecord is written**

- **Priority:** P0 (unrecorded possible egress) · **Status:** open · **Effort:** S
- **Evidence:** Non-streaming, `src/hearth/router/route.py:225-236`: the degrade path calls
  `self._generate(self.local, ...)` at `:236` outside any handler, so a `ProviderError` there
  propagates past `self.metrics.record` at `:262`. Streaming,
  `src/hearth/gateway/app.py:592-594`: the outer `except` yields `_stream_failure` and
  returns without recording. `/tmp` probe driving the real `Router` and `_stream_sse` with
  `DeadRemote` + `DeadLocal` from `tests/test_escalation_fallback.py`:
  ```
  route raised ProviderError: provider 'mlx' failed: weights missing
  non-stream requests recorded after double failure: 0
  stream tail: ['data: {"error": {"message": "provider \'mlx\' failed: weights missing", ', 'data: [DONE]']
  stream requests recorded after double failure: 0
  ```
  Same probe, local-only failure (no escalation): also `0` recorded. The broader rule is
  that a failed request is never recorded. Commit `2066db8` states the double-failure gap
  explicitly.
- **Impact:** The remote was called and may already have received the prompt
  (docs/PRIVACY.md:68). Yet `hearth stats` shows no request, no escalation and no
  `escalations_failed`. An outage combined with a broken local model reads as zero
  traffic, so the egress audit trail misses exactly the case where something went wrong.
- **Fix outline:** In `Router.route`, wrap the degraded `_generate` and, on failure, record
  a `RequestRecord` with `escalated=True`, `escalation_failed=<remote error>`,
  `served_by="none"` (or an explicit failure field), zero completion tokens, then re-raise.
  Mirror it in `_stream_sse`'s outer `except` when `escalation_failed` is set. Consider
  recording plain local failures too, so the error rate is visible.
- **Acceptance test:** Drive the real Router and gateway with `DeadRemote` + `DeadLocal`
  (both paths). The client still gets the error (503 / error event + `[DONE]`), and
  `metrics.rollup()` shows `requests == 1` and `escalations_failed == 1`. With the fix
  reverted, the rollup assertion must fail (it reads `0` today).

---

## P1: correctness

### B-004
**MLXProvider ignores `GenRequest.model`; the gateway serves the default model for every id**

- **Priority:** P1 · **Status:** in-progress-on-branch (another agent's worktree) ·
  **Effort:** M
- **Evidence:**
  - The router does pick a per-request model. `route.py:349-350` honours a client pin,
    `:351-352` a class `local_model` rung, `:353-357` the defaults. That choice goes into
    `GenRequest(model=decision.model)` at `route.py:306-312`.
  - The provider never reads it. `grep -n "req\.model" src/hearth/providers/*.py` hits only
    `echo.py:40`. `mlx.py:252` always loads `resolve_local_model(self.model_id)`.
  - The gateway's `ModelManager` factory returns the one app provider for any id:
    `gateway/app.py:102-104`, `factory=lambda _model_id: provider`.
  - Labels then disagree. Non-streaming reports the truth (`mlx.py:305`
    `model=self.model_id`, recorded via `route.py:251`). Streaming labels the chunks, the
    final `hearth` telemetry and the `RequestRecord` with `decision.model`
    (`app.py:510, 610, 627, 632`), which is a model that did **not** serve.
  - The adapter base-model guard compares against `decision.model`, not the served model
    (`route.py:398`). It can refuse an adapter that matches the weights actually loaded, or
    admit one that doesn't.
  - The agent route forwards `AgentRunRequest.model` the same way (`agent_route.py:118`,
    `schemas.py:246`).
- **Impact:** The `/chat` model dropdown, a client `model=` pin, and every per-class
  `local_model` rung all silently run the default model. On the streaming path the
  telemetry names the model that was asked for, not the one that ran. That is the
  CLAUDE.md §3 shape: the label is checked, the outcome differs.
- **Fix outline:** Make the gateway factory build (or fetch) a provider per model id through
  `ModelManager.get(decision.model)`, and route local generation through the manager so
  residency and eviction apply. Alternatively, have MLXProvider load `req.model` when it
  differs. Either way, report the model that actually generated (`GenResult.model`) on
  every path, streaming included.
- **Acceptance test:** With a fake `mlx_lm` that records which path `load()` received, send
  two requests pinning two different registered ids, streaming and non-streaming. Each
  load must receive the matching resolved path, and the response and `RequestRecord` must
  name it. Reverting to the single-provider lambda must fail the test.

### B-005
**`/v1/hearth/admin/ready` returns 200 without the weights being loaded**

- **Priority:** P1 · **Status:** in-progress-on-branch (another agent's worktree) ·
  **Effort:** S
- **Evidence:** `serving/manager.py:102-104` calls `provider.load(model_id)` only
  `if callable(getattr(provider, "load", None))`, then admits the model as resident
  regardless (`:105`). MLXProvider has no `load` method (`grep -n "def load"
  src/hearth/providers/*.py`: no hits). Its weights load lazily on first generate
  (`mlx.py:281-284`). `_warmup` (`app.py:318-330`) therefore "succeeds" without touching
  weights, and `/ready` (`app.py:142`) reads `manager.is_resident`. Probe
  `/tmp/bugs_probe/probe_ready.py`: `create_app` with
  `MLXProvider("nonexistent-org/no-such-weights")`, `backend=mlx`, `warmup=True`:
  ```
  ready: 200 {'status': 'ready', 'backend': 'mlx', 'model': 'mlx-community/Qwen2.5-Coder-7B-Instruct-4bit', 'resident': ['mlx-community/Qwen2.5-Coder-7B-Instruct-4bit']}
  provider weights loaded: False has load(): False
  ```
  The probe reports ready, for a model id the provider isn't even configured with (see
  B-004).
- **Impact:** Readiness gating (launchd, scripts, `cmux` gates) sends traffic to a server
  that can't serve. The first request pays the full load, or fails with
  `ModelNotOnDiskError`. Warmup's "degraded mode" log path never triggers for missing
  weights.
- **Fix outline:** Give MLXProvider a `load(model_id)` that runs `_load_variant(None)` on
  the MLX thread. Have `ModelManager._load` treat a provider without `load` as not-yet-loaded,
  or require the method. `/ready` should reflect a successful real load.
- **Acceptance test:** The probe above returns **503** and warmup logs the
  `ModelNotOnDiskError`. With a fake `mlx_lm` whose `load` succeeds, `/ready` returns 200
  and the fake recorded exactly one load before the first request. Remove the new `load`
  method and the 503 assertion must fail.

### B-006
**`HEARTH_BACKEND=auto` (the default) silently falls back to the echo stub, which reports ready**

- **Priority:** P1 · **Status:** open · **Effort:** S
- **Evidence:** `config.py:28` `backend: str = "auto"`. `providers/__init__.py:30-31`:
  `return MLXProvider(default_model) if mlx_available() else EchoProvider()`, with no log
  and no warning. `mlx_available()` is just `find_spec("mlx_lm")` (`mlx.py:110-114`). Then
  `/ready` treats echo as always ready (`app.py:142`
  `provider.name == "echo" or ...`), and echo stamps the requested model id on its reply
  (`echo.py:40` `model=req.model`).
- **Impact:** A bare `uv run` or a partial `uv sync` uninstalls `mlx_lm` (CLAUDE.md §1).
  After that, `hearth serve` comes up green, `/ready` is 200, and every answer is an echo
  labelled with a real model name. Only `/health`'s `backend` field reveals it.
- **Fix outline:** Under `auto`, log a WARNING naming the fallback and the canonical sync
  command, and make `/ready` return 503 (or `status: "stub"`) when the echo backend came
  from an `auto` fallback rather than an explicit `HEARTH_BACKEND=echo`. Optionally make
  `hearth serve` refuse `auto→echo` unless `--allow-stub`.
- **Acceptance test:** With `find_spec("mlx_lm")` patched to `None` and `backend=auto`,
  `/ready` is not 200 and a WARNING is captured. With `backend=echo` set explicitly, `/ready`
  stays 200, so the test can tell the two cases apart. Revert and the first assertion fails.

### B-007
**An exception after the stream relay ends the SSE stream with no `[DONE]`**

- **Priority:** P1 · **Status:** open · **Effort:** S
- **Evidence:** In `_stream_sse` the `try/except` that guarantees a terminal event closes at
  `app.py:594`. The accounting after it, `router.budget.spend` (`:601`),
  `router.metrics.record` (`:606`) and the final chunk build, is unguarded, and so is
  `router._resolve_adapter` (`:495`) before the first chunk. Probe: the real Router + echo
  local with a `MetricsStore` whose `record` raises `OSError("disk full")`:
  ```
  generator raised: OSError disk full
  DONE emitted: False chunks before death: 4
  ```
- **Impact:** The `/chat` page has already shown the full answer, then gets a dropped stream
  with no `[DONE]` and no error event, the failure mode `6cdcee9` set out to remove. The
  request also goes unrecorded.
- **Fix outline:** Put the post-relay accounting in its own `try`. On failure, log, still
  emit the final chunk if possible, then an error event (`hearth.metrics.unavailable`), then
  `[DONE]`. Wrap `_resolve_adapter` at `:495` the same way (it already swallows internally,
  but an injected store could raise).
- **Acceptance test:** The probe above, written as a test, asserts the last event is
  `[DONE]` and an error event precedes it. With the guard removed it must fail (today it
  raises).

### B-008
**A relative `HEARTH_ROUTING_YAML` resolves from the current working directory**

- **Priority:** P1 · **Status:** **FIXED in `2550766`** (with B-025). One resolver,
  `router/policy.py:resolve_routing_selection`, used by the router, the status probe and
  doctor: `~` expanded, relative paths against the **repo root**, and a missing
  `HEARTH_ROUTING_YAML` file raises `RoutingProfileNotFoundError` (doctor: fatal FAIL).
  Not done: logging the resolved path at startup / exposing it on `/health` (needs
  `gateway/app.py`, owned elsewhere). · **Effort:** S
- **Evidence:** `router/policy.py:94-96` returns `Path(override)` unchanged. `load_policy`
  (`policy.py:125-131`) falls back to `_safe_defaults()` (`:84-89`, zero remotes) on a
  missing file, with only a log warning. The status probe makes the same cwd-relative read
  (`status/probes.py:384`). `scripts/hearth_private.sh:36` absolutises its own value, and
  `scripts/cmux/cmux-open:42` passes an absolute path, so the scripts are safe. A hand-set
  `HEARTH_ROUTING_YAML=config/routing.remote.yaml hearth serve` run outside the repo root
  is not.
- **Impact:** Started from another directory, the operator's chosen profile is silently
  replaced. Either escalation quietly turns off (fails safe for privacy, but the operator
  believes they opted in), or a *different* `config/routing*.yaml` under the cwd is loaded,
  which could be one that has remotes. The status script, run from the repo root, can
  report a profile the daemon never loaded.
- **Fix outline:** Resolve a relative override against the repo root (as the default path
  already does), or reject relative paths at startup. Make a missing explicitly-selected
  profile an error, not a fallback. Log the resolved absolute path once at startup and
  expose it on `/health`.
- **Acceptance test:** `chdir` to a temp dir holding a decoy `config/routing.remote.yaml`
  with a sentinel remote, set the override to the relative path, and `load_policy()` must
  return the repo profile (or raise), never the decoy. A missing explicit profile must
  raise. Revert and the decoy test must fail.

### B-009
**Router confidence is a prompt-length stub; under `routing.remote.yaml` short messages escalate**

- **Priority:** P1 · **Status:** open · **Effort:** M (a real signal), S (a doc warning)
- **Evidence:** `router/route.py:425-437`: `min(1.0, 0.4 + length / 300.0)` over the last user
  message. The `task_class` argument is unused. `config/routing.remote.yaml:47-50` sets
  thresholds draft 0.6, code 0.7, chat 0.65, so a user message shorter than 60 / 90 / 75
  characters escalates. The yaml's own comment says so (`routing.remote.yaml:5-7`). This
  is inert under the default no-egress `config/routing.yaml` (status: `NO EGRESS: 0
  remotes`).
- **Impact:** Under the opt-in remote profile, "fix this" or "what does this do?" go to
  the frontier and long pasted documents stay local. That is inverted for privacy and for
  cost (LEARNING_plan F11).
- **Fix outline:** Per LEARNING_plan §F11: a cheap local signal (classifier margin,
  logprob of the first tokens), selected by config, with the length stub kept only as an
  explicit opt-in. Until then, document the inversion next to the thresholds.
- **Acceptance test:** A golden set of short-but-easy and long-but-hard prompts with known
  correct routes. The new signal must route them better than the stub, with significance
  over paired vectors (the CLAUDE.md §7 gate). A regression test that a 20-char "hi"
  under `routing.remote.yaml` does **not** escalate fails against the stub.

### B-010
**A promoted adapter with no significance proof is served by default; golden sets are below the gate's floor**

- **Priority:** P1 · **Status:** open · **Effort:** M
- **Evidence:** `scripts/hearth_status.py --section learning`:
  ```
  [warn ] data/extract_golden.jsonl: n=6 — gates only in the near-degenerate case (best p=0.0156)
  [warn ] data/route_golden.jsonl: n=5 — gates only in the near-degenerate case (best p=0.0312)
  [warn ] adapter:classify-20260710T020135Z: promoted, task=classify, weights present
  ```
  `~/.hearth/adapters.json`: the classify adapter's `promotion_proof` is
  `{"candidate_score": 1.0, "gate_passed": true, "incumbent_score": 0.2}`. It has no
  `p_value` and no `gate: "unverified"` stamp, because it predates
  `registry/adapters.py:171`. Its `base_model` is Coder-7B, the default
  (`config/models.yaml:16`), so `Router._resolve_adapter` (`route.py:395-406`) serves it
  for **every** `classify` request with no adapter named. Gate default `min_n`:
  `training/eval.py:53` `DEFAULT_MIN_N = 30`.
- **Impact:** An adapter promoted on two typed floats (the path `hearth adapters promote`
  has since removed, see B-015) is live by default. No golden set in the repo is big enough
  for any future promotion to pass the default gate.
- **Fix outline:** Retire the classify adapter (`hearth adapters retire`) or re-evaluate it
  via `hearth eval --prereg <committed> --promote`. Grow both golden sets to ≥30, kept
  disjoint from the training corpora. Consider having the router refuse to auto-serve a
  promoted adapter whose proof lacks `p_value`.
- **Acceptance test:** The status learning section reports no `warn`. A `classify` request
  with no adapter named serves base weights unless the promoted entry carries a `p_value`.
  `tests/test_eval_gate_replay.py` must still refuse the historical promotion.

---

## P2: quality / UX

### B-011
**The MLX embedder cannot work: the default id 404s, and mlx-lm has no BERT architecture**

- **Priority:** P2 · **Status:** open · **Effort:** M
- **Evidence:**
  - `config.py:44` `embed_model = "mlx-community/bge-small-en-v1.5-mlx"`.
    `config/models.yaml:65-67` says that id "404s upstream" and registers
    `mlx-community/bge-small-en-v1.5-bf16` instead (`:70`). The status script finds the
    `-bf16` weights on disk, so the setting names a model the resolver will refuse.
  - The `-bf16` weights are BERT: their `config.json` has `"architectures": ["BertModel"]`
    and `"model_type": "bert"`. `MLXEmbedder` loads through `mlx_lm.load`
    (`memory/embed.py:142-153`). mlx-lm 0.29.1 ships 102 model modules, none of them
    `bert`:
    ```
    ImportError: No module named 'mlx_lm.models.bert'
    ValueError: Model type bert not supported.   # mlx_lm.utils._get_classes({'model_type':'bert'})
    ```
- **Impact:** `HEARTH_EMBEDDER=mlx` fails with the default setting (not on disk), and fails
  with the correct id too (unsupported architecture). RAG only ever runs on the
  `HashEmbedder` (`config.py:40`), which has no semantic recall.
- **Fix outline:** Load BERT-family embedders with a BERT-capable MLX implementation (e.g.
  `mlx-embeddings`, or a small in-repo BERT forward pass) rather than `mlx_lm`. Fix the
  `config.py:44` default to the registered `-bf16` id. Keep disk-only resolution and the
  MLX thread.
- **Acceptance test:** With the real `-bf16` weights present, `MLXEmbedder.embed(["a","b"])`
  returns two 384-d vectors, and a retrieval check (a query nearer its paraphrase than an
  unrelated passage) passes where `HashEmbedder` fails. Pointing it back at `mlx_lm.load`
  must fail the test.

### B-012
**Gateway agent route: no finance tools, and `rag_search` is not pinned to a collection**

- **Priority:** P2 · **Status:** open · **Effort:** S
- **Evidence:** `gateway/agent_route.py:93-97` builds the toolset with
  `finance=getattr(state, "finance", None)`, and nothing assigns `app.state.finance`
  (`grep -rn "\.finance\s*=" src/hearth`: no hits). The same call passes no `collection`,
  so `rag_search_tool` registers unpinned (`agent/builtins.py:182-199, 374-387`) and the
  model has to name a collection itself. `AgentRunRequest` has no collection field
  (`gateway/schemas.py:242-249`, `extra: forbid`). The CLI pins both
  (`cli.py:344-360, 401`).
- **Impact:** Over HTTP (the `/chat` agent toggle) the agent can't answer ledger questions,
  and RAG depends on the model guessing a collection name. `hearth agent` at the CLI behaves
  differently.
- **Fix outline:** At `create_app`, attach a `FinanceStore` when a ledger is configured
  (respecting the finance profile's no-egress rule). Add a server-side setting for the
  agent's RAG collection. Don't let the wire choose the collection or the store.
- **Acceptance test:** With a synthetic ledger configured (never real data, CLAUDE.md §4),
  the route's start event lists `finance_total`, and a `rag_search` call with no collection
  argument searches the configured one. Removing the attachment must fail the test.

### B-013
**Agent: no search tool, an unmeasured step cap, and no tool-calling eval set**

- **Priority:** P2 · **Status:** open · **Effort:** M
- **Evidence:** The built-in tools are `read_file`, `list_files` (a glob over *names*,
  `agent/builtins.py:119`), `rag_search`, and the three `finance_*` tools
  (`builtins.py:87, 153, 232, 324-344`). None searches file contents.
  `agent/loop.py:141` `max_iterations: int = 8`. `docs/AGENT.md` §7, items 1-2
  (`AGENT.md:413-418`): the step default "is a judgement, not a measurement", and
  prompt-based vs native tool calling is unmeasured for lack of an eval set. *Reported, not
  reproduced here:* a live run that invented a filename ("meeting notes 4:" for `n4.txt`).
  This needs a real model, which this check did not load.
- **Impact:** Finding a fact means reading files one at a time against an 8-step cap, so
  tasks over more than a handful of files stop at the bound. There is no measurement to
  say whether 8 is too few, or whether prompting beats native tool calls.
- **Fix outline:** Add a read-only, root-confined `grep_files(pattern, root)` with capped
  output (same deny-by-default roots and AST no-network test). Build a tool-calling golden
  set (tasks with known-correct tool sequences) at n ≥ 30. Measure the step cap per model
  under the §7 gate rules.
- **Acceptance test:** With 20 files under a root and one containing the target fact, an
  agent run finds it within the cap on a scripted fake model that calls `grep_files`. The
  same task with grep removed hits `max_iterations`. Eval-set acceptance is the §7 gate
  itself.

### B-014
**Untested end to end: an actual cmux open-tier launch; the `/chat` agent toggle in a browser**

- **Priority:** P2 · **Status:** blocked (needs cmux.app plus a `tiers.yaml` `open` rule;
  needs a browser) · **Effort:** S each
- **Evidence:** Commit `816d824`: "Not verified: an actual open-tier launch (needs a
  tiers.yaml 'open' rule and cmux.app)". `tests/test_gateway_chat_ui.py` (15 tests) asserts
  on served markup, e.g. a regex for the `agentmode` input at `:93-102`. No test executes
  the page's JavaScript (no playwright/selenium/jsdom in `tests/`). Only the
  `/v1/hearth/agent` API was exercised live.
- **Impact:** The open tier's profile switch (`routing.remote.yaml`) and the toggle's
  `localStorage` / rendering logic (`docs/AGENT.md` §9.5) could be broken with all tests
  green.
- **Fix outline:** Run one real open-tier launch and record the result in
  `docs/cmux/RESULTS.md`. Add a headless-browser smoke test (optional extra), or record a
  manual check.
- **Acceptance test:** Open tier: inside the launched workspace, `hearth stats` / status
  reports `active_profile` = `routing.remote.yaml`. Toggle: with it on, an agent run
  renders each step as a turn, and with it off the page makes no `/v1/hearth/agent` request
  (watched at the network layer, not inferred from markup).

---

## P3: hygiene

### B-015
**`scripts/train_lora_real.sh --promote` always fails: it passes flags the CLI has removed**

- **Priority:** P3 · **Status:** open · **Effort:** S
- **Evidence:** `scripts/train_lora_real.sh:147,153-154` requires `--candidate-score` and
  calls `hearth adapters promote <id> --candidate-score ... [--incumbent-score ...]`.
  `cli.py:1270-1301` keeps those options hidden only to reject them: it prints "have been
  removed" and exits **2**. `hearth adapters promote` actually requires `--report` and a
  committed `--prereg`, then recomputes the gate from per-example vectors
  (`cli.py:1302-1376`). **This is not a gate bypass.** The script's promote path is dead.
  (A Python caller can still use the legacy `AdapterStore.promote(gate_passed=True)`, but
  the proof gets stamped `gate: "unverified"`, `registry/adapters.py:132-135, 171`.
  No `src/` caller uses it: `grep -rn "gate_passed=" src/hearth`, no hits.)
- **Impact:** `train_lora_real.sh --promote` trains for minutes, then fails at the last
  step. Its help text (`:47-48, 60`) teaches the removed workflow.
- **Fix outline:** Replace the promote block with `hearth eval <adapter> --golden <set>
  --prereg <committed prereg> --promote`, and drop `--candidate-score/--incumbent-score`
  from the script and its help.
- **Acceptance test:** With `hearth` stubbed to record argv, `train_lora_real.sh --promote
  --prereg X --golden Y` (training stubbed) invokes `hearth eval ... --prereg X --promote`
  and never `adapters promote --candidate-score`. The current script fails it.

### B-016
**`docs/RESULTS.md` and `docs/HANDOFF.md` are stale, and the status script doesn't track HANDOFF**

- **Priority:** P3 · **Status:** open · **Effort:** S
- **Evidence:** Status: `docs/RESULTS.md: 2026-07-20 @ 5cbd061, 87 commits since`.
  `docs/HANDOFF.md` was last changed at `e5ed305` (2026-07-10), **91** commits ago
  (`git rev-list --count e5ed305..HEAD`). It isn't in `KEY_DOCS`
  (`status/probes.py:1174-1181`, which lists `docs/cmux/HANDOFF.md` instead), so nothing
  flags it. HANDOFF still describes an agent with "the ability to download model weights
  from Hugging Face" (`HANDOFF.md:3-6`). Commits `c91ec42` and `54b152f` left both files
  unchanged on purpose, as historical records.
- **Impact:** A fresh agent reading them takes them as current: downloads allowed, the
  typed-score promotion path, old test counts.
- **Fix outline:** Add a top banner to each: "Historical record of <date>; not current. See
  docs/STATUS.md and run scripts/hearth_status.py". Add `docs/HANDOFF.md` to `KEY_DOCS`.
- **Acceptance test:** The status staleness section lists `docs/HANDOFF.md`. A test asserts
  each historical doc's first 5 lines contain the banner, and removing the banner fails it.

### B-017
**`docs/LEARNING_plan.md` cites stale line numbers and a wrong class count**

- **Priority:** P3 · **Status:** open · **Effort:** S
- **Evidence:** It cites `_confidence` at `router/route.py:348-360` (`:245, 281, 977, 1100`),
  but it is now `route.py:425-437`. It cites `Router.route` at `route.py:232-245`
  (`:295, 330, 523`), now `route.py:189-263`. Adapter resolution at `route.py:310-333`
  (`:866, 964`) is now `_resolve_adapter` at `:364-410`. `_OBJECTIVE_CLASSES` at
  `eval.py:28` and `objective_metric_for` at `eval.py:147-149` (`:88-89`) are now
  `eval.py:45` and `:612-614`. `LEARNING_plan.md:962` says "**Six of nine classes are pinned
  local**" but lists four. `config/routing.remote.yaml` defines **8** classes, four of them
  `local / never`. The ninth `TASK_CLASSES` entry, `embed` (`router/classify.py:15-25`),
  is absent from the yaml and falls back to the local default, so the true count is 5 of 9.
- **Impact:** A reader follows the citations to the wrong code. The headline count is wrong.
- **Fix outline:** Cite symbols (`route.py:_confidence`) rather than line ranges, and fix
  the count.
- **Acceptance test:** For each `file.py:NNN` citation in the doc, a script checks the named
  symbol appears within the cited range. Today it reports the mismatches above.

### B-018
**ruff: 82 findings outside `src/`** (the "7 pre-existing E501" undercounts)

- **Priority:** P3 · **Status:** open · **Effort:** S
- **Evidence:** `.venv/bin/ruff check --no-cache --statistics <repo>`: `63 E501, 13 E702, 1
  B905, 1 E402, 1 E731, 1 F401, 1 I001, 1 UP035 — Found 82 errors`, across
  `scripts/cmux/orchestrator.py` (18), `scripts/cmux/tier_classify.py` (13),
  `scripts/coreml_stateful_reference.py` (11), `tests/test_cmux_orchestrator.py` (10),
  `scripts/cmux/lulu_rule_check.py` (10), `scripts/cmux/orchestrator_demo.py` (7),
  `scripts/coreml_stateful_repro.py` (6), `tests/test_cmux_lulu_rule_check.py` (4) and
  `examples/cmux/offload_demo.py` (3). `src/` is clean. The 7 in `orchestrator_demo.py` are
  the ones commit `3919f7a` noted. Its check covered only files changed since `950f3ce`.
- **Impact:** Repo-wide `ruff check` can't serve as a gate, so new lint hides among the old.
- **Fix outline:** Fix them (3 are auto-fixable), or scope `extend-exclude` explicitly, then
  gate CI on a clean `ruff check .`.
- **Acceptance test:** `ruff check .` exits 0, and re-adding one over-long line makes it
  exit non-zero.

### B-019
**Unregistered weights on disk: `Qwen/Qwen2.5-0.5B-Instruct`**

- **Priority:** P3 · **Status:** open · **Effort:** S
- **Evidence:** Status: `[warn ] Qwen/Qwen2.5-0.5B-Instruct: on disk but UNREGISTERED (953.3
  MiB) — weights in the hub cache name no entry in config/models.yaml`.
- **Impact:** 953 MiB nothing can serve and nothing will garbage-collect. Someone may also
  assume it is servable.
- **Fix outline:** The operator decides: register it in `config/models.yaml` (if wanted,
  e.g. as a draft/test model) or delete it from the hub cache. Agents should not delete
  operator files unasked.
- **Acceptance test:** The status models section shows no `UNREGISTERED` line.

### B-020
**Starlette deprecation: `httpx` with `starlette.testclient`**

- **Priority:** P3 · **Status:** open · **Effort:** S
- **Evidence:** Importing `fastapi.testclient` prints `StarletteDeprecationWarning: Using
  httpx with starlette.testclient is deprecated; install httpx2 instead.` (seen while running
  the B-003 probe).
- **Impact:** None today. It breaks when Starlette drops httpx support.
- **Fix outline:** Move to the replacement Starlette names, under the pyproject ceiling.
  Pin the version that does.
- **Acceptance test:** `pytest -W error::DeprecationWarning` over the gateway tests passes.

### B-021
**`providers/mlx.py:audit_resolution` swaps `socket.socket.connect` process-wide**

- **Priority:** P3 · **Status:** open (latent) · **Effort:** S
- **Evidence:** `mlx.py:199-208` replaces `socket.socket.connect` / `connect_ex` for the
  whole process during a resolve. The docstring says "Single-threaded use only" (`:189`).
  Its only caller is the status probe (`status/probes.py:541-545`), and nothing in
  `gateway/` imports `hearth.status`, so the hazard is not live. Nothing enforces that
  either.
- **Impact:** If it is ever called inside the server, every concurrent connection in the
  process (a remote escalation, the HTTP client) is refused, and those refusals are counted
  as resolver egress. That would be a false FAIL plus a broken request.
- **Fix outline:** Assert at entry that only one thread is alive, or that we're not inside
  the server (e.g. `threading.active_count() == 1`, or a module flag set by `create_app`).
  Better: audit in a subprocess.
- **Acceptance test:** Calling `audit_resolution` while a second thread is alive raises.
  An AST test asserts `gateway/` never imports `audit_resolution` or `hearth.status`.

### B-022
**`REASON_LOCAL_FAILURE` is declared and never used**

- **Priority:** P3 · **Status:** open · **Effort:** S
- **Evidence:** `router/route.py:40`. `grep -rn REASON_LOCAL_FAILURE src tests` hits only the
  declaration.
- **Impact:** It suggests local failures get recorded with a reason. They don't (B-003).
- **Fix outline:** Use it when recording failed requests (B-003), or delete it.
- **Acceptance test:** Folded into B-003's test: a recorded local failure carries this
  reason.

### B-023
**Test-infra trap: with `NO_PROXY` empty, loopback HTTP is proxied and client disconnects are hidden**

- **Priority:** P3 · **Status:** open · **Effort:** S
- **Evidence:** In the agent shell, `NO_PROXY=[] no_proxy=[]` while `HTTP_PROXY`,
  `http_proxy` and `HTTPS_PROXY` are set (measured with `echo`). Commit `92d8f19` recorded
  the result. curl to `127.0.0.1` went through the proxy, which held the upstream
  connection open, so the server saw no disconnect ("666 successful sends in 8 s, no
  connection_lost") and the fix appeared to fail.
- **Impact:** Any live disconnect / streaming-cancel check run with curl or a
  proxy-honouring client measures the proxy, not HEARTH.
- **Fix outline:** Document it in CLAUDE.md / the runbooks. Live disconnect tests use a
  raw socket (as `92d8f19` does), or export `NO_PROXY=127.0.0.1,localhost` for the test
  process.
- **Acceptance test:** A live disconnect harness asserts the server logged
  `connection_lost` before it judges cancellation, so a proxied run reports
  "inconclusive" instead of "fix failed".

### B-024
**`scripts/train_lora_real.sh` refuses models fetched with `hearth models pull`**

- **Priority:** P2 · **Status:** **FIXED in `e00e5e1` (`scripts/check_base_on_disk.py`, called by the script)** · **Effort:** S
- **Evidence:** The script's "is the base cached?" heredoc calls
  `snapshot_download(repo_id=repo, local_files_only=True)` with no `cache_dir`
  (`scripts/train_lora_real.sh` ~line 121), so it looks only in the hub cache. `hearth models
  pull` writes to `~/.hearth/models`. Reported by the B-001 agent for
  `mlx-community/Qwen2.5-3B-Instruct-4bit`, which lives only in `~/.hearth/models`.
- **Impact:** The documented real-training script rejects a correctly pulled base model.
- **Fix outline:** Replace the heredoc with `hearth.providers.mlx.resolve_local_model(repo,
  allow_downloads=False)`, the resolver every other load path uses.
- **Acceptance test:** With a model planted only in a temp `~/.hearth/models`, the pre-check
  passes; with it absent, it fails with the `hearth models pull` hint and zero connects.

### B-025
**Status probe expands `~` in `HEARTH_ROUTING_YAML`; the router does not**

- **Priority:** P2 · **Status:** **FIXED in `2550766` (shared resolver; see B-008)** · **Effort:** S
- **Evidence:** `src/hearth/status/probes.py:414` uses `Path(active).expanduser()`;
  `src/hearth/router/policy.py:96` returns `Path(override)` unexpanded.
- **Impact:** For `HEARTH_ROUTING_YAML=~/x.yaml` the status report describes a file the
  router cannot open; the router then falls back to its built-in safe defaults (all local),
  so it fails safe, but the report and the running policy disagree — the CLAUDE.md §3 shape.
  `hearth doctor --offline` uses the router's own function and is not affected. Related: B-008.
- **Fix outline:** One shared resolver (expand `~`; decide relative-path base per B-008) used
  by both the router and the probe.
- **Acceptance test:** For a `~/...` path, the probe's reported path equals the path
  `load_policy()` actually read; reverting either side fails the test.

### B-026
**A failed `hearth train` leaves an empty run directory**

- **Priority:** P3 · **Status:** **FIXED in `3fae564`** · **Effort:** S
- **Evidence:** `src/hearth/training/lora.py:121` creates `data_dir` before
  `runner_invocation` (`:212`) resolves the base model, so a `ModelNotOnDiskError` exits
  after the directory exists.
- **Fix outline:** Resolve the base model before creating any directory.
- **Acceptance test:** A train with a not-on-disk base exits 1 and creates nothing under the
  train root.

### B-027
**A failed training subprocess ends `hearth train` in a traceback**

- **Priority:** P2 · **Status:** **FIXED in `0f1bbe2`** · **Effort:** S
- **Evidence:** `src/hearth/cli.py` (train command, ~line 846) catches only `RuntimeError`;
  the runner uses `subprocess.run(..., check=True)`, whose `CalledProcessError` is a
  `SubprocessError`, not a `RuntimeError`.
- **Impact:** A training failure (OOM, bad data) prints a Python traceback instead of a
  one-line error and exit code.
- **Fix outline:** Catch `subprocess.CalledProcessError` and report the return code and the
  child's last stderr lines; exit 1.
- **Acceptance test:** A runner that exits non-zero yields exit 1 and no traceback in output.

### B-028
**`hearth models export-coreml` reports missing coremltools before a missing model**

- **Priority:** P3 · **Status:** open (reported by the B-001 agent; not reproduced here —
  this venv has no coremltools) · **Effort:** S
- **Impact:** On a machine without coremltools, a mistyped model id is not reported until
  coremltools is installed.
- **Fix outline:** Resolve the source (disk-only) before the coremltools availability check.

### B-029
**An unregistered `HEARTH_DEFAULT_MODEL` is silently ignored; `doctor --offline` does not warn**

- **Priority:** P2 · **Status:** **FIXED in `d931473` (doctor WARN row + one registry log line; fallback unchanged. The row shows FAIL under `--offline`, see B-031)** · **Effort:** S
- **Evidence:** `src/hearth/registry/__init__.py:59-73` applies `HEARTH_DEFAULT_MODEL` only
  when it names a registry entry. Measured: `HEARTH_DEFAULT_MODEL=mlx-community/
  Qwen2.5-Coder-32B-Instruct-4bit hearth doctor --offline` → "SAFE", exit 0, checking the
  catalog default (Coder-7B). The verdict is correct (it checked what would serve), but
  nothing tells the operator their setting was discarded.
- **Impact:** The operator believes a different model is serving — the trap CLAUDE.md §2
  describes for misspelled variable names, one level down.
- **Fix outline:** `doctor` (and `serve` startup) WARN when `HEARTH_DEFAULT_MODEL` is set but
  unregistered, naming the model that will actually serve.
- **Acceptance test:** With the variable set to an unregistered id, doctor output contains a
  WARN naming both ids; removing the check fails the test.

### B-030
**Test-infra: `sandbox-exec` is unavailable to agents and inside harness worktrees**

- **Priority:** P3 · **Status:** open (environment) · **Effort:** —
- **Evidence:** `sandbox-exec -f /tmp/hearth_noegress.sb /usr/bin/true` → `sandbox_apply:
  Operation not permitted` (exit 71) from subagents, and from the main session once the
  harness had switched it into a worktree (2026-10-03). It worked earlier in the same
  session.
- **Impact:** Kernel-level no-egress proof cannot be produced by an agent. Agents fall back to
  a Python-level connect guard (`sitecustomize`), which does not see DNS or native sockets.
- **How to apply:** The operator runs the kernel-level check from their own shell
  (docs/PRIVACY.md, "Verifying no egress yourself"). Label agent evidence as Python-level.

### B-031
**`hearth doctor --offline` renders a non-fatal WARN row as FAIL**

- **Priority:** P3 · **Status:** open · **Effort:** S
- **Evidence:** `src/hearth/cli.py` `_doctor_offline` sets
  `mark = "[green]PASS[/green]" if c.ok else "[red]FAIL[/red]"`, ignoring `Check.fatal`
  (plain `hearth doctor` distinguishes WARN). Measured: the B-029 `default_model` row reads
  `FAIL` with a `WARN:` detail while the verdict is `SAFE offline`.
- **Fix outline:** Same three-way mark as `doctor`: PASS / FAIL (fatal) / WARN (non-fatal).
- **Acceptance test:** With an unregistered `HEARTH_DEFAULT_MODEL`, the `--offline`
  `default_model` row's status column is `WARN`.

### B-032
**`scripts/train_lora_real.sh` runs `uv run --no-sync` from the caller's cwd**

- **Priority:** P3 · **Status:** **FIXED** in `554923d` — reproduced first: from /tmp with no
  VIRTUAL_ENV the old pre-check exited 1 (mlx-lm "missing"), the fixed one exits 0 ·
  **Effort:** S
- **Evidence:** Unlike `scripts/hearth_private.sh` (`cd "$REPO_ROOT"`), the training script
  never changes directory, and every `uv run --no-sync ...` discovers its project from the
  cwd. Started outside the repo, the mlx-lm probe would run against a different (or no)
  project and could report "mlx-lm is not installed" for a correctly synced venv.
- **Fix outline:** `cd` to the repo root (as `hearth_private.sh` does), or pass
  `--project "$REPO_ROOT"` to each `uv run`.

---

## Fixed recently, do not re-open

Closed by the 2026-10 offline work (`git log 950f3ce..HEAD`, merged at `d803f1a`). Each commit
message carries its own WHAT / WHY / HOW VERIFIED.

| Commit | What it closed |
|---|---|
| `49eadd6` | Default `config/routing.yaml` is no-egress (zero remotes). Escalation is opt-in via `config/routing.remote.yaml`. |
| `41a42c8` | Serving loads (serve, chat, agent, MCP, RAG, embedder, `bench.py`) resolve from disk only. `ModelNotOnDiskError` unless `HEARTH_ALLOW_DOWNLOADS=1`. Previously mlx-lm called `snapshot_download` with the network on. |
| `6cdcee9` | A failed escalation is served locally instead of returning 503. `RequestRecord.escalation_failed` and `hearth stats` `escalations_failed` were added. A failed stream ends with an error event plus `[DONE]`. |
| `fd5ae11` | All MLX work runs on one process-wide thread. Concurrent chat no longer fails with "There is no Stream(gpu, 0) in current thread". |
| `f1bc78f` | No script calls bare `uv run` (it uninstalled mlx). |
| `c91ec42` | Docs state the no-egress default and which load paths are disk-only. `docs/PRIVACY.md:68-69` carries the caveat that a failed remote call may already have sent the prompt ("served local" describes the answer, not the egress). |
| `3dfec67` | Status `serving_load_egress` judges disk-only by **counted connects**, not by exception type. `HEARTH_ALLOW_DOWNLOADS` is parsed with Settings' own bool rules. |
| `2066db8` | A remote stream that dies mid-answer is billed and recorded. A bad adapter on the streaming path is retried on base weights. |
| `816d824` | The cmux open tier points at `routing.remote.yaml` (had silently lost escalation). No bare `uv run` in `scripts/cmux/`. |
| `92d8f19` | An abandoned stream is closed, so generation stops instead of holding the single MLX thread to `max_tokens`. |
| `15a6b02` | `MLXEmbedder` disk-only load and MLX-thread confinement are now tested. (Its architecture problem is still open, B-011.) |
| `54b152f` | Install hints use the one-command sync, so they no longer prune the other extras. |
| `3919f7a` | Lint introduced on the branch was cleared. (The pre-existing findings are B-018.) |
| `42ebf1f` | **B-001.** `hearth train`, `models convert`, `models export-coreml` resolve from disk and pin the hub offline in child processes; `models pull` is the only download path. Also fixed: every real `models convert` failed because the output dir was pre-created. |
| `2868160` | `hearth doctor --offline`: a measured safe/unsafe verdict (exit 1 when unsafe) over routing, serving resolution, every reachable model, every load path, bind host and `HEARTH_ALLOW_DOWNLOADS`. |
| `2550766` | **B-008, B-025.** One `HEARTH_ROUTING_YAML` resolver for router, status probe and doctor: `~` expanded, relative paths against the repo root (not the cwd), and a missing selected profile is a startup error instead of a silent fall back to safe defaults. |
| `e00e5e1` | **B-024.** `train_lora_real.sh` checks the base model with `resolve_local_model` (via `scripts/check_base_on_disk.py`), so models from `hearth models pull` are accepted. |
| `3fae564` | **B-026.** `hearth train` resolves the base model (and checks the mlx extra) before creating the run dir; a failed run leaves nothing behind. |
| `0f1bbe2` | **B-027.** A failed `mlx_lm.lora` child ends `hearth train` with exit 1, its return code and any captured stderr tail, no traceback. |
| `d931473` | **B-029.** `hearth doctor` / `doctor --offline` WARN when `HEARTH_DEFAULT_MODEL` is unregistered and ignored, naming the model that serves; the registry logs it once (seen at `hearth serve` startup). |
