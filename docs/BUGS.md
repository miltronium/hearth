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
- ~~[B-003](#b-003) Escalation fails, then local fails: no record that the prompt may have left~~ — fixed in `48982b0`

**P1: correctness**
- ~~[B-004](#b-004) MLXProvider ignores `GenRequest.model`; the model ladder silently serves the default (in progress)~~ — fixed in `2d186dd`
- ~~[B-005](#b-005) `/ready` returns 200 with no weights loaded (in progress)~~ — fixed in `2d186dd`
- ~~[B-006](#b-006) `HEARTH_BACKEND=auto` silently becomes the echo stub, and the stub reports ready~~ — fixed in `12d5366`
- ~~[B-007](#b-007) An exception after the stream relay ends the SSE stream with no `[DONE]`~~ — fixed in `b6303c6`
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
- ~~[B-031](#b-031) `hearth doctor --offline` renders a non-fatal WARN row as FAIL (P3)~~ — fixed in `2cf9af6`
- [B-032](#b-032) `train_lora_real.sh` runs `uv run --no-sync` from the caller's cwd (P3)

**Added 2026-10-05 (found by the docs agent and the B-008 merge check)**
- ~~[B-033](#b-033) `hearth serve` with a missing named routing profile prints a full traceback (P2)~~ — fixed in `51b9a30`
- ~~[B-034](#b-034) Unknown adapter: the response claims the adapter it did not use (P1)~~ — fixed in `25aae9f`
- ~~[B-035](#b-035) `HEARTH_WARMUP=false` leaves `/ready` at 503 "loading" forever on mlx (P2)~~ — fixed in `927b686`
- ~~[B-036](#b-036) `hearth rag ingest` with `HEARTH_EMBEDDER=mlx` ends in a traceback (P2)~~ — fixed in `41bf026`
- ~~[B-037](#b-037) `docs/API.md` documents endpoints that do not exist, and the wrong error envelope (P2)~~ — fixed in `2c8f5f9`
- ~~[B-038](#b-038) `docs/PRIVACY.md` "Formats" row says text/CSV only (P3)~~ — fixed in `48634a5`
- ~~[B-039](#b-039) `training/dataset.py` promises headerless datasets, then refuses them (P3)~~ — fixed in `46bddeb`
- ~~[B-040](#b-040) Example docs give install/run commands that prune the venv (P2)~~ — fixed in `0c3b5f9`
- ~~[B-041](#b-041) `docs/RUNBOOK_training.md` still teaches the removed `--candidate-score` promote path (P2)~~ — fixed in `e5761b1`
- ~~[B-042](#b-042) `Settings.default_model` is never read (P3)~~ — fixed in `eef3bd5`
- ~~[B-043](#b-043) `hearth_peek.py` output no longer matches the sample in `RUNBOOK_finance.md` §2 (P3)~~ — fixed in `cf3b6e9`

**Added 2026-10-05 (second batch)**
- ~~[B-044](#b-044) `hearth_peek.py` printed preamble values as headers~~ — fixed in `1883397` (P0)
- ~~[B-045](#b-045) unedited prereg template / empty baselines could gate a promotion~~ — fixed in `9165034` (P1)
- ~~[B-046](#b-046) `hearth train` steers to `adapters promote` without `--report/--prereg` (P3)~~ — runtime message fixed in `21b30d6` (the `train` docstring is left to the cli.py help pass)

**Added 2026-10-05 (after the model-selection merge)**
- ~~[B-047](#b-047) bogus `HEARTH_DEFAULT_MODEL`: `/ready` failed but `auto` served by the default (P1)~~ — fixed in `45dc5e1` (CLI refuses to start; see the item for what `create_app` still does)
- ~~[B-048](#b-048) `/ready` 503 when the default is evicted (P2)~~ — fixed in `927b686`
- ~~[B-049](#b-049) finance example computes money with float (P1)~~ — fixed in `9d9177e`

**Added 2026-10-05 (CLI-polish batch; recorded as found and fixed)**
- ~~[B-050](#b-050) `--help` prose is ragged at 80 columns (P3)~~ — fixed in `9fd539c`
- ~~[B-051](#b-051) `hearth stats` hides `failed` / `failure_rate`; "(served local)" label is false (P2)~~ — fixed in `6eb4fac`
- ~~[B-052](#b-052) `hearth train` with a 1-record dataset ends in a traceback (P3)~~ — fixed in `eff5a9c`
- ~~[B-053](#b-053) `hearth serve` printed "Serving on" before `create_app` raised (P3)~~ — fixed in `51b9a30`, pinned by `ff192b5`
- ~~[B-054](#b-054) `ModelNotOnDiskError` says `hearth models pull` for paths and unregistered ids (P2)~~ — fixed in `5556f5d`
- ~~[B-055](#b-055) MLX embedder error: "..", and sandbox advice (P3)~~ — fixed in `1d4da3f`
- ~~[B-056](#b-056) `rag query` embeds before checking the collection is empty (P2)~~ — fixed in `0f72c06`
- ~~[B-057](#b-057) `serve`'s stderr log handler: stale stream, suppressed by any foreign handler (P3)~~ — fixed in `258ff98`
- ~~[B-058](#b-058) `doctor --offline` `serving_resolution` says `~/.hearth/models` under any `HEARTH_HOME` (P3)~~ — fixed in `f014ff6`
- ~~[B-059](#b-059) An unknown `HEARTH_BACKEND` ends every command in a traceback (P3)~~ — fixed in `3951d2e`
- ~~[B-060](#b-060) An unknown `hearth.intent` / `--intent` is silently ignored (P3)~~ — fixed in `3951d2e`

**Added 2026-10-05 (adversarial review of the day's merges)**
- ~~[B-061](#b-061) `hearth adapters promote --report` accepts a hand-written report: nothing ties it to the a… (P0)~~ — fixed in `13038c5`
- ~~[B-062](#b-062) Prereg `bar` is not range-checked: NaN alpha/min_effect or alpha=1, min_effect<0, min_n=1 … (P0)~~ — fixed in `a731e8c`
- [B-063](#b-063) `hearth_peek.py` still prints cell values: a full-width all-text preamble row, a headerles… (P0)
- ~~[B-064](#b-064) `/ready` judges only the registry default; under a routing ladder the default may never se… (P1)~~ — fixed in `9645883`
- ~~[B-065](#b-065) Routing validation gaps: `defaults.local_model` is never validated; a class rung may name … (P1)~~ — fixed in `ed923e1`
- ~~[B-066](#b-066) Failure accounting gaps: an unservable ladder rung (404 / stream error event) writes no Re… (P1)~~ — fixed in `03f79b9`
- [B-067](#b-067) `doctor --offline` says SAFE with a plugin embedder or vector store, which receive every R… (P1)
- ~~[B-068](#b-068) The auto→echo fallback stub labels its echo with the requested real model and credits toke… (P2)~~ — fixed in `d5a7055`
- ~~[B-069](#b-069) LoRA adapter variants are full base reloads the ModelManager never counts (P2)~~ — fixed in `e8fcb8a`
- ~~[B-070](#b-070) B-047 enforced at CLI call sites, not where the model is chosen (P2)~~ — eval half fixed in `58890fa`, serving side in `36157af`
- ~~[B-071](#b-071) An abandoned agent run keeps the single MLX thread busy for up to its budget (P2)~~ — fixed in `1a872a2`
- ~~[B-072](#b-072) ModelManager evicts residents before knowing the new load will succeed (P3)~~ — fixed in `7c9e510`
- ~~[B-073](#b-073) Low: finance ladder example resolves HEARTH_ROUTING_YAML itself and silently falls back; c… (P3)~~ — fixed in `d4eb825` (2 of 3) and `dd4b774` (adapter field)
- ~~[B-074](#b-074) `hearth_map_draft.py` printed a preamble as column names, drafted skip_rows 0, printed file names and a total~~ — fixed in `052d3af` (P0)
- ~~[B-075](#b-075) `routing.finance.yaml` does not pin the `embed` class: `intent: embed` reaches the registry default 7B (P3)~~ — fixed in `c3cb201`
- ~~[B-076](#b-076) CLI prints "Routing profile not found:" for a profile that exists but names an unservable rung (P3)~~ — fixed in `c3cb201`
- ~~[B-077](#b-077) Status/doctor report "policy loader unavailable" for a profile with a bad model rung, hiding the reason (P3)~~ — fixed in `c3cb201`

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

- **Priority:** P0 (unrecorded possible egress) · **Status:** **FIXED** in `48982b0` (`RequestRecord.failed`; rollup `failed` / `failure_rate`) · **Effort:** S
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

- **Priority:** P1 · **Status:** **FIXED** in `2d186dd` (merge of the model-selection branch) ·
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

- **Priority:** P1 · **Status:** **FIXED** in `2d186dd` (merge of the model-selection branch) ·
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

- **Priority:** P1 · **Status:** **FIXED** in `12d5366` (WARNING at startup; `/ready` 503 `stub`; `/health` `backend_fallback`; `auto` still falls back) · **Effort:** S
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

- **Priority:** P1 · **Status:** **FIXED** in `b6303c6` (final chunk + `hearth.metrics.unavailable` + `[DONE]`; `_guarantee_done` backstop) · **Effort:** S
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

- **Priority:** P2 · **Status:** open (the `config.py` default id was fixed in `d4eb825`;
  the BERT architecture problem below is not) · **Effort:** M
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

- **Priority:** P3 · **Status:** **FIXED** in `2cf9af6` — same three-way PASS/FAIL/WARN mark as
  plain `doctor`; test asserts the status cell, reverting the hunk fails it · **Effort:** S
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

### B-033
**`hearth serve` with a missing named routing profile prints a full traceback**

- **Priority:** P2 · **Status:** **FIXED** in `51b9a30` — caught where the router is built in
  `serve`, `run`, `agent`, `mcp`, `rag query` (with or without `--answer`); exit 2, the router's
  message, no traceback. `stats`, `eval`, `rag ingest`, `models *` build no router · **Effort:** S
- **Evidence:** Measured: `HEARTH_ROUTING_YAML=config/no-such.yaml hearth serve` exits 1 with the right message, but only after a Rich traceback from `cli.py` `serve` → `create_app` raising `RoutingProfileNotFoundError` (introduced by `2550766`, B-008).
- **Impact:** Fails safe (server refuses to start) but buries the one-line fix under a traceback.
- **Fix outline:** Catch `RoutingProfileNotFoundError` in `serve` (and `run`/`agent`/`mcp`), print the message, exit 2.
- **Acceptance test:** Missing profile → exit 2, output contains the fix line and no 'Traceback'.

### B-034
**Unknown adapter: the response claims the adapter it did not use**

- **Priority:** P1 · **Status:** **FIXED** in `25aae9f` (explicit unknown adapter → 404 `adapter_not_found`; `hearth.adapter` = what served, else null) · **Effort:** S
- **Evidence:** Measured in-process: `hearth.adapter="no-such-adapter"` → server logs "adapter 'no-such-adapter' unresolved; serving base weights", response says `hearth.adapter: "no-such-adapter"`, 200. Telemetry echoes the request (`gateway/app.py` ~251 non-stream, ~633 stream) instead of the adapter that loaded.
- **Impact:** A client (or the eval/A-B flow) believes an adapter served that never did — the CLAUDE.md §3 bug class.
- **Fix outline:** Report the resolved adapter path/id (or null) from the generation, and decide whether an explicitly requested unknown adapter should be a 404 rather than a silent base-weights fallback.
- **Acceptance test:** Request an unknown adapter: response must not name it as served (null or 404); revert → test fails.

### B-035
**`HEARTH_WARMUP=false` leaves `/ready` at 503 "loading" forever on mlx**

- **Priority:** P2 · **Status:** **FIXED** in `927b686` (warmup off: 200 when the default's weights resolve on disk, `loaded: false`) · **Effort:** S
- **Evidence:** Reported by the docs agent, measured in-process on the pre-B-005 code (`gateway/app.py` ~117, ~142); the B-005 work (item 2) rewrites readiness — re-verify after it merges.
- **Impact:** Orchestrators that gate on `/ready` never route traffic to a server that is serving fine.
- **Fix outline:** With warmup off, `/ready` should report "not warmed (warmup disabled)" distinctly, or load lazily-on-first-request and then go ready.
- **Acceptance test:** With HEARTH_WARMUP=false, after one served request `/ready` is 200 (or a documented distinct state).

### B-036
**`hearth rag ingest` with `HEARTH_EMBEDDER=mlx` ends in a traceback**

- **Priority:** P2 · **Status:** **FIXED** in `41bf026` — `EmbeddingUnavailableError` caught
  around `ingest`/`query`; "Embedder unavailable: ..." and exit 1. The embedder itself still
  cannot work (B-011) · **Effort:** S
- **Evidence:** `src/hearth/cli.py` rag ingest (~line 740) has no handler for `EmbeddingUnavailableError` (checked: no `except` in the command body). The embedder itself cannot work (B-011).
- **Impact:** A clean, actionable error becomes a stack trace.
- **Fix outline:** Catch `EmbeddingUnavailableError` in `rag ingest`/`rag query`, print it, exit 1.
- **Acceptance test:** With the mlx embedder unavailable, exit 1 and no 'Traceback'.

### B-037
**`docs/API.md` documents endpoints that do not exist, and the wrong error envelope**

- **Priority:** P2 · **Status:** **FIXED** in `2c8f5f9` — the endpoint list matches
  `create_app().routes`; the never-built endpoints sit under "Not implemented", each marked;
  the error model documents the 401 `detail` nesting and the real types. Drift test
  `tests/test_api_doc_routes.py` checks both directions and the 401 body · **Effort:** S
- **Evidence:** Checked: `docs/API.md` names `/v1/hearth/classify`, `/v1/hearth/summarize`, `/v1/hearth/train/`, `/v1/hearth/train/{run_id}`; no route defines them (`grep @app.(get|post)` in `src/hearth/gateway`). Docs agent also reports `/admin/models/{id}/load|unload` and `/admin/adapters/...`, and that the real 401 is nested under `detail` (`gateway/auth.py:34`).
- **Impact:** Integrators build against an API that 404s.
- **Fix outline:** Regenerate the endpoint list from the FastAPI app (`app.routes`) and add a test that every documented path exists.
- **Acceptance test:** A doc/route drift test fails when API.md names a path the app does not serve.

### B-038
**`docs/PRIVACY.md` "Formats" row says text/CSV only**

- **Priority:** P3 · **Status:** **FIXED in `48634a5` (Formats row lists the real suffix table from `mcp/files.py`)** · **Effort:** S
- **Evidence:** Reported by the docs agent: `src/hearth/mcp/files.py:566-575` also reads `.json`, `.xlsx`, `.pdf`.
- **Impact:** Understates what the path-taking tools can read.
- **Fix outline:** Update the row from the code.
- **Acceptance test:** —

### B-039
**`training/dataset.py` promises headerless datasets, then refuses them**

- **Priority:** P3 · **Status:** **FIXED in `46bddeb` (docstring fixed; a headerless file is refused with an error naming the missing header)** · **Effort:** S
- **Evidence:** Reported by the docs agent: `src/hearth/training/dataset.py:182` docstring vs `load_dataset` raising "dataset task must be non-empty".
- **Impact:** Doc/code disagreement in the training path.
- **Fix outline:** Either support headerless files with an explicit `--task`, or fix the docstring.
- **Acceptance test:** A headerless file either loads with `--task` or the docstring no longer claims it does.

### B-040
**Example docs give install/run commands that prune the venv**

- **Priority:** P2 · **Status:** **FIXED in `0c3b5f9` (one-command sync and `uv run --no-sync [--project]` across `examples/`)** · **Effort:** S
- **Evidence:** Checked: `examples/claude_code_mcp.md:17,24,52,105` use `uv sync --extra mcp` and bare `uv run hearth ...`; `examples/cmux/hearth.mcp.json:8` says `uv sync --extra mlx --extra mcp` (prunes dev/files). The 54b152f sweep excluded `examples/`.
- **Impact:** Following the MCP setup example uninstalls mlx (CLAUDE.md §1).
- **Fix outline:** Apply the one-command sync and `uv run --no-sync` to `examples/`.
- **Acceptance test:** grep finds no single-extra `uv sync` or bare `uv run` in examples/.

### B-041
**`docs/RUNBOOK_training.md` still teaches the removed `--candidate-score` promote path**

- **Priority:** P2 · **Status:** **FIXED in `e5761b1` (promotion section rewritten around `prereg init/check` and `eval --promote --prereg`)** · **Effort:** S
- **Evidence:** Reported by the docs agent: `docs/RUNBOOK_training.md:182,198,204` (a current runbook). `hearth adapters promote` rejects those flags (cli.py, see B-015).
- **Impact:** Following the runbook fails at the promotion step.
- **Fix outline:** Rewrite the promotion section around `hearth eval --promote --prereg ...` (CLAUDE.md §7).
- **Acceptance test:** Every command in the runbook's promotion section runs (or is marked as needing real weights).

### B-042
**`Settings.default_model` is never read**

- **Priority:** P3 · **Status:** **FIXED in `eef3bd5` (dead field removed; the registry is the single source)** · **Effort:** S
- **Evidence:** Checked: no reader of `.default_model` in `src/` (only a docstring in `registry/__init__.py:68`); the registry reads `HEARTH_DEFAULT_MODEL` from `os.environ` directly.
- **Impact:** A dead setting invites a second source of truth.
- **Fix outline:** Remove the field, or make the registry read it from Settings.
- **Acceptance test:** —

### B-043
**`hearth_peek.py` output no longer matches the sample in `RUNBOOK_finance.md` §2**

- **Priority:** P3 · **Status:** **FIXED in `cf3b6e9` (sample refreshed from a synthetic run)** · **Effort:** S
- **Evidence:** Reported by the docs agent.
- **Impact:** Cosmetic doc drift.
- **Fix outline:** Refresh the sample from a synthetic run.
- **Acceptance test:** —

### B-044
**`hearth_peek.py` printed a statement's preamble (cell values) as "header names"**

- **Priority:** P0 (privacy) · **Status:** **FIXED** in `1883397` · **Effort:** S
- **Evidence:** It took `rows[0]` as the header. On the synthetic `examples/finance/statements.csv`
  it printed the `#` preamble text as headers, then announced "No cell values were printed".
  Real exports put account name/number above the header; the output is meant to be pasted to a
  cloud agent. Also, unexpected-error reasons echoed the exception message (can quote content).
- **Fix:** real header detection (modal width + every cell label-like); preamble counted, never
  printed; no confident header → no names; error type only. Tests plant a marker in every
  non-header cell and assert none appears in the output.
  **Superseded:** that label heuristic still printed values (B-063); replaced by an allowlist
  in `a4ad61c`.

### B-045
**An unedited `prereg init` template, or `must_beat_baselines: []`, could gate a promotion**

- **Priority:** P1 (promotion gate) · **Status:** **FIXED** in `9165034` · **Effort:** S
- **Evidence:** `training/prereg.py` required only task/golden_sha/metric; blank hypothesis /
  stopping_rule / kill_condition loaded, and `[]` baselines loaded as `()` (measured).
- **Fix:** both refused at load; defaults may be extended, never dropped.

### B-046
**`hearth train` help and output steer users to `adapters promote` without `--report/--prereg`**

- **Priority:** P3 · **Status:** **FIXED (runtime message)** in `21b30d6` — the post-train
  message names `hearth eval <id> --golden <set> --prereg <committed prereg> --promote`, or
  `--report-json` then `adapters promote <id> --report <file> --prereg <prereg>`. The `train`
  docstring (~"can be promoted (`hearth adapters promote`)") is left to the cli.py help pass ·
  **Effort:** S
- **Evidence:** Reported by the docs-hygiene agent: `src/hearth/cli.py` train docstring (~815)
  and the post-train message (~879) say "eval it, then `hearth adapters promote`" with no
  mention of the report/prereg the command requires.
- **Fix outline:** Point at `hearth eval ... --prereg ... --promote` (or `--report-json` then
  `adapters promote --report --prereg`). Do with the cli.py help pass.

### B-047
**A bogus `HEARTH_DEFAULT_MODEL`: `/ready` says failed, but `model=auto` is still served by the catalog default**

- **Priority:** P1 · **Status:** **FIXED** in `45dc5e1` — `Registry.require_default()`; `serve`,
  `run`, `agent`, `mcp` and `rag query --answer` refuse to start (exit 2) naming the bad id and
  the registered ids; plain `doctor` FAILs (fatal); `doctor --offline` shows it as a non-fatal
  WARN row (a refusal to start is not an egress path), verdict unchanged. Unset → catalog
  default. The gateway half followed in `7e6e12e`: `create_app()` itself calls
  `registry.require_default()` (`gateway/app.py:94`) and raises `UnregisteredDefaultModelError`
  (re-measured in the CLI-polish batch) · **Effort:** S
- **Evidence:** Confirmed live by the model-selection agent (in-process, real weights): with
  `HEARTH_DEFAULT_MODEL=bogus/no-such-model`, `/ready` → 503 `failed`, yet a `model=auto` chat
  is answered 200 by Coder-7B and `/admin/health` reports the 7B (`registry/__init__.py:59`
  falls back; B-029 added a warning only).
- **Impact:** Two parts of the server disagree about what is serving — the CLAUDE.md §3 shape.
- **Fix outline:** Treat an explicitly set but unregistered default like a missing named
  routing profile (B-008): refuse to start `serve`/`run`/`agent` with a clear message (exit 2);
  doctor FAIL. Keep the catalog default only when the variable is unset.
- **Acceptance test:** with the variable bogus, `serve` exits 2 naming it; revert → test fails.

### B-048
**`/ready` turns 503 when the default model is evicted to make room for another**

- **Priority:** P2 · **Status:** **FIXED** in `927b686` (ready = loaded once + on disk; residency reported as `loaded` / `resident`) · **Effort:** S
- **Evidence:** `src/hearth/gateway/app.py` `_weights_loaded` (~426) requires the default to be
  resident; LRU eviction under `ram_ceiling_gb` (observed live under a 12 GB ceiling) unloads it.
- **Impact:** A healthy server reports not-ready; orchestrators gating on `/ready` pull it.
- **Fix outline:** Ready = the default has loaded successfully once and can be reloaded (its
  weights resolve on disk); report residency separately (e.g. in the body or admin/models).
- **Acceptance test:** after evicting the default, `/ready` stays 200 with a "not resident,
  reloads on demand" detail; a default whose weights were deleted goes 503.

### B-049
**The finance ladder example computes money with `float`**

- **Priority:** P1 (CLAUDE.md §4: "Decimal in Python, never a float") · **Status:** **FIXED** in
  `9d9177e` — parsed by `hearth.finance.parse_money`, summed/compared/formatted as Decimal; an
  AST guard test fails on any `float(` in the file · **Effort:** S
- **Evidence:** Checked: `examples/finance/run_finance_ladder.py` `amount: float`, `float(r["amount"])`,
  and income/spend/net/by_category/largest totals all float (lines ~88, 111-117, 127, 236).
- **Impact:** The shipped example teaches the exact practice the finance package forbids; float
  sums drift (e.g. 0.1+0.2).
- **Fix outline:** Parse with `Decimal`, sum Decimals, format with quantize; reuse
  `hearth.finance` parsing if it fits.
- **Acceptance test:** amounts that sum exactly in Decimal but not in float (e.g. 0.10 + 0.20)
  produce the exact total; a test fails if `float(` reappears in money paths.

### B-050
**`--help` prose is ragged at an 80-column terminal**

- **Priority:** P3 · **Status:** **FIXED** in `9fd539c` · **Effort:** S
- **Evidence:** Typer/rich keeps every single newline after a help text's summary paragraph,
  so the ~90-column source lines wrapped again at 80 columns: `hearth --help` read
  "…127.0.0.1:8080, with / a / chat page (/chat)…" with one-word orphan lines. Examples over
  78 columns wrapped mid-command.
- **Fix:** `cli.py:ReflowGroup` reflows every command's help once (unindented lines join;
  indented example/command lines and list items stay verbatim, the convention
  `scripts/gen_manpage.py` already reads). Long examples use `\` continuations.
- **Acceptance test:** `tests/test_cli_help.py` at 80 columns: greedy-wrap invariant over every
  page, and each indented docstring line renders as one line. Typer reads `TERMINAL_WIDTH`
  once at import, so the test patches `typer.rich_utils.MAX_WIDTH` (without it the check ran
  at 300 columns and passed vacuously).

### B-051
**`hearth stats` hides `failed` / `failure_rate`, and labels `escalations_failed` "(served local)"**

- **Priority:** P2 · **Status:** **FIXED** in `6eb4fac` · **Effort:** S
- **Evidence:** `cli.py` `stats` printed neither rollup key B-003 added, and the
  `escalations_failed` count includes records whose local fallback also failed
  (`RequestRecord.escalation_failed` and `failed` both set): nothing was served there.
- **Fix:** rows "failed (error, no answer)", "failure rate", and "escalations failed (remote
  errored; prompt may have left)".
- **Acceptance test:** `tests/test_cli_stats.py` renders a real `MetricsStore` with a double
  failure and asserts the rows; "served local" never appears.

### B-052
**`hearth train` with a 1-record dataset ends in a traceback**

- **Priority:** P3 · **Status:** **FIXED** in `eff5a9c` · **Effort:** S
- **Evidence:** `LoRAConfig.validate` raises `DatasetError` ("need at least 2 records…"), a
  `ValueError`; `train` caught only `RuntimeError` / `CalledProcessError`.
- **Fix:** validate before announcing the run; `Dataset error: …` / `Invalid training config:
  …`, exit 1; the runner's batch-size `DatasetError` is caught too.
- **Acceptance test:** `tests/test_cli_training.py::test_train_with_one_record_is_a_clean_dataset_error`.

### B-053
**`hearth serve` printed "Serving on http://…" before `create_app` raised**

- **Priority:** P3 · **Status:** **FIXED** in `51b9a30` (B-033 moved `create_app` above the
  banner); re-checked on the merged code and pinned by tests in `ff192b5` · **Effort:** S
- **Acceptance test:** `tests/test_cli_startup_errors.py`: missing profile or a raising
  `create_app` → no banner, uvicorn never called; a healthy start prints it.

### B-054
**`ModelNotOnDiskError` tells you to `hearth models pull` a path or an unregistered repo id**

- **Priority:** P2 · **Status:** **FIXED** in `5556f5d` · **Effort:** S
- **Evidence:** `providers/mlx.py:resolve_local_model` always said "`hearth models pull
  <id>`", but `models pull` accepts registry ids only ("Unknown model id", exit 1).
- **Fix:** path → "no such path"; registry entry (by id or source) → pull its registry id;
  unregistered repo id → register in `config/models.yaml` then pull, or
  `HEARTH_ALLOW_DOWNLOADS=1`.
- **Acceptance test:** `tests/test_offline_model_resolution.py` (three cases, zero connects).

### B-055
**MLX embedder load error: a doubled period and sandbox-only advice**

- **Priority:** P3 · **Status:** **FIXED** in `1d4da3f` · **Effort:** S
- **Evidence:** `memory/embed.py` rendered "…HEARTH_ALLOW_DOWNLOADS=1.. Pre-pull it from an
  unrestricted terminal (network is blocked here)."
- **Fix:** one period, the resolver's own fix (or "`hearth models pull <registry id>`"), and a
  B-011 note: the bge embedder cannot load under mlx-lm even on disk.
- **Acceptance test:** `tests/test_mlx_embedder.py` (not-on-disk and "Model type bert not
  supported." cases).

### B-056
**`rag query` embeds the query before checking that the collection has chunks**

- **Priority:** P2 · **Status:** **FIXED** in `0f72c06` · **Effort:** S
- **Evidence:** `RagIndex.query` embedded first, so with `HEARTH_EMBEDDER=mlx` (B-011) an empty
  collection was "Embedder unavailable", exit 1, not "No chunks", exit 0.
- **Fix:** `store.count(collection) == 0` → no chunks, no embedding (library: HTTP and MCP
  too); the CLI checks first and skips `--answer` generation.
- **Acceptance test:** `tests/test_cli_startup_errors.py` (empty collection with an unloadable
  embedder; `RagIndex` with an embedder that raises on use).

### B-057
**`serve`'s stderr log handler kept a stale stream, and any foreign handler suppressed it**

- **Priority:** P3 · **Status:** **FIXED** in `258ff98` · **Effort:** S
- **Evidence:** `cli.py:_log_hearth_to_stderr` added a handler only `if not log.handlers`: a
  repeat call kept the first handler (bound to a possibly closed stderr, the "--- Logging
  error ---" trap), and a handler attached by anything else meant INFO was never enabled.
- **Fix:** a tagged handler, replaced on each call, on the current stderr; level INFO.
- **Acceptance test:** `tests/test_cli_log_handler.py`.

### B-058
**`doctor --offline` `serving_resolution` said `~/.hearth/models` whatever `HEARTH_HOME` was**

- **Priority:** P3 · **Status:** **FIXED** in `f014ff6` · **Effort:** S
- **Evidence:** hardcoded text in `status/probes.py:_serving_load_fact`.
- **Fix:** `ModelNotOnDiskError.models_dir` carries the directory the resolver searched; the
  row prints it (falls back to `$HEARTH_HOME/models`).
- **Acceptance test:** `tests/test_status_probes.py::test_serving_resolution_names_the_models_dir_the_resolver_searched`.

### B-059
**An unknown `HEARTH_BACKEND` ends every command in a traceback**

- **Priority:** P3 · **Status:** **FIXED** in `3951d2e` · **Effort:** S
- **Evidence:** Measured: `HEARTH_BACKEND=bogus hearth run hi` → a Rich traceback ending
  `ValueError: Unknown HEARTH_BACKEND: 'bogus' (use auto|mlx|echo, or install a plugin …)`,
  exit 1. Raised at `src/hearth/providers/__init__.py:99` (`select_provider`); no CLI command
  that calls it (`serve`, `run`, `agent`, `mcp`, `rag query`, `eval`) catches it.
- **Impact:** a typo in one variable buries the one-line fix, as B-033 did for profiles.
- **Fix outline:** a `_backend_required()` context manager like `_routing_profile_required`
  around `select_provider` in each command: print the message, exit 2.
- **Acceptance test:** each command with `HEARTH_BACKEND=bogus` → exit 2, the message, no
  `Traceback`; revert → fails.

### B-060
**An unknown `hearth.intent` / `--intent` is silently ignored**

- **Priority:** P3 · **Status:** **FIXED** in `3951d2e` · **Effort:** S
- **Evidence:** `src/hearth/router/classify.py:58` uses the intent only if it is in
  `TASK_CLASSES`, else falls through to keyword rules. Measured [in-process]:
  `POST /v1/hearth/route` with `"intent": "bogus"` → 200 `{"class": "chat", "method":
  "rules", …}`; `hearth run --intent bogus` prints `intent=bogus` and serves class `chat`.
  docs/GUIDE.md §4.6 says the intent "must be one of" the classes.
- **Impact:** a typo'd hint (e.g. `"clasify"`) silently routes by keywords instead, possibly to
  a different model rung, while the caller believes the hint applied. CLAUDE.md §3 shape.
- **Fix outline:** reject an unknown intent: 400 `invalid_request_error` (`param:
  hearth.intent`) over HTTP, exit 2 at the CLI, naming the valid classes.
- **Acceptance test:** an unknown intent → 400 / exit 2; a valid one still routes with
  `method: "intent"`; revert → fails.

### B-061
**`hearth adapters promote --report` accepts a hand-written report: nothing ties it to the adapter, and a prereg committed seconds earlier in any repo passes**

- **Priority:** P0 · **Status:** **FIXED** in `13038c5` · **Effort:** S–M
- **Evidence:** cli.py adapters promote (~1875-1990) recomputes the gate from the report's per_example vectors but never checks the report's candidate/task against ADAPTER_ID's entry; prereg.py verify_committed (~209) accepts any repo and records rev-parse HEAD, not the introducing commit. Reviewer repro promoted `bogus-ad` (no golden set, no model run, task mismatch, adapter_path=/nonexistent) → `gate: verified`. Mutations deleting the mismatches/verify_committed checks in this command fail 0 tests. (adversarial review 2026-10-05, 13b1438..69aa06e.)
- **Fix:** `hearth eval --report-json` HMAC-signs the report with a per-install key
  (`HEARTH_HOME/eval-report.key`, 0600, `training/attest.py`); `adapters promote` verifies it
  first. The report records candidate id, task, base model, adapter path, a SHA-256 of the
  adapter weights (hashed before scoring; `eval` refuses an adapter with no weights), the
  incumbent and its weights hash, `measured_at` and the golden set's git status;
  `training/promotion.report_problems` checks each against the registry and disk now
  (incl. "the incumbent is still the incumbent"). Both promotion paths require the prereg's
  last-changing commit to be no later than the measurement, in the git repo holding the
  committed golden set (`prereg.check_provenance`); the proof records that commit (not
  HEAD), its time, the introducing commit, the weights hash and the report sha.
  Residual: a user who reads the key can forge a MAC (they can also edit `adapters.json`);
  committer timestamps are settable, so a deliberate backdate is not caught.
- **Verified:** the reviewer repro → `Unusable eval report: no report-signing key`, exit 1,
  still a candidate. `tests/test_promotion_evidence.py` (31 attacks) promoted on the pre-fix
  tree; every guard mutation-killed.

### B-062
**Prereg `bar` is not range-checked: NaN alpha/min_effect or alpha=1, min_effect<0, min_n=1 disables every gate clause**

- **Priority:** P0 · **Status:** **FIXED** in `a731e8c` · **Effort:** S–M
- **Evidence:** prereg.py (~198-201); eval.py comparisons (~513, 549, 557) are False under NaN; negative margin makes the baseline clause vacuous. Reviewer: a candidate at 0.033 vs base 1.0 PASSES with reasons=() — defeats B-045, the n≥5 floor and min_n=30 (CLAUDE.md §7). Works through `hearth eval --promote` too. (adversarial review 2026-10-05, 13b1438..69aa06e.)
- **Fix:** `load_prereg` refuses (no coercion) a non-finite number, `alpha` outside
  (0, 0.05], `min_effect` < 0, `min_n` not an integer ≥ 30, an unknown `test`.
  `evaluate_gate` re-checks the bar (`check_bar`; its min_n floor is `min_n_for_alpha`, 5 at
  0.05), refuses a report with a non-finite/out-of-range score, a `score` that is not the
  mean of its vector, or a baseline from another golden set, and writes every comparison
  so a NaN adds a refusal. `prereg init` will not scaffold a bar `load_prereg` refuses.
- **Verified:** `/tmp/hr/nan.py` → both bars now raise `PreRegError`; 50 new tests failed
  on the pre-fix code; every guard mutation-killed; the replay test still refuses.

### B-063
**`hearth_peek.py` still prints cell values: a full-width all-text preamble row, a headerless all-text file, or JSON keys that are values pass `_looks_like_label`**

- **Priority:** P0 · **Status:** **FIXED** in `a4ad61c` · **Effort:** S–M
- **Fix:** no heuristic left. A header cell prints only if every word is in a fixed column
  vocabulary (`VOCAB`; no digits), else `(withheld)`; header row = first row with ≥2 such
  names (JSON: the key row); preamble counted as `skip_rows`, never printed; files print as
  ids `F1..Fn` (`--index-out` writes the id→path list locally); refusals print fixed reasons.
  Tests: the reviewer's three files (`tests/fixtures/peek/`) and a property test over 150
  random tables (no random token or 3+ digit run may appear); 6 mutants killed.
- **Evidence:** Reviewer: printed "Jane Q Public", "Premier Checking", "SECRETMERCHANT ONE" then "No cell values were printed". Heuristic label detection cannot distinguish text values from labels; tests only used rows containing a date and an amount. File names/paths also printed verbatim (may carry account numbers). (adversarial review 2026-10-05, 13b1438..69aa06e.)

### B-064
**`/ready` judges only the registry default; under a routing ladder the default may never serve**

- **Priority:** P1 · **Status:** **FIXED** in `9645883` · **Effort:** S–M
- **Fix:** `/ready` judges every model `router.route.policy_rungs` says an `auto` request can
  reach (each class rung, else `defaults.local_model`, else the registry default only if some
  class falls through), with the existing outcome rules; the body carries per-model `models`
  (`status`, `loaded`, `serves`, `reason`/`detail`). Warmup loads the most-used rung first, then
  each further rung that fits under the ceiling without evicting. Measured [in-process, fake
  weights]: finance profile with the 14B absent → `503 failed` naming the 14B, `models[3B]`
  ready, and the auto chat really 503s. Finding filed as B-075 (unpinned `embed` class).
- **Evidence:** app.py readiness (~185-249) vs Router._local_model (route.py ~439-462). Finance profile with the 14B absent: /ready 200 while model=auto → 503 not on disk; warmup loads a 7B the profile never serves. (adversarial review 2026-10-05, 13b1438..69aa06e.)

### B-065
**Routing validation gaps: `defaults.local_model` is never validated; a class rung may name an embed model or echo**

- **Priority:** P1 · **Status:** **FIXED** in `ed923e1` · **Effort:** S–M
- **Fix:** `defaults.local_model` and every class rung must be a registered chat model
  (`load_policy`) servable by the active backend (`check_policy_servable`, run by `Router()`
  for a file-loaded policy and by `create_app` always). A violation raises
  `RoutingPolicyError` (a `RoutingProfileNotFoundError` subclass, so the CLI exits 2) and is
  NOT degraded to the safe defaults — the fallback silently replaced the profile's ladder. A
  structurally broken file still falls back per ADR-005 (see "Remains").
- **Remains:** an unparseable or structurally invalid *selected* profile still degrades to the
  safe defaults with only a log line — the same shape as B-008 (no egress, but a different
  ladder than the operator selected). Left as ADR-005 behaviour; deciding it touches doctor
  and status, which read the fallback. CLI wording is B-076; status wording is B-077.
- **Evidence:** policy.py (~210-213, ~234). Typo'd defaults.local_model → policy loads, /ready 200, model=auto → 404; embed/echo rung → 404 at request time. (adversarial review 2026-10-05, 13b1438..69aa06e.)

### B-066
**Failure accounting gaps: an unservable ladder rung (404 / stream error event) writes no RequestRecord; that stream branch's [DONE] is untested; BudgetExhausted is never recorded**

- **Priority:** P1 · **Status:** **FIXED** in `03f79b9` · **Effort:** S–M
- **Fix:** `Router.route` records an UnknownModelError from the local rung, and from a failed
  escalation's local fallback (with `escalation_failed`), before re-raising;
  `Router.provider_for` (shared with the stream) records a denied escalation (`failed`,
  `served_by: remote`, `escalated: false`) before raising BudgetExhaustedError; the stream's
  `model_not_found` branch records. Tests assert that branch's own error event then `[DONE]`;
  deleting that `[DONE]` now fails 2 tests.
- **Evidence:** route.py ~415-418 re-raises UnknownModelError around record_failure; app.py ~871-884 no record; mutations deleting [DONE] in that branch fail 0 tests. A remote failure then local UnknownModelError leaves the escalation unrecorded (PLAUSIBLE). (adversarial review 2026-10-05, 13b1438..69aa06e.)

### B-067
**`doctor --offline` says SAFE with a plugin embedder or vector store, which receive every RAG chunk and query**

- **Priority:** P1 · **Status:** **FIXED** in `74e5943` · **Effort:** S–M
- **Fix:** fatal `embedder` / `vector_store` rows judge the exact type returned by the real
  `select_embedder` / `select_vector_store`: built-ins pass; a plugin (even a subclass of a
  built-in) or an unresolvable name FAILs. The reviewer's command is now UNSAFE, exit 1.
- **Evidence:** doctor.py FAILs a plugin backend (~398-411) but not HEARTH_EMBEDDER / HEARTH_VECTOR_STORE plugins. Reviewer: `HEARTH_EMBEDDER=evil-cloud-embedder HEARTH_VECTOR_STORE=evil-store hearth doctor --offline` → SAFE. (adversarial review 2026-10-05, 13b1438..69aa06e.)

### B-068
**The auto→echo fallback stub labels its echo with the requested real model and credits token savings**

- **Priority:** P2 · **Status:** **FIXED** in `d5a7055` · **Effort:** S–M
- **Fix:** the echo stub reports `model: "echo"` (result and stream terminal delta) whatever
  was requested, and `is_stub_backend` zeroes `estimated_frontier_tokens_saved` on the plain
  and stream paths. Same for an explicit `HEARTH_BACKEND=echo`.
- **Evidence:** echo.py ~40, ~59 report req.model. /ready is 503 (correct) but POST model=Qwen-14B → 200 `model: ...14B`, text "[echo] hi", metrics credit estimated_frontier_tokens_saved. (adversarial review 2026-10-05, 13b1438..69aa06e.)

### B-069
**LoRA adapter variants are full base reloads the ModelManager never counts**

- **Priority:** P2 · **Status:** **FIXED** in `e8fcb8a` · **Effort:** S–M
- **Fix:** each (model, adapter) variant is its own ModelManager resident
  (`<id>@adapter:<path>`), sized at the model's full `ram_gb`, LRU-evicted and refused like any
  model; `/admin/models` shows each resident's `adapter`. Measured: base + 3 adapters → 4 loads
  and `resident_ram_gb` 36.0 (was 9.0). Base weights are deliberately not shared: mlx_lm
  0.29.1's `remove_lora_layers` restores only plain `LoRALinear` (not DoRA / embedding /
  switch layers) and a `full` adapter overwrites base weights, so an in-place swap could leave
  the previous adapter applied.
- **Evidence:** MLXProvider._load_variant (mlx.py ~335-361) loads base+adapter per variant into _cache; 14B + 3 adapters ≈ 36 GB real vs resident_ram_gb 9.0 — above the 24 GB ceiling and the 30.15 GB working set. (adversarial review 2026-10-05, 13b1438..69aa06e.)

### B-070
**B-047 enforced at CLI call sites, not where the model is chosen**

- **Priority:** P2 · **Status:** **FIXED** — eval half in `58890fa`, serving side in `36157af` · **Effort:** S–M
- **Evidence:** mlx_pool().resolve("") with a bogus HEARTH_DEFAULT_MODEL → catalog default (warning only). Not gated: `hearth eval`, example scripts, direct API users; a mutation deleting eval's `_require_known_model` fails 0 tests. (adversarial review 2026-10-05, 13b1438..69aa06e.)
- **Eval half (fixed):** `hearth eval` exits 2 on an unregistered HEARTH_DEFAULT_MODEL
  (`require_default`), on an empty/`auto` base model, and on a base the registry cannot
  serve for any provider (not only a ModelPool); mutations deleting each check now fail
  tests (`tests/test_cli_eval.py`).
- **Serving side (fixed, `36157af`):** `serving.pool.resolve_model_id` (so `ModelPool.resolve`
  of `""`/`"auto"`/`None`) and `Router._local_model`'s fall-through both use
  `require_default()`, so the code that chooses the model raises
  `UnregisteredDefaultModelError` for every entry point (examples, direct API users), with no
  load performed.

### B-071
**An abandoned agent run keeps the single MLX thread busy for up to its budget**

- **Priority:** P2 · **Status:** **FIXED** in `1a872a2` · **Effort:** S–M
- **Fix:** the agent body is wrapped in `_close_on_disconnect(on_close=cancel.set)`; the run
  executes under `providers.base.cancel_scope`, which the MLX thread inherits (jobs run in a
  copy of the caller's contextvars) and checks per token; the loop checks the flag before every
  step (new stop reason `cancelled`); the router neither retries nor 503s a cancellation and
  records it failed. A `: keepalive` SSE comment every second is what lets Starlette notice the
  disconnect mid-step (in both ASGI modes it learns only between chunks). Tested through
  Starlette's own `StreamingResponse.__call__` with held generators.
- **Evidence:** agent_route.py wraps with _guarantee_done but not _close_on_disconnect, and closing would not stop work(); chat queues behind it (PLAUSIBLE). (adversarial review 2026-10-05, 13b1438..69aa06e.)

### B-072
**ModelManager evicts residents before knowing the new load will succeed**

- **Priority:** P3 · **Status:** **FIXED** in `7c9e510` · **Effort:** S–M
- **Fix:** `_admit` runs the provider's `preflight` (MLX: weights — and, since `e8fcb8a`, the
  adapter path — resolve on disk, nothing loaded) after the ceiling check and before evicting;
  a missing model leaves every resident loaded.
- **Remains (by decision):** a load that passes preflight and then fails (corrupt checkpoint)
  has already freed its victims; restoring them would overshoot the ceiling on every load or
  cost a second reload. They stay `loaded_once` with weights on disk and reload on demand.
- **Evidence:** manager.py _admit (~145-149): a request for a registered-but-not-pulled model evicts working residents, then fails (PLAUSIBLE). (adversarial review 2026-10-05, 13b1438..69aa06e.)

### B-073
**Low: finance ladder example resolves HEARTH_ROUTING_YAML itself and silently falls back; config embed_model default is the 404 id; failed record `adapter` field differs by path**

- **Priority:** P3 · **Status:** **FIXED** — 2 of 3 in `d4eb825` (the example resolves
  `HEARTH_ROUTING_YAML` with `resolve_routing_selection` and exits 2 on a missing named
  profile; `embed_model` defaults to the registered `-bf16` id); the failed record's `adapter`
  field in `dd4b774`: on every failure path it is the adapter SELECTED for the attempt that
  failed (explicit request, else the promoted default), documented on `RequestRecord.adapter`
  · **Effort:** S–M
- **Evidence:** run_finance_ladder.py ~400-402 (B-008 bypass); config.py ~46; route.py ~266 vs app.py ~889. (adversarial review 2026-10-05, 13b1438..69aa06e.)

### B-074
**`hearth_map_draft.py` printed a preamble as column names, drafted `skip_rows: 0` for every file with a preamble, printed file names, and printed a value-derived total on request**

- **Priority:** P0 (privacy) + P1 (correctness) · **Status:** **FIXED** in `052d3af` · **Effort:** M
- **Evidence:** it took `rows[0]` as the header (grouping and profiling) and printed raw names
  in `ColumnProfile.describe()`. Integrator, `--no-model`, synthetic file whose first rows were
  `Account Holder,Jane Q Public,Premier Checking` and `Acct 4417123412341234,Open,x` above
  `Date,Description,Amount`: printed `'Jane Q Public'` and `'Premier Checking'` as columns,
  then "No cell value was printed above." On `examples/finance/statements.csv` it printed the
  `#` comment text as column names and drafted nothing usable. `skip_rows` was hard-coded 0,
  so every draft for a file with a preamble was wrong. Refusals and parse failures printed
  `path.name`; `--show-total` printed the trial-parse sum; confirm items and verification
  details interpolated raw header names (and a `MappingError` message that lists the whole
  header).
- **Fix:** the peek rule moved into `src/hearth/finance/shape.py` and is used by both scripts
  (one implementation). map_draft finds the header with it (first row in 30 with >=2
  vocabulary names; JSON key row), writes the rows above as `skip_rows`, groups by (header
  row, header), and refuses a file with no identifiable header. Terminal: column names only
  from the vocabulary, else `column N (withheld)`, through `Note` slots for every sentence
  that names a column; files as `F1..Fn` (`--index-out` local); fixed reasons instead of
  exception messages; the model-proposed draft file name prints only if it is vocabulary; no
  counts of negatives/zeros, only yes/no. `--show-total` is refused (exit 2): a sum of the
  amounts is a value-derived figure, and CLAUDE.md §4 says values are not safe; the sum stays
  in the local draft. The closing lines say exactly what was printed and that the draft file
  holds real header names, file names and the sum. Also fixed: a repeated header name crashed
  the balance check (IndexError); such columns are now `repeated-name` and take no role.
- **Acceptance test:** `tests/test_map_draft_privacy.py`: the integrator's file (no
  Jane/Public/Premier/4417/file-name digits/3+ digit run in stdout; draft has `skip_rows: 2`,
  the real names, and parses with `parse_rows`), a hostile fake local model, failure paths,
  and a property test over 120 random tables (half with an echoing fake model). 19 mutants
  killed (each guard reverted → a test fails).
- **Remains:** a preamble row holding two vocabulary names (e.g. `Account Type,Checking,Account
  Number,...`) would be taken as the header. Privacy holds (only vocabulary prints) and the
  trial parse refuses the draft, but the operator then writes that mapping by hand. A header
  with fewer than two vocabulary names (e.g. a non-English export) is refused, not drafted.

### B-075
**`config/routing.finance.yaml` does not pin the `embed` class: a chat with `intent: embed` reaches the registry default 7B**

- **Priority:** P3 · **Status:** **FIXED** in `c3cb201` · **Effort:** S
- **Evidence:** `TASK_CLASSES` includes `embed` (router/classify.py:14-24) and an explicit
  intent of `embed` is accepted; the finance profile pins the other 8 classes only, so
  `Router.decide(..., intent="embed")` → `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`
  (asserted in tests/test_ready_routing_ladder.py). Since B-064's fix readiness therefore
  judges, and warmup loads, the 7B too — honest, but probably not the ladder's intent.
- **Fix outline:** pin `embed` to a ladder rung in the profile (config owner), or have the
  chat router refuse `embed` (embeddings go through /v1/embeddings).

### B-076
**The CLI says "Routing profile not found:" for a profile that exists but names an unservable rung**

- **Priority:** P3 · **Status:** **FIXED** in `c3cb201` · **Effort:** S
- **Evidence:** B-065's `RoutingPolicyError` subclasses `RoutingProfileNotFoundError` so
  `cli._routing_profile_required` (src/hearth/cli.py ~614-629) catches it and exits 2 — but
  prints the fixed prefix "Routing profile not found:" before the (correct) message.
- **Fix outline:** catch `RoutingPolicyError` first with a "Routing profile unusable:" prefix.

### B-077
**Status / `doctor` report "policy loader unavailable" for a profile with a bad model rung**

- **Priority:** P3 · **Status:** **FIXED** in `c3cb201` · **Effort:** S
- **Evidence:** `status/probes.py:_policy_outcome` (~333-337) catches any exception from
  `load_policy` as `{"error": "policy loader unavailable"}`; since B-065 a bad rung raises
  `RoutingPolicyError`, so the routing row FAILs (correct outcome) with a reason that hides
  the real one (which rung, which file).
- **Fix outline:** report `str(exc)` for a `RoutingProfileNotFoundError`.

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
| `48634a5` | **B-038.** `docs/PRIVACY.md` Formats row matches `mcp/files.py`: UTF-8 text suffixes and extension-less, `.csv`, `.json`, `.xlsx` (refuses uncached formulas), `.pdf` (text layer only, refuses scans), the `[files]` extra, and what `read_table` accepts. |
| `46bddeb` | **B-039.** `load_dataset` no longer claims headerless files load. A headerless file is refused with an error naming the missing header (the header is the only place the task lives; `hearth train --task` is not passed in). |
| `0c3b5f9` | **B-040.** `examples/` uses the one-command sync and `uv run --no-sync`. The Claude Code MCP registration is `uv run --no-sync --project <repo> hearth mcp`, so it works from any cwd. |
| `e5761b1` | **B-041.** `docs/RUNBOOK_training.md` promotion is `hearth prereg init/check`, `hearth eval --prereg --report-json`, then `eval --promote` or `adapters promote --report --prereg`. Every no-weights command was run on the echo backend. |
| `eef3bd5` | **B-042.** `Settings.default_model` removed. `Registry.default_id` is the single source of the default model; the status probe lists `HEARTH_DEFAULT_MODEL` as read outside Settings. |
| `cf3b6e9` | **B-043.** `RUNBOOK_finance.md` §2 `hearth_peek.py` sample regenerated from a synthetic CSV. |
| `1883397` | **B-044.** `hearth_peek.py` never prints preamble or value-like cells; output asserted marker-free. |
| `9165034` | **B-045.** A prereg needs written hypothesis/stopping_rule/kill_condition and cannot drop default baselines. |
| `2d186dd` | **B-004, B-005.** ModelPool: the requested model serves (404 for unknown ids); telemetry from the provider that ran; warmup loads and `/ready` reports loading/failed with a reason. |
| `2cf9af6` | **B-031.** `hearth doctor --offline` marks a non-fatal failed row WARN (fatal → FAIL), so the table cannot contradict the SAFE/UNSAFE verdict. |
| `51b9a30` | **B-033.** A missing selected routing profile ends `serve`/`run`/`agent`/`mcp`/`rag query` with "Routing profile not found: ..." and exit 2, no traceback (caught where the router is built). |
| `41bf026` | **B-036.** `rag ingest`/`rag query` with an embedder that cannot load print "Embedder unavailable: ..." and exit 1, no traceback. |
| `21b30d6` | **B-046.** `hearth train`'s post-train message names `eval ... --prereg ... --promote` or `--report-json` + `adapters promote --report --prereg` (runtime message; docstring left to the help pass). |
| `45dc5e1` | **B-047.** An explicitly set, unregistered `HEARTH_DEFAULT_MODEL` makes `serve`/`run`/`agent`/`mcp`/`rag query --answer` refuse to start (exit 2, bad id + registered ids); `doctor` FAIL, `doctor --offline` WARN. `Registry.default_id` stays lenient; `require_default()` is the strict form. |
| `9d9177e` | **B-049.** `examples/finance/run_finance_ladder.py` parses with `hearth.finance.parse_money` and keeps every amount Decimal; tests assert exact totals where float sums drift, plus an AST no-`float(` guard. |
| `48982b0` | **B-003.** A failed request writes a `RequestRecord` with `failed` set (plain local failure, escalation-then-local failure, a remote stream dying mid-answer), on both the plain and streaming paths. Rollup adds `failed` / `failure_rate`; `backend_mix` and latency count served answers only. Client behaviour unchanged. |
| `b6303c6` | **B-007.** Every SSE stream ends with `[DONE]`: post-relay accounting failures still send the final chunk, then `hearth.metrics.unavailable`; any other exception becomes `hearth.stream.internal_error` + `[DONE]`. A metrics failure no longer 500s a served non-streaming answer. |
| `25aae9f` | **B-034.** An explicitly requested unknown/retired adapter is a 404 `adapter_not_found` before anything runs; `hearth.adapter` and the record name the adapter that actually generated (null for base weights, a base retry, a remote, or a backend that ignores adapters). Promoted default unchanged. |
| `927b686` | **B-035, B-048.** `/ready` = the default loaded with weights at least once, its last load did not fail, and its weights resolve on disk; `loaded` reports residency. Warmup off is ready when on disk; an evicted default stays ready; deleted weights are 503 `failed`. |
| `12d5366` | **B-006.** `auto` → echo fallback logs a WARNING and `/ready` is 503 `stub` (with `/health` `backend_fallback`); explicit `HEARTH_BACKEND=echo` stays ready. |
| `9fd539c` | **B-050.** `--help` prose reflows to the terminal width; example lines stay intact (80-column render tests). |
| `6eb4fac` | **B-051.** `hearth stats` shows `failed` / `failure rate`; `escalations_failed` labelled "(remote errored; prompt may have left)". |
| `eff5a9c` | **B-052.** `hearth train` with one record: `Dataset error: need at least 2 records …`, exit 1, no traceback. |
| `ff192b5` | **B-053.** Tests pin that `hearth serve` prints its banner only after `create_app` succeeds (the reorder itself was `51b9a30`). |
| `5556f5d` | **B-054.** `ModelNotOnDiskError` hint per kind of id: path → no such path; registry id → pull; unregistered → register, then pull. |
| `1d4da3f` | **B-055.** MLX embedder load error: one period, the real fix, the B-011 note. |
| `0f72c06` | **B-056.** An empty RAG collection answers "No chunks" without embedding the query (CLI, HTTP, MCP). |
| `258ff98` | **B-057.** `serve`'s stderr log handler is idempotent, on the current stderr, and not suppressed by foreign handlers. |
| `f014ff6` | **B-058.** `doctor --offline` `serving_resolution` names the models dir the resolver searched. |
| `2c8f5f9` | **B-037.** `docs/API.md` matches the app's routes and error envelopes; `tests/test_api_doc_routes.py` checks both ways. |
| `05c4db5` | Docs: `docs/GUIDE.md` has no pending-change markers left; B-003/006/031/033–036/046–049 described as merged, with measured output. |
| `3951d2e` | **B-059, B-060.** Unknown HEARTH_BACKEND → exit 2 / doctor FAIL; unknown intent → 422 / exit 2 / UnknownIntentError. |
| `a4ad61c` | **B-063.** `hearth_peek.py` prints only allowlisted column names, counts, type guesses and file ids; no heuristic, no file names; property-tested. |
| `74e5943` | **B-067.** `doctor --offline` FAILs a plugin (or unresolvable) embedder / vector store, judged on the type the real selector returns. |
| `d4eb825` | **B-073 (2 of 3).** Finance example uses the shared routing resolver and exits on a missing profile; `embed_model` default is the registered `-bf16` id. |
| `a731e8c` | **B-062.** Prereg bar range-checked at load (finite; alpha ∈ (0, 0.05]; min_effect ≥ 0; min_n ≥ 30; known test); `evaluate_gate` re-checks it and fails closed on non-finite input. |
| `13038c5` | **B-061.** `adapters promote --report` needs an HMAC-signed report from `hearth eval` on this install, about this adapter (id, task, base, weights hash, incumbent), with a prereg committed before the measurement in the golden set's repo. |
| `58890fa` | **B-070 (eval half).** `hearth eval` exits 2 on an unregistered HEARTH_DEFAULT_MODEL or an empty/`auto`/unservable base; serving side still open. |
| `052d3af` | **B-074.** `hearth_map_draft.py` finds the real header (shared `hearth.finance.shape` rule with peek), drafts `skip_rows`, prints only vocabulary column names / file ids / fixed reasons; `--show-total` refused; property-tested, 19 mutants killed. |
| `ed923e1` | **B-065.** A routing rung (`defaults.local_model` or a class `local_model`) that is unregistered, not chat, or not servable by the active backend raises `RoutingPolicyError` at load / router build / `create_app` instead of falling back. |
| `9645883` | **B-064.** `/ready` judges every model the routing profile can route `auto` to, reports per-model `models`; warmup loads the most-used rung first and never evicts what it warmed. |
| `03f79b9` | **B-066.** Unservable-rung 404s, denied escalations (429) and a failed escalation's local 404 are recorded (plain and stream); the stream branch's `[DONE]` is tested. |
| `dd4b774` | **B-073 (adapter field).** A failed record's `adapter` is the adapter selected for the failed attempt on every path. |
| `d5a7055` | **B-068.** The echo stub reports `model: "echo"` and never credits frontier-token savings. |
| `7c9e510` | **B-072.** ModelManager runs the provider's `preflight` (weights on disk) before evicting anything. |
| `e8fcb8a` | **B-069.** Each LoRA adapter variant is its own counted, LRU-evicted resident at the model's full `ram_gb`. |
| `36157af` | **B-070 (serving side).** `ModelPool.resolve("")`/`("auto")` and the router's default fall-through raise `UnregisteredDefaultModelError`. |
| `1a872a2` | **B-071.** An abandoned `/v1/hearth/agent` stream cancels the run between steps and mid-generation (keepalive + `_close_on_disconnect` + `cancel_scope`). |
| `c3cb201` | **B-075, B-076, B-077.** Finance profile pins `embed` (no 7B warmup); 'Routing profile unusable' label; status probe shows the routing error. |
