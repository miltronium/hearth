# HEARTH — Learning Subsystem: Assessment & Improvement Plan

**Status:** Design/analysis only. Nothing in this document has been implemented.
**Date:** 2026-09-02.
**Scope:** `src/hearth/training/`, `src/hearth/registry/adapters.py`, the eval gate, the
correction-signal loop that does not yet exist, and the router's learnable surfaces.
**Constraint:** every proposal here must run entirely on-box and survive sealed private mode
(`docs/PRIVACY.md`). Nothing proposed requires network egress.

Every claim about existing code cites `file:line`. Every dataset number is an exact count
taken from the files on disk on 2026-09-02.

---

## 0. TL;DR

The Phase 4 stack is a **correct, well-tested skeleton with an honest lifecycle and no
statistical spine**. It can carry a promotion decision from end to end; it cannot yet tell
you whether that decision was justified. Three things are structurally missing, in order of
how much they cost you:

1. **There is no capture.** `RequestRecord` (`src/hearth/observability/metrics.py:49-64`)
   stores counters and no text. Every prompt and every completion HEARTH has ever served is
   gone. The corpus that would make everything else possible does not exist and cannot be
   back-filled.
2. **The gate is an honor system.** `hearth adapters promote --candidate-score 1.0
   --incumbent-score 0.2` (`src/hearth/cli.py:715-758`) constructs an `EvalReport` out of two
   floats you type on the command line (`cli.py:739`) and promotes on that. `AdapterStore.promote`
   takes `gate_passed` as a caller assertion (`registry/adapters.py:117-140`). Nothing verifies
   that the two numbers came from the same golden set, the same metric, the same model, or from
   any measurement at all.
3. **The golden sets total 11 examples across the whole project**, and the eval harness runs at
   `temperature=0.7` (`providers/base.py:31`, never overridden by `cli.py:616-619` or
   `scripts/eval_candidate.py:54-59`) — so it is both underpowered and non-reproducible.

The good news is that all three are cheap to fix relative to their leverage, and two of the
three are pure-Python work with no GPU time.

---

## 1. Honest assessment

### 1.1 What is genuinely solid

These are not consolation prizes — they are the parts that make the rest fixable.

- **The lifecycle is right and it is enforced at the store.** `candidate → promoted → retired`,
  exactly one promoted adapter per task (`registry/adapters.py:141-144`), retired adapters never
  resolve (`adapters.py:190-191`), and candidates are servable behind an explicit A/B flag
  (`adapters.py:181-196`). The A/B flag is the hook that makes online shadow evaluation possible
  later (§2.6) — that was good foresight.
- **Promotion is auditable after the fact.** `promotion_proof` persists to
  `~/.hearth/adapters.json` (`adapters.py:145-146`). The *shape* is right; only the *content* is
  untrustworthy. Fixing §3 means filling this dict with things that can be checked, not inventing
  a new mechanism.
- **The training orchestrator is honestly seeded and deterministic where it can be.**
  `LoRAConfig.seed` flows to mlx-lm (`training/lora.py:52`, `lora.py:152`), the train/valid split
  does not shuffle (`lora.py:110-121`), and dataset serialization is byte-reproducible with
  caller-supplied timestamps (`training/dataset.py:123-132`). This is better discipline than most
  fine-tuning code.
- **`_preflight_batch_size` (`lora.py:158-182`) is exactly the right kind of fix** — it converts
  an opaque downstream failure into an actionable one *before* spending GPU time, and it lives in
  the real runner so the fake-runner tests stay valid. That instinct should be applied to the eval
  path too (§3.5).
- **The dataset header carries provenance and a schema version** (`dataset.py:111-121`) and
  `load_dataset` refuses a version it does not understand (`dataset.py:197-201`). The versioning
  primitive you need for golden-set identity already exists — it is just not applied to golden
  sets.
- **RESULTS.md is unusually honest.** Task A records the gate *refusing* a real candidate
  (`RESULTS.md:57-64`) and names the base-model-already-saturated problem out loud. Finding 2b
  (`RESULTS.md:294-307`) is a real root-cause investigation that rejected the convenient
  explanation. This is the methodology the eval harness needs to catch up to.

### 1.2 What is weak, naive, or gameable

**F1 — `default_judge` is a verbosity contest.** `training/eval.py:152-162`:

```python
return bool(cand) and len(cand) >= len(ref)
```

It returns `True` iff the candidate is non-empty and **at least as long as the reference**. It
never reads `prompt`. It never reads the *content* of either string. A candidate that emits a
fixed 10 KB of whitespace-separated lorem ipsum scores a 1.0 judge win-rate on every example and
passes the gate. The docstring is honest that it is a stub, but two things make it more dangerous
than a stub should be:

- `objective_metric_for()` returns `"judge"` for `draft`, `code`, `reason`, and `chat`
  (`eval.py:147-149`, `_OBJECTIVE_CLASSES` at `eval.py:28`), so the *documented default* for four
  of nine task classes routes to a length heuristic.
- In `score_candidate`, **a supplied judge silently overrides the `metric` argument**
  (`eval.py:105-109`): `if judge is not None:` comes first, so
  `score_candidate(golden, gen, metric="exact", judge=default_judge)` ignores `metric="exact"`
  entirely and reports `metric="judge_win_rate"`. A caller who passes both gets the length
  heuristic and no warning.

Minimum fix regardless of anything else in this document: make `default_judge` **raise**
`NotImplementedError` rather than return a boolean, and make `score_candidate` raise when both
`metric` and `judge` are supplied. A gate that cannot run is safer than a gate that always passes.

**F2 — `beats_incumbent` has a wide-open first-promotion path.** `eval.py:130-144`:

```python
if incumbent is None:
    return candidate.score > 0.0
```

The **first** adapter for any task is promoted on *any* non-zero score. Combined with `token_f1`
(`eval.py:70-87`), which awards partial credit for any shared whitespace token, an adapter that
emits the single word `"the"` on every prompt earns a non-zero F1 against most English references
and is promoted. There is no absolute floor, no comparison against the *base model* when there is
no incumbent, and no minimum-n requirement. The base model is the natural floor and it is already
computed everywhere else in the codebase (`scripts/eval_candidate.py:66-67` scores base explicitly;
`cli.py:629-636` only scores an incumbent adapter, never base). The fix is one line of intent: when
there is no incumbent, **the base model is the incumbent**.

**F3 — the gate compares two bare floats with no shared provenance.** `EvalReport`
(`eval.py:50-62`) carries `task`, `metric`, `score`, `per_example`, `n`. It does **not** carry:
which golden set, that golden set's content hash, the base model, the adapter id, `max_tokens`,
`temperature`, a seed, or a timestamp. `beats_incumbent` therefore cannot detect that it is
comparing a candidate scored on a 6-example set at `max_tokens=24` against an incumbent scored on
a 5-example set at `max_tokens=64`. And `cli.py:739` fabricates one wholesale:

```python
candidate = EvalReport(task="", metric="cli", score=candidate_score)
```

`task=""`, `metric="cli"`. This is the promotion path that `RESULTS.md:90-94` actually used. The
gate as shipped is: *type a number you like.*

**F4 — the eval harness is stochastic and unseeded.** `GenRequest.temperature` defaults to `0.7`
(`providers/base.py:31`). `hearth eval` builds its request at `cli.py:616-618` **without passing
temperature**, so it inherits 0.7. `scripts/eval_candidate.py:54-59` does the same. There is no
seed on the generation path at all. Consequences:

- Re-running `hearth eval` on the *same* adapter and the *same* golden set can produce a different
  score, so `beats_incumbent` near the margin is a coin flip you can re-roll until it passes. On a
  5-example exact-match set the score granularity is 0.2, so a single sampled token flip moves the
  score 20 points.
- `scripts/eval_candidate.py` generates each example **four times** — twice inside `score_candidate`
  (lines 66-68) and twice again for the per-example printout (lines 72-73) — so at temperature 0.7
  **the strings printed in the per-example table are not the strings that were scored.** The
  per-example table in `RESULTS.md:75-80` is therefore illustrative rather than evidentiary.

Eval must be greedy (`temperature=0.0`) and the temperature must be recorded in the report. This
is a two-line change with outsized effect on trust.

**F5 — the validation split is a single class.** `lora.py:124-133` holds out the **tail** of the
record list. `build_route_dataset.py:113-115` emits records grouped by label
(`for code, descs in ROUTES.items() for desc in descs`), with `QX-1` last in the dict
(`build_route_dataset.py:85`). For `data/route.jsonl` at `valid_fraction=0.1` (`lora.py:51`):
`n_valid = max(1, round(50 * 0.1)) = 5`, `cut = 45`, so `records[45:50]` is **the last five
records, all labeled `QX-1`**. The `Val loss 0.764` reported at `RESULTS.md:70` was computed on
five examples of one label out of five. It is not a measure of generalization across the label
space; it is a measure of how well the model memorized the frontend/UI bucket. The split is
deterministic (good) and stratified (absent). Fix: stratify the split by label, or shuffle under
the config seed before splitting. ~5 lines in `lora.py:124-133`.

**F6 — nothing prevents train/test contamination, now or as the corpus grows.** I checked the
current state directly and it is clean *by author discipline*, not by machinery:

- Zero exact prompt overlap between `data/route.jsonl` and `data/route_golden.jsonl`, and zero
  between `data/extract.jsonl` and `data/extract_golden.jsonl`.
- Golden extract ids are reserved by a hard-coded set (`build_extract_dataset.py:46`,
  `_GOLDEN_IDS`) and filtered out of the training pool (`build_extract_dataset.py:55-56`).
- Golden route prompts are a separate literal dict (`build_route_dataset.py:92-99`).

But: `training/dataset.py` has **no dedup, no content hashing, no split manifest, and no
train/eval barrier of any kind**. `build_dataset` (`dataset.py:135-168`) will happily accept the
same pair twice. `load_dataset` (`dataset.py:179-213`) validates shape and version, never overlap.
The disjointness is asserted in comments (`build_route_dataset.py:91`, `build_extract_dataset.py:5`)
and enforced by nothing. **The moment a dataset is assembled from live traffic (§2), this
protection evaporates**, because the same request can be captured twice, or a near-paraphrase of a
golden item can land in the training pool. Contamination will be silent and will present as a
suspiciously good eval score.

**F7 — the golden sets do not measure generalization; they measure template memorization.** Both
golden sets were written by the *same generator* as their training sets, from the same templates
and the same author, in the same sitting. For `extract`, every golden prompt uses the identical
frame `"Extract the ticket id from: '<template>'."` drawn from the same 12 templates
(`build_extract_dataset.py:27-41`) and the same 10-project pool (`build_extract_dataset.py:42`);
the *only* novelty is the integer. That set cannot distinguish "the model learned to extract a
ticket id" from "the model learned to copy the `[A-Z]+-[0-9]+` substring", which is why the base
model scored a perfect 1.0 with no training at all (`RESULTS.md:57-63`). It is not a hard set; it
is a regex test administered to a 7B model.

For `route`, I measured nearest-neighbour token Jaccard from each golden prompt to its closest
training prompt: **0.36, 0.31, 0.17, 0.15, 0.14**. The top item ("The write-ahead log partition ran
out of space" vs. training's "The database ran out of storage mid-write") shares a third of its
tokens with a same-label training example. That is a legitimate held-out set by the usual standard,
but it is *in-distribution paraphrase*, not independent provenance. It cannot tell you whether the
adapter generalizes to how an incident is actually described in your ticket tracker.

The distinction that matters going forward: **held-out ≠ independent**. Held-out means "the model
did not train on this string." Independent means "this item was produced by a different process
than the training data." Only the second licenses a claim about production behaviour. Real captured
traffic (§2) is the only source of genuinely independent eval items available to you.

**F8 — the golden sets are far too small to support the claims made on them.** Exact counts on
disk, 2026-09-02:

| File | Lines | Header? | Records / examples |
| --- | ---: | --- | ---: |
| `data/route.jsonl` | 51 | yes (`hearth.dataset.header`) | **50** chat records |
| `data/route_golden.jsonl` | 5 | no | **5** examples |
| `data/extract.jsonl` | 49 | yes | **48** instruction records |
| `data/extract_golden.jsonl` | 6 | no | **6** examples |
| **total** | **111** | | **98 training records, 11 golden examples** |

Label balance: `route.jsonl` is exactly 10 per class across 5 classes; `route_golden.jsonl` is
exactly 1 per class. **One flipped example moves the route score by 20 absolute points.** The
statistical consequence is developed in §3.3, but the headline is worth stating here because it
concerns the project's single validated promotion:

> The `0.20 → 1.00` lift recorded at `RESULTS.md:81-83` is 4 discordant pairs in the candidate's
> favour and 0 against. An exact one-sided McNemar test on that gives **p = 0.0625**. The
> observed effect is almost certainly real — the mechanism is understood and the base model is
> provably at chance on an arbitrary convention — but the golden set is **one example too small**
> for the result to clear a conventional p < 0.05 bar. With `n = 6` and the same clean sweep it
> would be p = 0.031.

That is not a criticism of the finding. It is a precise statement of what a 5-example golden set
can and cannot license, and it is exactly the kind of bar the APEX pre-registration habit is
designed to enforce.

**F9 — there is no capture of anything, and the metrics that do exist evaporate.**
`RequestRecord` (`metrics.py:49-64`) has no prompt text, no completion text, no request id, and no
feedback field. `MetricsStore` is an in-memory `deque(maxlen=10_000)` (`metrics.py:70-72`) that is
never persisted — the module docstring admits it (`metrics.py:5-7`) and `RESULTS.md:171-174`
records the consequence in the field (`hearth stats` reports zeros while the daemon has served 19
requests, because they are different processes). `BudgetAccountant` is the same
(`observability/budget.py:5-7`). The gateway *does* mint a per-response id at `app.py:199`
(`chatcmpl-<hex>`) but never attaches it to the `RequestRecord`, so even if records were persisted
you could not correlate a client-side correction back to the interaction that produced it. This is
the single largest gap and it is the subject of §2.

**F10 — the streamed path fabricates its token counts.** `app.py:406-408` estimates
`completion_tokens = text_len // 4` and `prompt_tokens = sum(len(content)) // 4` rather than
tokenizing. Those estimates feed `estimated_tokens_saved` (`metrics.py:39-46`) and therefore the
headline savings figure. Any streamed traffic makes the `estimated_frontier_tokens_saved` number
(`RESULTS.md:154`) a mix of measured and estimated quantities with no flag distinguishing them.
Minor, but it should be marked in the record, not silently blended.

**F11 — the confidence gate is a string-length proxy, and it is inverted.**
`router/route.py:348-360`:

```python
return min(1.0, 0.4 + length / 300.0)
```

Solving against the configured thresholds in `config/routing.yaml:31-36`: `draft` (0.6) escalates
only when the last user message is **shorter than 60 characters**; `chat` (0.65) below 75; `code`
(0.7) below 90. The entire confidence-based escalation policy reduces to *"is the prompt short?"*
And the sign is backwards for the use case: a long, hard, ambiguous reasoning prompt reads as
maximally confident and stays local, while a terse but well-specified request escalates. This is
the heuristic actually deciding where your frontier tokens go. §6.3 proposes a free, principled
replacement.

**F12 — `data/route.jsonl` is not router data.** Worth flagging because the name invites exactly
the wrong conclusion. It is an **incident-description → internal queue-code** dataset for the
`classify` *task class* (`build_route_dataset.py:1-18`). It contains no `prompt → task_class`
labels and cannot train `router/classify.py`. **HEARTH currently has zero training data for its
own task classifier.** See §6.

### 1.3 Summary table

| # | Finding | Location | Severity |
| --- | --- | --- | --- |
| F1 | `default_judge` = length heuristic; silently overrides `metric` | `training/eval.py:105-109,152-162` | High |
| F2 | First promotion passes on any score > 0.0; no base-model floor | `training/eval.py:142-143` | High |
| F3 | Gate compares bare floats; `EvalReport` has no provenance; CLI fabricates one | `training/eval.py:50-62`, `cli.py:739` | High |
| F4 | Eval runs at temperature 0.7, unseeded, non-reproducible | `providers/base.py:31`, `cli.py:616-618` | High |
| F5 | Validation split is the tail → a single label class | `training/lora.py:124-133` | Medium |
| F6 | No dedup / hashing / train-eval barrier anywhere | `training/dataset.py` (absent) | High (latent) |
| F7 | Golden sets share provenance with training sets | `scripts/build_*_dataset.py` | Medium |
| F8 | 11 golden examples total; underpowered for any gate | `data/*_golden.jsonl` | High |
| F9 | No text capture; metrics in-memory only; no correlation id | `observability/metrics.py:49-72` | Critical |
| F10 | Streamed token counts are `chars // 4` estimates | `gateway/app.py:406-408` | Low |
| F11 | Confidence = prompt length; inverted for the use case | `router/route.py:348-360` | Medium |
| F12 | No training data exists for the router's own classifier | — | Medium |

---

## 2. The data flywheel

The suspicion in the brief is correct and it is worse than suspected: **there is no capture at
all**, not even lossy capture. This is the only gap in the system that gets strictly more expensive
every day it stays open, because captured traffic cannot be back-filled. Fixing it is the one item
where "start today" beats "start it right."

### 2.1 Where capture hooks in

**Primary hook: `Router.route`, `router/route.py:232-245.`** This is the single choke point.
Every path converges here:

- gateway non-streaming → `app.py:187` → `router.route(...)`
- MCP tools → `mcp/tools.py:24` `_route_local` → `router.route(...)`
- `hearth run` CLI → same router

`RequestRecord` is already constructed at this exact point with the decision, the result, the
adapter, the latency and the token counts in scope. Extending it is a small diff.

**Second hook, easy to miss: the streaming path builds its own record.** `app.py:412-425` inside
`_stream_sse` constructs and records a separate `RequestRecord`, bypassing `Router.route` entirely.
Any capture implemented only in `route()` will silently lose 100% of streamed traffic. Both sites
need the hook, or `_stream_sse` needs to be refactored to share a single record-building helper —
the latter is better and is a prerequisite worth doing first.

**Not the MetricsStore.** `MetricsStore` (`metrics.py:67-117`) is a rollup engine over a bounded
ring buffer and should stay that. Capture is a different concern with different retention, a
different privacy posture, and a different access pattern. Introduce a sibling:
`src/hearth/observability/capture.py` with an `InteractionLog` backed by SQLite at
`~/.hearth/interactions.db`, following the same connection/schema pattern as
`memory/store.py:132-147`. `MetricsStore.record()` and `InteractionLog.append()` are called from the
same place; the first is always on, the second is opt-in.

### 2.2 Schema

One table, one row per served request. `~/.hearth/interactions.db`:

```sql
CREATE TABLE IF NOT EXISTS interactions (
  interaction_id   TEXT PRIMARY KEY,   -- reuse the gateway's chatcmpl-<hex> (app.py:199)
  ts               REAL NOT NULL,
  source           TEXT NOT NULL,      -- gateway | mcp | cli
  consumer         TEXT,               -- client-supplied tag (cambot, claude-code, ...)

  -- routing decision (all already in scope at route.py:232)
  task_class       TEXT NOT NULL,
  method           TEXT NOT NULL,      -- intent | rules   <- the free label, see 2.5
  intent_hint      TEXT,               -- non-null iff the caller asserted a class
  confidence       REAL,
  backend          TEXT NOT NULL,
  model            TEXT NOT NULL,
  adapter          TEXT,
  served_by        TEXT NOT NULL,
  escalated        INTEGER NOT NULL,
  escalation_reason TEXT,

  -- content (the part that does not exist today)
  messages_json    TEXT NOT NULL,      -- full request messages
  completion       TEXT NOT NULL,
  system_prompt    TEXT,
  temperature      REAL, max_tokens INTEGER,

  -- accounting
  prompt_tokens INTEGER, completion_tokens INTEGER,
  tokens_estimated INTEGER NOT NULL DEFAULT 0,   -- 1 on the streamed path (F10)
  latency_ms       REAL,

  -- corpus discipline (see 2.4)
  content_hash     TEXT NOT NULL,      -- sha256 of the canonicalized prompt
  source_key       TEXT,               -- stable id of the underlying entity, if known
  split            TEXT NOT NULL,      -- train | eval  -- assigned AT WRITE TIME, immutable

  -- correction signal (see 2.3)
  verdict          TEXT,               -- NULL | accepted | corrected | rejected | reasked
  corrected_text   TEXT,
  corrected_class  TEXT,               -- for router-classifier labels
  feedback_ts      REAL,
  feedback_source  TEXT,               -- cli | mcp | review | implicit
  reviewer_note    TEXT
);
CREATE INDEX IF NOT EXISTS ix_task_split ON interactions(task_class, split, verdict);
CREATE INDEX IF NOT EXISTS ix_hash       ON interactions(content_hash);
CREATE INDEX IF NOT EXISTS ix_ts         ON interactions(ts);
```

The `interaction_id` must also be returned to the caller, or corrections have nothing to point at.
Add one field to `HearthTelemetry` (`gateway/schemas.py:37-45`):
`interaction_id: str | None = None`, populated from the same value used at `app.py:199`. MCP tool
results should append it too (`mcp/tools.py:50-88`) so an agent can quote it back.

**Privacy.** This stores raw prompts and completions at rest, which changes the posture documented
at `PRIVACY.md:52-60`. It must therefore be:

- **Off by default**, via `Settings.capture: str = "off"` (`config.py`) with values
  `off | counters | full`. `counters` writes every column except `messages_json`, `completion`,
  `system_prompt`, `corrected_text` — enough for router learning on hashes and for rate analysis,
  with no content at rest.
- File mode `0600`, under `~/.hearth`, never outside it.
- `hearth capture purge --before <date>` and `hearth capture stats` shipped in the same change,
  not later.
- Documented as a new ADR (**ADR-012 — Interaction capture is opt-in, local-only, and never
  leaves the box**) rather than an edit to ADR-007. It does not contradict any existing ADR; it
  extends ARCHITECTURE §8's "structured logs (JSON lines) for later analysis" from an aspiration to
  a schema, and it is the missing input to ARCHITECTURE §7's "curated from ... past sessions, and
  accepted outputs."

### 2.3 How a correction gets recorded — three friction tiers

The design constraint is that **the correction must cost less than the annoyance of the wrong
answer**, or it will not happen.

**Tier 0 — implicit, zero keystrokes.** Three signals are free:

- **`escalated=True, reason=low_confidence`** is already a label: *the system did not trust local
  here.* Every one is a training candidate for the confidence model (§6.3).
- **Re-ask detection.** The same or near-same prompt submitted again within a short window is
  strong evidence the first answer was rejected. Detectable purely from `content_hash` plus a
  10-minute window; write `verdict='reasked'` on the earlier row. Cheap, and it captures the most
  common real-world rejection (the user just tries again).
- **Escalation-after-local.** A local answer at time *t* followed within the window by a
  semantically similar request that escalates is the strongest implicit rejection available. Same
  detector, different arm.

None of these are ground truth. They are *priors for what to review*, which is their job.

**Tier 1 — explicit, one command.** The interaction id is in the response, so:

```
hearth fix <interaction_id> "QX-2"       # the right answer
hearth ok  <interaction_id>              # confirm
hearth fix <interaction_id> --class code # the right task class (router label)
```

`hearth fix` with no id defaults to the most recent interaction from this shell, which is the
common case and removes the copy-paste step entirely.

**Tier 1b — the highest-value path: an MCP correction tool.** Add `hearth_correct` to
`mcp/tools.py` alongside the existing five (`mcp/tools.py:109-127`). When Claude Code offloads an
extraction and you reply "no, it's `CORE-77`", the agent is *already holding both the interaction
id and the corrected value in natural language*. It can call `hearth_correct(interaction_id,
corrected)` with zero keystrokes from you. This is the single highest-leverage capture surface in
the whole design: **the correction signal is already being spoken out loud in the transcript; it
just needs a place to land.** It costs one tool definition and one MCP handler.

**Tier 2 — batch review, deliberately sampled.** `hearth review --task extract --since 7d --n 20`
walks a sampled set and takes accept / edit / skip. The sampling is the interesting part and it
should **not** be recency or random. Stratify and oversample where the information is:

1. all `verdict='reasked'` rows (implicit rejections),
2. all `escalated=True, reason=low_confidence` rows,
3. rows in the lowest confidence quartile that were served locally anyway,
4. rows where a shadow candidate disagreed with the incumbent (§2.6),
5. a small uniform-random stratum so the sample stays estimable and unbiased.

This is active learning at personal scale: strata 1–4 are the disagreement set, and labelling
disagreements buys far more per unit of your attention than labelling random traffic.

### 2.4 Keeping the held-out set uncontaminated as the corpus grows

This is the mechanism that makes the whole thing safe, and it is one line of arithmetic:

> **Assign `split` deterministically from a content hash at write time, and never reassign it.**
>
> ```python
> split = "eval" if int(sha256(canonical_key).hexdigest()[:8], 16) % 100 < 20 else "train"
> ```

Properties that matter:

- **Stable under growth.** An item's assignment depends only on its own content, so adding a
  million rows never moves an existing row across the barrier. The eval set grows organically at
  20% and stays valid.
- **Duplicate-proof.** Identical prompts hash identically, so the same content can *never* land on
  both sides. This is the exact failure that killed F6's implicit protection.
- **Reproducible.** Anyone can recompute the split from the content alone; it needs no manifest to
  be trusted.

Two refinements:

- **Hash a `source_key`, not the prompt, when one exists.** If ten prompts derive from the same
  source document or the same transaction, they must move together, or a paraphrase of an eval item
  ends up in training. Where the caller can supply a stable entity id, hash that; fall back to the
  canonicalized prompt.
- **Canonicalize before hashing:** casefold, collapse whitespace, strip the system prompt. Two
  requests differing only in whitespace must not straddle the barrier.

**Plus a near-duplicate tripwire at dataset-build time.** Deterministic hashing catches exact
collisions; it cannot catch paraphrase. Before emitting any training dataset, compute char-5-gram
Jaccard (or MinHash for speed once the corpus is large) between every train candidate and every
eval item, and **refuse the build** if any pair exceeds a threshold. Introduce it at **0.80**,
which is deliberately loose: the current maximum in the repo is **0.36** (§1.2/F7), so the check
passes today, is non-binding, and only fires when something genuinely changes. Report the observed
maximum on every build so the number is visible and its drift is legible.

### 2.5 Curation and training

`hearth dataset build --task extract --from-corrections --since 30d` should:

1. select `split='train' AND verdict IN ('corrected','accepted')` for the task,
2. drop duplicates by `content_hash`,
3. run the near-duplicate tripwire against the current eval split,
4. emit a versioned `Dataset` via the existing builder (`dataset.py:135-168`) with `provenance`
   carrying the interaction id range, the corpus sha, the counts by verdict, and the tripwire's
   observed maximum Jaccard,
5. **stratify** the train/valid split by label rather than taking the tail (fixes F5).

Two curation rules that are not obvious and matter a great deal:

- **Accepted outputs are weak evidence; corrections are strong evidence.** An `accepted` row often
  means "the user did not look." Training predominantly on the model's own accepted outputs is
  self-distillation and drifts toward the model's existing biases. Concrete rule: **cap the
  accepted:corrected ratio at 3:1** in any built dataset, and record the realized ratio in
  provenance. If you have 400 accepted and 12 corrected, you ship 36 accepted and 12 corrected —
  not 412 rows of mostly self-agreement.
- **Never train on `split='eval'` rows, at any verdict.** The corrected version of an eval item is
  the most tempting training example in the corpus and the one that destroys the measurement. The
  hash split makes this mechanical rather than a matter of care.

### 2.6 Gate → promote, with an online arm

The A/B flag already exists (`adapters.py:181-196`) and is currently only used manually. Two
additions turn it into a measurement instrument:

- **Shadow scoring.** For a sampled fraction of live traffic in a task class, run *both* the
  promoted adapter and the candidate. For the structured tasks in question the second generation is
  4–20 tokens (`RESULTS.md:302` records a 4-token answer) — genuinely cheap. Log both outputs.
- **Disagreement queue.** Where the two disagree, enqueue the item for Tier-2 review. You only
  spend attention where the models actually differ, which is precisely the set where a label
  changes the promotion decision. Everything else is labelled for free by agreement.

This produces an *independently-provenanced, production-distribution* eval set as a by-product of
running the candidate — which is exactly the thing §1.2/F7 says the hand-written golden sets can
never be.

### 2.7 The loop, end to end

```
  serve ──► Router.route (route.py:232) ──► InteractionLog.append(split=hash%100<20)
              + _stream_sse (app.py:412)          │
                                                  ├─ Tier 0: reask / low-confidence flags (free)
                                                  ├─ Tier 1: hearth fix / hearth_correct (1 action)
                                                  └─ Tier 2: hearth review, disagreement-sampled
                                                          │
      hearth dataset build --from-corrections  ◄──────────┘   (train split only, 3:1 cap,
              │                                                near-dup tripwire)
              ▼
      hearth train (lora.py, stratified split)
              │
              ▼
      hearth eval --prereg <file>   ◄── golden = eval split + curated goldens, sha-pinned
              │                          greedy, paired test vs incumbent AND base
              ▼
      promote (proof carries golden_sha + prereg_sha + config fingerprint)
              │
              ▼
      shadow-serve next candidate ──► disagreement queue ──► back to Tier 2
```

---

## 3. Eval authority

"What makes an eval trustworthy enough to gate a promotion." Six mechanisms, all local, all
enforceable in code rather than by discipline.

### 3.1 Golden-set identity and versioning

Golden sets are currently bare JSONL with no header (`data/route_golden.jsonl`,
`data/extract_golden.jsonl` — 5 and 6 lines, no header record), loaded by two different ad-hoc
parsers (`cli.py:63-86` and `scripts/eval_candidate.py:29-34`).

Give them the header treatment `Dataset` already has (`dataset.py:111-121`) plus the fields a gate
needs:

```json
{"kind": "hearth.golden.header", "schema_version": 1, "task": "classify",
 "version": "v3", "created_at": "...", "count": 42,
 "golden_sha": "sha256 of the canonicalized, sorted items",
 "provenance": {"origin": "capture", "split": "eval",
                "window": "2026-08-01..2026-08-31", "labeler": "human",
                "independent_of": "data/route.jsonl"},
 "blocked_from_training": true}
```

Then extend `EvalReport` (`eval.py:50-62`) with `golden_sha`, `golden_version`, `n`, `base_model`,
`adapter_id`, `temperature`, `max_tokens`, `system_hash`, `seed`, `measured_at`. And make
`beats_incumbent` **refuse rather than compare** when `candidate.golden_sha != incumbent.golden_sha`
or the config fingerprints differ. Right now it compares two floats from unrelated universes
without complaint (F3).

`hearth adapters promote --candidate-score <float>` (`cli.py:715-758`) should then be deleted or
demoted to `--force` behind an explicit `--i-know-what-this-is` flag that records
`gate: "manual_override"` in the proof. The normal path becomes `hearth eval --promote`, which
already scores both sides properly (`cli.py:623-638`).

### 3.2 Contamination and leakage detection

Beyond the hash barrier of §2.4, add a `hearth eval --audit` preflight that runs *before* scoring
and refuses on failure:

1. **Exact-overlap check** — any golden `content_hash` present in the training dataset the adapter
   was trained on. The `train_run_id` is already recorded on the adapter entry
   (`adapters.py:51-56`) so the training dataset is recoverable; store its `corpus_sha` alongside.
2. **Near-duplicate check** — max char-5-gram Jaccard between any golden item and any training
   item. Report the max; refuse above 0.80. (Current corpus max: 0.36.)
3. **Answer-leakage check** — for closed-label tasks, whether the golden *answers* appear verbatim
   in training prompts. Currently clean: I verified zero of the six `extract_golden` expected ids
   appear in any `extract.jsonl` completion.
4. **Degenerate-baseline check** — score three trivial baselines on the golden set and record them
   in the report: the empty string, the majority label, and the most common training completion.
   If the candidate does not beat *all three* by the required margin, the gate fails regardless of
   the incumbent comparison. This is the direct fix for F2's `score > 0.0` hole, and it catches the
   `token_f1`-rewards-"the" degenerate case automatically.

### 3.3 Statistical significance on small golden sets

*Is a 3-example improvement real?* Usually: no, and the arithmetic is simple enough to be exact
rather than approximate.

Candidate and incumbent are scored on the **same items** (`eval.py:119`), so the correct test is
**paired**, not a two-proportion comparison. `per_example` already stores exactly the paired vectors
needed (`eval.py:61`) — the harness computes the mean at `eval.py:120` and throws the pairing away.

**For binary metrics (exact-match): exact McNemar.** Let `b` = items the candidate gets right and
the incumbent gets wrong; `c` = the reverse. Concordant items carry no information at all. The
one-sided exact p-value is a binomial test of `b` out of `b+c` at 0.5.

| b (cand wins) | c (inc wins) | one-sided exact p | verdict at α=0.05 |
| ---: | ---: | ---: | --- |
| 3 | 0 | 0.125 | not significant |
| 4 | 0 | 0.0625 | not significant ← **the RESULTS.md promotion** |
| 5 | 0 | 0.031 | significant |
| 6 | 1 | 0.062 | not significant |
| 8 | 1 | 0.020 | significant |
| 10 | 3 | 0.046 | significant |

**Two rules fall straight out of that table:**

1. **You need at least 5 net discordant wins.** No amount of cleverness gets a significant paired
   result from fewer, because `0.5^4 = 0.0625 > 0.05`.
2. **Therefore a golden set must have `n ≥ 30`, and 50 is the working target.** A 5-item set can at
   absolute best produce `b=5, c=0` — requiring the incumbent to fail *every single item* the
   candidate passes. Any incumbent competence at all makes significance unreachable. **On an
   11-example corpus, no honest gate can ever fire.** This is the concrete reason the golden sets
   must grow, and §2 is where the items come from.

**For continuous metrics (token-F1): paired bootstrap.** Resample item indices with replacement
B=10,000 times, recompute `mean(cand) - mean(inc)` on each resample, and require the 2.5th
percentile of that distribution to exceed the margin. On `n=5` the interval will be enormous and
the test will honestly refuse — which is the point. Report the CI in the eval table next to the
point estimate so the width is always visible.

**API shape:**

```python
def beats_incumbent(candidate, incumbent, *, margin=0.0, alpha=0.05, min_n=30,
                    test="auto") -> GateResult
```

returning a `GateResult(passed, test, p_value, ci_low, ci_high, b, c, n, reasons)` rather than a
bare bool, so the proof dict has something worth persisting. Keep the current boolean behaviour
available as `test="none"` for the existing tests (`tests/test_training_eval.py:60-72`) — but make
`"none"` stamp `gate: "unverified"` into the proof so a weak gate is never indistinguishable from a
strong one in the audit trail.

### 3.4 Pre-registration, mechanically enforced

You already do this by habit in APEX. The upgrade is to make the harness refuse to run without it,
so the discipline cannot erode under deadline pressure.

`evals/<task>/prereg-<YYYY-MM-DD>-<slug>.yaml`, committed **before** the training run:

```yaml
task: classify
hypothesis: >
  A LoRA adapter trained on the Aug-2026 correction log will beat the incumbent on
  the Sep-2026 held-out eval split.
golden_sha: "a1b2c3..."          # pinned; the harness refuses a different set
golden_version: v3
n: 52
metric: exact
generation: { temperature: 0.0, max_tokens: 24, seed: 0, system_hash: "9f8e..." }
incumbent_score_measured_before_training: 0.827   # measured FIRST, recorded here
bar:
  test: mcnemar_exact
  alpha: 0.05
  min_net_discordant_wins: 5
  min_absolute: 0.85            # must also clear this floor
  must_beat_baselines: [empty, majority_label, most_common_completion]
tie_rule: "a tie fails"
stopping_rule: >
  One training run at seed 0. No seed re-rolls. If the bar is not met the hypothesis
  is killed and recorded as killed; a second attempt requires a new prereg with a
  stated change to the training recipe.
kill_condition: >
  If the candidate fails to beat the base model (not just the incumbent), the
  task-specific-adapter hypothesis is killed for this task class.
```

Enforcement, all mechanical:

- `hearth eval` **requires** `--prereg <file>` and refuses if `golden_sha` does not match the
  golden set it was handed, or if the generation config differs from the registered one.
- The incumbent score is read from the prereg, not re-measured, so it cannot be re-rolled after
  seeing the candidate.
- `promote` refuses unless `promotion_proof.prereg_sha` matches a file that exists **and is
  committed to git** (`git cat-file -e`) — so the bar provably predates the measurement.
- A failed run writes `evals/<task>/verdict-<date>.md` with `KILLED` and the numbers. Failures
  become artifacts, not silence.

### 3.5 Reproducibility

Fix F4 in the same change: `hearth eval` and `scripts/eval_candidate.py` must pass
`temperature=0.0` explicitly, record it in the report, and refuse to run at `temperature > 0`
without `--allow-sampling` (which stamps `gate: "unverified"`). Also drop
`scripts/eval_candidate.py`'s double generation (lines 66-73) — score once, keep the strings, print
the strings that were actually scored.

Then add a cheap self-check that mirrors the spirit of `_preflight_batch_size`
(`lora.py:158-182`): re-score 3 random golden items a second time and refuse if any answer differs.
That catches an unseeded or misconfigured provider before it produces a promotion.

### 3.6 What the proof should contain

Replace today's `{"candidate_score": 1.0, "incumbent_score": 0.2, "gate_passed": true}`
(`RESULTS.md:98`) with:

```json
{"gate": "verified", "gate_passed": true,
 "test": "mcnemar_exact", "p_value": 0.031, "b": 5, "c": 0, "n": 52,
 "candidate_score": 0.942, "incumbent_score": 0.827, "base_score": 0.211,
 "baselines": {"empty": 0.0, "majority": 0.20, "most_common_completion": 0.20},
 "golden_sha": "a1b2c3...", "golden_version": "v3",
 "prereg_sha": "d4e5f6...", "prereg_committed": true,
 "config": {"base_model": "...", "temperature": 0.0, "max_tokens": 24, "seed": 0},
 "audit": {"exact_overlap": 0, "max_train_eval_jaccard": 0.31, "answer_leakage": 0},
 "measured_at": "2026-09-14T..."}
```

Every field is checkable after the fact by someone who does not trust you — which is the operational
definition of eval authority.

---

## 4. Local LLM-as-judge

Needed for `draft`, `code`, `reason`, `chat` — four of nine classes (`eval.py:28`) — where no
reference answer exists and `default_judge` is currently a length heuristic (F1).

### 4.1 The failure modes, ranked by how badly they would bite *here*

1. **Same-family self-preference — the acute risk in this repo.** `config/models.yaml` contains
   **only Qwen models** (7B-Coder, 14B-Coder, 14B-Instruct, 3B-Instruct, plus a BGE embedder).
   There is no non-Qwen model in the registry at all. A Qwen-14B judging a Qwen-7B-LoRA candidate
   against a Qwen-7B base is scoring three samples from one token distribution; it will
   systematically prefer the phrasing its own pretraining made fluent, which correlates with the
   *candidate's* style precisely because the candidate shares that pretraining. **Mitigation:
   add a different-family judge to `models.yaml`** (a Llama-3.x-8B, Mistral, or Gemma MLX build) and
   pin it. This is a one-entry config change and it is the highest-value single action for judge
   validity.
2. **Position bias.** Judges prefer whichever candidate appears first (or last) at rates that can
   exceed the effect being measured. **Mitigation: every pair is judged twice, A/B and B/A.**
   Report *position-consistency* `c` = fraction of pairs where the verdict survives the swap. Then
   the fraction of verdicts carrying real signal is at most **`2c − 1`**: a judge that is only 70%
   self-consistent has ≤40% signal and 60% coin flip. **Pre-register `c ≥ 0.85`** as a precondition
   for the judge having gate authority at all. Score only the consistent pairs; count inconsistent
   pairs as ties.
3. **Verbosity bias.** Judges reward length. Note that `default_judge` does not merely suffer this
   bias — it *is* this bias, in closed form. **Mitigations:** (a) report the correlation between
   judged win and length delta and refuse to trust a judge whose Spearman ρ with length exceeds
   0.3; (b) include length-matched control pairs in the calibration set; (c) never let the judge
   see which side is the candidate.
4. **Prompt-order and label bias.** Judges over-select the first-listed option and over-use
   whichever label appears earliest in the rubric. **Mitigation:** randomize option order per item
   under a recorded seed, and force a three-way verdict `{A, B, TIE}` with `TIE` listed first so
   it is not the residual.
5. **Rubric drift.** A free-text "which is better?" judge silently changes criteria across runs.
   **Mitigation:** a fixed, versioned rubric with a `rubric_sha` in the `EvalReport`, and per-axis
   scores (correctness / completeness / format-compliance) rather than one holistic verdict.
   Aggregate the axes in code, not in the model.
6. **Non-determinism.** Same fix as §3.5: judge at `temperature=0.0`, record it, refuse otherwise.
7. **Judge is stronger at the task than at judging it.** A 14B judging a 7B on code is not
   obviously more reliable than the 7B itself. **Mitigation:** the calibration in §4.2 is not
   optional — an uncalibrated judge has no gate authority, full stop.

### 4.2 Calibration against human labels — the concrete protocol

A judge earns gate authority by passing a measurement, once, and re-passing it whenever the judge
model or rubric changes.

1. **Build the calibration set: 60 pairs**, drawn from the disagreement queue (§2.6) so they are
   real and hard rather than easy and synthetic. Include **10 deliberate controls**: 5 pairs where
   one side is verifiably wrong (the judge must catch it), and 5 length-mismatched pairs with equal
   quality (the judge must call a tie).
2. **You label them once, blind** — sides anonymized and order randomized, verdict in
   `{A, B, TIE}`. Budget: at ~20 s/pair, about 20 minutes. This is the only human cost in the entire
   plan and it is paid once per judge version.
3. **Run the judge twice per pair (A/B and B/A).** Compute:
   - **position-consistency `c`** — pre-registered bar `c ≥ 0.85`;
   - **Cohen's κ** between judge and human on the consistent pairs — pre-registered bar
     **κ ≥ 0.6** (substantial agreement). Use κ, not raw accuracy: with three classes and an
     unbalanced tie rate, raw agreement flatters badly;
   - **control accuracy** — must be 5/5 on the verifiably-wrong pairs; anything less is
     disqualifying regardless of κ;
   - **verbosity ρ** — Spearman correlation between judged win and length delta; bar |ρ| ≤ 0.3;
   - **self-preference delta** — κ computed separately on pairs where one side came from the
     judge's own family vs. pairs where neither did. A gap > 0.15 means the judge is scoring
     family, not quality, and must be replaced.
4. **Kill condition, pre-registered:** if κ < 0.6 or c < 0.85 or controls < 5/5, **this judge does
   not gate promotions.** It may still be used to *rank items for review* (a biased ranker is
   useful; a biased gate is not). Try a different family, a per-axis rubric, or accept that
   `draft`/`code` are gated by human spot-check for now.
5. **Re-calibrate on any change** to judge model, quantization, rubric, or system prompt. The
   `judge_sha = sha256(model_id + rubric + system_prompt + temperature)` goes in the `EvalReport`,
   and `beats_incumbent` refuses to compare two judge-scored reports with different `judge_sha` —
   the same rule as `golden_sha` in §3.1.

### 4.3 Two cheaper things to do first

- **Reference-anchored judging beats open-ended judging** at this scale. Where you can write a
  reference answer, ask the judge "does the candidate contain the same facts as the reference?"
  rather than "which is better?" Factual-containment judgements are far more reliable than quality
  preferences and correlate better with human labels.
- **For anything with structure, do not judge at all — validate.** JSON parseability, schema
  conformance, label-set membership, regex match, does-the-code-compile, do-the-tests-pass. These
  are free, exact, deterministic, and unbiased. A large share of what `draft`/`code` actually needs
  gating on is format compliance, which a judge is a bad and expensive way to measure. Reach for
  the judge only for what is genuinely irreducible to a check.

---

## 5. Task-specific small models: the quantitative case

### 5.1 What is actually measured on this box

| Model | Decode | Resident | Source |
| --- | ---: | ---: | --- |
| Qwen2.5-Coder-7B-4bit | 26.1 tok/s | ~4.5 GB | `docs/ROADMAP.md:32-35` |
| Qwen2.5-Coder-14B-4bit | 12.4 tok/s | ~9.0 GB | `docs/ROADMAP.md:32-35` |
| 7B LoRA train, 200 iters | 0.64 it/s, ~5 min | 6.4 GB peak | `docs/RESULTS.md:53-55` |
| 7B LoRA trainable params | 0.151% (11.5M / 7616M) | — | `docs/RESULTS.md:52` |
| Served p50 / p95 latency | 1857 / 23821 ms | — | `docs/RESULTS.md:159` |

2× the parameters cost 2.10× the decode time — near-perfect memory-bandwidth scaling, as expected
for 4-bit decode on unified memory. **No 3B or smaller model has ever been benchmarked on this
machine**, so everything below the 7B line is extrapolation and is flagged as such.

### 5.2 Extrapolated (to be measured, not asserted)

Assuming decode ∝ 1/params, from the 26.1 tok/s @ 7B anchor:

| Model | Est. decode | Est. resident | Est. LoRA train (200 it) |
| --- | ---: | ---: | ---: |
| 3B-4bit | ~61 tok/s | ~2.0 GB (`models.yaml`) | ~2 min, ~2.5 GB peak |
| 1.5B-4bit | ~122 tok/s | ~1.0 GB | ~1 min |
| 0.5B-4bit | ~365 tok/s | ~0.35 GB | ~30 s |

### 5.3 The three arguments, in order of actual strength

**Argument 1 (weak): latency.** For a structured task the answer is 4–20 tokens
(`RESULTS.md:302` records a **4-token** `QX-2` answer after the stop-token fix). Decode for 4
tokens: 7B = 153 ms, 3B = 66 ms, 0.5B = 11 ms. The absolute saving is ~90 ms per call and is
swamped by prefill and framework overhead. **Do not justify small models on single-call latency.**
It is real but it is not the reason.

**Argument 2 (strong): throughput at volume.** 10,000 transaction categorizations at 4 tokens
each: 7B ≈ 26 min of pure decode, 3B ≈ 11 min, 0.5B ≈ 2 min. For a nightly batch
categorization job the difference is the difference between "runs while you sleep" and "runs while
you get coffee." This is where the model size actually pays.

**Argument 3 (strongest, and the non-obvious one): retrain cycle time is what makes the flywheel
viable.** A 7B LoRA run is ~5 min at 200 iters (`RESULTS.md:53-55`); a 3B is ~2 min; a 0.5B ~30 s.
At 30 seconds you can retrain **on every correction batch**, nightly, automatically, and gate it
without ceremony. At 5 minutes you retrain when you remember to. The small model's real advantage
is not that it answers faster — it is that **it learns faster, so the loop in §2 can close on a
daily cadence instead of a monthly one.** That is the argument that should decide the architecture.

**The memory argument, which is better than it first looks.** `ram_ceiling_gb` defaults to 24.0
(`config.py:57`) and `ModelManager` LRU-evicts to stay under it. Naively, N specialists means N
resident models. But HEARTH already supports per-request adapter hot-swap over a shared base
(`route.py:310-333`, `GenRequest.adapter` at `providers/base.py:35`). LoRA at 7B was 11.5M params;
at 3B with the same rank and `num_layers=16` (`lora.py:50`) it is roughly 5M ≈ **10–20 MB in fp16**.
So:

- one generalist 7B, no adapters: **4.5 GB**, one behaviour;
- one 3B base + **ten** task adapters: **~2.2 GB**, ten specialised behaviours, hot-swappable
  per request.

**The fleet is cheaper than the generalist.** That is the architectural conclusion, and the
existing code already supports it.

### 5.4 Does the small model actually win on quality? The repo already answers half of it

The most important quality evidence is already in `RESULTS.md` and it reframes the question:

- On plain ticket-id extraction, the **base 7B scores 1.0 with no training at all**
  (`RESULTS.md:57-63`). There is no headroom. A fine-tune here can only regress — and did.
- On the arbitrary QX routing convention, the **base 7B scores 0.20 = chance** and the LoRA scores
  **1.00** (`RESULTS.md:81-83`). The base cannot do it, and *no amount of scale would fix that*,
  because the convention appears in no pretraining corpus. A 14B would score ~0.2. Opus would score
  ~0.2.

So the axis that matters is not *small vs. large*. It is **convention-bound vs. knowledge-bound**:

| Task type | Winner | Why |
| --- | --- | --- |
| Convention-bound (your queue codes, your category taxonomy, your field names) | **fine-tuned small** | The knowledge is not in any pretraining corpus; only training puts it there. Scale does not help. |
| Format-bound (valid JSON, one of N labels, a regex-shaped id) | **constrained decoding, no training** | See §5.5. |
| Knowledge- or reasoning-bound (novel code, open synthesis) | **large / frontier** | Fine-tuning a 3B does not create capability it lacks. |

For transaction categorization into *your* chart of accounts and extraction of *your* field names,
that table says: **fine-tuned small model, and confidently.** It is the convention-bound case, which
is exactly where the repo's one honest measurement shows a 5× lift.

### 5.5 The alternative you should evaluate before fine-tuning anything

**Constrained decoding.** For a closed label set, restrict the sampler to the valid continuations
at decode time. Zero training, zero data, exactly-valid output always. But be precise about what it
buys, because the repo's own numbers make the distinction sharp:

- **Constrained decoding fixes format failures, not knowledge failures.** Constraining the QX task
  to `{QX-1, QX-2, QX-4, QX-7, QX-9}` guarantees a *valid* code and gets you to chance (0.20) — it
  does not get you to the *right* code. Only training does (1.00).
- **Conversely, for the `extract` task it is strictly better than the LoRA.** Base already scores
  1.0; the LoRA *regressed* to 0.0 (`RESULTS.md:57-63`). A regex validator plus a retry costs
  nothing, cannot regress, and needs no adapter, no registry entry and no gate.

**Verdict: do not fine-tune the extraction task. Fine-tune the taxonomy task.** That decision
alone saves a training run, a golden set, and a promotion cycle.

**Self-consistency voting.** With a 3B at ~61 tok/s and 4-token answers, five samples cost ~330 ms
total. Majority-vote them. This typically buys several points of accuracy on classification for
pure latency you already have, needs no training, and — critically — the vote *dispersion* is a
free, calibrated confidence signal (§6.3). Nothing in the repo currently does this and it is one of
the cheapest wins available.

---

## 6. Router learning

### 6.1 First, clear up the naming trap

`data/route.jsonl` is **not** router data (F12). It is incident-description → internal queue-code
for the `classify` *task class* (`build_route_dataset.py:1-18`). It contains no
`prompt → task_class` labels. **HEARTH has zero training data for `router/classify.py` today.**

### 6.2 Can the classifier be learned? Yes — and the labels are already free

`classify()` returns `method="intent"` when the caller asserted a class
(`router/classify.py:58-59`). **Every `method="intent"` request is a human-labeled training
example, flowing through the system right now, and being discarded.** The MCP tools supply an
explicit intent on every call (`mcp/tools.py:54, 65, 82, 88`), and any gateway consumer using the
`hearth.intent` block does the same. Add capture (§2) and the labeled corpus accumulates for free
with zero additional friction.

Caveat worth stating: MCP intents are constant per tool, so MCP traffic alone yields a corpus where
the label is perfectly predictable from the caller — useful for *coverage* of prompt phrasings
within a class, useless for learning the decision boundary. The valuable labels come from gateway
consumers that vary their intent, and from `hearth fix --class` corrections (§2.3) where a *human
disagreed with the rules*. Those disagreement labels are worth 10× the agreement labels.

**What it would take:** ~2,000 labeled prompts, then a **logistic regression over character
n-grams** — not a fine-tune. At nine classes and a few thousand examples, a linear model trains in
under a second, runs in microseconds, is fully deterministic, is inspectable (you can read the
weights and see *why* it fired), and needs no GPU and no adapter lifecycle. `memory/embed.py`'s
`HashEmbedder` already provides a dependency-free feature map if you prefer embeddings to n-grams.
Fine-tuning a 0.5B for 9-way classification would be strictly worse on every axis that matters
here. The "tiny-model classifier hook" comment at `classify.py:66-69` should be read as *"a tiny
model here"*, and a linear model is the right size of tiny.

### 6.3 Is it worth it? Mostly no — and the honest answer points somewhere better

Count where the task class actually changes behaviour, per `config/routing.yaml:31-38`:

- `summarize`, `extract`, `classify`, `rank` → `local / never`. **Six of nine classes are pinned
  local.** Misclassifying among them changes *nothing* except which adapter loads
  (`route.py:310-333`) and which savings multiplier is applied (`metrics.py:22-32`). No cost, no
  quality change.
- `reason` → `remote / always`. This one costs money.
- `draft`, `code`, `chat` → `on_low_confidence`. These *might* cost money.

So the entire ROI of a learned 9-way classifier is concentrated on one boundary: **"does this need
the frontier?"** Everything else is bookkeeping. The rigorous move is therefore:

> Do not build a 9-way classifier. **Measure the misclassification rate on the
> local-vs-escalate boundary specifically**, from captured traffic, and build only if that rate is
> materially costly.

And once you look at that boundary, the classifier is not the weak link — **`_confidence` is**.
It is `min(1.0, 0.4 + len/300)` (`route.py:348-360`), i.e. "is the prompt shorter than ~75
characters," inverted relative to what you want (F11). It is the function actually deciding where
your frontier tokens go.

**Two replacements, in increasing order of cost:**

1. **Self-consistency dispersion — free, no training, available today.** Sample the local model
   *k*=5 times at moderate temperature and use the agreement rate as the confidence score. High
   agreement → the model knows; high dispersion → it is guessing → escalate. This is a genuinely
   calibrated uncertainty signal, it costs ~330 ms on a 3B for short answers (§5.5), it requires no
   data, no labels, and no training, and it drops straight into `_confidence`'s signature. It is,
   in my judgement, the single best ratio of leverage to effort in this entire document after the
   eval-gate fix.
2. **A learned "will local get this right?" predictor — the principled endpoint.** Every captured
   interaction with a `verdict` is a label for exactly this binary question. Train a small
   calibrated classifier on `(prompt features, task_class, model, adapter) → P(correct)` and set
   the escalation threshold by *expected cost*: escalate when
   `P(wrong) × cost(wrong) > cost(frontier call)`. That is the correct decision rule and it makes
   the escalation threshold a business quantity you can tune, rather than a magic number in YAML.
   It requires the §2 corpus, which is why it is downstream — but it is what the flywheel is *for*.

**Verdict on Q6:** learning the task classifier is a low-value project dressed as an interesting
one. Learning the *confidence* function is a high-value project hiding behind a one-line heuristic.
Do the second, and get the first for free as a by-product of capture if the measured boundary error
turns out to justify it.

---

## 7. Prioritized plan

### 7.1 Ranking by leverage ÷ effort

Leverage and effort both 1–5. Effort is in sessions of focused work.

| # | Proposal | Lev | Eff | **Ratio** | Depends on |
| --- | --- | ---: | ---: | ---: | --- |
| **P1** | **Eval authority: un-gameable gate** (§3.1-3.6, F1-F4) | 5 | 2 | **2.5** | — |
| **P2** | **Self-consistency confidence gate** (§6.3.1, F11) | 4 | 2 | **2.0** | — |
| **P3** | **Interaction capture + correction ingress** (§2) | 5 | 3 | **1.7** | — |
| P4 | Stratified train/valid split + dedup + near-dup tripwire (F5, F6) | 3 | 1 | 3.0* | — |
| P5 | 3B-vs-7B specialist bake-off (§5, pre-registered) | 4 | 3 | 1.3 | P1 |
| P6 | Constrained decoding + validators for structured tasks (§5.5) | 3 | 2 | 1.5 | — |
| P7 | Local judge calibration protocol (§4.2) | 4 | 4 | 1.0 | P3 (labels), P1 |
| P8 | Shadow-mode candidate serving + disagreement queue (§2.6) | 4 | 3 | 1.3 | P3 |
| P9 | Learned "will local get this right" predictor (§6.3.2) | 4 | 4 | 1.0 | P3 (weeks of data) |
| P10 | Learned 9-way task classifier (§6.2) | 1 | 3 | 0.3 | P3; measure first |

\* P4 has the highest raw ratio but is a ~1 hour bug-fix bundle, not a project. **Do it inside P1's
session** rather than tracking it separately.

### 7.2 Execution order ≠ ratio order — and why

P1 has the better ratio, but **P3 should be started first in wall-clock terms.** Capture's value is
*time-integrated*: a month of traffic is worth a month of waiting, and it **cannot be
retroactively recovered**. Every day the hook is missing is a day of correction signal permanently
destroyed. P1's value, by contrast, is available in full whenever you get to it.

**Recommended sequence:** land P3's *write path* first (even in `counters` mode, even before the
correction commands exist — just start recording), then do P1 properly, then P2, then P5.

### 7.3 Top 3, in implementable detail

---

#### **P1 — Eval authority: make the gate un-gameable**

*Effort: one session, no GPU time. Files: `training/eval.py`, `registry/adapters.py`, `cli.py`,
`scripts/eval_candidate.py`, `tests/test_training_eval.py`, plus `training/lora.py` and
`training/dataset.py` for the P4 bundle.*

**Changes:**

1. `default_judge` (`eval.py:152-162`) → raises `NotImplementedError`. `score_candidate`
   (`eval.py:105-117`) raises when both `metric` and `judge` are given.
2. `EvalReport` (`eval.py:50-62`) gains `golden_sha`, `golden_version`, `base_model`, `adapter_id`,
   `temperature`, `max_tokens`, `system_hash`, `seed`, `measured_at`, `judge_sha`.
3. `beats_incumbent` (`eval.py:130-144`) returns a `GateResult`; runs exact McNemar (binary) or
   paired bootstrap (continuous) over `per_example`; refuses on `golden_sha` / config mismatch;
   enforces `min_n`; **when `incumbent is None`, the base model is the incumbent** (kills F2).
4. Degenerate-baseline scoring (empty / majority / most-common-completion) computed on every eval
   and required to be beaten (§3.2.4).
5. Golden-set header + `golden_sha`; one loader shared by `cli.py:63-86` and
   `scripts/eval_candidate.py:29-34`.
6. `temperature=0.0` explicit at `cli.py:616-618` and `eval_candidate.py:54-59`; refuse sampling
   without `--allow-sampling`; drop the double generation at `eval_candidate.py:66-73`.
7. `--prereg` required by `hearth eval`; `promote` refuses without a committed matching prereg;
   `hearth adapters promote --candidate-score` demoted behind `--force` and stamped
   `gate: "manual_override"`.
8. **(P4 bundle)** stratified split in `lora.py:124-133`; content-hash dedup and the near-duplicate
   tripwire in `dataset.py`.

**Pre-registered acceptance criteria** *(written before implementation; each is falsifiable and
each has a kill condition):*

- **A1.** A candidate whose generator returns a fixed 10 KB constant string is **refused** by the
  gate on every task class, including `draft`. *(Today it passes via `default_judge`.)*
- **A2.** With no incumbent registered, a candidate that does not beat the **base model** on the
  golden set is **refused**. *(Today `score > 0.0` promotes it.)*
- **A3.** `hearth eval` run twice on the same adapter and golden set produces **byte-identical**
  `per_example` vectors.
- **A4.** A candidate scored on golden `v3` cannot be compared against an incumbent scored on `v2`;
  the gate raises rather than returning a boolean.
- **A5.** Replaying the `RESULTS.md` Task A promotion (`b=4, c=0, n=5`) through the new gate yields
  `p = 0.0625` and **`passed = False`** under the default `alpha=0.05, min_n=30`. *This is the
  self-consistency test: the new gate must be willing to refuse the project's own celebrated
  result.* If it passes that promotion, the gate is still not real.
- **A6.** `promote` refuses when `promotion_proof.prereg_sha` names a file not committed to git.
- **A7.** `lora._split` on `data/route.jsonl` yields a validation set containing **all five** QX
  labels. *(Today: five records, all `QX-1`.)*
- **A8.** The near-duplicate tripwire run on the current corpus reports max Jaccard **0.36**
  (route) and passes at threshold 0.80. *(Pre-registering the number means a future change that
  moves it is visible.)*
- **Kill condition:** if satisfying A5 means no adapter HEARTH can currently train would ever be
  promotable, **that is the finding** — record it, and treat "grow the golden sets to n≥30" as a
  hard blocker on the entire training subsystem rather than working around the gate.

---

#### **P2 — Replace the length-proxy confidence with self-consistency dispersion**

*Effort: one session, some GPU time for measurement. Files: `router/route.py`, `config/routing.yaml`,
a new `scripts/confidence_calibration.py`.*

**Change:** `_confidence` (`route.py:348-360`) gains a second implementation selected by config:
sample the local model *k*=5 times at `temperature≈0.7` for the `on_low_confidence` classes and
return the modal-answer agreement rate (`count(mode)/k`). Keep the length heuristic as
`confidence: length` for a fallback and for the echo backend; add `confidence: self_consistency`
to `routing.yaml` defaults. Cache by prompt hash so a repeated prompt does not re-sample.

**Pre-registered acceptance criteria:**

- **B1.** On a held-out labeled set of **n ≥ 100** captured or constructed prompts spanning
  `draft`/`code`/`chat`, self-consistency confidence achieves **AUC ≥ 0.70** for predicting
  local-answer correctness. *(Bar chosen because a length proxy should be near 0.5; anything below
  0.65 is not worth the latency.)*
- **B2.** The **length** proxy is measured on the identical set and its AUC recorded. If length
  scores ≥ 0.65, the whole premise is wrong and P2 is **killed** — publish that instead.
- **B3.** Added p50 latency on `on_low_confidence` classes is **< 400 ms** at k=5 for answers
  ≤ 32 tokens.
- **B4.** At a threshold tuned to hold the escalation rate constant, self-consistency produces
  **strictly fewer** wrong-local answers than length on the same traffic. If it does not, P2 is
  **killed** regardless of B1.
- **Kill condition:** if AUC < 0.65 or B4 fails, revert to length, record the negative result, and
  note that the confidence gate needs the learned predictor (P9) rather than a free heuristic.

---

#### **P3 — Interaction capture + correction ingress**

*Effort: one session for the write path; a second for the correction surfaces. Files: new
`observability/capture.py`, `router/route.py`, `gateway/app.py`, `gateway/schemas.py`,
`mcp/tools.py`, `cli.py`, `config.py`, `docs/DECISIONS.md` (ADR-012), `docs/PRIVACY.md`.*

**Changes, in the order they should land:**

1. Refactor `_stream_sse`'s record construction (`app.py:412-425`) and `Router.route`'s
   (`route.py:232-245`) to share one helper — **before** adding capture, so it cannot be added to
   only one path.
2. `InteractionLog` (SQLite, schema in §2.2), `Settings.capture = off|counters|full`,
   `hearth capture stats|purge`.
3. `interaction_id` threaded into `HearthTelemetry` (`schemas.py:37-45`) and MCP results.
4. Hash-based immutable `split` assignment at write time (§2.4).
5. `hearth fix` / `hearth ok`; the `hearth_correct` MCP tool.
6. Tier-0 implicit detectors (re-ask window, low-confidence flag).
7. ADR-012 + a `PRIVACY.md` section on data at rest.

**Pre-registered acceptance criteria:**

- **C1.** After driving the `RESULTS.md` Task B workload (19 requests, mixed classes, streaming and
  non-streaming), `interactions.db` contains **exactly 19 rows** with non-empty `messages_json` and
  `completion` under `capture=full`. *(Specifically checks that the streaming path is not silently
  dropped.)*
- **C2.** With `capture=off` (the default), the DB file **is not created**, and p50 latency is
  within **2%** of the pre-change baseline.
- **C3.** With `capture=counters`, no row contains any request or response text in any column.
- **C4.** `split` is stable: re-running the same 19 requests produces 19 new rows whose `split`
  values match the originals **1:1** by `content_hash`.
- **C5.** A correction issued via `hearth_correct` from an MCP session lands on the correct row,
  matched by `interaction_id`, with `verdict='corrected'` and `feedback_source='mcp'`.
- **C6.** `hearth dataset build --from-corrections` on a seeded corpus emits a dataset containing
  **zero** rows with `split='eval'`, verified by hash-set intersection being empty.
- **C7.** Sealed-mode check: with the private profile active, driving traffic with capture on
  produces **zero** outbound connections (the `lsof` procedure at `PRIVACY.md:84-87`).
- **Kill condition:** if C2 fails (capture-off costs measurable latency) the design is wrong —
  move the write off the request path onto a queue before proceeding.

### 7.4 What I deliberately am *not* proposing

- **RLHF / DPO / preference optimization.** At 11 golden examples and no preference corpus, the
  data does not exist and would not for a year. LoRA SFT on corrections is the right tool at this
  scale.
- **A bigger base model as the answer to quality.** `RESULTS.md:81-83` shows the failure mode is
  convention-boundedness, which scale does not fix (§5.4).
- **A learned 9-way task classifier as a near-term project.** §6.3 — six of nine classes are pinned
  local; the ROI is a rounding error until the escalation boundary is measured.
- **Fine-tuning the `extract` task.** Base scores 1.0 and the LoRA regressed to 0.0
  (`RESULTS.md:57-63`). Constrained decoding plus a validator is strictly better (§5.5).
- **Any cloud eval, cloud judge, or telemetry upload.** Non-negotiable per `docs/PRIVACY.md`; every
  mechanism above is local-only by construction.

### 7.5 Prior decisions this respects

Nothing here contradicts an existing ADR. Explicitly:

- **ADR-006** (PEFT-only, local, eval-gated) is *strengthened*: §3 makes "eval-gated" mean something
  checkable. The "revisit if PEFT plateaus" clause is exactly what P5's kill condition tests.
- **ADR-007** (escalation explicit and measured) is extended by §2 from counters to content, and
  §6.3 replaces the heuristic behind the escalation decision without changing its contract.
- **ADR-005** (routing is data, not code) is honoured: P2 adds a `confidence:` key to
  `routing.yaml` rather than branching in code.
- **ARCHITECTURE §7's** "datasets curated from ... past sessions, and accepted outputs" is the
  documented intent that §2 finally implements; **§8's** "structured logs (JSON lines) for later
  analysis" is the same for §2.2.
- **One new ADR is required: ADR-012 — Interaction capture is opt-in, local-only, and never leaves
  the box**, because storing raw prompts and completions at rest changes the posture documented at
  `PRIVACY.md:52-60` and should be a deliberate, recorded decision rather than a side effect.
