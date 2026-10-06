# RUNBOOK — Validating the LoRA fine-tuning path on real weights (Phase 4, G4)

This runbook takes the Phase 4 fine-tuning path from "exercised only with fakes" to
"validated end-to-end on real weights": build a dataset, run a **real** LoRA fine-tune,
confirm the candidate registers, evaluate it, and confirm the eval-gated promotion
lifecycle. It calls only commands and functions that exist in the codebase today.

> **Hardware gate.** The training run itself (steps 3–4) requires an Apple-Silicon GPU, the
> `[mlx]` extra, and a base model already in the local Hugging Face cache. **These steps
> cannot run in CI or a locked-down sandbox** — network model downloads are blocked there.
> Steps that ARE verifiable without hardware are marked *(CI-safe)*; steps that need real
> hardware are marked *(hardware)*.

The harness that automates step 3 is `scripts/train_lora_real.sh`
(run `scripts/train_lora_real.sh --help`). Its `--promote` option is broken (B-015); promote
with `hearth eval --promote` (step 6).

---

## 0. Prerequisites *(hardware)*

- macOS on Apple Silicon (`uname -m` → `arm64`).
- `uv` on PATH.
- The training backend installed:

  ```sh
  uv sync --extra mlx --extra mcp --extra dev --extra files
  ```

- A base model cached locally. HEARTH's default is
  `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` (see `config/models.yaml`). Pre-warm the
  cache **once** from an unrestricted network, then work offline:

  ```sh
  # from an unrestricted terminal (network allowed):
  hearth models pull mlx-community/Qwen2.5-Coder-7B-Instruct-4bit
  # or: huggingface-cli download mlx-community/Qwen2.5-Coder-7B-Instruct-4bit
  ```

- Offline is the default: `hearth train` resolves `--base` from `~/.hearth/models` or the hub
  cache and runs `mlx_lm.lora` with `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` set for the
  child, failing with `ModelNotOnDiskError` if the base is not on disk (unless
  `HEARTH_ALLOW_DOWNLOADS=1`). Exporting the vars yourself is harmless but no longer needed.
  `uv run --no-sync hearth doctor --offline` confirms it before you spend GPU time.

  `scripts/train_lora_real.sh` sets these itself and refuses to run if the base model is
  not already cached, so it can never silently download.

---

## 1. Build (or bring) a dataset *(CI-safe)*

Training reads a JSONL dataset produced by `hearth.training.dataset`. Each record is either
**instruction** shape (`{"prompt": ..., "completion": ...}`) or **chat** shape
(`{"messages": [...]}`). The file **must** start with the header line that `write_dataset`
writes (`{"kind": "hearth.dataset.header", "task": ...}`); a headerless file is refused,
because the header is the only place the task is recorded (see `dataset.py`). Build one
programmatically:

```python
from pathlib import Path
from hearth.training.dataset import build_dataset, write_dataset

pairs = [
    ("Extract the ticket id from: 'Fixed in PROJ-1234, see PR #90'.", "PROJ-1234"),
    ("Extract the ticket id from: 'Closes PROJ-5678 after review'.",  "PROJ-5678"),
    # ... see the size note below — a real run needs ~40+ records, not just 2
]
ds = build_dataset(task="extract", pairs=pairs, created_at="2026-07-09T00:00:00Z")
write_dataset(ds, Path("data/extract.jsonl"))
```

> **Dataset size — read this before a real run.** `LoRAConfig.validate()` accepts **≥ 2
> records**, but that is *not enough for a real `mlx_lm.lora` run*. HEARTH holds out
> `valid_fraction` (default 0.1) as a validation split, and mlx-lm iterates that split in
> `batch_size` (default **4**) chunks — so the **validation split must have ≥ `batch_size`
> examples**, or the run aborts with `Dataset must have at least batch_size=4 examples but
> only has N`. In practice you need **~40+ records** (≈ `batch_size / valid_fraction`).
> `scripts/build_extract_dataset.py` / `scripts/build_route_dataset.py` generate ~48–50.
> The real runner now preflights this and raises an actionable error before spending GPU
> time (see `hearth.training.lora._preflight_batch_size`). Validated live on Apple Silicon
> — see [RESULTS.md](RESULTS.md) → Finding 1.

You can validate a hand-written file without a GPU:

```sh
uv run --no-sync python -c "from hearth.training.dataset import load_dataset; print(len(load_dataset('data/extract.jsonl')))"
```

---

## 2. Assemble a golden eval set *(CI-safe)*

Promotion is gated on a golden set (`hearth.training.eval`). Keep it separate from the
training data. Objective classes (`extract`, `classify`, `summarize`, `rank`) score with
exact-match or token-F1; subjective classes (`draft`, `code`) need a judge hook.

`hearth eval` and `hearth prereg` read the golden set as a JSONL file of
`{"prompt": ..., "expected": ...}` rows (a `hearth.dataset` or `hearth.golden` header line
is skipped). It needs **at least 30 rows** to clear the gate's default `min_n`. The
mathematical floor for any promotion at α=0.05 is 5, since the smallest achievable p is 0.5ⁿ.
For scripting, the same set in Python:

```python
from hearth.training.eval import as_golden_set
golden = as_golden_set("extract", [
    ("Extract the ticket id from: 'See PROJ-4242 for context'.", "PROJ-4242"),
    # ...
])
```

---

## 3. Run the real training + candidate registration *(hardware)*

Use the harness (recommended — it enforces every prerequisite and stays offline):

```sh
scripts/train_lora_real.sh --data data/extract.jsonl --task extract --iters 200
```

Or drive the CLI directly (equivalent):

```sh
HF_HUB_OFFLINE=1 uv run --no-sync hearth train --task extract \
    --base mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \
    --data data/extract.jsonl --iters 200
```

Under the hood (`hearth.training.lora`): HEARTH validates the config, lays out the run dir,
writes the `train.jsonl`/`valid.jsonl` split, then shells out to
`python -m mlx_lm.lora --train ...` and registers the result as a **candidate** adapter
named `<task>-<run-id>` (`hearth.cli:train`).

### Expected artifacts *(hardware)*

- Run directory: `~/.hearth/train/<run-id>/` (override with `--out`), containing:
  - `data/train.jsonl`, `data/valid.jsonl` — the deterministic split.
  - `adapters/` — the produced LoRA adapter weights (`--adapter-path`).
- A candidate row in the adapter registry (`~/.hearth/adapters.json`). Confirm:

  ```sh
  uv run --no-sync hearth adapters list --task extract
  ```

  The new adapter shows `status = candidate` and an empty `eval` column.

---

## 4. Pre-register the bar, then commit it *(CI-safe)*

An adapter cannot be promoted on a score you typed (CLAUDE.md §7). The bar is declared
**before** the measurement, in a pre-registration file that git has committed and that is
unmodified when the gate runs. Scaffold one from the golden set (run, synthetic set):

```sh
uv run --no-sync hearth prereg init --task extract --golden data/extract_golden.jsonl \
    --metric exact --system "Reply with only the answer, nothing else." \
    --out prereg/extract.yaml
# -> Wrote prereg/extract.yaml. Fill in hypothesis/stopping_rule/kill_condition, then
#    git commit it — an uncommitted prereg cannot gate a promotion.
```

The file pins the golden set by **content sha** and the decode parameters (temperature 0,
`--max-tokens`, a hash of `--system`) by **fingerprint**, and declares the bar: `alpha`
(default 0.05), `min_effect` (0.0), `min_n` (30), `test: auto`, and the degenerate baselines
the candidate must beat (`empty`, `majority_label`, `copy_input`). The `--metric`,
`--max-tokens` and `--system` you register here must be the ones you pass to `hearth eval`
in step 5, or the run is refused as "not the registered experiment".

The bar may only be **stricter** than the gate's defaults (B-062): `alpha` in (0, 0.05],
`min_effect` ≥ 0, `min_n` an integer ≥ 30, `test` one of `auto` / `mcnemar` / `bootstrap`,
every number finite. `prereg init`, `prereg check` and `eval` refuse anything else
(`pre-registered bar refused: alpha must be in (0, 0.05] ...`).

**Where the prereg lives.** Commit it, and the golden set, in the **evals repository** this
install is anchored to (B-081). By default that is HEARTH's own repository — next to
`data/<task>_golden.jsonl`; `hearth prereg anchor` prints it and `hearth prereg anchor
<repo>` pins another one (stored in `~/.hearth/evals-repo`). Every eval records the anchor
in force, and promotion requires the prereg to live in the anchor recorded at the adapter's
**first** measurement: a copy of the golden set in a throwaway repo does not count, and
re-anchoring after a score has been seen does not rescue an adapter already measured.
Promotion also refuses a golden set that was not committed and unmodified when it was
measured. "Committed and unmodified" means the bytes on disk are the committed blob — HEARTH
hashes them itself rather than asking `git diff`, so `git update-index --assume-unchanged`
or `--skip-worktree` cannot hide an edit (B-078) — and at promotion the golden set is
re-read from the committed blob and must hash to what was scored. Every golden row must be
a distinct prompt (whitespace- and case-insensitive): a repeated prompt is one observation
counted twice, and `eval` / `prereg init` refuse the set (B-080). A prereg YAML with a
duplicated key is refused too (B-083).

Fill in `hypothesis`, `stopping_rule` and `kill_condition` by hand — `prereg check` and `eval --promote` refuse a prereg with any of them blank, and one whose `bar.must_beat_baselines` drops a default baseline (empty / majority / copy-input) — then commit **before the adapter is measured at all**. Every `hearth eval` — with or without `--prereg`, on any golden set — is recorded in a signed, append-only ledger (`~/.hearth/measurements.jsonl`) *before* it scores anything, and promotion requires the prereg's last commit to precede the adapter's **first** recorded measurement, both in committer time and in history (the commit must be an ancestor of the HEAD recorded at that first measurement, which a backdated commit cannot be) (B-079). An exploratory `hearth eval` run before the bar is committed therefore makes that adapter unpromotable; retrain or measure a fresh adapter under the committed bar:

```sh
git add prereg/extract.yaml data/extract_golden.jsonl && git commit -m "prereg: extract adapter bar"
uv run --no-sync hearth prereg check prereg/extract.yaml --golden data/extract_golden.jsonl
# -> prints the registered bar, "Golden set matches (N examples).", and
#    "git: committed at <sha> and unmodified."  (exit 0; <sha> is the commit that last
#    changed the prereg, not HEAD)
```

`prereg check` exits 1 if the file is not committed or has local edits
(`git: not committed — ...`), or if the golden set no longer hashes to the registered sha
(`Golden set has changed: ...`). The uncommitted path was run against a synthetic set; the
committed path was not run here (it needs a commit in a real repo) and is covered by
`tests/test_cli_eval.py`. Editing the golden set after registering it is exactly the
thing the sha is there to catch.

---

## 5. Evaluate the candidate *(hardware for a real score; refusal paths CI-safe)*

`hearth eval` runs the candidate through `MLXProvider`'s per-request adapter slot
(`GenRequest.adapter`, via `AdapterStore.resolve_path(..., allow_candidate=True)`), scores it
at temperature 0, scores the **incumbent** the same way (the currently-promoted adapter for
the task, or **the base model** when none is promoted), scores the degenerate baselines,
and runs the gate: a paired significance test over the per-example vectors (exact McNemar
for exact-match, paired bootstrap otherwise), lift above `min_effect`, `n >= min_n`, and a
win over every registered baseline.

Measure and write the report without promoting *(needs real weights — not run)*:

```sh
HEARTH_BACKEND=mlx uv run --no-sync hearth eval extract-<run-id> \
    --golden data/extract_golden.jsonl --metric exact \
    --system "Reply with only the answer, nothing else." \
    --prereg prereg/extract.yaml --report-json reports/extract-<run-id>.json
```

It prints a table (candidate, incumbent or base, each baseline), then
`gate: PASS|FAIL n=... alpha=... mcnemar_exact p=... (b=..., c=...)`, one line per failed
condition, and the `golden_sha` / `config` fingerprint. The JSON report carries the
per-example vectors and provenance, so the gate can be recomputed from it later. It also
records what was measured — adapter id, task, base model, adapter path, a SHA-256 of the
adapter's weights (hashed before scoring), the incumbent and its weights hash,
`measured_at`, and the golden set's git status — and is **HMAC-signed** with this install's
key (`~/.hearth/eval-report.key`, created 0600 on first use). Do not edit it: an edited
report is refused at promotion. A report only promotes on the install that wrote it.

`hearth eval` refuses to measure (exit 1) when the adapter's weights are missing — scoring
an adapter with no weights scores the base model — and exits 2 when HEARTH_DEFAULT_MODEL
names an unregistered model or the adapter's base model is empty, `auto` or not servable
(B-070): an eval never runs on a silent fallback model.

The same command on the `echo` backend runs the plumbing offline with no weights. Its
scores are meaningless, so the gate fails. That is a useful check that the gate refuses a
non-lift (run, synthetic 30-row golden set):

```text
gate: FAIL n=30 alpha=0.05 mcnemar_exact p=1.0000 (b=0, c=0)
  · no lift: candidate 0.0000 does not exceed base 0.0000 + margin 0
  · not significant: mcnemar_exact p=1.0000 > alpha=0.05 (b=0, c=0)
  · fails degenerate baseline 'copy_input': ...
```

The refusals, all CI-safe (run on the echo backend; each exits 1):

| You run | You get |
|---|---|
| `hearth eval ... --temperature 0.7` | `Refusing to score at temperature > 0: the gate would be re-rollable.` (`--allow-sampling` measures anyway; it can never be promoted under a temperature-0 prereg) |
| `hearth eval ... --prereg prereg/extract.yaml` with a different `--system`/`--metric`/`--max-tokens` | `This run is not the registered experiment: decode config ... != registered ...` |
| `hearth eval ... --promote` with no `--prereg` | `Promotion refused: --promote requires --prereg.` |
| `hearth eval ... --prereg <uncommitted file> --promote` | `Promotion refused: pre-registration is not git-committed: <file> is not tracked by git ...` (or `not inside a git repository`) |
| `hearth eval ... --prereg <file committed after the adapter was first measured> --promote` | `Promotion refused: pre-registration was committed at ..., AFTER the measurement started at ... — the first recorded measurement of ...` or `... was not in the history of ... when ... was first measured ...` |
| `hearth eval ... --prereg <file in a repo other than the golden set's> --promote` | `Promotion refused: the pre-registration lives in ..., but the golden set is versioned in ...` |
| `hearth eval ... --prereg <file + golden set in a repo that is not the anchored evals repository> --promote` | `Promotion refused: ... which is not the evals repository HEARTH was anchored to when ... was first measured ...` |
| `hearth eval ... --prereg <file edited, hidden with --assume-unchanged> --promote` | `Promotion refused: pre-registration is not git-committed: ... has uncommitted modifications ...` |
| `hearth eval --golden <set that repeats a prompt> ...` | `Golden set error: golden set repeats N prompt(s) ...` |
| `hearth eval ... --promote` on the echo backend or a plugin | `Promotion refused: the scores were generated by backend '...', not the MLX model pool ...` |
| `hearth eval ... --metric exactt` | `Unknown metric: 'exactt' ...` (refused before anything is recorded) |
| `hearth eval <adapter with no weights on disk> ...` | `Refusing to measure: adapter weights not found: ...` |
| `hearth eval ... --alpha 0.5` (or `--margin -1`, `--min-n 1`) | `Gate refused to compare: alpha must be in (0, 0.05] ...` |

`--check-determinism` re-generates a few prompts and refuses if any answer changes: a score
that re-rolls is not a measurement.

Without `--prereg`, `eval` still prints the gate verdict, followed by `Exploratory
measurement (no --prereg): recorded in the measurement ledger. <id> can only be promoted
under a pre-registration committed BEFORE its first measurement (<time>)`. With `--prereg`
but no `--promote` it says up front whether the adapter is promotable under that prereg.
The ledger is append-only and chained: an edited, removed or reordered record makes every
later measurement and promotion refuse (`cannot record the measurement: ...`,
`the measurement ledger is not intact: ...`). Only scores generated by the MLX model pool
(`HEARTH_BACKEND=mlx`) can promote; the report records the backend (B-084). A promoted
incumbent is scored on **its own** registered base model, not the candidate's (B-085).

For scripted or custom scoring, the same API is in `hearth.training.eval`. Use
`evaluate_gate`, not `beats_incumbent` (a legacy mean-only comparison with no significance
or provenance check, which returns `False` when there is no incumbent):

```python
from hearth.training.eval import EvalConfig, baseline_reports, evaluate_gate, score_candidate

cfg = EvalConfig.for_system("Reply with only the answer, nothing else.")  # temperature 0
candidate = score_candidate(golden, generate_with_candidate, metric="exact", config=cfg)
incumbent = score_candidate(golden, generate_with_base, metric="exact", config=cfg)  # or the promoted adapter
gate = evaluate_gate(candidate, incumbent, incumbent_role="base",
                     baselines=baseline_reports(golden, metric="exact", config=cfg))
print(gate.passed, gate.reasons)
```

Pass the same `config` to every report. Without it the reports carry no provenance and the
gate refuses them (`candidate report has no provenance (golden_sha/config) — unverifiable`).
This snippet was run against a synthetic 30-row set with stub generators.

---

## 6. Promote the winning candidate *(needs real weights — not run)*

Promotion goes through the gate, under the committed pre-registration. Two equivalent ways.

Measure and promote in one step:

```sh
HEARTH_BACKEND=mlx uv run --no-sync hearth eval extract-<run-id> \
    --golden data/extract_golden.jsonl --metric exact \
    --system "Reply with only the answer, nothing else." \
    --prereg prereg/extract.yaml --promote
# -> Promoted extract-<run-id> (gate passed, candidate=..., p=...).
```

Or promote later from the step-5 report. `adapters promote` does **not** trust the report:
it verifies the HMAC signature (only a report `hearth eval --report-json` wrote on this
install, unedited), checks the report is about **this** adapter (id, task, base model,
weights path, and weights on disk that still hash to what was measured) against the
incumbent that is **still** in place, recomputes the gate from the stored per-example
vectors under the bar in the prereg, requires the report's measurement to be in this
install's ledger, and re-checks the prereg's provenance (committed, unmodified, committed
before the adapter's first recorded measurement, in the anchored evals repository that
holds the golden set). The incumbent check and the registry write happen under one lock, so
an adapter promoted in between is not silently retired (B-082):

```sh
uv run --no-sync hearth adapters promote extract-<run-id> \
    --report reports/extract-<run-id>.json --prereg prereg/extract.yaml
# -> Promoted extract-<run-id> (gate passed, mcnemar_exact p=..., n=...).
```

Refusals here (all exit 1, all in `tests/test_promotion_evidence.py`): `Unusable eval report:
report is not signed ...` / `... was edited after hearth eval wrote it` / `... signed by a
different install's key`; `Promotion refused: the report is not evidence for '<id>' — report
measured adapter ..., weights changed since the measurement ..., ... is the promoted adapter
for '<task>' now: re-run hearth eval against it`. If you retrain, or another adapter is
promoted for the task in between, re-run step 5.

Both `--report` and `--prereg` are required (`Promotion requires --report and --prereg.`,
exit 1). The old `--candidate-score` / `--incumbent-score` flags are **removed**. Passing
them prints `--candidate-score/--incumbent-score have been removed. An operator-typed score
is not evidence...` and exits 2 (run). `scripts/train_lora_real.sh --promote` still passes
them and therefore always fails (B-015). Train with the harness, then promote with
`hearth eval --promote`.

Confirm the lifecycle transitioned and any prior promoted adapter for the task was retired
(the store keeps exactly one promoted adapter per task):

```sh
uv run --no-sync hearth adapters list --task extract
# candidate -> promoted; a previously-promoted adapter for `extract` shows `retired`.
```

The promotion is auditable: `~/.hearth/adapters.json` records the `promotion_proof`, which
holds the gate result (test, p-value, n, alpha, baselines), the pre-registration it ran
under (its sha; `prereg_commit`, the commit that last changed it — not HEAD — with
`prereg_committed_at` and `prereg_introduced_commit`; `prereg_repo`; `golden_commit`), the
`candidate_weights_sha` that was measured, `evidence` (`measured` for `eval --promote`,
`signed-report` for `adapters promote`), the ledger positions (`ledger_seq`,
`first_ledger_seq`, `first_measured_at`) and, for `adapters promote`, the `report_sha`.

What this does not cover (residual trust): a user who reads `~/.hearth/eval-report.key` can
sign anything — reports and ledger records alike — and the same user can edit
`adapters.json` directly, so no file-based check could do better. Anyone who can write
`~/.hearth` can delete the ledger or truncate its tail (an orphaned report is refused, but a
fresh measurement after the deletion starts a new history). The anchor is install
configuration: re-pinning it before an adapter's first measurement is allowed (and
recorded). Perturbing an adapter's weights makes a new candidate with a fresh history.
Committer timestamps are settable, which is why the ancestry rule exists — but a bar
committed *before* the first measurement and never looked at again is what is enforced,
not that nobody peeked at a sibling adapter trained the same way.

**Measurements the ledger cannot see (B-123).** The ledger records every `hearth eval` on
*this* install. It does not record, and cannot stop you learning from:

- **another install** — a second `HEARTH_HOME` has its own key and ledger, so a full PASS
  measured there never appears here. Measure only on the install you promote from.
- **serving the candidate** — `/v1/chat/completions` with `hearth.adapter=<candidate id>` (the
  A/B path) answers with the candidate; chatting with it is an informal measurement.
- **library calls** — `hearth.training.eval.score_candidate` scores without writing the ledger.

These are deliberate: closing them would block legitimate A/B serving and testing. The gate
guards against *self-deception by an operator using this install the normal way*; it is not a
defence against a determined operator with filesystem access. If you have looked at a
candidate by any of these routes, treat it as measured: retrain or register a fresh adapter
before committing a bar for it.

---

## 7. Serve the promoted adapter *(hardware)*

Once promoted, HEARTH's router resolves task → promoted adapter and the `MLXProvider`
hot-swaps it per request (Phase 4). Start the gateway with the MLX backend and route an
`extract` task; it degrades to base weights if the adapter fails to load.

```sh
HF_HUB_OFFLINE=1 HEARTH_BACKEND=mlx uv run --no-sync hearth serve
```

---

## What CI can and cannot verify

| Step | CI-safe? | Why |
| ---- | -------- | --- |
| 1 dataset build/validate | ✅ | Pure Python (`hearth.training.dataset`), no model. |
| 2 golden set build | ✅ | Pure Python / a JSONL file. |
| 3 real train | ❌ | Needs Apple-Silicon GPU + cached weights + `[mlx]`. |
| 4 prereg init/check | ✅ | No model; `check` needs the file committed in a git repo. |
| 5 eval | ⚠️ | The plumbing and every refusal run on the `echo` backend; a meaningful score needs real weights. |
| 6 promote | ⚠️ | Every refusal is CI-safe (`tests/test_cli_eval.py`, `tests/test_cli_training.py`); a real promotion needs a real, significant lift. |
| 7 serve with adapter | ❌ | Needs the MLX backend + real weights. |
