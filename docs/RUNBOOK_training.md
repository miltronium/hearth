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

The harness that automates steps 3 + 6 is `scripts/train_lora_real.sh`
(run `scripts/train_lora_real.sh --help`).

---

## 0. Prerequisites *(hardware)*

- macOS on Apple Silicon (`uname -m` → `arm64`).
- `uv` on PATH.
- The training backend installed:

  ```sh
  uv sync --extra mlx
  ```

- A base model cached locally. HEARTH's default is
  `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` (see `config/models.yaml`). Pre-warm the
  cache **once** from an unrestricted network, then work offline:

  ```sh
  # from an unrestricted terminal (network allowed):
  hearth models pull mlx-community/Qwen2.5-Coder-7B-Instruct-4bit
  # or: huggingface-cli download mlx-community/Qwen2.5-Coder-7B-Instruct-4bit
  ```

- Offline mode for all subsequent steps (load weights from cache, never download):

  ```sh
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  ```

  `scripts/train_lora_real.sh` sets these itself and refuses to run if the base model is
  not already cached, so it can never silently download.

---

## 1. Build (or bring) a dataset *(CI-safe)*

Training reads a JSONL dataset produced by `hearth.training.dataset`. Each record is either
**instruction** shape (`{"prompt": ..., "completion": ...}`) or **chat** shape
(`{"messages": [...]}`); the file may carry a header line (see `dataset.py`). Build one
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
uv run python -c "from hearth.training.dataset import load_dataset; print(len(load_dataset('data/extract.jsonl')))"
```

---

## 2. Assemble a golden eval set *(CI-safe)*

Promotion is gated on a golden set (`hearth.training.eval`). Keep it separate from the
training data. Objective classes (`extract`, `classify`, `summarize`, `rank`) score with
exact-match or token-F1; subjective classes (`draft`, `code`) need a judge hook.

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
HF_HUB_OFFLINE=1 uv run hearth train --task extract \
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
  uv run hearth adapters list --task extract
  ```

  The new adapter shows `status = candidate` and an empty `eval` column.

---

## 4. Evaluate the candidate *(hardware)*

Score the candidate against your golden set. Wire the candidate adapter into a generate
function and use `score_candidate`. Serving a candidate adapter requires the A/B flag — see
`AdapterStore.resolve_path(adapter_id, allow_candidate=True)` — and the per-request adapter
hot-swap in `MLXProvider` (`GenRequest.adapter`). Sketch:

```python
from hearth.training.eval import score_candidate, beats_incumbent, EvalReport

candidate_report = score_candidate(golden, generate_with_candidate, metric="f1")
# `generate_with_candidate` routes each prompt through the candidate adapter (A/B flag on).
incumbent_report = None  # or score the currently-promoted adapter the same way
assert beats_incumbent(candidate_report, incumbent_report), "candidate did not beat incumbent"
print("candidate score:", candidate_report.score)
```

Record `candidate_report.score` (and the incumbent's, if any) — you pass these to the
promote step as *proof the gate passed*.

---

## 5. Confirm the eval gate blocks a regression *(CI-safe)*

The safety guarantee is that a candidate that does **not** beat the incumbent cannot be
promoted. You can prove this with the CLI and no GPU by supplying scores directly:

```sh
# A weaker candidate is refused (gate not passed):
uv run hearth adapters promote extract-badrun --candidate-score 0.40 --incumbent-score 0.71
# -> "Promotion refused: ... candidate did not beat the incumbent"  (exit 1)
```

`hearth adapters promote` computes `beats_incumbent(candidate, incumbent)` and the store
raises `GateNotPassedError` unless it passes (`registry/adapters.py`,
`cli.py:adapters_promote`).

---

## 6. Promote the winning candidate *(hardware for a real adapter; gate logic CI-safe)*

With real scores that clear the gate:

```sh
scripts/train_lora_real.sh --data data/extract.jsonl --task extract \
    --promote --candidate-score 0.82 --incumbent-score 0.71
```

or directly:

```sh
uv run hearth adapters promote extract-<run-id> --candidate-score 0.82 --incumbent-score 0.71
```

Confirm the lifecycle transitioned and any prior promoted adapter for the task was retired
(the store keeps exactly one promoted adapter per task):

```sh
uv run hearth adapters list --task extract
# candidate -> promoted; a previously-promoted adapter for `extract` shows `retired`.
```

The promotion is auditable: `~/.hearth/adapters.json` records the `promotion_proof`
(candidate/incumbent scores + `gate_passed`).

---

## 7. Serve the promoted adapter *(hardware)*

Once promoted, HEARTH's router resolves task → promoted adapter and the `MLXProvider`
hot-swaps it per request (Phase 4). Start the gateway with the MLX backend and route an
`extract` task; it degrades to base weights if the adapter fails to load.

```sh
HF_HUB_OFFLINE=1 HEARTH_BACKEND=mlx uv run hearth serve
```

---

## What CI can and cannot verify

| Step | CI-safe? | Why |
| ---- | -------- | --- |
| 1 dataset build/validate | ✅ | Pure Python (`hearth.training.dataset`), no model. |
| 2 golden set build | ✅ | Pure Python (`hearth.training.eval`). |
| 3–4 real train + eval | ❌ | Needs Apple-Silicon GPU + cached weights + `[mlx]`. |
| 5 gate blocks a regression | ✅ | `beats_incumbent` + `promote` refusal, scores supplied directly. |
| 6 promote lifecycle | ⚠️ | Gate/lifecycle logic is CI-safe with supplied scores; promoting a *real* trained adapter is hardware. |
| 7 serve with adapter | ❌ | Needs the MLX backend + real weights. |

## Prerequisites not yet in the codebase

- There is **no** `hearth eval` CLI command. Step 4 uses the `hearth.training.eval` Python
  API (`score_candidate` / `beats_incumbent`) directly; wire the candidate through the
  MLX provider's per-request adapter slot yourself. If you want a one-command eval, that is
  a genuine follow-up to add, not something to invoke as if it exists.
