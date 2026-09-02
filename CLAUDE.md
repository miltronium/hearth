# HEARTH — working notes for agents

Read this before touching the repo. It is the short list of things that have actually bitten
people here, plus the design rules the codebase is held to.

---

## 1. Environment: the venv footgun

**Always run `uv run --no-sync`.** A bare `uv run` (or `uv sync` with a subset of extras)
syncs to the DEFAULT dependency set and **uninstalls 33 packages including `mlx`, `mlx-lm`,
`mcp` and `pytest`** — silently killing local inference. Extras are never default in uv, so
this is permanent, not a transient state.

To install or repair, sync every extra **in one command**; syncing one prunes the others:

```sh
uv sync --extra mlx --extra mcp --extra dev --extra files
```

The README's step-by-step `uv sync --extra dev` … `uv sync --extra mlx` sequence is wrong for
this reason — the second command removes what the first installed.

Verify after any sync:

```sh
uv run --no-sync python -c "import mlx_lm, mcp, openpyxl, pypdf; print('ok')"
```

## 2. Where the models live

`hearth models pull` writes to `~/.hearth/models`; huggingface_hub resolves
`HF_HUB_CACHE` → `HF_HOME/hub` → `~/.cache/huggingface/hub`. These disagreed until
`providers/mlx.py:resolve_local_model`, which checks HEARTH's directory first and then falls
through. Do not "fix" this by exporting `HF_HUB_CACHE` globally — a hidden global makes the
provider disagree with `scripts/hearth_status.py`, which is the bug class below.

Also: the model setting is **`HEARTH_DEFAULT_MODEL`**, not `HEARTH_MODEL`. Settings use
`env_prefix="HEARTH_"` with `extra="ignore"`, so a wrong name is silently discarded and you
will get a result from a different model than you think. `scripts/hearth_status.py` flags
`HEARTH_*` vars nothing reads.

## 3. The bug class this codebase keeps finding

> **A gate must assert on the OUTCOME, never on a CONFIGURATION that implies the outcome.**

Nine instances found so far, all the same shape — the check and the checked thing were
different objects:

| Where | The check said | The reality |
|---|---|---|
| cmux seal (ADR-C009) | `SEALED` | pane children egressed freely |
| HEARTH no-egress | router verified sealed | the *calling agent* had already leaked the file |
| `APEX llm/client.py:156` | privacy gates green | `OLLAMA_HOST` accepted any remote host |
| `gateway/app.py:202` | `finish_reason: "stop"` | output was truncated at 512 tokens |
| `training/eval.py:161` | judge passed | the candidate was merely *longer* |
| `training/eval.py:142` | beats incumbent | any score `> 0.0` promoted |
| `hearth_private.sh` | sealed | verified a hardcoded profile, not the one being run |
| pulled models | downloaded | invisible to the provider |
| a would-be PCC backend | "no router egress path exists" | a live TLS connection |

When adding a check, ask what it would report if the thing it guards were broken in the most
plausible way. If the answer is "still green", it is not a gate.

## 4. Privacy rules (non-negotiable)

- **Never read the operator's real financial data.** Not from `~/hearth-statements/`, not
  from `APEX/FINANCES/`. Build against synthetic fixtures; the operator runs the pipeline.
  An agent reading those files sends them to a cloud provider, which defeats the entire
  architecture. Column *headers* are safe; *values* are not.
- **HEARTH stays no-egress by construction.** `config/routing.private.yaml` and
  `config/routing.finance.yaml` define zero remotes, so the router has structurally nowhere
  to send a task. Do not add a PCC or frontier backend — see `docs/TIERS.md` for why, and
  `src/hearth/handoff/` for the human-carried alternative.
- Packages with a no-network invariant (`hearth.handoff`, `hearth.finance`) enforce it with
  AST tests over their own source. Keep them passing; do not add convenience imports.
- Financial arithmetic is **Decimal in Python, never a model and never a float**. A model may
  categorize (a judgement); it must never compute a figure.

## 5. Status, and the docs

Run `uv run --no-sync python scripts/hearth_status.py` rather than trusting a handoff doc.
It measures: which weights are loadable, which routing profiles are structurally no-egress,
golden-set sizes against the minimum that could ever license a promotion, the GPU working-set
ceiling (30.15 GB on this M3 Pro, not the advertised 36), and how many commits each doc is
behind. Written status rots; this does not. See `docs/STATUS.md`.

Key docs: `docs/TIERS.md` (why tiers 3–4 are not backends) · `docs/LEARNING_plan.md` (the
training/eval audit) · `docs/MODELS_local.md` (model choice, measured) · `docs/APEX_seam.md`
(the sibling project boundary) · `docs/STATUS.md`.

## 6. Concurrency

Other agents edit this repo live. **Stage explicit paths; never `git add -A` or `git add .`**.
`config/cmux/tiers.yaml` and `examples/repos/` are intentionally untracked. If tests fail in
files you do not own, report them — do not fix them.

## 7. Promotion gate

An adapter cannot be promoted on a score you typed. `hearth eval --promote` requires a
`--prereg` that is git-committed and unmodified, evaluation at temperature 0, an incumbent
(the base model when no adapter is promoted), significance over paired per-example vectors
(exact McNemar / paired bootstrap), and beating empty/majority/copy-input baselines.
**n ≥ 5 is the mathematical floor** for any promotion at α=0.05, since the smallest
achievable p is 0.5ⁿ; the gate's default `min_n=30` is a power floor above that.
`tests/test_eval_gate_replay.py` replays this repo's own historical promotion and asserts it
is REFUSED — if that test ever passes the promotion, the gate has stopped being real.
