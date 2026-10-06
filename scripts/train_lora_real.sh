#!/usr/bin/env bash
# train_lora_real.sh — end-to-end REAL LoRA training harness for HEARTH (Phase 4, G4).
#
# Runs the ACTUAL fine-tuning path on Apple Silicon: `hearth train` (mlx_lm.lora under the
# hood) → register a candidate adapter → evaluate → `hearth adapters promote` behind the
# eval gate. Everything here calls real, in-repo commands — nothing is faked.
#
# This script CANNOT run in CI or a locked-down sandbox: it needs the [mlx] extra, an
# Apple-Silicon GPU, and a base model already on disk — ~/.hearth/models (`hearth models
# pull`) or the HF cache (network downloads are blocked). It is written to FAIL FAST with
# a clear message when a prerequisite is missing, and it NEVER attempts a network download (it forces
# HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 and verifies the base model is on disk first).
#
# See docs/RUNBOOK_training.md for the full walkthrough and expected artifacts.

set -euo pipefail

# --- defaults (override via flags or env) --------------------------------------------
BASE_MODEL="${HEARTH_BASE_MODEL:-mlx-community/Qwen2.5-Coder-7B-Instruct-4bit}"
DATA="${HEARTH_TRAIN_DATA:-}"
TASK="${HEARTH_TRAIN_TASK:-extract}"
ITERS="${HEARTH_TRAIN_ITERS:-200}"
OUT="${HEARTH_TRAIN_OUT:-}"
GOLDEN=""
PREREG=""
DO_PROMOTE=0

usage() {
  cat <<'USAGE'
Usage: scripts/train_lora_real.sh --data <dataset.jsonl> [options]

Runs a REAL LoRA fine-tune on Apple Silicon and (optionally) promotes the resulting
adapter through HEARTH's eval gate. Requires: `uv sync --extra mlx --extra mcp --extra dev --extra files`, an Apple-Silicon GPU,
and the base model already on disk: ~/.hearth/models (`hearth models pull`) or the HF cache
(this script runs OFFLINE and will not download anything).

Required:
  --data <path>            Dataset JSONL (built by hearth.training.dataset; see the runbook).

Options:
  --base <model-id>        Base model to fine-tune. Default: the HEARTH default 7B coder
                           (mlx-community/Qwen2.5-Coder-7B-Instruct-4bit). Must be on disk.
  --task <name>            Task class the adapter targets (extract|classify|summarize|draft|
                           code). Default: extract.
  --iters <n>              Training iterations. Default: 200.
  --out <dir>              Run output dir. Default: ~/.hearth/train/<timestamp>.
  --promote                After training, measure the candidate and promote it only if it
                           passes the gate: runs `hearth eval <id> --golden G --prereg P
                           --promote`. Requires --golden and --prereg. The prereg must be
                           committed BEFORE this run (this run is the adapter's first
                           measurement) — see docs/RUNBOOK_training.md §4.
  --golden <file>          Golden set JSONL (committed in the evals repository).
  --prereg <file>          Committed pre-registration YAML (`hearth prereg init`).
  -h, --help               Show this help.

Environment equivalents: HEARTH_BASE_MODEL, HEARTH_TRAIN_DATA, HEARTH_TRAIN_TASK,
HEARTH_TRAIN_ITERS, HEARTH_TRAIN_OUT.

Example:
  scripts/train_lora_real.sh --data data/extract.jsonl --task extract --iters 300
  # train, then measure + promote under a bar committed beforehand:
  scripts/train_lora_real.sh --data data/extract.jsonl --task extract \
      --promote --golden data/extract_golden.jsonl --prereg prereg/extract.yaml
USAGE
}

# --- parse args ----------------------------------------------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    --data) DATA="$2"; shift 2 ;;
    --base) BASE_MODEL="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --iters) ITERS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --promote) DO_PROMOTE=1; shift ;;
    --golden) GOLDEN="$2"; shift 2 ;;
    --prereg) PREREG="$2"; shift 2 ;;
    --candidate-score|--incumbent-score)
      echo "error: $1 was removed: a typed score cannot promote an adapter (B-015)." >&2
      echo "       Use --promote --golden <file> --prereg <file>; see docs/RUNBOOK_training.md §4-6." >&2
      exit 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

die() { echo "error: $*" >&2; exit 1; }

# --promote is checked BEFORE training: a missing golden set or prereg must not cost a GPU run.
if [ "${DO_PROMOTE}" -eq 1 ]; then
  [ -n "${GOLDEN}" ] && [ -n "${PREREG}" ] || { echo "error: --promote requires --golden and --prereg" >&2; exit 2; }
  [ -f "${GOLDEN}" ] || { echo "error: golden set not found: ${GOLDEN}" >&2; exit 2; }
  [ -f "${PREREG}" ] || { echo "error: pre-registration not found: ${PREREG}" >&2; exit 2; }
fi

# --- offline enforcement: never touch the network ------------------------------------
# Force HF into offline mode so a missing cache errors out instead of silently downloading.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "==> HEARTH real LoRA training harness"
echo "    base=${BASE_MODEL} task=${TASK} iters=${ITERS}"
echo "    HF_HUB_OFFLINE=${HF_HUB_OFFLINE} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}"

# --- prereq: dataset -----------------------------------------------------------------
[ -n "${DATA}" ] || { echo "error: --data is required" >&2; usage >&2; exit 2; }
[ -f "${DATA}" ] || die "dataset not found: ${DATA}"

# Every `uv run` names the project explicitly: without it uv picks the project from the
# CALLER's working directory, so a run started outside the repo (with no VIRTUAL_ENV, as in a
# fresh terminal) used the wrong interpreter and reported mlx-lm "not installed" (B-032).
# --project rather than `cd`, so relative --data / --out paths keep meaning the caller's cwd.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- prereq: uv + hearth CLI ---------------------------------------------------------
command -v uv >/dev/null 2>&1 || die "uv not found on PATH. Install uv, then: uv sync --extra mlx --extra mcp --extra dev --extra files"

# --- prereq: mlx extra installed (the real training backend) -------------------------
# hearth.training.lora._mlx_lm_runner requires mlx_lm; check it is importable up front so
# we fail with the fix hint before spending GPU time laying out the run dir.
if ! uv run --no-sync --project "$REPO_ROOT" python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('mlx_lm') else 1)"; then
  die "mlx-lm is not installed. Install the training backend with: uv sync --extra mlx --extra mcp --extra dev --extra files"
fi

# --- prereq: Apple Silicon -----------------------------------------------------------
if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  die "real LoRA training needs an Apple-Silicon (arm64 macOS) GPU. Detected: $(uname -s)/$(uname -m)"
fi

# --- prereq: base model on disk (NO network) -----------------------------------------
# Resolved with hearth's own resolver (providers/mlx.py:resolve_local_model, downloads off),
# the one `hearth train` uses: ~/.hearth/models (where `hearth models pull` writes) first,
# then the huggingface hub cache. Checking the hub cache alone rejected pulled models (B-024).
echo "==> Verifying base model is on disk (offline)…"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! uv run --no-sync --project "$REPO_ROOT" python "${SCRIPT_DIR}/check_base_on_disk.py" "$BASE_MODEL"; then
  die "base model '${BASE_MODEL}' is not on disk (looked in ~/.hearth/models and the HF cache).
      Fetch it ONCE, deliberately:
          hearth models pull ${BASE_MODEL}
      then re-run this script (it stays offline)."
fi

# --- train (REAL) --------------------------------------------------------------------
echo "==> Training (this uses the GPU and can take a while)…"
TRAIN_CMD=(uv run --no-sync --project "$REPO_ROOT" hearth train --task "${TASK}" --base "${BASE_MODEL}" --data "${DATA}" --iters "${ITERS}")
[ -n "${OUT}" ] && TRAIN_CMD+=(--out "${OUT}")
echo "    ${TRAIN_CMD[*]}"
"${TRAIN_CMD[@]}"

echo "==> Registered candidate adapter(s):"
uv run --no-sync --project "$REPO_ROOT" hearth adapters list --task "${TASK}"

# --- promote (optional, eval-gated) --------------------------------------------------
if [ "${DO_PROMOTE}" -eq 1 ]; then
  # hearth train names the candidate <task>-<run-id>; the newest one is what we just made.
  ADAPTER_ID="$(uv run --no-sync --project "$REPO_ROOT" hearth adapters list --task "${TASK}" --status candidate \
    | awk 'NR>3 {print $1}' | grep -v '^$' | tail -1 || true)"
  [ -n "${ADAPTER_ID}" ] || die "could not find a candidate adapter to promote for task '${TASK}'"
  echo "==> Measuring ${ADAPTER_ID} under ${PREREG}; promoting only if the gate passes…"
  # Exit status is the gate's: a FAIL or any refusal leaves the adapter a candidate.
  uv run --no-sync --project "$REPO_ROOT" hearth eval "${ADAPTER_ID}" \
    --golden "${GOLDEN}" --prereg "${PREREG}" --promote
  echo "==> Final adapter state:"
  uv run --no-sync --project "$REPO_ROOT" hearth adapters list --task "${TASK}"
fi

echo "==> Done. See docs/RUNBOOK_training.md for how to serve the promoted adapter."
