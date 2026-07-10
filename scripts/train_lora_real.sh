#!/usr/bin/env bash
# train_lora_real.sh — end-to-end REAL LoRA training harness for HEARTH (Phase 4, G4).
#
# Runs the ACTUAL fine-tuning path on Apple Silicon: `hearth train` (mlx_lm.lora under the
# hood) → register a candidate adapter → evaluate → `hearth adapters promote` behind the
# eval gate. Everything here calls real, in-repo commands — nothing is faked.
#
# This script CANNOT run in CI or a locked-down sandbox: it needs the [mlx] extra, an
# Apple-Silicon GPU, and a base model already present in the local HF cache (network
# downloads are blocked). It is written to FAIL FAST with a clear message when a
# prerequisite is missing, and it NEVER attempts a network download (it forces
# HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 and verifies the base model is cached first).
#
# See docs/RUNBOOK_training.md for the full walkthrough and expected artifacts.

set -euo pipefail

# --- defaults (override via flags or env) --------------------------------------------
BASE_MODEL="${HEARTH_BASE_MODEL:-mlx-community/Qwen2.5-Coder-7B-Instruct-4bit}"
DATA="${HEARTH_TRAIN_DATA:-}"
TASK="${HEARTH_TRAIN_TASK:-extract}"
ITERS="${HEARTH_TRAIN_ITERS:-200}"
OUT="${HEARTH_TRAIN_OUT:-}"
CANDIDATE_SCORE="${HEARTH_CANDIDATE_SCORE:-}"
INCUMBENT_SCORE="${HEARTH_INCUMBENT_SCORE:-}"
DO_PROMOTE=0

usage() {
  cat <<'USAGE'
Usage: scripts/train_lora_real.sh --data <dataset.jsonl> [options]

Runs a REAL LoRA fine-tune on Apple Silicon and (optionally) promotes the resulting
adapter through HEARTH's eval gate. Requires: `uv sync --extra mlx`, an Apple-Silicon GPU,
and the base model already cached under ~/.cache/huggingface (this script runs OFFLINE and
will not download anything).

Required:
  --data <path>            Dataset JSONL (built by hearth.training.dataset; see the runbook).

Options:
  --base <model-id>        Base model to fine-tune. Default: the HEARTH default 7B coder
                           (mlx-community/Qwen2.5-Coder-7B-Instruct-4bit). Must be cached.
  --task <name>            Task class the adapter targets (extract|classify|summarize|draft|
                           code). Default: extract.
  --iters <n>              Training iterations. Default: 200.
  --out <dir>              Run output dir. Default: ~/.hearth/train/<timestamp>.
  --promote                After training, promote the candidate. Requires --candidate-score.
  --candidate-score <f>    Candidate eval score in [0,1] proving it beat the incumbent.
  --incumbent-score <f>    Incumbent eval score (omit if no incumbent for this task).
  -h, --help               Show this help.

Environment equivalents: HEARTH_BASE_MODEL, HEARTH_TRAIN_DATA, HEARTH_TRAIN_TASK,
HEARTH_TRAIN_ITERS, HEARTH_TRAIN_OUT, HEARTH_CANDIDATE_SCORE, HEARTH_INCUMBENT_SCORE.

Example:
  scripts/train_lora_real.sh --data data/extract.jsonl --task extract --iters 300
  # inspect: hearth adapters list --task extract
  # eval it (see runbook), then:
  scripts/train_lora_real.sh --data data/extract.jsonl --task extract \
      --promote --candidate-score 0.82 --incumbent-score 0.71
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
    --candidate-score) CANDIDATE_SCORE="$2"; shift 2 ;;
    --incumbent-score) INCUMBENT_SCORE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

die() { echo "error: $*" >&2; exit 1; }

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

# --- prereq: uv + hearth CLI ---------------------------------------------------------
command -v uv >/dev/null 2>&1 || die "uv not found on PATH. Install uv, then: uv sync --extra mlx"

# --- prereq: mlx extra installed (the real training backend) -------------------------
# hearth.training.lora._mlx_lm_runner requires mlx_lm; check it is importable up front so
# we fail with the fix hint before spending GPU time laying out the run dir.
if ! uv run python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('mlx_lm') else 1)"; then
  die "mlx-lm is not installed. Install the training backend with: uv sync --extra mlx"
fi

# --- prereq: Apple Silicon -----------------------------------------------------------
if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  die "real LoRA training needs an Apple-Silicon (arm64 macOS) GPU. Detected: $(uname -s)/$(uname -m)"
fi

# --- prereq: base model already cached (NO network) ----------------------------------
# We verify the base model resolves from the local HF cache with downloads disabled. If it
# is not cached, huggingface_hub raises under HF_HUB_OFFLINE=1 and we abort with guidance
# instead of hanging or (worse) downloading.
echo "==> Verifying base model is cached (offline)…"
if ! uv run python - "$BASE_MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo = sys.argv[1]
try:
    # local_files_only mirrors HF_HUB_OFFLINE=1: resolve from cache or raise.
    path = snapshot_download(repo_id=repo, local_files_only=True)
except Exception as exc:  # noqa: BLE001 - surface any cache-miss as a clean failure
    print(f"NOT CACHED: {exc}", file=sys.stderr)
    sys.exit(3)
print(path)
PY
then
  die "base model '${BASE_MODEL}' is not in the local HF cache.
      Pre-warm it ONCE from an unrestricted network, e.g.:
          HF_HUB_OFFLINE=0 uv run huggingface-cli download ${BASE_MODEL}
      or: hearth models pull ${BASE_MODEL}
      then re-run this script (it stays offline)."
fi

# --- train (REAL) --------------------------------------------------------------------
echo "==> Training (this uses the GPU and can take a while)…"
TRAIN_CMD=(uv run hearth train --task "${TASK}" --base "${BASE_MODEL}" --data "${DATA}" --iters "${ITERS}")
[ -n "${OUT}" ] && TRAIN_CMD+=(--out "${OUT}")
echo "    ${TRAIN_CMD[*]}"
"${TRAIN_CMD[@]}"

echo "==> Registered candidate adapter(s):"
uv run hearth adapters list --task "${TASK}"

# --- promote (optional, eval-gated) --------------------------------------------------
if [ "${DO_PROMOTE}" -eq 1 ]; then
  [ -n "${CANDIDATE_SCORE}" ] || die "--promote requires --candidate-score (prove the eval gate passed)"
  # hearth train names the candidate <task>-<run-id>; the newest one is what we just made.
  ADAPTER_ID="$(uv run hearth adapters list --task "${TASK}" --status candidate \
    | awk 'NR>3 {print $1}' | grep -v '^$' | tail -1 || true)"
  [ -n "${ADAPTER_ID}" ] || die "could not find a candidate adapter to promote for task '${TASK}'"
  echo "==> Promoting ${ADAPTER_ID} (candidate=${CANDIDATE_SCORE} incumbent=${INCUMBENT_SCORE:-none})…"
  PROMOTE_CMD=(uv run hearth adapters promote "${ADAPTER_ID}" --candidate-score "${CANDIDATE_SCORE}")
  [ -n "${INCUMBENT_SCORE}" ] && PROMOTE_CMD+=(--incumbent-score "${INCUMBENT_SCORE}")
  "${PROMOTE_CMD[@]}"
  echo "==> Final adapter state:"
  uv run hearth adapters list --task "${TASK}"
fi

echo "==> Done. See docs/RUNBOOK_training.md for how to serve the promoted adapter."
