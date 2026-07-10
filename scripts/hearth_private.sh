#!/usr/bin/env bash
# hearth_private.sh — run HEARTH in a locked-down, NO-EGRESS mode for confidential work.
#
# Enforces every knob that could leak data OFF this machine, then (unless --check) starts the
# daemon. The privacy posture is verified BEFORE serving, so a misconfiguration fails closed
# rather than silently leaking. See docs/PRIVACY.md for the full model.
#
#   Local-only guarantees enforced here:
#     * HEARTH_ROUTING_YAML=config/routing.private.yaml  -> no remotes, every class local/never
#     * HEARTH_HOST=127.0.0.1                            -> loopback bind, never off-box
#     * HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1        -> load cached weights, never download
#     * HEARTH_BACKEND=mlx                               -> real local inference (no echo stub)
#
# Usage:
#   scripts/hearth_private.sh            # verify posture, then serve on 127.0.0.1:8080
#   scripts/hearth_private.sh --check    # verify posture only (no daemon); exit 0 if sealed
#
# NOTE: this seals HEARTH. It does NOT seal the *calling agent* — an agent that reads
# confidential files into its own context has already handled that data before HEARTH sees
# it. Choose which agent touches which repo accordingly (docs/PRIVACY.md § "The caller caveat").

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ROUTING="config/routing.private.yaml"
export HEARTH_ROUTING_YAML="$REPO_ROOT/$ROUTING"
export HEARTH_HOST="127.0.0.1"
export HEARTH_BACKEND="mlx"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

echo "==> HEARTH private mode — verifying NO-EGRESS posture"
echo "    routing=$HEARTH_ROUTING_YAML host=$HEARTH_HOST backend=$HEARTH_BACKEND"
echo "    HF_HUB_OFFLINE=$HF_HUB_OFFLINE TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"

# 1) loopback bind only.
case "$HEARTH_HOST" in
  127.0.0.1|::1|localhost) ;;
  *) echo "error: HEARTH_HOST=$HEARTH_HOST is not loopback — refusing (would expose off-box)." >&2; exit 1 ;;
esac

# 2) the routing policy actually parses to a no-egress config (no remotes, all local/never).
uv run python - "$HEARTH_ROUTING_YAML" <<'PY'
import sys
from pathlib import Path
from hearth.router.policy import load_policy

policy = load_policy(Path(sys.argv[1]))
problems = []
if policy.remotes:
    problems.append(f"remotes defined: {sorted(policy.remotes)}")
if policy.remote_for() is not None:
    problems.append("a default remote resolves")
escapable = [c for c, r in policy.classes.items() if r.backend != "local" or r.escalate != "never"]
if escapable:
    problems.append(f"classes can leave local: {escapable}")
if problems:
    print("SEALED CHECK FAILED:", "; ".join(problems), file=sys.stderr)
    sys.exit(2)
print("    OK: 0 remotes, no default remote, all classes local/never-escalate")
PY

echo "==> Posture verified: no router egress path exists."

if [ "$CHECK_ONLY" -eq 1 ]; then
  echo "==> --check only; not starting the daemon."
  exit 0
fi

echo "==> Starting sealed HEARTH daemon (loopback, offline, no remotes)…"
exec uv run hearth serve
