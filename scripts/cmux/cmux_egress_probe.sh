#!/usr/bin/env bash
# cmux_egress_probe.sh — dynamic egress verification for a running cmux (C0 §9, docs/cmux/AUDIT.md).
#
# The static audit (docs/cmux/AUDIT.md) established WHAT code paths can egress and how to close them.
# This probe confirms WHAT ACTUALLY DOES, by watching a running cmux's established TCP connections and
# flagging anything that is not loopback. It cannot build or sign in/out of cmux for you — those are
# the interactive states you drive; the probe reports what it observes in whatever state you launch.
#
# It does NOT itself make any network connection. It only reads local process/socket state (lsof) and,
# if available, nettop. Safe to run on the confidential machine.
#
# Usage:
#   scripts/cmux/cmux_egress_probe.sh                 # watch all cmux processes for 60s, list non-loopback conns
#   scripts/cmux/cmux_egress_probe.sh --seconds 300   # longer capture (e.g. idle baseline)
#   scripts/cmux/cmux_egress_probe.sh --pattern cmux  # override the process-match pattern
#
# Exit status: 0 = only loopback/local connections observed (SEALED-clean for this run);
#              3 = at least one off-box connection observed (NOT sealed);
#              2 = no matching cmux process found.
#
# The four states to run (see AUDIT.md §9), recording each result in AUDIT.md §10:
#   1. signed-out + sealed flags on   -> expect exit 0 (the seal works)
#   2. no flags (negative control)    -> expect exit 3 seeing posthog/sentry/sparkle (proves the probe detects egress)
#   3. signed-in                      -> expect exit 3 seeing *.relay.cmux.dev (proves sign-out is load-bearing)
#   4. loopback-only firewall on      -> expect exit 0 even under state 2 (proves the OS backstop fails closed)

set -euo pipefail

SECONDS_TO_WATCH=60
PATTERN="cmux"
while [ $# -gt 0 ]; do
  case "$1" in
    --seconds) SECONDS_TO_WATCH="$2"; shift 2 ;;
    --pattern) PATTERN="$2"; shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 64 ;;
  esac
done

# Hosts the static audit says a DEFAULT (unsealed) cmux reaches (AUDIT.md §5) — annotated in output.
KNOWN_EGRESS_RE='posthog|sentry\.io|relay\.cmux\.dev|cmux\.com|objects\.githubusercontent|suggestqueries|duckduckgo|bing|githubusercontent'

pids="$(pgrep -f "$PATTERN" || true)"
if [ -z "$pids" ]; then
  echo "==> no process matching '$PATTERN' found. Launch cmux first, then re-run." >&2
  exit 2
fi

echo "==> cmux egress probe — watching ALL processes matching '$PATTERN' (re-scanned each sample)"
echo "    initial pids: $(echo "$pids" | tr '\n' ' ')  duration=${SECONDS_TO_WATCH}s"
echo "    (loopback = 127.0.0.1/::1/localhost = local, OK)"
echo "    reference denylist (AUDIT.md §5): posthog / sentry.io / *.relay.cmux.dev / cmux.com / suggestqueries / duckduckgo / bing"
echo

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

# Sample established connections repeatedly over the window; union the results. cmux is MULTI-PROCESS
# (main app + "cmux Helper" + webview/networking children), so re-scan pids EACH sample to catch
# children spawned after start — a single-pid snapshot would miss egress in a helper.
end=$(( $(date +%s) + SECONDS_TO_WATCH ))
while [ "$(date +%s)" -lt "$end" ]; do
  for p in $(pgrep -f "$PATTERN" 2>/dev/null); do
    lsof -nP -iTCP -a -p "$p" -sTCP:ESTABLISHED 2>/dev/null | awk 'NR>1 {print $9}' >> "$tmp" || true
  done
  sleep 3
done

# Non-loopback = anything whose remote endpoint is not 127.0.0.1 / ::1 / localhost.
offbox="$(sort -u "$tmp" | awk -F'->' 'NF==2 {print $2}' \
  | grep -Ev '127\.0\.0\.1|\[::1\]|localhost' || true)"

if [ -z "$offbox" ]; then
  echo "==> RESULT: only loopback/local connections observed. SEALED-clean for this run. (exit 0)"
  exit 0
fi

echo "==> RESULT: OFF-BOX connections observed — NOT sealed: (exit 3)"
echo "$offbox" | while read -r endpoint; do
  [ -z "$endpoint" ] && continue
  if echo "$endpoint" | grep -Eiq "$KNOWN_EGRESS_RE"; then
    echo "    [KNOWN cmux egress] $endpoint"
  else
    echo "    [other off-box]     $endpoint"
  fi
done
echo
echo "    Note: endpoints show as IP:port; resolve with 'nslookup'/'host' or re-run with Little Snitch"
echo "    to see the domain. Cross-reference AUDIT.md §5 for expected default-cmux hosts."
exit 3
