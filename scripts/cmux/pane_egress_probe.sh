#!/usr/bin/env bash
# pane_egress_probe.sh — outcome probe for the sealed workspace: what can a CHILD OF A PANE reach?
#
# Thin wrapper over scripts/cmux/pane_containment.py (verdict logic is unit-tested there with a fake
# runner). Design: docs/cmux/DESIGN_C7_containment.md §5; finding: docs/cmux/FINDING_pane_egress.md.
# Unlike cmux_egress_probe.sh (which samples cmux's OWN sockets and cannot see a pane's curl), this
# sends six attempts INTO a pane — curl, TCP/UDP to an IP literal, ICMP, IPv6 literal, DNS — and runs
# the same six outside the seal as a control.
#
# WARNING: OPERATOR-RUN. It makes REAL network attempts. Agents must not run it.
#
# Usage:
#   scripts/cmux/pane_egress_probe.sh --surface SURFACE_ID --sealed-uid UID
#   (or set CMUX_SEALED_SURFACE and CMUX_SEALED_UID; with neither, prints INCONCLUSIVE and exits 4)
#
# Exit status: 0 = CONTAINED    (all blocked in pane, all reached in control, identity matched)
#              3 = LEAKS         (a pane attempt reached the network)
#              4 = INCONCLUSIVE  (offline control, wrong uid/pane, missing tool, not configured)
#              2 = usage error
#
# Limitations: runs at launch only (a rule flushed later is not seen); loopback is trusted, so a
# forwarding listener is invisible; pane ancestry is only checked as "the surface answered".
# A CONTAINED result is a measurement of these six attempts on this run, nothing more.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$HERE/pane_containment.py" "$@"
