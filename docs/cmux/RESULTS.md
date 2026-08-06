# HEARTH × cmux — Validation results

Real-hardware validation of the cmux integration. Headless results (unit tests + on-device model
runs) are recorded from the build phases; the **on-hardware** slots are filled during the C6 session
(`docs/cmux/RUNBOOK_onhardware.md`). Graduation to `main` requires every on-hardware slot green.

**Machine:** _(fill: model / chip / RAM / macOS)_ · **cmux ref:** _(commit)_ · **Date:** _____

---

## Headless results (done — from C0–C5)

| Phase | Evidence | Result |
| --- | --- | --- |
| C0 static audit | 23-agent workflow `wf_efc49410-455` | 123 findings; 26 blockers all disableable; seal invariant derived |
| C1 ADRs | `docs/cmux/DECISIONS.md` | ADR-C001…C006 Accepted; tier policy concrete |
| C2 offload | `examples/cmux/offload_demo.py` (MLX) | 4 subtasks, **1053 est. frontier tokens saved**, 0 escalations, all local |
| C3 classifier + gates | `tests/test_cmux_tier_classify.py` (14) | fail-closed demonstrated: open repo→exit 3, pf absent→exit 2 |
| C4 orchestrator | `orchestrator_demo.py` (MLX) + 8 tests | 4 panes triaged correctly, 3/4 flagged, 0 frontier tokens |
| C5 open gate | `cmux-open` live demo | sealed/unclassified refused (exit 2); open granted + logged |
| Suite | `uv run pytest -q` | 246 passed, 1 skipped (standalone HEARTH untouched) |

---

## On-hardware results (fill during the C6 run)

### Part 1 — Sealed tier

- **§1.1 sealed HEARTH + pf loaded** — _(paste `hearth_private.sh --check` + `pfctl -a cmux-sealed -sr`)_
- **§1.2 preflight `--check --strict`** — _(paste output + exit code; expect all PASS, exit 0)_
- **§1.3 loopback-only under load** — ⬅ **primary privacy gate** — _first run (2026-07-22, M-series, cmux
  from brew cask, telemetry-off defaults + signed out):_
  - **App-level seal ONLY → NOT sealed.** With telemetry off + signed out, cmux still opened **one**
    outbound connection: `cmux → 140.82.116.6:443` = `lb-140-82-116-6-sea.github.com` (**GitHub**), i.e.
    **Sparkle's launch-time update check** to `github.com/manaflow-ai/cmux/releases`. `SUEnableAutomaticChecks=false`
    did NOT stop it — confirms **AUDIT §3 A5** (the launch probe is not covered by that flag) and validates
    **ADR-C006** (app flags are insufficient; the structural seal is required).
  - **Sampling probe gave a FALSE "SEALED-clean".** `cmux_egress_probe.sh --seconds 150` reported exit 0,
    but the GitHub connection fired *after* the window (Sparkle delays its first check) and was caught by a
    later `lsof -nP -iTCP -sTCP:ESTABLISHED -a -c cmux` snapshot. → the sampling probe is a quick look, not
    proof; LuLu / continuous capture is authoritative.
  - **Remediation:** install LuLu, block cmux's outbound. _(PENDING re-verify: `lsof -a -c cmux` empty with
    LuLu active → then §1.3 = sealed.)_
- **§1.4 pf backstop (signed-in, firewall on)** — _(paste probe result; expect still loopback-only)_
- **§1.5 negative control** — _(paste result showing the probe DOES see egress, or "skipped")_
- **§1.6 C2 live pane offload** — _(subtask run + confirm probe stayed clean during it)_
- **§1.7 C4 orchestrator on live socket** — ✅ **functional half done (2026-08-06, cmux 0.64.20)**;
  egress half still pending the §1.3 seal.
  - Ran from an ordinary terminal via `socketControlMode=password` + `scripts/cmux/cmux-auth-env`
    (**ADR-C007**) against 4 workspaces parked in distinct states. HEARTH classified **4/4 correctly**
    (done / working / waiting / error), flagged **3/4**, **0 frontier tokens**; three real badges
    confirmed with `cmux list-notifications`. Full table in
    [RUNBOOK_orchestrator.md](RUNBOOK_orchestrator.md) "Live validation".
  - **Live-JSON fixups (3, all fixed + regression-tested):** (1) surface refs are positional and go
    stale between enumerate and notify → client now requests `--id-format both` and uses UUIDs;
    (2) `--dry-run` reported a flat `0/N` flagged (counted notifications sent, not panes warranting
    attention); (3) cmux writes errors to stdout with rc=1, so failures surfaced as a bare "non-zero
    exit status" — `_run` now propagates cmux's message. Suite: **248 passed, 1 skipped**.
  - ⏳ **Not yet done:** re-run the sweep under `cmux-sealed` with the probe to prove it stays
    loopback-only. Blocked on the same parked firewall work as §1.3.

### Part 2 — Open tier

- **§2.1 gate refuses sealed repo** — _(paste `cmux-open --check $CONF_REPO` + exit code; expect REFUSED/2)_
- **§2.2 open workspace works** — _(note cloud/Docker workspace ran + paste audit `open-GRANTED` line)_

---

## Graduation decision

| Criterion | Met? | Note |
| --- | --- | --- |
| Works (sealed cockpit + offload + orchestrator + open workspace) | ☐ | |
| Verified private (§1.3 clean, §1.4 backstop, open gate) | ☐ | |
| Beneficial (offload savings + triage useful; UX win) | ☐ | |
| Reversible (archive tag restores standalone; suite passes sans cmux) | ☐ | |
| Documented (this file complete; README tracker all ☑) | ☐ | |

**Decision:** ☐ graduate `cmux/integration` → `main`  ·  ☐ hold (record blockers above)

_Merge command and tag are in `RUNBOOK_onhardware.md` Part 3._
