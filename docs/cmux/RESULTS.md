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
  - **Remediation attempted 2026-08-17 — STILL NOT SEALED.** LuLu's rule for `com.cmuxterm.app` was
    changed from `ALLOW *:*` to `BLOCK *:*` (verified in the rule store). cmux was quit, relaunched,
    and watched for 280 s: **probe exit 3**, off-box connection to `172.182.252.137:443`.
  - **Why it failed — the finding (ADR-C008):** LuLu's network extension was
    `[terminated waiting to uninstall on reboot]` and LuLu.app was not running, so **nothing was
    enforcing the rule**. The rule was correct and irrelevant. Note the first version of the new
    rule-reading check reported `SEALED` in exactly this state — which is why the check now also
    verifies extension liveness and returns a distinct exit 3 for "blocked but unenforced".
  - ✅ **CLOSED 2026-08-19 — SEALED.** LuLu enforcement was restored (extension no longer queued for
    removal; `lulu_rule_check.py` exits **0** = `BLOCK *:*` for `com.cmuxterm.app` **and** "network
    extension is loaded and running"). Preflight `cmux-sealed --check --strict` returned **sealed posture
    verified**, all 7 gates PASS. cmux (v0.64.20, pid 96728) was launched cold against the empty
    non-confidential `examples/repos/oss_repo` with the probe already watching:
    `cmux_egress_probe.sh --seconds 280` → **exit 0, "only loopback/local connections observed."**
    A post-hoc `lsof -nP -iTCP -sTCP:ESTABLISHED -a -c cmux` snapshot (the method that caught the
    2026-07-22 false clean) was also empty. LuLu re-verified as still enforcing *after* the run.
  - **Why this run is trustworthy — the controlled comparison.** A clean probe cannot by itself
    distinguish "blocked" from "never attempted", and no LuLu block log was reachable on this machine
    to prove the block fired. But the 2026-08-17 failed run and this one differ in **exactly one
    variable**: both had the `BLOCK *:*` rule set and both had `SUEnableAutomaticChecks=0`; on
    2026-08-17 the extension was dead (**probe exit 3**, `172.182.252.137:443`), today it is live
    (**exit 0**). Enforcement liveness is the only thing that changed, so it is what produced the seal.
  - **The `SUEnableAutomaticChecks=0` confound is closed by precedent:** the 2026-07-22 run had that
    same flag at `0` and **still** egressed to GitHub via Sparkle, so flag state alone does not explain
    today's clean result. This is the same point **ADR-C006/C008** make — flags are not the seal.
  - **Window adequacy:** 280 s, versus the 150 s that missed Sparkle on 2026-07-22 and the 75 s in which
    the §1.5 negative control *did* catch it. The late-Sparkle failure mode had ample room to appear.
  - ⚠️ **SCOPE — added 2026-08-19, read this before citing §1.3.** This result is about
    **`com.cmuxterm.app` only**. It is **not** a workspace-containment result. Measured the same day:
    a pane in this very sealed workspace reached the public internet (`curl → example.com` = **HTTP
    200**) while every seal gate reported `SEALED`/`PASS` — because the LuLu rule is scoped to the cmux
    binary, not to the processes a pane spawns, and `cmux_egress_probe.sh` only watches processes
    matching `-c cmux`, so it cannot see a pane's `curl`. §1.3 stands as written; **do not read it as
    "confidential work cannot leave the machine."** See [FINDING_pane_egress.md](FINDING_pane_egress.md)
    and **ADR-C009**.
- **§1.4 pf backstop (signed-in, firewall on)** — _(paste probe result; expect still loopback-only)_
- **§1.5 negative control** — ✅ **DONE 2026-08-17.** With cmux permitted (LuLu rule still `ALLOW *:*`),
  `cmux_egress_probe.sh --seconds 75` returned **exit 3**, reporting `140.82.116.6:443` =
  `lb-140-82-116-6-sea.github.com`. The probe demonstrably detects egress, so a later exit 0 means
  something. This also **reproduces the 2026-07-22 Sparkle finding** 26 days later, on a machine where
  `SUEnableAutomaticChecks` had silently regressed from `0` back to `1` with no app update
  (same binary, same v0.64.20) — further evidence for **ADR-C006/C008** that app flags do not stay set.
- **§1.6 C2 live pane offload** — ✅ **functional half done (2026-08-17, cmux 0.64.20)**;
  egress half still pending the §1.3 seal.
  - Real cmux pane on a non-confidential empty test dir, driven over the socket
    (`socketControlMode=password`, **ADR-C007**). Both wiring surfaces exercised live:
    **OpenAI** — in-pane request to `$OPENAI_BASE_URL` returned `served_by=local`,
    `escalated=False`, **49** est. frontier tokens saved;
    **MCP** — Claude Code offloaded a 180-line/13 KB log summary, `VERDICT= OFFLOADED` with
    `mcp__hearth__hearth_summarize` in the agent's own `tool_use` transcript (reproduced twice).
  - Assertion is **transcript-based, not self-reported** — the harness
    (`examples/cmux/pane_offload_live.sh`) parses `--output-format stream-json` tool_use records.
    Locality is structural: `allow_escalation=False` + `routing.private.yaml` `remotes: {}`.
  - **Live-only findings (3, all fixed/documented):** (1) the `mcp` extra is required and Claude Code
    drops the server **silently** without it; (2) `uv sync --extra mcp` alone **prunes** mlx/dev —
    sync extras together; (3) `max_words` is a soft hint (~40 words for a 25 limit). Suite after
    the dependency churn: **248 passed, 1 skipped**. Detail in RUNBOOK_wiring §5.
  - ⏳ **Not yet done:** confirm the probe stays clean during the run (needs the §1.3 seal).
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
- **§1.8 workspace containment (pane-child egress)** — ❌ **FAILED 2026-08-19 — new gate, open.**
  The sealed workspace does **not** contain the processes running in it. From a pane in a sealed
  workspace, with `lulu_rule_check.py` at exit 0 and `cmux-sealed --check`'s firewall gate at `PASS`:

  | Probe (sent over the socket into a live pane) | Result |
  | --- | --- |
  | `curl -m 8 https://example.com` (Apple-signed binary) | **http=200** — full round-trip to the public internet |
  | `python3` → TCP `1.1.1.1:443` | **CONNECTED** |
  | same URL from the agent's own shell (outside cmux) | `(56) CONNECT tunnel failed, response 403` |

  The pane was **less contained than the agent session driving it**. Cause: LuLu rules key on
  executable identity, no rule exists for `node`/`claude`/`curl`/`git`/`ssh`/`python3`, and LuLu here
  is allow-by-default (`passiveMode=true`, `allowApple=true`, `allowInstalled=true`, `blockMode=false`).
  Full evidence, root cause, options analysis and next steps:
  [FINDING_pane_egress.md](FINDING_pane_egress.md); decision recorded as **ADR-C009**.
  **Graduation implication:** C6 must not proceed on the current sealed claim.
  Re-run this section once C7 workspace containment lands — the gate is "all three probes fail closed."

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
