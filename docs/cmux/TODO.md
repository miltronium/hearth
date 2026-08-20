# HEARTH × cmux — Parked work & TODO

Tracks work deliberately deferred so we can proceed with the functional integration. Nothing here
blocks the cockpit+engine value (C2 wiring, C4 orchestrator) — it's the confidential-sealing hardening
and the graduation gate. Pick this back up when the sealed tier matters for real confidential use.

---

## 🚨 OPEN — C7 workspace containment (the sealed tier does not contain panes)

**Status:** found 2026-08-19, parked with full context the same day. **This is now the top-priority
item and a graduation blocker** — it outranks the two parked items below.

**The finding in one line:** the seal blocks **cmux's own binary**; it does **not** block the processes
a pane spawns. A pane in a sealed workspace reached the public internet (`curl → example.com` =
**HTTP 200**) while `lulu_rule_check.py` returned exit 0 (`SEALED`) and `cmux-sealed --check`'s
firewall gate reported `PASS`.

**Everything you need is in [FINDING_pane_egress.md](FINDING_pane_egress.md)** — evidence, exact
repro, three-layer root cause, what it does/doesn't invalidate, an options analysis with a
recommendation, tooling + doc deliverables, and an ordered next-steps list. Decision: **ADR-C009**.
Negative result recorded as **RESULTS §1.8**; **§1.3 gained a scope warning** (it stands, but only as a
statement about `com.cmuxterm.app`).

**Start here:** reproduce it first (FINDING §9), confirm the gates still say `SEALED` at the same
moment, *then* prototype §8.1 option 1 (dedicated sealed uid + the existing pf uid anchor) on a
throwaway account.

- [ ] Reproduce + confirm the contradiction (FINDING §9 steps 1–2)
- [ ] Prototype uid-scoped containment; measure with an in-pane probe, not by reasoning
- [ ] `scripts/cmux/pane_egress_probe.sh` + extend `cmux_egress_probe.sh` to cmux's descendants
- [ ] Mandatory `workspace-containment` and `pane-agent` gates in `cmux-sealed --check`
- [ ] `cmux-sealed --purge` (scrollback at rest, still unimplemented per PRIVACY.md)
- [ ] Fill RESULTS §1.8 with the fixed-state re-run; amend ADR-C006; AUDIT §4 6th condition

> **Until this lands: do not put confidential material through cmux.** The functional work below and
> in "ACTIVE" remains fine on non-confidential/empty dirs — that constraint has not changed, only our
> understanding of *why* it is necessary.

---

## PARKED — Sealed-tier egress hardening (completes C3 on-hardware + §1.3/§1.4)

**Status:** paused 2026-07-22 (tedious; not needed for functional work). The *design* is done and the
key finding is validated; what remains is finishing the deny-by-default firewall setup and recording it.

**What we proved on hardware (keep this — it's the important result):**
- With cmux's app-level seal (telemetry-off defaults + signed out), cmux **still** phones home: it
  connected to GitHub (`140.82.116.6` / `185.199.110.215` = github.com / githubusercontent) — **Sparkle's
  launch-time update check**, which `SUEnableAutomaticChecks=false` did **not** stop. Confirms AUDIT §3
  **A5** and validates **ADR-C006**: app flags are insufficient; a structural egress firewall is required.
- The sampling probe (`cmux_egress_probe.sh`) gave a false "clean" (Sparkle fired after the window) →
  a deny-by-default firewall (LuLu / Little Snitch) is the authoritative seal, not the probe.

**Progress 2026-08-17 (see RESULTS §1.3/§1.5 + ADR-C008):**
- ✅ §1.5 negative control captured (probe exit 3 → github.com); Sparkle finding reproduced.
- ✅ LuLu rule for cmux set to `BLOCK *:*` (was `ALLOW *:*` — the gate had been passing on that).
- ✅ `cmux-sealed --check` firewall gate hardened: verifies the rule store **and** extension liveness
      (`scripts/cmux/lulu_rule_check.py`, 15 tests), fails closed with distinct exit codes.
- ❌ §1.3 was still open at end of 2026-08-17 — the block did not take effect: LuLu's network extension
      was `[terminated waiting to uninstall on reboot]`, so nothing enforced the rule (probe exit 3).

**Progress 2026-08-19 — §1.3 CLOSED (see RESULTS §1.3):**
- ✅ LuLu enforcement restored — `lulu_rule_check.py` exits **0** (BLOCK rule *and* extension live).
- ✅ `cmux-sealed --check --strict` → sealed posture verified, all 7 gates PASS.
- ✅ **RESULTS §1.3 filled** — cmux launched cold under the block, `cmux_egress_probe.sh --seconds 280`
      → **exit 0, loopback-only**; post-hoc `lsof` snapshot also empty. The 2026-08-17 (exit 3) and
      2026-08-19 (exit 0) runs differ in exactly one variable — extension liveness — which is what
      makes this a seal rather than a quiet app.

**TODO to finish:**
- [ ] Sign into cmux with LuLu active, confirm `*.relay.cmux.dev` (iroh) is blocked → fill **RESULTS §1.4** (backstop proof).
- [ ] (optional) Investigate fully neutralizing Sparkle without a firewall — likely needs removing `SUFeedURL`
      from `Info.plist` (breaks notarization → re-sign) or a launch wrapper; LuLu is simpler, so low priority.
- [ ] (optional) pf route: a named anchor is enforced only if `/etc/pf.conf` references `anchor "cmux-sealed"`.
      Document the exact /etc/pf.conf edit, or keep recommending LuLu (current stance).

## PARKED — C5 open-tier live cloud/Docker workspace

**Status:** guard + launcher done and gate-tested; the live run needs a cloud account.
- [ ] Spin up a real cloud VM / networked-Docker workspace via `cmux-open` for a non-confidential repo → fill **RESULTS §2.2**.

## PARKED — C6 graduation to `main`

**Status:** gated on the two above **plus C7 workspace containment** (added 2026-08-19 — a sealed tier
that does not contain its panes cannot graduate on a privacy claim). Do NOT merge
`cmux/integration` → `main` until:
- [ ] **C7 workspace containment closed (RESULTS §1.8 green)** — see the OPEN section at the top.
- [ ] RESULTS §1.3 + §1.4 sealed (LuLu), §2.2 open workspace ran, graduation checklist all green.
- [ ] Then follow `RUNBOOK_onhardware.md` Part 3 merge/tag commands.

---

## ACTIVE — proceed now (no firewall needed)

These are the functional cockpit+engine wins; safe to do with empty test dirs (no confidential data):
- ✅ **C2 live — DONE 2026-08-17.** A real Claude Code agent in a real cmux pane offloaded a log
  summary to local HEARTH over MCP (`VERDICT= OFFLOADED`, proven from the agent's own `tool_use`
  transcript), and the OpenAI surface served in-pane at `served_by=local`. Harness:
  `examples/cmux/pane_offload_live.sh`. Three live-only findings fixed/documented (silent MCP-extra
  failure, `uv sync` extra-pruning, soft `max_words`). See **RESULTS §1.6** and RUNBOOK_wiring §5.
  - [ ] Remaining C2 sliver: re-run the offload under `cmux-sealed` + probe to prove loopback-only.
        Blocked on the parked firewall work above, not on C2 itself.
  - [ ] (HEARTH-side, optional) add a path-taking `hearth_summarize` variant — today the tool takes
        `text`, so a pane must Read the file into its own context first, capping the savings.
- ✅ **C4 live — DONE 2026-08-06.** `orchestrator.py` drove the live socket: enumerated real panes,
  triaged 4/4 correctly on-device, fired 3 real notify badges. Three live-only bugs found and fixed
  (stale positional surface refs, `--dry-run` miscount, swallowed cmux errors). See **RESULTS §1.7**
  and RUNBOOK_orchestrator "Live validation". Access via `socketControlMode=password` (**ADR-C007**).
  - [ ] Remaining C4 sliver: re-run the sweep under `cmux-sealed` + probe to prove loopback-only.
        Blocked on the parked firewall work above, not on C4 itself.
- Anything else that makes cmux + HEARTH useful together (browser-DOM→HEARTH summarize, notification
  triage, socket-driven automation).

> Reminder for confidential use later: none of the ACTIVE work above is sealed — do it on
> non-confidential repos until the PARKED firewall hardening is finished.
