# HEARTH × cmux — Parked work & TODO

Tracks work deliberately deferred so we can proceed with the functional integration. Nothing here
blocks the cockpit+engine value (C2 wiring, C4 orchestrator) — it's the confidential-sealing hardening
and the graduation gate. Pick this back up when the sealed tier matters for real confidential use.

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

**TODO to finish:**
- [ ] Install + approve **LuLu** (or Little Snitch); add a rule blocking **cmux** → any remote.
- [ ] Re-verify: `lsof -nP -iTCP -sTCP:ESTABLISHED -a -c cmux` is **empty** with LuLu active → fill **RESULTS §1.3**.
- [ ] Sign into cmux with LuLu active, confirm `*.relay.cmux.dev` (iroh) is blocked → fill **RESULTS §1.4** (backstop proof).
- [ ] (optional) Investigate fully neutralizing Sparkle without a firewall — likely needs removing `SUFeedURL`
      from `Info.plist` (breaks notarization → re-sign) or a launch wrapper; LuLu is simpler, so low priority.
- [ ] (optional) pf route: a named anchor is enforced only if `/etc/pf.conf` references `anchor "cmux-sealed"`.
      Document the exact /etc/pf.conf edit, or keep recommending LuLu (current stance).

## PARKED — C5 open-tier live cloud/Docker workspace

**Status:** guard + launcher done and gate-tested; the live run needs a cloud account.
- [ ] Spin up a real cloud VM / networked-Docker workspace via `cmux-open` for a non-confidential repo → fill **RESULTS §2.2**.

## PARKED — C6 graduation to `main`

**Status:** gated on the two above. Do NOT merge `cmux/integration` → `main` until:
- [ ] RESULTS §1.3 + §1.4 sealed (LuLu), §2.2 open workspace ran, graduation checklist all green.
- [ ] Then follow `RUNBOOK_onhardware.md` Part 3 merge/tag commands.

---

## ACTIVE — proceed now (no firewall needed)

These are the functional cockpit+engine wins; safe to do with empty test dirs (no confidential data):
- **C2 live** — wire a real cmux pane's agent to HEARTH (MCP / OpenAI base_url), demonstrate offload.
- **C4 live** — point `orchestrator.py` at the live cmux socket; enumerate real panes, triage, notify.
- Anything else that makes cmux + HEARTH useful together (browser-DOM→HEARTH summarize, notification
  triage, socket-driven automation).

> Reminder for confidential use later: none of the ACTIVE work above is sealed — do it on
> non-confidential repos until the PARKED firewall hardening is finished.
