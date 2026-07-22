# HEARTH × cmux — Sealed-tier runbook (C3)

**Phase:** C3 · **Branch:** `cmux/sealed-profile`. How to run a cmux workspace in the **sealed**
(confidential, no-egress) tier and verify it. Implements the C0 §4 seal invariant and ADR-C001/C003/C006.
The launcher **fails closed** — if it cannot prove no-egress, it does not launch.

> **What "sealed" guarantees.** Inside a sealed workspace there is *no* path off the machine: a
> loopback-only pf firewall (the structural seal), a HEARTH with 0 remotes, and a repo that
> classifies sealed. Telemetry/Sparkle/signed-out are defense-in-depth on top. See `docs/cmux/AUDIT.md`.

---

## Components (all under `scripts/cmux/` + `config/cmux/`)

| File | Role |
| --- | --- |
| `scripts/cmux/cmux-sealed` | launcher + fail-closed preflight (`man scripts/cmux/cmux-sealed.1`) |
| `scripts/cmux/cmux-sealed.pf.conf` | pf rules for the loopback-only egress seal |
| `scripts/cmux/tier_classify.py` | tier classifier (ADR-C003), tested in `tests/test_cmux_tier_classify.py` |
| `scripts/cmux/cmux_egress_probe.sh` | dynamic lsof/nettop confirmation (AUDIT §9) |
| `config/cmux/tiers.yaml` | your tier policy (copy from `tiers.example.yaml`) |

---

## One-time setup

```sh
cp config/cmux/tiers.example.yaml config/cmux/tiers.yaml   # then edit for your machine
# rules match the CANONICAL path (symlinks resolved: /tmp -> /private/tmp)

# load the structural egress seal (pf anchor; needs sudo). Recommended: run cmux as a dedicated
# uid and scope the pf rule to it (see comments in cmux-sealed.pf.conf).
scripts/cmux/cmux-sealed --install-firewall
```

## Every sealed session

```sh
# 1. start a sealed HEARTH (loopback, no remotes, offline weights)
scripts/hearth_private.sh &                     # or the explicit env from RUNBOOK_wiring.md §2

# 2. verify posture (fails closed if anything is off)
scripts/cmux/cmux-sealed --check /path/to/confidential/repo
scripts/cmux/cmux-sealed --check --strict /path/to/confidential/repo   # also enforce advisory checks

# 3. launch (only if --check passes)
scripts/cmux/cmux-sealed /path/to/confidential/repo

# 4. confirm loopback-only while you work (AUDIT §9)
scripts/cmux/cmux_egress_probe.sh --seconds 120

# 5. after the session: quit cmux, then purge scrollback (docs/cmux/PRIVACY.md)
rm -rf ~/Library/Application\ Support/cmux/
```

---

## The three mandatory gates (each fails closed — demonstrated)

`cmux-sealed --check` exits non-zero unless **all** pass:

| Gate | Passes when | Demonstrated failure |
| --- | --- | --- |
| **tier** | repo classifies sealed (`tiers.yaml`) | an `open`-classified repo → `require-sealed` exits 3 (refused) |
| **hearth** | `hearth_private.sh --check` passes (0 remotes) | a resolvable remote → non-zero |
| **firewall** | pf anchor `cmux-sealed` loaded with a block rule | anchor absent → `--check` exits 2, refuses to launch |

Advisory (WARN by default, FAIL under `--strict`): `sendAnonymousTelemetry=0`, `SUEnableAutomaticChecks=0`,
`CMUX_CLI_SENTRY_DISABLED=1`+`CMUX_CLAUDE_HOOK_SENTRY_DISABLED=1`, no cmux auth item in keychain (best-effort signed-out).

**Validated (2026-07-21):** with the firewall not yet loaded, `cmux-sealed --check` correctly failed
closed (tier PASS, hearth PASS, firewall FAIL → exit 2). The tier gate refused an `open`-classified
repo (exit 3). Classifier: `tests/test_cmux_tier_classify.py`, 14 tests green.

---

## Why pf and not just app flags (ADR-C006)

The C0 audit found the in-app **browser** and **iroh** transport have **no in-code off switch**. So
the sealed guarantee cannot rest on cmux settings alone — the loopback-only pf firewall is the
structural fence that holds even if a flag regresses or the app is signed in. The app-level flags
(telemetry/Sparkle/sign-out) reduce what *tries* to leave; pf guarantees nothing *can*.

---

## C3 status & next

- ✅ Launcher + fail-closed preflight (`cmux-sealed`), pf template, tier classifier + tests, man page,
  scrollback purge guidance. Fail-closed demonstrated for all three mandatory gates.
- ⏳ On-hardware: load pf, launch a real cmux, confirm `cmux_egress_probe.sh` is loopback-only under a
  confidential-style workload (this is the shared C0 §9 / C3 / C6 dynamic run).
- **Next:** C4 `cmux/orchestrator` (local control loop over cmux's socket, HEARTH-decided) and/or
  C5 `cmux/open-tier` (gated cloud/Docker for non-confidential repos).
