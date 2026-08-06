# HEARTH × cmux — Orchestrator runbook (C4)

**Phase:** C4 · **Branch:** `cmux/orchestrator`. A local control loop that reads each cmux pane's
recent output, asks **on-device HEARTH** what state the pane is in, and fires `cmux notify` for the
panes that need a human — so you can run many agents in parallel and be told which one to look at.
Fully local; safe under the sealed tier.

---

## Files (`scripts/cmux/`)

| File | Role |
| --- | --- |
| `orchestrator.py` | client abstraction (real CLI + fake) + triage + `run_once` sweep + `main` |
| `orchestrator_demo.py` | seeds 4 fake panes, runs the triage on the real MLX model |
| `cmux-orchestrator.1` | man page |
| `tests/test_cmux_orchestrator.py` | 8 tests (decision logic, argv build, browser-skip, dry-run) |

## How it works (one sweep)

1. `cmux --json list-workspaces` + `list-pane-surfaces` → enumerate terminal surfaces (browsers skipped).
2. `cmux --json read-screen --surface <id> --lines N` → the pane's recent text.
3. **Local HEARTH** classifies it into `working | waiting | done | error` (`build_toolset().classify`,
   escalation off), and summarizes a one-line body for panes that warrant a notification.
4. Priority: `waiting`/`error` → **attention**, `done` → **info**, `working` → **none** (quiet).
   Unknown ⇒ none, so the model never produces a spurious notification.
5. `cmux notify --surface <id> --title … --body …` for attention/info panes.

## Run — two ways onto the socket

cmux gates its automation socket with `automation.socketControlMode` (Settings ▸ Automation, or
`~/.config/cmux/cmux.json`). Three values: `cmuxOnly` (default) · `password` · `allowAll`.

**Route A — inside a cmux pane (default, `cmuxOnly`).** The socket accepts only **processes descended
from cmux**; it injects `CMUX_SOCKET_PASSWORD` into the shells it spawns. From an external terminal
you get `Access denied - only processes started inside cmux can connect`, so the orchestrator has to
run in a pane. Nothing to configure.

**Route B — password auth (`password`), any local terminal.** Set the mode to `password` and give it
a secret; then any local process presenting that secret may drive the socket — which is what lets the
orchestrator run from an ordinary shell, a script, or an outside agent session.

```sh
# one-time: set the mode, then relaunch cmux so it picks it up
#   Settings ▸ Automation ▸ Socket control mode → "Password",  or in ~/.config/cmux/cmux.json:
#     "automation": { "socketControlMode": "password", "socketPassword": "<random secret>" }
# On launch cmux MIGRATES the secret out of cmux.json (blanking the key, stamping
# socketControlPasswordMigrationVersion) into ~/.local/state/cmux/socket-control-password, mode 0600.
# That file is the source of truth from then on.

. scripts/cmux/cmux-auth-env      # exports CMUX_SOCKET_PASSWORD, reading the 0600 store
cmux ping                          # expect: PONG
```

`scripts/cmux/cmux-auth-env` reads the secret at run time so it never lands in shell history or a
process argument list (`--password` is visible in `ps`). Source it; don't execute it.

> **Security trade.** Route B is a deliberate relaxation: it swaps "must be a cmux descendant" for
> "must be able to read a 0600 file in your home directory" — i.e. **any process running as you** can
> drive cmux (send keystrokes to panes, read screens). On a single-user machine that is close to what
> same-uid processes could already do, but it is strictly weaker than `cmuxOnly`. Prefer Route A for
> confidential work; see **ADR-C007**. Never use `allowAll` — it drops authentication entirely.

The cmux CLI also isn't on PATH by default; the orchestrator auto-locates it
(`$CMUX_BIN` → PATH → `/Applications/cmux.app/Contents/Resources/bin/cmux`).

```sh
# Route A: open a pane in cmux, then in that pane.  Route B: any terminal, after sourcing cmux-auth-env.
cmux --version                 # confirm the CLI is reachable
cd /path/to/HEARTH
HEARTH_BACKEND=mlx uv run python scripts/cmux/orchestrator.py --dry-run  # triage only, no notify
HEARTH_BACKEND=mlx uv run python scripts/cmux/orchestrator.py            # one sweep, notifies

# offline demo (no cmux GUI): triage 4 realistic panes on the local model
HEARTH_BACKEND=mlx uv run python scripts/cmux/orchestrator_demo.py

# loop it (a sweep every 20s) — a cmux pane itself is a fine host:
while :; do HEARTH_BACKEND=mlx uv run python scripts/cmux/orchestrator.py; sleep 20; done
```

## Privacy (code-reviewed — C4 gate)

Pane text goes to **exactly one place**: the local HEARTH router (`build_toolset`,
`allow_escalation=False`). The cmux client only shells to the local `cmux` CLI (local Unix socket).
`orchestrator.py` imports no network library and has no other outbound path — so under `cmux-sealed`
the sweep stays loopback-only (confirm with `cmux_egress_probe.sh`). **Do not** add a transport that
sends pane contents off-box; that would break the sealed guarantee.

## Validation (2026-07-21, Apple M3 Pro / MLX)

`orchestrator_demo.py` over 4 panes — the local model triaged **all correctly**:

| Pane | Screen | State | Action |
| --- | --- | --- | --- |
| build | compiling (3/7) | working | quiet ✓ |
| adapters | `Delete …? [y/N]` | waiting | 🔔 notify ✓ |
| tests | 238 tests passed | done | 🔔 notify (info) ✓ |
| coreml-fix | `error: cannot find 'writePos'` | error | 🔔 notify ✓ |

**3/4 flagged for attention, 0 frontier tokens.** Deterministic decision logic + CLI argv build:
`tests/test_cmux_orchestrator.py`, 8 green.

## Live validation (2026-08-06, cmux 0.64.20, Route B)

Ran against a **real cmux** with four workspaces deliberately parked in different states, from an
ordinary terminal (not a pane) via password auth. HEARTH triaged **all four correctly**:

| Workspace | Screen | State | Result |
| --- | --- | --- | --- |
| oss_repo | idle shell at prompt | done | 🔔 notify (info) ✓ |
| build-running | `[2/5] compiling module_2.rs` | working | quiet ✓ |
| awaiting-input | `Type yes to continue:` | waiting | 🔔 notify ✓ |
| test-failing | `FAILED: …test_escalation_denied` | error | 🔔 notify ✓ |

**3/4 flagged, 0 frontier tokens**, and three real notification badges confirmed via
`cmux list-notifications`. This closes the on-hardware C4 gate.

Three things the live run corrected, all now fixed in `orchestrator.py`:

1. **Surface refs are positional and go stale.** cmux returns short refs (`surface:1`) by default and
   omits UUIDs. `notify --surface surface:1` failed with `Surface ref not found` even though
   enumeration had just returned that ref. The client now passes `--id-format both` and uses the
   stable UUID. *(The code already preferred `s["id"]` — it just never asked cmux to include it.)*
2. **`--dry-run` always reported `0/N` flagged**, because the summary counted notifications *sent*
   rather than panes *warranting* attention. A correct sweep looked like it had found nothing.
3. **Failures were unreadable.** cmux reports errors on stdout with rc=1, so `check_returncode()`
   surfaced only "returned non-zero exit status 1", hiding `Surface ref not found`. `_run` now
   propagates cmux's own message.

The live `list-workspaces` / `list-pane-surfaces` JSON otherwise matched the mapped shapes
(`workspaces[].ref`, `surfaces[].ref/title/type`) — no parser rewrite was needed.

## Status & next

- ✅ Orchestrator + triage on real model + tests (`tests/test_cmux_orchestrator.py`, 10 green) + man
  page + runbook. Config-only; HEARTH untouched.
- ✅ **On-hardware live run complete** (above) — enumerate → read → triage → notify against cmux 0.64.20.
- ⏳ Still open: confirm the sweep is **loopback-only under `cmux-sealed`** (shared C0 §9 / C3 / C6
  dynamic run). That is part of the parked sealed-tier hardening, not a C4 gap — see [TODO.md](TODO.md).
