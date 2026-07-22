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

## Run

```sh
# live (needs a running cmux; set the socket for determinism)
export CMUX_SOCKET_PATH=~/.local/state/cmux/cmux.sock
HEARTH_BACKEND=mlx uv run python scripts/cmux/orchestrator.py            # one sweep, notifies
HEARTH_BACKEND=mlx uv run python scripts/cmux/orchestrator.py --dry-run  # triage only, no notify

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

## Status & next

- ✅ Orchestrator + triage on real model + tests + man page + runbook. Config-only; HEARTH untouched.
- ⏳ On-hardware: point it at a live cmux socket and confirm the sweep is loopback-only under
  `cmux-sealed` (shared C0 §9 / C3 / C6 dynamic run). The CLI-client parsing against the live `tree`/
  `list-*` JSON is validated then.
- **Next:** C5 `cmux/open-tier` (gated cloud/Docker for non-confidential repos), then C6 graduation.
