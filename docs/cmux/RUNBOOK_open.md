# HEARTH × cmux — Open-tier runbook (C5)

**Phase:** C5 · **Branch:** `cmux/open-tier`. How to run a cmux workspace in the **open**
(non-confidential) tier — cloud VMs, networked Docker, frontier escalation — for the ~majority of work
that isn't confidential. The gate is the point: the open tier is reachable **only** for repos you
explicitly classify `open`, and `cmux-open` **fails closed to sealed** for everything else.

> This is the other half of the two-tier model (ARCHITECTURE §3). Sealed is the default and the
> fallback; open is opt-in, per-repo, and logged. See `cmux-sealed`(1) for the confidential side.

---

## Components

| File | Role |
| --- | --- |
| `scripts/cmux/cmux-open` | open-tier launcher + fail-closed-to-sealed guard + audit log (`man scripts/cmux/cmux-open.1`) |
| `scripts/cmux/tier_classify.py` | the shared classifier (ADR-C003), `--assert-open` here |
| `config/cmux/tiers.yaml` | your tier policy |
| `$CMUX_TIER_AUDIT_LOG` | append-only log of open grants/refusals (default `~/.hearth/cmux-tier-audit.log`) |

## Opt a (non-confidential) repo into the open tier

Edit `config/cmux/tiers.yaml` — add an `open:` rule that matches the repo, and make sure **no**
`sealed_override` matches it:

```yaml
default: sealed
open:
  - path: "~/oss/**"                      # rules match the CANONICAL path (see tiers.example.yaml)
  - remote_host: "github.com/my-public-org/**"
sealed_override:
  - remote_host_contains: "apple"          # confidential markers ALWAYS win
```

## Run

```sh
scripts/cmux/cmux-open --check ~/oss/my-repo    # verify it may open (exit 0) or is refused (exit 2)
scripts/cmux/cmux-open ~/oss/my-repo            # verify + log the grant + launch cmux in the open tier
scripts/cmux/cmux-open --cost                   # HEARTH escalation/cost rollup for open-tier frontier use
```

In the open tier the launcher sets `HEARTH_ROUTING_YAML=config/routing.yaml` (frontier **escalation
permitted**), installs **no** pf egress seal, and permits cloud/networked-Docker panes.

## The gate (fail-closed to sealed) — demonstrated

| Repo | Classifies | `cmux-open` |
| --- | --- | --- |
| unclassified (e.g. this HEARTH repo) | sealed (default) | **REFUSED**, exit 2, logged `open-REFUSED` ✓ |
| confidential (matches `sealed_override`) | sealed | **REFUSED**, exit 2 ✓ |
| explicit `open:` match, no override | open | allowed, logged `open-GRANTED`, launches ✓ |

**Validated (2026-07-21):** the HEARTH repo (unclassified) was refused (exit 2); a repo with an explicit
`open:` rule was granted and logged. Audit log sample:

```
2026-07-…Z	open-REFUSED	/Users/…/HEARTH	remote=git@github.com:miltronium/hearth.git	user=…
2026-07-…Z	open-GRANTED	/…/oss	remote=-	user=…
```

The `--assert-open` exit-code contract the guard depends on is unit-tested
(`tests/test_cmux_tier_classify.py::test_main_assert_open_exit_codes`).

## Sealed vs open at a glance

| | `cmux-sealed` (C3) | `cmux-open` (C5) |
| --- | --- | --- |
| Requires | repo classifies **sealed** (default) | repo classifies **open** (explicit) |
| On wrong classification | n/a (sealed is default) | **fails closed to sealed** |
| Egress | pf loopback-only seal (mandatory) | none (cloud/Docker allowed) |
| Docker | `--network none` | networked OK |
| HEARTH | `routing.private.yaml` (no remotes) | `routing.yaml` (escalation allowed) |
| Logged | — | every grant + refusal |

## Status & next

- ✅ `cmux-open` launcher + fail-closed-to-sealed guard + audit logging + `--cost` + man page + runbook.
  Both gate outcomes demonstrated. Config-only; HEARTH untouched.
- ⏳ On-hardware: actually spin up a cloud/Docker open workspace (needs the cmux GUI + a cloud account).
- **Next:** C6 graduation — the full on-hardware run (sealed probe + open workspace), then merge
  `cmux/integration` → `main` when everything is proven, working, and beneficial.
