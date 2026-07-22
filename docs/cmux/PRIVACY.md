# HEARTH × cmux — Privacy & the gated model

**Status:** Draft (planning). Extends `docs/PRIVACY.md` (standalone HEARTH). Read that first —
this document only covers what *changes* when cmux and its cloud/Docker capability enter the picture.

> **One-line summary.** cmux adds a cockpit and new capabilities — including egress-capable ones
> (cloud VMs, networked Docker, remote panes). We keep those capabilities but put them behind a
> **fail-closed sealed/open tier gate** whose default is sealed. Confidential work runs in the
> sealed tier, which is *structurally* incapable of egress and *proves* it before launching.

---

## The honest tradeoff (read this first)

Standalone HEARTH in private mode offers an **airtight** guarantee: in `routing.private.yaml`
there is *no remote in the process at all* — nowhere for a task to go. Adding cmux with cloud &
Docker capability on the same machine changes the shape of the guarantee for the machine as a whole:

- **Airtight (what standalone sealed HEARTH gives):** the leak path does not exist.
- **Gated (what a dual-tier machine gives):** leak paths exist (cloud VMs, remote panes) but are
  fenced off from confidential work by a gate.

A gated guarantee is **strictly weaker** than an airtight one, because correctness now depends on
the gate never misrouting a confidential repo into the open tier. We adopt this consciously — it
buys real cloud/Docker capability for the large majority of work that isn't confidential — and we
concentrate the rigor exactly where the weakness is: **default-sealed, fail-closed, verified.**

Crucially, **the sealed tier still gives the airtight guarantee.** For a confidential repo, a
sealed workspace runs the same no-remote HEARTH plus network-less containers — the leak path does
not exist *inside that workspace*. The gate's job is to guarantee confidential work only ever runs
there.

---

## What changes vs standalone HEARTH

| Concern | Standalone HEARTH | With cmux |
| --- | --- | --- |
| Egress vectors in scope | 2 (remote escalation; HF weight download) | + cloud VM workspaces, networked Docker, SSH/remote panes, cmux update/telemetry, cmux browser fetches, cmux cloud/AI features |
| Guarantee | airtight in private mode | airtight **inside the sealed tier**; gated at the machine level |
| Third-party code on box | HEARTH only | + cmux (audited in C0) |
| Enforcement | `hearth_private.sh` fails closed | sealed launcher (cmux analog) fails closed on the whole cockpit |

---

## Egress vectors cmux introduces — and how the gate closes each

| Vector | Sealed tier (closed) | Open tier (allowed) |
| --- | --- | --- |
| **Cloud VM workspace** | not configured; preflight refuses launch if a cloud endpoint is set | opt-in per repo |
| **Networked Docker pane** | every container `--network none` (or internal-only, no gateway); verified | networked containers permitted |
| **SSH / remote pane** | disabled | permitted |
| **cmux update check / telemetry** | disabled (from C0 findings) | may run |
| **cmux browser outbound** | local pages only; DOM piped to local HEARTH | may fetch remote |
| **cmux AI / Founders cloud** | disabled | may be enabled |
| **HEARTH escalation** | `routing.private.yaml` — 0 remotes, all `local`/`never` | `routing.yaml` — escalation allowed |

The C0 audit (see ROADMAP) **enumerates cmux's actual outbound paths** so this table is grounded
in what the binary does, not what the README says.

---

## The gate: default-sealed, fail-closed, verified

Three properties, non-negotiable:

1. **Default-sealed.** A workspace with no explicit classification is sealed. You opt *into* the
   open tier per repo; you never opt out of sealed. Unknown/ambiguous ⇒ sealed.
2. **Fail-closed.** The sealed launcher verifies posture *before* opening a confidential workspace
   and **refuses to start** if it can't prove no-egress — a misconfiguration fails closed, never
   silently leaks. This is exactly `hearth_private.sh`'s behavior, extended to the cockpit.
3. **Verified, not assumed.** "No network" on a container is checked, not trusted to a flag; "no
   remote" in HEARTH reuses the existing `hearth_private.sh --check`; "no cloud endpoint" is read
   from cmux config; bind is confirmed loopback.

## The caller caveat — unchanged, now enforced structurally

`docs/PRIVACY.md` § "The caller caveat": HEARTH seals the subtask that runs *on* HEARTH; it does
not seal a frontier agent that already read a confidential file into its own context. cmux does not
change this — but the sealed tier **enforces the right choice structurally**: a sealed workspace's
panes run local/sealed HEARTH agents, so there is no frontier context to leak into. Never run a bare
frontier agent over confidential files in a sealed workspace — and the gate is designed so you
can't.

---

## Verifying no egress yourself (planned — finalized in C3)

The intended verification story, mirroring standalone HEARTH's:

```sh
# 1. Sealed-launcher posture check (the cmux analog of hearth_private.sh --check):
#    - no cloud endpoint configured for this workspace
#    - all Docker panes resolve to no-network
#    - HEARTH routing resolves 0 remotes, all classes local/never
#    - HEARTH bind is loopback
cmux-sealed --check          # (C3 deliverable) exit 0 only if fully sealed

# 2. Reuse HEARTH's own sealed check underneath:
scripts/hearth_private.sh --check

# 3. Watch the whole cockpit make no off-box connections under load:
#    (loopback :8080 is HEARTH itself; expect nothing else)
lsof -nP -iTCP -a -p "$(pgrep -f cmux)"   -sTCP:ESTABLISHED
lsof -nP -iTCP -a -p "$(pgrep -f 'hearth serve')" -sTCP:ESTABLISHED
#    and per confidential container:
docker inspect <ctr> --format '{{json .NetworkSettings.Networks}}'   # expect none / no gateway
```

The exact commands are finalized when the sealed launcher exists (C3) and after the C0 audit tells
us which cmux paths to disable.

---

## Data at rest (additions)

Standalone HEARTH's data-at-rest notes still apply (RAG index, adapters, token — keep `~/.hearth`
on FileVault). cmux adds:

- **cmux session state** (`~/Library/Application Support/cmux/`) — restores panes, working dirs,
  **scrollback**. Scrollback of a confidential pane is a copy of confidential output on disk. Keep
  it on the encrypted volume; know how to purge it. (Retention/purge policy finalized in C3.)
- **Docker volumes / workspace mounts** for sealed panes — same handling as any confidential
  working copy.

---

## Checkpoint / returning later

If you return to this work: run `scripts/hearth_private.sh --check` (still the ground truth for the
engine), then — once it exists — `cmux-sealed --check` for the cockpit, before pointing any
workspace at a confidential repo. Re-read "The honest tradeoff" and "The caller caveat" above. The
durable state lives in git (this doc set, the sealed launcher), `~/.hearth/`, `~/Library/Application
Support/cmux/`, and the agent's local memory — all local on purpose.
