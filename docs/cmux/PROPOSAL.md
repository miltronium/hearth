# HEARTH × cmux — Proposal

**Status:** Draft (planning). The pitch: why bring cmux into HEARTH, what "done" means, and
what we explicitly will *not* do.

---

## The problem

HEARTH is deliberately **headless**. It ships a CLI, an OpenAI-compatible HTTP gateway, an MCP
server, and a Swift/Core ML embedded path — but **no GUI and no orchestrator**. Running several
local/frontier agents in parallel, seeing which one needs attention, driving them, and keeping
each one's work isolated is all left to the user and a pile of terminal tabs.

Separately, the token-savings thesis (offload cheap high-volume work to on-device models, reserve
frontier tokens for genuine reasoning) only pays off when agents actually *route* their grunt work
to HEARTH. Today that wiring is manual and per-agent.

## The opportunity

cmux is, almost exactly, the front-end and orchestrator HEARTH never built — and it has **no
model of its own** in its local edition. It is a native macOS terminal purpose-built to run many
coding agents in parallel with vertical tabs, notifications, a scriptable browser, and a
programmable Unix socket. The two projects compose cleanly:

- **cmux = cockpit.** The UI, the notifications, the automation surface, session restore.
- **HEARTH = engine.** Local inference, routing, escalation policy, RAG, adapters — *and the
  gate* that decides what may leave the machine.

Bringing cmux into HEARTH gives the ecosystem a polished multi-agent cockpit *and* makes the
token-savings offload the default path, while keeping HEARTH's privacy guarantees intact through
a tier model that governs cmux's cloud/Docker capability rather than banning it.

## Vision

> **A HEARTH-native cockpit.** You open one native app, spin up as many agents as the task needs,
> and each one offloads its grunt work to a local HEARTH brain by default. Confidential repos run
> in a **sealed** tier that is structurally incapable of egress; everything else can opt into a
> **cloud/Docker** tier for isolation and scale. A local, HEARTH-decided control loop watches the
> panes and tells you which agent actually needs a human. All of cmux's ability — built for HEARTH.

## Goals

1. **Adopt cmux's full capability set** — parallel panes, notifications, scriptable browser,
   socket automation, session restore, *and* Docker/cloud workspaces.
2. **Make HEARTH the default local brain** behind every pane (MCP + OpenAI base_url), so offload
   is automatic, not manual.
3. **A two-tier gated model** (sealed / open) that lets cloud & Docker exist while keeping
   confidential work provably on-device. Default sealed, fail-closed.
4. **A local orchestrator** built on cmux's socket, using HEARTH for cheap triage/decisions.
5. **Comprehensive, durable docs** so the path is fully defined and nothing is lost.
6. **Zero regression to standalone HEARTH.** `main` keeps shipping the working Phases 0–7 build.

## Non-goals (for now)

- **Forking cmux.** We integrate with and configure cmux; we do not maintain a hard fork unless a
  gate requirement forces a patch we can't achieve via config/wrapper. Any such patch is an ADR.
- **Rebuilding cmux's terminal/UI in HEARTH.** We reuse cmux; we don't reimplement libghostty.
- **Making HEARTH depend on cmux.** The dependency arrow points one way: the cockpit uses the
  engine. HEARTH's conformance suite still passes with no cmux present (same rule as CAMBOT).
- **Shipping the cloud tier for confidential repos.** Cloud is for non-confidential work only.

## Success criteria

The integration graduates to `main` when **all** of these hold:

1. **Works:** a cmux cockpit runs multiple agents, each offloading to HEARTH; the sealed launcher
   starts a confidential workspace and the orchestrator drives panes — end to end, on real hardware.
2. **Verified private:** the sealed tier is `lsof`-clean under load; the gate demonstrably
   fails closed when a cloud/network path is present; egress audit (C0) is documented and re-run green.
3. **Beneficial:** measured frontier-token savings from cockpit offload (the same kind of number
   `hearth stats` already produces), plus a qualitative UX win over the bare-terminal status quo.
4. **Reversible:** the archive tag/branch restores standalone HEARTH cleanly; nothing in `main`
   hard-requires cmux to be installed.
5. **Documented:** RESULTS captured; ADRs ratified; runbooks written so a cold pickup succeeds.

## Risks & mitigations

| Risk | Mitigation |
| --- | --- |
| cmux phones home / auto-updates / cloud features leak | **C0 egress audit** before anything touches a confidential box; sealed launcher disables update/cloud paths; periodic `lsof` spot-check. |
| Misrouting: a confidential repo lands in the cloud tier | **Default-sealed, fail-closed** gate; explicit per-repo opt-in to cloud; unknown → sealed. |
| Weaker guarantee than standalone (gated vs airtight) | Named openly in PRIVACY.md as a conscious tradeoff; sealed tier keeps the airtight path for confidential work. |
| cmux upstream changes break the wiring | Pin a known-good cmux version; wiring is config/wrapper, re-verified per bump. |
| Scope creep swallows the standalone guarantee | Branch isolation + graduation rule: `main` untouched until the whole thing is proven. |
| The caller caveat is forgotten | Restated in every relevant doc; sealed-tier panes run local agents by construction. |

## Why this is worth it

The token-savings thesis and the privacy model are both *already built* in HEARTH — they just lack
a cockpit that makes them the path of least resistance. cmux is that cockpit, and its cloud/Docker
capability is genuinely useful for the majority of (non-confidential) work. Gating rather than
banning that capability is consistent with how HEARTH already treats frontier escalation: the
capability exists, it is fenced, and the fence is verifiable.
