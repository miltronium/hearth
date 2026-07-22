# HEARTH × cmux integration — master map

> **This directory is the single source of truth for the cmux integration effort.**
> If you are picking this work up cold, read this file top to bottom first. Nothing about
> the effort should live only in someone's head or a chat log — if it matters, it is written
> down here.

**Goal, in one sentence:** bring *all* of cmux's ability and UX into the HEARTH ecosystem —
as a HEARTH-native cockpit and orchestrator — including cloud and Docker capability, with
those egress-capable features **gated** behind a fail-closed sealed/open tier model so
confidential work is structurally prevented from leaving the machine.

This is HEARTH's **second major build** after the standalone Phases 0–7. It does not alter
the standalone guarantee: it is developed on its own branch and merges into `main` only when
**fully verified, confirmed working, successful, and beneficial**.

---

## What cmux is (and the one clarification that governs everything)

`manaflow-ai/cmux` is a **native macOS terminal** (Swift/AppKit, `libghostty` rendering) built
to **orchestrate multiple terminal coding agents in parallel** — Claude Code, Codex, opencode,
Gemini CLI — as first-class vertical tabs and split panes. It provides:

- **Notifications** — panes ring / tabs light up when an agent needs attention (OSC 9/99/777, or a `cmux notify` CLI).
- **A scriptable in-app browser** (from agent-browser) — DOM snapshot, click, fill, eval JS.
- **A Unix-socket API + CLI** — create workspaces, split panes, send keystrokes, **read screen contents**, drive the browser.
- **Session restore** — windows, panes, working dirs, scrollback survive restarts.
- Runs **fully locally**; GPL-3.0; no documented telemetry.

**Governing clarification:** the same vendor (Manaflow) also ships a *separate* capability that
spins up **per-agent workspaces in cloud VMs or Docker**, plus Founders-Edition cloud features
(cmux AI, iOS sync, Cloud VMs). We **want** the Docker and cloud capability too — but those are
egress vectors and must be **gated**, not adopted blindly. Local Docker with no network is an
*isolation feature* and belongs in the sealed tier; cloud VMs are the vector the gate fences.
See [ARCHITECTURE.md](ARCHITECTURE.md) and [PRIVACY.md](PRIVACY.md).

---

## The mental model: cockpit + engine

cmux and HEARTH are **complementary layers, not competitors.**

| | **cmux** | **HEARTH** |
| --- | --- | --- |
| Role | **Cockpit** — multi-pane UI, notifications, browser, automation socket | **Engine** — local inference, routing/escalation, RAG, adapters |
| Interface | Unix socket + CLI, OSC notifications | HTTP (OpenAI) + `hearth` CLI + MCP + Swift SDK |
| Stack | Swift/AppKit, libghostty | Python gateway + Swift embedded (Apple Silicon) |
| Missing piece it needs | a local brain | a front-end + orchestrator |

cmux supplies the UI/CLI/orchestration surface HEARTH never built; HEARTH supplies the local,
token-saving, gate-able brain cmux's agents offload to. Same hardware, both Swift-native.

---

## Document map (read in this order)

| Doc | What's in it |
| --- | --- |
| **README.md** (this) | Master map, status tracker, branch model. Start here. |
| [PROPOSAL.md](PROPOSAL.md) | Why we're doing this: problem, vision, goals/non-goals, success criteria, risks. |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Cockpit/engine design, the two-tier **sealed/open** model, component mapping, the gate. |
| [PRIVACY.md](PRIVACY.md) | The gated privacy model, fail-closed enforcement, the honest structural-vs-policy tradeoff, verification. |
| [ROADMAP.md](ROADMAP.md) | Phased plan (C0–C6), per-phase sub-branch, deliverables, **acceptance gates**. |
| [WORKFLOW.md](WORKFLOW.md) | Branch discipline, merge-up gates, how work graduates to `main`, sub-agent rules. |
| [DECISIONS.md](DECISIONS.md) | ADR log for this effort (ADR-C001…). The *why* behind the big choices. |

---

## Branch model (see [WORKFLOW.md](WORKFLOW.md) for the full discipline)

```
main                     ← standalone HEARTH; cmux merges ONLY when fully proven
archive/hearth-pre-cmux  ← dormant frozen original (branch) + tag archive/hearth-pre-cmux-2026-07-21
cmux/integration         ← the work trunk; sub-branches merge up into it
  cmux/<task>            ← one sub-branch per unit of work (planning-docs, egress-audit, adr, …)
```

> **git naming note:** git cannot have a branch `cmux/integration` *and* branches literally nested
> under it, so all cmux branches are **siblings under the `cmux/` namespace**. They are still
> branched *from* `cmux/integration` (the ancestry that matters); the flat naming is the only
> scheme git permits. Never create a bare branch named `cmux`.

**The graduation rule:** a sub-branch merges into `cmux/integration` when *its* acceptance gate
passes. `cmux/integration` merges into `main` only when the **whole** integration is verified,
working, successful, and beneficial — never piecemeal.

---

## Status tracker

Update this table as phases land. `☐` not started · `◐` in progress · `☑` done & merged to `cmux/integration`.

| Phase | Sub-branch | Deliverable | Gate | Status |
| --- | --- | --- | --- | --- |
| — | `cmux/planning-docs` | This doc set (map + plan) | Docs reviewed & merged to `cmux/integration` | ☑ |
| C0 | `cmux/egress-audit` | cmux egress audit ([AUDIT.md](AUDIT.md)) + `lsof` probe | Static: **done** (123 findings, all disableable). Dynamic `lsof` run: pending real-hardware exec | ◐ |
| C1 | `cmux/adr` | ADRs ratifying the two-tier gated model | ADR-C001…C006 **Accepted**; tier-classification concrete ([tiers.example.yaml](../../config/cmux/tiers.example.yaml)) | ☑ |
| C2 | `cmux/wiring` | HEARTH-as-brain per pane (MCP + OpenAI base_url), config-only | A cmux pane offloads to sealed HEARTH; 0 frontier tokens | ☐ |
| C3 | `cmux/sealed-profile` | Sealed-tier launcher + preflight (cmux analog of `hearth_private.sh`) | Fails closed on any cloud/network path; egress-verified | ☐ |
| C4 | `cmux/orchestrator` | Local control loop over cmux socket, HEARTH-decided | Reads panes → HEARTH triage → notify/drive, fully local | ☐ |
| C5 | `cmux/open-tier` | Gated cloud/Docker tier for non-confidential work | Tier selection default-sealed; cloud reachable only when explicitly classified | ☐ |
| C6 | `cmux/graduation` | Merge to `main` | Full integration proven, working, beneficial; RESULTS captured | ☐ |

Phases are described in full in [ROADMAP.md](ROADMAP.md). This table is the at-a-glance index;
the roadmap holds the detail and acceptance criteria.

---

## Non-negotiables (carried from standalone HEARTH)

1. **`main` always ships a working standalone HEARTH.** The archive tag/branch is the escape hatch.
2. **Default posture is sealed.** Unknown/unclassified repo → sealed tier. You opt *into* cloud, never out of it.
3. **The gate fails closed.** A sealed workspace that cannot *prove* no-egress does not launch.
4. **The caller caveat still holds.** HEARTH seals the subtask on HEARTH; it does not seal a
   frontier agent cmux orchestrates. Confidential-repo panes run local/sealed agents. See PRIVACY.md.
5. **Nothing gets lost.** Every decision, result, and dead-end lands in this directory.
