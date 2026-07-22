# HEARTH × cmux — Roadmap

**Status:** Draft (planning). Phased build plan. Each phase = one sub-branch off
`cmux/integration`, one deliverable, one **acceptance gate** that must pass before it merges up.
`cmux/integration` merges to `main` only at C6, when the *whole* integration is proven.

Mirrors the standalone HEARTH roadmap style (`docs/ROADMAP.md`): deliverables + acceptance criteria,
nothing "done" until its gate is green.

---

## Phase map

| Phase | Name | Sub-branch | Depends on |
| --- | --- | --- | --- |
| — | Planning docs | `cmux/planning-docs` | — |
| C0 | Egress audit | `cmux/egress-audit` | — |
| C1 | Ratify decisions (ADRs) | `cmux/adr` | C0 |
| C2 | HEARTH-as-brain wiring | `cmux/wiring` | C1 |
| C3 | Sealed tier + gate | `cmux/sealed-profile` | C0, C1 |
| C4 | Local orchestrator | `cmux/orchestrator` | C2, C3 |
| C5 | Open (cloud/Docker) tier | `cmux/open-tier` | C1, C3 |
| C6 | Graduation to `main` | `cmux/graduation` | all |

Phases C2/C3 can proceed in parallel after C1; C4 needs both; C5 needs the gate (C3).

---

## Phase — Planning docs  ·  `cmux/planning-docs`

**Deliverable:** this `docs/cmux/` set (README, PROPOSAL, ARCHITECTURE, PRIVACY, ROADMAP, WORKFLOW,
DECISIONS) — the full map so the path is defined and nothing is lost.

**Gate:** docs reviewed and merged into `cmux/integration`.

---

## Phase C0 — Egress audit  ·  `cmux/egress-audit`

**Why first:** cmux is third-party code that will run on a confidential machine. Before we trust it,
we must know exactly what it sends off-box. This is the gate before cmux touches any confidential repo.

**Deliverable:** `docs/cmux/AUDIT.md` — a grounded report:
- cmux cloned to a scratch location (not the HEARTH repo), built/run in isolation.
- **`lsof`/network capture** of a running cmux doing nothing, then driving a local agent.
- **Source/config grep** for: update checks, telemetry/analytics, cloud-SDK endpoints, the browser's
  fetch paths, cmux AI / Founders calls, SSH/remote code paths.
- An **enumerated list of every outbound path**, each labeled: always-on / opt-in / cloud-feature.
- The concrete list of knobs the **sealed launcher must disable** (feeds C3).

**Acceptance gate:**
- Every outbound path enumerated and classified. ✅ **Done** — `docs/cmux/AUDIT.md` (123 verified
  findings; 26 blockers, all disableable; native-core-vs-cloud split; seal invariant).
- A "sealed candidate" config demonstrated `lsof`-clean at idle and under a local-agent workload.
  ⏳ **Pending** — `scripts/cmux/cmux_egress_probe.sh` written; needs a real-hardware run (build +
  launch cmux under the probe in the four AUDIT §9 states). This is the remaining C0 sub-step.
- Report reviewed; any blocker (e.g. an un-disableable phone-home) escalated as an ADR before proceeding.
  ✅ **Done** — no blocker is un-disableable, but the two no-code-switch paths (browser, iroh)
  drove **ADR-C006** (config-only seal insufficient → signed-out + OS-level containment).

---

## Phase C1 — Ratify decisions (ADRs)  ·  `cmux/adr`

**Deliverable:** ADRs in `docs/cmux/DECISIONS.md` making the planning choices binding:
- ADR-C001 two-tier gated model (sealed default, fail-closed).
- ADR-C002 cockpit/engine boundary (dependency one-way; no HEARTH→cmux coupling).
- ADR-C003 tier-classification mechanism (how a repo is marked sealed vs open).
- ADR-C004 configure > wrap > patch policy for cmux.
- (further ADRs as C0 findings demand.)

**Acceptance gate:** each ADR has context/decision/consequences and a status; the tier-classification
mechanism (C003) is concrete enough to implement in C3/C5.

---

## Phase C2 — HEARTH-as-brain wiring  ·  `cmux/wiring`

**Deliverable:** config-only integration so a cmux pane's agent offloads to HEARTH:
- MCP registration (`hearth mcp`) for MCP-speaking agents (Claude Code).
- `OPENAI_BASE_URL=http://127.0.0.1:8080/v1` recipe for OpenAI-shaped agents.
- A `docs/cmux/RUNBOOK_wiring.md` and example config under `examples/cmux/`.

**Acceptance gate:**
- A live cmux pane offloads a summarize/classify subtask to a **sealed** HEARTH (escalation off) —
  demonstrated, with `hearth stats` showing the frontier tokens saved (0 spent locally).
- No changes required to HEARTH's own code (config-only), or any change is minimal + tested.

---

## Phase C3 — Sealed tier + gate  ·  `cmux/sealed-profile`

**Deliverable:** the cmux-side analog of `scripts/hearth_private.sh`:
- `cmux-sealed` launcher: opens a workspace with cloud off, Docker panes `--network none`, HEARTH on
  `routing.private.yaml`, loopback bind.
- `cmux-sealed --check`: fail-closed preflight verifying all of the above (reuses
  `hearth_private.sh --check` underneath).
- Container no-network **verification** (not just the flag).
- Scrollback/session-state retention + purge guidance finalized in `docs/cmux/PRIVACY.md`.

**Acceptance gate:**
- `cmux-sealed --check` exits non-zero if *any* egress path is present (cloud endpoint set, a
  networked container, a resolvable HEARTH remote, non-loopback bind) — demonstrated for each failure.
- A sealed workspace is `lsof`-clean at idle and under a real confidential-style workload.
- Fails closed: a deliberately mis-set config does **not** launch.

---

## Phase C4 — Local orchestrator  ·  `cmux/orchestrator`

**Deliverable:** a local control loop over cmux's Unix socket:
- Reads pane screen contents; asks HEARTH (local) to classify "blocked / waiting / done / needs
  human" and/or summarize; drives `cmux notify` and/or sends keystrokes.
- Runs inside the sealed tier without egress (pane contents go only to local HEARTH).

**Acceptance gate:**
- Demonstrated: N panes, orchestrator correctly triages which need attention and notifies — using
  only on-device HEARTH.
- Runs under `cmux-sealed` with the workload still `lsof`-clean.
- Orchestrator has no path that ships pane contents anywhere but local HEARTH (code-reviewed).

---

## Phase C5 — Open (cloud/Docker) tier  ·  `cmux/open-tier`

**Deliverable:** the gated cloud/Docker capability for non-confidential work:
- Tier classification wired (ADR-C003): explicit per-repo opt-in to open; default sealed.
- Cloud VM / networked Docker workspaces reachable **only** when a repo is classified open.
- `hearth stats`-style cost view for open-tier frontier usage.

**Acceptance gate:**
- A non-confidential repo runs an open-tier workspace (cloud/Docker) successfully.
- A confidential (or unclassified) repo **cannot** reach the open tier — the gate blocks it;
  demonstrated.
- Switching a repo sealed→open is explicit and logged; open→ default remains sealed.

---

## Phase C6 — Graduation to `main`  ·  `cmux/graduation`

**Deliverable:** merge the proven integration into `main`.

**Acceptance gate (all of the Proposal's success criteria):**
- **Works** end-to-end on real hardware (cockpit + wiring + sealed launcher + orchestrator; open
  tier for non-confidential).
- **Verified private**: sealed tier `lsof`-clean under load; gate fails closed; C0 audit re-run green.
- **Beneficial**: measured frontier-token savings + qualitative UX win captured in
  `docs/cmux/RESULTS.md`.
- **Reversible**: archive tag/branch still restores standalone; `main` does not hard-require cmux.
- **Documented**: ADRs ratified, runbooks written, RESULTS captured, status tracker in README all `☑`.

Only when every box is checked does `cmux/integration` merge to `main`.

---

## Deferred / future (not in scope for C0–C6)

- Deep Swift-native embedding (cmux "AI" features pointed at HEARTH's embedded Core ML path).
- Upstreaming any sealed-mode patches to cmux.
- Multi-machine / team cockpit.

Logged here so they aren't lost, but explicitly out of the initial build.
