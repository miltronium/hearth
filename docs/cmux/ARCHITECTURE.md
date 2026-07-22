# HEARTH × cmux — Architecture

**Status:** Draft (planning). System design for the cockpit/engine integration and the two-tier
gated model. Grounded in the standalone HEARTH design (`docs/ARCHITECTURE.md`, `docs/PRIVACY.md`).

---

## 1. Layered view

```
┌──────────────────────────────────────────────────────────────────────┐
│  COCKPIT  — cmux (native macOS, Swift/AppKit, libghostty)             │
│  • vertical-tab panes (one coding agent each)                         │
│  • notifications (OSC 9/99/777, `cmux notify`)                        │
│  • scriptable browser (DOM snapshot / click / fill / eval)            │
│  • Unix-socket API + CLI (spawn panes, send keys, read screen)        │
│  • session restore                                                     │
└──────────────┬─────────────────────────────┬─────────────────────────┘
               │ each pane runs an agent      │ socket drives orchestration
               ▼                              ▼
┌──────────────────────────┐   ┌─────────────────────────────────────────┐
│  AGENTS (per pane)        │   │  ORCHESTRATOR (local control loop)       │
│  Claude Code / Codex /    │   │  reads pane screens → asks HEARTH        │
│  opencode / Gemini / …    │   │  "blocked? done? next action?" → acts    │
│  or a sealed HEARTH agent │   │  (fully on-device; C4)                   │
└──────────────┬───────────┘   └──────────────────┬──────────────────────┘
               │ MCP / OpenAI base_url            │ router (local, escalation off)
               ▼                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  ENGINE  — HEARTH                                                      │
│  gateway (OpenAI HTTP) · MCP server · router/policy · RAG · adapters   │
│  embedded Swift/Core ML (offline) · the EGRESS GATE (routing policy)   │
└──────────────────────────────────────────────────────────────────────┘
```

**Dependency arrow points one way:** cockpit → engine. HEARTH exposes no cmux types; its
conformance suite passes with no cmux present (same boundary rule as CAMBOT in `docs/INTEGRATION.md`).

---

## 2. Component mapping (cmux capability → HEARTH integration)

| cmux capability | How HEARTH plugs in | Egress class |
| --- | --- | --- |
| Agent pane (terminal coding agent) | Agent registers HEARTH MCP (`hearth mcp`) or `OPENAI_BASE_URL=http://127.0.0.1:8080/v1`; grunt subtasks route on-device | **local** |
| `cmux notify` / OSC notifications | Orchestrator classifies pane output via HEARTH → sets notification urgency | **local** |
| Scriptable browser (DOM snapshot) | Pipe DOM text through `hearth_summarize`/`hearth_extract` before it enters a frontier context | **local** |
| Unix-socket automation | Orchestrator control loop: read screen → HEARTH decide → send keys / spawn panes | **local** |
| Local Docker workspace | Run agent in `docker --network none` (or internal-only net) — isolation without egress | **local (sealed-eligible)** |
| Cloud VM workspace | Remote agent workspace — real egress; **open tier only** | **remote (gated)** |
| SSH / remote panes | Agent + browser routed through remote network — **open tier only** | **remote (gated)** |
| cmux AI / Founders cloud features | Cloud model calls — **open tier only, or disabled** | **remote (gated)** |

The mapping is the heart of the design: **every cmux capability is labeled local or remote**, and
the tier model (below) decides which are reachable in a given workspace.

---

## 3. The two-tier gated model

This mirrors HEARTH's existing `routing.yaml` (open) vs `routing.private.yaml` (sealed) split and
extends it to the whole cockpit.

| | **Tier 0 — sealed** (confidential) | **Tier 1 — open** (non-confidential) |
| --- | --- | --- |
| Panes | native local, or Docker `--network none` | cloud VMs, networked Docker, SSH panes |
| Model backend | HEARTH sealed (`routing.private.yaml`), escalation **off** | frontier escalation allowed (`routing.yaml`) |
| cmux cloud features | disabled | enabled |
| Browser | local only | may route through remote |
| Guarantee | **structural** — no network path exists | **policy-fenced** — remotes exist, gated |
| Default? | **yes** (unknown repo ⇒ sealed) | opt-in per repo |

### Tier selection

- **Default is sealed.** A workspace with no explicit classification is Tier 0.
- **Opt into open** per repo via an explicit classification (mechanism decided in C1/C5 — likely a
  small policy file mapping repo path / git remote → tier, defaulting to sealed; candidate:
  reuse the `routing.*.yaml` convention so there is one policy language, not two).
- **Unknown / ambiguous ⇒ sealed.** Fail safe, not fail open.

### The gate (fail-closed preflight)

The sealed launcher is the cmux-side analog of `scripts/hearth_private.sh`. Before it opens a
sealed workspace it **verifies posture and refuses to launch otherwise**:

1. **No cloud endpoint configured** in cmux for this workspace (cloud/Founders features off).
2. **Containers pinned to no network** — every Docker pane runs `--network none` (or an
   internal-only bridge with no gateway); verified, not assumed.
3. **HEARTH resolves no remote** — the workspace's HEARTH uses `routing.private.yaml`; the same
   check `hearth_private.sh --check` already performs (0 remotes, all classes `local`/`never`).
4. **Loopback only** — any HEARTH bind is `127.0.0.1`.

If any check fails, the sealed workspace **does not start**. This is the whole ballgame: the gate
must be structural (a sealed workspace *cannot* reach cloud) and self-verifying (it *proves* it
before serving), never a toggle a human must remember.

---

## 4. Data flow — sealed workspace (confidential)

```
confidential repo ──▶ cmux pane runs a SEALED HEARTH agent (escalation off)
                         │
                         ├─ subtask offload ──▶ HEARTH router (local) ──▶ on-device model
                         ├─ browser DOM     ──▶ hearth_summarize (local)
                         └─ orchestrator    ──▶ read screen → HEARTH decide (local) → send keys
                      (Docker panes: --network none · HEARTH bind 127.0.0.1 · no cloud endpoint)
                      NOTHING in this path has a route off the machine — verified by the gate.
```

**The caller caveat, restated structurally:** in a sealed workspace the *pane's own agent* is a
local/sealed HEARTH agent, so there is no frontier context to leak into. That is why sealed panes
must not run a bare frontier agent (Claude Code hitting a cloud model) over confidential files — the
gate enforces the local-agent choice rather than trusting the user to make it each time.

## 5. Data flow — open workspace (non-confidential)

```
OSS / non-confidential repo ──▶ cmux pane runs any agent (frontier allowed)
                                   ├─ cloud VM / networked Docker workspace
                                   ├─ escalation to frontier model per routing.yaml
                                   └─ HEARTH still available for cheap offload (cost, not privacy)
```

Here HEARTH is a **cost** optimization (save frontier tokens), not a privacy boundary.

---

## 6. What we build vs configure vs (maybe) patch

| Approach | Where used |
| --- | --- |
| **Configure** cmux (existing knobs, `.mcp.json`, env) | C2 wiring — HEARTH-as-brain per pane |
| **Wrap** cmux (a launcher + preflight script; orchestrator over its socket) | C3 sealed profile, C4 orchestrator |
| **Patch** cmux (fork/PR) | Only if a gate requirement can't be met by config/wrapper — each patch is an ADR, and we prefer upstreaming |

Order of preference: **configure > wrap > patch.** Minimize the surface we own.

---

## 7. Open architecture questions (resolved in later phases)

- **Tier classification mechanism** — policy file vs cmux workspace metadata vs git-remote
  inspection. (C1/C5)
- **Docker no-egress enforcement** — `--network none` vs internal bridge; how the preflight
  *verifies* it rather than trusting the flag. (C3)
- **cmux's own outbound paths** — update checks, browser fetches, cloud SDK — enumerated by the
  C0 audit; determines exactly what the sealed launcher must disable. (C0)
- **Orchestrator transport** — how the local control loop speaks to cmux's Unix socket, and how it
  reads pane screens without shipping them anywhere. (C4)

These are tracked in [ROADMAP.md](ROADMAP.md) and answered as ADRs in [DECISIONS.md](DECISIONS.md).
