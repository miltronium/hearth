# HEARTH × cmux — Decisions (ADR log)

Architecture Decision Records for the cmux integration, numbered `ADR-C###` to keep them distinct
from standalone HEARTH's `docs/DECISIONS.md` (ADR-0xx / ADR-011…). Same format: **Context → Decision
→ Consequences → Status.**

Statuses: `Proposed` · `Accepted` · `Superseded` · `Rejected`. **C1 (`cmux/adr`) is complete:**
ADR-C001…C006 are all **Accepted** (2026-07-21); ADR-C003's mechanism is concrete
(`config/cmux/tiers.example.yaml`). Later phases add ADRs as new constraints surface.

---

## ADR-C001 — Two-tier gated model (sealed default, fail-closed)

**Status:** Accepted (ratified C1, 2026-07-21)

**Context.** We want *all* of cmux's ability, including cloud VMs and Docker workspaces — which are
egress-capable. Standalone HEARTH's privacy model is airtight only because private mode removes
every remote. We need a way to keep cloud/Docker capability without losing confidentiality for
sensitive repos.

**Decision.** Adopt a two-tier model that mirrors HEARTH's `routing.yaml` (open) vs
`routing.private.yaml` (sealed) split, extended to the whole cockpit:
- **Tier 0 sealed** (default): native/Docker-`--network none` panes, HEARTH sealed, cloud off —
  structurally no egress.
- **Tier 1 open** (opt-in per repo): cloud/networked Docker, frontier escalation allowed.
- Default is sealed; unknown/ambiguous ⇒ sealed; you opt *into* open, never out of sealed.
- The sealed launcher is **fail-closed**: it verifies no-egress before opening a confidential
  workspace and refuses otherwise.

**Consequences.** The machine-level guarantee becomes *gated* (weaker than airtight) — correctness
depends on the gate not misrouting. The **sealed tier remains airtight**. We accept the tradeoff
consciously (see PRIVACY.md § "The honest tradeoff") and concentrate rigor on the gate.

---

## ADR-C002 — Cockpit/engine boundary (one-way dependency)

**Status:** Accepted (ratified C1, 2026-07-21)

**Context.** cmux (cockpit) and HEARTH (engine) are complementary. HEARTH's design rule (from
CAMBOT) is that consumers depend on HEARTH, never the reverse, and HEARTH's conformance suite passes
with no consumer present.

**Decision.** The dependency arrow points **cockpit → engine** only. HEARTH exposes no cmux types,
gains no cmux dependency, and its conformance suite continues to pass with no cmux installed. cmux is
a *consumer* of HEARTH like CAMBOT and Claude Code — not a privileged special case.

**Consequences.** Standalone HEARTH stays independently shippable and testable. Integration code
(wiring, launchers, orchestrator) lives on the cmux side / in `examples/` and docs, not baked into
HEARTH's core.

---

## ADR-C003 — Tier classification mechanism

**Status:** Accepted (ratified C1, 2026-07-21). **Artifact:** `config/cmux/tiers.example.yaml`.

**Context.** Something must decide whether a given cmux workspace is sealed or open, with a
fail-safe default, in a language consistent with the rest of HEARTH.

**Decision (concrete).** A single YAML policy file `config/cmux/tiers.yaml` (shipped example:
`config/cmux/tiers.example.yaml`), in the spirit of the existing `routing.*.yaml`, with these keys
and **invariants**:
- `default: sealed` — a workspace with no matching rule is sealed.
- `open:` — a list of opt-in rules (`path` glob and/or `remote_host` pattern). A repo reaches the
  **open** tier *only* by matching one, **and** matching no `sealed_override`.
- `sealed_override:` — rules (`path` glob / `remote_host_contains`) that force **sealed** even when
  an `open` rule matches (the confidential belt-and-suspenders).
- Resolution: by the workspace's working-dir **path** and, when present, its `git remote … origin`
  **host**. **Most-restrictive-wins** — any sealed signal, no match, or ambiguity ⇒ sealed. Open
  never wins a tie. Unknown/unresolvable ⇒ sealed.

The C3 `cmux-sealed` launcher reads this file and **fails closed**: it will not open a workspace in
the open tier unless classification unambiguously resolves to open.

**Consequences.** One mental model for "what may leave the machine" across engine and cockpit.
Unknown repos are sealed by default; open-classification is explicit, auditable, and overridable by
a confidential marker. Implemented/wired in C3 (sealed enforcement) and C5 (open-tier enablement).

---

## ADR-C004 — Configure > wrap > patch

**Status:** Accepted (ratified C1, 2026-07-21)

**Context.** cmux is third-party GPL software. We can integrate by configuring it, wrapping it
(launcher + socket orchestrator), or patching/forking it.

**Decision.** Prefer, in order: **configure** (existing knobs, `.mcp.json`, env) → **wrap**
(launcher/preflight/orchestrator around cmux) → **patch** (fork/PR) only when a gate requirement
can't be met otherwise. Every patch is its own ADR and we prefer upstreaming over carrying a fork.

**Consequences.** We minimize the surface we own and the maintenance burden of tracking cmux
upstream. Pin a known-good cmux version; re-verify wiring per bump.

---

## ADR-C005 — cmux stays out of the HEARTH repo

**Status:** Accepted (ratified C1, 2026-07-21)

**Context.** The audit and build need cmux present, but committing a third-party GPL codebase into
HEARTH would entangle licensing, bloat the repo, and blur the boundary.

**Decision.** cmux is cloned to a scratch location for audit/build and **never committed into
HEARTH**. We commit only *our* artifacts: wiring/config (`examples/cmux/`), launchers, orchestrator,
and docs. Reference cmux by pinned version/commit.

**Consequences.** Clean licensing and boundary. Reproducing the build requires cloning cmux at the
pinned ref (documented in the relevant runbook).

---

## ADR-C006 — Sealing cmux requires signed-out + OS-level egress control (config alone is insufficient)

**Status:** Accepted (ratified C1, 2026-07-21). **Source:** C0 egress audit (`docs/cmux/AUDIT.md`).

**Context.** The C0 audit (123 verified findings) established that cmux's native core is *not*
egress-clean out of the box — a Release build makes always-on connections to PostHog, Sentry (app
*and* CLI, on separate gates), Sparkle, and (when signed in) iroh relay servers. Critically, **two
capabilities have no in-code off switch**: the in-app browser (`BrowserNavigationDelegate.swift:445`
— no local-only mode) and iroh mobile-host (`MobileHostService.swift:542` — no runtime toggle). But
the entire cloud surface (cloud VMs, presence, iroh, billing, push) is gated behind **Stack Auth
sign-in** — signed-out ⇒ none of it activates.

**Decision.** The sealed tier's guarantee rests on **(1) running cmux signed-out + (5) OS-level
loopback-only egress containment** (pf / Little Snitch / restricted launch), with telemetry-off,
auto-update-off, and browser-pinned as **defense-in-depth** (#2–#4 of the AUDIT §4 invariant).
Config/flags alone are **insufficient** because of the no-switch paths. The `cmux-sealed` launcher
(C3) must enforce **and verify** all five conditions and **fail closed** — e.g. refuse to launch if
cmux is signed in or the firewall profile is inactive.

**Consequences.** This refines **ADR-C004**: we will `wrap` cmux *and* depend on OS-level controls,
and may need a small **build-time patch** to stub the browser/iroh for a hardened sealed build
(prefer upstreaming a `--sealed`/local-only mode to cmux). Also establishes a **build-offline,
run-sealed** posture (AUDIT §8): build cmux on an unrestricted machine, run the artifact sealed;
never build/`bun install`/`zig build` on the confidential box. Dynamic `lsof` verification
(`scripts/cmux/cmux_egress_probe.sh`) gates C3 trust.

---

## (Further ADRs land here as C3+ surface real constraints — e.g. the exact container-network
## enforcement, or an upstream cmux sealed-mode patch decision.)
