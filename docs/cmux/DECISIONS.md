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

## ADR-C007 — Socket access: `cmuxOnly` for confidential work, `password` for automation

**Status:** Accepted (2026-08-06) · **Context:** C4 live run.

**Context.** cmux gates its automation socket with `automation.socketControlMode`:

| Mode | Who may drive the socket |
| --- | --- |
| `cmuxOnly` (default) | only processes **descended from cmux** — it injects `CMUX_SOCKET_PASSWORD` into pane shells |
| `password` | any local process presenting the secret |
| `allowAll` | anyone local, unauthenticated |

Under the default, the C4 orchestrator can only run **inside a cmux pane**. That is fine for a human
babysitting one machine, but it blocks the case the orchestrator exists for: an unattended sweep
driven from a script, a scheduler, or an agent session that is not a cmux child.

We initially recorded this as an absolute ("the orchestrator must run in a pane"). It is not — it is a
**default**, and cmux ships a supported way to lift it.

**Decision.** Keep **`cmuxOnly` as the posture for confidential/sealed work**. Use **`password`** for
automation, on non-confidential material, via `scripts/cmux/cmux-auth-env`. **Never `allowAll`.**

Password handling: cmux migrates the secret out of `cmux.json` on launch into
`~/.local/state/cmux/socket-control-password` (mode 0600), which is authoritative. `cmux-auth-env`
reads it at run time and exports `CMUX_SOCKET_PASSWORD`, so the secret stays out of shell history,
committed files, and `ps` output (which `--password` would expose).

**Consequences.** `password` mode trades "must be a cmux descendant" for "must be able to read a 0600
file in `$HOME`" — so **any process running as you** can read pane screens and send keystrokes to
panes. On a single-user Mac that is near what same-uid processes could already reach, but it is
strictly weaker than the default, and pane contents are exactly what the sealed tier protects. Hence
the split by tier rather than a blanket switch. This is consistent with **ADR-C004** (configure >
wrap > patch): a supported setting, no patching. It does **not** weaken **ADR-C006** — socket control
is local-process authorization and is orthogonal to egress; the sealed tier's guarantee still rests on
signed-out + OS-level egress containment.

---

## ADR-C008 — Seal gates must verify ENFORCEMENT, not configuration

**Status:** Accepted (2026-08-17) · **Context:** firewall-hardening session; two live failures in
one hour.

**Context.** The mandatory firewall gate in `cmux-sealed --check` was written to accept a *running*
LuLu/Little Snitch process as the structural seal (ADR-C006 #5). Hardening it on real hardware
surfaced two distinct ways that reasoning fails — both of which reported a seal that did not exist:

1. **Config lies.** LuLu was running, and the gate passed — while LuLu held an explicit
   `ALLOW *:*` rule for `com.cmuxterm.app`. A firewall that is installed and permits the app is not
   a seal. Reading the rule store fixed this.
2. **Rules lie too.** With the rule corrected to `BLOCK *:*`, the rule-reading check reported
   **SEALED** — while LuLu's network extension was `[terminated waiting to uninstall on reboot]` and
   LuLu.app was not running. cmux connected off-box anyway (probe exit 3, `172.182.252.137:443`).
   **A rule that nothing enforces is not a seal.**

A third, independent instance of the same class appeared in the same session: cmux's
`SUEnableAutomaticChecks` had silently regressed from `0` (set 2026-07-22) back to `1`, with no app
update — so even a correctly-applied *app* setting does not stay applied.

**Decision.** Every seal gate must verify the property it claims, as close to the observable outcome
as it can get, and **fail closed on anything it cannot confirm**. Concretely, the layers are ordered
by strength, and a weaker layer may never stand in for a stronger one:

| Layer | Question | Verified by |
| --- | --- | --- |
| 1. config | is the tool configured to block? | `lulu_rule_check.py` rule scan |
| 2. enforcement | is the enforcer actually running? | `lulu_rule_check.py --` extension liveness |
| 3. outcome | did anything actually leave? | `cmux_egress_probe.sh` (authoritative) |

`lulu_rule_check.py` implements 1+2 and returns distinct exit codes (1 = not blocked, 2 =
undetermined, 3 = blocked-but-unenforced) so the gate can say *which* layer failed. Layer 3 remains
the only proof; layers 1–2 exist to stop us from *claiming* a seal we never had.

**Consequences.** The gate is now noisier and will refuse to launch in states it previously waved
through — that is the point. "A security tool is installed" is demoted to evidence, not proof.
This strengthens rather than revises **ADR-C006**: the structural seal is still required; what
changes is that we now *check* it instead of assuming it. The same principle applies to the pf route,
where an anchor can be loaded yet dormant because `/etc/pf.conf` never references it — the identical
failure shape (see `cmux-sealed.pf.conf`).

---

## (Further ADRs land here as later phases surface real constraints — e.g. the exact
## container-network enforcement, or an upstream cmux sealed-mode patch decision.)
