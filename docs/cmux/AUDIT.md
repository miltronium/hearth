# HEARTH × cmux — C0 Egress Audit

**Phase:** C0 · **Branch:** `cmux/egress-audit` · **cmux ref:** shallow clone of
`github.com/manaflow-ai/cmux@main` (2026-07-21) · **Method:** 23-agent static source audit
(10 egress dimensions × find + adversarial-verify, plus a completeness critic).

> **Headline verdict.** cmux's native core is **not** egress-clean out of the box: a Release
> build makes **always-on** connections to PostHog (analytics + feature flags), Sentry (crash
> reporting — app *and* CLI, on separate gates), Sparkle (auto-update), and — **when signed in** —
> iroh relay servers. **Every** one is disableable, and the whole *cloud* surface (cloud VMs,
> presence, mobile-host iroh, billing, push, remote panes) is gated behind **one thing: Stack Auth
> sign-in.** That yields a simple, verifiable seal invariant (below). Two capabilities have **no
> in-code off switch** (the in-app browser and iroh), so a config-only seal is **insufficient** —
> the sealed tier must add OS-level egress containment and/or a build-time stub. This refines
> ADR-C004 and adds **ADR-C006**.

Counts: **123 verified findings** — 26 always-on *blockers*, 59 *gated*, 15 *benign*, 23 *info*.
By sealed-verdict: **1 must-exclude, 70 must-disable, 40 safe-if-local, 12 not-in-native-core.**
Full machine-readable results: workflow `wf_efc49410-455` (journal.jsonl); this doc is the synthesis.

---

## 1. Method & confidence

A background Workflow fanned out **one finder per egress dimension** — native-app network calls,
telemetry, auto-update, iroh/p2p, cloud endpoints, cloud-VM/billing, browser/webviews, AI/LLM,
build/supply-chain, CLI/daemon/SSH, plus a broad hardcoded-host sweep. Each finder's output was
**adversarially re-verified** against the actual code (drop false positives, correct
component/trigger/severity, add misses), then a **completeness critic** hunted for anything the
dimensions missed (Info.plist, launch-time paths, SPM deps). Blockers were independently
re-discovered by 4–5 dimensions each — high confidence.

**Scope limit (honest):** this is a **static** source audit. It establishes *what code paths can
egress and how to close them*. The **dynamic** confirmation — building the app and watching it
under `lsof`/`nettop` in each state — is specified in §9 and must be executed on real hardware
before the sealed tier (C3) is trusted. Static + dynamic together is the C0→C3 gate.

---

## 2. What we port vs what we don't

| Bucket | Dirs | Disposition |
| --- | --- | --- |
| **Native terminal core** (port target) | `Sources/` (Swift app), `CLI/`, local automation socket, bundled webview UIs | Port — after sealing (§3–§4) |
| **Cloud / backend / mobile** (exclude) | `web/`, `workers/`, `services/`, `ios/`, `daemon/remote` | Not run locally; open-tier only |
| **Build toolchain** (offline) | `scripts/`, submodules, CI | Build on an unrestricted machine, run the artifact sealed (§8) |

---

## 3. Section A — Always-on egress in the native core (the blockers)

Deduped to the distinct issues (each was found by multiple dimensions). All fire in a **Release**
build **without user opt-in** unless disabled. `file:line` are repo-relative to the cmux clone.

| # | Issue | Host | Where | Trigger | How to disable |
| --- | --- | --- | --- | --- | --- |
| A1 | **PostHog analytics** (active-user beacons, 30-min timer) | `us.i.posthog.com` | `Sources/PostHogAnalytics.swift:14,120`; started `AppDelegate.swift:1417` | always-on (Release; default **opt-out** = ON) | `sendAnonymousTelemetry=false` **before launch** (launch-frozen at `cmuxApp.swift:4717`); or build `defaultValue:false` at `CmuxSettings/.../AppCatalogSection.swift:116`; or blank the embedded PostHog key |
| A2 | **PostHog remote feature-flags** | `us.i.posthog.com` | `Sources/FeatureFlags.swift:242` | always-on | transitively off with A1 (SDK never set up → flags fall back to local defaults) |
| A3 | **Sentry crash/error/hang reporting (app)** | `o4507547940749312.ingest.us.sentry.io` | `Sources/AppDelegate.swift:1371-1408`; also rides `GhosttyTerminalView.swift:721` | always-on | same `sendAnonymousTelemetry=false`; the `if telemetryEnabled { SentrySDK.start }` block then never runs |
| A4 | **Sentry telemetry (cmux CLI)** — *separate gate* | same Sentry host | `CLI/CLISocketSentryTelemetry.swift:55,116` | always-on | **env only:** `CMUX_CLI_SENTRY_DISABLED=1` **and** `CMUX_CLAUDE_HOOK_SENTRY_DISABLED=1`. The app telemetry flag does **not** cover the CLI. |
| A5 | **Sparkle auto-update** (launch probe + 24 h checks) | `github.com` appcast → releases / `objects.githubusercontent.com` | `Resources/Info.plist:245-246` (`SUFeedURL`, `SUEnableAutomaticChecks`); `CmuxUpdater/UpdateController.swift`; started `AppDelegate.swift:1451` | always-on | `SUEnableAutomaticChecks=false` + remove `SUFeedURL` in Info.plist, or don't call `startUpdaterIfNeeded()`, or exclude the `CmuxUpdater` package. **Not** covered by the telemetry flag. |
| A6 | **iroh mobile-host relay** (persistent QUIC to relay fleet + broker) | `*.relay.cmux.dev` + `cmux.com` broker | `Sources/Mobile/MobileHostService.swift:542`; `MobileHostIrohRuntime+Activation.swift:88`; `CmxIrohLibEndpointFactory.swift:78` | **only when signed in** | **No runtime/env off switch.** Sealed only by (a) **not signing in** (activation requires an account) or (b) a build-time stub of iroh activation. `CMUX_IROH_BROKER_BASE_URL` redirects but does not stop it. |
| A7 | **In-app browser — arbitrary remote navigation** | any host | `Sources/Panels/BrowserNavigationDelegate.swift:445` | user/agent-driven | **No local-only/sealed mode exists in code.** Exclude the browser panel, pin to `about:blank`/local scheme, or contain at the OS level. |
| A8 | **Omnibar search suggestions** (streams *typed text* to search engines) | `suggestqueries.google.com`, `duckduckgo.com`, `www.bing.com` (+kagi/startpage) | `Sources/Panels/BrowserPanel.swift:1714,1765` | default **ON** | `browser.showSearchSuggestions=false` (or `CMUX_UI_TEST_DISABLE_REMOTE_SUGGESTIONS=1`) |

**Reading A6/A7:** these are the two with **no code switch**. iroh is *behaviorally* sealable
(sign-out), but the browser is not — a signed-out cmux with telemetry off will still navigate the
in-app browser anywhere it's told. That is why §4 requires OS-level egress enforcement, not just flags.

---

## 4. Section B — The seal invariant (feeds C3)

The audit collapses to a small, verifiable invariant. A **sealed** cmux workspace is one where **all** hold:

1. **Signed out.** The entire cloud surface — cloud VMs, presence (`presence.cmux.dev`), mobile-host
   **iroh**, billing, push — requires **Stack Auth** sign-in (`Sources/Auth/AuthEnvironment.swift`).
   Signed-out ⇒ none of it activates. This single condition neutralizes the largest egress class (A6 + §6).
2. **Telemetry off, before first launch.** `sendAnonymousTelemetry=false` (kills A1–A3) **and** the CLI
   env vars `CMUX_CLI_SENTRY_DISABLED=1` + `CMUX_CLAUDE_HOOK_SENTRY_DISABLED=1` (kills A4). Launch-frozen,
   so it must be set *before* the process starts.
3. **Auto-update off.** `SUEnableAutomaticChecks=false`, no `SUFeedURL`, updater not started (kills A5).
4. **Browser pinned or excluded**, search suggestions off (A7, A8).
5. **OS-level egress containment** as the backstop: a loopback-only firewall profile (pf / Little
   Snitch / network-restricted launch) so that A6/A7 — the no-code-switch paths — **cannot** leave the
   box even if a flag regresses. This is the fail-closed enforcement; the flags are defense-in-depth.

> **Config-only is insufficient (ADR-C006).** Because A6 and A7 have no in-code off switch, the
> sealed tier's guarantee rests on **#1 (signed-out) + #5 (OS containment)**, with #2–#4 as
> defense-in-depth. C3's `cmux-sealed` launcher must *enforce and verify* all five, and **fail
> closed** if it cannot (e.g. refuses to launch if the process is signed in, or if the firewall
> profile isn't active).

---

## 5. Section C — Distinct always-on native-core hosts (the firewall denylist / allowlist input)

For the OS-level profile, these are the hosts a **default** (unsealed) native core reaches at/near launch:

- `us.i.posthog.com` (A1, A2)
- `o4507547940749312.ingest.us.sentry.io` (A3, A4)
- `github.com` + `objects.githubusercontent.com` (A5 appcast + release assets)
- `*.relay.cmux.dev`, `cmux.com` (A6 — signed-in only)
- `suggestqueries.google.com`, `duckduckgo.com`, `www.bing.com` (A8 — browser)

The sealed profile is an **allowlist** (loopback + explicitly-approved local model hosts only), which
is stronger than denylisting these; the list is recorded so the dynamic check (§9) knows what to look for.

---

## 6. Section D — Gated cloud surface (open-tier only; not in sealed)

All of the following are **auth-required or explicitly opt-in** — correctly *not* reachable without
sign-in / user action. They are the Tier-1 (open) capabilities and are **out of scope for sealed**:

| Capability | Host | Where |
| --- | --- | --- |
| Cloud VM provisioning / PTY | `cmux.com` `/api/vm`, dynamic VM WS | `Sources/Auth/AuthEnvironment.swift:365`; `CLI/cmux.swift:11581` |
| Presence service | `presence.cmux.dev` | `PresenceSettings.swift:27` |
| Billing / Stripe checkout | `cmux.com/api/billing`, hosted Stripe | `Sources/PricingPlansScreen.swift:58` |
| Device / push notifications | `cmux.com/api/{devices,notifications/push}` | `AuthEnvironment` |
| Auth provider | `api.stack-auth.com` | `Sources/Auth/AuthEnvironment.swift:382` |
| Agent-chat model catalog | `cmux.com/api/agent-models` (sidecar fetch) | `agent-chat/catalog.ts:59` |
| Remote panes / SSH / remote-browser proxy | user SSH host; arbitrary target on the **remote** daemon | `CLI/cmux.swift:8897`; `daemon/remote/.../main.go:1601` |
| iOS app, web app, workers, relay-minter service | hosted (Vercel/Cloudflare) | `web/`, `workers/`, `services/`, `ios/` |

**Note:** there is **no compiled-in LLM client in the native core** — "cmux AI" agent-chat points at a
**local sidecar** (`127.0.0.1:7739`) by default (`Sources/CmuxAgentChatConfig.swift:94`); the only
cloud dependency is the model *catalog* fetch (cloud component, gated). cmux does not itself send your
prompts to a model — spawned agents (your `claude`/`codex`) reach their own providers, which is the
existing HEARTH "caller caveat," not new cmux egress.

---

## 7. Section E — Safe / local-only primitives (good news for orchestration, C4)

The orchestration surface we most want is **local**:

- **`cmux notify`** — local **Unix-domain-socket IPC, no network** (`CLI/cmux.swift:5051`).
- **Automation Unix socket** — local IPC; network-facing *only* via an authenticated SSH
  reverse-forward you opt into (`daemon/remote/.../cli.go:1094`).
- **Bundled webview UIs** (agent-session, diff, markdown) — served from the **app bundle**, not hosted
  (`Sources/Panels/AgentSessionWebRendererCoordinator.swift:120`).
- **Agent-chat sidecar** — `127.0.0.1` by default; CodexTeams approval bridge is loopback to a locally
  spawned server (`CLI/cmux.swift:21234`).

So the C4 orchestrator (read panes → HEARTH decide → notify/drive) can be built entirely on **local**
cmux primitives. Minor `safe-if-local` items to be aware of: markdown **remote image** loading
(`MarkdownRemoteImageLoader.swift`), a `react-grab` script pulled from `unpkg.com` and injected into
the browser (`Sources/Panels/ReactGrab.swift:122`), and a settings-schema `$ref` to
`raw.githubusercontent.com` — all browser/sidebar-adjacent, all covered by §4 #4–#5.

---

## 8. Section F — Build & supply-chain egress → build offline, run sealed

The **build** reaches many hosts (this is the single `must-exclude`, and it is build-host, not runtime):

- Toolchain bootstrap: `sh.rustup.rs`, `static.rust-lang.org`, `ziglang.org` (+ mirrors), `index.crates.io`.
- Dependencies: SPM (`github.com`), `iroh-ffi` fork, GhosttyKit xcframework (`github.com/manaflow-ai/ghostty`), `registry.npmjs.org`.
- CI release **uploads**: `sentry-cli` sends **dSYMs + source** to Sentry; artifacts to Cloudflare R2 / `files.cmux.com`, Apple notary, GitHub Releases, App Store (`scripts/ensure-sentry-cli.sh`, `.github/workflows/release.yml:550`).

**Posture:** mirror HEARTH's "pre-cache once, then go offline" model — **build cmux on an unrestricted
machine, then run the built artifact sealed on the confidential box.** Never build (or `bun install`,
or `zig build`) on the confidential machine online. The dSYM/source upload path must be stripped from
any build we run ourselves.

---

## 9. Dynamic verification plan (must run on real hardware before C3 trust)

Static analysis says *what can* egress; the following confirms *what does*, in each state. This is an
**interactive** step (a signed GUI app under a network monitor) — scripted in
`scripts/cmux/cmux_egress_probe.sh` (C0 deliverable), executed by a human:

1. **Baseline, signed-out, sealed flags on** — launch cmux; capture 5 min idle + a local-agent workload:
   `lsof -nP -iTCP -a -p "$(pgrep -f cmux)" -sTCP:ESTABLISHED` and `nettop -p <pid>`. **Expect: loopback only.**
2. **Negative control** — launch *without* the flags; confirm the probe **sees** PostHog/Sentry/Sparkle
   (proves the probe actually detects egress; a clean result in step 1 then means something).
3. **Signed-in** — confirm iroh relay (`*.relay.cmux.dev`) appears, proving §4 #1 is load-bearing.
4. **Firewall on** — enable the loopback-only pf/Little-Snitch profile; confirm even step-2 conditions
   produce **no** off-box connection (proves §4 #5 fail-closed backstop).
5. Record results in this file under "§10 Dynamic results" and gate C3 on step 1 + step 4 being clean.

---

## 10. Conclusions & hand-off to C3 / ADRs

1. **cmux is sealable, not clean OOTB.** The native core can be run with zero off-box egress via the §4
   invariant, verified by §9.
2. **The seal rests on signed-out + OS-level containment**, because the browser and iroh have no code
   switch. → **ADR-C006** (new): *sealed-tier enforcement requires signed-out + OS egress control +
   feature exclusion; config alone is insufficient.* Refines **ADR-C004** (we will `wrap` + rely on OS
   controls, and may need a small build-time patch to stub the browser/iroh — prefer upstreaming).
3. **Build offline, run sealed** (§8) — a new build-posture note for the C3 runbook.
4. **Orchestration primitives are local** (§7) — C4 is unblocked and safe.
5. **The cloud surface is cleanly gated behind sign-in** (§6) — this is exactly the Tier-0/Tier-1
   boundary from ARCHITECTURE.md, and it maps to a single enforceable condition. Good.

**No blocker is un-disableable** → we proceed to C1 (ratify ADRs incl. C006) and C3 (build the
`cmux-sealed` launcher + `cmux_egress_probe.sh`). Nothing here changes the standalone HEARTH guarantee.
