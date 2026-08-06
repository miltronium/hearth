# HEARTH × cmux — Session handoff / resume-here

> **Read this first when resuming.** Snapshot of exactly where the cmux integration stands as of
> **2026-08-06**, so a fresh conversation can pick up with no context loss. The full plan is
> in [README.md](README.md); this file is the "what's live right now + what to do next."

---

## TL;DR

- **All build phases C0–C5 are done** (headless: code + tests + docs + man pages, each on its own
  `cmux/<task>` sub-branch, merged into `cmux/integration`). **`main` is untouched** — standalone HEARTH
  still ships (**248 passed, 1 skipped**).
- ✅ **C4 is now live-validated (2026-08-06)** — the orchestrator drove a real cmux 0.64.20, triaged
  four panes 4/4 correctly on-device, and fired three real notify badges. **RESULTS §1.7** functional
  half is filled. Three live-only bugs were found and fixed; see below.
- **The "must run inside a cmux pane" rule was wrong** — it's a *default*, not a hard gate. See
  **ADR-C007** and the two access routes in [RUNBOOK_orchestrator.md](RUNBOOK_orchestrator.md).
- The **confidentiality lockdown (firewall sealing) is still parked** — see [TODO.md](TODO.md).
  LuLu is installed and active, so the remaining §1.3/§1.4 work is now short.
- ⚠ **Commits are still local-only, not pushed** — **LuLu is still blocking `ssh → github.com`**
  (verified today: `git ls-remote` times out). Allow `ssh` in LuLu, or pause it, then push.

---

## Resume checklist (do these first)

```sh
cd /Users/miltronix/Claude/apps/HEARTH
git branch --show-current          # expect: cmux/integration
git log --oneline -1               # expect: 1093c63 (or later) — the C4 live-validation merge
git log --oneline origin/cmux/integration..cmux/integration   # the UNPUSHED backlog (see below)

# 1) unblock git push: in LuLu, allow `ssh`→github (or pause LuLu — the lockdown is parked anyway), then:
git push origin cmux/integration
#    ...and push the sub-branches if you want them on origin (optional).

# 2) pick up either track:
#    functional  → live C2 offload from a pane (TODO.md "ACTIVE")
#    graduation  → the parked firewall hardening (TODO.md "PARKED"); LuLu is already installed
```

---

## Git state (exact)

- **On branch `cmux/integration`.** Working tree clean except **untracked `config/cmux/tiers.yaml`**
  (machine-local tier policy you created via `cp`; intentionally not committed — may list private paths).
- **`origin/cmux/integration` = `0e4668f`; local is 8 commits ahead, UNPUSHED** (latest first):
  ```
  1093c63 merge(cmux): C4 orchestrator live-validated — socket password auth (ADR-C007) + 3 live-only fixes
  1cc95b3 feat(cmux): C4 orchestrator validated live — socket password auth + 3 live-only fixes
  b9382af merge(cmux): session handoff / resume-here doc
  032aabf docs(cmux): session handoff — resume-here snapshot for a fresh conversation
  4a04262 merge(cmux): orchestrator auto-locates CLI + in-pane socket requirement (live finding)
  73026b2 feat(cmux): orchestrator auto-locates cmux CLI + documents in-pane requirement
  5924cd3 merge(cmux): park sealed-tier hardening (TODO) — proceed with functional integration
  5249be0 docs(cmux): park sealed-tier firewall hardening; capture TODO + finding
  ```
  Plus local sub-branches not yet pushed: `cmux/park-lockdown`, `cmux/onhardware-findings`,
  `cmux/orchestrator-live`, `cmux/handoff`, `cmux/socket-auth`. **Nothing is lost — it's all committed locally.**
- **Why unpushed:** LuLu (installed for the parked sealing) now blocks `ssh → github.com:22`. Earlier
  phases pushed fine; only LuLu changed. Fix: allow `ssh` in LuLu, or pause LuLu.
- `main` = `75ba6ad` (untouched). Archive tag `archive/hearth-pre-cmux-2026-07-21` still restores standalone.

---

## Phase status

| Phase | What | Status |
| --- | --- | --- |
| C0 | Egress audit ([AUDIT.md](AUDIT.md)) + probe | ✅ static (123 findings). Dynamic probe: partly run (see findings) |
| C1 | ADRs (C001–C006) | ✅ Accepted |
| C2 | HEARTH-as-brain wiring ([RUNBOOK_wiring.md](RUNBOOK_wiring.md)) | ✅ validated (1053 tokens saved). Live pane offload: optional to redo |
| C3 | Sealed launcher + classifier + pf/LuLu ([RUNBOOK_sealed.md](RUNBOOK_sealed.md)) | ✅ built/tested; on-hardware firewall sealing **PARKED** → TODO |
| C4 | Orchestrator ([RUNBOOK_orchestrator.md](RUNBOOK_orchestrator.md)) | ✅ built/tested; ✅ **live run done 2026-08-06** (RESULTS §1.7). Loopback-only-under-seal check still pending |
| C5 | Open tier ([RUNBOOK_open.md](RUNBOOK_open.md)) | ✅ gate demonstrated; live cloud run **PARKED** → TODO |
| C6 | Graduation to `main` | ◐ runbook+RESULTS ready; **PARKED** on sealed hardening |

---

## ✅ Done 2026-08-06 — C4 orchestrator live

**Reproduce it in one line** (cmux running, `socketControlMode=password` already set on this machine):
```sh
cd /Users/miltronix/Claude/apps/HEARTH
. scripts/cmux/cmux-auth-env                                              # exports CMUX_SOCKET_PASSWORD
HEARTH_BACKEND=mlx uv run python scripts/cmux/orchestrator.py --dry-run   # add nothing to actually notify
```

**Result:** 4 workspaces parked in distinct states → classified done / working / waiting / error,
**4/4 correct**, 3/4 flagged, **0 frontier tokens**, three badges confirmed via `cmux list-notifications`.

**What the live run corrected (don't re-derive):**
1. **Socket access is configurable.** `automation.socketControlMode` = `cmuxOnly` (default,
   ancestry-gated) | `password` | `allowAll`. We set **`password`**, so the sweep runs from any
   terminal. cmux **migrates** a `socketPassword` written into `cmux.json` out of that file on launch
   into `~/.local/state/cmux/socket-control-password` (0600) — that file is the source of truth, and
   the cmux CLI auto-reads it (which is why a passwordless `cmux ping` appears to work). **ADR-C007**
   records the trade: `cmuxOnly` for confidential work, `password` for automation, never `allowAll`.
2. **Surface refs (`surface:1`) are positional and go stale** between enumerate and notify → always
   pass `--id-format both` and use the UUID. This was the actual cause of `Surface ref not found`.
3. **cmux reports errors on stdout with rc=1**, so a plain `check_returncode()` hides them.

**Next (functional):** live C2 offload from a pane (`examples/cmux/sealed-pane.env` or MCP), then
anything else in [TODO.md](TODO.md) "ACTIVE" (browser-DOM→HEARTH summarize, notification triage).

**Next (to unblock graduation):** the parked firewall hardening — LuLu is already installed and
active, so §1.3/§1.4 are now mostly a matter of adding the cmux block rule and re-verifying.

---

## Environment facts (this machine, discovered 2026-07-22)

- **cmux:** v0.64.20; app at `/Applications/cmux.app`; **CLI at `/Applications/cmux.app/Contents/Resources/bin/cmux`**
  (the `Contents/MacOS/cmux` binary is the GUI). Socket: `~/.local/state/cmux/cmux.sock`. Config:
  `~/.config/cmux/cmux.json` (JSONC). Event log: `~/.cmuxterm/events.jsonl`.
- **cmux app-level seal applied:** `defaults write com.cmuxterm.app sendAnonymousTelemetry -bool false`
  and `SUEnableAutomaticChecks -bool false` are set; signed out.
- **Socket auth (set 2026-08-06):** `automation.socketControlMode = "password"` in
  `~/.config/cmux/cmux.json`; secret in `~/.local/state/cmux/socket-control-password` (0600).
  Revert to the stricter default by setting the mode back to `"cmuxOnly"` and relaunching cmux.
  Note cmux **rewrites `cmux.json` on launch** (it strips the commented template and drops migrated
  secrets), keeping its own `cmux.<timestamp>.bak`; a pre-edit backup is at
  `~/.config/cmux/cmux.json.20260805-164608.bak`.
- **HEARTH operational:** MLX backend, `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` pulled to
  `~/.hearth/models`. `HEARTH_BACKEND=mlx uv run python …` / `uv run hearth …` work.
- **LuLu installed + active** (system extension approved). It blocks unknown egress → currently blocking
  `ssh→github` (git push). It's the intended structural seal for the PARKED lockdown; for now it's just
  friction — allow ssh or pause it.
- **Test dirs:** `CONF_REPO`/`OSS_REPO` were `examples/repos/{conf_repo,oss_repo}` (empty dirs are fine).

---

## Key findings this session (don't re-derive)

1. **cmux is NOT egress-clean on app flags alone.** With telemetry off + signed out, cmux still hit
   **GitHub** (`140.82.116.6` / `185.199.110.215`) — **Sparkle's launch-time update check**;
   `SUEnableAutomaticChecks=false` does not stop it. Confirms **AUDIT §3 A5**, validates **ADR-C006**
   (structural firewall required, not just flags). Fix = LuLu blocking cmux (parked; see TODO/RESULTS §1.3).
2. **The sampling probe can give a false "clean"** (Sparkle fired after the window). A deny-by-default
   firewall (LuLu/Little Snitch) is the authoritative seal — noted in the probe header + RESULTS §1.3.
3. **cmux socket = ancestry-gated _by default only_.** `automation.socketControlMode` also accepts
   `password` (and `allowAll`). With `password` the orchestrator runs from any local terminal — see
   **ADR-C007**. *(Supersedes the earlier "must run inside a pane" absolute.)*
4. **Surface refs are positional and go stale** — enumerate with `--id-format both`, act on UUIDs.

---

## Guardrails to remember

- **Sub-branch discipline:** every unit of work on `cmux/<task>`, merged into `cmux/integration`. Create
  the sub-branch BEFORE starting (slipped once on C4, corrected).
- **Do NOT merge to `main`** until the parked graduation gate (RESULTS complete) is green.
- **Functional work is on non-confidential/empty dirs** — the sealed guarantee is parked, so don't put
  confidential data through cmux until the firewall hardening (TODO) is finished.
- cmux clone lives at `…/scratchpad/cmux` (NOT committed, per ADR-C005).
