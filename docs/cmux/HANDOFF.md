# HEARTH × cmux — Session handoff / resume-here

> **Read this first when resuming.** Snapshot of exactly where the cmux integration stands as of
> **2026-07-22 (evening)**, so a fresh conversation can pick up with no context loss. The full plan is
> in [README.md](README.md); this file is the "what's live right now + what to do next."

---

## TL;DR

- **All build phases C0–C5 are done** (headless: code + tests + docs + man pages, each on its own
  `cmux/<task>` sub-branch, merged into `cmux/integration`). **`main` is untouched** — standalone HEARTH
  still ships (247 passed, 1 skipped).
- We started the **on-hardware validation** and then **parked the confidentiality lockdown** (firewall
  sealing) as too tedious — see [TODO.md](TODO.md). Everything needed to resume it is recorded.
- **cmux is installed and running** (v0.64.20). We pivoted to the **functional** cockpit+engine work.
- **Immediate next step:** run the **C4 orchestrator live inside a cmux pane** (steps below). That's the
  fun payoff — watch on-device HEARTH triage your real panes.
- ⚠ **4 commits are local-only, not pushed** — **LuLu is blocking `ssh → github.com`**. Resolve LuLu
  (allow ssh, or pause it) then push. Details below.

---

## Resume checklist (do these first tomorrow)

```sh
cd /Users/miltronix/Claude/apps/HEARTH
git branch --show-current          # expect: cmux/integration
git log --oneline -1               # expect: 4a04262 (or later) — the orchestrator-live merge
git log --oneline origin/cmux/integration..cmux/integration   # the UNPUSHED backlog (see below)

# 1) unblock git push: in LuLu, allow `ssh`→github (or pause LuLu — the lockdown is parked anyway), then:
git push origin cmux/integration
#    ...and push the sub-branches if you want them on origin (optional).

# 2) run the live orchestrator test (THE next task) — see "Immediate next task" below.
```

---

## Git state (exact)

- **On branch `cmux/integration`.** Working tree clean except **untracked `config/cmux/tiers.yaml`**
  (machine-local tier policy you created via `cp`; intentionally not committed — may list private paths).
- **`origin/cmux/integration` = `0e4668f`; local = `4a04262` → 4 commits ahead, UNPUSHED:**
  ```
  4a04262 merge(cmux): orchestrator auto-locates CLI + in-pane socket requirement (live finding)
  73026b2 feat(cmux): orchestrator auto-locates cmux CLI + documents in-pane requirement
  5924cd3 merge(cmux): park sealed-tier hardening (TODO) — proceed with functional integration
  5249be0 docs(cmux): park sealed-tier firewall hardening; capture TODO + finding
  ```
  Plus local sub-branches not yet pushed: `cmux/park-lockdown`, `cmux/onhardware-findings`,
  `cmux/orchestrator-live`, `cmux/handoff`. **Nothing is lost — it's all committed locally.**
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
| C4 | Orchestrator ([RUNBOOK_orchestrator.md](RUNBOOK_orchestrator.md)) | ✅ built/tested; **live in-pane run = NEXT** |
| C5 | Open tier ([RUNBOOK_open.md](RUNBOOK_open.md)) | ✅ gate demonstrated; live cloud run **PARKED** → TODO |
| C6 | Graduation to `main` | ◐ runbook+RESULTS ready; **PARKED** on sealed hardening |

---

## Immediate next task — C4 orchestrator, live in a cmux pane

**Why in a pane:** cmux's socket only accepts processes **descended from cmux** (ancestry check; it
injects `CMUX_SOCKET_PASSWORD` into pane shells). From an outside terminal you get
`Access denied - only processes started inside cmux can connect`. The cmux CLI also isn't on PATH — the
orchestrator auto-locates it (`$CMUX_BIN` → PATH → `/Applications/cmux.app/Contents/Resources/bin/cmux`).

**Steps (run inside a cmux pane):**
```sh
# leave a couple of OTHER panes in different states first (one running a cmd, one idle, one erroring)
cmux --version                       # confirm CLI reachable from inside the pane
cd /Users/miltronix/Claude/apps/HEARTH
HEARTH_BACKEND=mlx uv run python scripts/cmux/orchestrator.py --dry-run   # triage only (first run loads the model)
```
**Expected:** enumerates live panes, reads each screen, HEARTH classifies working/waiting/done/error,
prints which it would flag. **If the live `tree`/`list-pane-surfaces` JSON differs from the mapped
shapes** (`scripts/cmux/orchestrator.py` `CmuxCliClient.list_surfaces`/`read_screen`), adjust the parser
— that's the one thing that may need a tweak against the real 0.64.20 output. Then drop `--dry-run` to
see real `cmux notify` badges. This closes the live C4 validation.

**After C4:** optional live C2 (offload from a pane via `examples/cmux/sealed-pane.env` or MCP), then
anything else in [TODO.md](TODO.md) "ACTIVE" (browser-DOM→HEARTH summarize, notification triage).

---

## Environment facts (this machine, discovered 2026-07-22)

- **cmux:** v0.64.20; app at `/Applications/cmux.app`; **CLI at `/Applications/cmux.app/Contents/Resources/bin/cmux`**
  (the `Contents/MacOS/cmux` binary is the GUI). Socket: `~/.local/state/cmux/cmux.sock`. Config:
  `~/.config/cmux/cmux.json` (JSONC). Event log: `~/.cmuxterm/events.jsonl`.
- **cmux app-level seal applied:** `defaults write com.cmuxterm.app sendAnonymousTelemetry -bool false`
  and `SUEnableAutomaticChecks -bool false` are set; signed out.
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
3. **cmux socket = ancestry-gated** → orchestrator must run inside a pane (fixed + documented this session).

---

## Guardrails to remember

- **Sub-branch discipline:** every unit of work on `cmux/<task>`, merged into `cmux/integration`. Create
  the sub-branch BEFORE starting (slipped once on C4, corrected).
- **Do NOT merge to `main`** until the parked graduation gate (RESULTS complete) is green.
- **Functional work is on non-confidential/empty dirs** — the sealed guarantee is parked, so don't put
  confidential data through cmux until the firewall hardening (TODO) is finished.
- cmux clone lives at `…/scratchpad/cmux` (NOT committed, per ADR-C005).
