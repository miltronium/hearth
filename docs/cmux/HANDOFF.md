# HEARTH × cmux — Session handoff / resume-here

> **Read this first when resuming.** Snapshot of exactly where the cmux integration stands as of
> **2026-08-19**, so a fresh conversation can pick up with no context loss. The full plan is
> in [README.md](README.md); this file is the "what's live right now + what to do next."

---

## TL;DR

- **All build phases C0–C5 are done** (headless: code + tests + docs + man pages, each on its own
  `cmux/<task>` sub-branch, merged into `cmux/integration`). **`main` is untouched** — standalone HEARTH
  still ships (**248 passed, 1 skipped**).
- ✅ **C4 is live-validated (2026-08-06)** — the orchestrator drove a real cmux 0.64.20, triaged
  four panes 4/4 correctly on-device, and fired three real notify badges. **RESULTS §1.7** functional
  half is filled. Three live-only bugs were found and fixed; see below.
- ✅ **C2 is live-validated (2026-08-17)** — a real Claude Code agent in a real cmux pane offloaded a
  log summary to local HEARTH over MCP, proven from the agent's own `tool_use` transcript; the
  OpenAI surface served in-pane at `served_by=local`. **RESULTS §1.6** functional half is filled.
  Three more live-only findings fixed/documented. **Both functional tracks are now closed.**
- **The "must run inside a cmux pane" rule was wrong** — it's a *default*, not a hard gate. See
  **ADR-C007** and the two access routes in [RUNBOOK_orchestrator.md](RUNBOOK_orchestrator.md).
- ✅ **LuLu enforcement is RESTORED (2026-08-19).** `lulu_rule_check.py` exits **0** — `BLOCK *:*`
  for `com.cmuxterm.app` **and** "network extension is loaded and running". The 2026-08-17
  "no application firewall is active" blocker is **resolved**; disregard it.
- ✅ **§1.3 — the primary privacy gate — is CLOSED (2026-08-19).** cmux launched cold under the block;
  `cmux_egress_probe.sh --seconds 280` → **exit 0, loopback-only**, post-hoc `lsof` snapshot also empty.
  **RESULTS §1.3** filled. The sealed tier now has its central guarantee on record.
- ✅ **Push is unblocked** (as of 2026-08-17) — `ssh → github.com` works again and
  `origin/cmux/integration` is level with local. The 2026-08-06 "commits are local-only / LuLu
  blocks ssh" warning is **stale**; disregard it.

---

## Resume checklist (do these first)

**The 2026-08-17 LuLu blocker is cleared and §1.3 is closed.** Next up is **§1.4**, which needs a
human decision (signing into cmux with the block active — your account, your call).

```sh
cd /Users/miltronix/Claude/apps/HEARTH
git branch --show-current                                       # expect: cmux/integration

# 0) re-confirm the seal is still in force (check EVERY session — flags and the extension both regress)
python3 scripts/cmux/lulu_rule_check.py                         # want exit 0 "filter loaded and running"
defaults read com.cmuxterm.app SUEnableAutomaticChecks          # want 0 — this has self-regressed before
CMUX_CLI_SENTRY_DISABLED=1 CMUX_CLAUDE_HOOK_SENTRY_DISABLED=1 \
  scripts/cmux/cmux-sealed --check --strict /Users/miltronix/Claude/apps/HEARTH   # want all 7 PASS

# 1) §1.4 — sign into cmux with the block active, prove the iroh relay is still blocked
scripts/cmux/cmux_egress_probe.sh --seconds 280 &   # start FIRST, it waits for cmux
/Applications/cmux.app/Contents/Resources/bin/cmux ~/Claude/apps/HEARTH/examples/repos/oss_repo
#    then sign in inside the app; want probe exit 0 (no *.relay.cmux.dev) -> fills RESULTS §1.4

# 2) then the two egress slivers that were only ever waiting on the seal:
#    - re-run examples/cmux/pane_offload_live.sh under the seal   -> §1.6 egress half
#    - re-run scripts/cmux/orchestrator.py under the seal          -> §1.7 egress half
```

> ⚠️ **Running `--check` from a bare shell always FAILs `cli-telemetry`** — that gate reads two env
> vars the launcher exports itself at `cmux-sealed:170`. Pass them inline (as above) or you will
> chase a phantom failure. This is a check-mode artifact, not a real posture gap.

> ⚠️ While the seal holds, **`git push` from inside a cmux pane fails** — push from an ordinary
> terminal, which is unaffected.

Both functional tracks (C2 live, C4 live) are **done** — nothing functional is waiting on you.
Everything outstanding is validation/graduation.

---

## Git state (exact)

- **On branch `cmux/integration`.** Working tree clean except two intentionally-untracked paths:
  **`config/cmux/tiers.yaml`** (machine-local tier policy; may list private paths) and
  **`examples/repos/`** (the empty non-confidential test dirs + throwaway live-run evidence).
- **`origin/cmux/integration` is level with local** — everything is pushed. Some local-only
  sub-branches remain unpushed (`cmux/handoff`, `cmux/handoff-update`, `cmux/orchestrator-live`,
  `cmux/park-lockdown`, `cmux/socket-auth`, `cmux/c2-live`); that's optional housekeeping, and
  their content is already merged into `cmux/integration`.
- `main` = `75ba6ad` (untouched). Archive tag `archive/hearth-pre-cmux-2026-07-21` still restores standalone.
- **Local venv note:** this machine now has `uv sync --extra mlx --extra mcp --extra dev` applied
  (the `mcp` extra is required for `hearth mcp`). Syncing a single extra prunes the others.

---

## Phase status

| Phase | What | Status |
| --- | --- | --- |
| C0 | Egress audit ([AUDIT.md](AUDIT.md)) + probe | ✅ static (123 findings). Dynamic probe: partly run (see findings) |
| C1 | ADRs (C001–C006) | ✅ Accepted |
| C2 | HEARTH-as-brain wiring ([RUNBOOK_wiring.md](RUNBOOK_wiring.md)) | ✅ validated (1053 tokens saved); ✅ **live pane offload done 2026-08-17** (RESULTS §1.6). Loopback-only-under-seal check still pending |
| C3 | Sealed launcher + classifier + pf/LuLu ([RUNBOOK_sealed.md](RUNBOOK_sealed.md)) | ✅ built/tested; gate hardened 2026-08-17 (**ADR-C008**); ✅ **on-hardware seal proven 2026-08-19** (RESULTS §1.3). §1.4 remains |
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

---

## ✅ Done 2026-08-17 — C2 live pane offload

**Reproduce it** (cmux running, gateway up, `uv sync --extra mlx --extra mcp --extra dev` applied):
```sh
# 1) sealed local gateway (loopback, no remotes, offline weights)
HEARTH_ROUTING_YAML=config/routing.private.yaml HEARTH_HOST=127.0.0.1 HEARTH_BACKEND=mlx \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run hearth serve

# 2) from inside a cmux pane (or via `cmux send`), against a non-confidential dir:
examples/cmux/pane_offload_live.sh device.log hearth.live.mcp.json    # expect: VERDICT= OFFLOADED
```

**Result:** both wiring surfaces exercised from a real pane. **OpenAI** — `served_by=local`,
`escalated=False`, 49 est. tokens saved. **MCP** — a real Claude Code agent called
`mcp__hearth__hearth_summarize` on a 180-line/13 KB log; **proven from the agent's own `tool_use`
transcript**, reproduced twice. Suite still **248 passed, 1 skipped**.

**What the live run corrected (don't re-derive):**
1. **The `mcp` extra is required, and Claude Code drops the server SILENTLY without it.** `hearth mcp`
   exits with `The MCP server requires the 'mcp' extra.`; the agent then reports "no hearth tools are
   registered", which misleads you into debugging `--mcp-config` paths. Probe the server standalone.
2. **`uv sync --extra mcp` alone PRUNES the other extras** — it stripped `mlx-lm`, `transformers`,
   `torch`, `pytest`, `ruff`. Always `uv sync --extra mlx --extra mcp --extra dev` together.
3. **`--output-format json` shows no tool_use records** — only the final result. Any "did it call the
   tool?" assertion needs `--output-format stream-json --verbose`. Don't trust an agent's self-report:
   an early harness asked it to print `TOOL_USED=yes|no` and it answered `yes`, proving nothing.
4. **`hearth_summarize`'s `max_words` is a soft hint** (~40 words returned for a 25 limit).
5. **MCP tools take `text`, not a path** — so a pane must Read the file into its own context first,
   capping savings on the "pre-digest a large file" pattern. A path-taking variant would close this.

**Next (to unblock graduation):** §1.3 is now closed (2026-08-19). What remains is **§1.4** (sign in
with the block active), the §1.6/§1.7 egress halves re-run under the seal, and §2.2's open-tier cloud
run. **No functional work is outstanding.**

---

## ✅ Done 2026-08-17 → 2026-08-19 — firewall hardening (§1.5 and §1.3 both filled)

**What shipped:** `scripts/cmux/lulu_rule_check.py` (+15 tests) and a rewritten mandatory firewall
gate in `cmux-sealed --check`. **ADR-C008** records the principle. Suite **263 passed, 1 skipped**.

**What was proven — three things lied in one session (this is the finding):**
1. **Config lies.** LuLu was running and the old gate passed — while LuLu held an explicit
   `ALLOW *:*` rule for `com.cmuxterm.app`. The gate had been reporting a seal that never existed.
2. **Rules lie too.** With the rule corrected to `BLOCK *:*`, the new rule-reading check said
   `SEALED` — while LuLu's extension was dead. cmux egressed anyway (probe exit 3,
   `172.182.252.137:443`). Hence the liveness layer and exit 3.
3. **App flags regress.** `SUEnableAutomaticChecks` had flipped `0` → `1` since 2026-07-22 with no
   app update (same binary, same v0.64.20). Re-applied to `0` on 2026-08-17.

→ Layers are **config < enforcement < outcome**; only `cmux_egress_probe.sh` is proof.

**What did NOT happen on 2026-08-17:** §1.3 was not filled — the seal was never in force that day, and
it was correctly refused rather than back-filled. §1.4 remains untouched (needs a decision about signing in).

**Closed 2026-08-19:** enforcement came back (extension live), and §1.3 was filled on a real run —
probe **exit 0**, loopback-only. See RESULTS §1.3 for the controlled comparison that makes it a seal
rather than a quiet app. The cmux LuLu rule stays **BLOCK**; note **`git push` from inside a cmux pane
fails** while the seal holds — push from an ordinary terminal, which is unaffected.

---

## Open decisions for tomorrow

- **§1.4 (iroh relay backstop)** needs you to *sign into cmux* with the block active, to prove
  sign-out is not load-bearing. Your account → your call. Skippable, at the cost of one graduation
  criterion.
- **When to restore the LuLu rule to Allow.** §1.6/§1.7's egress halves still need the seal active,
  so the cheap path is unchanged: leave cmux blocked through §1.4 → §1.7, then restore once at the end.

---

## Environment facts (this machine, discovered 2026-07-22)

- **cmux:** v0.64.20; app at `/Applications/cmux.app`; **CLI at `/Applications/cmux.app/Contents/Resources/bin/cmux`**
  (the `Contents/MacOS/cmux` binary is the GUI). Socket: `~/.local/state/cmux/cmux.sock`. Config:
  `~/.config/cmux/cmux.json` (JSONC). Event log: `~/.cmuxterm/events.jsonl`.
- **cmux app-level seal applied:** `defaults write com.cmuxterm.app sendAnonymousTelemetry -bool false`
  and `SUEnableAutomaticChecks -bool false` are set; signed out. ⚠️ **`SUEnableAutomaticChecks`
  regressed to `1` on its own** between 2026-07-22 and 2026-08-17 (no app update) — re-applied, but
  **re-check it every session**; this is why ADR-C008 exists.
- **Socket auth (set 2026-08-06):** `automation.socketControlMode = "password"` in
  `~/.config/cmux/cmux.json`; secret in `~/.local/state/cmux/socket-control-password` (0600).
  Revert to the stricter default by setting the mode back to `"cmuxOnly"` and relaunching cmux.
  Note cmux **rewrites `cmux.json` on launch** (it strips the commented template and drops migrated
  secrets), keeping its own `cmux.<timestamp>.bak`; a pre-edit backup is at
  `~/.config/cmux/cmux.json.20260805-164608.bak`.
- **HEARTH operational:** MLX backend, `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` pulled to
  `~/.hearth/models`. `HEARTH_BACKEND=mlx uv run python …` / `uv run hearth …` work.
- ✅ **LuLu (as of 2026-08-19): installed AND enforcing.** `lulu_rule_check.py` exits 0 — extension
  "loaded and running", rule store holds `BLOCK *:*` for `com.cmuxterm.app`. The 2026-08-17
  `[terminated waiting to uninstall on reboot]` state is resolved. **Re-verify every session**
  (`python3 scripts/cmux/lulu_rule_check.py`; exit 0 = blocked *and* enforced) — this machine has
  regressed both the extension and the Sparkle flag once each already.
  LuLu does not block `ssh→github` — pushes work from an ordinary terminal.
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
