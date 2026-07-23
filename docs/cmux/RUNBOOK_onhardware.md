# HEARTH × cmux — On-hardware validation runbook (C6)

**Phase:** C6 · **Branch:** `cmux/graduation`. The single hands-on session that closes every
"pending on-hardware" item across C0–C5 and decides graduation to `main`. Everything upstream was
built and unit/model-validated headlessly; this run confirms the parts that need a real cmux GUI, the
pf firewall, a live socket, and a cloud account.

> **Do not merge `cmux/integration` → `main` until this runbook is green and `docs/cmux/RESULTS.md`
> is filled in.** That is the graduation gate (PROPOSAL success criteria).

**What this run closes:** C0 §9 (dynamic egress probe) · C2 (live pane offload) · C3 (sealed launch +
loopback-only under load + pf backstop) · C4 (orchestrator on a live socket) · C5 (open-tier cloud/Docker).

Record every result inline in `docs/cmux/RESULTS.md` as you go (it has a slot per step).

---

## 0. Prerequisites — build offline, then seal (AUDIT §8, ADR-C006)

The cmux **build** reaches many hosts and CI uploads dSYMs/source to Sentry. So **build and install
cmux while online, before any confidential work**, then go sealed. Never `bun install` / `zig build`
on the confidential box mid-session.

```sh
# --- while ONLINE (unrestricted), one time ---
# 1. Install cmux. RECOMMENDED: the official Apple-notarized app (no build toolchain, no build-time
#    egress on your box; the on-hardware probe validates the real binary regardless of source):
brew tap manaflow-ai/cmux && brew install --cask cmux
#    or the DMG: https://github.com/manaflow-ai/cmux/releases/latest/download/cmux-macos.dmg
#    (Optional, for binary==audited-source provenance: build from source per cmux's README —
#     submodules, zig, rust, bun, Xcode; heavier, fetches deps, do it online.)
#    Result: /Applications/cmux.app (set CMUX_APP below if your path differs).
# 2. Pre-cache the HEARTH model so sealed mode never needs the network:
uv run hearth models pull mlx-community/Qwen2.5-Coder-7B-Instruct-4bit
# 3. Set cmux telemetry/auto-update OFF in defaults BEFORE first launch (C0 A1–A5). cmux SPARKLE-
#    AUTO-UPDATES by default (README: "download once"), so this is mandatory; re-apply after any
#    `brew upgrade --cask cmux`:
defaults write com.cmuxterm.app sendAnonymousTelemetry -bool false
defaults write com.cmuxterm.app SUEnableAutomaticChecks -bool false
# 4. Make sure cmux is SIGNED OUT (no account) — this alone disables cloud + iroh (AUDIT §4 #1).
# 5. Note the installed version (`cmux --version`); if it differs from the audited main, the probe
#    still validates the real binary, but you can re-run the AUDIT §5 host greps against that tag.
```

Pick two directories to test the two tiers with, and `export` them (plus cmux's app + socket) so every
later command can refer to them by name. **They need NOT be git repos — any directory works, even an
empty one.** The tier classifier matches on the directory *path* (and only consults `git remote origin`
if you write a `remote_host` rule); cmux opens any dir as a workspace (it just won't show git branch info
for a non-repo). Contents are irrelevant here — you're validating egress and tier-gating, not doing work.

```sh
# NOTE: `export` lasts for THIS terminal session only (nothing is saved to disk). If you open a new
# terminal mid-run, re-run these four lines. Edit the paths to real directories on your machine.

# SEALED-tier test (Part 1). Any dir — even empty/non-git. Unclassified ⇒ sealed (the default), so
# nothing to configure. Use a confidential repo, or a throwaway you treat as confidential for a shakeout.
export CONF_REPO="$HOME/path/to/any/dir"

# OPEN-tier test (Part 2). Must classify `open`, so add a PATH rule matching it to config/cmux/tiers.yaml
# below (a `remote_host` rule would need a real git remote). An empty dir is fine to test the GATE; for
# the "cloud/Docker workspace actually runs" step, a real public git repo is more representative.
export OSS_REPO="$HOME/oss/any/dir"

export CMUX_APP="/Applications/cmux.app"                      # where you installed cmux (step 1)
export CMUX_SOCKET_PATH="$HOME/.local/state/cmux/cmux.sock"   # cmux's socket; the probe + orchestrator target it
```

Add an `open:` rule for `$OSS_REPO` (and confirm no `sealed_override` matches it):

```sh
cp -n config/cmux/tiers.example.yaml config/cmux/tiers.yaml   # then edit: add an open: path/remote rule
```

---

## Part 1 — Sealed tier (closes C0 §9 · C2 · C3 · C4)

### 1.1 Start a sealed HEARTH + load the egress firewall

```sh
scripts/hearth_private.sh --check                 # expect: "Posture verified: no router egress path"
scripts/hearth_private.sh &                        # sealed gateway on 127.0.0.1:8080
scripts/cmux/cmux-sealed --install-firewall        # loads the pf loopback-only anchor (needs sudo)
sudo pfctl -a cmux-sealed -sr                       # expect: block rules present
```
**RESULTS §1.1:** paste the `--check` line and the `pfctl -sr` output.

### 1.2 Preflight — must pass all mandatory gates

```sh
scripts/cmux/cmux-sealed --check --strict "$CONF_REPO"
```
**Expect:** `PASS tier`, `PASS hearth`, `PASS firewall`, and (under `--strict`) the advisory checks
PASS too → `RESULT: sealed posture verified (strict)`, exit 0.
**RESULTS §1.2:** paste the full check output + `echo exit=$?`.

### 1.3 Launch sealed + confirm loopback-only under load (the C0 §9 core)

```sh
scripts/cmux/cmux-sealed "$CONF_REPO"              # launches cmux at the confidential repo
# in another terminal, watch egress for 5 min while you actually use cmux (open panes, run an agent):
scripts/cmux/cmux_egress_probe.sh --seconds 300
```
**Expect:** `RESULT: only loopback/local connections observed. SEALED-clean.` exit 0. No
`posthog / sentry.io / *.relay.cmux.dev / cmux.com` endpoints.
**RESULTS §1.3:** paste the probe result. **This is the primary privacy gate — it must be clean.**

### 1.4 pf backstop proof (strongest evidence; emits nothing)

Even if a flag regressed or you were signed in, pf must block egress. With the firewall still loaded:

```sh
# (optional, strongest) sign IN to cmux, then re-run the probe. iroh will TRY *.relay.cmux.dev but
# pf drops it — the probe must STILL be loopback-only:
scripts/cmux/cmux_egress_probe.sh --seconds 120
# then sign back out.
```
**Expect:** still exit 0 (loopback-only) — proves the pf seal holds regardless of app state (ADR-C006).
**RESULTS §1.4:** paste result + note signed-in state.

### 1.5 (Optional) negative control — prove the probe can see egress

> Caveat: this deliberately lets cmux's **own** telemetry out (not your data). Do it on a throwaway
> context, or skip it and rely on 1.4. If you run it: remove the firewall + flip telemetry on, launch,
> probe — the probe should report OFF-BOX `posthog`/`sentry` (exit 3), proving 1.3's clean result is real.
> Then restore the seal (`--install-firewall`, telemetry off) before any confidential work.

**RESULTS §1.5:** paste result or "skipped".

### 1.6 C2 live — a pane offloads to sealed HEARTH

In a cmux pane at `$CONF_REPO`:
```sh
source examples/cmux/sealed-pane.env               # OPENAI_BASE_URL + cmux telemetry-off env
# then use a pane agent (Claude Code with examples/cmux/hearth.mcp.json, or an OpenAI-shaped agent).
# Trigger a summarize/classify subtask; confirm it is served locally (probe still clean).
```
**RESULTS §1.6:** note the subtask + that the probe stayed loopback-only during it.

### 1.7 C4 live — orchestrator on the live socket

With several panes running agents. (The workspace dir can be empty — the orchestrator triages each
pane's *output*, not the repo. To exercise it, leave panes in different states: one mid-command, one at
a `[y/N]` prompt, one finished, one showing an error — then run the sweep.)
```sh
HEARTH_BACKEND=mlx uv run python scripts/cmux/orchestrator.py --dry-run   # triage only, verify states
HEARTH_BACKEND=mlx uv run python scripts/cmux/orchestrator.py             # actually notify
scripts/cmux/cmux_egress_probe.sh --seconds 60                            # still loopback-only
```
**Expect:** the sweep enumerates live panes, classifies them, notifies the ones needing attention;
probe stays clean.
**RESULTS §1.7:** paste the orchestrator output + probe result. Note any parsing fixups needed against
the live `tree`/`list-*` JSON (the client was built to the mapped shapes; adjust if the live JSON differs).

---

## Part 2 — Open tier (closes C5)

### 2.1 Gate refuses a sealed repo

```sh
scripts/cmux/cmux-open --check "$CONF_REPO"        # expect: REFUSED, exit 2
```
**RESULTS §2.1:** paste output + exit code.

### 2.2 Open a non-confidential repo (cloud/Docker)

```sh
scripts/cmux/cmux-open --check "$OSS_REPO"         # expect: PASS (exit 0)
scripts/cmux/cmux-open "$OSS_REPO"                 # grants (logged) + launches open workspace
# in cmux: create a cloud VM or networked-Docker workspace for $OSS_REPO and confirm it works.
tail -3 ~/.hearth/cmux-tier-audit.log              # expect an open-GRANTED line for $OSS_REPO
```
**RESULTS §2.2:** note the cloud/Docker workspace worked + paste the audit line.

---

## Part 3 — Fill in RESULTS and decide graduation

Complete every `RESULTS §` slot in `docs/cmux/RESULTS.md`, then check the graduation gate below.

### C6 graduation checklist (all must be true — PROPOSAL success criteria)

- [ ] **Works** — sealed cockpit + pane offload + live orchestrator ran end-to-end (Part 1); open-tier
      cloud/Docker workspace ran (Part 2).
- [ ] **Verified private** — §1.3 probe loopback-only under load; §1.4 pf backstop held; C0 audit
      re-checked; open gate refused a sealed repo (§2.1).
- [ ] **Beneficial** — measured local offload savings (C2 / §1.6) and the orchestrator triage were
      useful; qualitative UX win over bare terminals noted in RESULTS.
- [ ] **Reversible** — `git switch -c restore archive/hearth-pre-cmux-2026-07-21` still restores
      standalone HEARTH; nothing in the integration hard-requires cmux for HEARTH's own suite.
- [ ] **Documented** — RESULTS.md filled; README status tracker all ☑.

### When every box is checked, graduate:

```sh
uv run pytest -q                                   # standalone HEARTH still green (no cmux needed)
git switch main
git merge --no-ff cmux/integration -m "feat(cmux): integrate cmux cockpit into HEARTH (C0–C6, gated sealed/open tiers)"
git tag -a cmux/integrated-$(date +%Y-%m-%d) -m "cmux integration graduated to main: proven, sealed-verified, beneficial."
git push origin main --tags
```

If any box fails: **do not merge.** Record what failed in RESULTS, open a fix on a new `cmux/<task>`
sub-branch, re-run the relevant Part. `main` stays the working standalone HEARTH until it's all green.

---

## Teardown

```sh
scripts/cmux/cmux-sealed --remove-firewall         # unload the pf anchor
pkill -f "hearth serve"                             # stop the sealed gateway
# after confidential work: quit cmux, then purge scrollback (docs/cmux/PRIVACY.md):
rm -rf ~/Library/Application\ Support/cmux/
```
