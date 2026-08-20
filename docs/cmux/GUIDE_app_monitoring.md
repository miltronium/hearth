# Driving your app from cmux — run, watch, and monitor log output

> **Audience: an agent (or a human) who has never used cmux and wants it as the harness for
> *their own* app.** You do not need to know anything about HEARTH to use this. Everything here is
> the cmux **CLI / Unix-socket API** — no GUI clicking, no fork, no patch.
>
> Written against **cmux 0.64.20** on macOS (`/Applications/cmux.app`). Verified state and
> hard-won gotchas come from this repo's live validations (2026-08-06, 2026-08-17, 2026-08-19).
>
> **Read the [verification ledger](#12-verification-ledger--what-is-proven-vs-read-from---help)
> before you trust a command in here.** Some commands are live-proven in this repo; others are
> transcribed from `cmux --help` and marked as unexercised. Do not report an unexercised command
> as working until you have run it.

---

## 0. TL;DR — the six primitives you actually need

| Want | Command |
| --- | --- |
| Give the app its own tab, running a command | `cmux new-workspace --name app --cwd <dir> --command "<cmd>"` |
| Read what it printed | `cmux --json read-screen --surface <id> --lines 200` |
| Keep a durable log file of the stream | `cmux pipe-pane --surface <id> --command "tee -a /tmp/app.log"` |
| Answer a prompt it's blocked on | `cmux send --surface <id> "y"` · `cmux send-key --surface <id> Enter` |
| Tell the human / yourself something happened | `cmux notify --surface <id> --title "…" --body "…"` |
| Watch without polling | `cmux events --cursor-file /tmp/cur --reconnect` |

Everything else in this doc is those six with the sharp edges filed off.

---

## 1. Get on the socket (do this first, it is where everyone gets stuck)

### 1.1 The CLI is not on `PATH`

The `cmux` on `PATH` may not exist; `Contents/MacOS/cmux` is the **GUI binary**, not the CLI.
The CLI lives inside the bundle:

```sh
/Applications/cmux.app/Contents/Resources/bin/cmux
```

Resolution order worth copying (this is what `scripts/cmux/orchestrator.py:default_cmux_bin` does):
`$CMUX_BIN` → `PATH` → `/Applications/cmux.app/Contents/Resources/bin/cmux` →
`~/Applications/cmux.app/…`.

Give yourself a handle for the rest of the session:

```sh
cx() { "${CMUX_BIN:-/Applications/cmux.app/Contents/Resources/bin/cmux}" "$@"; }
cx --version        # cmux 0.64.20 (100) …
```

### 1.2 Two routes onto the socket

cmux gates its automation socket with `automation.socketControlMode` in `~/.config/cmux/cmux.json`.
Three values: **`cmuxOnly`** (default) · **`password`** · `allowAll`.

- **Route A — run inside a cmux pane.** Under the default `cmuxOnly`, the socket accepts **only
  processes descended from cmux** (it injects `CMUX_SOCKET_PASSWORD` into the shells it spawns).
  From an outside terminal you get
  `Access denied - only processes started inside cmux can connect`. Nothing to configure — just be
  in a pane.
- **Route B — password auth, from any local terminal.** Set `socketControlMode: "password"` and a
  secret, relaunch cmux. On launch cmux **migrates** the secret out of `cmux.json` into
  `~/.local/state/cmux/socket-control-password` (mode 0600) and that file becomes the source of
  truth — the CLI auto-reads it, which is why a "passwordless" `cmux ping` appears to work.

```sh
. scripts/cmux/cmux-auth-env    # exports CMUX_SOCKET_PASSWORD from the 0600 store — SOURCE it, don't exec it
cx ping                          # expect: PONG
cx identify --json               # who am I / which workspace+surface am I in
```

> Never pass `--password` on the command line — it is visible in `ps`. Never use `allowAll`; it
> drops authentication entirely. Route B means *any process running as you* can read your panes and
> type into them; that is a real relaxation of Route A. This repo chose `password` deliberately —
> see `docs/cmux/DECISIONS.md` **ADR-C007**.

**On this machine (2026-08-19): `access_mode` is already `password`.** Route B works from an
ordinary terminal; you do not need to configure anything.

### 1.3 If a sandbox is in the way

A sandboxed shell (e.g. an agent harness with default Bash sandboxing) will fail to `connect(2)` the
Unix socket with `Operation not permitted, errno 1` — that is the sandbox, **not** cmux auth, and no
amount of password fiddling fixes it. Verify by checking the app is alive and the socket exists:

```sh
pgrep -fl "cmux.app/Contents/MacOS"     # the GUI process
ls -l ~/.local/state/cmux/cmux.sock     # srw------- owned by you
```

If both are fine and you still get `errno 1`, escalate out of the sandbox for socket calls.

---

## 2. The target model (window → workspace → pane → surface)

```
window          a macOS window
└── workspace   a vertical tab (has a cwd; this is "one project / one job")
    └── pane    a split region
        └── surface   the actual content: a terminal, a browser, or an agent session
```

**You almost always target a `--surface`** — that is the thing with output to read and a keyboard to
type into. Enumerate with:

```sh
cx --json --id-format both list-workspaces                       # alias of `cmux workspace list`
cx --json --id-format both list-pane-surfaces --workspace <id>   # surfaces[]: id/ref/title/type
cx tree --all                                                    # human-readable whole topology
```

### ⚠️ The single most expensive gotcha: refs go stale

cmux returns **short positional refs** (`surface:1`, `workspace:2`) by default and **omits UUIDs
entirely**. Refs renumber as workspaces and surfaces come and go, so a ref you captured during
enumeration can resolve to a *different* surface — or to nothing (`Surface ref not found`) — a few
seconds later. This cost a live debugging session.

> **Always pass `--id-format both` and act on the UUID.** Only ever hand a bare `surface:1` to a
> throwaway interactive command you type yourself.

Inside a cmux pane these are pre-set for you and used as the defaults for *all* commands:

| Env var | Meaning |
| --- | --- |
| `CMUX_WORKSPACE_ID` | default `--workspace` |
| `CMUX_SURFACE_ID` | default `--surface` |
| `CMUX_TAB_ID` | default `--tab` for `tab-action` / `rename-tab` |
| `CMUX_SOCKET_PATH` | override the socket (default `~/.local/state/cmux/cmux.sock`) |
| `CMUX_QUIET=1` | silence deprecation/alias notices |

---

## 3. Pattern 1 — give your app its own workspace, running your command

```sh
# a dedicated tab, cwd'd into the project, running the app immediately
cx --json --id-format both new-workspace \
   --name "app: api-server" \
   --cwd /path/to/your/app \
   --command "npm run dev"        # or: uv run myapp serve / cargo run / ./gradlew bootRun

# add a second surface beside it for logs/tests, without disturbing the app
cx new-split right --workspace <ws>
cx new-pane --type terminal --direction down --workspace <ws>

# restart the app in place (same surface, new process)
cx respawn-pane --surface <sid> --command "npm run dev"

# tidy up when done
cx close-surface --surface <sid>
cx close-workspace --workspace <ws>
```

Why a workspace per job: it carries its own cwd, its own notification badge, its own status chips and
its own log panel (§7), and it survives restarts via session restore. One workspace = one thing you
are watching.

Useful cousins:

- `cx new-surface --type agent-session --provider claude|codex|opencode` — a first-class coding-agent
  surface instead of a bare shell.
- `cx rename-tab --surface <sid> "api-server"` — make your enumeration output readable.
- `cx --json top --processes --sort cpu` / `cx --json memory` — per-surface CPU/RSS, i.e. "is my app
  spinning?" without leaving cmux.
- `cx --json surface-health` — cmux's own view of whether surfaces are alive.

---

## 4. Pattern 2 — read the output

```sh
# the visible screen (fast, bounded, what a human sees)
cx --json read-screen --surface <sid> --lines 200        # -> {"text": "..."}

# include scrollback when you need history the screen has already lost
cx --json read-screen --surface <sid> --scrollback --lines 2000

# tmux-compatible alias, same thing
cx --json capture-pane --surface <sid> --scrollback --lines 2000

# search across surfaces by content, and optionally jump to the hit
cx find-window --content "Traceback (most recent call last)" --select

# reset a noisy pane before a fresh run so your next read is unambiguous
cx clear-history --surface <sid>
```

**Know what `read-screen` is.** It is a *terminal screen dump*, not a log file: lines are
hard-wrapped at the pane width, ANSI/TUI redraws may have overwritten earlier text, and anything past
the scrollback limit is gone. That is fine for "what state is this in right now" and bad for "parse
every log record". For the latter, use §5.

Practical shape for a poll loop: read the **tail** (`text[-2000:]`) and match on it. Full-screen
matching gets confused by prior runs still on screen.

---

## 5. Pattern 3 — a durable log file (`pipe-pane`)

`read-screen` samples; `pipe-pane` **streams**. It pipes the surface's output into a shell command of
your choosing, which is how you get a real, complete, greppable log:

```sh
# tee everything this surface emits into a file, from now on
cx pipe-pane --surface <sid> --command "tee -a /tmp/app.log"

# then monitor like any other log — no cmux involved
tail -f /tmp/app.log
grep -nE "ERROR|FATAL|Traceback|panicked at" /tmp/app.log

# or filter at the source
cx pipe-pane --surface <sid> --command "grep --line-buffered -E 'ERROR|WARN' >> /tmp/app.errors.log"
```

> 📖 `pipe-pane` is transcribed from `cmux --help` (tmux-compatibility section) and has **not** been
> exercised in this repo. Verify the exact semantics on first use — in particular whether it toggles
> off when re-invoked without `--command` (tmux behaviour) and whether it captures raw bytes
> including escape sequences (in tmux it does; `sed`-stripping ANSI may be needed for clean logs).

**If `pipe-pane` disappoints, don't fight it.** The boring alternative is better and fully in your
control: have your app write its own log and let cmux run the tail.

```sh
cx new-workspace --name "app" --cwd /path/to/app --command "npm run dev 2>&1 | tee -a /tmp/app.log"
cx new-workspace --name "logs" --cwd /tmp --command "tail -F /tmp/app.log"
```

Then you get both: a real file for parsing, and a pane a human can glance at.

---

## 6. Pattern 4 — react (answer prompts, unblock the app)

```sh
cx send     --surface <sid> "yes"      # type text
cx send-key --surface <sid> Enter      # a named key: Enter, Tab, Escape, Up, ctrl+c …
cx send-key --surface <sid> ctrl+c     # stop a runaway process
```

Two rules learned the hard way:

1. **Read before you send.** Confirm the pane is actually at the prompt you think it is
   (`read-screen`), then send. Blind sends land in whatever has focus of that surface's PTY.
2. **Never send into a surface you did not just enumerate by UUID.** See §2.

There is also a synchronization primitive:

```sh
cx wait-for mybuild                    # block until signalled
cx wait-for -S mybuild                 # signal it (from inside the app's pane, post-build)
cx wait-for mybuild --timeout 300      # bounded wait
```

📖 Unexercised in this repo. It is the clean way to say "wait until the app says it's ready" instead
of `sleep 5` — worth verifying if you need it.

---

## 7. Pattern 5 — put state where it's visible (this is cmux's real edge)

A monitor nobody reads is not a monitor. cmux gives you four distinct UI channels, all
socket-driven — use them instead of printing into your own scrollback:

```sh
# 1. Notifications — badge the tab, ring the pane. THIS is "come look at me".
cx notify --surface <sid> --title "api-server — crashed" --body "OOM at 14:32, restarted 3×"
cx --json list-notifications                       # ✅ this is how you PROVE a notify landed
cx mark-notification-read --all ; cx clear-notifications

# 2. Status chips — persistent key/value on the workspace (build state, git branch, error count)
cx set-status build "failing" --icon xmark --color '#ff3b30' --priority 10 --workspace <ws>
cx --json list-status --workspace <ws> ; cx clear-status build --workspace <ws>

# 3. Progress — a real progress bar for long runs
cx set-progress 0.42 --label "migrating 42/100" --workspace <ws> ; cx clear-progress --workspace <ws>

# 4. The workspace log panel — structured entries in the UI, separate from terminal noise
cx log --level error --source api-server "500 on /v1/orders (12 in 60s)" --workspace <ws>
cx --json list-log --workspace <ws> --limit 50 ; cx clear-log --workspace <ws>
```

Also: `cx workspace status set lane|auto`, `cx todo add|list|check` (a per-workspace checklist an
agent can drive), `cx trigger-flash` (visual ping), and `cx markdown open <path>` — a
**live-reloading** markdown viewer, which makes "write findings to a file and keep it open beside the
app" a genuinely good monitoring UI.

> 📖 Everything in §7 except `notify` / `list-notifications` is transcribed from `--help` and
> unexercised here. `notify` + `list-notifications` are **live-proven** (three real badges, 2026-08-06).

---

## 8. Pattern 6 — stop polling, subscribe

```sh
# stream cmux's own event bus; --cursor-file makes it resumable across restarts
cx events --cursor-file /tmp/cmux.cursor --reconnect

# filter, or replay from a sequence number
cx events --name <event> --category <category> --after <seq> --limit 100 --no-heartbeat --no-ack
```

There is also a hook mechanism — cmux runs *your* command when an event fires:

```sh
cx set-hook --list
cx set-hook <event> '<shell-command>'
cx set-hook --unset <event>
```

📖 Both unexercised here; the event/category names are not documented in `--help`, so **discover them
empirically**: run `cx events --limit 20 --no-ack --no-heartbeat`, do the thing you care about, and
read what comes back. Also note `~/.cmuxterm/events.jsonl` on disk as an offline record of the same
bus. Until you have verified event names, a 10–20s `read-screen` poll loop (§10) is the safe default —
that is what is proven in this repo.

---

## 9. "Is my server actually up?" — cmux knows your ports

Verified live: each workspace object in `cmux --json list-workspaces` carries a **`listening_ports`**
array (plus `remote.conflicted_ports`). cmux tracks what the processes in that workspace are
listening on, so you get liveness without `lsof` gymnastics:

```sh
cx --json --id-format both list-workspaces \
  | python3 -c 'import json,sys; [print(w["ref"], w["current_directory"], w["listening_ports"]) for w in json.load(sys.stdin)["workspaces"]]'
```

Poll that until your port appears → "app is up" without parsing a startup banner out of a screen
dump. Other observed fields worth knowing: `current_directory`, `latest_submitted_at`,
`latest_conversation_message` (for agent surfaces), `pinned`, `index`.

---

## 10. Putting it together — the monitoring loop

This is the shape that is **live-proven** in this repo (`scripts/cmux/orchestrator.py`, validated
2026-08-06 against real cmux): *enumerate → read → decide → act*. Copy the shape; swap the decision
step for whatever your app needs.

```sh
#!/bin/sh
# watch.sh — one sweep. Loop it from a cmux pane: while :; do ./watch.sh; sleep 20; done
set -u
CX="${CMUX_BIN:-/Applications/cmux.app/Contents/Resources/bin/cmux}"

# 1. enumerate — UUIDs, not refs
WS=$("$CX" --json --id-format both list-workspaces | python3 -c 'import json,sys;print(json.load(sys.stdin)["workspaces"][0]["id"])')
SID=$("$CX" --json --id-format both list-pane-surfaces --workspace "$WS" \
      | python3 -c 'import json,sys;print(next(s["id"] for s in json.load(sys.stdin)["surfaces"] if s["type"]=="terminal"))')

# 2. read the tail
TEXT=$("$CX" --json read-screen --surface "$SID" --lines 120 | python3 -c 'import json,sys;print(json.load(sys.stdin)["text"][-2000:])')

# 3. decide (cheap deterministic rules first; escalate to a model only if you must)
case "$TEXT" in
  *"Traceback"*|*"FATAL"*|*"panicked at"*)
      "$CX" notify --surface "$SID" --title "app — error" --body "crash detected in tail" ;;
  *"[y/N]"*|*"Press Enter"*)
      "$CX" notify --surface "$SID" --title "app — waiting" --body "blocked on a prompt" ;;
esac
```

Design notes that came out of the live run:

- **Skip non-terminal surfaces.** `type` can be `browser` / `agent-session`; reading a browser
  surface as a log is meaningless. Filter on `type == "terminal"`.
- **Default to silence.** Map only `waiting` / `error` → notify. An unknown state must produce *no*
  notification, or you train yourself to ignore the badges.
- **Count what warrants attention, not what you sent.** A dry-run mode that counts notifications
  *sent* always reports `0/N` and makes a working sweep look broken. (Real bug, real hour lost.)
- **A cmux pane is a fine host for the loop** — and under Route A it is the *only* host.

### Optional: let a local model do the triage (this repo's angle)

If pattern-matching is too brittle for your app's output, `scripts/cmux/orchestrator.py` already does
the whole sweep with an **on-device** model classifying each pane into
`working | waiting | done | error` and summarizing a one-line notification body — **0 frontier
tokens**, no pane text leaving the machine:

```sh
cd /Users/miltronix/Claude/apps/HEARTH
. scripts/cmux/cmux-auth-env
HEARTH_BACKEND=mlx uv run python scripts/cmux/orchestrator.py --dry-run   # drop --dry-run to notify
```

Live result: 4 workspaces in distinct states, **4/4 classified correctly**, 3 real notification
badges. It is ~270 lines with a `FakeCmuxClient` for tests — read it as the reference implementation
of everything in §2/§4/§7. Details: `docs/cmux/RUNBOOK_orchestrator.md`.

To wire your app's *agent* (not the monitor) to the same local model — MCP tools or an
`OPENAI_BASE_URL` swap, config-only — see `docs/cmux/RUNBOOK_wiring.md` and
`examples/cmux/hearth.mcp.json`.

---

## 11. Web app? Use the built-in scriptable browser

If the thing you are monitoring has a UI, cmux has a Playwright-shaped browser **inside** the
terminal, on the same socket:

```sh
cx browser open http://localhost:3000
cx browser wait --selector "#app" --load-state complete --timeout-ms 10000
cx browser --json console list          # console messages — front-end log monitoring
cx browser --json errors list            # uncaught JS errors
cx browser screenshot --out /tmp/app.png --json
cx browser snapshot --interactive --compact   # a compact DOM/a11y tree, token-cheap
cx browser eval 'window.__APP_STATE__'
cx browser click "#retry" --snapshot-after
```

`browser console list` + `browser errors list` are the browser-side equivalent of tailing a log — for
a web app they are often the highest-signal monitor you can get. `browser snapshot --compact` exists
precisely so an agent can look at a page without burning context on raw HTML.

📖 Live-unexercised in this repo (the browser subsystem is documented and audited here, but our
validations covered terminal surfaces). `cmux docs browser` prints the upstream reference.

---

## 12. Verification ledger — what is proven vs read from `--help`

| Area | Status |
| --- | --- |
| CLI path, `--version`, socket path, `access_mode: password` | ✅ verified live 2026-08-19 |
| `list-workspaces` / `list-pane-surfaces` / `tree` JSON shapes, `listening_ports` | ✅ verified live 2026-08-19 |
| `read-screen --lines`, `send`, `send-key`, `notify`, `list-notifications` | ✅ live-validated 2026-08-06 / 2026-08-17 |
| Stale positional refs → use `--id-format both` + UUIDs | ✅ proven by failure, then fixed |
| Errors on **stdout** with rc=1 | ✅ proven by failure, then fixed |
| `pipe-pane`, `wait-for`, `events`, `set-hook`, `set-status`, `set-progress`, `log`/`list-log`, `capture-pane --scrollback` | 📖 from `cmux --help` (0.64.20) — **unexercised here, verify on first use** |
| `browser *` | 📖 documented upstream, unexercised in this repo's validations |

Ground truth beyond this doc: `cmux --help` (the full command list), `cmux docs api|browser|agents|settings`,
`cmux capabilities` (the live JSON-RPC method list — the real contract), and upstream
`docs/cli-contract.md` + `skills/cmux/SKILL.md` in `manaflow-ai/cmux`.

---

## 13. Gotchas ledger (each one cost someone real time)

1. **The CLI isn't on `PATH`;** `Contents/MacOS/cmux` is the GUI. Use `Contents/Resources/bin/cmux`.
2. **Refs are positional and go stale.** `--id-format both`, act on UUIDs. Symptom:
   `Surface ref not found` for a ref you *just* enumerated.
3. **cmux reports failures on stdout with rc=1.** A bare `check_returncode()` hides
   `Surface ref not found` behind "returned non-zero exit status 1". Capture and re-raise stdout.
4. **Socket auth is a *default*, not a law.** "Must run inside a pane" is only true under
   `cmuxOnly`. `password` lets you drive it from anywhere (ADR-C007).
5. **`errno 1` on connect is your sandbox, not cmux.** See §1.3.
6. **Legacy command names print an alias notice** (`list-workspaces` → `workspace list`). It does not
   appear to pollute `--json` stdout, but if you merge `2>&1` into a JSON parser it will. Set
   `CMUX_QUIET=1` in any script that parses output.
7. **`read-screen` is a screen, not a log.** Wrapped, redrawn, truncated. Use `--scrollback`, or a
   real file (§5), when completeness matters.
8. **Don't trust an agent's self-report.** A harness that asked an agent to print
   `TOOL_USED=yes|no` got `yes` — proving nothing. Assert on machine-readable evidence
   (`list-notifications`, a log file, `--output-format stream-json` tool_use records).
9. **cmux rewrites `~/.config/cmux/cmux.json` on launch** (strips comments, migrates secrets, keeps
   its own `.bak`). Back up before editing, and don't expect your edits to survive verbatim.
10. **`cmux reload-config`** reloads both `cmux.json` and `~/.config/ghostty/config` in place — no app
    restart needed. Prefer Ghostty config for anything terminal-level (font, theme, keybinds).
11. **App flags regress.** `SUEnableAutomaticChecks` flipped `0` → `1` on its own with no app update.
    If you depend on a cmux flag, re-verify it every session; do not assume it stuck.

---

## 14. Before you point cmux at anything sensitive (read this)

This repo runs cmux under a **two-tier gated model** (sealed / open). Three facts about the machine
you are on right now:

- **🚨 The seal does NOT contain your panes. Do not put confidential data in a cmux pane.**
  Measured 2026-08-19: from inside a pane, `curl https://example.com` returned **HTTP 200** and
  `python3` opened a TCP session to **1.1.1.1:443** — at the same moment
  `scripts/cmux/lulu_rule_check.py` reported `SEALED` and `cmux-sealed`'s firewall gate reported
  `PASS`. The LuLu rule is scoped to the **`com.cmuxterm.app` binary**; the processes a pane spawns
  (`node`, `claude`, `curl`, `git`, `ssh`, `python3`) are evaluated by LuLu independently, have **no
  rules at all**, and LuLu is configured `passiveMode=true` + `allowApple`/`allowInstalled=true` —
  i.e. **allow-by-default**. A pane has ordinary, unrestricted internet access.
- **`git push` from a pane works.** An earlier session note claiming it fails under the seal is
  **wrong** — same root cause as above.
- **cmux is not egress-clean on app flags alone.** With telemetry off and signed out it still hit
  GitHub via Sparkle's launch-time update check (and `SULastCheckTime` shows a check ran 2026-08-18
  with `SUEnableAutomaticChecks=0`). Only the firewall stops cmux's *own* traffic — and only its own.

What the seal **does** cover, verified: cmux.app's own outbound traffic (block rule + live network
extension, `lulu_rule_check.py` exit 0; RESULTS §1.3 probe loopback-only). That is a much narrower
claim than "confidential work is contained", and it must not be read as the latter.

Rules of engagement:

- Launch through the tier launchers rather than raw `cmux <path>`:
  `scripts/cmux/cmux-sealed [REPO]` (confidential; fails closed if it cannot prove no-egress) or
  `scripts/cmux/cmux-open [REPO]` (only for a repo explicitly classified `open` in
  `config/cmux/tiers.yaml`; falls back to sealed for anything unclassified). `--check` on either
  verifies without launching. **Note `cmux-sealed --check --strict` currently exits 2 (NOT SEALED)** —
  it fails the `cli-telemetry` env gate — so by its own design it will refuse to launch sealed today.
- **Nothing enforces which agent runs in a sealed pane.** `cmux-sealed`'s checks cover tier, HEARTH
  routing, firewall, telemetry, auto-update, CLI telemetry and sign-in — **not** what you launch in
  the pane. A frontier coding agent in a "sealed" pane ships whatever it reads to its own cloud
  model, and no gate stops you. PRIVACY.md's "the gate is designed so you can't" describes the
  intent, not the current implementation.
- **Unclassified ⇒ sealed.** You opt *into* cloud, never out of it.
- **Never add a transport that ships pane text off-box.** Pane contents are the sensitive artifact in
  this whole design; the monitor is allowed to send them to a local model and nowhere else.

Background, if you need the reasoning: `docs/cmux/PRIVACY.md`, `docs/cmux/ARCHITECTURE.md` §3,
`docs/cmux/DECISIONS.md` (ADR-C001/C003/C006/C007/C008), `docs/cmux/TODO.md`.
