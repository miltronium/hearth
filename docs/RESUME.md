# HEARTH — RESUME HERE (handoff, 2026-10-03)

Read this first, then CLAUDE.md. Written for a human or an agent (Claude, or `hearth agent`)
picking up mid-stream. Every claim below was measured on 2026-10-03; re-verify with the
commands in §2 before trusting it — status rots, measurements don't.

## ⏩ RESUME HERE

- **Branch:** `cmux/integration` — code as of `35708f5`; this doc is the commit right after
  it (local only — **nothing has been pushed**).
- **State:** HEARTH is safe to use offline by default (§1). Item 2 ("the model you pick is the
  model that serves" + truthful `/ready`) is **code-complete but not live-verified**, committed
  as WIP `d1d8434` on branch `worktree-agent-ac0ade7fd1fa99aba` (§4.1). Not merged.
- **Do next, in order:**
  1. Verify state (§2).
  2. Finish item 2 (§4.1), verify live, merge, update `docs/BUGS.md` B-004/B-005.
  3. Docs / man page / CLI help phase (§4.2) — the operator wants to learn HEARTH ASAP.
  4. Independent adversarial verification of everything merged (§4.3).
  5. Work the backlog in `docs/BUGS.md` by priority (P0 B-002, B-003 first).

## 1. What is true now (merged on `cmux/integration`)

| Area | State | Evidence |
|---|---|---|
| Routing | `config/routing.yaml` (the default) is **no-egress**: 0 remotes, every class local/never. Claude escalation is opt-in: `HEARTH_ROUTING_YAML=config/routing.remote.yaml`. | `tests/test_policy.py::test_bundled_default_is_no_egress` |
| Model loads | Every load path is **disk-only** (serve, chat, agent, MCP, RAG, embedder, `train`, `models convert`, `models export-coreml`). Only `hearth models pull` downloads. Opt-in: `HEARTH_ALLOW_DOWNLOADS=1`. | `tests/test_offline_model_resolution.py`, `tests/test_offline_load_paths.py` |
| Offline check | `hearth doctor --offline` → measured SAFE/UNSAFE verdict, exit 1 when unsafe. | SAFE under a kernel deny-egress sandbox; red for remote profile, downloads on, `HEARTH_HOST=0.0.0.0` |
| Failed escalation | Served locally, recorded as `escalations_failed` in `hearth stats`. | `tests/test_escalation_fallback.py` |
| Concurrency | All MLX work on one thread; concurrent chat works (was 2/3 failing). | live: 9/9 concurrent requests 200 on Coder-14B |
| Disconnects | Abandoned stream stops generating (was holding GPU ~71 s). | live: next request 0.57–0.62 s |
| `uv` | No bare `uv run` anywhere; install hints use the one-command sync. | grep |
| Backlog | `docs/BUGS.md`: 30 items, verified with file:line, priorities, acceptance tests. B-001 closed. | — |

Test suite on `35708f5`: **1024 passed, 1 skipped**.

Key commits: `d803f1a` (merge of the 13-commit offline branch), `8c7220e` (BUGS.md),
`13105ef` (merge: disk-only train/convert/coreml + `doctor --offline`), `35708f5` (BUGS update).
Each commit message has WHAT / WHY / HOW VERIFIED.

## 2. Verify the state before doing anything

```sh
cd ~/Claude/apps/HEARTH
git status --short && git log --oneline -5          # expect this doc's commit, then 35708f5
uv run --no-sync python -c "import mlx_lm, mcp, openpyxl, pypdf; print('venv ok')"
uv run --no-sync pytest -q -p no:cacheprovider      # expect 1024 passed, 1 skipped
uv run --no-sync hearth doctor --offline            # expect: SAFE offline, exit 0
uv run --no-sync python scripts/hearth_status.py    # full measured status
git worktree list                                   # see §5
```

If the venv is broken: `uv sync --extra mlx --extra mcp --extra dev --extra files` (ONE
command — syncing one extra uninstalls the others; CLAUDE.md §1).

## 3. Using HEARTH offline today (operator quick start)

```sh
uv run --no-sync hearth doctor --offline            # is it safe right now?
uv run --no-sync hearth serve                       # http://127.0.0.1:8080/chat (token: ~/.hearth/token)
HEARTH_FILE_ROOTS=~/some/dir uv run --no-sync hearth agent "how many CSV files are there?"
uv run --no-sync hearth run "summarize: ..."        # one-shot local completion
```

Caveat until item 2 lands: the `/chat` model dropdown and per-request `model` are ignored —
the default model (Coder-7B, or `HEARTH_DEFAULT_MODEL` if it names a registry entry) serves.

## 4. Work queue

### 4.1 Item 2 — model selection is real; `/ready` is truthful (IN PROGRESS)

**Committed as WIP `d1d8434`** on branch `worktree-agent-ac0ade7fd1fa99aba` (worktree
`~/Claude/apps/HEARTH/.claude/worktrees/agent-ac0ade7fd1fa99aba`, based on `d803f1a`, clean
tree). Read its commit message first (`git -C <worktree> log -1`) — it lists what is done,
what is not, and mutation results. Independently re-run at handoff: **1023 passed, 1 skipped**.

Done in the WIP (unit-tested with a fake mlx_lm, 30 tests, 11 of 12 mutants killed):
`serving/pool.py` `ModelPool` (one `MLXProvider` per registry id, LRU under the RAM ceiling,
everything on the MLX thread); unknown ids → 404 `model_not_found` / CLI exit 2; telemetry
from the provider that ran; warmup really loads; `/ready` 503 "loading"/"failed" with reason;
new `GET /v1/hearth/admin/models`; `hearth run/agent --model` default to `auto` (the ladder).

**Remaining before merge:**
1. Rebase/merge onto current `cmux/integration` (it is based on `d803f1a`; integration has
   since gained `doctor --offline`, disk-only train/convert, BUGS.md). Expect conflicts in
   `cli.py` and `providers/mlx.py`; re-run the full suite after.
2. Add a timeout to `test_concurrent_churn_never_holds_weights_outside_the_ceiling` so mutant
   M10 fails cleanly instead of HANGING (deadlock), then run M10b (stream path) — not run.
3. **Live verification — none done yet.** No server was started, no real weights loaded.
   Do the live acceptance below, reading `/v1/hearth/admin/models` (loaded_path,
   generations per instance) and the server INFO log for server-side proof.
4. `/chat` dropdown lists echo and the embed model (now 404 instead of silent default) —
   filter to servable models. `examples/finance/run_finance_ladder.py` still has its own
   `LadderProvider`. `docs/API.md` lacks the 404, `/ready` reasons and admin/models.
5. Then merge, update BUGS.md B-004/B-005 (and B-029: `/ready` now reports a typo'd
   `HEARTH_DEFAULT_MODEL`, but `Registry.default_id` still silently ignores it).

Goal (BUGS.md B-004, B-005):
- `MLXProvider` serves the requested model; one provider per model id held by the
  `ModelManager` (LRU under `ram_ceiling_gb`); loading/eviction on the single MLX thread.
- Unknown model id → clear error (OpenAI-style 404 `model_not_found` / CLI error), never a
  silent fallback. `auto`/empty → the per-class ladder / default.
- Telemetry reports the model that actually generated (derived from the provider that ran).
- Warmup actually loads the default weights; `/ready` is 503 until loaded and when the load
  failed.
- Live acceptance: serve with 7B default; alternate `model=3B` / `model=7B` (stream,
  non-stream, concurrent) and prove server-side which weights generated each; show
  `routing.finance.yaml` sends a classify prompt to the 3B; `/ready` before/after warmup and
  with a bogus default.

### 4.2 Docs, man page, CLI help (NOT STARTED — start after 4.1 merges)

Survey done 2026-10-03: there is no `hearth` man page (only `scripts/cmux/cmux-orchestrator.1`),
no single user guide, and thin help (`serve --help` says nothing about profiles, offline,
token or `/chat`; `doctor --help` has no description). Plan:
1. `docs/GUIDE.md` — start-to-finish user guide: install/verify, models (`models pull/list`),
   `doctor --offline`, `serve` + `/chat` + token, chat API, `hearth agent`, `hearth run`,
   RAG, MCP (Claude Code offload), finance pointer, training pointer, routing profiles,
   full `HEARTH_*` env-var reference, troubleshooting, glossary.
2. `man/hearth.1` generated from the Typer app (every command, option, env var) + a test that
   fails if a command or option is missing from it (drift = red).
3. Complete `--help` for every command/option, with examples and env vars; a test that every
   command has a description and every option has help text.
4. Refresh `README.md` quick start; add `docs/README.md` index.
5. Verify every documented command by running it (offline).

### 4.3 Final adversarial verification (NOT STARTED)

A fresh agent tries to REFUTE the merged work (as done for the offline branch — it found 8
real issues). Then fix, re-verify, update BUGS.md.

## 5. Worktrees and branches

| Path | Branch | State |
|---|---|---|
| `~/Claude/apps/HEARTH` | `cmux/integration` | main checkout; untracked `config/cmux/tiers.yaml`, `examples/repos/` are intentional |
| `~/Claude/apps/HEARTH-wt-offline` | `fix/offline-by-default` | merged (`d803f1a`) — safe to remove |
| `.claude/worktrees/agent-a62e0c3304b322af6` | `worktree-agent-a62e0c…` | merged (`13105ef`) — safe to remove |
| `.claude/worktrees/agent-acee173b3fb359e44` | `worktree-agent-acee…` | merged (`8c7220e`) — safe to remove |
| `.claude/worktrees/agent-ac0ade7fd1fa99aba` | `worktree-agent-ac0ade…` | **item 2 WIP `d1d8434` — keep** |

Remove merged ones with `git worktree remove <path>` (check `git -C <path> status` is clean first).

## 6. Gotchas that cost real time this session

- **`uv run --no-sync` always.** Verified: a default sync uninstalls 33 packages incl. mlx.
- **Agent worktrees start stale.** The Agent tool's `isolation: worktree` created worktrees at
  `75ba6ad`, 86 commits behind. Every agent must first fast-forward to `cmux/integration`
  (`git merge --ff-only cmux/integration`) and confirm `import hearth` resolves into its worktree.
- **Agents in worktrees use the main venv:** `PYTHONPATH=<worktree>/src
  ~/Claude/apps/HEARTH/.venv/bin/python …` (there is no venv per worktree).
- **Loopback curl is proxied.** `NO_PROXY` is empty in the Claude shell, so curl to
  127.0.0.1 goes through the sandbox proxy, which hides client disconnects. Disconnect tests
  need a raw socket (script below). The hook blocks `curl --noproxy`.
- **`sandbox-exec` became unavailable** to agents and to the session after the harness moved
  it into a worktree (exit 71, BUGS.md B-030). For kernel-level proof, the operator runs it.
- **Mutation-check every test.** Twice this session a test passed with its fix removed
  (refcounting closed a generator; a broad `except` swallowed a network attempt).
- **zsh**: `$VAR` holding a command line is not word-split; `echo ===` errors; failed globs abort.

### Recreate the test instruments (`/tmp` is cleared on reboot)

Deny-egress sandbox profile (validate it: raw TCP to huggingface.co must fail under it and
succeed without it):

```sh
cat > /tmp/hearth_noegress.sb <<'EOF'
(version 1)
(allow default)
(deny network-outbound)
(allow network-outbound (remote ip "localhost:*"))
(allow network-outbound (remote unix-socket))
EOF
# run a command under it, proxies and HF offline vars cleared:
sandbox-exec -f /tmp/hearth_noegress.sb env -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY \
  -u https_proxy -u ALL_PROXY -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
  uv run --no-sync hearth doctor --offline
```

Raw-socket SSE client (read N bytes then hang up — for disconnect tests):

```python
import json, socket
def abandon(port, body, read_bytes=400):
    payload = json.dumps(body).encode()
    s = socket.create_connection(("127.0.0.1", port), timeout=600)
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
              b"Content-Type: application/json\r\n"
              + f"Content-Length: {len(payload)}\r\n\r\n".encode() + payload)
    got = b""
    while len(got) < read_bytes:
        chunk = s.recv(4096)
        if not chunk:
            break
        got += chunk
    s.close()
    return got
```

Live server recipe (auth off for local testing only; pick a free port):

```sh
HEARTH_PORT=8765 HEARTH_REQUIRE_AUTH=false HEARTH_BACKEND=mlx \
HEARTH_DEFAULT_MODEL=mlx-community/Qwen2.5-Coder-14B-Instruct-4bit uv run --no-sync hearth serve
```

Models on disk (2026-10-03): Qwen2.5-Coder-7B (default), Qwen2.5-Coder-14B (hub cache only),
Qwen2.5-14B, Qwen2.5-3B, bge-small-en-v1.5-bf16 (unusable — B-011), plus unregistered
Qwen2.5-0.5B. GPU working-set ceiling 30.15 GB.

## 7. Operator's standing instructions (from this session)

"Assume nothing, verify everything." Spin up subagents by priority, isolate writers in
worktrees, and have independent agents check and verify all work. Priority order: safe
offline use first; then the bug backlog doc; then comprehensive docs, man page and CLI help
so the operator can learn HEARTH fast. Merge to `cmux/integration` locally; ask before any
push or PR.
