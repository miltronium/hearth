# HEARTH — RESUME HERE (handoff, parked 2026-10-06)

Read this first, then CLAUDE.md. Written for a human or an agent picking up mid-stream. §1–§3
were measured on 2026-10-05; the RESUME HERE block and §4 on 2026-10-06. Re-verify with §2.

## ⏩ RESUME HERE

- **Branch:** `cmux/integration`. Pushed through `17c2559`. **`1ec3b60` (B-129) is local,
  not pushed.** Suite: 1840 passed, 2 skipped.
- **Parked 2026-10-06 mid-verification of B-129.** The fix is in. One real-socket test was not
  runnable from the Claude sandbox (no `bind()`). Next, the operator runs, in their terminal:
  `uv run --no-sync pytest -v -p no:cacheprovider tests/test_cli_startup_errors.py -k real_port_conflict`
  (expect `1 passed`, not `skipped`). Then, with a server holding 8080,
  `uv run --no-sync hearth serve; echo "exit=$?"` should print `Cannot listen on 127.0.0.1:8080`,
  no `Serving on`, no `loaded`, and `exit=1`. Record the result in BUGS B-129, then ask
  before pushing `1ec3b60`.
- **Operator verification done 2026-10-06:** YOUR_TURN §1–§4, §6, §7 and §8 all passed or were
  decided. Results are recorded in `docs/YOUR_TURN.md` ("Done 2026-10-06") and `docs/BUGS.md`.
- **Next after that:** grow `data/agent_eval.yaml` toward n ≥ 30 (B-013); §5 golden sets
  (operator's examples); B-002 cmux containment (P0, design first).
- **Learn it:** `docs/GUIDE.md`, `man ./man/hearth.1`, `uv run --no-sync hearth --help`.

## 1. What is true now

| Area | State |
|---|---|
| Offline | Default routing has zero remotes; every model load is disk-only (`HEARTH_ALLOW_DOWNLOADS=1` to opt in; `hearth models pull` is the one download path). `hearth doctor --offline` gives a measured SAFE / UNSAFE / NOT-SAFE-TO-RUN verdict (backend, embedder and vector-store plugins, routing, every reachable model, every load path, bind host). |
| Models | The requested model serves (ModelPool, LRU under `HEARTH_RAM_CEILING_GB`, adapter variants counted); unknown ids → 404 `model_not_found` (recorded); `auto` = the per-class ladder; routing rungs validated at load (registered, chat-capable, servable, fits the ceiling). |
| Readiness | `/v1/hearth/admin/ready` judges every model the active profile can route `auto` to (on disk, fits, loaded once, last load ok), per-model detail in the body; warmup loads the most-used rungs that fit. |
| Accounting | Every request — served, failed, refused (404/429), abandoned — writes one RequestRecord; `hearth stats` shows failures; every SSE stream ends with `[DONE]`; echo always reports `echo`. |
| Privacy | `hearth_peek.py` and `hearth_map_draft.py` share `hearth.finance.shape`: only allowlisted column-vocabulary names, counts, type guesses and file ids are printed (stdout AND stderr), by construction; drafts get the right `skip_rows`. |
| Promotion gate | Bar range-checked (finite, alpha ∈ (0,0.05], min_effect ≥ 0, min_n ≥ 30); `adapters promote --report` requires an HMAC-signed report from this install tied to adapter id/task/base/weights hash; prereg must be committed (bytes on disk = committed blob) before the measurement; `tests/test_eval_gate_replay.py` still refuses the historical promotion. See the latest gate commits for round-two hardening (measurement ledger, distinct-example n, atomic promote). |
| Docs | `docs/GUIDE.md`, `man/hearth.1` (generated; drift test), complete `--help` (test walks the app), `docs/API.md` (route drift test), `docs/BUGS.md` (verified backlog with "Fixed recently"). |

## 2. Verify before doing anything

```sh
cd ~/Claude/apps/HEARTH
git status --short && git log --oneline -3
uv run --no-sync python -c "import mlx_lm, mcp, openpyxl, pypdf; print('venv ok')"
uv run --no-sync pytest -q -p no:cacheprovider        # all green expected
uv run --no-sync hearth doctor --offline              # SAFE offline, exit 0
uv run --no-sync python scripts/gen_manpage.py --check
```

Kernel-level proof that nothing leaves the machine (agents could not run this — B-030):

```sh
cat > /tmp/hearth_noegress.sb <<'EOF'
(version 1)
(allow default)
(deny network-outbound)
(allow network-outbound (remote ip "localhost:*"))
(allow network-outbound (remote unix-socket))
EOF
sandbox-exec -f /tmp/hearth_noegress.sb env -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY \
  -u https_proxy -u ALL_PROXY -u HF_HUB_OFFLINE uv run --no-sync hearth doctor --offline
```

Real-TCP smoke test of model selection (agents could not bind ports this session):

```sh
HEARTH_PORT=8771 uv run --no-sync hearth serve &          # wait for "Serving on"
TOKEN=$(cat ~/.hearth/token)
for m in mlx-community/Qwen2.5-3B-Instruct-4bit mlx-community/Qwen2.5-Coder-7B-Instruct-4bit; do
  curl -s http://127.0.0.1:8771/v1/chat/completions -H "Authorization: Bearer $TOKEN" \
    -H 'content-type: application/json' \
    -d "{\"model\":\"$m\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}" \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['model'])"; done
curl -s -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8771/v1/hearth/admin/models
kill %1
```

## 3. What happened on 2026-10-05 (133 commits since 13b1438; suite 1791 passed, 1 skipped)

Model selection (B-004/B-005) merged and live-verified in-process with real 3B/7B/14B weights;
three bug batches (CLI, gateway, docs hygiene); the user guide, man page and full `--help`;
then **four adversarial review rounds**, each of which found real defects that were fixed with
tests that fail on the old code and survive mutation checks. Notable catches:
- **Promotion gate:** a hand-typed report could promote a nonexistent adapter; a NaN /
  vacuous bar passed a 0.033-vs-1.0 candidate; `git update-index --assume-unchanged` hid a
  post-hoc prereg/golden edit; a prereg could be written after seeing the score.
- **Privacy:** `hearth_peek.py` and `hearth_map_draft.py` printed statement preambles
  ("Jane Q Public") as headers while claiming "no cell values were printed"; openpyxl leaked
  workbook metadata (incl. an SSN-shaped property) to stderr; and (round 4) MCP tool errors
  returned spreadsheet cell text verbatim to the calling agent (B-115).
- **Gate, rounds 3–4:** `git replace` could fake prereg history; extra files reset an adapter's
  history; a "menu" of preregs allowed picking the bar after seeing scores (B-120..B-125).
- **Serving:** `/ready` green while a whole class 503'd (ladder; RAM ceiling); concurrent chat
  failing on MLX thread-locality; abandoned streams holding the GPU; a bad adapter forcing a
  base reload on every request.
Full record: `docs/BUGS.md` → "Fixed recently".

## 4. Work queue

**First: `docs/YOUR_TURN.md`** — the operator's list (kernel-level + real-TCP verification, the
`/chat` browser check, the classify-adapter decision, golden sets, cleanups, push/PR).

Done 2026-10-06 (agent, alone): B-126 ledger test gaps closed; B-123 documented as residual
trust (RUNBOOK_training §6); all seven round-3 gate fixes independently mutation-verified;
`search_files` agent tool (B-013, live: 2 steps instead of reading files in order); HTTP agent
gets ledger tools + a pinnable RAG collection (B-012); train_lora_real.sh --promote uses the
real gate (B-015); dead constant removed (B-022); stale docs marked historical and monitored
(B-016/B-017).

Done 2026-10-06 (with the operator): kernel-sandbox no-egress proof (§1); real-TCP model
selection, 404 and disconnect (§2); `/chat` incl. agent mode in a browser (§3, B-014 half);
classify adapter retired (§4, B-010 half); httpx2 (B-020); 0.5B weights kept (B-019); pktap
capture, no HEARTH traffic (§8); branch pushed; agent smoke eval `data/agent_eval.yaml`
(B-013); `read_file` ranges and truthful truncation hints (B-130); serve binds before the
banner and warmup (B-129, real-socket test pending). B-009: operator said "maybe", so it stays
open.

Open as of 2026-10-05 (superseded in part by the line above): P0 **B-002** (cmux pane containment — no confidential work
through cmux); P1 B-009 (confidence stub, remote profile only), B-010 (promoted classify
adapter — operator decision, YOUR_TURN §4); P2 B-014 (browser check — YOUR_TURN §3); P3
B-019, B-020 (operator), B-021, B-023, B-028, B-030 (environment / minor). Remaining agent work
when the budget allows: one more independent review of the round-3 gate changes; B-028; the
agent tool-calling eval set once the operator supplies tasks (B-013 remainder).

## 5. Gotchas that cost real time

- **`uv run --no-sync` always** (a default sync uninstalls 33 packages incl. mlx).
- **Agent worktrees start stale** (Agent `isolation: worktree` created them at `75ba6ad`): every
  agent must `git merge --ff-only cmux/integration` first and confirm `import hearth` resolves
  into its worktree. Agents use the main venv: `PYTHONPATH=<wt>/src ~/Claude/apps/HEARTH/.venv/bin/python`.
- **Loopback curl is proxied** in the Claude shell (`NO_PROXY` empty) and the proxy hides client
  disconnects — use a raw socket for lifecycle tests. The hook blocks `curl --noproxy`.
- **`sandbox-exec` and `bind()` became unavailable** to agents and to the session after the
  harness moved it into a worktree (exit 71 / EPERM).
- **Mutation-check every test.** Repeatedly this session a test passed with its fix removed
  (refcounting closed a generator; a broad `except` swallowed a network attempt; Typer read
  `TERMINAL_WIDTH` once at import so an "80-column" test ran at 300).
- **Validate the instrument**: several "results" were artifacts (zsh not word-splitting `$VAR`;
  `pip` missing so a package count read 0; `"$c"` passing `run hi` as one argument).
- **2026-10-06:**
  - The agent eval extracted its corpus to a random temp path. The path is in every
    observation, so identical code scored 9/10 and then 8/10. It now uses a fixed path; keep
    eval prompts byte-stable.
  - pktap tags many *outbound* packets `()`; only replies carry `eproc`.
  - `-W error::<dotted class>` is ignored by plain `python` (resolved only by pytest).
  - `StarletteDeprecationWarning` is a `UserWarning`.
- **Never run `uv add`/`uv lock` from a Claude session.** The session sets `UV_DEFAULT_INDEX`
  to an internal mirror and rewrites `uv.lock`. The repo is public, so the operator runs
  lockfile changes.
- **Commands for the operator must be paste-ready.** zsh treats a pasted `# comment` as an
  error or as extra arguments, so put expectations in prose.

## 6. Security note for the operator

`~/.claude/apple/tool_allowlist.csv` has 31 `contains_match` entries (as of 2026-10-06),
including `python3`, `bash`, `sudo`, `curl`, `git` and `acc`. Any command *containing* one of
them runs outside the Claude Code sandbox. It is the operator's config; agents must not edit
it. See YOUR_TURN §6.

## 7. Operator's standing instructions

"Assume nothing, verify everything." Subagents by priority, writers isolated in worktrees,
independent adversarial reviewers check all work. Priority: safe offline use; the bug backlog;
docs/man/help to learn HEARTH fast. Merge to `cmux/integration` locally; ask before push/PR.
