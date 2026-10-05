# HEARTH — RESUME HERE (handoff, 2026-10-05)

Read this first, then CLAUDE.md. Written for a human or an agent picking up mid-stream. Every
claim was measured on 2026-10-05; re-verify with §2 before trusting it.

## ⏩ RESUME HERE

- **Branch:** `cmux/integration` (local only — **nothing has been pushed**). See `git log -1`.
- **State:** HEARTH is no-egress and disk-only by default, model selection is real, the
  promotion gate has been hardened through two adversarial review rounds, and the operator
  docs (guide, man page, `--help`) are complete and drift-tested.
- **Learn it:** `docs/GUIDE.md` ("Learn HEARTH in 15 minutes" at the top), `man ./man/hearth.1`,
  `uv run --no-sync hearth --help`.
- **Do next:** see §4.

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

## 3. What happened on 2026-10-05 (110+ commits since 13b1438)

Model selection (B-004/B-005) merged and live-verified in-process with real 3B/7B/14B weights;
three bug batches (CLI, gateway, docs hygiene); the user guide, man page and full `--help`;
then **three adversarial review rounds**, each of which found real defects that were fixed with
tests that fail on the old code and survive mutation checks. Notable catches:
- **Promotion gate:** a hand-typed report could promote a nonexistent adapter; a NaN /
  vacuous bar passed a 0.033-vs-1.0 candidate; `git update-index --assume-unchanged` hid a
  post-hoc prereg/golden edit; a prereg could be written after seeing the score.
- **Privacy:** `hearth_peek.py` and `hearth_map_draft.py` printed statement preambles
  ("Jane Q Public") as headers while claiming "no cell values were printed"; openpyxl leaked
  workbook metadata (incl. an SSN-shaped property) to stderr.
- **Serving:** `/ready` green while a whole class 503'd (ladder; RAM ceiling); concurrent chat
  failing on MLX thread-locality; abandoned streams holding the GPU; a bad adapter forcing a
  base reload on every request.
Full record: `docs/BUGS.md` → "Fixed recently".

## 4. Work queue

1. **Kernel-level + real-TCP verification by the operator** (§2) — the two things no agent could
   measure this session.
2. **Open backlog** (`docs/BUGS.md`, 16 open at handoff): P0 **B-002** (cmux panes not contained
   by the seal — do not run confidential work through cmux); P1 B-009 (confidence stub under
   the remote profile), B-010 (promoted classify adapter lacks a significance proof; golden
   sets below min_n); P2 B-012/B-013 (agent: finance tools over HTTP, search tool, eval set),
   B-014 (untested: cmux open-tier launch, `/chat` agent toggle in a browser); P3 hygiene.
3. **Golden sets:** grow `data/*_golden.jsonl` to ≥ 30 distinct examples before any training.
4. **Graduation:** `cmux/integration` → `main` only after the graduation gate is green
   (docs/cmux/TODO.md). Ask before any push or PR.

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

## 6. Security note for the operator

`~/.claude/apple/tool_allowlist.csv` contains `acc,contains_match`: **any** shell command
containing the letters "acc" (accept, access, accuracy…) runs outside the Claude Code sandbox.
Found by an agent; nothing was edited and it was not used deliberately. Consider narrowing it.

## 7. Operator's standing instructions

"Assume nothing, verify everything." Subagents by priority, writers isolated in worktrees,
independent adversarial reviewers check all work. Priority: safe offline use; the bug backlog;
docs/man/help to learn HEARTH fast. Merge to `cmux/integration` locally; ask before push/PR.
