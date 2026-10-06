# HEARTH — what only you can do (operator guide, 2026-10-06)

Everything an agent could do alone is done and committed on `cmux/integration` (local; nothing
pushed). What remains needs **your machine without the Claude sandbox**, **your judgement**, or
**your data**. Ordered by value. Each item says why it is yours, the exact commands, and what
"done" looks like. Run everything from `~/Claude/apps/HEARTH`; always `uv run --no-sync`.

---

## 1. Prove nothing leaves the machine — kernel level (10 min) — *needs your terminal*

The Claude session could not run `sandbox-exec` or bind ports today (its own process was
sandboxed). All offline evidence so far is HEARTH's own connect-counting plus one kernel-level
`doctor` run on 2026-10-02. Do the real thing:

```sh
cat > /tmp/hearth_noegress.sb <<'EOF'
(version 1)
(allow default)
(deny network-outbound)
(allow network-outbound (remote ip "localhost:*"))
(allow network-outbound (remote unix-socket))
EOF
alias sb='sandbox-exec -f /tmp/hearth_noegress.sb env -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY -u https_proxy -u ALL_PROXY -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE'

# a. validate the instrument: this MUST fail with "Operation not permitted" ...
sb python3 -c "import socket; socket.create_connection(('huggingface.co',443),timeout=5)"
# ... and this (no sandbox) should connect — otherwise your network is down and step b proves nothing
python3 -c "import socket; socket.create_connection(('huggingface.co',443),timeout=5); print('net ok')"

# b. the verdict, under the kernel sandbox
sb uv run --no-sync hearth doctor --offline        # expect: SAFE offline, exit 0
# c. a real local answer, under the kernel sandbox (loads real weights)
sb uv run --no-sync hearth run "Say hello in five words."
```

**Done when:** (a) fails then succeeds as stated, (b) says SAFE, (c) prints an answer. If (c)
errors with a network message, that is a real egress bug — file it in `docs/BUGS.md`.

## 2. Real-TCP serving + model selection + disconnect (15 min) — *needs your terminal*

Model selection was live-verified with real weights but only in-process (no TCP socket).

```sh
sb uv run --no-sync hearth serve &          # wait for "Serving on http://127.0.0.1:8080"
TOKEN=$(cat ~/.hearth/token); H="Authorization: Bearer $TOKEN"; U=http://127.0.0.1:8080
curl -s $U/v1/hearth/admin/ready | python3 -m json.tool          # 200 "ready" once warm
for m in mlx-community/Qwen2.5-3B-Instruct-4bit mlx-community/Qwen2.5-Coder-7B-Instruct-4bit; do
  curl -s $U/v1/chat/completions -H "$H" -H 'content-type: application/json' \
    -d "{\"model\":\"$m\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}" \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['model'])"; done
curl -s -H "$H" $U/v1/hearth/admin/models | python3 -m json.tool  # generations moved per model
curl -s $U/v1/chat/completions -H "$H" -H 'content-type: application/json' \
  -d '{"model":"no/such","messages":[{"role":"user","content":"hi"}]}'   # 404 model_not_found
# disconnect test: start a long stream, Ctrl-C it after a second, then time a short request
curl -sN $U/v1/chat/completions -H "$H" -H 'content-type: application/json' \
  -d '{"stream":true,"max_tokens":1500,"messages":[{"role":"user","content":"write a long essay"}]}'
time curl -s $U/v1/chat/completions -H "$H" -H 'content-type: application/json' \
  -d '{"max_tokens":5,"messages":[{"role":"user","content":"ok"}]}' >/dev/null   # ~1 s, not ~70 s
kill %1
```

**Done when:** each model id comes back as itself, admin/models shows the matching counters
moving, the bogus id is a 404, and the short request after Ctrl-C returns in about a second.

## 3. Use `/chat` in a browser, incl. agent mode (10 min) — *needs a browser* (B-014)

With the server from §2 running: open `http://127.0.0.1:8080/chat`, paste the token, chat with
two different models from the dropdown (only servable chat models are listed). Then set
`HEARTH_FILE_ROOTS=~/some/non-confidential/folder` (restart serve), turn **agent mode** on, and
ask "which file mentions X?" — it should use `search_files`. **Done when** both work; note
anything confusing in `docs/BUGS.md`.

## 4. Decide about the promoted `classify` adapter (5 min) — *your adapter* (B-010)

`classify-20260710T020135Z` is **promoted, and the router applies it to every classify request
under the default profile** (it was trained on Coder-7B, the default model; measured
2026-10-06 via the router's own adapter selection — under `routing.finance.yaml` classify runs
on the 3B and the adapter is not used). But it
was promoted before the current gate existed: no significance proof, on a golden set far below
30 examples. Under today's rules it could not be promoted. Options:

```sh
uv run --no-sync hearth adapters list
uv run --no-sync hearth adapters retire classify-20260710T020135Z   # stop serving it by default
```

or keep it knowingly, or re-qualify it later (needs §5 first; then a prereg committed before
its first ledger measurement — see `docs/RUNBOOK_training.md` §4–6).

**What it is:** a July 2026 demo adapter that learned an arbitrary ticket-routing convention
(incident text → queue codes `QX-1`..`QX-9`) from 45 synthetic examples; its "proof" is a typed
1.0-vs-0.2 score. **Measured 2026-10-06 (real weights, same prompt with and without it):** on
three ordinary classify prompts (spam / sentiment / spending category, labels given in the
prompt) it changed nothing — it does not hijack classifications that name their own labels.
That is n=3, not proof of no effect. **Recommendation:** retire it when convenient (it serves
no real task of yours and is unverified); not urgent.

## 5. Grow the golden sets to ≥ 30 distinct examples — *needs your judgement*

`data/extract_golden.jsonl` has 6 rows, `data/route_golden.jsonl` 5. The gate refuses anything
below `min_n = 30` distinct prompts (and near-duplicates are now caught). These must be **real,
representative examples with correct answers you vouch for** — not something an agent should
invent. Format and rules: `docs/RUNBOOK_training.md` §2. Never put real financial values in
them (CLAUDE.md §4). Commit them in the repo (the evals anchor).

## 6. Optional cleanups — *your files / your config*

- **Unregistered weights (B-019):** `Qwen/Qwen2.5-0.5B-Instruct` (953 MiB) sits in the HF cache,
  served by nothing. Delete it, or register it in `config/models.yaml` if you want it:
  `rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct`
- **Claude sandbox allowlist:** `~/.claude/apple/tool_allowlist.csv` has `acc,contains_match`
  — any command containing "acc" (accept, access, accuracy…) runs outside the sandbox. Narrow
  it to the exact command you meant (e.g. `acc,exact` or the full command line).
- **Starlette deprecation (B-020):** fixing it means adding a dependency (`httpx2`); your call
  whether to change the lockfile (`uv add --dev httpx2`, then
  `uv sync --extra mlx --extra mcp --extra dev --extra files`).

## 7. Decisions that are yours

- **Push / PR.** Nothing is pushed; ~150 commits sit on local `cmux/integration`. Earlier
  pushes were blocked by LuLu (ssh→github:22). When ready: allow it, then
  `git push origin cmux/integration`, and open a PR to `main` only after the graduation gate
  (`docs/cmux/TODO.md`).
- **cmux (B-002, P0).** Sealing a cmux workspace does not contain pane child processes. Until
  that is solved, do not run confidential work through cmux; plain `hearth` is fine.
- **Remote escalation (B-009).** The confidence score that decides escalation is a stub, but it
  only matters under the opt-in `config/routing.remote.yaml`. If you never use remote
  escalation, ignore it; if you will, it needs a real design.
- **Agent eval set (rest of B-013).** To measure the agent's step cap and tool use, it needs a
  small set of tasks you actually care about with known answers. Your examples, your call.
- **A real training run end to end**, when you have ≥ 30 golden examples:
  commit a prereg first, then
  `scripts/train_lora_real.sh --data data/extract.jsonl --task extract --promote --golden data/extract_golden.jsonl --prereg prereg/extract.yaml`.

---

When you have done §1–§3, tell Claude the outcome (or paste the output) and it will record the
evidence; anything that fails is a real bug to fix.

## 8. Watch the wire during a real session (optional, 15 min) — *needs sudo*

§1 proves HEARTH *cannot* reach the network (kernel sandbox). This shows what *did* happen while
you used it, attributed per process — useful as independent evidence, and the right tool for
cmux panes, where containment is still open (B-002). A quiet capture only covers what you did
during the window; it is not a substitute for §1.

```sh
# macOS pktap tags every packet with the sending process; loopback (HEARTH's own API) excluded.
sudo tcpdump -i pktap,all -k NP -w ~/hearth.pcap 'not (host 127.0.0.1 or host ::1)' &
# ... now use HEARTH normally: serve, /chat, an agent run over a folder, a file summary ...
sudo kill %1
tcpdump -r ~/hearth.pcap -k NP 2>/dev/null | grep -iE 'proc (python|hearth|uv)' || echo "no packets from HEARTH"
# open ~/hearth.pcap in Wireshark for detail (DNS queries count: a lookup leaks a hostname)
```

Live views: `nettop -m tcp` (per-process, live) and LuLu alerts (already installed). **Done when**
no packet in the capture comes from `python`/`hearth`/`uv`. For cmux, run the same capture while
a pane runs `curl https://example.com`: today it WILL show egress (that is B-002), and it is the
measurement any containment fix must turn silent.
