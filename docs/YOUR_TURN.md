# HEARTH — what only you can do (operator guide, 2026-10-06)

Everything an agent could do alone is done and committed on `cmux/integration` (local; nothing
pushed). What remains needs **your machine without the Claude sandbox**, **your judgement**, or
**your data**. Ordered by value. Each item says why it is yours, the exact commands, and what
"done" looks like. Run everything from `~/Claude/apps/HEARTH`; always `uv run --no-sync`.

---

## 1. Prove nothing leaves the machine — kernel level (10 min) — *needs your terminal*

The Claude session could not run `sandbox-exec` or bind ports today (its own process was
sandboxed). All offline evidence so far is HEARTH's own connect-counting plus one kernel-level
`doctor` run on 2026-10-02. Do the real thing.

Every block in this guide is paste-ready: no comments inside, expectations are in the prose.
Use a fresh Terminal window (not a `!` command in a Claude session — its shell is sandboxed
and drops aliases) and keep it open; §2 reuses the `sb` alias.

Set up the sandbox profile and alias:

```sh
cd ~/Claude/apps/HEARTH
cat > /tmp/hearth_noegress.sb <<'EOF'
(version 1)
(allow default)
(deny network-outbound)
(allow network-outbound (remote ip "localhost:*"))
(allow network-outbound (remote unix-socket))
EOF
alias sb='sandbox-exec -f /tmp/hearth_noegress.sb env -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY -u https_proxy -u ALL_PROXY -u all_proxy -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE'
```

**a. Validate the instrument.** Under the sandbox this MUST fail with "Operation not permitted":

```sh
sb python3 -c "import socket; socket.create_connection(('huggingface.co',443),timeout=5)"
```

Without the sandbox this should print `net ok` — otherwise your network is down and the steps
below prove nothing:

```sh
python3 -c "import socket; socket.create_connection(('huggingface.co',443),timeout=5); print('net ok')"
```

**b. The verdict, under the kernel sandbox.** Expect `SAFE` and `exit=0`:

```sh
sb uv run --no-sync hearth doctor --offline; echo "exit=$?"
```

**c. A real local answer, under the kernel sandbox** (loads real weights; first load is slow):

```sh
sb uv run --no-sync hearth run "Say hello in five words."
```

**Done when:** (a) fails then succeeds as stated, (b) says SAFE, (c) prints an answer. If (c)
errors with a network message, that is a real egress bug — file it in `docs/BUGS.md`.

**What this does not cover: DNS.** The profile allows Unix sockets, and macOS name lookups go
through mDNSResponder over one. In (a) the sandboxed `huggingface.co` lookup *succeeded*; only
the `connect` was refused. So §1 proves no outbound connection, not no outbound packet: a
lookup can still leak a hostname. §8's capture is what shows one: DNS replies there are
attributed to the process that asked, and on 2026-10-06 none went to HEARTH.

**Done 2026-10-06 (operator):** (a) `PermissionError: [Errno 1] Operation not permitted`, then
`net ok`; (b) `SAFE offline`, `exit=0`, profile `config/routing.yaml` with 0 remotes; (c)
`Hello there!` served by Coder-7B via mlx, under the sandbox.

## 2. Real-TCP serving + model selection + disconnect (15 min) — *needs your terminal*

Model selection was live-verified with real weights but only in-process (no TCP socket).

Every `curl` below uses `--noproxy '*'`: if your shell has `HTTP_PROXY` set, curl would send
127.0.0.1 through the proxy, which hides client disconnects and can make the disconnect test
falsely pass.

Start the server in the same window as §1 (needs the `sb` alias), then wait for
`Uvicorn running on http://127.0.0.1:8080`. The earlier `Serving on` line prints before the
port is bound (B-129), so it does not mean the server is up. If you see
`address already in use`, something else holds 8080: find it with
`lsof -nP -iTCP:8080 -sTCP:LISTEN` before going on, or every request below tests the wrong
process.

```sh
sb uv run --no-sync hearth serve &
```

The server window is now its log. Open a second window for everything below. Shell variables
do not carry over, so set them there:

```sh
TOKEN=$(cat ~/.hearth/token); H="Authorization: Bearer $TOKEN"; U=http://127.0.0.1:8080
```

Readiness — expect `"ready"` once the model is warm (repeat until it is):

```sh
curl -s --noproxy '*' $U/v1/hearth/admin/ready | python3 -m json.tool
```

Model selection — expect each model id printed back as itself, in order:

```sh
for m in mlx-community/Qwen2.5-3B-Instruct-4bit mlx-community/Qwen2.5-Coder-7B-Instruct-4bit; do
  curl -s --noproxy '*' $U/v1/chat/completions -H "$H" -H 'content-type: application/json' \
    -d "{\"model\":\"$m\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}" \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['model'])"; done
```

Per-model counters — expect the generation counts for both models above to have moved:

```sh
curl -s --noproxy '*' -H "$H" $U/v1/hearth/admin/models | python3 -m json.tool
```

Unknown model: expect `404`:

```sh
curl -s --noproxy '*' -o /dev/null -w '%{http_code}\n' $U/v1/chat/completions -H "$H" -H 'content-type: application/json' -d '{"model":"no/such","messages":[{"role":"user","content":"hi"}]}'
```

Disconnect test, part 1. curl hangs up by itself after 2 s, mid-stream. Expect
`curl exit=28` (timed out). Any other exit code means the stream ended on its own before the
hang-up, and part 2 then proves nothing. The prompt has to force a long answer: "write a long
essay" made the model ask clarifying questions and finish in ~250 tokens.

```sh
curl -sN --noproxy '*' --max-time 2 $U/v1/chat/completions -H "$H" -H 'content-type: application/json' -d '{"stream":true,"max_tokens":1500,"messages":[{"role":"user","content":"Write a 1500-word essay on the history of bridges. Start the essay immediately; do not ask any questions."}]}' >/dev/null; echo "curl exit=$?"
```

Disconnect test, part 2. Run it immediately. Expect a total of about 1 s. Tens of seconds
means the server kept generating for the client that left:

```sh
time curl -s --noproxy '*' $U/v1/chat/completions -H "$H" -H 'content-type: application/json' -d '{"max_tokens":5,"messages":[{"role":"user","content":"ok"}]}' >/dev/null
```

Leave the server running if you are going on to §3. Otherwise stop it **from the server
window** (`%1` only exists in the shell that started it):

```sh
kill %1
```

**Done when:** each model id comes back as itself, admin/models shows the matching counters
moving, the bogus id is a 404, part 1 exits 28 and part 2 returns in about a second.

**Done 2026-10-06 (operator, server under the §1 sandbox):** ready; 3B and Coder-7B each came
back as themselves; `generations: 1` on each; `404` with `model_not_found`; part 1
`curl exit=28`, part 2 `0.659 s` against a `0.66 s` baseline with no disconnect.

## 3. Use `/chat` in a browser, incl. agent mode (10 min) — *needs a browser* (B-014)

With the server from §2 running: open `http://127.0.0.1:8080/chat`, paste the token, chat with
two different models from the dropdown (only servable chat models are listed). Then set
`HEARTH_FILE_ROOTS=~/some/non-confidential/folder` (restart serve), turn **agent mode** on, and
ask "which file mentions X?" — it should use `search_files`. **Done when** both work; note
anything confusing in `docs/BUGS.md`.

The token field is in the header bar ("paste ~/.hearth/token"); then click **Load models**.
`pbcopy < ~/.hearth/token` puts the token on the clipboard. The page loads nothing
off-machine: all its requests are same-origin.

**Done 2026-10-06 (operator, sandboxed server, `HEARTH_FILE_ROOTS=docs/`):** 4 models
listed; chat replies from Coder-7B, Coder-14B and 14B (server log), each labelled with the
model picked; agent mode answered
"Which file mentions pktap?" correctly (`search_files`, then `docs/YOUR_TURN.md`; 2 steps,
32.3 s). Server log: toggle-off messages hit only `/v1/chat/completions`, agent runs
`/v1/hearth/agent`. Recorded under B-014.

## 4. Decide about the promoted `classify` adapter (5 min) — *your adapter* (B-010)

**Done 2026-10-06:** the operator retired it. `hearth_status.py --section learning` reports
`retired`. The text below is kept for the record.

`classify-20260710T020135Z` is **promoted, and the router applies it to every classify request
under the default profile** (it was trained on Coder-7B, the default model; measured
2026-10-06 via the router's own adapter selection — under `routing.finance.yaml` classify runs
on the 3B and the adapter is not used). But it
was promoted before the current gate existed: no significance proof, on a golden set far below
30 examples. Under today's rules it could not be promoted. Options — see what is there:

```sh
uv run --no-sync hearth adapters list
```

then retire it (stops serving it by default):

```sh
uv run --no-sync hearth adapters retire classify-20260710T020135Z
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

- **0.5B weights (B-019): kept, 2026-10-06.** `Qwen/Qwen2.5-0.5B-Instruct` (953 MiB, HF
  cache) is not served by HEARTH but is the reference model of
  `scripts/coreml_stateful_reference.py`. The status `UNREGISTERED` warn is expected.
- **Claude sandbox allowlist:** `~/.claude/apple/tool_allowlist.csv` is wider than one entry.
  As of 2026-10-06 it has 31 `contains_match` entries, and each one lets *any* command that
  contains the string run outside the sandbox. Among them: `python3`, `bash`, `sudo`, `curl`,
  `git`, `ssh`, `docker`, `node`, `x.com` and `acc` (which also matches accept, access,
  accuracy…). The valid match types are `contains_match` and `exact_match`; there is no prefix
  match, so a short CLI name cannot be narrowed to "commands starting with it". Either remove
  an entry, or replace it with `exact_match` lines for the full commands you actually run.
  Review in the ACC Dashboard's **Tools** tab, which shows what each entry opens. This is your
  config: an agent should not edit it.
- **Starlette deprecation (B-020): done 2026-10-06.** It added `httpx2` to the `dev` extra. Kept here as the procedure for any future dependency change. **Run this from your own
  shell, not a Claude session.** Claude sessions set `UV_DEFAULT_INDEX` to an internal mirror,
  and `uv add` writes it into `pyproject.toml` and every `uv.lock` source. This should print
  nothing:

  ```sh
  env | grep '^UV_'
  ```

  Add it without syncing. `dev` is an extra, so it takes `--optional dev`, not `--dev`:

  ```sh
  uv add --optional dev httpx2 --no-sync
  ```

  Check: a small diff, and `0` lockfile lines naming any registry other than pypi.org:

  ```sh
  git diff --stat pyproject.toml uv.lock; grep 'registry = ' uv.lock | grep -vc 'pypi.org/simple'
  ```

  Then sync every extra in one command (CLAUDE.md §1):

  ```sh
  uv sync --extra mlx --extra mcp --extra dev --extra files
  ```

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
you used it, attributed per process — independent evidence, and the right tool for cmux panes,
where containment is still open (B-002). A quiet capture only covers what you did during the
window; it is not a substitute for §1.

**What pktap attribution can and cannot see (measured 2026-10-06).** On this Mac, packets that
receive a reply carry the process: `eproc nc:86598` on the inbound SYN-ACK. Many *outbound*
packets carry no metadata, `()`. A send that is never answered (an unanswered SYN, one-way UDP)
can therefore go unattributed. DNS is better than expected: the query is unattributed, but the
reply is tagged with the requester (`proc mDNSResponder:441, eproc Safari:24038`). So the
capture proves "no *answered* traffic from HEARTH, including DNS lookups", not "no packet".
The old check (`grep 'proc (python…'`) only worked because `proc nc` is a substring of
`eproc nc`.

Run HEARTH **unsandboxed** here: §1 already showed the sandbox blocks; this measures what HEARTH
does on its own. Strip proxy variables. Otherwise HEARTH talks to a local proxy over loopback,
the filter below hides that, and the proxy egresses under its own name.

**Window 1.** Start the server, then note the PID in `Started server process [NNNN]`:

```sh
env -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY -u https_proxy -u ALL_PROXY -u all_proxy HEARTH_FILE_ROOTS=$HOME/Claude/apps/HEARTH/docs uv run --no-sync hearth serve &
```

**Window 3. Validate the instrument first.** This waits for 5 packets to or from example.com
and prints each one with its process. `-n` stops tcpdump making DNS lookups of its own:

```sh
sudo tcpdump -i pktap,all -k NP -n -c 5 'tcp port 443 and host example.com'
```

In window 2, while it waits. Window 3 should print lines naming `eproc nc:<pid>`:

```sh
nc -z -w 5 example.com 443
```

**Window 3. Start the real capture** in the foreground (stop it later with Ctrl-C):

```sh
sudo tcpdump -i pktap,all -n -w ~/hearth.pcap 'not (host 127.0.0.1 or host ::1)'
```

**Window 2. A control inside the window**, with a fresh hostname so its lookup is visible too:

```sh
nc -z -w 5 example.org 443
```

Now use HEARTH: in `/chat`, one plain message and one agent-mode question. Then a CLI run:

```sh
env -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY -u https_proxy -u ALL_PROXY -u all_proxy uv run --no-sync hearth run "Say hello in five words."
```

Stop the capture with Ctrl-C in window 3. Then, in window 2, list attributed traffic from the
processes that matter. Expect `eproc nc:` and no `python`/`hearth` line:

```sh
tcpdump -r ~/hearth.pcap -n -k NP 2>/dev/null | grep -oE 'e?proc [^,)]+' | grep -iE 'python|hearth|nc:' | sort | uniq -c | sort -rn
```

List the relevant hostnames looked up. Expect `example.org` and nothing else. A missing
`example.org` means lookups are not visible (e.g. encrypted DNS), so that half is unmeasured:

```sh
tcpdump -r ~/hearth.pcap -n 'port 53' 2>/dev/null | grep -oiE '(example\.org|huggingface[a-z.]*|hf\.co|pypi[a-z.]*|github[a-z.]*|openai[a-z.]*|anthropic[a-z.]*)' | sort | uniq -c
```

If a `python` line or a hostname shows up, identify it before concluding anything. Show where a
process connected (put its `Name:PID` from the first list in place of `Python:94611`):

```sh
tcpdump -r ~/hearth.pcap -n -k NP 2>/dev/null | grep 'Python:94611' | head -4
```

A DNS reply does not repeat the hostname. Find the query line, then grep its numeric ID (here
42211) to get the reply, which names the requester as `eproc`:

```sh
tcpdump -r ~/hearth.pcap -n -k NP 'port 53' 2>/dev/null | grep -E ' 42211[ +]'
```

**Done when** the control appears in both lists, and nothing attributed belongs to the server's
PID or the CLI run. For cmux, run the same capture while a pane runs `curl https://example.com`:
today it WILL show egress (that is B-002), and it is the measurement any containment fix must
turn silent.

**Done 2026-10-06 (operator).** Instrument: outbound packets `()`, the reply `eproc nc:86598`.
Capture: 20,965 packets. Attributed traffic among the relevant names: `eproc nc:93913` (the
control) and `Python:94611`, a different program's client connecting to a third-party API on
443 (host deliberately not recorded here). It is not HEARTH: the server was PID 94602 (port
8080's listener). DNS: `example.org`
(the control) and `glb-…github.com`, whose replies went to `eproc Safari:24038`. Nothing
attributed to the HEARTH server or the CLI run, DNS replies included.
