# HEARTH — User Guide

The start-to-finish guide for using HEARTH on this Mac. It is task-oriented: each section is a
thing you want to do, with commands you can paste. Every command and variable here was
checked against the code at `d628c4f` (`cmux/integration`, 2026-10-05, model selection merged).
Everything the B-0xx fixes merged since then changed (B-003, B-006, B-031, B-033 to B-037, B-046
to B-049) was re-run on the merged code (after `c97df9e`, with the CLI-polish batch) and is
described as it behaves now.

**New here? Do [§0, Learn HEARTH in 15 minutes](#0-learn-hearth-in-15-minutes) first.** The
same material is in the tool itself: `hearth --help`, `hearth COMMAND --help`, and the man page
(`man ./man/hearth.1`, §2.6).

**How to read the markers.**

- **Unmarked commands** were run during docs verification, and their output was checked
  against what this guide says. The run used an isolated `HEARTH_HOME` under `/tmp` and
  `HEARTH_BACKEND=echo` wherever a model would otherwise have answered. Outputs below are
  trimmed with `…`.
- **[in-process]** means the HTTP request was checked through FastAPI's `TestClient` against
  the real app, not over a live socket. The verification sandbox could not bind a port.
- **(not run in docs verification)** means the command was not run. Usually it needs real
  weights, the network, or changes to your Claude Code configuration.
- **B-0xx** numbers are entries in [docs/BUGS.md](BUGS.md): open limits are named where they
  bite, and a "(fixed)" note says the behaviour described is the fixed one.

Always run HEARTH from the repo with **`uv run --no-sync hearth …`**. Never use a bare
`uv run` or `uv sync` (see §2 and §11).

---

## Contents

0. [Learn HEARTH in 15 minutes](#0-learn-hearth-in-15-minutes)
1. [What HEARTH is, and the privacy model](#1-what-hearth-is-and-the-privacy-model)
2. [Install and verify](#2-install-and-verify)
3. [Models](#3-models)
4. [Chatting: the server, `/chat`, and the OpenAI-compatible API](#4-chatting-the-server-chat-and-the-openai-compatible-api)
5. [`hearth run` and `hearth agent`](#5-hearth-run-and-hearth-agent)
6. [RAG](#6-rag)
7. [MCP: using HEARTH from Claude Code](#7-mcp-using-hearth-from-claude-code)
8. [Routing profiles and escalation](#8-routing-profiles-and-escalation)
9. [Finance, and training / eval / promotion](#9-finance-and-training--eval--promotion)
10. [Environment variable reference](#10-environment-variable-reference)
11. [Troubleshooting](#11-troubleshooting)
12. [Command cheat-sheet](#12-command-cheat-sheet)

---

## 0. Learn HEARTH in 15 minutes

Eight steps, in order. Each shows what to type and what you should see. Run them from the repo
root (`~/Claude/apps/HEARTH`). Where a real model would answer, the output shown is from the
`echo` stub, which repeats your prompt back as `[echo] …`; on your machine (`HEARTH_BACKEND`
unset, so `auto` → MLX) a real model answers instead.

**Step 1: the venv is whole.**

```sh
uv run --no-sync python -c "import mlx_lm, mcp, openpyxl, pypdf; print('ok')"
```

```text
ok
```

Anything else (a `ModuleNotFoundError`) means a partial sync pruned the venv. Repair it with
the one-command sync in §2.1, then re-run this.

**Step 2: is it safe to use offline right now?**

```sh
uv run --no-sync hearth doctor --offline; echo "exit=$?"
```

```text
                     hearth doctor --offline
┃ check                                  ┃ status ┃ detail
│ routing_profile                        │ PASS   │ …/config/routing.yaml: 0 remotes, every class local/never — the router has nowhere to send
│ backend                                │ PASS   │ HEARTH_BACKEND=auto
│ bind_host                              │ PASS   │ HEARTH_HOST=127.0.0.1 (loopback)
│ allow_downloads                        │ PASS   │ HEARTH_ALLOW_DOWNLOADS is off
│ serving_resolution                     │ PASS   │ disk-only: a model on neither /Users/…/.hearth/models nor the hub cache fails to load …
│ model mlx-community/Qwen2.5-Coder-7B-… │ PASS   │ default model: on disk at /Users/…/.cache/huggingface/hub/models--mlx-community--…
│ load path: hearth train                │ PASS   │ disk-only: an absent model fails with no connect; …
│ load path: hearth models convert       │ PASS   │ …
│ load path: hearth models export-coreml │ PASS   │ …
│ hf_hub_offline                         │ PASS   │ HF_HUB_OFFLINE=unset — not required: every load path above is measured disk-only without it
│ models_pull                            │ PASS   │ `hearth models pull` is the one path that downloads, by design …
not measured: Measured in THIS command's environment. …
not measured: Covers HEARTH's router and loaders only — …
SAFE offline — no check found a path off this machine.
exit=0
```

`SAFE offline` and exit 0 is the answer you want. Every FAIL row names one cause; §2.4 explains
each row.

**Step 3: which models exist, and which one answers.**

```sh
uv run --no-sync hearth models list
```

```text
┃ id                                                     ┃ backend ┃ quant ┃ context ┃ ram_gb ┃ capabilities ┃
│ mlx-community/Qwen2.5-Coder-7B-Instruct-4bit (default) │ mlx     │ 4bit  │   32768 │    4.5 │ chat         │
│ mlx-community/Qwen2.5-Coder-14B-Instruct-4bit          │ mlx     │ 4bit  │   32768 │      9 │ chat         │
│ mlx-community/Qwen2.5-14B-Instruct-4bit                │ mlx     │ 4bit  │   32768 │      9 │ chat         │
│ mlx-community/Qwen2.5-3B-Instruct-4bit                 │ mlx     │ 4bit  │   32768 │      2 │ chat         │
│ echo                                                   │ echo    │ none  │    8192 │      0 │ chat         │
│ mlx-community/bge-small-en-v1.5-bf16                   │ mlx     │ none  │     512 │    0.2 │ embed        │
```

`(default)` is the model that serves when a request names none. Listing does not check that
the weights are on disk; step 2 did.

**Step 4: one completion.**

```sh
uv run --no-sync hearth run "Say hello in five words."
```

```text
[echo] Say hello in five words.
[served by mlx-community/Qwen2.5-Coder-7B-Instruct-4bit via echo; class=chat]
```

The first line (stdout) is the answer. The second (stderr) says which model and backend
produced it. On your machine it should say `via mlx`; **`via echo` means you are talking to the
stub** (§11). Pick a model with `--model <id>`; an id that is not servable exits 2 and lists
the ones that are.

**Step 5: the server, and the chat page.**

```sh
uv run --no-sync hearth serve
```

```text
HEARTH 0.0.1 — backend=mlx model=mlx-community/Qwen2.5-Coder-7B-Instruct-4bit
Serving on http://127.0.0.1:8080  (OpenAI-compatible /v1)
```
*[in-process]: banner observed with `backend=echo`; binding the port could not be tested.*

In a second terminal:

```sh
curl -s http://127.0.0.1:8080/v1/hearth/admin/ready
pbcopy < ~/.hearth/token
open http://127.0.0.1:8080/chat
```

`/ready` answers 503 `{"status":"loading","reason":"warmup in progress",…}` while the default
model loads, then 200 `{…,"status":"ready"}` (about 1 s for the 7B from page cache, per
docs/API.md; with real weights: not run in docs verification). If it says `failed`, the
`reason` tells you what to fix (§4.7). On echo it is 200 at once [in-process].

Paste the token into the page's **token** field once, pick a model, and chat. Leave the server
running for steps 6–8. (§4 covers the API, the token and `/ready` in full.)

**Step 6: the agent reads files you allow.**

```sh
mkdir -p /tmp/hearth-demo && printf 'a,b\n1,2\n' > /tmp/hearth-demo/demo.csv
HEARTH_FILE_ROOTS=/tmp/hearth-demo uv run --no-sync hearth agent "how many CSV files are there?"; echo "exit=$?"
```

```text
HEARTH agent — backend=mlx model=auto tools=list_files, read_file, search_files
rag_search not offered — no --collection named
finance tools not offered — no ledger at /Users/…/.hearth/finance/ledger.db
                       agent steps
┃ # ┃ step       ┃ arguments                 ┃ observation        ┃ tokens ┃ model/tool s ┃
…
answer
…
exit=0
```

*Checked on echo: the header and notes are exactly as above; the echo stub cannot follow the
agent's JSON contract, so on echo the run ends `NO ANSWER — the run stopped because
'invalid_output'` with exit 1. A real model answering is (not run in docs verification).*

The exit code is the verdict: 0 only when the model answered. Without `HEARTH_FILE_ROOTS` the
agent refuses to start (exit 2), because it could read nothing.

**Step 7: what it saved.**

```sh
curl -s -H "Authorization: Bearer $(cat ~/.hearth/token)" "http://127.0.0.1:8080/v1/hearth/admin/metrics?since=24h"
# {"requests":2,"estimated_frontier_tokens_saved":26,"escalations":0,"escalation_rate":0.0,"escalations_failed":0,
#  "failed":0,"failure_rate":0.0,"backend_mix":{"local":2},"class_mix":{"chat":2},"latency_ms":{"p50":0.01,"p95":0.01}}
uv run --no-sync hearth stats
```
*[in-process] for the metrics call (two echo requests).*

`hearth stats` in a fresh shell always prints zeros: metrics live in the memory of the process
that served the requests. The metrics endpoint asks the running server.

**Step 8: where to look things up.**

```sh
uv run --no-sync hearth --help          # commands grouped: Start here / Use / Train and evaluate / Extend
uv run --no-sync hearth serve --help    # every command: examples, env vars, exit codes
man ./man/hearth.1                      # the full reference, generated from the CLI
```

You now know the core loop. Next: §3 (choosing models), §8 (routing profiles), §7 (Claude
Code), §9 (finance and training).

---

## 1. What HEARTH is, and the privacy model

HEARTH is a local-first model gateway for Apple Silicon:

- It runs local LLMs (MLX, 4-bit Qwen2.5 3B–14B) behind an **OpenAI-compatible HTTP API** on
  `127.0.0.1:8080`, so any OpenAI client can use it by changing `base_url`.
- It adds a **local chat page** (`/chat`) and a **bounded, read-only, tool-using agent**
  (`hearth agent`, plus agent mode in `/chat`).
- It adds **local RAG** (`hearth rag`), and an **MCP server** (`hearth mcp`) so Claude Code can
  hand routine work to the local model.
- It ships a **finance pipeline** (parse, reconcile, store; all arithmetic in Python `Decimal`)
  and **LoRA fine-tuning** behind a statistical promotion gate.

**The privacy model, in plain words.**

- **The default is no-egress.** The shipped routing profile (`config/routing.yaml`) defines
  zero remote models. Every task class runs locally, so the router has nowhere to send a
  prompt. Escalation to Claude is opt-in (§8).
- **Model loads are disk-only.** Serving, chat, agent, MCP, RAG, `train`, `models convert`
  and `models export-coreml` load weights from disk. A missing model fails with an error
  instead of downloading. Only `hearth models pull` downloads, and you run it on purpose.
  `HEARTH_ALLOW_DOWNLOADS=1` overrides this; leave it off.
- **The gateway binds to loopback** (`127.0.0.1`) and requires a bearer token.
- **Metrics hold no prompt or response text**, only token counts and metadata, in memory.

**What this does NOT cover:**

- **The calling agent.** HEARTH protects the subtask that runs *on* HEARTH. If Claude Code (or
  any cloud agent) reads a confidential file into its own context and then asks HEARTH to
  summarize it, the file has already gone wherever that agent runs. Hand over a **path**
  instead (the `*_file` MCP tools, §7), so HEARTH opens the file and only the result crosses
  back.
- **Machine-level containment.** `hearth doctor --offline` checks HEARTH's router and loaders.
  It does not inspect firewalls, sockets or other processes. To prove that no packet leaves,
  run under a deny-egress sandbox (docs/PRIVACY.md, "Verifying no egress yourself").
- **Data at rest.** RAG collections (`~/.hearth/rag/*.db`), training runs and the finance
  ledger hold real copies of your text. Keep `~/.hearth` on FileVault.

Full model: [docs/PRIVACY.md](PRIVACY.md). Why there is no cloud backend at all:
[docs/TIERS.md](TIERS.md).

---

## 2. Install and verify

Everything runs from the repo checkout (`~/Claude/apps/HEARTH`).

### 2.1 Install: one command, every extra

```sh
uv sync --extra mlx --extra mcp --extra dev --extra files
```
*(not run in docs verification; it would modify the shared venv)*

**This must be one command.** `uv sync --extra X` syncs the environment to exactly that set
and **uninstalls every extra you did not name**. A bare `uv run` syncs to the default set and
removes 33 packages, including `mlx`, `mlx-lm`, `mcp` and `pytest`. Local inference then
stops: HEARTH falls back to the echo stub, with a startup WARNING and `/ready` 503 `stub`
(§11). To add an optional extra, append it to the full command, e.g.
`… --extra files --extra remote`.

| Extra | What it enables |
|---|---|
| `mlx` | real local inference (required for actual answers) |
| `mcp` | `hearth mcp` (Claude Code offload) |
| `files` | PDF / XLSX reading in the file tools |
| `dev` | the test suite |
| `remote` | the Anthropic SDK, needed only for the opt-in Claude escalation profile |
| `embeddings`, `vec`, `coreml` | MLX embedder (currently unusable, B-011), sqlite-vec store, Core ML export |

### 2.2 Verify the venv

```sh
uv run --no-sync python -c "import mlx_lm, mcp, openpyxl, pypdf; print('ok')"
```

It must print `ok`. Run this after any `uv` command that might have synced.

### 2.3 `hearth doctor`: the environment preflight

```sh
uv run --no-sync hearth doctor
```

The checks are `apple_silicon`, `memory`, `mlx_backend` ("mlx-lm importable") and `state_dir`
(`~/.hearth` writable). It ends with `Ready. (warnings are non-fatal)` and exits 0. A FAIL row
is fatal (exit 1); a WARN row is not. An unregistered `HEARTH_DEFAULT_MODEL` adds a
`default_model` **FAIL** row, because `serve`, `run`, `agent`, `mcp` and `rag query --answer`
refuse to start on it (§3.4). Measured with `HEARTH_DEFAULT_MODEL=bogus/x`:

```text
│ default_model │ FAIL   │ HEARTH_DEFAULT_MODEL='bogus/x' is not in the model registry (…/config/models.yaml). Registered
│               │        │ ids: mlx-community/Qwen2.5-Coder-7B-Instruct-4bit, … Fix the id, register the model, or unset
│               │        │ HEARTH_DEFAULT_MODEL to use the catalog default. `hearth serve`/`run`/`agent`/`mcp` refuse to start …
Fatal checks failed.
```

and exit 1.

### 2.4 `hearth doctor --offline`: is it safe to use offline right now?

```sh
uv run --no-sync hearth doctor --offline
echo $?        # 0 = SAFE, 1 = UNSAFE
```

Each row is a measurement, not a reading of a config value:

| Row | PASS means |
|---|---|
| `routing_profile` | the profile the router would load now exists and has 0 remotes, every class local/never. A `HEARTH_ROUTING_YAML` that names a missing file is a FAIL |
| `backend` | the `HEARTH_BACKEND` value is a built-in backend |
| `bind_host` | `HEARTH_HOST` is loopback |
| `allow_downloads` | `HEARTH_ALLOW_DOWNLOADS` is off |
| `serving_resolution` | a model that is not on disk fails to load, and no connection is attempted; the detail names the models directory the resolver searched (`$HEARTH_HOME/models`) |
| `model <id>` | each reachable model (the default, plus every class's `local_model` in the profile) is on disk; the row names the path. Not shown on the echo backend, which loads nothing |
| `load path: hearth train / models convert / models export-coreml` | each is handed a local path with the hub pinned offline |
| `default_model` | only present when `HEARTH_DEFAULT_MODEL` is set but unregistered. Here it is a non-fatal **WARN** (a command that refuses to start is not an egress path); plain `doctor` FAILs it (§2.3) |
| `hf_hub_offline`, `models_pull` | informational |

The last line is either `SAFE offline — no check found a path off this machine.` (exit 0), or
`UNSAFE offline: <failed rows>` (exit 1). Measured with three unsafe settings at once:

```sh
HEARTH_ROUTING_YAML=config/routing.remote.yaml HEARTH_HOST=0.0.0.0 HEARTH_ALLOW_DOWNLOADS=1 \
  uv run --no-sync hearth doctor --offline
# UNSAFE offline: routing_profile, bind_host, allow_downloads, serving_resolution,
#   load path: hearth train, load path: hearth models convert, load path: hearth models export-coreml
```

A non-fatal row is drawn **WARN**, never FAIL, so the table cannot contradict the verdict
(B-031, fixed). Measured with `HEARTH_DEFAULT_MODEL=bogus/x HEARTH_BACKEND=echo`:

```text
│ default_model                          │ WARN   │ HEARTH_DEFAULT_MODEL='bogus/x' is not in the model registry …
SAFE offline — no check found a path off this machine.
```

and exit 0.

**Two limits, both printed as `not measured:` lines.** First, doctor measures **this
command's environment**. A `hearth serve` you started earlier with different variables is not
what it checked: run doctor with the same variables you serve with. Second, it covers
HEARTH's router and loaders, not machine-level containment (§1).

### 2.5 Measured status (optional)

```sh
uv run --no-sync python scripts/hearth_status.py
uv run --no-sync python scripts/hearth_status.py --section environment
```

This prints which weights are on disk, which routing profiles are no-egress, the GPU
working-set ceiling, golden-set sizes, and how stale each doc is. It also flags `HEARTH_*`
variables that nothing reads (measured: `HEARTH_MODEL=x` → `hearth_env: HEARTH_MODEL …
SILENTLY IGNORED`). Prefer it over any written status, including this guide.

### 2.6 Built-in help and the man page

Every command documents itself, with examples, the `HEARTH_*` variables it reads and its exit
codes:

```sh
uv run --no-sync hearth --help              # commands grouped in learning order, first steps, pointers
uv run --no-sync hearth agent --help        # any command or subcommand
uv run --no-sync hearth models pull --help
```

The man page is generated from the same CLI (`scripts/gen_manpage.py`) and committed as
`man/hearth.1`. Read it in place:

```sh
man ./man/hearth.1
```

To make plain `man hearth` work, link it into a man directory on your `MANPATH`
(`man/hearth.1` is not in a `man1/` subdirectory, so pointing `MANPATH` at the repo's `man/`
does not work):

```sh
mkdir -p ~/.local/share/man/man1
ln -sf ~/Claude/apps/HEARTH/man/hearth.1 ~/.local/share/man/man1/hearth.1
export MANPATH="$HOME/.local/share/man:"     # trailing colon keeps the system pages; put it in ~/.zshrc
man hearth
```
*(the same recipe was run with a `/tmp` man directory; `man -w hearth` found the page)*

If you change a command or option, regenerate it (`uv run --no-sync python
scripts/gen_manpage.py`); `tests/test_manpage.py` fails until you do.

---

## 3. Models

### 3.1 List, pull, remove

```sh
uv run --no-sync hearth models list
uv run --no-sync hearth models pull mlx-community/Qwen2.5-3B-Instruct-4bit   # (not run in docs verification: downloads)
uv run --no-sync hearth models rm   mlx-community/Qwen2.5-3B-Instruct-4bit   # (not run against real weights)
```

- `list` prints the registry. The serving default is marked `(default)`, and this reflects
  `HEARTH_DEFAULT_MODEL` when it names a registered model.
- `pull` only accepts **registry ids**. An unknown id prints `Unknown model id: …` and exits 1;
  `pull echo` says `echo has no downloadable source (nothing to pull).` and exits 0. Pull
  respects `HF_ENDPOINT` (a mirror) and `HF_HUB_OFFLINE`.
- `rm` deletes only the copy under `~/.hearth/models`. It does **not** touch the Hugging Face
  hub cache. If the model is not there it prints `Not cached locally: <path>` and exits 1.

### 3.2 Where models live

`hearth models pull` writes to **`~/.hearth/models`** (i.e. `$HEARTH_HOME/models`), using the
hub layout `models--<org>--<name>/snapshots/<sha>`. Every load resolves a model id in this
order (`providers/mlx.py:resolve_local_model`):

1. an existing filesystem path, as given;
2. `~/.hearth/models`;
3. the Hugging Face hub cache: `HF_HUB_CACHE`, else `HF_HOME/hub`, else
   `~/.cache/huggingface/hub`.

Both cache lookups are local-only. A model found in none of them raises `ModelNotOnDiskError`
(§11). Do not "fix" a split between the two locations by exporting `HF_HUB_CACHE` globally:
the resolver already checks both. `hearth doctor --offline` and `scripts/hearth_status.py`
show which location each model resolved from.

### 3.3 The registry: `config/models.yaml`

The registry is data. Edit it to add, re-tier or retire a model, then `pull` it. Fields:
`id`, `backend` (`mlx` | `echo`), `quant`, `context`, `ram_gb`, `capabilities` (`chat` |
`embed`), `source` (HF repo id). `HEARTH_MODELS_YAML=/abs/path.yaml` points HEARTH at a
different registry file.

Shipped entries:

| id | ram_gb | role |
|---|---|---|
| `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` | 4.5 | **catalog default** (`default:` key) |
| `mlx-community/Qwen2.5-Coder-14B-Instruct-4bit` | 9.0 | higher-quality coder |
| `mlx-community/Qwen2.5-14B-Instruct-4bit` | 9.0 | general instruct; finance tier 2 |
| `mlx-community/Qwen2.5-3B-Instruct-4bit` | 2.0 | small/fast; finance tier 1; LoRA target |
| `echo` | 0 | deterministic stub, no weights |
| `mlx-community/bge-small-en-v1.5-bf16` | 0.2 | embedding model (unusable today, B-011) |

### 3.4 Choosing the model: the default, per request, per class

Three ways, from broadest to narrowest:

1. **The default** (`HEARTH_DEFAULT_MODEL`, else the registry's `default:` key) serves any
   request that names no model and has no per-class rung.
2. **Per class** (the routing profile's `local_model` for a task class, §8) applies when a
   request says `"model": "auto"` (the default for the API, `hearth run` and `hearth agent`).
3. **Per request**: `"model": "<id>"` in the API, `--model <id>` on `run` / `agent`, or the
   `/chat` model picker. **The model you name is the model that serves** (§4.7).

```sh
HEARTH_DEFAULT_MODEL=mlx-community/Qwen2.5-3B-Instruct-4bit uv run --no-sync hearth models list
# … mlx-community/Qwen2.5-3B-Instruct-4bit (default) …
```

**Trap 1: the name.** The variable is `HEARTH_DEFAULT_MODEL`, not `HEARTH_MODEL`. Settings
ignore unknown `HEARTH_*` names silently (§2.5 flags them).

**Trap 2: an unregistered id refuses to start** (B-047, fixed). Measured with
`HEARTH_DEFAULT_MODEL=bogus/x` (echo backend):

- `serve`, `run`, `agent`, `mcp` and `rag query --answer` print, and exit **2**:

  ```text
  Refusing to start: HEARTH_DEFAULT_MODEL='bogus/x' is not in the model registry (…/config/models.yaml).
  Registered ids: mlx-community/Qwen2.5-Coder-7B-Instruct-4bit, …, echo, mlx-community/bge-small-en-v1.5-bf16.
  Fix the id, register the model, or unset HEARTH_DEFAULT_MODEL to use the catalog default.
  ```

- an app built directly with `create_app()` raises `UnregisteredDefaultModelError` with the
  same message [in-process];
- `hearth doctor` shows a `default_model` **FAIL** row and exits 1 (§2.3); `doctor --offline`
  shows it as a non-fatal WARN and keeps its SAFE/UNSAFE verdict (§2.4);
- commands that only need an id keep working: `hearth models list` still marks the catalog
  default and adds `HEARTH_DEFAULT_MODEL='bogus/x' is not in the model registry:
  hearth serve/run/agent/mcp refuse to start on it; anything asking only for an id gets the
  catalog default '…Coder-7B…'`; plain `rag query` (no `--answer`) exits 0.

Unset, the variable means the catalog default.

**Always confirm with `hearth models list`.** To use a new model, register it first (§3.3).

### 3.5 Memory: the real ceiling

This M3 Pro advertises 36 GB, but the GPU working-set ceiling (the driver's limit on resident
GPU memory) is **30.15 GB**. That is what `scripts/hearth_status.py` reports as
`gpu_working_set`. Size models against that figure.

On the mlx backend the server holds **one provider per model id** (`hearth.serving.ModelPool`),
loads a model the first time a request names it, and keeps the set of resident models under
`HEARTH_RAM_CEILING_GB` (default 24.0, using the registry's `ram_gb` estimates): loading one
more model evicts the least-recently-used one. `GET /v1/hearth/admin/models` (§4.7) shows what
is resident. Background: [docs/MODELS_local.md](MODELS_local.md).

### 3.6 Convert and Core ML export (advanced)

```sh
uv run --no-sync hearth models convert --source <id-or-path> --out ~/.hearth/models/<name>-q4 --q-bits 4
uv run --no-sync hearth models export-coreml --source <id-or-path> --out ~/.hearth/coreml/<name>
```
*(not run in docs verification: need real weights; `export-coreml` also needs `--extra coreml`)*

Both are disk-only: a source that is not on disk fails with `ModelNotOnDiskError` and no
download. Register a converted model in `config/models.yaml` to serve it.

---

## 4. Chatting: the server, `/chat`, and the OpenAI-compatible API

### 4.1 Start the server

```sh
uv run --no-sync hearth serve                        # http://127.0.0.1:8080
uv run --no-sync hearth serve --port 8081            # another port
```
*[in-process]: the startup banner was observed. Binding the port could not be tested in the sandbox.*

It prints `HEARTH <version> — backend=<mlx|echo> model=<default id>`, then `Serving on
http://127.0.0.1:8080 (OpenAI-compatible /v1)`. **Check `backend=`**: if it says `echo`, you
are talking to the stub, not a model (§11). Settings are read once at startup, so restart
after changing any `HEARTH_*` variable.

**Model loading.** With `HEARTH_WARMUP` on (the default), the default model starts loading in
the background as soon as the server starts; the server accepts requests immediately.
`/v1/hearth/admin/ready` turns 200 once the weights are in memory (§4.7). Other models load
the first time a request names them; that request pays the load time (about 15 s cold for a
14B model).

### 4.2 The token

- On first run HEARTH creates **`~/.hearth/token`** (mode `0600`; `$HEARTH_HOME/token` under a
  custom home).
- Every `/v1/*` route requires `Authorization: Bearer <token>`. The exceptions are
  `/v1/hearth/admin/health`, `/v1/hearth/admin/ready` and the `/chat` page itself.
- A missing or wrong token returns **401**, with body
  `{"detail":{"error":{"message":"missing or invalid bearer token","type":"invalid_request_error","code":"hearth.auth.unauthorized"}}}`
  [in-process].
- `HEARTH_REQUIRE_AUTH=false` turns auth off. Use it only for throwaway local testing.

```sh
export HEARTH_TOKEN="$(cat ~/.hearth/token)"
```

### 4.3 The `/chat` page

Open **http://127.0.0.1:8080/chat** (served without a token [in-process]). In the header:

- **token**: paste the contents of `~/.hearth/token` once (`pbcopy < ~/.hearth/token`). It is
  kept in the page's `localStorage` on the loopback origin. The server never embeds it in the
  page.
- **Load models** / **model**: the dropdown is filled from `/v1/models`, which lists exactly
  the chat models this backend can serve. **The model you pick is the model that answers**;
  the reply's provenance line names it.
- **temp**, **max tokens** (default 512). A reply cut off at max tokens is flagged
  **Truncated**.
- **agent mode** (off by default) switches from plain chat to the local agent loop. That loop
  has read-only tools (list and read files under `HEARTH_FILE_ROOTS`, RAG search), and each
  step shows as its own turn. **steps** caps the iterations (max 12). A run that hits a bound
  is shown as **NOT AN ANSWER**. With agent mode off, plain chat has no tools at all: a
  question about your files gets an invented answer. Turn agent mode on for those.
- Each reply shows its provenance: served on-device or remotely, model, backend, adapter.

The page is self-contained (no CDN, no fonts) and talks only to `/v1/models`,
`/v1/chat/completions` and, in agent mode, `/v1/hearth/agent`. *(The page itself in a browser
was not driven in docs verification, B-014.)*

> Agent mode over HTTP differs from the CLI agent (B-012). It offers `rag_search` without a
> pinned collection, so the model must guess the collection name. It never offers the finance
> tools.

### 4.4 The OpenAI-compatible API

| Endpoint | Auth | What |
|---|---|---|
| `POST /v1/chat/completions` | yes | chat, streaming or not |
| `GET /v1/models` | yes | the chat models this backend can serve (§4.7) |
| `POST /v1/embeddings` | yes | embeddings from the configured embedder (`hash` by default); ignores `model` |
| `POST /v1/hearth/route` | yes | dry run: which class, backend and model the router *would* use |
| `POST /v1/hearth/rag/ingest`, `/rag/query` | yes | RAG over HTTP (§6) |
| `POST /v1/hearth/agent` | yes | the agent loop, streamed as SSE (docs/AGENT.md §9) |
| `GET /v1/hearth/admin/metrics?since=24h` | yes | rollups: requests, tokens saved, escalations, latency |
| `GET /v1/hearth/admin/models` | yes | resident models: loaded path, generation count, RAM budget (§4.7) |
| `GET /v1/hearth/admin/health` | no | liveness: version, backend, default model |
| `GET /v1/hearth/admin/ready` | no | readiness, 200 or 503 with a reason (§4.7) |
| `GET /chat` | no | the chat page |

This is every `/v1` route the gateway serves. `docs/API.md` lists the same set, and
`tests/test_api_doc_routes.py` fails if the two drift (B-037, fixed). Endpoints that earlier
drafts promised (`/v1/hearth/classify`, `/summarize`, `/train/*`, admin model load/unload and
adapter promote/retire) are listed there under **Not implemented**: use chat completions with
a `hearth.intent`, the MCP tools, or the CLI instead.

**curl** [in-process]:

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $HEARTH_TOKEN" -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":64}'
```

Response (echo backend shown; on mlx `backend` is `mlx` and the content is a real answer):

```json
{"id":"chatcmpl-…","object":"chat.completion","created":…,
 "model":"mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
 "choices":[{"index":0,"message":{"role":"assistant","content":"[echo] Say hello in five words."},"finish_reason":"stop"}],
 "usage":{"prompt_tokens":6,"completion_tokens":7,"total_tokens":13},
 "hearth":{"served_by":"local","backend":"echo","model":"mlx-community/Qwen2.5-Coder-7B-Instruct-4bit","adapter":null,
           "escalated":false,"estimated_frontier_tokens_saved":13}}
```

Request defaults: `model: "auto"`, `max_tokens: 512`, `temperature: 0.7`, `stream: false`.
`response_format: {"type": "json_object"}` is supported. HEARTH instructs the model and then
validates the output. If the model's reply is not valid JSON, you get **422**
`hearth.response_format.invalid_json` with the raw content, never a half-parsed object
[in-process].

**Python, `openai` SDK** *(not run in docs verification: `openai` is not installed in the
HEARTH venv; install it in your own project)*:

```python
from pathlib import Path
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1",
                api_key=Path("~/.hearth/token").expanduser().read_text().strip())
resp = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Summarize: HEARTH keeps inference on this Mac."}],
    extra_body={"hearth": {"intent": "summarize", "allow_escalation": False}},
)
print(resp.choices[0].message.content)
```

**Python, HEARTH's own client** (`hearth.client.HearthClient`, needs only `httpx`)
[in-process: its HTTP client was swapped for a `TestClient`]:

```python
from pathlib import Path
from hearth.client import HearthClient

hc = HearthClient("http://127.0.0.1:8080", token=Path("~/.hearth/token").expanduser().read_text().strip())
print(hc.summarize("HEARTH keeps inference on this Mac.", max_words=20))
print("".join(hc.chat([{"role": "user", "content": "hello"}], stream=True)))
```

### 4.5 Streaming

With `"stream": true`, the response is SSE (`text/event-stream`). The stream sends `data:`
chunks in OpenAI `chat.completion.chunk` shape: a role delta, then content deltas. The
**final chunk before `data: [DONE]`** carries `finish_reason` and the `hearth` telemetry
block, whose `model` names the model that generated [in-process]. If generation fails
mid-stream, the stream ends with an `error` event (`hearth.provider.unavailable`) and then
`[DONE]`. Dropping the connection stops generation.

### 4.6 The `hearth` extension fields

Request (optional; plain OpenAI clients never need it):

```json
"hearth": {"intent": "summarize", "allow_escalation": false, "adapter": "<adapter id>"}
```

- `intent` is a routing hint that skips classification. It must be one of `summarize`,
  `extract`, `classify`, `rank`, `draft`, `code`, `reason`, `chat`. Without it, keyword rules
  over the last user message pick the class.
- `allow_escalation` (default `true`): `false` hard-pins this call to local. Under the default
  profile it is local anyway.
- `adapter` names a LoRA adapter (`hearth adapters list`), served even if it is still a
  candidate. Without `adapter`, a **promoted** adapter for the request's class is applied
  automatically when its base model is the one serving. On this machine
  `classify-20260710T020135Z` is promoted on Coder-7B, so `classify` requests use it (B-010).

An **unknown or retired** adapter id is refused before anything runs (B-034, fixed).
Measured [in-process], echo backend, `"hearth": {"adapter": "no-such-adapter"}`:

```json
HTTP 404
{"error":{"message":"adapter 'no-such-adapter' cannot be served: unknown adapter: 'no-such-adapter'",
          "type":"invalid_request_error","param":"hearth.adapter","code":"adapter_not_found"}}
```

The response's `hearth.adapter` names the adapter whose weights actually generated, or `null`
for base weights (no adapter, a base-weights retry after an adapter failed to load, a remote,
or a backend that ignores adapters such as echo: measured `"adapter": null` on echo).

Response `hearth` block: `served_by` (`local` | `remote`), `backend`, `model`, `adapter`,
`escalated`, `estimated_frontier_tokens_saved`.

### 4.7 Choosing a model per request, and readiness

**The model a request names is the model that serves.** On the mlx backend:

- `"model": "<registered chat id>"` is served by **those weights**, loaded on first use and
  evicted least-recently-used under `HEARTH_RAM_CEILING_GB` (§3.5). The `hearth.model`
  telemetry is filled in by the provider that generated, on streaming responses too.
- An **id the server cannot serve** returns **404** `model_not_found` before anything is
  generated, for a typo, `echo` on the mlx backend, or the embedding model. Measured
  [in-process], mlx backend: `echo` → 404 `model_not_found`;
  `mlx-community/bge-small-en-v1.5-bf16` → 404 `model_not_found`; `bogus/model` → 404 with
  the message `The model 'bogus/model' does not exist in the HEARTH model registry
  (config/models.yaml). Servable models: […]`. The CLI applies the same rule:
  `hearth run --model bogus/x` prints `Unknown model: …` and exits 2.
- A **registered chat model whose weights are not on disk** is not a 404: the load fails and
  the request gets **503** `hearth.provider.unavailable`, whose message names the
  `hearth models pull <id>` to run [in-process].
- `"model": "auto"` (or omitted) uses the routing profile's **per-class ladder** (each class's
  `local_model`, e.g. under the finance profile classify → 3B, summarize → 14B), then the
  default. Measured with `hearth run --intent classify` under `routing.finance.yaml`:
  `served by mlx-community/Qwen2.5-3B-Instruct-4bit`; `--intent summarize`:
  `…Qwen2.5-14B-Instruct-4bit`.
- `GET /v1/models` lists exactly the servable chat models: on mlx, the four Qwen chat models
  (not `echo`, not the embedder) [in-process]. On the echo backend it also lists `echo`.

**`GET /v1/hearth/admin/models`** (token required) shows what is resident right now, read off
the provider instances: `backend`, `default`, `ram_ceiling_gb`, `resident_ram_gb`, and per
resident model its `loaded_path` (the directory the weights were read from) and `generations`
(how many generations that instance ran). Measured on a fresh server: `"resident": []`
[in-process]; with real weights loaded (not run in docs verification) each entry names its
snapshot path.

**`GET /v1/hearth/admin/ready`** (no token) answers **200 `ready`** when the default model can
serve a request now: it has loaded with weights at least once (warmup or any request), its
last load did not fail, and its weights still resolve on disk. Residency is reported
separately as `loaded`, so a default that was evicted to make room for another model stays
200 and reloads on demand (B-035, B-048, fixed). Otherwise **503**, with a `status` and a
`reason`. On the mlx backend [in-process]:

| Situation | Response |
|---|---|
| warmup still loading | 503 `loading`, `warmup in progress` *(from the code; too brief to catch with no weights on disk)* |
| weights not on disk (warmup on) | 503 `failed`, `weights for '…Coder-7B…' do not resolve on disk: ModelNotOnDiskError: … Fetch it deliberately with hearth models pull mlx-community/Qwen2.5-Coder-7B-Instruct-4bit.` (measured, with the warmup's `warmup of … failed … NOT READY; serving in degraded mode` in the log); a load that fails for another reason reads `warmup of '<id>' failed: …` |
| `HEARTH_WARMUP=false`, weights on disk | 200 `ready`, `loaded: false`, detail `warmup disabled (HEARTH_WARMUP=false); '…Coder-7B…' is on disk and loads on the first request` (measured with planted weights; nothing was loaded) |
| `HEARTH_WARMUP=false`, weights not on disk (or deleted) | 503 `failed`, `weights for '…Coder-7B…' do not resolve on disk: ModelNotOnDiskError: … Fetch it deliberately with hearth models pull mlx-community/Qwen2.5-Coder-7B-Instruct-4bit.` (measured) |
| default evicted (LRU) after loading | 200 `ready`, `loaded: false`, detail `… loaded before and is not resident now (evicted to make room); it reloads on demand` *(from `tests/test_model_selection.py`, which drives a fake mlx_lm; not run with real weights)* |
| echo chosen explicitly (`HEARTH_BACKEND=echo`) | 200 `ready` (nothing to load) |
| `auto` fell back to echo (`mlx_lm` missing) | 503 `stub`, reason `HEARTH_BACKEND=auto found no importable mlx_lm, so the echo STUB is serving …` (§11) |

Every body also carries `backend`, `model` (the default id), `loaded` and `resident`. An
unregistered `HEARTH_DEFAULT_MODEL` no longer gets this far: the server refuses to start
(§3.4).

---

## 5. `hearth run` and `hearth agent`

### 5.1 `hearth run`: one-shot completion

```sh
uv run --no-sync hearth run "summarize: <text>"
uv run --no-sync hearth run --file notes.txt --max-tokens 256
echo "prompt from stdin" | uv run --no-sync hearth run
uv run --no-sync hearth run "label this ticket: …" --intent classify
uv run --no-sync hearth run --model mlx-community/Qwen2.5-3B-Instruct-4bit "hello"
```

- It prints the model's text on stdout, and `[served by <model> via <backend>; class=<c>]` on
  stderr. With `--intent`, a dim `intent=<x>` line comes first.
- An empty prompt prints `No prompt provided.` and exits 1. An unservable `--model` prints
  `Unknown model: …` and exits 2; so do an unregistered `HEARTH_DEFAULT_MODEL` (§3.4) and a
  missing routing profile (§8).
- Defaults: `--max-tokens 512`, `--model auto` (the per-class ladder, §4.7).
- **Always local.** `hearth run` never escalates, whatever the routing profile.
- It has **no tools**: it cannot read files you mention. Use `--file` to send a file's
  contents as the prompt, or use `hearth agent`.

### 5.2 `hearth agent`: bounded, read-only, tool-using

```sh
HEARTH_FILE_ROOTS=~/some/dir uv run --no-sync hearth agent "how many CSV files are there?"
```
*(the mechanics, refusals, exit codes and `--json` were run on echo. Real answers need the mlx backend: not run in docs verification)*

The agent plans, calls **one** tool, reads the result, and repeats until it answers or hits a
bound. Every generation is local, and it **never** escalates.

**Tools.** These are assembled from what is present, and the header line lists them:

| Tool | Offered when | Governed by |
|---|---|---|
| `list_files(root, pattern)`, `read_file(path)`, `search_files(text, root, pattern)` | always (refuse everything without roots) | `HEARTH_FILE_ROOTS`, `HEARTH_FILE_MAX_BYTES`, the format table |
| `rag_search(query, …)` | `--collection NAME` given (and it is non-empty) | the RAG index |
| `finance_total / finance_explain / finance_rows` | a ledger exists at `~/.hearth/finance/ledger.db` (disable with `--no-finance`) | Decimal arithmetic, integrity checks |

There are no write, shell or network tools, and no flag to add them.

**`HEARTH_FILE_ROOTS`** is a colon-separated list of directories. It is **deny-by-default**:
unset means every read is refused, with no implicit root (not the current directory, not
`$HOME`). Paths are fully resolved (`..`, symlinks) before the containment check. Readable
formats: `.txt .text .md .markdown .rst .log` and extension-less files as text, plus `.csv`,
`.json`, `.xlsx` and `.pdf` (XLSX/PDF need `--extra files`).

**Options:**

| Option | Default | |
|---|---|---|
| `--max-iterations` | 8 | model turns |
| `--max-seconds` | 180 | wall clock |
| `--max-tokens` | 24000 | prompt + completion, all steps |
| `--model` | `auto` | registry id for every step; `auto` = per-class ladder |
| `--collection` | none | offer `rag_search` pinned to this collection |
| `--finance/--no-finance` | on | offer the ledger tools when a ledger exists |
| `--steps/--no-steps` | on | the step table (tool, arguments, truncated observation, tokens, timings) |
| `--full` | off | the raw per-step transcript instead of the table |
| `--json` | off | the whole run as one JSON document (`completed`, `stopped_reason`, `answer` = `null` unless answered, every step) |

**Exit codes are the stop reason:**

| Exit | Meaning |
|---|---|
| `0` | the model answered, and only then |
| `1` | stopped at a bound or a failure: `max_iterations`, `timeout`, `token_budget`, `invalid_output` (3 unusable replies in a row), `provider_error`, `egress_refused`. It prints `NO ANSWER — the run stopped because '<reason>'`, and the steps are a partial trace, not a result |
| `2` | never started: an impossible bound (`max_iterations must be at least 1`), an empty `--collection`, an unservable `--model`, or nothing reachable (no roots, no collection, no ledger: `Refusing to start`) |

A typo'd root is caught too. Measured: `HEARTH_FILE_ROOTS=/tmp/no-such-dir` prints
`HEARTH_FILE_ROOTS is set to '/tmp/no-such-dir', but none of those are existing directories`
and exits 2.

**What it can't do.** It handles short, checkable step sequences, not long-horizon work. The
small local models re-try failed ideas, answer from partial results, and invent plausible
filenames. There is no content-search tool (B-013), so finding a fact means reading files one
at a time within 8 steps. Read the step table before trusting the answer. Background:
[docs/AGENT.md](AGENT.md).

---

## 6. RAG

```sh
uv run --no-sync hearth rag ingest ~/notes --collection notes          # file or directory
uv run --no-sync hearth rag query "how often to descale the kettle" --collection notes --k 2
uv run --no-sync hearth rag query "descale" --collection notes --answer  # local model answers from the chunks
```

- `ingest` prints `Ingesting <path> → collection <name> (embedder=hash) …`, then
  `Done. N file(s), M chunk(s) in collection <name>.` Options: `--collection` (default
  `default`), `--size 800`, `--overlap 100` (characters).
- `query` prints a table of score, source and text. `--k` defaults to 6. A missing or empty
  collection prints `No chunks in collection '<name>'.` and exits 0, without embedding the
  query or running `--answer` (measured with `HEARTH_EMBEDDER=mlx`, which cannot load).
- `--answer` sends the retrieved chunks plus your question to the local model, never escalated.
  *(Checked on echo only; a real answer needs the mlx backend.)*
- Storage: `~/.hearth/rag/<collection>.db` holds the **raw chunk text**. Delete that file to
  purge the collection.
- Over HTTP: `POST /v1/hearth/rag/ingest` `{"collection","paths":[…],"chunk":{"size","overlap"}}`
  and `POST /v1/hearth/rag/query` `{"collection","query","k","answer"}`.
- From the agent: `hearth agent --collection notes "…"`. From Claude Code: `hearth_rag_query`
  (§7).

**Embedders.** The default `HEARTH_EMBEDDER=hash` is offline and dependency-free, but it is a
**hashing embedder: lexical overlap, not semantic recall**. Queries need to share words with
the text. `HEARTH_EMBEDDER=mlx` **does not work today (B-011)**. Its default model id
(`…-mlx`) is not on disk, and the registered `…-bf16` weights are a BERT model that `mlx_lm`
cannot load. Stay on `hash`.

With `HEARTH_EMBEDDER=mlx`, `rag ingest` (and `rag query` on a non-empty collection) ends
with one message and exit 1, no traceback (B-036, fixed). Measured:

```text
Ingesting /tmp/a5db_doc.txt → collection k (embedder=mlx) …
Embedder unavailable: could not load embedding model 'mlx-community/bge-small-en-v1.5-mlx': model
'mlx-community/bge-small-en-v1.5-mlx' is not on disk (looked in /tmp/…/models and the huggingface hub cache) and
HEARTH does not download on load. It is not in the model registry, so `hearth models pull` refuses it: … Note:
mlx-lm has no BERT architecture, so the registered bge embedder cannot load even when it is on disk
(docs/BUGS.md B-011); HEARTH_EMBEDDER=hash works offline.
```

`HEARTH_VECTOR_STORE` is `sqlite` (default) or `sqlite-vec` (needs `--extra vec`).

---

## 7. MCP: using HEARTH from Claude Code

`hearth mcp` starts a **stdio** MCP server named `hearth`. Claude Code can then hand routine
subtasks to the local model, with zero frontier tokens spent. Every tool runs with escalation
**disabled**, under any routing profile, in-process: no HTTP and no token.

**Tools** (verified by listing them over stdio with the MCP client SDK, echo backend):

| Tool | Arguments | Notes |
|---|---|---|
| `hearth_summarize` | `text`, `max_words?` | |
| `hearth_classify` | `text`, `labels` | |
| `hearth_extract` | `text`, `fields` | returns `{field: value}` |
| `hearth_draft` | `instruction`, `context?` | |
| `hearth_summarize_file` | `path`, `max_words?` | HEARTH opens the file; the content never enters Claude's context |
| `hearth_classify_file` | `path`, `labels` | same |
| `hearth_extract_file` | `path`, `fields` | same |
| `hearth_rag_query` | `collection`, `query`, `k` (default 6), `answer` (default false) | returns chunks, plus an answer if asked |

**The `*_file` tools need `HEARTH_FILE_ROOTS`** in the MCP server's environment. Without it,
every read is refused. A path outside the roots fails with `path is outside every allowed root
in HEARTH_FILE_ROOTS: '<path>'`, and refusal messages never quote file content. **Prefer the
`*_file` tools for anything confidential.** Pasting text into `hearth_summarize` means Claude
has already read it (§1).

**Register with Claude Code** *(not run in docs verification: it changes your Claude Code
config)*. Use the venv's absolute `hearth` path. An MCP subprocess has no activated venv and
no guaranteed working directory, and this path never invokes `uv` (so it can never sync):

```sh
claude mcp add hearth \
  -e HEARTH_FILE_ROOTS="$HOME/some/dir" \
  -- ~/Claude/apps/HEARTH/.venv/bin/hearth mcp
```

Or in `.mcp.json`:

```json
{
  "mcpServers": {
    "hearth": {
      "command": "/Users/<you>/Claude/apps/HEARTH/.venv/bin/hearth",
      "args": ["mcp"],
      "env": { "HEARTH_FILE_ROOTS": "/Users/<you>/some/dir" }
    }
  }
}
```

(`uv run --no-sync --project ~/Claude/apps/HEARTH hearth mcp` also works from any directory;
see [examples/claude_code_mcp.md](../examples/claude_code_mcp.md).) If you add
`HEARTH_ROUTING_YAML` to `env`, a relative path resolves against the repo root (§8). Without
the `mcp` extra, `hearth mcp` prints `The MCP server requires the 'mcp' extra.` and exits 1.
Claude Code then just shows no hearth tools. Re-run the one-command sync (§2.1).

---

## 8. Routing profiles and escalation

A routing profile is a YAML file that decides, per task class, where a request runs and which
local model serves it. Select one with **`HEARTH_ROUTING_YAML`**:

- unset → the repo's `config/routing.yaml`;
- `~` is expanded;
- a **relative path resolves against the repo root**, not the current directory, so
  `HEARTH_ROUTING_YAML=config/routing.finance.yaml` means the same file from anywhere;
- a named file that **does not exist is an error**: `serve`, `run`, `agent`, `mcp` and
  `rag query` refuse to start with one line and exit **2**, no traceback (B-033, fixed), and
  `serve` prints no `Serving on …` banner. Measured:

  ```text
  Routing profile not found: HEARTH_ROUTING_YAML='config/no-such.yaml' selects <repo>/config/no-such.yaml,
  which does not exist (relative paths resolve against the repo root, <repo>). Fix or unset HEARTH_ROUTING_YAML.
  ```

  `doctor --offline` fails its `routing_profile` row. (A file that exists but does not parse
  still falls back to all-local defaults with a warning.)

| Profile | Egress | What it is for |
|---|---|---|
| `config/routing.yaml` (default when unset) | **none**: 0 remotes, all classes local/never | everyday use |
| `config/routing.private.yaml` | **none** | confidential work; used by `scripts/hearth_private.sh` |
| `config/routing.finance.yaml` | **none** | a two-tier **local ladder**: classify/extract/rank → Qwen2.5-3B; summarize/draft/reason/chat/code → Qwen2.5-14B |
| `config/routing.remote.yaml` | **yes**: Claude (`claude-opus-4-8`, Anthropic SDK), 200k tokens/day | opt-in escalation, only for work that may leave the machine |
| `config/routing.escalation-demo.yaml` | to `127.0.0.1:8099` only | demo: escalation goes to a local stub (`scripts/frontier_stub.py`) |

```sh
HEARTH_ROUTING_YAML=config/routing.finance.yaml uv run --no-sync hearth doctor --offline
HEARTH_ROUTING_YAML=config/routing.finance.yaml uv run --no-sync hearth serve              # [in-process]
```

`doctor --offline` under the finance profile checks that the 3B **and** the 14B are on disk.
Measured with neither present: `model mlx-community/Qwen2.5-3B-Instruct-4bit … class classify,
class extract, class rank: NOT on disk — fetch it with hearth models pull …` and `UNSAFE
offline: model …14B…, model …3B…`. Pull both, then it reports SAFE.

**Sealed private mode** *(not run in docs verification)*:

```sh
scripts/hearth_private.sh --check                                         # verify only; exit 0 if sealed
scripts/hearth_private.sh --profile config/routing.finance.yaml --check
scripts/hearth_private.sh                                                 # verify, then serve
```

The script forces loopback, `HEARTH_BACKEND=mlx` and offline HF. It refuses to start if the
selected profile has any remote or escapable class.

**What escalation does (remote profile only).** Under `routing.remote.yaml`, `reason` always
goes to Claude. `draft`, `code` and `chat` escalate "on low confidence", but that confidence
is a **prompt-length stub** (B-009). Short messages (under ~60–90 characters) escalate, while
long pasted documents stay local: the inverse of what you would want for privacy. It needs
`--extra remote` and `ANTHROPIC_API_KEY` (or an `ant auth login` profile). If the remote call
fails, the request is **served locally** and counted as `escalations_failed`, not reported as
an error. A failed remote call may still have transmitted the prompt. Per request,
`"hearth": {"allow_escalation": false}` pins the call local. `hearth run`, `hearth agent` and
all MCP tools never escalate.

A request that fails outright (the local provider fails, a failed escalation is followed by
a failed local fallback, or a remote stream dies mid-answer) **is recorded** (B-003, fixed):
it counts in `requests` and in `failed` / `failure_rate`, and, after a failed escalation, in
`escalations_failed`. `backend_mix` and latency count answered requests only. Measured
[in-process] with a provider that always fails: the client gets 503
`hearth.provider.unavailable`, and `/admin/metrics` reads
`{"requests":1, … "failed":1,"failure_rate":1.0,"backend_mix":{}, …}`.

**Dry-run a routing decision** [in-process]:

```sh
curl -s http://127.0.0.1:8080/v1/hearth/route -H "Authorization: Bearer $HEARTH_TOKEN" \
  -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"Summarize this diff"}]}'
# {"class":"summarize","method":"rules","backend":"local","model":"…Coder-7B…","would_escalate":false,"reason":"class policy: summarize->local","confidence":null}
```

**`hearth stats`.**

```sh
uv run --no-sync hearth stats --since 24h
```

The rows are requests, estimated frontier tokens saved, escalations, escalation rate,
escalations failed (remote errored; prompt may have left), failed (error, no answer), failure
rate, backend mix, class mix, and p50/p95 latency. "Escalations failed" counts every request
whose remote call errored, whether the local fallback then answered or failed too. **Metrics are
in-memory per process**, so `hearth stats` in a fresh shell always shows zeros. For a running
server, read `GET /v1/hearth/admin/metrics?since=24h` instead. It returns the same rollup as
JSON [in-process].

---

## 9. Finance, and training / eval / promotion

### 9.1 Finance

Read [docs/RUNBOOK_finance.md](RUNBOOK_finance.md) before using real statements. The rules
that matter:

- **Never paste statement contents into a cloud agent.** Column headers are safe; values are
  not.
- **Python computes every figure** (`Decimal`, verified twice). A model may only categorize
  rows or write prose about finished numbers.
- Always supply a **control total**. Without one, the sum is reported `UNVERIFIED`, not
  passing.

Quickstart on **synthetic** data. The flow is the runbook's §§1–4.1; the paths here are under
`/tmp`:

```sh
mkdir -p /tmp/hg_fin/incoming /tmp/hg_fin/mappings
printf 'Posting Date,Description,Amount,Balance\n01/03/2026,SYNTHETIC PAYROLL,2500.00,2500.00\n01/05/2026,SYNTHETIC GROCER #12,(84.20),2415.80\n01/09/2026,SYNTHETIC CAFE,(6.75),2409.05\n' > /tmp/hg_fin/incoming/demo.csv
export HEARTH_FILE_ROOTS=/tmp/hg_fin HEARTH_HOME=/tmp/hg_home
uv run --no-sync python scripts/hearth_peek.py /tmp/hg_fin/incoming/demo.csv   # headers + type guesses, never a value
```

`hearth_peek.py` prints column headers and type guesses only (here: `Posting Date` date-like,
`Description` text, `Amount` number-like with accounting negatives, `Balance` number-like); it
never prints a preamble line or a value-like cell (B-044, fixed).

Write `/tmp/hg_fin/mappings/demo.yaml`:

```yaml
bank: Demo Bank (synthetic)
date_column: Posting Date
description_column: Description
amount_column: Amount
date_format: "%m/%d/%Y"
sign: as_written
negative_notation: [parens]
decimal_separator: "."
thousands_separator: ","
skip_rows: 0
currency: USD
```

Then ingest, reconcile against the printed total, store, and total:

```python
from datetime import date
from decimal import Decimal
from hearth.config import Settings
from hearth.finance.mapping import ColumnMapping
from hearth.finance.parse import data_row_count, parse_rows, read_table
from hearth.finance.store import FinanceStore, hash_file, mapping_fingerprint
from hearth.finance.validate import reconcile

s = Settings(); path = "/tmp/hg_fin/incoming/demo.csv"
m = ColumnMapping.from_yaml("/tmp/hg_fin/mappings/demo.yaml")
rows = read_table(path, s); txns = parse_rows(rows, m)
recon = reconcile(txns, rows_read=data_row_count(rows, m), control_total=Decimal("2409.05"))
print(recon.describe())                                  # reconciliation: PASS … control total 2409.05 [verified]
store = FinanceStore(settings=s)                         # $HEARTH_HOME/finance/ledger.db
print(store.ingest(txns, recon, source_path=path, content_sha256=hash_file(path, s),
                   mapping_id="demo", mapping_version="1", fingerprint=mapping_fingerprint(m)).reason)
print(store.total(start=date(2026, 1, 1), end=date(2026, 1, 31)))   # Figure(label='total', amount=Decimal('2409.05'), count=3, …)
```

Once a ledger exists under your `HEARTH_HOME`, `hearth agent` offers `finance_total`,
`finance_explain` and `finance_rows` (measured: the header line then reads
`tools=finance_explain, finance_rows, finance_total, list_files, read_file`). For real data, use `scripts/hearth_private.sh --profile
config/routing.finance.yaml --check` and the runbook's sealed setup.
`examples/finance/run_finance_ladder.py` parses amounts with `hearth.finance.parse_money` and
keeps every figure `Decimal` (B-049, fixed; an AST test fails on any `float(` in it).

### 9.2 Training, eval and promotion

Read [docs/RUNBOOK_training.md](RUNBOOK_training.md) for the real-weights walkthrough (its
promotion steps use the commands below). `scripts/train_lora_real.sh --promote` still uses
the removed typed-score flags and always fails (B-015). The commands:

```sh
uv run --no-sync hearth train --task classify --base mlx-community/Qwen2.5-3B-Instruct-4bit --data data.jsonl   # (not run: real training)
uv run --no-sync hearth prereg anchor                      # the evals repository bars must live in
uv run --no-sync hearth prereg init --task classify --golden golden.jsonl --out prereg/classify.yaml
uv run --no-sync hearth prereg check prereg/classify.yaml --golden golden.jsonl
uv run --no-sync hearth eval <adapter-id> --golden golden.jsonl --prereg prereg/classify.yaml --report-json r.json
uv run --no-sync hearth eval <adapter-id> --golden golden.jsonl --prereg prereg/classify.yaml --promote          # (promotion with real scores: not run)
uv run --no-sync hearth adapters list
uv run --no-sync hearth adapters retire <adapter-id>
```

*(`prereg init/check`, `eval` without `--promote`, `eval --promote` without a prereg,
`adapters list/retire/promote` were run on echo with a synthetic 6-row golden set and a fake
registered adapter, before B-061; `eval` now also needs the adapter's weights on disk and a
registered base model.)*

**Datasets and `hearth train`.**

- A **dataset JSONL** needs a header line, e.g.
  `{"kind":"hearth.dataset.header","schema_version":1,"task":"classify","version":"v1"}`,
  followed by at least **two** `{"prompt","completion"}` or `{"messages":[…]}` rows. A
  headerless file is refused (exit 1) with `Dataset error: dataset task must be non-empty: the
  file has no header line. …`. A file with one row is refused before training starts, exit 1,
  no traceback (measured): `Dataset error: need at least 2 records to split into train/valid`.
- `train` only produces a **candidate** (`~/.hearth/train/<run-id>/`). It needs the base model
  on disk; one that is not fails cleanly (exit 1) with `model '…' is not on disk (looked in
  ~/.hearth/models and the huggingface hub cache) and HEARTH does not download on load. Fetch
  it deliberately with hearth models pull …`, and **leaves no run directory behind** (measured).
  If the training process itself fails, `train` exits 1 with its return code and the tail of
  its stderr, not a traceback.
- After a successful run it prints `Registered candidate <id>. It is not served until it
  passes the eval gate:` followed by `hearth eval <id> --golden <set> --prereg <committed
  prereg> --promote`, or `--report-json <file>` then `hearth adapters promote <id> --report
  <file> --prereg <committed prereg>` (B-046, fixed; from the code and
  `tests/test_cli_training.py`, since a real run needs weights).

**Pre-registration.** `prereg init` writes a template pinning the golden set (content sha) and
the decode parameters, with `hypothesis`, `stopping_rule` and `kill_condition` blank. **You
must write those three** before anything accepts the file. Measured on the unedited template:

```text
Pre-registration error: pre-registration has blank ['hypothesis', 'stopping_rule', 'kill_condition']: write
the hypothesis, when you will stop, and what result would kill the idea BEFORE training — an unedited template
is not a pre-registration
```

With the prose written, `prereg check` prints the bar (task, metric, golden_sha, alpha 0.05,
min_effect 0, min_n 30, test auto, baselines `empty, majority_label, copy_input`), `Golden set
matches (6 examples).`, then `git: not committed — …` and exits 1 until you `git commit` it.
Removing a default baseline is also refused.

**`hearth eval`** prints a score table (candidate, the incumbent or `base`, and each baseline),
then `gate: PASS|FAIL n=… alpha=… <test> p=…` and, on FAIL, one line per reason. Measured on
echo with a 6-row set:

```text
gate: FAIL n=6 alpha=0.05 mcnemar_exact p=1.0000 (b=0, c=0)
  · golden set too small: n=6 < min_n=30 (see stats.min_n_for_alpha for what a set this size can license)
  · no lift: candidate 0.0000 does not exceed base 0.0000 + margin 0
  · not significant: mcnemar_exact p=1.0000 > alpha=0.05 (b=0, c=0)
  · fails degenerate baseline 'majority_label': candidate 0.0000 does not exceed 1.0000 + margin 0
  …
```

**Every eval is recorded first.** Before it scores anything, `eval` appends a signed record
to `~/.hearth/measurements.jsonl` (adapter id, weights hash, golden set and its repo's HEAD,
metric, decode fingerprint, backend, time), with or without `--prereg`. Promotion needs the
prereg committed before the adapter's **first** record (B-079), so **commit the bar before
you run `eval` on the adapter at all**. Without `--prereg`, the gate line is followed by
`Exploratory measurement (no --prereg): … can only be promoted under a pre-registration
committed BEFORE its first measurement (…)`.

Without `--promote`, `eval` **exits 0 once it has measured, even when the gate FAILs**: read
the `gate:` line, not the exit code. It exits 1 when it refuses to measure (unknown adapter
or `--metric`, adapter weights missing on disk, bad golden set or prereg, a golden set that
repeats a prompt, a bar looser than the gate allows, `--temperature` above 0 without
`--allow-sampling`, a measurement ledger that cannot be written or does not verify), and 2
when `HEARTH_DEFAULT_MODEL` names
an unregistered model or the adapter's base model is empty, `auto` or not servable — it never
measures on a silent fallback model (B-070). With `--promote`, it exits 0 only if the adapter
was promoted.

**The promotion gate** (CLAUDE.md §7). You cannot promote an adapter on a score you typed.
`hearth eval --promote` requires all of the following:

- a `--prereg` that is **git-committed and unmodified** — its bytes are the committed blob,
  checked by hashing, so `--assume-unchanged` cannot hide an edit (B-078) (without one:
  `Promotion refused: --promote requires --prereg. …`, exit 1) — whose last commit
  **precedes the adapter's first recorded measurement**, in time and in history (B-079), and
  which lives, with the committed golden set, in the **anchored evals repository**
  (`hearth prereg anchor`; HEARTH's own repo by default) recorded at that first measurement
  (B-061, B-081). The golden set is re-read from its committed blob and must hash to what
  was scored, with every prompt distinct (B-080);
- a bar no looser than the defaults: α in (0, 0.05], `min_effect` ≥ 0, `min_n` ≥ 30, a
  known `test`, every number finite (B-062: a NaN or α = 1 used to disable the gate);
- evaluation at **temperature 0** (`--temperature > 0` is refused unless `--allow-sampling`,
  and then it cannot gate);
- an **incumbent**: the promoted adapter for the task, or the **base model** when none is
  promoted;
- **significance over paired per-example vectors** (exact McNemar or paired bootstrap) at the
  prereg's α;
- beating the **empty / majority-label / copy-input** baselines;
- the adapter's base model as registered (not a `--base` override), and weights that hash the
  same when promoted as when scored;
- scores generated by the MLX model pool (`HEARTH_BACKEND=mlx`) — not echo, not a plugin
  (B-084). A promoted incumbent is scored on its own base model (B-085), and it must still
  be the incumbent when the registry is written, checked under the registry lock (B-082).

**n ≥ 5 is the mathematical floor** at α = 0.05, because the smallest achievable p is 0.5ⁿ.
The default `min_n = 30` is a power floor above that; a prereg may raise it, never lower it.
The repo's golden sets are n = 5 and n = 6, below that floor (B-010).

`hearth adapters promote` requires `--report` and `--prereg`. The report must be one
`hearth eval --report-json` wrote **on this install**: it is HMAC-signed with
`~/.hearth/eval-report.key` (created 0600 on first use), and an unsigned, edited or
other-install report is refused (`Unusable eval report: …`). It must also be about this
adapter — same id, task, base model and weights path, weights that still hash to what was
measured — and the incumbent it beat must still be the incumbent; then the gate is recomputed
from its vectors, the report's measurement must be in this install's ledger, and the
prereg's provenance is re-checked as above. The proof records the commit that last changed
the prereg (not HEAD), the weights hash, the report sha and the ledger positions
(`first_measured_at`). The
removed `--candidate-score/--incumbent-score` flags exit 2. *(These paths run in CI on a fake
provider — `tests/test_promotion_evidence.py`, `tests/test_cli_eval.py`; a promotion with
real scores was not run.)* Background: [docs/LEARNING_plan.md](LEARNING_plan.md).

---

## 10. Environment variable reference

Settings come from `HEARTH_*` environment variables (`src/hearth/config.py`,
`env_prefix="HEARTH_"`, `extra="ignore"`). **A misspelled name is silently ignored.** Run
`scripts/hearth_status.py --section environment` to flag it. `hearth serve` and `hearth mcp`
read settings once at startup; restart them after a change. Booleans accept
`1/0/true/false/yes/no`. The man page's ENVIRONMENT section (§2.6) is generated from the same
fields.

### 10.1 Settings fields

| Variable | Default | Read by | Notes |
|---|---|---|---|
| `HEARTH_HOST` | `127.0.0.1` | `hearth serve` bind; `doctor --offline` `bind_host` | non-loopback makes doctor UNSAFE; `--host` overrides |
| `HEARTH_PORT` | `8080` | `hearth serve` | `--port` overrides |
| `HEARTH_BACKEND` | `auto` | `providers/__init__.py:select_provider` (serve, run, agent, mcp, rag query, eval) | `auto` = mlx if `mlx_lm` imports, else the echo stub **with a WARNING** at startup, `/ready` 503 `stub` and `/health` `backend_fallback` (B-006, fixed); `mlx`; `echo`; or a plugin name. An unknown value makes the command fail with a traceback (`Unknown HEARTH_BACKEND`) |
| `HEARTH_REQUIRE_AUTH` | `true` | `gateway/auth.py` | `false` disables bearer auth on `/v1/*` |
| `HEARTH_EMBEDDER` | `hash` | `memory/embed.py:select_embedder` (rag, `/v1/embeddings`, agent `rag_search`) | `mlx` is broken (B-011) |
| `HEARTH_EMBED_DIM` | `256` | hash embedder (`memory/embed.py`) | vector dimension |
| `HEARTH_EMBED_MODEL` | `mlx-community/bge-small-en-v1.5-mlx` | MLX embedder; doctor | this default id is not on disk (B-011) |
| `HEARTH_VECTOR_STORE` | `sqlite` | `memory/store.py:select_vector_store` | `sqlite-vec` (`--extra vec`) or a plugin name |
| `HEARTH_RAM_CEILING_GB` | `24.0` | `ModelPool` / `ModelManager` (`serving/`) | resident-model budget; LRU eviction above it (§3.5) |
| `HEARTH_WARMUP` | `true` | `gateway/app.py` | load the default model in the background at `serve` start; no-op on echo. Off: nothing loads until the first request, and `/ready` is 200 (`loaded: false`) while the default's weights resolve on disk (§4.7) |
| `HEARTH_FILE_ROOTS` | `""` (deny all) | `mcp/files.py:allowed_roots`: MCP `*_file` tools, agent `read_file`/`list_files`/`search_files` (CLI and HTTP), finance `read_table`, `scripts/hearth_peek.py` | colon-separated directories; `~` is expanded; non-existent entries are dropped |
| `HEARTH_FILE_MAX_BYTES` | `2000000` | `mcp/files.py` | larger files are refused, not truncated |
| `HEARTH_ALLOW_DOWNLOADS` | `false` | `providers/mlx.py:resolve_local_model`; doctor | `1` lets loads fetch missing models; makes doctor UNSAFE |
| `HEARTH_HOME` | `~/.hearth` | `config.py` (token, `models/`, `rag/`, `adapters.json`, `train/`, `finance/ledger.db`); also read directly by `handoff/store.py`, `finance/store.py` and the status probes | point at a scratch directory for experiments |

### 10.2 Read directly from the environment (not Settings)

From `status/probes.py:_EXTRA_ENV_NAMES`:

| Variable | Default | Read by | Notes |
|---|---|---|---|
| `HEARTH_DEFAULT_MODEL` | the `default:` key of `config/models.yaml` | `registry/__init__.py:Registry.default_id` (the only reader; there is no Settings field, B-042); `/ready` | an unregistered id makes `serve`/`run`/`agent`/`mcp`/`rag query --answer` refuse to start (exit 2) and `create_app()` raise; `doctor` FAIL, `doctor --offline` WARN (§3.4) |
| `HEARTH_ROUTING_YAML` | `<repo>/config/routing.yaml` | `router/policy.py:resolve_routing_selection` (router, doctor, status probe); `scripts/hearth_private.sh` | `~` expanded; relative paths resolve against the repo root; a missing named file is an error (§8) |
| `HEARTH_MODELS_YAML` | `<repo>/config/models.yaml` | `registry/__init__.py` | alternate registry file |
| `HEARTH_BASE_MODEL` | `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` | `scripts/train_lora_real.sh` only | |
| `HEARTH_TRAIN_DATA` | — | `scripts/train_lora_real.sh` only | |
| `HEARTH_TRAIN_TASK` | `extract` | `scripts/train_lora_real.sh` only | |
| `HEARTH_TRAIN_ITERS` | `200` | `scripts/train_lora_real.sh` only | |
| `HEARTH_TRAIN_OUT` | — | `scripts/train_lora_real.sh` only | |
| `HEARTH_CANDIDATE_SCORE`, `HEARTH_INCUMBENT_SCORE` | — | `scripts/train_lora_real.sh` only | feed the removed typed-score promote path; promotion fails (B-015) |

Used only by examples (not HEARTH itself): `HEARTH_URL`, `HEARTH_TOKEN`
(`examples/cambot_offload.py`), `HEARTH_REPO` (`examples/cmux/pane_offload_live.sh`).
`HEARTH_MODEL` is read by nothing (see §3.4).

### 10.3 Non-HEARTH variables that matter

| Variable | Effect |
|---|---|
| `HF_HUB_CACHE`, `HF_HOME` | where the resolver's hub-cache lookup looks (§3.2) |
| `HF_ENDPOINT` | mirror used by `hearth models pull` |
| `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` | not needed by HEARTH's own load paths; `hearth_private.sh` sets them for any other hub call |
| `ANTHROPIC_API_KEY` | Claude escalation under `routing.remote.yaml` (or the remote's `api_key_env`) |

---

## 11. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Answers start with `[echo] …`; serve banner, `served by … via echo` or `/v1/hearth/admin/health` says `echo` | `mlx_lm` is missing, so `HEARTH_BACKEND=auto` fell back to the echo stub. It says so: a startup WARNING `HEARTH_BACKEND=auto found no importable mlx_lm, so the echo STUB is serving …`, `/ready` 503 `stub` and `/health` `backend_fallback` (B-006, fixed; measured [in-process] with `mlx_lm` hidden). Usually a bare `uv run`/`uv sync` pruned the venv | `uv sync --extra mlx --extra mcp --extra dev --extra files`, then the verify import (§2.2). Set `HEARTH_BACKEND=mlx` so a missing MLX fails loudly instead of degrading |
| `ModuleNotFoundError: mlx_lm` / `mcp`, or `The MCP server requires the 'mcp' extra.` | the venv was pruned by a partial sync | same one-command sync |
| `model '<id>' is not on disk (looked in ~/.hearth/models and the huggingface hub cache) and HEARTH does not download on load` (`ModelNotOnDiskError`); chat returns 503 `hearth.provider.unavailable`; `/ready` 503 `failed` | the weights are not in either location | `uv run --no-sync hearth models pull <id>`, then `hearth doctor --offline` shows where it resolved. Do not set `HEARTH_ALLOW_DOWNLOADS=1` to "fix" it |
| `404` `model_not_found` from the API, or `Unknown model: …` and exit 2 from `run` / `agent` | the requested id is not a chat model this backend can serve (a typo, `echo` on mlx, the embedder) | pick one from the message's `Servable models:` list, or `GET /v1/models`; register new models in `config/models.yaml` |
| `401` `hearth.auth.unauthorized` | missing or wrong bearer token, or `/chat` has a stale token | send `Authorization: Bearer $(cat ~/.hearth/token)`; re-paste the token in `/chat`. Under a custom `HEARTH_HOME` the token is `$HEARTH_HOME/token` |
| `/v1/hearth/admin/ready` returns 503 | read `status` and `reason` in the body (§4.7): `loading` + `warmup in progress` (wait a few seconds); `failed` + `ModelNotOnDiskError` (pull it); `failed` + `… do not resolve on disk` (pulled weights deleted, or never pulled with warmup off); `stub` (the echo fallback, row above) | as the reason says |
| `hearth doctor --offline` says UNSAFE | each FAIL row names one cause: `routing_profile` (a remote profile is selected, or the named file does not exist), `bind_host` (`HEARTH_HOST` not loopback), `allow_downloads` (`HEARTH_ALLOW_DOWNLOADS` on, which also fails `serving_resolution` and the `load path` rows), `model <id>` (weights not on disk) | unset the offending variable, or pull the model; re-run until `SAFE offline` |
| `Refusing to start: HEARTH_DEFAULT_MODEL='…' is not in the model registry`, exit 2; `doctor` FAIL `default_model` | the variable names an unregistered id (§3.4) | fix the id, register the model, or unset the variable |
| The model you set as default is not the one answering | `HEARTH_DEFAULT_MODEL` misspelled as `HEARTH_MODEL` (silently ignored, §3.4) | `uv run --no-sync hearth models list` shows the real `(default)` |
| `404` `adapter_not_found` | the request names an adapter that is unknown or retired (§4.6) | check the id with `hearth adapters list` |
| `Routing profile not found: … which does not exist (relative paths resolve against the repo root …)`, exit 2 | `HEARTH_ROUTING_YAML` names a missing file (§8) | fix the path (relative to the repo root) or unset it |
| `error while attempting to bind on address ('127.0.0.1', 8080): address already in use` *(message not reproduced in docs verification)* | another `hearth serve` (or other process) holds the port | `lsof -nP -iTCP:8080 -sTCP:LISTEN` to find it, or `hearth serve --port 8081` |
| Agent exits 2 `No readable file roots` / `Refusing to start` | `HEARTH_FILE_ROOTS` is unset or names no existing directory | `HEARTH_FILE_ROOTS=/abs/dir hearth agent …` |
| Agent exits 1 `NO ANSWER — … 'max_iterations'` (or `'invalid_output'`) | the task needs more steps than the budget, or the model wandered; on echo, always `invalid_output` | narrow the task, raise `--max-iterations`, read the step table; check `backend=` |
| `hearth stats` shows all zeros | metrics are per-process, in memory | query `GET /v1/hearth/admin/metrics` on the running server |
| `hearth train`: `Dataset error: need at least 2 records to split into train/valid`, exit 1 | the dataset has only one row | add rows: at least 2 after the header |
| `Embedder unavailable: could not load embedding model …`, exit 1 | `HEARTH_EMBEDDER=mlx`: its default id is not on disk, and even the registered bge weights cannot load (B-011) | use `HEARTH_EMBEDDER=hash` |

---

## 12. Command cheat-sheet

Prefix every command with `uv run --no-sync` from the repo root. `hearth COMMAND --help` and
`man ./man/hearth.1` have the details.

```text
hearth --help                                    commands by panel, first steps, pointers
hearth version                                   print version
hearth doctor                                    environment preflight (exit 1 on a fatal FAIL)
hearth doctor --offline                          offline-safety verdict (exit 1 = UNSAFE)

hearth models list                               registry; (default) marks what serves
hearth models pull <registry-id>                 the ONLY command that downloads
hearth models rm <registry-id>                   delete from ~/.hearth/models (not the hub cache)
hearth models convert --source X --out DIR [--q-bits 4]
hearth models export-coreml --source X --out DIR

hearth serve [--host H] [--port P]               gateway + /chat on 127.0.0.1:8080
hearth run "prompt" [--file F] [--intent C] [--model ID] [--max-tokens N]
hearth agent "task" [--collection C] [--no-finance] [--model ID] [--max-iterations 8]
             [--max-seconds 180] [--max-tokens 24000] [--no-steps|--full|--json]
hearth mcp                                       stdio MCP server for Claude Code
hearth stats [--since 24h]                       this process's rollups (zeros in a fresh shell)

hearth rag ingest PATH [--collection C] [--size 800] [--overlap 100]
hearth rag query "q" [--collection C] [--k 6] [--answer]

hearth train --task T --base MODEL --data D.jsonl [--iters 200] [--out DIR] [--no-register]
hearth prereg init --task T --golden G.jsonl [--out P.yaml]
hearth prereg check P.yaml [--golden G.jsonl]
hearth eval ADAPTER --golden G.jsonl [--prereg P.yaml --promote] [--report-json R.json]
hearth adapters list [--task T] [--status S]
hearth adapters promote ADAPTER --report R.json --prereg P.yaml
hearth adapters retire ADAPTER
hearth plugins list

man ./man/hearth.1                                          the reference manual
python scripts/gen_manpage.py [--check]                     regenerate (or check) man/hearth.1
python scripts/hearth_status.py [--section environment]     measured status
python scripts/hearth_peek.py FILE                          headers + type guesses, no values
scripts/hearth_private.sh [--profile P] [--check]           sealed no-egress serve
```

HTTP, with `H="Authorization: Bearer $(cat ~/.hearth/token)"`:

```text
curl -s localhost:8080/v1/hearth/admin/ready                 200 ready / 503 loading|failed + reason (no token)
curl -s -H "$H" localhost:8080/v1/models                     servable chat models
curl -s -H "$H" localhost:8080/v1/hearth/admin/models        resident models, loaded paths
curl -s -H "$H" "localhost:8080/v1/hearth/admin/metrics?since=24h"
```

Key files: `~/.hearth/token` · `~/.hearth/models/` · `~/.hearth/rag/<c>.db` ·
`~/.hearth/adapters.json` · `~/.hearth/finance/ledger.db` · `config/models.yaml` ·
`config/routing*.yaml` · `man/hearth.1`.

Other docs: [docs/README.md](README.md).
