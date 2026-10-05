# HEARTH — User Guide

The start-to-finish guide for using HEARTH on this Mac. It is task-oriented: each section is a
thing you want to do, with commands you can paste. Every command and variable here was
checked against the code as of `13b1438` (`cmux/integration`, 2026-10-05).

**How to read the markers.**

- **Unmarked commands** were run during docs verification, and their output was checked
  against what this guide says. The run used offline mode, an isolated `HEARTH_HOME` under
  `/tmp`, and `HEARTH_BACKEND=echo` wherever a model would otherwise have answered.
- **[in-process]** means the HTTP request was checked through FastAPI's `TestClient` against
  the real app (echo backend), not over a live socket. The verification sandbox could not
  bind a port.
- **(not run in docs verification)** means the command was not run. Usually it needs real
  weights, the network, or changes to your Claude Code configuration.
- **⚠ pending merge of B-004 / B-005** marks behaviour that is coded on a branch but not yet
  merged. Those sections describe how it **will** work. Today's behaviour is stated next to
  each one.

Always run HEARTH from the repo with **`uv run --no-sync hearth …`**. Never use a bare
`uv run` or `uv sync` (see §2 and §11).

---

## Contents

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
removes 33 packages, including `mlx`, `mlx-lm`, `mcp` and `pytest`. Local inference then dies
silently: HEARTH falls back to the echo stub (§11). To add an optional extra, append it to the
full command, e.g. `… --extra files --extra remote`.

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
(`~/.hearth` writable). It ends with `Ready.` and exits 0. A FAIL row is fatal (exit 1). A
WARN row is not.

### 2.4 `hearth doctor --offline`: is it safe to use offline right now?

```sh
uv run --no-sync hearth doctor --offline
echo $?        # 0 = SAFE, 1 = UNSAFE
```

Each row is a measurement, not a reading of a config value:

| Row | PASS means |
|---|---|
| `routing_profile` | the profile the router would load now has 0 remotes, every class local/never |
| `backend` | the `HEARTH_BACKEND` value |
| `bind_host` | `HEARTH_HOST` is loopback |
| `allow_downloads` | `HEARTH_ALLOW_DOWNLOADS` is off |
| `serving_resolution` | a model that is not on disk fails to load, and no connection is attempted |
| `model <id>` | each reachable model (the default, plus every class's `local_model` in the profile) is on disk; the row names the path |
| `load path: hearth train / models convert / models export-coreml` | each is handed a local path with the hub pinned offline |
| `hf_hub_offline`, `models_pull` | informational |

The last line is either `SAFE offline — no check found a path off this machine.` (exit 0), or
`UNSAFE offline: <failed rows>` (exit 1). For example, this was measured with three unsafe
settings at once:

```sh
HEARTH_ROUTING_YAML=$PWD/config/routing.remote.yaml HEARTH_HOST=0.0.0.0 HEARTH_ALLOW_DOWNLOADS=1 \
  uv run --no-sync hearth doctor --offline
# UNSAFE offline: routing_profile, bind_host, allow_downloads, serving_resolution,
#   load path: hearth train, load path: hearth models convert, load path: hearth models export-coreml
```

**Two limits, both printed as `not measured:` lines.** First, doctor measures **this
command's environment**. A `hearth serve` you started earlier with different variables is not
what it checked: run doctor with the same variables you serve with. Second, it covers
HEARTH's router and loaders, not machine-level containment (§1). It also does not warn when
`HEARTH_DEFAULT_MODEL` names an unregistered model. That value is silently ignored, and doctor
checks the model that will actually serve (B-029, §3.4).

### 2.5 Measured status (optional)

```sh
uv run --no-sync python scripts/hearth_status.py
uv run --no-sync python scripts/hearth_status.py --section environment
```

This prints which weights are on disk, which routing profiles are no-egress, the GPU
working-set ceiling, golden-set sizes, and how stale each doc is. It also flags `HEARTH_*`
variables that nothing reads (e.g. `HEARTH_MODEL` → `SILENTLY IGNORED`). Prefer it over any
written status, including this guide.

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
  `pull echo` says there is nothing to pull. Pull respects `HF_ENDPOINT` (a mirror) and
  `HF_HUB_OFFLINE`.
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

### 3.4 Choosing the default model: `HEARTH_DEFAULT_MODEL`

```sh
HEARTH_DEFAULT_MODEL=mlx-community/Qwen2.5-3B-Instruct-4bit uv run --no-sync hearth models list
# … mlx-community/Qwen2.5-3B-Instruct-4bit (default) …
```

**Trap 1: the name.** The variable is `HEARTH_DEFAULT_MODEL`, not `HEARTH_MODEL`. Settings
ignore unknown `HEARTH_*` names silently.

**Trap 2: unregistered ids are ignored (B-029).** If the value is not an `id` in
`config/models.yaml`, HEARTH silently keeps the catalog default. Nothing warns you, and
`doctor --offline` still says SAFE (it checks what will really serve). Measured: with
`HEARTH_DEFAULT_MODEL=mlx-community/Qwen2.5-Coder-32B-Instruct-4bit`, `models list` still
marks Coder-7B `(default)`. **Always confirm with `hearth models list`.** To use a new model,
register it first (§3.3).

### 3.5 Memory: the real ceiling

This M3 Pro advertises 36 GB, but the GPU working-set ceiling (the driver's limit on resident
GPU memory) is **30.15 GB**. That is what `scripts/hearth_status.py` reports as
`gpu_working_set`. Size models against that figure. `HEARTH_RAM_CEILING_GB` (default 24.0) is
the budget HEARTH's model manager keeps resident models under. Until B-004 merges, the server
holds one model, so the setting has little effect. Background: [docs/MODELS_local.md](MODELS_local.md).

### 3.6 Convert and Core ML export (advanced)

```sh
uv run --no-sync hearth models convert --source <id-or-path> --out ~/.hearth/models/<name>-q4 --q-bits 4
uv run --no-sync hearth models export-coreml --source <id-or-path> --out ~/.hearth/coreml/<name>
```
*(not run in docs verification: need real weights; `export-coreml` also needs `--extra coreml`)*

Both are disk-only. Register a converted model in `config/models.yaml` to serve it.

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

**Model loading.** Today, the weights load lazily on the **first request**. That request pays
the full load time, about 15 s cold for a 14B model. `HEARTH_WARMUP` (on by default) does not
yet touch the weights (B-005). ⚠ After B-005 merges, warmup loads the default model in the
background at startup, and `/ready` turns 200 once it has loaded (§4.7).

### 4.2 The token

- On first run HEARTH creates **`~/.hearth/token`** (mode `0600`).
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

Open **http://127.0.0.1:8080/chat**. In the header:

- **token**: paste the contents of `~/.hearth/token` once (`pbcopy < ~/.hearth/token`). It is
  kept in the page's `localStorage` on the loopback origin. The server never embeds it in the
  page.
- **Load models** / **model**: the dropdown is filled from `/v1/models`.
  ⚠ **Today the selection is ignored; the default model answers (B-004).** After B-004 merges,
  the model you pick is the model that serves.
- **temp**, **max tokens** (default 512). A reply cut off at max tokens is flagged
  **Truncated**.
- **agent mode** (off by default) switches from plain chat to the local agent loop. That loop
  has read-only tools (list and read files under `HEARTH_FILE_ROOTS`, RAG search), and each
  step shows as its own turn. **steps** caps the iterations (max 12). A run that hits a bound
  is shown as **NOT AN ANSWER**. With agent mode off, plain chat has no tools at all: a
  question about your files gets an invented answer. Turn agent mode on for those.
- Each reply shows its provenance: served on-device or remotely, model, backend, adapter.

The page is self-contained (no CDN, no fonts) and talks only to `/v1/models`,
`/v1/chat/completions` and, in agent mode, `/v1/hearth/agent`.

> Agent mode over HTTP differs from the CLI agent (B-012). It offers `rag_search` without a
> pinned collection, so the model must guess the collection name. It never offers the finance
> tools.

### 4.4 The OpenAI-compatible API

| Endpoint | Auth | What |
|---|---|---|
| `POST /v1/chat/completions` | yes | chat, streaming or not |
| `GET /v1/models` | yes | the registry |
| `POST /v1/embeddings` | yes | embeddings from the configured embedder (`hash` by default) |
| `POST /v1/hearth/route` | yes | dry run: which class, backend and model the router *would* use |
| `POST /v1/hearth/rag/ingest`, `/rag/query` | yes | RAG over HTTP (§6) |
| `POST /v1/hearth/agent` | yes | the agent loop, streamed as SSE (docs/AGENT.md §9) |
| `GET /v1/hearth/admin/metrics?since=24h` | yes | rollups: requests, tokens saved, escalations, latency |
| `GET /v1/hearth/admin/health` | no | liveness: version, backend, default model |
| `GET /v1/hearth/admin/ready` | no | readiness, 200 or 503 (§4.7) |
| `GET /chat` | no | the chat page |

`docs/API.md` also describes `/v1/hearth/classify`, `/v1/hearth/summarize`, `/v1/hearth/train/*`
and `/admin/models/{id}/load|unload`. **These do not exist in the current gateway.**

**curl** [in-process]:

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $HEARTH_TOKEN" -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":64}'
```

Response (echo backend shown; the content differs on a real model):

```json
{"id":"chatcmpl-…","object":"chat.completion","created":…,
 "model":"mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
 "choices":[{"index":0,"message":{"role":"assistant","content":"…"},"finish_reason":"stop"}],
 "usage":{"prompt_tokens":6,"completion_tokens":7,"total_tokens":13},
 "hearth":{"served_by":"local","backend":"mlx","model":"…","adapter":null,
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
[in-process]:

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
block [in-process]. If generation fails mid-stream, the stream ends with an `error` event
(`hearth.provider.unavailable`) and then `[DONE]`. Dropping the connection stops generation.

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
  candidate. **Caveat:** an unknown adapter id silently falls back to base weights, yet the
  response's `hearth.adapter` still echoes the id you asked for [in-process]. Without
  `adapter`, a **promoted** adapter for the request's class is applied automatically when its
  base model is the one serving. On this machine `classify-20260710T020135Z` is promoted on
  Coder-7B, so `classify` requests use it (B-010).

Response `hearth` block: `served_by` (`local` | `remote`), `backend`, `model`, `adapter`,
`escalated`, `estimated_frontier_tokens_saved`.

### 4.7 Choosing a model per request, and readiness

> **⚠ pending merge of B-004 / B-005.** This describes the behaviour coded on branch
> `worktree-agent-ac0ade7fd1fa99aba` (WIP `d1d8434`). Finalize this section after it merges.

**How it will work:**

- `"model": "<registered chat id>"` is served by **those weights**, loaded on demand and
  evicted least-recently-used under `HEARTH_RAM_CEILING_GB`. Telemetry names the model that
  actually generated, on streaming responses too.
- An **unknown or non-chat id** (a typo, `echo` on the mlx backend, the embedding model)
  returns **404** with `code: "model_not_found"`. It never falls back silently. The CLI exits
  2.
- `"model": "auto"` (or omitted) uses the routing profile's **per-class ladder**
  (`local_model` per class, e.g. finance: classify→3B, summarize→14B), then the default.
  `hearth run` and `hearth agent` gain `--model`, defaulting to `auto`.
- `GET /v1/hearth/admin/ready` returns 200 only when the default model's weights are really
  loaded. Otherwise it returns 503 with status `loading`, or `failed` with a reason (the load
  raised, or `HEARTH_DEFAULT_MODEL` names an unregistered id).
- New: `GET /v1/hearth/admin/models` (auth) lists the resident models, their loaded paths and
  generation counts.

**How it works today (measured):**

- `model` is ignored by the mlx backend: the default model answers every request. A bogus
  `model` gets a **200**, and on the echo backend the response is labelled with the bogus id
  [in-process]. On streaming responses the label can name a model that did not run.
- `/ready` returns **200 even when the weights are not on disk**. The first chat request then
  fails with 503 `hearth.provider.unavailable` / `ModelNotOnDiskError` [in-process]. On echo,
  `/ready` is always 200. With `HEARTH_WARMUP=false` on mlx, `/ready` stays at 503 `loading`
  permanently [in-process].

---

## 5. `hearth run` and `hearth agent`

### 5.1 `hearth run`: one-shot completion

```sh
uv run --no-sync hearth run "summarize: <text>"
uv run --no-sync hearth run --file notes.txt --max-tokens 256
echo "prompt from stdin" | uv run --no-sync hearth run
uv run --no-sync hearth run "label this ticket: …" --intent classify
```

- It prints only the model's text. With `--intent`, a dim `intent=<x>` line comes first.
- An empty prompt prints `No prompt provided.` and exits 1.
- Defaults: `--max-tokens 512`.
- **Always local.** `hearth run` never escalates, whatever the routing profile.
- It has **no tools**: it cannot read files you mention. Use `--file` to send a file's
  contents as the prompt, or use `hearth agent`.
- Today it always uses the default model (there is no `--model` yet, B-004), and it pins that
  model explicitly. A profile's per-class ladder therefore does not apply to `hearth run`
  until B-004 merges.

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
| `list_files(root, pattern)`, `read_file(path)` | always (refuse everything without roots) | `HEARTH_FILE_ROOTS`, `HEARTH_FILE_MAX_BYTES`, the format table |
| `rag_search(query, …)` | `--collection NAME` given (and it is non-empty) | the RAG index |
| `finance_total / finance_explain / finance_rows` | a ledger exists at `~/.hearth/finance/ledger.db` (disable with `--no-finance`) | Decimal arithmetic, integrity checks |

There are no write, shell or network tools, and no flag to add them.

**`HEARTH_FILE_ROOTS`** is a colon-separated list of directories. It is **deny-by-default**:
unset means every read is refused, with no implicit root (not the current directory, not
`$HOME`). Paths are fully resolved (`..`, symlinks) before the containment check. Readable
formats: `.txt .text .md .markdown .rst .log` and extension-less files as text, plus `.csv`,
`.json`, `.xlsx` and `.pdf` (XLSX/PDF need `--extra files`).

**Budgets:**

| Option | Default | |
|---|---|---|
| `--max-iterations` | 8 | model turns |
| `--max-seconds` | 180 | wall clock |
| `--max-tokens` | 24000 | prompt + completion, all steps |
| `--steps/--no-steps` | on | the step table (tool, arguments, truncated observation, tokens, timings) |
| `--full` | off | the raw per-step transcript instead of the table |
| `--json` | off | the whole run as one JSON document (`completed`, `stopped_reason`, `answer` = `null` unless answered, every step) |

**Exit codes are the stop reason:**

| Exit | Meaning |
|---|---|
| `0` | the model answered, and only then |
| `1` | stopped at a bound or a failure: `max_iterations`, `timeout`, `token_budget`, `invalid_output` (3 unusable replies in a row), `provider_error`, `egress_refused`. It prints `NO ANSWER — the run stopped because '<reason>'`, and the steps are a partial trace, not a result |
| `2` | never started: an impossible bound (`max_iterations must be at least 1`), an empty `--collection`, or nothing reachable (no roots, no collection, no ledger: `Refusing to start`) |

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
  `Done. N file(s), M chunk(s)`. Options: `--collection` (default `default`), `--size 800`,
  `--overlap 100` (characters).
- `query` prints a table of score, source and text. `--k` defaults to 6. A missing or empty
  collection prints `No chunks in collection '<name>'.` and exits 0.
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
cannot load. Measured: `rag ingest` with `HEARTH_EMBEDDER=mlx` exits 1 with a traceback ending
in `EmbeddingUnavailableError` / `ModelNotOnDiskError`. Stay on `hash`.

`HEARTH_VECTOR_STORE` is `sqlite` (default) or `sqlite-vec` (needs `--extra vec`).

---

## 7. MCP: using HEARTH from Claude Code

`hearth mcp` starts a **stdio** MCP server named `hearth`. Claude Code can then hand routine
subtasks to the local model, with zero frontier tokens spent. Every tool runs with escalation
**disabled**, under any routing profile, in-process: no HTTP and no token.

**Tools** (verified by listing them over stdio):

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

If you add `HEARTH_ROUTING_YAML` to `env`, use an **absolute** path (§8). Without the `mcp`
extra, `hearth mcp` prints `The MCP server requires the 'mcp' extra.` and exits 1. Claude Code
then just shows no hearth tools. Re-run the one-command sync (§2.1).

---

## 8. Routing profiles and escalation

A routing profile is a YAML file that decides, per task class, where a request runs. Select
one with **`HEARTH_ROUTING_YAML`**. Use an **absolute path**: a relative path resolves from
the current directory, not the repo (B-008), and `~` is not expanded (B-025). A missing file
does not crash anything. The router logs a warning and falls back to built-in safe defaults
(all local), so a typo silently loses the profile you meant.

| Profile | Egress | What it is for |
|---|---|---|
| `config/routing.yaml` (default when unset) | **none**: 0 remotes, all classes local/never | everyday use |
| `config/routing.private.yaml` | **none** | confidential work; used by `scripts/hearth_private.sh` |
| `config/routing.finance.yaml` | **none** | a two-tier **local ladder**: classify/extract/rank → Qwen2.5-3B; summarize/draft/reason/chat/code → Qwen2.5-14B |
| `config/routing.remote.yaml` | **yes**: Claude (`claude-opus-4-8`, Anthropic SDK), 200k tokens/day | opt-in escalation, only for work that may leave the machine |
| `config/routing.escalation-demo.yaml` | to `127.0.0.1:8099` only | demo: escalation goes to a local stub (`scripts/frontier_stub.py`) |

```sh
HEARTH_ROUTING_YAML=$PWD/config/routing.finance.yaml uv run --no-sync hearth doctor --offline   # SAFE; checks the 3B and 14B are on disk
HEARTH_ROUTING_YAML=$PWD/config/routing.finance.yaml uv run --no-sync hearth serve              # [in-process]
```

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
an error. Measured on echo without the SDK: the response had `served_by: local`,
`escalated: false`, and metrics showed `escalations_failed: 1` [in-process]. A failed remote
call may still have transmitted the prompt. Per request, `"hearth": {"allow_escalation":
false}` pins the call local. `hearth run`, `hearth agent` and all MCP tools never escalate.

**Dry-run a routing decision** [in-process]:

```sh
curl -s http://127.0.0.1:8080/v1/hearth/route -H "Authorization: Bearer $HEARTH_TOKEN" \
  -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"Summarize this diff"}]}'
# {"class":"summarize","method":"rules","backend":"local","model":"…Coder-7B…","would_escalate":false,"reason":"class policy: summarize->local",…}
```

**`hearth stats`.**

```sh
uv run --no-sync hearth stats --since 24h
```

The rows are requests, estimated frontier tokens saved, escalations, escalation rate,
escalations failed (served local), backend mix, class mix, and p50/p95 latency. **Metrics are
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
`finance_explain` and `finance_rows`. For real data, use `scripts/hearth_private.sh --profile
config/routing.finance.yaml --check` and the runbook's sealed setup.

### 9.2 Training, eval and promotion

Read [docs/RUNBOOK_training.md](RUNBOOK_training.md) for the walkthrough. Note that its
promote steps (and `scripts/train_lora_real.sh --promote`) still use the removed
`--candidate-score` flags, which now exit 2 (B-015). Use the `hearth eval … --promote` form
below. The commands:

```sh
uv run --no-sync hearth train --task classify --base mlx-community/Qwen2.5-3B-Instruct-4bit --data data.jsonl   # (not run: real training)
uv run --no-sync hearth prereg init --task classify --golden golden.jsonl --out prereg/classify.yaml
uv run --no-sync hearth prereg check prereg/classify.yaml --golden golden.jsonl
uv run --no-sync hearth eval <adapter-id> --golden golden.jsonl --prereg prereg/classify.yaml --promote          # (not run: real scoring)
uv run --no-sync hearth adapters list
uv run --no-sync hearth adapters retire <adapter-id>
```

- **Dataset JSONL** needs a header line, e.g.
  `{"kind":"hearth.dataset.header","schema_version":1,"task":"classify","version":"v1"}`,
  followed by `{"prompt","completion"}` or `{"messages":[…]}` rows. A headerless file is
  refused with `dataset task must be non-empty`.
- `train` only produces a **candidate** (`~/.hearth/train/<run-id>/`). A base model that is not
  on disk fails with the `ModelNotOnDiskError` message (exit 1).
- **Golden set JSONL** has rows of the form `{"prompt","expected"}`.

**The promotion gate** (CLAUDE.md §7). You cannot promote an adapter on a score you typed.
`hearth eval --promote` requires all of the following:

- a `--prereg` that is **git-committed and unmodified**. `prereg check` reports
  `git: not committed …` and exits 1 otherwise;
- evaluation at **temperature 0** (`--temperature > 0` is refused unless `--allow-sampling`,
  and then it cannot gate);
- an **incumbent**: the promoted adapter for the task, or the **base model** when none is
  promoted;
- **significance over paired per-example vectors** (exact McNemar or paired bootstrap) at the
  prereg's α;
- beating the **empty / majority-label / copy-input** baselines.

**n ≥ 5 is the mathematical floor** at α = 0.05, because the smallest achievable p is 0.5ⁿ.
The default `min_n = 30` is a power floor above that. The repo's golden sets are n = 5 and
n = 6, below that floor (B-010). `hearth adapters promote` requires `--report` (from
`hearth eval --report-json`) and `--prereg`, and recomputes the gate itself. Background:
[docs/LEARNING_plan.md](LEARNING_plan.md).

---

## 10. Environment variable reference

Settings come from `HEARTH_*` environment variables (`src/hearth/config.py`,
`env_prefix="HEARTH_"`, `extra="ignore"`). **A misspelled name is silently ignored.** Run
`scripts/hearth_status.py --section environment` to flag it. `hearth serve` and `hearth mcp`
read settings once at startup; restart them after a change. Booleans accept
`1/0/true/false/yes/no`.

### 10.1 Settings fields

| Variable | Default | Read by | Notes |
|---|---|---|---|
| `HEARTH_HOST` | `127.0.0.1` | `hearth serve` bind (`cli.py:252`); `doctor --offline` `bind_host` | non-loopback makes doctor UNSAFE; `--host` overrides |
| `HEARTH_PORT` | `8080` | `hearth serve` (`cli.py:253`) | `--port` overrides |
| `HEARTH_BACKEND` | `auto` | `providers/__init__.py:select_provider` (serve, run, agent, mcp, rag query, eval) | `auto` = mlx if `mlx_lm` imports, **else echo, silently** (B-006); `mlx`; `echo`; or a plugin name. An unknown value makes the command fail with a traceback (`Unknown HEARTH_BACKEND`) |
| `HEARTH_REQUIRE_AUTH` | `true` | `gateway/auth.py` | `false` disables bearer auth on `/v1/*` |
| `HEARTH_DEFAULT_MODEL` | (Settings field: `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`) | **read directly from the environment** by `registry/__init__.py:default_id` | applied only if it names a registered id, else silently ignored (B-029). The effective default without it is `config/models.yaml` `default:`. The Settings field itself is read by nothing |
| `HEARTH_EMBEDDER` | `hash` | `memory/embed.py:select_embedder` (rag, `/v1/embeddings`, agent `rag_search`) | `mlx` is broken (B-011) |
| `HEARTH_EMBED_DIM` | `256` | hash embedder (`memory/embed.py`) | vector dimension |
| `HEARTH_EMBED_MODEL` | `mlx-community/bge-small-en-v1.5-mlx` | MLX embedder; doctor | this default id is not on disk (B-011) |
| `HEARTH_VECTOR_STORE` | `sqlite` | `memory/store.py:select_vector_store` | `sqlite-vec` (`--extra vec`) or a plugin name |
| `HEARTH_RAM_CEILING_GB` | `24.0` | `serving/manager.py` via `gateway/app.py` | resident-model budget; little effect until B-004 merges |
| `HEARTH_WARMUP` | `true` | `gateway/app.py` | preload the default on `serve`; no-op on echo; today it does not load weights (B-005) |
| `HEARTH_FILE_ROOTS` | `""` (deny all) | `mcp/files.py:allowed_roots`: MCP `*_file` tools, agent `read_file`/`list_files` (CLI and HTTP), finance `read_table`, `scripts/hearth_peek.py` | colon-separated directories; `~` is expanded; non-existent entries are dropped |
| `HEARTH_FILE_MAX_BYTES` | `2000000` | `mcp/files.py` | larger files are refused, not truncated |
| `HEARTH_ALLOW_DOWNLOADS` | `false` | `providers/mlx.py:resolve_local_model`; doctor | `1` lets loads fetch missing models; makes doctor UNSAFE |
| `HEARTH_HOME` | `~/.hearth` | `config.py` (token, `models/`, `rag/`, `adapters.json`, `train/`, `finance/ledger.db`); also read directly by `handoff/store.py` and the status probes | point at a scratch directory for experiments |

### 10.2 Read directly from the environment (not Settings)

From `status/probes.py:_EXTRA_ENV_NAMES`:

| Variable | Default | Read by | Notes |
|---|---|---|---|
| `HEARTH_ROUTING_YAML` | `<repo>/config/routing.yaml` | `router/policy.py:94` (router, doctor), status probe; `scripts/hearth_private.sh` | use an absolute path: relative resolves from the current directory (B-008); `~` is not expanded by the router (B-025); a missing file falls back to all-local defaults with a warning |
| `HEARTH_MODELS_YAML` | `<repo>/config/models.yaml` | `registry/__init__.py:78` | alternate registry file |
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
| Answers start with `[echo] …`; serve banner or `/v1/hearth/admin/health` says `backend: echo` | `mlx_lm` is missing, so `HEARTH_BACKEND=auto` fell back to the echo stub **silently**, and `/ready` still says 200 (B-006). Usually a bare `uv run`/`uv sync` pruned the venv | `uv sync --extra mlx --extra mcp --extra dev --extra files`, then the verify import (§2.2). Set `HEARTH_BACKEND=mlx` so a missing MLX fails loudly instead of degrading |
| `ModuleNotFoundError: mlx_lm` / `mcp`, or `The MCP server requires the 'mcp' extra.` | the venv was pruned by a partial sync | same one-command sync |
| `model '<id>' is not on disk (looked in ~/.hearth/models and the huggingface hub cache) and HEARTH does not download on load` (`ModelNotOnDiskError`); chat returns 503 `hearth.provider.unavailable` | the weights are not in either location | `uv run --no-sync hearth models pull <id>`, then `hearth doctor --offline` shows where it resolved. Do not set `HEARTH_ALLOW_DOWNLOADS=1` to "fix" it |
| `401` `hearth.auth.unauthorized` | missing or wrong bearer token, or `/chat` has a stale token | send `Authorization: Bearer $(cat ~/.hearth/token)`; re-paste the token in `/chat`. Under a custom `HEARTH_HOME` the token is `$HEARTH_HOME/token` |
| `/v1/hearth/admin/ready` returns 503 | Today: `HEARTH_WARMUP=false` on the mlx backend leaves `/ready` at 503 `loading` permanently, even though requests are served [in-process]. ⚠ After B-005: the default is still loading, its load failed, or `HEARTH_DEFAULT_MODEL` is unregistered, and the body names the reason | today: leave `HEARTH_WARMUP` on. After B-005: wait, or read the reason and fix the model (`models list`, `models pull`) |
| `/ready` 200 but the first request fails | today `/ready` does not prove the weights loaded (B-005) | check `hearth doctor --offline`'s `model …` row |
| `hearth doctor --offline` says UNSAFE | each FAIL row names one cause: `routing_profile` (a remote profile is selected), `bind_host` (`HEARTH_HOST` not loopback), `allow_downloads` (`HEARTH_ALLOW_DOWNLOADS` on, which also fails `serving_resolution` and the `load path` rows), `model <id>` (weights not on disk) | unset the offending variable, or pull the model; re-run until `SAFE offline` |
| The model you set is not the one answering | `HEARTH_DEFAULT_MODEL` misspelled as `HEARTH_MODEL`, or set to an unregistered id (B-029); or a per-request `model`, which is ignored until B-004 merges | `uv run --no-sync hearth models list` shows the real `(default)` |
| `error while attempting to bind on address ('127.0.0.1', 8080): address already in use` *(message not reproduced in docs verification)* | another `hearth serve` (or other process) holds the port | `lsof -nP -iTCP:8080 -sTCP:LISTEN` to find it, or `hearth serve --port 8081` |
| Agent exits 2 `No readable file roots` / `Refusing to start` | `HEARTH_FILE_ROOTS` is unset or names no existing directory | `HEARTH_FILE_ROOTS=/abs/dir hearth agent …` |
| Agent exits 1 `NO ANSWER — … 'max_iterations'` | the task needs more steps than the budget, or the model wandered | narrow the task, raise `--max-iterations`, read the step table |
| My routing profile seems ignored | relative `HEARTH_ROUTING_YAML` resolved from another directory, or `~` not expanded, so it fell back to all-local defaults (B-008, B-025) | use an absolute path; `hearth doctor --offline` prints the profile it loaded |
| `hearth stats` shows all zeros | metrics are per-process, in memory | query `GET /v1/hearth/admin/metrics` on the running server |
| `HEARTH_EMBEDDER=mlx` traceback | B-011 | use `hash` |

---

## 12. Command cheat-sheet

Prefix every command with `uv run --no-sync` from the repo root.

```text
hearth version                                   print version
hearth doctor                                    environment preflight (exit 1 on a fatal FAIL)
hearth doctor --offline                          offline-safety verdict (exit 1 = UNSAFE)

hearth models list                               registry; (default) marks what serves
hearth models pull <registry-id>                 the ONLY command that downloads
hearth models rm <registry-id>                   delete from ~/.hearth/models (not the hub cache)
hearth models convert --source X --out DIR [--q-bits 4]
hearth models export-coreml --source X --out DIR

hearth serve [--host H] [--port P]               gateway + /chat on 127.0.0.1:8080
hearth run "prompt" [--file F] [--intent C] [--max-tokens N]
hearth agent "task" [--collection C] [--no-finance] [--max-iterations 8]
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

python scripts/hearth_status.py [--section environment]     measured status
python scripts/hearth_peek.py FILE                          headers + type guesses, no values
scripts/hearth_private.sh [--profile P] [--check]           sealed no-egress serve
```

Key files: `~/.hearth/token` · `~/.hearth/models/` · `~/.hearth/rag/<c>.db` ·
`~/.hearth/adapters.json` · `~/.hearth/finance/ledger.db` · `config/models.yaml` ·
`config/routing*.yaml`.

Other docs: [docs/README.md](README.md).
