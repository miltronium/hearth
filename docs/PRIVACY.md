# HEARTH — Privacy & confidential-work model

HEARTH is designed to let an agent offload work to a **local** model so sensitive content
never leaves the machine. This document states exactly what that does and does **not**
guarantee, how to run HEARTH in a sealed no-egress mode, and how to verify it — grounded in
the code, not aspiration.

> **One-line summary.** In private mode HEARTH is a sealed local box: inference, embeddings,
> RAG, and metrics all stay on-device, and the router has **no** path to send a task off the
> machine. The remaining responsibility is the *calling agent* — see § "The caller caveat".
>
> **The default is no-egress too.** The shipped `config/routing.yaml` (used whenever
> `HEARTH_ROUTING_YAML` is unset) has zero remotes, every class `local`/`never`, and a zero
> remote budget. Escalation to a frontier model is opt-in — see § "Opting in to remote
> escalation". Private mode adds the loopback/backend pins and the fail-closed check on top.

---

## What stays local (verified in the code)

| Concern | Behavior | Where |
| --- | --- | --- |
| **Inference** | Runs on-device via MLX; no network at generate time. | `providers/mlx.py` |
| **Embeddings / RAG** | Local embedder + local SQLite store; no network. | `memory/embed.py`, `memory/store.py` |
| **Observability** | `RequestRecord` stores **token counts + metadata only — no prompt/response text**, in an in-memory ring (10k). Nothing content-bearing is persisted. | `observability/metrics.py` |
| **Gateway bind** | Loopback `127.0.0.1` by default; never off-box unless you set `HEARTH_HOST`. | `cli.py`, `gateway/app.py` |
| **Logs** | Router/provider log **metadata** (task class, model, latency), not prompt/response content. | `router/route.py` |

## The only ways data can leave the machine — and how private mode closes them

There are exactly **two** egress vectors in the entire codebase:

1. **Escalation to a configured remote** (`providers/remote.py`) — if routing sends a class to a
   remote (Anthropic or an OpenAI-compatible endpoint), that task's content is sent there.
   → **The default profile (`config/routing.yaml`) and private mode
   (`config/routing.private.yaml`) both define zero remotes and make every class
   `local`/`never`**, so the router has nowhere to send a task. Only a profile you select
   deliberately — `config/routing.remote.yaml` — permits this vector. The MCP tools run with
   `allow_escalation=False` regardless (`mcp/tools.py`), so agent offload is local under any
   profile; the profile is what governs the HTTP/CAMBOT/`/chat` path.
2. **Model-weight download** from HuggingFace — weights, *not your data*. → The serving load
   path no longer downloads by default: `providers/mlx.py:resolve_local_model` resolves a model
   only from disk, and so do `hearth train`, `hearth models convert` and `export-coreml` (see
   § "Model loading: what is disk-only and what is not"). Private mode also sets
   `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`; no HEARTH load path depends on them any more,
   but they still cover code that calls `huggingface_hub` directly. Pre-cache once
   (`hearth models pull …`) from an unrestricted terminal. `hearth doctor --offline` measures
   all of this in one command.

No analytics, telemetry, or phone-home exists anywhere else.

### Opting in to remote escalation

The escalating profile — `reason` → remote `always`; `draft`/`code`/`chat` escalate
`on_low_confidence` (a prompt-*length* stub today, so short messages escalate); remote = Claude
`claude-opus-4-8` via the `anthropic` SDK; 200k tokens/day — lives in
`config/routing.remote.yaml`. Until 2026-10 it was the default; it is now selected explicitly:

```sh
HEARTH_ROUTING_YAML=config/routing.remote.yaml hearth serve   # relative paths resolve from CWD
```

Only use it for work that may leave the machine. If an escalation's remote call fails
(unreachable, SDK missing, rejected), `Router.route` degrades to the local provider
(`Router.degrade_to_local`) and records `served_by=local`, `escalated=false`; the streaming
path does the same if the remote fails before any text. The failure is recorded, not just
logged: the request's `RequestRecord.escalation_failed` holds the remote's error, and the
rollup (`/v1/hearth/admin/metrics`, `hearth stats`) counts `escalations_failed`, so a remote
outage is visible rather than reading as a policy that never escalated. A stream that fails mid-answer, or a local provider that fails, ends with
an error event (`hearth.provider.unavailable`) and then `[DONE]`. Note what the degrade does
**not** mean: a remote call that failed may still have transmitted the prompt before failing.
"Served local" describes the answer, not the egress.

### Model loading: what is disk-only and what is not

`providers/mlx.py:resolve_local_model` resolves a model id from disk only: an existing path
as-is, then `~/.hearth/models`, then the huggingface hub cache (`HF_HUB_CACHE` → `HF_HOME/hub`
→ `~/.cache/huggingface/hub`), both cache lookups `local_files_only`. A model in neither place
raises `ModelNotOnDiskError` instead of downloading, unless `HEARTH_ALLOW_DOWNLOADS=1`
(`settings.allow_downloads`, default off). Fetching is the explicit act `hearth models pull`.

| Path | Disk-only without `HF_HUB_OFFLINE`? |
| --- | --- |
| `MLXProvider` — `hearth serve`, `/v1/chat/completions`, `/chat`, `hearth run`, `hearth agent`, `hearth eval`, MCP tools | **Yes** (resolver) |
| `MLXEmbedder` (`HEARTH_EMBEDDER=mlx`) | **Yes** (same resolver) |
| `hearth train` — `mlx_lm.lora` in a child process | **Yes** — `--model` is the resolved path; the child runs with `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` (`training/lora.py:runner_invocation`) |
| `hearth models convert` — `mlx_lm convert` in a child process | **Yes** — `--hf-path` is the resolved path; same offline child env (`convert.py:convert_invocation`) |
| `hearth models export-coreml` — `transformers` in-process | **Yes** — resolved path + `local_files_only=True` on every `from_pretrained` (`coreml.py:load_hf_source`) |
| `scripts/bench.py`, `scripts/eval_candidate.py`, `scripts/hearth_map_draft.py` | **Yes** (same resolver, via `mlx_lm.load` / `MLXProvider`) |
| `scripts/coreml_stateful_reference.py` | **Yes** — sets the offline vars for itself and passes `local_files_only=True`; reads the hub cache only, not `~/.hearth/models` |
| `hearth models pull` | Downloads by design — the only path that does |

Every path except `pull` honours the same opt-in, `HEARTH_ALLOW_DOWNLOADS=1`: with it set, a
model that is not on disk is handed on by id and the child is not pinned offline. The path
matters beyond the weights: handed a repo id, `mlx_lm`'s `save()` also fetches the source's
model card from the hub, so the old `convert` reached huggingface.co even with the weights
cached (measured: 28 connects vs 0). `tests/test_offline_load_paths.py` asserts each path's
outcome with every connect refused and counted, plus a census that `models pull` is the only
non-`local_files_only` hub fetch in `src/` and `scripts/`.

`hearth doctor --offline` measures all of it and exits non-zero when anything is unsafe: the
routing profile the router would load now (and which classes can escape it), serving
resolution under a connect-counting audit, every reachable model (default, each class's
`local_model`, the MLX embedder) resolving to weights on disk, each load path above handing
its tool a local path with the hub pinned offline, a loopback bind host, and
`HEARTH_ALLOW_DOWNLOADS` off. It measures this command's environment, not a running daemon's.

This is a guarantee about HEARTH's **loader and router**, not machine-level containment.
Nothing here inspects a firewall or a socket, and it says nothing about other processes,
other libraries, or code that calls `huggingface_hub` / `providers/remote.py` directly. For
that, measure (§ "Verifying no egress yourself", step 3).

## The caller caveat (read this)

**HEARTH protects the subtask that runs *on* HEARTH — it does not protect the calling agent's
own context.** If an agent reads a confidential file into its context and *then* calls
`hearth_summarize`, that file content already went wherever that agent runs *before* HEARTH saw
it. HEARTH cannot un-send it.

Practical rule: choose the *agent* by the data's sensitivity. Use an agent with an approved data
path for the confidential repo; let it offload subtasks to HEARTH's local model so those
subtasks stay fully on-device (with escalation off). HEARTH is the on-device sink, not a shield
in front of a frontier agent.

### Path-taking tools — closing the caveat for files

For the common case where the sensitive thing *is a file*, the MCP server also exposes
**path-taking** tools — `hearth_summarize_file`, `hearth_classify_file`, `hearth_extract_file`
(`mcp/files.py`, `mcp/tools.py`). The agent passes a **path**; HEARTH opens the file itself,
locally, and only the task's result (a summary, a label, the requested fields) crosses back. The
agent never holds a byte of the content, so nothing to un-send.

That makes them an arbitrary-file-read primitive in an agent's hands, so they are gated:

| Control | Behavior |
| --- | --- |
| **Allowlist, deny by default** | `HEARTH_FILE_ROOTS` (colon-separated dirs) is the *only* way to enable reads. Unset ⇒ every read refused. No implicit root — not CWD, not `$HOME`. |
| **Full resolution first** | The path is `expanduser()`-ed and `resolve()`-d (flattening `..`, following symlinks) and the **result** must be inside a resolved root — so a traversal or a symlink planted inside a root escapes nothing. |
| **Type + size** | Regular files only (directories, devices, FIFOs refused); over `HEARTH_FILE_MAX_BYTES` (default 2 MB) is refused, not truncated. |
| **Formats** | Plain text (`.txt`, `.md`, `.log`, `.rst`, extension-less) and `.csv` today; anything else is refused by name. PDF/XLSX/JSON are one handler + one table entry away (`mcp/files.py`). |
| **Errors** | Refusals name the path and the reason and **never quote file content** — the error travels back to the agent we're keeping the content from. |

```sh
# Enable for one directory, then let the agent summarize without ever reading it:
export HEARTH_FILE_ROOTS="$HOME/statements"
#   agent: hearth_summarize_file(path="~/statements/aug.csv")
```

Note this closes the *file* half of the caveat only. An agent that pastes confidential text it
already read into `hearth_summarize` is still leaking — the path-taking tool is what you point it
at instead.

## Data at rest

- **RAG index** — `hearth rag ingest` writes the **raw chunk text** to
  `~/.hearth/rag/<collection>.db`. That is a real copy of your source on disk. Keep `~/.hearth`
  on an encrypted volume (FileVault). Purge a collection with `rm ~/.hearth/rag/<collection>.db`.
- **Adapters / training runs** — `~/.hearth/adapters.json` and `~/.hearth/train/<id>/` hold
  adapter weights and the train/valid split (which contains your training text). Same handling.
- **Bearer token** — `~/.hearth/token` (mode 0600) gates the HTTP API.

## Running HEARTH in sealed private mode

```sh
scripts/hearth_private.sh --check     # verify the no-egress posture only (exit 0 if sealed)
scripts/hearth_private.sh             # verify, then serve on 127.0.0.1:8080
```

The script defaults `HEARTH_ROUTING_YAML=config/routing.private.yaml`, forces `HEARTH_HOST=127.0.0.1`,
`HEARTH_BACKEND=mlx`, and offline HF, and **fails closed** (won't start) if the routing policy
resolves any remote or any escapable class. It honours an inherited `HEARTH_ROUTING_YAML` or
`--profile PATH` and re-runs the same assertions against that profile — so pointing it at
`config/routing.remote.yaml` refuses to start. The default profile is no-egress already; what
the script adds is that the posture is *verified* before serving rather than assumed.

## Verifying no egress yourself

```sh
# 0. One command, every HEARTH-side path measured (exit 1 when unsafe):
uv run --no-sync hearth doctor --offline

# 1. Posture check (no remotes, all classes local/never):
scripts/hearth_private.sh --check

# 2. Confirm a would-escalate class stays local under this profile:
HEARTH_ROUTING_YAML=config/routing.private.yaml uv run --no-sync python -c "
from hearth.router.policy import load_policy; p=load_policy()
print('remotes:', p.remotes, '| reason ->', p.classes['reason'])"
#   -> remotes: {} | reason -> ClassRule(backend='local', escalate='never')

# 3. Optional: watch the box make no outbound connections while you drive a workload
#    (loopback :8080 is HEARTH itself):
#    lsof -nP -iTCP -a -p "$(pgrep -f 'hearth serve')" -sTCP:ESTABLISHED
```

## Checkpoint / returning later

State that matters for resuming this work lives in three durable, **local** places (kept local
on purpose — routing confidential-project notes through a cloud service would itself be egress):

- **Git** — this scaffolding (`config/routing.private.yaml`, `scripts/hearth_private.sh`, this
  doc). Tagged as a checkpoint.
- **`~/.hearth/`** — runtime state (RAG collections, adapters) on your machine.
- **The agent's local memory** — project context (which repos are confidential, the caller-agent
  decision, private-mode invariants) is recorded so a future session recalls the constraints.

If you return to this: run `scripts/hearth_private.sh --check` first to confirm the box is still
sealed, then re-read § "The caller caveat" before pointing any agent at a confidential repo.
