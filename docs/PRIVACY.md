# HEARTH — Privacy & confidential-work model

HEARTH is designed to let an agent offload work to a **local** model so sensitive content
never leaves the machine. This document states exactly what that does and does **not**
guarantee, how to run HEARTH in a sealed no-egress mode, and how to verify it — grounded in
the code, not aspiration.

> **One-line summary.** In private mode HEARTH is a sealed local box: inference, embeddings,
> RAG, and metrics all stay on-device, and the router has **no** path to send a task off the
> machine. The remaining responsibility is the *calling agent* — see § "The caller caveat".

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
   → **Private mode removes every remote and makes every class `local`/`never`** so the router
   has nowhere to send a task (`config/routing.private.yaml`). The MCP tools already run with
   `allow_escalation=False` regardless (`mcp/tools.py`), so agent offload is local even without
   this profile — the profile also seals the HTTP/CAMBOT path.
2. **Model-weight download** from HuggingFace — weights, *not your data*. → Private mode sets
   `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` so it loads only cached weights and never hits
   the network. Pre-cache once (`hearth models pull …`) from an unrestricted terminal.

No analytics, telemetry, or phone-home exists anywhere else.

## The caller caveat (read this)

**HEARTH protects the subtask that runs *on* HEARTH — it does not protect the calling agent's
own context.** If an agent reads a confidential file into its context and *then* calls
`hearth_summarize`, that file content already went wherever that agent runs *before* HEARTH saw
it. HEARTH cannot un-send it.

Practical rule: choose the *agent* by the data's sensitivity. Use an agent with an approved data
path for the confidential repo; let it offload subtasks to HEARTH's local model so those
subtasks stay fully on-device (with escalation off). HEARTH is the on-device sink, not a shield
in front of a frontier agent.

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

The script forces `HEARTH_ROUTING_YAML=config/routing.private.yaml`, `HEARTH_HOST=127.0.0.1`,
`HEARTH_BACKEND=mlx`, and offline HF, and **fails closed** (won't start) if the routing policy
resolves any remote or any escapable class.

## Verifying no egress yourself

```sh
# 1. Posture check (no remotes, all classes local/never):
scripts/hearth_private.sh --check

# 2. Confirm a would-escalate class stays local under this profile:
HEARTH_ROUTING_YAML=config/routing.private.yaml uv run python -c "
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
