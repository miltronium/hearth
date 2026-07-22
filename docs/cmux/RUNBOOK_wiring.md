# HEARTH × cmux — Wiring runbook (C2)

**Phase:** C2 · **Branch:** `cmux/wiring`. How a cmux pane's coding agent offloads routine subtasks
to a **local, sealed** HEARTH — the config that makes the token-savings the default path. This is
**configure-only** (ADR-C004): no change to HEARTH's code. Full sealed *enforcement* (signed-out +
OS-level egress containment) is the C3 launcher; this runbook is the offload wiring + the measured proof.

> **Validated on real hardware (Apple M3 Pro, MLX, Qwen2.5-Coder-4bit), 2026-07-21.** See §4.

---

## The model

A cmux pane runs a coding agent. Wire that agent to HEARTH by one of two surfaces; HEARTH does the
volume (bulk reads/summaries/classification), the agent keeps the reasoning:

| Agent in the pane | Wire via | Artifact |
| --- | --- | --- |
| Claude Code (MCP) | HEARTH MCP server | `examples/cmux/hearth.mcp.json` |
| codex / opencode / any OpenAI SDK | `OPENAI_BASE_URL` → local gateway | `examples/cmux/sealed-pane.env` |

Both keep work local: the MCP tools run `allow_escalation=False` by construction (`mcp/tools.py`),
and the sealed serve uses `routing.private.yaml` (0 remotes). Nothing offloaded here leaves the box.

---

## 1. MCP wiring (Claude Code panes)

Copy `examples/cmux/hearth.mcp.json` into the pane's Claude Code MCP config, editing
`HEARTH_ROUTING_YAML` to the **absolute** path of `config/routing.private.yaml` (the MCP server's
cwd isn't guaranteed to be the repo). Claude then has these local tools:

`hearth_summarize` · `hearth_classify` · `hearth_extract` · `hearth_draft` · `hearth_rag_query`

Workflow pattern: let Claude *orchestrate*; have it call `hearth_summarize` to pre-digest large files
**before** reading them into its own context, and `hearth_classify`/`hearth_extract` for routing/labeling
— each one a frontier round-trip avoided.

> If `hearth` isn't on PATH, use `"command": "uv", "args": ["run", "hearth", "mcp"]` with the repo as cwd.

## 2. OpenAI wiring (codex / opencode / OpenAI-SDK panes)

Start a sealed gateway once, then source the env in each pane:

```sh
# start the sealed local gateway (loopback, no remotes, offline weights)
HEARTH_ROUTING_YAML=config/routing.private.yaml HEARTH_HOST=127.0.0.1 HEARTH_BACKEND=mlx \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run hearth serve      # http://127.0.0.1:8080

# in each cmux pane, before launching the agent:
source examples/cmux/sealed-pane.env    # sets OPENAI_BASE_URL/KEY + cmux telemetry-off env
```

The agent's existing OpenAI client now transparently hits the local model. Verified request/response:

```sh
curl -sf http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $(cat ~/.hearth/token)" -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"Classify into exactly one of [query, action, config]. Reply with only the label. Text: restart the thermal daemon on device 7"}],
       "extra_body":{"hearth":{"intent":"classify","allow_escalation":false}}}'
#  -> "action"   (served by mlx-community/Qwen2.5-Coder-7B-Instruct-4bit, local)
```

---

## 3. Reading the savings (and an honest caveat)

`hearth stats` reads an **in-memory, per-process** metrics ring (`cli.py:218`). A running
`hearth serve` **does** accumulate metrics — but a *separate* `hearth stats` process can't see them
(they aren't persisted across processes yet). So today, measure with the in-process demo:

```sh
HEARTH_BACKEND=mlx uv run python examples/cmux/offload_demo.py
```

It drives the **same** `build_toolset` → Router path an MCP-wired pane uses, then prints the rollup —
so the number is measured, not asserted. (A future HEARTH phase persists records to JSONL; then a
daemon-wide `hearth stats` becomes possible and is the natural cmux-cockpit "tokens saved" readout.)

---

## 4. Validation results (2026-07-21, Apple M3 Pro)

`examples/cmux/offload_demo.py` ran 4 representative cmux-pane subtasks through **sealed local** HEARTH:

| Subtask | Result | Backend |
| --- | --- | --- |
| `summarize` (device log → 25 words) | coherent local summary | local (Qwen2.5-Coder) |
| `classify` ("restart thermal daemon" → query/action/config) | **action** ✓ | local |
| `extract` (error/component/fallback from a log) | all 3 fields correct ✓ | local |
| `draft` (conventional-commit from a diff) | valid commit line | local |

**Rollup (this run):** `requests=4 · estimated frontier tokens saved=1053 · escalations=0 (0%) ·
backend mix={local:4}`. HTTP path (§2) independently verified against a live sealed serve.

**Interpretation:** a single pane offloading four subtasks avoided ~1k frontier tokens. A cmux
session runs many such subtasks across many panes — that multiplies. This is the C2 gate met:
a pane-equivalent offload to sealed HEARTH, escalation off, with measured savings, config-only.

---

## 5. C2 status & next

- ✅ MCP + OpenAI wiring artifacts (`examples/cmux/`), validated on real hardware.
- ✅ Measured savings via the in-process demo (mirrors the MCP path).
- ⏳ Live *GUI* proof (a real cmux pane, not the equivalent code path) folds into the C3/C6
  on-hardware runs alongside the AUDIT §9 egress probe.
- **Next:** C3 `cmux/sealed-profile` — the `cmux-sealed` launcher + fail-closed preflight that turns
  this wiring into an enforced sealed tier (signed-out, telemetry/Sparkle off, OS egress containment,
  reads `config/cmux/tiers.yaml`).
