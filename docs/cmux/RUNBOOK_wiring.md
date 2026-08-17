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

## 5. Live pane validation (2026-08-17) — the GUI proof

§4 measured the *equivalent code path*. This section closes the gap: a real coding agent, in a real
cmux pane, offloading to local HEARTH. Harness: **`examples/cmux/pane_offload_live.sh`**.

**Setup:** cmux 0.64.20 launched on a non-confidential empty test dir
(`examples/repos/oss_repo`, untracked) with a synthetic 180-line / 13 KB `device.log`. Pane driven
over the socket (`cmux send` / `read-screen`, `socketControlMode=password` per **ADR-C007**).

| Surface | What ran in the pane | Result |
| --- | --- | --- |
| **OpenAI** (§2) | `source sealed-pane.env` → live request to `$OPENAI_BASE_URL` | `served_by=local`, `escalated=False`, **49** est. tokens saved. Env verified in-pane: base=`http://127.0.0.1:8080/v1`, key loaded, `CMUX_CLI_SENTRY_DISABLED=1` |
| **MCP** (§1) | Claude Code `-p` with `--mcp-config`, offloading a `device.log` summary | **`VERDICT= OFFLOADED`** — `mcp__hearth__hearth_summarize` present in the agent's own `tool_use` records; coherent local summary returned |

**The assertion is transcript-based, not self-reported.** The harness parses the agent's
`--output-format stream-json` records for `mcp__hearth__*` rather than trusting the agent's prose —
an early version asked the agent to print `TOOL_USED=<yes|no>` and it answered `yes`, which is
evidence of nothing. Reproduced twice with identical verdicts.

**Locality is structural, not observed:** MCP tools run `allow_escalation=False` by construction
(`mcp/tools.py`) *and* `config/routing.private.yaml` declares `remotes: {}` with every class
`escalate: never` — the router has no egress target to choose even if asked.

### Live-only findings (3)

1. **The `mcp` extra is required and fails silently.** Without it, `hearth mcp` exits with
   `The MCP server requires the 'mcp' extra.`; Claude Code then drops the server with no error and
   the agent reports "no hearth tools are registered" — which reads like a bad `--mcp-config` path.
   Fixed in `examples/cmux/hearth.mcp.json` (prereq documented) + the harness's failure-mode probe.
2. **`uv sync --extra mcp` alone PRUNES the other extras** — it removed `mlx-lm`, `transformers`,
   `torch`, `pytest`, and `ruff`, breaking the MLX backend. Always sync extras together:
   `uv sync --extra mlx --extra mcp --extra dev`. (Suite re-verified after: **248 passed, 1 skipped**.)
3. **`hearth_summarize`'s `max_words` is a soft hint** — a `max_words: 25` call returned ~40 words.
   Fine for offload, but don't rely on it for width-constrained UI (e.g. a cockpit status line).

**Honest caveat:** the MCP tools take `text`, not a path, so a Claude Code pane must `Read` the file
into its own context before handing it over. That caps the savings for the "pre-digest a large file"
pattern — the offload saves the *generation* and downstream reasoning, not the initial read. A
path-taking tool variant would close this; noted as a HEARTH-side improvement, not a cmux blocker.

## 6. C2 status & next

- ✅ MCP + OpenAI wiring artifacts (`examples/cmux/`), validated on real hardware.
- ✅ Measured savings via the in-process demo (mirrors the MCP path).
- ✅ **Live GUI proof done 2026-08-17** (§5) — both surfaces exercised from a real pane.
- ⏳ Egress half still pending: re-run §5 under `cmux-sealed` + probe to prove loopback-only.
  Blocked on the parked firewall hardening (TODO.md), not on C2.
- **Next:** C3 `cmux/sealed-profile` — the `cmux-sealed` launcher + fail-closed preflight that turns
  this wiring into an enforced sealed tier (signed-out, telemetry/Sparkle off, OS egress containment,
  reads `config/cmux/tiers.yaml`).
