# HEARTH documentation index

**Start with [GUIDE.md](GUIDE.md)**, §0 "Learn HEARTH in 15 minutes". It covers using HEARTH
from install to training. The same reference is built into the tool: `hearth --help`,
`hearth COMMAND --help`, and the generated man page `man ./man/hearth.1`.

Status key:

- **current**: matches the code as of 2026-10-05 (`d628c4f`, model selection merged).
- **partly stale**: mostly right, with known wrong spots (named).
- **design/reference**: background and rationale, not instructions.
- **historical**: a record of a past state. Do not follow its commands.

When a doc and the code disagree, trust `uv run --no-sync python scripts/hearth_status.py`
and `hearth doctor --offline`. They measure; docs describe.

## Using HEARTH

| Doc | What it is for | Status |
|---|---|---|
| [GUIDE.md](GUIDE.md) | Start-to-finish user guide: install, models, serve, `/chat`, API, run, agent, RAG, MCP, routing, finance, training, env-var reference, troubleshooting, cheat-sheet | current. Areas another branch is changing right now carry a "⚠ changing (B-0xx)" note (B-003, B-031, B-033, B-034, B-035, B-036, B-047, B-048) |
| [PRIVACY.md](PRIVACY.md) | The privacy model: what stays local, the two egress vectors, disk-only loads, the caller caveat, sealed mode, verifying no egress | current |
| [AGENT.md](AGENT.md) | The local agent loop: tools, bounds, exit codes, security model, `hearth agent`, `POST /v1/hearth/agent`, `/chat` agent mode | current |
| [API.md](API.md) | HTTP contract: OpenAI-compatible endpoints, `hearth` extension fields, `/chat`, streaming | partly stale (B-037): the model-selection 404, `/ready` reasons and `GET /admin/models` are current, but it still describes `/v1/hearth/classify`, `/summarize`, `/train/*` and `/admin/adapters/*` routes that do not exist, and the 401 body is nested under `detail` |
| [RUNBOOK_finance.md](RUNBOOK_finance.md) | Your bank statements on this machine: staging, mappings, sealed run, ingest, reconcile, categorize, aggregate, audit | current |
| [RUNBOOK_training.md](RUNBOOK_training.md) | Real-weights LoRA validation: dataset, train, eval, promote, serve | current (promotion via `prereg` + `eval --prereg --promote`, B-041). `scripts/train_lora_real.sh --promote` still uses removed flags (B-015) |
| [RUNBOOK_consumer_wiring.md](RUNBOOK_consumer_wiring.md) | Wiring CAMBOT (HTTP) and Claude Code (MCP) to a live HEARTH and reading token savings | current |
| [INTEGRATION.md](INTEGRATION.md) | How any client consumes HEARTH (CAMBOT, Claude Code MCP, OpenAI SDKs, shell) | current in substance. Its `.mcp.json` uses a bare `hearth` command, so use the venv's absolute path (GUIDE §7) |
| [STATUS.md](STATUS.md) | What `scripts/hearth_status.py` measures and what it cannot see | current |
| [BUGS.md](BUGS.md) | Known defects and gaps, prioritized, with file:line evidence and acceptance tests | current (the backlog) |
| [RESUME.md](RESUME.md) | Handoff for picking up HEARTH development mid-stream | current as of 2026-10-06 |
| [YOUR_TURN.md](YOUR_TURN.md) | What only the operator can do, with exact commands | current as of 2026-10-06 |

## Design and reference

| Doc | What it is for | Status |
|---|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Target system design: layers, interfaces, backends, data flow | design/reference (draft; some parts aspirational) |
| [DECISIONS.md](DECISIONS.md) | Architecture Decision Records: the why behind big choices | design/reference |
| [TIERS.md](TIERS.md) | The four-tier ladder; why tiers 3–4 (PCC / frontier) are not backends; the handoff mechanism | design/reference (current policy) |
| [PLUGINS.md](PLUGINS.md) | Writing third-party providers, vector stores and embedders via entry points | design/reference (current) |
| [MODELS_local.md](MODELS_local.md) | Local model roster for this M3 Pro, with measured sizes | design/reference (researched 2026-09-02) |
| [LEARNING_plan.md](LEARNING_plan.md) | Audit of training/eval and the improvement plan | design/reference. Says "nothing implemented", but the prereg/significance gate has since shipped. It quotes the removed `--candidate-score` path as the problem it fixes |
| [APEX_seam.md](APEX_seam.md) | Boundary analysis between HEARTH (model layer) and APEX (finance domain) | design/reference (2026-09-02) |

## Historical

| Doc | What it is for | Status |
|---|---|---|
| [PROPOSAL.md](PROPOSAL.md) | The original pitch: problem, goals, non-goals | historical (pre-implementation) |
| [ROADMAP.md](ROADMAP.md) | Phase 0–7 build plan and result log | historical (test counts and "status" are from the phase build) |
| [RESULTS.md](RESULTS.md) | Real-hardware validation, 2026-07-10: LoRA run, consumer token savings | historical. Its promotion used typed scores; that path is now removed, and `tests/test_eval_gate_replay.py` asserts that promotion is refused |
| [HANDOFF.md](HANDOFF.md) | Brief for a real-hardware Claude Code to run the hardware-blocked follow-ups | historical. Uses bare `uv run` (prunes the venv) and removed promote flags |

## Elsewhere in the repo

| Path | What it is for | Status |
|---|---|---|
| [`../man/hearth.1`](../man/hearth.1) | The reference manual, generated from the CLI by `scripts/gen_manpage.py` (`man ./man/hearth.1`) | current (a test fails when it drifts from the CLI) |
| [`../README.md`](../README.md) | Project overview and quick start | current quick start; the "Status" test counts are historical |
| [`../CLAUDE.md`](../CLAUDE.md) | Working notes and rules for agents editing the repo | current |
| [`cmux/README.md`](cmux/README.md) | Master map of the HEARTH × cmux integration effort (its own docs set) | current for that effort |
| [`../examples/claude_code_mcp.md`](../examples/claude_code_mcp.md) | Registering `hearth mcp` with Claude Code | current (one-command sync, `uv run --no-sync --project`, B-040) |
| [`../examples/finance/README.md`](../examples/finance/README.md) | The two-tier finance ladder, measured on synthetic data | design/reference |
| [`../swift/README.md`](../swift/README.md), [`../swift/OFFLINE.md`](../swift/OFFLINE.md) | Swift SDK and the offline Core ML path | design/reference |
