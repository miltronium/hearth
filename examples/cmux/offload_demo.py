#!/usr/bin/env python3
"""cmux × HEARTH offload demo — the measurable proof for C2 (docs/cmux/RUNBOOK_wiring.md).

A cmux pane runs a coding agent (Claude Code / codex / opencode). When that agent is wired to
HEARTH (MCP or OpenAI base_url), its routine subtasks — summarize-before-read, classify, extract,
draft — run on the LOCAL model instead of spending frontier tokens. This script drives the *exact
same code path* an MCP-wired pane triggers (``build_toolset`` -> Router, ``allow_escalation=False``)
and then prints HEARTH's own rollup, so the token savings are measured, not asserted.

Why in-process: ``hearth stats`` reads an in-memory, per-process metrics ring (cli.py:218), so a
separate CLI can't see a running daemon's numbers. Running the offloads and reading the rollup in
ONE process is the honest way to measure until metrics are persisted (a future HEARTH phase).

Run (real local inference):
    HEARTH_BACKEND=mlx uv run python examples/cmux/offload_demo.py

With no MLX/model it falls back to the echo backend (savings still tallied, text is stubbed).
"""

from __future__ import annotations

from hearth.mcp.tools import build_toolset
from hearth.observability import get_metrics

# Representative subtasks a cmux pane's agent would offload instead of burning frontier tokens.
DEVICE_LOG = (
    "2026-07-21T18:03:11Z WARN thermal: SoC 92C throttling; 2026-07-21T18:03:12Z INFO gpu: clock "
    "1.1GHz->0.8GHz; 2026-07-21T18:03:19Z ERROR ane: mmap SIGBUS at 0x0 during fp16 predict; "
    "2026-07-21T18:03:19Z INFO fallback: switching compute_units cpuAndNeuralEngine->cpu"
)
COMMIT_DIFF = (
    "wired cmux panes to HEARTH: added examples/cmux/{hearth.mcp.json, sealed-pane.env}, a wiring "
    "runbook, and an in-process offload demo that reports estimated frontier tokens saved."
)


def main() -> int:
    tools = build_toolset()  # same Router + RAG the MCP server and gateway use

    print("== running cmux-pane offload subtasks locally (escalation off) ==\n")

    summary = tools.summarize(DEVICE_LOG, max_words=25)
    print(f"[summarize] {summary}\n")

    intent = tools.classify("restart the thermal daemon on device 7", labels=["query", "action", "config"])
    print(f"[classify]  -> {intent}\n")

    fields = tools.extract(DEVICE_LOG, fields=["error", "component", "fallback"])
    print(f"[extract]   -> {fields}\n")

    commit = tools.draft("Write a one-line conventional-commit message for this diff.", context=COMMIT_DIFF)
    print(f"[draft]     {commit}\n")

    roll = get_metrics().rollup()
    print("== HEARTH rollup (this process) ==")
    print(f"  requests ............... {roll['requests']}")
    print(f"  frontier tokens saved .. {roll['estimated_frontier_tokens_saved']}")
    print(f"  escalations ............ {roll['escalations']} ({roll['escalation_rate']:.0%})")
    print(f"  backend mix ............ {roll['backend_mix']}")
    print(f"  class mix .............. {roll['class_mix']}")
    print(f"  latency p50/p95 (ms) ... {roll['latency_ms']['p50']:g} / {roll['latency_ms']['p95']:g}")
    # A cmux session runs many such subtasks across many panes; this is the per-run slice.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
