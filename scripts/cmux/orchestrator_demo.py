#!/usr/bin/env python3
"""C4 demo — run the cmux×HEARTH orchestrator's triage on realistic pane transcripts, locally.

Seeds a FakeCmuxClient with four panes in different states, runs the SAME triage/run_once path the
live orchestrator uses, but against on-device HEARTH (MLX). Shows which panes the local model flags
for attention and the notifications it would fire — no cmux GUI and no frontier tokens required.

    HEARTH_BACKEND=mlx uv run python scripts/cmux/orchestrator_demo.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location("orchestrator", Path(__file__).with_name("orchestrator.py"))
orch = importlib.util.module_from_spec(_spec)
sys.modules["orchestrator"] = orch  # dataclasses + `from __future__ import annotations` need this
_spec.loader.exec_module(orch)

PANES = {
    "surface:1": "==> Building target HearthCoreML (3/7)\nCompiling CoreMLGeneration.swift\nCompiling CoreMLProvider.swift\n",
    "surface:2": "Delete adapters.json and retire adapter qwen-lora-3? This cannot be undone. [y/N] ",
    "surface:3": "Test Suite 'All tests' passed at 2026-07-21 18:42.\n\t Executed 238 tests, with 0 failures.\n$ ",
    "surface:4": "error: cannot find 'writePos' in scope\n  let mask = causalMask(writePos)\n                        ^~~~~~~~\nBuild failed (exit 65)\n$ ",
}
TITLES = {"surface:1": "build", "surface:2": "adapters", "surface:3": "tests", "surface:4": "coreml-fix"}


def main() -> int:
    client = orch.FakeCmuxClient(PANES, titles=TITLES)
    classify, summarize = orch.hearth_callables()

    print("== orchestrator triage sweep (4 panes, on-device HEARTH, escalation off) ==\n")
    results = orch.run_once(client, classify, summarize, lines=80, do_notify=True)

    for t in results:
        flag = "🔔 NOTIFY" if t.notified else "·  quiet "
        print(f"{flag}  [{t.state:>7}/{t.priority:>9}]  {t.title:<10}  {t.message}")

    print(f"\n== {len(client.notifications)} notifications fired ==")
    for sid, title, body in client.notifications:
        print(f"  → ({sid}) {title}: {body}")

    flagged = sum(1 for t in results if t.notified)
    print(f"\n{flagged}/{len(results)} panes flagged for attention by the local model — 0 frontier tokens.")
    print("Expected: 'build' quiet (working); 'adapters' + 'coreml-fix' + 'tests' flagged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
