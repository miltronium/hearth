#!/usr/bin/env python3
"""cambot_offload.py — a CAMBOT-style consumer offloading work to a running HEARTH gateway.

Demonstrates the REAL ``hearth.client.HearthClient`` API (Phase 5, G2/G8): a consumer
delegates a summarize/extract subtask to the local model over HTTP instead of spending
frontier tokens on it. The delegated call runs with ``allow_escalation=False`` so HEARTH
never makes a surprise remote call on the consumer's behalf.

------------------------------------------------------------------------------------------
Pointing this at a LIVE daemon
------------------------------------------------------------------------------------------
1. Start HEARTH in another terminal:

       uv run hearth serve                       # binds 127.0.0.1:8080 by default

   On first start HEARTH writes a bearer token to ``~/.hearth/token`` (0600). Read it:

       export HEARTH_TOKEN="$(cat ~/.hearth/token)"
       export HEARTH_URL="http://127.0.0.1:8080"

2. Run this example against the live daemon:

       uv run python examples/cambot_offload.py --live

   (Without ``--live`` it only prints what it *would* send — safe to run offline/in CI.)

------------------------------------------------------------------------------------------
Measuring token savings afterwards (G2/G8)
------------------------------------------------------------------------------------------
Every offloaded task is a task that did NOT hit a frontier model. HEARTH's observability
layer estimates the frontier tokens it saved. Read it two ways:

  * CLI rollup (per-process; reflects the running daemon):

        uv run hearth stats --since 24h

  * Admin metrics endpoint (same numbers, JSON; needs the bearer token):

        curl -s -H "Authorization: Bearer $HEARTH_TOKEN" \
             "$HEARTH_URL/v1/hearth/admin/metrics?since=24h" | python -m json.tool

  Look at ``estimated_frontier_tokens_saved`` and ``escalation_rate``. See
  docs/RUNBOOK_consumer_wiring.md for the end-to-end wiring + how to read the numbers.
"""

from __future__ import annotations

import argparse
import os

# A stand-in for the kind of text CAMBOT would hand off — e.g. a captured log or a diff.
SAMPLE_TEXT = (
    "The nightly build finished in 12m4s. Unit tests: 1,204 passed, 0 failed. "
    "The linter flagged 3 style warnings in the payments module, all auto-fixable. "
    "Disk usage on the CI runner peaked at 78%. No regressions were detected."
)


def run_offload(base_url: str, token: str | None) -> None:
    """Offload a summarize + classify subtask to a live HEARTH gateway and print results.

    Imported lazily so ``--help`` and a dry run work with no network and no live daemon.
    """
    from hearth.client import HearthClient

    with HearthClient(base_url, token=token) as hearth:
        # Both calls are hard-local (allow_escalation=False inside these helpers): the
        # consumer's frontier budget is untouched.
        summary = hearth.summarize(SAMPLE_TEXT, max_words=25)
        label = hearth.classify(SAMPLE_TEXT, labels=["healthy", "degraded", "failing"])

    print("SUMMARY:", summary)
    print("STATUS :", label)
    print(
        "\nOffloaded 2 subtasks locally (0 frontier tokens). "
        "Run `hearth stats` to see estimated_frontier_tokens_saved."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--url",
        default=os.environ.get("HEARTH_URL", "http://127.0.0.1:8080"),
        help="HEARTH base URL (default: $HEARTH_URL or http://127.0.0.1:8080).",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HEARTH_TOKEN"),
        help="Bearer token (default: $HEARTH_TOKEN; see ~/.hearth/token).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Actually call the running daemon. Omit for a safe offline dry run.",
    )
    args = parser.parse_args(argv)

    if not args.live:
        print("[dry run] Would offload 2 subtasks to HEARTH at", args.url)
        print("[dry run] Sample text:", SAMPLE_TEXT[:60], "...")
        print("[dry run] Re-run with --live against `hearth serve` to execute.")
        return 0

    run_offload(args.url, args.token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
