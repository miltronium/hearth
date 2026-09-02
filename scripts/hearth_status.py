#!/usr/bin/env python
"""Measure HEARTH's real state and print it — the antidote to a rotting status doc.

Everything this prints was measured when you ran it: bytes on disk, a routing policy after
the loader actually resolved it, lines in a golden file, the GPU's own working-set ceiling,
git history. Nothing is read from ``docs/RESULTS.md``, ``docs/cmux/HANDOFF.md`` or any other
hand-maintained file, because those record what somebody *believed*. See ``docs/STATUS.md``
for why that distinction is load-bearing here.

**This command is read-only.** It writes no file, creates no directory, opens no socket and
mutates nothing — so it is always safe to run first, including on a machine you suspect is
broken and in a sealed no-egress session.

Usage:
    uv run python scripts/hearth_status.py                # human-readable
    uv run python scripts/hearth_status.py --json         # machine-readable
    uv run python scripts/hearth_status.py --section learning egress
    uv run python scripts/hearth_status.py --strict       # exit 1 if anything is warn/fail

Exit codes: 0 always, unless ``--strict`` is given and a ``warn``/``fail`` fact was measured
(``unverified`` never fails the command — an unmeasured thing is not a broken thing).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:  # runnable without an editable install
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from hearth.status import collect_status, render_text  # noqa: E402
from hearth.status.report import LEVEL_FAIL, LEVEL_WARN, StatusReport  # noqa: E402


def _filtered(report: StatusReport, keys: list[str] | None) -> StatusReport:
    """Return ``report`` narrowed to ``keys`` (order preserved), or unchanged if none given."""
    if not keys:
        return report
    wanted = [s for s in report.sections if s.key in keys]
    unknown = sorted(set(keys) - {s.key for s in report.sections})
    if unknown:
        known = ", ".join(s.key for s in report.sections)
        raise SystemExit(f"unknown section(s): {', '.join(unknown)} (known: {known})")
    return StatusReport(
        generated_at=report.generated_at, root=report.root, sections=tuple(wanted)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report HEARTH's measured state (read-only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument(
        "--section",
        nargs="+",
        metavar="KEY",
        help="only these sections: models egress learning tests environment staleness",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=_REPO_ROOT,
        help="repo checkout to probe (default: the one this script lives in)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 when any warn/fail fact was measured (unverified never fails)",
    )
    args = parser.parse_args(argv)

    report = _filtered(collect_status(root=args.repo), args.section)
    if args.json:
        print(json.dumps(report.to_json(), indent=2, sort_keys=False))
    else:
        print(render_text(report))

    if args.strict and report.worst() in (LEVEL_FAIL, LEVEL_WARN):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
