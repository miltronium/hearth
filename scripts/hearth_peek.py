#!/usr/bin/env python3
"""Show a statement file's SHAPE without ever printing its contents.

Authoring a column mapping needs the header row and a sense of what each column holds.
It does NOT need the values — and the values are the whole reason this pipeline exists.
So this prints headers, row counts, and a per-column type *guess*, and never emits a cell.

That distinction is the point. Header names ("Posting Date", "Amount", "Balance") are safe
to read aloud, paste into a chat, or hand to a cloud agent for help writing a mapping.
The cells underneath are not. Keeping them apart mechanically means the safe path is also
the convenient one, instead of relying on somebody remembering to be careful.

    HEARTH_FILE_ROOTS=~/hearth-statements uv run --no-sync python scripts/hearth_peek.py \
        ~/hearth-statements/incoming/august.csv

Reads go through the same HEARTH_FILE_ROOTS allowlist as every other path-taking read;
this script has no privileges of its own.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from hearth.mcp.files import FileAccessError, read_table  # noqa: E402

_DATE = re.compile(r"^\s*\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}")
_MONEY = re.compile(r"^\s*[-+(]?\s*[$€£]?\s*[\d,]+\.?\d*\s*\)?-?\s*$")


def _guess(cells: list[str]) -> str:
    """Classify a column from its cells, returning only the CLASS — never a cell."""
    seen = [c.strip() for c in cells if c.strip()]
    if not seen:
        return "empty"
    if all(_DATE.match(c) for c in seen):
        return "date-like"
    if all(_MONEY.match(c) for c in seen):
        neg = any(c.strip().startswith("(") or c.strip().endswith("-") for c in seen)
        return "number-like (accounting negatives present)" if neg else "number-like"
    return "text"


def peek(path: str) -> int:
    rows = read_table(path)
    if not rows:
        print(f"{path}: no rows")
        return 1
    header, data = rows[0], rows[1:]
    width = max(len(r) for r in rows)
    print(f"\n{path}")
    print(f"  {len(data)} data rows, {width} columns\n")
    print(f"  {'#':>3}  {'header':<34} {'looks like'}")
    print(f"  {'-' * 3}  {'-' * 34} {'-' * 40}")
    for i in range(width):
        name = header[i] if i < len(header) else "(no header)"
        col = [r[i] for r in data if i < len(r)]
        print(f"  {i:>3}  {name[:34]:<34} {_guess(col)}")
    print("\n  No cell values were printed. Header names are safe to share; values are not.")
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    rc = 0
    for arg in sys.argv[1:]:
        try:
            rc |= peek(arg)
        except FileAccessError as exc:
            print(f"\n{arg}\n  refused: {exc}")
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
