#!/usr/bin/env python3
"""Show statement files' SHAPE without ever printing their contents.

Authoring a column mapping needs the header row and a sense of what each column holds. It
does NOT need the values — and the values are the whole reason this pipeline exists. So this
prints headers, row counts and per-column type *guesses*, and never emits a cell.

That distinction is the point. Header names ("Posting Date", "Amount", "Balance") are safe to
read aloud, paste into a chat, or hand to a cloud agent for help writing a mapping. The cells
underneath are not. Keeping them apart mechanically means the safe path is also the
convenient one, instead of relying on somebody remembering to be careful.

Accepts files *and* directories; directories are walked recursively, because real statement
archives are nested by year and account. The output groups files by their **header
signature**, since that is the question that actually matters once there is more than a
handful: you need one mapping per distinct format, not one per file. Twenty statements from
one bank are one mapping and this will say so.

    HEARTH_FILE_ROOTS=~/Documents/Banking \
        uv run --no-sync python scripts/hearth_peek.py ~/Documents/Banking

Reads go through the same HEARTH_FILE_ROOTS allowlist as every other path-taking read; this
script has no privileges of its own, and a file outside the roots is refused here too.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from hearth.mcp.files import FileAccessError, read_table, read_text_file  # noqa: E402

TABLE_EXTS = {".csv", ".xlsx", ".json"}
TEXT_EXTS = {".pdf", ".txt", ".text", ".md", ".log"}
DEFAULT_EXTS = TABLE_EXTS | TEXT_EXTS

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
        neg = any(c.startswith("(") or c.endswith("-") for c in seen)
        return "number-like (accounting negatives present)" if neg else "number-like"
    return "text"


def _collect(targets: list[str], exts: set[str]) -> list[Path]:
    """Expand files and directories into a sorted list of candidate files."""
    found: list[Path] = []
    for target in targets:
        path = Path(target).expanduser()
        if path.is_dir():
            found.extend(
                p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in exts
            )
        else:
            found.append(path)
    # Deduplicate while keeping a stable, human-scannable order.
    return sorted(set(found))


def _inspect(path: Path) -> tuple[tuple[str, ...] | None, dict]:
    """Return (header signature, details). Signature is None for non-tabular files."""
    if path.suffix.lower() in TABLE_EXTS:
        rows = read_table(path)
        if not rows:
            return (), {"kind": "table", "rows": 0, "columns": []}
        header, data = rows[0], rows[1:]
        width = max(len(r) for r in rows)
        columns = []
        for i in range(width):
            name = header[i] if i < len(header) else "(no header)"
            columns.append((i, name, _guess([r[i] for r in data if i < len(r)])))
        return tuple(c[1] for c in columns), {
            "kind": "table",
            "rows": len(data),
            "columns": columns,
        }
    # Text documents (PDF and friends) have no header row to group on. Report only that the
    # text layer exists and roughly how much of it — never any of the text itself.
    text = read_text_file(path)
    return None, {"kind": "text", "chars": len(text), "lines": text.count("\n") + 1}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show statement files' shape (headers and column types) — never their values.",
    )
    parser.add_argument("targets", nargs="+", help="files or directories (walked recursively)")
    parser.add_argument(
        "--ext",
        action="append",
        default=None,
        help="limit to this extension (repeatable), e.g. --ext .csv --ext .xlsx",
    )
    args = parser.parse_args()

    exts = {e if e.startswith(".") else f".{e}" for e in (args.ext or [])} or DEFAULT_EXTS
    files = _collect(args.targets, exts)
    if not files:
        print("no matching files found")
        return 1

    groups: dict[tuple[str, ...], list[tuple[Path, dict]]] = defaultdict(list)
    texts: list[tuple[Path, dict]] = []
    refused: list[tuple[Path, str]] = []

    for path in files:
        try:
            signature, details = _inspect(path)
        except FileAccessError as exc:
            refused.append((path, str(exc)))
            continue
        except Exception as exc:  # a malformed file should not abort the sweep
            refused.append((path, f"{type(exc).__name__}: {exc}"))
            continue
        if signature is None:
            texts.append((path, details))
        else:
            groups[signature].append((path, details))

    common = Path(files[0]).parent
    for path in files[1:]:
        while common != common.parent and not str(path).startswith(str(common) + "/"):
            common = common.parent

    print(f"\nscanned {len(files)} file(s) under {common}")
    print(f"{len(groups)} distinct table format(s), {len(texts)} text document(s), "
          f"{len(refused)} unreadable\n")

    ranked = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    for n, (_signature, members) in enumerate(ranked, 1):
        total_rows = sum(d["rows"] for _, d in members)
        print(f"── format {n} ── {len(members)} file(s), {total_rows} data rows "
              f"── needs ONE mapping ──")
        for path, details in members:
            try:
                shown = path.relative_to(common)
            except ValueError:
                shown = path
            print(f"     {shown}  ({details['rows']} rows)")
        print()
        print(f"     {'#':>3}  {'header':<34} {'looks like'}")
        print(f"     {'-' * 3}  {'-' * 34} {'-' * 40}")
        for i, name, guess in members[0][1]["columns"]:
            print(f"     {i:>3}  {name[:34]:<34} {guess}")
        print()

    if texts:
        print("── text documents (no header row; read as text, not as a table) ──")
        for path, details in texts:
            try:
                shown = path.relative_to(common)
            except ValueError:
                shown = path
            print(f"     {shown}  ({details['chars']} chars, {details['lines']} lines)")
        print()

    if refused:
        print("── could not read ──")
        for path, reason in refused:
            try:
                shown = path.relative_to(common)
            except ValueError:
                shown = path
            print(f"     {shown}\n       {reason}")
        print()

    print("No cell values were printed. Header names are safe to share; values are not.")
    if groups:
        print(f"You need {len(groups)} column mapping(s), one per format above.")
    return 0 if not refused else 1


if __name__ == "__main__":
    raise SystemExit(main())
