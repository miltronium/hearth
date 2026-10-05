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
_DIGIT_RUN = re.compile(r"\d{4,}")
_HEADER_SCAN = 30  # how far down a file the real header may sit below a preamble
WITHHELD = "(withheld: looks like a value)"
NO_HEADER = "(header not identified: withheld)"


def _looks_like_label(cell: str) -> bool:
    """True if a cell reads as a column NAME rather than a value.

    Deliberately conservative: anything date- or money-shaped, long, carrying a run of four
    or more digits (account and card numbers, years, amounts), or an e-mail is treated as a
    value. A real header that trips this is shown as withheld — a missing label costs the
    operator a glance at the file; a printed value costs the whole premise of this script.
    """
    c = cell.strip()
    return bool(c) and not (
        _DATE.match(c)
        or _MONEY.match(c)
        or _DIGIT_RUN.search(c)
        or "@" in c
        or len(c) > 40
    )


def _find_header(rows: list[list[str]]) -> tuple[int | None, int]:
    """Return (header row index or None, table width).

    The width is the most common count of non-empty cells among rows with two or more. The
    header is the first row, within the first ``_HEADER_SCAN``, that has exactly that many
    non-empty cells and whose every non-empty cell looks like a label. Bank exports often put
    an account-name/number preamble above the header (what a mapping's ``skip_rows`` is for);
    taking ``rows[0]`` printed that preamble — cell values — as "headers".
    """
    counts = [sum(1 for c in r if c.strip()) for r in rows]
    widths = [n for n in counts if n >= 2]
    if not widths:
        return None, max((len(r) for r in rows), default=0)
    width = max(set(widths), key=widths.count)
    for i, row in enumerate(rows[:_HEADER_SCAN]):
        cells = [c for c in row if c.strip()]
        if counts[i] == width and all(_looks_like_label(c) for c in cells):
            return i, width
    return None, width


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
        at, _width = _find_header(rows)
        if at is None:
            header: list[str] = []
            data = rows
            preamble = 0
        else:
            header, data, preamble = rows[at], rows[at + 1 :], at
        span = max((len(r) for r in [header, *data]), default=0)
        columns = []
        for i in range(span):
            if at is None:
                name = NO_HEADER
            elif i < len(header) and _looks_like_label(header[i]):
                name = header[i].strip()
            elif i < len(header) and header[i].strip():
                # Unreachable while _find_header requires every header cell to be a label;
                # kept as the backstop if that rule is ever loosened.
                name = WITHHELD
            else:
                name = "(no header)"
            columns.append((i, name, _guess([r[i] for r in data if i < len(r)])))
        return tuple(c[1] for c in columns), {
            "kind": "table",
            "rows": len(data),
            "columns": columns,
            "preamble": preamble,
            "header_found": at is not None,
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
            # The exception TYPE only: a parser's message can quote the bytes or line it
            # choked on, which would be file content.
            reason = f"{type(exc).__name__} (message withheld: may quote file content)"
            refused.append((path, reason))
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
            skipped = details.get("preamble", 0)
            note = (
                f", {skipped} preamble line(s) above the header skipped, not shown"
                if skipped
                else ""
            )
            if not details.get("header_found", True):
                note += ", no header row identified"
            print(f"     {shown}  ({details['rows']} rows{note})")
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

    # True by construction: the only file-derived strings printed above are header cells that
    # passed _looks_like_label, and type CLASSES. Preamble rows, data rows, cells that look like
    # values, and parser error messages are never printed.
    print("No cell values were printed: only header cells that read as labels, and type")
    print("guesses. Preamble lines and value-like cells were withheld.")
    print("Values are not safe to share.")
    if groups:
        print(f"You need {len(groups)} column mapping(s), one per format above.")
    return 0 if not refused else 1


if __name__ == "__main__":
    raise SystemExit(main())
