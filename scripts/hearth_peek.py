#!/usr/bin/env python3
"""Show statement files' SHAPE without ever printing their contents.

Authoring a column mapping needs column positions, a sense of what each column holds, and the
names of the columns you map. It does NOT need the values, and the values are the reason this
pipeline exists. This script's output is meant to be safe to paste into a chat with a cloud
agent, so what it may print is fixed **by construction**, not judged per cell:

  * **header names only from a fixed vocabulary.** A header cell is printed only if it is made
    entirely of words from :data:`VOCAB` (statement column words: date, posting, description,
    amount, debit, credit, balance, ...) plus spacing/punctuation, with no digit anywhere. Any
    other header cell prints as ``(withheld)``. An earlier version tried to tell a text VALUE
    from a LABEL heuristically; "Jane Q Public", "Premier Checking" and a headerless file's
    merchant names all looked like labels (B-063). No heuristic can make that distinction, so
    there is none: unknown text is withheld, whatever it looks like.
  * **no preamble rows, no data rows, no raw JSON keys.** Only their *counts*. JSON keys are
    header cells and go through the same vocabulary rule.
  * **no file names or paths.** File names carry account numbers and holder names, and so do
    directories. Files are shown as ordinal ids (``F1``, ``F2``, ...) plus their extension from
    a fixed list. ``--index-out FILE`` writes the id -> path list to a local file for you.
  * **no exception messages.** Parser messages can quote the bytes they choked on; a refusal
    prints a fixed reason string or the exception type.

Everything else printed is a count (rows, columns, preamble lines) or a type CLASS
("date-like", "number-like", "text", "empty").

What that leaves an operator is enough to write a mapping: the position of every column, its
type guess, the names of the conventionally named columns, and ``skip_rows`` (the preamble
count). For a column shown as ``(withheld)``, open the file locally and copy the header text
yourself; do not paste it into a cloud chat.

Accepts files *and* directories; directories are walked recursively. Files are grouped by
header signature (computed from the raw header internally, never printed), because you need
one mapping per distinct format, not one per file.

    HEARTH_FILE_ROOTS=~/Documents/Banking \\
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

#: The only words a printed header may contain. Each is a generic statement-column word, so a
#: cell made only of these words says nothing about the account or its owner even in the worst
#: case (a data row mistaken for the header prints, at most, "Debit" or "Check").
#: Adding a word here widens what this script may print: keep it to column vocabulary, never a
#: name, merchant, place or institution.
VOCAB: frozenset[str] = frozenset(
    """
    date dates day time posting posted post transaction transactions trans tran txn effective
    value settlement settled processed booking booked cleared pending
    description desc details detail memo memos narrative notes note payee payer merchant name
    amount amounts amt debit debits credit credits withdrawal withdrawals deposit deposits
    money in out paid received payment payments charge charges purchase fee fees tax
    balance running available ledger closing opening total net gross principal interest
    type category categories subcategory class tag tags label method mode
    reference ref check cheque number no num nbr id code mcc sequence seq line item
    status account acct card currency original
    """.split()
)
_MAX_HEADER_WORDS = 4
_MAX_HEADER_CHARS = 40
#: Characters allowed in a printed header besides vocabulary letters. No digits, ever.
_HEADER_CHARS = re.compile(r"^[A-Za-z .,_/()#&:$*'-]+$")

_DATE = re.compile(r"^\s*\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}")
_MONEY = re.compile(r"^\s*[-+(]?\s*[$€£]?\s*[\d,]+\.?\d*\s*\)?-?\s*$")
_HEADER_SCAN = 30  # how far down a file the real header may sit below a preamble
_MIN_KNOWN = 2  # known column names a row needs before it is taken as the header
WITHHELD = "(withheld)"
NO_HEADER = "(withheld: no header found)"
EMPTY_HEADER = "(blank)"

# A FileAccessError message names the requested path, so it is never printed. These prefixes
# map it to a fixed reason; anything unmatched prints only that it was refused.
_ACCESS_REASONS = (
    ("file reads are disabled", "file reads disabled: HEARTH_FILE_ROOTS is not set"),
    ("path is outside every allowed root", "outside HEARTH_FILE_ROOTS"),
    ("file not found or not readable", "not found or not readable"),
    ("path is a directory", "is a directory"),
    ("path is not a regular file", "not a regular file"),
    ("file is too large", "too large (HEARTH_FILE_MAX_BYTES)"),
    ("file could not be read", "could not be read"),
    ("PDF is password-protected", "password-protected PDF"),
    ("file is not a readable PDF", "not a readable PDF"),
)


def is_known_column_name(cell: str) -> bool:
    """True only if ``cell`` is made of :data:`VOCAB` words, spacing and punctuation.

    This is an allowlist, not a guess about whether the cell "looks like a label": case,
    whitespace and punctuation are normalized away, every remaining word must be in the
    vocabulary, and a digit anywhere refuses the cell outright.
    """
    c = cell.strip()
    if not c or len(c) > _MAX_HEADER_CHARS or not _HEADER_CHARS.match(c):
        return False
    words = [w for w in re.split(r"[^a-z]+", c.lower()) if w]
    return 0 < len(words) <= _MAX_HEADER_WORDS and all(w in VOCAB for w in words)


def _find_header(rows: list[list[str]]) -> int | None:
    """Return the index of the first row, within ``_HEADER_SCAN``, holding at least
    ``_MIN_KNOWN`` known column names; None if there is none.

    Every row above it is preamble and is only counted. A wrong pick costs nothing in privacy:
    whichever row is chosen, only its vocabulary cells are printed.
    """
    for i, row in enumerate(rows[:_HEADER_SCAN]):
        if sum(1 for c in row if is_known_column_name(c)) >= _MIN_KNOWN:
            return i
    return None


def _guess(cells: list[str]) -> str:
    """Classify a column from its cells, returning only the CLASS, never a cell."""
    seen = [c.strip() for c in cells if c.strip()]
    if not seen:
        return "empty"
    if all(_DATE.match(c) for c in seen):
        return "date-like"
    if all(_MONEY.match(c) for c in seen):
        neg = any(c.startswith("(") or c.endswith("-") for c in seen)
        return "number-like (accounting negatives present)" if neg else "number-like"
    return "text"


def _header_label(cell: str) -> str:
    """What may be printed for one header cell: itself if known, else a fixed placeholder."""
    if is_known_column_name(cell):
        return cell.strip()
    return WITHHELD if cell.strip() else EMPTY_HEADER


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
    return sorted(set(found))


def _inspect(path: Path) -> tuple[tuple | None, dict]:
    """Return (grouping signature, details). Signature is None for non-tabular files.

    The signature may hold raw header text: it is a dict key, never printed. ``details``
    holds only counts, type classes and :func:`_header_label` output.
    """
    if path.suffix.lower() in TABLE_EXTS:
        rows = read_table(path)
        if not rows:
            return ("empty",), {"kind": "table", "rows": 0, "columns": [], "header_at": None}
        # A JSON array of objects has its keys as row 0 by construction (read_table); for
        # every other table the header is searched for. Either way only vocabulary cells of
        # the chosen row are ever printed.
        at = 0 if path.suffix.lower() == ".json" else _find_header(rows)
        data = rows if at is None else rows[at + 1 :]
        header = [] if at is None else rows[at]
        span = max((len(r) for r in [header, *data]), default=0)
        columns = []
        for i in range(span):
            if at is None:
                name = NO_HEADER
            else:
                name = _header_label(header[i]) if i < len(header) else EMPTY_HEADER
            columns.append((i, name, _guess([r[i] for r in data if i < len(r)])))
        if at is None:
            signature = ("no-header", tuple(g for _, _, g in columns))
        else:
            signature = ("header", tuple(c.strip() for c in header))
        return signature, {
            "kind": "table",
            "rows": len(data),
            "columns": columns,
            "header_at": at,
        }
    # Text documents (PDF and friends) have no header row to group on. Report only how much
    # text there is, never any of it.
    text = read_text_file(path)
    return None, {"kind": "text", "chars": len(text), "lines": text.count("\n") + 1}


def _ext(path: Path) -> str:
    """The file's extension if it is one we know; never free text from the file name."""
    suffix = path.suffix.lower()
    return suffix if suffix in DEFAULT_EXTS else "other"


def _refusal(exc: Exception) -> str:
    """A fixed reason string for a refusal; never the exception's message."""
    if isinstance(exc, FileAccessError):
        msg = str(exc)
        for prefix, reason in _ACCESS_REASONS:
            if msg.startswith(prefix):
                return reason
        return "refused by the file gate (reason withheld: the message names the path)"
    return f"{type(exc).__name__} (message withheld: may quote file content)"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Show statement files' shape (column positions, type guesses, known column names) "
            "without printing values, other header text, or file names."
        ),
    )
    parser.add_argument("targets", nargs="+", help="files or directories (walked recursively)")
    parser.add_argument(
        "--ext",
        action="append",
        default=None,
        help="limit to this extension (repeatable), e.g. --ext .csv --ext .xlsx",
    )
    parser.add_argument(
        "--index-out",
        metavar="FILE",
        default=None,
        help=(
            "write the file id -> real path list to FILE (local only; it holds file names, "
            "so do not paste it anywhere). Nothing about it is printed."
        ),
    )
    args = parser.parse_args()

    exts = {e if e.startswith(".") else f".{e}" for e in (args.ext or [])} or DEFAULT_EXTS
    files = _collect(args.targets, exts)
    if not files:
        print("no matching files found")
        return 1
    ids = {path: f"F{n}" for n, path in enumerate(files, 1)}

    groups: dict[tuple, list[tuple[Path, dict]]] = defaultdict(list)
    texts: list[tuple[Path, dict]] = []
    refused: list[tuple[Path, str]] = []

    for path in files:
        try:
            signature, details = _inspect(path)
        except Exception as exc:  # a malformed file should not abort the sweep
            refused.append((path, _refusal(exc)))
            continue
        if signature is None:
            texts.append((path, details))
        else:
            groups[signature].append((path, details))

    if args.index_out:
        lines = [f"{ids[p]}\t{p}" for p in files]
        Path(args.index_out).expanduser().write_text("\n".join(lines) + "\n")

    print(f"\nscanned {len(files)} file(s); shown as ids F1..F{len(files)} "
          "(names and paths are not printed)")
    print(f"{len(groups)} distinct table format(s), {len(texts)} text document(s), "
          f"{len(refused)} unreadable\n")

    ranked = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    for n, (_signature, members) in enumerate(ranked, 1):
        total_rows = sum(d["rows"] for _, d in members)
        print(f"── format {n} ── {len(members)} file(s), {total_rows} data rows "
              f"── needs ONE mapping ──")
        for path, details in members:
            at = details["header_at"]
            if not details["columns"]:
                where = "empty"
            elif at is None:
                where = "no header row identified (names withheld)"
            elif at:
                where = (f"header is row {at + 1}; skip_rows: {at} "
                         f"({at} preamble line(s) counted, not shown)")
            else:
                where = "header is row 1; skip_rows: 0"
            print(f"     {ids[path]} ({_ext(path)})  {details['rows']} rows, {where}")
        print()
        print(f"     {'#':>3}  {'header':<36} {'looks like'}")
        print(f"     {'-' * 3}  {'-' * 36} {'-' * 40}")
        for i, name, guess in members[0][1]["columns"]:
            print(f"     {i:>3}  {name:<36} {guess}")
        print()

    if texts:
        print("── text documents (no header row; read as text, not as a table) ──")
        for path, details in texts:
            print(f"     {ids[path]} ({_ext(path)})  {details['chars']} chars, "
                  f"{details['lines']} lines")
        print()

    if refused:
        print("── could not read ──")
        for path, reason in refused:
            print(f"     {ids[path]} ({_ext(path)})  {reason}")
        print()

    # True by construction: every file-derived string above is either a header cell that
    # passed is_known_column_name (VOCAB words + punctuation, no digits), a count, a type
    # class, an extension from DEFAULT_EXTS, or a fixed reason string.
    print("Printed: file ids and extensions, row/column/preamble counts, a type guess per")
    print("column, and header cells made only of HEARTH's fixed column vocabulary (date,")
    print("description, amount, debit, credit, balance, ...). Not printed: any cell value,")
    print("preamble line, other header text, JSON key outside that vocabulary, file name,")
    print("path, or error message.")
    print("To write a mapping, column positions, type guesses, skip_rows and the names shown")
    print("are enough. For a (withheld) name, open the file locally and copy the header text")
    print("yourself; do not paste it into a cloud chat.")
    if args.index_out:
        print("The file id -> path list was written to the --index-out file (local only).")
    else:
        print("To see which file is which locally, re-run with --index-out FILE.")
    if groups:
        print(f"You need {len(groups)} column mapping(s), one per format above.")
    return 0 if not refused else 1


if __name__ == "__main__":
    raise SystemExit(main())
