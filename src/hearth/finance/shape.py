"""What a statement-shape report may print, decided by construction. One rule, two scripts.

``scripts/hearth_peek.py`` and ``scripts/hearth_map_draft.py`` both print to a terminal whose
output an operator is likely to paste into a chat with a *cloud* agent. Both used to take
``rows[0]`` as the header and print it raw, so a preamble ("Account Holder, Jane Q Public,
Premier Checking") came out as "column names" followed by a promise that no value had been
printed (B-044, B-063, B-074). This module is the single implementation of the rule that
replaced that, so the two scripts cannot drift apart:

* **A header cell is printable only if it is made of :data:`VOCAB` words** (generic statement
  column words) plus spacing and punctuation, with no digit anywhere. Anything else is
  :data:`WITHHELD`, whatever it looks like. No heuristic decides whether text "looks like a
  label"; no heuristic can tell a merchant name from a column name.
* **The header row is the first row (within :data:`HEADER_SCAN`) holding at least
  :data:`MIN_KNOWN` vocabulary names**; for keyed data (a JSON array of objects) it is the key
  row. Rows above it are preamble: they become ``skip_rows`` and are never printed. A wrong
  pick costs nothing in privacy, because whichever row is picked only its vocabulary cells can
  print.
* **Files are ordinal ids** (``F1``, ``F2``, ...) plus an extension from a fixed list, never a
  name or path: file and directory names carry account numbers and holder names.
* **Refusals print a fixed reason or an exception type**, never the message: parser messages
  quote the bytes they choked on and file-gate messages quote the path.

This module reads nothing and imports only the standard library, so it sits inside
``hearth.finance``'s no-network invariant (tests/test_finance_aggregate.py checks the package).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path

#: The only words a printed header may contain. Each is a generic statement-column word, so a
#: cell made only of these words says nothing about the account or its owner even in the worst
#: case (a data row mistaken for the header prints, at most, "Debit" or "Check").
#: Adding a word here widens what BOTH scripts may print: keep it to column vocabulary, never a
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

HEADER_SCAN = 30  # how far down a file the real header may sit below a preamble
MIN_KNOWN = 2  # known column names a row needs before it is taken as the header

WITHHELD = "(withheld)"
NO_HEADER = "(withheld: no header found)"
EMPTY_HEADER = "(blank)"

#: The extensions a file label may show. Anything else prints as ``other``.
KNOWN_EXTS: frozenset[str] = frozenset(
    {".csv", ".xlsx", ".json", ".pdf", ".txt", ".text", ".md", ".log"}
)

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
ACCESS_REFUSED = "refused by the file gate (reason withheld: the message names the path)"


def is_known_column_name(cell: str) -> bool:
    """True only if ``cell`` is made of :data:`VOCAB` words, spacing and punctuation.

    This is an allowlist, not a guess about whether the cell "looks like a label": case,
    whitespace and punctuation are normalized away, every remaining word must be in the
    vocabulary, and a digit anywhere refuses the cell outright.
    """
    c = str(cell).strip()
    if not c or len(c) > _MAX_HEADER_CHARS or not _HEADER_CHARS.match(c):
        return False
    words = [w for w in re.split(r"[^a-z]+", c.lower()) if w]
    return 0 < len(words) <= _MAX_HEADER_WORDS and all(w in VOCAB for w in words)


def find_header(rows: Sequence[Sequence[str]], *, keyed: bool = False) -> int | None:
    """Return the index of the header row, or None if no row qualifies.

    ``keyed`` tables (a JSON array of objects, as ``read_table`` returns it) have their keys
    as row 0 by construction. Otherwise the header is the first row within
    :data:`HEADER_SCAN` holding at least :data:`MIN_KNOWN` known column names; every row above
    it is preamble, i.e. ``skip_rows``.
    """
    if keyed:
        return 0 if rows else None
    for i, row in enumerate(rows[:HEADER_SCAN]):
        if sum(1 for c in row if is_known_column_name(c)) >= MIN_KNOWN:
            return i
    return None


def header_label(cell: str) -> str:
    """What may be printed for one header cell: itself if known, else a fixed placeholder."""
    if is_known_column_name(cell):
        return str(cell).strip()
    return WITHHELD if str(cell).strip() else EMPTY_HEADER


def column_ref(cell: str, index: int) -> str:
    """A printable reference to a column inside a sentence: ``'Amount'`` or ``column 3 (withheld)``.

    Withheld columns are told apart by position, which is all an operator needs to find them
    in the file locally.
    """
    if is_known_column_name(cell):
        return repr(str(cell).strip())
    return f"column {index} {WITHHELD if str(cell).strip() else EMPTY_HEADER}"


def is_printable_slug(slug: str, *, extra: Iterable[str] = ()) -> bool:
    """True when every hyphen-separated word of ``slug`` is vocabulary, ``extra`` or a number.

    For names that are not header cells but may echo one (a model-proposed format name). A
    number is at most two digits, so no account-number fragment passes.
    """
    allowed = VOCAB | frozenset(extra)
    words = [w for w in slug.split("-") if w]
    return bool(words) and all(w in allowed or re.fullmatch(r"\d{1,2}", w) for w in words)


def file_ids(paths: Iterable[Path]) -> dict[Path, str]:
    """Ordinal ids ``F1..Fn`` in the order given. The only way a file is ever named in output."""
    return {path: f"F{n}" for n, path in enumerate(paths, 1)}


def ext_label(path: Path) -> str:
    """The file's extension if it is one we know; never free text from the file name."""
    suffix = path.suffix.lower()
    return suffix if suffix in KNOWN_EXTS else "other"


def file_label(path: Path, ids: dict[Path, str]) -> str:
    """``F3 (.csv)``: an id and a fixed-list extension. Nothing from the name or the path."""
    return f"{ids[path]} ({ext_label(path)})"


def write_index(target: Path, ids: dict[Path, str], extra: Iterable[tuple[str, Path]] = ()) -> None:
    """Write the id -> real path list to a LOCAL file. Nothing about it is printed by callers."""
    lines = [f"{fid}\t{path}" for path, fid in ids.items()]
    lines += [f"{label}\t{path}" for label, path in extra]
    target.expanduser().write_text("\n".join(lines) + "\n", encoding="utf-8")


def refusal_reason(exc: BaseException, access_error: type[BaseException] | None = None) -> str:
    """A fixed reason string for a refusal; never the exception's message.

    ``access_error`` is the file gate's exception type (``hearth.mcp.files.FileAccessError``),
    passed in so this module stays free of the reader's imports. Its message is matched against
    known prefixes and mapped to fixed text; any other exception prints only its type.
    """
    if access_error is not None and isinstance(exc, access_error):
        msg = str(exc)
        for prefix, reason in _ACCESS_REASONS:
            if msg.startswith(prefix):
                return reason
        return ACCESS_REFUSED
    return f"{type(exc).__name__} (message withheld: may quote file content)"
