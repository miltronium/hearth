"""Rows plus a mapping become :class:`Transaction` records — or the parse stops with an index.

Two properties matter more than anything else in this module, and both exist because of the
same failure mode: **a mis-parsed statement produces a plausible number, not an error.**

**1. Every amount is a :class:`decimal.Decimal`, from the string, all the way through.** Never
a float. ``float("0.1") + float("0.2")`` is not ``0.3``, and the error compounds through
aggregation until a reconciliation that should tie out by construction is off by cents for
reasons nobody can find. APEX's ``_coerce_money`` routes even floats through ``str()`` for
exactly this reason (``docs/APEX_seam.md`` §5.2); here no float is ever constructed at all.

**2. A row that cannot be parsed is a hard error carrying its row index — never a skip.** A
skipped row is invisible: the totals still add up, the reconciliation still ties to itself,
and the number is simply wrong by however much that row was worth. So :func:`parse_rows`
raises :class:`ParseError` on the first unparseable row and names it, and the caller fixes the
mapping or the file. There is no ``errors="ignore"`` mode and there should never be one.

The transform itself is narrow: cells in, normalized records out, using only what the
:class:`~hearth.finance.mapping.ColumnMapping` states. No format is inferred, no fallback date
format is tried, and no sign is deduced from the data.
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .mapping import (
    CURRENCY_SYMBOLS,
    NOTATION_MINUS,
    NOTATION_PARENS,
    NOTATION_TRAILING_MINUS,
    SIGN_AS_WRITTEN,
    SIGN_DEBIT_NEGATIVE,
    SIGN_NEGATE,
    ColumnMapping,
    MappingError,
    inspect_header,
    normalize_header,
)

# Whitespace that shows up in exported money cells. NBSP and narrow-NBSP are real thousands
# separators in some locales, so they are stripped from the *edges* only — never from inside a
# number, where removing them would silently reinterpret the value.
_EDGE_TRIM = " \t\r\n\u00a0\u202f"

ZERO = Decimal("0")


class ParseError(ValueError):
    """A table or one of its rows could not be parsed under the given mapping.

    ``row_index`` is the offset into the row sequence handed to :func:`parse_rows` — index 0
    is the first row of the table (usually the header), so the number can be turned straight
    into a line to look at. It is ``None`` for table-level failures (a missing column, an
    absent reader) that belong to no single row.

    ``column`` names the column whose cell failed, when there is one. The message never quotes
    a whole row: an exception can travel further than the data it came from.
    """

    def __init__(
        self, message: str, *, row_index: int | None = None, column: str | None = None
    ) -> None:
        super().__init__(message)
        self.row_index = row_index
        self.column = column


@dataclass(frozen=True)
class Transaction:
    """One normalized line item. Immutable, exact, and traceable back to its row.

    ``amount`` follows HEARTH's single normal form — **money in is positive, money out is
    negative** — reached by applying the mapping's stated sign convention, never by inspecting
    the data. Its precision is whatever the file wrote: a cell reading ``12.50`` becomes
    ``Decimal("12.50")`` and stays that way, so a total can be compared to a printed control
    total digit for digit.

    ``row_index`` is kept so that every downstream number can be walked back to the line it
    came from. A reconciliation that cannot point at its inputs is a claim, not a check.

    ``category`` is assigned elsewhere (that is a judgement, and this package makes none); it
    rides along so :mod:`hearth.finance.aggregate` can group without a second lookup table.
    """

    date: datetime.date
    description: str
    amount: Decimal
    row_index: int
    currency: str | None = None
    category: str | None = None

    def with_category(self, category: str | None) -> Transaction:
        """Return a copy carrying ``category`` — the record itself stays immutable."""
        return replace(self, category=category)


# -- money ---------------------------------------------------------------------------------


def parse_money(text: str, mapping: ColumnMapping) -> Decimal:
    """Parse one money cell into a :class:`Decimal`, or raise :class:`ParseError`.

    Handles only what the mapping declares. A parenthesized or trailing-minus negative is
    refused unless the operator enabled that notation, because the alternative — reading
    ``(50.00)`` as ``50.00`` — turns a charge into a deposit and still balances to a number
    that looks fine. Grouping is checked positionally, so ``1,50`` is refused under a
    comma-thousands mapping instead of quietly becoming one hundred and fifty.

    The returned sign is the cell's own; the mapping's sign *convention* is applied by
    :func:`parse_rows`, which knows whether the cell came from an amount, debit or credit
    column.
    """
    raw = str(text).strip(_EDGE_TRIM)
    if not raw:
        raise ParseError("amount cell is empty")

    negative = False
    body = raw

    if body.startswith("(") and body.endswith(")"):
        if NOTATION_PARENS not in mapping.negative_notation:
            raise ParseError(
                f"{raw!r} is a parenthesized accounting negative, but this mapping does not "
                f"enable it; add {NOTATION_PARENS!r} to negative_notation if that is what "
                "this bank writes"
            )
        negative = True
        body = body[1:-1].strip(_EDGE_TRIM)
    elif body.endswith("-"):
        if NOTATION_TRAILING_MINUS not in mapping.negative_notation:
            raise ParseError(
                f"{raw!r} has a trailing minus, but this mapping does not enable it; add "
                f"{NOTATION_TRAILING_MINUS!r} to negative_notation if that is what this bank "
                "writes"
            )
        negative = True
        body = body[:-1].strip(_EDGE_TRIM)

    body = body.strip(CURRENCY_SYMBOLS + _EDGE_TRIM)

    if body.startswith(("-", "+")):
        if body[0] == "-":
            if NOTATION_MINUS not in mapping.negative_notation:
                raise ParseError(
                    f"{raw!r} has a leading minus, but this mapping does not enable "
                    f"{NOTATION_MINUS!r}"
                )
            if negative:
                raise ParseError(f"{raw!r} states its sign twice; which one is authoritative?")
            negative = True
        body = body[1:].strip(CURRENCY_SYMBOLS + _EDGE_TRIM)

    if not body:
        raise ParseError(f"{raw!r} has no digits — an empty or placeholder cell is not a zero")

    pattern = _numeric_pattern(mapping.decimal_separator, mapping.thousands_separator)
    if not pattern.match(body):
        raise ParseError(
            f"{raw!r} is not a number under this mapping (decimal_separator="
            f"{mapping.decimal_separator!r}, thousands_separator="
            f"{mapping.thousands_separator!r}). Digit grouping must be in threes, so a "
            "separator in any other position is refused rather than assumed"
        )

    cleaned = body
    if mapping.thousands_separator:
        cleaned = cleaned.replace(mapping.thousands_separator, "")
    if mapping.decimal_separator != ".":
        cleaned = cleaned.replace(mapping.decimal_separator, ".")

    try:
        value = Decimal(cleaned)
    except InvalidOperation:  # pragma: no cover - the pattern above already guarantees this
        raise ParseError(f"{raw!r} is not a decimal number") from None
    return -value if negative else value


_PATTERN_CACHE: dict[tuple[str, str], re.Pattern[str]] = {}


def _numeric_pattern(decimal_sep: str, thousands_sep: str) -> re.Pattern[str]:
    """Build (and cache) the strict numeric pattern for one separator pair.

    Either an ungrouped run of digits, or digits grouped strictly in threes — so a mapping
    that declares ``,`` as the thousands separator rejects ``1,50`` outright. That is the
    difference between an error the operator sees and a hundredfold overstatement they do not.
    """
    key = (decimal_sep, thousands_sep)
    cached = _PATTERN_CACHE.get(key)
    if cached is not None:
        return cached
    dec = re.escape(decimal_sep)
    frac = rf"(?:{dec}\d+)?"
    if thousands_sep:
        thou = re.escape(thousands_sep)
        whole = rf"(?:\d{{1,3}}(?:{thou}\d{{3}})+|\d+)"
    else:
        whole = r"\d+"
    pattern = re.compile(rf"^{whole}{frac}$")
    _PATTERN_CACHE[key] = pattern
    return pattern


# -- rows ------------------------------------------------------------------------------------


def data_row_count(rows: list[list[str]], mapping: ColumnMapping) -> int:
    """Return how many rows of ``rows`` are data — everything after skip_rows and the header.

    :func:`~hearth.finance.validate.reconcile` requires the caller to state this separately
    from the parsed records, so "rows read equals rows parsed" is a real comparison against
    the table rather than a list compared with itself.
    """
    return max(0, len(rows) - mapping.skip_rows - 1)


def resolve_columns(header: list[str] | tuple[str, ...], mapping: ColumnMapping) -> dict[str, int]:
    """Return ``role -> column index``, raising :class:`MappingError` if the mapping misfits.

    Runs before any row is touched, so a mapping written for last year's export format fails
    on the header rather than half way down the file with a partial total already computed.
    """
    report = inspect_header(header, mapping)
    if report.missing:
        detail = ", ".join(f"{role}={name!r}" for role, name in sorted(report.missing.items()))
        raise MappingError(
            f"mapped column(s) not present in this header: {detail}. The header has "
            f"{list(report.header)}"
        )
    if report.ambiguous:
        raise MappingError(
            f"this header repeats mapped column name(s) {list(report.ambiguous)}; there is no "
            "stated rule for which occurrence wins, and picking one is a guess"
        )
    cells = normalize_header(header)
    return {role: cells.index(name) for role, name in report.mapped.items()}


def parse_rows(
    rows: list[list[str]], mapping: ColumnMapping, *, start_index: int | None = None
) -> list[Transaction]:
    """Parse a whole table into transactions. Any unparseable row raises with its index.

    ``rows`` is the table as :func:`hearth.mcp.files.read_table` returns it: the header row
    followed by data rows, each a list of cell strings. ``mapping.skip_rows`` accounts for any
    preamble above the header.

    ``start_index`` overrides the index reported for the first row of ``rows`` — pass it when
    handing over a slice, so the indices in errors and in
    :attr:`Transaction.row_index` still refer to the original file.

    Raises :class:`ParseError` (with ``row_index``) for a bad row, and
    :class:`~hearth.finance.mapping.MappingError` for a header the mapping does not fit.
    Returns whole or not at all: a partial result would be a silent skip wearing a list.
    """
    base = 0 if start_index is None else start_index
    if len(rows) <= mapping.skip_rows:
        raise ParseError(
            f"table has {len(rows)} row(s), which is not enough for skip_rows="
            f"{mapping.skip_rows} plus a header row"
        )

    header_index = mapping.skip_rows
    columns = resolve_columns(rows[header_index], mapping)
    width = len(rows[header_index])

    out: list[Transaction] = []
    for offset, row in enumerate(rows[header_index + 1 :], start=header_index + 1):
        out.append(_parse_row(row, columns, mapping, row_index=base + offset, width=width))
    return out


def _parse_row(
    row: list[str],
    columns: dict[str, int],
    mapping: ColumnMapping,
    *,
    row_index: int,
    width: int,
) -> Transaction:
    """Parse one data row, attaching ``row_index`` to every failure it can produce."""
    if not any(str(cell).strip() for cell in row):
        raise ParseError(
            "row is blank. A blank row is an error rather than a skip: skipping it silently "
            "is indistinguishable from skipping a real transaction",
            row_index=row_index,
        )
    if len(row) < width:
        raise ParseError(
            f"row has {len(row)} cell(s) but the header has {width}; a short row would read "
            "some other column's value as the amount",
            row_index=row_index,
        )

    def cell(role: str) -> str:
        return str(row[columns[role]])

    date_text = cell("date").strip()
    if not date_text:
        raise ParseError("date cell is empty", row_index=row_index, column=mapping.date_column)
    try:
        when = datetime.datetime.strptime(date_text, mapping.date_format).date()
    except ValueError:
        raise ParseError(
            f"{date_text!r} does not match date_format {mapping.date_format!r}. No other "
            "format is tried: two formats can both parse a date and only one is right",
            row_index=row_index,
            column=mapping.date_column,
        ) from None

    description = " ".join(cell("description").split())

    try:
        amount = (
            _paired_amount(cell("debit"), cell("credit"), mapping)
            if mapping.uses_pair
            else _single_amount(cell("amount"), mapping)
        )
    except ParseError as exc:
        raise ParseError(
            str(exc),
            row_index=row_index,
            column=mapping.amount_column if not mapping.uses_pair else mapping.debit_column,
        ) from None

    return Transaction(
        date=when,
        description=description,
        amount=amount,
        row_index=row_index,
        currency=mapping.currency,
    )


def _single_amount(text: str, mapping: ColumnMapping) -> Decimal:
    """Apply a single-column sign convention to one parsed cell."""
    value = parse_money(text, mapping)
    if mapping.sign == SIGN_AS_WRITTEN:
        return value
    if mapping.sign == SIGN_NEGATE:
        return -value
    raise ParseError(f"sign convention {mapping.sign!r} does not apply to a single column")


def _paired_amount(debit_text: str, credit_text: str, mapping: ColumnMapping) -> Decimal:
    """Combine a debit/credit pair into one signed amount under the declared convention.

    A zero in the unused column is treated as absent — exports routinely write ``0.00`` rather
    than leaving the cell blank, and a zero cannot change the answer either way. Two *non-zero*
    values in one row is an error: the file is saying two contradictory things and there is no
    stated rule for which wins.

    A signed value in either column is also an error. Under a debit/credit convention the
    columns carry magnitudes; a minus sign there means the declared convention is wrong, and
    the resulting number would be right in size and wrong in direction.
    """
    debit = parse_money(debit_text, mapping) if debit_text.strip(_EDGE_TRIM) else None
    credit = parse_money(credit_text, mapping) if credit_text.strip(_EDGE_TRIM) else None

    for name, value in (("debit", debit), ("credit", credit)):
        if value is not None and value < ZERO:
            raise ParseError(
                f"the {name} column holds a negative value ({value}); under sign="
                f"{mapping.sign!r} these columns carry magnitudes and the direction comes from "
                "which column is filled. A signed value here means the mapping is wrong"
            )

    debit = None if debit == ZERO else debit
    credit = None if credit == ZERO else credit

    if debit is not None and credit is not None:
        raise ParseError(
            f"both the debit ({debit}) and credit ({credit}) columns are non-zero; the row "
            "states two amounts and the mapping does not say which one is the transaction"
        )
    if debit is None and credit is None:
        return ZERO

    outflow = mapping.sign == SIGN_DEBIT_NEGATIVE
    if debit is not None:
        return -debit if outflow else debit
    credit_value = credit if credit is not None else ZERO
    return credit_value if outflow else -credit_value


# -- reading a table off disk ---------------------------------------------------------------


def read_table(path: str | Path, settings: Any | None = None) -> list[list[str]]:
    """Read an allowlisted local file as a table via :mod:`hearth.mcp.files`.

    A thin seam, not a reader. HEARTH's allowlisted reader owns bytes-to-rows — the deny-by-
    default roots, the size cap, the format dispatch and the errors that never quote file
    content (``docs/PRIVACY.md``). This package owns rows-to-records and must not grow a
    second copy of any of that.

    The import is local and late so that :mod:`hearth.finance` stays importable, testable and
    auditable while ``read_table`` is still landing next door. If it is absent, the error says
    so plainly rather than falling back to a reader of our own.
    """
    try:
        from ..mcp.files import read_table as _read_table
    except ImportError:  # pragma: no cover - exercised by the skip guard in the tests
        raise ParseError(
            "hearth.mcp.files.read_table is not available; this package deliberately has no "
            "reader of its own — file access goes through the allowlisted reader or not at all"
        ) from None
    return _read_table(path, settings=settings)


def parse_file(
    path: str | Path, mapping: ColumnMapping, settings: Any | None = None
) -> list[Transaction]:
    """Read ``path`` through the allowlisted reader and parse it under ``mapping``."""
    return parse_rows(read_table(path, settings), mapping)


__all__ = [
    "ZERO",
    "ParseError",
    "Transaction",
    "data_row_count",
    "parse_file",
    "parse_money",
    "parse_rows",
    "read_table",
    "resolve_columns",
]
