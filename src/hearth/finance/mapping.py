"""The operator's explicit column mapping — the one place bank-export semantics are stated.

**HEARTH may parse structure. It must never infer financial semantics.** That line is the
whole reason this module exists, and ``docs/APEX_seam.md`` §5.2 is the argument for it: APEX's
``backend/processors/base.py`` already owns bank-export parsing, and its alias tables
(``DATE_ALIASES``, ``AMOUNT_ALIASES``, ``DEBIT_ALIASES``, …, resolved with alias *priority*)
are **empirical** — every entry is a bank that formatted its export differently, and the only
way to obtain that table is to have hit them. A second implementation that guesses will drift
from the first, and the drift is silent: **the wrong column parsed as an amount yields a
plausible number, not an error.**

So this module contains no alias table, no header sniffing, and no fallbacks. A
:class:`ColumnMapping` is written once per bank by a human, in YAML, and every field that
could change a number is required:

  * which column is the date, and its **exact** ``strptime`` format — never a list of formats
    tried in order, because ``03/04/2026`` parses under two of them and only one is right;
  * which column is the description;
  * either a single signed amount column **or** a debit/credit pair — never both, never
    neither;
  * an explicit **sign convention** saying which direction of money the file writes as
    positive. HEARTH's normalized output is fixed: *money in is positive, money out is
    negative*. The convention says how to get there from this bank's file;
  * the decimal and thousands separators, so ``1,50`` is never silently read as ``150``.

Construction validates and raises :class:`MappingError`; there is no partially-valid mapping
and no default that stands in for a decision. The one helper that touches a real header,
:func:`inspect_header`, *reports* — it lists what is mapped, unmapped and missing so the
operator can author the mapping quickly. It never picks a column.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

# -- sign conventions ---------------------------------------------------------------------
#
# HEARTH's normalized amount is always: money into the account positive, money out negative.
# These four names say what the *file* does, so the transform to that normal form is a stated
# fact rather than something inferred from column headers or from the sign of the first row.

#: Single amount column, already money-in-positive. Taken verbatim.
SIGN_AS_WRITTEN = "as_written"
#: Single amount column where a positive number means money *out* (a "Withdrawal" column, or
#: a statement that prints every charge unsigned). Negated on the way in.
SIGN_NEGATE = "negate"
#: Debit/credit pair, customer's view: the debit column is money out. The common case.
SIGN_DEBIT_NEGATIVE = "debit_negative"
#: Debit/credit pair, bookkeeping view: the debit column is money in. Rare, and exactly the
#: kind of reversal that makes every total wrong while every total still looks reasonable.
SIGN_DEBIT_POSITIVE = "debit_positive"

SIGN_CONVENTIONS: tuple[str, ...] = (
    SIGN_AS_WRITTEN,
    SIGN_NEGATE,
    SIGN_DEBIT_NEGATIVE,
    SIGN_DEBIT_POSITIVE,
)
#: The conventions that require a debit/credit column pair rather than one amount column.
PAIRED_SIGNS: tuple[str, ...] = (SIGN_DEBIT_NEGATIVE, SIGN_DEBIT_POSITIVE)

# -- how a negative is written ------------------------------------------------------------
#
# Accounting notations are *lexical*, not semantic — "(1,234.56)" means minus one thousand
# two hundred etc. in every ledger ever printed, so reading it is parsing, not inferring.
# They are still opt-in, because the failure of a missing opt-in is the safe one: an
# undeclared "(50.00)" raises with its row index, where a helpfully-guessed one would flip a
# charge into a deposit and balance to a plausible wrong number.

#: A leading ``-`` (or ``+``). On by default: it is unambiguous in every locale.
NOTATION_MINUS = "minus"
#: Parenthesized accounting negative — ``(1,234.56)``.
NOTATION_PARENS = "parens"
#: Trailing minus — ``1234.56-``. Common in mainframe-era exports.
NOTATION_TRAILING_MINUS = "trailing_minus"

NEGATIVE_NOTATIONS: tuple[str, ...] = (
    NOTATION_MINUS,
    NOTATION_PARENS,
    NOTATION_TRAILING_MINUS,
)

#: Roles a mapping can bind to a column, in report order.
ROLES: tuple[str, ...] = ("date", "description", "amount", "debit", "credit")

# Symbols stripped from around a money cell before it is parsed. Removing a currency mark
# cannot change a value, so this is the one normalization here that needs no declaration.
CURRENCY_SYMBOLS = "$€£¥₹"

_YAML_KEYS = frozenset(
    {
        "date_column",
        "description_column",
        "amount_column",
        "debit_column",
        "credit_column",
        "date_format",
        "sign",
        "currency",
        "negative_notation",
        "decimal_separator",
        "thousands_separator",
        "skip_rows",
        "bank",
        "notes",
    }
)


class MappingError(ValueError):
    """A mapping is incomplete, self-contradictory, or does not fit the table it was used on.

    Subclasses :class:`ValueError` to match :class:`hearth.mcp.files.FileAccessError` and the
    rest of HEARTH's argument-validation errors, so one ``except ValueError`` at a call site
    catches the whole family. Every message names the field and says what to write instead —
    an operator hitting this is mid-way through authoring a mapping, not debugging HEARTH.
    """


@dataclass(frozen=True)
class ColumnMapping:
    """One bank's export layout, stated in full by the operator. Validated on construction.

    Frozen because a mapping is a claim about a file format; mutating one after a table has
    been parsed against it would leave the resulting numbers unattributable. Use
    :meth:`replace_columns` to derive a variant.

    Every field that could change a number is required or has a default that is a *literal
    reading* rather than a guess: ``skip_rows=0`` means "no preamble", ``negative_notation``
    starts at plain minus only. There is deliberately no default for ``sign`` or
    ``date_format``.
    """

    date_column: str
    description_column: str
    date_format: str
    sign: str
    amount_column: str | None = None
    debit_column: str | None = None
    credit_column: str | None = None
    currency: str | None = None
    negative_notation: tuple[str, ...] = (NOTATION_MINUS,)
    decimal_separator: str = "."
    thousands_separator: str = ","
    skip_rows: int = 0
    bank: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        # Validation runs in __post_init__ rather than in a separate validate() the caller
        # must remember: an unvalidated ColumnMapping must not be constructible at all, or
        # "construction fails on an incomplete mapping" becomes a convention instead of a
        # property.
        self.validate()

    # -- validation -----------------------------------------------------------------------

    def validate(self) -> None:
        """Raise :class:`MappingError` unless this mapping is complete and consistent."""
        for field_name in ("date_column", "description_column", "date_format", "sign"):
            if not str(getattr(self, field_name) or "").strip():
                raise MappingError(
                    f"{field_name} is required — a mapping with no {field_name} would make "
                    "HEARTH choose one, and choosing is exactly what it must not do"
                )

        if self.sign not in SIGN_CONVENTIONS:
            raise MappingError(
                f"sign must be one of {SIGN_CONVENTIONS}, got {self.sign!r} — it states "
                "which direction of money this bank writes as positive; HEARTH normalizes "
                "to money-in-positive and cannot work that out from the data"
            )

        self._validate_amount_columns()
        self._validate_date_format()
        self._validate_notation()
        self._validate_separators()

        if self.currency is not None:
            code = self.currency.strip()
            if len(code) != 3 or not code.isalpha():
                raise MappingError(
                    f"currency must be a three-letter code (e.g. USD) or omitted, got "
                    f"{self.currency!r}"
                )

        if self.skip_rows < 0:
            raise MappingError(f"skip_rows must be >= 0, got {self.skip_rows}")

    def _validate_amount_columns(self) -> None:
        """Enforce exactly one of: a signed amount column, or a debit/credit pair."""
        has_amount = bool((self.amount_column or "").strip())
        has_debit = bool((self.debit_column or "").strip())
        has_credit = bool((self.credit_column or "").strip())
        paired = self.sign in PAIRED_SIGNS

        if has_amount and (has_debit or has_credit):
            raise MappingError(
                "set either amount_column or the debit_column/credit_column pair, never "
                "both — with both present there is no stated rule for which one wins, and "
                "picking one is a guess"
            )
        if not has_amount and not (has_debit and has_credit):
            if has_debit or has_credit:
                missing = "credit_column" if has_debit else "debit_column"
                raise MappingError(
                    f"a debit/credit mapping needs both columns; {missing} is missing. A row "
                    "whose only populated cell is in the column you did not map would parse "
                    "as a zero"
                )
            raise MappingError(
                "no amount source: set amount_column, or both debit_column and credit_column"
            )
        if paired and has_amount:
            raise MappingError(
                f"sign={self.sign!r} describes a debit/credit pair but amount_column is set; "
                f"use {SIGN_AS_WRITTEN!r} or {SIGN_NEGATE!r} for a single amount column"
            )
        if not paired and not has_amount:
            raise MappingError(
                f"sign={self.sign!r} describes a single amount column but a debit/credit "
                f"pair was given; use one of {PAIRED_SIGNS}"
            )

    def _validate_date_format(self) -> None:
        """Reject a format that cannot round-trip a date — most usefully, one with no year.

        A format like ``%m/%d`` parses without complaint and silently dates every row to
        1900, which then sails through a period check and lands in the wrong month bucket.
        """
        probe = datetime.date(2026, 3, 4)
        try:
            rendered = probe.strftime(self.date_format)
            parsed = datetime.datetime.strptime(rendered, self.date_format).date()
        except (ValueError, TypeError) as exc:
            raise MappingError(
                f"date_format {self.date_format!r} is not a usable strptime format: {exc}"
            ) from None
        if parsed != probe:
            raise MappingError(
                f"date_format {self.date_format!r} loses information — a date written with "
                f"it reads back as {parsed.isoformat()}, not {probe.isoformat()}. It most "
                "likely has no year, which would silently date every row to 1900"
            )

    def _validate_notation(self) -> None:
        """Reject unknown or empty negative notations."""
        if not self.negative_notation:
            raise MappingError(
                "negative_notation cannot be empty — with no notation enabled, no negative "
                f"amount is parseable at all; the baseline is {(NOTATION_MINUS,)!r}"
            )
        unknown = [n for n in self.negative_notation if n not in NEGATIVE_NOTATIONS]
        if unknown:
            raise MappingError(
                f"unknown negative_notation {unknown} — known notations are "
                f"{list(NEGATIVE_NOTATIONS)}"
            )

    def _validate_separators(self) -> None:
        """Reject separator choices that make a number ambiguous rather than readable."""
        if len(self.decimal_separator) != 1:
            raise MappingError(
                f"decimal_separator must be a single character, got "
                f"{self.decimal_separator!r}"
            )
        if self.thousands_separator and len(self.thousands_separator) != 1:
            raise MappingError(
                f"thousands_separator must be a single character (or empty for none), got "
                f"{self.thousands_separator!r}"
            )
        if self.decimal_separator == self.thousands_separator:
            raise MappingError(
                f"decimal_separator and thousands_separator are both "
                f"{self.decimal_separator!r}; nothing could be read unambiguously"
            )
        if self.decimal_separator.isdigit() or self.thousands_separator.isdigit():
            raise MappingError("separators cannot be digits")

    # -- accessors ------------------------------------------------------------------------

    @property
    def uses_pair(self) -> bool:
        """True when amounts come from a debit/credit column pair."""
        return self.sign in PAIRED_SIGNS

    def columns(self) -> dict[str, str]:
        """Return ``role -> column name`` for every role this mapping binds."""
        bound = {
            "date": self.date_column,
            "description": self.description_column,
            "amount": self.amount_column,
            "debit": self.debit_column,
            "credit": self.credit_column,
        }
        return {role: name.strip() for role, name in bound.items() if name and name.strip()}

    def replace_columns(self, **changes: Any) -> ColumnMapping:
        """Return a validated copy with ``changes`` applied (``dataclasses.replace``)."""
        return replace(self, **changes)

    # -- serialization --------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return the YAML-shaped dict this mapping round-trips through."""
        out: dict[str, Any] = {
            "date_column": self.date_column,
            "description_column": self.description_column,
            "date_format": self.date_format,
            "sign": self.sign,
        }
        for name in ("amount_column", "debit_column", "credit_column", "currency"):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        out["negative_notation"] = list(self.negative_notation)
        out["decimal_separator"] = self.decimal_separator
        out["thousands_separator"] = self.thousands_separator
        if self.skip_rows:
            out["skip_rows"] = self.skip_rows
        for name in ("bank", "notes"):
            if getattr(self, name):
                out[name] = getattr(self, name)
        return out

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> ColumnMapping:
        """Build a mapping from a plain dict, refusing unknown keys.

        An unknown key is an error rather than a shrug because the likely cause is a typo —
        ``ammount_column:`` — and ignoring it would fall through to "no amount source" at
        best, or to a *different* mapped column at worst.
        """
        if not isinstance(obj, dict):
            raise MappingError(f"a mapping must be a YAML mapping (dict), got {type(obj).__name__}")
        unknown = sorted(set(obj) - _YAML_KEYS)
        if unknown:
            raise MappingError(
                f"unknown mapping key(s) {unknown} — known keys are {sorted(_YAML_KEYS)}. A "
                "misspelled key would otherwise be silently ignored"
            )
        for required in ("date_column", "description_column", "date_format", "sign"):
            if required not in obj:
                raise MappingError(f"mapping is missing required key {required!r}")

        notation = obj.get("negative_notation", [NOTATION_MINUS])
        if isinstance(notation, str):
            notation = [notation]
        try:
            notation = tuple(str(n) for n in notation)
        except TypeError:
            raise MappingError(
                f"negative_notation must be a list of names from {list(NEGATIVE_NOTATIONS)}"
            ) from None

        return cls(
            date_column=str(obj["date_column"]),
            description_column=str(obj["description_column"]),
            date_format=str(obj["date_format"]),
            sign=str(obj["sign"]),
            amount_column=_opt_str(obj.get("amount_column")),
            debit_column=_opt_str(obj.get("debit_column")),
            credit_column=_opt_str(obj.get("credit_column")),
            currency=_opt_str(obj.get("currency")),
            negative_notation=notation,
            decimal_separator=str(obj.get("decimal_separator", ".")),
            thousands_separator=str(obj.get("thousands_separator", ",")),
            skip_rows=int(obj.get("skip_rows", 0)),
            bank=str(obj.get("bank", "")),
            notes=str(obj.get("notes", "")),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ColumnMapping:
        """Load a mapping from a YAML file — written once per bank, kept beside the exports."""
        import yaml  # local: keeps the module's import graph to the stdlib at import time

        text = Path(path).expanduser().read_text(encoding="utf-8")
        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise MappingError(f"{path}: not valid YAML: {exc}") from None
        if raw is None:
            raise MappingError(f"{path}: file is empty — a mapping must state every field")
        return cls.from_dict(raw)


def _opt_str(value: Any) -> str | None:
    """Coerce an optional YAML scalar to a stripped string, or ``None`` when absent/blank."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


# -- header reporting ---------------------------------------------------------------------


@dataclass(frozen=True)
class HeaderReport:
    """What a real header row looks like next to a mapping — a report, never a suggestion.

    This is the authoring aid. It answers "which columns have I not accounted for yet?" and
    "does the mapping I wrote actually fit this file?" without ever proposing that *this*
    column looks like an amount. ``unmapped`` in particular is the operator's to-do list;
    ``missing`` and ``ambiguous`` are hard problems that will stop a parse.
    """

    header: tuple[str, ...]
    mapped: dict[str, str]
    unmapped: tuple[str, ...]
    missing: dict[str, str]
    ambiguous: tuple[str, ...]

    @property
    def fits(self) -> bool:
        """True when every mapped column exists exactly once in this header."""
        return not self.missing and not self.ambiguous

    def describe(self) -> str:
        """Render the report as operator-readable lines."""
        lines = [f"header: {len(self.header)} column(s)"]
        for role in ROLES:
            if role in self.mapped:
                lines.append(f"  {role:<12} -> {self.mapped[role]!r}")
        if self.missing:
            lines.append("  MISSING (mapped, but not in this header):")
            lines.extend(f"    {role} -> {name!r}" for role, name in sorted(self.missing.items()))
        if self.ambiguous:
            lines.append(
                "  AMBIGUOUS (this header repeats a mapped name): "
                + ", ".join(repr(n) for n in self.ambiguous)
            )
        if self.unmapped:
            lines.append("  unmapped columns (you decide whether any of these matter):")
            lines.extend(f"    {name!r}" for name in self.unmapped)
        else:
            lines.append("  unmapped columns: none")
        return "\n".join(lines)


def normalize_header(header: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Strip surrounding whitespace from each header cell. That is the whole normalization.

    No case-folding, no punctuation stripping, no aliasing. Every softening of the match is a
    step towards HEARTH choosing the column for you, and a header that does not match should
    fail loudly so the operator fixes the mapping rather than discovering later that
    ``Amount`` and ``Amount (USD)`` were treated as the same thing.
    """
    return tuple(str(cell).strip() for cell in header)


def inspect_header(
    header: list[str] | tuple[str, ...], mapping: ColumnMapping | None = None
) -> HeaderReport:
    """Report how ``mapping`` lines up with ``header``; with no mapping, list every column.

    Call it with ``mapping=None`` on a fresh export to see the columns you have to account
    for, then again once the mapping is written to confirm it fits before any number is
    computed from it.
    """
    cells = normalize_header(header)
    bound = mapping.columns() if mapping is not None else {}

    counts: dict[str, int] = {}
    for cell in cells:
        counts[cell] = counts.get(cell, 0) + 1

    mapped = {role: name for role, name in bound.items() if counts.get(name, 0) == 1}
    missing = {role: name for role, name in bound.items() if counts.get(name, 0) == 0}
    ambiguous = tuple(sorted({name for name in bound.values() if counts.get(name, 0) > 1}))

    claimed = set(bound.values())
    seen: set[str] = set()
    unmapped: list[str] = []
    for cell in cells:
        if cell in claimed or cell in seen:
            continue
        seen.add(cell)
        unmapped.append(cell)

    return HeaderReport(
        header=cells,
        mapped=mapped,
        unmapped=tuple(unmapped),
        missing=missing,
        ambiguous=ambiguous,
    )


def mapping_template(header: list[str] | tuple[str, ...]) -> str:
    """Emit a blank mapping YAML stub listing ``header``'s columns as comments.

    Deliberately blank. Pre-filling ``amount_column`` from something that looks like an
    amount is the single most attractive convenience in this package and the one that would
    reintroduce the failure mode it exists to prevent, so the stub hands the operator the
    vocabulary and the column list and stops there.
    """
    cells = normalize_header(header)
    lines = [
        "# HEARTH column mapping. Every field below is a decision; there are no defaults",
        "# that guess. See src/hearth/finance/mapping.py for what each one means.",
        "#",
        "# Columns found in this file:",
    ]
    lines.extend(f"#   - {cell!r}" for cell in cells)
    lines += [
        "",
        "bank: ",
        "date_column: ",
        "description_column: ",
        "",
        "# Exactly one of: amount_column, or both debit_column and credit_column.",
        "amount_column: ",
        "# debit_column: ",
        "# credit_column: ",
        "",
        "# strptime format, exact. Must include a year.",
        "date_format: ",
        "",
        f"# One of: {', '.join(SIGN_CONVENTIONS)}",
        "sign: ",
        "",
        f"# Any of: {', '.join(NEGATIVE_NOTATIONS)}",
        "negative_notation:",
        f"  - {NOTATION_MINUS}",
        "",
        "decimal_separator: '.'",
        "thousands_separator: ','",
        "# currency: USD",
        "# skip_rows: 0",
    ]
    return "\n".join(lines) + "\n"


__all__ = [
    "CURRENCY_SYMBOLS",
    "NEGATIVE_NOTATIONS",
    "NOTATION_MINUS",
    "NOTATION_PARENS",
    "NOTATION_TRAILING_MINUS",
    "PAIRED_SIGNS",
    "ROLES",
    "SIGN_AS_WRITTEN",
    "SIGN_CONVENTIONS",
    "SIGN_DEBIT_NEGATIVE",
    "SIGN_DEBIT_POSITIVE",
    "SIGN_NEGATE",
    "ColumnMapping",
    "HeaderReport",
    "MappingError",
    "inspect_header",
    "mapping_template",
    "normalize_header",
]
