#!/usr/bin/env python3
"""Draft a :class:`ColumnMapping` per statement format — locally, with the values staying put.

Authoring one mapping per bank by hand is the friction in ``docs/RUNBOOK_finance.md`` §2, and
it is friction a *cloud* agent structurally cannot remove: deciding whether ``03/04/2026`` is
March or April, or whether a debit column carries magnitudes, requires looking at the
operator's actual transactions. A local model may look at them freely. That asymmetry is the
entire reason HEARTH exists, and this script is it applied to the one job that needs it.

**The rule this script is built around.** HEARTH may parse structure but must never *infer*
financial semantics, because a mis-read amount column yields a plausible number rather than an
error (``src/hearth/finance/mapping.py``). A model **drafting** a mapping does not break that
rule — provided nothing unverified is ever used. So the labour is split three ways and the
split is visible in every file this writes:

1. **Mechanically decidable → decided in code, never asked of the model.** The date format
   above all: every date cell of every row is scanned, and the format is whichever *single*
   candidate parses all of them. If two candidates both parse every row, the answer is
   **AMBIGUOUS** and the draft will not load until a human picks one. This is the strictest
   treatment in the file and it is deliberate: **reconciliation cannot catch a wrong date
   format.** Sums do not depend on dates, so ``%m/%d`` where ``%d/%m`` was meant passes every
   arithmetic check while silently moving transactions between months. It is the one semantic
   error with no downstream gate. Also mechanical: which columns are numeric, whether
   accounting negatives are present, the decimal/thousands separators, whether the file has a
   debit/credit pair or one signed column, and which columns are entirely empty.

2. **The model proposes the judgement calls** — which numeric column is the transaction amount
   rather than a running balance, which text column is the description, which of two date
   columns dates the transaction, and a human-readable name for the format. It runs locally
   (``hearth.providers.mlx.MLXProvider``) at temperature 0, is shown **headers and structural
   facts only** — never a cell — and **every field it returns is validated against the
   mechanical facts before use**: a column name it did not copy exactly, or one whose measured
   type contradicts the role, is discarded rather than written.

3. **Verified before it is written.** A complete draft is trial-parsed against the real files
   with :func:`~hearth.finance.parse.parse_rows` and reconciled with
   :func:`~hearth.finance.validate.reconcile`. A draft that cannot parse its own file is
   **not written** — it is reported instead. No control total is ever invented, so the sum is
   reported as UNVERIFIED, in those words, in the draft.

Anything still unresolved is written into the YAML as a sentinel (``AMBIGUOUS`` /
``UNRESOLVED``) that :meth:`ColumnMapping.from_yaml` **refuses to load**. A draft that needs a
human is therefore inert until it gets one, rather than being a plausible file that runs.

**Privacy.** The tool reads values locally — that is the point — but it never prints one. Only
headers, measured types, counts and the model's structural conclusions reach stdout, because
that output is exactly what an operator is likely to paste into a cloud chat. This extends to
failures: :class:`~hearth.finance.parse.ParseError` messages quote the offending cell, so a
trial-parse failure is reported by row index and column and the message is withheld. The
computed total goes into the draft file (local, and where a reviewer needs it) and reaches
stdout only under ``--show-total``.

    HEARTH_FILE_ROOTS=~/hearth-statements \\
        uv run --no-sync python scripts/hearth_map_draft.py ~/hearth-statements/incoming

Reads go through :func:`hearth.mcp.files.read_table`, so the ``HEARTH_FILE_ROOTS`` allowlist
applies here as everywhere else; this script has no privileges of its own.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from hearth.config import get_settings  # noqa: E402
from hearth.finance.mapping import (  # noqa: E402
    NEGATIVE_NOTATIONS,
    NOTATION_MINUS,
    NOTATION_PARENS,
    NOTATION_TRAILING_MINUS,
    SIGN_AS_WRITTEN,
    SIGN_DEBIT_NEGATIVE,
    SIGN_DEBIT_POSITIVE,
    SIGN_NEGATE,
    ColumnMapping,
    MappingError,
    normalize_header,
)
from hearth.finance.parse import (  # noqa: E402
    ParseError,
    data_row_count,
    parse_money,
    parse_rows,
)
from hearth.finance.validate import SUM_UNVERIFIED, reconcile  # noqa: E402
from hearth.mcp.files import FileAccessError, read_table  # noqa: E402

TABLE_EXTS = {".csv", ".xlsx", ".json"}

#: Written into any field that is not settled. Both are deliberately *not* loadable: a draft
#: that still needs a human must fail loudly at ``from_yaml``, not run with a plausible guess.
AMBIGUOUS = "AMBIGUOUS"
UNRESOLVED = "UNRESOLVED"

# -- mechanical determination: dates --------------------------------------------------------
#
# The candidate list is closed and explicit. A format is accepted only if it parses EVERY date
# cell in EVERY file of the format, which is why the scan is exhaustive rather than sampled: a
# single 25/12 in row 900 is what separates day-first from month-first, and a sampler that
# misses it reports a confident wrong answer.

DATE_CANDIDATES: tuple[str, ...] = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y.%m.%d",
    "%Y-%m-%d %H:%M:%S",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%m-%d-%Y",
    "%d-%m-%Y",
    "%m.%d.%Y",
    "%d.%m.%Y",
    "%m/%d/%y",
    "%d/%m/%y",
    "%d-%b-%Y",
    "%d %b %Y",
    "%b %d, %Y",
    "%d %B %Y",
    "%B %d, %Y",
    "%Y%m%d",
)

# A bank statement is not from the year 3 or the year 9999. The bound exists because strptime
# is lenient about widths — "01/02/03" parses under %m/%d/%Y as the year 3 — and a candidate
# that only survives by inventing an impossible year is not a candidate.
_MIN_YEAR = 1900
_MAX_YEAR = 2100

#: Numeric date shapes, for reporting *why* a format was chosen (counts only, never a cell).
_NUMERIC_DATE = re.compile(r"^\s*(\d{1,4})([-/.])(\d{1,4})\2(\d{1,4})(?:[ T].*)?\s*$")

DATE_DECIDED = "decided"
DATE_AMBIGUOUS = "ambiguous"
DATE_UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class DateFinding:
    """What an exhaustive scan of one column's date cells established — and what it did not.

    ``candidates`` holds every format that parsed *all* cells. One is a decision; two or more
    is the ambiguity that must reach the operator intact, because nothing downstream can catch
    it. ``evidence`` says which observation settled it, in counts.
    """

    status: str
    candidates: tuple[str, ...]
    cells_scanned: int
    evidence: str

    @property
    def chosen(self) -> str | None:
        """The single format, or ``None`` when the scan did not settle on exactly one."""
        return self.candidates[0] if self.status == DATE_DECIDED else None


def _parses_every(cells: list[str], fmt: str) -> bool:
    """True when ``fmt`` parses every cell to a plausible statement year."""
    for cell in cells:
        try:
            when = datetime.datetime.strptime(cell, fmt)
        except ValueError:
            return False
        if not _MIN_YEAR <= when.year <= _MAX_YEAR:
            return False
    return True


def _shape_evidence(cells: list[str]) -> str:
    """Describe, in counts, what the cells' components rule out. Never quotes a cell."""
    lead_year = first_over_12 = second_over_12 = 0
    for cell in cells:
        match = _NUMERIC_DATE.match(cell)
        if not match:
            continue
        first, _, second, _third = match.groups()
        if len(first) == 4:
            lead_year += 1
        elif int(first) > 12:
            first_over_12 += 1
        if len(first) != 4 and int(second) > 12:
            second_over_12 += 1
    parts = []
    if lead_year:
        parts.append(f"{lead_year} cell(s) lead with a 4-digit year")
    if first_over_12:
        parts.append(f"{first_over_12} cell(s) have a first component > 12 (day-first)")
    if second_over_12:
        parts.append(f"{second_over_12} cell(s) have a second component > 12 (month-first)")
    if not parts:
        parts.append("no cell has a component > 12, so nothing in the data separates "
                     "day-first from month-first")
    return "; ".join(parts)


def detect_date_format(cells: list[str]) -> DateFinding:
    """Determine the strptime format of a date column from **every** cell it holds.

    Returns :data:`DATE_AMBIGUOUS` rather than a preference whenever more than one candidate
    survives the whole column. Picking the popular one here would be the single most damaging
    guess this script could make: sums are indifferent to dates, so a wrong choice reconciles
    perfectly and quietly reassigns transactions to other months. There is no arithmetic gate
    downstream of this decision, so it is the operator's.
    """
    seen = [c.strip() for c in cells if str(c).strip()]
    if not seen:
        return DateFinding(DATE_UNRESOLVED, (), 0, "no populated cells")
    survivors = tuple(fmt for fmt in DATE_CANDIDATES if _parses_every(seen, fmt))
    evidence = _shape_evidence(seen)
    if not survivors:
        return DateFinding(
            DATE_UNRESOLVED,
            (),
            len(seen),
            f"no candidate format parses all {len(seen)} cell(s); {evidence}",
        )
    if len(survivors) == 1:
        return DateFinding(
            DATE_DECIDED,
            survivors,
            len(seen),
            f"exactly one candidate parses all {len(seen)} cell(s); {evidence}",
        )
    return DateFinding(
        DATE_AMBIGUOUS,
        survivors,
        len(seen),
        f"{len(survivors)} candidates parse all {len(seen)} cell(s); {evidence}",
    )


# -- mechanical determination: numbers ------------------------------------------------------

KIND_EMPTY = "empty"
KIND_DATE = "date"
KIND_NUMBER = "number"
KIND_TEXT = "text"

#: The separator pairs tried, in order. The first that parses every numeric cell wins; two
#: pairs cannot both parse the same cell set except where the cells are unambiguous anyway,
#: because :func:`~hearth.finance.parse.parse_money` requires grouping in strict threes.
SEPARATOR_PAIRS: tuple[tuple[str, str], ...] = ((".", ","), (",", "."))


def _probe_mapping(decimal_sep: str, thousands_sep: str) -> ColumnMapping:
    """A throwaway mapping used only to reach :func:`parse_money`'s number grammar.

    Reusing the real parser to answer "is this a number?" keeps one definition of a number in
    the codebase. A second regex here would drift from the one that computes the totals, and
    the drift would show up as a column classified as numeric that then fails to parse.
    """
    return ColumnMapping(
        date_column="_probe_date",
        description_column="_probe_description",
        date_format="%Y-%m-%d",
        sign=SIGN_AS_WRITTEN,
        amount_column="_probe_amount",
        negative_notation=NEGATIVE_NOTATIONS,
        decimal_separator=decimal_sep,
        thousands_separator=thousands_sep,
    )


def _number_or_none(cell: str, probe: ColumnMapping) -> Decimal | None:
    """Parse one cell as money under ``probe``, or ``None`` if it is not a number."""
    try:
        return parse_money(cell, probe)
    except ParseError:
        return None


@dataclass(frozen=True)
class ColumnProfile:
    """One column measured across every row of every file in its format.

    ``numbers`` is excluded from ``repr`` on purpose: it holds parsed cell values, and a
    dataclass repr is exactly the sort of thing that ends up in a log line or a traceback.
    """

    index: int
    name: str
    kind: str
    populated: int
    rows: int
    date: DateFinding | None = None
    negatives: int = 0
    positives: int = 0
    zeros: int = 0
    parens: bool = False
    trailing_minus: bool = False
    numbers: tuple[Decimal | None, ...] = field(default=(), repr=False)

    @property
    def fully_populated(self) -> bool:
        """True when no row leaves this column blank."""
        return self.rows > 0 and self.populated == self.rows

    def describe(self) -> str:
        """One safe line for the operator: header, measured kind and counts only."""
        bits = [f"type={self.kind}", f"populated={self.populated}/{self.rows}"]
        if self.kind == KIND_NUMBER:
            bits.append(f"negative={self.negatives}")
            bits.append(f"zero={self.zeros}")
            if self.parens:
                bits.append("accounting-negatives=parens")
            if self.trailing_minus:
                bits.append("accounting-negatives=trailing-minus")
        if self.kind == KIND_DATE and self.date is not None:
            bits.append(f"date-format={self.date.status}")
        return f"[{self.index}] {self.name!r:<32} " + "  ".join(bits)


def _classify(name: str, index: int, cells: list[str], probe: ColumnMapping) -> ColumnProfile:
    """Measure one column: kind, fill, sign counts and negative notation. No inference."""
    populated = [c for c in cells if str(c).strip()]
    if not populated:
        return ColumnProfile(index=index, name=name, kind=KIND_EMPTY, populated=0, rows=len(cells))

    date = detect_date_format(cells)
    if date.status in (DATE_DECIDED, DATE_AMBIGUOUS):
        return ColumnProfile(
            index=index,
            name=name,
            kind=KIND_DATE,
            populated=len(populated),
            rows=len(cells),
            date=date,
        )

    numbers = tuple(_number_or_none(str(c), probe) if str(c).strip() else None for c in cells)
    if all(n is not None for c, n in zip(cells, numbers, strict=True) if str(c).strip()):
        stripped = [str(c).strip() for c in populated]
        return ColumnProfile(
            index=index,
            name=name,
            kind=KIND_NUMBER,
            populated=len(populated),
            rows=len(cells),
            negatives=sum(1 for n in numbers if n is not None and n < 0),
            positives=sum(1 for n in numbers if n is not None and n > 0),
            zeros=sum(1 for n in numbers if n is not None and n == 0),
            parens=any(c.startswith("(") and c.endswith(")") for c in stripped),
            trailing_minus=any(c.endswith("-") for c in stripped),
            numbers=numbers,
        )

    return ColumnProfile(
        index=index, name=name, kind=KIND_TEXT, populated=len(populated), rows=len(cells)
    )


def detect_separators(columns: list[list[str]]) -> tuple[str, str, str]:
    """Return ``(decimal, thousands, evidence)`` — the pair under which every number parses.

    Tried rather than assumed, because ``1,50`` under a comma-thousands mapping is refused by
    the parser instead of quietly becoming one hundred and fifty, and that refusal is the
    signal. If both conventions parse everything the file contains no grouped number at all;
    the default is then a statement about the notation, not about the file, and says so.
    """
    scored: list[tuple[int, str, str]] = []
    for decimal_sep, thousands_sep in SEPARATOR_PAIRS:
        probe = _probe_mapping(decimal_sep, thousands_sep)
        parsed = sum(
            1
            for column in columns
            for cell in column
            if str(cell).strip() and _number_or_none(str(cell), probe) is not None
        )
        scored.append((parsed, decimal_sep, thousands_sep))
    best, runner_up = sorted(scored, reverse=True)[:2]
    if best[0] > runner_up[0]:
        return (
            best[1],
            best[2],
            f"{best[0]} cell(s) parse as numbers under {best[1]!r}/{best[2]!r} against "
            f"{runner_up[0]} under {runner_up[1]!r}/{runner_up[2]!r}",
        )
    return ".", ",", (
        "no number in this file parses under one convention and not the other, so this is a "
        "statement about the default notation and not about the file"
    )


# -- mechanical determination: amount structure ---------------------------------------------


@dataclass(frozen=True)
class AmountStructure:
    """The amount sources the *file's shape* permits — never which one means what.

    ``pairs`` are column pairs that behave like a debit/credit split: both carry magnitudes
    only, and no row fills both. ``singles`` are numeric columns present on every row, which
    is the shape of a signed amount — and equally the shape of a running balance. Telling
    those two apart is a judgement, so it is not made here.
    """

    pairs: tuple[tuple[str, str], ...]
    singles: tuple[str, ...]
    evidence: tuple[str, ...]


def detect_amount_structure(columns: list[ColumnProfile]) -> AmountStructure:
    """Find every debit/credit-shaped pair and every signed-amount-shaped column."""
    numeric = [c for c in columns if c.kind == KIND_NUMBER]
    pairs: list[tuple[str, str]] = []
    evidence: list[str] = []

    for i, left in enumerate(numeric):
        for right in numeric[i + 1 :]:
            if left.negatives or right.negatives:
                continue
            both = sum(
                1
                for a, b in zip(left.numbers, right.numbers, strict=False)
                if a not in (None, 0) and b not in (None, 0)
            )
            covered = sum(
                1
                for a, b in zip(left.numbers, right.numbers, strict=False)
                if a not in (None, 0) or b not in (None, 0)
            )
            if both == 0 and covered == left.rows:
                pairs.append((left.name, right.name))
                evidence.append(
                    f"columns {left.name!r} and {right.name!r} carry magnitudes only and no "
                    f"row fills both — the shape of a debit/credit split ({covered} row(s) "
                    "covered)"
                )

    singles = tuple(c.name for c in numeric if c.fully_populated)
    for name in singles:
        evidence.append(f"column {name!r} is numeric on every row — the shape of a signed "
                        "amount column, and equally of a running balance")
    return AmountStructure(pairs=tuple(pairs), singles=singles, evidence=tuple(evidence))


# -- mechanical evidence: a running balance -------------------------------------------------


@dataclass(frozen=True)
class BalanceEvidence:
    """A numeric column whose successive differences track an amount source, row for row.

    This is arithmetic, not opinion: either ``balance[i] - balance[i-1]`` equals the amount on
    row *i* or it does not. What it cannot establish is that the column *is* the account
    balance — that reading is the operator's, and every rendering of this says so.
    """

    balance_column: str
    source: tuple[str, ...]
    sign: str
    matched: int
    comparable: int

    @property
    def source_label(self) -> str:
        """The source's columns as one readable name."""
        return " + ".join(self.source)

    @property
    def total(self) -> bool:
        """True when every comparable row agreed."""
        return self.comparable > 0 and self.matched == self.comparable


def _source_series(
    columns: dict[str, ColumnProfile], source: tuple[str, ...], sign: str
) -> list[Decimal | None]:
    """Render one candidate amount source as a signed series, per the sign convention."""
    if len(source) == 1:
        values = columns[source[0]].numbers
        return [None if v is None else (v if sign == SIGN_AS_WRITTEN else -v) for v in values]
    debit, credit = columns[source[0]].numbers, columns[source[1]].numbers
    out: list[Decimal | None] = []
    for d, c in zip(debit, credit, strict=False):
        d_value = d or Decimal(0)
        c_value = c or Decimal(0)
        out.append(c_value - d_value if sign == SIGN_DEBIT_NEGATIVE else d_value - c_value)
    return out


def detect_balance_evidence(
    columns: list[ColumnProfile],
    structure: AmountStructure,
    spans: list[tuple[int, int]],
) -> tuple[BalanceEvidence, ...]:
    """Compare each fully-populated numeric column's row-to-row change against each source.

    ``spans`` are the per-file row ranges: a difference taken across a file boundary compares
    two unrelated statements, so those rows are simply not comparable and are skipped rather
    than counted as disagreements.
    """
    by_name = {c.name: c for c in columns}
    balances = [c for c in columns if c.kind == KIND_NUMBER and c.fully_populated]
    sources: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
        ((name,), (SIGN_AS_WRITTEN, SIGN_NEGATE)) for name in structure.singles
    ]
    sources += [(pair, (SIGN_DEBIT_NEGATIVE, SIGN_DEBIT_POSITIVE)) for pair in structure.pairs]

    found: list[BalanceEvidence] = []
    for balance in balances:
        for source, signs in sources:
            if source == (balance.name,):
                continue
            for sign in signs:
                series = _source_series(by_name, source, sign)
                matched = comparable = 0
                for start, end in spans:
                    for i in range(start + 1, end):
                        previous, current = balance.numbers[i - 1], balance.numbers[i]
                        amount = series[i]
                        if previous is None or current is None or amount is None:
                            continue
                        comparable += 1
                        if current - previous == amount:
                            matched += 1
                if comparable and matched == comparable:
                    found.append(
                        BalanceEvidence(
                            balance_column=balance.name,
                            source=source,
                            sign=sign,
                            matched=matched,
                            comparable=comparable,
                        )
                    )
    return tuple(found)


# -- the format profile ---------------------------------------------------------------------


@dataclass
class FormatProfile:
    """Everything measured about one header signature, across every file that shares it."""

    signature: tuple[str, ...]
    files: list[Path]
    rows: int
    columns: list[ColumnProfile]
    decimal_separator: str
    thousands_separator: str
    separator_evidence: str
    structure: AmountStructure
    balance_evidence: tuple[BalanceEvidence, ...]

    def by_name(self, name: str) -> ColumnProfile | None:
        """Look up a measured column by header name, or ``None``."""
        return next((c for c in self.columns if c.name == name), None)

    @property
    def negative_notation(self) -> tuple[str, ...]:
        """The notations this file actually uses. Lexical, so reading them is not inference."""
        notation = [NOTATION_MINUS]
        if any(c.parens for c in self.columns):
            notation.append(NOTATION_PARENS)
        if any(c.trailing_minus for c in self.columns):
            notation.append(NOTATION_TRAILING_MINUS)
        return tuple(notation)


def profile_files(paths: list[Path], settings: Any | None = None) -> FormatProfile:
    """Measure a group of same-header files as one format. Every row of every file is read."""
    header: tuple[str, ...] = ()
    grid: list[list[str]] = []
    spans: list[tuple[int, int]] = []
    for path in paths:
        rows = read_table(path, settings)
        if not rows:
            continue
        if not header:
            header = normalize_header(rows[0])
        start = len(grid)
        grid.extend(rows[1:])
        spans.append((start, len(grid)))

    width = len(header)
    columns_cells = [[row[i] if i < len(row) else "" for row in grid] for i in range(width)]
    decimal_sep, thousands_sep, separator_evidence = detect_separators(columns_cells)
    probe = _probe_mapping(decimal_sep, thousands_sep)
    columns = [
        _classify(name, i, columns_cells[i], probe) for i, name in enumerate(header)
    ]
    structure = detect_amount_structure(columns)
    return FormatProfile(
        signature=header,
        files=list(paths),
        rows=len(grid),
        columns=columns,
        decimal_separator=decimal_sep,
        thousands_separator=thousands_sep,
        separator_evidence=separator_evidence,
        structure=structure,
        balance_evidence=detect_balance_evidence(columns, structure, spans),
    )


# -- the model's half ------------------------------------------------------------------------


class Proposer(Protocol):
    """Anything that can answer a structural question about a header. Injectable for tests."""

    name: str

    def propose(self, prompt: str) -> str:
        """Return the model's raw text for ``prompt``."""


class ModelUnavailable(RuntimeError):
    """The local model could not be loaded (no mlx, no weights). Never fatal here."""


class LocalModel:
    """The local MLX model, at temperature 0, used for judgement calls only.

    Temperature 0 because a mapping is not a creative act: the same file must produce the same
    proposal, or "the operator reviewed the draft" stops meaning anything. The provider is
    injectable so the tests exercise this wiring without weights on disk.
    """

    def __init__(self, model_id: str, provider: Any | None = None) -> None:
        self.model_id = model_id
        self.name = model_id
        self._provider = provider

    def _load(self) -> Any:
        if self._provider is not None:
            return self._provider
        try:
            from hearth.providers.mlx import MLXProvider, mlx_available
        except ImportError as exc:  # pragma: no cover - import guard
            raise ModelUnavailable(str(exc)) from None
        if not mlx_available():
            raise ModelUnavailable("mlx-lm is not installed (uv sync --extra mlx)")
        self._provider = MLXProvider(self.model_id)
        return self._provider

    def propose(self, prompt: str) -> str:
        """Run one completion locally, deterministically."""
        from hearth.providers.base import GenRequest, Message

        provider = self._load()
        request = GenRequest(
            messages=[Message(role="user", content=prompt)],
            model=self.model_id,
            max_tokens=400,
            temperature=0.0,
        )
        try:
            return provider.generate(request).text
        except ModelUnavailable:
            raise
        except Exception as exc:  # a missing snapshot surfaces from deep inside mlx-lm
            raise ModelUnavailable(f"{type(exc).__name__}: {exc}") from None


def build_prompt(profile: FormatProfile) -> str:
    """Render the structural question for the model. Contains no cell and no date format.

    The date format is absent by construction, not by instruction: it was settled in code
    before this prompt existed, and a model asked for it would answer plausibly for a file it
    cannot check. What is left here are the genuine judgement calls — which numeric column is
    the transaction rather than the balance, which text column reads as a description.
    """
    lines = [
        "You are labelling the columns of a bank-statement export for a strict parser.",
        "You are given the column headers and facts measured from the file. No cell values",
        "are shown and none are needed.",
        "",
        f"Columns ({profile.rows} data row(s) across {len(profile.files)} file(s)):",
    ]
    lines += [f"  {c.describe()}" for c in profile.columns]
    if profile.structure.evidence or profile.balance_evidence:
        lines += ["", "Measured facts:"]
        lines += [f"  - {fact}" for fact in profile.structure.evidence]
        lines += [
            f"  - column {e.balance_column!r} changes by exactly the value of "
            f"{e.source_label!r} on all {e.comparable} comparable row(s)"
            for e in profile.balance_evidence
        ]
    date_columns = [c.name for c in profile.columns if c.kind == KIND_DATE]
    lines += [
        "",
        "Reply with ONE JSON object and nothing else:",
        '{"format_name": "...", "date_column": "...", "description_column": "...",',
        ' "amount_column": null, "debit_column": null, "credit_column": null}',
        "",
        "Rules:",
        "  - Copy every column name EXACTLY from the list above, or use null.",
        "  - Set EITHER amount_column, OR both debit_column and credit_column.",
        "  - The amount column holds the transaction, not a running account balance.",
        "  - The debit column is money leaving the account.",
        f"  - date_column must be one of: {date_columns}. Pick the one that dates the",
        "    transaction.",
        "  - format_name: two to four lowercase hyphenated words naming this layout.",
    ]
    return "\n".join(lines)


_JSON_KEYS = (
    "format_name",
    "date_column",
    "description_column",
    "amount_column",
    "debit_column",
    "credit_column",
)


def parse_proposal(text: str) -> dict[str, str | None]:
    """Extract the model's JSON object, or return an empty proposal. Never raises.

    A model that answered with prose, or with a key we did not ask for, has not produced a
    usable proposal — and an unusable proposal is the same as no model at all, which is a
    supported state. It is never worth salvaging by guessing what it meant.
    """
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        raw = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str | None] = {}
    for key in _JSON_KEYS:
        value = raw.get(key)
        out[key] = str(value).strip() if isinstance(value, (str, int, float)) else None
    return out


def validate_proposal(
    proposal: dict[str, str | None], profile: FormatProfile
) -> dict[str, str | None]:
    """Keep only the proposed fields the measurements support. Everything else is dropped.

    This is where "the model may propose, but nothing unverified is used" is enforced. A
    column name the model invented, or one whose measured type contradicts the role it was
    given, is discarded — leaving the field UNRESOLVED for a human, which is a worse draft to
    look at and a much better one to trust.
    """
    kept: dict[str, str | None] = {}

    date_name = proposal.get("date_column")
    date_column = profile.by_name(date_name) if date_name else None
    if date_column is not None and date_column.kind == KIND_DATE:
        kept["date_column"] = date_column.name

    description = proposal.get("description_column")
    description_column = profile.by_name(description) if description else None
    if description_column is not None and description_column.kind in (KIND_TEXT, KIND_DATE):
        kept["description_column"] = description_column.name

    amount = proposal.get("amount_column")
    debit, credit = proposal.get("debit_column"), proposal.get("credit_column")
    if amount and amount in profile.structure.singles and not (debit or credit):
        kept["amount_column"] = amount
    elif debit and credit and (debit, credit) in profile.structure.pairs:
        kept["debit_column"], kept["credit_column"] = debit, credit
    elif debit and credit and (credit, debit) in profile.structure.pairs:
        kept["debit_column"], kept["credit_column"] = debit, credit

    name = proposal.get("format_name")
    if name:
        kept["format_name"] = name
    return kept


# -- assembling a draft -----------------------------------------------------------------------

SOURCE_MECHANICAL = "mechanical"
SOURCE_MODEL = "model"
SOURCE_CONFIRM = "confirm"

VERIFIED = "verified"
VERIFIED_EACH = "verified under every candidate for the fields left open"
NOT_ATTEMPTED = "not attempted"
FAILED = "failed"


@dataclass(frozen=True)
class Provenance:
    """One field of a draft, and where its value came from. Rendered into the YAML header."""

    field_name: str
    value: str
    source: str
    note: str


@dataclass(frozen=True)
class Verification:
    """The result of trial-parsing a draft against the real files it was drafted from."""

    status: str
    rows_read: int = 0
    rows_parsed: int = 0
    total: Decimal | None = None
    detail: str = ""

    @property
    def blocks_write(self) -> bool:
        """A draft that could be tried and failed is not written. That is the whole gate."""
        return self.status == FAILED


@dataclass
class Draft:
    """A drafted mapping for one format: its fields, their provenance, and its verification."""

    profile: FormatProfile
    name: str
    fields: dict[str, Any]
    provenance: list[Provenance]
    confirm: list[str]
    verification: Verification
    mapping: ColumnMapping | None = None

    @property
    def complete(self) -> bool:
        """True when no field is still a sentinel — i.e. the draft loads as a mapping."""
        return not any(
            isinstance(v, str) and v in (AMBIGUOUS, UNRESOLVED) for v in self.fields.values()
        )


def _first_or_none(values: tuple[str, ...]) -> str | None:
    return values[0] if len(values) == 1 else None


def draft_for_profile(
    profile: FormatProfile,
    proposer: Proposer | None,
    *,
    settings: Any | None = None,
    rank: int = 1,
) -> Draft:
    """Draft one mapping: mechanics first, the model second, verification last."""
    provenance: list[Provenance] = []
    confirm: list[str] = []
    fields: dict[str, Any] = {}

    proposal: dict[str, str | None] = {}
    model_note = ""
    if proposer is not None:
        try:
            raw = proposer.propose(build_prompt(profile))
        except ModelUnavailable as exc:
            model_note = f"the local model was unavailable ({exc}); no field below is a model "\
                         "proposal"
        else:
            proposal = validate_proposal(parse_proposal(raw), profile)
            if not proposal:
                model_note = ("the local model answered, but nothing it proposed survived "
                              "validation against the measurements")
    else:
        model_note = "run without a model (--no-model); no field below is a model proposal"

    # -- date column, then its format (mechanical, always) --------------------------------
    date_candidates = tuple(c.name for c in profile.columns if c.kind == KIND_DATE)
    forced_date = _first_or_none(date_candidates)
    if forced_date is not None:
        fields["date_column"] = forced_date
        provenance.append(
            Provenance("date_column", forced_date, SOURCE_MECHANICAL,
                       "the only column whose every cell parses as a date")
        )
    elif proposal.get("date_column"):
        fields["date_column"] = proposal["date_column"]
        provenance.append(
            Provenance("date_column", str(proposal["date_column"]), SOURCE_MODEL,
                       f"chosen among {list(date_candidates)}; posting vs transaction date "
                       "is a judgement")
        )
        confirm.append(f"date_column — {len(date_candidates)} columns parse as dates")
    else:
        fields["date_column"] = UNRESOLVED
        provenance.append(
            Provenance("date_column", UNRESOLVED, SOURCE_CONFIRM,
                       f"candidates: {list(date_candidates) or 'none'}")
        )
        confirm.append("date_column — not settled")

    chosen_date = profile.by_name(str(fields["date_column"]))
    finding = chosen_date.date if chosen_date is not None else None
    if finding is not None and finding.status == DATE_DECIDED:
        fields["date_format"] = finding.chosen
        provenance.append(
            Provenance("date_format", str(finding.chosen), SOURCE_MECHANICAL, finding.evidence)
        )
    elif finding is not None and finding.status == DATE_AMBIGUOUS:
        fields["date_format"] = AMBIGUOUS
        provenance.append(
            Provenance(
                "date_format",
                AMBIGUOUS,
                SOURCE_CONFIRM,
                f"{list(finding.candidates)} all parse every cell — {finding.evidence}",
            )
        )
        confirm.append(
            f"date_format — AMBIGUOUS between {list(finding.candidates)}. Nothing downstream "
            "can catch a wrong choice: sums do not depend on dates, so the wrong one "
            "reconciles perfectly and files transactions in the wrong months"
        )
    else:
        fields["date_format"] = UNRESOLVED
        provenance.append(
            Provenance("date_format", UNRESOLVED, SOURCE_CONFIRM,
                       finding.evidence if finding else "no date column was settled")
        )
        confirm.append("date_format — no candidate format parses every row")

    # -- description ------------------------------------------------------------------------
    text_columns = tuple(c.name for c in profile.columns if c.kind == KIND_TEXT)
    forced_text = _first_or_none(text_columns)
    if forced_text is not None:
        fields["description_column"] = forced_text
        provenance.append(
            Provenance("description_column", forced_text, SOURCE_MECHANICAL,
                       "the only non-numeric, non-date, non-empty column in this file")
        )
    elif proposal.get("description_column"):
        fields["description_column"] = proposal["description_column"]
        provenance.append(
            Provenance("description_column", str(proposal["description_column"]), SOURCE_MODEL,
                       f"chosen among {list(text_columns)}")
        )
        confirm.append("description_column — a model proposal")
    else:
        fields["description_column"] = UNRESOLVED
        provenance.append(
            Provenance("description_column", UNRESOLVED, SOURCE_CONFIRM,
                       f"text columns: {list(text_columns) or 'none'}")
        )
        confirm.append("description_column — not settled")

    # -- amount source and sign ---------------------------------------------------------------
    _draft_amount(profile, proposal, fields, provenance, confirm)

    # -- lexical, mechanical, unconditional ---------------------------------------------------
    fields["negative_notation"] = list(profile.negative_notation)
    provenance.append(
        Provenance("negative_notation", str(list(profile.negative_notation)), SOURCE_MECHANICAL,
                   "the notations that actually occur in this file; a notation not listed "
                   "here is refused rather than reinterpreted")
    )
    fields["decimal_separator"] = profile.decimal_separator
    fields["thousands_separator"] = profile.thousands_separator
    provenance.append(
        Provenance("decimal_separator/thousands_separator",
                   f"{profile.decimal_separator!r}/{profile.thousands_separator!r}",
                   SOURCE_MECHANICAL, profile.separator_evidence)
    )
    fields["skip_rows"] = 0
    provenance.append(
        Provenance("skip_rows", "0", SOURCE_MECHANICAL,
                   "the allowlisted reader returned the header as the first row; preamble "
                   "above a header is NOT detected by this tool — check it yourself")
    )

    name = _slug(str(proposal.get("format_name") or "")) or f"format-{rank}"
    if proposal.get("format_name"):
        provenance.append(
            Provenance("(file name)", name, SOURCE_MODEL, "a label only; it changes no number")
        )

    mapping, verification = _verify(profile, fields, finding, settings=settings)
    if model_note:
        confirm.insert(0, model_note)
    return Draft(
        profile=profile,
        name=name,
        fields=fields,
        provenance=provenance,
        confirm=confirm,
        verification=verification,
        mapping=mapping,
    )


def _sole_balance_evidence(profile: FormatProfile) -> BalanceEvidence | None:
    """Return the one amount source a running-balance column tracks exactly, if there is one.

    ``balance[i] - balance[i-1] == amount[i]`` on every comparable row is arithmetic, and when
    exactly one source satisfies it the file has effectively told us which columns carry the
    transaction and which way the money runs. Two sources satisfying it would mean the file
    contains the same series twice, and that is a choice, not a measurement — so it falls
    through to the model.
    """
    evidenced = [e for e in profile.balance_evidence if e.total]
    sources = {e.source for e in evidenced}
    if len(sources) != 1:
        return None
    return evidenced[0]


def _apply_balance_evidence(
    evidence: BalanceEvidence,
    fields: dict[str, Any],
    provenance: list[Provenance],
    confirm: list[str],
) -> None:
    """Write the source and sign a running balance settles.

    ``debit=A, credit=B, sign=debit_negative`` and ``debit=B, credit=A, sign=debit_positive``
    describe the same file, so a reversal is normalized to the customer's view: the pair is
    always written with the money-out column as ``debit_column``.
    """
    if len(evidence.source) == 2:
        left, right = evidence.source
        debit, credit = (left, right) if evidence.sign == SIGN_DEBIT_NEGATIVE else (right, left)
        fields["debit_column"], fields["credit_column"] = debit, credit
        fields["sign"] = SIGN_DEBIT_NEGATIVE
        value = f"{debit} / {credit} / {SIGN_DEBIT_NEGATIVE}"
        field_name = "debit_column/credit_column/sign"
        detail = f"changes by exactly {credit!r} minus {debit!r}"
    else:
        fields["amount_column"] = evidence.source[0]
        fields["sign"] = evidence.sign
        value = f"{evidence.source[0]} / {evidence.sign}"
        field_name = "amount_column/sign"
        detail = f"changes by exactly this column's value under {evidence.sign!r}"

    provenance.append(
        Provenance(field_name, value, SOURCE_MECHANICAL,
                   f"column {evidence.balance_column!r} {detail} on all {evidence.comparable} "
                   "comparable row(s), and no other arrangement of this file's numeric columns "
                   f"does — which also rules {evidence.balance_column!r} out as the amount. It "
                   f"fixes the DIRECTION only if {evidence.balance_column!r} is the account "
                   "balance")
    )
    confirm.append(
        f"sign — settled by treating {evidence.balance_column!r} as a running account balance. "
        "Confirm that reading; reversed, every total is right in size and wrong in direction"
    )


def _draft_amount(
    profile: FormatProfile,
    proposal: dict[str, str | None],
    fields: dict[str, Any],
    provenance: list[Provenance],
    confirm: list[str],
) -> None:
    """Settle the amount source *and* the sign convention — they are one decision, not two.

    Which column holds the amount and which direction of money the file writes as positive
    cannot be separated: naming a debit column already states the direction. So they are
    drafted together, and ``sign`` lands in the confirm list whatever the outcome — no control
    total is ever invented here, and without one the arithmetic downstream cannot tell a
    reversed convention from a correct one.
    """
    structure = profile.structure
    evidence = _sole_balance_evidence(profile)
    if evidence is not None:
        _apply_balance_evidence(evidence, fields, provenance, confirm)
        return

    only_pair = structure.pairs[0] if len(structure.pairs) == 1 and not structure.singles else None
    only_single = _first_or_none(structure.singles) if not structure.pairs else None

    if only_pair is not None:
        if proposal.get("debit_column") and proposal.get("credit_column"):
            _model_pair(proposal, fields, provenance, confirm, note="the file's only pair")
            return
        fields["debit_column"] = UNRESOLVED
        fields["credit_column"] = UNRESOLVED
        fields["sign"] = UNRESOLVED
        provenance.append(
            Provenance("debit_column/credit_column/sign", UNRESOLVED, SOURCE_CONFIRM,
                       f"columns {list(only_pair)} are a debit/credit-shaped pair (measured), "
                       "but nothing in the file says which of the two is money LEAVING the "
                       "account. Write the money-out column as debit_column, the other as "
                       f"credit_column, and sign: {SIGN_DEBIT_NEGATIVE}")
        )
        confirm.append(
            f"which of {list(only_pair)} is money out — the pair is measured, the direction is "
            "yours"
        )
        return

    if only_single is not None:
        fields["amount_column"] = only_single
        provenance.append(
            Provenance("amount_column", only_single, SOURCE_MECHANICAL,
                       "the only numeric column populated on every row")
        )
    elif proposal.get("amount_column"):
        fields["amount_column"] = proposal["amount_column"]
        provenance.append(
            Provenance("amount_column", str(proposal["amount_column"]), SOURCE_MODEL,
                       f"chosen among {list(structure.singles)}; a running balance has the "
                       "same shape as a signed amount, which is why this is a proposal")
        )
        confirm.append(
            f"amount_column is {proposal['amount_column']!r} — a model proposal. A balance "
            "column read as the amount still sums, and still reconciles against a control "
            "total derived from itself"
        )
    elif proposal.get("debit_column") and proposal.get("credit_column"):
        _model_pair(proposal, fields, provenance, confirm,
                    note=f"chosen among {[list(p) for p in structure.pairs]}")
        return
    else:
        fields["amount_column"] = UNRESOLVED
        fields["sign"] = UNRESOLVED
        provenance.append(
            Provenance("amount_column", UNRESOLVED, SOURCE_CONFIRM,
                       f"single-column candidates: {list(structure.singles) or 'none'}; "
                       f"debit/credit pairs: {[list(p) for p in structure.pairs] or 'none'}")
        )
        confirm.append("amount_column and sign — not settled; no source could be named")
        return

    _draft_single_sign(profile, fields, provenance, confirm)


def _model_pair(
    proposal: dict[str, str | None],
    fields: dict[str, Any],
    provenance: list[Provenance],
    confirm: list[str],
    *,
    note: str,
) -> None:
    """Record a model-named debit/credit pair. The sign follows from the naming, definitionally."""
    debit, credit = str(proposal["debit_column"]), str(proposal["credit_column"])
    fields["debit_column"], fields["credit_column"] = debit, credit
    fields["sign"] = SIGN_DEBIT_NEGATIVE
    provenance.append(
        Provenance("debit_column/credit_column/sign", f"{debit} / {credit} / "
                   f"{SIGN_DEBIT_NEGATIVE}", SOURCE_MODEL,
                   f"{note}; the model was asked which column is money LEAVING the account, "
                   "and the sign convention follows from that answer rather than being a "
                   "second choice")
    )
    confirm.append(
        f"debit_column is {debit!r} — a model proposal, and it alone decides the direction of "
        "every amount. Reversed, every total is right in size and wrong in direction, and no "
        "arithmetic here can see it"
    )


def _draft_single_sign(
    profile: FormatProfile,
    fields: dict[str, Any],
    provenance: list[Provenance],
    confirm: list[str],
) -> None:
    """Choose the sign convention for a single amount column, or leave it for the operator."""
    name = str(fields.get("amount_column", ""))
    column = profile.by_name(name)

    if column is not None and column.negatives and column.positives:
        fields["sign"] = SIGN_AS_WRITTEN
        note = (f"presumed: this column holds {column.negatives} negative and "
                f"{column.positives} positive value(s), so it is already signed and "
                f"{SIGN_AS_WRITTEN!r} takes it verbatim. If a negative here means money IN, "
                f"the convention is {SIGN_NEGATE!r}")
    elif column is not None and not column.negatives:
        fields["sign"] = UNRESOLVED
        note = ("this column has no negative value at all, so it states magnitudes and the "
                "direction of each transaction is not in it — say which way it goes")
    else:
        fields["sign"] = UNRESOLVED
        note = "no amount column was settled"

    provenance.append(Provenance("sign", str(fields["sign"]), SOURCE_CONFIRM, note))
    confirm.append(
        f"sign={fields['sign']} — {note}. Nothing here checked it: no control total was "
        "supplied, and without one the arithmetic cannot see a reversed convention"
    )


def _verify(
    profile: FormatProfile,
    fields: dict[str, Any],
    finding: DateFinding | None,
    *,
    settings: Any | None = None,
) -> tuple[ColumnMapping | None, Verification]:
    """Trial-parse the draft against the real files, and reconcile what comes out.

    A draft that is complete must survive this or it is not written. Two fields are allowed to
    be unsettled and still verified — the date format and the description column — because
    **neither changes a number**: the draft is then tried under every combination of the
    surviving candidates, which proves the arithmetic half without pretending either question
    was answered. Anything else unsettled means there is no mapping to try, and this says so
    rather than reporting a pass it did not earn.
    """
    unsettled = [
        k for k, v in fields.items() if isinstance(v, str) and v in (AMBIGUOUS, UNRESOLVED)
    ]
    if set(unsettled) - {"date_format", "description_column"}:
        return None, Verification(
            NOT_ATTEMPTED,
            detail=f"unsettled field(s) {sorted(unsettled)} — there is no complete mapping to try",
        )

    formats = (
        list(finding.candidates)
        if "date_format" in unsettled and finding is not None
        else [str(fields["date_format"])]
    )
    descriptions = (
        [c.name for c in profile.columns if c.kind == KIND_TEXT]
        if "description_column" in unsettled
        else [str(fields["description_column"])]
    )
    combinations = [(f, d) for f in formats for d in descriptions]
    if not combinations:
        return None, Verification(
            NOT_ATTEMPTED,
            detail=f"unsettled field(s) {sorted(unsettled)} have no candidate to stand in for "
                   "them, so there is no mapping to try",
        )

    last: ColumnMapping | None = None
    rows_read = rows_parsed = 0
    total: Decimal | None = None
    for date_format, description in combinations:
        candidate = dict(fields)
        candidate["date_format"] = date_format
        candidate["description_column"] = description
        try:
            mapping = ColumnMapping.from_dict({k: v for k, v in candidate.items() if v != ""})
        except MappingError as exc:
            return None, Verification(FAILED, detail=f"mapping is invalid: {exc}")

        rows_read = rows_parsed = 0
        transactions = []
        for path in profile.files:
            try:
                rows = read_table(path, settings)
                parsed = parse_rows(rows, mapping)
            except ParseError as exc:
                # ParseError messages quote the offending cell. Report the coordinates only.
                where = f"row {exc.row_index}" if exc.row_index is not None else "the table"
                column = f" column {exc.column!r}" if exc.column else ""
                return None, Verification(
                    FAILED,
                    detail=f"{path.name}: parse failed at {where}{column} (the parser's "
                           "message is withheld: it quotes the cell)",
                )
            except (MappingError, FileAccessError) as exc:
                return None, Verification(FAILED, detail=f"{path.name}: {exc}")
            rows_read += data_row_count(rows, mapping)
            rows_parsed += len(parsed)
            transactions.extend(parsed)

        result = reconcile(transactions, rows_read=rows_read)
        if not result.passed:
            return None, Verification(
                FAILED,
                rows_read=result.rows_read,
                rows_parsed=result.rows_parsed,
                detail=f"reconciliation failed: rows read {result.rows_read} vs parsed "
                       f"{result.rows_parsed}; problems {list(result.problems)}",
            )
        last, total = mapping, result.total

    settled = len(combinations) == 1
    varied = ", ".join(
        part
        for part in (
            f"date format {formats}" if len(formats) > 1 else "",
            f"description column {descriptions}" if len(descriptions) > 1 else "",
        )
        if part
    )
    detail = (
        "parsed and reconciled"
        if settled
        else f"parsed and reconciled under every combination of {varied}; those choices change "
             "no total, which is exactly why nothing here proves them"
    )
    return (last if settled else None), Verification(
        VERIFIED if settled else VERIFIED_EACH,
        rows_read=rows_read,
        rows_parsed=rows_parsed,
        total=total,
        detail=detail,
    )


def _slug(name: str) -> str:
    """Reduce a model-proposed name to a safe file stem.

    The model chooses this string and it becomes a path, so it is stripped to lowercase
    alphanumerics and hyphens — a proposal containing ``../`` must not be able to write
    outside ``--out``.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug[:48]


# -- rendering ---------------------------------------------------------------------------------


def _yaml_scalar(value: Any) -> str:
    """Render a scalar as YAML. JSON strings are valid YAML double-quoted scalars."""
    if isinstance(value, bool | int):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(json.dumps(str(v), ensure_ascii=False) for v in value) + "]"
    return json.dumps(str(value), ensure_ascii=False)


_FIELD_ORDER = (
    "bank",
    "date_column",
    "description_column",
    "amount_column",
    "debit_column",
    "credit_column",
    "date_format",
    "sign",
    "negative_notation",
    "decimal_separator",
    "thousands_separator",
    "skip_rows",
)


def render_draft(draft: Draft) -> str:
    """Render the draft as commented YAML: the fields, and where every one of them came from."""
    profile = draft.profile
    lines = [
        "# ================================================================================",
        "# HEARTH column mapping — DRAFT. NOT CONFIRMED. NOT YET TRUSTED.",
        "# ================================================================================",
        "#",
        f"# Drafted by scripts/hearth_map_draft.py on {datetime.date.today().isoformat()} from",
        f"# {len(profile.files)} file(s) sharing one header, {profile.rows} data row(s).",
        "#",
        "# A drafted mapping is a proposal about what a bank's export means. Read every line",
        "# below before using it: a mis-read amount column yields a plausible number, not an",
        "# error. Fields marked AMBIGUOUS or UNRESOLVED will REFUSE to load until you replace",
        "# them, which is the point.",
        "#",
        "# Files in this format:",
    ]
    lines += [f"#   - {path.name}" for path in profile.files]
    lines += ["#", "# Columns as measured (headers and types only; no value was recorded):"]
    lines += [f"#   {column.describe()}" for column in profile.columns]

    for title, source in (
        ("MECHANICALLY DETERMINED — read off every row, not a guess and not a model output",
         SOURCE_MECHANICAL),
        ("PROPOSED BY THE LOCAL MODEL — a judgement, validated against the measurements above",
         SOURCE_MODEL),
        ("NOT SETTLED — you must decide these", SOURCE_CONFIRM),
    ):
        entries = [p for p in draft.provenance if p.source == source]
        if not entries:
            continue
        lines += ["#", f"# {title}:"]
        for entry in entries:
            lines.append(f"#   {entry.field_name}: {entry.value}")
            lines += [f"#       {chunk}" for chunk in _wrap(entry.note)]

    verification = draft.verification
    lines += ["#", "# VERIFICATION (run against the real files before this draft was written):"]
    lines.append(f"#   status         : {verification.status}")
    if verification.detail:
        lines += [f"#       {chunk}" for chunk in _wrap(verification.detail)]
    if verification.status in (VERIFIED, VERIFIED_EACH):
        lines.append(f"#   rows read      : {verification.rows_read}")
        lines.append(f"#   rows parsed    : {verification.rows_parsed}")
        lines.append(f"#   sum of amounts : {verification.total}   [{SUM_UNVERIFIED}]")
        lines.append("#       No control total was supplied, so this sum has NOT been checked")
        lines.append("#       against anything. Supply one when you ingest.")

    if draft.confirm:
        lines += ["#", "# BEFORE YOU USE THIS FILE, CONFIRM:"]
        for n, item in enumerate(draft.confirm, 1):
            wrapped = _wrap(item)
            lines.append(f"#   {n}. {wrapped[0]}")
            lines += [f"#      {chunk}" for chunk in wrapped[1:]]

    lines += ["", f"bank: {_yaml_scalar('DRAFT — name this account')}"]
    for key in _FIELD_ORDER:
        if key == "bank" or key not in draft.fields:
            continue
        value = draft.fields[key]
        if value in (None, ""):
            continue
        lines.append(f"{key}: {_yaml_scalar(value)}")
    lines.append(
        "notes: " + _yaml_scalar("DRAFT from scripts/hearth_map_draft.py — confirm the fields "
                                 "listed above, then delete this line")
    )
    return "\n".join(lines) + "\n"


def _wrap(text: str, width: int = 84) -> list[str]:
    """Wrap a note for the comment block, keeping every line inside the file's margin."""
    words, out, current = text.split(), [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            out.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        out.append(current)
    return out or [""]


# -- the sweep ----------------------------------------------------------------------------------


def collect(targets: list[str], exts: set[str]) -> list[Path]:
    """Expand files and directories into a sorted list of candidate tables."""
    found: list[Path] = []
    for target in targets:
        path = Path(target).expanduser()
        if path.is_dir():
            found.extend(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in exts)
        elif path.suffix.lower() in exts:
            found.append(path)
    return sorted(set(found))


def group_by_signature(
    paths: list[Path], settings: Any | None = None
) -> tuple[dict[tuple[str, ...], list[Path]], list[tuple[Path, str]]]:
    """Group files by header signature — one mapping is needed per group, not per file."""
    groups: dict[tuple[str, ...], list[Path]] = defaultdict(list)
    refused: list[tuple[Path, str]] = []
    for path in paths:
        try:
            rows = read_table(path, settings)
        except FileAccessError as exc:
            refused.append((path, str(exc)))
            continue
        except Exception as exc:  # a malformed file must not abort the sweep
            refused.append((path, f"{type(exc).__name__}"))
            continue
        if not rows:
            refused.append((path, "no rows"))
            continue
        groups[normalize_header(rows[0])].append(path)
    return dict(groups), refused


def unique_name(name: str, claimed: set[str], rank: int) -> str:
    """Keep two formats in one sweep from claiming the same file name.

    The model names the format, and asked twice about two different layouts it will happily
    answer "bank statement" both times. Without this the second draft collides with the first
    and is reported as "an existing mapping I will not overwrite" — which is true of the path
    and false about the situation.
    """
    return f"{name}-{rank}" if name in claimed else name


def write_draft(draft: Draft, out_dir: Path, *, force: bool) -> tuple[Path, str]:
    """Write one draft, refusing to clobber a reviewed mapping. Returns (path, outcome)."""
    if draft.verification.blocks_write:
        return out_dir / f"{draft.name}.yaml", "refused"
    path = out_dir / f"{draft.name}.yaml"
    if path.exists() and not force:
        return path, "exists"
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(render_draft(draft), encoding="utf-8")
    return path, "written"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Draft a HEARTH column mapping per statement format, locally. "
                    "Never prints a cell value.",
    )
    parser.add_argument("targets", nargs="+", help="files or directories (walked recursively)")
    parser.add_argument(
        "--out",
        default="~/hearth-statements/mappings",
        help="directory to write drafts into (default: ~/hearth-statements/mappings)",
    )
    parser.add_argument("--model", default=None, help="local model id (default: registry default)")
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="mechanical determination only; model-proposed fields stay UNRESOLVED",
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing mapping of the same name"
    )
    parser.add_argument(
        "--show-total",
        action="store_true",
        help="also print the trial-parse total (it is always written into the draft)",
    )
    parser.add_argument("--ext", action="append", default=None, help="limit to this extension")
    args = parser.parse_args(argv)

    exts = {e if e.startswith(".") else f".{e}" for e in (args.ext or [])} or TABLE_EXTS
    files = collect(args.targets, exts)
    if not files:
        print("no matching table files found")
        return 1

    # Resolved once and threaded through, rather than each read reaching for the process
    # settings on its own: one place decides which allowlist this run used.
    settings = get_settings()
    groups, refused = group_by_signature(files, settings)
    print(f"\nscanned {len(files)} file(s): {len(groups)} distinct format(s), "
          f"{len(refused)} unreadable")

    proposer: Proposer | None = None
    if not args.no_model:
        model_id = args.model
        if model_id is None:
            try:
                from hearth.registry import get_registry

                model_id = get_registry().default_id
            except Exception:
                model_id = ""
        proposer = LocalModel(model_id) if model_id else None
        print(f"local model: {model_id or 'none available'}  (temperature 0)")
    else:
        print("local model: disabled (--no-model)")

    out_dir = Path(args.out).expanduser()
    exit_code = 0
    claimed: set[str] = set()
    for rank, (_signature, paths) in enumerate(
        sorted(groups.items(), key=lambda kv: -len(kv[1])), 1
    ):
        profile = profile_files(paths, settings)
        print(f"\n── format {rank} ── {len(paths)} file(s), {profile.rows} data row(s) ──")
        for column in profile.columns:
            print(f"     {column.describe()}")

        draft = draft_for_profile(profile, proposer, settings=settings, rank=rank)
        draft.name = unique_name(draft.name, claimed, rank)
        claimed.add(draft.name)
        path, outcome = write_draft(draft, out_dir, force=args.force)
        print(f"     verification: {draft.verification.status}"
              + (f" — {draft.verification.detail}" if draft.verification.detail else ""))
        if draft.verification.status in (VERIFIED, VERIFIED_EACH):
            print(f"     rows read {draft.verification.rows_read} / "
                  f"parsed {draft.verification.rows_parsed}")
            if args.show_total:
                print(f"     sum of amounts {draft.verification.total}  [{SUM_UNVERIFIED}]")
            else:
                print(f"     sum of amounts: written into the draft [{SUM_UNVERIFIED}] "
                      "(pass --show-total to print it here)")

        if outcome == "written":
            print(f"     draft: {path}")
        elif outcome == "exists":
            print(f"     NOT written: {path} already exists — your reviewed mapping outranks a "
                  "fresh draft (use --force to replace it)")
            exit_code = 1
        else:
            print("     NOT written: this draft did not parse its own files, and a mapping "
                  "that cannot parse its file is worse than none")
            exit_code = 1

        if draft.confirm:
            print("     needs your confirmation:")
            for item in draft.confirm:
                print(f"       - {item}")
        if not draft.complete:
            print("     this draft will REFUSE to load until you replace every AMBIGUOUS / "
                  "UNRESOLVED field")

    if refused:
        print("\n── could not read ──")
        for path, reason in refused:
            print(f"     {path.name}: {reason}")
        exit_code = 1

    print("\nNo cell value was printed above. Every draft is unconfirmed until you read it.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
