"""Reconciliation — arithmetic that proves the parse, checkable without a model.

This is the load-bearing module. Everything upstream of it is a *claim* that a file was read
correctly; this is where the claim is tested against something the file did not produce
itself. Three properties define it:

**It is arithmetic, not inference.** Every figure in a :class:`Reconciliation` is a count, a
sum, or a comparison of two exact :class:`~decimal.Decimal` values. No model is consulted, no
threshold is tuned, and :meth:`Reconciliation.describe` renders it as lines a person can check
by hand against a statement. A validation step a model performs is not a validation step.

**It fails closed.** ``rows_read`` is supplied by the caller from the *table*
(:func:`hearth.finance.parse.data_row_count`), not derived from the parsed records, so
"every row was parsed" is a real comparison and not a list measured against itself. A control
total that misses by a cent fails; there is no default tolerance, and any tolerance is stated
by the operator.

**It never implies a check it did not perform.** With no control total supplied, the sum is
reported as :data:`SUM_UNVERIFIED` and says so in every rendering. A total that nothing was
compared against is an unchecked number, and calling it "reconciled" would be the same class
of lie as a silently skipped row.

What it can prove, given a control total: that the rows in the file were all read, that the
amounts read from them sum to the figure the bank printed, and that every date falls in the
period claimed. What it cannot prove, and does not pretend to: that the file is complete
(a statement missing a page reconciles perfectly against its own printed total), that the
mapping named the right columns (a mapping that reads a "Balance" column as the amount can
still tie out against a control total derived from that same column), or that any flagged
duplicate is or is not a genuine repeated charge.
"""

from __future__ import annotations

import datetime
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal

from .parse import Transaction

ZERO = Decimal("0")

#: The sum was compared against an operator-supplied control total and matched.
SUM_VERIFIED = "verified"
#: It was compared and did not match. The reconciliation fails.
SUM_MISMATCH = "mismatch"
#: No control total was supplied. The sum is a number nobody checked, and says so.
SUM_UNVERIFIED = "unverified"


class ReconciliationError(ValueError):
    """Raised by :func:`require_pass` when a reconciliation did not pass.

    Separate from the report itself so that inspecting a failed reconciliation is always
    possible, while a pipeline that must not proceed on unverified numbers gets one call that
    stops it.
    """


@dataclass(frozen=True)
class DuplicateGroup:
    """Rows sharing a date, description and amount — flagged, never removed.

    Genuine repeats exist: two coffees on one day, a split payment, a subscription billed
    twice after a plan change. Dropping them would understate the total by exactly the amount
    that looked suspicious, so this package surfaces the group and its row indices and leaves
    the judgement where the knowledge is.
    """

    date: datetime.date
    description: str
    amount: Decimal
    row_indices: tuple[int, ...]

    @property
    def count(self) -> int:
        """How many rows are in this group (always at least two)."""
        return len(self.row_indices)


@dataclass(frozen=True)
class Reconciliation:
    """The result of checking one parsed table. Every field is a count, a sum or a comparison.

    ``passed`` is derived rather than stored, so a report cannot be constructed claiming a
    pass it does not support.
    """

    rows_read: int
    rows_parsed: int
    total: Decimal
    control_total: Decimal | None = None
    tolerance: Decimal = ZERO
    sum_status: str = SUM_UNVERIFIED
    duplicates: tuple[DuplicateGroup, ...] = ()
    first_date: datetime.date | None = None
    last_date: datetime.date | None = None
    period_start: datetime.date | None = None
    period_end: datetime.date | None = None
    out_of_period: tuple[int, ...] = ()
    credits: Decimal = ZERO
    debits: Decimal = ZERO
    problems: tuple[str, ...] = field(default_factory=tuple)

    @property
    def difference(self) -> Decimal | None:
        """``total - control_total``, or ``None`` when nothing was supplied to compare."""
        if self.control_total is None:
            return None
        return self.total - self.control_total

    @property
    def sum_verified(self) -> bool:
        """True only when a control total was supplied *and* matched."""
        return self.sum_status == SUM_VERIFIED

    @property
    def all_rows_parsed(self) -> bool:
        """True when the table's data rows and the parsed records are the same count."""
        return self.rows_read == self.rows_parsed

    @property
    def passed(self) -> bool:
        """Overall pass/fail.

        Fails on a row count mismatch, a control-total mismatch, or a date outside the stated
        period. Deliberately does **not** fail on duplicates (they are frequently real) or on
        an absent control total (that is not a failed check, it is no check — reported as
        such rather than dressed up as either outcome).
        """
        return (
            self.all_rows_parsed
            and self.sum_status != SUM_MISMATCH
            and not self.out_of_period
            and not self.problems
        )

    def describe(self) -> str:
        """Render the reconciliation as lines an operator can check against a statement."""
        verdict = "PASS" if self.passed else "FAIL"
        if self.sum_status == SUM_UNVERIFIED:
            verdict += "  (sum UNVERIFIED — no control total was supplied)"
        lines = [
            f"reconciliation: {verdict}",
            f"  rows read      : {self.rows_read}",
            f"  rows parsed    : {self.rows_parsed}"
            + ("" if self.all_rows_parsed else "   <- MISMATCH"),
            f"  sum of amounts : {self.total}",
            f"  money in       : {self.credits}",
            f"  money out      : {self.debits}",
        ]
        if self.control_total is None:
            lines.append(
                "  control total  : none supplied — the sum above has NOT been checked "
                "against anything"
            )
        else:
            lines.append(f"  control total  : {self.control_total}  [{self.sum_status}]")
            lines.append(f"  difference     : {self.difference}")
            if self.tolerance != ZERO:
                lines.append(f"  tolerance      : {self.tolerance}")
        if self.first_date and self.last_date:
            lines.append(f"  date range     : {self.first_date} .. {self.last_date}")
        if self.period_start or self.period_end:
            lines.append(f"  expected period: {self.period_start} .. {self.period_end}")
        if self.out_of_period:
            lines.append(f"  OUT OF PERIOD  : row(s) {list(self.out_of_period)}")
        if self.duplicates:
            lines.append(f"  duplicates     : {len(self.duplicates)} group(s), flagged not removed")
            for group in self.duplicates:
                lines.append(
                    f"    {group.date} {group.description!r} {group.amount} "
                    f"x{group.count} rows {list(group.row_indices)}"
                )
        else:
            lines.append("  duplicates     : none")
        for problem in self.problems:
            lines.append(f"  PROBLEM        : {problem}")
        return "\n".join(lines)


def control_total_from_balances(opening: Decimal, closing: Decimal) -> Decimal:
    """Return the control total implied by two statement balances: ``closing - opening``.

    Both must already be :class:`~decimal.Decimal`. Passing floats is refused rather than
    coerced — a control total built from a binary float is not the figure printed on the
    statement, and it is the one number in the system that has to be exact.
    """
    for name, value in (("opening", opening), ("closing", closing)):
        if not isinstance(value, Decimal):
            raise TypeError(
                f"{name} balance must be a Decimal, got {type(value).__name__} — build it "
                'from the printed string, e.g. Decimal("1284.37")'
            )
    return closing - opening


def find_duplicates(transactions: Sequence[Transaction]) -> tuple[DuplicateGroup, ...]:
    """Group transactions sharing date, description and amount exactly.

    Matching is exact on all three: no fuzzy description matching and no date window. A near
    match is a judgement about whether two merchant strings mean the same thing, which is
    precisely the inference this package does not make.

    Amount equality is Decimal equality, so ``12.50`` and ``12.5`` group together — they are
    the same quantity of money written two ways, which is a fact and not an opinion.
    """
    buckets: dict[tuple[datetime.date, str, Decimal], list[int]] = {}
    for txn in transactions:
        buckets.setdefault((txn.date, txn.description, txn.amount), []).append(txn.row_index)
    groups = [
        DuplicateGroup(date=key[0], description=key[1], amount=key[2], row_indices=tuple(rows))
        for key, rows in buckets.items()
        if len(rows) > 1
    ]
    groups.sort(key=lambda g: (g.date, g.description, g.amount, g.row_indices))
    return tuple(groups)


def reconcile(
    transactions: Iterable[Transaction],
    *,
    rows_read: int,
    control_total: Decimal | None = None,
    tolerance: Decimal = ZERO,
    period_start: datetime.date | None = None,
    period_end: datetime.date | None = None,
) -> Reconciliation:
    """Check a parsed table and return a :class:`Reconciliation`. Never raises on a bad result.

    ``rows_read`` is required and keyword-only: it comes from the table
    (:func:`hearth.finance.parse.data_row_count`), so the row-count check compares the parse
    against the file rather than against itself. Defaulting it to ``len(transactions)`` would
    turn the single most important invariant here into a tautology.

    ``control_total`` is the figure to check the sum against — a printed statement total, or
    :func:`control_total_from_balances` of the opening and closing balances. Omitting it is
    allowed and is reported as :data:`SUM_UNVERIFIED`; it is never reported as a pass.

    ``tolerance`` is an absolute Decimal, default exact zero. Money adds up exactly; a
    tolerance is the operator saying they know why it will not, and it appears in the report.
    """
    txns = list(transactions)

    problems: list[str] = []
    if rows_read < 0:
        problems.append(f"rows_read is negative ({rows_read})")
    for name, value in (("control_total", control_total), ("tolerance", tolerance)):
        if value is not None and not isinstance(value, Decimal):
            problems.append(
                f"{name} must be a Decimal, got {type(value).__name__}; a float control total "
                "is not the figure printed on the statement"
            )
    if isinstance(tolerance, Decimal) and tolerance < ZERO:
        problems.append(f"tolerance must be >= 0, got {tolerance}")

    total = sum((t.amount for t in txns), ZERO)
    credits = sum((t.amount for t in txns if t.amount > ZERO), ZERO)
    debits = sum((-t.amount for t in txns if t.amount < ZERO), ZERO)

    if control_total is None or problems:
        sum_status = SUM_UNVERIFIED
    else:
        sum_status = (
            SUM_VERIFIED if abs(total - control_total) <= tolerance else SUM_MISMATCH
        )

    dates = sorted(t.date for t in txns)
    first_date = dates[0] if dates else None
    last_date = dates[-1] if dates else None

    out_of_period = tuple(
        t.row_index
        for t in txns
        if (period_start is not None and t.date < period_start)
        or (period_end is not None and t.date > period_end)
    )

    return Reconciliation(
        rows_read=rows_read,
        rows_parsed=len(txns),
        total=total,
        control_total=control_total,
        tolerance=tolerance if isinstance(tolerance, Decimal) else ZERO,
        sum_status=sum_status,
        duplicates=find_duplicates(txns),
        first_date=first_date,
        last_date=last_date,
        period_start=period_start,
        period_end=period_end,
        out_of_period=tuple(sorted(out_of_period)),
        credits=credits,
        debits=debits,
        problems=tuple(problems),
    )


def require_pass(result: Reconciliation, *, require_control_total: bool = False) -> Reconciliation:
    """Return ``result`` if it passed, else raise :class:`ReconciliationError` with the report.

    ``require_control_total`` promotes :data:`SUM_UNVERIFIED` from "reported" to "refused" —
    the setting for a pipeline that must never act on a sum nothing was compared against.
    It is opt-in rather than the default so that the two states stay distinguishable:
    *unchecked* and *checked and wrong* are different facts and this package never conflates
    them.
    """
    if require_control_total and result.sum_status == SUM_UNVERIFIED:
        raise ReconciliationError(
            "the sum was never checked against a control total, and this caller requires "
            f"one:\n{result.describe()}"
        )
    if not result.passed:
        raise ReconciliationError(f"reconciliation failed:\n{result.describe()}")
    return result


__all__ = [
    "SUM_MISMATCH",
    "SUM_UNVERIFIED",
    "SUM_VERIFIED",
    "DuplicateGroup",
    "Reconciliation",
    "ReconciliationError",
    "control_total_from_balances",
    "find_duplicates",
    "reconcile",
    "require_pass",
]
