"""Deterministic grouping and summing. Every number here is computed by Python, in Decimal.

**A model must never compute a number in this package.** Not "should not" — the whole point of
the two-tier design in ``examples/finance/`` is that arithmetic is done in Python and the model
only ever sees *finished* figures to write prose about. A model asked to add a column will
produce a total that is right often enough to be trusted and wrong often enough to matter, and
nothing downstream can tell the two apart. So this module is plain summation over
:class:`~decimal.Decimal`, with fixed tie-breaks so the same input always yields the same
ordering.

Two boundaries it deliberately does not cross:

**Categories are an input.** Assigning a category to a transaction is a judgement — it is the
job of the tier-1 classifier, or of the operator's own rules. This module reads
:attr:`Transaction.category` and groups by it. It never derives one, and an unset category is
grouped under :data:`UNCATEGORIZED` rather than being guessed at or dropped.

**Merchant strings are not normalized.** :func:`top_merchants` groups by the *exact*
description. Collapsing ``SQ *BLUE BOTTLE #4412`` and ``SQ *BLUE BOTTLE #0031`` into one
merchant requires knowing that the trailing digits are a store number rather than part of the
name — that is empirical merchant knowledge of the kind ``docs/APEX_seam.md`` §5.2 describes,
and a plausible-looking guess at it produces a plausible-looking merchant total. Callers who
own that knowledge pass their own ``key`` function.
"""

from __future__ import annotations

import datetime
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from decimal import Decimal

from .parse import Transaction

ZERO = Decimal("0")

#: The bucket for transactions with no category assigned. Never silently dropped: a spend
#: breakdown that omits what it could not label understates every share it reports.
UNCATEGORIZED = "uncategorized"


@dataclass(frozen=True)
class Totals:
    """Sums over one set of transactions. All Decimal, all computed here.

    ``income`` and ``spend`` are both **positive magnitudes** — ``spend`` is the absolute
    value of the money that left — and ``net`` is ``income - spend``, matching the convention
    the existing ``examples/finance`` harness reports. ``total`` is the plain signed sum, so
    it can be compared directly with a :class:`~hearth.finance.validate.Reconciliation`.
    """

    count: int = 0
    income: Decimal = ZERO
    spend: Decimal = ZERO
    total: Decimal = ZERO
    first_date: datetime.date | None = None
    last_date: datetime.date | None = None

    @property
    def net(self) -> Decimal:
        """Money in minus money out. Equal to :attr:`total` by construction."""
        return self.income - self.spend


@dataclass(frozen=True)
class MerchantTotal:
    """One merchant string and what it came to. ``description`` is verbatim from the file."""

    description: str
    total: Decimal
    count: int


def totals(transactions: Iterable[Transaction]) -> Totals:
    """Sum a set of transactions into income, spend, net and a date range."""
    txns = list(transactions)
    if not txns:
        return Totals()
    income = sum((t.amount for t in txns if t.amount > ZERO), ZERO)
    spend = sum((-t.amount for t in txns if t.amount < ZERO), ZERO)
    dates = sorted(t.date for t in txns)
    return Totals(
        count=len(txns),
        income=income,
        spend=spend,
        total=sum((t.amount for t in txns), ZERO),
        first_date=dates[0],
        last_date=dates[-1],
    )


def by_category(transactions: Iterable[Transaction]) -> dict[str, Decimal]:
    """Return ``category -> signed net``, largest absolute movement first.

    Signed rather than spend-only, so a category holding both a charge and its refund reports
    what actually happened to the balance instead of a spend figure that ignores the refund.
    Ordering is by absolute size then by name, so the result is stable across runs and across
    machines — an ordering that depends on dict insertion is a diff that looks like a change.
    """
    sums: dict[str, Decimal] = {}
    for txn in transactions:
        key = txn.category or UNCATEGORIZED
        sums[key] = sums.get(key, ZERO) + txn.amount
    return {k: sums[k] for k in sorted(sums, key=lambda k: (-abs(sums[k]), k))}


def spend_by_category(transactions: Iterable[Transaction]) -> dict[str, Decimal]:
    """Return ``category -> total money out`` as positive magnitudes, largest first.

    Only negative amounts contribute, so an income category never appears here — a breakdown
    of "where the money went" that includes a salary credit is not a breakdown of spending.
    """
    sums: dict[str, Decimal] = {}
    for txn in transactions:
        if txn.amount >= ZERO:
            continue
        key = txn.category or UNCATEGORIZED
        sums[key] = sums.get(key, ZERO) + -txn.amount
    return {k: sums[k] for k in sorted(sums, key=lambda k: (-sums[k], k))}


def month_key(when: datetime.date) -> str:
    """Return the ``YYYY-MM`` bucket for a date — sortable as a string, by construction."""
    return f"{when.year:04d}-{when.month:02d}"


def by_month(transactions: Iterable[Transaction]) -> dict[str, Totals]:
    """Return ``YYYY-MM -> Totals``, in chronological order.

    Months with no transactions are absent rather than zero-filled: this module reports what
    the data contains, and inventing an empty January would be asserting that the file covers
    it. Callers wanting a dense series supply the calendar themselves.
    """
    grouped: dict[str, list[Transaction]] = {}
    for txn in transactions:
        grouped.setdefault(month_key(txn.date), []).append(txn)
    return {key: totals(grouped[key]) for key in sorted(grouped)}


def top_merchants(
    transactions: Iterable[Transaction],
    limit: int = 10,
    *,
    spend_only: bool = True,
    key: Callable[[Transaction], str] | None = None,
) -> list[MerchantTotal]:
    """Return the biggest merchants by total, grouping on the **exact** description.

    ``key`` overrides the grouping function for callers who own real merchant knowledge — a
    normalizer that strips store numbers and auth codes belongs to whoever can verify it, not
    here. Without one, two spellings of the same shop are two entries; that is visibly wrong
    to a human reading the list and invisibly wrong if this module had merged them by guess.

    ``spend_only`` (the default) considers outflows and reports positive magnitudes, which is
    what "top merchants" means. With it off, the signed net per description is ranked by
    absolute size and income sources appear alongside spend.
    """
    if limit < 0:
        raise ValueError(f"limit must be >= 0, got {limit}")
    group_of = key or (lambda t: t.description)

    sums: dict[str, Decimal] = {}
    counts: dict[str, int] = {}
    for txn in transactions:
        if spend_only and txn.amount >= ZERO:
            continue
        name = group_of(txn)
        sums[name] = sums.get(name, ZERO) + (-txn.amount if spend_only else txn.amount)
        counts[name] = counts.get(name, 0) + 1

    ordered = sorted(sums, key=lambda n: (-abs(sums[n]), n))
    return [
        MerchantTotal(description=name, total=sums[name], count=counts[name])
        for name in ordered[:limit]
    ]


def render_totals(result: Totals) -> str:
    """Render :class:`Totals` as operator-readable lines — figures only, already computed."""
    span = (
        f"{result.first_date} .. {result.last_date}"
        if result.first_date and result.last_date
        else "no transactions"
    )
    return "\n".join(
        [
            f"transactions : {result.count}  ({span})",
            f"money in     : {result.income}",
            f"money out    : {result.spend}",
            f"net          : {result.net}",
        ]
    )


__all__ = [
    "UNCATEGORIZED",
    "MerchantTotal",
    "Totals",
    "by_category",
    "by_month",
    "month_key",
    "render_totals",
    "spend_by_category",
    "top_merchants",
    "totals",
]
