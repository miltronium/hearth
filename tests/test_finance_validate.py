"""Tests for the reconciliation layer (hearth.finance.validate).

This is the module the whole finance path leans on. Everything downstream — categorization,
aggregation, a narrative from a local model — is only as trustworthy as the claim that the
rows were read correctly and that they add up to what the bank says. Reconciliation is
arithmetic, not inference: no model is involved, and none can be, because the failure being
guarded against is precisely the one that yields a confident plausible number.

It arrived untested (the session that wrote it was interrupted). Untested validation code is
a gate nobody has checked, which is the same shape as the bugs this codebase spent the
session removing — so these tests pin the behaviour that makes the gate real.
"""

import datetime
from decimal import Decimal

import pytest

from hearth.finance.parse import Transaction
from hearth.finance.validate import (
    SUM_MISMATCH,
    SUM_UNVERIFIED,
    SUM_VERIFIED,
    Reconciliation,
    ReconciliationError,
    control_total_from_balances,
    find_duplicates,
    reconcile,
    require_pass,
)

D = Decimal


def tx(day: int, description: str, amount: str, row: int, category: str | None = None):
    return Transaction(
        date=datetime.date(2026, 8, day),
        description=description,
        amount=D(amount),
        row_index=row,
        category=category,
    )


def sample():
    """Three debits and one credit; sums to 3050.15 exactly."""
    return [
        tx(2, "SQ *BLUE BOTTLE COFFEE", "-6.75", 1),
        tx(3, "SHELL OIL 574839201", "-52.10", 2),
        tx(5, "ACH DEPOSIT PAYROLL", "3250.00", 3),
        tx(7, "PACIFIC GAS + ELECTRIC", "-141.00", 4),
    ]


# -- the sum: three distinct states, never conflated ----------------------------------------

def test_no_control_total_is_unverified_not_passed_silently():
    """The most important distinction in this module.

    'Nobody checked' and 'checked and correct' must never render the same. A pipeline that
    reports a total as good because nothing contradicted it is the plausible-wrong-number
    failure wearing a friendly face.
    """
    result = reconcile(sample(), rows_read=4)
    assert result.sum_status == SUM_UNVERIFIED
    assert result.sum_verified is False
    assert result.difference is None
    assert result.passed is True  # nothing FAILED — but nothing was checked either
    assert "NOT been checked" in result.describe()


def test_matching_control_total_verifies_the_sum():
    result = reconcile(sample(), rows_read=4, control_total=D("3050.15"))
    assert result.total == D("3050.15")
    assert result.sum_status == SUM_VERIFIED
    assert result.sum_verified is True
    assert result.difference == D("0")
    assert result.passed is True


def test_control_total_mismatch_fails_and_reports_the_difference():
    result = reconcile(sample(), rows_read=4, control_total=D("3000.00"))
    assert result.sum_status == SUM_MISMATCH
    assert result.difference == D("50.15")
    assert result.passed is False


def test_tolerance_admits_a_stated_rounding_slack_only():
    inside = reconcile(sample(), rows_read=4, control_total=D("3050.14"), tolerance=D("0.01"))
    assert inside.sum_status == SUM_VERIFIED
    outside = reconcile(sample(), rows_read=4, control_total=D("3050.00"), tolerance=D("0.01"))
    assert outside.sum_status == SUM_MISMATCH


# -- row counts ------------------------------------------------------------------------------

def test_a_dropped_row_fails_even_when_the_sum_was_never_checked():
    """A silently skipped row is the canonical way to produce a plausible wrong total."""
    result = reconcile(sample(), rows_read=5)
    assert result.all_rows_parsed is False
    assert result.passed is False
    assert "MISMATCH" in result.describe()


def test_row_counts_matching_is_required_even_with_a_good_sum():
    result = reconcile(sample(), rows_read=9, control_total=D("3050.15"))
    assert result.sum_status == SUM_VERIFIED
    assert result.passed is False  # the sum is right and the read still went wrong


# -- money in / money out --------------------------------------------------------------------

def test_credits_and_debits_are_split_exactly():
    """`debits` is a positive MAGNITUDE ("money out"), not a negative sum — so the identity
    that must hold is credits - debits == total, not credits + debits."""
    result = reconcile(sample(), rows_read=4)
    assert result.credits == D("3250.00")
    assert result.debits == D("199.85")
    assert result.credits - result.debits == result.total


def test_decimal_precision_survives_reconciliation():
    """Money is Decimal end to end. A float would make 0.1 + 0.2 != 0.3 and compound."""
    rows = [tx(1, "A", "0.10", 1), tx(1, "B", "0.20", 2)]
    result = reconcile(rows, rows_read=2, control_total=D("0.30"))
    assert result.total == D("0.30")
    assert str(result.total) == "0.30"
    assert result.sum_status == SUM_VERIFIED


# -- duplicates: flagged, never removed --------------------------------------------------------

def test_duplicates_are_flagged_with_their_rows_and_do_not_fail_the_check():
    """Two identical coffees on one day are usually real. Dropping them would understate the
    total by exactly the amount that looked suspicious, so the judgement stays with the human."""
    rows = sample() + [tx(2, "SQ *BLUE BOTTLE COFFEE", "-6.75", 5)]
    result = reconcile(rows, rows_read=5)
    assert len(result.duplicates) == 1
    group = result.duplicates[0]
    assert group.count == 2
    assert group.row_indices == (1, 5)
    assert result.passed is True
    assert "flagged not removed" in result.describe()


def test_find_duplicates_ignores_rows_differing_in_any_field():
    rows = [
        tx(2, "COFFEE", "-6.75", 1),
        tx(3, "COFFEE", "-6.75", 2),      # different date
        tx(2, "COFFEE", "-7.75", 3),      # different amount
        tx(2, "TEA", "-6.75", 4),         # different description
    ]
    assert find_duplicates(rows) == ()


# -- period bounds ------------------------------------------------------------------------------

def test_a_row_outside_the_stated_period_fails_and_names_the_row():
    result = reconcile(
        sample(),
        rows_read=4,
        period_start=datetime.date(2026, 8, 3),
        period_end=datetime.date(2026, 8, 31),
    )
    assert result.out_of_period == (1,)
    assert result.passed is False
    assert "OUT OF PERIOD" in result.describe()


def test_date_range_is_reported():
    result = reconcile(sample(), rows_read=4)
    assert result.first_date == datetime.date(2026, 8, 2)
    assert result.last_date == datetime.date(2026, 8, 7)


# -- control_total_from_balances -----------------------------------------------------------------

def test_control_total_from_balances_is_the_net_movement():
    assert control_total_from_balances(D("4217.30"), D("7267.45")) == D("3050.15")


# -- require_pass -----------------------------------------------------------------------------

def test_require_pass_returns_a_passing_result_unchanged():
    result = reconcile(sample(), rows_read=4, control_total=D("3050.15"))
    assert require_pass(result) is result


def test_require_pass_raises_with_the_report_attached():
    result = reconcile(sample(), rows_read=4, control_total=D("1.00"))
    with pytest.raises(ReconciliationError) as exc:
        require_pass(result)
    assert "reconciliation failed" in str(exc.value)
    assert "difference" in str(exc.value)


def test_require_control_total_promotes_unverified_from_reported_to_refused():
    """Opt-in, so that 'unchecked' and 'checked and wrong' stay distinguishable by default."""
    result = reconcile(sample(), rows_read=4)
    assert require_pass(result) is result                       # tolerated by default
    with pytest.raises(ReconciliationError) as exc:
        require_pass(result, require_control_total=True)
    assert "never checked" in str(exc.value)


# -- the report cannot lie -------------------------------------------------------------------

def test_passed_is_derived_and_cannot_be_asserted_by_construction():
    """`passed` is a property, not a field: a Reconciliation cannot be built claiming a pass
    it does not support. Pinning this because a stored boolean is exactly how a gate becomes
    a thing that reports success independently of the outcome."""
    fabricated = Reconciliation(rows_read=5, rows_parsed=4, total=D("0"))
    assert fabricated.passed is False
    with pytest.raises(TypeError):
        Reconciliation(rows_read=4, rows_parsed=4, total=D("0"), passed=True)
