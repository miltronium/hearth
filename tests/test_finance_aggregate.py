"""Tests for deterministic aggregation (hearth.finance.aggregate), plus the package's
no-network invariant.

The design rule these pin: **a model never computes a number here.** Categorization is a
judgement and a local model may make it; arithmetic is not, and Python does all of it in
Decimal. That split is what lets a narrative claim be audited — the figure came from a sum
over rows, and the rows can be listed.

Arrived untested when the authoring session was interrupted; aggregation feeds every number
an operator will read, so it gets pinned here.
"""

from __future__ import annotations

import ast
import datetime
from decimal import Decimal
from pathlib import Path

import hearth.finance
from hearth.finance.aggregate import (
    UNCATEGORIZED,
    by_category,
    by_month,
    month_key,
    spend_by_category,
    top_merchants,
    totals,
)
from hearth.finance.parse import Transaction

D = Decimal


def tx(month: int, day: int, description: str, amount: str, row: int, category=None):
    return Transaction(
        date=datetime.date(2026, month, day),
        description=description,
        amount=D(amount),
        row_index=row,
        category=category,
    )


def sample():
    return [
        tx(8, 2, "SQ *BLUE BOTTLE COFFEE", "-6.75", 1, "dining"),
        tx(8, 3, "SHELL OIL 574839201", "-52.10", 2, "fuel"),
        tx(8, 5, "ACH DEPOSIT PAYROLL", "3250.00", 3, "income"),
        tx(9, 1, "TRADER JOES #182", "-96.31", 4, "groceries"),
        tx(9, 2, "SQ *BLUE BOTTLE COFFEE", "-4.25", 5, None),
    ]


# -- totals ---------------------------------------------------------------------------------

def test_totals_split_income_and_spend_as_positive_magnitudes():
    result = totals(sample())
    assert result.count == 5
    assert result.income == D("3250.00")
    assert result.spend == D("159.41")
    assert result.total == D("3090.59")
    assert result.net == result.total


def test_totals_of_nothing_is_zero_not_an_error():
    result = totals([])
    assert result.count == 0
    assert result.total == D("0")
    assert result.first_date is None


def test_decimal_precision_survives_aggregation():
    """The reason money is never a float. 0.1 + 0.2 must be exactly 0.30 here."""
    rows = [tx(8, 1, "A", "-0.10", 1), tx(8, 1, "B", "-0.20", 2)]
    assert totals(rows).spend == D("0.30")
    assert str(totals(rows).spend) == "0.30"


def test_many_small_amounts_do_not_drift():
    rows = [tx(8, 1, f"M{i}", "-0.01", i) for i in range(1, 101)]
    assert totals(rows).spend == D("1.00")


# -- grouping ---------------------------------------------------------------------------------

def test_by_category_sums_signed_amounts_per_category():
    grouped = by_category(sample())
    assert grouped["dining"] == D("-6.75")
    assert grouped["income"] == D("3250.00")


def test_an_unassigned_category_is_surfaced_not_guessed_or_dropped():
    """Dropping it would understate spend; guessing it would invent a fact. Neither is this
    module's call to make."""
    grouped = by_category(sample())
    assert grouped[UNCATEGORIZED] == D("-4.25")
    assert sum(grouped.values()) == totals(sample()).total


def test_spend_by_category_excludes_income_and_reports_magnitudes():
    spend = spend_by_category(sample())
    assert "income" not in spend
    assert spend["fuel"] == D("52.10")


def test_by_month_partitions_and_each_partition_reconciles():
    months = by_month(sample())
    assert set(months) == {"2026-08", "2026-09"}
    assert months["2026-08"].total == D("3191.15")
    assert months["2026-09"].total == D("-100.56")
    assert sum(m.total for m in months.values()) == totals(sample()).total


def test_month_key_is_sortable():
    assert month_key(datetime.date(2026, 9, 30)) == "2026-09"
    assert month_key(datetime.date(2026, 8, 1)) < month_key(datetime.date(2026, 9, 1))


# -- merchants ---------------------------------------------------------------------------------

def test_top_merchants_groups_on_the_exact_description_by_default():
    """No normalization: stripping store numbers is merchant knowledge this package does not
    have, and a wrong normalizer silently merges unrelated spending."""
    top = top_merchants(sample(), limit=3)
    assert [m.description for m in top] == [
        "TRADER JOES #182",          # 96.31
        "SHELL OIL 574839201",       # 52.10
        "SQ *BLUE BOTTLE COFFEE",    # 11.00 across two rows
    ]
    assert top[0].total == D("96.31")
    coffee = [m for m in top if m.description == "SQ *BLUE BOTTLE COFFEE"]
    assert coffee and coffee[0].count == 2 and coffee[0].total == D("11.00")


def test_top_merchants_respects_the_limit_and_excludes_income_by_default():
    assert len(top_merchants(sample(), limit=2)) == 2
    assert all(m.description != "ACH DEPOSIT PAYROLL" for m in top_merchants(sample()))


def test_a_caller_owning_merchant_knowledge_can_supply_its_own_key():
    top = top_merchants(sample(), key=lambda t: t.description.split()[0])
    assert any(m.description == "SQ" and m.count == 2 for m in top)


# -- the no-network invariant -------------------------------------------------------------------

BANNED_CALLS = frozenset(
    {"system", "popen", "urlopen", "socket", "connect", "sendall", "eval", "exec", "__import__"}
)
BANNED_IMPORTS = frozenset(
    {"socket", "ssl", "http", "urllib", "requests", "httpx", "aiohttp", "subprocess", "ftplib",
     "smtplib", "telnetlib", "xmlrpc", "asyncio"}
)
SOURCES = sorted(Path(hearth.finance.__file__).parent.glob("*.py"))


def test_the_package_has_sources_to_check():
    assert SOURCES, "no sources found — the invariant test would pass vacuously"


def test_no_module_imports_anything_that_could_reach_the_network():
    """Financial records are the most sensitive thing HEARTH touches. This package must have
    no way to send them anywhere, and that has to be checked mechanically rather than trusted:
    a convenience import is exactly how such a guarantee erodes."""
    for path in SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            for name in names:
                root = name.split(".")[0]
                assert root not in BANNED_IMPORTS, f"{path.name} imports {name}"


def test_no_module_can_reach_a_shell_or_a_dynamic_import():
    for path in SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = getattr(func, "attr", None) or getattr(func, "id", None)
                assert name not in BANNED_CALLS, f"{path.name} calls {name}"
