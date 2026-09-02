"""Hermetic tests for the two-tier ladder example (examples/finance/).

No model, no weights, no network. These pin the properties the example exists to prove:
the aggregates are computed in **Python** and are exact, the synthetic answer key is
well-formed, and the no-egress seal refuses a leaky profile before anything loads.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from hearth.router.policy import ClassRule, Defaults, RemoteConfig, RoutingPolicy

_EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "finance"


def _load_module():
    """Import the harness by path — examples/ is not an installed package."""
    spec = importlib.util.spec_from_file_location(
        "finance_ladder", _EXAMPLE / "run_finance_ladder.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ladder = _load_module()


# -- the synthetic data --------------------------------------------------------------------


def test_statements_csv_is_well_formed():
    txns = ladder.load_transactions(_EXAMPLE / "statements.csv")
    assert 30 <= len(txns) <= 50
    assert all(t.expected in ladder.CATEGORIES for t in txns), "answer key uses an unknown label"
    assert all(t.difficulty in ("easy", "hard") for t in txns)
    assert any(t.difficulty == "hard" for t in txns), "the point is the hard cases"
    assert any(t.amount > 0 for t in txns) and any(t.amount < 0 for t in txns)


def test_answer_key_covers_every_category():
    """Every label the model may choose is actually exercised by at least one row."""
    txns = ladder.load_transactions(_EXAMPLE / "statements.csv")
    assert {t.expected for t in txns} == set(ladder.CATEGORIES)


# -- stage 2: the arithmetic is Python's, and it is exact ----------------------------------


def _rows(*items):
    """Build categorized rows from ``(amount, category)`` pairs."""
    return [
        ladder.Categorized(
            txn=ladder.Transaction("2026-06-01", f"row {i}", amount, category, "easy"),
            predicted=category,
            model="fake",
            latency_ms=0.0,
        )
        for i, (amount, category) in enumerate(items)
    ]


def test_aggregates_are_exact():
    agg = ladder.aggregate(
        _rows((1000.00, "income"), (-25.50, "dining"), (-10.25, "dining"), (-100.00, "groceries"))
    )
    assert agg.total_income == pytest.approx(1000.00)
    assert agg.total_spend == pytest.approx(135.75)
    assert agg.net == pytest.approx(864.25)
    assert agg.transaction_count == 4
    assert agg.by_category["dining"] == pytest.approx(35.75)
    assert agg.by_category["groceries"] == pytest.approx(100.00)
    assert agg.counts_by_category["dining"] == 2
    assert agg.largest == ("row 3", 100.00)
    # Credits never land in the spend breakdown.
    assert "income" not in agg.by_category


def test_by_category_is_ordered_by_spend():
    agg = ladder.aggregate(_rows((-5.0, "dining"), (-50.0, "groceries"), (-20.0, "transport")))
    assert list(agg.by_category) == ["groceries", "transport", "dining"]


def test_fact_sheet_quotes_only_computed_figures():
    """Tier 2 sees finished numbers and shares — never raw rows to add up itself."""
    agg = ladder.aggregate(_rows((100.0, "income"), (-75.0, "groceries"), (-25.0, "dining")))
    facts = ladder.render_facts(agg)
    assert "Total spend: $100.00" in facts
    assert "groceries: $75.00 across 1 transactions (75.0%)" in facts
    assert "dining: $25.00 across 1 transactions (25.0%)" in facts


def test_aggregate_handles_an_empty_month():
    agg = ladder.aggregate([])
    assert (agg.total_income, agg.total_spend, agg.net, agg.transaction_count) == (0.0, 0.0, 0.0, 0)
    assert ladder.render_facts(agg)  # renders without dividing by zero


# -- stage 1: label parsing ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("dining", "dining"),
        ("  Dining\n", "dining"),
        ("Category: transport.", "transport"),
        ("subscriptions (recurring)", "subscriptions"),
        ("I think this is groceries, not dining", "groceries"),  # earliest match wins
        ("no idea", "uncategorized"),
        ("", "uncategorized"),
    ],
)
def test_parse_label(reply, expected):
    assert ladder._parse_label(reply) == expected


# -- the seal ------------------------------------------------------------------------------


def test_verify_no_egress_accepts_the_bundled_finance_profile():
    from hearth.router.policy import load_policy

    path = Path(__file__).resolve().parent.parent / "config" / "routing.finance.yaml"
    ladder.verify_no_egress(load_policy(path), path)  # must not raise


@pytest.mark.parametrize(
    "policy",
    [
        RoutingPolicy(
            defaults=Defaults(),
            classes={"classify": ClassRule(backend="local", escalate="never")},
            remotes={"default": RemoteConfig(protocol="anthropic", model="x")},
        ),
        RoutingPolicy(
            defaults=Defaults(),
            classes={"reason": ClassRule(backend="remote", escalate="always")},
            remotes={},
        ),
        RoutingPolicy(
            defaults=Defaults(),
            classes={"chat": ClassRule(backend="local", escalate="on_low_confidence")},
            remotes={},
        ),
    ],
    ids=["remote-defined", "class-pinned-remote", "class-can-escalate"],
)
def test_verify_no_egress_fails_closed(policy):
    """A leaky profile exits non-zero *before* any weights load."""
    with pytest.raises(SystemExit) as exc:
        ladder.verify_no_egress(policy, Path("test.yaml"))
    assert exc.value.code == 2
