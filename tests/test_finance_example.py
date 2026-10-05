"""Hermetic tests for the two-tier ladder example (examples/finance/).

No model, no weights, no network. These pin the properties the example exists to prove:
the aggregates are computed in **Python** and are exact, the synthetic answer key is
well-formed, and the no-egress seal refuses a leaky profile before anything loads.
"""

from __future__ import annotations

import importlib.util
import sys
from decimal import Decimal
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
    """Build categorized rows from ``(amount, category)`` pairs; amounts given as strings."""
    return [
        ladder.Categorized(
            txn=ladder.Transaction("2026-06-01", f"row {i}", Decimal(amount), category, "easy"),
            predicted=category,
            model="fake",
            latency_ms=0.0,
        )
        for i, (amount, category) in enumerate(items)
    ]


def test_aggregates_are_exact():
    agg = ladder.aggregate(
        _rows(("1000.00", "income"), ("-25.50", "dining"), ("-10.25", "dining"),
              ("-100.00", "groceries"))
    )
    assert agg.total_income == Decimal("1000.00")
    assert agg.total_spend == Decimal("135.75")
    assert agg.net == Decimal("864.25")
    assert agg.transaction_count == 4
    assert agg.by_category["dining"] == Decimal("35.75")
    assert agg.by_category["groceries"] == Decimal("100.00")
    assert agg.counts_by_category["dining"] == 2
    assert agg.largest == ("row 3", Decimal("100.00"))
    # Credits never land in the spend breakdown.
    assert "income" not in agg.by_category


def test_by_category_is_ordered_by_spend():
    agg = ladder.aggregate(_rows(("-5.0", "dining"), ("-50.0", "groceries"),
                                 ("-20.0", "transport")))
    assert list(agg.by_category) == ["groceries", "transport", "dining"]


def test_fact_sheet_quotes_only_computed_figures():
    """Tier 2 sees finished numbers and shares — never raw rows to add up itself."""
    agg = ladder.aggregate(_rows(("100.0", "income"), ("-75.0", "groceries"),
                                 ("-25.0", "dining")))
    facts = ladder.render_facts(agg)
    assert "Total spend: $100.00" in facts
    assert "groceries: $75.00 across 1 transactions (75.0%)" in facts
    assert "dining: $25.00 across 1 transactions (25.0%)" in facts


def test_aggregate_handles_an_empty_month():
    agg = ladder.aggregate([])
    assert (agg.total_income, agg.total_spend, agg.net, agg.transaction_count) == (0, 0, 0, 0)
    assert ladder.render_facts(agg)  # renders without dividing by zero


# -- B-049: money is Decimal end to end ------------------------------------------------------

_CSV_HEADER = "date,description,amount,expected_category,difficulty\n"


def _csv(tmp_path, amounts: list[str]) -> Path:
    path = tmp_path / "statements.csv"
    path.write_text("# synthetic\n" + _CSV_HEADER + "".join(
        f"2026-06-01,row {i},{a},{'income' if not a.startswith('-') else 'dining'},easy\n"
        for i, a in enumerate(amounts)
    ))
    return path


def test_totals_are_exact_where_float_sums_are_not(tmp_path):
    """Parse -> sum -> compare -> format, through the CSV loader. Each total below is one
    float addition gets wrong (0.1+0.2 = 0.30000000000000004; ten 0.10s = 0.9999999999999999),
    so equality with the exact Decimal fails on any float in the path."""
    assert 0.10 + 0.20 != 0.30 and sum([0.10] * 10) != 1.0  # the trap is real
    txns = ladder.load_transactions(_csv(tmp_path, ["0.10", "0.20"] + ["-0.10"] * 10))
    assert all(type(t.amount) is Decimal for t in txns)
    agg = ladder.aggregate([ladder.Categorized(t, t.expected, "fake", 0.0) for t in txns])
    assert agg.total_income == Decimal("0.30")
    assert agg.total_spend == Decimal("1.00")
    assert agg.net == Decimal("-0.70")
    assert agg.by_category["dining"] == Decimal("1.00")
    for figure in (agg.total_income, agg.total_spend, agg.net, *agg.by_category.values()):
        assert type(figure) is Decimal
    facts = ladder.render_facts(agg)
    assert "Total income: $0.30" in facts and "Net: $-0.70" in facts


def test_bundled_statement_totals_match_an_independent_exact_sum():
    """The bundled synthetic CSV, summed independently in integer cents."""
    txns = ladder.load_transactions(_EXAMPLE / "statements.csv")
    agg = ladder.aggregate([ladder.Categorized(t, t.expected, "fake", 0.0) for t in txns])
    cents = [int(t.amount * 100) for t in txns]
    assert all(t.amount * 100 == c for t, c in zip(txns, cents, strict=True))
    assert agg.total_income * 100 == sum(c for c in cents if c > 0)
    assert agg.total_spend * 100 == -sum(c for c in cents if c < 0)


def test_an_unparseable_amount_is_refused_with_its_row(tmp_path):
    with pytest.raises(ValueError, match="data row 2"):
        ladder.load_transactions(_csv(tmp_path, ["1.00", "12.3.4"]))


def test_no_float_is_applied_to_money_in_the_example():
    """Guard: the example must not construct a float from anything, and its money fields
    must be declared Decimal — the shipped example teaches what the finance package does."""
    import ast

    source = (_EXAMPLE / "run_finance_ladder.py").read_text()
    tree = ast.parse(source)
    float_calls = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "float"
    ]
    assert float_calls == [], f"float(...) at line(s) {float_calls}"
    money_fields = {
        "Transaction": {"amount"},
        "Aggregates": {"total_income", "total_spend", "net", "by_category", "largest"},
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in money_fields:
            annotations = {
                stmt.target.id: ast.unparse(stmt.annotation)
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
            }
            for name in money_fields.pop(node.name):
                assert "Decimal" in annotations[name], (node.name, name, annotations[name])
                assert "float" not in annotations[name], (node.name, name, annotations[name])
    assert money_fields == {}, f"classes not found: {money_fields}"


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
