"""Token-budget accountant tests (ADR-007)."""

from __future__ import annotations

from hearth.observability.budget import BudgetAccountant


def test_starts_full():
    b = BudgetAccountant(1000)
    assert b.remaining() == 1000
    assert b.spent() == 0
    assert b.can_afford(1000)
    assert not b.can_afford(1001)


def test_spend_decrements():
    b = BudgetAccountant(1000)
    b.spend(300)
    assert b.spent() == 300
    assert b.remaining() == 700
    assert b.can_afford(700)
    assert not b.can_afford(701)


def test_exhaustion_denies():
    b = BudgetAccountant(100)
    b.spend(100)
    assert b.remaining() == 0
    assert not b.can_afford(1)


def test_zero_budget_disables_escalation():
    b = BudgetAccountant(0)
    assert not b.can_afford(0)
    assert not b.can_afford(1)


def test_negative_and_zero_spends_ignored():
    b = BudgetAccountant(100)
    b.spend(0)
    b.spend(-5)
    assert b.spent() == 0


def test_day_rollover_resets(monkeypatch):
    import hearth.observability.budget as budget_mod

    day = {"value": "2026-07-08"}
    monkeypatch.setattr(budget_mod, "_utc_day", lambda: day["value"])
    b = BudgetAccountant(100)
    b.spend(80)
    assert b.spent() == 80  # same day
    # Advance the clock a day -> the next query resets the counter.
    day["value"] = "2026-07-09"
    assert b.remaining() == 100
    assert b.spent() == 0
