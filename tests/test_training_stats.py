"""Exact paired statistics behind the promotion gate (LEARNING_plan §3.3).

These are the numbers the gate's authority rests on, so they are pinned against values
computed by hand rather than against the implementation.
"""

from __future__ import annotations

import pytest

from hearth.training.stats import (
    binomial_sf,
    discordant_pairs,
    is_binary,
    mcnemar_exact_p,
    min_n_for_alpha,
    paired_bootstrap,
    smallest_achievable_p,
)


def test_binomial_sf_matches_hand_computation():
    # P(X >= 8) for X ~ Bin(9, 0.5) = (C(9,8) + C(9,9)) / 2**9 = 10/512.
    assert binomial_sf(8, 9) == pytest.approx(10 / 512)
    assert binomial_sf(0, 5) == 1.0
    assert binomial_sf(6, 5) == 0.0
    assert binomial_sf(5, 5) == pytest.approx(1 / 32)


@pytest.mark.parametrize(
    ("b", "c", "expected"),
    [
        (3, 0, 0.125),
        (4, 0, 0.0625),  # <- the docs/RESULTS.md promotion; see test_eval_gate_replay.py
        (5, 0, 0.03125),
        (6, 1, 0.0625),
        (8, 1, 10 / 512),
        (10, 3, 378 / 8192),
    ],
)
def test_mcnemar_exact_p_reproduces_the_plan_table(b, c, expected):
    """Every row of the LEARNING_plan §3.3 table, recomputed exactly."""
    assert mcnemar_exact_p(b, c) == pytest.approx(expected)


def test_mcnemar_with_no_discordant_pairs_is_not_evidence():
    # Concordant items carry no information: two identical systems prove nothing.
    assert mcnemar_exact_p(0, 0) == 1.0


def test_discordant_pairs_counts_both_directions():
    cand = [1.0, 1.0, 0.0, 1.0]
    inc = [1.0, 0.0, 1.0, 0.0]
    assert discordant_pairs(cand, inc) == (2, 1)


def test_discordant_pairs_refuses_unpaired_or_continuous_vectors():
    with pytest.raises(ValueError):
        discordant_pairs([1.0, 0.0], [1.0])
    with pytest.raises(ValueError):
        discordant_pairs([0.5, 1.0], [1.0, 1.0])


def test_is_binary():
    assert is_binary([0.0, 1.0, 1.0])
    assert not is_binary([0.0, 0.4])
    assert not is_binary([])


def test_paired_bootstrap_is_deterministic_under_a_seed():
    cand = [0.9, 0.8, 0.7, 1.0, 0.6]
    inc = [0.4, 0.5, 0.3, 0.6, 0.2]
    a = paired_bootstrap(cand, inc, iterations=500, seed=7)
    b = paired_bootstrap(cand, inc, iterations=500, seed=7)
    assert (a.diff, a.p_value, a.ci_low, a.ci_high) == (b.diff, b.p_value, b.ci_low, b.ci_high)


def test_paired_bootstrap_separates_a_real_lift_from_no_lift():
    big = paired_bootstrap([1.0] * 40, [0.0] * 40, iterations=500, seed=0)
    assert big.diff == pytest.approx(1.0)
    assert big.p_value < 0.05
    assert big.ci_low > 0.0

    none = paired_bootstrap([0.5] * 40, [0.5] * 40, iterations=500, seed=0)
    assert none.diff == 0.0
    assert none.p_value > 0.05  # a tie is not a win


def test_paired_bootstrap_interval_is_huge_on_a_tiny_set():
    """n=5 must produce an honestly enormous interval, not a confident-looking one."""
    boot = paired_bootstrap([1.0, 1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0, 1.0],
                            iterations=1000, seed=0)
    assert boot.ci_high - boot.ci_low > 0.8
    assert boot.p_value > 0.05


def test_paired_bootstrap_rejects_bad_input():
    with pytest.raises(ValueError):
        paired_bootstrap([1.0], [1.0, 0.0])
    with pytest.raises(ValueError):
        paired_bootstrap([], [])


def test_the_minimum_golden_set_size_that_can_ever_clear_alpha():
    """At n=5 the best possible p is 0.5**5 = 0.03125 — so 5 is the hard floor.

    Below five paired items no outcome exists that clears alpha=0.05: four clean wins give
    0.0625. Five is the *mathematical* minimum and it demands a clean sweep (the incumbent
    wrong on every item the candidate gets right); the gate's default ``min_n`` sits well
    above it because any incumbent competence at all pushes the requirement up.
    """
    assert smallest_achievable_p(5) == pytest.approx(0.03125)
    assert smallest_achievable_p(4) == pytest.approx(0.0625)
    assert min_n_for_alpha(0.05) == 5
    assert min_n_for_alpha(0.01) == 7
    assert smallest_achievable_p(min_n_for_alpha(0.05)) <= 0.05
    assert smallest_achievable_p(min_n_for_alpha(0.05) - 1) > 0.05
