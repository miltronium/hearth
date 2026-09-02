"""Exact small-sample statistics for the promotion gate (LEARNING_plan §3.3).

The gate compares a candidate and an incumbent scored on the **same** golden items, so the
correct test is *paired*: concordant items carry no information and must not be counted as
evidence. Two tests cover the metrics HEARTH actually produces:

  * binary outcomes (exact-match, judge win/loss) — **exact one-sided McNemar**, i.e. a
    binomial test of the discordant pairs against 0.5. Exact, not the chi-square
    approximation, because a golden set here has single-digit discordance.
  * continuous outcomes (token-F1) — **paired bootstrap** over item indices, seeded so the
    same vectors always yield the same interval.

Everything is stdlib ``math``/``random``: no scipy, no numpy, no new dependency (ADR: the
core install stays light, and the gate must run in sealed private mode).
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

# Iterations for the paired bootstrap. 10k is the plan's number: enough resolution for a
# 2.5th percentile, cheap enough that the gate stays interactive on a golden set of any
# size HEARTH will ever hand it.
DEFAULT_BOOTSTRAP_ITERATIONS = 10_000


def binomial_sf(k: int, n: int, p: float = 0.5) -> float:
    """``P(X >= k)`` for ``X ~ Binomial(n, p)``, summed exactly (no normal approximation).

    Returns 1.0 for ``k <= 0`` and 0.0 for ``k > n``. ``n = 0`` is the empty experiment and
    yields 1.0 — no evidence, no significance.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    total = 0.0
    for i in range(k, n + 1):
        total += math.comb(n, i) * (p**i) * ((1.0 - p) ** (n - i))
    return min(1.0, total)


def mcnemar_exact_p(b: int, c: int) -> float:
    """One-sided exact McNemar p-value for ``b`` candidate wins vs. ``c`` incumbent wins.

    ``b`` = items the candidate gets right and the incumbent gets wrong; ``c`` = the
    reverse. Concordant items are dropped: the p-value is ``P(X >= b)`` for
    ``X ~ Binomial(b + c, 0.5)`` — the probability of seeing this lopsided a split of the
    discordant pairs if the two systems were equally good.

    With no discordant pairs at all there is nothing to test and this returns 1.0.
    """
    n = b + c
    if n == 0:
        return 1.0
    return binomial_sf(b, n, 0.5)


def is_binary(values: Sequence[float]) -> bool:
    """True when every value is exactly 0.0 or 1.0 (so McNemar is the right test)."""
    return bool(values) and all(v in (0.0, 1.0) for v in values)


def discordant_pairs(
    candidate: Sequence[float], incumbent: Sequence[float]
) -> tuple[int, int]:
    """Return ``(b, c)`` — candidate-wins and incumbent-wins over the paired vectors.

    Requires equal-length binary vectors; raises otherwise, because a silent
    length-mismatch here is exactly the "comparing unrelated universes" failure the gate
    exists to prevent.
    """
    if len(candidate) != len(incumbent):
        raise ValueError(
            f"paired vectors must be the same length: {len(candidate)} vs {len(incumbent)}"
        )
    if not is_binary(candidate) or not is_binary(incumbent):
        raise ValueError("McNemar needs binary (0.0/1.0) per-example scores")
    b = sum(1 for x, y in zip(candidate, incumbent, strict=True) if x > y)
    c = sum(1 for x, y in zip(candidate, incumbent, strict=True) if y > x)
    return b, c


@dataclass(frozen=True)
class BootstrapResult:
    """Paired-bootstrap outcome over ``mean(candidate) - mean(incumbent)``."""

    diff: float
    ci_low: float
    ci_high: float
    p_value: float
    iterations: int
    seed: int


def paired_bootstrap(
    candidate: Sequence[float],
    incumbent: Sequence[float],
    *,
    margin: float = 0.0,
    alpha: float = 0.05,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = 0,
) -> BootstrapResult:
    """Resample item indices with replacement and test ``diff > margin`` one-sided.

    Both vectors are resampled at the *same* indices on every draw — that is what makes it
    paired and what keeps per-item difficulty from inflating the interval. ``p_value`` is
    the (add-one smoothed) fraction of resamples in which the difference failed to exceed
    ``margin``, so it can never report an impossible 0.0. The interval is the
    ``[alpha, 1 - alpha]`` quantile range of the resampled differences, reported so the
    width of a small-``n`` estimate is always visible next to the point estimate.
    """
    if len(candidate) != len(incumbent):
        raise ValueError(
            f"paired vectors must be the same length: {len(candidate)} vs {len(incumbent)}"
        )
    if not candidate:
        raise ValueError("cannot bootstrap an empty vector")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    n = len(candidate)
    diff = (sum(candidate) - sum(incumbent)) / n
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(iterations):
        total = 0.0
        for _ in range(n):
            i = rng.randrange(n)
            total += candidate[i] - incumbent[i]
        draws.append(total / n)
    draws.sort()
    failures = sum(1 for d in draws if d <= margin)
    return BootstrapResult(
        diff=diff,
        ci_low=_quantile(draws, alpha),
        ci_high=_quantile(draws, 1.0 - alpha),
        p_value=(failures + 1) / (iterations + 1),
        iterations=iterations,
        seed=seed,
    )


def smallest_achievable_p(n: int) -> float:
    """The best (smallest) exact one-sided McNemar p-value reachable on ``n`` paired items.

    The optimum is a clean sweep — the candidate right on every item, the incumbent wrong
    on every item — giving ``b = n, c = 0`` and ``p = 0.5**n``. Use it to answer "can this
    golden set license a promotion at all?" before spending GPU time on the candidate.
    """
    if n < 1:
        return 1.0
    return 0.5**n


def min_n_for_alpha(alpha: float = 0.05) -> int:
    """Smallest golden-set size whose *best case* clears ``alpha`` under exact McNemar.

    Solves ``0.5**n <= alpha``. At ``alpha = 0.05`` this is **5** — and only in the
    degenerate case where the incumbent misses every item the candidate hits; any
    incumbent competence at all pushes the requirement well above it (hence the gate's
    separate, larger ``min_n`` policy floor).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    n = 1
    while 0.5**n > alpha:
        n += 1
    return n


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile of an already-sorted sequence."""
    if not sorted_values:
        raise ValueError("empty sequence")
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    pos = q * (len(sorted_values) - 1)
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return sorted_values[int(pos)]
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (pos - low)


__all__ = [
    "DEFAULT_BOOTSTRAP_ITERATIONS",
    "BootstrapResult",
    "binomial_sf",
    "discordant_pairs",
    "is_binary",
    "mcnemar_exact_p",
    "min_n_for_alpha",
    "paired_bootstrap",
    "smallest_achievable_p",
]
