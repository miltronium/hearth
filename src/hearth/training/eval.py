"""Eval harness + promotion gate (ARCHITECTURE §7, ADR-006, Phase 4).

Scores a candidate adapter against a golden set per task class and encodes the gate that
protects promotion: **a candidate must BEAT the incumbent to be promotable** (ADR-006).

Metrics (ARCHITECTURE §7):
  * ``extract`` / ``classify`` — exact-match / token-F1 (deterministic, no judge).
  * ``draft`` / ``code``       — win-rate via a pluggable judge hook (stubbed here).

Scoring is done against a caller-supplied ``generate`` function ``(prompt) -> text`` so
this harness is fully testable with fakes and never touches a real model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

# Generates a candidate answer for a prompt. Injected — a real caller wires a provider;
# tests pass a dict-backed fake.
Generate = Callable[[str], str]

# Judges whether ``candidate`` beats ``reference`` for ``prompt``; returns True on a win.
# Stubbed/injected for draft/code classes (no judge model in Phase 4).
Judge = Callable[[str, str, str], bool]

# Classes scored by objective string metrics vs. a judge (ARCHITECTURE §7).
_OBJECTIVE_CLASSES = frozenset({"extract", "classify", "rank", "summarize"})


@dataclass(frozen=True)
class GoldenExample:
    """One golden (prompt, expected) pair for a task class."""

    prompt: str
    expected: str


@dataclass(frozen=True)
class GoldenSet:
    """A golden evaluation set for one task class."""

    task: str
    examples: list[GoldenExample]

    def __len__(self) -> int:
        return len(self.examples)


@dataclass(frozen=True)
class EvalReport:
    """Aggregate score of a candidate over a golden set.

    ``score`` is in [0, 1] (mean per-example score). ``per_example`` keeps the individual
    scores for inspection. ``metric`` names how it was computed.
    """

    task: str
    metric: str
    score: float
    per_example: list[float] = field(default_factory=list)
    n: int = 0


def exact_match_score(candidate: str, expected: str) -> float:
    """1.0 if the (stripped, case-folded) strings match exactly, else 0.0."""
    return 1.0 if candidate.strip().casefold() == expected.strip().casefold() else 0.0


def token_f1_score(candidate: str, expected: str) -> float:
    """Token-level F1 over whitespace tokens (case-folded). 0.0 when either side empty."""
    cand = candidate.casefold().split()
    gold = expected.casefold().split()
    if not cand or not gold:
        return 1.0 if not cand and not gold else 0.0
    # Multiset overlap (bounded by counts on each side).
    overlap = 0
    remaining = list(gold)
    for tok in cand:
        if tok in remaining:
            remaining.remove(tok)
            overlap += 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(cand)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def score_candidate(
    golden: GoldenSet,
    generate: Generate,
    *,
    metric: str = "f1",
    judge: Judge | None = None,
) -> EvalReport:
    """Score ``generate`` over ``golden`` and return an :class:`EvalReport`.

    ``metric`` selects the objective scorer for extract/classify-style tasks:
    ``"exact"`` or ``"f1"``. For subjective classes (draft/code) pass a ``judge``; each
    example scores 1.0 on a judged win, else 0.0. A judge, when supplied, always wins over
    the string metric.
    """
    if not golden.examples:
        raise ValueError("cannot score an empty golden set")

    if judge is not None:
        used_metric = "judge_win_rate"
        scorer = lambda cand, ex: 1.0 if judge(ex.prompt, cand, ex.expected) else 0.0  # noqa: E731
    elif metric == "exact":
        used_metric = "exact_match"
        scorer = lambda cand, ex: exact_match_score(cand, ex.expected)  # noqa: E731
    elif metric == "f1":
        used_metric = "token_f1"
        scorer = lambda cand, ex: token_f1_score(cand, ex.expected)  # noqa: E731
    else:
        raise ValueError(f"unknown metric: {metric!r} (use 'exact', 'f1', or pass a judge)")

    per_example = [scorer(generate(ex.prompt), ex) for ex in golden.examples]
    mean = sum(per_example) / len(per_example)
    return EvalReport(
        task=golden.task,
        metric=used_metric,
        score=mean,
        per_example=per_example,
        n=len(per_example),
    )


def beats_incumbent(
    candidate: EvalReport,
    incumbent: EvalReport | None,
    *,
    margin: float = 0.0,
) -> bool:
    """The promotion gate: does ``candidate`` beat the ``incumbent`` by at least ``margin``?

    A missing incumbent (no promoted adapter for the task yet) means any candidate scoring
    above the base floor is an improvement — treated as a pass. ``margin`` requires a
    strict lift to avoid promoting noise-level ties (default 0.0 = strictly greater).
    """
    if incumbent is None:
        return candidate.score > 0.0
    return candidate.score > incumbent.score + margin


def objective_metric_for(task: str) -> str:
    """Default objective metric name for a task class (F1 for objective, else judge)."""
    return "f1" if task in _OBJECTIVE_CLASSES else "judge"


def default_judge(prompt: str, candidate: str, reference: str) -> bool:
    """STUB judge for draft/code (ARCHITECTURE §7): no judge model in Phase 4.

    A real implementation calls a strong local model (or an escalated frontier judge) to
    pick a winner. This deterministic placeholder counts a win when the candidate is at
    least as long as the reference and non-empty — enough to exercise the plumbing and
    obviously replaceable. Callers wanting a real gate must inject their own judge.
    """
    cand = candidate.strip()
    ref = reference.strip()
    return bool(cand) and len(cand) >= len(ref)


def as_golden_set(task: str, pairs: Sequence[tuple[str, str]]) -> GoldenSet:
    """Convenience: build a :class:`GoldenSet` from ``(prompt, expected)`` pairs."""
    return GoldenSet(task=task, examples=[GoldenExample(p, e) for p, e in pairs])


__all__ = [
    "GoldenExample",
    "GoldenSet",
    "EvalReport",
    "Generate",
    "Judge",
    "exact_match_score",
    "token_f1_score",
    "score_candidate",
    "beats_incumbent",
    "objective_metric_for",
    "default_judge",
    "as_golden_set",
]
