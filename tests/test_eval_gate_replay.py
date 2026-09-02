"""A5 — replay HEARTH's own celebrated promotion through the new gate. It must FAIL.

docs/RESULTS.md "Run 2 — `classify` (ticket-routing) adapter" records the project's single
validated lift: on a 5-example golden set the base model scored 0.20 (it parrots ``QX-7``
for everything) and the LoRA candidate scored 1.00, and the gate of the day said
``beats_incumbent(candidate, base) = True``. It was then promoted with two floats typed on
the command line.

**Why this test asserts failure.** The effect is almost certainly real — the mechanism is
understood, and the base model is provably at chance on an arbitrary org convention it has
never seen. But *the golden set is one example too small to prove it*. The contingency is
b=4 (items the candidate wins and the base loses), c=0, and the exact one-sided McNemar
p-value is 0.5**4 = 0.0625, which does not clear alpha=0.05. Four clean wins is the most
that 5 items with a partially-correct incumbent can produce, and four clean wins is not
significant at the conventional bar.

So the new gate has to be willing to refuse the project's own best result. A gate that
cannot refuse a promotion its authors are proud of is not a gate, it is a formality —
which is exactly what the old one was. If this test ever starts passing the candidate, the
statistics have been quietly loosened and the gate is no longer real.

(The plan's kill condition applies and is the honest reading: if nothing HEARTH can
currently train is promotable under this gate, that *is* the finding — grow the golden sets
to n>=30, do not weaken the gate.)
"""

from __future__ import annotations

import pytest

from hearth.training.eval import (
    EvalConfig,
    as_golden_set,
    baseline_reports,
    evaluate_gate,
    score_candidate,
)
from hearth.training.stats import mcnemar_exact_p, min_n_for_alpha, smallest_achievable_p

# The exact per-example table printed in docs/RESULTS.md, in order.
#   expected | base            | candidate
#   QX-7     | 'QX-7'  correct | 'QX-7'  correct
#   QX-2     | 'QX-7'  wrong   | 'QX-2'  correct
#   QX-9     | 'QX-7'  wrong   | 'QX-9'  correct
#   QX-4     | 'QX-7'  wrong   | 'QX-4'  correct
#   QX-1     | 'QX-7'  wrong   | 'QX-1'  correct
RESULTS_GOLDEN = [
    ("The API gateway is returning 502s for all downstream calls", "QX-7"),
    ("The DNS resolver keeps timing out on internal lookups", "QX-2"),
    ("Nightly ETL job failed with a schema mismatch", "QX-9"),
    ("The write-ahead log partition ran out of space", "QX-4"),
    ("The settings modal renders behind the nav bar on mobile", "QX-1"),
]

CONFIG = EvalConfig(temperature=0.0, max_tokens=24)


def _replay_reports():
    """Score the RESULTS.md run exactly as it happened, through the real harness."""
    golden = as_golden_set("classify", RESULTS_GOLDEN, version="results-md")
    answers = {prompt: expected for prompt, expected in RESULTS_GOLDEN}
    candidate = score_candidate(
        golden,
        lambda prompt: answers[prompt],  # the adapter learned the convention: 5/5
        metric="exact",
        model_id="Qwen2.5-Coder-7B+classify-20260710T020135Z",
        config=CONFIG,
    )
    base = score_candidate(
        golden,
        lambda prompt: "QX-7",  # the base model parrots the one example it saw
        metric="exact",
        model_id="Qwen2.5-Coder-7B",
        config=CONFIG,
    )
    return golden, candidate, base


def test_replay_reproduces_the_published_scores():
    """Sanity: the replay really is the RESULTS.md run — 0.20 vs 1.00 on 5 examples."""
    _, candidate, base = _replay_reports()
    assert base.score == pytest.approx(0.20)
    assert candidate.score == pytest.approx(1.00)
    assert candidate.n == 5
    assert candidate.per_example == [1.0, 1.0, 1.0, 1.0, 1.0]
    assert base.per_example == [1.0, 0.0, 0.0, 0.0, 0.0]


def test_the_contingency_is_b4_c0_and_p_is_0_0625():
    assert mcnemar_exact_p(4, 0) == pytest.approx(0.0625)


def test_results_md_promotion_is_REFUSED_by_the_new_gate():
    """The headline acceptance test: passed must be False, at p = 0.0625 > alpha = 0.05.

    Two independent reasons fire, and both are load-bearing:
      1. n = 5 is below the gate's min_n policy floor;
      2. even ignoring that floor, the result is not significant (see the next test).
    """
    golden, candidate, base = _replay_reports()
    gate = evaluate_gate(
        candidate,
        base,
        incumbent_role="base",
        baselines=baseline_reports(golden, metric="exact", config=CONFIG),
        candidate_id="classify-20260710T020135Z",
        incumbent_id="Qwen2.5-Coder-7B",
    )

    assert gate.passed is False, (
        "the gate promoted the RESULTS.md result — it is not a real gate. "
        f"reasons={gate.reasons}"
    )
    assert gate.test == "mcnemar_exact"
    assert (gate.b, gate.c, gate.n) == (4, 0, 5)
    assert gate.p_value == pytest.approx(0.0625)
    assert any("not significant" in r for r in gate.reasons)
    assert any("too small" in r for r in gate.reasons)
    # And the proof records the refusal in a form an auditor can re-derive.
    proof = gate.as_proof()
    assert proof["gate_passed"] is False
    assert proof["p_value"] == pytest.approx(0.0625)
    assert proof["incumbent_role"] == "base"


def test_it_still_fails_when_the_size_floor_is_taken_out_of_the_argument():
    """Isolate the statistics: with min_n lowered to 5, significance alone still refuses.

    This is the clause that matters. The size floor is policy and could be argued about;
    p = 0.0625 > 0.05 is arithmetic and cannot.
    """
    golden, candidate, base = _replay_reports()
    gate = evaluate_gate(
        candidate,
        base,
        incumbent_role="base",
        baselines=baseline_reports(golden, metric="exact", config=CONFIG),
        min_n=5,
    )
    assert gate.passed is False
    assert gate.reasons == (
        "not significant: mcnemar_exact p=0.0625 > alpha=0.05 (b=4, c=0)",
    )


def test_one_more_golden_example_would_have_licensed_it():
    """The finding is not "the lift is fake" — it is "the set is one example too small".

    Same clean sweep with a sixth item the base also misses gives b=5, c=0, p=0.031, and
    the gate passes on the statistics. This is what "grow the golden sets" buys.
    """
    sixth = ("The billing webhook retries forever on a 409", "QX-2")
    golden = as_golden_set("classify", [*RESULTS_GOLDEN, sixth])
    answers = {p: e for p, e in golden_pairs(golden)}
    candidate = score_candidate(golden, lambda p: answers[p], metric="exact", config=CONFIG)
    base = score_candidate(golden, lambda p: "QX-7", metric="exact", config=CONFIG)
    gate = evaluate_gate(candidate, base, incumbent_role="base", min_n=6)
    assert (gate.b, gate.c) == (5, 0)
    assert gate.p_value == pytest.approx(0.03125)
    assert gate.passed is True


def test_five_is_the_smallest_golden_set_that_could_ever_promote_anything():
    """The minimum-n answer, stated as an assertion rather than a claim in a doc.

    At n=5 the best reachable p is 0.5**5 = 0.03125 — but only via b=5, c=0, i.e. an
    incumbent that misses *every* item the candidate hits. RESULTS.md's incumbent got one
    item right, which caps it at b=4 and p=0.0625. Below n=5 no outcome clears alpha=0.05
    at all.
    """
    assert smallest_achievable_p(4) == pytest.approx(0.0625)  # nothing at n=4 clears 0.05
    assert smallest_achievable_p(5) == pytest.approx(0.03125)
    assert min_n_for_alpha(0.05) == 5


def golden_pairs(golden):
    """(prompt, expected) pairs of a golden set — small local helper for readability."""
    return [(ex.prompt, ex.expected) for ex in golden.examples]
