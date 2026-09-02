"""Eval harness tests — scoring metrics, report provenance, and the promotion gate.

The gate's job is to refuse: most of these tests are about the ways a candidate that
*looks* better is not promotable (LEARNING_plan §3). The replay of HEARTH's own celebrated
promotion lives in ``tests/test_eval_gate_replay.py``.
"""

from __future__ import annotations

import pytest

from hearth.training.eval import (
    DEFAULT_MIN_N,
    EvalConfig,
    EvalReport,
    GateProvenanceError,
    as_golden_set,
    baseline_reports,
    beats_incumbent,
    check_determinism,
    default_judge,
    evaluate_gate,
    exact_match_score,
    objective_metric_for,
    score_candidate,
    token_f1_score,
)

CONFIG = EvalConfig(temperature=0.0, max_tokens=24)


def _report(
    per_example,
    *,
    task="classify",
    metric="exact_match",
    golden_sha="sha-A",
    config=CONFIG,
    model_id="m",
) -> EvalReport:
    """Build an EvalReport straight from a per-example vector (no generation)."""
    values = [float(v) for v in per_example]
    return EvalReport(
        task=task,
        metric=metric,
        score=sum(values) / len(values),
        per_example=values,
        n=len(values),
        golden_sha=golden_sha,
        config=config,
        model_id=model_id,
    )


def _significant_pair():
    """A candidate that genuinely beats an incumbent: n=40, b=6, c=0, p=0.0156."""
    incumbent = [1.0] * 30 + [0.0] * 10
    candidate = [1.0] * 36 + [0.0] * 4
    return _report(candidate, model_id="cand"), _report(incumbent, model_id="inc")


# -- metrics ---------------------------------------------------------------------------


def test_exact_match_is_case_and_space_insensitive():
    assert exact_match_score("Yes", " yes ") == 1.0
    assert exact_match_score("yes", "no") == 0.0


def test_token_f1_partial_overlap():
    assert token_f1_score("the cat sat", "the cat sat") == 1.0
    assert token_f1_score("", "") == 1.0
    assert token_f1_score("cat", "") == 0.0
    # Partial overlap is between 0 and 1.
    score = token_f1_score("the cat", "the dog")
    assert 0.0 < score < 1.0


def test_score_candidate_exact_metric():
    golden = as_golden_set("classify", [("q1", "yes"), ("q2", "no")])
    answers = {"q1": "yes", "q2": "wrong"}
    report = score_candidate(golden, lambda p: answers[p], metric="exact")
    assert report.metric == "exact_match"
    assert report.n == 2
    assert report.score == 0.5
    assert report.per_example == [1.0, 0.0]  # the pairing is what the gate needs


def test_score_candidate_with_injected_judge():
    golden = as_golden_set("draft", [("write intro", "a fine intro")])
    # Judge: candidate wins whenever it is non-empty.
    report = score_candidate(
        golden, lambda p: "my draft", judge=lambda prompt, cand, ref: bool(cand)
    )
    assert report.metric == "judge_win_rate"
    assert report.score == 1.0


def test_score_candidate_refuses_metric_and_judge_together():
    """A judge used to silently override `metric`, so `exact` quietly became a judge."""
    golden = as_golden_set("draft", [("p", "e")])
    with pytest.raises(ValueError, match="not both"):
        score_candidate(golden, lambda p: "x", metric="exact", judge=lambda a, b, c: True)


def test_score_candidate_rejects_empty_golden_set():
    with pytest.raises(ValueError):
        score_candidate(as_golden_set("x", []), lambda p: "")


def test_objective_metric_for():
    assert objective_metric_for("extract") == "f1"
    assert objective_metric_for("draft") == "judge"


# -- provenance ------------------------------------------------------------------------


def test_report_carries_the_identity_of_what_it_measured():
    golden = as_golden_set("classify", [("q1", "yes")], version="v3")
    report = score_candidate(
        golden, lambda p: "yes", metric="exact", model_id="base+cand-1", config=CONFIG
    )
    assert report.golden_sha == golden.sha
    assert report.golden_version == "v3"
    assert report.model_id == "base+cand-1"
    assert report.config_fingerprint == CONFIG.fingerprint
    assert report.has_provenance


def test_golden_sha_is_content_identity_not_file_identity():
    a = as_golden_set("classify", [("q1", "yes"), ("q2", "no")])
    reordered = as_golden_set("classify", [("q2", "no"), ("q1", "yes")])
    edited = as_golden_set("classify", [("q1", "yes"), ("q2", "maybe")])
    assert a.sha == reordered.sha  # same items, any order
    assert a.sha != edited.sha  # one changed answer is a different set


def test_config_fingerprint_separates_decode_settings():
    assert EvalConfig(max_tokens=24).fingerprint != EvalConfig(max_tokens=64).fingerprint
    assert EvalConfig(temperature=0.7).fingerprint != EvalConfig(temperature=0.0).fingerprint
    assert EvalConfig(temperature=0.0).deterministic
    assert not EvalConfig(temperature=0.7).deterministic


def test_report_round_trips_through_json():
    report = _report([1.0, 0.0, 1.0])
    again = EvalReport.from_json(report.to_json())
    assert again == report


# -- the gate --------------------------------------------------------------------------


def test_gate_passes_a_real_significant_lift():
    candidate, incumbent = _significant_pair()
    gate = evaluate_gate(candidate, incumbent)
    assert gate.passed
    assert bool(gate) is True
    assert gate.test == "mcnemar_exact"
    assert (gate.b, gate.c, gate.n) == (6, 0, 40)
    assert gate.p_value == pytest.approx(0.5**6)
    assert gate.reason == "gate passed"
    proof = gate.as_proof()
    assert proof["gate"] == "verified" and proof["gate_passed"] is True
    assert proof["golden_sha"] == "sha-A"


def test_gate_refuses_a_higher_mean_that_is_not_significant():
    """A2/A5 in miniature: more wins than losses is not the same as a real difference."""
    candidate = _report([1.0, 1.0, 1.0] + [1.0] * 30 + [0.0] * 7)
    incumbent = _report([0.0, 0.0, 0.0] + [1.0] * 30 + [0.0] * 7)
    gate = evaluate_gate(candidate, incumbent, min_n=10)
    assert candidate.score > incumbent.score  # the old gate would have promoted this
    assert not gate.passed
    assert gate.p_value == pytest.approx(0.125)
    assert any("not significant" in r for r in gate.reasons)


def test_gate_refuses_when_there_is_no_incumbent_report():
    """F2: the first adapter is not free — with nothing promoted, base IS the incumbent."""
    candidate = _report([1.0] * 40)
    gate = evaluate_gate(candidate, None)
    assert not gate.passed
    assert gate.incumbent_role == "none"
    assert any("BASE MODEL is the incumbent" in r for r in gate.reasons)


def test_gate_refuses_a_candidate_that_only_beats_zero():
    """The old rule promoted anything scoring > 0.0; an adapter emitting "the" qualified."""
    golden = as_golden_set("extract", [(f"q{i}", "the ticket id is ABC-1") for i in range(40)])
    candidate = score_candidate(golden, lambda p: "the", metric="f1", config=CONFIG)
    base = score_candidate(golden, lambda p: "the ticket id is ABC-1", metric="f1", config=CONFIG)
    assert candidate.score > 0.0
    gate = evaluate_gate(candidate, base, incumbent_role="base")
    assert not gate.passed


def test_gate_enforces_a_minimum_golden_set_size():
    candidate = _report([1.0] * 6)
    incumbent = _report([0.0] * 6)
    small = evaluate_gate(candidate, incumbent)
    assert not small.passed
    assert any(f"min_n={DEFAULT_MIN_N}" in r for r in small.reasons)
    # The very same vectors clear the bar once the policy floor is lowered to the
    # mathematical minimum — the refusal above is about power, not about the data.
    assert evaluate_gate(candidate, incumbent, min_n=5).passed


def test_gate_requires_a_lift_not_just_significance():
    candidate = _report([1.0] * 30)
    incumbent = _report([1.0] * 30)
    gate = evaluate_gate(candidate, incumbent)
    assert not gate.passed
    assert any("no lift" in r for r in gate.reasons)


def test_gate_honours_a_margin():
    candidate, incumbent = _significant_pair()
    assert evaluate_gate(candidate, incumbent, margin=0.10).passed  # lift is 0.15
    assert not evaluate_gate(candidate, incumbent, margin=0.20).passed


def test_gate_raises_on_a_golden_set_mismatch():
    """A4: v3 vs v2 is not a losing candidate, it is an ill-posed question."""
    candidate, _ = _significant_pair()
    other = _report([0.0] * 40, golden_sha="sha-B")
    with pytest.raises(GateProvenanceError, match="golden-set mismatch"):
        evaluate_gate(candidate, other)


def test_gate_raises_on_a_decode_config_mismatch():
    candidate, _ = _significant_pair()
    other = _report([0.0] * 40, config=EvalConfig(max_tokens=64))
    with pytest.raises(GateProvenanceError, match="decode-config mismatch"):
        evaluate_gate(candidate, other)


def test_gate_raises_on_a_metric_mismatch():
    candidate, _ = _significant_pair()
    other = _report([0.0] * 40, metric="token_f1")
    with pytest.raises(GateProvenanceError, match="metric mismatch"):
        evaluate_gate(candidate, other)


def test_gate_raises_without_a_per_example_vector():
    bare = EvalReport(task="classify", metric="exact_match", score=1.0, golden_sha="sha-A",
                      config=CONFIG)
    incumbent = EvalReport(task="classify", metric="exact_match", score=0.2, golden_sha="sha-A",
                           config=CONFIG)
    with pytest.raises(GateProvenanceError):
        evaluate_gate(bare, incumbent)


def test_gate_refuses_reports_with_no_provenance_at_all():
    """Two anonymous vectors are comparable to each other and to nothing else."""
    candidate = _report([1.0] * 36 + [0.0] * 4, golden_sha="", config=None)
    incumbent = _report([1.0] * 30 + [0.0] * 10, golden_sha="", config=None)
    gate = evaluate_gate(candidate, incumbent)
    assert not gate.passed
    assert any("no provenance" in r for r in gate.reasons)


def test_gate_uses_the_bootstrap_for_continuous_metrics():
    candidate = _report([0.9, 0.85, 0.95] * 14, metric="token_f1")
    incumbent = _report([0.2, 0.25, 0.15] * 14, metric="token_f1")
    gate = evaluate_gate(candidate, incumbent, bootstrap_iterations=500)
    assert gate.test == "paired_bootstrap"
    assert gate.passed
    assert gate.ci_low is not None and gate.ci_low > 0.0


def test_gate_rejects_an_unknown_test_name():
    candidate, incumbent = _significant_pair()
    with pytest.raises(ValueError, match="unknown test"):
        evaluate_gate(candidate, incumbent, test="t-test")


# -- degenerate baselines --------------------------------------------------------------


def test_baseline_reports_score_the_three_degenerate_answers():
    golden = as_golden_set("classify", [("p1", "A"), ("p2", "A"), ("p3", "B")])
    baselines = baseline_reports(golden, metric="exact", config=CONFIG)
    assert baselines["empty"].score == 0.0
    assert baselines["majority_label"].score == pytest.approx(2 / 3)
    assert baselines["copy_input"].score == 0.0
    assert all(b.golden_sha == golden.sha for b in baselines.values())


def test_gate_fails_a_candidate_that_cannot_beat_the_majority_label():
    """A candidate can beat a weak incumbent and still have learned nothing."""
    # Every second item is the majority label, so always-guessing it scores 0.5.
    golden = as_golden_set(
        "classify", [(f"p{i}", "A" if i % 2 == 0 else f"B{i}") for i in range(40)]
    )
    # The candidate has learned exactly one thing: say "A". It still beats an incumbent
    # that says nothing useful, significantly and by 0.5 — and it is worth nothing.
    candidate = score_candidate(golden, lambda p: "A", metric="exact", config=CONFIG)
    incumbent = score_candidate(golden, lambda p: "nothing", metric="exact", config=CONFIG)
    baselines = baseline_reports(golden, metric="exact", config=CONFIG)
    gate = evaluate_gate(candidate, incumbent, incumbent_role="base", baselines=baselines)
    assert candidate.score > incumbent.score
    assert not gate.passed
    assert any("majority_label" in r for r in gate.reasons)
    assert gate.baselines["majority_label"] == pytest.approx(0.5)


def test_a_constant_string_candidate_is_refused_on_every_task_class():
    """A1: a fixed 10 KB blob used to score a perfect judge win-rate on draft/code."""
    blob = "lorem ipsum " * 900
    golden = as_golden_set("draft", [(f"write {i}", "a short reference") for i in range(40)])

    # The judged path no longer exists by default: there is nothing to pass.
    with pytest.raises(NotImplementedError):
        score_candidate(golden, lambda p: blob, judge=default_judge)

    # And on an objective metric the blob loses to the baselines and to the base model.
    candidate = score_candidate(golden, lambda p: blob, metric="f1", config=CONFIG)
    base = score_candidate(golden, lambda p: "a short reference", metric="f1", config=CONFIG)
    baselines = baseline_reports(golden, metric="f1", config=CONFIG)
    gate = evaluate_gate(candidate, base, incumbent_role="base", baselines=baselines)
    assert not gate.passed


def test_default_judge_refuses_to_run():
    """F1: a gate that cannot run is safer than a gate that always passes."""
    with pytest.raises(NotImplementedError, match="no default judge"):
        default_judge("p", "a longer candidate", "short")


# -- determinism -----------------------------------------------------------------------


def test_check_determinism_catches_a_sampling_generator():
    """A3: a re-rollable score must be caught before it reaches a promotion."""
    golden = as_golden_set("classify", [(f"p{i}", "A") for i in range(5)])
    assert check_determinism(golden, lambda p: "A") == ()

    calls = {"n": 0}

    def _sampling(prompt: str) -> str:
        calls["n"] += 1
        return f"answer-{calls['n']}"

    drift = check_determinism(golden, _sampling)
    assert len(drift) == 3  # the default sample size


# -- the legacy wrapper ----------------------------------------------------------------


def test_beats_incumbent_still_answers_the_mean_only_question():
    inc = EvalReport(task="extract", metric="f1", score=0.80)
    better = EvalReport(task="extract", metric="f1", score=0.85)
    worse = EvalReport(task="extract", metric="f1", score=0.75)
    tie = EvalReport(task="extract", metric="f1", score=0.80)
    assert beats_incumbent(better, inc) is True
    assert beats_incumbent(worse, inc) is False
    assert beats_incumbent(tie, inc) is False  # a tie does not beat


def test_beats_incumbent_no_longer_promotes_on_a_missing_incumbent():
    """F2 at the wrapper: "no incumbent" used to mean "any score above zero wins"."""
    cand = EvalReport(task="extract", metric="f1", score=0.5)
    assert beats_incumbent(cand, None) is False
