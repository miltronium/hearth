"""Eval harness tests — scoring metrics and the promotion gate (candidate must beat)."""

from __future__ import annotations

import pytest

from hearth.training.eval import (
    EvalReport,
    as_golden_set,
    beats_incumbent,
    default_judge,
    exact_match_score,
    score_candidate,
    token_f1_score,
)


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


def test_score_candidate_with_injected_judge():
    golden = as_golden_set("draft", [("write intro", "a fine intro")])
    # Judge: candidate wins whenever it is non-empty.
    report = score_candidate(
        golden, lambda p: "my draft", judge=lambda prompt, cand, ref: bool(cand)
    )
    assert report.metric == "judge_win_rate"
    assert report.score == 1.0


def test_score_candidate_rejects_empty_golden_set():
    with pytest.raises(ValueError):
        score_candidate(as_golden_set("x", []), lambda p: "")


def test_gate_requires_strict_win_over_incumbent():
    inc = EvalReport(task="extract", metric="f1", score=0.80)
    better = EvalReport(task="extract", metric="f1", score=0.85)
    worse = EvalReport(task="extract", metric="f1", score=0.75)
    tie = EvalReport(task="extract", metric="f1", score=0.80)
    assert beats_incumbent(better, inc) is True
    assert beats_incumbent(worse, inc) is False
    assert beats_incumbent(tie, inc) is False  # a tie does not beat


def test_gate_passes_any_positive_when_no_incumbent():
    cand = EvalReport(task="extract", metric="f1", score=0.5)
    assert beats_incumbent(cand, None) is True
    zero = EvalReport(task="extract", metric="f1", score=0.0)
    assert beats_incumbent(zero, None) is False


def test_default_judge_is_a_stub_length_heuristic():
    assert default_judge("p", "a longer candidate", "short") is True
    assert default_judge("p", "", "anything") is False
