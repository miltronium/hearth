"""Pre-registration tests — the bar is declared, committed, and then enforced (§3.4).

The git checks are exercised against a real throwaway repository under ``tmp_path``: the
whole point of shelling out to git is that the answer is the one a reviewer would get, so
mocking it would test nothing.
"""

from __future__ import annotations

import subprocess

import pytest
import yaml

from hearth.training.eval import EvalConfig, EvalReport, as_golden_set, score_candidate
from hearth.training.prereg import (
    DEFAULT_BASELINES,
    PreRegError,
    load_prereg,
    require_prereg,
    template,
    verify_committed,
)

CONFIG = EvalConfig(temperature=0.0, max_tokens=24)
GOLDEN = as_golden_set("classify", [(f"p{i}", "A" if i % 2 else "B") for i in range(40)])


def _prereg_body(**overrides) -> dict:
    body = {
        "task": "classify",
        "hypothesis": "the adapter learns the QX convention",
        "golden_sha": GOLDEN.sha,
        "golden_version": "v1",
        "n": len(GOLDEN),
        "metric": "exact",
        "generation": {"temperature": 0.0, "max_tokens": 24, "seed": None, "system_hash": ""},
        "bar": {"test": "auto", "alpha": 0.05, "min_effect": 0.0, "min_n": 30},
        "stopping_rule": "one run at seed 0, no re-rolls",
        "kill_condition": "no lift over the base model at alpha 0.05",
    }
    body.update(overrides)
    return body


def _write(tmp_path, body: dict, name: str = "prereg.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return path


def _git(tmp_path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=test", *args],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )


def _repo(tmp_path):
    """An initialised git repo with an initial commit, so HEAD exists."""
    _git(tmp_path, "init", "-q")
    (tmp_path / "README").write_text("seed\n")
    _git(tmp_path, "add", "README")
    _git(tmp_path, "commit", "-qm", "seed")
    return tmp_path


def _candidate_report() -> EvalReport:
    return score_candidate(
        GOLDEN, lambda p: "A", metric="exact", model_id="base+cand", config=CONFIG
    )


# -- parsing ---------------------------------------------------------------------------


def test_load_prereg_reads_the_declared_bar(tmp_path):
    prereg = load_prereg(_write(tmp_path, _prereg_body()))
    assert prereg.task == "classify"
    assert prereg.golden_sha == GOLDEN.sha
    assert prereg.alpha == 0.05
    assert prereg.min_n == 30
    assert prereg.must_beat_baselines == ("empty", "majority_label", "copy_input")
    assert prereg.generation.fingerprint == CONFIG.fingerprint
    assert len(prereg.sha) == 64  # the file's own hash, for the promotion proof


def test_load_prereg_refuses_a_missing_required_field(tmp_path):
    body = _prereg_body()
    del body["golden_sha"]
    with pytest.raises(PreRegError, match="golden_sha"):
        load_prereg(_write(tmp_path, body))


def test_load_prereg_refuses_a_sampled_generation_config(tmp_path):
    """A bar registered at temperature 0.7 registers a re-rollable score (F4)."""
    body = _prereg_body(generation={"temperature": 0.7, "max_tokens": 24})
    with pytest.raises(PreRegError, match="temperature must be 0.0"):
        load_prereg(_write(tmp_path, body))


def test_load_prereg_refuses_a_missing_or_malformed_file(tmp_path):
    with pytest.raises(PreRegError, match="cannot read"):
        load_prereg(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a list\n")
    with pytest.raises(PreRegError, match="mapping"):
        load_prereg(bad)


def test_an_unedited_template_is_not_a_prereg_until_the_prose_is_written(tmp_path):
    """The tool writes the numbers; the operator must write the claim before it can gate.

    An unedited `prereg init` file used to load cleanly and, once committed, could gate a
    promotion — "a bar written by the tool is not a prereg", enforced only in a comment.
    """
    text = template(task="classify", golden_sha=GOLDEN.sha, metric="exact", max_tokens=24)
    path = tmp_path / "scaffold.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(PreRegError, match="hypothesis"):
        load_prereg(path)
    body = yaml.safe_load(text)
    body.update(
        hypothesis="the adapter learns the QX convention",
        stopping_rule="one run at seed 0, no re-rolls",
        kill_condition="no lift over the base model at alpha 0.05",
    )
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    prereg = load_prereg(path)
    assert prereg.golden_sha == GOLDEN.sha
    assert prereg.generation.fingerprint == CONFIG.fingerprint


@pytest.mark.parametrize("baselines", [[], ["empty", "majority"], "copy_input"])
def test_a_prereg_cannot_drop_a_default_baseline(tmp_path, baselines):
    body = _prereg_body()
    body["bar"] = {**body["bar"], "must_beat_baselines": baselines}
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    with pytest.raises(PreRegError, match="must include every default baseline"):
        load_prereg(path)


def test_a_prereg_may_add_baselines(tmp_path):
    body = _prereg_body()
    body["bar"] = {**body["bar"], "must_beat_baselines": [*DEFAULT_BASELINES, "extra"]}
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    assert "extra" in load_prereg(path).must_beat_baselines


# -- matching --------------------------------------------------------------------------


def test_mismatches_are_empty_for_the_registered_experiment(tmp_path):
    prereg = load_prereg(_write(tmp_path, _prereg_body()))
    assert prereg.mismatches(_candidate_report()) == ()


def test_mismatches_catch_a_swapped_golden_set(tmp_path):
    prereg = load_prereg(_write(tmp_path, _prereg_body(golden_sha="deadbeef" * 8)))
    problems = prereg.mismatches(_candidate_report())
    assert any("golden_sha" in p for p in problems)


def test_mismatches_catch_a_different_metric_or_decode_config(tmp_path):
    prereg = load_prereg(_write(tmp_path, _prereg_body(metric="f1")))
    assert any("metric" in p for p in prereg.mismatches(_candidate_report()))

    prereg = load_prereg(
        _write(tmp_path, _prereg_body(generation={"temperature": 0.0, "max_tokens": 64}))
    )
    assert any("decode config" in p for p in prereg.mismatches(_candidate_report()))


# -- git enforcement -------------------------------------------------------------------


def test_uncommitted_prereg_is_refused(tmp_path):
    _repo(tmp_path)
    path = _write(tmp_path, _prereg_body())
    status = verify_committed(path)
    assert not status.committed
    assert "not tracked" in status.reason
    with pytest.raises(PreRegError, match="not git-committed"):
        require_prereg(path, _candidate_report())


def test_committed_prereg_is_accepted(tmp_path):
    _repo(tmp_path)
    path = _write(tmp_path, _prereg_body())
    _git(tmp_path, "add", "prereg.yaml")
    _git(tmp_path, "commit", "-qm", "prereg: classify")

    status = verify_committed(path)
    assert status.committed
    assert len(status.commit) == 40
    prereg = require_prereg(path, _candidate_report())
    assert prereg.golden_sha == GOLDEN.sha


def test_a_prereg_edited_after_commit_is_refused(tmp_path):
    """Moving the bar after seeing the score is exactly what this prevents."""
    _repo(tmp_path)
    path = _write(tmp_path, _prereg_body())
    _git(tmp_path, "add", "prereg.yaml")
    _git(tmp_path, "commit", "-qm", "prereg: classify")
    _write(tmp_path, _prereg_body(bar={"alpha": 0.5, "min_n": 1}))  # alpha moved to 0.5

    status = verify_committed(path)
    assert not status.committed
    assert "uncommitted modifications" in status.reason


def test_a_prereg_outside_any_repository_is_refused(tmp_path):
    """Fail closed: no repo means the bar cannot be shown to predate the measurement."""
    path = _write(tmp_path, _prereg_body())
    status = verify_committed(path)
    assert not status.committed


def test_require_prereg_refuses_a_run_that_drifted_from_the_plan(tmp_path):
    _repo(tmp_path)
    path = _write(tmp_path, _prereg_body(golden_sha="deadbeef" * 8))
    _git(tmp_path, "add", "prereg.yaml")
    _git(tmp_path, "commit", "-qm", "prereg")
    with pytest.raises(PreRegError, match="does not match"):
        require_prereg(path, _candidate_report())


# -- B-062: the bar is range-checked at load --------------------------------------------
#
# Every comparison in the gate is False under NaN, a negative margin makes the baseline
# clause vacuous, and alpha=1 / min_n=1 switch off significance and the size floor. A
# prereg is the operator's bar, so it may only make the gate STRICTER than CLAUDE.md §7:
# alpha in (0, 0.05], min_effect >= 0, min_n >= 30, a known test, every number finite.


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("alpha", float("nan"), "alpha"),
        ("alpha", float("inf"), "alpha"),
        ("alpha", 1.0, "alpha"),
        ("alpha", 0.051, "alpha"),
        ("alpha", 0.0, "alpha"),
        ("alpha", -0.01, "alpha"),
        ("alpha", "loose", "alpha"),
        ("alpha", True, "alpha"),
        ("min_effect", float("nan"), "min_effect"),
        ("min_effect", float("inf"), "min_effect"),
        ("min_effect", -1.0, "min_effect"),
        ("min_effect", -1e-9, "min_effect"),
        ("min_n", 1, "min_n"),
        ("min_n", 5, "min_n"),
        ("min_n", 29, "min_n"),
        ("min_n", 30.5, "min_n"),
        ("min_n", float("nan"), "min_n"),
        ("min_n", "thirty", "min_n"),
        ("test", "t-test", "test"),
        ("test", "", "test"),
    ],
)
def test_a_prereg_bar_that_would_loosen_the_gate_is_refused(tmp_path, field, value, match):
    body = _prereg_body()
    body["bar"] = {**body["bar"], field: value}
    with pytest.raises(PreRegError, match=match):
        load_prereg(_write(tmp_path, body))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("alpha", 0.05),
        ("alpha", 0.01),
        ("min_effect", 0.0),
        ("min_effect", 0.1),
        ("min_n", 30),
        ("min_n", 200),
        ("test", "auto"),
        ("test", "mcnemar"),
        ("test", "bootstrap"),
    ],
)
def test_a_prereg_may_tighten_the_bar(tmp_path, field, value):
    body = _prereg_body()
    body["bar"] = {**body["bar"], field: value}
    prereg = load_prereg(_write(tmp_path, body))
    assert getattr(prereg, field) == value


def test_the_reviewer_nan_and_loose_bars_no_longer_load(tmp_path):
    """The B-062 reproducer verbatim: both bars used to load and PASS a 0.033 vs 1.0 run."""
    for bar in ("{alpha: .nan, min_effect: .nan, min_n: 30}",
                "{alpha: 1.0, min_effect: -1.0, min_n: 1}"):
        path = tmp_path / "p.yaml"
        path.write_text(
            f"task: t\ngolden_sha: {'cd' * 32}\nmetric: exact\nhypothesis: h\n"
            f"stopping_rule: s\nkill_condition: k\nbar: {bar}\n",
            encoding="utf-8",
        )
        with pytest.raises(PreRegError):
            load_prereg(path)
