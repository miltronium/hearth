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


def test_template_round_trips_through_the_loader(tmp_path):
    text = template(task="classify", golden_sha=GOLDEN.sha, metric="exact", max_tokens=24)
    path = tmp_path / "scaffold.yaml"
    path.write_text(text, encoding="utf-8")
    prereg = load_prereg(path)
    assert prereg.golden_sha == GOLDEN.sha
    assert prereg.generation.fingerprint == CONFIG.fingerprint
    # The prose is left for the operator — a bar written by the tool is not a prereg.
    assert prereg.hypothesis == ""


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
