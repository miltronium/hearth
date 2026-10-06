"""B-126: guards in the measurement ledger / provenance that no test protected.

Each test below was written against a mutant that survived the round-4 review's mutation run:
the ledger lookup by MAC (vs. "the latest record"), and the committed-blob duplicate check.
(The seq / prev / flock cases live in tests/test_training_ledger.py.)
"""

from __future__ import annotations

import subprocess

import pytest
import test_promotion_evidence as pe
from test_promotion_evidence import World

from hearth.training.eval import parse_golden_jsonl
from hearth.training.prereg import committed_golden_problems


@pytest.fixture(autouse=True)
def fake_provider(monkeypatch):
    monkeypatch.setattr("hearth.cli.select_provider", lambda settings: pe._Provider())
    pe.allow_test_backends(monkeypatch)


def test_a_later_measurement_of_another_adapter_does_not_block_a_legitimate_promotion(tmp_path):
    """The report is matched to ITS ledger record by MAC. Taking "the latest record" instead
    would compare the report with whatever was measured afterwards and refuse a valid
    promotion (or, worse, accept a report against an unrelated record)."""
    world = World(tmp_path)
    world.adapter("extract-1")
    world.registered()
    world.eval_report("extract-1")
    world.adapter("extract-2")
    world.eval("extract-2")  # appends a later, unrelated ledger record
    result = world.promote("extract-1")
    assert result.exit_code == 0, result.output
    assert world.status("extract-1") == "promoted"


def test_a_committed_golden_set_that_repeats_a_prompt_is_a_problem(tmp_path):
    repo = tmp_path / "evals"
    repo.mkdir()
    text = "".join(
        f'{{"task": "extract", "prompt": "{p}", "expected": "A"}}\n' for p in ("q0", "q0", "q1")
    )
    (repo / "golden.jsonl").write_text(text)
    for args in (["init", "-q"], ["add", "golden.jsonl"],
                 ["-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qm", "g"]):
        subprocess.run(["git", *args], cwd=repo, check=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                            capture_output=True, text=True).stdout.strip()
    sha = parse_golden_jsonl(text, task="extract").sha
    problems = committed_golden_problems(
        {"repo_root": str(repo), "commit": commit, "rel_path": "golden.jsonl"},
        task="extract", golden_sha=sha,
    )
    assert any("repeats a prompt" in p for p in problems), problems
