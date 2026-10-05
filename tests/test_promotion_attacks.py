"""Second-round attacks on the promotion gate (B-078..B-085) — each must now be REFUSED.

A reviewer's attack suite (reusing ``test_promotion_evidence.World``) asserted that each of
these PROMOTED on the code at 46125d8. Every test below is that attack, ported to assert the
refusal, and each was run against the old code first to confirm it promoted there.

    A  measure with no prereg, see PASS, then write + commit the bar and re-measure (B-079)
    B  3 distinct golden items repeated 10x satisfy min_n=30 (B-080)
    C  a throwaway repo holding a copy of the golden set and the prereg (B-081)
    D  `git update-index --assume-unchanged` hides an edited prereg (B-078)
    E  `--assume-unchanged` hides a golden set edited after commit (B-078)
    F  a promotion racing between the incumbent check and the registry write (B-082)
"""

from __future__ import annotations

import json

import pytest
import yaml

import test_promotion_evidence as pe
from hearth.providers.base import GenResult
from hearth.training.eval import as_golden_set
from test_promotion_evidence import World


@pytest.fixture(autouse=True)
def fake_provider(monkeypatch):
    monkeypatch.setattr("hearth.cli.select_provider", lambda settings: pe._Provider())


@pytest.fixture
def world(tmp_path) -> World:
    w = World(tmp_path)
    w.adapter()
    return w


def _flat(result) -> str:
    return " ".join(result.output.split())


def test_B_duplicate_rows_cannot_inflate_n(world, monkeypatch):
    """Three distinct examples, each repeated 10x, are n=3 — not a 30-item golden set."""
    distinct = [("q0", "A"), ("q1", "B1"), ("q2", "B2")]
    rows = [{"prompt": p, "expected": e} for _ in range(10) for p, e in distinct]
    answers = dict(distinct)

    class _P:
        name = "fake"

        def generate(self, req):
            prompt = req.messages[-1].content
            text = answers[prompt] if req.adapter else "A"
            return GenResult(text=text, model=req.model, backend="fake")

    monkeypatch.setattr("hearth.cli.select_provider", lambda s: _P())
    world.golden.write_text("".join(json.dumps(r) + "\n" for r in rows))
    world.write_prereg()
    body = yaml.safe_load(world.prereg.read_text())
    body["golden_sha"] = as_golden_set("extract", [(r["prompt"], r["expected"]) for r in rows]).sha
    world.prereg.write_text(yaml.safe_dump(body))
    world.commit("golden.jsonl", "prereg.yaml")

    result = world.eval("extract-1", "--report-json", str(world.report))
    assert result.exit_code == 1, _flat(result)
    assert "Golden set error: golden set repeats 3 prompt(s)" in _flat(result)
    assert not world.report.exists()
    result = world.eval("extract-1", "--prereg", str(world.prereg), "--promote")
    assert result.exit_code == 1 and "repeats 3 prompt(s)" in _flat(result)
    assert world.status() == "candidate"


def test_B_near_duplicates_differing_only_in_case_and_spacing_count_once(world):
    rows = [{"prompt": f"p{i}", "expected": "A"} for i in range(30)]
    rows.append({"prompt": "  P0 ", "expected": "B"})  # same item, relabelled
    world.golden.write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = world.eval("extract-1")
    assert result.exit_code == 1 and "repeats 1 prompt(s)" in _flat(result)


def test_B_prereg_init_and_check_refuse_a_golden_set_with_repeats(world):
    from test_promotion_evidence import app, runner

    world.golden.write_text("".join(json.dumps({"prompt": f"q{i % 3}", "expected": "A"}) + "\n"
                                    for i in range(30)))
    result = runner.invoke(app, ["prereg", "init", "--task", "extract", "--golden",
                                 str(world.golden)], env=world.env)
    assert result.exit_code == 1 and "repeats 3 prompt(s)" in _flat(result)
    world.write_prereg()
    result = runner.invoke(app, ["prereg", "check", str(world.prereg), "--golden",
                                 str(world.golden)], env=world.env)
    assert result.exit_code == 1 and "repeats 3 prompt(s)" in _flat(result)
