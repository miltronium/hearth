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


# -- D / E: an edit hidden from `git diff` by the index (B-078) ---------------------------


def test_D_an_assume_unchanged_prereg_edit_is_refused(world):
    """Commit an honest bar, edit the working tree, hide the edit from `git diff`."""
    world.write_prereg()
    honest = yaml.safe_load(world.prereg.read_text())
    real_sha = honest["golden_sha"]
    honest["golden_sha"] = "0" * 64  # the committed bar pins a different golden set
    world.prereg.write_text(yaml.safe_dump(honest))
    world.commit("golden.jsonl", "prereg.yaml")
    honest["golden_sha"] = real_sha  # after seeing the score: move the bar...
    world.prereg.write_text(yaml.safe_dump(honest))
    pe._git(world.repo, "update-index", "--assume-unchanged", "prereg.yaml")  # ...and hide it
    assert pe._git(world.repo, "status", "--porcelain") == ""  # git itself is fooled
    world.eval_report()
    result = world.promote()
    assert result.exit_code == 1, _flat(result)
    assert "uncommitted modifications" in _flat(result)
    assert world.status() == "candidate"
    result = world.eval("extract-1", "--prereg", str(world.prereg), "--promote")
    assert result.exit_code == 1 and "uncommitted modifications" in _flat(result)
    assert world.status() == "candidate"


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_E_a_golden_set_edited_and_hidden_from_git_is_refused(world, flag):
    """Committed golden = 40 items the candidate loses; working tree = the cherry-picked set."""
    lose = [{"prompt": f"p{i}", "expected": "Z"} for i in range(40)]
    world.golden.write_text("".join(json.dumps(r) + "\n" for r in lose))
    world.commit("golden.jsonl")
    world.golden.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in pe.ROWS))
    pe._git(world.repo, "update-index", flag, "golden.jsonl")
    world.write_prereg()
    world.commit("prereg.yaml")
    assert '"Z"' in pe._git(world.repo, "show", "HEAD:golden.jsonl")
    payload = world.eval_report()
    assert payload["golden_git"]["committed"] is False
    result = world.promote()
    assert result.exit_code == 1, _flat(result)
    assert "golden set was not committed" in _flat(result)
    assert world.status() == "candidate"
    result = world.eval("extract-1", "--prereg", str(world.prereg), "--promote")
    assert result.exit_code == 1 and "golden set was not committed" in _flat(result)


def test_E_a_report_claiming_a_committed_golden_set_is_checked_against_the_blob(world):
    """A key-holder re-signs `golden_git.committed: true`: promotion re-reads the blob itself."""
    lose = [{"prompt": f"p{i}", "expected": "Z"} for i in range(40)]
    world.golden.write_text("".join(json.dumps(r) + "\n" for r in lose))
    world.commit("golden.jsonl")
    world.golden.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in pe.ROWS))
    world.write_prereg()
    world.commit("prereg.yaml")
    payload = world.eval_report()
    del payload["signature"]
    golden_commit = pe._git(world.repo, "log", "-1", "--format=%H", "--", "golden.jsonl")
    payload["golden_git"].update(committed=True, commit=golden_commit, rel_path="golden.jsonl",
                                 reason="committed and unmodified")
    world.write_report(payload, resign=True)
    result = world.promote()
    assert result.exit_code == 1, _flat(result)
    assert "what was scored is not what was committed" in _flat(result)
    assert world.status() == "candidate"


# -- F: the incumbent check and the registry write are one step (B-082) -------------------


def test_F_a_promotion_landing_between_check_and_write_is_not_retired(world, monkeypatch):
    """A report that beat the BASE; another adapter is promoted after the check, before the
    write. store.promote used to retire it anyway."""
    world.registered()
    world.eval_report()
    import hearth.training.promotion as promo

    orig = promo.report_problems
    fired = []

    def racing(*a, **k):
        out = orig(*a, **k)
        if not fired:  # the concurrent `adapters promote` of extract-2 lands once, here
            fired.append(True)
            world.adapter("extract-2")
            world.store.promote("extract-2", gate_passed=True)
        return out

    monkeypatch.setattr(promo, "report_problems", racing)
    result = world.promote()
    assert result.exit_code == 1, _flat(result)
    assert "measured against 'the base model'" in _flat(result)
    assert world.status() == "candidate"
    assert world.store.get("extract-2").status == "promoted"


def test_F_incumbent_weights_swapped_between_check_and_write_are_caught(world, monkeypatch):
    """The precondition re-runs every binding check under the lock, weights included."""
    old = world.adapter("extract-0")
    world.store.promote("extract-0", gate_passed=True)
    world.registered()
    payload = world.eval_report()
    assert payload["incumbent_id"] == "extract-0"
    import hearth.training.promotion as promo

    orig = promo.report_problems
    calls = []

    def swapping(*a, **k):
        out = orig(*a, **k)
        if not calls:
            calls.append(True)
            (old / "adapters.safetensors").write_bytes(b"incumbent retrained meanwhile")
        return out

    monkeypatch.setattr(promo, "report_problems", swapping)
    result = world.promote()
    assert result.exit_code == 1, _flat(result)
    assert "the evidence changed while promoting" in _flat(result)
    assert "incumbent 'extract-0' weights changed" in _flat(result)
    assert world.status() == "candidate"
    assert world.store.get("extract-0").status == "promoted"


def test_F_eval_promote_rechecks_the_incumbent_under_the_lock(world, monkeypatch):
    """The measuring path: an adapter promoted while the eval ran is not silently retired."""
    world.registered()
    import hearth.training.eval as ev

    orig = ev.evaluate_gate

    def racing(*a, **k):
        world.adapter("extract-2")
        world.store.promote("extract-2", gate_passed=True)
        return orig(*a, **k)

    monkeypatch.setattr(ev, "evaluate_gate", racing)
    result = world.eval("extract-1", "--prereg", str(world.prereg), "--promote")
    assert result.exit_code == 1, _flat(result)
    assert "measured against 'the base model'" in _flat(result)
    assert world.status() == "candidate"
    assert world.store.get("extract-2").status == "promoted"


def test_F_eval_promote_refuses_an_incumbent_retrained_while_it_was_scored(world, monkeypatch):
    old = world.adapter("extract-0")
    world.store.promote("extract-0", gate_passed=True)
    world.registered()

    class _Swapping(pe._Provider):
        def generate(self, req):
            if req.adapter and req.adapter.endswith("extract-0"):
                (old / "adapters.safetensors").write_bytes(b"incumbent retrained mid-eval")
            return super().generate(req)

    monkeypatch.setattr("hearth.cli.select_provider", lambda settings: _Swapping())
    result = world.eval("extract-1", "--prereg", str(world.prereg), "--promote")
    assert result.exit_code == 1, _flat(result)
    assert "incumbent 'extract-0''s weights changed during the eval" in _flat(result)
    assert world.status() == "candidate"


# -- L5: the incumbent is scored on its own base (B-085) ---------------------------------

OTHER_BASE = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"  # registered in models.yaml


def test_L5_the_incumbent_adapter_is_scored_on_its_own_base(world, monkeypatch):
    world.adapter("extract-0", base=OTHER_BASE)
    world.store.promote("extract-0", gate_passed=True)
    world.registered()
    seen = []

    class _Recording(pe._Provider):
        def generate(self, req):
            seen.append((req.adapter and req.adapter.rsplit("/", 1)[-1], req.model))
            return super().generate(req)

    monkeypatch.setattr("hearth.cli.select_provider", lambda settings: _Recording())
    payload = world.eval_report()
    assert {m for a, m in seen if a == "extract-0"} == {OTHER_BASE}
    assert {m for a, m in seen if a == "extract-1"} == {pe.BASE}
    assert payload["incumbent"]["model_id"] == f"{OTHER_BASE}+extract-0"
    assert payload["incumbent_base_model"] == OTHER_BASE
    result = world.promote()
    assert result.exit_code == 0, _flat(result)
    assert world.store.get("extract-0").status == "retired"


def test_L5_a_report_that_scored_the_incumbent_on_the_candidate_base_is_refused(world):
    world.adapter("extract-0", base=OTHER_BASE)
    world.store.promote("extract-0", gate_passed=True)
    world.registered()
    payload = world.eval_report()
    del payload["signature"]
    payload["incumbent"]["model_id"] = f"{pe.BASE}+extract-0"  # the old, wrong configuration
    world.write_report(payload, resign=True)
    result = world.promote()
    assert result.exit_code == 1 and "incumbent report model_id" in _flat(result)
    assert world.status() == "candidate"


@pytest.mark.parametrize("base", ["auto", "not-a/registered-model"])
def test_L5_an_incumbent_with_no_servable_base_is_not_measured(world, base):
    world.adapter("extract-0", base=base)
    world.store.promote("extract-0", gate_passed=True)
    result = world.eval("extract-1")
    assert result.exit_code == 2, _flat(result)
