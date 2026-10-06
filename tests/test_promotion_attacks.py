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
from pathlib import Path

import pytest
import yaml

import test_promotion_evidence as pe
from hearth.providers.base import GenResult
from hearth.training.eval import as_golden_set
from test_promotion_evidence import World


@pytest.fixture(autouse=True)
def fake_provider(monkeypatch):
    monkeypatch.setattr("hearth.cli.select_provider", lambda settings: pe._Provider())
    pe.allow_test_backends(monkeypatch)


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


# -- L4: only the MLX pool's scores can license a promotion (B-084) -----------------------


def test_L4_the_report_records_the_backend_that_generated_it(world):
    world.registered()
    payload = world.eval_report()
    assert payload["backend"] == "test_promotion_evidence._Provider:fake"


def test_L4_a_plugin_or_stub_backend_cannot_promote(world, monkeypatch):
    """Without the test allowlist, the very same (passing) run is refused on both paths."""
    from hearth.training import promotion

    world.registered()
    world.eval_report()
    monkeypatch.setattr(promotion, "PROMOTABLE_BACKENDS",
                        frozenset({"hearth.serving.pool.ModelPool:mlx"}))
    result = world.promote()
    assert result.exit_code == 1, _flat(result)
    assert "not the MLX model pool" in _flat(result)
    result = world.eval("extract-1", "--prereg", str(world.prereg), "--promote")
    assert result.exit_code == 1, _flat(result)
    assert "not the MLX model pool" in _flat(result)
    assert world.status() == "candidate"


def test_L4_a_provider_that_calls_itself_mlx_is_not_the_pool(monkeypatch):
    from hearth.providers.echo import EchoProvider
    from hearth.training import promotion
    from hearth.training.promotion import backend_identity, backend_problems

    monkeypatch.setattr(promotion, "PROMOTABLE_BACKENDS",
                        frozenset({"hearth.serving.pool.ModelPool:mlx"}))

    class Plugin:
        name = "mlx"

    assert backend_problems(backend_identity(Plugin()))
    assert backend_problems(backend_identity(EchoProvider()))
    assert backend_problems(None)


def test_L4_the_mlx_pool_is_promotable():
    from hearth.providers import mlx_pool
    from hearth.training.promotion import backend_identity, backend_problems

    assert backend_problems(backend_identity(mlx_pool())) == []


# -- A: the prereg must predate the adapter's FIRST measurement (B-079) -------------------


def test_A_a_bar_registered_after_seeing_the_score_is_refused(world):
    """Measure with no prereg (PASS + p printed), then commit the bar and re-measure."""
    result = world.eval("extract-1")
    assert result.exit_code == 0, _flat(result)
    assert "PASS" in _flat(result)
    assert "Exploratory measurement (no --prereg)" in _flat(result)
    world.registered()  # the bar, written after the outcome was seen
    world.eval_report()  # a deterministic re-measurement, "after the prereg"
    result = world.promote()
    assert result.exit_code == 1, _flat(result)
    assert "first measured" in _flat(result) or "first recorded measurement" in _flat(result)
    assert world.status() == "candidate"
    result = world.eval("extract-1", "--prereg", str(world.prereg), "--promote")
    assert result.exit_code == 1, _flat(result)
    assert world.status() == "candidate"


def test_A_a_measurement_on_any_other_golden_set_counts_as_first(world, tmp_path):
    """Peeking on a scratch set outside git still fixes when the adapter was first seen."""
    scratch = tmp_path / "scratch.jsonl"
    scratch.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in pe.ROWS[:35]))
    peek = pe.runner.invoke(pe.app, ["eval", "extract-1", "--golden", str(scratch), "--metric",
                                     "exact", "--max-tokens", "24"], env=world.env)
    assert peek.exit_code == 0, _flat(peek)
    world.registered()
    world.eval_report()
    result = world.promote()
    assert result.exit_code == 1, _flat(result)
    assert world.status() == "candidate"


def test_A_a_backdated_prereg_commit_is_not_in_the_history_seen_at_first_measurement(world):
    """GIT_COMMITTER_DATE can lie about WHEN; the HEAD recorded at measurement cannot."""
    from datetime import UTC, datetime, timedelta

    world.commit("golden.jsonl")
    world.eval("extract-1")  # first measurement: HEAD = the golden-only commit
    world.write_prereg()
    world.commit("prereg.yaml", when=datetime.now(tz=UTC) - timedelta(days=30))  # backdated
    world.eval_report()
    result = world.promote()
    assert result.exit_code == 1, _flat(result)
    assert "was not in the history of" in _flat(result)
    assert world.status() == "candidate"


def test_A_the_same_weights_under_a_fresh_id_are_not_a_fresh_adapter(world):
    """Re-registering measured weights under a new id does not reset the first measurement."""
    world.eval("extract-1")  # peek
    path = world.store.get("extract-1").adapter_path
    world.store.register("extract-1b", base_model=pe.BASE, task="extract", train_run_id="r",
                         adapter_path=path)
    world.registered()
    world.eval_report("extract-1b")  # the provider answers by weights dir: same winner
    result = world.promote("extract-1b")
    assert result.exit_code == 1, _flat(result)
    assert world.status("extract-1b") == "candidate"


def test_A_a_report_whose_measurement_is_not_in_the_ledger_is_refused(world):
    """Deleting the ledger does not make an adapter "never measured": the report is orphaned."""
    from hearth.training.ledger import ledger_path

    world.registered()
    world.eval_report()
    ledger_path(world.home).unlink()
    result = world.promote()
    assert result.exit_code == 1, _flat(result)
    assert "not in the measurement ledger" in _flat(result)
    assert world.status() == "candidate"


def test_A_a_tampered_ledger_refuses_promotion_and_measurement(world):
    from hearth.training.ledger import ledger_path

    world.registered()
    world.eval("extract-1")
    world.eval_report()
    path = ledger_path(world.home)
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[1:]) + "\n")  # drop the first measurement
    result = world.promote()
    assert result.exit_code == 1, _flat(result)
    assert "not intact" in _flat(result)
    result = world.eval("extract-1")
    assert result.exit_code == 1 and "cannot record the measurement" in _flat(result)
    assert world.status() == "candidate"


def test_A_an_unknown_metric_is_refused_before_anything_is_recorded(world):
    from hearth.training.ledger import ledger_path

    result = pe.runner.invoke(pe.app, ["eval", "extract-1", "--golden", str(world.golden),
                                       "--metric", "exactt"], env=world.env)
    assert result.exit_code == 1 and "Unknown metric" in _flat(result)
    assert not ledger_path(world.home).exists()


def test_A_the_legitimate_order_records_the_first_measurement_in_the_proof(world):
    world.registered()
    first = world.eval_report()
    world.eval_report()  # a second, later measurement under the same bar
    result = world.promote()
    assert result.exit_code == 0, _flat(result)
    proof = world.store.get("extract-1").promotion_proof
    assert proof["first_ledger_seq"] == first["ledger_seq"] == 0
    assert proof["ledger_seq"] == 1
    assert proof["first_measured_at"] == first["measured_at"]


# -- C: the evals repository is anchored before the first measurement (B-081) -------------


def test_C_a_throwaway_repo_holding_a_golden_copy_is_refused(world, tmp_path):
    other = tmp_path / "throwaway"
    other.mkdir()
    pe._git(other, "init", "-q")
    world.golden = other / "golden.jsonl"
    world.golden.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in pe.ROWS))
    world.prereg = other / "prereg.yaml"
    world.repo = other
    world.registered()
    world.eval_report()
    result = world.promote()
    assert result.exit_code == 1, _flat(result)
    assert "not the evals repository HEARTH was anchored to" in _flat(result)
    assert world.status() == "candidate"
    result = world.eval("extract-1", "--prereg", str(world.prereg), "--promote")
    assert result.exit_code == 1 and "not the evals repository" in _flat(result)
    assert world.status() == "candidate"


def test_C_re_anchoring_after_the_first_measurement_does_not_help(world, tmp_path):
    from hearth.training.prereg import pin_anchor

    world.eval("extract-1")  # first measured while anchored to world.repo
    other = tmp_path / "throwaway"
    other.mkdir()
    pe._git(other, "init", "-q")
    pin_anchor(world.home, other)  # ...then point HEARTH somewhere convenient
    world.golden = other / "golden.jsonl"
    world.golden.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in pe.ROWS))
    world.prereg = other / "prereg.yaml"
    world.repo = other
    world.write_prereg()  # backdated, so only the anchor rule stands between it and a promotion
    from datetime import UTC, datetime, timedelta

    world.commit("golden.jsonl", "prereg.yaml", when=datetime.now(tz=UTC) - timedelta(days=1))
    world.eval_report()
    result = world.promote()
    assert result.exit_code == 1, _flat(result)
    assert "not the evals repository" in _flat(result)
    assert world.status() == "candidate"


def test_C_a_worktree_of_the_anchored_repo_is_the_same_repository(world, tmp_path):
    """Identity is the common git dir: a worktree is the anchored repo, a copy is not."""
    world.registered()
    tree = tmp_path / "tree"
    pe._git(world.repo, "worktree", "add", "-q", str(tree))
    world.golden = tree / "golden.jsonl"
    world.prereg = tree / "prereg.yaml"
    world.eval_report()
    result = world.promote()
    assert result.exit_code == 0, _flat(result)


def test_C_prereg_anchor_shows_sets_and_refuses_a_non_repository(world, tmp_path):
    show = pe.runner.invoke(pe.app, ["prereg", "anchor"], env=world.env)
    assert show.exit_code == 0 and str(world.repo.resolve()) in _flat(show)
    plain = tmp_path / "plain"
    plain.mkdir()
    bad = pe.runner.invoke(pe.app, ["prereg", "anchor", str(plain)], env=world.env)
    assert bad.exit_code == 1 and "not inside a git repository" in _flat(bad)
    other = tmp_path / "other"
    other.mkdir()
    pe._git(other, "init", "-q")
    ok = pe.runner.invoke(pe.app, ["prereg", "anchor", str(other)], env=world.env)
    assert ok.exit_code == 0, _flat(ok)
    show = pe.runner.invoke(pe.app, ["prereg", "anchor"], env=world.env)
    assert str(other.resolve()) in _flat(show)
    (world.home / "evals-repo").write_text(str(plain))
    show = pe.runner.invoke(pe.app, ["prereg", "anchor"], env=world.env)
    assert show.exit_code == 1 and "No evals repository" in _flat(show)


def test_C_unset_the_anchor_is_hearths_own_repository(tmp_path):
    import hearth
    from hearth.training.prereg import resolve_anchor

    anchor = resolve_anchor(tmp_path / "empty-home")
    assert "default" in anchor["source"]
    assert anchor["common_dir"]  # this test runs from a checkout of HEARTH
    assert Path(hearth.__file__).resolve().is_relative_to(Path(anchor["path"]).resolve())


# -- B-086: guards whose deletion no test noticed (reviewer mutation run) -----------------


@pytest.mark.parametrize(
    ("flags", "text"),
    [
        (("--max-tokens", "32"), "decode config"),
        (("--metric", "f1"), "metric 'token_f1' != registered 'exact_match'"),
    ],
)
def test_gap_a_report_that_is_not_the_registered_experiment_is_refused(world, flags, text):
    """`adapters promote` must compare the report with the prereg (registration.mismatches)."""
    world.registered()
    world.eval_report("extract-1", *flags, exploratory=True)  # after the bar, not under it
    result = world.promote()
    assert result.exit_code == 1, _flat(result)
    assert "not the registered experiment" in _flat(result) and text in _flat(result)
    assert world.status() == "candidate"


@pytest.mark.parametrize(
    ("flags", "text"),
    [
        (("--max-tokens", "32"), "decode config"),
        (("--metric", "f1"), "metric 'token_f1' != registered 'exact_match'"),
    ],
)
def test_B122_eval_refuses_a_run_that_is_not_the_registered_experiment_before_recording(
    world, flags, text
):
    """A ledger record says which prereg a run was made under, so --prereg with a different
    golden set / metric / decode config is refused BEFORE anything is measured or recorded."""
    from hearth.training.ledger import ledger_path

    world.registered()
    result = world.eval("extract-1", *flags, "--prereg", str(world.prereg))
    assert result.exit_code == 1, _flat(result)
    assert "nothing was measured" in _flat(result) and text in _flat(result)
    assert "PASS" not in _flat(result) and "FAIL" not in _flat(result)
    assert not ledger_path(world.home).exists()


def test_B122_a_prereg_for_another_task_is_not_this_experiment(world):
    """Same golden content (the sha does not include the task), registered for another task."""
    from hearth.training.ledger import ledger_path

    world.write_prereg(task="classify")
    world.commit("golden.jsonl", "prereg.yaml")
    result = world.eval("extract-1", "--prereg", str(world.prereg))
    assert result.exit_code == 1, _flat(result)
    assert "task 'extract' != registered 'classify'" in _flat(result)
    assert not ledger_path(world.home).exists()


def test_gap_the_proof_names_the_commit_that_last_changed_the_bar(world):
    """Introduced at one commit, tightened at a later one (both before measuring): the bar in
    force is the later commit, and that is the commit the proof and the checks use."""
    world.write_prereg()
    introduced = world.commit("golden.jsonl", "prereg.yaml")
    world.prereg.write_text(world.prereg.read_text().replace("min_n: 30", "min_n: 35"))
    last = world.commit("prereg.yaml")
    world.eval_report()
    result = world.promote()
    assert result.exit_code == 0, _flat(result)
    proof = world.store.get("extract-1").promotion_proof
    assert proof["prereg_commit"] == last != introduced
    assert proof["prereg_introduced_commit"] == introduced
    assert proof["min_n"] == 35


def test_gap_a_bar_tightened_after_the_first_measurement_is_refused_even_in_the_same_second(world):
    world.registered()
    world.eval_report()
    world.prereg.write_text(world.prereg.read_text().replace("min_n: 30", "min_n: 35"))
    world.commit("prereg.yaml")  # no backdating needed: same-second commits are common
    world.eval_report()
    result = world.promote()
    assert result.exit_code == 1, _flat(result)
    assert world.status() == "candidate"


def test_gap_a_candidate_block_whose_measured_at_disagrees_is_refused(world):
    """The ledger binds the report's top-level measured_at; report_problems binds the
    candidate block to it. Edit only the block (key-holder re-sign): still refused."""
    world.registered()
    payload = world.eval_report()
    del payload["signature"]
    payload["candidate"]["measured_at"] = "2020-01-01T00:00:00+00:00"
    world.write_report(payload, resign=True)
    result = world.promote()
    assert result.exit_code == 1 and "measured_at" in _flat(result)
    assert world.status() == "candidate"
