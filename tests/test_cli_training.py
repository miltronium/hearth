"""CLI tests for Phase 4 — `hearth train` and `hearth adapters list|promote|retire`.

All use an isolated HEARTH_HOME so the real ~/.hearth is never touched, and a fake
training run (via --out + a dataset) so no MLX/model download happens.
"""

from __future__ import annotations

from typer.testing import CliRunner

from hearth.cli import app

runner = CliRunner()


def _env(tmp_path) -> dict[str, str]:
    return {"COLUMNS": "200", "HEARTH_HOME": str(tmp_path / ".hearth")}


def _seed_adapter(tmp_path, adapter_id="extract-1", task="extract", promote=False):
    """Write an adapters.json directly under the isolated home."""
    from hearth.registry import AdapterStore

    store = AdapterStore(path=tmp_path / ".hearth" / "adapters.json")
    store.register(
        adapter_id, base_model="org/base", task=task, train_run_id="r", adapter_path="/a/x"
    )
    if promote:
        store.promote(adapter_id, gate_passed=True)


def test_help_lists_train_and_adapters():
    result = runner.invoke(app, ["--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "train" in result.stdout
    assert "adapters" in result.stdout


def test_train_reports_dataset_error_cleanly(tmp_path):
    # A malformed dataset must fail with a clean message and exit 1 — this exercises the
    # CLI wiring up to (but never launching) a real training run, so no MLX/network.
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"prompt": "only-a-prompt-no-completion"}\n')
    result = runner.invoke(
        app,
        ["train", "--task", "extract", "--base", "org/base", "--data", str(bad),
         "--out", str(tmp_path / "run")],
        env=_env(tmp_path),
    )
    assert result.exit_code == 1
    assert "Dataset error" in result.stdout


def test_adapters_list_renders(tmp_path):
    _seed_adapter(tmp_path)
    result = runner.invoke(app, ["adapters", "list"], env=_env(tmp_path))
    assert result.exit_code == 0
    assert "extract-1" in result.stdout
    assert "candidate" in result.stdout


def test_adapters_promote_refused_without_gate(tmp_path):
    # candidate=0.5 does NOT beat incumbent=0.9 → gate fails → promote refused.
    _seed_adapter(tmp_path)
    result = runner.invoke(
        app,
        ["adapters", "promote", "extract-1", "--candidate-score", "0.5",
         "--incumbent-score", "0.9"],
        env=_env(tmp_path),
    )
    assert result.exit_code == 1
    assert "refused" in result.stdout.lower()


def test_adapters_promote_succeeds_when_gate_passes(tmp_path):
    _seed_adapter(tmp_path)
    result = runner.invoke(
        app,
        ["adapters", "promote", "extract-1", "--candidate-score", "0.95",
         "--incumbent-score", "0.80"],
        env=_env(tmp_path),
    )
    assert result.exit_code == 0
    assert "promoted" in result.stdout.lower()


def test_adapters_promote_requires_candidate_score(tmp_path):
    _seed_adapter(tmp_path)
    result = runner.invoke(app, ["adapters", "promote", "extract-1"], env=_env(tmp_path))
    assert result.exit_code == 1
    assert "candidate-score" in result.stdout


def test_adapters_retire(tmp_path):
    _seed_adapter(tmp_path)
    result = runner.invoke(app, ["adapters", "retire", "extract-1"], env=_env(tmp_path))
    assert result.exit_code == 0
    assert "retired" in result.stdout.lower()
