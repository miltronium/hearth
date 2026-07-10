"""CLI tests for `hearth eval` — offline via the echo backend (no MLX / no model).

The echo backend returns ``[echo] <prompt>`` deterministically, so a golden set whose
``expected`` matches that lets us drive the eval gate PASS/FAIL paths without a real model.
Each run uses an isolated HEARTH_HOME and forces HEARTH_BACKEND=echo.
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from hearth.cli import app

runner = CliRunner()


def _env(tmp_path) -> dict[str, str]:
    return {
        "COLUMNS": "200",
        "HEARTH_HOME": str(tmp_path / ".hearth"),
        "HEARTH_BACKEND": "echo",
    }


def _seed_adapter(tmp_path, adapter_id="extract-1", task="extract", promote=False):
    from hearth.registry import AdapterStore

    store = AdapterStore(path=tmp_path / ".hearth" / "adapters.json")
    store.register(
        adapter_id, base_model="org/base", task=task, train_run_id="r", adapter_path="/a/x"
    )
    if promote:
        store.promote(adapter_id, gate_passed=True)


def _golden(tmp_path, rows) -> str:
    path = tmp_path / "golden.jsonl"
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    return str(path)


def test_help_lists_eval():
    result = runner.invoke(app, ["--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "eval" in result.stdout


def test_eval_scores_and_reports_gate(tmp_path):
    _seed_adapter(tmp_path)
    # echo returns "[echo] foo" for prompt "foo" -> exact match with expected below.
    golden = _golden(tmp_path, [{"prompt": "foo", "expected": "[echo] foo"}])
    result = runner.invoke(
        app, ["eval", "extract-1", "--golden", golden, "--metric", "exact"], env=_env(tmp_path)
    )
    assert result.exit_code == 0
    assert "candidate" in result.stdout
    assert "PASS" in result.stdout


def test_eval_promotes_when_it_beats_incumbent(tmp_path):
    _seed_adapter(tmp_path)
    golden = _golden(tmp_path, [{"prompt": "foo", "expected": "[echo] foo"}])
    result = runner.invoke(
        app,
        ["eval", "extract-1", "--golden", golden, "--metric", "exact", "--promote"],
        env=_env(tmp_path),
    )
    assert result.exit_code == 0
    assert "promoted" in result.stdout.lower()
    # Registry reflects the promotion.
    from hearth.registry import AdapterStore

    store = AdapterStore(path=tmp_path / ".hearth" / "adapters.json")
    assert store.get("extract-1").status == "promoted"


def test_eval_promote_refused_on_regression(tmp_path):
    _seed_adapter(tmp_path)
    # expected never matches the echo output -> score 0.0 -> gate FAIL.
    golden = _golden(tmp_path, [{"prompt": "foo", "expected": "totally-different"}])
    result = runner.invoke(
        app,
        ["eval", "extract-1", "--golden", golden, "--metric", "exact", "--promote"],
        env=_env(tmp_path),
    )
    assert result.exit_code == 1
    assert "refused" in result.stdout.lower()
    from hearth.registry import AdapterStore

    store = AdapterStore(path=tmp_path / ".hearth" / "adapters.json")
    assert store.get("extract-1").status == "candidate"  # unchanged


def test_eval_unknown_adapter(tmp_path):
    golden = _golden(tmp_path, [{"prompt": "foo", "expected": "bar"}])
    result = runner.invoke(
        app, ["eval", "nope-1", "--golden", golden], env=_env(tmp_path)
    )
    assert result.exit_code == 1
    assert "unknown adapter" in result.stdout.lower()


def test_eval_bad_golden_row(tmp_path):
    _seed_adapter(tmp_path)
    golden = _golden(tmp_path, [{"prompt": "only-a-prompt"}])  # missing "expected"
    result = runner.invoke(
        app, ["eval", "extract-1", "--golden", golden], env=_env(tmp_path)
    )
    assert result.exit_code == 1
    assert "golden set error" in result.stdout.lower()
