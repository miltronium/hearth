"""CLI tests for `hearth eval` / `hearth prereg` — offline (no MLX, no model download).

The echo backend ignores the per-request adapter slot, so base and candidate always score
*identically* through it. That is not a limitation of these tests, it is the new gate being
honest: with the echo backend there is no lift, so nothing is promotable. The one promotion
path exercised here injects a fake provider that really does answer differently with the
adapter attached, plus a git-committed pre-registration.

Each run uses an isolated HEARTH_HOME and forces HEARTH_BACKEND=echo.
"""

from __future__ import annotations

import json
import subprocess

import pytest
import yaml
from typer.testing import CliRunner

from hearth.cli import app
from hearth.providers.base import GenResult
from hearth.training.eval import EvalConfig, as_golden_set

runner = CliRunner()

# 40 examples: half expect "A" (so the majority-label baseline scores 0.5), half expect a
# per-item label the base model cannot guess. Large enough to clear the gate's min_n.
ROWS = [
    {"prompt": f"p{i}", "expected": "A" if i % 2 == 0 else f"B{i}"} for i in range(40)
]
CONFIG = EvalConfig(temperature=0.0, max_tokens=24)


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


def _golden_sha(rows, task="extract") -> str:
    return as_golden_set(task, [(r["prompt"], r["expected"]) for r in rows]).sha


def _git(tmp_path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=test", *args],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )


def _committed_prereg(tmp_path, rows, *, task="extract", name="prereg.yaml", **bar) -> str:
    """Write a matching pre-registration and commit it to a throwaway repo."""
    _git(tmp_path, "init", "-q")
    body = {
        "task": task,
        "hypothesis": "the adapter learns the labels the base model cannot guess",
        "golden_sha": _golden_sha(rows, task),
        "golden_version": "",
        "n": len(rows),
        "metric": "exact",
        "generation": {"temperature": 0.0, "max_tokens": 24, "seed": None, "system_hash": ""},
        "bar": {"test": "auto", "alpha": 0.05, "min_effect": 0.0, "min_n": 30, **bar},
        "stopping_rule": "one run, no re-rolls",
    }
    path = tmp_path / name
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    _git(tmp_path, "add", name)
    _git(tmp_path, "commit", "-qm", "prereg")
    return str(path)


class _FakeProvider:
    """Base weights parrot "A"; the adapter (A/B slot) answers every item correctly."""

    name = "fake"

    def __init__(self, answers: dict[str, str]) -> None:
        self._answers = answers

    def generate(self, req):
        prompt = req.messages[-1].content
        text = self._answers[prompt] if req.adapter else "A"
        return GenResult(text=text, model=req.model, backend=self.name)


@pytest.fixture
def real_lift(monkeypatch):
    """Install a provider where the adapter genuinely beats the base model."""
    answers = {r["prompt"]: r["expected"] for r in ROWS}
    monkeypatch.setattr("hearth.cli.select_provider", lambda settings: _FakeProvider(answers))


def test_help_lists_eval_and_prereg():
    result = runner.invoke(app, ["--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "eval" in result.stdout
    assert "prereg" in result.stdout


def test_eval_refuses_a_tiny_golden_set(tmp_path):
    """One example cannot license a promotion, however good the score looks."""
    _seed_adapter(tmp_path)
    golden = _golden(tmp_path, [{"prompt": "foo", "expected": "[echo] foo"}])
    result = runner.invoke(
        app, ["eval", "extract-1", "--golden", golden, "--metric", "exact"], env=_env(tmp_path)
    )
    assert result.exit_code == 0
    assert "FAIL" in result.stdout
    assert "min_n" in result.stdout


def test_eval_scores_the_base_model_as_the_incumbent(tmp_path):
    """F2: with nothing promoted the base model is the incumbent, not a free pass."""
    _seed_adapter(tmp_path)
    golden = _golden(tmp_path, [{"prompt": f"p{i}", "expected": f"[echo] p{i}"} for i in range(40)])
    result = runner.invoke(
        app, ["eval", "extract-1", "--golden", golden, "--metric", "exact"], env=_env(tmp_path)
    )
    assert result.exit_code == 0
    assert "base" in result.stdout  # the incumbent row is the base model
    # Echo ignores the adapter, so candidate == base: a perfect 1.0 that is still no lift.
    assert "FAIL" in result.stdout
    assert "no lift" in result.stdout


def test_eval_refuses_to_score_at_a_sampling_temperature(tmp_path):
    _seed_adapter(tmp_path)
    golden = _golden(tmp_path, ROWS)
    result = runner.invoke(
        app,
        ["eval", "extract-1", "--golden", golden, "--metric", "exact", "--temperature", "0.7"],
        env=_env(tmp_path),
    )
    assert result.exit_code == 1
    assert "re-rollable" in result.stdout


def test_eval_promote_requires_a_prereg(tmp_path, real_lift):
    _seed_adapter(tmp_path)
    golden = _golden(tmp_path, ROWS)
    result = runner.invoke(
        app,
        ["eval", "extract-1", "--golden", golden, "--metric", "exact", "--max-tokens", "24",
         "--promote"],
        env=_env(tmp_path),
    )
    assert result.exit_code == 1
    assert "requires --prereg" in result.stdout
    from hearth.registry import AdapterStore

    store = AdapterStore(path=tmp_path / ".hearth" / "adapters.json")
    assert store.get("extract-1").status == "candidate"  # unchanged


def test_eval_promote_refused_when_the_prereg_is_not_committed(tmp_path, real_lift):
    _seed_adapter(tmp_path)
    golden = _golden(tmp_path, ROWS)
    path = _committed_prereg(tmp_path, ROWS)
    (tmp_path / "prereg.yaml").write_text(
        (tmp_path / "prereg.yaml").read_text().replace("min_n: 30", "min_n: 1")
    )
    result = runner.invoke(
        app,
        ["eval", "extract-1", "--golden", golden, "--metric", "exact", "--max-tokens", "24",
         "--prereg", path, "--promote"],
        env=_env(tmp_path),
    )
    assert result.exit_code == 1
    assert "uncommitted modifications" in result.stdout


def test_eval_promote_refused_when_the_run_is_not_the_registered_one(tmp_path, real_lift):
    """A different golden set than the one registered is not the experiment."""
    _seed_adapter(tmp_path)
    golden = _golden(tmp_path, ROWS[:38])  # the registered sha pins all 40
    path = _committed_prereg(tmp_path, ROWS)
    result = runner.invoke(
        app,
        ["eval", "extract-1", "--golden", golden, "--metric", "exact", "--max-tokens", "24",
         "--prereg", path, "--promote"],
        env=_env(tmp_path),
    )
    assert result.exit_code == 1
    assert "not the registered experiment" in result.stdout


def test_eval_promotes_a_real_significant_lift_under_a_committed_prereg(tmp_path, real_lift):
    _seed_adapter(tmp_path)
    golden = _golden(tmp_path, ROWS)
    path = _committed_prereg(tmp_path, ROWS)
    result = runner.invoke(
        app,
        ["eval", "extract-1", "--golden", golden, "--metric", "exact", "--max-tokens", "24",
         "--prereg", path, "--promote"],
        env=_env(tmp_path),
    )
    assert result.exit_code == 0, result.stdout
    assert "PASS" in result.stdout

    from hearth.registry import AdapterStore

    entry = AdapterStore(path=tmp_path / ".hearth" / "adapters.json").get("extract-1")
    assert entry.status == "promoted"
    proof = entry.promotion_proof
    assert proof["gate"] == "verified"
    assert proof["test"] == "mcnemar_exact"
    assert proof["b"] == 20 and proof["c"] == 0 and proof["n"] == 40
    assert proof["p_value"] < 0.05
    assert proof["incumbent_role"] == "base"
    assert proof["prereg_committed"] is True
    assert len(proof["prereg_sha"]) == 64
    assert proof["golden_sha"] == _golden_sha(ROWS)
    assert proof["baselines"]["majority_label"] == 0.5


def test_eval_report_json_carries_the_vectors_and_provenance(tmp_path, real_lift):
    _seed_adapter(tmp_path)
    golden = _golden(tmp_path, ROWS)
    out = tmp_path / "reports" / "extract-1.json"
    result = runner.invoke(
        app,
        ["eval", "extract-1", "--golden", golden, "--metric", "exact", "--max-tokens", "24",
         "--report-json", str(out)],
        env=_env(tmp_path),
    )
    assert result.exit_code == 0
    payload = json.loads(out.read_text())
    assert payload["candidate"]["golden_sha"] == _golden_sha(ROWS)
    assert payload["candidate"]["config_fingerprint"] == CONFIG.fingerprint
    assert len(payload["candidate"]["per_example"]) == 40
    assert payload["incumbent_role"] == "base"
    assert set(payload["baselines"]) == {"empty", "majority_label", "copy_input"}
    assert payload["gate"]["gate_passed"] is True


def test_adapters_promote_consumes_a_report_and_recomputes_the_gate(tmp_path, real_lift):
    """The offline path: `eval --report-json` then `adapters promote --report --prereg`."""
    _seed_adapter(tmp_path)
    golden = _golden(tmp_path, ROWS)
    path = _committed_prereg(tmp_path, ROWS)
    out = tmp_path / "report.json"
    runner.invoke(
        app,
        ["eval", "extract-1", "--golden", golden, "--metric", "exact", "--max-tokens", "24",
         "--report-json", str(out)],
        env=_env(tmp_path),
    )
    result = runner.invoke(
        app,
        ["adapters", "promote", "extract-1", "--report", str(out), "--prereg", path],
        env=_env(tmp_path),
    )
    assert result.exit_code == 0, result.stdout
    from hearth.registry import AdapterStore

    entry = AdapterStore(path=tmp_path / ".hearth" / "adapters.json").get("extract-1")
    assert entry.status == "promoted"
    assert entry.promotion_proof["gate"] == "verified"


def test_adapters_promote_refuses_a_report_that_fails_the_gate(tmp_path):
    """Echo backend: candidate == base, so the persisted report has no lift to promote."""
    _seed_adapter(tmp_path)
    golden = _golden(tmp_path, [{"prompt": f"p{i}", "expected": f"[echo] p{i}"} for i in range(40)])
    path = _committed_prereg(
        tmp_path, [{"prompt": f"p{i}", "expected": f"[echo] p{i}"} for i in range(40)]
    )
    out = tmp_path / "report.json"
    runner.invoke(
        app,
        ["eval", "extract-1", "--golden", golden, "--metric", "exact", "--max-tokens", "24",
         "--report-json", str(out)],
        env=_env(tmp_path),
    )
    result = runner.invoke(
        app,
        ["adapters", "promote", "extract-1", "--report", str(out), "--prereg", path],
        env=_env(tmp_path),
    )
    assert result.exit_code == 1
    assert "refused" in result.stdout.lower()


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


# -- hearth prereg ---------------------------------------------------------------------


def test_prereg_init_pins_the_golden_set_by_content_sha(tmp_path):
    golden = _golden(tmp_path, ROWS)
    out = tmp_path / "evals" / "prereg.yaml"
    result = runner.invoke(
        app,
        ["prereg", "init", "--task", "extract", "--golden", golden, "--out", str(out),
         "--metric", "exact", "--max-tokens", "24"],
        env=_env(tmp_path),
    )
    assert result.exit_code == 0
    body = yaml.safe_load(out.read_text())
    assert body["golden_sha"] == _golden_sha(ROWS)
    assert body["generation"]["temperature"] == 0.0
    assert body["hypothesis"] == ""  # left for the operator


def test_prereg_check_reports_an_uncommitted_bar(tmp_path):
    golden = _golden(tmp_path, ROWS)
    out = tmp_path / "prereg.yaml"
    runner.invoke(
        app,
        ["prereg", "init", "--task", "extract", "--golden", golden, "--out", str(out),
         "--metric", "exact", "--max-tokens", "24"],
        env=_env(tmp_path),
    )
    result = runner.invoke(
        app, ["prereg", "check", str(out), "--golden", golden], env=_env(tmp_path)
    )
    assert result.exit_code == 1
    assert "not committed" in result.stdout


def test_prereg_check_detects_a_golden_set_edited_after_registration(tmp_path):
    golden = _golden(tmp_path, ROWS)
    path = _committed_prereg(tmp_path, ROWS)
    _golden(tmp_path, [*ROWS, {"prompt": "p40", "expected": "A"}])  # set grew
    result = runner.invoke(
        app, ["prereg", "check", path, "--golden", golden], env=_env(tmp_path)
    )
    assert result.exit_code == 1
    assert "Golden set has changed" in result.stdout
