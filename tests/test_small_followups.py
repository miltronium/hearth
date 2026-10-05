"""B-075 (finance profile pins embed), B-076 (CLI label for an unusable profile),
B-077 (status probe shows the real routing error) — asserted on outcomes."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from hearth.cli import app
from hearth.config import get_settings
from hearth.registry import get_registry
from hearth.router.policy import load_policy
from hearth.router.route import policy_rungs
from hearth.status.probes import _policy_outcome

REPO = Path(__file__).resolve().parent.parent
SEVEN_B = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
BAD_RUNG = (
    "classes:\n  chat: { backend: local, escalate: never, "
    "local_model: mlx-community/bge-small-en-v1.5-bf16 }\n"
)


def test_the_finance_profile_never_routes_auto_to_the_registry_default():
    policy = load_policy(REPO / "config" / "routing.finance.yaml")
    rungs = policy_rungs(policy, get_registry().default_id)
    assert SEVEN_B not in rungs, rungs  # warmup/readiness no longer load a 7B nobody serves


def test_an_unusable_rung_is_labelled_unusable_not_missing(tmp_path, monkeypatch):
    profile = tmp_path / "bad.yaml"
    profile.write_text(BAD_RUNG)
    monkeypatch.setenv("HEARTH_ROUTING_YAML", str(profile))
    monkeypatch.setenv("HEARTH_BACKEND", "echo")
    get_settings.cache_clear()
    from hearth.router.policy import get_policy

    get_policy.cache_clear()
    try:
        result = CliRunner().invoke(app, ["run", "hi"])
    finally:
        get_policy.cache_clear()
        get_settings.cache_clear()
    assert result.exit_code == 2, result.output
    assert "Routing profile unusable" in result.output
    assert "not found" not in result.output


def test_the_status_probe_reports_the_routing_error_itself(tmp_path):
    profile = tmp_path / "bad.yaml"
    profile.write_text(BAD_RUNG)
    policy, meta = _policy_outcome(profile)
    assert policy is None
    assert "policy loader unavailable" not in meta["error"]
    assert "bge-small" in meta["error"]  # the operator sees which rung is wrong
