"""B-029: an unregistered ``HEARTH_DEFAULT_MODEL`` is ignored — and that is now reported.

The fallback itself is unchanged (a typo must not take the catalog default away); what
changes is that `hearth doctor` WARNs, and the registry logs once, naming the ignored id
and the model that will actually serve.
"""

from __future__ import annotations

import logging

from typer.testing import CliRunner

from hearth.cli import app
from hearth.registry import load_registry

CATALOG_DEFAULT = "org/catalog-default"
OTHER = "org/other-registered"
UNREGISTERED = "org/not-in-the-catalog"

_MODELS_YAML = f"""
default: {CATALOG_DEFAULT}
models:
  - {{id: {CATALOG_DEFAULT}, backend: echo}}
  - {{id: {OTHER}, backend: echo}}
"""


def _write_models(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(_MODELS_YAML)
    return path


def _env(tmp_path, override: str) -> dict[str, str]:
    return {
        "COLUMNS": "400",
        "HEARTH_HOME": str(tmp_path / "home"),
        "HEARTH_MODELS_YAML": str(_write_models(tmp_path)),
        "HEARTH_DEFAULT_MODEL": override,
    }


def test_hearth_doctor_warns_naming_the_ignored_id_and_the_serving_model(tmp_path):
    result = CliRunner().invoke(app, ["doctor"], env=_env(tmp_path, UNREGISTERED))
    line = next((ln for ln in result.output.splitlines() if "default_model" in ln), "")
    assert "WARN" in line, result.output
    assert UNREGISTERED in line and CATALOG_DEFAULT in line, result.output


def test_hearth_doctor_is_quiet_when_the_override_is_registered(tmp_path):
    result = CliRunner().invoke(app, ["doctor"], env=_env(tmp_path, OTHER))
    assert "default_model" not in result.output


def test_the_registry_logs_the_ignored_override_once_and_keeps_the_fallback(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setenv("HEARTH_DEFAULT_MODEL", UNREGISTERED)
    registry = load_registry(_write_models(tmp_path))
    with caplog.at_level(logging.WARNING, logger="hearth.registry"):
        assert registry.default_id == CATALOG_DEFAULT  # fallback semantics unchanged
        assert registry.default_id == CATALOG_DEFAULT
    warnings = [r.getMessage() for r in caplog.records if r.name == "hearth.registry"]
    assert len(warnings) == 1, warnings
    assert UNREGISTERED in warnings[0] and CATALOG_DEFAULT in warnings[0]
    assert registry.ignored_default_override == UNREGISTERED
