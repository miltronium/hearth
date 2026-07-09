"""CLI tests for Phase 7 commands — `hearth plugins list` and `hearth models convert`."""

from __future__ import annotations

from typer.testing import CliRunner

from hearth import plugins
from hearth.cli import app

runner = CliRunner()


class _FakeEntryPoint:
    def __init__(self, name: str, value: str, factory) -> None:
        self.name = name
        self.value = value
        self._factory = factory

    def load(self):
        return self._factory


class _PluginProvider:
    name = "pluginbackend"

    def capabilities(self):
        from hearth.providers.base import Capabilities

        return Capabilities(chat=True, stream=True)

    def generate(self, req):
        from hearth.providers.base import GenResult

        return GenResult(text="ok", model=req.model, backend=self.name)

    def stream(self, req):
        yield "ok"

    def footprint(self, model_id):
        from hearth.providers.base import ResourceEstimate

        return ResourceEstimate(ram_gb=1.0)


def test_plugins_list_empty(monkeypatch):
    monkeypatch.setattr(plugins, "_iter_entry_points", lambda group: [])
    result = runner.invoke(app, ["plugins", "list"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "No plugins installed" in result.stdout


def test_plugins_list_shows_ok_plugin(monkeypatch):
    def fake(group):
        if group == plugins.PROVIDER_GROUP:
            return [_FakeEntryPoint("hello", "hearth_hello:build", _PluginProvider)]
        return []

    monkeypatch.setattr(plugins, "_iter_entry_points", fake)
    result = runner.invoke(app, ["plugins", "list"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "hello" in result.stdout
    assert "hearth.providers" in result.stdout
    assert "ok" in result.stdout


def test_models_convert_rejects_bad_bits():
    result = runner.invoke(
        app,
        ["models", "convert", "--source", "x", "--out", "/tmp/hearth-conv", "--q-bits", "5"],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 1
    assert "Invalid conversion config" in result.stdout
