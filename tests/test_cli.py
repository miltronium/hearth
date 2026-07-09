"""CLI tests — the `hearth models list` subcommand renders the registry."""

from __future__ import annotations

from typer.testing import CliRunner

from hearth.cli import app
from hearth.registry import get_registry

runner = CliRunner()


def test_models_list():
    # Force a wide terminal so Rich doesn't truncate the long model ids.
    result = runner.invoke(app, ["models", "list"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    default_id = get_registry().default_id
    assert "echo" in result.stdout
    assert default_id.split("/")[-1] in result.stdout
    assert "(default)" in result.stdout


def test_help_lists_mcp_command():
    result = runner.invoke(app, ["--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "mcp" in result.stdout


def test_mcp_without_extra_prints_hint_and_exits_nonzero(monkeypatch):
    """`hearth mcp` with the `mcp` SDK absent must fail loudly, not launch a server.

    Force the import to fail regardless of whether the extra happens to be installed, so
    this asserts the graceful-degradation path deterministically.
    """
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "hearth.mcp.server" or name.startswith("mcp"):
            raise ModuleNotFoundError("No module named 'mcp'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = runner.invoke(app, ["mcp"], env={"COLUMNS": "200"})
    assert result.exit_code == 1
    assert "uv sync --extra mcp" in result.stdout
