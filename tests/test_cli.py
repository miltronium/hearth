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
