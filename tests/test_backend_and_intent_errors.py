"""B-059 (unknown HEARTH_BACKEND) and B-060 (unknown intent) are errors, not tracebacks or
silent fallbacks — asserted on what the operator / client actually sees."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from hearth.cli import app
from hearth.config import Settings, get_settings
from hearth.gateway import create_app
from hearth.providers.echo import EchoProvider
from hearth.router.classify import UnknownIntentError, classify

runner = CliRunner()


@pytest.fixture(autouse=True)
def _fresh_settings():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.parametrize(
    "argv", [["run", "hi"], ["agent", "hi"], ["rag", "query", "x", "--answer"]]
)
def test_an_unknown_backend_is_a_clean_exit_2(argv, monkeypatch, tmp_path):
    monkeypatch.setenv("HEARTH_BACKEND", "bogus")
    monkeypatch.setenv("HEARTH_FILE_ROOTS", str(tmp_path))
    result = runner.invoke(app, argv)
    assert result.exit_code == 2, result.output
    assert "Unknown HEARTH_BACKEND: 'bogus'" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_doctor_fails_an_unknown_backend(monkeypatch):
    monkeypatch.setenv("HEARTH_BACKEND", "bogus")
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "Unknown HEARTH_BACKEND" in result.output


def test_classify_refuses_an_unknown_intent_and_accepts_any_case():
    with pytest.raises(UnknownIntentError):
        classify([], intent="bogus")
    assert classify([], intent="Summarize") == ("summarize", "intent")
    assert classify([], intent="  ")[1] == "rules"  # blank = no hint


def test_run_refuses_an_unknown_intent(monkeypatch):
    monkeypatch.setenv("HEARTH_BACKEND", "echo")
    result = runner.invoke(app, ["run", "--intent", "bogus", "hi"])
    assert result.exit_code == 2
    assert "unknown intent 'bogus'" in result.output


@pytest.mark.parametrize(
    "path,body",
    [
        ("/v1/hearth/route", {"messages": [{"role": "user", "content": "hi"}], "intent": "bogus"}),
        ("/v1/chat/completions",
         {"messages": [{"role": "user", "content": "hi"}], "hearth": {"intent": "bogus"}}),
    ],
)
def test_the_api_rejects_an_unknown_intent(tmp_path, path, body):
    client = TestClient(create_app(
        provider=EchoProvider(),
        settings=Settings(backend="echo", require_auth=False, home=tmp_path / "h"),
    ))
    r = client.post(path, json=body)
    assert r.status_code == 422
    assert "unknown intent" in r.text
