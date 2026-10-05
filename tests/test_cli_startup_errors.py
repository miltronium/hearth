"""Startup errors from the CLI end in one actionable line and a deliberate exit code.

B-033: a ``HEARTH_ROUTING_YAML`` that names a missing profile made every command that builds
the router (``serve``, ``run``, ``agent``, ``mcp``, ``rag query``) print a full traceback
before the message that says how to fix it. The error is caught where the router is built,
so these tests trip the router's real check (no stub of the policy loader).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hearth.cli import app
from hearth.config import get_settings
from hearth.router import policy as policy_mod

_SRC = Path(__file__).resolve().parent.parent / "src"
MISSING_PROFILE = "config/hearth-test-no-such-profile.yaml"


@pytest.fixture(autouse=True)
def fresh_policy_cache():
    """``get_policy`` and ``get_settings`` are lru_cached per process: an earlier test's
    policy/settings would mask this test's env. Clear both on both sides."""
    policy_mod.get_policy.cache_clear()
    get_settings.cache_clear()
    yield
    policy_mod.get_policy.cache_clear()
    get_settings.cache_clear()


@pytest.fixture
def no_server(monkeypatch):
    """Never bind a port: record a `uvicorn.run` call instead of serving.

    Also restores the ``hearth`` logger: `serve` attaches a stderr handler, and under
    CliRunner that stream is closed after the invocation — a handler left behind makes a
    later test's background log line print "--- Logging error ---".
    """
    import logging

    import uvicorn

    log = logging.getLogger("hearth")
    handlers, level = list(log.handlers), log.level
    calls: list = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: calls.append((a, k)))
    yield calls
    log.handlers[:] = handlers
    log.setLevel(level)


def _env(tmp_path, **extra) -> dict[str, str]:
    roots = tmp_path / "roots"
    roots.mkdir(exist_ok=True)
    return {
        "COLUMNS": "300",
        "HEARTH_HOME": str(tmp_path / "home"),
        "HEARTH_BACKEND": "echo",
        "HEARTH_FILE_ROOTS": str(roots),
        "HEARTH_EMBEDDER": "hash",
        **extra,
    }


ROUTER_COMMANDS = [
    pytest.param(["serve"], id="serve"),
    pytest.param(["run", "hello"], id="run"),
    pytest.param(["agent", "hello"], id="agent"),
    pytest.param(["rag", "query", "hello"], id="rag-query"),
    pytest.param(["rag", "query", "hello", "--answer"], id="rag-query-answer"),
    pytest.param(["mcp"], id="mcp"),
]


@pytest.mark.parametrize("argv", ROUTER_COMMANDS)
def test_a_missing_routing_profile_exits_2_with_the_fix_line(argv, tmp_path, no_server):
    if argv == ["mcp"]:
        pytest.importorskip("mcp")
    result = CliRunner().invoke(
        app, argv, env=_env(tmp_path, HEARTH_ROUTING_YAML=MISSING_PROFILE)
    )
    assert result.exit_code == 2, (result.exit_code, result.output, result.exception)
    # Exit 2 came from typer.Exit, not an uncaught exception CliRunner swallowed.
    assert isinstance(result.exception, SystemExit), repr(result.exception)
    flat = " ".join(result.output.split())
    assert "Routing profile not found" in flat
    assert "Fix or unset HEARTH_ROUTING_YAML" in flat
    assert MISSING_PROFILE in flat
    assert "Traceback" not in result.output
    assert no_server == []  # serve never got as far as binding


def test_serve_with_a_missing_profile_prints_no_traceback_in_a_real_process(tmp_path):
    """The rendered terminal output of a real `hearth serve`, not CliRunner's capture."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("HEARTH_")}
    env.update(_env(tmp_path, HEARTH_ROUTING_YAML=MISSING_PROFILE, PYTHONPATH=str(_SRC)))
    proc = subprocess.run(
        [sys.executable, "-c", "from hearth.cli import app; app()", "serve"],
        env=env, capture_output=True, text=True, timeout=60, cwd=str(tmp_path),
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 2, out
    assert "Traceback" not in out
    assert "Fix or unset HEARTH_ROUTING_YAML" in " ".join(out.split())


# -- serve: the banner is printed only once the app has been built ------------------------


def test_serve_with_a_missing_profile_never_prints_the_serving_banner(tmp_path, no_server):
    """"Serving on http://..." used to be printed before create_app raised, so a server that
    never started announced that it was serving."""
    result = CliRunner().invoke(
        app, ["serve"], env=_env(tmp_path, HEARTH_ROUTING_YAML=MISSING_PROFILE)
    )
    assert result.exit_code == 2, result.output
    assert "Serving on" not in result.output
    assert "backend=" not in result.output  # the banner's first line, too
    assert no_server == []


def test_serve_prints_no_banner_when_create_app_raises(tmp_path, no_server, monkeypatch):
    """Any failure building the app (not only the routing profile) precedes the banner."""
    import hearth.gateway as gateway

    def broken(**_kw):
        raise RuntimeError("synthetic create_app failure")

    monkeypatch.setattr(gateway, "create_app", broken)
    result = CliRunner().invoke(app, ["serve"], env=_env(tmp_path))
    assert isinstance(result.exception, RuntimeError), repr(result.exception)
    assert "Serving on" not in result.output
    assert no_server == []


def test_serve_prints_the_banner_once_the_app_is_built(tmp_path, no_server):
    """Guard the guard: on a healthy start the banner is there, so its absence above means
    something."""
    result = CliRunner().invoke(app, ["serve", "--port", "18999"], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    assert "Serving on http://127.0.0.1:18999" in result.output
    assert len(no_server) == 1


# -- B-036: an embedder that cannot embed -------------------------------------------------

#: An embedding model id that cannot be on disk, so MLXEmbedder's disk-only load fails
#: (EmbeddingUnavailableError) whether or not mlx-lm is installed.
ABSENT_EMBEDDER = "hearth-test/no-such-embedder"


@pytest.mark.parametrize("argv", [
    pytest.param(["rag", "ingest", "{doc}"], id="ingest"),
    pytest.param(["rag", "query", "hello"], id="query"),
    pytest.param(["rag", "query", "hello", "--answer"], id="query-answer"),
])
def test_an_unavailable_embedder_exits_1_with_its_message(argv, tmp_path):
    doc = tmp_path / "doc.txt"
    doc.write_text("some text worth embedding\n")
    argv = [a.format(doc=doc) for a in argv]
    result = CliRunner().invoke(app, argv, env=_env(
        tmp_path, HEARTH_EMBEDDER="mlx", HEARTH_EMBED_MODEL=ABSENT_EMBEDDER
    ))
    assert result.exit_code == 1, (result.exit_code, result.output, result.exception)
    assert isinstance(result.exception, SystemExit), repr(result.exception)
    flat = " ".join(result.output.split())
    assert "Embedder unavailable" in flat
    assert "Traceback" not in result.output
