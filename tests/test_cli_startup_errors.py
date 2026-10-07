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
    """Never bind a port: record a serve call instead of serving.

    Also restores the ``hearth`` logger: `serve` attaches a stderr handler, and under
    CliRunner that stream is closed after the invocation — a handler left behind makes a
    later test's background log line print "--- Logging error ---".
    """
    import logging

    import hearth.cli as cli

    log = logging.getLogger("hearth")
    handlers, level = list(log.handlers), log.level
    calls: list = []
    # serve binds first (B-129), then hands the socket to uvicorn: fake both seams, so no
    # port is bound (this sandbox cannot bind at all) and serving is only recorded.
    monkeypatch.setattr(cli, "_bind", lambda host, port: _FakeSocket())
    monkeypatch.setattr(cli, "_run_uvicorn", lambda *a, **k: calls.append((a, k)))
    yield calls
    log.handlers[:] = handlers
    log.setLevel(level)


class _FakeSocket:
    """Stands in for the listening socket serve binds; records being closed."""

    closed = False

    def close(self) -> None:
        self.closed = True


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



# -- B-129: the port is bound before the banner, and before warmup can start ----------------


def test_a_port_in_use_exits_1_before_the_app_or_its_warmup_exist(tmp_path, no_server,
                                                                   monkeypatch):
    """With the port taken: no banner, no create_app (warmup starts inside it), exit 1 with
    the way to find the holder."""
    import errno

    import hearth.cli as cli
    import hearth.gateway as gateway

    def taken(host, port):
        raise OSError(errno.EADDRINUSE, "Address already in use")

    built: list = []
    monkeypatch.setattr(cli, "_bind", taken)
    monkeypatch.setattr(gateway, "create_app", lambda **kw: built.append(kw))
    result = CliRunner().invoke(app, ["serve", "--port", "18998"], env=_env(tmp_path))
    out = " ".join(result.output.split())
    assert result.exit_code == 1, result.output
    assert "Serving on" not in out and "backend=" not in out
    assert "Cannot listen on 127.0.0.1:18998: Address already in use." in out
    assert "lsof -nP -iTCP:18998 -sTCP:LISTEN" in out
    assert built == [] and no_server == []


def test_a_broken_profile_is_reported_before_the_port_is_tried(tmp_path, no_server,
                                                               monkeypatch):
    """Configuration errors win over port errors: exit 2 naming the profile."""
    import hearth.cli as cli

    tried: list = []
    monkeypatch.setattr(cli, "_bind", lambda host, port: tried.append(port))
    result = CliRunner().invoke(
        app, ["serve"], env=_env(tmp_path, HEARTH_ROUTING_YAML=MISSING_PROFILE)
    )
    assert result.exit_code == 2, result.output
    assert tried == []


def test_the_socket_is_closed_when_the_app_cannot_be_built(tmp_path, no_server, monkeypatch):
    import hearth.cli as cli
    import hearth.gateway as gateway

    sock = _FakeSocket()
    monkeypatch.setattr(cli, "_bind", lambda host, port: sock)

    def broken(**_kw):
        raise RuntimeError("synthetic create_app failure")

    monkeypatch.setattr(gateway, "create_app", broken)
    result = CliRunner().invoke(app, ["serve"], env=_env(tmp_path))
    assert isinstance(result.exception, RuntimeError), repr(result.exception)
    assert sock.closed and no_server == []


def test_serve_hands_uvicorn_the_socket_it_bound(tmp_path, no_server, monkeypatch):
    import hearth.cli as cli

    sock = _FakeSocket()
    monkeypatch.setattr(cli, "_bind", lambda host, port: sock)
    result = CliRunner().invoke(app, ["serve", "--port", "18997"], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    (args, _kwargs), = no_server
    assert args[1] is sock and args[2:] == ("127.0.0.1", 18997)


def _can_bind() -> bool:
    import socket

    try:
        socket.create_server(("127.0.0.1", 0)).close()
    except OSError:
        return False
    return True


@pytest.mark.skipif(not _can_bind(), reason="this environment cannot bind sockets")
def test_a_real_port_conflict_in_a_real_process(tmp_path):
    """The end-to-end case B-129 was found in: another listener already holds the port."""
    import socket

    holder = socket.create_server(("127.0.0.1", 0))
    port = holder.getsockname()[1]
    try:
        env = {k: v for k, v in os.environ.items() if not k.startswith("HEARTH_")}
        env.update(_env(tmp_path, PYTHONPATH=str(_SRC)))
        proc = subprocess.run(
            [sys.executable, "-c", "from hearth.cli import app; app()",
             "serve", "--port", str(port)],
            env=env, capture_output=True, text=True, timeout=60, cwd=str(tmp_path),
        )
    finally:
        holder.close()
    out = " ".join((proc.stdout + proc.stderr).split())
    assert proc.returncode == 1, out
    assert "Serving on" not in out and "Traceback" not in out
    assert f"Cannot listen on 127.0.0.1:{port}" in out


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
    if argv[1] == "query":
        # A query embeds only when the collection has chunks (an empty one says "No chunks"
        # without embedding, below), so index the default collection with the hash embedder.
        seeded = CliRunner().invoke(app, ["rag", "ingest", str(doc)], env=_env(tmp_path))
        assert seeded.exit_code == 0, seeded.output
        get_settings.cache_clear()  # the seed run cached HEARTH_EMBEDDER=hash
    result = CliRunner().invoke(app, argv, env=_env(
        tmp_path, HEARTH_EMBEDDER="mlx", HEARTH_EMBED_MODEL=ABSENT_EMBEDDER
    ))
    assert result.exit_code == 1, (result.exit_code, result.output, result.exception)
    assert isinstance(result.exception, SystemExit), repr(result.exception)
    flat = " ".join(result.output.split())
    assert "Embedder unavailable" in flat
    assert "Traceback" not in result.output


@pytest.mark.parametrize("answer", [False, True], ids=["query", "query-answer"])
def test_an_empty_collection_says_no_chunks_even_when_the_embedder_cannot_load(tmp_path, answer):
    """`rag query` embedded the query before looking at the collection, so with the mlx
    embedder (B-011) an empty collection was an "Embedder unavailable" error instead of
    "No chunks"."""
    argv = ["rag", "query", "hello", "--collection", "empty"] + (["--answer"] if answer else [])
    result = CliRunner().invoke(app, argv, env=_env(
        tmp_path, HEARTH_EMBEDDER="mlx", HEARTH_EMBED_MODEL=ABSENT_EMBEDDER
    ))
    assert result.exit_code == 0, (result.output, result.exception)
    flat = " ".join(result.output.split())
    assert "No chunks in collection 'empty'" in flat
    assert "Embedder unavailable" not in flat


def test_rag_index_does_not_embed_for_an_empty_collection(tmp_path):
    """The library path too (HTTP /v1/hearth/rag/query and MCP use RagIndex.query)."""
    from hearth.memory import RagIndex
    from hearth.memory.store import SQLiteVectorStore

    class Exploding:
        dim = 4

        def embed(self, texts):
            raise AssertionError("embedded a query for an empty collection")

    index = RagIndex(embedder=Exploding(), store=SQLiteVectorStore(root=tmp_path / "rag"))
    assert index.query("nothing-here", "hello").chunks == []
