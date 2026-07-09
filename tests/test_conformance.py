"""Client-agnostic conformance suite (Phase 5, ADR-001, docs/INTEGRATION.md "Testing").

The point: HEARTH is a genuinely standalone service. This suite exercises the FULL public
API surface end-to-end over the FastAPI app on the echo backend, with **no CAMBOT, no MLX /
remote / embeddings / mcp extras** — if it passes, HEARTH works for any client.

It also drives the Python :class:`~hearth.client.HearthClient` against the app in-process
(via ``httpx.ASGITransport``) and the MCP ``tools.py`` functions against the echo router,
so all three integration surfaces are covered without a running server or a live model.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hearth.client import HearthClient
from hearth.config import Settings, get_or_create_token
from hearth.gateway import create_app
from hearth.mcp.tools import build_toolset
from hearth.observability.budget import BudgetAccountant
from hearth.observability.metrics import MetricsStore
from hearth.providers.echo import EchoProvider
from hearth.router import Router, RoutingPolicy
from hearth.router.classify import TASK_CLASSES
from hearth.router.policy import ClassRule, Defaults


def _local_policy() -> RoutingPolicy:
    return RoutingPolicy(
        defaults=Defaults(),
        classes={c: ClassRule(backend="local", escalate="never") for c in TASK_CLASSES},
        remotes={},
    )


@pytest.fixture
def app(tmp_path):
    settings = Settings(backend="echo", home=tmp_path / ".hearth", require_auth=False)
    provider = EchoProvider()
    metrics = MetricsStore()
    router = Router(
        local_provider=provider,
        policy=_local_policy(),
        budget=BudgetAccountant(200000),
        metrics=metrics,
    )
    return create_app(
        provider=provider, settings=settings, router=router, metrics=metrics
    )


@pytest.fixture
def conf_client(app) -> TestClient:
    return TestClient(app)


# -- full HTTP surface on the echo backend ------------------------------------------------


def test_chat_non_streaming(conf_client):
    r = conf_client.post(
        "/v1/chat/completions",
        json={"model": "auto", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "[echo] hello"
    assert body["hearth"]["served_by"] == "local"


def test_chat_streaming(conf_client):
    r = conf_client.post(
        "/v1/chat/completions",
        json={"model": "auto", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.text.rstrip().endswith("data: [DONE]")


def test_models(conf_client):
    r = conf_client.get("/v1/models")
    assert r.status_code == 200
    ids = [m["id"] for m in r.json()["data"]]
    assert "echo" in ids


def test_embeddings(conf_client):
    r = conf_client.post("/v1/embeddings", json={"model": "auto", "input": ["a", "b"]})
    assert r.status_code == 200
    data = r.json()["data"]
    assert [d["index"] for d in data] == [0, 1]
    assert all(len(d["embedding"]) > 0 for d in data)


def test_route_dry_run(conf_client):
    r = conf_client.post(
        "/v1/hearth/route",
        json={"messages": [{"role": "user", "content": "summarize this"}]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["class"] == "summarize"
    assert body["backend"] == "local"


def test_rag_ingest_and_query(conf_client, tmp_path):
    doc = tmp_path / "note.txt"
    doc.write_text("HEARTH routes tasks to a local model to save frontier tokens.\n")
    ingest = conf_client.post(
        "/v1/hearth/rag/ingest",
        json={"collection": "conf", "paths": [str(doc)]},
    )
    assert ingest.status_code == 200
    assert ingest.json()["chunks"] >= 1

    query = conf_client.post(
        "/v1/hearth/rag/query",
        json={"collection": "conf", "query": "what saves tokens?", "k": 3},
    )
    assert query.status_code == 200
    assert len(query.json()["chunks"]) >= 1


def test_admin_health_and_metrics(conf_client):
    assert conf_client.get("/v1/hearth/admin/health").json()["status"] == "ok"
    # Drive one request so metrics are non-empty.
    conf_client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "summarize this"}]},
    )
    metrics = conf_client.get("/v1/hearth/admin/metrics").json()
    assert metrics["requests"] >= 1


def test_auth_401_and_200(tmp_path):
    settings = Settings(backend="echo", home=tmp_path / ".hearth", require_auth=True)
    client = TestClient(create_app(provider=EchoProvider(), settings=settings))
    assert client.get("/v1/hearth/admin/health").status_code == 200  # always open
    assert client.get("/v1/models").status_code == 401
    token = get_or_create_token(settings)
    ok = client.get("/v1/models", headers={"Authorization": f"Bearer {token}"})
    assert ok.status_code == 200


# -- Python client in-process (httpx.ASGITransport) ---------------------------------------


def _asgi_client(app) -> HearthClient:
    """A HearthClient whose HTTP calls hit the ASGI app in-process (no live server).

    Starlette's ``TestClient`` is an ``httpx.Client`` subclass with a sync-compatible ASGI
    transport, so we back the (synchronous) ``HearthClient`` with one — exercising the real
    client code against the real app without a socket.
    """
    hc = HearthClient("http://testserver")
    hc._client = TestClient(app)
    return hc


def test_python_client_chat_and_summarize(app):
    hc = _asgi_client(app)
    resp = hc.chat([{"role": "user", "content": "hello"}])
    assert resp["choices"][0]["message"]["content"] == "[echo] hello"
    assert hc.summarize("some long text", max_words=20).startswith("[echo]")


def test_python_client_stream(app):
    hc = _asgi_client(app)
    deltas = list(hc.chat([{"role": "user", "content": "hi there"}], stream=True))
    assert "".join(deltas) == "[echo] hi there"


def test_python_client_embed(app):
    hc = _asgi_client(app)
    vectors = hc.embed(["a", "b", "c"])
    assert len(vectors) == 3 and all(len(v) > 0 for v in vectors)


def test_python_client_rag_query(app, tmp_path):
    doc = tmp_path / "d.txt"
    doc.write_text("grounded context lives here\n")
    hc = _asgi_client(app)
    hc._client.post(
        "/v1/hearth/rag/ingest",
        json={"collection": "cc", "paths": [str(doc)]},
    )
    result = hc.rag_query("cc", "where does context live", k=2)
    assert "chunks" in result


# -- MCP tools against the echo router ----------------------------------------------------


def test_mcp_tools_surface(tmp_path):
    settings = Settings(backend="echo", home=tmp_path / ".hearth", require_auth=False)
    tools = build_toolset(settings=settings)
    assert tools.summarize("hello").startswith("[echo]")
    assert isinstance(tools.classify("x", ["a", "b"]), str)
    assert set(tools.extract("t", ["f1"]).keys()) == {"f1"}
    assert tools.rag_query("empty", "q")["chunks"] == []
