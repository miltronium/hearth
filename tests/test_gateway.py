"""End-to-end gateway tests — the walking skeleton must stay green at every phase."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from hearth.config import Settings, get_or_create_token
from hearth.gateway import create_app
from hearth.providers.echo import EchoProvider


def test_health(client):
    r = client.get("/v1/hearth/admin/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["backend"] == "echo"


def test_list_models(client, settings):
    r = client.get("/v1/models")
    assert r.status_code == 200
    ids = [m["id"] for m in r.json()["data"]]
    assert settings.default_model in ids
    assert "echo" in ids


def test_chat_completion_roundtrip(client):
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "hello world"}],
        },
    )
    assert r.status_code == 200
    body = r.json()
    # OpenAI-compatible shape
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "[echo] hello world"
    assert body["usage"]["total_tokens"] > 0
    # HEARTH telemetry block rides along
    assert body["hearth"]["served_by"] == "local"
    assert body["hearth"]["backend"] == "echo"
    assert body["hearth"]["estimated_frontier_tokens_saved"] > 0


def _parse_sse(text: str) -> list:
    """Return the parsed JSON payloads of each `data:` event (excluding [DONE])."""
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            payload = line[len("data: ") :]
            if payload != "[DONE]":
                events.append(json.loads(payload))
    return events


def test_chat_completion_streaming(client):
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "stream": True,
            "messages": [{"role": "user", "content": "hello world"}],
        },
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.text.rstrip().endswith("data: [DONE]")

    chunks = _parse_sse(r.text)
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    # First chunk announces the assistant role.
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
    # Reassembled deltas equal the full echo text.
    content = "".join(
        c["choices"][0]["delta"].get("content") or "" for c in chunks
    )
    assert content == "[echo] hello world"
    # Final chunk carries finish_reason + hearth telemetry.
    last = chunks[-1]
    assert last["choices"][0]["finish_reason"] == "stop"
    assert last["hearth"]["backend"] == "echo"
    assert last["hearth"]["estimated_frontier_tokens_saved"] > 0


def test_embeddings_stub_501(client):
    r = client.post("/v1/embeddings", json={"model": "auto", "input": "x"})
    assert r.status_code == 501
    assert r.json()["error"]["code"] == "hearth.embeddings.not_implemented"


def test_auth_enforced(tmp_path):
    """With require_auth on, protected routes 401 without a token and 200 with it."""
    settings = Settings(backend="echo", home=tmp_path / ".hearth", require_auth=True)
    app = create_app(provider=EchoProvider(), settings=settings)
    auth_client = TestClient(app)

    # Health is always open.
    assert auth_client.get("/v1/hearth/admin/health").status_code == 200

    # Protected route rejects a missing/bad token.
    assert auth_client.get("/v1/models").status_code == 401
    bad = auth_client.get("/v1/models", headers={"Authorization": "Bearer nope"})
    assert bad.status_code == 401

    # The generated token unlocks it.
    token = get_or_create_token(settings)
    ok = auth_client.get("/v1/models", headers={"Authorization": f"Bearer {token}"})
    assert ok.status_code == 200
