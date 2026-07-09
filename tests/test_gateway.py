"""End-to-end gateway tests — the walking skeleton must stay green at every phase."""

from __future__ import annotations


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
