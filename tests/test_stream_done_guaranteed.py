"""Every SSE stream ends with ``[DONE]``, even when something fails after the relay (B-007).

The answer has already reached the client when the post-relay accounting runs. A metrics
store that raised there (a full disk) used to kill the generator: the /chat page had shown
the whole answer and then saw a dropped connection with no ``[DONE]`` and no error event.
These tests drive the real Router + gateway and assert on the bytes the client receives.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from hearth.config import Settings
from hearth.gateway import create_app
from hearth.gateway.app import _guarantee_done
from hearth.observability.budget import BudgetAccountant
from hearth.observability.metrics import MetricsStore
from hearth.providers.echo import EchoProvider
from hearth.router import Router
from hearth.router.policy import Defaults, RoutingPolicy


class DiskFullMetrics(MetricsStore):
    def record(self, rec):
        raise OSError("disk full")


def _client(tmp_path, metrics: MetricsStore) -> TestClient:
    local = EchoProvider()
    router = Router(
        local_provider=local,
        policy=RoutingPolicy(defaults=Defaults(), classes={}, remotes={}),
        budget=BudgetAccountant(10_000),
        metrics=metrics,
    )
    settings = Settings(backend="echo", home=tmp_path / ".hearth", require_auth=False)
    return TestClient(
        create_app(provider=local, settings=settings, router=router, metrics=metrics)
    )


def _events(body: str) -> list:
    out = []
    for line in body.splitlines():
        if line.startswith("data: "):
            data = line[len("data: "):]
            out.append(data if data == "[DONE]" else json.loads(data))
    return out


def _post(client, stream=True):
    return client.post(
        "/v1/chat/completions",
        json={"stream": stream, "messages": [{"role": "user", "content": "hello there"}]},
    )


def test_a_metrics_failure_after_the_relay_still_ends_with_done(tmp_path):
    events = _events(_post(_client(tmp_path, DiskFullMetrics())).text)
    assert events[-1] == "[DONE]"
    text = "".join(
        e["choices"][0]["delta"].get("content") or ""
        for e in events
        if isinstance(e, dict) and e.get("choices")
    )
    assert text  # the answer was delivered
    assert any(isinstance(e, dict) and e.get("hearth") for e in events)  # final chunk too
    errors = [e for e in events if isinstance(e, dict) and "error" in e]
    assert [e["error"]["code"] for e in errors] == ["hearth.metrics.unavailable"]
    assert "disk full" in errors[0]["error"]["message"]
    assert events.index(errors[0]) == len(events) - 2  # the error precedes [DONE]


def test_a_metrics_failure_does_not_500_a_served_non_streaming_answer(tmp_path):
    r = _post(_client(tmp_path, DiskFullMetrics()), stream=False)
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"]


def test_an_unexpected_error_before_the_first_chunk_still_ends_with_done(tmp_path):
    client = _client(tmp_path, MetricsStore())
    router = client.app.state.router

    def explode(*a, **k):
        raise RuntimeError("adapter store corrupt")

    router.select_adapter = explode
    events = _events(_post(client).text)
    assert events[-1] == "[DONE]"
    errors = [e for e in events if isinstance(e, dict) and "error" in e]
    assert len(errors) == 1
    assert "adapter store corrupt" in errors[0]["error"]["message"]


def test_closing_the_guard_closes_the_inner_stream():
    """An abandoned stream must still cancel generation (92d8f19): close() passes through."""
    closed = []

    def inner():
        try:
            yield "data: one\n\n"
            yield "data: two\n\n"
        finally:
            closed.append(True)

    guarded = _guarantee_done(inner())
    assert next(guarded) == "data: one\n\n"
    guarded.close()
    assert closed == [True]
