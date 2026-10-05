"""B-106: every way a chat request can end leaves a record — abandonment and refusals too.

Before:
* a chat stream the client walked away from left NO record (``GeneratorExit`` is not an
  ``Exception``, so it skipped every ``except`` in ``_stream_sse``) — while an abandoned agent
  run was recorded as failed;
* an explicitly requested adapter refused with a 404 (``check_adapter``) left no record;
* an adapter that ``select_adapter`` could not resolve after the up-front check raised out
  of the stream before the first chunk: an internal-error event and no record.

Ports reviewer probe p6 and extends it to the refusals.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from hearth.config import Settings
from hearth.gateway import create_app
from hearth.gateway.app import CLIENT_DISCONNECTED, _guarantee_done, _stream_sse
from hearth.observability.budget import BudgetAccountant
from hearth.observability.metrics import MetricsStore
from hearth.providers.base import GenRequest, Message
from hearth.providers.echo import EchoProvider
from hearth.router import Router
from hearth.router.route import UnknownAdapterError


class _FlakyStore:
    """Resolves an adapter once (the up-front check), then it is gone (retired meanwhile)."""

    def __init__(self) -> None:
        self.calls = 0

    def resolve_path(self, adapter_id, allow_candidate=False):
        self.calls += 1
        if self.calls > 1:
            raise LookupError(f"adapter {adapter_id!r} was retired")
        return "/a/x"

    def promoted_for(self, task_class):
        return None


class _EmptyStore:
    def resolve_path(self, adapter_id, allow_candidate=False):
        raise LookupError(f"no adapter {adapter_id!r}")

    def promoted_for(self, task_class):
        return None


def _router(local_policy, adapters=None) -> Router:
    return Router(local_provider=EchoProvider(), policy=local_policy,
                  budget=BudgetAccountant(0), metrics=MetricsStore(), adapters=adapters)


def _client(tmp_path, router) -> TestClient:
    settings = Settings(backend="echo", home=tmp_path / ".hearth", require_auth=False)
    return TestClient(create_app(provider=router.local, settings=settings, router=router,
                                 metrics=router.metrics))


def _req(text="hello there, a long enough message to stream in several words"):
    return GenRequest(messages=[Message(role="user", content=text)], model="auto")


def _records(router):
    return list(router.metrics._records)


def test_an_abandoned_chat_stream_is_recorded_as_failed(local_policy):
    router = _router(local_policy)
    stream = _guarantee_done(_stream_sse(router, _req(), None, True, None))
    next(stream)  # the role chunk
    next(stream)  # one content chunk — then the client goes away
    stream.close()
    (rec,) = _records(router)
    assert rec.failed == CLIENT_DISCONNECTED
    assert (rec.served_by, rec.escalated, rec.model) == ("local", False, "echo")
    assert rec.completion_tokens >= 1  # the part that was generated is counted


def test_a_stream_abandoned_after_the_role_chunk_is_recorded_too(local_policy):
    router = _router(local_policy)
    stream = _stream_sse(router, _req(), None, True, None)
    next(stream)
    stream.close()
    (rec,) = _records(router)
    assert rec.failed == CLIENT_DISCONNECTED and rec.completion_tokens == 0


def test_a_completed_stream_closed_late_is_recorded_once_as_served(local_policy):
    router = _router(local_policy)
    stream = _stream_sse(router, _req(), None, True, None)
    events = []
    for event in stream:
        events.append(event)
        if '"hearth"' in event:
            break  # the final chunk is out: the outcome is recorded
    stream.close()
    (rec,) = _records(router)
    assert rec.failed is None


def test_a_stream_never_started_records_nothing(local_policy):
    router = _router(local_policy)
    _stream_sse(router, _req(), None, True, None).close()
    assert _records(router) == []


def _chat(client, stream, adapter):
    return client.post("/v1/chat/completions", json={
        "model": "auto", "stream": stream, "hearth": {"adapter": adapter},
        "messages": [{"role": "user", "content": "hello there"}]})


def _sse_events(resp):
    return [json.loads(line[6:]) for line in resp.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"]


def test_an_adapter_404_is_recorded(tmp_path, local_policy):
    for stream in (False, True):
        router = _router(local_policy, adapters=_EmptyStore())
        resp = _chat(_client(tmp_path, router), stream, "ghost")
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "adapter_not_found"
        (rec,) = _records(router)
        assert rec.failed and "ghost" in rec.failed
        assert (rec.adapter, rec.served_by, rec.escalated) == ("ghost", "local", False)


def test_an_adapter_unresolvable_at_selection_is_recorded_and_named(tmp_path, local_policy):
    # Streaming: the up-front check passes, selection fails before the first chunk.
    router = _router(local_policy, adapters=_FlakyStore())
    resp = _chat(_client(tmp_path, router), True, "x")
    events = _sse_events(resp)
    assert resp.text.rstrip().endswith("data: [DONE]")
    assert events == [events[-1]] and events[-1]["error"]["code"] == "adapter_not_found"
    (rec,) = _records(router)
    assert rec.failed and "retired" in rec.failed and rec.adapter == "x"


def test_route_records_an_adapter_unresolvable_at_selection(local_policy):
    router = _router(local_policy, adapters=_FlakyStore())
    with pytest.raises(UnknownAdapterError):
        router.route(_req(), adapter="x")
    (rec,) = _records(router)
    assert rec.failed and "retired" in rec.failed
