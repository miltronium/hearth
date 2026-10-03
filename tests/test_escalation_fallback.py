"""A failed escalation degrades to local; a failed stream still ends, and says why.

Before: a remote failure (offline, SDK missing, outage) surfaced as a 503 on the plain path
and, on the streaming path the /chat page uses, killed the generator mid-response with no
error event and no ``[DONE]``. These tests drive the real Router and gateway with remotes
and locals that raise, and assert on what the client actually receives and what the
metrics actually record — not on which branch the code took.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from hearth.config import Settings
from hearth.gateway import create_app
from hearth.observability.budget import BudgetAccountant
from hearth.observability.metrics import MetricsStore
from hearth.providers.base import Capabilities, GenRequest, GenResult, Message, ResourceEstimate
from hearth.providers.echo import EchoProvider
from hearth.router import ProviderError, Router
from hearth.router.policy import ClassRule, Defaults, RemoteConfig, RoutingPolicy
from hearth.router.route import REASON_REMOTE_FAILURE

PROMPT = "prove this step by step"  # classifies as `reason`


class _Provider:
    def capabilities(self) -> Capabilities:
        return Capabilities(chat=True, stream=True)

    def footprint(self, model_id: str) -> ResourceEstimate:
        return ResourceEstimate()


class DeadRemote(_Provider):
    """A remote that fails before producing anything — the offline / no-SDK case."""

    name = "remote"

    def __init__(self, config: RemoteConfig) -> None:
        self.config = config

    def generate(self, req: GenRequest) -> GenResult:
        raise ConnectionError("remote unreachable (offline)")

    def stream(self, req: GenRequest) -> Iterator[str]:
        raise ConnectionError("remote unreachable (offline)")
        yield  # pragma: no cover — makes this a generator, so it raises on first next()


class FlakyRemote(_Provider):
    """A remote that streams a little and then drops the connection."""

    name = "remote"

    def __init__(self, config: RemoteConfig) -> None:
        self.config = config

    def generate(self, req: GenRequest) -> GenResult:  # pragma: no cover — stream-only
        raise AssertionError("not used")

    def stream(self, req: GenRequest) -> Iterator[str]:
        yield "[remote] partial"
        raise ConnectionError("connection reset mid-answer")


class DeadLocal(_Provider):
    name = "mlx"

    def generate(self, req: GenRequest) -> GenResult:
        raise RuntimeError("weights missing")

    def stream(self, req: GenRequest) -> Iterator[str]:
        raise RuntimeError("weights missing")
        yield  # pragma: no cover


def _escalating_policy() -> RoutingPolicy:
    return RoutingPolicy(
        defaults=Defaults(),
        classes={"reason": ClassRule(backend="remote", escalate="always")},
        remotes={"default": RemoteConfig(protocol="anthropic", model="frontier")},
    )


def _router(remote_factory, local=None, metrics=None) -> Router:
    return Router(
        local_provider=local or EchoProvider(),
        policy=_escalating_policy(),
        budget=BudgetAccountant(10_000),
        metrics=metrics or MetricsStore(),
        remote_factory=remote_factory,
    )


def _client(tmp_path, remote_factory, local=None) -> TestClient:
    metrics = MetricsStore()
    local = local or EchoProvider()
    router = _router(remote_factory, local=local, metrics=metrics)
    settings = Settings(backend="echo", home=tmp_path / ".hearth", require_auth=False)
    app = create_app(provider=local, settings=settings, router=router, metrics=metrics)
    return TestClient(app)


def _events(body: str) -> list:
    """Parse an SSE body into its ``data:`` payloads (JSON where possible)."""
    out = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        data = line[len("data: ") :]
        out.append(data if data == "[DONE]" else json.loads(data))
    return out


# -- Router.route ---------------------------------------------------------------------


def test_a_dead_remote_is_served_locally_and_recorded_as_local():
    router = _router(DeadRemote)
    routed = router.route(GenRequest(messages=[Message(role="user", content=PROMPT)], model="auto"))
    assert routed.result.backend == "echo"  # the local provider actually answered
    assert routed.record.served_by == "local"
    assert routed.record.escalated is False
    assert routed.decision.would_escalate is False
    assert routed.decision.reason.startswith(REASON_REMOTE_FAILURE)
    assert "offline" in routed.decision.reason
    assert router.budget.spent() == 0  # nothing was billed for a call that never happened
    assert router.metrics.rollup()["backend_mix"] == {"local": 1}
    # ...and the outage is visible in the metrics, not only in a log line.
    assert "offline" in routed.record.escalation_failed
    assert router.metrics.rollup()["escalations_failed"] == 1
    assert router.metrics.rollup()["escalations"] == 0


def test_when_local_also_fails_the_error_still_surfaces():
    router = _router(DeadRemote, local=DeadLocal())
    with pytest.raises(ProviderError):
        router.route(GenRequest(messages=[Message(role="user", content=PROMPT)], model="auto"))


def test_a_local_failure_is_not_retried_anywhere_else():
    router = _router(DeadRemote, local=DeadLocal())
    with pytest.raises(ProviderError, match="mlx"):
        router.route(
            GenRequest(messages=[Message(role="user", content="summarize this")], model="auto"),
            allow_escalation=False,
        )


# -- gateway, non-streaming -----------------------------------------------------------


def test_chat_completions_answers_locally_when_the_remote_is_down(tmp_path):
    client = _client(tmp_path, DeadRemote)
    body = {"messages": [{"role": "user", "content": PROMPT}]}
    r = client.post("/v1/chat/completions", json=body)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["hearth"]["served_by"] == "local"
    assert body["hearth"]["escalated"] is False
    assert body["hearth"]["backend"] == "echo"
    assert body["choices"][0]["message"]["content"]


# -- gateway, streaming ---------------------------------------------------------------


def _stream(client: TestClient) -> list:
    r = client.post(
        "/v1/chat/completions",
        json={"stream": True, "messages": [{"role": "user", "content": PROMPT}]},
    )
    assert r.status_code == 200
    return _events(r.text)


def test_stream_degrades_to_local_when_the_remote_fails_before_any_text(tmp_path):
    client = _client(tmp_path, DeadRemote)
    events = _stream(client)
    assert events[-1] == "[DONE]"
    assert not any(isinstance(e, dict) and "error" in e for e in events)
    text = "".join(
        e["choices"][0]["delta"].get("content") or ""
        for e in events
        if isinstance(e, dict) and e.get("choices")
    )
    assert text  # the local model's answer actually streamed
    final = next(e for e in events if isinstance(e, dict) and e.get("hearth"))
    assert final["hearth"]["served_by"] == "local"
    assert final["hearth"]["escalated"] is False
    assert client.app.state.metrics.rollup()["backend_mix"] == {"local": 1}
    assert client.app.state.metrics.rollup()["escalations_failed"] == 1
    assert client.app.state.router.budget.spent() == 0


def test_stream_that_dies_mid_answer_ends_with_an_error_and_done(tmp_path):
    client = _client(tmp_path, FlakyRemote)
    events = _stream(client)
    assert events[-1] == "[DONE]"
    errors = [e for e in events if isinstance(e, dict) and "error" in e]
    assert len(errors) == 1
    assert errors[0]["error"]["code"] == "hearth.provider.unavailable"
    assert "connection reset" in errors[0]["error"]["message"]
    # No local answer was spliced onto the remote's partial one, and no success telemetry.
    assert not any(isinstance(e, dict) and e.get("hearth") for e in events)


def test_stream_whose_local_provider_fails_ends_with_an_error_and_done(tmp_path):
    client = _client(tmp_path, DeadRemote, local=DeadLocal())
    events = _stream(client)
    assert events[-1] == "[DONE]"
    errors = [e for e in events if isinstance(e, dict) and "error" in e]
    assert len(errors) == 1
    assert "weights missing" in errors[0]["error"]["message"]


def test_a_remote_that_dies_mid_stream_is_billed_and_recorded(tmp_path):
    """It received the prompt and produced tokens: that is spend, and a failed escalation."""
    client = _client(tmp_path, FlakyRemote)
    _stream(client)
    roll = client.app.state.metrics.rollup()
    assert roll["requests"] == 1
    assert roll["escalations"] == 1
    assert roll["escalations_failed"] == 1
    assert roll["backend_mix"] == {"remote": 1}
    assert client.app.state.router.budget.spent() > 0


class AdapterSensitiveLocal(_Provider):
    """A local backend whose (promoted) adapter is broken; base weights work."""

    name = "mlx"

    def generate(self, req: GenRequest) -> GenResult:
        if req.adapter:
            raise RuntimeError("bad adapter shape")
        return GenResult(text="base answer", model=req.model, backend=self.name,
                         prompt_tokens=1, completion_tokens=2)

    def stream(self, req: GenRequest) -> Iterator[str]:
        if req.adapter:
            raise RuntimeError("bad adapter shape")
        yield "base answer"


def _adapter_client(tmp_path) -> TestClient:
    client = _client(tmp_path, DeadRemote, local=AdapterSensitiveLocal())
    router = client.app.state.router
    router._resolve_adapter = lambda requested, task_class, model=None: "/adapters/broken"
    return client


def test_a_broken_adapter_falls_back_to_base_weights_when_streaming(tmp_path):
    client = _adapter_client(tmp_path)
    r = client.post(
        "/v1/chat/completions",
        json={"stream": True, "messages": [{"role": "user", "content": "summarize this"}]},
    )
    events = _events(r.text)
    assert events[-1] == "[DONE]"
    assert not any(isinstance(e, dict) and "error" in e for e in events)
    text = "".join(
        e["choices"][0]["delta"].get("content") or ""
        for e in events
        if isinstance(e, dict) and e.get("choices")
    )
    assert text == "base answer"


def test_streaming_and_non_streaming_agree_on_a_broken_adapter(tmp_path):
    client = _adapter_client(tmp_path)
    body = {"messages": [{"role": "user", "content": "summarize this"}]}
    plain = client.post("/v1/chat/completions", json=body)
    assert plain.status_code == 200
    assert plain.json()["choices"][0]["message"]["content"] == "base answer"
