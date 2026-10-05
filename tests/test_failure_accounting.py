"""B-066: every request that ends in an error writes a RequestRecord.

Before: a local rung nobody could serve (404 / stream ``model_not_found`` event), a denied
escalation (429 / stream ``budget_exhausted`` event), and a remote failure followed by a local
UnknownModelError all reached the client as errors with **no record** — so ``hearth stats``
read them as zero traffic, and the last one lost the fact that the remote had been called
(and may hold the prompt). The stream branch's ``[DONE]`` was untested: deleting it failed
nothing.

The fakes below raise from the provider itself, so these assert on the client's response and
the metrics store's contents — not on which branch the code took.
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
from hearth.router import BudgetExhaustedError, Router
from hearth.router.classify import TASK_CLASSES
from hearth.router.policy import ClassRule, Defaults, RemoteConfig, RoutingPolicy
from hearth.serving import UnknownModelError

RUNG = "mlx-community/Qwen2.5-3B-Instruct-4bit"  # a registered chat id: passes startup checks
REASON = "prove this step by step"  # classifies as `reason`
CHAT = "hello there"  # classifies as `chat`


class _Provider:
    def capabilities(self) -> Capabilities:
        return Capabilities(chat=True, stream=True, adapters=True)

    def footprint(self, model_id: str) -> ResourceEstimate:
        return ResourceEstimate()


class UnservableLocal(_Provider):
    """A local backend that refuses the model it is asked for, as a pool does an unknown id."""

    name = "mlx"

    def generate(self, req: GenRequest) -> GenResult:
        raise UnknownModelError(req.model, f"The model {req.model!r} cannot be served here")

    def stream(self, req: GenRequest) -> Iterator[str]:
        raise UnknownModelError(req.model, f"The model {req.model!r} cannot be served here")
        yield  # pragma: no cover


class DeadRemote(_Provider):
    name = "remote"

    def __init__(self, config: RemoteConfig) -> None:
        self.config = config

    def generate(self, req: GenRequest) -> GenResult:
        raise ConnectionError("remote unreachable (offline)")

    def stream(self, req: GenRequest) -> Iterator[str]:
        raise ConnectionError("remote unreachable (offline)")
        yield  # pragma: no cover


def _local_policy(rung: str | None = RUNG) -> RoutingPolicy:
    return RoutingPolicy(
        defaults=Defaults(),
        classes={
            c: ClassRule(backend="local", escalate="never", local_model=rung)
            for c in TASK_CLASSES
        },
        remotes={},
    )


def _remote_policy() -> RoutingPolicy:
    classes = {c: ClassRule(backend="local", escalate="never", local_model=RUNG)
               for c in TASK_CLASSES}
    classes["reason"] = ClassRule(backend="remote", escalate="always", local_model=RUNG)
    return RoutingPolicy(
        defaults=Defaults(remote="frontier"),
        classes=classes,
        remotes={"frontier": RemoteConfig(protocol="anthropic", model="frontier-x")},
    )


def _router(local, policy, *, budget: int = 1_000_000) -> Router:
    return Router(
        local_provider=local, policy=policy, budget=BudgetAccountant(budget),
        metrics=MetricsStore(), remote_factory=DeadRemote,
    )


def _client(tmp_path, router: Router) -> TestClient:
    settings = Settings(backend="echo", home=tmp_path / ".hearth", require_auth=False)
    app = create_app(provider=router.local, settings=settings, router=router,
                     metrics=router.metrics)
    return TestClient(app)


def _post(client, text: str, *, stream: bool, **hearth):
    body = {"messages": [{"role": "user", "content": text}], "stream": stream}
    if hearth:
        body["hearth"] = hearth
    return client.post("/v1/chat/completions", json=body)


def _events(resp) -> list:
    out = []
    for line in resp.text.splitlines():
        if line.startswith("data: "):
            payload = line[6:]
            out.append(payload if payload == "[DONE]" else json.loads(payload))
    return out


def _records(router: Router):
    return list(router.metrics._records)


def _req(text: str) -> GenRequest:
    return GenRequest(messages=[Message(role="user", content=text)], model="auto")


# -- an unservable local rung --------------------------------------------------------------


def test_route_records_an_unservable_rung_before_raising():
    router = _router(UnservableLocal(), _local_policy())
    with pytest.raises(UnknownModelError):
        router.route(_req(CHAT))
    (rec,) = _records(router)
    assert rec.failed and RUNG in rec.failed
    assert (rec.served_by, rec.model, rec.escalated) == ("local", RUNG, False)
    assert router.metrics.rollup()["failed"] == 1


def test_http_unservable_rung_is_a_404_and_is_recorded(tmp_path):
    router = _router(UnservableLocal(), _local_policy())
    resp = _post(_client(tmp_path, router), CHAT, stream=False)
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "model_not_found"
    (rec,) = _records(router)
    assert rec.failed


def test_stream_unservable_rung_ends_with_error_then_done_and_is_recorded(tmp_path):
    router = _router(UnservableLocal(), _local_policy())
    events = _events(_post(_client(tmp_path, router), CHAT, stream=True))
    assert events[-1] == "[DONE]"  # the branch's own [DONE], not a backstop's
    assert events[-2]["error"]["code"] == "model_not_found"
    assert not any(
        isinstance(e, dict) and e.get("error", {}).get("code") == "hearth.stream.internal_error"
        for e in events
    )
    (rec,) = _records(router)
    assert rec.failed and rec.served_by == "local"


# -- a denied escalation ------------------------------------------------------------------


def test_route_records_a_budget_denial_before_raising():
    router = _router(UnservableLocal(), _remote_policy(), budget=0)
    with pytest.raises(BudgetExhaustedError):
        router.route(_req(REASON))
    (rec,) = _records(router)
    assert rec.failed and "budget" in rec.failed
    assert rec.served_by == "remote"
    # Denied: nothing left the machine, so it is not counted as an escalation.
    assert rec.escalated is False and rec.escalation_failed is None
    assert rec.estimated_frontier_tokens_saved == 0


def test_route_records_a_missing_remote_before_raising():
    policy = _remote_policy()
    policy = RoutingPolicy(defaults=Defaults(remote="nowhere"), classes=policy.classes,
                           remotes={})
    router = _router(UnservableLocal(), policy)
    with pytest.raises(BudgetExhaustedError, match="no remote configured"):
        router.route(_req(REASON))
    (rec,) = _records(router)
    assert rec.failed and "no remote configured" in rec.failed


def test_http_budget_denial_is_a_429_and_is_recorded(tmp_path):
    router = _router(UnservableLocal(), _remote_policy(), budget=0)
    resp = _post(_client(tmp_path, router), REASON, stream=False)
    assert resp.status_code == 429
    (rec,) = _records(router)
    assert rec.failed


def test_stream_budget_denial_ends_with_error_then_done_and_is_recorded(tmp_path):
    router = _router(UnservableLocal(), _remote_policy(), budget=0)
    events = _events(_post(_client(tmp_path, router), REASON, stream=True))
    assert events[-1] == "[DONE]"
    assert events[-2]["error"]["code"] == "hearth.budget.exhausted"
    (rec,) = _records(router)
    assert rec.failed and rec.served_by == "remote"


# -- a remote failure, then a local UnknownModelError ---------------------------------------


def test_route_records_the_failed_escalation_when_the_local_fallback_404s():
    router = _router(UnservableLocal(), _remote_policy())
    with pytest.raises(UnknownModelError):
        router.route(_req(REASON))
    (rec,) = _records(router)
    assert rec.failed
    # The remote WAS called and may hold the prompt: the audit trail must say so.
    assert rec.escalation_failed and "unreachable" in rec.escalation_failed
    assert rec.served_by == "local"


def test_stream_records_the_failed_escalation_when_the_local_fallback_404s(tmp_path):
    router = _router(UnservableLocal(), _remote_policy())
    events = _events(_post(_client(tmp_path, router), REASON, stream=True))
    assert events[-1] == "[DONE]"
    assert events[-2]["error"]["code"] == "model_not_found"
    (rec,) = _records(router)
    assert rec.failed and rec.escalation_failed and "unreachable" in rec.escalation_failed
