"""Phase 2 gateway route tests: /v1/hearth/route dry-run, /admin/metrics, escalation."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi.testclient import TestClient

from hearth.config import Settings
from hearth.gateway import create_app
from hearth.observability.budget import BudgetAccountant
from hearth.observability.metrics import MetricsStore
from hearth.providers.base import Capabilities, GenRequest, GenResult, ResourceEstimate
from hearth.providers.echo import EchoProvider
from hearth.router import Router
from hearth.router.classify import TASK_CLASSES
from hearth.router.policy import ClassRule, Defaults, RemoteConfig, RoutingPolicy


class FakeRemote:
    name = "remote"

    def __init__(self, config: RemoteConfig) -> None:
        self.config = config

    def capabilities(self) -> Capabilities:
        return Capabilities(chat=True, stream=True)

    def footprint(self, model_id: str) -> ResourceEstimate:
        return ResourceEstimate()

    def generate(self, req: GenRequest) -> GenResult:
        return GenResult(
            text="[remote] " + req.messages[-1].content,
            model=req.model,
            backend=self.name,
            prompt_tokens=10,
            completion_tokens=20,
        )

    def stream(self, req: GenRequest) -> Iterator[str]:
        yield "[remote] "
        yield req.messages[-1].content


def _make_client(policy: RoutingPolicy, budget_tokens: int = 10_000, tmp_path=None) -> TestClient:
    settings = Settings(backend="echo", home=(tmp_path or "/tmp") / ".hearth", require_auth=False)
    metrics = MetricsStore()
    router = Router(
        local_provider=EchoProvider(),
        policy=policy,
        budget=BudgetAccountant(budget_tokens),
        metrics=metrics,
        remote_factory=FakeRemote,
    )
    app = create_app(provider=EchoProvider(), settings=settings, router=router, metrics=metrics)
    return TestClient(app)


def _local_policy() -> RoutingPolicy:
    return RoutingPolicy(
        defaults=Defaults(),
        classes={c: ClassRule(backend="local", escalate="never") for c in TASK_CLASSES},
        remotes={"default": RemoteConfig(protocol="anthropic", model="fake")},
    )


def _reason_remote_policy() -> RoutingPolicy:
    return RoutingPolicy(
        defaults=Defaults(),
        classes={"reason": ClassRule(backend="remote", escalate="always")},
        remotes={"default": RemoteConfig(protocol="anthropic", model="fake")},
    )


def test_route_dry_run_local(tmp_path):
    client = _make_client(_local_policy(), tmp_path=tmp_path)
    r = client.post(
        "/v1/hearth/route",
        json={"messages": [{"role": "user", "content": "summarize this document"}]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["class"] == "summarize"
    assert body["backend"] == "local"
    assert body["would_escalate"] is False


def test_route_dry_run_escalates(tmp_path):
    client = _make_client(_reason_remote_policy(), tmp_path=tmp_path)
    r = client.post(
        "/v1/hearth/route",
        json={"messages": [{"role": "user", "content": "prove this step by step"}]},
    )
    body = r.json()
    assert body["class"] == "reason"
    assert body["backend"] == "remote"
    assert body["would_escalate"] is True


def test_route_dry_run_does_not_execute(tmp_path):
    # A dry-run must not spend budget or record a request.
    client = _make_client(_reason_remote_policy(), tmp_path=tmp_path)
    client.post(
        "/v1/hearth/route",
        json={"messages": [{"role": "user", "content": "prove this"}]},
    )
    metrics = client.app.state.metrics
    assert metrics.rollup()["requests"] == 0
    assert client.app.state.router.budget.spent() == 0


def test_chat_escalates_through_gateway(tmp_path):
    client = _make_client(_reason_remote_policy(), tmp_path=tmp_path)
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "prove this step by step"}]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"].startswith("[remote]")
    assert body["hearth"]["served_by"] == "remote"
    assert body["hearth"]["escalated"] is True
    assert body["hearth"]["estimated_frontier_tokens_saved"] == 0


def test_streaming_escalates_through_gateway(tmp_path):
    client = _make_client(_reason_remote_policy(), tmp_path=tmp_path)
    r = client.post(
        "/v1/chat/completions",
        json={"stream": True, "messages": [{"role": "user", "content": "prove this step by step"}]},
    )
    assert r.status_code == 200
    assert "[remote]" in r.text
    assert r.text.rstrip().endswith("data: [DONE]")
    # The stream recorded a remote request and billed the budget.
    metrics = client.app.state.metrics
    assert metrics.rollup()["backend_mix"] == {"remote": 1}
    assert client.app.state.router.budget.spent() > 0


def test_metrics_endpoint_reflects_requests(tmp_path):
    client = _make_client(_local_policy(), tmp_path=tmp_path)
    client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "summarize this long document please"}]},
    )
    r = client.get("/v1/hearth/admin/metrics")
    assert r.status_code == 200
    roll = r.json()
    assert roll["requests"] == 1
    assert roll["estimated_frontier_tokens_saved"] > 0
    assert roll["backend_mix"] == {"local": 1}


def test_budget_exhausted_returns_error(tmp_path):
    client = _make_client(_reason_remote_policy(), budget_tokens=0, tmp_path=tmp_path)
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "prove this step by step"}]},
    )
    assert r.status_code == 429
    assert r.json()["error"]["code"] == "hearth.budget.exhausted"


def test_intent_hint_routes(tmp_path):
    client = _make_client(_reason_remote_policy(), tmp_path=tmp_path)
    # An intent hint of "reason" forces the remote class even for plain text.
    r = client.post(
        "/v1/hearth/route",
        json={"messages": [{"role": "user", "content": "whatever"}], "intent": "reason"},
    )
    body = r.json()
    assert body["class"] == "reason"
    assert body["method"] == "intent"
    assert body["would_escalate"] is True
