"""Hardening tests — readiness probe, warmup degradation, provider degradation (Phase 7).

All offline on the echo skeleton plus small fakes. Verifies:
  * ``/v1/hearth/admin/ready`` returns 200 only once the default model is resident; 503
    otherwise.
  * a failing warmup leaves the server up (degraded) instead of crashing.
  * a provider raising on generate yields a clean 503 envelope, not a 500 traceback, and
    a bad adapter degrades to base weights.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from fastapi.testclient import TestClient

from hearth.config import Settings
from hearth.gateway import create_app
from hearth.observability.budget import BudgetAccountant
from hearth.observability.metrics import MetricsStore
from hearth.providers.base import GenResult
from hearth.router import Router, RoutingPolicy
from hearth.router.classify import TASK_CLASSES
from hearth.router.policy import ClassRule, Defaults
from hearth.serving import ModelManager


@dataclass(frozen=True)
class _Caps:
    chat: bool = True
    embed: bool = False
    stream: bool = True
    adapters: bool = True


@dataclass(frozen=True)
class _Estimate:
    ram_gb: float = 1.0
    extra: dict = field(default_factory=dict)


class LoadableProvider:
    """A non-echo provider whose load succeeds; used to exercise warmup/readiness."""

    name = "loadable"

    def __init__(self) -> None:
        self.loaded = False

    def load(self, model_id: str) -> None:
        self.loaded = True

    def unload(self, model_id: str) -> None:
        self.loaded = False

    def capabilities(self) -> _Caps:
        return _Caps()

    def generate(self, req) -> GenResult:
        return GenResult(text="ok", model=req.model, backend=self.name, prompt_tokens=1,
                         completion_tokens=1)

    def stream(self, req) -> Iterator[str]:
        yield "ok"

    def footprint(self, model_id: str) -> _Estimate:
        return _Estimate()


class FailingProvider(LoadableProvider):
    """Loads fine but always fails to generate — exercises graceful degradation."""

    name = "failing"

    def generate(self, req) -> GenResult:
        raise RuntimeError("backend exploded")

    def stream(self, req) -> Iterator[str]:
        raise RuntimeError("backend exploded")


class FailingLoadProvider(LoadableProvider):
    """Fails on load — exercises the warmup degrade path."""

    name = "failload"

    def load(self, model_id: str) -> None:
        raise RuntimeError("cannot load weights")


def _local_policy() -> RoutingPolicy:
    return RoutingPolicy(
        defaults=Defaults(),
        classes={c: ClassRule(backend="local", escalate="never") for c in TASK_CLASSES},
        remotes={},
    )


def _app(provider, settings, *, manager=None, warmup=None):
    policy = _local_policy()
    router = Router(
        local_provider=provider,
        policy=policy,
        budget=BudgetAccountant(policy.defaults.remote_budget_tokens_per_day),
        metrics=MetricsStore(),
    )
    return create_app(provider=provider, settings=settings, router=router, manager=manager)


def test_ready_503_before_warm(tmp_path):
    # A non-echo provider with warmup off: default model not resident → 503.
    settings = Settings(backend="echo", home=tmp_path / ".hearth", require_auth=False, warmup=False)
    provider = LoadableProvider()
    client = TestClient(_app(provider, settings))
    r = client.get("/v1/hearth/admin/ready")
    assert r.status_code == 503
    assert r.json()["status"] == "loading"


def test_ready_200_after_warm(tmp_path):
    settings = Settings(backend="echo", home=tmp_path / ".hearth", require_auth=False, warmup=False)
    provider = LoadableProvider()
    manager = ModelManager(factory=lambda _m: provider, ram_ceiling_gb=10.0)
    client = TestClient(_app(provider, settings, manager=manager))
    # Warm the default model by hand, then the probe flips to 200.
    from hearth.registry import get_registry

    manager.get(get_registry().default_id)
    r = client.get("/v1/hearth/admin/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_echo_always_ready(client):
    # The echo skeleton has nothing to load, so it is ready immediately.
    r = client.get("/v1/hearth/admin/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_warmup_failure_does_not_crash_startup(tmp_path):
    # warmup on + a provider that fails to load → app still builds, /health ok, /ready 503.
    settings = Settings(
        backend="mlx", home=tmp_path / ".hearth", require_auth=False, warmup=True
    )
    provider = FailingLoadProvider()
    client = TestClient(_app(provider, settings))  # must not raise
    assert client.get("/v1/hearth/admin/health").status_code == 200
    assert client.get("/v1/hearth/admin/ready").status_code == 503


def test_provider_generate_failure_returns_clean_503(tmp_path):
    settings = Settings(backend="echo", home=tmp_path / ".hearth", require_auth=False, warmup=False)
    client = TestClient(_app(FailingProvider(), settings))
    r = client.post(
        "/v1/chat/completions",
        json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "hearth.provider.unavailable"


def test_bad_adapter_degrades_to_base(tmp_path):
    """A generate that fails only when an adapter is set retries on base weights."""

    class AdapterSensitiveProvider(LoadableProvider):
        name = "adaptersensitive"

        def generate(self, req) -> GenResult:
            if req.adapter is not None:
                raise RuntimeError("adapter load failed")
            return GenResult(text="base ok", model=req.model, backend=self.name,
                             prompt_tokens=1, completion_tokens=1)

    settings = Settings(backend="echo", home=tmp_path / ".hearth", require_auth=False, warmup=False)
    provider = AdapterSensitiveProvider()
    policy = _local_policy()
    router = Router(
        local_provider=provider,
        policy=policy,
        budget=BudgetAccountant(policy.defaults.remote_budget_tokens_per_day),
        metrics=MetricsStore(),
    )
    client = TestClient(create_app(provider=provider, settings=settings, router=router))
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "hi"}],
            "hearth": {"adapter": "some-adapter-id"},
        },
    )
    # The router can't resolve the id to a path (no store) → base weights → success.
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "base ok"
