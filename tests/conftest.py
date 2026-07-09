"""Shared test fixtures."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hearth.config import Settings
from hearth.gateway import create_app
from hearth.observability.budget import BudgetAccountant
from hearth.observability.metrics import MetricsStore
from hearth.providers.echo import EchoProvider
from hearth.router import Router, RoutingPolicy
from hearth.router.classify import TASK_CLASSES
from hearth.router.policy import ClassRule, Defaults


@pytest.fixture
def settings(tmp_path) -> Settings:
    # Force the echo backend, an isolated home, and no auth so tests never touch
    # ~/.hearth, MLX, or need a bearer token. Auth is exercised explicitly elsewhere.
    return Settings(backend="echo", home=tmp_path / ".hearth", require_auth=False)


@pytest.fixture
def local_policy() -> RoutingPolicy:
    """A policy that keeps every class local — keeps the walking skeleton deterministic."""
    return RoutingPolicy(
        defaults=Defaults(),
        classes={c: ClassRule(backend="local", escalate="never") for c in TASK_CLASSES},
        remotes={},
    )


@pytest.fixture
def client(settings, local_policy) -> TestClient:
    provider = EchoProvider()
    router = Router(
        local_provider=provider,
        policy=local_policy,
        budget=BudgetAccountant(local_policy.defaults.remote_budget_tokens_per_day),
        metrics=MetricsStore(),
    )
    app = create_app(provider=provider, settings=settings, router=router)
    return TestClient(app)
