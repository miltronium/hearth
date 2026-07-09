"""Shared test fixtures."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hearth.config import Settings
from hearth.gateway import create_app
from hearth.providers.echo import EchoProvider


@pytest.fixture
def settings(tmp_path) -> Settings:
    # Force the echo backend, an isolated home, and no auth so tests never touch
    # ~/.hearth, MLX, or need a bearer token. Auth is exercised explicitly elsewhere.
    return Settings(backend="echo", home=tmp_path / ".hearth", require_auth=False)


@pytest.fixture
def client(settings) -> TestClient:
    app = create_app(provider=EchoProvider(), settings=settings)
    return TestClient(app)
