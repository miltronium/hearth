"""Shared test fixtures."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hearth.config import Settings
from hearth.gateway import create_app
from hearth.providers.echo import EchoProvider


@pytest.fixture
def settings(tmp_path) -> Settings:
    # Force the echo backend and an isolated home so tests never touch ~/.hearth or MLX.
    return Settings(backend="echo", home=tmp_path / ".hearth")


@pytest.fixture
def client(settings) -> TestClient:
    app = create_app(provider=EchoProvider(), settings=settings)
    return TestClient(app)
