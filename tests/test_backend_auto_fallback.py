"""``HEARTH_BACKEND=auto`` falling back to the echo stub is loud, and not ready (B-006).

A bare ``uv run`` prunes mlx_lm (CLAUDE.md §1). ``auto`` then silently selected the echo
stub, ``/ready`` said 200, and every answer was an echo labelled with a real model id. The
fallback is kept (the server, its admin surface and non-inference CLI paths still start),
but it is announced with a WARNING and ``/ready`` reports 503 ``stub``. An explicit
``HEARTH_BACKEND=echo`` stays ready, so the probe tells the two apart.
"""

from __future__ import annotations

import importlib.util
import logging

from fastapi.testclient import TestClient

from hearth.config import Settings
from hearth.gateway import create_app
from hearth.providers import select_provider

_real_find_spec = importlib.util.find_spec


def _without_mlx_lm(monkeypatch):
    def find_spec(name, *args, **kwargs):
        if name == "mlx_lm":
            return None
        return _real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)


def _client(tmp_path, backend: str) -> TestClient:
    settings = Settings(backend=backend, home=tmp_path / ".hearth", require_auth=False)
    return TestClient(create_app(settings=settings))


def test_auto_without_mlx_lm_is_a_loud_stub_that_is_not_ready(tmp_path, monkeypatch, caplog):
    _without_mlx_lm(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="hearth.providers"):
        client = _client(tmp_path, "auto")
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("echo STUB" in r.getMessage() and "uv sync --extra mlx" in r.getMessage()
               for r in warnings)
    ready = client.get("/v1/hearth/admin/ready")
    assert ready.status_code == 503
    assert ready.json()["status"] == "stub"
    assert ready.json()["backend"] == "echo"
    assert "mlx_lm" in ready.json()["reason"]
    health = client.get("/v1/hearth/admin/health")
    assert health.status_code == 200  # liveness: the process is up
    assert health.json()["backend"] == "echo"
    assert "echo STUB" in health.json()["backend_fallback"]


def test_an_explicit_echo_backend_stays_ready(tmp_path, monkeypatch, caplog):
    _without_mlx_lm(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="hearth.providers"):
        client = _client(tmp_path, "echo")
    assert not [r for r in caplog.records if r.name == "hearth.providers"]
    ready = client.get("/v1/hearth/admin/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert "backend_fallback" not in client.get("/v1/hearth/admin/health").json()


def test_auto_with_mlx_lm_is_not_the_stub(monkeypatch, tmp_path):
    monkeypatch.setattr("hearth.providers.mlx_available", lambda: True)
    provider = select_provider(Settings(backend="auto", home=tmp_path / ".hearth"))
    assert provider.name == "mlx"
    assert getattr(provider, "fallback_reason", None) is None
