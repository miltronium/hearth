"""Unit tests for the Python HTTP client (Phase 5).

Construction/normalization + header logic that doesn't need a server. End-to-end behavior
against the app (in-process) lives in ``tests/test_conformance.py``.
"""

from __future__ import annotations

from hearth.client import HearthClient


def test_base_url_trims_trailing_v1_and_slash():
    assert HearthClient("http://127.0.0.1:8080/v1").base_url == "http://127.0.0.1:8080"
    assert HearthClient("http://127.0.0.1:8080/").base_url == "http://127.0.0.1:8080"
    assert HearthClient("http://127.0.0.1:8080").base_url == "http://127.0.0.1:8080"


def test_auth_header_only_when_token_given():
    with HearthClient("http://x", token="secret") as with_token:
        assert with_token._client.headers["authorization"] == "Bearer secret"
    with HearthClient("http://x") as no_token:
        assert "authorization" not in no_token._client.headers
