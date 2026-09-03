"""The ``/chat`` UI: served, self-contained, and not a hole in the auth surface.

The self-containment tests here assert on the *bytes the route serves*, not on "we did
not add a CDN link" — a configuration claim that would stay green if a font import crept
back in (CLAUDE.md §3). One external font request from a page discussing the operator's
bank statements is an egress channel and would falsify the project's central guarantee.
"""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from hearth.config import Settings, get_or_create_token
from hearth.gateway import create_app
from hearth.gateway.chat_ui import chat_ui_html
from hearth.providers.echo import EchoProvider

# Any attribute that could pull a byte from somewhere else.
_URL_ATTRS = re.compile(r"""\b(?:src|href|action|data|poster|srcset)\s*=\s*["']([^"']*)["']""")
# CSS's own fetch verbs.
_CSS_FETCH = re.compile(r"""(?:url\s*\(|@import)""", re.IGNORECASE)


def _auth_client(tmp_path) -> tuple[TestClient, Settings]:
    """A client with auth ON — the posture `hearth serve` actually runs in."""
    settings = Settings(backend="echo", home=tmp_path / ".hearth", require_auth=True)
    app = create_app(provider=EchoProvider(), settings=settings)
    return TestClient(app), settings


def test_chat_route_serves_html(client):
    r = client.get("/chat")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    body = r.text
    assert body.lstrip().lower().startswith("<!doctype html>")
    assert "</html>" in body


def test_chat_page_makes_no_external_requests(client):
    """Zero off-origin references in the served bytes: no CDN, no fonts, no images."""
    html = client.get("/chat").text
    lowered = html.lower()
    for scheme in ("http://", "https://", "ftp://", "ws://", "wss://"):
        assert scheme not in lowered, f"served page contains an absolute {scheme} URL"

    for url in _URL_ATTRS.findall(html):
        assert not url.startswith("//"), f"protocol-relative reference: {url!r}"
        assert url.startswith(("/", "#")), f"off-origin or ambiguous reference: {url!r}"

    # No external subresource elements at all, and no CSS-level fetching.
    assert "<script src" not in lowered
    assert "<link" not in lowered
    assert "<img" not in lowered
    assert "<iframe" not in lowered
    assert not _CSS_FETCH.search(html), "page uses url()/@import, which can fetch off-origin"


def test_chat_page_calls_only_same_origin_paths():
    """Every fetch target is a relative path on this gateway."""
    html = chat_ui_html()
    targets = re.findall(r"""fetch\(\s*["']([^"']+)["']""", html)
    assert targets, "expected the page to call the gateway"
    assert set(targets) == {"/v1/models", "/v1/chat/completions"}


def test_chat_page_streams_and_surfaces_truncation():
    """The UI must consume SSE incrementally and report a `length` finish."""
    html = chat_ui_html()
    assert "getReader" in html and "TextDecoder" in html  # incremental, not await-the-body
    assert "[DONE]" in html
    assert "finish_reason" in html
    assert 'finish === "length"' in html
    assert "max_tokens" in html
    # Telemetry the operator is meant to see at a glance.
    assert "served_by" in html
    assert "served on-device" in html
    assert "hearth.model" in html
    assert "hearth.backend" in html
    # Error paths are rendered, not hung on.
    assert "401" in html and "422" in html


def test_chat_route_needs_no_token_and_leaks_none(tmp_path):
    """`/chat` is reachable unauthenticated *because* it carries no credential."""
    auth_client, settings = _auth_client(tmp_path)
    token = get_or_create_token(settings)

    r = auth_client.get("/chat")
    assert r.status_code == 200

    # The chosen design's whole justification: an unauthenticated document is only safe
    # if it contains no secret. Assert the outcome, not the intent.
    assert token not in r.text
    assert settings.token_path.read_text().strip() == token
    body = r.text.lower()
    assert "bearer " + token.lower() not in body

    # And it reads the same with a valid token: no server-side injection path exists.
    with_token = auth_client.get("/chat", headers={"Authorization": f"Bearer {token}"})
    assert with_token.status_code == 200
    assert with_token.text == r.text


def test_v1_routes_still_require_a_token_after_the_ui_lands(tmp_path):
    """Adding /chat must not have loosened auth on any data-bearing route."""
    auth_client, settings = _auth_client(tmp_path)
    token = get_or_create_token(settings)

    assert auth_client.get("/v1/models").status_code == 401
    assert (
        auth_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        ).status_code
        == 401
    )
    assert auth_client.get("/v1/hearth/admin/metrics").status_code == 401

    headers = {"Authorization": f"Bearer {token}"}
    assert auth_client.get("/v1/models", headers=headers).status_code == 200
    # allow_escalation=False pins this to the echo backend so the assertion is about auth
    # and not about whichever remote the default policy would otherwise reach for.
    ok = auth_client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "hearth": {"allow_escalation": False},
        },
        headers=headers,
    )
    assert ok.status_code == 200


def test_no_cors_middleware_was_added(client):
    """Same-origin is the reason this is safe; a CORS grant would undo it."""
    r = client.options(
        "/v1/chat/completions",
        headers={
            "Origin": "null",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}


def test_the_bare_origin_redirects_to_the_chat_page(client):
    """Opening http://127.0.0.1:8080 should land on the UI, not a 404.

    Pinned because it did not, the first time this shipped: the operator started the daemon,
    opened the root, and got two 404s that read as a broken server rather than a wrong path.
    """
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/chat"


def test_the_redirect_does_not_bypass_auth_on_the_api(tmp_path):
    """The redirect is navigational only — /v1/* keeps its token requirement.

    Uses the auth-ON client deliberately: the shared `client` fixture runs with auth off, so
    asserting against it would have passed without proving anything.
    """
    auth_client, _ = _auth_client(tmp_path)
    assert auth_client.get("/", follow_redirects=False).status_code == 307
    assert auth_client.post("/v1/chat/completions", json={"messages": []}).status_code == 401
