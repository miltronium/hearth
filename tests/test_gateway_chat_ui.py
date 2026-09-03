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
    """Every fetch target is a relative path on this gateway.

    An exact set, not a subset: a new endpoint appearing in this page is a decision, and it
    should have to be made here rather than slipping in under a looser assertion.
    """
    html = chat_ui_html()
    targets = re.findall(r"""fetch\(\s*["']([^"']+)["']""", html)
    assert targets, "expected the page to call the gateway"
    assert set(targets) == {"/v1/models", "/v1/chat/completions", "/v1/hearth/agent"}


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


# -- agent mode ---------------------------------------------------------------------------


def test_agent_mode_defaults_to_off_in_the_served_markup(client):
    """The toggle ships unchecked, and a browser with no stored preference gets plain chat.

    Asserted on the served bytes rather than on "we didn't set it": the default is what the
    document says, and the document is what an operator's first visit renders. Both halves
    matter — an unchecked box that a script ticks on load is still on by default.
    """
    html = client.get("/chat").text
    tag = re.search(r"<input[^>]*id=\"agentmode\"[^>]*>", html)
    assert tag, "the agent-mode toggle is missing from the page"
    assert "checked" not in tag.group(0), "agent mode must ship OFF"

    # And the stored preference has to say "on" explicitly; anything else (absent, corrupt,
    # localStorage unavailable) falls to the safe side.
    assert 'window.localStorage.getItem(AGENT_KEY) === "on"' in html
    assert re.search(r"catch \(e\) \{ return false; \}", html)


def test_agent_mode_off_touches_only_chat_completions():
    """With the toggle off the page is the page that shipped before agent mode existed."""
    html = chat_ui_html()
    # The single branch: nothing agent-shaped runs unless agentOn() is true.
    assert "if (agentOn()) { runAgent(text); return; }" in html
    # /v1/hearth/agent is reachable from exactly one function, and that function is the
    # branch above — not from send(), not from loadModels().
    assert html.count('fetch("/v1/hearth/agent"') == 1


def test_agent_mode_renders_every_step_in_the_conversation():
    """Tool, arguments and observation, per step — the operator sees which files opened."""
    html = chat_ui_html()
    assert "hearth.agent.step" in html
    assert "stepTurn" in html
    assert 'ev.tool + "(" + JSON.stringify(ev.arguments || {}) + ")"' in html
    assert "ev.observation" in html
    assert "observation_truncated" in html
    # File content is untrusted input: it must never become markup. The assertion is on
    # assignment, not on the word — the module comment explaining the rule mentions it.
    assert not re.search(r"\.(?:innerHTML|outerHTML)\s*=", html)
    assert "insertAdjacentHTML" not in html
    assert "document.write" not in html


def test_a_run_that_stopped_short_never_renders_as_an_answer():
    """The `finish_reason` lesson at the presentation layer (CLAUDE.md §3).

    `AgentRun` refuses to carry an answer it did not earn; the last place that guarantee can
    be thrown away is the page that draws it, so the page must gate on `completed` and say
    plainly that a partial trace is not a result.
    """
    html = chat_ui_html()
    assert "if (ev.completed && ev.answer)" in html
    assert "NOT AN ANSWER" in html
    assert "partial trace, not a result" in html
    assert "stopped: " in html  # the stop reason is rendered, every run
    # A stream that dies before the terminal event is reported, not read as an empty answer.
    assert "without a stop reason" in html


def test_agent_mode_warns_about_prompt_injection_where_the_operator_will_see_it():
    """The note is in the UI, next to the toggle — not buried in a doc nobody opens.

    Bounded, not eliminated, is the honest framing and the wording has to say both halves:
    a warning that only lists mitigations reads as an all-clear.
    """
    html = chat_ui_html()
    assert "content becomes model input" in html
    assert "shaped like an instruction can steer the run" in html
    assert "they do not eliminate it" in html
    assert "read-only, no shell, no writes, no network" in html
    # It is attached to agent mode, so it appears exactly when the risk does.
    assert 'class="agentbar"' in html
    assert "body.agent .agentbar" in html


def test_the_agent_route_is_reachable_from_the_page_and_still_needs_a_token(tmp_path):
    """The UI's new endpoint is authenticated like the rest of /v1 — the page is not a way in."""
    auth_client, settings = _auth_client(tmp_path)
    token = get_or_create_token(settings)
    assert (
        auth_client.post("/v1/hearth/agent", json={"task": "read my files"}).status_code == 401
    )
    ok = auth_client.post(
        "/v1/hearth/agent",
        json={"task": "read my files"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert ok.status_code == 200


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
