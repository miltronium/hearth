"""docs/API.md and the FastAPI app agree on which endpoints exist (B-037).

API.md used to document ``/v1/hearth/classify``, ``/summarize``, ``/train/*`` and admin
adapter/model lifecycle endpoints that no route served — integrators would build against
404s. This test reads every backticked ``METHOD /path`` in API.md and asserts on the app's
real routes (``app.routes``), both ways:

* every endpoint API.md documents is served, with that method;
* every ``/v1/...`` route the app serves is documented;
* the endpoints listed under the ``### Not implemented`` heading (each marked
  ``(not implemented)``) are really absent, so the list cannot rot either.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

API_MD = Path(__file__).resolve().parent.parent / "docs" / "API.md"
_ENDPOINT = re.compile(r"`(GET|POST|PUT|PATCH|DELETE) (/[^`\s]*)`")
NOT_IMPLEMENTED_HEADING = "### Not implemented"
MARKER = "(not implemented)"


def _norm(path: str) -> str:
    """Path parameters compare by position, not by name: ``{id}`` == ``{model_id}``."""
    return re.sub(r"\{[^}]*\}", "{}", path.split("?")[0])


def _documented() -> tuple[set[tuple[str, str]], set[tuple[str, str]], str]:
    text = API_MD.read_text(encoding="utf-8")
    assert NOT_IMPLEMENTED_HEADING in text
    head, rest = text.split(NOT_IMPLEMENTED_HEADING, 1)
    nxt = re.search(r"^#{1,3} ", rest, flags=re.MULTILINE)
    missing_section = rest[: nxt.start()] if nxt else rest
    documented_text = head + (rest[nxt.start():] if nxt else "")
    real = {(m, _norm(p)) for m, p in _ENDPOINT.findall(documented_text)}
    absent = {(m, _norm(p)) for m, p in _ENDPOINT.findall(missing_section)}
    return real, absent, missing_section


@pytest.fixture(scope="module")
def served(tmp_path_factory) -> set[tuple[str, str]]:
    import os

    from hearth.config import Settings
    from hearth.gateway import create_app
    from hearth.providers.echo import EchoProvider

    home = tmp_path_factory.mktemp("home")
    old = os.environ.get("HEARTH_HOME")
    os.environ["HEARTH_HOME"] = str(home)
    try:
        app = create_app(provider=EchoProvider(), settings=Settings(home=home, backend="echo"))
    finally:
        if old is None:
            os.environ.pop("HEARTH_HOME", None)
        else:
            os.environ["HEARTH_HOME"] = old
    routes = set()
    for route in app.routes:
        for method in getattr(route, "methods", None) or ():
            if method != "HEAD":
                routes.add((method, _norm(route.path)))
    return routes


def test_the_doc_parse_found_the_endpoints():
    """Guard the guard: a parse that found nothing would pass everything below."""
    real, absent, _ = _documented()
    assert ("POST", "/v1/chat/completions") in real
    assert ("GET", "/v1/hearth/admin/ready") in real
    assert len(real) >= 10 and len(absent) >= 5


def test_every_documented_endpoint_is_served(served):
    real, _, _ = _documented()
    missing = sorted(real - served)
    assert not missing, f"docs/API.md documents endpoints the app does not serve: {missing}"


def test_every_served_v1_route_is_documented(served):
    real, _, _ = _documented()
    undocumented = sorted(r for r in served - real if r[1].startswith("/v1/"))
    assert not undocumented, f"routes missing from docs/API.md: {undocumented}"


def test_the_not_implemented_list_is_really_not_served(served):
    _, absent, section = _documented()
    wrongly = sorted(absent & served)
    assert not wrongly, f"listed as not implemented but served: {wrongly}"
    items = [i for i in re.split(r"\n(?=- )", section) if _ENDPOINT.search(i)]
    assert items and all(MARKER in i for i in items), "each item needs the marker"


def test_the_documented_401_envelope_is_the_real_one(tmp_path):
    """API.md said a 401 used the top-level ``{"error": ...}`` envelope; FastAPI's
    HTTPException nests it under ``detail`` (gateway/auth.py)."""
    import json

    from fastapi.testclient import TestClient

    from hearth.config import Settings
    from hearth.gateway import create_app
    from hearth.providers.echo import EchoProvider

    settings = Settings(home=tmp_path, backend="echo")
    client = TestClient(create_app(provider=EchoProvider(), settings=settings))
    resp = client.get("/v1/models")
    assert resp.status_code == 401
    body = resp.json()
    assert set(body) == {"detail"}
    assert body["detail"]["error"]["code"] == "hearth.auth.unauthorized"
    text = API_MD.read_text(encoding="utf-8")
    block = text.split("**401**", 1)[1].split("```jsonc", 1)[1].split("```", 1)[0]
    assert json.loads(block) == body
