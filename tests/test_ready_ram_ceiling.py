"""B-100: a rung that can never fit under the RAM ceiling makes /ready 503, naming it.

Before: readiness never compared a rung's ``ram_gb`` with ``ram_ceiling_gb``. A profile
routing ``chat`` to the 14B (9 GB) under an 8 GB ceiling said 200 — with warmup off (the 14B
is "on disk, loads on the first request") and with warmup on (warmup "skipped" it as not
fitting beside the rungs already loaded, which readiness also read as fine) — while every
``auto`` chat request 503'd ``needs 9.0 GB > ceiling 8.0 GB``. The probe and the request
judged different things (CLAUDE.md §3). Ports reviewer probes p2/p2b.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from test_model_selection import BIG, SMALL, _app, _chat, fake  # noqa: F401

from hearth.router.classify import TASK_CLASSES
from hearth.router.policy import load_policy
from hearth.serving import ModelManager

READY = "/v1/hearth/admin/ready"


def _chat_on_big(tmp_path):
    lines = "".join(
        f"  {c}: {{backend: local, escalate: never, "
        f"local_model: {BIG if c == 'chat' else SMALL}}}\n"
        for c in TASK_CLASSES
    )
    path = tmp_path / "routing.yaml"
    path.write_text("classes:\n" + lines)
    return load_policy(path)


@pytest.mark.parametrize("warmup", [False, True])
def test_a_rung_over_the_ceiling_is_not_ready_and_is_named(fake, tmp_path, warmup):  # noqa: F811
    app, _, _ = _app(tmp_path, _chat_on_big(tmp_path), ram_ceiling_gb=8.0, warmup=warmup)
    if app.state.warmup.thread is not None:
        app.state.warmup.thread.join(10)
    client = TestClient(app)
    resp = client.get(READY)
    body = resp.json()
    # The outcome readiness must agree with: a chat request routed to the 14B fails.
    chat = _chat(client, "auto", "hi there, let's chat", hearth={"intent": "chat"})
    assert chat.status_code == 503 and "ceiling" in chat.json()["error"]["message"]
    assert resp.status_code == 503, body
    assert body["status"] == "failed"
    assert BIG in body["reason"] and "ceiling" in body["reason"]
    assert body["models"][BIG]["status"] == "failed"
    assert body["models"][BIG]["serves"] == ["class:chat"]
    assert body["models"][SMALL]["status"] == "ready"


def test_the_same_profile_under_a_ceiling_that_fits_is_ready(fake, tmp_path):  # noqa: F811
    app, _, _ = _app(tmp_path, _chat_on_big(tmp_path), ram_ceiling_gb=9.0)
    client = TestClient(app)
    assert client.get(READY).status_code == 200
    assert _chat(client, "auto", "hi there", hearth={"intent": "chat"}).status_code == 200


def test_size_problem_is_the_rule_admission_refuses_on():
    manager = ModelManager(factory=lambda _m: None, ram_ceiling_gb=8.0)
    assert manager.size_problem("m", 8.0) is None  # exactly at the ceiling fits
    assert "ceiling" in manager.size_problem("m", 8.01)
