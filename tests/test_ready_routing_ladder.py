"""B-064: /ready judges every model the active routing profile can route ``auto`` to.

Before: readiness judged only the registry default. Under ``config/routing.finance.yaml``
(3B for classify/extract/rank, 14B for everything else) the default 7B is never served, so
with the 14B absent ``/ready`` said 200 while every ``model=auto`` chat 503'd "not on disk"
— and warmup loaded the 7B the profile never serves. The check and the checked thing were
different models (CLAUDE.md §3).

The fake ``mlx_lm`` from test_model_selection records every load, so "warmup loaded X" is
measured, not inferred from a status string.
"""

from __future__ import annotations

from fastapi.testclient import TestClient
from test_model_selection import (  # noqa: F401
    BIG,
    CODER7,
    FINANCE_YAML,
    SMALL,
    _app,
    _chat,
    fake,
    weights,
)

from hearth.providers.base import GenRequest
from hearth.router.classify import TASK_CLASSES
from hearth.router.policy import load_policy

READY = "/v1/hearth/admin/ready"


def _wait_warmup(app) -> None:
    thread = app.state.warmup.thread
    if thread is not None:
        thread.join(10)


def test_finance_profile_with_the_14b_absent_is_not_ready_and_names_it(fake, tmp_path):  # noqa: F811
    fake.missing.add(BIG)
    app, _, _ = _app(tmp_path, load_policy(FINANCE_YAML))
    client = TestClient(app)
    resp = client.get(READY)
    body = resp.json()
    assert resp.status_code == 503, body
    assert body["status"] == "failed"
    assert BIG in body["reason"]
    assert body["models"][BIG]["status"] == "failed"
    assert body["models"][SMALL]["status"] == "ready"
    # ...and that IS the outcome a client sees: an auto chat routed to the 14B rung fails.
    assert _chat(client, "auto", "hello there, tell me something").status_code == 503


def test_finance_profile_with_the_3b_absent_is_not_ready_either(fake, tmp_path):  # noqa: F811
    fake.missing.add(SMALL)
    app, _, _ = _app(tmp_path, load_policy(FINANCE_YAML))
    body = TestClient(app).get(READY).json()
    assert body["status"] == "failed" and SMALL in body["reason"]
    assert "class:classify" in body["models"][SMALL]["serves"]


def test_the_finance_profile_pins_every_class_so_the_default_is_not_judged(fake, tmp_path):  # noqa: F811
    """routing.finance.yaml pins all 9 classes (B-075 pinned `embed` to the tier-1 3B), so
    the registry default is not a model `auto` can reach and readiness does not judge it."""
    app, _, router = _app(tmp_path, load_policy(FINANCE_YAML))
    body = TestClient(app).get(READY).json()
    assert CODER7 not in body["models"], body["models"]
    assert router.decide(GenRequest(messages=[], model="auto"), intent="embed").model == SMALL


def test_a_default_the_profile_never_serves_is_not_judged(fake, tmp_path):  # noqa: F811
    path = tmp_path / "routing.pinned.yaml"
    path.write_text(
        "classes:\n"
        + "".join(
            f"  {c}: {{backend: local, escalate: never, local_model: {BIG}}}\n"
            for c in TASK_CLASSES
        )
    )
    fake.missing.add(CODER7)  # the registry default; every class is pinned elsewhere
    app, _, _ = _app(tmp_path, load_policy(path))
    resp = TestClient(app).get(READY)
    assert resp.status_code == 200, resp.json()
    assert set(resp.json()["models"]) == {BIG}


def test_warmup_loads_the_most_used_rung_first_not_the_unserved_default(fake, tmp_path):  # noqa: F811
    app, _, _ = _app(tmp_path, load_policy(FINANCE_YAML), warmup=True)
    _wait_warmup(app)
    loaded = fake.loaded_paths()
    assert loaded, "warmup loaded nothing"
    # 5 of 9 classes route to the 14B, 3 to the 3B, only `embed` to the registry default.
    # The 7B registry default is not a finance rung (B-075), so warmup never loads it.
    assert loaded == [weights(BIG), weights(SMALL)]
    body = TestClient(app).get(READY).json()
    assert body["status"] == "ready", body
    assert body["model"] == BIG and body["loaded"] is True
    assert body["models"][BIG]["loaded"] is True


def test_warmup_does_not_evict_what_it_already_warmed(fake, tmp_path):  # noqa: F811
    # Room for the 14B (9 GB) but not also the 3B (2 GB) or the 7B (4.5 GB): those are left
    # to load on demand instead of evicting the 14B warmup just loaded.
    app, pool, _ = _app(tmp_path, load_policy(FINANCE_YAML), warmup=True, ram_ceiling_gb=10.0)
    _wait_warmup(app)
    assert fake.loaded_paths() == [weights(BIG)]
    body = TestClient(app).get(READY).json()
    assert body["status"] == "ready", body
    assert body["models"][SMALL]["loaded"] is False
    assert "loads on the first request" in body["models"][SMALL]["detail"]


def test_unpinned_profile_judges_the_registry_default_as_before(fake, tmp_path, local_policy):  # noqa: F811
    fake.missing.add(CODER7)
    app, _, _ = _app(tmp_path, local_policy)
    body = TestClient(app).get(READY).json()
    assert body["status"] == "failed" and body["model"] == CODER7
    assert set(body["models"]) == {CODER7}
