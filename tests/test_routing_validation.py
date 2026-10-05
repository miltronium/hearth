"""B-065: every model rung of a routing profile must be a chat model the backend can serve.

Before: ``defaults.local_model`` was read verbatim and never checked, a class rung could name
an embed model, and ``echo`` passed under mlx. Each loaded, ``/ready`` said 200, and every
``model=auto`` request routed to that rung 404'd. A typo'd *class* rung was rejected, but the
ValueError was swallowed by the safe-defaults fallback, so the profile's whole ladder was
replaced by the registry default with nothing but a log line (CLAUDE.md §3: the server said
it was running the profile; it was running something else).

These assert on the outcome — the policy refuses to load / the app refuses to build — and on
the reviewer's reproduction (a typo'd defaults rung answering /ready 200 then 404).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from test_model_selection import EMBED, FINANCE_YAML, SMALL, _app, _chat, fake  # noqa: F401

from hearth.config import Settings
from hearth.gateway import create_app
from hearth.observability.budget import BudgetAccountant
from hearth.observability.metrics import MetricsStore
from hearth.providers.echo import EchoProvider
from hearth.router import Router, RoutingPolicyError
from hearth.router import policy as policy_mod
from hearth.router.policy import RoutingProfileNotFoundError, load_policy


def _profile(tmp_path, text: str):
    path = tmp_path / "routing.custom.yaml"
    path.write_text(text)
    return path


# -- load_policy: registered, chat-capable ------------------------------------------------


def test_typo_in_defaults_local_model_is_refused_at_load(tmp_path):
    path = _profile(tmp_path, "defaults: {local_model: mlx-community/TYPO}\nclasses: {}\n")
    with pytest.raises(RoutingPolicyError) as excinfo:
        load_policy(path)
    message = str(excinfo.value)
    assert "defaults.local_model" in message
    assert "mlx-community/TYPO" in message
    assert str(path) in message  # names the file, so the operator knows what to fix


def test_typo_in_a_class_rung_is_refused_not_replaced_by_safe_defaults(tmp_path):
    path = _profile(
        tmp_path, "classes:\n  classify: {backend: local, local_model: mlx-community/typo}\n"
    )
    with pytest.raises(RoutingPolicyError, match="class 'classify'"):
        load_policy(path)


def test_an_embed_model_as_a_rung_is_refused(tmp_path):
    path = _profile(
        tmp_path, f"classes:\n  chat: {{backend: local, escalate: never, local_model: {EMBED}}}\n"
    )
    with pytest.raises(RoutingPolicyError, match="not chat-capable"):
        load_policy(path)
    path = _profile(tmp_path, f"defaults: {{local_model: {EMBED}}}\n")
    with pytest.raises(RoutingPolicyError, match="defaults.local_model"):
        load_policy(path)


def test_a_selected_profile_with_a_bad_rung_stops_the_router_from_building(
    tmp_path, monkeypatch
):
    """What ``hearth serve`` hits: ``Router()`` → ``get_policy()`` with the env selection."""
    path = _profile(tmp_path, "defaults: {local_model: mlx-community/TYPO}\n")
    monkeypatch.setenv("HEARTH_ROUTING_YAML", str(path))
    policy_mod.get_policy.cache_clear()
    try:
        with pytest.raises(RoutingPolicyError) as excinfo:
            Router(local_provider=EchoProvider(), metrics=MetricsStore())
    finally:
        policy_mod.get_policy.cache_clear()
    # Every existing "refuse to start on an unusable profile" handler catches it.
    assert isinstance(excinfo.value, RoutingProfileNotFoundError)


def test_auto_and_registered_chat_rungs_still_load(tmp_path):
    path = _profile(
        tmp_path,
        f"defaults: {{local_model: {SMALL}}}\n"
        "classes:\n  chat: {backend: local, escalate: never, local_model: auto}\n",
    )
    policy = load_policy(path)
    assert policy.defaults.local_model == SMALL
    assert policy.rule_for("chat").local_model == "auto"
    assert load_policy(FINANCE_YAML).rule_for("chat").local_model  # bundled ladder loads


def test_a_structurally_broken_file_still_degrades_to_safe_defaults(tmp_path):
    """ADR-005 is unchanged for a file that does not parse into a policy at all."""
    path = _profile(tmp_path, "classes:\n  chat: {backend: bogus}\n")
    assert load_policy(path).rule_for("chat").backend == "local"


# -- the active backend: echo under mlx -----------------------------------------------------


def test_echo_as_a_rung_under_mlx_refuses_to_build_the_app(fake, tmp_path):  # noqa: F811
    path = _profile(tmp_path, "classes:\n  code: {backend: local, local_model: echo}\n")
    policy = load_policy(path)  # echo IS a registered chat model: only the backend says no
    with pytest.raises(RoutingPolicyError) as excinfo:
        _app(tmp_path, policy)
    assert "class 'code'" in str(excinfo.value)
    assert "'echo'" in str(excinfo.value)


def test_echo_as_a_rung_under_mlx_refuses_to_build_a_router_from_the_profile(
    fake, tmp_path, monkeypatch  # noqa: F811
):
    from hearth.providers import mlx_pool

    path = _profile(tmp_path, "defaults: {local_model: echo}\n")
    monkeypatch.setenv("HEARTH_ROUTING_YAML", str(path))
    policy_mod.get_policy.cache_clear()
    try:
        with pytest.raises(RoutingPolicyError, match="defaults.local_model"):
            Router(
                local_provider=mlx_pool(Settings(backend="mlx", home=tmp_path / "h")),
                metrics=MetricsStore(),
            )
    finally:
        policy_mod.get_policy.cache_clear()


def test_echo_backend_serves_any_registered_chat_rung(tmp_path):
    """Under the echo backend every registered chat id is answered by the stub: no refusal."""
    policy = load_policy(FINANCE_YAML)
    settings = Settings(backend="echo", home=tmp_path / ".hearth", require_auth=False)
    router = Router(
        local_provider=EchoProvider(), policy=policy, budget=BudgetAccountant(0),
        metrics=MetricsStore(),
    )
    app = create_app(provider=router.local, settings=settings, router=router)
    assert TestClient(app).get("/v1/hearth/admin/ready").status_code == 200


def test_reviewer_repro_typoed_defaults_no_longer_reaches_ready_200(fake, tmp_path):  # noqa: F811
    """Was: policy loads, /ready 200, model=auto → 404 with no record. Now: refused at load."""
    path = _profile(tmp_path, "defaults: {local_model: mlx-community/TYPO}\nclasses: {}\n")
    with pytest.raises(RoutingPolicyError):
        policy = load_policy(path)
        app, _, _ = _app(tmp_path, policy)
        client = TestClient(app)
        assert client.get("/v1/hearth/admin/ready").status_code == 200
        assert _chat(client, "auto").status_code == 404
