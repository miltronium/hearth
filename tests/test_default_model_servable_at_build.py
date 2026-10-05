"""B-104: HEARTH_DEFAULT_MODEL naming a model the backend cannot serve is refused at build.

``check_policy_servable`` (B-065) checked only the rungs a profile NAMES. Under an unpinned
profile every class falls through to the registry default, so ``HEARTH_DEFAULT_MODEL=echo``
under the mlx backend built an app whose every ``auto`` request 404'd — while the same
mistake spelled ``defaults.local_model: echo`` was refused at build. Two spellings of one
configuration, two behaviours. Ports reviewer probe p3.
"""

from __future__ import annotations

import pytest
from test_model_selection import CODER7, SMALL, _app, _settings, fake  # noqa: F401

from hearth.providers import mlx_pool
from hearth.router import Router, RoutingPolicy, RoutingPolicyError
from hearth.router.classify import TASK_CLASSES
from hearth.router.policy import ClassRule, Defaults, get_policy


def _policy(local_model: str = "auto", pinned: dict | None = None) -> RoutingPolicy:
    pinned = pinned or {}
    return RoutingPolicy(
        defaults=Defaults(local_model=local_model),
        classes={
            c: ClassRule(backend="local", escalate="never", local_model=pinned.get(c, "auto"))
            for c in TASK_CLASSES
        },
        remotes={},
    )


@pytest.mark.parametrize("spelling", ["HEARTH_DEFAULT_MODEL", "defaults.local_model"])
def test_both_spellings_of_echo_under_mlx_are_refused_at_build(
    fake, tmp_path, monkeypatch, spelling  # noqa: F811
):
    if spelling == "HEARTH_DEFAULT_MODEL":
        monkeypatch.setenv("HEARTH_DEFAULT_MODEL", "echo")
        policy = _policy()
    else:
        policy = _policy(local_model="echo")
    with pytest.raises(RoutingPolicyError) as caught:
        _app(tmp_path, policy)
    message = str(caught.value)
    assert "'echo'" in message and "mlx" in message
    if spelling == "HEARTH_DEFAULT_MODEL":
        assert "registry default" in message and "class:chat" in message


def test_a_router_loading_its_profile_refuses_it_too(fake, tmp_path, monkeypatch):  # noqa: F811
    """`hearth serve` builds Router() from the profile file: exit 2 there, not a 404 later."""
    monkeypatch.setenv("HEARTH_DEFAULT_MODEL", "echo")
    profile = tmp_path / "routing.yaml"
    profile.write_text("classes: {}\n")
    monkeypatch.setenv("HEARTH_ROUTING_YAML", str(profile))
    get_policy.cache_clear()
    try:
        with pytest.raises(RoutingPolicyError):
            Router(local_provider=mlx_pool(_settings(tmp_path)))
    finally:
        get_policy.cache_clear()


def test_a_default_no_class_falls_through_to_is_not_judged(fake, tmp_path, monkeypatch):  # noqa: F811
    """Every class pinned: the registry default serves nothing, so it cannot block startup."""
    monkeypatch.setenv("HEARTH_DEFAULT_MODEL", "echo")
    _app(tmp_path, _policy(pinned={c: SMALL for c in TASK_CLASSES}))


def test_a_servable_default_still_builds(fake, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setenv("HEARTH_DEFAULT_MODEL", CODER7)
    _app(tmp_path, _policy())
