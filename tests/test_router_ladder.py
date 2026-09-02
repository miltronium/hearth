"""Local model ladder tests — per-class ``local_model`` resolution (ARCHITECTURE §3).

Covers :meth:`hearth.router.route.Router._local_model`: a per-class rung lets one policy
express "small fast model for classify/extract, larger model for summarize/draft" without
touching code. All hermetic — the echo provider serves, no weights are loaded.
"""

from __future__ import annotations

from hearth.observability.budget import BudgetAccountant
from hearth.observability.metrics import MetricsStore
from hearth.providers.base import GenRequest, Message
from hearth.providers.echo import EchoProvider
from hearth.registry import get_registry
from hearth.router import Router
from hearth.router.policy import ClassRule, Defaults, RoutingPolicy

TIER1 = "small-3b"
TIER2 = "big-14b"


def _ladder(default_local: str = "auto") -> RoutingPolicy:
    """A no-egress policy whose classify rung differs from its summarize rung."""
    return RoutingPolicy(
        defaults=Defaults(local_model=default_local, remote="none"),
        classes={
            "classify": ClassRule(backend="local", escalate="never", local_model=TIER1),
            "extract": ClassRule(backend="local", escalate="never", local_model=TIER1),
            "summarize": ClassRule(backend="local", escalate="never", local_model=TIER2),
            "chat": ClassRule(backend="local", escalate="never"),
        },
        remotes={},
    )


def _router(policy: RoutingPolicy) -> Router:
    return Router(
        local_provider=EchoProvider(),
        policy=policy,
        budget=BudgetAccountant(0),
        metrics=MetricsStore(),
    )


def _req(text: str, model: str = "auto") -> GenRequest:
    return GenRequest(messages=[Message(role="user", content=text)], model=model)


def test_per_class_rung_selects_the_model():
    router = _router(_ladder())
    assert router.decide(_req("x", ), intent="classify").model == TIER1
    assert router.decide(_req("x"), intent="extract").model == TIER1
    assert router.decide(_req("x"), intent="summarize").model == TIER2


def test_explicit_request_model_beats_the_class_rung():
    """A client pin is step 1 of the order and outranks policy."""
    router = _router(_ladder())
    decision = router.decide(_req("x", model="pinned-by-client"), intent="classify")
    assert decision.model == "pinned-by-client"


def test_unpinned_class_falls_through_to_defaults():
    """Step 3: a rule with no rung inherits ``defaults.local_model``."""
    router = _router(_ladder(default_local="policy-default"))
    assert router.decide(_req("hello there"), intent="chat").model == "policy-default"
    # …and the pinned classes still win over that default.
    assert router.decide(_req("x"), intent="classify").model == TIER1


def test_unpinned_everywhere_falls_through_to_registry_default():
    """Step 4 (unchanged behaviour): no rung, no policy default → the registry's default."""
    policy = RoutingPolicy(
        defaults=Defaults(remote="none"),
        classes={"classify": ClassRule(backend="local", escalate="never")},
        remotes={},
    )
    router = _router(policy)
    assert router.decide(_req("x"), intent="classify").model == get_registry().default_id


def test_auto_rung_is_treated_as_unset():
    policy = RoutingPolicy(
        defaults=Defaults(local_model="policy-default", remote="none"),
        classes={"classify": ClassRule(backend="local", escalate="never", local_model="auto")},
        remotes={},
    )
    assert _router(policy).decide(_req("x"), intent="classify").model == "policy-default"


def test_ladder_model_reaches_the_provider_and_telemetry():
    """The decided rung is what actually gets generated with, and what gets recorded."""
    router = _router(_ladder())
    routed = router.route(_req("categorize this transaction"), intent="classify")
    assert routed.decision.model == TIER1
    assert routed.result.model == TIER1  # echo echoes back the requested model id
    assert routed.record.model == TIER1
    assert not routed.record.escalated


def test_ladder_does_not_change_escalation_decisions():
    """``local_model`` is orthogonal to backend/escalate — a remote class still escalates."""
    policy = RoutingPolicy(
        defaults=Defaults(),
        classes={"reason": ClassRule(backend="remote", escalate="always", local_model=TIER2)},
        remotes={},
    )
    decision = _router(policy).decide(_req("prove this"), intent="reason")
    assert decision.backend == "remote"
    assert decision.would_escalate
