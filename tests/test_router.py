"""Router orchestration tests (ARCHITECTURE §3, ADR-007).

Exercise routing decisions, confidence + budget gating, and escalation to a FAKE remote
provider — never a real network call (constraint: api.anthropic.com is blocked here).
"""

from __future__ import annotations

from collections.abc import Iterator

from hearth.observability.budget import BudgetAccountant
from hearth.observability.metrics import MetricsStore
from hearth.providers.base import Capabilities, GenRequest, GenResult, Message, ResourceEstimate
from hearth.providers.echo import EchoProvider
from hearth.router import Router
from hearth.router.policy import ClassRule, Defaults, RemoteConfig, RoutingPolicy


class FakeRemote:
    """A stand-in for RemoteProvider — records that it was called, no network."""

    name = "remote"

    def __init__(self, config: RemoteConfig) -> None:
        self.config = config

    def capabilities(self) -> Capabilities:
        return Capabilities(chat=True, stream=True)

    def footprint(self, model_id: str) -> ResourceEstimate:
        return ResourceEstimate()

    def generate(self, req: GenRequest) -> GenResult:
        return GenResult(
            text="[remote] " + req.messages[-1].content,
            model=req.model,
            backend=self.name,
            prompt_tokens=10,
            completion_tokens=20,
        )

    def stream(self, req: GenRequest) -> Iterator[str]:
        yield self.generate(req).text


def _policy(**class_rules) -> RoutingPolicy:
    remotes = {"default": RemoteConfig(protocol="anthropic", model="fake-model")}
    return RoutingPolicy(defaults=Defaults(), classes=class_rules, remotes=remotes)


def _router(policy: RoutingPolicy, budget_tokens: int = 10_000) -> Router:
    return Router(
        local_provider=EchoProvider(),
        policy=policy,
        budget=BudgetAccountant(budget_tokens),
        metrics=MetricsStore(),
        remote_factory=FakeRemote,
    )


def _req(text: str) -> GenRequest:
    return GenRequest(messages=[Message(role="user", content=text)], model="auto")


def test_local_class_serves_local():
    policy = _policy(summarize=ClassRule(backend="local", escalate="never"))
    router = _router(policy)
    routed = router.route(_req("summarize this long and detailed document please"))
    assert routed.decision.backend == "local"
    assert routed.result.backend == "echo"
    assert not routed.record.escalated
    assert routed.record.estimated_frontier_tokens_saved > 0


def test_always_class_escalates_to_remote():
    policy = _policy(reason=ClassRule(backend="remote", escalate="always"))
    router = _router(policy)
    routed = router.route(_req("prove that P equals NP step by step"))
    assert routed.decision.backend == "remote"
    assert routed.record.escalated
    assert routed.record.escalation_reason == "class_policy"
    assert routed.result.text.startswith("[remote]")
    # Remote spend is billed; local savings are not counted for remote hits.
    assert routed.record.estimated_frontier_tokens_saved == 0


def test_low_confidence_escalates_short_prompt():
    # chat with a high threshold: a very short prompt scores low -> escalate.
    policy = _policy(chat=ClassRule(backend="local", escalate="on_low_confidence", threshold=0.9))
    router = _router(policy)
    routed = router.route(_req("hi"))
    assert routed.decision.backend == "remote"
    assert routed.record.escalation_reason == "low_confidence"


def test_high_confidence_stays_local():
    policy = _policy(chat=ClassRule(backend="local", escalate="on_low_confidence", threshold=0.5))
    router = _router(policy)
    routed = router.route(_req("here is a nicely detailed message " * 5))
    assert routed.decision.backend == "local"
    assert not routed.record.escalated


def test_allow_escalation_false_pins_local():
    policy = _policy(reason=ClassRule(backend="remote", escalate="always"))
    router = _router(policy)
    routed = router.route(_req("reason about this"), allow_escalation=False)
    assert routed.decision.backend == "local"
    assert not routed.record.escalated


def test_budget_scarce_on_low_confidence_serves_local():
    # on_low_confidence class would escalate, but no budget -> gracefully serve local.
    policy = _policy(chat=ClassRule(backend="local", escalate="on_low_confidence", threshold=0.9))
    router = _router(policy, budget_tokens=0)
    routed = router.route(_req("hi"))
    assert routed.decision.backend == "local"
    assert not routed.record.escalated


def test_budget_exhausted_on_always_class_raises():
    from hearth.router import BudgetExhaustedError

    policy = _policy(reason=ClassRule(backend="remote", escalate="always"))
    router = _router(policy, budget_tokens=0)
    try:
        router.route(_req("reason about the universe"))
    except BudgetExhaustedError:
        pass
    else:
        raise AssertionError("expected BudgetExhaustedError when budget is exhausted")


def test_remote_spend_is_billed_to_budget():
    policy = _policy(reason=ClassRule(backend="remote", escalate="always"))
    router = _router(policy, budget_tokens=10_000)
    router.route(_req("reason about this"))
    # FakeRemote reports 10 + 20 tokens.
    assert router.budget.spent() == 30
