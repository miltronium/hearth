"""Router orchestration (ARCHITECTURE §3, ADR-005, ADR-007).

Turns a :class:`GenRequest` into an executed response plus telemetry. The pipeline:

  1. **classify** → task class (intent hint short-circuits; else rules).
  2. **select** backend/model from policy.
  3. **confidence gate** — for ``on_low_confidence`` classes, a heuristic score decides.
  4. **budget gate** — prefer local when remote budget is scarce; deny/serve-local when
     exhausted per the class policy.
  5. **execute** via the chosen provider.
  6. **record** a telemetry :class:`RequestRecord`.

Escalation is always a first-class, logged event carrying a reason (ADR-007).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from ..observability.budget import BudgetAccountant, get_budget
from ..observability.metrics import (
    MetricsStore,
    RequestRecord,
    estimated_tokens_saved,
    get_metrics,
)
from ..providers.base import GenRequest, GenResult, ModelProvider
from .classify import classify
from .policy import RoutingPolicy, get_policy

logger = logging.getLogger("hearth.router")

# Escalation reasons (ADR-007). Every escalation records exactly one.
REASON_INTENT = "intent"
REASON_CLASS_POLICY = "class_policy"
REASON_LOW_CONFIDENCE = "low_confidence"
REASON_EXPLICIT = "explicit"
REASON_LOCAL_FAILURE = "local_failure"


class BudgetExhaustedError(RuntimeError):
    """Raised when escalation is required by policy but the remote budget is exhausted.

    Maps to the API's ``hearth.budget.exhausted`` error (docs/API.md).
    """


class ProviderError(RuntimeError):
    """Raised when the chosen provider fails to load or generate (Phase 7 hardening).

    A provider raising (missing weights, backend crash, remote unreachable) is turned into
    this clean error rather than a bare traceback, so the gateway can return a tidy 503
    envelope instead of a 500. The router first attempts a local degrade (see
    :meth:`Router.route`); this surfaces only when even that fails.
    """


@dataclass(frozen=True)
class RouteDecision:
    """What the router decided (before execution). Surfaced by ``POST /v1/hearth/route``."""

    task_class: str
    method: str  # how the class was resolved: "intent" | "rules"
    backend: str  # "local" | "remote"
    model: str
    would_escalate: bool
    reason: str
    confidence: float | None = None


@dataclass(frozen=True)
class RouteResult:
    """An executed route: the generation plus the recorded telemetry."""

    result: GenResult
    decision: RouteDecision
    record: RequestRecord = field(repr=False)


class Router:
    """Executes the routing policy for each request (ARCHITECTURE §3)."""

    def __init__(
        self,
        local_provider: ModelProvider,
        policy: RoutingPolicy | None = None,
        budget: BudgetAccountant | None = None,
        metrics: MetricsStore | None = None,
        remote_factory=None,
        adapters=None,
    ) -> None:
        self.local = local_provider
        self.policy = policy or get_policy()
        self.budget = budget or get_budget()
        self.metrics = metrics or get_metrics()
        # Injectable so tests supply a fake remote without importing the anthropic SDK.
        # Default resolves lazily to avoid a providers<->router import cycle.
        self._remote_factory = remote_factory
        # Adapter registry for resolving a requested adapter id -> on-disk path (Phase 4).
        # Optional/lazy so the router works with no adapters (the echo skeleton) and tests
        # can inject a fake store.
        self._adapters = adapters

    def _make_remote(self, config) -> ModelProvider:
        """Build a remote provider from config (lazy import breaks the providers cycle)."""
        if self._remote_factory is not None:
            return self._remote_factory(config)
        from ..providers.remote import RemoteProvider

        return RemoteProvider(config)

    # -- decision (no execution) ------------------------------------------------------

    def decide(
        self,
        req: GenRequest,
        intent: str | None = None,
        allow_escalation: bool = True,
    ) -> RouteDecision:
        """Classify + apply policy/confidence/budget gates. Does not execute (dry-run)."""
        task_class, method = classify(req.messages, intent=intent)
        rule = self.policy.rule_for(task_class)
        local_model = self._local_model(req)

        # A class pinned to `remote` (e.g. reason) escalates by class policy.
        base_backend = rule.backend
        escalate = False
        reason = f"class policy: {task_class}->{base_backend}"
        confidence: float | None = None

        if rule.escalate == "always" or base_backend == "remote":
            escalate = True
            reason = REASON_CLASS_POLICY
        elif rule.escalate == "on_low_confidence":
            confidence = _confidence(req, task_class)
            if confidence < rule.threshold:
                escalate = True
                reason = REASON_LOW_CONFIDENCE

        # Explicit client pin overrides everything: hard-local for this call.
        if not allow_escalation and escalate:
            escalate = False
            reason = REASON_EXPLICIT + ": escalation disabled by client (allow_escalation=false)"

        # Budget gate: prefer local when remote budget can't cover an estimated call.
        if escalate and not self.budget.can_afford(_estimate_remote_cost(req)):
            # Budget exhausted. `always`/remote classes must fail closed (API contract);
            # `on_low_confidence` classes gracefully serve local instead.
            if rule.escalate == "always" or base_backend == "remote":
                return RouteDecision(
                    task_class=task_class,
                    method=method,
                    backend="remote",
                    model=self._remote_model(),
                    would_escalate=True,
                    reason="budget exhausted; escalation required by policy",
                    confidence=confidence,
                )
            escalate = False
            reason = "budget scarce; served local instead of escalating"

        if escalate:
            return RouteDecision(
                task_class=task_class,
                method=method,
                backend="remote",
                model=self._remote_model(),
                would_escalate=True,
                reason=reason,
                confidence=confidence,
            )
        return RouteDecision(
            task_class=task_class,
            method=method,
            backend="local",
            model=local_model,
            would_escalate=False,
            reason=reason,
            confidence=confidence,
        )

    # -- execution --------------------------------------------------------------------

    def route(
        self,
        req: GenRequest,
        intent: str | None = None,
        allow_escalation: bool = True,
        adapter: str | None = None,
    ) -> RouteResult:
        """Decide, execute via the chosen provider, and record telemetry (non-streaming)."""
        decision = self.decide(req, intent=intent, allow_escalation=allow_escalation)
        if decision.would_escalate and decision.backend == "remote":
            remote_cfg = self.policy.remote_for()
            if remote_cfg is None or not self.budget.can_afford(_estimate_remote_cost(req)):
                raise BudgetExhaustedError(
                    "remote budget exhausted; escalation denied"
                    if remote_cfg is not None
                    else "no remote configured for escalation"
                )
            logger.info(
                "escalating class=%s reason=%s model=%s",
                decision.task_class,
                decision.reason,
                decision.model,
            )
            provider: ModelProvider = self._make_remote(remote_cfg)
        else:
            provider = self.local

        # Adapters only layer over the LOCAL backend; resolve the id -> path here so the
        # provider gets a concrete adapter_path to load (hot-swap; ARCHITECTURE §5).
        adapter_path = None
        if not decision.would_escalate:
            adapter_path = self._resolve_adapter(adapter, decision.task_class)

        started = time.perf_counter()
        result = self._generate(provider, decision, req, adapter_path)
        latency_ms = (time.perf_counter() - started) * 1000.0

        served_by = "remote" if decision.would_escalate else "local"
        if served_by == "remote":
            self.budget.spend(result.prompt_tokens + result.completion_tokens)
            saved = 0
        else:
            saved = estimated_tokens_saved(
                decision.task_class, result.prompt_tokens, result.completion_tokens
            )

        record = RequestRecord(
            task_class=decision.task_class,
            backend=result.backend,
            model=result.model,
            served_by=served_by,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            latency_ms=latency_ms,
            escalated=decision.would_escalate,
            escalation_reason=decision.reason if decision.would_escalate else None,
            adapter=adapter,
            estimated_frontier_tokens_saved=saved,
        )
        self.metrics.record(record)
        return RouteResult(result=result, decision=decision, record=record)

    # -- helpers ----------------------------------------------------------------------

    def _generate(
        self,
        provider: ModelProvider,
        decision: RouteDecision,
        req: GenRequest,
        adapter_path: str | None,
    ) -> GenResult:
        """Run ``provider.generate`` with graceful degradation (Phase 7 hardening).

        If generation fails *with* an adapter, retry once on base weights — a bad adapter
        must not sink an otherwise-servable request. If it still fails, wrap the error as a
        :class:`ProviderError` so the gateway returns a clean 503 rather than a 500.
        """
        gen = GenRequest(
            messages=req.messages,
            model=decision.model,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            adapter=adapter_path,
        )
        try:
            return provider.generate(gen)
        except Exception as exc:  # noqa: BLE001 — degrade rather than 500 the request
            if adapter_path is not None:
                logger.warning(
                    "generate failed with adapter; retrying on base weights: %s", exc
                )
                try:
                    return provider.generate(
                        GenRequest(
                            messages=req.messages,
                            model=decision.model,
                            max_tokens=req.max_tokens,
                            temperature=req.temperature,
                            adapter=None,
                        )
                    )
                except Exception as retry_exc:  # noqa: BLE001
                    exc = retry_exc
            logger.error("provider %s failed to generate: %s", provider.name, exc)
            raise ProviderError(f"provider {provider.name!r} failed: {exc}") from exc

    def _local_model(self, req: GenRequest) -> str:
        """Local model to serve with: an explicit non-'auto' request model wins over policy.

        Resolution order: explicit request model → policy ``local_model`` → registry default.
        """
        if req.model and req.model not in ("auto", ""):
            return req.model
        configured = self.policy.defaults.local_model
        if configured and configured != "auto":
            return configured
        from ..registry import get_registry

        return get_registry().default_id

    def _remote_model(self) -> str:
        cfg = self.policy.remote_for()
        return cfg.model if cfg else self.policy.defaults.remote

    def _resolve_adapter(self, requested: str | None, task_class: str) -> str | None:
        """Resolve the adapter to actually load for a local request → an on-disk path.

        Resolution:
          * an explicit ``requested`` id (``hearth.adapter``) wins — served behind the A/B
            flag even if it's still a candidate (ARCHITECTURE §5);
          * otherwise the promoted adapter for the task class serves by default, if any;
          * else ``None`` (base weights).

        Returns ``None`` (and never raises) when there's no adapter store or the requested
        id can't be resolved — routing must not fail because an adapter is missing; it
        degrades to the base model.
        """
        store = self._adapter_store()
        if store is None:
            return None
        try:
            if requested:
                return store.resolve_path(requested, allow_candidate=True)
            promoted = store.promoted_for(task_class)
            return store.resolve_path(promoted.id) if promoted else None
        except Exception:  # noqa: BLE001 — degrade to base weights; never fail the request
            logger.warning("adapter %r unresolved; serving base weights", requested)
            return None

    def _adapter_store(self):
        """The adapter store (injected, or lazily the default). ``None`` if unavailable."""
        if self._adapters is not None:
            return self._adapters
        try:
            from ..registry import AdapterStore

            self._adapters = AdapterStore()
        except Exception:  # noqa: BLE001 — no store ⇒ no adapters, base weights only
            return None
        return self._adapters


def _confidence(req: GenRequest, task_class: str) -> float:
    """Heuristic confidence score in [0, 1] — STUB (ARCHITECTURE §3, step 3).

    Phase 2 has no judge model. This proxy: longer, well-formed prompts read as more
    confident local hits; very short/empty prompts score low so they escalate under an
    ``on_low_confidence`` policy. Replace with a lightweight judge/logprob score later.
    """
    last = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    length = len(last.strip())
    if length == 0:
        return 0.0
    # Saturating curve: ~200+ chars → high confidence; a handful of chars → low.
    return min(1.0, 0.4 + length / 300.0)


def _estimate_remote_cost(req: GenRequest) -> int:
    """Rough pre-call token estimate for the budget gate (prompt chars/4 + max_tokens)."""
    prompt_chars = sum(len(m.content) for m in req.messages)
    return max(1, prompt_chars // 4) + req.max_tokens


__all__ = ["Router", "RouteDecision", "RouteResult", "BudgetExhaustedError", "ProviderError"]
