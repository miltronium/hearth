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
from ..providers.base import GenerationCancelledError, GenRequest, GenResult, ModelProvider
from ..providers.echo import is_stub_backend
from ..serving.pool import UnknownModelError, check_model
from .classify import TASK_CLASSES, classify
from .policy import ClassRule, RoutingPolicy, RoutingPolicyError, get_policy

logger = logging.getLogger("hearth.router")

# Escalation reasons (ADR-007). Every escalation records exactly one.
REASON_INTENT = "intent"
REASON_CLASS_POLICY = "class_policy"
REASON_LOW_CONFIDENCE = "low_confidence"
REASON_EXPLICIT = "explicit"
REASON_LOCAL_FAILURE = "local_failure"
# Not an escalation reason: the reason a request that tried to escalate was served LOCAL.
REASON_REMOTE_FAILURE = "remote_failed; served local"


class BudgetExhaustedError(RuntimeError):
    """Raised when escalation is required by policy but the remote budget is exhausted.

    Maps to the API's ``hearth.budget.exhausted`` error (docs/API.md).
    """


class ProviderError(RuntimeError):
    """Raised when the chosen provider fails to load or generate (Phase 7 hardening).

    A provider raising (missing weights, backend crash, remote unreachable) is turned into
    this clean error rather than a bare traceback, so the gateway can return a tidy 503
    envelope instead of a 500. When an *escalation's* remote call fails, the router
    degrades to the local provider (:meth:`Router.degrade_to_local`); this surfaces only
    when the local provider itself fails.
    """


class UnknownAdapterError(LookupError):
    """An explicitly requested adapter id cannot be served (maps to a 404).

    Not registered, retired, or there is no adapter store at all. An explicit
    ``hearth.adapter`` used to fall back to base weights silently while the response named
    the adapter as served — the A/B flow then compared base weights against themselves.
    """

    def __init__(self, adapter_id: str, message: str) -> None:
        super().__init__(message)
        self.adapter_id = adapter_id

    def __str__(self) -> str:
        return str(self.args[0])


@dataclass(frozen=True)
class AdapterChoice:
    """The adapter selected for a local request: its registry ``id`` and on-disk ``path``.

    Both ``None`` means base weights. This is what was *selected*; whether it actually
    served is decided by the generation (an adapter that fails is retried on base).
    """

    id: str | None = None
    path: str | None = None


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
        loaded_from_file = policy is None
        self.policy = policy or get_policy()
        if loaded_from_file:
            # The profile the operator selected, checked against the backend that will
            # actually serve it (B-065). A policy built in code is the caller's own business.
            check_policy_servable(self.policy, local_provider)
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
        local_model = self._local_model(req, rule)

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
        """Decide, execute via the chosen provider, and record telemetry (non-streaming).

        Raises :class:`UnknownAdapterError` before anything runs when ``adapter`` names an
        adapter that cannot be served. The record's ``adapter`` is the adapter that actually
        served (``None`` for base weights, a remote, or an adapter that failed and was
        retried on base) — never merely the one requested.
        """
        try:
            self.check_adapter(adapter)
        except UnknownAdapterError as exc:
            self.record_refused(req, exc, intent=intent, adapter=adapter)  # B-106
            raise
        started = time.perf_counter()
        decision = self.decide(req, intent=intent, allow_escalation=allow_escalation)
        # Records a denied escalation (BudgetExhaustedError) before raising it (B-066).
        provider = self.provider_for(req, decision, started=started)

        # Adapters only layer over the LOCAL backend; resolve the id -> path here so the
        # provider gets a concrete adapter_path to load (hot-swap; ARCHITECTURE §5).
        choice = AdapterChoice()
        if not decision.would_escalate:
            try:
                choice = self.select_adapter(adapter, decision.task_class, decision.model)
            except UnknownAdapterError as exc:  # unservable since check_adapter (B-106)
                self.record_failure(
                    req, decision, provider, exc, started=started, adapter=adapter
                )
                raise

        escalation_failed: str | None = None
        try:
            result, used_path = self._generate(provider, decision, req, choice.path)
        except (UnknownModelError, GenerationCancelledError) as exc:
            # A rung nobody serves (the client gets a 404), or a caller that went away
            # mid-generation (B-071): either way the record says so (B-066).
            self.record_failure(
                req, decision, provider, exc, started=started, adapter=choice.id
            )
            raise
        except ProviderError as exc:
            if not decision.would_escalate:
                # A plain local failure: the client gets a 503, and the record says so.
                self.record_failure(
                    req, decision, provider, exc, started=started, adapter=choice.id
                )
                raise
            # The remote failed (unreachable, offline, SDK missing, rejected the call).
            # Serve the request locally rather than turning a frontier outage into an
            # error: local is always the more private answer. The record says local served
            # AND that an escalation was attempted and failed (``escalation_failed``).
            escalation_failed = str(exc)
            decision = self.degrade_to_local(req, decision, exc)
            choice = AdapterChoice(id=adapter)  # what the record names if selection fails
            provider = self.local
            try:
                choice = self.select_adapter(adapter, decision.task_class, decision.model)
                result, used_path = self._generate(self.local, decision, req, choice.path)
            except (
                ProviderError, UnknownModelError, UnknownAdapterError, GenerationCancelledError
            ) as local_exc:
                # Both failed. The remote was CALLED and may already hold the prompt
                # (docs/PRIVACY.md), so this is exactly the request the audit trail must
                # not lose: record the failed escalation and the failed fallback, re-raise
                # — a 404 from the local rung included (B-066).
                self.record_failure(
                    req, decision, self.local, local_exc, started=started,
                    adapter=choice.id, escalation_failed=escalation_failed,
                )
                raise
        latency_ms = (time.perf_counter() - started) * 1000.0
        served_adapter = self.served_adapter(provider, choice, used_path)

        served_by = "remote" if decision.would_escalate else "local"
        if served_by == "remote":
            self.budget.spend(result.prompt_tokens + result.completion_tokens)
            saved = 0
        elif is_stub_backend(result.backend):
            saved = 0  # an echo replaced no frontier call with inference (B-068)
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
            escalation_failed=escalation_failed,
            adapter=served_adapter,
            estimated_frontier_tokens_saved=saved,
        )
        try:
            self.metrics.record(record)
        except Exception as exc:  # noqa: BLE001 — the answer exists; accounting can't 500 it
            logger.error("request served but could not be recorded: %s", exc)
        return RouteResult(result=result, decision=decision, record=record)

    # -- helpers ----------------------------------------------------------------------

    def provider_for(
        self, req: GenRequest, decision: RouteDecision, *, started: float | None = None
    ) -> ModelProvider:
        """The provider that executes ``decision`` — or :class:`BudgetExhaustedError`.

        Shared by :meth:`route` and the gateway's streaming path, so both apply the same
        escalation and budget rule. A denied escalation (no remote configured, or the budget
        cannot cover the estimate) is RECORDED before it is raised (B-066): it used to reach
        the client as a 429 / stream error event with nothing in the metrics. The record has
        ``failed`` set and ``escalated=False`` — nothing left the machine.
        """
        if not (decision.would_escalate and decision.backend == "remote"):
            return self.local
        remote_cfg = self.policy.remote_for()
        if remote_cfg is None or not self.budget.can_afford(_estimate_remote_cost(req)):
            exc = BudgetExhaustedError(
                "remote budget exhausted; escalation denied"
                if remote_cfg is not None
                else "no remote configured for escalation"
            )
            self.record_failure(
                req, decision, None, exc,
                started=time.perf_counter() if started is None else started,
                backend=remote_cfg.protocol if remote_cfg is not None else "none",
                escalated=False,
            )
            raise exc
        logger.info(
            "escalating class=%s reason=%s model=%s",
            decision.task_class,
            decision.reason,
            decision.model,
        )
        return self._make_remote(remote_cfg)

    def record_failure(
        self,
        req: GenRequest,
        decision: RouteDecision,
        provider: ModelProvider | None,
        exc: Exception,
        *,
        started: float,
        adapter: str | None = None,
        escalation_failed: str | None = None,
        completion_tokens: int = 0,
        backend: str | None = None,
        escalated: bool | None = None,
        model: str | None = None,
    ) -> RequestRecord | None:
        """Record a request that ended in an error instead of an answer (``failed`` set).

        Shared by :meth:`route` and the gateway's streaming path. ``served_by`` names the
        tier that was tried and failed; ``backend_mix`` does not count it (nothing was
        served). ``adapter`` is the adapter SELECTED for the attempt that failed (an explicit
        request, else the promoted default; ``None`` for base weights or a remote) — the
        same meaning on every path (B-073). ``backend`` overrides ``provider.name`` when no
        provider was built (a denied escalation); ``escalated`` overrides
        ``decision.would_escalate``. ``model`` defaults to :func:`generating_model` of the
        provider that was tried — the same identity a success would have reported (B-101).
        Never raises: a metrics store that fails here must not
        replace the provider's error the client is about to receive with its own.
        """
        prompt_tokens = max(1, sum(len(m.content) for m in req.messages) // 4)
        if model is None:
            # The model that was ATTEMPTED — what the provider generates for this request —
            # not what was asked for: an echo stub that fails is "echo", as its answer would
            # have been (B-101). No provider (a denied escalation): the decision's model.
            model = (
                generating_model(provider, decision.model)
                if provider is not None
                else decision.model
            )
        record = RequestRecord(
            task_class=decision.task_class,
            backend=backend if backend is not None else getattr(provider, "name", "none"),
            model=model,
            served_by="remote" if decision.would_escalate else "local",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            escalated=decision.would_escalate if escalated is None else escalated,
            escalation_reason=decision.reason if decision.would_escalate else None,
            escalation_failed=escalation_failed,
            adapter=adapter,
            estimated_frontier_tokens_saved=0,
            failed=str(exc),
        )
        try:
            self.metrics.record(record)
        except Exception as rec_exc:  # noqa: BLE001 — the original error must surface
            logger.error("could not record failed request (%s): %s", exc, rec_exc)
            return None
        return record

    def record_refused(
        self,
        req: GenRequest,
        exc: Exception,
        *,
        intent: str | None = None,
        adapter: str | None = None,
    ) -> RequestRecord | None:
        """Record a request refused before anything ran — an unservable adapter (a 404).

        Such a request used to leave no record at all (B-106): ``hearth stats`` could not
        see a client hammering a retired adapter. The record is attributed to the LOCAL
        decision for the request (adapters only layer over local; nothing left the
        machine), with ``failed`` set. Never raises.
        """
        started = time.perf_counter()
        try:
            decision = self.decide(req, intent=intent, allow_escalation=False)
        except Exception as dec_exc:  # noqa: BLE001 — the refusal itself must still surface
            logger.error("could not record refused request (%s): %s", exc, dec_exc)
            return None
        return self.record_failure(
            req, decision, self.local, exc, started=started, adapter=adapter
        )

    def degrade_to_local(
        self, req: GenRequest, decision: RouteDecision, exc: Exception
    ) -> RouteDecision:
        """The local decision to serve with after an escalation's remote call failed.

        Shared by :meth:`route` and the gateway's streaming path so both degrade the same
        way. The returned decision is a plain local one (``would_escalate=False``) whose
        reason names the failure, so telemetry records local as what served.
        """
        logger.warning(
            "escalation failed (class=%s model=%s); serving locally instead: %s",
            decision.task_class,
            decision.model,
            exc,
        )
        rule = self.policy.rule_for(decision.task_class)
        return RouteDecision(
            task_class=decision.task_class,
            method=decision.method,
            backend="local",
            model=self._local_model(req, rule),
            would_escalate=False,
            reason=f"{REASON_REMOTE_FAILURE}: {exc}",
            confidence=decision.confidence,
        )

    def _generate(
        self,
        provider: ModelProvider,
        decision: RouteDecision,
        req: GenRequest,
        adapter_path: str | None,
    ) -> tuple[GenResult, str | None]:
        """Run ``provider.generate`` with graceful degradation (Phase 7 hardening).

        If generation fails *with* an adapter, retry once on base weights — a bad adapter
        must not sink an otherwise-servable request. If it still fails, wrap the error as a
        :class:`ProviderError` so the gateway returns a clean 503 rather than a 500.

        Returns ``(result, adapter_path_used)``: the path is ``None`` when base weights
        answered, including after a failed adapter was retried on base.
        """
        gen = GenRequest(
            messages=req.messages,
            model=decision.model,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            adapter=adapter_path,
        )
        try:
            return provider.generate(gen), adapter_path
        except (UnknownModelError, GenerationCancelledError):
            # The request named a model nobody serves — the caller's error (a 404) — or the
            # caller went away (B-071). Neither is a provider outage to retry or wrap as 503.
            raise
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
                    ), None
                except (UnknownModelError, GenerationCancelledError):
                    # Same rule as the first attempt: a caller that went away while the
                    # adapter was failing is a cancellation, never a provider outage (503).
                    raise
                except Exception as retry_exc:  # noqa: BLE001
                    exc = retry_exc
            logger.error("provider %s failed to generate: %s", provider.name, exc)
            raise ProviderError(f"provider {provider.name!r} failed: {exc}") from exc

    def _local_model(self, req: GenRequest, rule: ClassRule) -> str:
        """Local model to serve this request with — where the local model *ladder* resolves.

        Resolution order, first hit wins:

          1. an explicit non-``auto`` request model — a client pin always wins;
          2. the class rule's ``local_model`` — the per-class rung (e.g. a small fast model
             for ``classify``/``extract``, a larger one for ``summarize``/``draft``);
          3. ``defaults.local_model`` from the routing policy;
          4. the model registry's default id (``require_default``: an unregistered
             ``HEARTH_DEFAULT_MODEL`` raises instead of falling back, B-070).

        A rule with no ``local_model`` (or ``"auto"``) is a no-op: steps 3–4 decide exactly
        as they did before the field existed, so existing configs are unaffected.
        """
        if req.model and req.model not in ("auto", ""):
            return req.model
        if rule.local_model and rule.local_model != "auto":
            return rule.local_model
        configured = self.policy.defaults.local_model
        if configured and configured != "auto":
            return configured
        from ..registry import get_registry

        # Strict: an unregistered HEARTH_DEFAULT_MODEL raises UnregisteredDefaultModelError
        # rather than routing to the catalog default (B-070).
        return get_registry().require_default()

    def _remote_model(self) -> str:
        cfg = self.policy.remote_for()
        return cfg.model if cfg else self.policy.defaults.remote

    def check_adapter(self, requested: str | None) -> None:
        """Refuse an explicitly requested adapter that cannot be served.

        Raises :class:`UnknownAdapterError` (a 404 at the gateway, like an unknown model)
        when ``requested`` is set and is not a servable registered adapter. Called before
        anything is generated or a stream is opened.
        """
        if not requested:
            return
        store = self._adapter_store()
        if store is None:
            raise UnknownAdapterError(
                requested,
                f"adapter {requested!r} was requested but no adapter registry is available",
            )
        try:
            store.resolve_path(requested, allow_candidate=True)
        except Exception as exc:  # noqa: BLE001 — AdapterError and friends: not servable
            raise UnknownAdapterError(
                requested, f"adapter {requested!r} cannot be served: {exc}"
            ) from exc

    def select_adapter(
        self, requested: str | None, task_class: str, model: str | None = None
    ) -> AdapterChoice:
        """Select the adapter to load for a local request → its id and on-disk path.

        Resolution:
          * an explicit ``requested`` id (``hearth.adapter``) wins — served behind the A/B
            flag even if it's still a candidate (ARCHITECTURE §5). One that cannot be
            served raises :class:`UnknownAdapterError`: an explicit request is never
            quietly answered by base weights;
          * otherwise the promoted adapter for the task class serves by default, **if it was
            trained on the model actually being served**;
          * else base weights (an empty :class:`AdapterChoice`).

        The base-model check matters once a class pins its own ``local_model``: a ladder can
        route a class to a different base than its promoted adapter was tuned on (e.g. a
        7B-trained ``classify`` adapter landing on a 3B rung), and LoRA weights of the wrong
        shape fail at generation time — every request then pays a doomed load plus the
        degrade-and-retry in :meth:`_generate`. Skipping the mismatch keeps that cost off the
        request; an explicitly requested adapter is still honoured, since that is a
        deliberate operator choice. ``model=None`` means "caller didn't say what is serving",
        which skips the check and preserves the pre-ladder behaviour.

        The promoted-default path never raises: a missing store or an unresolvable promoted
        entry degrades to base weights (and the response then reports no adapter).
        """
        if requested:
            self.check_adapter(requested)
            store = self._adapter_store()
            try:
                path = store.resolve_path(requested, allow_candidate=True)
            except Exception as exc:  # noqa: BLE001 — retired/removed since check_adapter
                raise UnknownAdapterError(
                    requested, f"adapter {requested!r} cannot be served: {exc}"
                ) from exc
            return AdapterChoice(id=requested, path=path)
        store = self._adapter_store()
        if store is None:
            return AdapterChoice()
        try:
            promoted = store.promoted_for(task_class)
            if promoted is None:
                return AdapterChoice()
            if model is not None and promoted.base_model and promoted.base_model != model:
                logger.debug(
                    "adapter %s was trained on %s but %s is served by %s; serving base weights",
                    promoted.id,
                    promoted.base_model,
                    task_class,
                    model,
                )
                return AdapterChoice()
            return AdapterChoice(id=promoted.id, path=store.resolve_path(promoted.id))
        except Exception:  # noqa: BLE001 — degrade to base weights; never fail the request
            logger.warning(
                "promoted adapter for %r unresolved; serving base weights", task_class
            )
            return AdapterChoice()

    def _resolve_adapter(
        self, requested: str | None, task_class: str, model: str | None = None
    ) -> str | None:
        """The on-disk path of :meth:`select_adapter`'s choice (``None`` = base weights)."""
        return self.select_adapter(requested, task_class, model).path

    @staticmethod
    def served_adapter(
        provider: ModelProvider, choice: AdapterChoice, used_path: str | None
    ) -> str | None:
        """The adapter id that actually served, for telemetry — or ``None``.

        ``None`` unless the generation really ran with the chosen adapter's path (not
        retried on base) on a provider that applies adapters at all: a provider whose
        capabilities say ``adapters=False`` ignores ``GenRequest.adapter`` (the echo stub),
        so naming an adapter for its answer would be the false claim B-034 was about.
        """
        if choice.id is None or used_path is None or used_path != choice.path:
            return None
        try:
            applies = bool(provider.capabilities().adapters)
        except Exception:  # noqa: BLE001 — a provider that can't say has not shown it applies
            applies = False
        return choice.id if applies else None

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


def generating_model(provider: ModelProvider, model: str) -> str:
    """The model that GENERATES when ``provider`` is handed a request for ``model``.

    Known before generation starts, so a stream can label its role and content chunks with
    it (B-101) instead of the decision's model — which names what was asked for:

    * the echo stub generates as ``"echo"`` whatever was asked (B-068);
    * a provider that resolves ids (:class:`~hearth.serving.ModelPool`) generates the id it
      resolves to (an unresolvable id is left as asked: that request fails anyway);
    * a provider bound to one model (``model_id``) generates that model;
    * anything else (a remote, a test fake) is taken at the requested id.

    The terminal ``StreamDelta.model`` / ``GenResult.model`` stays the authority once the
    generation reports it; this is the same derivation, made before the first token.
    """
    name = getattr(provider, "name", None)
    if is_stub_backend(name):
        return name
    resolve = getattr(provider, "resolve", None)
    if callable(resolve):
        try:
            return resolve(model)
        except Exception:  # noqa: BLE001 — the request itself will fail and say why
            return model
    return getattr(provider, "model_id", None) or model


def policy_rungs(policy: RoutingPolicy, default_id: str) -> dict[str, list[str]]:
    """Every local model an ``auto`` request can be routed to → where each comes from.

    Mirrors :meth:`Router._local_model` for a request naming no model, over every task
    class: the class rung, else ``defaults.local_model``, else ``default_id`` (the registry
    default). A remote class is included too — :meth:`Router.degrade_to_local` serves it
    on the same rung when its escalation fails. Keys are model ids, ordered most-used first
    (ties in task-class order); values name the classes, e.g. ``"class:chat"`` or
    ``"class:chat (defaults.local_model)"``. The registry default appears only if some class
    actually falls through to it — a default the profile never serves is not its model.
    """
    configured = policy.defaults.local_model
    if configured and configured != "auto":
        fallback, how = configured, "defaults.local_model"
    else:
        fallback, how = default_id, "registry default"
    found: dict[str, list[str]] = {}
    for task_class in TASK_CLASSES:
        rung = policy.rule_for(task_class).local_model
        if rung and rung != "auto":
            found.setdefault(rung, []).append(f"class:{task_class}")
        else:
            found.setdefault(fallback, []).append(f"class:{task_class} ({how})")
    order = {model_id: i for i, model_id in enumerate(found)}
    return dict(sorted(found.items(), key=lambda kv: (-len(kv[1]), order[kv[0]])))


def check_policy_servable(policy: RoutingPolicy, local: ModelProvider, registry=None) -> None:
    """Refuse a policy whose model rungs the ACTIVE backend cannot serve (B-065).

    ``load_policy`` already checks each rung is a registered chat model; this adds what
    only the live provider knows — a :class:`~hearth.serving.ModelPool` serves the chat
    models of its own backend, so ``echo`` (or a plugin's model) as a rung under mlx is a
    request-time 404 for every task of that class. Judged with :func:`check_model`, the
    same function that 404s the request, so this check and the request cannot disagree.
    The registry default counts as a rung when some class falls through to it (B-104).
    Raises :class:`RoutingPolicyError` naming every bad rung.
    """
    if registry is None:
        registry = getattr(local, "registry", None)
    if registry is None:
        from ..registry import get_registry

        registry = get_registry()
    named: dict[str, list[str]] = {}
    if policy.defaults.local_model and policy.defaults.local_model != "auto":
        named.setdefault(policy.defaults.local_model, []).append("defaults.local_model")
    for task_class, rule in policy.classes.items():
        if rule.local_model and rule.local_model != "auto":
            named.setdefault(rule.local_model, []).append(f"class {task_class!r}")
    # The registry default an UNPINNED class falls through to is as much a rung as a named
    # one (B-104): HEARTH_DEFAULT_MODEL=echo under mlx used to build an app whose every
    # unpinned request 404'd, while `defaults.local_model: echo` was refused. Which classes
    # fall through is read off policy_rungs — the same derivation readiness and warmup use.
    # Judged only for a provider that validates ids when it serves them (a ModelPool's
    # ``resolve``): any other provider (echo, a plugin, a test fake) answers the default
    # whatever it is, so refusing it would refuse a server whose requests succeed.
    validates_ids = callable(getattr(local, "resolve", None))
    for model_id, sources in policy_rungs(policy, registry.default_id).items():
        through = [s.split(" ", 1)[0] for s in sources if s.endswith("(registry default)")]
        if through and validates_ids:
            named.setdefault(model_id, []).append(
                "the registry default (HEARTH_DEFAULT_MODEL, else the config/models.yaml "
                f"default) that {', '.join(through)} fall through to"
            )
    problems = []
    for model_id, where in named.items():
        try:
            check_model(local, registry, model_id)
        except UnknownModelError as exc:
            problems.append(f"{', '.join(where)} -> {exc}")
    if problems:
        raise RoutingPolicyError(
            f"the routing profile routes to a model the {getattr(local, 'name', '?')!r} backend "
            "cannot serve: " + "; ".join(problems)
        )


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


__all__ = [
    "check_policy_servable",
    "generating_model",
    "policy_rungs",
    "AdapterChoice",
    "BudgetExhaustedError",
    "ProviderError",
    "RouteDecision",
    "RouteResult",
    "Router",
    "UnknownAdapterError",
]
