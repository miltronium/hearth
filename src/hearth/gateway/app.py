"""FastAPI application factory and routes.

Serves the OpenAI-compatible core (``/v1/chat/completions`` incl. streaming,
``/v1/embeddings`` stub, ``/v1/models``) plus the ``/v1/hearth/`` extension + admin
surface. From Phase 2, chat completions go through the :class:`~hearth.router.Router`
(classify → select → gate → execute → record) rather than calling a provider directly.
A local bearer token (see :mod:`hearth.gateway.auth`) gates every route except the
liveness probe ``/v1/hearth/admin/health``.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field

from fastapi import Depends, FastAPI, Query
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import iterate_in_threadpool

from .. import __version__
from ..config import Settings, get_settings
from ..memory import RagIndex, select_embedder, select_vector_store
from ..observability.metrics import (
    MetricsStore,
    RequestRecord,
    estimated_tokens_saved,
    get_metrics,
)
from ..providers import select_provider
from ..providers.base import GenRequest, Message, ModelProvider, iter_stream
from ..providers.echo import is_stub_backend
from ..registry import Registry, get_registry
from ..router import BudgetExhaustedError, ProviderError, Router, check_policy_servable
from ..router.route import (
    AdapterChoice,
    UnknownAdapterError,
    generating_model,
    policy_rungs,
)
from ..serving import ModelManager, UnknownModelError, check_model, servable_for
from .agent_route import register_agent_route
from .auth import require_token
from .chat_ui import register_chat_ui
from .json_mode import (
    InvalidJsonResponseError,
    UnsupportedResponseFormatError,
    json_instruction,
    parse_json_object,
    resolve_format,
)
from .schemas import (
    ChatChoice,
    ChatChoiceMessage,
    ChatChunkChoice,
    ChatChunkDelta,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    HearthTelemetry,
    ModelCard,
    ModelList,
    RagChunk,
    RagIngestRequest,
    RagIngestResponse,
    RagQueryRequest,
    RagQueryResponse,
    RouteRequest,
    RouteResponse,
    Usage,
)

logger = logging.getLogger("hearth.gateway")


def create_app(
    provider: ModelProvider | None = None,
    settings: Settings | None = None,
    registry: Registry | None = None,
    router: Router | None = None,
    metrics: MetricsStore | None = None,
    rag: RagIndex | None = None,
    manager: ModelManager | None = None,
) -> FastAPI:
    """Build the HEARTH FastAPI app. Pass ``provider``/``router`` to inject stubs in tests."""
    settings = settings or get_settings()
    provider = provider or select_provider(settings)
    registry = registry or get_registry()
    # B-047: an explicitly set HEARTH_DEFAULT_MODEL that names no registered model is an
    # error when the app is BUILT, not a readiness footnote. Before, an app built directly
    # (not via `hearth serve`, which already refused) reported /ready "failed" while
    # model=auto was answered 200 by the catalog default — one server, two answers.
    registry.require_default()
    metrics = metrics or get_metrics()
    router = router or Router(local_provider=provider, metrics=metrics)
    # B-065: every model rung of the policy this app routes with must be servable by the
    # backend that serves it — checked whether the router was built here or injected, so a
    # ladder naming echo (or a plugin's model) under mlx refuses to start instead of 404ing
    # every request of that class.
    check_policy_servable(router.policy, router.local, registry)
    # RAG defaults to the offline embedder + SQLite store (rooted at settings.home/rag so
    # tests stay isolated); the index reuses the router so `answer=True` runs the local
    # model (allow_escalation=False). Injectable for tests.
    rag = rag or RagIndex(
        embedder=select_embedder(settings),
        store=select_vector_store(settings),
        router=router,
    )
    # Memory-aware residency (Phase 7): the manager owns which models are loaded within the
    # RAM ceiling. The MLX backend is a ModelPool, which brings its own manager holding one
    # provider per model id — readiness must look at THAT manager, the one requests are
    # served from. Any other provider (echo, a plugin, a test fake) serves every id itself,
    # so a manager over it is only the residency record for warmup/readiness. A pool's own
    # manager always wins, even over an injected one: readiness that inspected any other
    # manager would be checking a different object from the one that serves (CLAUDE.md §3).
    # Injectable so tests supply a manager over fake providers.
    manager = getattr(provider, "manager", None) or manager or ModelManager(
        factory=lambda _model_id: provider, ram_ceiling_gb=settings.ram_ceiling_gb
    )
    warmup_state = _WarmupState()

    app = FastAPI(title="HEARTH", version=__version__)
    app.state.provider = provider
    app.state.settings = settings
    app.state.registry = registry
    app.state.router = router
    app.state.metrics = metrics
    app.state.rag = rag
    app.state.manager = manager
    app.state.warmup = warmup_state

    # Warm what the routing profile serves (most-used rung first, B-064) so the first request
    # is fast and /ready flips to 200. Runs on a
    # background thread so serve starts listening at once (it never blocks on a load); the
    # load itself runs on the MLX thread. Never fatal: a failed warmup is logged loudly and
    # /ready reports 503 with the reason — a missing model shows up at startup, not on the
    # first request.
    if settings.warmup and provider.name != "echo":
        warmup_state.running = True
        warmup_state.thread = threading.Thread(
            target=_warmup,
            args=(provider, manager, registry, warmup_state, router.policy),
            name="hearth-warmup",
            daemon=True,
        )
        warmup_state.thread.start()

    # Auth gates everything except the health/ready liveness+readiness probes.
    auth = Depends(require_token)

    @app.get("/v1/hearth/admin/health")
    def health() -> dict:
        body = {
            "status": "ok",
            "version": __version__,
            "backend": provider.name,
            "model": registry.default_id,
        }
        # Liveness stays "ok" (the process is up), but an auto->echo fallback is named here
        # too, so no probe can read a stub as real inference (B-006).
        stub_reason = getattr(provider, "fallback_reason", None)
        if stub_reason:
            body["backend_fallback"] = stub_reason
        return body

    @app.get("/v1/hearth/admin/ready")
    def ready():
        """Readiness probe (distinct from liveness /health).

        **Ready means every model the active routing profile can route an ``auto`` request
        to can serve now** (B-064) — each class rung, ``defaults.local_model``, and the
        registry default only if some class falls through to it (see
        :func:`~hearth.router.route.policy_rungs`). Under an unpinned profile that is just
        the registry default, exactly as before. Each model is judged on outcomes:

        * it fits under the RAM ceiling at all (``HEARTH_RAM_CEILING_GB``): a rung larger
          than the whole ceiling is refused on every request, so it is ``failed`` whether
          or not warmup ran (B-100); **and**
        * its last load attempt did not fail, and a load of it has completed with weights in
          memory at least once (warmup, or any request) — residency granted without a load
          never counts (B-005); **and**
        * its weights still resolve on disk (for a backend that can check without loading),
          so an evicted model reloads on demand (B-048). A model that is on disk but was not
          loaded (warmup off, or left unloaded to stay under the RAM ceiling) is ready, with
          a ``detail`` saying it loads on the first request (B-035).

        A backend that cannot check the disk stays 503 ``loading`` until its first load. The
        body carries ``models`` — each judged model's ``status``, ``loaded``, the classes it
        ``serves`` and its ``reason``/``detail`` — and the top-level ``status``/``reason``
        aggregate them: any ``failed`` -> 503 ``failed`` naming the model(s); else any
        ``loading`` -> 503 ``loading``; else 200. ``model``/``loaded`` are the primary model
        (the most-used rung, the one warmup loads first). ``HEARTH_DEFAULT_MODEL`` naming no
        registered model is 503 ``failed``. The echo backend is ready when chosen explicitly
        and 503 ``stub`` when ``HEARTH_BACKEND=auto`` fell back to it.
        """
        code, body = _readiness(provider, manager, registry, router.policy, settings,
                                warmup_state)
        return JSONResponse(status_code=code, content=body)

    @app.get("/v1/hearth/admin/models", dependencies=[auth])
    def admin_models() -> dict:
        """What is resident right now, read off the provider instances themselves.

        ``loaded_path`` is the directory ``mlx_lm.load`` read and ``generations`` counts the
        generations THAT instance ran — evidence of which weights answered, independent of
        anything a response says about itself.
        """
        return {
            "backend": provider.name,
            "default": registry.default_id,
            "ram_ceiling_gb": manager.ram_ceiling_gb,
            "resident_ram_gb": manager.resident_ram_gb(),
            "resident": [
                {
                    "model": r.model_id,
                    "ram_gb": r.ram_gb,
                    "provider_model": getattr(r.provider, "model_id", None),
                    # The LoRA adapter this resident layers over the base (None = base): each
                    # variant is a full reload, counted separately (B-069).
                    "adapter": getattr(r.provider, "adapter", None),
                    "loaded": getattr(r.provider, "is_loaded", None),
                    "loaded_path": getattr(r.provider, "loaded_path", None),
                    "generations": getattr(r.provider, "generations", None),
                }
                for r in manager.residents()
            ],
        }

    @app.get("/v1/models", dependencies=[auth])
    def list_models() -> ModelList:
        # Only what a chat request can name and be SERVED by: OpenAI clients (and /chat)
        # build their model picker from this list, so listing echo under the mlx backend, or
        # an embed model, offers picks that can only 404. /v1/embeddings ignores ``model``
        # and always uses the configured embedder, so an embed entry here serves nobody.
        # Same rule as check_model (servable_for mirrors it), so list and 404 agree.
        servable = set(servable_for(router.local, registry))
        return ModelList(
            data=[
                ModelCard(
                    id=e.id,
                    backend=e.backend,
                    context=e.context,
                    capabilities=e.capabilities,
                )
                for e in registry.list()
                if e.id in servable
            ]
        )

    @app.post("/v1/embeddings", dependencies=[auth])
    def embeddings(req: EmbeddingRequest) -> EmbeddingResponse:
        # OpenAI-compatible embeddings via the active EmbeddingProvider (Phase 3). The
        # default is the offline hashing embedder, so this works with no extras/network.
        texts = [req.input] if isinstance(req.input, str) else list(req.input)
        vectors = rag.embedder.embed(texts)
        prompt_tokens = sum(_approx_tokens(t) for t in texts)
        return EmbeddingResponse(
            data=[
                EmbeddingData(embedding=vec, index=i) for i, vec in enumerate(vectors)
            ],
            model=rag.embedder.name,
            usage=EmbeddingUsage(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens),
        )

    @app.post("/v1/chat/completions", dependencies=[auth])
    def chat_completions(req: ChatCompletionRequest):
        opts = req.hearth
        intent = opts.intent if opts else None
        allow_escalation = opts.allow_escalation if opts else True
        adapter = opts.adapter if opts else None
        # response_format is validated before anything is generated: an unsupported value
        # is a bad request, not something to quietly downgrade to plain text.
        try:
            response_format = resolve_format(req.response_format)
        except UnsupportedResponseFormatError as exc:
            return _response_format_error(str(exc))
        # An unknown model is a 404 before anything runs — streaming included, where a
        # failure after the 200 could only be an in-band error event.
        try:
            check_model(router.local, registry, req.model)
        except UnknownModelError as exc:
            return _model_not_found(exc)
        # Same for an explicitly requested adapter: refused up front, never silently
        # answered by base weights under the adapter's name (B-034).
        try:
            router.check_adapter(adapter)
        except UnknownAdapterError as exc:
            return _adapter_not_found(exc)
        messages = (
            json_instruction(req.messages)
            if response_format == "json_object"
            else req.messages
        )
        gen_req = GenRequest(
            messages=[Message(role=m.role, content=m.content) for m in messages],
            model=req.model,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
        if req.stream:
            return StreamingResponse(
                _close_on_disconnect(
                    _guarantee_done(
                        _stream_sse(
                            router, gen_req, intent, allow_escalation, adapter,
                            response_format,
                        )
                    )
                ),
                media_type="text/event-stream",
            )

        try:
            routed = router.route(
                gen_req, intent=intent, allow_escalation=allow_escalation, adapter=adapter
            )
        except BudgetExhaustedError as exc:
            return _budget_error(str(exc))
        except UnknownModelError as exc:
            return _model_not_found(exc)
        except UnknownAdapterError as exc:
            return _adapter_not_found(exc)
        except ProviderError as exc:
            return _provider_error(str(exc))

        result = routed.result
        rec = routed.record
        total = result.prompt_tokens + result.completion_tokens
        # JSON mode validates before responding: a truncated object is unparseable, so this
        # turns the worst failure shape (plausible-looking partial data) into a named error.
        if response_format == "json_object":
            try:
                parse_json_object(result.text, result.finish_reason)
            except InvalidJsonResponseError as exc:
                # The client gets an error, so the request is a failure in the metrics too.
                router.metrics.mark_failed(rec, f"invalid_json_response: {exc}")
                return _json_mode_error(str(exc), exc.content, result.finish_reason)
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
            created=int(time.time()),
            model=result.model,
            choices=[
                ChatChoice(
                    message=ChatChoiceMessage(content=result.text),
                    finish_reason=result.finish_reason,
                )
            ],
            usage=Usage(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=total,
            ),
            hearth=HearthTelemetry(
                served_by=rec.served_by,
                backend=result.backend,
                model=result.model,
                adapter=rec.adapter,  # what served (None = base weights), not the request
                escalated=rec.escalated,
                estimated_frontier_tokens_saved=rec.estimated_frontier_tokens_saved,
            ),
        )

    @app.post("/v1/hearth/route", dependencies=[auth])
    def route_dry_run(req: RouteRequest) -> RouteResponse:
        gen_req = GenRequest(
            messages=[Message(role=m.role, content=m.content) for m in req.messages],
            model="auto",
        )
        d = router.decide(gen_req, intent=req.intent, allow_escalation=req.allow_escalation)
        return RouteResponse(
            **{"class": d.task_class},
            method=d.method,
            backend=d.backend,
            model=d.model,
            would_escalate=d.would_escalate,
            reason=d.reason,
            confidence=d.confidence,
        )

    @app.get("/v1/hearth/admin/metrics", dependencies=[auth])
    def admin_metrics(since: str | None = Query(None)) -> dict:
        return metrics.rollup(since_s=_parse_since(since))

    @app.post("/v1/hearth/rag/ingest", dependencies=[auth])
    def rag_ingest(req: RagIngestRequest) -> RagIngestResponse:
        files = 0
        chunks = 0
        for path in req.paths:
            result = rag.ingest(
                path, req.collection, size=req.chunk.size, overlap=req.chunk.overlap
            )
            files += result.files
            chunks += result.chunks
        return RagIngestResponse(collection=req.collection, files=files, chunks=chunks)

    @app.post("/v1/hearth/rag/query", dependencies=[auth])
    def rag_query(req: RagQueryRequest) -> RagQueryResponse:
        result = rag.query(req.collection, req.query, k=req.k, answer=req.answer)
        return RagQueryResponse(
            chunks=[
                RagChunk(text=c.text, source=c.source, score=c.score) for c in result.chunks
            ],
            answer=result.answer,
        )

    # The bounded agent loop at POST /v1/hearth/agent: authenticated like every other /v1
    # route, read-only vetted tools only, budgets clamped server-side, streamed step by step.
    # See hearth.gateway.agent_route and docs/AGENT.md §9.
    register_agent_route(app)

    # Operator chat UI at GET /chat: static, self-contained, no credential in the document
    # and no auth dependency (a browser navigation cannot send a bearer header). It changes
    # nothing about /v1/* auth — see hearth.gateway.chat_ui and docs/API.md.
    register_chat_ui(app)

    return app


def _approx_tokens(text: str) -> int:
    """Cheap ~4-chars-per-token estimate for embeddings usage accounting."""
    return max(1, len(text) // 4)


@dataclass
class _WarmupState:
    """What the warmup thread found. Read by /ready; written by :func:`_warmup`.

    ``error`` is a failure that sinks every model (``HEARTH_DEFAULT_MODEL`` unregistered);
    ``errors`` is per model; ``skipped`` holds rungs left unloaded to stay under the RAM
    ceiling (they load on their first request).
    """

    running: bool = False
    error: str | None = None
    errors: dict[str, str] = field(default_factory=dict)
    skipped: set[str] = field(default_factory=set)
    plan: list[str] = field(default_factory=list)
    thread: threading.Thread | None = None


def _default_model_problem(registry: Registry) -> str | None:
    """Why the configured default model cannot be the one serving, or ``None``.

    ``Registry.default_id`` ignores a ``HEARTH_DEFAULT_MODEL`` that names no registered
    model and falls back to the catalog default — so the operator asked for one model and
    would be served another with every probe green. Readiness says so instead.
    """
    from ..registry import UnregisteredDefaultModelError

    try:  # create_app already refused this; kept so readiness can never disagree with it
        registry.require_default()
    except UnregisteredDefaultModelError as exc:
        return str(exc)
    return None


def _routed_models(provider: ModelProvider, registry: Registry, policy) -> dict[str, list[str]]:
    """The models readiness judges and warmup loads -> the classes each serves, primary first.

    A :class:`~hearth.serving.ModelPool` (a provider with its own ``manager``) holds one
    provider per model id, so every rung is a different set of weights and each is judged.
    Any other provider answers every id itself: one judgement, under the primary rung's id,
    covers all classes.
    """
    rungs = policy_rungs(policy, registry.default_id)
    if getattr(provider, "manager", None) is not None:
        return rungs
    primary = next(iter(rungs))
    return {primary: [source for sources in rungs.values() for source in sources]}


def _judge_model(
    model_id: str,
    provider: ModelProvider,
    manager: ModelManager,
    settings: Settings,
    state: _WarmupState,
) -> dict:
    """One model's readiness: ``status`` (ready|loading|failed), ``loaded``, reason/detail."""
    loaded = _weights_loaded(manager, model_id)
    out: dict = {"status": "ready", "loaded": loaded}

    def verdict(status: str, reason: str | None = None, detail: str | None = None) -> dict:
        out["status"] = status
        if reason is not None:
            out["reason"] = reason
        if detail is not None:
            out["detail"] = detail
        return out

    resolve = getattr(provider, "resolve", None)
    if callable(resolve):
        try:
            resolve(model_id)
        except UnknownModelError as exc:
            return verdict("failed", f"{model_id!r} is not servable: {exc}")
    # A rung larger than the whole RAM ceiling can never load: the manager refuses it on
    # every request (ModelTooLargeError -> 503), warmup on or off. Judged with the manager's
    # own rule over the footprint it will size the load by (B-100).
    too_large = _size_problem(model_id, provider, manager)
    if too_large is not None:
        return verdict(
            "failed",
            f"{model_id!r} can never load under this RAM ceiling ({too_large}); every request "
            "routed to it fails — raise HEARTH_RAM_CEILING_GB or route its classes to a "
            "smaller model",
        )
    can_locate = bool(getattr(provider, "can_locate", False))
    if can_locate:
        missing = provider.weights_problem(model_id)
        if missing is not None:
            return verdict(
                "failed", f"weights for {model_id!r} do not resolve on disk: {missing}"
            )
    last_error = getattr(manager, "last_load_error", lambda _m: None)(model_id)
    if last_error is not None:
        return verdict("failed", f"last load of {model_id!r} failed: {last_error}")
    if loaded:
        return verdict("ready")
    if getattr(manager, "loaded_once", lambda _m: False)(model_id):
        return verdict(
            "ready",
            detail=f"{model_id!r} loaded before and is not resident now (evicted to make "
            "room); it reloads on demand",
        )
    if model_id in state.errors:
        return verdict("failed", state.errors[model_id])
    if state.running and model_id not in state.skipped:
        return verdict("loading", "warmup in progress")
    if not settings.warmup:
        if can_locate:
            return verdict(
                "ready",
                detail=f"warmup disabled (HEARTH_WARMUP=false); {model_id!r} is on disk and "
                "loads on the first request",
            )
        return verdict(
            "loading",
            f"weights for {model_id!r} are not loaded (HEARTH_WARMUP is off) and the "
            f"{provider.name!r} backend cannot verify them without loading",
        )
    if model_id in state.skipped and can_locate:
        return verdict(
            "ready",
            detail=f"{model_id!r} is on disk and loads on the first request (warmup left it "
            "unloaded: it did not fit under the RAM ceiling beside the rungs loaded first)",
        )
    return verdict("loading", f"weights for {model_id!r} are not loaded yet")


def _readiness(
    provider: ModelProvider,
    manager: ModelManager,
    registry: Registry,
    policy,
    settings: Settings,
    state: _WarmupState,
) -> tuple[int, dict]:
    """``(status_code, body)`` for ``/ready`` — see the route's docstring for the rules."""
    routed = _routed_models(provider, registry, policy)
    primary = next(iter(routed))
    payload: dict = {
        "backend": provider.name,
        "model": primary,
        "loaded": _weights_loaded(manager, primary),
        "resident": manager.resident_ids(),
    }

    def respond(code: int, status: str, reason: str | None = None,
                detail: str | None = None, models: dict | None = None) -> tuple[int, dict]:
        body = {**payload, "status": status}
        if reason is not None:
            body["reason"] = reason
        if detail is not None:
            body["detail"] = detail
        if models is not None:
            body["models"] = models
        return code, body

    if provider.name == "echo":
        # An echo that `auto` fell back to (mlx_lm not importable) is not inference: a
        # pruned venv must not come up green answering every request with an echo
        # labelled as a real model (B-006). An explicit HEARTH_BACKEND=echo is ready.
        stub_reason = getattr(provider, "fallback_reason", None)
        if stub_reason:
            return respond(503, "stub", stub_reason)
        return respond(200, "ready")
    problem = _default_model_problem(registry) or state.error
    if problem is not None:
        return respond(503, "failed", problem)

    models: dict[str, dict] = {}
    for model_id, serves in routed.items():
        models[model_id] = {
            **_judge_model(model_id, provider, manager, settings, state), "serves": serves
        }
    for status in ("failed", "loading"):
        hits = [m for m in models.values() if m["status"] == status]
        if hits:
            return respond(503, status, "; ".join(m["reason"] for m in hits), models=models)
    details = [m["detail"] for m in models.values() if "detail" in m]
    return respond(200, "ready", detail="; ".join(details) if details else None,
                   models=models)


def _size_problem(model_id: str, provider: ModelProvider, manager: ModelManager) -> str | None:
    """Why ``model_id`` can never be admitted by ``manager``, or ``None``.

    Sized by ``provider.footprint`` — what the manager's admission sizes the load by (a
    pool's per-model providers report the same registry ``ram_gb`` the pool does). A manager
    without :meth:`~hearth.serving.ModelManager.size_problem` or a provider that cannot size
    the model is not judged here: its load outcome still is.
    """
    size_problem = getattr(manager, "size_problem", None)
    if not callable(size_problem):
        return None
    try:
        ram_gb = provider.footprint(model_id).ram_gb
    except Exception:  # noqa: BLE001 — an unsizable model is judged by its load instead
        return None
    return size_problem(model_id, ram_gb)


def _weights_loaded(manager: ModelManager, model_id: str) -> bool:
    """True when ``model_id`` is resident AND its provider reports weights in memory."""
    resident = manager.peek(model_id)
    if resident is None:
        return False
    loaded = getattr(resident, "is_loaded", None)
    return True if loaded is None else bool(loaded)


def _warmup(
    provider: ModelProvider,
    manager: ModelManager,
    registry: Registry,
    state: _WarmupState,
    policy=None,
) -> bool:
    """Load the weights the routing profile serves now; record (never raise) a failure.

    Loads the most-used rung first (B-064: it used to load the registry default, which a
    pinned ladder may never serve), then each further rung that fits under the RAM ceiling
    beside what is already loaded — warmup never evicts a model it just warmed. A rung that
    does not fit is left to load on its first request. Returns whether every load it
    attempted succeeded. A failed load must not take the server down — `hearth serve`
    stays up in degraded mode and `/ready` reports 503 with the reason (Phase 7 graceful
    degradation). A pool loads through its own path so the load runs on the MLX thread.
    """
    state.running = True
    ok = True
    try:
        problem = _default_model_problem(registry)
        if problem is not None:
            state.error = f"warmup refused: {problem}"
            logger.error("%s — NOT READY; serving in degraded mode", state.error)
            return False
        if policy is None:
            from ..router.policy import get_policy

            policy = get_policy()
        state.plan = list(_routed_models(provider, registry, policy))
        warm = getattr(provider, "warm", None)
        via_pool = callable(warm) and getattr(provider, "manager", None) is manager
        for index, model_id in enumerate(state.plan):
            if index > 0 and not _fits_beside_residents(manager, registry, model_id):
                state.skipped.add(model_id)
                logger.info("warmup left %s unloaded: it does not fit under the RAM ceiling "
                            "beside the rungs already warmed; it loads on demand", model_id)
                continue
            try:
                if via_pool:
                    warm(model_id)
                else:
                    manager.get(model_id)
                if not _weights_loaded(manager, model_id):
                    raise RuntimeError(f"warmup returned but {model_id!r} holds no weights")
                state.errors.pop(model_id, None)
                logger.info("warmed %s", model_id)
            except Exception as exc:  # noqa: BLE001 — warmup is best-effort; never fatal
                ok = False
                state.errors[model_id] = (
                    f"warmup of {model_id!r} failed: {type(exc).__name__}: {exc}"
                )
                logger.error("%s — NOT READY; serving in degraded mode",
                             state.errors[model_id])
        return ok
    finally:
        state.running = False


def _fits_beside_residents(manager: ModelManager, registry: Registry, model_id: str) -> bool:
    """Whether loading ``model_id`` now would stay under the ceiling without evicting."""
    entry = registry.get(model_id)
    need = entry.ram_gb if entry is not None else 0.0
    return manager.resident_ram_gb() + need <= manager.ram_ceiling_gb


def _model_not_found(exc: UnknownModelError) -> JSONResponse:
    """OpenAI-style 404 for a model id this server cannot serve (docs/API.md)."""
    return JSONResponse(
        status_code=404,
        content={
            "error": {
                "message": str(exc),
                "type": "invalid_request_error",
                "param": "model",
                "code": "model_not_found",
            }
        },
    )


def _adapter_not_found(exc: UnknownAdapterError) -> JSONResponse:
    """404 for an explicitly requested adapter this server cannot serve (docs/API.md)."""
    return JSONResponse(
        status_code=404,
        content={
            "error": {
                "message": str(exc),
                "type": "invalid_request_error",
                "param": "hearth.adapter",
                "code": "adapter_not_found",
            }
        },
    )


def _budget_error(message: str) -> JSONResponse:
    """OpenAI-style error envelope for the budget-exhausted case (docs/API.md)."""
    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "message": message,
                "type": "budget_exhausted",
                "code": "hearth.budget.exhausted",
            }
        },
    )


def _provider_error(message: str) -> JSONResponse:
    """OpenAI-style 503 envelope for a provider load/generate failure (Phase 7).

    The provider degraded (bad adapter → base retry) but still failed; return a clean,
    retryable 503 rather than a 500 traceback so clients can back off or escalate.
    """
    return JSONResponse(
        status_code=503,
        content={
            "error": {
                "message": message,
                "type": "provider_unavailable",
                "code": "hearth.provider.unavailable",
            }
        },
    )


def _response_format_error(message: str) -> JSONResponse:
    """OpenAI-style 400 envelope for an unsupported ``response_format`` (docs/API.md)."""
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": "response_format",
                "code": "hearth.response_format.unsupported",
            }
        },
    )


def _json_mode_error(message: str, content: str, finish_reason: str) -> JSONResponse:
    """422 envelope for a ``json_object`` completion that doesn't parse.

    The offending completion rides along in the additive ``hearth`` block: the caller
    needs to see what the model actually produced (and whether ``max_tokens`` cut it off)
    to decide between retrying, raising the cap, or splitting the request.
    """
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "message": message,
                "type": "invalid_json_response",
                "code": "hearth.response_format.invalid_json",
            },
            "hearth": {"finish_reason": finish_reason, "content": content},
        },
    )


def _sse(payload: object) -> str:
    """Serialize one payload as an SSE ``data:`` event."""
    if isinstance(payload, str):
        return f"data: {payload}\n\n"
    if isinstance(payload, dict):
        import json

        return f"data: {json.dumps(payload)}\n\n"
    return f"data: {payload.model_dump_json(exclude_none=True)}\n\n"


async def _close_on_disconnect(
    stream: Iterator[str], on_close: Callable[[], None] | None = None
) -> AsyncIterator[str]:
    """Relay a sync SSE generator, and CLOSE it however the response ends.

    Starlette drives a sync body from a threadpool and, when the client disconnects, simply
    stops pulling — it never calls ``close()``. The generator then sits suspended, so the
    provider's cancellation (``providers/mlx.py:iterate_on_mlx_thread``, which fires on
    close) never runs and the MLX thread keeps generating to ``max_tokens`` while every
    other request queues behind it. Measured against Coder-14B: a stream abandoned 1.6 s in
    made the next 0.5 s request take 69.8 s — the full generation.

    ``on_close`` runs first, synchronously, however the response ends: the agent route
    passes its cancellation flag's ``set`` here, because closing its generator alone cannot
    reach a run already executing on a worker thread (B-071).
    """
    try:
        async for chunk in iterate_in_threadpool(stream):
            yield chunk
    finally:
        if on_close is not None:
            on_close()
        # Not awaited: this runs inside a cancelled task. A helper thread closes the
        # generator as soon as any in-flight next() on a pool thread has returned.
        threading.Thread(
            target=_close_when_idle, args=(stream,), name="hearth-stream-close", daemon=True
        ).start()


def _close_when_idle(stream: Iterator[str]) -> None:
    """``close()`` a generator, retrying while a pool thread is still inside ``next()``."""
    close = getattr(stream, "close", None)
    if close is None:
        return
    while True:
        try:
            close()
            return
        except ValueError:  # "generator already executing" — the in-flight next() ends soon
            time.sleep(0.02)


def _guarantee_done(stream: Iterator[str]) -> Iterator[str]:
    """Relay an SSE generator so it ends with ``[DONE]`` however it ends.

    :func:`_stream_sse` handles every failure it knows about; this is the backstop for the
    ones it does not (an adapter store that raises before the first chunk, a bug in a
    chunk build). An exception becomes a ``hearth.stream.internal_error`` event and
    ``[DONE]`` instead of a dropped connection. ``yield from`` forwards ``close()`` to the
    inner generator, so an abandoned stream still cancels generation (``GeneratorExit`` is
    not an ``Exception`` and passes straight through).
    """
    try:
        yield from stream
    except Exception as exc:  # noqa: BLE001 — every stream ends with [DONE]
        logger.exception("stream aborted by an unexpected error")
        yield _sse(
            {
                "error": {
                    "message": f"stream aborted: {type(exc).__name__}: {exc}",
                    "type": "internal_error",
                    "code": "hearth.stream.internal_error",
                }
            }
        )
        yield _sse("[DONE]")


def _stream_sse(
    router: Router,
    gen_req: GenRequest,
    intent: str | None,
    allow_escalation: bool,
    adapter: str | None,
    response_format: str = "text",
):
    """Yield OpenAI-compatible SSE chunks, then a final hearth chunk, then ``[DONE]``.

    The router decides (classify/gate) up front; the chosen provider streams the text.
    The final chunk carries real ``served_by``/``escalated``/savings telemetry plus the
    provider's own ``finish_reason``, so a stream cut off at ``max_tokens`` reports
    ``"length"`` exactly as the non-streaming path does. A :class:`RequestRecord` is
    written to the metrics store when the stream completes — and when it fails (with
    ``failed`` set), so an error is never invisible to ``hearth stats``.

    Under ``response_format="json_object"`` deltas still stream as they arrive (a client
    asked to stream), and the accumulated text is validated once at the end: an
    unparseable object is reported as a trailing error event before ``[DONE]``.
    """
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def base_choice(delta: ChatChunkDelta, finish: str | None = None) -> ChatChunkChoice:
        return ChatChunkChoice(delta=delta, finish_reason=finish)

    try:
        decision, provider = _resolve_stream_provider(
            router, gen_req, intent, allow_escalation
        )
    except BudgetExhaustedError as exc:
        # Emit an OpenAI-style error event, then terminate the stream.
        yield _sse(
            {
                "error": {
                    "message": str(exc),
                    "type": "budget_exhausted",
                    "code": "hearth.budget.exhausted",
                }
            }
        )
        yield _sse("[DONE]")
        return

    # Adapters layer over the local backend only; resolve the requested id (or the task's
    # promoted default) to a concrete path so streaming hot-swaps like non-streaming does.
    choice = (
        AdapterChoice()
        if decision.would_escalate
        else router.select_adapter(adapter, decision.task_class, decision.model)
    )
    stream_req = GenRequest(
        messages=gen_req.messages,
        model=decision.model,
        max_tokens=gen_req.max_tokens,
        temperature=gen_req.temperature,
        adapter=choice.path,
    )

    # First chunk announces the assistant role (OpenAI convention). Every chunk names the
    # model that GENERATES (B-101): the echo stub is "echo" from the first chunk, not only in
    # the final one — decision.model names what was asked for.
    yield _sse(
        ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=generating_model(provider, decision.model),
            choices=[base_choice(ChatChunkDelta(role="assistant"))],
        )
    )

    started = time.perf_counter()
    parts: list[str] = []
    finish_reason = "stop"
    escalation_failed: str | None = None
    # The model that actually generated, as reported by the provider instance that ran it
    # (the terminal StreamDelta). Never the request's or the decision's model: those name
    # what was asked for, which is exactly what used to be reported when another model ran.
    served_model: str | None = None
    # The adapter path the answer was actually generated with (None after a base retry).
    used_path: str | None = None

    def relay(provider: ModelProvider, stream_req: GenRequest):
        nonlocal finish_reason, served_model
        model = generating_model(provider, stream_req.model)
        for event in iter_stream(provider, stream_req):
            if event.finish_reason:
                finish_reason = event.finish_reason
            if event.model:
                served_model = event.model
            if not event.text:
                continue
            parts.append(event.text)
            yield _sse(
                ChatCompletionChunk(
                    id=chunk_id,
                    created=created,
                    model=model,
                    choices=[base_choice(ChatChunkDelta(content=event.text))],
                )
            )

    def relay_local(stream_req: GenRequest):
        # Mirrors Router._generate: an adapter that fails before any text is retried once on
        # base weights, so a broken promoted adapter cannot fail /chat while the
        # non-streaming path quietly succeeds.
        nonlocal used_path
        used_path = stream_req.adapter
        try:
            yield from relay(router.local, stream_req)
        except Exception as exc:  # noqa: BLE001
            if stream_req.adapter is None or parts:
                raise
            logger.warning("stream failed with adapter; retrying on base weights: %s", exc)
            used_path = None
            yield from relay(
                router.local,
                GenRequest(
                    messages=stream_req.messages,
                    model=stream_req.model,
                    max_tokens=stream_req.max_tokens,
                    temperature=stream_req.temperature,
                    adapter=None,
                ),
            )

    try:
        if not decision.would_escalate:
            yield from relay_local(stream_req)
        else:
            try:
                yield from relay(provider, stream_req)
            except Exception as exc:  # noqa: BLE001
                if parts:
                    # The remote received the prompt and produced tokens before dying: that
                    # is spend and an escalation that failed, so it is billed and recorded —
                    # not left to a log line — and nothing local is spliced onto its answer.
                    _record_failed_remote_stream(
                        router, gen_req, decision, provider, "".join(parts), None,
                        f"provider {provider.name!r} failed mid-stream: {exc}",
                        (time.perf_counter() - started) * 1000.0,
                    )
                    yield from _stream_failure(provider, exc)
                    return
                # The remote failed before saying anything: serve the whole answer locally,
                # as Router.route does. The final hearth chunk reports local as what served.
                escalation_failed = f"provider {provider.name!r} failed: {exc}"
                decision = router.degrade_to_local(gen_req, decision, exc)
                provider = router.local
                choice = router.select_adapter(adapter, decision.task_class, decision.model)
                stream_req = GenRequest(
                    messages=gen_req.messages,
                    model=decision.model,
                    max_tokens=gen_req.max_tokens,
                    temperature=gen_req.temperature,
                    adapter=choice.path,
                )
                yield from relay_local(stream_req)
    except UnknownModelError as exc:
        logger.error("stream refused: %s", exc)
        # Recorded like any failed request (B-066) — after a failed escalation too, where the
        # remote may already hold the prompt.
        router.record_failure(
            gen_req, decision, provider, exc, started=started, adapter=choice.id,
            escalation_failed=escalation_failed,
        )
        yield _sse(
            {
                "error": {
                    "message": str(exc),
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            }
        )
        yield _sse("[DONE]")
        return
    except Exception as exc:  # noqa: BLE001 — a dead stream must still end, and say why
        # Recorded before the error event goes out: a failed request (and, after a failed
        # escalation, a prompt the remote may already hold) must reach the metrics.
        router.record_failure(
            gen_req, decision, provider, exc, started=started, adapter=choice.id,
            escalation_failed=escalation_failed,
            completion_tokens=_estimate_stream_tokens(gen_req, "".join(parts))[1]
            if parts else 0,
        )
        yield from _stream_failure(provider, exc)
        return
    latency_ms = (time.perf_counter() - started) * 1000.0
    text = "".join(parts)
    # A provider that cannot say which model ran (third-party, plain stream()) falls back to
    # the same derivation that labelled the chunks: its bound model id, else the decision.
    model_served = served_model or generating_model(provider, decision.model)
    # The adapter that actually served — not the one requested (B-034).
    served_adapter = router.served_adapter(provider, choice, used_path)

    prompt_tokens, completion_tokens = _estimate_stream_tokens(gen_req, text)
    served_by = "remote" if decision.would_escalate else "local"
    saved = (
        0
        # A remote answer saved nothing; neither did an echo stub's (B-068).
        if served_by == "remote" or is_stub_backend(provider.name)
        else estimated_tokens_saved(decision.task_class, prompt_tokens, completion_tokens)
    )
    # The answer has already been streamed. Accounting that fails now (a full disk under the
    # metrics store, an injected store that raises) must not drop the stream with no [DONE]:
    # the client still gets the final chunk, then a named error event, then [DONE].
    accounting_error: str | None = None
    stream_record: RequestRecord | None = None
    try:
        if served_by == "remote":
            router.budget.spend(prompt_tokens + completion_tokens)
        stream_record = RequestRecord(
            task_class=decision.task_class,
            backend=provider.name,
            model=model_served,
            served_by=served_by,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            escalated=decision.would_escalate,
            escalation_reason=decision.reason if decision.would_escalate else None,
            escalation_failed=escalation_failed,
            adapter=served_adapter,
            estimated_frontier_tokens_saved=saved,
        )
        router.metrics.record(stream_record)
    except Exception as exc:  # noqa: BLE001 — accounting must not kill a delivered answer
        accounting_error = f"{type(exc).__name__}: {exc}"
        logger.error("stream served but accounting failed: %s", accounting_error)

    yield _sse(
        ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=model_served,
            choices=[base_choice(ChatChunkDelta(), finish=finish_reason)],
            hearth=HearthTelemetry(
                served_by=served_by,
                backend=provider.name,
                model=model_served,
                adapter=served_adapter,
                escalated=decision.would_escalate,
                estimated_frontier_tokens_saved=saved,
            ),
        )
    )
    if accounting_error is not None:
        yield _sse(
            {
                "error": {
                    "message": f"the answer was served but could not be recorded: "
                    f"{accounting_error}",
                    "type": "metrics_unavailable",
                    "code": "hearth.metrics.unavailable",
                }
            }
        )
    # Post-hoc validation for JSON mode: the deltas are already out, so the honest move is
    # to tell the client the object they just assembled is not usable, not to stay quiet.
    if response_format == "json_object":
        try:
            parse_json_object(text, finish_reason)
        except InvalidJsonResponseError as exc:
            if stream_record is not None:
                try:
                    router.metrics.mark_failed(stream_record, f"invalid_json_response: {exc}")
                except Exception as mark_exc:  # noqa: BLE001 — the [DONE] still goes out
                    logger.error("could not mark stream failed: %s", mark_exc)
            yield _sse(
                {
                    "error": {
                        "message": str(exc),
                        "type": "invalid_json_response",
                        "code": "hearth.response_format.invalid_json",
                    },
                    "hearth": {"finish_reason": finish_reason},
                }
            )
    yield _sse("[DONE]")


def _estimate_stream_tokens(gen_req: GenRequest, text: str) -> tuple[int, int]:
    """(prompt, completion) tokens estimated at ~4 chars/token, without a tokenizer pass."""
    prompt_tokens = max(1, sum(len(m.content) for m in gen_req.messages) // 4)
    return prompt_tokens, max(1, len(text) // 4)


def _record_failed_remote_stream(
    router: Router,
    gen_req: GenRequest,
    decision,
    provider: ModelProvider,
    partial_text: str,
    adapter: str | None,
    error: str,
    latency_ms: float,
) -> None:
    """Bill and record a remote stream that died after emitting text."""
    prompt_tokens, completion_tokens = _estimate_stream_tokens(gen_req, partial_text)
    router.budget.spend(prompt_tokens + completion_tokens)
    router.metrics.record(
        RequestRecord(
            task_class=decision.task_class,
            backend=provider.name,
            model=decision.model,
            served_by="remote",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            escalated=True,
            escalation_reason=decision.reason,
            escalation_failed=error,
            adapter=adapter,
            estimated_frontier_tokens_saved=0,
            # The client gets an error event, not an answer: a failed request.
            failed=error,
        )
    )


def _stream_failure(provider: ModelProvider, exc: Exception):
    """Terminate a stream whose provider raised: an error event, then ``[DONE]``.

    Without this the generator died mid-response with no ``[DONE]``, which a client sees as
    a dropped connection, not as a failure it can name — the /chat page showed only
    "[stream failed]". Mirrors the non-streaming path's ``hearth.provider.unavailable``.
    """
    logger.error("provider %s failed mid-stream: %s", provider.name, exc)
    yield _sse(
        {
            "error": {
                "message": f"provider {provider.name!r} failed: {exc}",
                "type": "provider_unavailable",
                "code": "hearth.provider.unavailable",
            }
        }
    )
    yield _sse("[DONE]")


def _resolve_stream_provider(
    router: Router,
    gen_req: GenRequest,
    intent: str | None,
    allow_escalation: bool,
):
    """Decide the route and return ``(decision, provider)`` for streaming.

    The same :meth:`Router.provider_for` :meth:`Router.route` uses, so streaming and
    non-streaming share the same escalation + budget semantics and accounting.
    """
    decision = router.decide(gen_req, intent=intent, allow_escalation=allow_escalation)
    # Router.provider_for records a denied escalation before raising it (B-066).
    return decision, router.provider_for(gen_req, decision)


def _parse_since(since: str | None) -> float | None:
    """Parse a ``--since``-style window (e.g. ``7d``, ``24h``, ``30m``) into seconds."""
    if not since:
        return None
    since = since.strip().lower()
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if since[-1] in units and since[:-1].isdigit():
        return int(since[:-1]) * units[since[-1]]
    if since.isdigit():  # bare number = seconds
        return float(since)
    return None
