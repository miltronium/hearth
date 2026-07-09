"""FastAPI application factory and routes.

Serves the OpenAI-compatible core (``/v1/chat/completions`` incl. streaming,
``/v1/embeddings`` stub, ``/v1/models``) plus the ``/v1/hearth/`` extension + admin
surface. From Phase 2, chat completions go through the :class:`~hearth.router.Router`
(classify → select → gate → execute → record) rather than calling a provider directly.
A local bearer token (see :mod:`hearth.gateway.auth`) gates every route except the
liveness probe ``/v1/hearth/admin/health``.
"""

from __future__ import annotations

import time
import uuid

from fastapi import Depends, FastAPI, Query
from fastapi.responses import JSONResponse, StreamingResponse

from .. import __version__
from ..config import Settings, get_settings
from ..observability.metrics import (
    MetricsStore,
    RequestRecord,
    estimated_tokens_saved,
    get_metrics,
)
from ..providers import select_provider
from ..providers.base import GenRequest, Message, ModelProvider
from ..registry import Registry, get_registry
from ..router import BudgetExhaustedError, Router
from .auth import require_token
from .schemas import (
    ChatChoice,
    ChatChoiceMessage,
    ChatChunkChoice,
    ChatChunkDelta,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    HearthTelemetry,
    ModelCard,
    ModelList,
    RouteRequest,
    RouteResponse,
    Usage,
)


def create_app(
    provider: ModelProvider | None = None,
    settings: Settings | None = None,
    registry: Registry | None = None,
    router: Router | None = None,
    metrics: MetricsStore | None = None,
) -> FastAPI:
    """Build the HEARTH FastAPI app. Pass ``provider``/``router`` to inject stubs in tests."""
    settings = settings or get_settings()
    provider = provider or select_provider(settings)
    registry = registry or get_registry()
    metrics = metrics or get_metrics()
    router = router or Router(local_provider=provider, metrics=metrics)

    app = FastAPI(title="HEARTH", version=__version__)
    app.state.provider = provider
    app.state.settings = settings
    app.state.registry = registry
    app.state.router = router
    app.state.metrics = metrics

    # Auth gates everything except /v1/hearth/admin/health (declared without the dep).
    auth = Depends(require_token)

    @app.get("/v1/hearth/admin/health")
    def health() -> dict:
        return {
            "status": "ok",
            "version": __version__,
            "backend": provider.name,
            "model": registry.default_id,
        }

    @app.get("/v1/models", dependencies=[auth])
    def list_models() -> ModelList:
        return ModelList(
            data=[
                ModelCard(
                    id=e.id,
                    backend=e.backend,
                    context=e.context,
                    capabilities=e.capabilities,
                )
                for e in registry.list()
            ]
        )

    @app.post("/v1/embeddings", dependencies=[auth])
    def embeddings() -> JSONResponse:
        # Real implementation lands in Phase 3 (local embedder + vector store). The route
        # exists so the surface is complete, but it is honestly not implemented yet.
        return JSONResponse(
            status_code=501,
            content={
                "error": {
                    "message": "embeddings are not implemented yet (Phase 3)",
                    "type": "not_implemented",
                    "code": "hearth.embeddings.not_implemented",
                }
            },
        )

    @app.post("/v1/chat/completions", dependencies=[auth])
    def chat_completions(req: ChatCompletionRequest):
        opts = req.hearth
        intent = opts.intent if opts else None
        allow_escalation = opts.allow_escalation if opts else True
        adapter = opts.adapter if opts else None
        gen_req = GenRequest(
            messages=[Message(role=m.role, content=m.content) for m in req.messages],
            model=req.model,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
        if req.stream:
            return StreamingResponse(
                _stream_sse(router, gen_req, intent, allow_escalation, adapter),
                media_type="text/event-stream",
            )

        try:
            routed = router.route(
                gen_req, intent=intent, allow_escalation=allow_escalation, adapter=adapter
            )
        except BudgetExhaustedError as exc:
            return _budget_error(str(exc))

        result = routed.result
        rec = routed.record
        total = result.prompt_tokens + result.completion_tokens
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
            created=int(time.time()),
            model=result.model,
            choices=[ChatChoice(message=ChatChoiceMessage(content=result.text))],
            usage=Usage(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=total,
            ),
            hearth=HearthTelemetry(
                served_by=rec.served_by,
                backend=result.backend,
                model=result.model,
                adapter=adapter,
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

    return app


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


def _sse(payload: object) -> str:
    """Serialize one payload as an SSE ``data:`` event."""
    if isinstance(payload, str):
        return f"data: {payload}\n\n"
    if isinstance(payload, dict):
        import json

        return f"data: {json.dumps(payload)}\n\n"
    return f"data: {payload.model_dump_json(exclude_none=True)}\n\n"


def _stream_sse(
    router: Router,
    gen_req: GenRequest,
    intent: str | None,
    allow_escalation: bool,
    adapter: str | None,
):
    """Yield OpenAI-compatible SSE chunks, then a final hearth chunk, then ``[DONE]``.

    The router decides (classify/gate) up front; the chosen provider streams the text.
    The final chunk carries real ``served_by``/``escalated``/savings telemetry, and a
    :class:`RequestRecord` is written to the metrics store when the stream completes.
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

    stream_req = GenRequest(
        messages=gen_req.messages,
        model=decision.model,
        max_tokens=gen_req.max_tokens,
        temperature=gen_req.temperature,
    )

    # First chunk announces the assistant role (OpenAI convention).
    yield _sse(
        ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=decision.model,
            choices=[base_choice(ChatChunkDelta(role="assistant"))],
        )
    )

    started = time.perf_counter()
    text_len = 0
    for delta in provider.stream(stream_req):
        if not delta:
            continue
        text_len += len(delta)
        yield _sse(
            ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=decision.model,
                choices=[base_choice(ChatChunkDelta(content=delta))],
            )
        )
    latency_ms = (time.perf_counter() - started) * 1000.0

    # Estimate tokens from streamed text (~4 chars/token) without a second tokenizer pass.
    completion_tokens = max(1, text_len // 4)
    prompt_tokens = max(1, sum(len(m.content) for m in gen_req.messages) // 4)
    served_by = "remote" if decision.would_escalate else "local"
    if served_by == "remote":
        router.budget.spend(prompt_tokens + completion_tokens)
        saved = 0
    else:
        saved = estimated_tokens_saved(decision.task_class, prompt_tokens, completion_tokens)

    router.metrics.record(
        RequestRecord(
            task_class=decision.task_class,
            backend=provider.name,
            model=decision.model,
            served_by=served_by,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            escalated=decision.would_escalate,
            escalation_reason=decision.reason if decision.would_escalate else None,
            adapter=adapter,
            estimated_frontier_tokens_saved=saved,
        )
    )

    yield _sse(
        ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=decision.model,
            choices=[base_choice(ChatChunkDelta(), finish="stop")],
            hearth=HearthTelemetry(
                served_by=served_by,
                backend=provider.name,
                model=decision.model,
                adapter=adapter,
                escalated=decision.would_escalate,
                estimated_frontier_tokens_saved=saved,
            ),
        )
    )
    yield _sse("[DONE]")


def _resolve_stream_provider(
    router: Router,
    gen_req: GenRequest,
    intent: str | None,
    allow_escalation: bool,
):
    """Decide the route and return ``(decision, provider)`` for streaming.

    Mirrors :meth:`Router.route`'s provider selection so streaming and non-streaming
    share the same escalation + budget semantics.
    """
    decision = router.decide(gen_req, intent=intent, allow_escalation=allow_escalation)
    if decision.would_escalate and decision.backend == "remote":
        remote_cfg = router.policy.remote_for()
        from ..router.route import _estimate_remote_cost

        if remote_cfg is None or not router.budget.can_afford(_estimate_remote_cost(gen_req)):
            raise BudgetExhaustedError(
                "remote budget exhausted; escalation denied"
                if remote_cfg is not None
                else "no remote configured for escalation"
            )
        return decision, router._make_remote(remote_cfg)
    return decision, router.local


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
