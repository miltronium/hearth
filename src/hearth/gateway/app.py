"""FastAPI application factory and routes.

Serves the OpenAI-compatible core (``/v1/chat/completions`` incl. streaming,
``/v1/embeddings`` stub, ``/v1/models``) plus the ``/v1/hearth/`` admin surface. A local
bearer token (see :mod:`hearth.gateway.auth`) gates every route except the liveness probe
``/v1/hearth/admin/health``.
"""

from __future__ import annotations

import time
import uuid

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from .. import __version__
from ..config import Settings, get_settings
from ..providers import select_provider
from ..providers.base import GenRequest, Message, ModelProvider
from ..registry import Registry, get_registry
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
    Usage,
)


def create_app(
    provider: ModelProvider | None = None,
    settings: Settings | None = None,
    registry: Registry | None = None,
) -> FastAPI:
    """Build the HEARTH FastAPI app. Pass ``provider`` to inject a stub in tests."""
    settings = settings or get_settings()
    provider = provider or select_provider(settings)
    registry = registry or get_registry()

    app = FastAPI(title="HEARTH", version=__version__)
    app.state.provider = provider
    app.state.settings = settings
    app.state.registry = registry

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
        model = registry.default_id if req.model in ("auto", "") else req.model
        gen_req = GenRequest(
            messages=[Message(role=m.role, content=m.content) for m in req.messages],
            model=model,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
        if req.stream:
            return StreamingResponse(
                _stream_sse(provider, gen_req),
                media_type="text/event-stream",
            )

        result = provider.generate(gen_req)
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
                served_by="local",
                backend=result.backend,
                model=result.model,
                # Phase 0 estimate: everything served locally would otherwise have been
                # a frontier call. Phase 2 replaces this with a class-aware multiplier.
                estimated_frontier_tokens_saved=total,
            ),
        )

    return app


def _sse(payload: object) -> str:
    """Serialize one payload as an SSE ``data:`` event."""
    if isinstance(payload, str):
        return f"data: {payload}\n\n"
    return f"data: {payload.model_dump_json(exclude_none=True)}\n\n"


def _stream_sse(provider: ModelProvider, gen_req: GenRequest):
    """Yield OpenAI-compatible SSE chunks, then a final hearth chunk, then ``[DONE]``.

    The completion-token count is estimated from the streamed text (~4 chars/token) so the
    telemetry block stays populated without a second tokenizer pass over the stream.
    """
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def base_choice(delta: ChatChunkDelta, finish: str | None = None) -> ChatChunkChoice:
        return ChatChunkChoice(delta=delta, finish_reason=finish)

    # First chunk announces the assistant role (OpenAI convention).
    yield _sse(
        ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=gen_req.model,
            choices=[base_choice(ChatChunkDelta(role="assistant"))],
        )
    )

    text_len = 0
    for delta in provider.stream(gen_req):
        if not delta:
            continue
        text_len += len(delta)
        yield _sse(
            ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=gen_req.model,
                choices=[base_choice(ChatChunkDelta(content=delta))],
            )
        )

    # Final content chunk: empty delta + finish_reason + the hearth telemetry block.
    completion_tokens = max(1, text_len // 4)
    yield _sse(
        ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=gen_req.model,
            choices=[base_choice(ChatChunkDelta(), finish="stop")],
            hearth=HearthTelemetry(
                served_by="local",
                backend=provider.name,
                model=gen_req.model,
                estimated_frontier_tokens_saved=completion_tokens,
            ),
        )
    )
    yield _sse("[DONE]")
