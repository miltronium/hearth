"""FastAPI application factory and routes.

Phase 0 serves the OpenAI-compatible core (``/v1/chat/completions``, ``/v1/models``)
plus a couple of ``/v1/hearth/`` stubs that later phases flesh out. Auth is deferred to
Phase 1 per the roadmap, so these routes are open on the loopback interface.
"""

from __future__ import annotations

import time
import uuid

from fastapi import FastAPI

from .. import __version__
from ..config import Settings, get_settings
from ..providers import select_provider
from ..providers.base import GenRequest, Message, ModelProvider
from .schemas import (
    ChatChoice,
    ChatChoiceMessage,
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
) -> FastAPI:
    """Build the HEARTH FastAPI app. Pass ``provider`` to inject a stub in tests."""
    settings = settings or get_settings()
    provider = provider or select_provider(settings)

    app = FastAPI(title="HEARTH", version=__version__)
    app.state.provider = provider
    app.state.settings = settings

    @app.get("/v1/hearth/admin/health")
    def health() -> dict:
        return {
            "status": "ok",
            "version": __version__,
            "backend": provider.name,
            "model": settings.default_model,
        }

    @app.get("/v1/models")
    def list_models() -> ModelList:
        return ModelList(data=[ModelCard(id=settings.default_model)])

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest) -> ChatCompletionResponse:
        model = settings.default_model if req.model in ("auto", "") else req.model
        gen_req = GenRequest(
            messages=[Message(role=m.role, content=m.content) for m in req.messages],
            model=model,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
        result = provider.generate(gen_req)

        total = result.prompt_tokens + result.completion_tokens
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
            created=int(time.time()),
            model=result.model,
            choices=[
                ChatChoice(message=ChatChoiceMessage(content=result.text)),
            ],
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
