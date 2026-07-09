"""OpenAI-compatible request/response schemas (the subset Phase 0 implements).

Kept intentionally minimal: enough for any OpenAI SDK to call ``/v1/chat/completions``
and ``/v1/models``. The additive ``hearth`` telemetry block rides along on responses
(see docs/API.md).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class HearthRequestOptions(BaseModel):
    """Optional, HEARTH-specific request hints. Ignored by pure OpenAI clients."""

    intent: str | None = None
    allow_escalation: bool = True
    adapter: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str = "auto"
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = False
    hearth: HearthRequestOptions | None = None


class HearthTelemetry(BaseModel):
    """Additive block reporting how the request was served."""

    served_by: Literal["local", "remote"] = "local"
    backend: str
    model: str
    adapter: str | None = None
    escalated: bool = False
    estimated_frontier_tokens_saved: int = 0


class ChatChoiceMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatChoiceMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: Usage
    hearth: HearthTelemetry


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: str = "hearth"
    backend: str | None = None
    context: int | None = None
    capabilities: list[str] = Field(default_factory=list)


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard] = Field(default_factory=list)


class ChatChunkDelta(BaseModel):
    """A streamed delta. ``role`` appears only on the first chunk (OpenAI convention)."""

    role: Literal["assistant"] | None = None
    content: str | None = None


class ChatChunkChoice(BaseModel):
    index: int = 0
    delta: ChatChunkDelta = Field(default_factory=ChatChunkDelta)
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    """One ``chat.completion.chunk`` SSE event. The final chunk carries ``hearth``."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatChunkChoice]
    hearth: HearthTelemetry | None = None


class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embeddings request. ``input`` accepts a string or a list."""

    model: str = "auto"
    input: str | list[str]


class EmbeddingData(BaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: list[float]
    index: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingData] = Field(default_factory=list)
    model: str
    usage: EmbeddingUsage


class RagChunkSpec(BaseModel):
    """Chunking controls for ingest (docs/API.md)."""

    size: int = 800
    overlap: int = 100


class RagIngestRequest(BaseModel):
    """Ingest one or more paths into a named collection (docs/API.md)."""

    collection: str
    paths: list[str]
    chunk: RagChunkSpec = Field(default_factory=RagChunkSpec)


class RagIngestResponse(BaseModel):
    collection: str
    files: int
    chunks: int


class RagQueryRequest(BaseModel):
    """Query a collection; ``answer`` toggles retrieve-then-answer (docs/API.md)."""

    collection: str
    query: str
    k: int = 6
    answer: bool = False


class RagChunk(BaseModel):
    text: str
    source: str
    score: float


class RagQueryResponse(BaseModel):
    chunks: list[RagChunk] = Field(default_factory=list)
    answer: str | None = None


class RouteRequest(BaseModel):
    """Dry-run routing request for ``POST /v1/hearth/route``."""

    messages: list[ChatMessage]
    intent: str | None = None
    allow_escalation: bool = True


class RouteResponse(BaseModel):
    """What the router *would* do, without executing (docs/API.md)."""

    task_class: str = Field(alias="class")
    method: str
    backend: str
    model: str
    would_escalate: bool
    reason: str
    confidence: float | None = None

    model_config = {"populate_by_name": True}
