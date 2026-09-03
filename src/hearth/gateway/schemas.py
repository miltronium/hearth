"""OpenAI-compatible request/response schemas (the subset Phase 0 implements).

Kept intentionally minimal: enough for any OpenAI SDK to call ``/v1/chat/completions``
and ``/v1/models``. The additive ``hearth`` telemetry block rides along on responses
(see docs/API.md).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class HearthRequestOptions(BaseModel):
    """Optional, HEARTH-specific request hints. Ignored by pure OpenAI clients."""

    intent: str | None = None
    allow_escalation: bool = True
    adapter: str | None = None


class ResponseFormat(BaseModel):
    """OpenAI ``response_format``. ``type`` is kept a plain ``str`` on purpose.

    A ``Literal`` would reject an unsupported value as a generic 422 body-validation
    error; the route checks it instead and answers with a named error saying *which*
    formats HEARTH implements (see :mod:`hearth.gateway.json_mode`). Either way the field
    can no longer be dropped on the floor, which is what silently ignoring it amounted to.
    """

    type: str = "text"


class ChatCompletionRequest(BaseModel):
    model: str = "auto"
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = False
    response_format: ResponseFormat | None = None
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
    # Required, deliberately: this used to default to "stop", so a generation cut off at
    # max_tokens was reported to the caller as a clean completion. There is no safe default
    # for "why did this end" — the provider's real reason must be threaded in.
    finish_reason: str


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


# --- the agent route (POST /v1/hearth/agent, docs/AGENT.md §8) --------------------------
#
# What is *absent* here is the design. There is no `vetted_only`, no `tools`, no `guidance`
# and no root/path field, because every one of those would let a wire request widen what the
# agent may reach. The security posture is a property of the server's code, not of the
# request body, and `extra="forbid"` makes an attempt to smuggle one in a loud 422 rather
# than a silently ignored field that a reader might mistake for an honoured one.


class AgentBudgetRequest(BaseModel):
    """A client's *requested* bounds. Every one is clamped against a server ceiling.

    Requested, not applied: :mod:`hearth.gateway.agent_route` reduces each value to its
    ceiling and reports what it actually used in :class:`AgentBudgetApplied`. A caller asking
    for ten thousand iterations gets the cap and is told so — the alternative is a wire field
    that can pin a local machine's GPU for an afternoon.
    """

    max_iterations: int | None = Field(default=None, ge=1)
    max_seconds: float | None = Field(default=None, gt=0)
    max_total_tokens: int | None = Field(default=None, ge=1)

    model_config = {"extra": "forbid"}


class AgentRunRequest(BaseModel):
    """The body of ``POST /v1/hearth/agent``: a task, and at most a smaller budget."""

    task: str
    model: str = "auto"
    budget: AgentBudgetRequest | None = None

    model_config = {"extra": "forbid"}

    @field_validator("task")
    @classmethod
    def _task_is_not_blank(cls, value: str) -> str:
        # The loop raises AgentConfigError on an empty task; refusing here turns that into a
        # 422 naming the field rather than a 500 from inside a streaming response.
        if not value.strip():
            raise ValueError("task must not be empty")
        return value


class AgentBudgetApplied(BaseModel):
    """The bounds the run actually used, plus the names of the ones that were clamped.

    ``clamped`` exists so the reduction is visible in the stream. A budget silently smaller
    than the one requested makes a ``max_iterations`` stop look like a server bug; naming the
    clamp makes it an expected outcome the operator can act on.
    """

    max_iterations: int
    max_seconds: float
    max_total_tokens: int
    clamped: list[str] = Field(default_factory=list)


class AgentStartEvent(BaseModel):
    """The first SSE event: what this run may reach, before it spends any of its budget.

    Emitted *before* the first generation on purpose. If ``HEARTH_FILE_ROOTS`` names nothing,
    the file tools are deny-by-default and the run cannot read anything — the operator should
    learn that in the first event, not infer it from eight wasted steps.
    """

    object: Literal["hearth.agent.start"] = "hearth.agent.start"
    task: str
    tools: list[str] = Field(default_factory=list)
    #: Not configurable from the wire. Serialised so the stream *shows* the posture rather
    #: than leaving the client to assume it.
    vetted_only: Literal[True] = True
    budget: AgentBudgetApplied
    file_roots: int = 0
    warnings: list[str] = Field(default_factory=list)


class AgentStepEvent(BaseModel):
    """One step of the run, as the loop recorded it.

    Mirrors :class:`hearth.agent.Step` field for field (minus the raw model output, which is
    the model's private scratch and can be long) so the operator sees which tool ran with
    which arguments and what came back — a conclusion that cannot be traced to its steps is a
    claim, not a result.
    """

    object: Literal["hearth.agent.step"] = "hearth.agent.step"
    index: int
    kind: str
    thought: str = ""
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    observation: str | None = None
    #: True when the observation was shortened for the wire. Never silent: the text itself
    #: also carries the marker, so a client that ignores this field still sees the cut.
    observation_truncated: bool = False
    error: str | None = None
    model: str = ""
    backend: str = ""
    model_seconds: float = 0.0
    tool_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class AgentRunEvent(BaseModel):
    """The terminal SSE event: why the run stopped, and the answer only if it answered.

    :class:`hearth.agent.AgentRun` makes "ran out of budget" unrepresentable as an answer;
    this re-enforces the same rule at the presentation layer, because a type invariant that
    is dropped on the way out of the process is not an invariant. That is the
    ``finish_reason: "stop"`` lesson (``CLAUDE.md`` §3) — the place it was previously lost was
    exactly here, on the boundary between a result object and its serialisation.
    """

    object: Literal["hearth.agent.run"] = "hearth.agent.run"
    stopped_reason: str
    completed: bool
    answer: str | None = None
    detail: str = ""
    steps: int = 0
    elapsed_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    budget: AgentBudgetApplied
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _answer_requires_completion(self) -> AgentRunEvent:
        if self.completed and not (self.answer or "").strip():
            raise ValueError("a completed run must carry the answer text")
        if not self.completed and self.answer is not None:
            raise ValueError(
                f"a run that stopped as {self.stopped_reason!r} must not carry an answer; "
                "a partial trace serialised as a result is the failure this check exists to "
                "prevent"
            )
        return self
