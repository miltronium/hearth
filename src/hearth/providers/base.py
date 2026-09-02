"""Provider abstraction — the interface every inference backend implements.

This is ADR-004: the router and gateway only ever see :class:`ModelProvider`, so new
backends (Ollama, Core ML, Foundation Models, remote) drop in without touching upper
layers. Phase 0 ships two providers: :mod:`hearth.providers.echo` (deterministic stub)
and :mod:`hearth.providers.mlx` (real Apple Silicon inference).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

# Why a generation ended, in OpenAI's vocabulary. ``"length"`` means the backend stopped
# because it hit ``max_tokens`` — the caller is holding a *truncated* answer and must be
# able to tell. Reporting ``"stop"`` for a capped generation is the silent-truncation bug
# this type exists to make impossible.
FinishReason = Literal["stop", "length"]

FINISH_STOP: FinishReason = "stop"
FINISH_LENGTH: FinishReason = "length"


def normalize_finish_reason(raw: str | None) -> FinishReason:
    """Map a backend's stop reason onto :data:`FinishReason`.

    Backends spell truncation differently (mlx-lm says ``"length"``, Anthropic says
    ``"max_tokens"``); everything else — an EOS token, a stop sequence, a stream that
    ended before the backend reported anything — is a natural stop.
    """
    return FINISH_LENGTH if raw in ("length", "max_tokens") else FINISH_STOP


@dataclass(frozen=True)
class Message:
    """A single chat message."""

    role: str
    content: str


@dataclass(frozen=True)
class GenRequest:
    """A generation request handed to a provider."""

    messages: list[Message]
    model: str
    max_tokens: int = 512
    temperature: float = 0.7
    # Resolved LoRA adapter path to layer over the base weights for this request, or None
    # for base weights. The router resolves an adapter *id* to this path (Phase 4); a
    # provider without adapter support ignores it.
    adapter: str | None = None


@dataclass(frozen=True)
class GenResult:
    """A completed (non-streaming) generation."""

    text: str
    model: str
    backend: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Why generation ended. Defaults to a natural stop so a provider that cannot tell
    # keeps working; every provider that *can* tell is expected to report honestly.
    finish_reason: FinishReason = FINISH_STOP


@dataclass(frozen=True)
class StreamDelta:
    """One event from a streaming generation.

    Text deltas carry ``text`` and no reason; the single terminal event carries the
    :data:`FinishReason` and no text. Splitting them this way lets a streaming backend
    report truncation the same way :class:`GenResult` does, so the streaming and
    non-streaming paths cannot drift.
    """

    text: str = ""
    finish_reason: FinishReason | None = None


@dataclass(frozen=True)
class Capabilities:
    """What a provider can do — used later by the router for backend selection."""

    chat: bool = False
    embed: bool = False
    stream: bool = False
    adapters: bool = False


@dataclass(frozen=True)
class ResourceEstimate:
    """Rough footprint of a loaded model, for memory-aware scheduling (Phase 7)."""

    ram_gb: float = 0.0
    extra: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class ModelProvider(Protocol):
    """The single interface all inference backends implement.

    Phase 0 only exercises :meth:`generate` and :meth:`capabilities`; the rest are part
    of the stable contract that later phases (embeddings, memory scheduling) rely on.
    """

    name: str

    def capabilities(self) -> Capabilities:
        """Return what this provider supports."""
        ...

    def generate(self, req: GenRequest) -> GenResult:
        """Run a (non-streaming) chat completion."""
        ...

    def stream(self, req: GenRequest) -> Iterator[str]:
        """Yield generated text incrementally (one delta per chunk)."""
        ...

    def footprint(self, model_id: str) -> ResourceEstimate:
        """Estimate the resource footprint of ``model_id`` under this backend."""
        ...

    # Optional extension, deliberately *not* part of the structural contract above so
    # third-party providers keep satisfying it: a backend that knows why streaming ended
    # also implements ``stream_deltas(req) -> Iterator[StreamDelta]``. Callers reach it
    # through :func:`iter_stream`, which falls back to :meth:`stream` for the rest.


def iter_stream(provider: ModelProvider, req: GenRequest) -> Iterator[StreamDelta]:
    """Stream ``req`` through ``provider`` as :class:`StreamDelta` events.

    Prefers the provider's ``stream_deltas`` (which reports the real stop reason) and
    falls back to the plain :meth:`~ModelProvider.stream` contract, terminating that with
    a natural stop — the most a text-only stream can honestly claim.
    """
    rich = getattr(provider, "stream_deltas", None)
    if rich is not None:
        yield from rich(req)
        return
    for text in provider.stream(req):
        yield StreamDelta(text=text)
    yield StreamDelta(finish_reason=FINISH_STOP)
