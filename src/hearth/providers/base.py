"""Provider abstraction — the interface every inference backend implements.

This is ADR-004: the router and gateway only ever see :class:`ModelProvider`, so new
backends (Ollama, Core ML, Foundation Models, remote) drop in without touching upper
layers. Phase 0 ships two providers: :mod:`hearth.providers.echo` (deterministic stub)
and :mod:`hearth.providers.mlx` (real Apple Silicon inference).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


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


@dataclass(frozen=True)
class GenResult:
    """A completed (non-streaming) generation."""

    text: str
    model: str
    backend: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


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
