"""A minimal in-repo HEARTH provider plugin (Phase 7 reference + test fixture).

Demonstrates the zero-core-edit extension path: this package declares a
``hearth.providers`` entry point (see ``pyproject.toml``) whose target is the
:func:`build` factory below. Once the package is installed, ``HEARTH_BACKEND=hello``
resolves this provider through HEARTH's existing ``select_provider`` — no edits to
HEARTH's source.

The provider satisfies :class:`hearth.providers.base.ModelProvider` structurally (it is a
runtime-checkable Protocol), so it only needs the right attributes/methods, not a base
class. It does not import anything from HEARTH, keeping the plugin fully decoupled.

This example is a *reference*; it is intentionally NOT installed by the repo. HEARTH's
test suite installs an equivalent provider via a simulated entry point.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field


@dataclass(frozen=True)
class _Capabilities:
    chat: bool = True
    embed: bool = False
    stream: bool = True
    adapters: bool = False


@dataclass(frozen=True)
class _ResourceEstimate:
    ram_gb: float = 0.0
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _GenResult:
    text: str
    model: str
    backend: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class HelloProvider:
    """A trivial provider that greets the last user message. No models, no network."""

    name = "hello"

    def capabilities(self) -> _Capabilities:
        return _Capabilities()

    def generate(self, req: object) -> _GenResult:
        last = next(
            (m.content for m in reversed(req.messages) if m.role == "user"),  # type: ignore[attr-defined]
            "",
        )
        text = f"Hello from the example plugin! You said: {last.strip()}"
        return _GenResult(
            text=text,
            model=getattr(req, "model", "hello"),
            backend=self.name,
            prompt_tokens=max(1, len(last) // 4),
            completion_tokens=max(1, len(text) // 4),
        )

    def stream(self, req: object) -> Iterator[str]:
        yield self.generate(req).text

    def footprint(self, model_id: str) -> _ResourceEstimate:
        return _ResourceEstimate(ram_gb=0.0)


def build() -> HelloProvider:
    """Entry-point factory: return a ready :class:`HelloProvider` instance.

    HEARTH's plugin loader calls this with no arguments (see
    :func:`hearth.plugins.load_plugin`).
    """
    return HelloProvider()
