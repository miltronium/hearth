"""Deterministic stub provider.

Exists so the walking skeleton runs end-to-end with no models downloaded and no MLX
installed — used by the test suite and by ``hearth serve`` when the real backend is
unavailable. It never touches the network or the GPU; it echoes a summary of the input.
"""

from __future__ import annotations

from collections.abc import Iterator

from .base import Capabilities, GenRequest, GenResult, ResourceEstimate


class EchoProvider:
    """A no-op provider that returns a deterministic response derived from the prompt."""

    name = "echo"

    def capabilities(self) -> Capabilities:
        return Capabilities(chat=True, embed=False, stream=True, adapters=False)

    def generate(self, req: GenRequest) -> GenResult:
        last_user = next(
            (m.content for m in reversed(req.messages) if m.role == "user"),
            "",
        )
        text = f"[echo] {last_user.strip()}"
        return GenResult(
            text=text,
            model=req.model,
            backend=self.name,
            prompt_tokens=_approx_tokens(last_user),
            completion_tokens=_approx_tokens(text),
        )

    def stream(self, req: GenRequest) -> Iterator[str]:
        """Yield the echoed text word-by-word (whitespace re-attached to each word)."""
        result = self.generate(req)
        words = result.text.split(" ")
        for i, word in enumerate(words):
            yield word if i == 0 else " " + word

    def footprint(self, model_id: str) -> ResourceEstimate:
        return ResourceEstimate(ram_gb=0.0)


def _approx_tokens(text: str) -> int:
    """Cheap ~4-chars-per-token estimate; good enough for skeleton telemetry."""
    return max(1, len(text) // 4)
