"""Deterministic stub provider.

Exists so the walking skeleton runs end-to-end with no models downloaded and no MLX
installed — used by the test suite and by ``hearth serve`` when the real backend is
unavailable. It never touches the network or the GPU; it echoes a summary of the input.
"""

from __future__ import annotations

from .base import Capabilities, GenRequest, GenResult, ResourceEstimate


class EchoProvider:
    """A no-op provider that returns a deterministic response derived from the prompt."""

    name = "echo"

    def capabilities(self) -> Capabilities:
        return Capabilities(chat=True, embed=False, stream=False, adapters=False)

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

    def footprint(self, model_id: str) -> ResourceEstimate:
        return ResourceEstimate(ram_gb=0.0)


def _approx_tokens(text: str) -> int:
    """Cheap ~4-chars-per-token estimate; good enough for skeleton telemetry."""
    return max(1, len(text) // 4)
