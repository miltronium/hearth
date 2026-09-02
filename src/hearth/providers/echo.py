"""Deterministic stub provider.

Exists so the walking skeleton runs end-to-end with no models downloaded and no MLX
installed — used by the test suite and by ``hearth serve`` when the real backend is
unavailable. It never touches the network or the GPU; it echoes a summary of the input.
"""

from __future__ import annotations

from collections.abc import Iterator

from .base import (
    FINISH_LENGTH,
    FINISH_STOP,
    Capabilities,
    FinishReason,
    GenRequest,
    GenResult,
    ResourceEstimate,
    StreamDelta,
)


class EchoProvider:
    """A no-op provider that returns a deterministic response derived from the prompt."""

    name = "echo"

    def capabilities(self) -> Capabilities:
        return Capabilities(chat=True, embed=False, stream=True, adapters=False)

    def generate(self, req: GenRequest) -> GenResult:
        text, finish_reason = self._echo(req)
        last_user = next(
            (m.content for m in reversed(req.messages) if m.role == "user"),
            "",
        )
        return GenResult(
            text=text,
            model=req.model,
            backend=self.name,
            prompt_tokens=_approx_tokens(last_user),
            completion_tokens=_approx_tokens(text),
            finish_reason=finish_reason,
        )

    def stream(self, req: GenRequest) -> Iterator[str]:
        """Yield the echoed text word-by-word (whitespace re-attached to each word)."""
        for delta in self.stream_deltas(req):
            if delta.text:
                yield delta.text

    def stream_deltas(self, req: GenRequest) -> Iterator[StreamDelta]:
        """Stream the echoed text, then report the same stop reason :meth:`generate` does."""
        text, finish_reason = self._echo(req)
        words = text.split(" ")
        for i, word in enumerate(words):
            yield StreamDelta(text=word if i == 0 else " " + word)
        yield StreamDelta(finish_reason=finish_reason)

    def footprint(self, model_id: str) -> ResourceEstimate:
        return ResourceEstimate(ram_gb=0.0)

    def _echo(self, req: GenRequest) -> tuple[str, FinishReason]:
        """Build the echo text and the honest stop reason for it.

        The stub obeys ``max_tokens`` under the same ~4-chars-per-token estimate it bills
        with: an echo that doesn't fit is cut at the cap and reported as ``"length"``. That
        makes the truncation path reproducible end-to-end with no model downloaded — the
        one thing the skeleton has to be able to rehearse.
        """
        last_user = next(
            (m.content for m in reversed(req.messages) if m.role == "user"),
            "",
        )
        text = f"[echo] {last_user.strip()}"
        budget = max(1, req.max_tokens) * _CHARS_PER_TOKEN
        if len(text) > budget:
            return text[:budget], FINISH_LENGTH
        return text, FINISH_STOP


_CHARS_PER_TOKEN = 4


def _approx_tokens(text: str) -> int:
    """Cheap ~4-chars-per-token estimate; good enough for skeleton telemetry."""
    return max(1, len(text) // _CHARS_PER_TOKEN)
