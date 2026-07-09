"""MLX inference provider — the real Apple Silicon backend (ADR-003).

Wraps ``mlx-lm``. Import of the heavy dependency is deferred to load time so that the
package (and the echo-backed skeleton) works without the ``mlx`` extra installed.
Install it with: ``uv sync --extra mlx``.
"""

from __future__ import annotations

from .base import Capabilities, GenRequest, GenResult, ResourceEstimate


class MLXUnavailableError(RuntimeError):
    """Raised when the MLX backend is requested but ``mlx-lm`` isn't importable."""


def mlx_available() -> bool:
    """True if ``mlx-lm`` can be imported in this environment."""
    import importlib.util

    return importlib.util.find_spec("mlx_lm") is not None


class MLXProvider:
    """Loads a single model via ``mlx-lm`` and serves non-streaming completions.

    The model is loaded lazily on first :meth:`generate` and cached for the process
    lifetime (Phase 0 keeps exactly one resident model; ADR-003).
    """

    name = "mlx"

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._model = None
        self._tokenizer = None

    def capabilities(self) -> Capabilities:
        return Capabilities(chat=True, embed=False, stream=False, adapters=True)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not mlx_available():
            raise MLXUnavailableError(
                "mlx-lm is not installed. Install the backend with: uv sync --extra mlx"
            )
        from mlx_lm import load  # deferred heavy import

        self._model, self._tokenizer = load(self.model_id)

    def generate(self, req: GenRequest) -> GenResult:
        self._ensure_loaded()
        from mlx_lm import generate as mlx_generate

        prompt = self._format_prompt(req.messages)
        text = mlx_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=req.max_tokens,
            verbose=False,
        )
        return GenResult(
            text=text.strip(),
            model=self.model_id,
            backend=self.name,
            prompt_tokens=len(self._tokenizer.encode(prompt)),
            completion_tokens=len(self._tokenizer.encode(text)),
        )

    def footprint(self, model_id: str) -> ResourceEstimate:
        # Refined later from the registry; a 7B 4-bit model is ~4.5 GB resident.
        return ResourceEstimate(ram_gb=4.5)

    def _format_prompt(self, messages: list) -> str:
        """Render messages via the tokenizer's chat template when available."""
        chat = [{"role": m.role, "content": m.content} for m in messages]
        tmpl = getattr(self._tokenizer, "apply_chat_template", None)
        if tmpl is not None:
            return tmpl(chat, tokenize=False, add_generation_prompt=True)
        # Fallback for tokenizers without a chat template.
        return "\n".join(f"{m.role}: {m.content}" for m in messages) + "\nassistant:"
