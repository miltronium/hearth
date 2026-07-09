"""MLX inference provider — the real Apple Silicon backend (ADR-003).

Wraps ``mlx-lm``. Import of the heavy dependency is deferred to load time so that the
package (and the echo-backed skeleton) works without the ``mlx`` extra installed.
Install it with: ``uv sync --extra mlx``.
"""

from __future__ import annotations

from collections.abc import Iterator

from .base import Capabilities, GenRequest, GenResult, ResourceEstimate


class MLXUnavailableError(RuntimeError):
    """Raised when the MLX backend is requested but ``mlx-lm`` isn't importable."""


def mlx_available() -> bool:
    """True if ``mlx-lm`` can be imported in this environment."""
    import importlib.util

    return importlib.util.find_spec("mlx_lm") is not None


class MLXProvider:
    """Loads a single model via ``mlx-lm`` and serves streaming/non-streaming completions.

    The model is loaded lazily on first use and cached for the process lifetime (Phase 0
    keeps exactly one resident model; ADR-003). An optional LoRA ``adapter`` path is wired
    through :meth:`load` for Phase 4; it is ``None`` (base weights only) by default.
    """

    name = "mlx"

    def __init__(self, model_id: str, adapter: str | None = None) -> None:
        self.model_id = model_id
        self.adapter = adapter
        self._model = None
        self._tokenizer = None

    def capabilities(self) -> Capabilities:
        return Capabilities(chat=True, embed=False, stream=True, adapters=True)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not mlx_available():
            raise MLXUnavailableError(
                "mlx-lm is not installed. Install the backend with: uv sync --extra mlx"
            )
        from mlx_lm import load  # deferred heavy import

        # ``adapter_path`` layers a LoRA adapter over the base weights when provided;
        # unused by default (Phase 4 populates it from the adapter registry).
        kwargs = {"adapter_path": self.adapter} if self.adapter else {}
        self._model, self._tokenizer = load(self.model_id, **kwargs)

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
        text = self._strip_terminators(text)
        return GenResult(
            text=text.strip(),
            model=self.model_id,
            backend=self.name,
            prompt_tokens=len(self._tokenizer.encode(prompt)),
            completion_tokens=len(self._tokenizer.encode(text)),
        )

    def stream(self, req: GenRequest) -> Iterator[str]:
        """Yield decoded text deltas as ``mlx_lm.stream_generate`` produces them.

        A trailing chat terminator (e.g. ``<|im_end|>``) is buffered and stripped so it
        never reaches the client — matching :meth:`generate`'s terminator handling.
        """
        self._ensure_loaded()
        from mlx_lm import stream_generate

        prompt = self._format_prompt(req.messages)
        pending = ""
        markers = self._terminator_markers()
        max_marker = max((len(m) for m in markers), default=0)
        for response in stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=req.max_tokens,
        ):
            pending += response.text
            # Emit everything except a tail that could still be the start of a marker.
            safe = len(pending) - max_marker
            if safe > 0:
                yield pending[:safe]
                pending = pending[safe:]
        yield self._strip_terminators(pending)

    def footprint(self, model_id: str) -> ResourceEstimate:
        # Refined later from the registry; a 7B 4-bit model is ~4.5 GB resident.
        return ResourceEstimate(ram_gb=4.5)

    def _terminator_markers(self) -> list[str]:
        markers = ["<|im_end|>", "<|endoftext|>", "<|eot_id|>"]
        eos = getattr(self._tokenizer, "eos_token", None)
        if eos:
            markers.append(eos)
        return markers

    def _strip_terminators(self, text: str) -> str:
        """Remove a trailing chat end-of-turn / EOS token the decoder may emit verbatim.

        Some chat templates decode the terminator (e.g. ``<|im_end|>``) into the output
        string instead of stopping before it; trim it so callers get clean text.
        """
        for m in self._terminator_markers():
            if m and text.rstrip().endswith(m):
                text = text.rstrip()[: -len(m)]
        return text

    def _format_prompt(self, messages: list) -> str:
        """Render messages via the tokenizer's chat template when available."""
        chat = [{"role": m.role, "content": m.content} for m in messages]
        tmpl = getattr(self._tokenizer, "apply_chat_template", None)
        if tmpl is not None:
            return tmpl(chat, tokenize=False, add_generation_prompt=True)
        # Fallback for tokenizers without a chat template.
        return "\n".join(f"{m.role}: {m.content}" for m in messages) + "\nassistant:"
