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
    keeps exactly one resident base model; ADR-003). LoRA adapters are hot-swappable per
    request (Phase 4): :meth:`generate`/:meth:`stream` accept an optional ``adapter`` path
    that layers over the base weights; each distinct adapter path is loaded once and cached
    alongside the base so switching between them is cheap (ARCHITECTURE §5).
    """

    name = "mlx"

    def __init__(self, model_id: str, adapter: str | None = None) -> None:
        self.model_id = model_id
        self.adapter = adapter
        self._model = None
        self._tokenizer = None
        # adapter path -> (model, tokenizer); the base (no adapter) is keyed by "".
        self._cache: dict[str, tuple[object, object]] = {}

    def capabilities(self) -> Capabilities:
        return Capabilities(chat=True, embed=False, stream=True, adapters=True)

    def _load_variant(self, adapter: str | None):
        """Load (and cache) the (model, tokenizer) for ``adapter`` (``None`` = base weights).

        A distinct ``adapter_path`` layers a LoRA adapter over the base; loaded once per
        path and cached so per-request hot-swap is a dict lookup after the first use.
        """
        key = adapter or ""
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if not mlx_available():
            raise MLXUnavailableError(
                "mlx-lm is not installed. Install the backend with: uv sync --extra mlx"
            )
        from mlx_lm import load  # deferred heavy import

        kwargs = {"adapter_path": adapter} if adapter else {}
        loaded = load(self.model_id, **kwargs)
        self._cache[key] = loaded
        return loaded

    def _ensure_loaded(self, adapter: str | None = None) -> None:
        """Ensure the variant for ``adapter`` (or the provider default) is the active one."""
        selected = adapter if adapter is not None else self.adapter
        self._model, self._tokenizer = self._load_variant(selected)

    def generate(self, req: GenRequest) -> GenResult:
        self._ensure_loaded(req.adapter)
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
        self._ensure_loaded(req.adapter)
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
