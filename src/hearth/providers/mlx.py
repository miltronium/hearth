"""MLX inference provider — the real Apple Silicon backend (ADR-003).

Wraps ``mlx-lm``. Import of the heavy dependency is deferred to load time so that the
package (and the echo-backed skeleton) works without the ``mlx`` extra installed.
Install it with: ``uv sync --extra mlx``.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from ..config import get_settings
from .base import (
    FINISH_STOP,
    Capabilities,
    FinishReason,
    GenRequest,
    GenResult,
    ResourceEstimate,
    StreamDelta,
    normalize_finish_reason,
)


class MLXUnavailableError(RuntimeError):
    """Raised when the MLX backend is requested but ``mlx-lm`` isn't importable."""


def mlx_available() -> bool:
    """True if ``mlx-lm`` can be imported in this environment."""
    import importlib.util

    return importlib.util.find_spec("mlx_lm") is not None


def resolve_local_model(model_id: str) -> str:
    """Return a local snapshot path for ``model_id`` if HEARTH has one, else ``model_id``.

    ``hearth models pull`` downloads into ``settings.models_dir`` (``~/.hearth/models``),
    but ``mlx_lm.load`` asks huggingface_hub, which resolves ``HF_HUB_CACHE`` ->
    ``HF_HOME/hub`` -> ``~/.cache/huggingface/hub`` and knows nothing about HEARTH. The two
    halves never agreed: a freshly pulled model was invisible to the provider unless the
    operator happened to export ``HF_HUB_CACHE`` themselves, so ``hearth models pull X``
    followed by a run would report X missing — or quietly serve a *different* model that
    happened to sit in the default cache.

    Resolution is explicit rather than by mutating ``os.environ``: a hidden global would
    make this module disagree with anything that inspects the environment (such as
    ``scripts/hearth_status.py``), which is the class of bug this is fixing. HEARTH's own
    directory is checked first; anything else falls through to huggingface_hub's normal
    resolution, so a model in the default cache still loads and an operator who set
    ``HF_HUB_CACHE`` deliberately is unaffected.
    """
    try:  # deferred: huggingface_hub is only present with the mlx/embeddings extras
        from huggingface_hub import snapshot_download
    except ImportError:
        return model_id
    models_dir = get_settings().models_dir
    if not models_dir.is_dir():
        return model_id
    try:
        return snapshot_download(
            repo_id=model_id, cache_dir=str(models_dir), local_files_only=True
        )
    except Exception:
        # Not in HEARTH's directory (or not a repo id at all — it may already be a path).
        return model_id


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
        loaded = load(resolve_local_model(self.model_id), **kwargs)
        self._ensure_stop_tokens(loaded[1])
        self._cache[key] = loaded
        return loaded

    @staticmethod
    def _ensure_stop_tokens(tokenizer) -> None:
        """Make generation stop at the chat-template turn terminator.

        Root cause of the LoRA "ramble" (see ``_clean_stream``): mlx-lm's ``generate`` stops
        only on ``tokenizer.eos_token_ids`` — for Qwen that's ``<|endoftext|>`` (151643) — but
        the chat template ends an assistant turn with a *different* token, ``<|im_end|>``
        (151645, the tokenizer's ``eos_token``). A model that emits the turn terminator but not
        the base EOS therefore never stops: it decodes ``<|im_end|>`` to the literal string and
        runs to ``max_tokens``. Base models usually emit ``<|endoftext|>`` soon after and get
        lucky; a tuned adapter can loop the turn terminator forever. Adding ``eos_token_id`` to
        the stop set fixes it at the source — generation ends cleanly at end-of-turn (and stops
        burning tokens on the ramble). Best-effort: a tokenizer without a mutable
        ``eos_token_ids`` set is left as-is (the ``_strip_terminators`` safety net still runs).
        """
        eos_id = getattr(tokenizer, "eos_token_id", None)
        stop = getattr(tokenizer, "eos_token_ids", None)
        if eos_id is None or stop is None:
            return
        try:
            stop.add(eos_id)
        except AttributeError:
            pass

    def _ensure_loaded(self, adapter: str | None = None) -> None:
        """Ensure the variant for ``adapter`` (or the provider default) is the active one."""
        selected = adapter if adapter is not None else self.adapter
        self._model, self._tokenizer = self._load_variant(selected)

    def generate(self, req: GenRequest) -> GenResult:
        """Run a completion, reporting whether it ended at EOS or at the ``max_tokens`` cap.

        Built on :meth:`stream_deltas` rather than ``mlx_lm.generate``: the latter returns
        only the decoded string, which is exactly the information loss that let a truncated
        answer be reported as a clean stop.
        """
        deltas = list(self.stream_deltas(req))
        text = "".join(d.text for d in deltas).strip()
        finish_reason = next(
            (d.finish_reason for d in reversed(deltas) if d.finish_reason), FINISH_STOP
        )
        prompt = self._format_prompt(req.messages)
        return GenResult(
            text=text,
            model=self.model_id,
            backend=self.name,
            prompt_tokens=len(self._tokenizer.encode(prompt)),
            completion_tokens=len(self._tokenizer.encode(text)),
            finish_reason=finish_reason,
        )

    def stream(self, req: GenRequest) -> Iterator[str]:
        """Yield decoded text deltas as ``mlx_lm.stream_generate`` produces them.

        Chat terminators are handled exactly as in :meth:`generate`: streaming stops at the
        first terminator marker and never leaks it — a LoRA-tuned model that fails to stop at
        EOS can emit the literal marker mid-stream (see :meth:`_clean_stream`).
        """
        for delta in self.stream_deltas(req):
            if delta.text:
                yield delta.text

    def stream_deltas(self, req: GenRequest) -> Iterator[StreamDelta]:
        """Stream cleaned text deltas, then the real stop reason from ``mlx-lm``.

        ``mlx_lm.stream_generate`` tags each ``GenerationResponse`` with ``finish_reason``
        (``None`` while generating, then ``"stop"`` at an EOS token or ``"length"`` at the
        ``max_tokens`` cap); we pass that through so the gateway never has to guess. When
        :meth:`_clean_stream` cuts early at a literal terminator marker the loop ends before
        mlx-lm reports anything — that *is* an end-of-turn, so it normalizes to ``"stop"``.
        """
        self._ensure_loaded(req.adapter)
        from mlx_lm import stream_generate

        prompt = self._format_prompt(req.messages)
        # Populated as responses are consumed; read after _clean_stream drains (or cuts).
        raw_reason: list[str | None] = [None]

        def chunks() -> Iterator[str]:
            for response in stream_generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=req.max_tokens,
            ):
                raw_reason[0] = getattr(response, "finish_reason", None)
                yield response.text

        for text in self._clean_stream(chunks()):
            yield StreamDelta(text=text)
        yield StreamDelta(finish_reason=self._finish_reason(raw_reason[0]))

    @staticmethod
    def _finish_reason(raw: str | None) -> FinishReason:
        """Normalize mlx-lm's ``finish_reason`` onto the OpenAI vocabulary."""
        return normalize_finish_reason(raw)

    def _clean_stream(self, chunks: Iterable[str]) -> Iterator[str]:
        """Yield cleaned text deltas from raw model ``chunks`` (pure; no model needed).

        Two jobs, mirroring :meth:`generate`'s cut-at-first-marker: (1) hold back a tail that
        could be the *start* of a terminator so a marker split across chunk boundaries is
        never leaked, and (2) stop emitting at the first *complete* terminator — a LoRA-tuned
        model that fails to stop at EOS emits the literal marker mid-stream and then rambles
        (``QX-2<|im_end|> !<|im_end|> ...``), and streaming clients must not see that.
        """
        pending = ""
        max_marker = max((len(m) for m in self._terminator_markers() if m), default=0)
        for chunk in chunks:
            pending += chunk
            cut = self._first_terminator(pending)
            if cut is not None:
                if cut > 0:
                    yield pending[:cut]
                return
            # No complete marker yet: emit all but a tail that could still start one.
            safe = len(pending) - max_marker
            if safe > 0:
                yield pending[:safe]
                pending = pending[safe:]
        tail = self._strip_terminators(pending)
        if tail:
            yield tail

    def footprint(self, model_id: str) -> ResourceEstimate:
        # Refined later from the registry; a 7B 4-bit model is ~4.5 GB resident.
        return ResourceEstimate(ram_gb=4.5)

    def _terminator_markers(self) -> list[str]:
        markers = ["<|im_end|>", "<|endoftext|>", "<|eot_id|>"]
        eos = getattr(self._tokenizer, "eos_token", None)
        if eos:
            markers.append(eos)
        return markers

    def _first_terminator(self, text: str) -> int | None:
        """Index of the earliest terminator marker in ``text``, or ``None`` if none appear."""
        cut: int | None = None
        for m in self._terminator_markers():
            if not m:
                continue
            idx = text.find(m)
            if idx != -1:
                cut = idx if cut is None else min(cut, idx)
        return cut

    def _strip_terminators(self, text: str) -> str:
        """Cut the output at the first chat end-of-turn / EOS token emitted verbatim.

        Some chat templates decode the terminator (e.g. ``<|im_end|>``) into the output
        string instead of stopping before it. A base model usually stops at the real EOS
        token, but a LoRA-tuned model can emit the *literal* marker mid-stream and then
        ramble (``QX-2<|im_end|> !<|im_end|> ...``). Truncating at the earliest marker
        returns the clean answer in both cases (a bare trailing marker is just the special
        case where the cut is at the end).
        """
        cut = self._first_terminator(text)
        return text if cut is None else text[:cut]

    def _format_prompt(self, messages: list) -> str:
        """Render messages via the tokenizer's chat template when available."""
        chat = [{"role": m.role, "content": m.content} for m in messages]
        tmpl = getattr(self._tokenizer, "apply_chat_template", None)
        if tmpl is not None:
            return tmpl(chat, tokenize=False, add_generation_prompt=True)
        # Fallback for tokenizers without a chat template.
        return "\n".join(f"{m.role}: {m.content}" for m in messages) + "\nassistant:"
