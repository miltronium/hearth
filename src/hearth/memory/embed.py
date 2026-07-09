"""Embedding providers — turn text into vectors for RAG (ARCHITECTURE §6, Phase 3).

Mirrors the inference-provider pattern (ADR-004): a small :class:`EmbeddingProvider`
protocol with a dependency-free default (:class:`HashEmbedder`) and an optional real
backend (:class:`MLXEmbedder`) whose heavy import is deferred. Selection honors
``HEARTH_EMBEDDER`` (``hash`` default | ``mlx``), like :func:`~hearth.providers.select_provider`.

The default path must work fully offline with no extra deps — it is what the walking
skeleton and the test suite use. The real MLX embedder needs a pre-pulled model and is
never exercised in the sandbox (network is blocked, same as the coder models).
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, runtime_checkable

from ..config import Settings, get_settings

# Default embedding dimensionality for the offline hashing embedder. 256 is a good
# tradeoff: small enough to store/scan cheaply, wide enough to keep token collisions rare.
DEFAULT_HASH_DIM = 256

# Tokenizer for the hashing embedder: word-boundary tokens (letters/digits/underscore).
_TOKEN_RE = re.compile(r"[a-z0-9_]+")


class EmbeddingUnavailableError(RuntimeError):
    """Raised when an embedding backend is requested but its dep/model is missing."""


@runtime_checkable
class EmbeddingProvider(Protocol):
    """The single interface every embedding backend implements.

    ``embed`` returns one vector per input text; all vectors share :attr:`dim`.
    """

    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into fixed-length vectors."""
        ...


class HashEmbedder:
    """Deterministic, dependency-free hashing-trick embedder (the DEFAULT).

    Tokenizes on word boundaries, hashes each token into one of :attr:`dim` buckets with
    a signed value, then L2-normalizes the result. It is fully offline, needs no model
    download, and yields the same vector for the same text on every run — ideal for the
    walking skeleton, tests, and CI.

    **Quality tradeoff.** This is a bag-of-tokens representation: it captures lexical
    overlap (shared words → higher cosine similarity) but has *no* semantic understanding
    — synonyms, word order, and paraphrase are invisible to it, and hash collisions add
    noise. It is a functional stand-in, not a substitute for a real embedding model; swap
    in :class:`MLXEmbedder` (``HEARTH_EMBEDDER=mlx``) when retrieval quality matters.
    """

    name = "hash"

    def __init__(self, dim: int = DEFAULT_HASH_DIM) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN_RE.findall(text.lower()):
            bucket, sign = self._bucket(token)
            vec[bucket] += sign
        return _l2_normalize(vec)

    def _bucket(self, token: str) -> tuple[int, float]:
        """Map a token to a (bucket, ±1) pair via a stable hash.

        ``hash()`` is per-process salted, so we use a fixed digest (blake2b) to stay
        deterministic across runs. One digest byte picks the sign, keeping the mean of
        the hashed features near zero (the standard signed hashing trick).
        """
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        bucket = value % self.dim
        sign = 1.0 if (value >> 63) & 1 else -1.0
        return bucket, sign


class MLXEmbedder:
    """Real local embeddings via ``mlx-lm`` (OPTIONAL, ``[embeddings]`` extra).

    The heavy import is deferred to first use (like :class:`~hearth.providers.mlx.MLXProvider`),
    so the package installs and runs without the extra. It requires a pre-pulled embedding
    model; when the dependency or model is missing it raises :class:`EmbeddingUnavailableError`
    with a clear message. Never exercised in the sandbox (network is blocked).
    """

    name = "mlx"

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._model = None
        self._tokenizer = None
        # Dimensionality is only known once the model loads; expose a sentinel until then.
        self.dim = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._ensure_loaded()
        import mlx.core as mx  # deferred heavy import

        vectors: list[list[float]] = []
        for text in texts:
            tokens = mx.array([self._tokenizer.encode(text)])
            # Mean-pool the last hidden state into a single sentence vector, then normalize.
            hidden = self._model(tokens)
            pooled = hidden.mean(axis=1)[0]
            vec = [float(x) for x in pooled.tolist()]
            vectors.append(_l2_normalize(vec))
        return vectors

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not mlx_embeddings_available():
            raise EmbeddingUnavailableError(
                "the MLX embeddings backend is not installed. "
                "Install it with: uv sync --extra embeddings"
            )
        try:
            from mlx_lm import load  # deferred; part of the mlx-lm package
        except ImportError as exc:  # pragma: no cover - guarded by availability check
            raise EmbeddingUnavailableError(
                "mlx-lm is not importable for the embeddings backend "
                "(install: uv sync --extra embeddings)"
            ) from exc
        try:
            self._model, self._tokenizer = load(self.model_id)
        except Exception as exc:  # pragma: no cover - needs a pre-pulled model
            raise EmbeddingUnavailableError(
                f"could not load embedding model {self.model_id!r}: {exc}. "
                "Pre-pull it from an unrestricted terminal (network is blocked here)."
            ) from exc
        # Infer dim from a probe embedding so downstream stores size their columns right.
        self.dim = len(self.embed(["dim probe"])[0]) if self.dim == 0 else self.dim


def mlx_embeddings_available() -> bool:
    """True if the MLX embeddings dependency (``mlx-lm``) is importable."""
    import importlib.util

    return importlib.util.find_spec("mlx_lm") is not None


def select_embedder(settings: Settings | None = None) -> EmbeddingProvider:
    """Return the active embedding provider based on ``HEARTH_EMBEDDER`` config.

    ``hash`` (default) is the offline dependency-free embedder; ``mlx`` is the real model.
    The MLX model id comes from settings (``HEARTH_EMBED_MODEL``), matching how the model
    registry supplies the default coder model.
    """
    settings = settings or get_settings()
    choice = settings.embedder.lower()
    if choice == "hash":
        return HashEmbedder(dim=settings.embed_dim)
    if choice == "mlx":
        return MLXEmbedder(settings.embed_model)
    raise ValueError(f"Unknown HEARTH_EMBEDDER: {settings.embedder!r} (use hash|mlx)")


def _l2_normalize(vec: list[float]) -> list[float]:
    """Scale ``vec`` to unit length; a zero vector is returned unchanged.

    L2-normalizing up front means cosine similarity reduces to a dot product at query time.
    """
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


__all__ = [
    "EmbeddingProvider",
    "EmbeddingUnavailableError",
    "HashEmbedder",
    "MLXEmbedder",
    "select_embedder",
    "mlx_embeddings_available",
    "DEFAULT_HASH_DIM",
]
