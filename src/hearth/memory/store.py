"""Vector store — embedded, file-based similarity search (ADR-008, Phase 3).

A :class:`VectorStore` protocol with a dependency-free default,
:class:`SQLiteVectorStore`, that keeps one SQLite file per collection under
``~/.hearth/rag/<collection>.db``. Each row is ``(id, text, source, metadata_json,
embedding)``; ``query`` does brute-force cosine similarity in Python.

Brute force is fine at this scale — per-project code/doc collections, not millions of
vectors (ADR-008). ``numpy`` is used as an optional speedup when importable; the pure
stdlib ``math`` path is the fallback so the default install needs no extra deps.

Follow-up: a ``sqlite-vec`` / LanceDB backend can drop in behind this protocol without
touching the RAG layer above it.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..config import Settings, get_settings


@dataclass(frozen=True)
class Chunk:
    """One stored (or retrieved) chunk. ``score`` is set only on query results."""

    id: str
    text: str
    source: str
    metadata: dict
    score: float = 0.0


@runtime_checkable
class VectorStore(Protocol):
    """The interface every vector backend implements (per named collection)."""

    def add(self, collection: str, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        """Store ``chunks`` with their ``vectors`` into ``collection``; return count added."""
        ...

    def query(self, collection: str, vector: list[float], k: int) -> list[Chunk]:
        """Return the top-``k`` chunks by cosine similarity, most similar first."""
        ...

    def count(self, collection: str) -> int:
        """Return the number of chunks stored in ``collection``."""
        ...


class SQLiteVectorStore:
    """Embedded vector store backed by one SQLite file per collection (ADR-008).

    Vectors are stored as JSON text (portable, no binary-endianness concerns at this
    scale). Similarity is brute-force cosine over all rows — vectors are assumed already
    L2-normalized by the embedder, so cosine reduces to a dot product.
    """

    name = "sqlite"

    def __init__(self, root: Path | None = None, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        self.root = root or (settings.home / "rag")

    def add(self, collection: str, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")
        if not chunks:
            return 0
        conn = self._connect(collection)
        try:
            conn.executemany(
                "INSERT OR REPLACE INTO chunks (id, text, source, metadata, embedding) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        c.id,
                        c.text,
                        c.source,
                        json.dumps(c.metadata),
                        json.dumps(v),
                    )
                    for c, v in zip(chunks, vectors, strict=True)
                ],
            )
            conn.commit()
        finally:
            conn.close()
        return len(chunks)

    def query(self, collection: str, vector: list[float], k: int) -> list[Chunk]:
        conn = self._connect(collection)
        try:
            rows = conn.execute(
                "SELECT id, text, source, metadata, embedding FROM chunks"
            ).fetchall()
        finally:
            conn.close()
        if not rows:
            return []
        scored = _rank(vector, rows, k)
        return [
            Chunk(
                id=row["id"],
                text=row["text"],
                source=row["source"],
                metadata=json.loads(row["metadata"]),
                score=score,
            )
            for score, row in scored
        ]

    def count(self, collection: str) -> int:
        path = self._path(collection)
        if not path.exists():
            return 0
        conn = self._connect(collection)
        try:
            return int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        finally:
            conn.close()

    def _path(self, collection: str) -> Path:
        return self.root / f"{_safe_collection(collection)}.db"

    def _connect(self, collection: str) -> sqlite3.Connection:
        self.root.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._path(collection))
        conn.row_factory = sqlite3.Row
        conn.execute(
            "CREATE TABLE IF NOT EXISTS chunks ("
            "  id TEXT PRIMARY KEY,"
            "  text TEXT NOT NULL,"
            "  source TEXT NOT NULL,"
            "  metadata TEXT NOT NULL,"
            "  embedding TEXT NOT NULL"
            ")"
        )
        return conn


def _rank(
    query_vec: list[float], rows: list[sqlite3.Row], k: int
) -> list[tuple[float, sqlite3.Row]]:
    """Score every row by cosine similarity and return the top ``k`` (numpy if available)."""
    embeddings = [json.loads(r["embedding"]) for r in rows]
    scores = _cosine_scores(query_vec, embeddings)
    ranked = sorted(zip(scores, rows, strict=True), key=lambda pair: pair[0], reverse=True)
    return ranked[: max(0, k)]


def _cosine_scores(query_vec: list[float], embeddings: list[list[float]]) -> list[float]:
    """Cosine similarity of ``query_vec`` against each embedding.

    Vectors are stored L2-normalized, so cosine == dot product. Falls back to a pure
    stdlib computation when numpy is not importable (the default, no-extras path).
    """
    try:
        import numpy as np  # optional speedup only
    except ImportError:
        return [_dot(query_vec, e) for e in embeddings]
    q = np.asarray(query_vec, dtype=np.float32)
    mat = np.asarray(embeddings, dtype=np.float32)
    return (mat @ q).tolist()


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


def _safe_collection(name: str) -> str:
    """Sanitize a collection name into a safe filename stem (no path traversal)."""
    safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in name.strip())
    return safe or "default"


def select_vector_store(settings: Settings | None = None) -> VectorStore:
    """Return the active vector store based on ``HEARTH_VECTOR_STORE`` config (Phase 7).

    ``sqlite`` (default) is the embedded, file-based store (ADR-008). Any other value
    resolves against the ``hearth.vector_stores`` plugin entry-point group, so a
    third-party store (e.g. sqlite-vec, LanceDB) serves via ``HEARTH_VECTOR_STORE=<name>``
    with zero core edits. Mirrors :func:`~hearth.providers.select_provider`.
    """
    settings = settings or get_settings()
    choice = settings.vector_store.lower()
    if choice == "sqlite":
        return SQLiteVectorStore(settings=settings)

    from ..plugins import VECTOR_STORE_GROUP, load_plugin

    plugin = load_plugin(VECTOR_STORE_GROUP, settings.vector_store)
    if plugin is not None:
        return plugin
    raise ValueError(
        f"Unknown HEARTH_VECTOR_STORE: {settings.vector_store!r} "
        "(use sqlite, or install a plugin registering this name under hearth.vector_stores)"
    )


__all__ = ["Chunk", "VectorStore", "SQLiteVectorStore", "select_vector_store"]
