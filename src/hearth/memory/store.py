"""Vector store — embedded, file-based similarity search (ADR-008, Phase 3).

A :class:`VectorStore` protocol with a dependency-free default,
:class:`SQLiteVectorStore`, that keeps one SQLite file per collection under
``~/.hearth/rag/<collection>.db``. Each row is ``(id, text, source, metadata_json,
embedding)``; ``query`` does brute-force cosine similarity in Python.

Brute force is fine at this scale — per-project code/doc collections, not millions of
vectors (ADR-008). ``numpy`` is used as an optional speedup when importable; the pure
stdlib ``math`` path is the fallback so the default install needs no extra deps.

Follow-up: :class:`SqliteVecVectorStore` now drops a real ``sqlite-vec`` KNN backend in
behind this protocol (opt in via ``HEARTH_VECTOR_STORE=sqlite-vec``, needs the ``[vec]``
extra) without touching the RAG layer above it. A LanceDB backend could join it the same
way.
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


class SqliteVecUnavailableError(RuntimeError):
    """Raised when the sqlite-vec backend is requested but ``sqlite-vec`` isn't importable."""


class SqliteVecVectorStore:
    """Vector store backed by the ``sqlite-vec`` extension (opt-in, needs the ``[vec]`` extra).

    A drop-in behind :class:`VectorStore`: one SQLite file per collection under
    ``~/.hearth/rag/<collection>.db`` (same root/sanitization as :class:`SQLiteVectorStore`),
    but similarity search is delegated to sqlite-vec's ``vec0`` virtual table using an indexed
    KNN ``MATCH`` query instead of Python brute force. Each collection has a ``vec_chunks``
    virtual table holding the embedding keyed by an integer ``rowid``, alongside a companion
    ``chunks`` table for the ``(id, text, source, metadata)`` payload.

    The ``sqlite_vec`` import is deferred to :meth:`_connect` (not module import), so importing
    this module needs no extra; requesting the backend without it raises
    :class:`SqliteVecUnavailableError` with the fix hint ``uv sync --extra vec``. Construction
    itself is cheap and never loads the extension, so dispatch/protocol conformance are testable
    without the native library.

    Vectors are L2-normalized by the embedder, so sqlite-vec's default L2 distance ``d`` relates
    to cosine similarity by ``d^2 = 2 - 2*cos``; we return ``score = 1 - d^2/2`` so higher means
    more similar, matching the cosine convention of :class:`SQLiteVectorStore`.
    """

    name = "sqlite-vec"

    def __init__(self, root: Path | None = None, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        self.root = root or (settings.home / "rag")

    def add(self, collection: str, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")
        if not chunks:
            return 0
        conn = self._connect(collection, dim=len(vectors[0]))
        try:
            import sqlite_vec  # deferred; already imported successfully by _connect

            for c, v in zip(chunks, vectors, strict=True):
                # Upsert the payload; keep vec_chunks in lockstep via the shared rowid so a
                # re-ingest with a stable id replaces rather than duplicates (mirrors the
                # brute-force store's INSERT OR REPLACE semantics).
                cur = conn.execute(
                    "INSERT INTO chunks (id, text, source, metadata) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(id) DO UPDATE SET text=excluded.text, source=excluded.source, "
                    "metadata=excluded.metadata RETURNING rowid",
                    (c.id, c.text, c.source, json.dumps(c.metadata)),
                )
                rowid = int(cur.fetchone()[0])
                conn.execute("DELETE FROM vec_chunks WHERE rowid = ?", (rowid,))
                conn.execute(
                    "INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)",
                    (rowid, sqlite_vec.serialize_float32(v)),
                )
            conn.commit()
        finally:
            conn.close()
        return len(chunks)

    def query(self, collection: str, vector: list[float], k: int) -> list[Chunk]:
        path = self._path(collection)
        if not path.exists() or k <= 0:
            return []
        conn = self._connect(collection, dim=len(vector))
        try:
            import sqlite_vec  # deferred; already imported successfully by _connect

            rows = conn.execute(
                "SELECT c.id, c.text, c.source, c.metadata, v.distance AS distance "
                "FROM vec_chunks v JOIN chunks c ON c.rowid = v.rowid "
                "WHERE v.embedding MATCH ? AND k = ? ORDER BY v.distance",
                (sqlite_vec.serialize_float32(vector), k),
            ).fetchall()
        finally:
            conn.close()
        return [
            Chunk(
                id=row["id"],
                text=row["text"],
                source=row["source"],
                metadata=json.loads(row["metadata"]),
                # L2 distance -> cosine similarity for unit vectors: cos = 1 - d^2/2.
                score=1.0 - (float(row["distance"]) ** 2) / 2.0,
            )
            for row in rows
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

    def _connect(self, collection: str, dim: int | None = None) -> sqlite3.Connection:
        """Open the collection DB with the sqlite-vec extension loaded.

        The ``sqlite_vec`` import is deferred here so importing this module needs no extra.
        The ``vec_chunks`` virtual table is created lazily on first write (its dimensionality
        is fixed at creation time, so ``dim`` must be known); ``count`` may connect without a
        ``dim`` when only the companion ``chunks`` table is needed.
        """
        try:
            import sqlite_vec
        except ImportError as exc:
            raise SqliteVecUnavailableError(
                "sqlite-vec is not installed. Install the backend with: uv sync --extra vec"
            ) from exc

        self.root.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._path(collection))
        conn.row_factory = sqlite3.Row
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS chunks ("
            "  rowid INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  id TEXT NOT NULL UNIQUE,"
            "  text TEXT NOT NULL,"
            "  source TEXT NOT NULL,"
            "  metadata TEXT NOT NULL"
            ")"
        )
        if dim is not None:
            conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
                f"  embedding float[{int(dim)}]"
                f")"
            )
        return conn


def _safe_collection(name: str) -> str:
    """Sanitize a collection name into a safe filename stem (no path traversal)."""
    safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in name.strip())
    return safe or "default"


def select_vector_store(settings: Settings | None = None) -> VectorStore:
    """Return the active vector store based on ``HEARTH_VECTOR_STORE`` config (Phase 7).

    ``sqlite`` (default) is the embedded, file-based store (ADR-008). ``sqlite-vec`` (aliased
    ``sqlitevec``) opts into the indexed :class:`SqliteVecVectorStore` KNN backend, which needs
    the ``[vec]`` extra (``uv sync --extra vec``). Any other value resolves against the
    ``hearth.vector_stores`` plugin entry-point group, so a third-party store (e.g. LanceDB)
    serves via ``HEARTH_VECTOR_STORE=<name>`` with zero core edits. Mirrors
    :func:`~hearth.providers.select_provider`.
    """
    settings = settings or get_settings()
    choice = settings.vector_store.lower()
    if choice == "sqlite":
        return SQLiteVectorStore(settings=settings)
    if choice in ("sqlite-vec", "sqlitevec"):
        return SqliteVecVectorStore(settings=settings)

    from ..plugins import VECTOR_STORE_GROUP, load_plugin

    plugin = load_plugin(VECTOR_STORE_GROUP, settings.vector_store)
    if plugin is not None:
        return plugin
    raise ValueError(
        f"Unknown HEARTH_VECTOR_STORE: {settings.vector_store!r} "
        "(use sqlite, or install a plugin registering this name under hearth.vector_stores)"
    )


__all__ = [
    "Chunk",
    "VectorStore",
    "SQLiteVectorStore",
    "SqliteVecVectorStore",
    "SqliteVecUnavailableError",
    "select_vector_store",
]
