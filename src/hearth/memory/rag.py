"""RAG layer — chunk, ingest, and query local collections (ARCHITECTURE §6, Phase 3).

Ties the embedder (:mod:`hearth.memory.embed`) and vector store
(:mod:`hearth.memory.store`) together:

  * :func:`chunk_text` — line-aware overlapping chunking.
  * :class:`RagIndex.ingest` — walk text files, chunk → embed → store into a collection.
  * :class:`RagIndex.query` — embed a query, retrieve top-k chunks; optionally answer with
    the local model (via the router, ``allow_escalation=False``) grounded in those chunks.

Everything defaults to the offline path (:class:`HashEmbedder` + :class:`SQLiteVectorStore`)
so ingest/query work with no extra deps and no network.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .embed import EmbeddingProvider, select_embedder
from .store import Chunk, SQLiteVectorStore, VectorStore

# Chunking defaults (docs/API.md): ~800-char chunks with 100-char overlap so retrieval
# has enough surrounding context and boundaries don't sever a relevant passage.
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100

# Directories never worth indexing — VCS metadata and common vendor/build trees.
_SKIP_DIRS = frozenset(
    {".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__",
     ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build", ".tox"}
)

# Read at most this many bytes to sniff whether a file is text (a NUL byte ⇒ binary).
_SNIFF_BYTES = 4096


@dataclass(frozen=True)
class IngestResult:
    """Counts from an ingest run."""

    collection: str
    files: int
    chunks: int


@dataclass(frozen=True)
class QueryResult:
    """A query response: retrieved chunks and, when requested, a grounded answer."""

    chunks: list[Chunk]
    answer: str | None = None


def chunk_text(
    text: str,
    size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split ``text`` into overlapping, line-aware chunks of about ``size`` chars.

    Chunks accumulate whole lines until adding the next line would exceed ``size``; the
    tail ``overlap`` characters (rounded to a line boundary) seed the next chunk so
    context isn't lost across the split. A single line longer than ``size`` is emitted
    as its own chunk rather than split mid-line.
    """
    if size <= 0:
        raise ValueError("chunk size must be positive")
    overlap = max(0, min(overlap, size - 1))

    lines = text.splitlines(keepends=True)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in lines:
        if current and current_len + len(line) > size:
            chunks.append("".join(current))
            current, current_len = _overlap_tail(current, overlap)
        current.append(line)
        current_len += len(line)

    if current and "".join(current).strip():
        chunks.append("".join(current))
    return chunks


class RagIndex:
    """Ingest and query per-collection RAG stores (ARCHITECTURE §6).

    Defaults wire the offline embedder + SQLite store; both are injectable so tests and
    later backends can swap them. ``router`` is optional and only needed for
    ``query(..., answer=True)``.
    """

    def __init__(
        self,
        embedder: EmbeddingProvider | None = None,
        store: VectorStore | None = None,
        router=None,
    ) -> None:
        self.embedder = embedder or select_embedder()
        self.store = store or SQLiteVectorStore()
        self.router = router

    # -- ingest -----------------------------------------------------------------------

    def ingest(
        self,
        path: Path | str,
        collection: str,
        size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> IngestResult:
        """Walk ``path`` (file or dir), chunk → embed → store into ``collection``."""
        root = Path(path)
        files = [root] if root.is_file() else sorted(_walk_text_files(root))
        total_chunks = 0
        ingested_files = 0
        for file in files:
            text = _read_text(file)
            if text is None:
                continue
            pieces = chunk_text(text, size=size, overlap=overlap)
            if not pieces:
                continue
            source = str(file)
            chunks = [
                Chunk(
                    id=_chunk_id(source, i, piece),
                    text=piece,
                    source=source,
                    metadata={"chunk_index": i},
                )
                for i, piece in enumerate(pieces)
            ]
            vectors = self.embedder.embed([c.text for c in chunks])
            total_chunks += self.store.add(collection, chunks, vectors)
            ingested_files += 1
        return IngestResult(collection=collection, files=ingested_files, chunks=total_chunks)

    # -- query ------------------------------------------------------------------------

    def query(
        self,
        collection: str,
        query: str,
        k: int = 6,
        answer: bool = False,
    ) -> QueryResult:
        """Embed ``query``, retrieve the top-``k`` chunks, optionally answer locally."""
        vector = self.embedder.embed([query])[0]
        chunks = self.store.query(collection, vector, k)
        if not answer:
            return QueryResult(chunks=chunks)
        return QueryResult(chunks=chunks, answer=self._answer(query, chunks))

    def _answer(self, query: str, chunks: list[Chunk]) -> str:
        """Answer ``query`` with the LOCAL model, grounded in the retrieved chunks.

        Escalation is disabled (``allow_escalation=False``) so RAG answers never make a
        surprise remote call — the whole point is cheap local context.
        """
        if self.router is None:
            raise ValueError("answer=True requires a router (RagIndex was built without one)")
        from ..providers.base import GenRequest, Message

        context = "\n\n".join(f"[{i + 1}] {c.source}\n{c.text}" for i, c in enumerate(chunks))
        system = (
            "You answer strictly from the provided context. If the context does not "
            "contain the answer, say so. Cite sources by their bracket number."
        )
        user = f"Context:\n{context}\n\nQuestion: {query}"
        messages = [
            Message(role="system", content=system),
            Message(role="user", content=user),
        ]
        routed = self.router.route(
            GenRequest(messages=messages, model="auto"),
            intent="summarize",
            allow_escalation=False,
        )
        return routed.result.text


def _walk_text_files(root: Path):
    """Yield text-like files under ``root``, skipping VCS/vendor dirs and binaries."""
    for entry in root.rglob("*"):
        if not entry.is_file():
            continue
        if any(part in _SKIP_DIRS for part in entry.parts):
            continue
        yield entry


def _read_text(path: Path) -> str | None:
    """Read a file as UTF-8 text, or return ``None`` if it looks binary/unreadable."""
    try:
        head = path.read_bytes()[:_SNIFF_BYTES]
    except OSError:
        return None
    if b"\x00" in head:  # NUL byte ⇒ almost certainly binary
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def _overlap_tail(lines: list[str], overlap: int) -> tuple[list[str], int]:
    """Return the trailing whole lines totaling up to ``overlap`` chars, as the next seed."""
    if overlap <= 0:
        return [], 0
    tail: list[str] = []
    length = 0
    for line in reversed(lines):
        if length + len(line) > overlap and tail:
            break
        tail.insert(0, line)
        length += len(line)
    return tail, length


def _chunk_id(source: str, index: int, text: str) -> str:
    """Stable id from source + position + content, so re-ingest replaces (not duplicates)."""
    digest = hashlib.blake2b(f"{source}:{index}:{text}".encode(), digest_size=16).hexdigest()
    return digest


__all__ = [
    "chunk_text",
    "RagIndex",
    "IngestResult",
    "QueryResult",
    "Chunk",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
]
