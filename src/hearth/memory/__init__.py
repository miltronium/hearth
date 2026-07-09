"""Memory / RAG layer (ARCHITECTURE §6, ADR-008, Phase 3).

Embeddings + an embedded vector store + a thin RAG orchestration on top. All of it
defaults to the offline, dependency-free path (:class:`HashEmbedder` +
:class:`SQLiteVectorStore`) so it works with no extras installed and no network.
"""

from __future__ import annotations

from .embed import (
    EmbeddingProvider,
    EmbeddingUnavailableError,
    HashEmbedder,
    MLXEmbedder,
    select_embedder,
)
from .rag import IngestResult, QueryResult, RagIndex, chunk_text
from .store import Chunk, SQLiteVectorStore, VectorStore, select_vector_store

__all__ = [
    "EmbeddingProvider",
    "EmbeddingUnavailableError",
    "HashEmbedder",
    "MLXEmbedder",
    "select_embedder",
    "Chunk",
    "VectorStore",
    "SQLiteVectorStore",
    "select_vector_store",
    "RagIndex",
    "IngestResult",
    "QueryResult",
    "chunk_text",
]
