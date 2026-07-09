"""Memory / RAG layer tests (Phase 3): embedder, vector store, chunker, ingest→query."""

from __future__ import annotations

import math

from hearth.memory.embed import DEFAULT_HASH_DIM, HashEmbedder, select_embedder
from hearth.memory.rag import RagIndex, chunk_text
from hearth.memory.store import Chunk, SQLiteVectorStore

# -- HashEmbedder ---------------------------------------------------------------------


def test_hash_embedder_dim_and_default():
    emb = HashEmbedder()
    assert emb.dim == DEFAULT_HASH_DIM
    (vec,) = emb.embed(["hello world"])
    assert len(vec) == DEFAULT_HASH_DIM


def test_hash_embedder_deterministic():
    emb = HashEmbedder()
    a = emb.embed(["the quick brown fox"])
    b = emb.embed(["the quick brown fox"])
    assert a == b


def test_hash_embedder_l2_normalized():
    (vec,) = HashEmbedder().embed(["vector store cosine similarity"])
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-6


def test_hash_embedder_empty_text_is_zero_vector():
    # No tokens → zero vector (norm 0), returned unchanged rather than dividing by zero.
    (vec,) = HashEmbedder().embed(["   "])
    assert all(x == 0.0 for x in vec)


def test_hash_embedder_similarity_ordering():
    emb = HashEmbedder()
    q, close, far = emb.embed(
        ["how does the vector store work", "the vector store uses sqlite", "bananas are yellow"]
    )
    sim_close = sum(a * b for a, b in zip(q, close, strict=False))
    sim_far = sum(a * b for a, b in zip(q, far, strict=False))
    assert sim_close > sim_far


def test_select_embedder_defaults_to_hash(settings):
    emb = select_embedder(settings)
    assert emb.name == "hash"


# -- SQLiteVectorStore ----------------------------------------------------------------


def _chunk(cid: str, text: str, source: str = "s.txt") -> Chunk:
    return Chunk(id=cid, text=text, source=source, metadata={"k": cid})


def test_store_add_count_query_roundtrip(tmp_path):
    store = SQLiteVectorStore(root=tmp_path / "rag")
    emb = HashEmbedder()
    texts = ["alpha vector store", "beta something else", "gamma unrelated topic"]
    chunks = [_chunk(f"c{i}", t) for i, t in enumerate(texts)]
    vectors = emb.embed(texts)

    added = store.add("col", chunks, vectors)
    assert added == 3
    assert store.count("col") == 3

    (qvec,) = emb.embed(["vector store"])
    results = store.query("col", qvec, k=2)
    assert len(results) == 2
    # The lexically closest chunk ranks first and carries a score + metadata.
    assert results[0].text == "alpha vector store"
    assert results[0].score >= results[1].score
    assert results[0].metadata == {"k": "c0"}


def test_store_count_empty_collection(tmp_path):
    store = SQLiteVectorStore(root=tmp_path / "rag")
    assert store.count("nope") == 0
    assert store.query("nope", [0.0] * 4, k=3) == []


def test_store_reingest_replaces_not_duplicates(tmp_path):
    store = SQLiteVectorStore(root=tmp_path / "rag")
    emb = HashEmbedder()
    chunk = _chunk("stable-id", "hello")
    store.add("col", [chunk], emb.embed(["hello"]))
    store.add("col", [chunk], emb.embed(["hello"]))
    assert store.count("col") == 1


# -- chunker --------------------------------------------------------------------------


def test_chunk_text_respects_size_and_overlaps():
    text = "\n".join(f"line number {i}" for i in range(50))
    chunks = chunk_text(text, size=80, overlap=20)
    assert len(chunks) > 1
    # No chunk is wildly over-size (line-aware: at most one extra line past the limit).
    assert all(len(c) <= 80 + 40 for c in chunks)
    # Reassembling with overlap still covers the whole document.
    joined = "".join(chunks)
    assert "line number 0" in joined
    assert "line number 49" in joined


def test_chunk_text_long_single_line_kept_whole():
    long_line = "x" * 500
    chunks = chunk_text(long_line, size=100, overlap=10)
    # A single over-long line is emitted rather than split mid-line.
    assert chunks == [long_line]


def test_chunk_text_empty():
    assert chunk_text("   \n  \n") == []


# -- ingest → query roundtrip ---------------------------------------------------------


def test_ingest_then_query_finds_planted_string(tmp_path, settings):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("def add(a, b):\n    return a + b\n")
    (repo / "b.md").write_text(
        "The astris backend authenticates using a bearer token stored on disk.\n"
    )
    # A binary file and a vendor dir must be skipped.
    (repo / "logo.bin").write_bytes(b"\x00\x01\x02binary")
    (repo / "node_modules").mkdir()
    (repo / "node_modules" / "junk.js").write_text("console.log('vendor')\n")

    index = RagIndex(store=SQLiteVectorStore(root=tmp_path / "rag"))
    result = index.ingest(repo, "t", size=200, overlap=40)
    assert result.files == 2  # a.py + b.md only
    assert result.chunks >= 2

    hits = index.query("t", "how does the backend authenticate", k=3)
    assert hits.chunks
    assert any("bearer token" in c.text for c in hits.chunks)
    assert hits.answer is None  # answer=False → chunks only


def test_ingest_single_file(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("hearth vector store notes")
    index = RagIndex(store=SQLiteVectorStore(root=tmp_path / "rag"))
    result = index.ingest(f, "single")
    assert result.files == 1
    assert result.chunks == 1
