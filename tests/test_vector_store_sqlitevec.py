"""Tests for the opt-in sqlite-vec vector store (Phase 7, ADR-008).

The real KNN backend needs the ``[vec]`` extra (``sqlite-vec``), which is very likely NOT
installed in this environment, so those tests ``pytest.importorskip`` cleanly. Dispatch and
protocol-conformance tests are designed to run WITHOUT the native extension.
"""

from __future__ import annotations

import pytest

from hearth.config import Settings
from hearth.memory.embed import HashEmbedder
from hearth.memory.store import (
    Chunk,
    SQLiteVectorStore,
    SqliteVecUnavailableError,
    SqliteVecVectorStore,
    VectorStore,
    select_vector_store,
)


def _chunk(cid: str, text: str, source: str = "s.txt") -> Chunk:
    return Chunk(id=cid, text=text, source=source, metadata={"k": cid})


def _has_sqlite_vec() -> bool:
    import importlib.util

    return importlib.util.find_spec("sqlite_vec") is not None


# -- dispatch + protocol conformance (run WITHOUT the extension) -----------------------


def test_select_vector_store_dispatches_to_sqlite_vec(tmp_path):
    # Construction is cheap and must not need the native extension: dispatch is verifiable
    # regardless of whether sqlite-vec is installed.
    settings = Settings(vector_store="sqlite-vec", home=tmp_path / ".hearth")
    store = select_vector_store(settings)
    assert isinstance(store, SqliteVecVectorStore)
    assert store.name == "sqlite-vec"


def test_select_vector_store_accepts_sqlitevec_alias(tmp_path):
    settings = Settings(vector_store="sqlitevec", home=tmp_path / ".hearth")
    store = select_vector_store(settings)
    assert isinstance(store, SqliteVecVectorStore)


def test_select_vector_store_default_is_plain_sqlite(tmp_path):
    settings = Settings(home=tmp_path / ".hearth")
    assert isinstance(select_vector_store(settings), SQLiteVectorStore)


def test_sqlite_vec_store_satisfies_protocol():
    # runtime_checkable Protocol conformance — no extension required.
    store = SqliteVecVectorStore(root=None, settings=Settings())
    assert isinstance(store, VectorStore)


def test_sqlite_vec_uses_same_root_convention(tmp_path):
    settings = Settings(home=tmp_path / ".hearth")
    store = SqliteVecVectorStore(settings=settings)
    assert store.root == settings.home / "rag"


@pytest.mark.skipif(
    _has_sqlite_vec(), reason="sqlite-vec IS installed; the no-extension error path is moot"
)
def test_add_without_extension_raises_fix_hint(tmp_path):
    store = SqliteVecVectorStore(root=tmp_path / "rag")
    with pytest.raises(SqliteVecUnavailableError) as excinfo:
        store.add("col", [_chunk("c0", "hello")], [[1.0, 0.0]])
    assert "uv sync --extra vec" in str(excinfo.value)


# -- real backend (skipped cleanly when the extra is absent) ---------------------------

sqlite_vec = pytest.importorskip("sqlite_vec")


def test_sqlite_vec_add_count_query_roundtrip(tmp_path):
    store = SqliteVecVectorStore(root=tmp_path / "rag")
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
    assert results[0].text == "alpha vector store"
    assert results[0].score >= results[1].score
    assert results[0].metadata == {"k": "c0"}


def test_sqlite_vec_count_empty_and_query_empty(tmp_path):
    store = SqliteVecVectorStore(root=tmp_path / "rag")
    assert store.count("nope") == 0
    assert store.query("nope", [1.0, 0.0], k=3) == []


def test_sqlite_vec_reingest_replaces_not_duplicates(tmp_path):
    store = SqliteVecVectorStore(root=tmp_path / "rag")
    emb = HashEmbedder()
    chunk = _chunk("stable-id", "hello")
    store.add("col", [chunk], emb.embed(["hello"]))
    store.add("col", [chunk], emb.embed(["hello"]))
    assert store.count("col") == 1


def test_sqlite_vec_topk_matches_brute_force_ordering(tmp_path):
    # Hand-built normalized vectors: sqlite-vec KNN must rank them identically to the
    # brute-force cosine store for the same query.
    import math

    def norm(v: list[float]) -> list[float]:
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    vectors = [
        norm([1.0, 0.0, 0.0]),
        norm([0.9, 0.1, 0.0]),
        norm([0.0, 1.0, 0.0]),
        norm([0.0, 0.0, 1.0]),
        norm([0.5, 0.5, 0.0]),
    ]
    chunks = [_chunk(f"c{i}", f"chunk {i}") for i in range(len(vectors))]
    query = norm([0.95, 0.05, 0.0])

    brute = SQLiteVectorStore(root=tmp_path / "brute")
    brute.add("col", chunks, vectors)
    brute_order = [c.id for c in brute.query("col", query, k=len(vectors))]

    vec = SqliteVecVectorStore(root=tmp_path / "vec")
    vec.add("col", chunks, vectors)
    vec_order = [c.id for c in vec.query("col", query, k=len(vectors))]

    assert vec_order == brute_order
    # Scores follow the cosine convention (higher = more similar), so descending.
    scores = [c.score for c in vec.query("col", query, k=len(vectors))]
    assert scores == sorted(scores, reverse=True)
