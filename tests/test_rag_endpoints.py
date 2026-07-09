"""RAG endpoint tests (Phase 3): /v1/hearth/rag/ingest and /v1/hearth/rag/query."""

from __future__ import annotations


def test_rag_ingest_then_query(client, tmp_path):
    repo = tmp_path / "docs"
    repo.mkdir()
    (repo / "auth.md").write_text(
        "The astris backend authenticates callers with a bearer token read from disk.\n"
    )
    (repo / "store.md").write_text("The vector store is an embedded SQLite file per collection.\n")

    ingest = client.post(
        "/v1/hearth/rag/ingest",
        json={"collection": "c", "paths": [str(repo)], "chunk": {"size": 200, "overlap": 40}},
    )
    assert ingest.status_code == 200
    body = ingest.json()
    assert body["collection"] == "c"
    assert body["files"] == 2
    assert body["chunks"] >= 2

    query = client.post(
        "/v1/hearth/rag/query",
        json={"collection": "c", "query": "how does the backend authenticate", "k": 3},
    )
    assert query.status_code == 200
    qbody = query.json()
    assert qbody["answer"] is None  # answer=False default
    assert qbody["chunks"]
    top = qbody["chunks"][0]
    assert set(top) == {"text", "source", "score"}
    assert any("bearer token" in c["text"] for c in qbody["chunks"])


def test_rag_query_with_answer_uses_local_model(client, tmp_path):
    doc = tmp_path / "note.txt"
    doc.write_text("HEARTH keeps one SQLite file per RAG collection.")

    client.post("/v1/hearth/rag/ingest", json={"collection": "a", "paths": [str(doc)]})
    r = client.post(
        "/v1/hearth/rag/query",
        json={"collection": "a", "query": "where are collections stored", "answer": True},
    )
    assert r.status_code == 200
    body = r.json()
    # The echo backend serves the grounded answer locally (allow_escalation=False).
    assert body["answer"] is not None
    assert body["answer"].startswith("[echo]")


def test_rag_query_empty_collection(client):
    r = client.post("/v1/hearth/rag/query", json={"collection": "empty", "query": "x"})
    assert r.status_code == 200
    assert r.json()["chunks"] == []
