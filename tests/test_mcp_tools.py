"""Tests for the MCP tool logic (Phase 5, ADR-010).

These exercise :mod:`hearth.mcp.tools` directly against an echo router — proving the tool
logic works with **no ``mcp`` package installed** (only ``server.py`` imports it). This is
the import-safety guarantee for the split.
"""

from __future__ import annotations

from hearth.config import Settings
from hearth.mcp.tools import HearthTools, _parse_fields, build_toolset
from hearth.memory import RagIndex, SQLiteVectorStore, select_embedder
from hearth.providers.echo import EchoProvider
from hearth.router import Router


def _tools(tmp_path) -> HearthTools:
    settings = Settings(backend="echo", home=tmp_path / ".hearth", require_auth=False)
    router = Router(local_provider=EchoProvider())
    rag = RagIndex(
        embedder=select_embedder(settings),
        store=SQLiteVectorStore(settings=settings),
        router=router,
    )
    return HearthTools(router=router, rag=rag)


def test_tools_module_imports_without_mcp():
    # Importing the tools module (and the package) must not require the `mcp` SDK.
    import hearth.mcp  # noqa: F401
    import hearth.mcp.tools  # noqa: F401


def test_summarize_runs_locally(tmp_path):
    tools = _tools(tmp_path)
    out = tools.summarize("the quick brown fox", max_words=10)
    # Echo backend prefixes [echo]; the summarize prompt text rides through.
    assert "[echo]" in out
    assert "the quick brown fox" in out


def test_classify_returns_text(tmp_path):
    tools = _tools(tmp_path)
    out = tools.classify("deploy the build", labels=["query", "action", "config"])
    assert isinstance(out, str) and out


def test_extract_returns_field_map(tmp_path):
    tools = _tools(tmp_path)
    out = tools.extract("ticket ABC-1", fields=["ticket", "assignee"])
    assert set(out.keys()) == {"ticket", "assignee"}


def test_draft_with_and_without_context(tmp_path):
    tools = _tools(tmp_path)
    assert "[echo]" in tools.draft("write a commit message")
    assert "diff here" in tools.draft("write a commit message", context="diff here")


def test_rag_query_returns_dict_shape(tmp_path):
    tools = _tools(tmp_path)
    tools.rag.ingest(tmp_path, "docs")  # ingest the (empty-ish) tmp tree; may yield 0 chunks
    result = tools.rag_query("docs", "anything", k=3)
    assert set(result.keys()) == {"chunks", "answer"}
    assert isinstance(result["chunks"], list)
    assert result["answer"] is None


def test_build_toolset_wires_defaults(tmp_path):
    settings = Settings(backend="echo", home=tmp_path / ".hearth", require_auth=False)
    tools = build_toolset(settings=settings)
    assert isinstance(tools, HearthTools)
    assert tools.summarize("hi")  # end-to-end through the wired echo router


def test_parse_fields_case_insensitive_and_defaults():
    parsed = _parse_fields("Ticket: ABC-1\nother: x", ["ticket", "assignee"])
    assert parsed == {"ticket": "ABC-1", "assignee": ""}
