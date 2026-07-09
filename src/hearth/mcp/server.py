"""FastMCP wiring for the HEARTH MCP server (ADR-010, Phase 5).

This is the *only* module that imports the ``mcp`` SDK, so it lives behind the ``[mcp]``
extra and is imported lazily by ``hearth mcp``. All tool logic lives in
:mod:`hearth.mcp.tools`; here we just register each bound method as an MCP tool under the
names Claude Code sees (``hearth_summarize`` etc., per docs/INTEGRATION.md) and run the
stdio transport.
"""

from __future__ import annotations

from .tools import HearthTools, build_toolset


def build_server(tools: HearthTools | None = None):
    """Build the FastMCP server with HEARTH's tools registered.

    Returns a ``FastMCP`` instance. Importing ``mcp`` is deferred to call time so merely
    importing this module (or ``hearth.mcp``) doesn't require the optional dependency.
    """
    from mcp.server.fastmcp import FastMCP

    tools = tools or build_toolset()
    mcp = FastMCP("hearth")

    @mcp.tool(name="hearth_summarize")
    def hearth_summarize(text: str, max_words: int | None = None) -> str:
        """Summarize text on the local HEARTH model (no frontier tokens spent)."""
        return tools.summarize(text, max_words=max_words)

    @mcp.tool(name="hearth_classify")
    def hearth_classify(text: str, labels: list[str]) -> str:
        """Classify text into one of the given labels, locally."""
        return tools.classify(text, labels)

    @mcp.tool(name="hearth_extract")
    def hearth_extract(text: str, fields: list[str]) -> dict[str, str]:
        """Extract the named fields from text, locally. Returns a field->value map."""
        return tools.extract(text, fields)

    @mcp.tool(name="hearth_draft")
    def hearth_draft(instruction: str, context: str | None = None) -> str:
        """Draft prose/boilerplate (e.g. a commit message) from an instruction, locally."""
        return tools.draft(instruction, context=context)

    @mcp.tool(name="hearth_rag_query")
    def hearth_rag_query(
        collection: str, query: str, k: int = 6, answer: bool = False
    ) -> dict:
        """Retrieve grounded chunks from a local RAG collection; optionally answer locally."""
        return tools.rag_query(collection, query, k=k, answer=answer)

    return mcp


def run() -> None:
    """Launch the HEARTH MCP server over stdio (the transport Claude Code speaks)."""
    build_server().run()


__all__ = ["build_server", "run"]
