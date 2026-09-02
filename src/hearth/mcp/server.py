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

    # Path-taking variants — the agent passes a path and never holds the content. Gated by
    # the HEARTH_FILE_ROOTS allowlist in hearth.mcp.files (deny-by-default when unset).
    # Formats come from that reader, not from here: text, CSV, JSON, XLSX and text-layer
    # PDF all arrive through the same three tools, so the MCP surface never grows a
    # per-format variant an agent would have to choose between.

    @mcp.tool(name="hearth_summarize_file")
    def hearth_summarize_file(path: str, max_words: int | None = None) -> str:
        """Summarize a local file WITHOUT reading it into your context. HEARTH opens it
        itself; the path must be inside a HEARTH_FILE_ROOTS directory."""
        return tools.summarize_file(path, max_words=max_words)

    @mcp.tool(name="hearth_classify_file")
    def hearth_classify_file(path: str, labels: list[str]) -> str:
        """Classify a local file into one of the given labels without reading it into your
        context. The path must be inside a HEARTH_FILE_ROOTS directory."""
        return tools.classify_file(path, labels)

    @mcp.tool(name="hearth_extract_file")
    def hearth_extract_file(path: str, fields: list[str]) -> dict[str, str]:
        """Extract the named fields from a local file without reading it into your context.
        Returns a field->value map. The path must be inside a HEARTH_FILE_ROOTS directory."""
        return tools.extract_file(path, fields)

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
