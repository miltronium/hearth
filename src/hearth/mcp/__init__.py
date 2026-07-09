"""HEARTH MCP server (ADR-010, Phase 5).

Exposes HEARTH's local model to MCP-speaking agents (Claude Code) as tools, so routine
subtasks — summarize, classify, extract, draft, RAG query — run on the local model and
never spend the agent's frontier budget.

Two layers, deliberately split so the tool *logic* is testable without the MCP SDK:

  * :mod:`hearth.mcp.tools` — pure functions that call HEARTH's router in-process with
    ``allow_escalation=False``. No ``mcp`` import; import-safe with only core deps.
  * :mod:`hearth.mcp.server` — thin FastMCP wiring (imports ``mcp``); ``hearth mcp`` runs it.
"""

from __future__ import annotations

__all__ = ["build_toolset", "HearthTools"]


def __getattr__(name: str):
    # Lazy re-export so `import hearth.mcp` never pulls the `mcp` SDK (only tools.py, which
    # is dependency-free). server.py is imported explicitly by the CLI when needed.
    if name in __all__:
        from . import tools

        return getattr(tools, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
