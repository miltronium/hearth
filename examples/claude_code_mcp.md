# Registering HEARTH's MCP server with Claude Code

HEARTH ships an [MCP](https://modelcontextprotocol.io) server so an agent like **Claude
Code** can delegate subtasks (summarize / classify / extract / draft / RAG query) to your
**local** model instead of spending frontier tokens on them (Phase 5, G5; ADR-010).

Every tool runs on HEARTH's router with escalation **disabled**, so the delegated work
never triggers a remote call — it is purely local. (Verify in `src/hearth/mcp/tools.py`:
`_route_local(..., allow_escalation=False)`.)

## Prerequisites

The MCP server lives behind the optional `mcp` extra (it is the only place the `mcp` SDK is
imported; the tool logic in `hearth.mcp.tools` needs no extras):

```sh
uv sync --extra mcp
```

Sanity-check the command exists and starts (Ctrl-C to exit — it speaks stdio and will wait
for a client):

```sh
uv run hearth mcp
```

If the extra is missing, `hearth mcp` fails loudly with the fix hint
(`uv sync --extra mcp`) rather than a traceback.

## The server command

`hearth mcp` launches a **stdio** MCP server named `hearth` (see
`src/hearth/mcp/server.py` → `FastMCP("hearth")`). It registers exactly these tools:

| MCP tool name       | Arguments                                        | What it does (locally)                    |
| ------------------- | ------------------------------------------------ | ----------------------------------------- |
| `hearth_summarize`  | `text: str`, `max_words: int?`                   | Summarize text on the local model         |
| `hearth_classify`   | `text: str`, `labels: [str]`                     | Pick exactly one label                    |
| `hearth_extract`    | `text: str`, `fields: [str]`                     | Return a `{field: value}` map             |
| `hearth_draft`      | `instruction: str`, `context: str?`              | Draft prose/boilerplate (e.g. a message)  |
| `hearth_rag_query`  | `collection: str`, `query: str`, `k: int=6`, `answer: bool=false` | Retrieve local RAG chunks; optionally answer |

## Register it with Claude Code

Claude Code speaks MCP over stdio. Add HEARTH as a **local** MCP server. Two equivalent
ways:

### Option A — the `claude mcp add` CLI

```sh
# Run this from the HEARTH repo root so `uv run` resolves the project env.
claude mcp add hearth -- uv run hearth mcp
```

### Option B — an MCP config JSON block

Add a `hearth` entry to the `mcpServers` map in your Claude Code MCP config
(`~/.claude.json`, or a project `.mcp.json`). Set `cwd` to your HEARTH checkout so
`uv run` finds the project:

```json
{
  "mcpServers": {
    "hearth": {
      "command": "uv",
      "args": ["run", "hearth", "mcp"],
      "cwd": "/absolute/path/to/HEARTH",
      "env": {
        "HEARTH_BACKEND": "mlx"
      }
    }
  }
}
```

Notes:

- `HEARTH_BACKEND` is optional. Omit it (or set `auto`) to use MLX when available and fall
  back to the deterministic `echo` backend otherwise; set `mlx` to force real inference.
  The MCP server reads the same `HEARTH_*` settings as `hearth serve`
  (`src/hearth/config.py`).
- No bearer token is needed here: the MCP server drives HEARTH's router **in-process**, so
  it does not go through the HTTP auth layer. (The HTTP API — used by `examples/cambot_offload.py`
  — does need the `~/.hearth/token` bearer.)
- If you prefer not to depend on `uv` at the MCP layer, use the console script directly:
  `"command": "hearth", "args": ["mcp"]` (requires `hearth` on the server's PATH with the
  `mcp` extra installed).

## Use it from Claude Code

Once registered and Claude Code is restarted, the `hearth_*` tools appear as available MCP
tools. Ask Claude Code to offload work, e.g.:

> Use the `hearth_summarize` tool to summarize this 4k-line log, then only reason over the
> summary.

Each such call runs on your local model — those tokens never hit the frontier budget.

## Confirm the savings

The MCP path shares HEARTH's observability with the HTTP gateway. After a session, read the
rollup (see `docs/RUNBOOK_consumer_wiring.md` for detail):

```sh
uv run hearth stats --since 24h
```

`estimated_frontier_tokens_saved` is the headline number.
