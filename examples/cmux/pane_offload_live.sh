#!/bin/zsh
# pane_offload_live.sh — C2 LIVE proof: a real coding agent, in a real cmux pane, offloading a
# subtask to local HEARTH over MCP. This is the GUI-path counterpart to offload_demo.py (which
# drives the same build_toolset -> Router path in-process). See docs/RUNBOOK_wiring.md §5.
#
# Unlike the demo, this asserts on the AGENT'S OWN TRANSCRIPT: it greps the stream-json tool_use
# records for `mcp__hearth__*`, so "the agent said it used the tool" is not accepted as evidence.
#
# Usage (from inside a cmux pane, or via `cmux send`):
#   ./pane_offload_live.sh <file-to-summarize> [mcp-config.json]
#
# Prereqs (both matter — see the FAILURE MODE note below):
#   uv sync --extra mlx --extra mcp        # the `mcp` extra is REQUIRED for `hearth mcp`
#   an mcp config pointing at an ABSOLUTE `hearth` path (see hearth.mcp.json)

set -u
TARGET="${1:?usage: pane_offload_live.sh <file> [mcp-config.json]}"
MCP_CONFIG="${2:-hearth.live.mcp.json}"
REPO="${HEARTH_REPO:-/Users/miltronix/Claude/apps/HEARTH}"

source "$REPO/examples/cmux/sealed-pane.env"

# Inline of the `claude-me` zsh function (~/.zshrc) — it is a shell FUNCTION, so it does not exist
# in a script shell (`command not found: claude-me`). Same isolation: strip Apple/corp env, point
# at the personal config dir. Adjust CLAUDE_BIN for a different agent install.
unset -m 'ANTHROPIC_*' 'APPLE_CLAUDE_CODE_*' 'CLAUDE_CODE_*'
unset NODE_EXTRA_CA_CERTS
export CLAUDE_CONFIG_DIR="${CLAUDE_CONFIG_DIR:-$HOME/.claude-me}"
CLAUDE_BIN="${CLAUDE_BIN:-$HOME/.claude-me-cli/node_modules/.bin/claude}"

OUT="${OFFLOAD_OUT:-offload_run.jsonl}"
echo "=== MCP-OFFLOAD-START target=$TARGET ==="

# --output-format stream-json is REQUIRED to see tool_use records. Plain `--output-format json`
# returns only the final result object, so a tool-call assertion against it always reports zero
# calls even when the offload demonstrably happened.
"$CLAUDE_BIN" -p "Read $TARGET and then summarize it in 25 words by calling the mcp__hearth__hearth_summarize tool with the file contents as the \`text\` argument. Do NOT write the summary yourself — the tool output IS the answer. Print the tool result verbatim." \
  --mcp-config "$MCP_CONFIG" \
  --allowedTools "Read,mcp__hearth__hearth_summarize" \
  --output-format stream-json --verbose > "$OUT" 2>&1
rc=$?

python3 - "$OUT" <<'PY'
import json, sys

# stream-json is JSONL: one object per line.
msgs = []
for line in open(sys.argv[1]):
    line = line.strip()
    if not line:
        continue
    try:
        msgs.append(json.loads(line))
    except json.JSONDecodeError:
        pass

if not msgs:
    print("VERDICT= NO_TRANSCRIPT (agent failed to start — check the head of the output file)")
    raise SystemExit(1)

calls = []
def walk(o):
    if isinstance(o, dict):
        if o.get("type") == "tool_use":
            calls.append(o.get("name"))
        for v in o.values():
            walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)
walk(msgs)

hearth = [c for c in calls if c and "hearth" in c]
print("TOOL_CALLS=", calls)
print("HEARTH_TOOL_CALLS=", hearth)
print("VERDICT=", "OFFLOADED" if hearth else "NOT_OFFLOADED")

for m in msgs:
    if isinstance(m, dict) and m.get("type") == "result" and m.get("result"):
        print("RESULT=", str(m["result"]).strip()[:400])
        break
PY
echo "=== MCP-OFFLOAD-END rc=$rc ==="

# FAILURE MODE (found live 2026-08-17): if the `mcp` extra is not installed, `hearth mcp` exits with
# "The MCP server requires the 'mcp' extra." — and Claude Code drops the server SILENTLY. The agent
# then reports "no hearth tools are registered" and the run looks like a wiring/config mistake
# rather than a missing dependency. Verify the server independently before blaming the config:
#   printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"probe","version":"1"}}}' \
#     '{"jsonrpc":"2.0","method":"notifications/initialized"}' '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
#     | .venv/bin/hearth mcp | grep -o '"name":"hearth_[a-z_]*"'
