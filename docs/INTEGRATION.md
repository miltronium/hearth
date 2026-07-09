# HEARTH — Integration Guide

**Status:** Draft. How consumers use HEARTH. The rule: **no consumer is privileged.**
CAMBOT and Claude Code are examples, not special cases.

---

## The mental model

HEARTH is a local resource that speaks OpenAI. Any client that can make an HTTP call — or
speak MCP — can use it. Three integration surfaces:

1. **HTTP (OpenAI-compatible)** — anything with an OpenAI SDK or `curl`.
2. **MCP server** — agents that speak MCP (Claude Code) get HEARTH as a tool.
3. **Swift SDK / embedded** — Swift apps (CAMBOT) call HEARTH over HTTP, or embed it
   in-process for offline on-device inference.

---

## 1. CAMBOT (Swift core + Python MCP)

CAMBOT is the first consumer. Two ways in:

**a. Over HTTP via the Swift SDK (Phases 5).** CAMBOT's Swift code calls HEARTH's local
endpoint for offload — e.g. summarizing device logs before they enter a frontier prompt,
classifying a command intent, drafting a report section.

```swift
// Illustrative — HEARTH Swift SDK
let hearth = HearthClient(baseURL: URL(string: "http://127.0.0.1:8080")!, token: token)
let summary = try await hearth.summarize(text: deviceLog, maxWords: 120)   // local, 0 frontier tokens
let intent  = try await hearth.classify(text: userCommand, labels: [.query, .action, .config])
```

**b. Embedded, offline (Phase 6).** For fully offline on-device inference, CAMBOT links the
HEARTH Swift package's embedded mode (Apple Foundation Models / Core ML) — no daemon,
no network. Same call shapes, in-process.

**Boundary rule:** HEARTH exposes *no* CAMBOT types. CAMBOT depends on HEARTH; never the
reverse. HEARTH's conformance suite runs with no CAMBOT present (see Testing below).

Where offload pays off in CAMBOT specifically:
- Condensing verbose device/tool output before it hits a frontier context window.
- Ranking/filtering search results across the device inventory.
- Drafting commit messages, changelog lines, status-report prose.
- Classifying intents / extracting structured fields from natural-language commands.
- Local RAG over CAMBOT's own docs (`docs/stack`, phase plans) for grounded answers.

---

## 2. Claude Code — delegate subtasks to save frontier tokens

This is the direct answer to "I keep hitting my token limit." HEARTH ships an **MCP server**
(Phase 5). Register it with Claude Code, and Claude can call HEARTH tools to hand routine
subtasks to the local model — the work never spends your frontier budget.

```jsonc
// .mcp.json (or Claude Code MCP config)
{
  "mcpServers": {
    "hearth": { "command": "hearth", "args": ["mcp"] }
  }
}
```

Tools the HEARTH MCP server exposes:

| Tool | Use from Claude Code |
| --- | --- |
| `hearth_summarize` | "Summarize this file/log locally" before reading it into context |
| `hearth_classify` | Route/label something without a frontier round-trip |
| `hearth_extract` | Pull structured fields from text locally |
| `hearth_rag_query` | Retrieve grounded chunks from a local collection instead of pasting whole files |
| `hearth_draft` | First-draft boilerplate/commit messages locally |

**Workflow pattern:** for a big task, let Claude *orchestrate* (the reasoning) while HEARTH
*does the bulk reads/summaries* (the volume). Claude asks HEARTH to pre-digest large inputs,
then reasons over the compact result. That's the split that stretches your limit furthest.

> Note: this complements — doesn't replace — Claude Code's own token hygiene (subagents that
> return conclusions, scoped reads, `/compact`). HEARTH is the "send the grunt work
> elsewhere" half of the strategy.

---

## 3. Generic OpenAI client

Any tool that speaks the OpenAI API:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8080/v1
export OPENAI_API_KEY=$(cat ~/.hearth/token)
# now your existing scripts/SDKs hit local models transparently
```

```python
from openai import OpenAI
client = OpenAI()  # picks up OPENAI_BASE_URL / OPENAI_API_KEY
r = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "extract the ticket ids from this text: ..."}],
    extra_body={"hearth": {"intent": "extract", "allow_escalation": False}},
)
```

---

## 4. Shell / CI

```bash
hearth run "write a conventional-commit message for this diff" --stdin < diff.txt
hearth rag ingest ./Sources      # index a repo
hearth rag query "where is auth handled?" --collection cambot
hearth stats --since 7d          # how many frontier tokens did we save this week?
```

---

## Testing: proving client-agnosticism

HEARTH's guarantee is that it works for *any* client. Enforce it:

- **Conformance suite** exercises the full API (core + extensions) with **no CAMBOT and no
  Claude Code present** — just HTTP. If it passes, HEARTH is genuinely standalone.
- **Contract tests** for each `ModelProvider` verify the interface, so backends stay swappable.
- **SDK parity tests** ensure the Swift SDK and Python client produce identical request
  shapes against the same endpoints.

---

## Configuration a consumer cares about

| Setting | Where | Effect |
| --- | --- | --- |
| `base_url` / token | client env or config | how to reach HEARTH |
| `intent` hint | per request | skip classification, pin task class |
| `allow_escalation` | per request | hard-pin local (privacy / cost) or permit remote |
| `adapter` | per request | request a specific fine-tuned adapter |
| routing policy | HEARTH `routing.yaml` | global local-vs-remote defaults (not per client) |

Clients express *intent and constraints*; HEARTH owns *policy*. That separation is what
keeps every client simple and the routing logic in one place.
