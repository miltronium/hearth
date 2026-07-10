#!/usr/bin/env python
"""A local OpenAI-compatible 'frontier' stub for the escalation demo.

Stands in for a real frontier endpoint so HEARTH's escalation path (RemoteProvider `openai`
protocol → httpx → budget accounting → metrics) can be exercised end-to-end with NO external
call and NO cost. Returns a canned answer + a fixed token usage so savings/escalation numbers
are deterministic. This is a demo stand-in only — a real escalation swaps base_url for a real
provider (Anthropic via the `anthropic` protocol, or any OpenAI-compatible endpoint).
"""

from __future__ import annotations

import time
import uuid

import uvicorn
from fastapi import FastAPI

app = FastAPI()


@app.post("/v1/chat/completions")
async def chat(body: dict):
    messages = body.get("messages", [])
    user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
    text = f"[frontier-stub] escalated reasoning answer for: {user[:80]}"
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "frontier-stub"),
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 150, "completion_tokens": 90, "total_tokens": 240},
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8099, log_level="warning")
