"""Task classification (ARCHITECTURE §3, step 1).

Classify a request into one task class so the policy can pick a backend for it. Phase 2 is
deliberately **deterministic and model-free**: an ``intent`` hint short-circuits, otherwise
cheap keyword/rule heuristics run over the last user message. A tiny local classifier model
is a later enhancement — the hook for it is marked below.
"""

from __future__ import annotations

from ..providers.base import Message

# The closed set of task classes (ARCHITECTURE §3). ``embed`` is included for completeness
# but embeddings route through /v1/embeddings, not the chat router, in Phase 2/3.
TASK_CLASSES: tuple[str, ...] = (
    "summarize",
    "extract",
    "classify",
    "rank",
    "draft",
    "code",
    "reason",
    "chat",
    "embed",
)

# Method labels for how a class was resolved (recorded in telemetry).
METHOD_INTENT = "intent"
METHOD_RULES = "rules"

# Keyword rules, checked in priority order. The first class whose keywords appear in the
# (lowercased) last user message wins. Order matters: more specific/greedy classes first
# so e.g. "rank these" beats the generic "chat" fallback. This is intentionally simple —
# swap in a tiny classifier model here later without changing the router contract.
_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("summarize", ("summarize", "summarise", "summary", "tl;dr", "tldr", "recap", "condense")),
    ("extract", ("extract", "parse out", "pull out", "list the", "find all", "identify all")),
    ("classify", ("classify", "categorize", "categorise", "label this", "which category")),
    ("rank", ("rank", "sort these", "order these", "prioritize", "prioritise", "top ")),
    ("code", ("code", "function", "refactor", "bug", "compile", "stack trace", "traceback",
              "unit test", "implement ", "def ", "class ", "```")),
    ("reason", ("prove", "derive", "reason about", "step by step", "think through",
                "why does", "explain why", "analyze the trade")),
    ("draft", ("draft", "write a", "write an", "compose", "rewrite", "commit message",
               "email", "paragraph")),
)


def classify(messages: list[Message], intent: str | None = None) -> tuple[str, str]:
    """Return ``(task_class, method)`` for a request.

    Resolution order (ARCHITECTURE §3):
      1. If ``intent`` is a valid class, use it (``method="intent"``) — short-circuit.
      2. Otherwise run keyword rules over the last user message (``method="rules"``).

    Falls back to ``chat`` when no rule matches. Never calls a model in Phase 2.
    """
    if intent and intent.lower() in TASK_CLASSES:
        return intent.lower(), METHOD_INTENT

    last_user = _last_user_text(messages).lower()
    for task_class, keywords in _RULES:
        if any(kw in last_user for kw in keywords):
            return task_class, METHOD_RULES

    # --- tiny-model classifier hook -------------------------------------------------
    # Later: when rules are ambiguous, consult a tiny local classifier here and return
    # its label with method="model". Kept out of Phase 2 to stay deterministic/fast.
    return "chat", METHOD_RULES


def _last_user_text(messages: list[Message]) -> str:
    """The content of the most recent ``user`` message, or the last message, or ``""``."""
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return messages[-1].content if messages else ""
