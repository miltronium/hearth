"""Task classifier tests (ARCHITECTURE §3, step 1)."""

from __future__ import annotations

from hearth.providers.base import Message
from hearth.router.classify import classify


def _user(text: str) -> list[Message]:
    return [Message(role="user", content=text)]


def test_intent_hint_short_circuits():
    # A valid intent wins regardless of message content, method="intent".
    cls, method = classify(_user("summarize this"), intent="code")
    assert cls == "code"
    assert method == "intent"


def test_invalid_intent_falls_through_to_rules():
    cls, method = classify(_user("summarize this document"), intent="not-a-class")
    assert cls == "summarize"
    assert method == "rules"


def test_rule_keywords_map_to_classes():
    cases = {
        "please summarize this diff": "summarize",
        "extract all email addresses": "extract",
        "classify this ticket": "classify",
        "rank these candidates": "rank",
        "refactor this function and fix the bug": "code",
        "prove this theorem step by step": "reason",
        "draft a commit message": "draft",
    }
    for text, expected in cases.items():
        cls, method = classify(_user(text))
        assert cls == expected, f"{text!r} -> {cls}"
        assert method == "rules"


def test_unmatched_falls_back_to_chat():
    cls, method = classify(_user("hey there"))
    assert cls == "chat"
    assert method == "rules"


def test_uses_last_user_message():
    messages = [
        Message(role="user", content="ignore me"),
        Message(role="assistant", content="ok"),
        Message(role="user", content="now summarize the report"),
    ]
    cls, _ = classify(messages)
    assert cls == "summarize"


def test_empty_messages_are_chat():
    cls, method = classify([])
    assert cls == "chat"
    assert method == "rules"
