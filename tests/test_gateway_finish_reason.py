"""``finish_reason`` must tell the truth on both the streaming and non-streaming paths.

The bug this file guards: ``/v1/chat/completions`` built every choice with the schema's
default ``"stop"``, so a generation cut off at ``max_tokens`` was reported as a normal
completion. For a programmatic consumer — which uses the *non-streaming* path — that
turns a truncated extraction into plausible, silently incomplete data.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from hearth.gateway.schemas import ChatChoice, ChatChoiceMessage


def _parse_sse(text: str) -> list:
    """Return the parsed JSON payloads of each ``data:`` event (excluding [DONE])."""
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            payload = line[len("data: ") :]
            if payload != "[DONE]":
                events.append(json.loads(payload))
    return events


def _chat(client, **overrides):
    body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "hello world"}],
    }
    body.update(overrides)
    return client.post("/v1/chat/completions", json=body)


# -- the schema itself refuses to guess -----------------------------------------------------

def test_chat_choice_requires_an_explicit_finish_reason():
    # The default was the bug: constructing a choice must force the caller to say why.
    with pytest.raises(ValidationError):
        ChatChoice(message=ChatChoiceMessage(content="x"))
    assert ChatChoice(message=ChatChoiceMessage(content="x"), finish_reason="length")


# -- non-streaming --------------------------------------------------------------------------

def test_non_streaming_reports_stop_on_natural_completion(client):
    body = _chat(client).json()
    assert body["choices"][0]["finish_reason"] == "stop"


def test_non_streaming_reports_length_when_the_cap_is_hit(client):
    # 2 tokens of budget for an 18-character echo: truncated, and it must say so.
    r = _chat(client, max_tokens=2)
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "[echo] h"
    assert body["choices"][0]["finish_reason"] == "length"


# -- streaming ------------------------------------------------------------------------------

def test_streaming_reports_stop_on_natural_completion(client):
    chunks = _parse_sse(_chat(client, stream=True).text)
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_streaming_reports_length_when_the_cap_is_hit(client):
    r = _chat(client, stream=True, max_tokens=2)
    assert r.status_code == 200
    chunks = _parse_sse(r.text)
    content = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
    assert content == "[echo] h"
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"


def test_streaming_and_non_streaming_agree(client):
    """The two paths must never disagree about how the same generation ended."""
    for max_tokens in (512, 2):
        non_streaming = _chat(client, max_tokens=max_tokens).json()
        streamed = _parse_sse(_chat(client, stream=True, max_tokens=max_tokens).text)
        assert (
            non_streaming["choices"][0]["finish_reason"]
            == streamed[-1]["choices"][0]["finish_reason"]
        )
