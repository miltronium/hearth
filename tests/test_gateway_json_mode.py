"""``response_format`` on ``/v1/chat/completions``.

Two guarantees, in order of importance:

  1. The field can never be **dropped**. It used to be absent from the request schema
     entirely, so an OpenAI client asking for JSON got prose and no indication that its
     request had been ignored. An unsupported value is now a named 400.
  2. ``json_object`` output is **validated**, not hoped for. HEARTH prompts and parses (it
     does not constrain decoding — see :mod:`hearth.gateway.json_mode`), so the parse is
     the only thing standing between a caller and a malformed or truncated object.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from hearth.gateway import create_app
from hearth.gateway.json_mode import (
    JSON_INSTRUCTION,
    InvalidJsonResponseError,
    UnsupportedResponseFormatError,
    json_instruction,
    parse_json_object,
    resolve_format,
)
from hearth.gateway.schemas import ChatMessage, ResponseFormat
from hearth.observability.budget import BudgetAccountant
from hearth.observability.metrics import MetricsStore
from hearth.providers.base import (
    Capabilities,
    GenRequest,
    GenResult,
    ResourceEstimate,
    StreamDelta,
)
from hearth.router import Router


class _ScriptedProvider:
    """Returns a fixed completion with a fixed stop reason, on both paths."""

    name = "scripted"

    def __init__(self, text: str, finish_reason: str = "stop") -> None:
        self.text = text
        self.finish_reason = finish_reason
        self.seen: list[GenRequest] = []

    def capabilities(self) -> Capabilities:
        return Capabilities(chat=True, embed=False, stream=True, adapters=False)

    def generate(self, req: GenRequest) -> GenResult:
        self.seen.append(req)
        return GenResult(
            text=self.text,
            model=req.model,
            backend=self.name,
            prompt_tokens=1,
            completion_tokens=1,
            finish_reason=self.finish_reason,
        )

    def stream_deltas(self, req: GenRequest) -> Iterator[StreamDelta]:
        self.seen.append(req)
        yield StreamDelta(text=self.text)
        yield StreamDelta(finish_reason=self.finish_reason)

    def stream(self, req: GenRequest) -> Iterator[str]:
        for delta in self.stream_deltas(req):
            if delta.text:
                yield delta.text

    def footprint(self, model_id: str) -> ResourceEstimate:
        return ResourceEstimate(ram_gb=0.0)


@pytest.fixture
def scripted(settings, local_policy):
    """Build ``(provider, client)`` over a provider whose output the test dictates."""

    def _build(text: str, finish_reason: str = "stop"):
        provider = _ScriptedProvider(text, finish_reason)
        router = Router(
            local_provider=provider,
            policy=local_policy,
            budget=BudgetAccountant(local_policy.defaults.remote_budget_tokens_per_day),
            metrics=MetricsStore(),
        )
        app = create_app(
            provider=provider,
            settings=settings.model_copy(update={"warmup": False}),
            router=router,
        )
        return provider, TestClient(app)

    return _build


def _chat(client, **overrides):
    body = {"model": "auto", "messages": [{"role": "user", "content": "extract totals"}]}
    body.update(overrides)
    return client.post("/v1/chat/completions", json=body)


def _parse_sse(text: str) -> list:
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            payload = line[len("data: ") :]
            if payload != "[DONE]":
                events.append(json.loads(payload))
    return events


# -- the field cannot be dropped -------------------------------------------------------------

def test_response_format_is_accepted_by_the_schema(client):
    # Absent from the schema, this used to be discarded by pydantic without a word.
    r = _chat(client, response_format={"type": "text"})
    assert r.status_code == 200


def test_unsupported_response_format_is_a_named_error(client):
    r = _chat(client, response_format={"type": "json_schema"})
    assert r.status_code == 400
    error = r.json()["error"]
    assert error["code"] == "hearth.response_format.unsupported"
    assert error["param"] == "response_format"
    # The message must name what *is* supported, not just what isn't.
    assert "json_object" in error["message"]


def test_unsupported_response_format_never_reaches_the_provider(scripted):
    provider, client = scripted('{"ok": true}')
    assert _chat(client, response_format={"type": "json_schema"}).status_code == 400
    assert provider.seen == []


def test_unsupported_response_format_errors_on_the_streaming_path_too(client):
    r = _chat(client, stream=True, response_format={"type": "json_schema"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "hearth.response_format.unsupported"


# -- json_object: the happy path -------------------------------------------------------------

def test_json_object_returns_the_parsed_object_unchanged(scripted):
    provider, client = scripted('{"total": 3, "rows": []}')
    r = _chat(client, response_format={"type": "json_object"})
    assert r.status_code == 200
    content = r.json()["choices"][0]["message"]["content"]
    assert json.loads(content) == {"total": 3, "rows": []}


def test_json_object_instructs_the_model(scripted):
    provider, client = scripted('{"ok": true}')
    _chat(client, response_format={"type": "json_object"})
    # Instruct-then-validate: the instruction is appended as its own system turn.
    assert provider.seen[0].messages[-1].role == "system"
    assert provider.seen[0].messages[-1].content == JSON_INSTRUCTION


def test_text_format_does_not_instruct(scripted):
    provider, client = scripted("plain prose")
    _chat(client, response_format={"type": "text"})
    assert [m.content for m in provider.seen[0].messages] == ["extract totals"]


def test_json_object_unwraps_a_markdown_fence(scripted):
    provider, client = scripted('```json\n{"total": 3}\n```')
    assert _chat(client, response_format={"type": "json_object"}).status_code == 200


# -- json_object: the failure that matters ---------------------------------------------------

def test_truncated_json_is_an_error_not_a_plausible_answer(scripted):
    # The financial-extraction failure shape: a JSON object cut off at max_tokens.
    provider, client = scripted('{"rows": [{"amount": 12.5}, {"amou', finish_reason="length")
    r = _chat(client, response_format={"type": "json_object"})
    assert r.status_code == 422
    body = r.json()
    assert body["error"]["code"] == "hearth.response_format.invalid_json"
    assert "max_tokens" in body["error"]["message"]
    # The raw completion and the stop reason ride along so the caller can act on it.
    assert body["hearth"]["finish_reason"] == "length"
    assert body["hearth"]["content"] == '{"rows": [{"amount": 12.5}, {"amou'


def test_prose_under_json_object_is_an_error(scripted):
    provider, client = scripted("Sure! Here are the totals.")
    r = _chat(client, response_format={"type": "json_object"})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "hearth.response_format.invalid_json"


def test_json_array_is_rejected_as_not_an_object(scripted):
    provider, client = scripted("[1, 2, 3]")
    r = _chat(client, response_format={"type": "json_object"})
    assert r.status_code == 422
    assert "object" in r.json()["error"]["message"]


def test_echo_backend_cannot_pretend_to_do_json_mode(client):
    # The stub emits "[echo] ...", so json_object over it must fail loudly rather than
    # hand back a string that is not JSON.
    r = _chat(client, response_format={"type": "json_object"})
    assert r.status_code == 422


# -- json_object: streaming -------------------------------------------------------------------

def test_streaming_json_object_passes_valid_output_through(scripted):
    provider, client = scripted('{"total": 3}')
    chunks = _parse_sse(_chat(client, stream=True, response_format={"type": "json_object"}).text)
    content = "".join(
        c["choices"][0]["delta"].get("content") or "" for c in chunks if "choices" in c
    )
    assert json.loads(content) == {"total": 3}
    assert not [c for c in chunks if "error" in c]


def test_streaming_json_object_reports_invalid_output_before_done(scripted):
    provider, client = scripted('{"rows": [{"amou', finish_reason="length")
    r = _chat(client, stream=True, response_format={"type": "json_object"})
    chunks = _parse_sse(r.text)
    errors = [c for c in chunks if "error" in c]
    assert len(errors) == 1
    assert errors[0]["error"]["code"] == "hearth.response_format.invalid_json"
    assert errors[0]["hearth"]["finish_reason"] == "length"
    # The error is the last event before the terminator.
    assert r.text.rstrip().endswith("data: [DONE]")


# -- json_mode unit level ---------------------------------------------------------------------

def test_resolve_format_defaults_to_text():
    assert resolve_format(None) == "text"
    assert resolve_format(ResponseFormat()) == "text"
    assert resolve_format(ResponseFormat(type="json_object")) == "json_object"


def test_resolve_format_rejects_unknown_types():
    for bad in ("json", "json_schema", "xml", ""):
        with pytest.raises(UnsupportedResponseFormatError):
            resolve_format(ResponseFormat(type=bad))


def test_json_instruction_appends_without_rewriting_the_caller_turns():
    original = [ChatMessage(role="user", content="extract totals")]
    result = json_instruction(original)
    assert len(original) == 1  # not mutated
    assert result[0] == original[0]
    assert result[-1].role == "system"


def test_parse_json_object_accepts_a_bare_object():
    assert parse_json_object('{"a": 1}', "stop") == {"a": 1}


def test_parse_json_object_error_names_truncation_only_when_truncated():
    with pytest.raises(InvalidJsonResponseError) as truncated:
        parse_json_object('{"a": ', "length")
    assert "max_tokens" in str(truncated.value)

    with pytest.raises(InvalidJsonResponseError) as refused:
        parse_json_object("no json here", "stop")
    assert "max_tokens" not in str(refused.value)
    assert refused.value.content == "no json here"


def test_parse_json_object_unwraps_fences():
    assert parse_json_object('```json\n{"a": 1}\n```', "stop") == {"a": 1}
    assert parse_json_object('```\n{"a": 1}\n```', "stop") == {"a": 1}
