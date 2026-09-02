"""Providers must report *why* generation ended, not just what it produced.

A completion cut off at ``max_tokens`` and a completion that ended at EOS are different
events with the same shape — a string. These tests pin the distinction at the provider
layer, where it originates: :class:`GenResult.finish_reason` for the non-streaming path
and the terminal :class:`StreamDelta` for the streaming one.
"""

from __future__ import annotations

from collections.abc import Iterator

from hearth.providers.base import (
    Capabilities,
    GenRequest,
    GenResult,
    Message,
    ResourceEstimate,
    StreamDelta,
    iter_stream,
    normalize_finish_reason,
)
from hearth.providers.echo import EchoProvider
from hearth.providers.mlx import MLXProvider


def _req(text: str = "hello world", **kwargs) -> GenRequest:
    return GenRequest(messages=[Message(role="user", content=text)], model="m", **kwargs)


# -- normalize_finish_reason ----------------------------------------------------------------

def test_normalize_maps_backend_spellings_of_the_cap():
    assert normalize_finish_reason("length") == "length"
    assert normalize_finish_reason("max_tokens") == "length"


def test_normalize_treats_everything_else_as_a_natural_stop():
    for raw in (None, "stop", "end_turn", "stop_sequence", "eos"):
        assert normalize_finish_reason(raw) == "stop"


# -- defaults -------------------------------------------------------------------------------

def test_gen_result_defaults_to_stop():
    # A provider that cannot tell keeps working; it must not have to opt in to a value.
    assert GenResult(text="x", model="m", backend="b").finish_reason == "stop"


# -- echo: simulates both outcomes ----------------------------------------------------------

def test_echo_reports_stop_when_the_answer_fits():
    result = EchoProvider().generate(_req())
    assert result.text == "[echo] hello world"
    assert result.finish_reason == "stop"


def test_echo_reports_length_when_it_hits_the_cap():
    # 2 tokens * ~4 chars = 8 chars of budget for an 18-char echo.
    result = EchoProvider().generate(_req(max_tokens=2))
    assert result.finish_reason == "length"
    assert result.text == "[echo] h"


def test_echo_streaming_agrees_with_generate():
    for max_tokens, expected in ((512, "stop"), (2, "length")):
        events = list(EchoProvider().stream_deltas(_req(max_tokens=max_tokens)))
        text = "".join(e.text for e in events)
        assert text == EchoProvider().generate(_req(max_tokens=max_tokens)).text
        assert events[-1].finish_reason == expected
        # Only the terminal event carries a reason.
        assert all(e.finish_reason is None for e in events[:-1])


def test_echo_plain_stream_still_yields_text_only():
    # The str-yielding contract other callers rely on is unchanged.
    assert "".join(EchoProvider().stream(_req())) == "[echo] hello world"


# -- iter_stream adapter --------------------------------------------------------------------

class _TextOnlyProvider:
    """A provider from before ``stream_deltas`` existed — yields bare strings."""

    name = "text-only"

    def capabilities(self) -> Capabilities:
        return Capabilities(chat=True, stream=True)

    def generate(self, req: GenRequest) -> GenResult:  # pragma: no cover - unused here
        return GenResult(text="", model=req.model, backend=self.name)

    def stream(self, req: GenRequest) -> Iterator[str]:
        yield from ("a", "b")

    def footprint(self, model_id: str) -> ResourceEstimate:  # pragma: no cover - unused
        return ResourceEstimate()


def test_iter_stream_uses_the_rich_stream_when_available():
    events = list(iter_stream(EchoProvider(), _req(max_tokens=2)))
    assert events[-1].finish_reason == "length"


def test_iter_stream_falls_back_to_a_text_only_provider():
    events = list(iter_stream(_TextOnlyProvider(), _req()))
    assert [e.text for e in events[:-1]] == ["a", "b"]
    assert events[-1] == StreamDelta(finish_reason="stop")


# -- mlx: normalization of mlx-lm's own finish_reason ---------------------------------------

def test_mlx_passes_through_mlx_lm_finish_reasons():
    # mlx-lm 0.29.1 tags each GenerationResponse "stop" | "length" | None.
    assert MLXProvider._finish_reason("length") == "length"
    assert MLXProvider._finish_reason("stop") == "stop"
    # None = the stream ended before mlx-lm reported (we cut at a terminator) = a stop.
    assert MLXProvider._finish_reason(None) == "stop"


class _FakeResponse:
    """The shape of ``mlx_lm.generate.GenerationResponse`` this provider reads."""

    def __init__(self, text: str, finish_reason: str | None) -> None:
        self.text = text
        self.finish_reason = finish_reason


class _FakeTokenizer:
    eos_token = None

    def encode(self, text):
        return list(text)

    def apply_chat_template(self, chat, tokenize, add_generation_prompt):
        return "PROMPT"


def _mlx_provider_over(responses: list[_FakeResponse], monkeypatch) -> MLXProvider:
    """An MLXProvider whose ``stream_generate`` replays ``responses`` (no model, no MLX)."""
    import sys
    import types

    provider = MLXProvider("org/model")
    provider._cache[""] = (object(), _FakeTokenizer())

    module = types.ModuleType("mlx_lm")
    module.stream_generate = lambda *a, **kw: iter(responses)
    monkeypatch.setitem(sys.modules, "mlx_lm", module)
    return provider


def test_mlx_reports_length_when_generation_hits_the_cap(monkeypatch):
    responses = [
        _FakeResponse("partial ", None),
        _FakeResponse("answer", "length"),
    ]
    provider = _mlx_provider_over(responses, monkeypatch)
    events = list(provider.stream_deltas(_req(max_tokens=2)))
    assert "".join(e.text for e in events) == "partial answer"
    assert events[-1].finish_reason == "length"

    # generate() is built on the same stream, so the two paths cannot disagree.
    provider = _mlx_provider_over(responses, monkeypatch)
    result = provider.generate(_req(max_tokens=2))
    assert result.text == "partial answer"
    assert result.finish_reason == "length"


def test_mlx_reports_stop_when_cut_at_a_terminator_marker(monkeypatch):
    # The LoRA ramble: mlx-lm would run to the cap, but the turn genuinely ended at
    # <|im_end|>, so the client sees the clean answer *and* an honest "stop".
    responses = [
        _FakeResponse("QX-2<|im_end|> !", None),
        _FakeResponse(" !", "length"),
    ]
    provider = _mlx_provider_over(responses, monkeypatch)
    result = provider.generate(_req(max_tokens=64))
    assert result.text == "QX-2"
    assert result.finish_reason == "stop"
