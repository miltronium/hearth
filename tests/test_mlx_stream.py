"""MLXProvider terminator handling — pure string logic, no MLX / no model.

MLXProvider.__init__ loads nothing (the heavy import is deferred), so the terminator
helpers can be exercised directly with a null tokenizer. Covers both the non-streaming
cut (`_strip_terminators`) and the streaming cleaner (`_clean_stream`), including the
LoRA-tuned failure mode where the model emits the literal `<|im_end|>` mid-output and
rambles instead of stopping at EOS.
"""

from __future__ import annotations

from hearth.providers.mlx import MLXProvider


def _provider() -> MLXProvider:
    # No load() is called; _tokenizer stays None and _terminator_markers falls back to the
    # static chat markers (<|im_end|>, <|endoftext|>, <|eot_id|>).
    return MLXProvider("org/model")


def _stream(chunks: list[str]) -> str:
    return "".join(_provider()._clean_stream(chunks))


# -- _first_terminator / _strip_terminators -------------------------------------------------

def test_strip_leaves_clean_text_untouched():
    assert _provider()._strip_terminators("just the answer") == "just the answer"


def test_strip_cuts_at_first_marker_and_drops_ramble():
    # The exact LoRA-tuned failure mode seen on real weights.
    assert _provider()._strip_terminators("QX-2<|im_end|> !<|im_end|> !") == "QX-2"


def test_strip_handles_trailing_marker():
    assert _provider()._strip_terminators("answer<|endoftext|>") == "answer"


def test_strip_marker_at_start_yields_empty():
    assert _provider()._strip_terminators("<|im_end|>anything") == ""


def test_first_terminator_reports_earliest():
    p = _provider()
    assert p._first_terminator("no markers here") is None
    assert p._first_terminator("a<|eot_id|>b<|im_end|>c") == 1


# -- _clean_stream --------------------------------------------------------------------------

def test_stream_reassembles_clean_output_in_order():
    assert _stream(["The quick ", "brown fox ", "jumps over"]) == "The quick brown fox jumps over"


def test_stream_stops_at_mid_stream_marker():
    # Model answers correctly then fails to stop at EOS and rambles — client sees only "QX-2".
    assert _stream(["QX-2<|im_end|> !", "<|im_end|> !", "<|im_end|> !"]) == "QX-2"


def test_stream_does_not_leak_marker_split_across_chunks():
    # "<|im_end|>" is split across three chunks; it must never reach the client.
    assert _stream(["hello ", "<|im", "_end|> trailing junk"]) == "hello "


def test_stream_empty_when_marker_leads():
    assert _stream(["<|im_end|>", " junk"]) == ""
