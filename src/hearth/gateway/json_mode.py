"""``response_format`` support for ``/v1/chat/completions``.

**What this is, precisely.** HEARTH does *not* do grammar-constrained decoding. mlx-lm
0.29.1 exposes a ``logits_processors`` hook on ``generate_step``, but it ships no JSON or
grammar constraint to plug into it, and no constraint library (outlines, xgrammar,
lm-format-enforcer) is installed — mlx-lm's own OpenAI server doesn't implement
``response_format`` either. So ``{"type": "json_object"}`` is implemented here as
**instruct-then-validate**: a system instruction steers the model at one end, and the
completion is parsed at the other. If it does not parse, the caller gets a named error
rather than a string that merely looks like JSON.

That distinction matters most in exactly the case this endpoint used to hide: a JSON
object cut off at ``max_tokens`` is *syntactically invalid*, so instruct-then-validate
catches the truncation the caller would otherwise have to notice for themselves. It does
not catch a well-formed object with missing rows — only ``finish_reason == "length"``
tells you that, which is why both signals ship together.

When true constrained decoding becomes available, only :func:`json_instruction` and the
provider call site need to change; the validation below stays as the belt to its braces.
"""

from __future__ import annotations

import json
from typing import Any

from .schemas import ChatMessage, ResponseFormat

# The formats HEARTH implements. Anything else (notably OpenAI's ``json_schema``, which
# needs the constrained decoding we don't have) is refused rather than ignored.
SUPPORTED_FORMATS: tuple[str, ...] = ("text", "json_object")

JSON_INSTRUCTION = (
    "Respond with a single valid JSON object and nothing else. "
    "Do not wrap it in Markdown code fences, and do not add commentary "
    "before or after the object."
)


class UnsupportedResponseFormatError(ValueError):
    """Raised when ``response_format.type`` names a format HEARTH does not implement."""


class InvalidJsonResponseError(ValueError):
    """Raised when a ``json_object`` completion does not parse as a JSON object."""

    def __init__(self, message: str, content: str) -> None:
        super().__init__(message)
        # The raw completion, so the caller can inspect what the model actually produced
        # instead of being left with an error and no output at all.
        self.content = content


def resolve_format(response_format: ResponseFormat | None) -> str:
    """Return the validated format type for a request (``"text"`` when unset).

    Raises :class:`UnsupportedResponseFormatError` for anything outside
    :data:`SUPPORTED_FORMATS` — silently downgrading to plain text would put the caller
    back in the dark, which is the whole failure mode this module exists to remove.
    """
    if response_format is None:
        return "text"
    fmt = response_format.type
    if fmt not in SUPPORTED_FORMATS:
        raise UnsupportedResponseFormatError(
            f"response_format type {fmt!r} is not supported; "
            f"HEARTH implements {', '.join(repr(f) for f in SUPPORTED_FORMATS)}. "
            "Grammar-constrained decoding (json_schema) is not available on this backend."
        )
    return fmt


def json_instruction(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Return ``messages`` with the JSON-mode instruction appended as a system turn.

    Appended rather than prepended so it wins over an earlier, vaguer system prompt, and
    added as a separate message so the caller's own turns are never rewritten. Task
    classification reads the last *user* message (``router.classify``), so this does not
    perturb routing.
    """
    return [*messages, ChatMessage(role="system", content=JSON_INSTRUCTION)]


def parse_json_object(text: str, finish_reason: str) -> Any:
    """Parse a ``json_object`` completion, or raise :class:`InvalidJsonResponseError`.

    Tolerates the one deviation models produce constantly under prompting — a Markdown
    code fence around the object — and nothing else. ``finish_reason`` only sharpens the
    error message: a truncated object and a chatty one fail the same parse, but they call
    for very different fixes on the caller's side.
    """
    candidate = _strip_code_fence(text.strip())
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise InvalidJsonResponseError(_failure_message(finish_reason, str(exc)), text) from exc
    if not isinstance(parsed, dict):
        raise InvalidJsonResponseError(
            "response_format 'json_object' requires a JSON object; the model returned "
            f"a {type(parsed).__name__}.",
            text,
        )
    return parsed


def _failure_message(finish_reason: str, detail: str) -> str:
    """Explain an unparseable completion, naming truncation when that's the cause."""
    if finish_reason == "length":
        return (
            "the model hit the max_tokens cap before completing the JSON object, so the "
            f"output is truncated and does not parse ({detail}). Raise max_tokens or ask "
            "for less in one call."
        )
    return (
        "the model did not return valid JSON under response_format 'json_object' "
        f"({detail}). HEARTH instructs and validates rather than constraining decoding, "
        "so a model can still refuse the format; retry or use a stronger model."
    )


def _strip_code_fence(text: str) -> str:
    """Unwrap a ```/```json fenced block, if the whole completion is one."""
    if not text.startswith("```") or not text.endswith("```"):
        return text
    body = text[3:-3]
    newline = body.find("\n")
    if newline == -1:
        return body.strip()
    # Drop the info string ("json", "JSON", "") that follows the opening fence.
    first_line = body[:newline].strip()
    return (body[newline + 1 :] if not first_line or first_line.isalpha() else body).strip()


__all__ = [
    "JSON_INSTRUCTION",
    "SUPPORTED_FORMATS",
    "InvalidJsonResponseError",
    "UnsupportedResponseFormatError",
    "json_instruction",
    "parse_json_object",
    "resolve_format",
]
