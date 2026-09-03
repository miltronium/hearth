"""The output contract between the loop and a local model, and its strict parser.

**Why prompt-based tool calling and not a native one.** mlx-lm 0.29.1 *does* carry tool-call
machinery, and the Qwen2.5 weights in ``~/.hearth/models`` do declare the ``<tool_call>`` /
``</tool_call>`` special tokens it keys off. But that machinery lives in ``mlx_lm.server`` —
the OpenAI-compatible HTTP server — not in ``mlx_lm.generate``, and it is reached by passing
``tools=`` through ``tokenizer.apply_chat_template`` and then watching the token stream for
those two sentinels. HEARTH's provider contract has no seam for it: :class:`GenRequest` has no
``tools`` field and :class:`GenResult` has no ``tool_calls``, so a native path would mean
changing every provider *and* would exist only for models whose tokenizer happens to declare
those tokens — untestable on the echo backend, and silently absent on anything else. A
capability that works on one model and vanishes on the next is worse than one that works
everywhere, because the failure is invisible at the call site.

So the contract is a prompt contract, and it is deliberately shaped to be the *same payload*
mlx-lm's native parser produces — ``{"name": ..., "arguments": {...}}`` — so a model that
emits a native ``<tool_call>`` block parses here too, unchanged. If a ``tools`` field ever
lands in the provider contract, the wire shape does not have to move.

**Parsing is strict where ambiguity would be dangerous and forgiving where it would only be
annoying.** Refused outright, because guessing would run the wrong thing:

  * no JSON object at all;
  * *more than one* JSON object — two tool calls in one turn must never resolve to "the first
    one", which is how an agent quietly skips a step it told you it took;
  * both ``tool`` and ``answer``, or neither — the model has not decided what it is doing;
  * ``arguments`` that is not an object.

Absorbed silently, because they are formatting noise a 3B model produces constantly and none
of them change meaning: markdown fences, a ``<think>`` block, a ``<tool_call>`` wrapper, prose
around the object, the ``name``/``parameters`` spelling, and ``arguments`` delivered as a
JSON *string* rather than an object.

Every refusal raises :class:`ProtocolError`, whose message is written for the model — it says
what was wrong and restates the contract — because the loop feeds it straight back as the next
observation. A parse failure is a recoverable turn, not a crash; :mod:`hearth.agent.loop`
decides when a run of them means the model cannot follow the contract at all.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .tools import ToolRegistry

# Reasoning models wrap scratch work in these. It is not part of the contract and stripping
# it is not a judgement call — the block is explicitly delimited by the tokenizer.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# The native mlx-lm / Qwen tool-call sentinels. When present we parse what is *between* them,
# which is the same JSON object this contract asks for in plain text.
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)

# ```json ... ``` and bare ``` ... ``` fences.
_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*\n?(.*?)```", re.DOTALL)

#: Key spellings accepted for the tool name. ``name`` is mlx-lm's native shape.
_TOOL_KEYS = ("tool", "name")
#: Key spellings accepted for the argument object.
_ARG_KEYS = ("arguments", "args", "parameters")
#: Key spellings accepted for a final answer.
_ANSWER_KEYS = ("answer", "final_answer")


class ProtocolError(ValueError):
    """The model's output did not satisfy the contract.

    Recoverable: the loop records it and hands the message back to the model. The message is
    therefore phrased as an instruction to the model, not as a diagnostic for a developer.
    """


@dataclass(frozen=True)
class ToolCall:
    """The model asked to run a tool. Not yet validated against that tool's schema."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    thought: str = ""


@dataclass(frozen=True)
class FinalAnswer:
    """The model says it is done. Whether it is *right* is a separate question the loop
    cannot answer — see ``docs/AGENT.md`` on verification staying outside the model."""

    text: str
    thought: str = ""


#: What one turn of model output resolves to.
ModelAction = ToolCall | FinalAnswer


CONTRACT = """\
Reply with exactly ONE JSON object and nothing else. No prose before or after it.

To use a tool:
{"thought": "why this step", "tool": "<tool name>", "arguments": {"<param>": <value>}}

To finish, when you can answer from what you have already observed:
{"thought": "why this is enough", "answer": "<your answer>"}

Rules:
- Exactly one JSON object per reply. Never two. Never a tool call and an answer together.
- Use only the tools listed above, spelled exactly as listed, with only their listed parameters.
- Arguments must be a JSON object, even when empty: "arguments": {}.
- Do not invent an observation. If you have not called a tool, you have not seen its result.
- If a step fails, read the error and either fix the call or answer with what you do know,
  saying what you could not determine. A wrong answer stated confidently is the worst outcome.\
"""


def render_system_prompt(
    registry: ToolRegistry,
    *,
    task: str,
    guidance: str = "",
) -> str:
    """Build the system prompt: the role, the tools, the contract, and the task.

    Rebuilt every turn from the registry rather than cached, so a tool cannot be described to
    the model and then be absent when it is called — the description and the dispatch table
    are the same object by construction (``CLAUDE.md`` §3).
    """
    blocks = [
        "You are HEARTH's local agent. You work on this machine only. You reach data solely "
        "through the tools below; you have no network, no shell, and no way to write files.",
        "",
        "Tools:",
        registry.render(),
        "",
        CONTRACT,
    ]
    if guidance.strip():
        blocks.extend(["", guidance.strip()])
    blocks.extend(["", "Task:", task.strip()])
    return "\n".join(blocks)


def parse_action(raw: str) -> ModelAction:
    """Parse one turn of model output into a :class:`ToolCall` or a :class:`FinalAnswer`.

    Raises :class:`ProtocolError` for anything the contract does not permit. See the module
    docstring for exactly which deviations are absorbed and which are refused.
    """
    payload = _sole_object(raw)

    tool_name = _first_key(payload, _TOOL_KEYS)
    answer = _first_key(payload, _ANSWER_KEYS)
    thought = _as_text(payload.get("thought", ""))

    if tool_name is not None and answer is not None:
        raise ProtocolError(
            "your reply contained both a tool call and an answer. Decide: call one tool, or "
            "give the answer. Emit exactly one JSON object with either `tool` or `answer`."
        )
    if tool_name is None and answer is None:
        keys = ", ".join(sorted(str(k) for k in payload)) or "(none)"
        raise ProtocolError(
            f"your JSON object had neither a `tool` key nor an `answer` key (it had: {keys}). "
            "Emit one object with either `tool` plus `arguments`, or `answer`."
        )

    if answer is not None:
        text = _as_text(answer)
        if not text.strip():
            raise ProtocolError(
                "your `answer` was empty. Either answer the task, or call a tool to get what "
                "you still need."
            )
        return FinalAnswer(text=text.strip(), thought=thought)

    name = _as_text(tool_name).strip()
    if not name:
        raise ProtocolError("your `tool` was empty. Name one of the tools listed above.")
    return ToolCall(name=name, arguments=_arguments(payload, name), thought=thought)


def _sole_object(raw: str) -> dict[str, Any]:
    """Extract the one JSON object in ``raw``, refusing zero or several.

    ``json.loads`` on the whole string is not enough: models wrap the object in fences, in a
    ``<tool_call>`` block, or in a sentence. So we normalise those away, then scan for
    top-level objects with ``raw_decode``. Finding two is an error rather than a
    take-the-first, because the second one is usually the step the model actually meant and
    executing the first silently drops it.
    """
    text = _THINK_RE.sub(" ", raw or "")

    wrapped = _TOOL_CALL_RE.findall(text)
    if wrapped:
        if len(wrapped) > 1:
            raise ProtocolError(
                "your reply contained more than one <tool_call> block. Emit exactly one tool "
                "call per reply."
            )
        text = wrapped[0]
    else:
        fenced = _FENCE_RE.findall(text)
        if len(fenced) == 1:
            text = fenced[0]
        elif len(fenced) > 1:
            raise ProtocolError(
                "your reply contained more than one code block. Emit exactly one JSON object "
                "and nothing else."
            )

    objects = _decode_objects(text)
    if not objects:
        raise ProtocolError(
            "your reply contained no JSON object. Reply with exactly one JSON object, for "
            'example {"thought": "...", "tool": "...", "arguments": {}}.'
        )
    if len(objects) > 1:
        raise ProtocolError(
            f"your reply contained {len(objects)} JSON objects. Emit exactly one — one step "
            "per reply, so each step's result can be observed before the next."
        )
    return objects[0]


def _decode_objects(text: str) -> list[dict[str, Any]]:
    """Return every top-level JSON *object* in ``text``, left to right, non-overlapping."""
    decoder = json.JSONDecoder()
    found: list[dict[str, Any]] = []
    index = 0
    while True:
        start = text.find("{", index)
        if start < 0:
            return found
        try:
            value, end = decoder.raw_decode(text, start)
        except ValueError:
            index = start + 1
            continue
        if isinstance(value, dict):
            found.append(value)
        index = max(end, start + 1)


def _first_key(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Return the value of the first present key in ``keys``, or ``None`` if none are."""
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def _arguments(payload: dict[str, Any], tool: str) -> dict[str, Any]:
    """Pull the argument object out, accepting a JSON *string* and an omitted key.

    A missing ``arguments`` is read as ``{}`` rather than refused: for a zero-parameter tool
    that is the correct call, and for one with required parameters the schema check in
    :meth:`hearth.agent.tools.Tool.validate` produces a far better message than a generic
    protocol complaint would — it names the parameter the model forgot.
    """
    value = _first_key(payload, _ARG_KEYS)
    if value is None:
        return {}
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            value = json.loads(stripped)
        except ValueError:
            raise ProtocolError(
                f"the `arguments` for {tool!r} were a string that is not valid JSON. Send "
                'arguments as a JSON object, e.g. "arguments": {"path": "/a/b.txt"}.'
            ) from None
    if not isinstance(value, dict):
        raise ProtocolError(
            f"the `arguments` for {tool!r} must be a JSON object of parameter names, not "
            f"{_json_shape(value)}."
        )
    return value


def _as_text(value: Any) -> str:
    """Render a contract field as text without letting a structured value through silently."""
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str, ensure_ascii=False)


def _json_shape(value: Any) -> str:
    """Name a JSON value's shape for a message — its type, never its content."""
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, str):
        return "a string"
    if isinstance(value, (int, float)):
        return "a number"
    if isinstance(value, list):
        return "an array"
    if value is None:
        return "null"
    return "an unsupported value"


__all__ = [
    "CONTRACT",
    "FinalAnswer",
    "ModelAction",
    "ProtocolError",
    "ToolCall",
    "parse_action",
    "render_system_prompt",
]
