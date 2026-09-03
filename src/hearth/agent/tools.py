"""The tool protocol — what an agent is allowed to reach, stated explicitly and typed.

A tool is the only way the loop touches anything outside the model. That makes this module
the agent's entire attack surface, so three properties are built in rather than bolted on:

**Tools are injected, never discovered.** There is no registry populated at import time, no
entry-point scan, no plugin hook. A :class:`ToolRegistry` starts empty and a caller adds
exactly the tools it intends the agent to have. "What can this agent do?" is answered by
reading one call site, not by auditing an import graph.

**Arguments are validated before dispatch, not inside the callable.** A local 3B model emits
malformed calls routinely — a missing field, an invented parameter, a number as a string.
:meth:`Tool.validate` catches all of that and raises :class:`ToolValidationError`, which the
loop feeds back to the model as a *recoverable* observation. The callable therefore only ever
runs on arguments that matched its declared schema, which is what lets the built-in tools be
thin wrappers with no defensive parsing of their own.

**Validation refuses rather than guesses — with one narrow, deliberate exception.** JSON has
no way to tell ``5`` from ``"5"``, and a small model picks between them almost at random. So a
string that is an *exact* literal of the declared scalar type is coerced (``"5"`` -> ``5``,
``"true"`` -> ``True``) and nothing else is: ``"5.7"`` is not an integer, ``"yes"`` is not a
boolean, and ``"~/notes"`` is never turned into anything but itself. The line is drawn at
round-tripping, so a coercion can never change the value the model meant. Everything outside
it — an unknown parameter, a missing required one, a value outside ``choices`` — is a refusal
naming the parameter, because a refusal the model can act on costs one iteration and a wrong
guess costs the whole run.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

#: The scalar types a tool parameter may declare. Deliberately not "object" or "array": a
#: small local model nests JSON badly, and a tool that needs structure is better expressed as
#: several flat parameters than as one the model gets wrong every third turn.
ParamType = Literal["string", "integer", "number", "boolean"]

PARAM_TYPES: tuple[ParamType, ...] = ("string", "integer", "number", "boolean")

# A tool name has to survive being written by a model, quoted in JSON, and grepped for in a
# transcript. Restricting it to this shape means a near-miss ("Read_File", "read file") fails
# the *unknown tool* check with a clear message instead of silently matching something else.
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class ToolError(RuntimeError):
    """Base class for every failure originating in the tool layer."""


class ToolDefinitionError(ToolError):
    """A tool was declared wrongly. Raised at construction — a caller bug, never a model one."""


class ToolValidationError(ToolError):
    """Parsed arguments did not match a tool's schema.

    Recoverable by design: the loop turns this into an observation and lets the model try
    again, so the message is written for the *model* to act on — it names the parameter and
    what was expected, never a Python type or a traceback.
    """


class UnknownToolError(ToolError):
    """The model asked for a tool that is not registered. Also recoverable, also fed back."""


@dataclass(frozen=True)
class ToolParam:
    """One typed parameter of a tool.

    ``description`` is not decoration: it is the only thing telling a 3B model what to put
    here, and it is rendered into every prompt. Write it as an instruction, not a label.
    """

    name: str
    type: ParamType
    description: str
    required: bool = True
    default: Any = None
    #: When non-empty, the value must be one of these (compared as strings). An enumerated
    #: parameter is the cheapest way to stop a small model inventing a mode that doesn't exist.
    choices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _NAME_RE.match(self.name):
            raise ToolDefinitionError(
                f"parameter name {self.name!r} must be lowercase snake_case"
            )
        if self.type not in PARAM_TYPES:
            raise ToolDefinitionError(
                f"parameter {self.name!r} declares unsupported type {self.type!r}; "
                f"supported: {', '.join(PARAM_TYPES)}"
            )
        if self.required and self.default is not None:
            raise ToolDefinitionError(
                f"parameter {self.name!r} is required but also carries a default; a default "
                "on a required parameter is dead code that hides which one actually applies"
            )
        if not self.description.strip():
            raise ToolDefinitionError(
                f"parameter {self.name!r} has no description; the model has nothing else to "
                "go on and an undescribed parameter is filled in by guesswork"
            )

    def render(self) -> str:
        """Render this parameter for the tool block in the prompt (one line)."""
        bits = [self.type]
        bits.append("required" if self.required else f"optional, default {self.default!r}")
        if self.choices:
            bits.append("one of: " + ", ".join(self.choices))
        return f"    {self.name} ({'; '.join(bits)}) — {self.description}"


@dataclass(frozen=True)
class Tool:
    """A named, typed, callable capability handed to an agent.

    ``call`` is invoked with the *validated* arguments as keyword arguments and must return
    something the loop can render as text (:func:`render_observation` handles str, numbers,
    lists and dicts). It may raise: a raised exception is caught by the loop, recorded in the
    transcript, and fed back as an observation, because a tool refusing (a path outside
    ``HEARTH_FILE_ROOTS``, an empty result set) is information the model can act on.

    ``returns`` describes the shape of what comes back, in one line, for the same reason
    parameter descriptions exist — a model that does not know what a tool returns cannot plan
    the step after it.
    """

    name: str
    description: str
    call: Callable[..., Any]
    params: tuple[ToolParam, ...] = ()
    returns: str = ""

    def __post_init__(self) -> None:
        if not _NAME_RE.match(self.name):
            raise ToolDefinitionError(
                f"tool name {self.name!r} must be lowercase snake_case (matched against the "
                "model's output verbatim, so near-misses must fail rather than fuzzy-match)"
            )
        if not self.description.strip():
            raise ToolDefinitionError(f"tool {self.name!r} has no description")
        if not callable(self.call):
            raise ToolDefinitionError(f"tool {self.name!r} has a non-callable `call`")
        seen: set[str] = set()
        for param in self.params:
            if param.name in seen:
                raise ToolDefinitionError(
                    f"tool {self.name!r} declares parameter {param.name!r} twice"
                )
            seen.add(param.name)

    @property
    def signature(self) -> str:
        """``name(a: string, b: integer)`` — the one-line form used in the prompt header."""
        inner = ", ".join(
            f"{p.name}: {p.type}" if p.required else f"{p.name}?: {p.type}" for p in self.params
        )
        return f"{self.name}({inner})"

    def render(self) -> str:
        """Render the full tool description block the model reads every turn."""
        lines = [f"  {self.signature}", f"    {self.description.strip()}"]
        lines.extend(p.render() for p in self.params)
        if self.returns.strip():
            lines.append(f"    returns: {self.returns.strip()}")
        return "\n".join(lines)

    def validate(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Check ``arguments`` against this tool's schema and return the call kwargs.

        Runs **before** dispatch so ``call`` never sees an argument it did not declare. Every
        refusal raises :class:`ToolValidationError` with a message addressed to the model.
        """
        if not isinstance(arguments, Mapping):
            raise ToolValidationError(
                f"arguments for {self.name!r} must be a JSON object of parameter names, "
                f"not a {type(arguments).__name__}"
            )

        declared = {p.name: p for p in self.params}
        unknown = sorted(set(arguments) - set(declared))
        if unknown:
            known = ", ".join(declared) or "(none)"
            raise ToolValidationError(
                f"{self.name!r} has no parameter(s) {unknown}; its parameters are: {known}"
            )

        kwargs: dict[str, Any] = {}
        for param in self.params:
            if param.name not in arguments:
                if param.required:
                    raise ToolValidationError(
                        f"{self.name!r} requires the parameter {param.name!r} "
                        f"({param.description.strip()})"
                    )
                kwargs[param.name] = param.default
                continue
            kwargs[param.name] = _coerce(self.name, param, arguments[param.name])
        return kwargs


class ToolRegistry:
    """The set of tools one agent may use. Empty until a caller fills it.

    Kept as an object rather than a module-level dict precisely so two agents in one process
    cannot share a capability by accident — the finance agent's ledger access is not reachable
    from the notes agent because they hold different registries, not because a flag says so.
    """

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        """Add ``tool``. A duplicate name is refused rather than silently overwriting.

        Overwriting would mean the tool the prompt describes and the tool that runs are two
        different objects — the exact shape of bug ``CLAUDE.md`` §3 catalogues.
        """
        if not isinstance(tool, Tool):
            raise ToolDefinitionError(
                f"expected a hearth.agent.Tool, got {type(tool).__name__}; tools must be "
                "declared through the Tool dataclass so their schema is validated"
            )
        if tool.name in self._tools:
            raise ToolDefinitionError(
                f"a tool named {tool.name!r} is already registered; names are matched against "
                "model output verbatim, so two tools cannot share one"
            )
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Return the tool called ``name``, or raise :class:`UnknownToolError`."""
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(sorted(self._tools)) or "(none)"
            raise UnknownToolError(
                f"there is no tool named {name!r}. Available tools: {available}"
            )
        return tool

    @property
    def names(self) -> tuple[str, ...]:
        """Registered tool names, sorted — the stable order the prompt renders them in."""
        return tuple(sorted(self._tools))

    def tools(self) -> tuple[Tool, ...]:
        """Every registered tool, in :attr:`names` order."""
        return tuple(self._tools[n] for n in self.names)

    def render(self) -> str:
        """Render the whole tool block for the prompt."""
        return "\n".join(self._tools[n].render() for n in self.names)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools


@dataclass(frozen=True)
class ToolOutcome:
    """What one dispatch produced — the value, or the refusal, plus how long it took.

    Both outcomes are first class: a tool that refused is not an error in the run, it is an
    observation. Only the loop decides whether a *sequence* of refusals means the agent is
    stuck.
    """

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    value: Any = None
    error: str | None = None
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        """True when the tool returned rather than raised."""
        return self.error is None


def render_observation(value: Any, limit: int) -> str:
    """Render a tool's return value as the text the model sees, truncated **visibly**.

    Truncation is the point. A single ``read_file`` on a large document can consume an
    agent's entire token budget in one step, so observations are capped — but a silently
    shortened observation is a model reasoning over data it thinks it has and doesn't. The
    marker states the cut and the original size, so the model can narrow its next call
    instead of confidently summarising a fragment.
    """
    text = value if isinstance(value, str) else _to_text(value)
    if limit > 0 and len(text) > limit:
        return (
            f"{text[:limit]}\n[... truncated: showing the first {limit} of {len(text)} "
            "characters. Narrow the request if you need the rest.]"
        )
    return text


def _to_text(value: Any) -> str:
    """Render a non-string tool result as text a model can read."""
    if value is None:
        return "(no result)"
    if isinstance(value, (list, tuple)):
        if not value:
            return "(empty list)"
        return "\n".join(f"- {_scalar(item)}" for item in value)
    if isinstance(value, Mapping):
        if not value:
            return "(empty object)"
        return "\n".join(f"{k}: {_scalar(v)}" for k, v in value.items())
    return str(value)


def _scalar(value: Any) -> str:
    """Render one nested value on a single line, JSON for anything structured."""
    if isinstance(value, str):
        return value
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, default=str, sort_keys=True, ensure_ascii=False)
    return str(value)


def _coerce(tool: str, param: ToolParam, value: Any) -> Any:
    """Type-check one argument, allowing only round-trippable string->scalar coercion."""
    if param.type == "string":
        if not isinstance(value, str):
            raise ToolValidationError(
                f"{tool}.{param.name} must be a string, got {_shape(value)}"
            )
        coerced: Any = value
    elif param.type == "boolean":
        coerced = _as_bool(tool, param, value)
    elif param.type == "integer":
        coerced = _as_int(tool, param, value)
    else:  # "number"
        coerced = _as_float(tool, param, value)

    if param.choices and str(coerced) not in param.choices:
        raise ToolValidationError(
            f"{tool}.{param.name} must be one of {', '.join(param.choices)}; got {coerced!r}"
        )
    return coerced


def _as_bool(tool: str, param: ToolParam, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    raise ToolValidationError(
        f"{tool}.{param.name} must be true or false, got {_shape(value)}"
    )


def _as_int(tool: str, param: ToolParam, value: Any) -> int:
    # bool is an int in Python; accepting it here would turn `true` into 1 silently.
    if isinstance(value, bool):
        raise ToolValidationError(f"{tool}.{param.name} must be a whole number, got a boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            pass
    raise ToolValidationError(
        f"{tool}.{param.name} must be a whole number, got {_shape(value)}"
    )


def _as_float(tool: str, param: ToolParam, value: Any) -> float:
    if isinstance(value, bool):
        raise ToolValidationError(f"{tool}.{param.name} must be a number, got a boolean")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            pass
    raise ToolValidationError(f"{tool}.{param.name} must be a number, got {_shape(value)}")


def _shape(value: Any) -> str:
    """Name a value's JSON shape for an error message — never its content.

    Same rule as :mod:`hearth.mcp.files`: these messages travel back into a prompt (and into
    a stored transcript), so they carry the *type* of what arrived, not the value.
    """
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, str):
        return "a string"
    if isinstance(value, int):
        return "a whole number"
    if isinstance(value, float):
        return "a number"
    if isinstance(value, Mapping):
        return "an object"
    if isinstance(value, (list, tuple)):
        return "an array"
    if value is None:
        return "null"
    return "an unsupported value"


__all__ = [
    "PARAM_TYPES",
    "ParamType",
    "Tool",
    "ToolDefinitionError",
    "ToolError",
    "ToolOutcome",
    "ToolParam",
    "ToolRegistry",
    "ToolValidationError",
    "UnknownToolError",
    "render_observation",
]
