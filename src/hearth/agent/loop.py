"""The agent loop — plan, call one tool, observe, repeat, and stop for a stated reason.

This is a **bounded** loop, not an autonomous one, and the difference is the whole design.
A local 3B-14B model is good at one well-specified step at a time and bad at long-horizon
planning with recovery: it does not reliably notice that its plan stopped working, and its
characteristic failure is not an error but a fluent answer built on a step it never took.
Every choice here follows from that:

  * **one tool per turn**, so every claim in the transcript has a result behind it;
  * **hard bounds** on iterations, wall clock and tokens, checked before each model call;
  * **a stop reason that is part of the result type**, so "finished" and "ran out of budget"
    cannot be confused — :class:`AgentRun` refuses to hold an answer unless it was answered
    (``__post_init__``), and :meth:`AgentRun.require_answer` raises rather than returning a
    partial result as a whole one. That is the ``finish_reason: "stop"`` lesson from
    ``gateway/app.py`` (``CLAUDE.md`` §3) applied to a whole run instead of one generation;
  * **a run of invalid outputs terminates the agent.** A model that cannot follow the output
    contract will not start following it on turn nine; looping until the iteration cap burns
    the operator's budget to produce nothing;
  * **no answer is ever synthesised by the loop.** If the model never answers, there is no
    answer — the run reports why it stopped and the caller sees the steps.

**Local by outcome, not by configuration.** Every generation goes through the router with
``allow_escalation=False``, and then the loop *checks what actually happened*: if the executed
route did not report a local backend, the run stops with :data:`STOPPED_EGRESS_REFUSED` and
the offending step is in the transcript. Asserting on the flag we passed in would be checking
our own request rather than the outcome — the exact bug class ``CLAUDE.md`` §3 catalogues.

**Tools are vetted by where their code lives.** By default an agent will only run tools whose
callable is defined inside ``src/hearth/agent/`` — the directory ``tests/test_agent_no_network.py``
proves, by AST over its own source, cannot reach the network or a shell. A caller who needs a
tool of their own passes ``vetted_only=False``, which is one greppable word and a documented
transfer of responsibility. See ``docs/AGENT.md`` for why this gate is by *code location*
rather than by tool name.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..providers.base import FINISH_LENGTH, GenRequest, Message
from ..router.route import ProviderError, Router
from .protocol import FinalAnswer, ProtocolError, ToolCall, parse_action, render_system_prompt
from .tools import (
    Tool,
    ToolOutcome,
    ToolRegistry,
    ToolValidationError,
    UnknownToolError,
    render_observation,
)

logger = logging.getLogger("hearth.agent")

#: The run reached an answer. **The only successful terminal state.**
STOPPED_ANSWERED = "answered"
#: The iteration cap was reached before the model answered.
STOPPED_MAX_ITERATIONS = "max_iterations"
#: The wall-clock cap was reached before the model answered.
STOPPED_TIMEOUT = "timeout"
#: The token cap was reached before the model answered.
STOPPED_TOKEN_BUDGET = "token_budget"
#: Too many consecutive turns the loop could not turn into a valid, dispatchable step.
STOPPED_INVALID_OUTPUT = "invalid_output"
#: The provider failed (missing weights, backend crash). Terminal, not recoverable here.
STOPPED_PROVIDER_ERROR = "provider_error"
#: A generation was not served locally. Terminal, and a bug worth waking someone for.
STOPPED_EGRESS_REFUSED = "egress_refused"

StopReason = Literal[
    "answered",
    "max_iterations",
    "timeout",
    "token_budget",
    "invalid_output",
    "provider_error",
    "egress_refused",
]

STOP_REASONS: tuple[StopReason, ...] = (
    STOPPED_ANSWERED,
    STOPPED_MAX_ITERATIONS,
    STOPPED_TIMEOUT,
    STOPPED_TOKEN_BUDGET,
    STOPPED_INVALID_OUTPUT,
    STOPPED_PROVIDER_ERROR,
    STOPPED_EGRESS_REFUSED,
)

#: What one turn resolved to, for the transcript.
StepKind = Literal["tool_call", "answer", "invalid"]

# Where vetted tool code must live. Resolved once, from this file's own location, so it
# cannot be pointed elsewhere by configuration.
_PACKAGE_DIR = Path(__file__).resolve().parent


class AgentError(RuntimeError):
    """Base class for agent-construction and result-misuse failures."""


class AgentConfigError(AgentError):
    """The agent was built in a way that cannot produce a trustworthy run. Raised eagerly."""


class UnvettedToolError(AgentConfigError):
    """A registered tool's code does not live inside ``hearth.agent``."""


class AgentIncompleteError(AgentError):
    """The run did not answer, and something asked for its answer anyway.

    Exists so a caller cannot write ``run.answer`` into a report without having decided what
    to do when there isn't one. An unanswered run has steps, not conclusions.
    """


class EgressRefusedError(AgentError):
    """A generation was served by something other than the local backend."""


@dataclass(frozen=True)
class Budget:
    """The hard bounds on one run. Every one of them ends the run with its own stop reason.

    Defaults are sized for a local 3B-14B doing bounded work: enough steps to read a couple of
    files and answer, not enough to wander. They are deliberately small — an agent that needs
    thirty steps is a task that needs decomposing, and the honest failure is a run that stops
    and says so.

    ``max_total_tokens`` counts prompt *and* completion tokens across every step, including
    the prompt re-sent each turn. That over-counts against a naive "tokens generated" reading
    and is meant to: re-sending a growing transcript is where an agent's cost actually goes,
    and a budget that ignores it does not bound anything.
    """

    max_iterations: int = 8
    max_seconds: float = 180.0
    max_total_tokens: int = 24_000
    #: ``max_tokens`` for each individual generation. A tool call is short; a cap this low
    #: also makes a rambling model hit ``finish_reason == "length"`` early, which the loop
    #: reports as a truncated turn rather than trying to parse a severed JSON object.
    max_step_tokens: int = 768
    #: Consecutive turns that produced no dispatchable step before the run is abandoned.
    max_consecutive_invalid: int = 3
    #: Characters of a tool result shown to the model. Truncation is marked, never silent.
    max_observation_chars: int = 4_000

    def __post_init__(self) -> None:
        for name in ("max_iterations", "max_total_tokens", "max_step_tokens",
                     "max_consecutive_invalid"):
            if getattr(self, name) < 1:
                raise AgentConfigError(f"{name} must be at least 1")
        if self.max_seconds <= 0:
            raise AgentConfigError("max_seconds must be positive")
        if self.max_observation_chars < 0:
            raise AgentConfigError("max_observation_chars must not be negative")


@dataclass(frozen=True)
class Step:
    """One turn, recorded whole: what the model said, what ran, and what came back.

    Every field is here so a run can be *reconstructed* rather than summarised. The principle
    is :meth:`hearth.finance.store.FinanceStore.explain`'s: a conclusion that cannot be walked
    back to the steps that produced it is a claim, not a result.
    """

    index: int
    kind: StepKind
    model_output: str
    thought: str = ""
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    observation: str | None = None
    error: str | None = None
    truncated_output: bool = False
    model_seconds: float = 0.0
    tool_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    backend: str = ""

    @property
    def ok(self) -> bool:
        """True when this turn produced a dispatchable step (a tool result, or an answer)."""
        return self.error is None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class AgentRun:
    """The result of one run: the stop reason, the answer if there is one, and every step.

    ``answer`` is ``None`` for every stop reason except :data:`STOPPED_ANSWERED`, and
    ``__post_init__`` enforces that both ways — an answered run must carry text and an
    unanswered one must not. The impossible state simply cannot be constructed, so no caller
    has to remember to check a flag before reading a field.
    """

    task: str
    stopped_reason: StopReason
    answer: str | None
    steps: tuple[Step, ...]
    budget: Budget
    detail: str = ""
    elapsed_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __post_init__(self) -> None:
        if self.stopped_reason not in STOP_REASONS:
            raise AgentConfigError(f"unknown stopped_reason {self.stopped_reason!r}")
        answered = self.stopped_reason == STOPPED_ANSWERED
        if answered and not (self.answer or "").strip():
            raise AgentConfigError(
                "a run that stopped as 'answered' must carry the answer text"
            )
        if not answered and self.answer is not None:
            raise AgentConfigError(
                f"a run that stopped as {self.stopped_reason!r} must not carry an answer; "
                "a partial result presented as a whole one is the failure this type exists "
                "to make unrepresentable"
            )

    @property
    def completed(self) -> bool:
        """True only when the model produced an answer within budget."""
        return self.stopped_reason == STOPPED_ANSWERED

    @property
    def iterations(self) -> int:
        """How many model turns the run took."""
        return len(self.steps)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def tool_calls(self) -> tuple[Step, ...]:
        """The steps that actually dispatched a tool, in order."""
        return tuple(s for s in self.steps if s.kind == "tool_call" and s.ok)

    def require_answer(self) -> str:
        """Return the answer, or raise :class:`AgentIncompleteError` explaining the stop.

        Use this wherever the answer is about to be *used* — written into a document, handed
        to another system, shown as a conclusion. It is the difference between a caller that
        handles an exhausted budget and one that reports whatever was lying around.
        """
        if self.answer is None:
            raise AgentIncompleteError(
                f"the agent did not answer: it stopped because {self.stopped_reason!r}"
                + (f" ({self.detail})" if self.detail else "")
                + f". {self.iterations} step(s) ran; call transcript() to see them."
            )
        return self.answer

    def transcript(self) -> str:
        """Render the whole run for a person to check, step by step.

        Deliberately shaped like :meth:`hearth.finance.store.FinanceStore.explain`: the
        headline is at the top *and* the evidence is below it, so the reader can see whether
        the conclusion is supported rather than being told that it is.
        """
        lines = [
            f"task           : {self.task.strip()}",
            f"stopped_reason : {self.stopped_reason}",
            f"answer         : {self.answer if self.answer is not None else '(none)'}",
            f"steps          : {self.iterations} of at most {self.budget.max_iterations}",
            f"tokens         : {self.total_tokens} of at most {self.budget.max_total_tokens}"
            f" (prompt {self.prompt_tokens}, completion {self.completion_tokens})",
            f"elapsed        : {self.elapsed_seconds:.2f}s of at most "
            f"{self.budget.max_seconds:.0f}s",
        ]
        if self.detail:
            lines.append(f"detail         : {self.detail}")
        for step in self.steps:
            lines.append("")
            head = f"  step {step.index} [{step.kind}]"
            if step.tool:
                head += f" {step.tool}({_render_args(step.arguments)})"
            lines.append(head)
            lines.append(
                f"    model      : {step.model or '?'} via {step.backend or '?'}  "
                f"({step.prompt_tokens}+{step.completion_tokens} tok, "
                f"{step.model_seconds:.2f}s model, {step.tool_seconds:.2f}s tool)"
            )
            if step.truncated_output:
                lines.append("    TRUNCATED  : the model's own output hit max_step_tokens")
            if step.thought:
                lines.append(f"    thought    : {step.thought}")
            lines.append(f"    output     : {_indent(step.model_output)}")
            if step.error:
                lines.append(f"    ERROR      : {_indent(step.error)}")
            if step.observation is not None:
                lines.append(f"    observation: {_indent(step.observation)}")
        if not self.completed:
            lines.append("")
            lines.append(
                "  NOT AN ANSWER — this run stopped before the model concluded. Anything "
                "above is a partial trace, not a result."
            )
        return "\n".join(lines)


class Agent:
    """A bounded, on-device, tool-using loop over one :class:`~hearth.router.Router`.

    The router (not a bare provider) is the seam, so an agent inherits HEARTH's class
    routing, adapter selection, telemetry and budget accounting for free — and runs on the
    echo backend in tests with no model downloaded.

    ``tools`` is required and must be non-empty. An agent with no tools is a chat completion
    wearing a loop: it can only produce assertions about data it never read, which is the
    failure mode this package exists to avoid. Call ``router.route`` directly for that.
    """

    def __init__(
        self,
        router: Router,
        tools: Iterable[Tool] | ToolRegistry,
        *,
        budget: Budget | None = None,
        model: str = "auto",
        intent: str = "reason",
        temperature: float = 0.0,
        guidance: str = "",
        vetted_only: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.router = router
        self.registry = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools)
        if len(self.registry) == 0:
            raise AgentConfigError(
                "an agent needs at least one tool; without one it can only assert, never "
                "check. Use router.route() for a plain completion."
            )
        if vetted_only:
            _require_vetted(self.registry.tools())
        self.budget = budget or Budget()
        self.model = model
        self.intent = intent
        # Temperature 0 by default: the contract asks for one exact JSON shape, and sampling
        # noise buys nothing but parse failures. Same reason the eval gate pins it.
        self.temperature = temperature
        self.guidance = guidance
        self._clock = clock

    # -- the loop ---------------------------------------------------------------------

    def run(self, task: str) -> AgentRun:
        """Run ``task`` to an answer or to a bound, and return the whole trace either way."""
        if not task.strip():
            raise AgentConfigError("an agent run needs a task")

        started = self._clock()
        messages: list[Message] = [
            Message(
                role="system",
                content=render_system_prompt(
                    self.registry, task=task, guidance=self.guidance
                ),
            ),
            Message(role="user", content=_FIRST_TURN),
        ]
        steps: list[Step] = []
        prompt_tokens = completion_tokens = 0
        consecutive_invalid = 0

        def finish(reason: StopReason, detail: str, answer: str | None = None) -> AgentRun:
            return AgentRun(
                task=task,
                stopped_reason=reason,
                answer=answer,
                steps=tuple(steps),
                budget=self.budget,
                detail=detail,
                elapsed_seconds=self._clock() - started,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        while True:
            # Bounds are checked *before* starting a turn, so the run never begins a step it
            # cannot pay for and the reported totals are the ones actually spent.
            if len(steps) >= self.budget.max_iterations:
                return finish(
                    STOPPED_MAX_ITERATIONS,
                    f"reached the {self.budget.max_iterations}-step limit without an answer",
                )
            elapsed = self._clock() - started
            if elapsed >= self.budget.max_seconds:
                return finish(
                    STOPPED_TIMEOUT,
                    f"ran for {elapsed:.1f}s, past the {self.budget.max_seconds:.0f}s limit, "
                    "without an answer",
                )
            spent = prompt_tokens + completion_tokens
            if spent >= self.budget.max_total_tokens:
                return finish(
                    STOPPED_TOKEN_BUDGET,
                    f"spent {spent} tokens, past the {self.budget.max_total_tokens}-token "
                    "limit, without an answer",
                )

            index = len(steps) + 1
            try:
                turn = self._generate(messages)
            except ProviderError as exc:
                steps.append(
                    Step(index=index, kind="invalid", model_output="", error=str(exc))
                )
                return finish(STOPPED_PROVIDER_ERROR, str(exc))
            except EgressRefusedError as exc:
                steps.append(
                    Step(index=index, kind="invalid", model_output="", error=str(exc))
                )
                logger.error("agent run aborted: %s", exc)
                return finish(STOPPED_EGRESS_REFUSED, str(exc))

            prompt_tokens += turn.prompt_tokens
            completion_tokens += turn.completion_tokens

            step, observation, answer = self._resolve(index, turn)
            steps.append(step)

            if answer is not None:
                return finish(STOPPED_ANSWERED, "the model answered", answer=answer)

            consecutive_invalid = 0 if step.ok else consecutive_invalid + 1
            messages.append(Message(role="assistant", content=turn.text))
            messages.append(Message(role="user", content=observation))

            if consecutive_invalid >= self.budget.max_consecutive_invalid:
                return finish(
                    STOPPED_INVALID_OUTPUT,
                    f"{consecutive_invalid} consecutive turns produced nothing runnable; the "
                    "model is not following the output contract",
                )

    # -- one turn ---------------------------------------------------------------------

    def _generate(self, messages: Sequence[Message]) -> _Turn:
        """Run one hard-local generation and *verify* it was served locally.

        ``allow_escalation=False`` states the intent; the check afterwards is what makes it
        true. A router misconfiguration, a policy edit, or a future backend that ignores the
        flag all surface here as a refusal rather than as a quietly remote agent step.
        """
        started = self._clock()
        routed = self.router.route(
            GenRequest(
                messages=list(messages),
                model=self.model,
                max_tokens=self.budget.max_step_tokens,
                temperature=self.temperature,
            ),
            intent=self.intent,
            allow_escalation=False,
        )
        seconds = self._clock() - started

        if routed.decision.backend != "local" or routed.record.served_by != "local":
            raise EgressRefusedError(
                "an agent step was served by "
                f"{routed.record.served_by!r}/{routed.decision.backend!r} rather than the "
                "local backend, despite allow_escalation=False. Refusing to continue: "
                "HEARTH agent runs are on-device only (docs/TIERS.md)."
            )

        return _Turn(
            text=routed.result.text or "",
            model=routed.result.model,
            backend=routed.result.backend,
            prompt_tokens=routed.result.prompt_tokens,
            completion_tokens=routed.result.completion_tokens,
            truncated=routed.result.finish_reason == FINISH_LENGTH,
            seconds=seconds,
        )

    def _resolve(self, index: int, turn: _Turn) -> tuple[Step, str, str | None]:
        """Turn one model output into a recorded :class:`Step` plus the next observation.

        Returns ``(step, observation, answer)``. ``answer`` is non-``None`` exactly when the
        run should stop successfully; ``observation`` is what the model is told next.
        """
        base = dict(
            index=index,
            model_output=turn.text,
            model_seconds=turn.seconds,
            prompt_tokens=turn.prompt_tokens,
            completion_tokens=turn.completion_tokens,
            model=turn.model,
            backend=turn.backend,
            truncated_output=turn.truncated,
        )

        # A truncated generation is reported as such and never parsed. Half a JSON object can
        # decode into a *valid-looking* call with the arguments cut off, which is a wrong
        # action taken with full confidence — the failure mode FinishReason exists to expose.
        if turn.truncated:
            error = (
                f"your reply was cut off at the {self.budget.max_step_tokens}-token limit, "
                "so it could not be read. Reply with one short JSON object only; put nothing "
                "before or after it and keep `thought` to a single sentence."
            )
            return Step(kind="invalid", error=error, **base), _observation_error(error), None

        try:
            action = parse_action(turn.text)
        except ProtocolError as exc:
            return (
                Step(kind="invalid", error=str(exc), **base),
                _observation_error(str(exc)),
                None,
            )

        if isinstance(action, FinalAnswer):
            step = Step(kind="answer", thought=action.thought, **base)
            return step, "", action.text

        outcome = self._dispatch(action)
        step = Step(
            kind="tool_call" if outcome.ok else "invalid",
            thought=action.thought,
            tool=action.name,
            arguments=dict(outcome.arguments),
            observation=(
                render_observation(outcome.value, self.budget.max_observation_chars)
                if outcome.ok
                else None
            ),
            error=outcome.error,
            tool_seconds=outcome.seconds,
            **base,
        )
        if outcome.ok:
            return step, _observation_result(action.name, step.observation or ""), None
        return step, _observation_error(outcome.error or "the tool failed"), None

    def _dispatch(self, call: ToolCall) -> ToolOutcome:
        """Look the tool up, validate the arguments, and run it. Never raises.

        The three failure modes — unknown tool, arguments that don't match the schema, and the
        tool itself refusing — are all recoverable turns, so they come back as a
        :class:`ToolOutcome` carrying the message rather than as an exception. Validation
        happens *before* dispatch, so a tool's callable only ever sees arguments matching its
        declared schema.
        """
        try:
            tool = self.registry.get(call.name)
        except UnknownToolError as exc:
            return ToolOutcome(tool=call.name, error=str(exc))

        try:
            kwargs = tool.validate(call.arguments)
        except ToolValidationError as exc:
            return ToolOutcome(tool=call.name, error=str(exc))

        started = self._clock()
        try:
            value = tool.call(**kwargs)
        except Exception as exc:  # noqa: BLE001 — a refusing tool is an observation
            logger.debug("tool %s raised: %s", call.name, exc)
            return ToolOutcome(
                tool=call.name,
                arguments=kwargs,
                error=f"{call.name} failed: {exc}",
                seconds=self._clock() - started,
            )
        return ToolOutcome(
            tool=call.name,
            arguments=kwargs,
            value=value,
            seconds=self._clock() - started,
        )


@dataclass(frozen=True)
class _Turn:
    """One completed generation, normalised for the loop."""

    text: str
    model: str = ""
    backend: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    truncated: bool = False
    seconds: float = 0.0


_FIRST_TURN = (
    "Begin. Reply with exactly one JSON object: either a tool call, or an answer if you "
    "already have everything you need."
)


def _observation_result(tool: str, text: str) -> str:
    return f"Observation from {tool}:\n{text}\n\nReply with exactly one JSON object."


def _observation_error(message: str) -> str:
    return (
        f"That step did not run. {message}\n\n"
        "Reply with exactly one JSON object: fix the call, or answer with what you do know "
        "and say what you could not determine."
    )


def _require_vetted(tools: Sequence[Tool]) -> None:
    """Refuse any tool whose callable is not defined inside ``src/hearth/agent/``.

    The check is on the **code**, not on the name: ``tool.name`` is a label a caller chooses
    freely, so an allowlist of names would compare the check against the wrong object
    (``CLAUDE.md`` §3) — anyone could register ``Tool(name="read_file", call=fetch)`` and pass
    it. A callable's ``__code__.co_filename`` says where its body was compiled from, and the
    bodies in this package are held network-free by ``tests/test_agent_no_network.py``, which
    walks their source. A C function or builtin has no ``__code__`` and is refused for that
    reason alone: there is no source for the invariant to be proven over.

    This bounds the *tool code*. It does not bound an object a built-in tool was constructed
    with — a fake RAG index that phoned home would still phone home. That is the caller's
    responsibility and is stated as such in ``docs/AGENT.md``.
    """
    for tool in tools:
        code = getattr(tool.call, "__code__", None) or getattr(
            getattr(tool.call, "__func__", None), "__code__", None
        )
        origin = Path(code.co_filename).resolve() if code is not None else None
        if origin is None or not origin.is_relative_to(_PACKAGE_DIR):
            where = str(origin) if origin is not None else "a non-Python callable"
            raise UnvettedToolError(
                f"tool {tool.name!r} is implemented in {where}, outside {_PACKAGE_DIR}. "
                "By default an agent runs only tools whose code lives in hearth.agent, "
                "because that is the source the no-network AST test covers. Pass "
                "vetted_only=False to take responsibility for a tool of your own."
            )


def _render_args(arguments: dict[str, Any] | None) -> str:
    """Render a call's arguments for the transcript header line."""
    if not arguments:
        return ""
    return ", ".join(f"{k}={v!r}" for k, v in arguments.items())


def _indent(text: str, width: int = 17) -> str:
    """Indent continuation lines so a multi-line field stays inside its column."""
    pad = " " * width
    return text.replace("\n", "\n" + pad)


__all__ = [
    "STOPPED_ANSWERED",
    "STOPPED_EGRESS_REFUSED",
    "STOPPED_INVALID_OUTPUT",
    "STOPPED_MAX_ITERATIONS",
    "STOPPED_PROVIDER_ERROR",
    "STOPPED_TIMEOUT",
    "STOPPED_TOKEN_BUDGET",
    "STOP_REASONS",
    "Agent",
    "AgentConfigError",
    "AgentError",
    "AgentIncompleteError",
    "AgentRun",
    "Budget",
    "EgressRefusedError",
    "Step",
    "StepKind",
    "StopReason",
    "UnvettedToolError",
]
