"""``POST /v1/hearth/agent`` — the bounded local agent loop, streamed step by step.

Why this route exists: ``/v1/chat/completions`` has no tools, so asking the chat UI to read a
directory of statements gets a fluent, specific, entirely invented answer — the exact failure
``hearth.agent`` was built to prevent. This route is the seam that lets the operator ask the
same question and get a run whose every claim has a tool result behind it.

``docs/AGENT.md`` §6 used to end with *"exposing this over HTTP means deciding who may pass
``vetted_only=False`` and what a hostile task string can reach through the tools."* This module
is that decision, and it is made by **removing the choice from the wire** (§9):

* ``vetted_only=True`` is a literal in the ``Agent(...)`` call. There is no request field,
  header or query parameter that reaches it — :class:`~hearth.gateway.schemas.AgentRunRequest`
  forbids extra keys, so an attempt to send one is a 422 rather than an ignored field.
* The toolset is exactly :func:`hearth.agent.local_toolset` over the app's own collaborators:
  the read-only built-ins, no shell, no writes, no network tool.
* File reads keep going through ``HEARTH_FILE_ROOTS``. This route grants no filesystem
  privilege of its own — it cannot, because it never passes a path or a root anywhere.
* Budgets are clamped to a server ceiling, so a client asking for ten thousand iterations gets
  :data:`MAX_ITERATIONS_CEILING` and is told which bound was reduced.

**Streamed, because a blocking request would look hung.** A local model at ~12 tok/s takes
minutes over eight steps. The stream reuses the gateway's existing SSE conventions (``data:``
lines, a terminal ``[DONE]`` sentinel, an ``error`` envelope for a failure mid-stream) and
emits one event per step as the loop records it, then one terminal event.

**The stop reason survives the wire.** :class:`~hearth.agent.AgentRun` cannot hold an answer
unless it stopped as ``answered``; :class:`~hearth.gateway.schemas.AgentRunEvent` re-asserts
that on the way out. An invariant dropped at the serialisation boundary is not an invariant,
and that boundary is precisely where this repository lost ``finish_reason`` once already
(``CLAUDE.md`` §3).
"""

from __future__ import annotations

import json
import logging
import queue
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Final

from fastapi import Depends, FastAPI, Request
from fastapi.responses import StreamingResponse

from ..agent import Agent, AgentRun, Budget, Step, local_toolset
from ..mcp.files import allowed_roots
from .auth import require_token
from .schemas import (
    AgentBudgetApplied,
    AgentBudgetRequest,
    AgentRunEvent,
    AgentRunRequest,
    AgentStartEvent,
    AgentStepEvent,
)

logger = logging.getLogger("hearth.gateway.agent")

#: Server-side ceilings. A client may ask for less; asking for more gets these. They are
#: module constants rather than settings on purpose: a budget cap that can be raised by
#: configuration is a cap the operator has to audit their configuration to trust.
MAX_ITERATIONS_CEILING: Final[int] = 12
MAX_SECONDS_CEILING: Final[float] = 600.0
MAX_TOTAL_TOKENS_CEILING: Final[int] = 48_000

#: How much of a step's observation goes on the wire. The loop has already capped it at
#: ``Budget.max_observation_chars`` for the *model*; this second, smaller cap is for the
#: *reader*, so one 4 000-character CSV dump does not bury the seven steps around it.
OBSERVATION_WIRE_CHARS: Final[int] = 1_200

_NO_ROOTS_WARNING: Final[str] = (
    "HEARTH_FILE_ROOTS names no readable directory, so read_file and list_files will refuse "
    "every path this run tries. Set it (e.g. HEARTH_FILE_ROOTS=/Users/you/statements) and "
    "restart the gateway if this task needs files."
)


def register_agent_route(app: FastAPI) -> None:
    """Mount ``POST /v1/hearth/agent`` on ``app``, authenticated like every other ``/v1``.

    Collaborators are read from ``app.state`` per request rather than captured at mount time,
    so a deployment that attaches a :class:`~hearth.finance.store.FinanceStore` later (or a
    test that removes the RAG index) changes what the agent can reach without a second
    registration path deciding it.
    """

    @app.post("/v1/hearth/agent", dependencies=[Depends(require_token)])
    def run_agent(request: Request, req: AgentRunRequest) -> StreamingResponse:
        """Run one bounded agent task over the local toolset, streaming every step."""
        state = request.app.state
        settings = state.settings
        registry = local_toolset(
            settings=settings,
            rag=getattr(state, "rag", None),
            finance=getattr(state, "finance", None),
        )
        budget, applied = _clamp(req.budget)
        roots = allowed_roots(settings)
        warnings = [] if roots else [_NO_ROOTS_WARNING]

        # `names` is a property, not a method.
        tool_names = list(registry.names)
        start = AgentStartEvent(
            task=req.task,
            tools=tool_names,
            budget=applied,
            file_roots=len(roots),
            warnings=warnings,
        )
        return StreamingResponse(
            _stream_agent(
                agent_factory=lambda on_step: _StreamingAgent(
                    state.router,
                    registry,
                    on_step=on_step,
                    budget=budget,
                    model=req.model,
                    # Hardcoded. Not a default, not a setting, not a request field: the one
                    # place this could be relaxed is this literal, and it is greppable.
                    vetted_only=True,
                ),
                task=req.task,
                start=start,
                applied=applied,
                warnings=warnings,
                reachable=_is_reachable(tool_names, roots),
            ),
            media_type="text/event-stream",
        )


class _StreamingAgent(Agent):
    """An :class:`~hearth.agent.Agent` that hands each recorded step to a callback.

    The override is on ``_resolve`` because that is the one point where the loop's own
    :class:`~hearth.agent.Step` exists — fully resolved, with the validated arguments and the
    observation the tool actually returned — before it is appended to the transcript.

    Reconstructing steps instead (by parsing the model text a wrapped router saw, say) would
    mean the stream reports the gateway's reading of the run while the transcript reports the
    loop's. Two objects, one check: ``CLAUDE.md`` §3. Emitting the loop's own object keeps the
    stream and the ``AgentRun`` the same thing.
    """

    def __init__(self, *args: Any, on_step: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._on_step = on_step

    def _resolve(self, index: int, turn: Any) -> tuple[Step, str, str | None]:
        resolved = super()._resolve(index, turn)
        self._on_step(resolved[0])
        return resolved


def _clamp(requested: AgentBudgetRequest | None) -> tuple[Budget, AgentBudgetApplied]:
    """Reduce a client's requested bounds to the server ceilings, naming what was reduced.

    Clamping rather than refusing: a caller who asks for more than the machine will give is
    not making an invalid request, they are making an optimistic one, and the honest answer is
    the smaller run plus a note saying so. Refusing would push callers toward guessing the cap.
    """
    default = Budget()
    clamped: list[str] = []

    def cap(name: str, asked: float | None, fallback: float, ceiling: float) -> float:
        if asked is None:
            return fallback
        if asked > ceiling:
            clamped.append(name)
            return ceiling
        return asked

    iterations = int(
        cap(
            "max_iterations",
            requested.max_iterations if requested else None,
            default.max_iterations,
            MAX_ITERATIONS_CEILING,
        )
    )
    seconds = float(
        cap(
            "max_seconds",
            requested.max_seconds if requested else None,
            default.max_seconds,
            MAX_SECONDS_CEILING,
        )
    )
    tokens = int(
        cap(
            "max_total_tokens",
            requested.max_total_tokens if requested else None,
            default.max_total_tokens,
            MAX_TOTAL_TOKENS_CEILING,
        )
    )
    budget = Budget(
        max_iterations=iterations, max_seconds=seconds, max_total_tokens=tokens
    )
    return budget, AgentBudgetApplied(
        max_iterations=iterations,
        max_seconds=seconds,
        max_total_tokens=tokens,
        clamped=clamped,
    )


def _is_reachable(tool_names: list[str], roots: list[Any]) -> bool:
    """Whether this run could observe anything at all.

    With no file roots and no RAG or finance collaborator, the toolset is two tools that both
    refuse every call. Such a run can only produce an assertion, which is the one output this
    package exists to prevent — so it is refused before the first generation rather than after
    eight of them.
    """
    if roots:
        return True
    return bool(set(tool_names) - {"read_file", "list_files"})


#: Every agent run executes on ONE long-lived worker thread, not a fresh thread per request.
#:
#: MLX's GPU stream is thread-local: a model loaded on one thread cannot be generated from
#: another, and a run on a fresh thread dies with "There is no Stream(gpu, 0) in current
#: thread". Measured, not theorised — against a live 14B, request 1 answered and requests 2
#: and 3 both failed with provider_error. A thread per request makes this route work exactly
#: once per server process, which is the kind of bug that looks like a flaky model.
#:
#: Reusing one worker also serialises runs, which is what we want regardless: two concurrent
#: 14B loops would contend for the same ~9 GB of unified memory and neither would finish
#: sooner. A queued run waits for the one ahead of it, itself bounded by its own max_seconds.
_RUNNER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hearth-agent")


def _stream_agent(
    *,
    agent_factory: Any,
    task: str,
    start: AgentStartEvent,
    applied: AgentBudgetApplied,
    warnings: list[str],
    reachable: bool,
) -> Iterator[str]:
    """Yield the start event, one event per step as it happens, a terminal event, ``[DONE]``.

    The loop is synchronous and a generator cannot be resumed from a callback, so the run goes
    on a worker thread and steps arrive over a queue. The thread is a daemon and the run is
    bounded by ``max_seconds``, so a client that disconnects mid-run cannot leave work behind
    that outlives the process.
    """
    yield _sse(start)

    if not reachable:
        yield _sse(
            {
                "error": {
                    "message": (
                        "this agent can reach nothing: HEARTH_FILE_ROOTS is unset and no RAG "
                        "index or finance store is attached, so every tool would refuse. "
                        "Refusing before spending the budget — an agent with no observations "
                        "can only assert."
                    ),
                    "type": "agent_no_tools_reachable",
                    "code": "hearth.agent.no_tools_reachable",
                }
            }
        )
        yield _sse("[DONE]")
        return

    events: queue.Queue[Any] = queue.Queue()
    _DONE = object()
    result: dict[str, Any] = {}

    def work() -> None:
        try:
            result["run"] = agent_factory(events.put).run(task)
        except Exception as exc:  # noqa: BLE001 — reported to the client, not swallowed
            logger.exception("agent run failed")
            result["error"] = exc
        finally:
            events.put(_DONE)

    _RUNNER.submit(work)

    streamed: set[int] = set()
    while True:
        item = events.get()
        if item is _DONE:
            break
        streamed.add(item.index)
        yield _sse(_step_event(item))

    error = result.get("error")
    if error is not None:
        yield _sse(
            {
                "error": {
                    "message": str(error),
                    "type": "agent_error",
                    "code": "hearth.agent.failed",
                }
            }
        )
        yield _sse("[DONE]")
        return

    run: AgentRun = result["run"]
    # A step the loop built outside `_resolve` — a provider error, a refused escalation — is
    # only in the finished run. Flushing it here keeps the stream a complete transcript
    # rather than one that silently omits the step that ended the run.
    for step in run.steps:
        if step.index not in streamed:
            yield _sse(_step_event(step))

    yield _sse(
        AgentRunEvent(
            stopped_reason=run.stopped_reason,
            completed=run.completed,
            answer=run.answer,
            detail=run.detail,
            steps=run.iterations,
            elapsed_seconds=run.elapsed_seconds,
            prompt_tokens=run.prompt_tokens,
            completion_tokens=run.completion_tokens,
            total_tokens=run.total_tokens,
            budget=applied,
            warnings=warnings,
        )
    )
    yield _sse("[DONE]")


def _step_event(step: Step) -> AgentStepEvent:
    """Project one recorded :class:`~hearth.agent.Step` onto the wire."""
    observation, truncated = _clip(step.observation)
    return AgentStepEvent(
        index=step.index,
        kind=step.kind,
        thought=step.thought,
        tool=step.tool,
        arguments=step.arguments,
        observation=observation,
        observation_truncated=truncated,
        error=step.error,
        model=step.model,
        backend=step.backend,
        model_seconds=step.model_seconds,
        tool_seconds=step.tool_seconds,
        prompt_tokens=step.prompt_tokens,
        completion_tokens=step.completion_tokens,
    )


def _clip(text: str | None) -> tuple[str | None, bool]:
    """Shorten an observation for display, **marking** the cut in the text itself.

    Same rule as :func:`hearth.agent.render_observation`: a silently shortened observation is
    a reader who thinks they saw the evidence and did not.
    """
    if text is None or len(text) <= OBSERVATION_WIRE_CHARS:
        return text, False
    return (
        f"{text[:OBSERVATION_WIRE_CHARS]}\n[... truncated for display: showing the first "
        f"{OBSERVATION_WIRE_CHARS} of {len(text)} characters.]",
        True,
    )


def _sse(payload: object) -> str:
    """Serialize one payload as an SSE ``data:`` event (the convention in ``app.py``)."""
    if isinstance(payload, str):
        return f"data: {payload}\n\n"
    if isinstance(payload, dict):
        return f"data: {json.dumps(payload)}\n\n"
    return f"data: {payload.model_dump_json(exclude_none=True)}\n\n"  # type: ignore[attr-defined]


__all__ = [
    "MAX_ITERATIONS_CEILING",
    "MAX_SECONDS_CEILING",
    "MAX_TOTAL_TOKENS_CEILING",
    "OBSERVATION_WIRE_CHARS",
    "register_agent_route",
]
