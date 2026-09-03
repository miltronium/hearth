"""The loop: a multi-step run, every recoverable failure, and every bound.

Everything here runs offline with no model. Two fakes do the work: a scripted provider that
emits a canned sequence of turns (so a "model" can be made to misbehave in exactly one way per
test) and a fake clock (so the wall-clock bound is deterministic rather than slept for). The
scripted provider is driven through the **real** :class:`~hearth.router.Router`, so the tests
exercise the same routing path a real run takes.

The load-bearing assertions are the ones about *stop reasons*: a run that ran out of budget
must be impossible to mistake for one that finished, and no bound may end a run quietly.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest

from hearth.agent import (
    STOPPED_ANSWERED,
    STOPPED_EGRESS_REFUSED,
    STOPPED_INVALID_OUTPUT,
    STOPPED_MAX_ITERATIONS,
    STOPPED_PROVIDER_ERROR,
    STOPPED_TIMEOUT,
    STOPPED_TOKEN_BUDGET,
    Agent,
    AgentConfigError,
    AgentIncompleteError,
    AgentRun,
    Budget,
    Tool,
    ToolParam,
    UnvettedToolError,
    local_toolset,
)
from hearth.observability.budget import BudgetAccountant
from hearth.observability.metrics import MetricsStore
from hearth.providers.base import (
    FINISH_LENGTH,
    FINISH_STOP,
    Capabilities,
    GenRequest,
    GenResult,
    ResourceEstimate,
)
from hearth.router import Router

# -- fakes -------------------------------------------------------------------------------


class ScriptedProvider:
    """A provider that replays a canned sequence of model turns.

    Satisfies :class:`~hearth.providers.base.ModelProvider` structurally so it can be driven
    through the real router. An entry is either a string (a natural stop) or a
    ``(text, finish_reason)`` pair, which is how the truncation path is rehearsed without a
    model that actually runs out of tokens.

    Running past the end of the script returns an unparseable marker rather than raising: a
    raise would be wrapped as a provider error and could be mistaken for the failure a test
    was actually asserting.
    """

    name = "scripted"

    def __init__(self, turns, *, prompt_tokens: int = 20, completion_tokens: int = 10) -> None:
        self._turns = list(turns)
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        self.requests: list[GenRequest] = []

    def capabilities(self) -> Capabilities:
        return Capabilities(chat=True, stream=False)

    def generate(self, req: GenRequest) -> GenResult:
        self.requests.append(req)
        if self._turns:
            turn = self._turns.pop(0)
        else:
            turn = "SCRIPT EXHAUSTED — the test asked for more turns than it scripted"
        text, finish = turn if isinstance(turn, tuple) else (turn, FINISH_STOP)
        return GenResult(
            text=text,
            model=req.model,
            backend=self.name,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            finish_reason=finish,
        )

    def stream(self, req: GenRequest) -> Iterator[str]:
        yield self.generate(req).text

    def footprint(self, model_id: str) -> ResourceEstimate:
        return ResourceEstimate()


class ExplodingProvider:
    """A provider that always fails to generate — the missing-weights / crashed-backend case."""

    name = "exploding"

    def capabilities(self) -> Capabilities:
        return Capabilities(chat=True)

    def generate(self, req: GenRequest) -> GenResult:
        raise RuntimeError("weights not loadable")

    def stream(self, req: GenRequest) -> Iterator[str]:
        raise RuntimeError("weights not loadable")

    def footprint(self, model_id: str) -> ResourceEstimate:
        return ResourceEstimate()


class FakeClock:
    """A monotonic clock that advances a fixed amount per read, so timeouts are exact."""

    def __init__(self, step: float = 0.0) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


# -- fixtures ----------------------------------------------------------------------------


@pytest.fixture
def notes():
    """Two tools over an in-memory 'filesystem', plus a record of what was actually called."""
    files = {"/notes/march.txt": "March total was 120.", "/notes/april.txt": "April: 90."}
    calls: list[tuple[str, dict]] = []

    def list_notes() -> list[str]:
        calls.append(("list_notes", {}))
        return sorted(files)

    def read_note(path: str) -> str:
        calls.append(("read_note", {"path": path}))
        if path not in files:
            raise FileNotFoundError(f"no such note: {path}")
        return files[path]

    tools = [
        Tool(name="list_notes", description="List the notes.", call=list_notes, returns="paths"),
        Tool(
            name="read_note",
            description="Read one note.",
            call=read_note,
            params=(ToolParam(name="path", type="string", description="Note path."),),
        ),
    ]
    return SimpleNamespace(tools=tools, calls=calls, files=files)


def _agent(turns, tools, policy, *, budget=None, clock=None, provider=None, **kwargs) -> Agent:
    """Build an agent whose 'model' replays ``turns`` through a real router."""
    provider = provider or ScriptedProvider(turns)
    router = Router(
        local_provider=provider,
        policy=policy,
        budget=BudgetAccountant(policy.defaults.remote_budget_tokens_per_day),
        metrics=MetricsStore(),
    )
    # vetted_only=False because these tools are declared in the test file, outside the
    # package the no-network invariant covers. The default is exercised separately below.
    return Agent(
        router,
        tools,
        budget=budget or Budget(),
        vetted_only=False,
        clock=clock or FakeClock(),
        **kwargs,
    )


def _call(tool: str, **arguments) -> str:
    import json

    return json.dumps({"thought": "next", "tool": tool, "arguments": arguments})


def _answer(text: str) -> str:
    import json

    return json.dumps({"thought": "done", "answer": text})


# -- a successful multi-step run -----------------------------------------------------------


def test_a_multi_step_run_calls_each_tool_and_ends_with_an_answer(notes, local_policy):
    agent = _agent(
        [_call("list_notes"), _call("read_note", path="/notes/march.txt"), _answer("120")],
        notes.tools,
        local_policy,
    )
    run = agent.run("What was the March total?")

    assert run.stopped_reason == STOPPED_ANSWERED
    assert run.completed is True
    assert run.answer == "120"
    assert run.require_answer() == "120"
    assert run.iterations == 3
    assert notes.calls == [
        ("list_notes", {}),
        ("read_note", {"path": "/notes/march.txt"}),
    ]
    assert [s.kind for s in run.steps] == ["tool_call", "tool_call", "answer"]
    assert run.total_tokens == run.prompt_tokens + run.completion_tokens > 0


def test_every_generation_is_hard_local_and_the_outcome_is_checked(notes, local_policy):
    provider = ScriptedProvider([_answer("ok")])
    agent = _agent([], notes.tools, local_policy, provider=provider)
    run = agent.run("anything")

    assert run.completed
    assert provider.requests, "the agent must actually have called the provider"
    assert run.steps[0].backend == "scripted"


def test_the_tool_result_reaches_the_next_prompt_so_the_model_can_use_it(notes, local_policy):
    provider = ScriptedProvider([_call("read_note", path="/notes/april.txt"), _answer("90")])
    agent = _agent([], notes.tools, local_policy, provider=provider)
    agent.run("What was April?")

    second_prompt = provider.requests[1].messages[-1].content
    assert "Observation from read_note" in second_prompt
    assert "April: 90." in second_prompt


# -- recoverable failures ------------------------------------------------------------------


def test_an_unknown_tool_is_fed_back_and_the_run_continues(notes, local_policy):
    agent = _agent(
        [_call("fetch_url", url="anything"), _answer("could not do that")],
        notes.tools,
        local_policy,
    )
    run = agent.run("Fetch something.")

    assert run.stopped_reason == STOPPED_ANSWERED
    assert run.iterations == 2
    first = run.steps[0]
    assert first.kind == "invalid"
    assert first.ok is False
    assert "no tool named 'fetch_url'" in (first.error or "")
    assert "list_notes" in (first.error or "")  # it is told what does exist
    assert notes.calls == []  # nothing ran


def test_malformed_output_is_fed_back_and_the_run_continues(notes, local_policy):
    agent = _agent(
        ["I will now read all of your notes, one moment.", _answer("done")],
        notes.tools,
        local_policy,
    )
    run = agent.run("Read the notes.")

    assert run.stopped_reason == STOPPED_ANSWERED
    assert run.steps[0].kind == "invalid"
    assert "no JSON object" in (run.steps[0].error or "")
    assert notes.calls == []


def test_arguments_failing_schema_validation_never_reach_the_tool(notes, local_policy):
    agent = _agent(
        [_call("read_note", filename="/notes/march.txt"), _answer("gave up")],
        notes.tools,
        local_policy,
    )
    run = agent.run("Read March.")

    assert run.steps[0].kind == "invalid"
    assert "has no parameter" in (run.steps[0].error or "")
    assert notes.calls == [], "validation must happen before dispatch"


def test_a_missing_required_argument_is_reported_by_name(notes, local_policy):
    agent = _agent([_call("read_note"), _answer("gave up")], notes.tools, local_policy)
    run = agent.run("Read something.")
    assert "requires the parameter 'path'" in (run.steps[0].error or "")


def test_a_tool_that_raises_becomes_an_observation_not_a_crash(notes, local_policy):
    agent = _agent(
        [_call("read_note", path="/notes/nope.txt"), _answer("that note does not exist")],
        notes.tools,
        local_policy,
    )
    run = agent.run("Read the missing note.")

    assert run.stopped_reason == STOPPED_ANSWERED
    assert run.steps[0].kind == "invalid"
    assert "no such note" in (run.steps[0].error or "")
    assert run.steps[0].arguments == {"path": "/notes/nope.txt"}


def test_a_truncated_turn_is_reported_and_never_parsed(notes, local_policy):
    # The severed text below is still valid JSON and would dispatch happily. Parsing it
    # would be a wrong action taken with full confidence — the exact failure FinishReason
    # exists to expose (CLAUDE.md §3, gateway/app.py).
    agent = _agent(
        [(_call("read_note", path="/notes/march.txt"), FINISH_LENGTH), _answer("stopped")],
        notes.tools,
        local_policy,
    )
    run = agent.run("Read March.")

    assert run.steps[0].kind == "invalid"
    assert run.steps[0].truncated_output is True
    assert "cut off" in (run.steps[0].error or "")
    assert notes.calls == []


# -- the bounds ----------------------------------------------------------------------------


def test_the_iteration_bound_stops_the_run_and_says_so(notes, local_policy):
    agent = _agent(
        [_call("list_notes")] * 5,
        notes.tools,
        local_policy,
        budget=Budget(max_iterations=2),
    )
    run = agent.run("Keep listing forever.")

    assert run.stopped_reason == STOPPED_MAX_ITERATIONS
    assert run.completed is False
    assert run.answer is None
    assert run.iterations == 2
    assert "2-step limit" in run.detail


def test_the_wall_clock_bound_stops_the_run_and_says_so(notes, local_policy):
    agent = _agent(
        [_call("list_notes")] * 5,
        notes.tools,
        local_policy,
        budget=Budget(max_seconds=3.0),
        clock=FakeClock(step=1.0),
    )
    run = agent.run("Take your time.")

    assert run.stopped_reason == STOPPED_TIMEOUT
    assert run.answer is None
    assert "3s limit" in run.detail


def test_the_token_bound_stops_the_run_and_says_so(notes, local_policy):
    provider = ScriptedProvider([_call("list_notes")] * 5, prompt_tokens=100, completion_tokens=100)
    agent = _agent(
        [],
        notes.tools,
        local_policy,
        provider=provider,
        budget=Budget(max_total_tokens=150),
    )
    run = agent.run("Spend everything.")

    assert run.stopped_reason == STOPPED_TOKEN_BUDGET
    assert run.answer is None
    assert run.iterations == 1
    assert run.total_tokens == 200  # what was actually spent, not what was budgeted
    assert "150-token limit" in run.detail


def test_a_model_that_cannot_follow_the_contract_terminates(notes, local_policy):
    agent = _agent(
        ["nope"] * 10,
        notes.tools,
        local_policy,
        budget=Budget(max_iterations=20, max_consecutive_invalid=3),
    )
    run = agent.run("Do something.")

    assert run.stopped_reason == STOPPED_INVALID_OUTPUT
    assert run.iterations == 3, "it must stop at the consecutive cap, not at max_iterations"
    assert "not following the output contract" in run.detail


def test_a_good_step_resets_the_consecutive_invalid_counter(notes, local_policy):
    agent = _agent(
        ["nope", _call("list_notes"), "nope", "nope"],
        notes.tools,
        local_policy,
        budget=Budget(max_iterations=20, max_consecutive_invalid=2),
    )
    run = agent.run("Do something.")

    assert run.stopped_reason == STOPPED_INVALID_OUTPUT
    assert run.iterations == 4


def test_a_provider_failure_ends_the_run_with_its_own_reason(notes, local_policy):
    agent = _agent([], notes.tools, local_policy, provider=ExplodingProvider())
    run = agent.run("Anything.")

    assert run.stopped_reason == STOPPED_PROVIDER_ERROR
    assert run.answer is None
    assert "weights not loadable" in run.detail


# -- "finished" and "out of budget" must be impossible to confuse ---------------------------


def test_an_unanswered_run_cannot_be_read_as_an_answer(notes, local_policy):
    agent = _agent([_call("list_notes")], notes.tools, local_policy,
                   budget=Budget(max_iterations=1))
    run = agent.run("Keep going.")

    assert run.answer is None
    with pytest.raises(AgentIncompleteError, match="max_iterations"):
        run.require_answer()


def test_the_result_type_refuses_to_hold_an_answer_it_did_not_earn():
    # The invariant is structural, not a convention a caller has to remember.
    with pytest.raises(AgentConfigError, match="must not carry an answer"):
        AgentRun(
            task="t",
            stopped_reason=STOPPED_MAX_ITERATIONS,
            answer="a plausible-looking summary",
            steps=(),
            budget=Budget(),
        )
    with pytest.raises(AgentConfigError, match="must carry the answer"):
        AgentRun(
            task="t", stopped_reason=STOPPED_ANSWERED, answer=None, steps=(), budget=Budget()
        )


def test_an_unknown_stop_reason_is_refused():
    with pytest.raises(AgentConfigError, match="unknown stopped_reason"):
        AgentRun(
            task="t",
            stopped_reason="probably_fine",  # type: ignore[arg-type]
            answer=None,
            steps=(),
            budget=Budget(),
        )


# -- the transcript ------------------------------------------------------------------------


def test_the_transcript_reconstructs_the_whole_run(notes, local_policy):
    agent = _agent(
        [
            _call("fetch_url", url="x"),
            _call("read_note", path="/notes/march.txt"),
            _answer("March totalled 120."),
        ],
        notes.tools,
        local_policy,
    )
    run = agent.run("What was the March total?")
    text = run.transcript()

    assert "task           : What was the March total?" in text
    assert f"stopped_reason : {STOPPED_ANSWERED}" in text
    assert "answer         : March totalled 120." in text
    # every step, in order, with what ran and what came back
    assert "step 1 [invalid]" in text
    assert "no tool named 'fetch_url'" in text
    assert "step 2 [tool_call] read_note(path='/notes/march.txt')" in text
    assert "March total was 120." in text  # the observation the answer rests on
    assert "step 3 [answer]" in text
    assert "tokens         :" in text
    assert "NOT AN ANSWER" not in text


def test_an_incomplete_transcript_says_so_where_the_answer_would_be(notes, local_policy):
    agent = _agent([_call("list_notes")], notes.tools, local_policy,
                   budget=Budget(max_iterations=1))
    text = agent.run("Keep going.").transcript()

    assert "answer         : (none)" in text
    assert "NOT AN ANSWER" in text
    assert "step 1 [tool_call] list_notes()" in text


def test_the_steps_carry_the_timings_and_token_counts_of_each_turn(notes, local_policy):
    agent = _agent([_call("list_notes"), _answer("two")], notes.tools, local_policy)
    run = agent.run("How many notes?")

    assert all(s.prompt_tokens == 20 and s.completion_tokens == 10 for s in run.steps)
    assert run.prompt_tokens == 40 and run.completion_tokens == 20
    assert run.tool_calls[0].tool == "list_notes"


# -- the egress gate -------------------------------------------------------------------------


def test_a_step_not_served_locally_aborts_the_run(notes, local_policy):
    """The gate asserts on the executed route, not on the flag we passed in.

    A router that ignored ``allow_escalation=False`` — a policy edit, a future backend, a bug
    — would otherwise give a quietly remote agent while every configuration still read
    "local". This is the ``CLAUDE.md`` §3 shape, so the check is on the outcome.
    """

    class RemoteServingRouter:
        def route(self, req, intent=None, allow_escalation=True):
            return SimpleNamespace(
                result=GenResult(text=_answer("42"), model="m", backend="remote"),
                decision=SimpleNamespace(backend="remote"),
                record=SimpleNamespace(served_by="remote"),
            )

    agent = Agent(RemoteServingRouter(), notes.tools, vetted_only=False, clock=FakeClock())
    run = agent.run("Anything.")

    assert run.stopped_reason == STOPPED_EGRESS_REFUSED
    assert run.answer is None
    assert "on-device only" in run.detail


# -- construction --------------------------------------------------------------------------


def test_an_agent_with_no_tools_is_refused(local_policy):
    with pytest.raises(AgentConfigError, match="at least one tool"):
        _agent([], [], local_policy)


def test_an_empty_task_is_refused(notes, local_policy):
    with pytest.raises(AgentConfigError, match="needs a task"):
        _agent([], notes.tools, local_policy).run("   ")


def test_a_tool_defined_outside_the_package_is_refused_by_default(notes, local_policy):
    provider = ScriptedProvider([])
    router = Router(local_provider=provider, policy=local_policy)
    with pytest.raises(UnvettedToolError, match="outside"):
        Agent(router, notes.tools)  # vetted_only defaults to True


def test_the_built_in_toolset_passes_the_vetting_gate(local_policy, settings):
    provider = ScriptedProvider([])
    router = Router(local_provider=provider, policy=local_policy)
    agent = Agent(router, local_toolset(settings=settings))  # must not raise
    assert set(agent.registry.names) == {"read_file", "list_files"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_iterations": 0},
        {"max_total_tokens": 0},
        {"max_step_tokens": 0},
        {"max_consecutive_invalid": 0},
        {"max_seconds": 0},
        {"max_observation_chars": -1},
    ],
)
def test_a_bound_that_could_never_stop_anything_is_refused(kwargs):
    with pytest.raises(AgentConfigError):
        Budget(**kwargs)
