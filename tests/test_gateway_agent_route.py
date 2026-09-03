"""``POST /v1/hearth/agent``: the security posture, the bounds, and the stop reason.

Everything here runs offline with no model. A scripted provider replays canned model turns
through the **real** router and the **real** agent loop, so these tests exercise the same path
a run on local weights takes — only the token source is fake.

The load-bearing assertions are not about the happy path. They are:

* there is no field, anywhere in the request model, that can set ``vetted_only=False``;
* a client's budget is clamped to a server ceiling rather than honoured;
* an unset ``HEARTH_FILE_ROOTS`` is reported in the first event, before the loop spends
  anything discovering it;
* and a run that hit a bound does not carry an answer *on the wire* — the invariant
  :class:`~hearth.agent.AgentRun` enforces in-process has to survive serialisation, because
  the serialisation boundary is where this repository lost ``finish_reason`` once already
  (``CLAUDE.md`` §3).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from hearth.config import Settings, get_or_create_token
from hearth.gateway import create_app
from hearth.gateway.agent_route import (
    _NO_ROOTS_WARNING,
    MAX_ITERATIONS_CEILING,
    MAX_SECONDS_CEILING,
    MAX_TOTAL_TOKENS_CEILING,
    OBSERVATION_WIRE_CHARS,
    _stream_agent,
)
from hearth.gateway.schemas import (
    AgentBudgetApplied,
    AgentBudgetRequest,
    AgentRunEvent,
    AgentRunRequest,
    AgentStartEvent,
)
from hearth.observability.budget import BudgetAccountant
from hearth.observability.metrics import MetricsStore
from hearth.providers.base import (
    FINISH_STOP,
    Capabilities,
    GenRequest,
    GenResult,
    ResourceEstimate,
)
from hearth.router import Router

# -- fakes -------------------------------------------------------------------------------


class ScriptedProvider:
    """Replays a canned sequence of model turns through the real router.

    Structurally a :class:`~hearth.providers.base.ModelProvider`. Running past the end of the
    script returns an unparseable marker rather than raising, so a test asserting on a bound
    never accidentally asserts on a provider error instead.
    """

    name = "scripted"

    def __init__(self, turns: list[str]) -> None:
        self._turns = list(turns)
        self.requests: list[GenRequest] = []

    def capabilities(self) -> Capabilities:
        return Capabilities(chat=True, stream=False)

    def generate(self, req: GenRequest) -> GenResult:
        self.requests.append(req)
        text = (
            self._turns.pop(0)
            if self._turns
            else "SCRIPT EXHAUSTED — the test asked for more turns than it scripted"
        )
        return GenResult(
            text=text,
            model=req.model,
            backend=self.name,
            prompt_tokens=20,
            completion_tokens=10,
            finish_reason=FINISH_STOP,
        )

    def stream(self, req: GenRequest) -> Iterator[str]:
        yield self.generate(req).text

    def footprint(self, model_id: str) -> ResourceEstimate:
        return ResourceEstimate()


def _call(tool: str, **arguments: Any) -> str:
    return json.dumps({"thought": f"use {tool}", "tool": tool, "arguments": arguments})


def _answer(text: str) -> str:
    return json.dumps({"thought": "enough", "answer": text})


# -- harness -----------------------------------------------------------------------------


def _build(
    tmp_path,
    local_policy,
    turns: list[str],
    *,
    file_roots: str = "",
    require_auth: bool = False,
) -> tuple[TestClient, Settings, ScriptedProvider]:
    settings = Settings(
        backend="echo",
        home=tmp_path / ".hearth",
        require_auth=require_auth,
        file_roots=file_roots,
    )
    provider = ScriptedProvider(turns)
    router = Router(
        local_provider=provider,
        policy=local_policy,
        budget=BudgetAccountant(local_policy.defaults.remote_budget_tokens_per_day),
        metrics=MetricsStore(),
    )
    app = create_app(provider=provider, settings=settings, router=router)
    return TestClient(app), settings, provider


def _events(response) -> list[Any]:
    """Parse an SSE body into payloads. Each event is one line: the models JSON-escape."""
    out: list[Any] = []
    for line in response.text.splitlines():
        if line.startswith("data: "):
            payload = line[len("data: ") :]
            out.append(payload if payload == "[DONE]" else json.loads(payload))
    return out


def _of(events: list[Any], kind: str) -> list[dict]:
    return [e for e in events if isinstance(e, dict) and e.get("object") == kind]


@pytest.fixture
def statements(tmp_path):
    """A root with one readable file, so a scripted run has something real to observe."""
    root = tmp_path / "statements"
    root.mkdir()
    (root / "march.txt").write_text("March total was 120.50\n", encoding="utf-8")
    return root


# -- the security posture ----------------------------------------------------------------


def test_no_request_field_can_disable_tool_vetting():
    """There is no wire path to ``vetted_only=False``. Asserted over the model's own fields.

    A test that merely checked the route passes ``vetted_only=True`` today would stay green
    the moment someone adds a pass-through field. This asserts on the request *shape*: the
    capability simply is not expressible.
    """
    names: set[str] = set()

    def walk(model) -> None:
        for name, field in model.model_fields.items():
            names.add(name)
            for candidate in (field.annotation, *getattr(field.annotation, "__args__", ())):
                if hasattr(candidate, "model_fields") and candidate is not model:
                    walk(candidate)

    walk(AgentRunRequest)
    assert "vetted_only" not in names
    assert not {n for n in names if "vet" in n or "unsafe" in n or "tool" in n}
    # Nor a way to name a root, a path, a shell or extra tools.
    assert names == {"task", "model", "budget", "max_iterations", "max_seconds",
                     "max_total_tokens"}


def test_an_unknown_field_is_refused_rather_than_ignored(tmp_path, local_policy):
    """Sending ``vetted_only`` is a 422, not a silently dropped key.

    Pydantic's default would ignore it — and an ignored field reads, to anyone inspecting a
    request log, exactly like an honoured one.
    """
    client, _, _ = _build(tmp_path, local_policy, [_answer("hi")])
    r = client.post("/v1/hearth/agent", json={"task": "hello", "vetted_only": False})
    assert r.status_code == 422
    r = client.post(
        "/v1/hearth/agent",
        json={"task": "hello", "budget": {"max_iterations": 2, "vetted_only": False}},
    )
    assert r.status_code == 422


def test_the_route_requires_a_token(tmp_path, local_policy, statements):
    """Authenticated like every other /v1 route — never exempted for convenience."""
    client, settings, _ = _build(
        tmp_path,
        local_policy,
        [_answer("done")],
        file_roots=str(statements),
        require_auth=True,
    )
    assert client.post("/v1/hearth/agent", json={"task": "hi"}).status_code == 401

    token = get_or_create_token(settings)
    ok = client.post(
        "/v1/hearth/agent",
        json={"task": "hi"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert ok.status_code == 200


def test_only_the_read_only_builtins_are_offered(tmp_path, local_policy, statements):
    """No shell, no write, no fetch — the advertised toolset is the read-only built-ins."""
    client, _, _ = _build(
        tmp_path, local_policy, [_answer("done")], file_roots=str(statements)
    )
    events = _events(client.post("/v1/hearth/agent", json={"task": "hi"}))
    start = _of(events, "hearth.agent.start")
    assert len(start) == 1
    tools = set(start[0]["tools"])
    assert {"read_file", "list_files"} <= tools
    assert not {t for t in tools if any(bad in t for bad in ("write", "shell", "exec", "fetch"))}
    # The posture is reported, not assumed by the client.
    assert start[0]["vetted_only"] is True


def test_file_reads_stay_governed_by_hearth_file_roots(tmp_path, local_policy, statements):
    """The route grants no filesystem privilege of its own: outside a root is still refused."""
    outside = tmp_path / "private.txt"
    outside.write_text("not yours", encoding="utf-8")
    client, _, _ = _build(
        tmp_path,
        local_policy,
        [_call("read_file", path=str(outside)), _answer("refused")],
        file_roots=str(statements),
    )
    events = _events(client.post("/v1/hearth/agent", json={"task": "read it"}))
    steps = _of(events, "hearth.agent.step")
    assert steps[0]["kind"] == "invalid"
    assert "HEARTH_FILE_ROOTS" in steps[0]["error"]
    assert "not yours" not in json.dumps(events)


# -- the bounds --------------------------------------------------------------------------


def test_a_client_budget_is_clamped_to_the_server_ceiling(tmp_path, local_policy, statements):
    """Ten thousand iterations gets the cap, and the stream says which bound was reduced."""
    client, _, _ = _build(
        tmp_path, local_policy, [_answer("done")], file_roots=str(statements)
    )
    r = client.post(
        "/v1/hearth/agent",
        json={
            "task": "hi",
            "budget": {
                "max_iterations": 10_000,
                "max_seconds": 99_999,
                "max_total_tokens": 10_000_000,
            },
        },
    )
    applied = _of(_events(r), "hearth.agent.start")[0]["budget"]
    assert applied["max_iterations"] == MAX_ITERATIONS_CEILING
    assert applied["max_seconds"] == MAX_SECONDS_CEILING
    assert applied["max_total_tokens"] == MAX_TOTAL_TOKENS_CEILING
    assert set(applied["clamped"]) == {"max_iterations", "max_seconds", "max_total_tokens"}


def test_a_smaller_budget_is_honoured_and_not_reported_as_clamped(
    tmp_path, local_policy, statements
):
    client, _, _ = _build(
        tmp_path, local_policy, [_answer("done")], file_roots=str(statements)
    )
    r = client.post(
        "/v1/hearth/agent", json={"task": "hi", "budget": {"max_iterations": 2}}
    )
    applied = _of(_events(r), "hearth.agent.start")[0]["budget"]
    assert applied["max_iterations"] == 2
    assert applied["clamped"] == []


def test_a_nonsense_budget_is_a_named_refusal(tmp_path, local_policy):
    """Zero iterations is not a small run, it is a bad request. Refused by field, loudly."""
    client, _, _ = _build(tmp_path, local_policy, [_answer("done")])
    r = client.post("/v1/hearth/agent", json={"task": "hi", "budget": {"max_iterations": 0}})
    assert r.status_code == 422
    assert client.post("/v1/hearth/agent", json={"task": "   "}).status_code == 422


# -- the preflight -----------------------------------------------------------------------


def test_unset_file_roots_is_reported_before_the_loop_runs(tmp_path, local_policy):
    """The deny-by-default state arrives in the first event, not after eight wasted steps."""
    client, _, provider = _build(tmp_path, local_policy, [_answer("done")], file_roots="")
    events = _events(client.post("/v1/hearth/agent", json={"task": "read my statements"}))

    assert events[0]["object"] == "hearth.agent.start"
    assert events[0]["file_roots"] == 0
    assert any("HEARTH_FILE_ROOTS" in w for w in events[0]["warnings"])
    # Also carried on the terminal event, for a client that reads only the outcome.
    assert any("HEARTH_FILE_ROOTS" in w for w in _of(events, "hearth.agent.run")[0]["warnings"])
    assert len(provider.requests) >= 1  # the run still proceeded; RAG remained reachable


def test_the_warning_is_emitted_before_the_agent_is_even_constructed():
    """"Up front" has to mean *before the run starts*, not merely first in a buffered body.

    The route's own generator is driven directly here — the one place the ordering is a fact
    rather than a race. Advancing it exactly once must produce the warning while the agent
    factory has not been called at all; over HTTP, Starlette writes each yielded chunk as it
    arrives, so that first chunk is on the wire before any token is spent. Asserting on the
    order of a fully-read response body would say nothing: a stream flushed only at the end
    would satisfy it, and the warning would arrive after the run it exists to pre-empt.
    """
    built: list[Any] = []

    def factory(on_step: Any) -> Any:
        built.append(on_step)
        raise AssertionError("the agent must not be built before the client is warned")

    applied = AgentBudgetApplied(max_iterations=8, max_seconds=180.0, max_total_tokens=24_000)
    stream = _stream_agent(
        agent_factory=factory,
        task="read my statements",
        start=AgentStartEvent(
            task="read my statements",
            tools=["list_files", "read_file"],
            budget=applied,
            file_roots=0,
            warnings=[_NO_ROOTS_WARNING],
        ),
        applied=applied,
        warnings=[_NO_ROOTS_WARNING],
        reachable=True,
    )
    try:
        first = next(stream)
        assert '"hearth.agent.start"' in first
        assert "HEARTH_FILE_ROOTS" in first
        assert built == []
    finally:
        stream.close()


def test_a_run_that_can_reach_nothing_is_refused_before_it_spends_anything(
    tmp_path, local_policy
):
    """No roots, no RAG, no ledger: every tool would refuse, so no generation is spent."""
    client, _, provider = _build(tmp_path, local_policy, [_answer("done")], file_roots="")
    client.app.state.rag = None

    events = _events(client.post("/v1/hearth/agent", json={"task": "read my statements"}))
    assert events[0]["object"] == "hearth.agent.start"
    assert "error" in events[1]
    assert events[1]["error"]["code"] == "hearth.agent.no_tools_reachable"
    assert events[-1] == "[DONE]"
    assert provider.requests == []


# -- the stream, and the stop reason -----------------------------------------------------


def test_a_completed_run_streams_steps_then_a_terminal_event_then_done(
    tmp_path, local_policy, statements
):
    """One event per step, in order, each naming the tool, its arguments and what came back."""
    path = str(statements / "march.txt")
    client, _, _ = _build(
        tmp_path,
        local_policy,
        [
            _call("list_files"),
            _call("read_file", path=path),
            _answer("March totalled 120.50"),
        ],
        file_roots=str(statements),
    )
    events = _events(client.post("/v1/hearth/agent", json={"task": "what did March total?"}))

    assert events[0]["object"] == "hearth.agent.start"
    steps = _of(events, "hearth.agent.step")
    assert [s["index"] for s in steps] == [1, 2, 3]
    assert steps[0]["tool"] == "list_files"
    assert steps[1]["tool"] == "read_file"
    assert steps[1]["arguments"] == {"path": path}
    assert "120.50" in steps[1]["observation"]
    assert steps[1]["model_seconds"] >= 0.0 and steps[1]["backend"] == "scripted"
    assert steps[2]["kind"] == "answer"

    terminal = _of(events, "hearth.agent.run")
    assert len(terminal) == 1
    assert terminal[0]["stopped_reason"] == "answered"
    assert terminal[0]["completed"] is True
    assert terminal[0]["answer"] == "March totalled 120.50"
    assert terminal[0]["steps"] == 3
    # Ordering: every step precedes the terminal event, which precedes [DONE].
    assert events.index(terminal[0]) > events.index(steps[-1])
    assert events[-1] == "[DONE]"


def test_a_budget_exhausted_run_carries_no_answer_on_the_wire(
    tmp_path, local_policy, statements
):
    """The invariant survives serialisation: no ``answer`` key at all, and a loud reason."""
    client, _, _ = _build(
        tmp_path,
        local_policy,
        [_call("list_files"), _answer("March totalled 120.50")],
        file_roots=str(statements),
    )
    events = _events(
        client.post(
            "/v1/hearth/agent",
            json={"task": "what did March total?", "budget": {"max_iterations": 1}},
        )
    )
    terminal = _of(events, "hearth.agent.run")[0]
    assert terminal["stopped_reason"] == "max_iterations"
    assert terminal["completed"] is False
    assert "answer" not in terminal
    assert terminal["detail"]
    assert len(_of(events, "hearth.agent.step")) == 1
    assert events[-1] == "[DONE]"


def test_the_terminal_event_type_refuses_an_answer_it_did_not_earn():
    """The presentation model re-asserts what ``AgentRun`` guarantees in-process.

    Without this the impossible state becomes representable again the moment the run object
    is projected onto a schema — which is exactly how ``finish_reason: "stop"`` happened.
    """
    budget = {"max_iterations": 8, "max_seconds": 180.0, "max_total_tokens": 24_000}
    with pytest.raises(ValueError):
        AgentRunEvent(
            stopped_reason="max_iterations",
            completed=False,
            answer="here is a total",
            budget=budget,
        )
    with pytest.raises(ValueError):
        AgentRunEvent(stopped_reason="answered", completed=True, answer=None, budget=budget)


def test_a_long_observation_is_truncated_visibly(tmp_path, local_policy, statements):
    """Cut for the reader, and the cut is stated — never a silently shortened observation."""
    big = statements / "big.txt"
    big.write_text("x" * (OBSERVATION_WIRE_CHARS * 3), encoding="utf-8")
    client, _, _ = _build(
        tmp_path,
        local_policy,
        [_call("read_file", path=str(big)), _answer("read it")],
        file_roots=str(statements),
    )
    events = _events(client.post("/v1/hearth/agent", json={"task": "read big"}))
    step = _of(events, "hearth.agent.step")[0]
    assert step["observation_truncated"] is True
    assert "truncated for display" in step["observation"]
    assert len(step["observation"]) < OBSERVATION_WIRE_CHARS + 200


def test_a_tool_refusal_is_a_visible_step_not_a_crash(tmp_path, local_policy, statements):
    """A failed call is an observation the operator can read, and the run continues."""
    client, _, _ = _build(
        tmp_path,
        local_policy,
        [_call("read_file", path=str(statements / "nope.txt")), _answer("could not find it")],
        file_roots=str(statements),
    )
    events = _events(client.post("/v1/hearth/agent", json={"task": "read nope"}))
    steps = _of(events, "hearth.agent.step")
    assert steps[0]["kind"] == "invalid"
    assert steps[0]["error"]
    assert _of(events, "hearth.agent.run")[0]["completed"] is True


def test_the_budget_request_model_bounds_nothing_upward():
    """The ceiling lives in the server, not in the schema — a client may always ask for more.

    Pinned so the clamp stays the thing that enforces the cap. Moving the ceiling into a
    ``le=`` would turn an optimistic request into a 422 and quietly delete the clamp path.
    """
    asked = AgentBudgetRequest(max_iterations=10_000)
    assert asked.max_iterations == 10_000


def test_every_run_executes_on_one_reused_worker(tmp_path, local_policy, statements):
    """All agent runs share a single worker thread, and consecutive runs both succeed.

    A thread per request made this route work exactly ONCE per server process: MLX's GPU
    stream is thread-local, so the model loaded on request one could not be generated from
    request two's fresh thread, which died with "There is no Stream(gpu, 0) in current
    thread". Measured against a live 14B — request 1 answered, requests 2 and 3 returned
    provider_error — and it presents as a flaky model rather than as a threading bug, which
    is why it is pinned here rather than left to be rediscovered.

    The thread identity is the assertion. A scripted provider cannot reproduce MLX's
    thread-local state, so testing "two runs succeed" alone would pass again the moment
    somebody reintroduces a per-request thread.
    """
    from concurrent.futures import ThreadPoolExecutor

    from hearth.gateway import agent_route

    assert isinstance(agent_route._RUNNER, ThreadPoolExecutor)
    assert agent_route._RUNNER._max_workers == 1

    client, _, _ = _build(
        tmp_path,
        local_policy,
        [_answer("first"), _answer("second")],
        file_roots=str(statements),
    )
    threads: set[str] = set()

    for expected in ("first", "second"):
        with client.stream("POST", "/v1/hearth/agent", json={"task": "t"}) as response:
            assert response.status_code == 200
            terminal = None
            for line in response.iter_lines():
                if not line.startswith("data: ") or "[DONE]" in line:
                    continue
                payload = json.loads(line[6:])
                if payload.get("object", "").endswith("run"):
                    terminal = payload
            assert terminal is not None
            assert terminal["completed"] is True
            assert terminal["answer"] == expected
        threads.add(agent_route._RUNNER._thread_name_prefix)

    assert len(threads) == 1
