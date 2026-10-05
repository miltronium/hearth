"""B-071: an abandoned ``/v1/hearth/agent`` stream stops the run — between steps AND mid-step.

Before: the agent route wrapped its SSE body with ``_guarantee_done`` but not the chat
stream's ``_close_on_disconnect``, so a client that went away left the run going: the current
generation ran to completion on the single MLX thread, then the loop started the next step,
up to its full budget (minutes on a 14B), while every chat request queued behind it. Closing
the generator would not have helped either — nothing told ``work()`` or the in-flight
generation to stop.

Held-reference pattern (tests/test_mlx_thread_confinement.py): the response object is held,
as the live server holds it, so CPython's refcounting cannot close the generator on the
test's behalf and make a mutant pass.
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import anyio
import pytest
from test_mlx_thread_confinement import ThreadLocalMLX

from hearth.agent import STOPPED_CANCELLED, Agent, Budget, local_toolset
from hearth.config import Settings
from hearth.gateway import agent_route, create_app
from hearth.gateway.schemas import AgentRunRequest
from hearth.observability.budget import BudgetAccountant
from hearth.observability.metrics import MetricsStore
from hearth.providers.base import (
    Capabilities,
    GenerationCancelledError,
    GenRequest,
    GenResult,
    Message,
    ResourceEstimate,
    cancel_scope,
)
from hearth.providers.mlx import MLXProvider, run_on_mlx_thread
from hearth.router import Router

CODER7 = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"


class EndlessMLX(ThreadLocalMLX):
    """Generates until told to stop; counts generation calls (one per agent step)."""

    def __init__(self) -> None:
        super().__init__(tokens=10**9, delay=0.001)
        self.calls = 0
        self.halt = False

    def stream_generate(self, model, tokenizer, prompt, max_tokens):
        self.calls += 1
        for response in super().stream_generate(model, tokenizer, prompt, max_tokens):
            if self.halt:  # teardown only: a failing build must not wedge the MLX thread
                return
            yield response


@pytest.fixture
def endless(monkeypatch):
    fake = EndlessMLX()
    fake.install(monkeypatch)
    yield fake
    # Teardown runs before monkeypatch restores the real mlx_lm: a run a failing build left
    # going must drain against THIS fake (halted: every turn is empty, so it ends quickly).
    fake.halt = True
    run_on_mlx_thread(lambda: None)
    agent_route._RUNNER.submit(lambda: None).result(timeout=60)


def _wait_until_still(read, seconds: float = 3.0) -> None:
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        before = read()
        time.sleep(0.2)
        if read() == before:
            return
    raise AssertionError(f"still running after {seconds}s ({read()} and counting)")


@pytest.mark.parametrize("close", ["closed", "close-unreachable"])
@pytest.mark.parametrize("spec_version", ["2.0", "2.4"])
def test_an_abandoned_agent_stream_stops_the_in_flight_generation_and_the_run(
    endless, tmp_path, local_policy, monkeypatch, spec_version, close
):
    """Driven through Starlette's own ``StreamingResponse.__call__``, in both of its modes:
    ASGI < 2.4 (a disconnect message cancels the response's task group) and >= 2.4 (the
    next ``send`` raises OSError). Either way the server learns of the disconnect only
    between body chunks.

    ``close-unreachable`` (B-111): two mechanisms stop the run — the route's
    ``on_close=cancel.set`` (run synchronously when the response ends) and the generator's
    own ``GeneratorExit`` handler (run when the helper thread manages to ``close()`` it).
    With both live, deleting ``on_close`` left every test green. Here the close never
    reaches the generator, so only ``on_close`` can stop the generation."""
    from starlette.requests import ClientDisconnect

    from hearth.gateway import app as app_mod

    if close == "close-unreachable":
        monkeypatch.setattr(app_mod, "_close_when_idle", lambda stream: None)

    root = tmp_path / "statements"
    root.mkdir()
    (root / "a.txt").write_text("x\n")
    provider = MLXProvider(CODER7)
    metrics = MetricsStore()
    router = Router(local_provider=provider, policy=local_policy, budget=BudgetAccountant(0),
                    metrics=metrics)
    settings = Settings(backend="mlx", home=tmp_path / ".hearth", require_auth=False,
                        warmup=False, file_roots=str(root))
    app = create_app(provider=provider, settings=settings, router=router, metrics=metrics)
    endpoint = next(r.endpoint for r in app.routes if getattr(r, "path", "") == "/v1/hearth/agent")

    # Held, as the live server evidently holds them: with no outside reference CPython's
    # refcounting closes an abandoned generator on its own, and a build that never wires the
    # disconnect would pass (a mutant without _close_on_disconnect did, before this).
    held_bodies: list = []
    real_guarantee = app_mod._guarantee_done

    def holding_guarantee(stream):
        body = real_guarantee(stream)
        held_bodies.append((stream, body))
        return body

    monkeypatch.setattr(app_mod, "_guarantee_done", holding_guarantee)
    response = endpoint(SimpleNamespace(app=app), AgentRunRequest(task="read my statements"))
    assert held_bodies, "the route no longer builds its body through _guarantee_done"

    # A failing build must fail, not hang: halt the fake generation after 8 s regardless.
    watchdog = threading.Timer(8.0, lambda: setattr(endless, "halt", True))
    watchdog.daemon = True
    watchdog.start()
    left: list[float] = []

    def client_gone() -> bool:
        if endless.emitted and not left:
            left.append(time.perf_counter())
        return bool(left)

    async def receive():
        while not client_gone():
            await anyio.sleep(0.01)
        return {"type": "http.disconnect"}

    async def send(message):
        if spec_version == "2.4" and client_gone():
            raise OSError("the client went away")

    scope = {"type": "http", "asgi": {"spec_version": spec_version}}
    try:
        anyio.run(response, scope, receive, send)
    except ClientDisconnect:
        pass
    noticed = time.perf_counter() - left[0]
    watchdog.cancel()
    assert noticed < 3.0, f"the disconnect took {noticed:.1f}s to reach the stream"
    _wait_until_still(lambda: endless.emitted)  # the in-flight generation stopped
    run_on_mlx_thread(lambda: None)  # the MLX thread is free for the next request
    agent_route._RUNNER.submit(lambda: None).result(timeout=3)  # and so is the agent runner
    assert endless.calls == 1  # no further step was started after the client left
    # The cut-short generation is accounted for as a failed request, not lost.
    (rec,) = list(metrics._records)
    assert rec.failed and "cancel" in rec.failed


# -- the loop and the router, without HTTP ------------------------------------------------


class _Scripted:
    name = "scripted"

    def __init__(self, on_generate=None) -> None:
        self.calls = 0
        self.on_generate = on_generate

    def capabilities(self) -> Capabilities:
        return Capabilities(chat=True)

    def footprint(self, model_id: str) -> ResourceEstimate:
        return ResourceEstimate()

    def generate(self, req: GenRequest) -> GenResult:
        self.calls += 1
        if self.on_generate is not None:
            self.on_generate()
        text = json.dumps({"thought": "look", "tool": "list_files", "arguments": {"path": "."}})
        return GenResult(text=text, model=req.model, backend=self.name, prompt_tokens=5,
                         completion_tokens=5)

    def stream(self, req):  # pragma: no cover
        yield self.generate(req).text


def _agent(provider, settings, cancel, local_policy) -> Agent:
    router = Router(local_provider=provider, policy=local_policy, budget=BudgetAccountant(0),
                    metrics=MetricsStore())
    return Agent(router, local_toolset(settings=settings), budget=Budget(max_iterations=5),
                 cancel=cancel)


def test_the_loop_checks_the_flag_between_steps(tmp_path, local_policy):
    cancel = threading.Event()
    provider = _Scripted(on_generate=cancel.set)  # the client leaves during step 1
    settings = Settings(backend="echo", home=tmp_path / "h", file_roots=str(tmp_path))
    run = _agent(provider, settings, cancel, local_policy).run("list my files")
    assert run.stopped_reason == STOPPED_CANCELLED
    assert provider.calls == 1 and run.answer is None
    assert run.iterations == 1  # step 1 is in the transcript; step 2 never started


def test_a_run_cancelled_before_it_starts_spends_nothing(tmp_path, local_policy):
    cancel = threading.Event()
    cancel.set()
    provider = _Scripted()
    settings = Settings(backend="echo", home=tmp_path / "h", file_roots=str(tmp_path))
    run = _agent(provider, settings, cancel, local_policy).run("list my files")
    assert run.stopped_reason == STOPPED_CANCELLED and provider.calls == 0


def test_a_cancelled_generation_is_not_retried_on_base_weights(local_policy):
    """The router's degrade-and-retry must not turn a cancellation into a second generation."""
    calls: list[str | None] = []

    class Cancelling(_Scripted):
        def generate(self, req):
            calls.append(req.adapter)
            raise GenerationCancelledError("cancelled")

    class Store:
        def resolve_path(self, adapter_id, allow_candidate=False):
            return "/a/x"

        def promoted_for(self, task):
            return None

    router = Router(local_provider=Cancelling(), policy=local_policy,
                    budget=BudgetAccountant(0), metrics=MetricsStore(), adapters=Store())
    with pytest.raises(GenerationCancelledError):
        router.route(GenRequest(messages=[Message(role="user", content="hi")], model="auto"),
                     adapter="x")
    assert calls == ["/a/x"]
    (rec,) = list(router.metrics._records)
    assert rec.failed


def test_the_mlx_token_loop_honours_the_scope(endless):
    provider = MLXProvider(CODER7)
    cancel = threading.Event()
    req = GenRequest(messages=[Message(role="user", content="go")], model=CODER7)
    timer = threading.Timer(0.3, cancel.set)
    timer.start()
    # A failing build must fail, not hang: the fake stops by itself after 8 s.
    watchdog = threading.Timer(8.0, lambda: setattr(endless, "halt", True))
    watchdog.daemon = True
    watchdog.start()
    started = time.perf_counter()
    try:
        with cancel_scope(cancel), pytest.raises(GenerationCancelledError):
            provider.generate(req)  # runs on the MLX thread, which must see this scope
    finally:
        watchdog.cancel()
    assert time.perf_counter() - started < 3.0
