"""`hearth agent` — the CLI seam onto the bounded local agent loop.

Everything here runs offline with no model. The "model" is a scripted provider replaying a
canned sequence of turns, injected by patching :func:`hearth.cli.select_provider`, so the
command exercises the real router, the real toolset assembly and the real loop while never
touching MLX or the network. ``HEARTH_HOME`` and ``HEARTH_FILE_ROOTS`` are pinned to a tmp
path for every invocation, so no test can see the operator's own ledger or files.

The load-bearing assertions are the ones a terminal and a pipeline both depend on:

  * a run that stopped at a bound **exits non-zero and prints no answer**, because the whole
    point of ``AgentRun``'s stop reason is that an exhausted budget cannot be read as a
    conclusion (``docs/AGENT.md`` §2.4);
  * the deny-by-default file gate is reported **before the loop starts**, not discovered by
    an agent spending eight turns on refusals;
  * and there is no flag, anywhere on this command, that can turn tool vetting off.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
import typer
from typer.testing import CliRunner

from hearth.cli import app
from hearth.providers.base import (
    FINISH_STOP,
    Capabilities,
    GenRequest,
    GenResult,
    ResourceEstimate,
)

runner = CliRunner()


# -- fakes -------------------------------------------------------------------------------


class ScriptedProvider:
    """A provider that replays canned model turns and records what it was asked.

    Satisfies :class:`~hearth.providers.base.ModelProvider` structurally so it can be driven
    through the real router. ``requests`` is what lets a test assert the loop never ran at
    all — the difference between refusing up front and refusing after eight wasted turns.
    """

    name = "scripted"

    def __init__(self, turns: list[str]) -> None:
        self._turns = list(turns)
        self.requests: list[GenRequest] = []

    def capabilities(self) -> Capabilities:
        return Capabilities(chat=True, stream=False)

    def generate(self, req: GenRequest) -> GenResult:
        self.requests.append(req)
        text = self._turns.pop(0) if self._turns else "SCRIPT EXHAUSTED"
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


def _call(tool: str, **arguments) -> str:
    return json.dumps({"thought": "next", "tool": tool, "arguments": arguments})


def _answer(text: str) -> str:
    return json.dumps({"thought": "enough", "answer": text})


# -- fixtures ----------------------------------------------------------------------------


@pytest.fixture
def statements(tmp_path):
    """An allowlisted directory with two CSVs and a file that is not one."""
    root = tmp_path / "statements"
    root.mkdir()
    (root / "january.csv").write_text("date,amount\n2026-01-04,12.00\n", encoding="utf-8")
    (root / "february.csv").write_text("date,amount\n2026-02-04,15.00\n", encoding="utf-8")
    (root / "readme.txt").write_text("not a statement", encoding="utf-8")
    return root


@pytest.fixture
def env(tmp_path, statements):
    """Env for one invocation: echo backend, isolated home, one allowed root, wide terminal.

    ``HEARTH_HOME`` matters as much as the roots do: it decides where the finance ledger is
    looked for, and a test that fell through to the real ``~/.hearth`` would both be
    non-deterministic and be reaching at the operator's own data.
    """
    return {
        "COLUMNS": "200",
        "HEARTH_BACKEND": "echo",
        "HEARTH_HOME": str(tmp_path / ".hearth"),
        "HEARTH_FILE_ROOTS": str(statements),
    }


@pytest.fixture
def script(monkeypatch):
    """Install a scripted 'model' and hand the test the provider that recorded the calls."""

    def install(turns: list[str]) -> ScriptedProvider:
        provider = ScriptedProvider(turns)
        monkeypatch.setattr("hearth.cli.select_provider", lambda *a, **k: provider)
        return provider

    return install


# -- a run that answers ------------------------------------------------------------------


def test_successful_run_prints_the_answer_and_exits_zero(env, script):
    provider = script(
        [_call("list_files", pattern="*.csv"), _answer("There are 2 CSV files.")]
    )
    result = runner.invoke(app, ["agent", "count the CSV files"], env=env)

    assert result.exit_code == 0
    assert "There are 2 CSV files." in result.stdout
    assert len(provider.requests) == 2


def test_successful_run_prints_the_steps_behind_the_answer(env, script):
    """The transcript is on by default: a conclusion with no traceable steps is a claim."""
    script([_call("list_files", pattern="*.csv"), _answer("There are 2 CSV files.")])
    result = runner.invoke(app, ["agent", "count the CSV files"], env=env)

    assert result.exit_code == 0
    assert "agent steps" in result.stdout
    assert "list_files" in result.stdout
    assert "pattern='*.csv'" in result.stdout
    assert "step(s) of at most 8" in result.stdout
    # What the observation *contained* is asserted against the --json rendering instead: the
    # table crops a cell to the terminal width, so a long absolute path is not a stable
    # string to match on here.


def test_no_steps_suppresses_the_transcript_but_not_the_answer(env, script):
    script([_answer("Nothing to look up.")])
    result = runner.invoke(app, ["agent", "say hello", "--no-steps"], env=env)

    assert result.exit_code == 0
    assert "Nothing to look up." in result.stdout
    assert "agent steps" not in result.stdout


def test_full_prints_the_loops_own_transcript_instead_of_the_table(env, script):
    script([_call("list_files", pattern="*.csv"), _answer("There are 2 CSV files.")])
    result = runner.invoke(app, ["agent", "count the CSV files", "--full"], env=env)

    assert result.exit_code == 0
    assert "stopped_reason : answered" in result.stdout
    assert "agent steps" not in result.stdout  # replaces the compact table, never doubles it


def test_toolset_states_what_it_could_not_offer(env, script):
    """Degrade cleanly: fewer tools, and the reason for each absence in the output."""
    script([_answer("done")])
    result = runner.invoke(app, ["agent", "anything"], env=env)

    assert result.exit_code == 0
    assert "tools=list_files, read_file" in result.stdout
    assert "rag_search not offered" in result.stdout
    assert "finance tools not offered" in result.stdout


def test_finance_tools_appear_when_a_ledger_exists(env, script, tmp_path):
    ledger = tmp_path / ".hearth" / "finance" / "ledger.db"
    ledger.parent.mkdir(parents=True)
    ledger.write_bytes(b"")  # existence is what local_toolset keys off
    script([_answer("done")])
    result = runner.invoke(app, ["agent", "anything"], env=env)

    assert result.exit_code == 0
    assert "finance_total" in result.stdout
    assert "finance_explain" in result.stdout
    assert "finance tools not offered" not in result.stdout


def test_no_finance_suppresses_the_ledger_tools(env, script, tmp_path):
    ledger = tmp_path / ".hearth" / "finance" / "ledger.db"
    ledger.parent.mkdir(parents=True)
    ledger.write_bytes(b"")
    script([_answer("done")])
    result = runner.invoke(app, ["agent", "anything", "--no-finance"], env=env)

    assert result.exit_code == 0
    assert "finance_total" not in result.stdout
    assert "--no-finance" in result.stdout


# -- a run that stopped short ------------------------------------------------------------


def test_exhausted_budget_exits_non_zero_and_prints_no_answer(env, script):
    """The whole reason the stop reason is in the result type, carried through to a shell.

    The script *would* have answered on turn two. The iteration cap stops the run first, so
    the answer must be nowhere in the output and the exit code must not be 0 — a pipeline
    that reads this run as finished is the failure this asserts against.
    """
    provider = script(
        [_call("list_files", pattern="*.csv"), _answer("There are 2 CSV files.")]
    )
    result = runner.invoke(
        app, ["agent", "count the CSV files", "--max-iterations", "1"], env=env
    )

    assert result.exit_code != 0
    assert "max_iterations" in result.stdout
    assert "NO ANSWER" in result.stdout
    assert "There are 2 CSV files." not in result.stdout
    # The bound was enforced, not merely reported: one generation, not two.
    assert len(provider.requests) == 1


def test_stop_reason_is_named_for_every_bound(env, script):
    """A wall-clock stop must be as legible as an iteration one, and equally non-zero."""
    script([_call("list_files", pattern="*.csv"), _answer("never reached")])
    result = runner.invoke(
        app, ["agent", "count the CSV files", "--max-seconds", "0.000001"], env=env
    )

    assert result.exit_code != 0
    assert "timeout" in result.stdout or "max_iterations" in result.stdout
    assert "never reached" not in result.stdout


def test_impossible_bound_is_refused_before_anything_runs(env, script):
    provider = script([_answer("unreachable")])
    result = runner.invoke(app, ["agent", "anything", "--max-iterations", "0"], env=env)

    assert result.exit_code == 2
    assert "max_iterations must be at least 1" in result.stdout
    assert provider.requests == []


# -- --json ------------------------------------------------------------------------------


def test_json_emits_the_whole_run_and_nothing_else(env, script):
    script([_call("list_files", pattern="*.csv"), _answer("There are 2 CSV files.")])
    result = runner.invoke(app, ["agent", "count the CSV files", "--json"], env=env)

    assert result.exit_code == 0
    payload = json.loads(result.stdout)  # stdout is the document, not prose plus a document
    assert payload["completed"] is True
    assert payload["stopped_reason"] == "answered"
    assert payload["answer"] == "There are 2 CSV files."
    assert payload["task"] == "count the CSV files"
    assert payload["tools"] == ["list_files", "read_file"]
    assert payload["iterations"] == len(payload["steps"]) == 2
    assert payload["total_tokens"] == payload["prompt_tokens"] + payload["completion_tokens"]
    assert payload["budget"]["max_iterations"] == 8

    step = payload["steps"][0]
    assert step["kind"] == "tool_call"
    assert step["tool"] == "list_files"
    assert step["arguments"] == {"root": "", "pattern": "*.csv"}
    assert "january.csv" in step["observation"]
    assert step["error"] is None


def test_json_carries_the_stop_reason_and_still_exits_non_zero(env, script):
    script([_call("list_files", pattern="*.csv"), _answer("There are 2 CSV files.")])
    result = runner.invoke(
        app, ["agent", "count them", "--json", "--max-iterations", "1"], env=env
    )

    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    assert payload["completed"] is False
    assert payload["stopped_reason"] == "max_iterations"
    assert payload["answer"] is None
    assert payload["detail"]


# -- the deny-by-default file gate -------------------------------------------------------


def test_unset_file_roots_is_reported_before_the_loop_runs(env, script):
    """Deny-by-default is stated up front, naming the env var, not discovered by the agent."""
    provider = script([_answer("unreachable")])
    result = runner.invoke(
        app, ["agent", "read my statements"], env={**env, "HEARTH_FILE_ROOTS": None}
    )

    assert result.exit_code == 2
    assert "HEARTH_FILE_ROOTS" in result.stdout
    assert "Refusing to start" in result.stdout
    assert provider.requests == []  # not one token was spent finding this out


def test_file_roots_pointing_nowhere_is_the_same_refusal(env, script, tmp_path):
    """The gate asserts on the roots that resolved, not on whether the variable is set.

    A typo'd root leaves ``HEARTH_FILE_ROOTS`` set and the agent still able to read nothing;
    a check that only tested "is it configured" would report green (``CLAUDE.md`` §3).
    """
    provider = script([_answer("unreachable")])
    result = runner.invoke(
        app,
        ["agent", "read my statements"],
        env={**env, "HEARTH_FILE_ROOTS": str(tmp_path / "typo")},
    )

    assert result.exit_code == 2
    assert "HEARTH_FILE_ROOTS" in result.stdout
    assert "none of those are existing directories" in result.stdout
    assert provider.requests == []


def test_missing_roots_only_warns_when_another_tool_can_still_reach_something(
    env, script, tmp_path
):
    """With a ledger present the run is still worth starting — but the gap is stated."""
    ledger = tmp_path / ".hearth" / "finance" / "ledger.db"
    ledger.parent.mkdir(parents=True)
    ledger.write_bytes(b"")
    script([_answer("I could not read any files.")])
    result = runner.invoke(
        app, ["agent", "what is in the ledger?"], env={**env, "HEARTH_FILE_ROOTS": None}
    )

    assert result.exit_code == 0
    assert "HEARTH_FILE_ROOTS" in result.stdout
    assert "will refuse every path" in result.stdout
    assert "I could not read any files." in result.stdout


# -- vetting has no off switch -----------------------------------------------------------


def test_no_cli_flag_can_disable_tool_vetting():
    """There is no ``--unvetted``, and no parameter that could carry ``vetted_only=False``.

    Vetting is by code location (``__code__.co_filename`` inside ``src/hearth/agent``)
    precisely so a tool cannot lie about what it is. A flag that switched it off from a shell
    would hand that back to whoever writes the command line, so the surface is asserted
    directly: neither the command's own signature nor any of its rendered options mentions it.
    """
    import inspect

    from hearth import cli

    parameters = inspect.signature(cli.agent).parameters
    assert not any("vetted" in name for name in parameters)

    command = next(
        c for c in typer.main.get_command(app).commands.values() if c.name == "agent"
    )
    options = [opt for param in command.params for opt in param.opts + param.secondary_opts]
    assert not any("vetted" in opt or "unvetted" in opt for opt in options)


def test_the_agent_the_command_builds_actually_refuses_an_outside_tool(env, script, monkeypatch):
    """The outcome, not the flag: the agent this command constructs rejects foreign tool code.

    Asserting only that no ``--unvetted`` option exists would be checking the command line
    while the construction could still pass ``vetted_only=False``. So the kwargs the command
    really used are captured and replayed against a tool declared *in this file* — outside
    ``src/hearth/agent``, and therefore outside the source the no-network AST test covers.
    If vetting were ever turned off here, this build would succeed and the test would fail.
    """
    from hearth.agent import Agent, Tool, ToolParam, UnvettedToolError

    captured: dict[str, object] = {}

    class SpyAgent(Agent):
        def __init__(self, router, tools, **kwargs):
            captured["router"] = router
            captured["kwargs"] = kwargs
            super().__init__(router, tools, **kwargs)

    monkeypatch.setattr("hearth.agent.Agent", SpyAgent)
    script([_answer("done")])
    assert runner.invoke(app, ["agent", "anything"], env=env).exit_code == 0

    def outside_tool(path: str) -> str:
        return path

    foreign = Tool(
        name="outside_tool",
        description="Declared in the test file, outside hearth.agent.",
        call=outside_tool,
        params=(ToolParam(name="path", type="string", description="Any path."),),
    )
    with pytest.raises(UnvettedToolError):
        Agent(captured["router"], [foreign], **captured["kwargs"])


# -- odds and ends -----------------------------------------------------------------------


def test_empty_task_is_refused(env, script):
    provider = script([_answer("unreachable")])
    result = runner.invoke(app, ["agent", "   "], env=env)

    assert result.exit_code == 1
    assert "No task provided." in result.stdout
    assert provider.requests == []


def test_help_shows_the_budget_defaults(env):
    result = runner.invoke(app, ["agent", "--help"], env={"COLUMNS": "200"})

    assert result.exit_code == 0
    assert "--max-iterations" in result.stdout
    assert "--max-seconds" in result.stdout
    assert "--max-tokens" in result.stdout
    assert "8" in result.stdout and "180" in result.stdout and "24000" in result.stdout


def test_unknown_collection_is_refused_before_the_loop_runs(env, script):
    provider = script([_answer("unreachable")])
    result = runner.invoke(
        app, ["agent", "search my notes", "--collection", "nothing-here"], env=env
    )

    assert result.exit_code == 2
    assert "Nothing indexed in collection" in result.stdout
    assert "hearth rag ingest" in result.stdout
    assert provider.requests == []
