"""A bounded, on-device agent loop: plan, call one tool, observe, repeat, stop for a reason.

This package lets the operator hand HEARTH a task over their own local data — "which of these
statements covers March, and what did it total?" — and have a local 3B-14B model work it out
in steps, with no cloud model anywhere in the path. It reaches data only through
:class:`Tool` objects a caller passes in, and every one of the built-ins is a thin wrapper
over a gate that already exists (``HEARTH_FILE_ROOTS``, the RAG index, the finance ledger's
Decimal arithmetic), so the agent inherits those rules rather than acquiring its own.

**What it is honestly good for.** Bounded, well-specified steps: find the file, read it,
retrieve the passage, pull the total, say what it says. Three to eight steps where each one is
individually checkable.

**What it is not.** It is not long-horizon autonomy. A local model of this size does not
reliably notice that its plan stopped working, and it does not recover well when it does. So
this loop is built to fail loudly instead of wandering: hard caps on iterations, wall clock
and tokens; a run of unparseable turns terminates it; and the result type
(:class:`AgentRun`) *cannot hold an answer* unless the model actually answered, so an
exhausted budget can never be read as a conclusion. ``docs/AGENT.md`` states the ceiling in
full, including what stays in Python and why.

**No network, and it is enforced rather than asserted.** No module here imports a transport or
a subprocess, there is no shell tool and no write tool, and
``tests/test_agent_no_network.py`` walks this package's own source to keep it that way. Every
generation goes through the router with ``allow_escalation=False`` *and* the loop then checks
that the executed route actually reported the local backend — asserting on the flag we passed
would be checking our own request instead of the outcome (``CLAUDE.md`` §3).

Typical use::

    from hearth.agent import Agent, Budget, local_toolset

    agent = Agent(router, local_toolset(settings=settings, finance=store),
                  budget=Budget(max_iterations=6))
    run = agent.run("What did groceries total in March 2026?")
    print(run.transcript())      # every step, always
    answer = run.require_answer()  # raises if the run stopped short
"""

from __future__ import annotations

from .builtins import (
    DEFAULT_LIST_LIMIT,
    DEFAULT_ROW_LIMIT,
    finance_tools,
    list_files_tool,
    local_toolset,
    rag_search_tool,
    read_file_tool,
)
from .loop import (
    STOP_REASONS,
    STOPPED_ANSWERED,
    STOPPED_EGRESS_REFUSED,
    STOPPED_INVALID_OUTPUT,
    STOPPED_MAX_ITERATIONS,
    STOPPED_PROVIDER_ERROR,
    STOPPED_TIMEOUT,
    STOPPED_TOKEN_BUDGET,
    Agent,
    AgentConfigError,
    AgentError,
    AgentIncompleteError,
    AgentRun,
    Budget,
    EgressRefusedError,
    Step,
    StepKind,
    StopReason,
    UnvettedToolError,
)
from .protocol import (
    CONTRACT,
    FinalAnswer,
    ModelAction,
    ProtocolError,
    ToolCall,
    parse_action,
    render_system_prompt,
)
from .tools import (
    PARAM_TYPES,
    ParamType,
    Tool,
    ToolDefinitionError,
    ToolError,
    ToolOutcome,
    ToolParam,
    ToolRegistry,
    ToolValidationError,
    UnknownToolError,
    render_observation,
)

__all__ = [
    "CONTRACT",
    "DEFAULT_LIST_LIMIT",
    "DEFAULT_ROW_LIMIT",
    "PARAM_TYPES",
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
    "FinalAnswer",
    "ModelAction",
    "ParamType",
    "ProtocolError",
    "Step",
    "StepKind",
    "StopReason",
    "Tool",
    "ToolCall",
    "ToolDefinitionError",
    "ToolError",
    "ToolOutcome",
    "ToolParam",
    "ToolRegistry",
    "ToolValidationError",
    "UnknownToolError",
    "UnvettedToolError",
    "finance_tools",
    "list_files_tool",
    "local_toolset",
    "parse_action",
    "rag_search_tool",
    "read_file_tool",
    "render_observation",
    "render_system_prompt",
]
