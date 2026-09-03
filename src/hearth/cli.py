"""The ``hearth`` command-line entrypoint.

Phase 0/1 commands:
  * ``hearth doctor``       — environment preflight
  * ``hearth serve``        — start the OpenAI-compatible gateway
  * ``hearth run``          — one-shot local completion (``--file``, ``--intent``)
  * ``hearth agent``        — bounded, tool-using local agent over your own data (docs/AGENT.md)
  * ``hearth models …``     — registry: ``list`` / ``pull`` / ``rm`` / ``convert`` / export-coreml
  * ``hearth rag …``        — local RAG: ``ingest`` / ``query`` (Phase 3)
  * ``hearth train …``      — LoRA fine-tune → register a candidate adapter (Phase 4)
  * ``hearth adapters …``   — adapter registry: ``list`` / ``promote`` / ``retire`` (Phase 4)
  * ``hearth prereg …``     — pre-register the eval bar before training (ADR-006)
  * ``hearth plugins list`` — third-party plugins discovered via entry points (Phase 7)
  * ``hearth mcp``          — MCP server for agent offload (Phase 5, needs ``[mcp]`` extra)
  * ``hearth stats``        — token-savings / escalation rollups (Phase 2)
  * ``hearth version``      — print version
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import ensure_home, get_or_create_token, get_settings
from .doctor import all_fatal_passed, run_checks
from .providers import select_provider
from .providers.base import GenRequest, Message
from .registry import get_registry
from .router import Router

app = typer.Typer(
    name="hearth",
    help="On-device intelligence layer — a local-first model gateway for Apple Silicon.",
    no_args_is_help=True,
    add_completion=False,
)
models_app = typer.Typer(help="Model registry: list, pull, remove, and convert models.")
app.add_typer(models_app, name="models")
rag_app = typer.Typer(help="Local RAG: ingest paths into a collection and query them.")
app.add_typer(rag_app, name="rag")
adapters_app = typer.Typer(help="LoRA adapter registry: list, promote, and retire adapters.")
app.add_typer(adapters_app, name="adapters")
prereg_app = typer.Typer(help="Pre-registration: declare the eval bar before training.")
app.add_typer(prereg_app, name="prereg")
plugins_app = typer.Typer(help="Third-party plugins discovered via entry points (Phase 7).")
app.add_typer(plugins_app, name="plugins")
console = Console()

#: Characters of one agent step's observation shown in the terminal transcript. The loop has
#: already capped what the *model* saw at ``Budget.max_observation_chars`` (4 000); this is the
#: much tighter cap on what a terminal gets, so a single large read cannot bury the run it is
#: supposed to make checkable. ``--full`` prints the loop's own transcript, still capped.
AGENT_OBSERVATION_PREVIEW = 200

#: Characters of a step's rendered arguments shown in that same table.
AGENT_ARGUMENT_PREVIEW = 60


def _adapter_store():
    """Build an :class:`AdapterStore` under the current ``HEARTH_HOME``.

    Reads a fresh :class:`Settings` (not the process-cached one) so a caller/test that
    sets ``HEARTH_HOME`` for a single invocation gets an isolated store.
    """
    from .config import Settings
    from .registry import AdapterStore

    return AdapterStore(settings=Settings())


def _load_golden_set(path: Path, task: str):
    """Load a golden set from a JSONL file of ``{"prompt", "expected"}`` rows.

    An optional leading header line (``kind == hearth.dataset.header`` or
    ``hearth.golden.header``) is skipped, so a file produced by ``hearth.training.dataset``
    and a bare hand-written list both work. A ``hearth.golden.header`` may carry a
    ``version`` label, which rides along in the report; the set's *identity* is always its
    content sha, so an unversioned file is still pinnable (LEARNING_plan §3.1).
    """
    import json

    from .training.eval import GoldenExample, GoldenSet

    examples = []
    version = ""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("kind") == "hearth.dataset.header":
            continue
        if obj.get("kind") == "hearth.golden.header":
            version = str(obj.get("version", ""))
            continue
        if "prompt" not in obj or "expected" not in obj:
            raise ValueError('each golden row needs "prompt" and "expected" fields')
        examples.append(GoldenExample(prompt=obj["prompt"], expected=obj["expected"]))
    if not examples:
        raise ValueError("golden set is empty")
    return GoldenSet(task=task, examples=examples, version=version)


def _agent_payload(run: Any, tools: tuple[str, ...]) -> dict[str, Any]:
    """Render an :class:`~hearth.agent.AgentRun` as the ``--json`` document.

    ``asdict`` carries the run *whole* — every step with its arguments, observation, error,
    tokens and timings, plus the budget it actually ran under — because a scripted caller that
    is handed a summary has to trust it. The derived fields are added here rather than left to
    be recomputed: ``completed`` is the single field to branch on, and ``answer`` is ``null``
    for every stop reason but ``answered``, so a run that hit a bound cannot be read as one
    that finished.
    """
    from dataclasses import asdict

    payload = asdict(run)
    payload["completed"] = run.completed
    payload["iterations"] = run.iterations
    payload["total_tokens"] = run.total_tokens
    payload["tools"] = list(tools)
    return payload


def _agent_steps_table(run: Any) -> Table:
    """Render the run's steps: what ran, with what, what came back, and what it cost.

    Printed by default. An agent's conclusion the operator cannot trace back to the steps
    behind it is a claim, not a result — the same reason ``AgentRun.transcript()`` puts the
    evidence under the headline rather than asserting the headline is supported.
    """
    table = Table(title="agent steps", show_header=True, header_style="bold")
    table.add_column("#", justify="right")
    table.add_column("step")
    table.add_column("arguments")
    table.add_column("observation")
    table.add_column("tokens", justify="right")
    table.add_column("model/tool s", justify="right")
    for step in run.steps:
        if step.error is not None:
            observation = f"[red]{_agent_snippet(step.error, AGENT_OBSERVATION_PREVIEW)}[/red]"
        elif step.kind == "answer":
            observation = "[dim](the answer, below)[/dim]"
        else:
            observation = _agent_snippet(step.observation or "", AGENT_OBSERVATION_PREVIEW)
        arguments = ", ".join(f"{k}={v!r}" for k, v in (step.arguments or {}).items())
        table.add_row(
            str(step.index),
            step.tool or step.kind,
            _agent_snippet(arguments, AGENT_ARGUMENT_PREVIEW),
            observation,
            f"{step.prompt_tokens}+{step.completion_tokens}",
            f"{step.model_seconds:.2f}/{step.tool_seconds:.2f}",
        )
    return table


def _agent_snippet(text: str, limit: int) -> str:
    """One-line, hard-capped rendering of a step field, with the cut marked rather than silent.

    Escaped for Rich markup: a step's arguments and observations are file paths and model
    output, and ``[... truncated ...]`` — which the loop itself appends — is close enough to a
    markup tag that rendering it raw is a crash waiting for the first large file.
    """
    from rich.markup import escape

    flat = " ".join(text.split())
    if len(flat) > limit:
        flat = flat[: limit - 1] + "…"
    return escape(flat)


@app.command()
def version() -> None:
    """Print the HEARTH version."""
    console.print(f"hearth {__version__}")


@app.command()
def doctor() -> None:
    """Run environment preflight checks."""
    checks = run_checks()
    table = Table(title="hearth doctor", show_header=True, header_style="bold")
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail")
    for c in checks:
        mark = "[green]PASS[/green]" if c.ok else (
            "[red]FAIL[/red]" if c.fatal else "[yellow]WARN[/yellow]"
        )
        table.add_row(c.name, mark, c.detail)
    console.print(table)

    if not all_fatal_passed(checks):
        console.print("[red]Fatal checks failed.[/red]")
        raise typer.Exit(code=1)
    console.print("[green]Ready.[/green] (warnings are non-fatal)")


@app.command()
def serve(
    host: str = typer.Option(None, help="Bind host (default from HEARTH_HOST / 127.0.0.1)."),
    port: int = typer.Option(None, help="Bind port (default from HEARTH_PORT / 8080)."),
) -> None:
    """Start the OpenAI-compatible gateway."""
    import uvicorn

    from .gateway import create_app

    settings = get_settings()
    ensure_home(settings)
    get_or_create_token(settings)  # ensure a token exists for bearer auth
    provider = select_provider(settings)

    bind_host = host or settings.host
    bind_port = port or settings.port
    console.print(
        f"[bold]HEARTH[/bold] {__version__} — backend=[cyan]{provider.name}[/cyan] "
        f"model=[cyan]{get_registry().default_id}[/cyan]"
    )
    console.print(f"Serving on http://{bind_host}:{bind_port}  (OpenAI-compatible /v1)")
    uvicorn.run(create_app(provider=provider, settings=settings), host=bind_host, port=bind_port)


@app.command()
def run(
    prompt: str = typer.Argument(None, help="Prompt text. Omit to read from stdin."),
    max_tokens: int = typer.Option(512, help="Max tokens to generate."),
    file: Path = typer.Option(
        None, "--file", help="Read the prompt from this file instead of the argument."
    ),
    intent: str = typer.Option(
        None, "--intent", help="Routing intent hint (recorded; used by the router in Phase 2)."
    ),
) -> None:
    """Run a one-shot local completion and print the result."""
    if file is not None:
        text = file.read_text()
    elif prompt is not None:
        text = prompt
    else:
        text = sys.stdin.read()
    if not text.strip():
        console.print("[red]No prompt provided.[/red]")
        raise typer.Exit(code=1)

    settings = get_settings()
    provider = select_provider(settings)
    # `intent` is recorded here for parity with the API's hearth.intent hint; the router
    # that consumes it arrives in Phase 2. Surface it so `--intent` is observably wired.
    if intent:
        console.print(f"[dim]intent={intent}[/dim]")
    router = Router(local_provider=provider)
    routed = router.route(
        GenRequest(
            messages=[Message(role="user", content=text)],
            model=get_registry().default_id,
            max_tokens=max_tokens,
        ),
        intent=intent,
        # A one-shot CLI stays local unless a daemon/policy escalates; keep it hard-local
        # so `hearth run` never makes a surprise remote call from a script.
        allow_escalation=False,
    )
    console.print(routed.result.text, markup=False, highlight=False)


@app.command()
def agent(
    task: str = typer.Argument(None, help="What the agent should do. Omit to read from stdin."),
    collection: str = typer.Option(
        None, "--collection", help="Offer rag_search, pinned to this indexed RAG collection."
    ),
    finance: bool = typer.Option(
        True,
        "--finance/--no-finance",
        help="Offer the read-only ledger tools when a finance ledger exists.",
    ),
    max_iterations: int = typer.Option(
        8, "--max-iterations", help="Hard cap on model turns; hitting it stops the run."
    ),
    max_seconds: float = typer.Option(
        180.0, "--max-seconds", help="Hard wall-clock cap in seconds; hitting it stops the run."
    ),
    max_tokens: int = typer.Option(
        24_000,
        "--max-tokens",
        help="Hard cap on prompt+completion tokens across every step of the run.",
    ),
    steps: bool = typer.Option(
        True, "--steps/--no-steps", help="Print the step-by-step transcript before the answer."
    ),
    full: bool = typer.Option(
        False, "--full", help="Print the loop's own full transcript (raw model output per step)."
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Emit the whole run as JSON instead of prose (for scripting)."
    ),
) -> None:
    """Run a bounded, tool-using local agent over your own data (docs/AGENT.md).

    The agent plans, calls **one** tool, observes the result, and repeats until it answers or
    hits a bound — every generation local, every step in the transcript. Unlike
    ``hearth run`` (and unlike ``/v1/chat/completions``), it can actually *read* the files it
    talks about, so "how many CSVs are under statements/" is answered from the filesystem
    rather than from the model's imagination:

        HEARTH_FILE_ROOTS=~/statements hearth agent "how many CSV files are there?"

    **The exit code is the stop reason.** ``0`` only when the model answered; ``1`` when the
    run stopped at a bound or a failure (and then there is no answer to print — ``AgentRun``
    cannot hold one); ``2`` when the agent was never started — an impossible bound, or a
    toolset that could reach nothing. A run that exhausted its budget must never read like a
    completed one, in a terminal or in a pipeline.

    Tools are assembled from the collaborators that are actually present — the file tools
    always, ``rag_search`` with ``--collection``, the ledger tools when a finance ledger
    exists — and what was assembled is printed. There is deliberately **no flag to disable
    tool vetting**: an agent here runs only tools whose code lives in ``hearth.agent``, which
    is the source the no-network AST test covers, and a flag that switched that off from a
    shell would hand back the one thing a tool cannot lie about. A caller who needs their own
    tool writes Python against the library (``docs/AGENT.md`` §4, §5.1).
    """
    from .agent import Agent, AgentConfigError, Budget, local_toolset
    from .config import Settings
    from .mcp.files import allowed_roots

    text = task if task is not None else sys.stdin.read()
    if not text.strip():
        console.print("[red]No task provided.[/red]")
        raise typer.Exit(code=1)

    # A fresh Settings() (not the lru_cached get_settings) so HEARTH_FILE_ROOTS, HEARTH_HOME
    # and HEARTH_BACKEND are read per invocation — the same reason `hearth eval` does it.
    settings = Settings()
    notes: list[str] = []

    rag = None
    if collection:
        from .memory import RagIndex, select_embedder, select_vector_store

        rag = RagIndex(
            embedder=select_embedder(settings), store=select_vector_store(settings)
        )
        if rag.store.count(collection) == 0:
            console.print(
                f"[red]Nothing indexed in collection[/red] {collection!r}. rag_search would "
                "return an empty result on every call and the agent would spend its whole "
                "budget discovering that.\n"
                f"  Ingest first:  [cyan]hearth rag ingest <path> --collection {collection}"
                "[/cyan]"
            )
            raise typer.Exit(code=2)
    else:
        notes.append("rag_search not offered — no --collection named")

    store = None
    if finance:
        from .finance.store import FinanceStore

        candidate = FinanceStore(settings=settings)
        if candidate.path.exists():
            store = candidate
        else:
            notes.append(f"finance tools not offered — no ledger at {candidate.path}")
    else:
        notes.append("finance tools not offered — --no-finance")

    # The file tools are deny-by-default, so an agent asked to read a directory with no roots
    # configured burns its entire budget discovering it may read nothing. Check the *outcome*
    # — the roots that actually resolved to existing directories — rather than whether the
    # variable is set, so a typo'd root is caught by the same gate (CLAUDE.md §3).
    roots = allowed_roots(settings)
    if not roots:
        console.print(
            "[red]No readable file roots.[/red] read_file and list_files are deny-by-default "
            "and will refuse every path: "
            + (
                f"HEARTH_FILE_ROOTS is set to {settings.file_roots!r}, but none of those are "
                "existing directories."
                if settings.file_roots.strip()
                else "HEARTH_FILE_ROOTS is unset, and there is no implicit root — not the "
                "current directory, not $HOME."
            )
            + "\n  Set it for this run:  "
            "[cyan]HEARTH_FILE_ROOTS=~/statements hearth agent \"…\"[/cyan]"
        )
        if rag is None and store is None:
            console.print(
                "[red]Refusing to start:[/red] with no file roots, no --collection and no "
                "ledger, this agent has nothing it can reach — it could only assert."
            )
            raise typer.Exit(code=2)
        notes.append("read_file/list_files will refuse every path — no roots resolved")

    tools = local_toolset(settings=settings, rag=rag, finance=store, collection=collection)
    provider = select_provider(settings)
    model_id = get_registry().default_id
    try:
        budget = Budget(
            max_iterations=max_iterations,
            max_seconds=max_seconds,
            max_total_tokens=max_tokens,
        )
        # `vetted_only` is left at its default of True and is not plumbed to a flag; see the
        # docstring. The router is entered with allow_escalation=False by the loop itself,
        # which then verifies the executed route actually reported the local backend.
        runner = Agent(Router(local_provider=provider), tools, budget=budget, model=model_id)
    except AgentConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from None

    if not as_json:
        console.print(
            f"[bold]HEARTH agent[/bold] — backend=[cyan]{provider.name}[/cyan] "
            f"model=[cyan]{model_id}[/cyan] tools=[cyan]{', '.join(tools.names)}[/cyan]"
        )
        for note in notes:
            console.print(f"[dim]{note}[/dim]")

    result = runner.run(text)

    if as_json:
        console.print_json(data=_agent_payload(result, tools.names), default=str, sort_keys=True)
    else:
        if full:
            console.print(result.transcript(), markup=False, highlight=False)
        elif steps:
            console.print(_agent_steps_table(result))
        console.print(
            f"[dim]{result.iterations} step(s) of at most {budget.max_iterations}, "
            f"{result.total_tokens} token(s) of at most {budget.max_total_tokens}, "
            f"{result.elapsed_seconds:.2f}s of at most {budget.max_seconds:g}s[/dim]"
        )
        if result.completed:
            console.print("\n[bold]answer[/bold]")
            console.print(result.require_answer(), markup=False, highlight=False)
        else:
            console.print(
                f"\n[red]NO ANSWER — the run stopped because "
                f"{result.stopped_reason!r}.[/red]"
            )
            if result.detail:
                console.print(f"[red]{result.detail}[/red]")
            console.print(
                "[yellow]The steps above are a partial trace, not a result.[/yellow]"
            )

    # One exit point for the verdict, so `--json` and the prose rendering cannot disagree
    # about whether the run finished.
    if not result.completed:
        raise typer.Exit(code=1)


@app.command()
def mcp() -> None:
    """Launch the HEARTH MCP server (stdio) so agents like Claude Code can offload subtasks.

    Registers HEARTH as an MCP tool provider (summarize/classify/extract/draft/rag_query),
    each running on the local model with escalation disabled — the delegated work never
    spends the agent's frontier budget (ADR-010, docs/INTEGRATION.md). Requires the ``mcp``
    extra; the tool logic itself lives in :mod:`hearth.mcp.tools` and needs no extras.
    """
    try:
        from .mcp import server

        server.run()
    except ModuleNotFoundError as exc:
        # The `mcp` SDK is an optional extra (server.py imports it lazily at run time, so
        # the failure surfaces here rather than at import). Fail loudly with the fix instead
        # of a bare traceback, and exit non-zero so callers/CI notice.
        if "mcp" not in str(exc):
            raise
        console.print(
            "[red]The MCP server requires the 'mcp' extra.[/red]\n"
            "Install it with:  [cyan]uv sync --extra mcp[/cyan]"
        )
        raise typer.Exit(code=1) from None


@app.command()
def stats(
    since: str = typer.Option(
        None, "--since", help="Rollup window, e.g. 7d / 24h / 30m (default: all)."
    ),
) -> None:
    """Show token-savings and escalation rollups (ARCHITECTURE §8).

    Phase 2 keeps metrics in-memory per process, so a fresh CLI invocation reports an
    empty store; the numbers accumulate within a running ``hearth serve`` daemon. A future
    phase persists records to JSONL so the CLI can roll up across restarts.
    """
    from .gateway.app import _parse_since
    from .observability import get_metrics

    roll = get_metrics().rollup(since_s=_parse_since(since))
    table = Table(title="hearth stats", show_header=True, header_style="bold")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("requests", str(roll["requests"]))
    table.add_row("estimated frontier tokens saved", str(roll["estimated_frontier_tokens_saved"]))
    table.add_row("escalations", str(roll["escalations"]))
    table.add_row("escalation rate", f"{roll['escalation_rate']:.2%}")
    backend_mix = ", ".join(f"{k}={v}" for k, v in roll["backend_mix"].items())
    class_mix = ", ".join(f"{k}={v}" for k, v in roll["class_mix"].items())
    table.add_row("backend mix", backend_mix or "-")
    table.add_row("class mix", class_mix or "-")
    table.add_row("latency p50 (ms)", f"{roll['latency_ms']['p50']:g}")
    table.add_row("latency p95 (ms)", f"{roll['latency_ms']['p95']:g}")
    console.print(table)


@models_app.command("list")
def models_list() -> None:
    """List models in the registry."""
    registry = get_registry()
    table = Table(title="hearth models", show_header=True, header_style="bold")
    table.add_column("id")
    table.add_column("backend")
    table.add_column("quant")
    table.add_column("context", justify="right")
    table.add_column("ram_gb", justify="right")
    table.add_column("capabilities")
    default_id = registry.default_id
    for e in registry.list():
        marker = " [green](default)[/green]" if e.id == default_id else ""
        table.add_row(
            e.id + marker,
            e.backend,
            e.quant,
            str(e.context),
            f"{e.ram_gb:g}",
            ",".join(e.capabilities),
        )
    console.print(table)


@models_app.command("pull")
def models_pull(model_id: str = typer.Argument(..., help="Registry model id to download.")) -> None:
    """Download a model's weights from its registry `source` repo.

    Respects the ``HF_ENDPOINT`` mirror and ``HF_HUB_OFFLINE`` env vars — hosts are never
    hardcoded, so a locked-down mirror works with no code change.
    """
    registry = get_registry()
    entry = registry.get(model_id)
    if entry is None:
        console.print(f"[red]Unknown model id:[/red] {model_id}")
        raise typer.Exit(code=1)
    if not entry.source:
        console.print(f"[yellow]{model_id} has no downloadable source (nothing to pull).[/yellow]")
        return

    settings = get_settings()
    ensure_home(settings)
    from huggingface_hub import snapshot_download  # deferred; keeps import cost off other cmds

    console.print(f"Pulling [cyan]{entry.source}[/cyan] → {settings.models_dir} …")
    path = snapshot_download(repo_id=entry.source, cache_dir=str(settings.models_dir))
    console.print(f"[green]Done.[/green] {path}")


@models_app.command("rm")
def models_rm(model_id: str = typer.Argument(..., help="Registry model id to remove.")) -> None:
    """Remove a model's cached weights from the local models dir."""
    import shutil

    registry = get_registry()
    entry = registry.get(model_id)
    if entry is None or not entry.source:
        console.print(f"[red]Unknown or non-downloadable model id:[/red] {model_id}")
        raise typer.Exit(code=1)

    settings = get_settings()
    # huggingface_hub lays caches out as models--<org>--<name> under the cache dir.
    cache_name = "models--" + entry.source.replace("/", "--")
    target = settings.models_dir / cache_name
    if not target.exists():
        console.print(f"[yellow]Not cached locally:[/yellow] {target}")
        raise typer.Exit(code=1)
    shutil.rmtree(target)
    console.print(f"[green]Removed[/green] {target}")


@models_app.command("convert")
def models_convert(
    source: str = typer.Option(
        ..., "--source", help="Source checkpoint: HF repo id or local path to convert."
    ),
    out: Path = typer.Option(..., "--out", help="Output dir for the MLX-format model."),
    quantize: bool = typer.Option(
        True, "--quantize/--no-quantize", help="Quantize the model (else format-convert only)."
    ),
    q_bits: int = typer.Option(4, "--q-bits", help="Quantization bit width (2/3/4/6/8)."),
    q_group_size: int = typer.Option(64, "--q-group-size", help="Quantization group size."),
) -> None:
    """Quantize/convert a checkpoint into an MLX-servable model (ARCHITECTURE §5, Phase 7).

    Real conversion needs the ``[mlx]`` extra, source weights, and (for cached inputs)
    offline HF:

        uv sync --extra mlx
        HF_HUB_OFFLINE=1 hearth models convert --source <id> --out ~/.hearth/models/<id> -q 4

    Add the produced model to ``config/models.yaml`` to serve it (registry is data, §5).
    """
    from .convert import ConvertConfig, ConvertUnavailableError
    from .convert import convert as run_convert

    config = ConvertConfig(
        source=source, output_dir=out, quantize=quantize, q_bits=q_bits, q_group_size=q_group_size
    )
    try:
        config.validate()
    except ValueError as exc:
        console.print(f"[red]Invalid conversion config:[/red] {exc}")
        raise typer.Exit(code=1) from None

    label = f"{q_bits}-bit" if quantize else "no quantization"
    console.print(f"Converting [cyan]{source}[/cyan] ({label}) -> {out} …")
    try:
        outcome = run_convert(config)
    except ConvertUnavailableError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    console.print(f"[green]Converted.[/green] model -> {outcome.output_dir}")


@models_app.command("export-coreml")
def models_export_coreml(
    source: str = typer.Option(
        ..., "--source", help="Source checkpoint: HF repo id or local path to export."
    ),
    out: Path = typer.Option(..., "--out", help="Output .mlpackage dir for the Core ML model."),
    compute_units: str = typer.Option(
        "cpuAndNeuralEngine",
        "--compute-units",
        help="Runtime placement: all / cpuAndNeuralEngine / cpuAndGPU / cpuOnly.",
    ),
    precision: str = typer.Option(
        "float16", "--precision", help="Weight precision: float16 / float32 / int8."
    ),
    max_seq_len: int = typer.Option(
        512, "--max-seq-len", help="Fixed sequence length to trace/export at (Core ML is static)."
    ),
    stateful: bool = typer.Option(
        False,
        "--stateful/--no-stateful",
        help="Export the stateful KV-cache model (Approach B, O(1)/token, CPU-only, Qwen2; "
        "ADR-011). Default off keeps Approach A (padded prefill, ANE) as the shipped path.",
    ),
) -> None:
    """Export a checkpoint to a Core ML ``.mlpackage`` for the on-device Swift path (Phase 6).

    The produced ``.mlpackage`` is loaded by the Swift ``CoreMLProvider`` (see swift/OFFLINE.md)
    for fully-offline, ANE-accelerated inference. Real export needs the ``[coreml]`` extra,
    source weights, and (for cached inputs) offline HF:

        uv sync --extra coreml
        HF_HUB_OFFLINE=1 hearth models export-coreml --source <id> --out ~/.hearth/coreml/<id>
    """
    from .coreml import CoreMLExportConfig, CoreMLExportUnavailableError
    from .coreml import export as run_export

    config = CoreMLExportConfig(
        source=source,
        output_dir=out,
        compute_units=compute_units,
        precision=precision,
        max_seq_len=max_seq_len,
        stateful=stateful,
    )
    try:
        config.validate()
    except ValueError as exc:
        console.print(f"[red]Invalid Core ML export config:[/red] {exc}")
        raise typer.Exit(code=1) from None

    approach = "stateful KV-cache (Approach B)" if stateful else "padded prefill (Approach A)"
    console.print(
        f"Exporting [cyan]{source}[/cyan] to Core ML "
        f"({precision}, {compute_units}, seq={max_seq_len}, {approach}) -> {out} …"
    )
    try:
        outcome = run_export(config)
    except CoreMLExportUnavailableError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    console.print(f"[green]Exported.[/green] model -> {outcome.output_dir}")
    console.print(f"  sidecar -> {outcome.manifest_path}")
    for path in outcome.tokenizer_paths:
        console.print(f"  tokenizer -> {path}")


@rag_app.command("ingest")
def rag_ingest(
    path: Path = typer.Argument(..., help="File or directory to ingest."),
    collection: str = typer.Option("default", "--collection", help="Target collection name."),
    size: int = typer.Option(800, "--size", help="Chunk size in characters."),
    overlap: int = typer.Option(100, "--overlap", help="Chunk overlap in characters."),
) -> None:
    """Chunk, embed, and store a path into a local RAG collection (ARCHITECTURE §6)."""
    from .memory import RagIndex

    index = RagIndex()
    console.print(
        f"Ingesting [cyan]{path}[/cyan] → collection [cyan]{collection}[/cyan] "
        f"(embedder=[cyan]{index.embedder.name}[/cyan]) …"
    )
    result = index.ingest(path, collection, size=size, overlap=overlap)
    console.print(
        f"[green]Done.[/green] {result.files} file(s), {result.chunks} chunk(s) "
        f"in collection [cyan]{result.collection}[/cyan]."
    )


@rag_app.command("query")
def rag_query(
    query: str = typer.Argument(..., help="Query text."),
    collection: str = typer.Option("default", "--collection", help="Collection to search."),
    k: int = typer.Option(6, "--k", help="Number of chunks to retrieve."),
    answer: bool = typer.Option(
        False, "--answer", help="Answer with the local model grounded in retrieved chunks."
    ),
) -> None:
    """Retrieve the top-k chunks for a query; optionally answer locally (ARCHITECTURE §6)."""
    from .memory import RagIndex

    provider = select_provider(get_settings())
    index = RagIndex(router=Router(local_provider=provider))
    result = index.query(collection, query, k=k, answer=answer)

    if not result.chunks:
        console.print(f"[yellow]No chunks in collection[/yellow] {collection!r}.")
        raise typer.Exit(code=0)

    table = Table(title=f"rag query — {collection}", show_header=True, header_style="bold")
    table.add_column("score", justify="right")
    table.add_column("source")
    table.add_column("text")
    for c in result.chunks:
        snippet = c.text.strip().replace("\n", " ")
        if len(snippet) > 120:
            snippet = snippet[:117] + "…"
        table.add_row(f"{c.score:.3f}", c.source, snippet)
    console.print(table)

    if result.answer is not None:
        console.print("\n[bold]answer[/bold]")
        console.print(result.answer, markup=False, highlight=False)


@app.command()
def train(
    task: str = typer.Option(..., "--task", help="Task class the adapter targets (e.g. extract)."),
    base: str = typer.Option(..., "--base", help="Base model id to fine-tune (LoRA)."),
    data: Path = typer.Option(..., "--data", help="Dataset JSONL (see hearth.training.dataset)."),
    out: Path = typer.Option(
        None, "--out", help="Output dir for the run (default: ~/.hearth/train/<run-id>)."
    ),
    iters: int = typer.Option(200, "--iters", help="Training iterations."),
    register: bool = typer.Option(
        True, "--register/--no-register", help="Register the result as a candidate adapter."
    ),
) -> None:
    """Train a LoRA adapter and register it as a *candidate* (ARCHITECTURE §7, ADR-006).

    Real training needs the ``[mlx]`` extra, a cached base model, and offline HF:

        uv sync --extra mlx
        HF_HUB_OFFLINE=1 hearth train --task extract --base <id> --data data.jsonl

    Training is eval-gated: a candidate must beat the incumbent on a golden set before it
    can be promoted (``hearth adapters promote``). This command only *produces a
    candidate*; promotion is a separate, deliberate step.
    """
    from datetime import UTC, datetime

    from .config import Settings
    from .registry import AdapterError
    from .training import LoRAConfig, load_dataset
    from .training import train as run_train

    try:
        dataset = load_dataset(data)
    except Exception as exc:  # dataset validation errors -> clean message, non-zero exit
        console.print(f"[red]Dataset error:[/red] {exc}")
        raise typer.Exit(code=1) from None

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = out or (Settings().home / "train" / run_id)
    config = LoRAConfig(
        base_model=base, task=task, dataset=dataset, output_dir=out_dir, iters=iters
    )
    console.print(
        f"Training [cyan]{task}[/cyan] adapter on [cyan]{base}[/cyan] "
        f"({len(dataset)} records) -> {out_dir}"
    )
    try:
        outcome = run_train(config, train_run_id=run_id)
    except RuntimeError as exc:
        # The real runner raises with the fix hint when the [mlx] extra is missing.
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    console.print(f"[green]Trained.[/green] adapter -> {outcome.adapter_path}")
    if not register:
        return
    adapter_id = f"{task}-{run_id}"
    try:
        _adapter_store().register(
            adapter_id,
            base_model=base,
            task=task,
            train_run_id=run_id,
            adapter_path=str(outcome.adapter_path),
        )
    except AdapterError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    console.print(
        f"Registered candidate [cyan]{adapter_id}[/cyan]. "
        "Eval it, then [bold]hearth adapters promote[/bold] to serve it."
    )


@app.command("eval")
def eval_adapter(
    adapter_id: str = typer.Argument(..., help="Candidate adapter id to score."),
    golden: Path = typer.Option(
        ..., "--golden", help='Golden set JSONL ({"prompt","expected"} per line).'
    ),
    metric: str = typer.Option("f1", "--metric", help="Objective metric: 'f1' or 'exact'."),
    system: str = typer.Option(
        None, "--system", help="Optional system prompt sent with every example."
    ),
    max_tokens: int = typer.Option(64, "--max-tokens", help="Max tokens per generation."),
    temperature: float = typer.Option(
        0.0,
        "--temperature",
        help="Decode temperature. 0.0 (greedy): a re-rollable score is not a measurement.",
    ),
    allow_sampling: bool = typer.Option(
        False, "--allow-sampling", help="Permit --temperature > 0 (scores become re-rollable)."
    ),
    base: str = typer.Option(
        None, "--base", help="Base model id (default: the adapter's recorded base_model)."
    ),
    alpha: float = typer.Option(
        0.05, "--alpha", help="Significance level (a --prereg overrides it)."
    ),
    margin: float = typer.Option(
        0.0, "--margin", help="Minimum effect the candidate must exceed (a --prereg overrides it)."
    ),
    min_n: int = typer.Option(
        30, "--min-n", help="Minimum golden-set size the gate licenses (--prereg overrides it)."
    ),
    prereg: Path = typer.Option(
        None, "--prereg", help="Pre-registration YAML: required by --promote, and its bar wins."
    ),
    report_json: Path = typer.Option(
        None, "--report-json", help="Write the reports + gate result here (feeds promote)."
    ),
    recheck: bool = typer.Option(
        False,
        "--check-determinism",
        help="Re-generate a few golden prompts and refuse if the answers differ.",
    ),
    promote: bool = typer.Option(
        False, "--promote", help="Promote the candidate if the gate passes (needs --prereg)."
    ),
) -> None:
    """Score a candidate adapter against a golden set and (optionally) promote it (ADR-006).

    Wires the candidate through the provider's per-request adapter slot (``GenRequest.adapter``),
    scores it with the objective metric at ``temperature=0`` and compares it against the
    incumbent — **and when no adapter is promoted for the task, the incumbent is the base
    model**, never a bare "any score above zero" (LEARNING_plan F2). The comparison is paired
    over the per-example vectors and must clear ``alpha``; the candidate must also beat the
    empty / majority-label / copy-input baselines. ``--promote`` additionally requires a
    ``--prereg`` that is git-committed and matches this run.

    Real scoring needs the MLX backend (``HEARTH_BACKEND=mlx``) + a cached base model; the
    echo backend runs the plumbing offline but won't produce meaningful scores.
    """
    import json as _json
    from datetime import UTC, datetime

    from .config import Settings
    from .registry import AdapterError
    from .training.eval import (
        EvalConfig,
        GateProvenanceError,
        baseline_reports,
        check_determinism,
        evaluate_gate,
        score_candidate,
    )
    from .training.prereg import PreRegError, load_prereg, verify_committed

    if temperature > 0.0 and not allow_sampling:
        console.print(
            "[red]Refusing to score at temperature > 0:[/red] the gate would be re-rollable. "
            "Use --temperature 0 (default), or --allow-sampling to measure anyway."
        )
        raise typer.Exit(code=1)

    store = _adapter_store()
    entry = store.get(adapter_id)
    if entry is None:
        console.print(f"[red]Unknown adapter:[/red] {adapter_id!r}")
        raise typer.Exit(code=1)
    try:
        candidate_path = store.resolve_path(adapter_id, allow_candidate=True)
    except AdapterError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    try:
        golden_set = _load_golden_set(golden, task=entry.task)
    except (OSError, ValueError) as exc:
        console.print(f"[red]Golden set error:[/red] {exc}")
        raise typer.Exit(code=1) from None

    registration = None
    if prereg is not None:
        try:
            registration = load_prereg(prereg)
        except PreRegError as exc:
            console.print(f"[red]Pre-registration error:[/red] {exc}")
            raise typer.Exit(code=1) from None

    base_model = base or entry.base_model
    # Fresh Settings() (not the lru_cached get_settings) so HEARTH_BACKEND is read per call.
    provider = select_provider(Settings())
    config = EvalConfig.for_system(system, temperature=temperature, max_tokens=max_tokens)
    measured_at = datetime.now(tz=UTC).isoformat(timespec="seconds")

    def _generate_with(adapter_path: str | None):
        def _gen(prompt: str) -> str:
            messages = []
            if system:
                messages.append(Message(role="system", content=system))
            messages.append(Message(role="user", content=prompt))
            req = GenRequest(
                messages=messages,
                model=base_model,
                max_tokens=max_tokens,
                temperature=temperature,
                adapter=adapter_path,
            )
            return provider.generate(req).text

        return _gen

    def _score(adapter_path: str | None, model_id: str):
        return score_candidate(
            golden_set,
            _generate_with(adapter_path),
            metric=metric,
            model_id=model_id,
            config=config,
            measured_at=measured_at,
        )

    try:
        candidate = _score(candidate_path, f"{base_model}+{adapter_id}")
    except ValueError as exc:  # unknown metric
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    if recheck:
        drift = check_determinism(golden_set, _generate_with(candidate_path))
        if drift:
            console.print(
                f"[red]Non-deterministic generation:[/red] {len(drift)} prompt(s) produced a "
                "different answer on a second pass — this score cannot gate a promotion."
            )
            raise typer.Exit(code=1)

    # The incumbent. No promoted adapter for the task does NOT mean "anything wins": the
    # base model becomes the incumbent and has to be beaten (LEARNING_plan F2).
    incumbent_entry = store.promoted_for(entry.task)
    if incumbent_entry is not None and incumbent_entry.id != adapter_id:
        incumbent_id = incumbent_entry.id
        incumbent_role = "incumbent"
        incumbent = _score(store.resolve_path(incumbent_id), f"{base_model}+{incumbent_id}")
    else:
        incumbent_id = base_model
        incumbent_role = "base"
        incumbent = _score(None, base_model)

    baselines = baseline_reports(
        golden_set, metric=metric, config=config, measured_at=measured_at
    )
    test = "auto"
    if registration is not None:
        problems = registration.mismatches(candidate)
        if problems:
            console.print(
                "[red]This run is not the registered experiment:[/red] " + "; ".join(problems)
            )
            raise typer.Exit(code=1)
        alpha, margin, min_n, test = (
            registration.alpha,
            registration.min_effect,
            registration.min_n,
            registration.test,
        )
        missing = [b for b in registration.must_beat_baselines if b not in baselines]
        if missing:
            console.print(
                "[red]Pre-registered baseline(s) not available:[/red] " + ", ".join(missing)
            )
            raise typer.Exit(code=1)
        baselines = {b: baselines[b] for b in registration.must_beat_baselines}

    try:
        gate = evaluate_gate(
            candidate,
            incumbent,
            incumbent_role=incumbent_role,
            baselines=baselines,
            margin=margin,
            alpha=alpha,
            min_n=min_n,
            test=test,
            candidate_id=adapter_id,
            incumbent_id=incumbent_id,
        )
    except (GateProvenanceError, ValueError) as exc:
        console.print(f"[red]Gate refused to compare:[/red] {exc}")
        raise typer.Exit(code=1) from None

    table = Table(title=f"hearth eval — {entry.task}", show_header=True, header_style="bold")
    table.add_column("adapter")
    table.add_column("role")
    table.add_column(f"{candidate.metric} score")
    table.add_row(adapter_id, "candidate", f"{candidate.score:.4f}")
    table.add_row(incumbent_id, incumbent_role, f"{incumbent.score:.4f}")
    for name, report in sorted(baselines.items()):
        table.add_row("—", f"baseline:{name}", f"{report.score:.4f}")
    console.print(table)

    stat = f"{gate.test} p={gate.p_value:.4f}" if gate.p_value is not None else gate.test
    if gate.test == "mcnemar_exact":
        stat += f" (b={gate.b}, c={gate.c})"
    elif gate.ci_low is not None:
        stat += f" (ci {gate.ci_low:+.4f}..{gate.ci_high:+.4f})"
    console.print(
        f"gate: [{'green' if gate.passed else 'red'}]{'PASS' if gate.passed else 'FAIL'}[/] "
        f"n={gate.n} alpha={gate.alpha:g} {stat}"
    )
    if not gate.passed:
        for reason in gate.reasons:
            console.print(f"  [yellow]·[/yellow] {reason}")
    console.print(
        f"[dim]golden_sha={candidate.golden_sha[:12]} "
        f"config={candidate.config_fingerprint}[/dim]"
    )

    if report_json is not None:
        payload = {
            "candidate": candidate.to_json(),
            "incumbent": incumbent.to_json(),
            "incumbent_role": incumbent_role,
            "incumbent_id": incumbent_id,
            "candidate_id": adapter_id,
            "baselines": {k: v.to_json() for k, v in baselines.items()},
            "gate": gate.as_proof(),
        }
        Path(report_json).parent.mkdir(parents=True, exist_ok=True)
        Path(report_json).write_text(
            _json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        console.print(f"[dim]report written to {report_json}[/dim]")

    if not promote:
        if gate.passed:
            console.print(
                "[dim]promote with:[/dim] hearth eval "
                f"{adapter_id} --golden {golden} --prereg <committed prereg> --promote"
            )
        return

    if registration is None:
        console.print(
            "[red]Promotion refused:[/red] --promote requires --prereg. The bar has to be "
            "declared and committed before the measurement (docs/LEARNING_plan.md §3.4); "
            "scaffold one with `hearth prereg init`."
        )
        raise typer.Exit(code=1)
    status = verify_committed(registration.path)
    if not status.committed:
        console.print(f"[red]Promotion refused:[/red] {status.reason}")
        raise typer.Exit(code=1)
    if not gate.passed:
        console.print(f"[red]Promotion refused:[/red] {gate.reason}")
        raise typer.Exit(code=1)

    proof = dict(registration.as_proof())
    proof["prereg_committed"] = True
    proof["prereg_commit"] = status.commit
    proof["measured_at"] = candidate.measured_at
    try:
        store.promote(adapter_id, gate=gate, proof=proof)
    except AdapterError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    console.print(
        f"[green]Promoted[/green] {adapter_id} (gate passed, candidate={candidate.score:.4f}, "
        f"p={gate.p_value:.4f})."
    )


@prereg_app.command("init")
def prereg_init(
    task: str = typer.Option(..., "--task", help="Task class the adapter serves."),
    golden: Path = typer.Option(..., "--golden", help="Golden set JSONL to pin by content sha."),
    out: Path = typer.Option(None, "--out", help="Write here instead of printing to stdout."),
    metric: str = typer.Option("exact", "--metric", help="Objective metric: 'exact' or 'f1'."),
    max_tokens: int = typer.Option(64, "--max-tokens", help="Registered generation max tokens."),
    system: str = typer.Option(None, "--system", help="System prompt (hashed into the config)."),
    alpha: float = typer.Option(0.05, "--alpha", help="Registered significance level."),
    min_effect: float = typer.Option(0.0, "--min-effect", help="Minimum lift that would count."),
    min_n: int = typer.Option(30, "--min-n", help="Minimum golden-set size."),
) -> None:
    """Scaffold a pre-registration for a golden set — then fill in the prose and commit it.

    The generated file pins the golden set by content sha and the decode parameters by
    fingerprint, so the harness can later prove the run it gated was the run that was
    registered. The hypothesis / stopping rule / kill condition are left blank on purpose:
    a bar written by the tool is not a pre-registration.
    """
    from .training.prereg import template

    try:
        golden_set = _load_golden_set(golden, task=task)
    except (OSError, ValueError) as exc:
        console.print(f"[red]Golden set error:[/red] {exc}")
        raise typer.Exit(code=1) from None

    text = template(
        task=task,
        golden_sha=golden_set.sha,
        golden_version=golden_set.version,
        n=len(golden_set),
        metric=metric,
        max_tokens=max_tokens,
        system=system,
        alpha=alpha,
        min_effect=min_effect,
        min_n=min_n,
    )
    if out is None:
        console.print(text)
        return
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(text, encoding="utf-8")
    console.print(
        f"Wrote [cyan]{out}[/cyan]. Fill in hypothesis/stopping_rule/kill_condition, then "
        "[bold]git commit[/bold] it — an uncommitted prereg cannot gate a promotion."
    )


@prereg_app.command("check")
def prereg_check(
    path: Path = typer.Argument(..., help="Pre-registration YAML to validate."),
    golden: Path = typer.Option(
        None, "--golden", help="Also check this golden set still hashes to the registered sha."
    ),
) -> None:
    """Validate a pre-registration and report whether git has it committed and unmodified."""
    from .training.prereg import PreRegError, load_prereg, verify_committed

    try:
        registration = load_prereg(path)
    except PreRegError as exc:
        console.print(f"[red]Pre-registration error:[/red] {exc}")
        raise typer.Exit(code=1) from None

    table = Table(title=f"prereg — {path}", show_header=True, header_style="bold")
    table.add_column("field")
    table.add_column("value")
    table.add_row("task", registration.task)
    table.add_row("metric", registration.metric)
    table.add_row("golden_sha", registration.golden_sha[:16])
    table.add_row("alpha", f"{registration.alpha:g}")
    table.add_row("min_effect", f"{registration.min_effect:g}")
    table.add_row("min_n", str(registration.min_n))
    table.add_row("test", registration.test)
    table.add_row("baselines", ", ".join(registration.must_beat_baselines))
    table.add_row("config", registration.generation.fingerprint)
    table.add_row("prereg_sha", registration.sha[:16])
    console.print(table)

    ok = True
    if golden is not None:
        try:
            golden_set = _load_golden_set(golden, task=registration.task)
        except (OSError, ValueError) as exc:
            console.print(f"[red]Golden set error:[/red] {exc}")
            raise typer.Exit(code=1) from None
        if golden_set.sha != registration.golden_sha:
            console.print(
                f"[red]Golden set has changed:[/red] {golden} now hashes to "
                f"{golden_set.sha[:16]}, registered {registration.golden_sha[:16]}"
            )
            ok = False
        else:
            console.print(f"[green]Golden set matches[/green] ({len(golden_set)} examples).")

    status = verify_committed(registration.path)
    if status.committed:
        console.print(f"[green]git: committed[/green] at {status.commit[:12]} and unmodified.")
    else:
        console.print(f"[red]git: not committed[/red] — {status.reason}")
        ok = False
    if not ok:
        raise typer.Exit(code=1)


@adapters_app.command("list")
def adapters_list(
    task: str = typer.Option(None, "--task", help="Filter by task class."),
    status: str = typer.Option(None, "--status", help="Filter by status."),
) -> None:
    """List adapters in the registry (candidate/promoted/retired)."""
    from .registry import AdapterError

    try:
        entries = _adapter_store().list(task=task, status=status)
    except AdapterError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    table = Table(title="hearth adapters", show_header=True, header_style="bold")
    table.add_column("id")
    table.add_column("task")
    table.add_column("base_model")
    table.add_column("status")
    table.add_column("eval")
    for e in entries:
        color = {"promoted": "green", "candidate": "yellow", "retired": "dim"}.get(
            e.status, "white"
        )
        scores = ", ".join(f"{k}={v:g}" for k, v in e.eval_scores.items())
        table.add_row(
            e.id, e.task, e.base_model, f"[{color}]{e.status}[/{color}]", scores or "-"
        )
    console.print(table)


@adapters_app.command("promote")
def adapters_promote(
    adapter_id: str = typer.Argument(..., help="Adapter id to promote."),
    report: Path = typer.Option(
        None, "--report", help="Eval report JSON written by `hearth eval --report-json`."
    ),
    prereg: Path = typer.Option(
        None, "--prereg", help="Committed pre-registration YAML declaring the bar."
    ),
    candidate_score: float = typer.Option(
        None, "--candidate-score", hidden=True, help="REMOVED — a typed score is not evidence."
    ),
    incumbent_score: float = typer.Option(
        None, "--incumbent-score", hidden=True, help="REMOVED — a typed score is not evidence."
    ),
) -> None:
    """Promote a candidate from a measured eval report (ARCHITECTURE §7, ADR-006).

    The gate is **recomputed here** from the persisted per-example vectors under the bar in
    the committed pre-registration — the report is evidence, not a verdict to be trusted.
    Both ``--report`` and ``--prereg`` are required; the normal path is
    ``hearth eval --prereg <file> --promote``, which measures and promotes in one step.

    ``--candidate-score`` / ``--incumbent-score`` are gone. They let an operator type two
    floats and promote on them, with no golden set, no metric, and no measurement behind
    either number (LEARNING_plan F3) — that was the path the promotion in docs/RESULTS.md
    actually used.
    """
    import json as _json

    from .registry import AdapterError, GateNotPassedError
    from .training.eval import EvalReport, GateProvenanceError, evaluate_gate
    from .training.prereg import PreRegError, load_prereg, verify_committed

    if candidate_score is not None or incumbent_score is not None:
        console.print(
            "[red]--candidate-score/--incumbent-score have been removed.[/red] An "
            "operator-typed score is not evidence: it names no golden set, no metric and no "
            "model. Measure instead:\n"
            "  hearth eval <adapter> --golden <set> --prereg <committed prereg> --promote"
        )
        raise typer.Exit(code=2)
    if report is None or prereg is None:
        console.print(
            "[red]Promotion requires --report and --prereg.[/red] Produce the report with "
            "`hearth eval ... --report-json <file>`, or promote directly with "
            "`hearth eval ... --prereg <file> --promote`."
        )
        raise typer.Exit(code=1)

    try:
        payload = _json.loads(Path(report).read_text(encoding="utf-8"))
        candidate = EvalReport.from_json(payload["candidate"])
        incumbent = EvalReport.from_json(payload["incumbent"])
        baselines = {
            name: EvalReport.from_json(obj)
            for name, obj in (payload.get("baselines") or {}).items()
        }
    except (OSError, KeyError, TypeError, ValueError) as exc:
        console.print(f"[red]Unusable eval report:[/red] {exc}")
        raise typer.Exit(code=1) from None

    try:
        registration = load_prereg(prereg)
    except PreRegError as exc:
        console.print(f"[red]Pre-registration error:[/red] {exc}")
        raise typer.Exit(code=1) from None
    status = verify_committed(registration.path)
    if not status.committed:
        console.print(f"[red]Promotion refused:[/red] {status.reason}")
        raise typer.Exit(code=1)
    problems = registration.mismatches(candidate)
    if problems:
        console.print(
            "[red]Promotion refused:[/red] the report is not the registered experiment — "
            + "; ".join(problems)
        )
        raise typer.Exit(code=1)

    missing = [b for b in registration.must_beat_baselines if b not in baselines]
    if missing:
        console.print(
            "[red]Promotion refused:[/red] report is missing pre-registered baseline(s): "
            + ", ".join(missing)
        )
        raise typer.Exit(code=1)

    try:
        gate = evaluate_gate(
            candidate,
            incumbent,
            incumbent_role=payload.get("incumbent_role", "incumbent"),
            baselines={b: baselines[b] for b in registration.must_beat_baselines},
            margin=registration.min_effect,
            alpha=registration.alpha,
            min_n=registration.min_n,
            test=registration.test,
            candidate_id=adapter_id,
            incumbent_id=payload.get("incumbent_id", ""),
        )
    except (GateProvenanceError, ValueError) as exc:
        console.print(f"[red]Gate refused to compare:[/red] {exc}")
        raise typer.Exit(code=1) from None

    proof = dict(registration.as_proof())
    proof["prereg_committed"] = True
    proof["prereg_commit"] = status.commit
    proof["measured_at"] = candidate.measured_at
    try:
        _adapter_store().promote(adapter_id, gate=gate, proof=proof)
    except GateNotPassedError:
        console.print(f"[red]Promotion refused:[/red] {gate.reason}")
        raise typer.Exit(code=1) from None
    except AdapterError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    console.print(
        f"[green]Promoted[/green] {adapter_id} (gate passed, {gate.test} "
        f"p={gate.p_value:.4f}, n={gate.n})."
    )


@adapters_app.command("retire")
def adapters_retire(
    adapter_id: str = typer.Argument(..., help="Adapter id to retire."),
) -> None:
    """Retire an adapter so it no longer serves."""
    from .registry import AdapterError

    try:
        _adapter_store().retire(adapter_id)
    except AdapterError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    console.print(f"[green]Retired[/green] {adapter_id}.")


@plugins_app.command("list")
def plugins_list() -> None:
    """List third-party plugins discovered via entry points (Phase 7, docs/PLUGINS.md).

    Shows every entry point registered in the ``hearth.providers`` /
    ``hearth.vector_stores`` / ``hearth.embedders`` groups, with its status: ``ok`` if it
    imported and satisfied the group's Protocol, else why it was skipped. A broken plugin
    is reported here rather than crashing the server.
    """
    from .plugins import discover_all

    found = discover_all()
    table = Table(title="hearth plugins", show_header=True, header_style="bold")
    table.add_column("name")
    table.add_column("group")
    table.add_column("target")
    table.add_column("status")
    for p in found:
        status = "[green]ok[/green]" if p.ok else f"[red]skipped[/red] — {p.detail}"
        table.add_row(p.name, p.group, p.value, status)
    console.print(table)
    if not found:
        console.print("[dim]No plugins installed. See docs/PLUGINS.md to write one.[/dim]")


if __name__ == "__main__":
    app()
