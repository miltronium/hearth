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
from contextlib import contextmanager
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

#: Help panels group the commands by what the operator is trying to do, in learning order.
PANEL_START = "Start here"
PANEL_USE = "Use"
PANEL_TRAIN = "Train and evaluate"
PANEL_EXTEND = "Extend"


def reflow_help(text: str | None) -> str | None:
    """Join a docstring's hard-wrapped prose lines so the terminal can wrap them itself.

    The docstrings are wrapped at ~90 columns in the source. Typer/rich keeps every single
    newline after the summary paragraph, so at an 80-column terminal each source line was
    wrapped again and left a one-word orphan ("with a" / "chat page"). The convention (the
    same one ``scripts/gen_manpage.py`` reads) is: paragraphs are separated by blank lines;
    a line that starts with whitespace (an example or command block) or a list marker stays
    on its own line, verbatim; an unindented line continues the prose line before it.
    Idempotent, and it never touches text after a ``\\f``.
    """
    if not text:
        return text
    import inspect

    body, sep, tail = inspect.cleandoc(text).partition("\f")
    paragraphs = []
    for para in body.split("\n\n"):
        out: list[str] = []
        prose_open = False  # is the last output line prose that the next one may continue?
        for line in para.split("\n"):
            verbatim = line[:1].isspace() or line.lstrip().startswith(("- ", "* ", "• "))
            if verbatim or not line.strip():
                out.append(line)
                prose_open = False
            elif prose_open:
                out[-1] = f"{out[-1]} {line.strip()}"
            else:
                out.append(line.strip())
                prose_open = True
        paragraphs.append("\n".join(out))
    return "\n\n".join(paragraphs) + sep + tail


class ReflowGroup(typer.core.TyperGroup):
    """The root group: reflows every command's and group's help once, when the CLI is built.

    Typer constructs subgroups first and hands them to the root in ``commands``, so walking
    the tree here covers every ``--help`` page and the man page generator, which reads the
    same click objects.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        def walk(cmd: Any) -> None:
            cmd.help = reflow_help(cmd.help)
            for sub in getattr(cmd, "commands", {}).values():
                walk(sub)

        walk(self)


app = typer.Typer(
    name="hearth",
    cls=ReflowGroup,
    help=(
        "HEARTH: a local-first, no-egress LLM gateway for Apple Silicon.\n\n"
        "Runs local MLX models behind an OpenAI-compatible API on 127.0.0.1:8080, with a\n"
        "chat page (/chat), a bounded read-only agent, local RAG, an MCP server for Claude\n"
        "Code, and LoRA training behind a statistical promotion gate. The shipped routing\n"
        "profile defines no remote models, and loading a model never downloads it.\n\n"
        "Learn it in this order:\n"
        "  hearth doctor --offline        is it safe to use offline right now?\n"
        "  hearth models list             which models exist; (default) is what serves\n"
        '  hearth run "hello"             one local completion\n'
        "  hearth serve                   API + chat page at http://127.0.0.1:8080/chat\n"
        '  hearth agent "..."             a bounded agent over files you allow\n'
        "  hearth stats                   token savings (per process)"
    ),
    epilog=(
        "Run it from the repo as: uv run --no-sync hearth COMMAND. A bare 'uv run' or a\n"
        "partial 'uv sync' uninstalls mlx and silently drops you onto the echo stub.\n\n"
        "Offline check: hearth doctor --offline (exit 0 = SAFE, 1 = UNSAFE).\n\n"
        "Every command: hearth COMMAND --help. Full reference: man ./man/hearth.1.\n\n"
        "Guide: docs/GUIDE.md (start with 'Learn HEARTH in 15 minutes')."
    ),
    no_args_is_help=True,
    add_completion=False,
)
models_app = typer.Typer(
    help=(
        "Model registry: list, download, remove, convert, export to Core ML.\n\n"
        "The registry is config/models.yaml (or HEARTH_MODELS_YAML). 'models pull' is the\n"
        "only command in HEARTH that downloads; every other load reads from disk only.\n\n"
        "Examples:\n"
        "  hearth models list\n"
        "  hearth models pull mlx-community/Qwen2.5-3B-Instruct-4bit"
    ),
)
app.add_typer(models_app, name="models", rich_help_panel=PANEL_START)
rag_app = typer.Typer(
    help=(
        "Local RAG: chunk and index files into a collection, then search it.\n\n"
        "Collections live in ~/.hearth/rag/NAME.db and hold the raw chunk text. The default\n"
        "embedder (HEARTH_EMBEDDER=hash) is lexical: queries must share words with the text.\n\n"
        "Examples:\n"
        "  hearth rag ingest ~/notes --collection notes\n"
        '  hearth rag query "descale the kettle" --collection notes'
    ),
)
app.add_typer(rag_app, name="rag", rich_help_panel=PANEL_USE)
adapters_app = typer.Typer(
    help=(
        "LoRA adapter registry: list, promote (from a measured report), retire.\n\n"
        "Adapters are recorded in ~/.hearth/adapters.json as candidate, promoted or retired.\n"
        "A promoted adapter is applied automatically to its task class when its base serves.\n\n"
        "Examples:\n"
        "  hearth adapters list --status candidate\n"
        "  hearth adapters retire ADAPTER_ID"
    ),
)
app.add_typer(adapters_app, name="adapters", rich_help_panel=PANEL_TRAIN)
prereg_app = typer.Typer(
    help=(
        "Pre-registration: declare the eval bar before training, then commit it.\n\n"
        "hearth eval --promote refuses unless a committed, unmodified pre-registration\n"
        "declares the bar the run is judged against.\n\n"
        "Examples:\n"
        "  hearth prereg init --task classify --golden golden.jsonl \\\n"
        "    --out prereg/classify.yaml\n"
        "  hearth prereg check prereg/classify.yaml --golden golden.jsonl"
    ),
)
app.add_typer(prereg_app, name="prereg", rich_help_panel=PANEL_TRAIN)
plugins_app = typer.Typer(
    help=(
        "Third-party providers, vector stores and embedders found via entry points.\n\n"
        "See docs/PLUGINS.md to write one.\n\n"
        "Examples:\n"
        "  hearth plugins list"
    ),
)
app.add_typer(plugins_app, name="plugins", rich_help_panel=PANEL_EXTEND)
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


@app.command(rich_help_panel=PANEL_START)
def version() -> None:
    """Print the HEARTH version.

    The same version appears in the serve banner and in GET /v1/hearth/admin/health.

    Examples:
      hearth version
    """
    console.print(f"hearth {__version__}")


@app.command(rich_help_panel=PANEL_START)
def doctor(
    offline: bool = typer.Option(
        False,
        "--offline",
        help="Answer 'is it safe to use HEARTH offline right now?' — measures the routing "
        "profile, model resolution and every load path; exits 1 when unsafe.",
    ),
) -> None:
    """Check the environment; with --offline, decide if HEARTH is safe to use offline.

    Plain doctor checks for Apple Silicon, enough memory, an importable mlx-lm, and a
    writable ~/.hearth. A FAIL row is fatal. A WARN row (for example an unregistered
    HEARTH_DEFAULT_MODEL, which is ignored) is not.

    doctor --offline measures the offline posture of THIS shell's environment instead of
    reading config: the routing profile the router would load has no remotes; the bind
    host is loopback; HEARTH_ALLOW_DOWNLOADS is off; a model that is not on disk fails
    with no connection attempted; every reachable model is on disk (the row names the
    path); and train / models convert / models export-coreml are disk-only. Run it with
    the same variables you serve with. It does not inspect firewalls or other processes.

    Examples:
      hearth doctor
      hearth doctor --offline; echo $?
      HEARTH_ROUTING_YAML=config/routing.finance.yaml hearth doctor --offline

    Env: HEARTH_HOME, HEARTH_BACKEND, HEARTH_HOST, HEARTH_ALLOW_DOWNLOADS,
    HEARTH_ROUTING_YAML, HEARTH_DEFAULT_MODEL, HEARTH_MODELS_YAML, HEARTH_EMBEDDER,
    HF_HUB_CACHE, HF_HOME.

    Exit: doctor: 0 ready (warnings allowed), 1 a fatal check failed.
    doctor --offline: 0 SAFE offline, 1 UNSAFE (the last line names the failed rows).
    """
    if offline:
        _doctor_offline()
        return
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


def _doctor_offline() -> None:
    """Render ``run_offline_checks`` and exit 1 when any safety check fails."""
    from rich.markup import escape

    from .doctor import OFFLINE_LIMITS, offline_verdict, run_offline_checks

    checks = run_offline_checks()
    table = Table(title="hearth doctor --offline", show_header=True, header_style="bold")
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail", overflow="fold")
    for c in checks:
        # Same three-way mark as plain `doctor`: a non-fatal failure is a WARN, not a FAIL —
        # rendering it FAIL beside a "SAFE offline" verdict contradicted the verdict (B-031).
        mark = "[green]PASS[/green]" if c.ok else (
            "[red]FAIL[/red]" if c.fatal else "[yellow]WARN[/yellow]"
        )
        table.add_row(c.name, mark, c.detail)
    console.print(table)
    for limit in OFFLINE_LIMITS:
        console.print(f"[dim]not measured: {limit}[/dim]")
    # The verdict names WHY (B-108): "unusable" (HEARTH refuses to run on it) is not "can
    # egress", though both exit 1 — neither is safe to run.
    safe, verdict = offline_verdict(checks)
    if not safe:
        console.print(f"[red]{escape(verdict)}[/red]")
        raise typer.Exit(code=1)
    console.print(f"[green]{escape(verdict)}[/green]")


@app.command(rich_help_panel=PANEL_USE)
def serve(
    host: str = typer.Option(
        None,
        help="Bind host (default from HEARTH_HOST / 127.0.0.1). Anything but loopback "
        "exposes the API to the network and makes doctor --offline UNSAFE.",
    ),
    port: int = typer.Option(None, help="Bind port (default from HEARTH_PORT / 8080)."),
) -> None:
    """Start the OpenAI-compatible gateway and the /chat page on 127.0.0.1:8080.

    Serves /v1/chat/completions, /v1/models, /v1/embeddings and the /v1/hearth/*
    extensions to any OpenAI client (base_url http://127.0.0.1:8080/v1). For the built-in
    chat page open http://127.0.0.1:8080/chat and paste the token once.

    Routing profile: config/routing.yaml unless HEARTH_ROUTING_YAML names another. The
    default defines zero remote models, so no request can leave this machine. A relative
    HEARTH_ROUTING_YAML resolves against the repo root, ~ is expanded, and a named file
    that does not exist stops serve from starting.

    Token: every /v1/* route needs 'Authorization: Bearer TOKEN'. The token is created on
    first run in ~/.hearth/token (mode 0600; $HEARTH_HOME/token). /chat,
    /v1/hearth/admin/health and /v1/hearth/admin/ready need no token.

    Readiness: on the mlx backend the default model loads in the background at startup
    (HEARTH_WARMUP). GET /v1/hearth/admin/ready answers 200 once the default has loaded
    (or, with HEARTH_WARMUP=false, once its weights are on disk) and stays 200 if it is
    later evicted to make room; it answers 503 'loading' during the first load, 'failed'
    with a reason when the load failed or the weights are missing, and 'stub' when the
    echo fallback is answering. Check backend= in the banner: 'echo' is a test stub.

    Settings are read once at startup, so restart after changing any HEARTH_* variable.

    Examples:
      hearth serve
      hearth serve --port 8081
      HEARTH_ROUTING_YAML=config/routing.finance.yaml hearth serve

    Env: HEARTH_HOST, HEARTH_PORT, HEARTH_BACKEND, HEARTH_ROUTING_YAML,
    HEARTH_DEFAULT_MODEL, HEARTH_MODELS_YAML, HEARTH_REQUIRE_AUTH, HEARTH_WARMUP,
    HEARTH_RAM_CEILING_GB, HEARTH_FILE_ROOTS, HEARTH_HOME.

    Exit: runs until interrupted (Ctrl-C). Exits non-zero without serving when the
    selected routing profile does not exist.
    """
    import uvicorn

    from .gateway import create_app

    settings = get_settings()
    ensure_home(settings)
    _require_registered_default()
    get_or_create_token(settings)  # ensure a token exists for bearer auth
    _log_hearth_to_stderr()
    with _backend_required():
        provider = select_provider(settings)

    bind_host = host or settings.host
    bind_port = port or settings.port
    with _routing_profile_required():
        gateway = create_app(provider=provider, settings=settings)
    console.print(
        f"[bold]HEARTH[/bold] {__version__} — backend=[cyan]{provider.name}[/cyan] "
        f"model=[cyan]{get_registry().default_id}[/cyan]"
    )
    console.print(f"Serving on http://{bind_host}:{bind_port}  (OpenAI-compatible /v1)")
    uvicorn.run(gateway, host=bind_host, port=bind_port)


def _log_hearth_to_stderr() -> None:
    """Show HEARTH's own INFO log in the serve console (uvicorn only configures its own).

    Without a handler, Python prints only WARNING and up, so the lines that say which
    weights were loaded, which model generated each request and what was evicted never
    appeared — the server's evidence of what it actually did was being discarded.

    Idempotent: exactly one such handler, however often it is called. Ours is tagged and
    replaced, so it always writes to the CURRENT ``sys.stderr`` (a repeat call in one
    process -- tests, an embedding caller -- used to keep a handler bound to a stream that
    may have been closed). A handler someone else attached no longer stops ours from being
    installed, nor the INFO level from being set.
    """
    import logging

    log = logging.getLogger("hearth")
    for old in [h for h in log.handlers if getattr(h, _CLI_LOG_HANDLER_TAG, False)]:
        log.removeHandler(old)
    handler = logging.StreamHandler()
    setattr(handler, _CLI_LOG_HANDLER_TAG, True)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)


#: Marks the stderr handler ``_log_hearth_to_stderr`` installed, so a repeat call replaces it.
_CLI_LOG_HANDLER_TAG = "_hearth_cli_stderr"


@app.command(rich_help_panel=PANEL_USE)
def run(
    prompt: str = typer.Argument(None, help="Prompt text. Omit to read from stdin."),
    max_tokens: int = typer.Option(512, help="Max tokens to generate."),
    file: Path = typer.Option(
        None, "--file", help="Read the prompt from this file instead of the argument."
    ),
    intent: str = typer.Option(
        None,
        "--intent",
        help="Task class hint that skips classification: summarize, extract, classify, rank, "
        "draft, code, reason or chat. Picks that class's model under 'auto'.",
    ),
    model: str = typer.Option(
        "auto",
        "--model",
        help="Registry model id to serve this prompt. 'auto' lets the routing ladder pick.",
    ),
) -> None:
    """Run one prompt through the local model and print the answer.

    Always local: run never escalates, whatever the routing profile. It has no tools, so
    it cannot open a file you mention: pass --file to send a file's text as the prompt,
    or use hearth agent. The answer goes to stdout; a 'served by MODEL via BACKEND' line
    goes to stderr, so piping the answer stays clean.

    Examples:
      hearth run "Summarize: HEARTH keeps inference on this Mac."
      hearth run --file notes.txt --max-tokens 256
      echo "crash when saving" | hearth run --intent classify
      hearth run --model mlx-community/Qwen2.5-3B-Instruct-4bit "hello"

    Env: HEARTH_BACKEND, HEARTH_DEFAULT_MODEL, HEARTH_MODELS_YAML, HEARTH_ROUTING_YAML,
    HEARTH_HOME.

    Exit: 0 answered; 1 empty prompt; 2 --model names no servable registry model,
    HEARTH_DEFAULT_MODEL names an unregistered id, or the selected routing profile
    is missing.
    """
    _require_registered_default()
    from .router.classify import UnknownIntentError, check_intent

    try:  # B-060: an unknown --intent is an error, not silently replaced by keyword rules
        intent = check_intent(intent)
    except UnknownIntentError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from None
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
    with _backend_required():
        provider = select_provider(settings)
    _require_known_model(provider, model)
    # Surface the hint so `--intent` is observably wired (the router consumes it below).
    if intent:
        console.print(f"[dim]intent={intent}[/dim]")
    with _routing_profile_required():
        router = Router(local_provider=provider)
    routed = router.route(
        GenRequest(
            messages=[Message(role="user", content=text)],
            # "auto" (not the registry default) so a per-class ladder rung applies; an
            # explicit --model pins it. Pinning the default here used to bypass the ladder.
            model=model,
            max_tokens=max_tokens,
        ),
        intent=intent,
        # A one-shot CLI stays local unless a daemon/policy escalates; keep it hard-local
        # so `hearth run` never makes a surprise remote call from a script.
        allow_escalation=False,
    )
    console.print(routed.result.text, markup=False, highlight=False)
    # Which weights answered, from the provider that ran — on stderr, so piping the answer
    # stays clean.
    typer.echo(
        f"[served by {routed.result.model} via {routed.result.backend}; "
        f"class={routed.decision.task_class}]",
        err=True,
    )


def _require_known_model(provider, model: str) -> None:
    """Exit 2 with the registry's answer when ``model`` is not servable here."""
    from .serving import UnknownModelError, check_model

    try:
        check_model(provider, get_registry(), model)
    except UnknownModelError as exc:
        console.print(f"[red]Unknown model:[/red] {exc}", markup=True, highlight=False)
        raise typer.Exit(code=2) from None


def _require_registered_default() -> None:
    """Exit 2 when ``HEARTH_DEFAULT_MODEL`` is set to an id the registry does not hold.

    Asked of the registry the serving code uses (``get_registry``), via the same
    ``require_default`` that ``hearth doctor`` reports, so the refusal and the report
    cannot disagree. Unset still means the catalog default (B-047).
    """
    from rich.markup import escape

    from .registry import UnregisteredDefaultModelError

    try:
        get_registry().require_default()
    except UnregisteredDefaultModelError as exc:
        console.print(f"[red]Refusing to start:[/red] {escape(str(exc))}")
        raise typer.Exit(code=2) from None


@contextmanager
def _routing_profile_required():
    """Exit 2 with the router's own message when the selected routing profile is missing.

    Wraps the code that actually builds the router (``Router()`` -> ``get_policy()``), so
    the error caught is the one the router raised, not a re-derivation of it (B-033: it used
    to surface as a full traceback before the one line that says how to fix it).
    """
    from rich.markup import escape

    from .router.policy import RoutingPolicyError, RoutingProfileNotFoundError

    try:
        yield
    except RoutingProfileNotFoundError as exc:
        # RoutingPolicyError subclasses it: the file exists but names an unusable rung (B-076).
        label = (
            "Routing profile unusable:"
            if isinstance(exc, RoutingPolicyError)
            else "Routing profile not found:"
        )
        console.print(f"[red]{label}[/red] {escape(str(exc))}")
        raise typer.Exit(code=2) from None


@contextmanager
def _backend_required():
    """Exit 2 with the provider factory's own message when HEARTH_BACKEND is unknown (B-059).

    Wraps the code that actually builds the provider, so the error caught is the one
    ``select_provider`` raised — it used to end every model-using command in a traceback.
    """
    from rich.markup import escape

    from .providers import UnknownBackendError

    try:
        yield
    except UnknownBackendError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=2) from None


@contextmanager
def _embedder_required():
    """Exit 1 with the embedder's own message when it cannot embed (B-036).

    ``MLXEmbedder`` loads lazily, so the failure surfaces at the first ``embed()`` inside
    ``ingest``/``query`` — catch it there rather than predicting it from the setting.
    """
    from rich.markup import escape

    from .memory import EmbeddingUnavailableError

    try:
        yield
    except EmbeddingUnavailableError as exc:
        console.print(f"[red]Embedder unavailable:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1) from None


@app.command(rich_help_panel=PANEL_USE)
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
    model: str = typer.Option(
        "auto",
        "--model",
        help="Registry model id for every step. 'auto' lets the routing ladder pick.",
    ),
) -> None:
    """Run a bounded, read-only, tool-using local agent over your own files.

    The agent plans, calls one tool, reads the result, and repeats until it answers or
    hits a bound. Every generation is local and it never escalates. Unlike hearth run
    (and plain /v1/chat/completions) it can actually read the files it talks about, so a
    question about a directory is answered from the filesystem, not from imagination.

    Tools are assembled from what is present, and the header line lists them:
    list_files and read_file always (deny-by-default: they refuse every path outside
    HEARTH_FILE_ROOTS); rag_search with --collection; finance_total / finance_explain /
    finance_rows when a ledger exists at ~/.hearth/finance/ledger.db. There are no write,
    shell or network tools, and deliberately no flag to add or un-vet tools: only tools
    whose code lives in hearth.agent run, which is what the no-network test covers. See
    docs/AGENT.md.

    Read the step table before trusting the answer: small local models retry failed
    ideas and invent plausible filenames.

    Examples:
      HEARTH_FILE_ROOTS=~/statements hearth agent "how many CSV files are there?"
      hearth agent --collection notes "when should I descale the kettle?"
      HEARTH_FILE_ROOTS=~/notes hearth agent --json "list the md files" > run.json

    Env: HEARTH_FILE_ROOTS (colon-separated readable directories; unset = none),
    HEARTH_FILE_MAX_BYTES, HEARTH_BACKEND, HEARTH_DEFAULT_MODEL, HEARTH_ROUTING_YAML,
    HEARTH_EMBEDDER, HEARTH_HOME.

    Exit (the stop reason): 0 the model answered, and only then; 1 the run stopped at a
    bound or a failure (max_iterations, timeout, token_budget, invalid_output,
    provider_error, egress_refused) and there is no answer; 2 never started (an
    impossible bound, an empty --collection, an unknown --model, or nothing reachable).
    """
    from .agent import Agent, AgentConfigError, Budget, local_toolset
    from .config import Settings
    from .mcp.files import allowed_roots

    _require_registered_default()
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
    with _backend_required():
        provider = select_provider(settings)
    _require_known_model(provider, model)
    # "auto" rather than the registry default: pinning the default bypassed the per-class
    # ladder. Each step's served model is in the transcript, read off the provider that ran.
    model_id = model
    try:
        budget = Budget(
            max_iterations=max_iterations,
            max_seconds=max_seconds,
            max_total_tokens=max_tokens,
        )
        # `vetted_only` is left at its default of True and is not plumbed to a flag; see the
        # docstring. The router is entered with allow_escalation=False by the loop itself,
        # which then verifies the executed route actually reported the local backend.
        with _routing_profile_required():
            router = Router(local_provider=provider)
        runner = Agent(router, tools, budget=budget, model=model_id)
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


@app.command(rich_help_panel=PANEL_USE)
def mcp() -> None:
    """Start the stdio MCP server so Claude Code can hand subtasks to the local model.

    Exposes hearth_summarize, hearth_classify, hearth_extract, hearth_draft, the
    path-taking hearth_summarize_file / hearth_classify_file / hearth_extract_file, and
    hearth_rag_query. Every tool runs on the local model with escalation disabled, under
    any routing profile, in-process (no HTTP, no token). Prefer the *_file tools for
    anything confidential: HEARTH opens the file, so its content never enters the calling
    agent's context. Those tools refuse every path outside HEARTH_FILE_ROOTS.

    Register it with the venv's absolute path; an MCP subprocess has no activated venv
    and no guaranteed working directory. Needs the mcp extra.

    Examples:
      hearth mcp
      claude mcp add hearth -e HEARTH_FILE_ROOTS="$HOME/docs" -- \\
        ~/Claude/apps/HEARTH/.venv/bin/hearth mcp

    Env: HEARTH_FILE_ROOTS, HEARTH_FILE_MAX_BYTES, HEARTH_BACKEND, HEARTH_DEFAULT_MODEL,
    HEARTH_ROUTING_YAML, HEARTH_EMBEDDER, HEARTH_HOME.

    Exit: 0 when the client closes stdin; 1 when the mcp extra is not installed.
    """
    _require_registered_default()
    try:
        from .mcp import server

        with _routing_profile_required(), _backend_required():
            server.run()
    except ModuleNotFoundError as exc:
        # The `mcp` SDK is an optional extra (server.py imports it lazily at run time, so
        # the failure surfaces here rather than at import). Fail loudly with the fix instead
        # of a bare traceback, and exit non-zero so callers/CI notice.
        if "mcp" not in str(exc):
            raise
        console.print(
            "[red]The MCP server requires the 'mcp' extra.[/red]\n"
            "Install it with:  "
            "[cyan]uv sync --extra mlx --extra mcp --extra dev --extra files[/cyan]"
        )
        raise typer.Exit(code=1) from None


@app.command(rich_help_panel=PANEL_USE)
def stats(
    since: str = typer.Option(
        None, "--since", help="Rollup window, e.g. 7d / 24h / 30m (default: all)."
    ),
) -> None:
    """Show token-savings, escalation and latency rollups for THIS process.

    Metrics are kept in memory per process and hold no prompt or response text. A fresh
    'hearth stats' therefore always shows zeros; the numbers accumulate inside a running
    hearth serve. For a running server, read GET /v1/hearth/admin/metrics?since=24h
    (same rollup, as JSON, token required).

    A request that ended in an error is counted in requests and in failed / failure rate;
    backend mix and latency count answered requests only. Escalations failed counts every
    request whose remote call errored, whether the local fallback then answered or failed
    too: the remote may have received the prompt either way.

    Examples:
      hearth stats
      hearth stats --since 24h
      curl -s -H "Authorization: Bearer $(cat ~/.hearth/token)" \\
        "http://127.0.0.1:8080/v1/hearth/admin/metrics?since=24h"
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
    # The remote errored. The request was then served locally, or -- when it is also counted
    # under "failed" -- the local fallback failed too. Either way the prompt may have left.
    table.add_row(
        "escalations failed (remote errored; prompt may have left)",
        str(roll["escalations_failed"]),
    )
    table.add_row("failed (error, no answer)", str(roll["failed"]))
    table.add_row("failure rate", f"{roll['failure_rate']:.2%}")
    backend_mix = ", ".join(f"{k}={v}" for k, v in roll["backend_mix"].items())
    class_mix = ", ".join(f"{k}={v}" for k, v in roll["class_mix"].items())
    table.add_row("backend mix", backend_mix or "-")
    table.add_row("class mix", class_mix or "-")
    table.add_row("latency p50 (ms)", f"{roll['latency_ms']['p50']:g}")
    table.add_row("latency p95 (ms)", f"{roll['latency_ms']['p95']:g}")
    console.print(table)


@models_app.command("list")
def models_list() -> None:
    """List the models in the registry; (default) marks the one that serves 'auto'.

    The registry is config/models.yaml (or HEARTH_MODELS_YAML). The (default) marker
    reflects HEARTH_DEFAULT_MODEL when it names a registered id, so this is the way to
    confirm which model will answer. Listing does not check that weights are on disk;
    hearth doctor --offline does.

    Examples:
      hearth models list
      HEARTH_DEFAULT_MODEL=mlx-community/Qwen2.5-3B-Instruct-4bit \\
        hearth models list

    Env: HEARTH_MODELS_YAML, HEARTH_DEFAULT_MODEL.
    """
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
    """Download a registry model's weights into ~/.hearth/models (needs the network).

    The ONE deliberate download path in HEARTH: serve, chat, agent, MCP, RAG, train,
    models convert and models export-coreml all load from disk only and fail rather
    than fetch. Only registry ids are accepted (hearth models list). Respects the
    HF_ENDPOINT mirror and HF_HUB_OFFLINE; no host is hardcoded.

    Examples:
      hearth models pull mlx-community/Qwen2.5-3B-Instruct-4bit
      HF_ENDPOINT=https://mirror.example \\
        hearth models pull mlx-community/Qwen2.5-14B-Instruct-4bit

    Env: HEARTH_HOME, HEARTH_MODELS_YAML, HF_ENDPOINT, HF_HUB_OFFLINE.

    Exit: 0 downloaded (or nothing to pull, e.g. echo); 1 unknown model id.
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
    """Delete a model's downloaded copy from ~/.hearth/models.

    Removes only HEARTH's own copy (models--ORG--NAME). It does not touch the Hugging
    Face hub cache (HF_HUB_CACHE / ~/.cache/huggingface/hub), which the loader also
    reads, so a model can still resolve after rm; hearth doctor --offline shows where.

    Examples:
      hearth models rm mlx-community/Qwen2.5-3B-Instruct-4bit

    Env: HEARTH_HOME, HEARTH_MODELS_YAML.

    Exit: 0 removed; 1 unknown id, or not present under ~/.hearth/models.
    """
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
    """Quantize or convert a checkpoint on disk into an MLX-servable model.

    Needs the mlx extra and the source on disk (hearth models pull, or a local path). It
    never downloads unless HEARTH_ALLOW_DOWNLOADS=1. To serve the result, add it to
    config/models.yaml (the registry is data).

    Examples:
      hearth models convert --source ./my-checkpoint \\
        --out ~/.hearth/models/mine-q8 --q-bits 8
      hearth models convert --source ./my-checkpoint \\
        --out ./my-checkpoint-mlx --no-quantize

    Env: HEARTH_HOME, HEARTH_ALLOW_DOWNLOADS, HF_HUB_CACHE, HF_HOME.

    Exit: 0 converted; 1 invalid options, source not on disk, output exists, or mlx
    missing.
    """
    from .convert import ConvertConfig, ConvertUnavailableError
    from .convert import convert as run_convert
    from .providers.mlx import ModelNotOnDiskError

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
    except (ConvertUnavailableError, ModelNotOnDiskError, FileExistsError) as exc:
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
    """Export a checkpoint on disk to a Core ML .mlpackage for the offline Swift path.

    The .mlpackage is loaded by the Swift CoreMLProvider (swift/OFFLINE.md) for
    fully-offline, daemon-free inference. Needs the coreml extra (add --extra coreml to
    the one-command sync) and the source on disk; it never downloads unless
    HEARTH_ALLOW_DOWNLOADS=1.

    Examples:
      hearth models export-coreml --source ./my-checkpoint \\
        --out ~/.hearth/coreml/mine
      hearth models export-coreml --source ./ckpt --out ./ckpt.mlpackage \\
        --compute-units cpuOnly

    Env: HEARTH_HOME, HEARTH_ALLOW_DOWNLOADS, HF_HUB_CACHE, HF_HOME.

    Exit: 0 exported; 1 invalid options, source not on disk, or coremltools missing.
    """
    from .coreml import CoreMLExportConfig, CoreMLExportUnavailableError
    from .coreml import export as run_export
    from .providers.mlx import ModelNotOnDiskError

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
    except (CoreMLExportUnavailableError, ModelNotOnDiskError) as exc:
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
    """Chunk, embed and store a file or directory into a local RAG collection.

    The collection is ~/.hearth/rag/COLLECTION.db and holds the raw chunk text; delete
    that file to purge it. The default embedder (HEARTH_EMBEDDER=hash) is offline and
    lexical, not semantic. HEARTH_EMBEDDER=mlx does not work today (docs/BUGS.md B-011).

    Examples:
      hearth rag ingest ~/notes --collection notes
      hearth rag ingest README.md --collection docs --size 400 --overlap 50

    Env: HEARTH_EMBEDDER, HEARTH_EMBED_DIM, HEARTH_EMBED_MODEL, HEARTH_VECTOR_STORE,
    HEARTH_HOME.
    """
    from .memory import RagIndex

    index = RagIndex()
    console.print(
        f"Ingesting [cyan]{path}[/cyan] → collection [cyan]{collection}[/cyan] "
        f"(embedder=[cyan]{index.embedder.name}[/cyan]) …"
    )
    with _embedder_required():
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
    """Search a RAG collection; with --answer, have the local model answer from the hits.

    Prints a table of score, source and text for the top --k chunks. --answer sends the
    retrieved chunks plus your question to the local model (never escalated).

    Examples:
      hearth rag query "how often to descale the kettle" --collection notes --k 2
      hearth rag query "descale" --collection notes --answer

    Env: HEARTH_EMBEDDER, HEARTH_EMBED_DIM, HEARTH_VECTOR_STORE, HEARTH_BACKEND,
    HEARTH_DEFAULT_MODEL, HEARTH_HOME.

    Exit: 0, including for a missing or empty collection (it says so).
    """
    from .memory import RagIndex

    if answer:  # --answer generates with the default model; retrieval alone does not
        _require_registered_default()
    with _backend_required():
        provider = select_provider(get_settings())
    with _routing_profile_required():
        index = RagIndex(router=Router(local_provider=provider))
    # Emptiness first: no embedding (the mlx embedder cannot load today, B-011) and no
    # --answer generation for a collection that has nothing to retrieve.
    if index.store.count(collection) == 0:
        console.print(f"[yellow]No chunks in collection[/yellow] {collection!r}.")
        raise typer.Exit(code=0)
    with _embedder_required():
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


@app.command(rich_help_panel=PANEL_TRAIN)
def train(
    task: str = typer.Option(..., "--task", help="Task class the adapter targets (e.g. extract)."),
    base: str = typer.Option(
        ..., "--base", help="Base model registry id to fine-tune (LoRA); must be on disk."
    ),
    data: Path = typer.Option(
        ...,
        "--data",
        help="Dataset JSONL: a hearth.dataset.header line, then at least 2 "
        '{"prompt","completion"} or {"messages"} rows.',
    ),
    out: Path = typer.Option(
        None, "--out", help="Output dir for the run (default: ~/.hearth/train/<run-id>)."
    ),
    iters: int = typer.Option(200, "--iters", help="Training iterations."),
    register: bool = typer.Option(
        True, "--register/--no-register", help="Register the result as a candidate adapter."
    ),
) -> None:
    """Train a LoRA adapter on disk-resident weights and register it as a candidate.

    Needs the mlx extra and the base model on disk (hearth models pull). It never
    downloads unless HEARTH_ALLOW_DOWNLOADS=1, and it checks the base model before
    creating the run directory, so a failed run leaves nothing behind. Training only
    produces a candidate; it serves nothing until it passes the promotion gate:
    hearth prereg init, commit the file, then hearth eval ADAPTER --prereg FILE --promote.

    Examples:
      hearth train --task classify --base mlx-community/Qwen2.5-3B-Instruct-4bit \\
        --data data.jsonl
      hearth train --task classify --base mlx-community/Qwen2.5-3B-Instruct-4bit \\
        --data data.jsonl --iters 50 --no-register

    Env: HEARTH_HOME, HEARTH_ALLOW_DOWNLOADS, HF_HUB_CACHE, HF_HOME.

    Exit: 0 trained (and registered); 1 dataset error, base model not on disk, mlx
    missing, or the training process failed (its exit code and stderr tail are shown).
    """
    import subprocess
    from datetime import UTC, datetime

    from rich.markup import escape

    from .config import Settings
    from .registry import AdapterError
    from .training import LoRAConfig, load_dataset
    from .training import train as run_train
    from .training.dataset import DatasetError

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
    try:
        # The trainability checks (e.g. "need at least 2 records to split into
        # train/valid") raise DatasetError / ValueError -- not RuntimeError -- so they used to
        # end in a traceback. Run them before announcing a training run.
        config.validate()
    except DatasetError as exc:
        console.print(f"[red]Dataset error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1) from None
    except ValueError as exc:
        console.print(f"[red]Invalid training config:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1) from None
    console.print(
        f"Training [cyan]{task}[/cyan] adapter on [cyan]{base}[/cyan] "
        f"({len(dataset)} records) -> {out_dir}"
    )
    try:
        outcome = run_train(config, train_run_id=run_id)
    except RuntimeError as exc:
        # The real runner raises with the fix hint when the [mlx] extra is missing, and
        # ModelNotOnDiskError (a RuntimeError) when the base model is not on disk.
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    except DatasetError as exc:
        # The real runner's batch-size preflight (validation split smaller than batch_size).
        console.print(f"[red]Dataset error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1) from None
    except subprocess.CalledProcessError as exc:
        # The mlx_lm.lora child failed (OOM, bad data…): one clean line, not a traceback.
        # Its stderr normally streamed to the terminal already; show a tail if captured.
        console.print(f"[red]Training failed: the training process exited with code "
                      f"{exc.returncode}.[/red]")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (
            exc.stderr or "")
        for line in stderr.strip().splitlines()[-20:]:
            console.print(f"  {line}", markup=False, highlight=False)
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
    # A candidate serves only once it passes the eval gate (CLAUDE.md §7). Name the commands
    # that can promote it, with the evidence each one requires — `adapters promote` alone
    # refuses without --report and --prereg (B-046).
    console.print(
        f"Registered candidate [cyan]{adapter_id}[/cyan]. It is not served until it passes "
        "the eval gate:\n"
        f"  [bold]hearth eval {adapter_id} --golden <set> --prereg <committed prereg> "
        "--promote[/bold]\n"
        f"  or: hearth eval {adapter_id} --golden <set> --prereg <committed prereg> "
        "--report-json <file>,\n"
        f"      then hearth adapters promote {adapter_id} --report <file> "
        "--prereg <committed prereg>",
        highlight=False,
    )


@app.command("eval", rich_help_panel=PANEL_TRAIN)
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
    """Score a candidate adapter against a golden set, and optionally promote it.

    The candidate is scored with an objective metric at temperature 0 and compared,
    pairwise per example, against the incumbent: the promoted adapter for the task, or
    the base model when none is promoted. To pass, it must clear the significance level
    (exact McNemar or paired bootstrap), exceed the margin, have at least min_n examples,
    and beat the empty / majority-label / copy-input baselines. --promote additionally
    requires a --prereg that is git-committed, unmodified, matches this run, was committed
    before the run started, and lives in the git repository that holds the committed
    golden set; its bar overrides --alpha / --margin / --min-n. The adapter's weights are
    hashed before scoring (no weights, no eval) and --report-json signs the report with
    this install's key, so adapters promote can tell it from a hand-written one. Real
    scores need HEARTH_BACKEND=mlx; the echo backend runs the plumbing only.

    Examples:
      hearth eval ADAPTER_ID --golden golden.jsonl
      hearth eval ADAPTER_ID --golden golden.jsonl --prereg prereg/c.yaml \\
        --report-json r.json
      hearth eval ADAPTER_ID --golden golden.jsonl --prereg prereg/c.yaml \\
        --promote

    Env: HEARTH_BACKEND, HEARTH_DEFAULT_MODEL, HEARTH_HOME, HEARTH_MODELS_YAML.

    Exit: without --promote, 0 once the gate was measured, whether it passed or failed
    (read the gate: line); 1 when it refused to measure (unknown adapter, missing adapter
    weights, bad golden set or prereg, a bar looser than the gate allows, temperature
    above 0, non-determinism); 2 when HEARTH_DEFAULT_MODEL names an unregistered model or
    the base model is empty, auto or not servable here (never a silent fallback). With
    --promote, 0 only when the adapter was promoted, else 1 (or 2 as above).
    """
    import json as _json
    from datetime import UTC, datetime

    from .config import Settings
    from .registry import AdapterError
    from .registry.adapters import adapter_weights_sha
    from .serving.pool import AUTO_MODEL_IDS
    from .training.attest import AttestationError, load_key, sign
    from .training.eval import (
        EvalConfig,
        GateProvenanceError,
        baseline_reports,
        check_determinism,
        evaluate_gate,
        score_candidate,
    )
    from .training.prereg import (
        PreRegError,
        check_provenance,
        golden_git_status,
        load_prereg,
        provenance_proof,
    )
    from .training.promotion import REPORT_SCHEMA

    if temperature > 0.0 and not allow_sampling:
        console.print(
            "[red]Refusing to score at temperature > 0:[/red] the gate would be re-rollable. "
            "Use --temperature 0 (default), or --allow-sampling to measure anyway."
        )
        raise typer.Exit(code=1)
    # An unregistered HEARTH_DEFAULT_MODEL is a misconfiguration to fix, not a default to
    # route around: the same refusal serve/run/agent make (B-047, B-070).
    _require_registered_default()

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
    # Hash the weights BEFORE scoring: this digest is what the report (and the promotion
    # proof) say was measured, and promotion re-hashes the disk against it (B-061). No
    # weights means the "candidate" would be scored as the bare base model.
    try:
        candidate_weights = adapter_weights_sha(candidate_path)
    except AdapterError as exc:
        console.print(
            f"[red]Refusing to measure:[/red] {exc} — an adapter with no weights would be "
            "scored as the base model."
        )
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

    base_model = (base or entry.base_model or "").strip()
    # Never evaluate on a silent fallback (B-070). An empty or "auto" base resolves to the
    # registry default — whatever HEARTH_DEFAULT_MODEL or the catalog says today — so the
    # report would name one model and measure another.
    if base_model in AUTO_MODEL_IDS:
        console.print(
            f"[red]Refusing to measure:[/red] {adapter_id!r} records no concrete base model "
            f"(base_model={entry.base_model!r}); pass --base <registered id>. An empty or "
            "'auto' base would silently evaluate the registry default."
        )
        raise typer.Exit(code=2)
    # Fresh Settings() (not the lru_cached get_settings) so HEARTH_BACKEND is read per call.
    with _backend_required():
        provider = select_provider(Settings())
    # The evaluated model must be the one that generates, and must be one the registry
    # serves. A pool resolves it against its backend; any other provider (echo, a plugin)
    # serves every id itself, so check_model asks the registry that the id is a registered
    # chat model. Either way an unservable base is exit 2, never a substitute.
    _require_known_model(provider, base_model)
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
    incumbent_weights = ""
    if incumbent_entry is not None and incumbent_entry.id != adapter_id:
        incumbent_id = incumbent_entry.id
        incumbent_role = "incumbent"
        incumbent_path = store.resolve_path(incumbent_id)
        try:
            incumbent_weights = adapter_weights_sha(incumbent_path)
        except AdapterError as exc:
            console.print(f"[red]Refusing to measure the incumbent {incumbent_id!r}:[/red] {exc}")
            raise typer.Exit(code=1) from None
        incumbent = _score(incumbent_path, f"{base_model}+{incumbent_id}")
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

    # Where the golden set stands in git at measurement time: promotion requires it committed
    # in the same repository as the pre-registration (training.prereg.check_provenance).
    golden_git = golden_git_status(golden)

    if report_json is not None:
        payload = {
            "schema": REPORT_SCHEMA,
            "candidate_id": adapter_id,
            "task": entry.task,
            "base_model": base_model,
            "adapter_path": entry.adapter_path,
            "candidate_weights_sha": candidate_weights,
            "measured_at": measured_at,
            "golden_git": golden_git,
            "candidate": candidate.to_json(),
            "incumbent": incumbent.to_json(),
            "incumbent_role": incumbent_role,
            "incumbent_id": incumbent_id,
            "incumbent_weights_sha": incumbent_weights,
            "baselines": {k: v.to_json() for k, v in baselines.items()},
            "gate": gate.as_proof(),
        }
        # Signed with this install's key so `adapters promote --report` can tell a report
        # this command wrote from one a human typed (B-061; training/attest.py).
        try:
            payload = sign(payload, load_key(Settings().home, create=True))
        except (AttestationError, OSError) as exc:
            console.print(f"[red]Cannot sign the eval report:[/red] {exc}")
            raise typer.Exit(code=1) from None
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
    if base_model != entry.base_model:
        console.print(
            f"[red]Promotion refused:[/red] measured on base {base_model!r}, but "
            f"{adapter_id!r} is registered on {entry.base_model!r} — an adapter is promoted "
            "on the base it serves on."
        )
        raise typer.Exit(code=1)
    try:
        status = check_provenance(registration, measured_at=measured_at, golden_git=golden_git)
    except PreRegError as exc:
        console.print(f"[red]Promotion refused:[/red] {exc}")
        raise typer.Exit(code=1) from None
    if not gate.passed:
        console.print(f"[red]Promotion refused:[/red] {gate.reason}")
        raise typer.Exit(code=1)
    # The bytes promoted must be the bytes measured.
    try:
        unchanged = adapter_weights_sha(candidate_path) == candidate_weights
    except AdapterError:
        unchanged = False
    if not unchanged:
        console.print(
            f"[red]Promotion refused:[/red] {adapter_id!r}'s weights changed during the eval."
        )
        raise typer.Exit(code=1)

    proof = dict(registration.as_proof())
    proof.update(provenance_proof(status, golden_git))
    proof["measured_at"] = candidate.measured_at
    proof["candidate_weights_sha"] = candidate_weights
    proof["evidence"] = "measured"
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
    """Scaffold a pre-registration for a golden set; then write the prose and commit it.

    The file pins the golden set by content sha and the decode parameters by
    fingerprint, so the gate can later prove the run it judged is the run that was
    registered. hypothesis, stopping_rule and kill_condition are left blank on purpose:
    prereg check and eval --promote refuse the file until you write them, because a bar
    written by the tool is not a pre-registration.

    Examples:
      hearth prereg init --task classify --golden golden.jsonl
      hearth prereg init --task classify --golden golden.jsonl \\
        --out prereg/classify.yaml

    Exit: 0 written (or printed); 1 the golden set is unreadable or empty, or the bar is
    looser than the gate allows (alpha above 0.05, negative min-effect, min-n below 30).
    """
    from .training.eval import DEFAULT_MIN_N, check_bar
    from .training.prereg import template

    try:
        check_bar(alpha=alpha, margin=min_effect, min_n=min_n, test="auto",
                  min_n_floor=DEFAULT_MIN_N)
    except ValueError as exc:
        console.print(f"[red]Refusing to scaffold this bar:[/red] {exc}")
        raise typer.Exit(code=1) from None

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
    """Validate a pre-registration and report whether git has it committed, unmodified.

    Prints the registered bar (task, metric, golden sha, alpha, min_effect, min_n, test,
    baselines). Refuses a file whose hypothesis / stopping_rule / kill_condition are
    blank or that drops a default baseline. With --golden, also checks the golden set
    still hashes to the registered sha.

    Examples:
      hearth prereg check prereg/classify.yaml
      hearth prereg check prereg/classify.yaml --golden golden.jsonl

    Exit: 0 valid, committed and unmodified (and the golden set matches); 1 otherwise.
    """
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
    status: str = typer.Option(
        None, "--status", help="Filter by status: candidate, promoted or retired."
    ),
) -> None:
    """List registered adapters with their task, base model, status and eval scores.

    Examples:
      hearth adapters list
      hearth adapters list --task classify --status promoted

    Env: HEARTH_HOME.

    Exit: 0; 1 an invalid --status.
    """
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
        None, "--report", help="Signed eval report written by `hearth eval --report-json`."
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
    """Promote a candidate from a signed eval report and a committed pre-registration.

    The report must be one hearth eval --report-json wrote on this install: it is
    HMAC-signed with a per-install key (HEARTH_HOME/eval-report.key, 0600), and an
    unsigned or edited report is refused. It must also be evidence for THIS adapter: same
    id, task, base model and weights path, weights on disk that still hash to what was
    measured, and an incumbent that is still the incumbent. The gate is then recomputed
    from the report's per-example vectors under the committed pre-registration's bar.
    The prereg must have been committed before the measurement started, in the git
    repository that holds the (committed) golden set. The usual one-step path is hearth
    eval ADAPTER --golden G --prereg P --promote. The old --candidate-score /
    --incumbent-score flags are removed: a typed score is not evidence.

    Examples:
      hearth adapters promote ADAPTER_ID --report r.json \\
        --prereg prereg/classify.yaml

    Env: HEARTH_HOME.

    Exit: 0 promoted; 1 refused (unsigned, edited or unusable report, a report about a
    different adapter or stale weights/incumbent, an uncommitted, late or mismatched
    prereg, gate failed); 2 the removed typed-score flags were used.
    """
    import json as _json

    from .config import Settings
    from .registry import AdapterError, GateNotPassedError
    from .training.attest import load_key, verify
    from .training.eval import EvalReport, GateProvenanceError, evaluate_gate
    from .training.prereg import (
        PreRegError,
        check_provenance,
        load_prereg,
        provenance_proof,
    )
    from .training.promotion import report_problems

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

    # The signature is checked before a single number in the report is read: an unsigned
    # or edited report is not evidence, whatever it claims (B-061; training/attest.py).
    try:
        payload = _json.loads(Path(report).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("the report is not a JSON object")
        report_sha = verify(payload, load_key(Settings().home))
        candidate = EvalReport.from_json(payload["candidate"])
        incumbent = EvalReport.from_json(payload["incumbent"])
        baselines = {
            name: EvalReport.from_json(obj)
            for name, obj in (payload.get("baselines") or {}).items()
        }
    except (OSError, KeyError, TypeError, ValueError) as exc:
        console.print(f"[red]Unusable eval report:[/red] {exc}")
        raise typer.Exit(code=1) from None

    store = _adapter_store()
    entry = store.get(adapter_id)
    if entry is None:
        console.print(f"[red]Unknown adapter:[/red] {adapter_id!r}")
        raise typer.Exit(code=1)
    problems = report_problems(
        payload,
        adapter_id=adapter_id,
        entry=entry,
        store=store,
        candidate=candidate,
        incumbent=incumbent,
    )
    if problems:
        console.print(
            f"[red]Promotion refused:[/red] the report is not evidence for {adapter_id!r} — "
            + "; ".join(problems)
        )
        raise typer.Exit(code=1)

    try:
        registration = load_prereg(prereg)
    except PreRegError as exc:
        console.print(f"[red]Pre-registration error:[/red] {exc}")
        raise typer.Exit(code=1) from None
    problems = list(registration.mismatches(candidate))
    if problems:
        console.print(
            "[red]Promotion refused:[/red] the report is not the registered experiment — "
            + "; ".join(problems)
        )
        raise typer.Exit(code=1)
    try:
        status = check_provenance(
            registration,
            measured_at=str(payload.get("measured_at") or ""),
            golden_git=dict(payload.get("golden_git") or {}),
        )
    except PreRegError as exc:
        console.print(f"[red]Promotion refused:[/red] {exc}")
        raise typer.Exit(code=1) from None

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
    proof.update(provenance_proof(status, dict(payload.get("golden_git") or {})))
    proof["measured_at"] = candidate.measured_at
    proof["candidate_weights_sha"] = payload.get("candidate_weights_sha")
    proof["evidence"] = "signed-report"
    proof["report_sha"] = report_sha
    try:
        store.promote(adapter_id, gate=gate, proof=proof)
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
    """Retire an adapter so it is no longer served, even if it was promoted.

    Examples:
      hearth adapters retire ADAPTER_ID

    Env: HEARTH_HOME.

    Exit: 0 retired; 1 unknown adapter.
    """
    from .registry import AdapterError

    try:
        _adapter_store().retire(adapter_id)
    except AdapterError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    console.print(f"[green]Retired[/green] {adapter_id}.")


@plugins_app.command("list")
def plugins_list() -> None:
    """List plugins registered under HEARTH's entry-point groups, and whether each loaded.

    Covers the hearth.providers, hearth.vector_stores and hearth.embedders groups. Status
    is 'ok' when the plugin imported and satisfied its Protocol, else why it was
    skipped; a broken plugin is reported here instead of crashing the server. Select one
    by name with HEARTH_BACKEND, HEARTH_VECTOR_STORE or HEARTH_EMBEDDER. See
    docs/PLUGINS.md.

    Examples:
      hearth plugins list
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
