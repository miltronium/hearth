"""The ``hearth`` command-line entrypoint.

Phase 0/1 commands:
  * ``hearth doctor``       — environment preflight
  * ``hearth serve``        — start the OpenAI-compatible gateway
  * ``hearth run``          — one-shot local completion (``--file``, ``--intent``)
  * ``hearth models …``     — registry: ``list`` / ``pull`` / ``rm`` / ``convert``
  * ``hearth rag …``        — local RAG: ``ingest`` / ``query`` (Phase 3)
  * ``hearth train …``      — LoRA fine-tune → register a candidate adapter (Phase 4)
  * ``hearth adapters …``   — adapter registry: ``list`` / ``promote`` / ``retire`` (Phase 4)
  * ``hearth plugins list`` — third-party plugins discovered via entry points (Phase 7)
  * ``hearth mcp``          — MCP server for agent offload (Phase 5, needs ``[mcp]`` extra)
  * ``hearth stats``        — token-savings / escalation rollups (Phase 2)
  * ``hearth version``      — print version
"""

from __future__ import annotations

import sys
from pathlib import Path

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
plugins_app = typer.Typer(help="Third-party plugins discovered via entry points (Phase 7).")
app.add_typer(plugins_app, name="plugins")
console = Console()


def _adapter_store():
    """Build an :class:`AdapterStore` under the current ``HEARTH_HOME``.

    Reads a fresh :class:`Settings` (not the process-cached one) so a caller/test that
    sets ``HEARTH_HOME`` for a single invocation gets an isolated store.
    """
    from .config import Settings
    from .registry import AdapterStore

    return AdapterStore(settings=Settings())


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
    candidate_score: float = typer.Option(
        None, "--candidate-score", help="Candidate eval score (proves it beat the incumbent)."
    ),
    incumbent_score: float = typer.Option(
        None, "--incumbent-score", help="Incumbent eval score (omit if no incumbent)."
    ),
) -> None:
    """Promote a candidate — refused unless the eval gate passed (ARCHITECTURE §7, ADR-006).

    Promotion requires proof the candidate beat the incumbent. Supply ``--candidate-score``
    (and ``--incumbent-score`` when one exists); the gate (``beats_incumbent``) must pass or
    this refuses. This is the safety guarantee: a bare "promote" never wins.
    """
    from .registry import AdapterError, GateNotPassedError
    from .training.eval import EvalReport, beats_incumbent

    if candidate_score is None:
        console.print("[red]--candidate-score is required to prove the eval gate passed.[/red]")
        raise typer.Exit(code=1)

    candidate = EvalReport(task="", metric="cli", score=candidate_score)
    incumbent = (
        EvalReport(task="", metric="cli", score=incumbent_score)
        if incumbent_score is not None
        else None
    )
    gate_passed = beats_incumbent(candidate, incumbent)
    proof = {
        "candidate_score": candidate_score,
        "incumbent_score": incumbent_score,
        "gate_passed": gate_passed,
    }
    try:
        _adapter_store().promote(adapter_id, gate_passed=gate_passed, proof=proof)
    except GateNotPassedError as exc:
        console.print(f"[red]Promotion refused:[/red] {exc}")
        raise typer.Exit(code=1) from None
    except AdapterError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    console.print(f"[green]Promoted[/green] {adapter_id} (gate passed).")


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
