"""The ``hearth`` command-line entrypoint.

Phase 0/1 commands:
  * ``hearth doctor``       — environment preflight
  * ``hearth serve``        — start the OpenAI-compatible gateway
  * ``hearth run``          — one-shot local completion (``--file``, ``--intent``)
  * ``hearth models …``     — registry: ``list`` / ``pull`` / ``rm``
  * ``hearth rag …``        — local RAG: ``ingest`` / ``query`` (Phase 3)
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
models_app = typer.Typer(help="Model registry: list, pull, and remove models.")
app.add_typer(models_app, name="models")
rag_app = typer.Typer(help="Local RAG: ingest paths into a collection and query them.")
app.add_typer(rag_app, name="rag")
console = Console()


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


if __name__ == "__main__":
    app()
