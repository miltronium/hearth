"""The ``hearth`` command-line entrypoint.

Phase 0/1 commands:
  * ``hearth doctor``       — environment preflight
  * ``hearth serve``        — start the OpenAI-compatible gateway
  * ``hearth run``          — one-shot local completion (``--file``, ``--intent``)
  * ``hearth models …``     — registry: ``list`` / ``pull`` / ``rm``
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

app = typer.Typer(
    name="hearth",
    help="On-device intelligence layer — a local-first model gateway for Apple Silicon.",
    no_args_is_help=True,
    add_completion=False,
)
models_app = typer.Typer(help="Model registry: list, pull, and remove models.")
app.add_typer(models_app, name="models")
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
    result = provider.generate(
        GenRequest(
            messages=[Message(role="user", content=text)],
            model=get_registry().default_id,
            max_tokens=max_tokens,
        )
    )
    console.print(result.text, markup=False, highlight=False)


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


if __name__ == "__main__":
    app()
