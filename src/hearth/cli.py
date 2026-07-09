"""The ``hearth`` command-line entrypoint.

Phase 0 commands:
  * ``hearth doctor``  — environment preflight
  * ``hearth serve``   — start the OpenAI-compatible gateway
  * ``hearth run``     — one-shot local completion
  * ``hearth version`` — print version
"""

from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import ensure_home, get_or_create_token, get_settings
from .doctor import all_fatal_passed, run_checks
from .providers import select_provider
from .providers.base import GenRequest, Message

app = typer.Typer(
    name="hearth",
    help="On-device intelligence layer — a local-first model gateway for Apple Silicon.",
    no_args_is_help=True,
    add_completion=False,
)
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
    get_or_create_token(settings)  # ensure a token exists for Phase 1 auth
    provider = select_provider(settings)

    bind_host = host or settings.host
    bind_port = port or settings.port
    console.print(
        f"[bold]HEARTH[/bold] {__version__} — backend=[cyan]{provider.name}[/cyan] "
        f"model=[cyan]{settings.default_model}[/cyan]"
    )
    console.print(f"Serving on http://{bind_host}:{bind_port}  (OpenAI-compatible /v1)")
    uvicorn.run(create_app(provider=provider, settings=settings), host=bind_host, port=bind_port)


@app.command()
def run(
    prompt: str = typer.Argument(None, help="Prompt text. Omit to read from stdin."),
    max_tokens: int = typer.Option(512, help="Max tokens to generate."),
) -> None:
    """Run a one-shot local completion and print the result."""
    text = prompt if prompt is not None else sys.stdin.read()
    if not text.strip():
        console.print("[red]No prompt provided.[/red]")
        raise typer.Exit(code=1)

    settings = get_settings()
    provider = select_provider(settings)
    result = provider.generate(
        GenRequest(
            messages=[Message(role="user", content=text)],
            model=settings.default_model,
            max_tokens=max_tokens,
        )
    )
    console.print(result.text, markup=False, highlight=False)


if __name__ == "__main__":
    app()
