"""Every command and option documents itself, and the rendered ``--help`` actually shows it.

WHY: help text that exists in the source but never reaches the terminal is the CLAUDE.md §3
shape (the check and the checked thing are different objects). So these tests walk the real
Typer app, render ``hearth ... --help`` through the CLI runner, and assert on the OUTPUT:

* every command and group has a docstring, and its summary line is in its rendered help;
* every visible option and argument has help text, and that text is in the rendered help;
* every leaf command's help carries an ``Examples:`` block with a runnable ``hearth`` line;
* the commands whose exit status means something (doctor, agent, eval, prereg check, train,
  serve) state it, and ``serve --help`` explains the routing profile, token, /chat and /ready;
* ``hearth --help`` points at the man page, the guide and the offline check.

Removing a docstring or a ``help=`` string, or hiding it from the renderer, fails here.
"""

from __future__ import annotations

import inspect
import re

import pytest
import typer
from typer.testing import CliRunner

from hearth.cli import app

runner = CliRunner()

#: Wide enough that rich never wraps a help string mid-sentence; the normaliser below also
#: copes with wrapping, so this is belt and braces rather than a precondition.
WIDE = {"COLUMNS": "300", "TERMINAL_WIDTH": "300", "NO_COLOR": "1"}

_BOX = re.compile(r"[│┃╭╮╰╯─━┏┓┗┛┡┩╇┳┻┃╷╵]")


def _norm(text: str) -> str:
    """Collapse rich's box drawing and wrapping so a help string reads contiguously."""
    return " ".join(_BOX.sub(" ", text).split())


def _commands():
    """(argv path, click command) for every command and group under ``hearth``."""
    root = typer.main.get_command(app)

    def walk(cmd, path):
        yield path, cmd
        for name, sub in getattr(cmd, "commands", {}).items():
            if not getattr(sub, "hidden", False):
                yield from walk(sub, [*path, name])

    for name, cmd in root.commands.items():
        yield from walk(cmd, [name])


COMMANDS = list(_commands())
LEAVES = [(p, c) for p, c in COMMANDS if not hasattr(c, "commands")]
IDS = [" ".join(p) for p, _ in COMMANDS]
LEAF_IDS = [" ".join(p) for p, _ in LEAVES]


def _help(path: list[str]) -> str:
    result = runner.invoke(app, [*path, "--help"], env=WIDE)
    assert result.exit_code == 0, result.output
    return result.output


def test_the_walk_found_the_cli():
    """Guard the guard: a walk that found nothing would pass every test below."""
    names = set(IDS)
    for expected in ("doctor", "serve", "run", "agent", "eval", "models pull", "prereg check"):
        assert expected in names
    assert len(LEAVES) >= 20


@pytest.mark.parametrize(("path", "cmd"), COMMANDS, ids=IDS)
def test_every_command_has_a_summary_that_renders(path, cmd):
    doc = inspect.cleandoc(cmd.help or "")
    assert doc, f"hearth {' '.join(path)} has no docstring / help"
    summary = doc.split("\n\n")[0].replace("\n", " ").strip()
    assert len(summary) <= 90, f"summary line too long for the command list: {summary!r}"
    assert _norm(summary) in _norm(_help(path))


@pytest.mark.parametrize(("path", "cmd"), COMMANDS, ids=IDS)
def test_every_option_and_argument_has_help_that_renders(path, cmd):
    rendered = _norm(_help(path))
    for param in cmd.params:
        if getattr(param, "hidden", False):
            continue
        text = (getattr(param, "help", None) or "").strip()
        assert text, f"hearth {' '.join(path)}: {param.name} has no help text"
        assert _norm(text) in rendered, f"{param.name}'s help is not in the rendered --help"


@pytest.mark.parametrize(("path", "cmd"), LEAVES, ids=LEAF_IDS)
def test_every_command_shows_a_runnable_example(path, cmd):
    rendered = _help(path)
    assert "Examples:" in rendered
    after = rendered.split("Examples:", 1)[1]
    assert re.search(rf"\bhearth {re.escape(path[0])}\b", after), (
        f"hearth {' '.join(path)} --help has no example invoking it"
    )


@pytest.mark.parametrize(
    "path",
    [["doctor"], ["agent"], ["eval"], ["prereg", "check"], ["train"], ["serve"]],
    ids=lambda p: " ".join(p),
)
def test_commands_with_meaningful_exit_codes_state_them(path):
    assert "Exit" in _help(path)


def test_doctor_help_states_both_verdict_codes():
    rendered = _norm(_help(["doctor"]))
    assert "0 SAFE" in rendered and "1 UNSAFE" in rendered


def test_serve_help_explains_profile_token_chat_and_ready():
    rendered = _norm(_help(["serve"]))
    for needle in (
        "HEARTH_ROUTING_YAML",
        "zero remote models",
        "~/.hearth/token",
        "http://127.0.0.1:8080/chat",
        "/v1/hearth/admin/ready",
        "Bearer",
    ):
        assert needle in rendered, needle


def test_top_level_help_points_at_man_page_guide_and_offline_check():
    rendered = _norm(runner.invoke(app, ["--help"], env=WIDE).output)
    assert "man ./man/hearth.1" in rendered
    assert "docs/GUIDE.md" in rendered
    assert "hearth doctor --offline" in rendered
    assert "uv run --no-sync" in rendered


# --- Help reflows to the terminal (an 80-column terminal used to show ragged prose) ---------

NARROW_COLS = 80
NARROW = {"COLUMNS": str(NARROW_COLS), "TERMINAL_WIDTH": str(NARROW_COLS), "NO_COLOR": "1"}
ALL_PAGES = [([], None), *COMMANDS]
ALL_PAGE_IDS = ["hearth", *IDS]


@pytest.fixture
def narrow(monkeypatch):
    """Render at 80 columns. Typer reads TERMINAL_WIDTH once, when ``typer.rich_utils`` is
    first imported, so whichever test rendered first would otherwise fix the width for all of
    them (the wide tests above run first: the narrow checks then passed at 300 columns)."""
    import typer.rich_utils

    monkeypatch.setattr(typer.rich_utils, "MAX_WIDTH", NARROW_COLS)


def _narrow_body(path: list[str]) -> list[str]:
    """The rendered description lines of ``hearth PATH --help`` at 80 columns.

    Everything above the first panel box, minus the Usage line; trailing padding stripped.
    """
    result = runner.invoke(app, [*path, "--help"], env=NARROW)
    assert result.exit_code == 0, result.output
    widest = max(len(ln) for ln in result.output.split("\n"))
    assert widest <= NARROW_COLS, f"rendered {widest} columns wide, not {NARROW_COLS}"
    body: list[str] = []
    for line in result.output.split("\n"):
        if line.lstrip().startswith("╭"):
            break
        body.append(line.rstrip())
    return [ln for ln in body if not ln.strip().startswith("Usage:")]


def _is_block(line: str) -> bool:
    # rich pads the help by one column; a docstring's indented block line has two more.
    return line.startswith("   ")


@pytest.mark.parametrize(("path", "cmd"), ALL_PAGES, ids=ALL_PAGE_IDS)
def test_prose_breaks_only_where_the_next_word_does_not_fit(path, cmd, narrow):
    """Greedy-wrap invariant: inside a prose paragraph a line ends only because the next
    word would not fit. A hard newline kept from the docstring breaks it: at 80 columns the
    source line wraps and its last word or two strand on a short line ("with a" / "chat
    page"), and the next line's first word would have fitted after them.
    """
    body = _narrow_body(path)
    assert sum(1 for ln in body if ln.strip()) >= 1, "nothing rendered: the check is vacuous"
    usable = NARROW_COLS - 1  # rich keeps the last column free
    for here, nxt in zip(body, body[1:], strict=False):
        if not here.strip() or not nxt.strip() or _is_block(here) or _is_block(nxt):
            continue
        first_word = nxt.split()[0]
        assert len(here) + 1 + len(first_word) > usable, (
            f"hearth {' '.join(path)} --help breaks prose early at 80 columns:\n"
            f"  {here!r}\n  {nxt!r}\n({first_word!r} fits on the first line)"
        )


@pytest.mark.parametrize(("path", "cmd"), ALL_PAGES, ids=ALL_PAGE_IDS)
def test_example_and_command_lines_each_keep_their_own_line(path, cmd, narrow):
    """Every indented block line in the docstring (examples, the learn-in-order list) is
    rendered as exactly one terminal line at 80 columns: not joined, not wrapped."""
    raw = (typer.main.get_command(app).help if cmd is None else cmd.help) or ""
    block = [ln.strip() for ln in inspect.cleandoc(raw).split("\n") if ln[:1].isspace()]
    rendered = [ln.strip() for ln in _narrow_body(path)]
    for want in block:
        assert want in rendered, (
            f"hearth {' '.join(path)} --help: block line not on its own line at 80 columns: "
            f"{want!r}"
        )


def test_reflow_joins_prose_and_keeps_blocks():
    from hearth.cli import reflow_help

    raw = "Summary.\n\nalpha beta gamma\ndelta epsilon\n\nExamples:\n  hearth x\n  hearth y"
    out = reflow_help(raw)
    assert "alpha beta gamma delta epsilon" in out
    assert "Examples:\n  hearth x\n  hearth y" in out
    assert reflow_help(out) == out
