"""The committed man page is the one the CLI generates, and it covers every command and var.

WHAT: ``scripts/gen_manpage.py`` renders ``man/hearth.1`` from the live Typer app and
``hearth.config.Settings``. These tests regenerate it in memory and compare it with the
committed file, so a command or option that changed without regenerating fails here, with
the fix in the message. They also check the page names every command, every visible option
and every ``HEARTH_*`` variable HEARTH reads, documents each one (no UNDOCUMENTED stubs),
and renders without errors when ``mandoc`` is available.

Regenerate with:  uv run --no-sync python scripts/gen_manpage.py
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MAN = REPO / "man" / "hearth.1"


def _load_generator():
    path = REPO / "scripts" / "gen_manpage.py"
    spec = importlib.util.spec_from_file_location("gen_manpage", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules.setdefault("gen_manpage", module)
    spec.loader.exec_module(module)
    return module


gen = _load_generator()


def _plain(roff: str) -> str:
    """Undo the escapes the generator applies, so names can be searched as typed."""
    return roff.replace("\\-", "-").replace("\\e", "\\").replace("\\fB", "").replace(
        "\\fI", ""
    ).replace("\\fR", "").replace("\\&", "")


@pytest.fixture(scope="module")
def rendered() -> str:
    return gen.render()


def test_committed_man_page_matches_a_fresh_render(rendered):
    assert MAN.exists(), "man/hearth.1 is missing: run scripts/gen_manpage.py"
    committed = MAN.read_text(encoding="utf-8")
    assert committed == rendered, (
        "man/hearth.1 is stale (a command, option, default or help text changed). "
        "Regenerate: uv run --no-sync python scripts/gen_manpage.py"
    )


def test_render_is_deterministic(rendered):
    assert gen.render() == rendered
    assert str(Path.home()) not in rendered, "a machine path leaked into the man page"


def test_every_command_and_visible_option_is_in_the_page(rendered):
    text = _plain(rendered)
    commands = list(gen.iter_commands())
    assert len(commands) >= 20, "the walk found too few commands to be the real CLI"
    for path, cmd in commands:
        name = " ".join(path)
        assert f'.SS "{name}"' in text, f"{name} has no COMMANDS entry"
        for param in cmd.params:
            if getattr(param, "hidden", False):
                continue
            for opt in [*getattr(param, "opts", []), *getattr(param, "secondary_opts", [])]:
                if opt.startswith("-"):
                    assert opt in text, f"{name}: option {opt} missing from the man page"


def test_every_hearth_env_var_is_in_the_page_and_documented(rendered):
    from hearth.config import Settings
    from hearth.status.probes import _EXTRA_ENV_NAMES

    text = _plain(rendered)
    names = {f"HEARTH_{f.upper()}" for f in Settings.model_fields} | set(_EXTRA_ENV_NAMES)
    assert len(names) >= 20
    env_section = text.split(".SH ENVIRONMENT", 1)[1].split(".SH FILES", 1)[0]
    for name in sorted(names):
        assert name in env_section, f"{name} missing from ENVIRONMENT"
    assert "UNDOCUMENTED" not in rendered, (
        "a Settings field or env var has no description: add it to SETTINGS_DOCS / "
        "EXTRA_DOCS in scripts/gen_manpage.py"
    )


def test_hand_written_sections_are_present(rendered):
    for section in (
        "NAME",
        "SYNOPSIS",
        "DESCRIPTION",
        "QUICK START",
        "COMMANDS",
        "PRIVACY AND OFFLINE USE",
        "ENVIRONMENT",
        "FILES",
        "EXIT STATUS",
        "EXAMPLES",
        "SEE ALSO",
    ):
        assert f".SH {section}\n" in rendered, section


def test_generator_check_mode_detects_drift(tmp_path):
    stale = tmp_path / "hearth.1"
    stale.write_text(gen.render().replace("Print the HEARTH version.", "stale"), "utf-8")
    assert gen.main(["--check", "--out", str(stale)]) == 1
    stale.write_text(gen.render(), "utf-8")
    assert gen.main(["--check", "--out", str(stale)]) == 0


@pytest.mark.skipif(shutil.which("mandoc") is None, reason="mandoc not installed")
def test_mandoc_renders_it_without_errors():
    lint = subprocess.run(
        ["mandoc", "-T", "lint", "-W", "error", str(MAN)], capture_output=True, text=True
    )
    assert lint.returncode < 3, lint.stdout + lint.stderr  # 2 = warnings, 3+ = errors
    out = subprocess.run(["mandoc", "-T", "utf8", str(MAN)], capture_output=True, text=True)
    assert out.returncode == 0
    assert "hearth - local-first, no-egress LLM gateway" in out.stdout.replace("‐", "-")
