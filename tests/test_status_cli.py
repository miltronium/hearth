"""Rendering and the ``scripts/hearth_status.py`` entry point.

The rendering assertions are about honesty rather than looks: an ``unverified`` field must
be as visible as a passing one, and every section must print what it did *not* measure. A
report that renders its gaps invisibly is a report that gets read as a clean bill of health,
which is the failure mode this package was written to end.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from hearth.status import collect_status, render_text
from hearth.status.render import render_section
from hearth.status.report import (
    LEVEL_FAIL,
    LEVEL_OK,
    LEVEL_UNVERIFIED,
    LEVEL_WARN,
    Fact,
    Section,
    StatusReport,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "hearth_status.py"


# ---------------------------------------------------------------------------------------
# report model
# ---------------------------------------------------------------------------------------


def test_a_fact_rejects_an_invented_level():
    with pytest.raises(ValueError, match="unknown level"):
        Fact("x", "y", "probably-fine")


def test_worst_ranks_fail_above_warn_above_unverified():
    def report(*levels: str) -> StatusReport:
        facts = tuple(Fact(f"f{i}", "v", lvl) for i, lvl in enumerate(levels))
        return StatusReport("now", "/tmp", (Section("s", "S", facts),))

    assert report(LEVEL_OK).worst() == LEVEL_OK
    assert report(LEVEL_OK, LEVEL_UNVERIFIED).worst() == LEVEL_UNVERIFIED
    assert report(LEVEL_UNVERIFIED, LEVEL_WARN).worst() == LEVEL_WARN
    assert report(LEVEL_WARN, LEVEL_FAIL).worst() == LEVEL_FAIL
    assert StatusReport("now", "/tmp", ()).worst() == LEVEL_OK


def test_json_round_trips_through_the_stdlib_encoder():
    report = StatusReport(
        "2026-01-01T00:00:00+00:00",
        "/repo",
        (Section("s", "S", (Fact("n", "v", LEVEL_WARN, "d", {"k": 1}),), ("nope",)),),
    )
    decoded = json.loads(json.dumps(report.to_json()))
    assert decoded["worst"] == LEVEL_WARN
    assert decoded["sections"][0]["facts"][0]["data"] == {"k": 1}
    assert decoded["sections"][0]["limits"] == ["nope"]


# ---------------------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------------------


def test_an_unverified_field_is_never_rendered_as_a_pass():
    section = Section(
        "s", "Title", (Fact("thing", "unmeasured", LEVEL_UNVERIFIED, "go look yourself"),)
    )
    text = "\n".join(render_section(section))
    assert "unver" in text
    assert "go look yourself" in text


def test_every_section_renders_what_it_did_not_measure():
    section = Section("s", "Title", (Fact("a", "b"),), ("the firewall was not inspected",))
    text = "\n".join(render_section(section))
    assert "not measured: the firewall was not inspected" in text


def test_render_text_tallies_levels_and_names_the_unmeasured_count():
    report = StatusReport(
        "2026-01-01T00:00:00+00:00",
        "/repo",
        (
            Section("a", "A", (Fact("x", "1"), Fact("y", "2", LEVEL_WARN))),
            Section("b", "B", (Fact("z", "3", LEVEL_UNVERIFIED),)),
        ),
    )
    text = render_text(report)
    assert "ok=1" in text and "warn=1" in text and "unverified=1" in text
    assert "worst=warn" in text
    assert "1 field(s) could not be measured" in text
    assert "not a claim copied from a doc" in text


def test_long_values_wrap_instead_of_running_off_the_terminal():
    long_value = " ".join(["escalates"] * 40)
    section = Section("s", "T", (Fact("profile", long_value),))
    lines = render_section(section)
    assert all(len(line) <= 100 for line in lines), "output must stay inside a sane width"


# ---------------------------------------------------------------------------------------
# collect_status wiring
# ---------------------------------------------------------------------------------------


def test_collect_status_returns_all_six_sections(tmp_path: Path):
    report = collect_status(root=tmp_path, home=tmp_path / "home", environ={})
    assert [s.key for s in report.sections] == [
        "models",
        "egress",
        "learning",
        "tests",
        "environment",
        "staleness",
    ]
    assert report.generated_at.endswith("+00:00")


def test_collect_status_survives_an_empty_directory(tmp_path: Path):
    """A status tool must work on a broken machine — that is when it is needed."""
    report = collect_status(root=tmp_path, home=tmp_path / "nothing-here", environ={})
    assert render_text(report)
    assert report.facts(level=LEVEL_UNVERIFIED), "gaps must be reported, not hidden"


# ---------------------------------------------------------------------------------------
# the script
# ---------------------------------------------------------------------------------------


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )


def test_script_help_is_clean():
    result = _run("--help")
    assert result.returncode == 0, result.stderr
    assert "read-only" in result.stdout


def test_script_emits_parseable_json():
    result = _run("--json", "--section", "learning", "staleness")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert [s["key"] for s in payload["sections"]] == ["learning", "staleness"]
    assert payload["worst"] in ("ok", "warn", "fail", "unverified")


def test_script_rejects_an_unknown_section():
    result = _run("--section", "wishful-thinking")
    assert result.returncode != 0
    assert "unknown section" in result.stderr


def test_script_human_output_names_the_measurement():
    result = _run("--section", "learning")
    assert result.returncode == 0, result.stderr
    assert "HEARTH status — measured" in result.stdout
    assert "not measured:" in result.stdout


def test_strict_mode_exits_nonzero_when_something_measured_badly(tmp_path: Path):
    result = _run("--strict", "--repo", str(tmp_path))
    assert result.returncode == 1
    assert "FAIL" in result.stdout or "warn" in result.stdout


def test_strict_mode_is_not_tripped_by_an_unverified_field(tmp_path: Path):
    """An unmeasured thing is not a broken thing.

    Conflating the two is how a CI signal gets ignored: if ``--strict`` went red every time
    the tool honestly admitted it could not see something, operators would learn to pass
    ``--no-strict`` and stop reading the report at all.
    """
    result = _run("--strict", "--repo", str(tmp_path), "--section", "tests")
    assert result.returncode == 0, result.stdout
    assert "unver" in result.stdout
