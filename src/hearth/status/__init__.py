"""Measured project status — what is *actually* true on this machine, right now.

HEARTH's institutional memory used to live in hand-written docs (``docs/RESULTS.md``,
``docs/cmux/HANDOFF.md``, ``docs/cmux/TODO.md``). Those record what somebody *believed*
when they typed them, and they rot silently: a "verified" claim stays green long after the
thing it describes stopped being true. This package replaces the belief with a probe.

The rule it is built on, learned the hard way six times over (see ``docs/STATUS.md``):

    **A gate must assert on the OUTCOME, never on a CONFIGURATION that implies it.**

So every field here is a measurement of an outcome — bytes on disk, a policy object after
the loader actually resolved it, lines in a golden file, a value returned by the GPU driver
— and where a thing genuinely cannot be measured from here, the report says ``unverified``
instead of asserting. A status line that cannot be wrong is worth more than ten that
merely have not been checked lately.

**Read-only invariant.** This package MUST never write, mutate, or transmit anything. It
opens no sockets, imports no HTTP client, creates no files or directories, and mutates no
state — running ``hearth_status.py`` against a broken machine must never be able to make it
worse, so that a nervous operator can always run it first. The one system call it makes is
read-only ``git`` plumbing (:mod:`hearth.status.gitmeta`), because history is a fact that
lives in ``.git`` and nowhere else. ``tests/test_status_readonly.py`` enforces all of this
by parsing this package's own source, so the invariant fails in CI rather than in
production.

Layout:
  * :mod:`hearth.status.report`  — the shape of a report (facts, sections, levels).
  * :mod:`hearth.status.gitmeta` — the single, allowlisted ``git`` call site.
  * :mod:`hearth.status.probes`  — the six probes that do the measuring.
  * :mod:`hearth.status.render`  — human-readable rendering (``--json`` uses ``to_json``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from .probes import (
    probe_egress,
    probe_environment,
    probe_learning,
    probe_models,
    probe_staleness,
    probe_tests,
)
from .render import render_text
from .report import (
    LEVEL_FAIL,
    LEVEL_OK,
    LEVEL_UNVERIFIED,
    LEVEL_WARN,
    Fact,
    Section,
    StatusReport,
)


def repo_root() -> Path:
    """The HEARTH checkout this package was imported from (four parents up)."""
    return Path(__file__).resolve().parents[3]


def collect_status(
    *,
    root: Path | None = None,
    home: Path | None = None,
    environ: dict[str, str] | None = None,
) -> StatusReport:
    """Run every probe and return the assembled report.

    ``root`` (repo checkout), ``home`` (``~/.hearth``) and ``environ`` are injectable so
    tests can probe a fixture tree instead of the operator's real machine. A probe that
    cannot measure its subject reports ``unverified`` facts rather than raising: a partial
    status report is useful, a traceback is not.
    """
    root = Path(root) if root is not None else repo_root()
    env = dict(environ) if environ is not None else None

    sections = (
        probe_models(root=root, home=home, environ=env),
        probe_egress(root=root, environ=env),
        probe_learning(root=root, home=home, environ=env),
        probe_tests(root=root),
        probe_environment(environ=env),
        probe_staleness(root=root),
    )
    return StatusReport(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        root=str(root),
        sections=sections,
    )


__all__ = [
    "Fact",
    "Section",
    "StatusReport",
    "LEVEL_OK",
    "LEVEL_WARN",
    "LEVEL_FAIL",
    "LEVEL_UNVERIFIED",
    "collect_status",
    "render_text",
    "repo_root",
    "probe_models",
    "probe_egress",
    "probe_learning",
    "probe_tests",
    "probe_environment",
    "probe_staleness",
]
