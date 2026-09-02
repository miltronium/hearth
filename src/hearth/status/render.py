"""Human rendering of a :class:`~hearth.status.report.StatusReport`.

Plain text on purpose — no ``rich``, no colour, no dependency. This output is meant to be
piped into a terminal, an issue, or another agent's context window unchanged, and a status
report that only reads correctly inside one TTY is a status report that gets screenshotted
instead of quoted.

Two rendering rules carry meaning rather than taste:

  * every fact prints its confidence level, so ``unverified`` is as visible as ``ok`` and
    an unmeasured field can never be skimmed as a passing one;
  * every section prints its ``limits``, so a reader finishes each block knowing what was
    *not* measured. That is the part a hand-written status doc always omits.
"""

from __future__ import annotations

import textwrap

from .report import (
    LEVEL_FAIL,
    LEVEL_OK,
    LEVEL_UNVERIFIED,
    LEVEL_WARN,
    LEVELS,
    Section,
    StatusReport,
)

_TAGS = {
    LEVEL_OK: "ok   ",
    LEVEL_WARN: "warn ",
    LEVEL_FAIL: "FAIL ",
    LEVEL_UNVERIFIED: "unver",
}

_WIDTH = 98
_INDENT = "        "


def _wrap(text: str, *, first: str = _INDENT, rest: str = _INDENT) -> list[str]:
    return textwrap.wrap(text, width=_WIDTH, initial_indent=first, subsequent_indent=rest)


def render_section(section: Section) -> list[str]:
    """Render one section: title, its facts, then what it did not measure."""
    lines = [section.title, "-" * min(len(section.title), _WIDTH)]
    if not section.facts:
        lines.append(f"{_INDENT}(nothing measured)")
    for fact in section.facts:
        prefix = f"  [{_TAGS.get(fact.level, fact.level)}] "
        lines.extend(_wrap(f"{fact.name}: {fact.value}", first=prefix, rest=_INDENT))
        if fact.detail:
            lines.extend(_wrap(fact.detail))
    for limit in section.limits:
        lines.extend(_wrap(limit, first="  not measured: ", rest="                 "))
    return lines


def render_text(report: StatusReport) -> str:
    """Render the whole report as plain text, ending with a level tally.

    The tally is deliberately last and deliberately blunt: the number of ``unverified``
    fields is the honest measure of how much of this report is a measurement and how much
    is an absence of one.
    """
    counts = {lvl: len(report.facts(level=lvl)) for lvl in LEVELS}
    out: list[str] = [
        f"HEARTH status — measured {report.generated_at}",
        f"repo: {report.root}",
        "",
        "Every line below is a measurement taken just now, not a claim copied from a doc.",
        "",
    ]
    for section in report.sections:
        out.extend(render_section(section))
        out.append("")

    out.append("Summary")
    out.append("-------")
    out.append(
        "  " + "  ".join(f"{lvl}={counts[lvl]}" for lvl in LEVELS) + f"  worst={report.worst()}"
    )
    out.extend(
        _wrap(
            f"{counts[LEVEL_UNVERIFIED]} field(s) could not be measured from here — see the "
            "'not measured' lines above for what a human still has to check.",
            first="  ",
            rest="  ",
        )
    )
    return "\n".join(out)


__all__ = ["render_text", "render_section"]
