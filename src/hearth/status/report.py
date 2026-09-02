"""The shape of a measured status report.

Three nested, frozen records — :class:`Fact` inside :class:`Section` inside
:class:`StatusReport` — with one deliberate design constraint: **a fact carries its own
confidence**. ``level`` is not decoration, it is the difference between "I measured this
and it is fine" (``ok``), "I measured this and it is wrong" (``warn``/``fail``), and "I
could not measure this from here, go look yourself" (``unverified``). A report that cannot
say the third thing ends up asserting the first, which is how status docs rot.

``to_json`` on each record gives the ``--json`` shape; ``data`` carries the structured
payload (counts, paths, ids) so a machine consumer never has to parse ``value`` prose.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Confidence levels. Ordered worst-first for reporting; see :meth:`StatusReport.worst`.
LEVEL_FAIL = "fail"
LEVEL_WARN = "warn"
LEVEL_UNVERIFIED = "unverified"
LEVEL_OK = "ok"
LEVELS = (LEVEL_FAIL, LEVEL_WARN, LEVEL_UNVERIFIED, LEVEL_OK)


@dataclass(frozen=True)
class Fact:
    """One measured thing.

    ``value`` is the short human answer, ``detail`` the sentence that explains it (why it
    matters, what to do), and ``data`` the machine-readable payload behind both.
    """

    name: str
    value: str
    level: str = LEVEL_OK
    detail: str = ""
    data: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise ValueError(f"unknown level {self.level!r} (expected one of {LEVELS})")

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "level": self.level,
            "detail": self.detail,
            "data": dict(self.data),
        }


@dataclass(frozen=True)
class Section:
    """One probe's output: its facts, plus the caveats a reader must carry away.

    ``limits`` is mandatory in spirit: every probe states what it did *not* measure, so a
    reader never mistakes silence for a clean bill of health.
    """

    key: str
    title: str
    facts: tuple[Fact, ...] = ()
    limits: tuple[str, ...] = ()

    def to_json(self) -> dict:
        return {
            "key": self.key,
            "title": self.title,
            "facts": [f.to_json() for f in self.facts],
            "limits": list(self.limits),
        }


@dataclass(frozen=True)
class StatusReport:
    """Every probe's output plus when and where it was taken."""

    generated_at: str
    root: str
    sections: tuple[Section, ...] = ()

    def section(self, key: str) -> Section | None:
        """Return the section with ``key``, or ``None``."""
        return next((s for s in self.sections if s.key == key), None)

    def facts(self, *, level: str | None = None) -> list[Fact]:
        """Every fact in the report, optionally filtered to one ``level``."""
        out = [f for s in self.sections for f in s.facts]
        return [f for f in out if f.level == level] if level else out

    def worst(self) -> str:
        """The most severe level present, ``ok`` if nothing worse was measured."""
        present = {f.level for f in self.facts()}
        return next((lvl for lvl in LEVELS if lvl in present), LEVEL_OK)

    def to_json(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "root": self.root,
            "worst": self.worst(),
            "sections": [s.to_json() for s in self.sections],
        }


__all__ = [
    "Fact",
    "Section",
    "StatusReport",
    "LEVELS",
    "LEVEL_OK",
    "LEVEL_WARN",
    "LEVEL_FAIL",
    "LEVEL_UNVERIFIED",
]
