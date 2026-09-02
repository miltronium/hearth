"""Redaction and human review — seeing exactly what would cross before it does.

The control that keeps a handoff honest is not the regex list below; it is a human reading
the payload. This module exists to make that reading *easy and complete*:
:func:`render_review` prints the entire envelope — every byte that would leave — in a form a
person can scan in one screen, and :func:`approve` binds the resulting decision to the
content hash so the approval cannot outlive the content it approved.

Redaction is an **aid to review, not a guarantee**. The default rules catch shapes that are
obviously secret (keys, tokens, home paths, email addresses); they cannot recognise a project
codename, an unreleased product, or a sentence that is confidential because of what it
implies. Treat a clean redaction report as "nothing obvious was found", never as "this is
safe to send" — see ``docs/TIERS.md`` §5.

No network code lives here (or anywhere in this package); redaction runs on a string in
memory and returns a string in memory.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from .envelope import (
    DECISION_APPROVED,
    DECISION_REJECTED,
    HandoffEnvelope,
    ReviewRecord,
    now_iso,
)

# Replacement marker. Uniform on purpose: a reviewer scanning for "did anything get removed?"
# looks for one token, and the rule name in the report says what kind.
MASK = "[REDACTED]"


@dataclass(frozen=True)
class RedactionRule:
    """One named pattern to mask. ``pattern`` is a regex applied with :func:`re.sub`."""

    name: str
    pattern: str
    replacement: str = MASK

    def apply(self, text: str) -> tuple[str, int]:
        """Return ``(masked_text, hit_count)``."""
        masked, count = re.subn(self.pattern, self.replacement, text)
        return masked, count


def _home_pattern() -> str:
    """Match the operator's home directory path (usernames leak org and identity)."""
    return re.escape(os.path.expanduser("~"))


DEFAULT_RULES: tuple[RedactionRule, ...] = (
    RedactionRule(
        "private_key_block",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----",
    ),
    RedactionRule("aws_access_key", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    RedactionRule(
        "keyed_secret",
        r"(?i)\b(?:api[_-]?key|secret|token|password|passwd|bearer)\b\s*[:=]\s*\S+",
    ),
    RedactionRule("long_hex_token", r"\b[0-9a-fA-F]{32,}\b"),
    RedactionRule("email", r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    RedactionRule("home_path", _home_pattern()),
)


@dataclass(frozen=True)
class RedactionReport:
    """What the redactor replaced, by rule name.

    Counts only — the report deliberately never carries the matched text, so it stays safe
    to keep, log, and store next to the envelope after the secret itself is gone.
    """

    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def clean(self) -> bool:
        """True when no rule fired. Means "nothing obvious", not "safe" — read the payload."""
        return self.total == 0


def redact_text(
    text: str, rules: tuple[RedactionRule, ...] = DEFAULT_RULES
) -> tuple[str, dict[str, int]]:
    """Apply ``rules`` in order to ``text``. Returns the masked text and per-rule hit counts."""
    counts: dict[str, int] = {}
    for rule in rules:
        text, hits = rule.apply(text)
        if hits:
            counts[rule.name] = counts.get(rule.name, 0) + hits
    return text, counts


def redact_envelope(
    envelope: HandoffEnvelope, rules: tuple[RedactionRule, ...] = DEFAULT_RULES
) -> tuple[HandoffEnvelope, RedactionReport]:
    """Return a redacted copy of ``envelope`` plus the report.

    Only the payload (prompt and inputs) is redacted — those are the fields that cross.
    ``local_attempt.result`` is *not* masked because it never leaves: it is context for the
    reviewer explaining why local was insufficient. Redacting the payload changes the content
    hash, which is exactly right: the redacted envelope is a different artifact and needs its
    own approval.
    """
    counts: dict[str, int] = {}
    prompt, prompt_counts = redact_text(envelope.prompt, rules)
    for name, hits in prompt_counts.items():
        counts[name] = counts.get(name, 0) + hits

    inputs: dict[str, str] = {}
    for key, value in envelope.inputs.items():
        masked, input_counts = redact_text(value, rules)
        inputs[key] = masked
        for name, hits in input_counts.items():
            counts[name] = counts.get(name, 0) + hits

    # Any prior approval is dropped: it approved different bytes.
    redacted = HandoffEnvelope(
        id=envelope.id,
        created_at=envelope.created_at,
        task_class=envelope.task_class,
        prompt=prompt,
        destination_tier=envelope.destination_tier,
        sensitivity=envelope.sensitivity,
        local_attempt=envelope.local_attempt,
        inputs=inputs,
        provenance=envelope.provenance,
        review=None,
        notes=envelope.notes,
        schema_version=envelope.schema_version,
    )
    redacted.validate()
    return redacted, RedactionReport(counts=counts)


def render_review(
    envelope: HandoffEnvelope, report: RedactionReport | None = None
) -> str:
    """Render the complete, human-readable review sheet for ``envelope``.

    Everything that would cross the boundary appears here in full — no truncation, no
    summarisation. A reviewer who reads this text has seen the whole payload; that is the
    property the whole mechanism rests on, so this function must never elide.
    """
    payload_bytes = len(envelope.prompt.encode("utf-8")) + sum(
        len(k.encode("utf-8")) + len(v.encode("utf-8")) for k, v in envelope.inputs.items()
    )
    attempt = envelope.local_attempt
    lines: list[str] = [
        "=" * 78,
        f"HANDOFF REVIEW — {envelope.id}",
        "=" * 78,
        f"  created            {envelope.created_at}",
        f"  task class         {envelope.task_class}",
        f"  destination        tier {envelope.destination_tier}",
        f"  sensitivity        {envelope.sensitivity}",
        f"  content hash       {envelope.content_hash}",
        f"  payload size       {payload_bytes} bytes",
        "",
        "-- why local was insufficient " + "-" * 48,
        f"  tier {attempt.tier} model {attempt.model or '(unnamed)'}"
        + (f"  confidence {attempt.confidence:.2f}" if attempt.confidence is not None else ""),
        f"  reason: {attempt.reason}",
    ]
    if attempt.result:
        lines.append("  local result (stays here, does NOT cross):")
        lines.extend(f"    | {line}" for line in attempt.result.splitlines() or [""])
    if report is not None:
        lines += ["", "-- redaction " + "-" * 65]
        if report.clean:
            lines.append("  no rule fired — this means 'nothing obvious', NOT 'safe to send'")
        else:
            lines.extend(f"  {name}: {hits}" for name, hits in sorted(report.counts.items()))
    lines += [
        "",
        "-- PAYLOAD: everything below this line would cross the boundary " + "-" * 15,
        "",
        "  [prompt]",
    ]
    lines.extend(f"  {line}" for line in envelope.prompt.splitlines() or [""])
    for key in sorted(envelope.inputs):
        lines += ["", f"  [input: {key}]"]
        lines.extend(f"  {line}" for line in envelope.inputs[key].splitlines() or [""])
    if envelope.notes:
        lines += ["", f"  [notes] {envelope.notes}"]
    lines += [
        "",
        "-- end of payload " + "-" * 59,
        "",
        "  HEARTH will not send this. Approving only writes the file to the release",
        "  directory; carrying it across is a separate, manual act.",
        "=" * 78,
    ]
    return "\n".join(lines)


def approve(
    envelope: HandoffEnvelope,
    *,
    reviewed_by: str,
    reviewed_at: str | None = None,
    report: RedactionReport | None = None,
    note: str = "",
) -> HandoffEnvelope:
    """Return a copy of ``envelope`` carrying an approval bound to its current content hash."""
    return _decide(
        envelope,
        DECISION_APPROVED,
        reviewed_by=reviewed_by,
        reviewed_at=reviewed_at,
        report=report,
        note=note,
    )


def reject(
    envelope: HandoffEnvelope,
    *,
    reviewed_by: str,
    reviewed_at: str | None = None,
    note: str = "",
) -> HandoffEnvelope:
    """Return a copy of ``envelope`` marked rejected. A rejected envelope never releases."""
    return _decide(
        envelope,
        DECISION_REJECTED,
        reviewed_by=reviewed_by,
        reviewed_at=reviewed_at,
        report=None,
        note=note,
    )


def _decide(
    envelope: HandoffEnvelope,
    decision: str,
    *,
    reviewed_by: str,
    reviewed_at: str | None,
    report: RedactionReport | None,
    note: str,
) -> HandoffEnvelope:
    record = ReviewRecord(
        reviewed_by=reviewed_by,
        reviewed_at=reviewed_at or now_iso(),
        payload_hash=envelope.content_hash,
        decision=decision,
        redactions=dict(report.counts) if report else {},
        note=note,
    )
    record.validate()
    reviewed = HandoffEnvelope(
        id=envelope.id,
        created_at=envelope.created_at,
        task_class=envelope.task_class,
        prompt=envelope.prompt,
        destination_tier=envelope.destination_tier,
        sensitivity=envelope.sensitivity,
        local_attempt=envelope.local_attempt,
        inputs=envelope.inputs,
        provenance=envelope.provenance,
        review=record,
        notes=envelope.notes,
        schema_version=envelope.schema_version,
    )
    reviewed.validate()
    return reviewed
