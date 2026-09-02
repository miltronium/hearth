"""Handoff envelopes — how work reaches tiers 3-4 without HEARTH gaining a way out.

HEARTH is no-egress **by construction**: ``config/routing.private.yaml`` defines zero
remotes, so the router has structurally nowhere to send a task, and that is verifiable on the
machine with ``lsof`` and a firewall probe (``docs/PRIVACY.md``). Tier 3 (Apple Private Cloud
Compute) and tier 4 (a frontier model) are off this machine, so wiring either in as a backend
would trade that property for convenience.

Instead HEARTH emits a **handoff envelope**: a local file describing the task, why the local
tiers were not enough, and exactly what would cross. A human reviews it and carries it over
using tools that live outside HEARTH. Answers come back through :mod:`hearth.handoff.ingest`
permanently tagged ``provenance="external"``. The engine stays sealed; the crossing is always
a human act.

**Invariant — no network code in this package.** No HTTP client, no socket, no URL fetch, no
subprocess. It imports the standard library only (not even other HEARTH modules), so the
property is checkable by reading four files. ``tests/test_handoff_no_network.py`` enforces it.

Design and open questions: ``docs/TIERS.md``.
"""

from __future__ import annotations

from .envelope import (
    DESTINATION_TIERS,
    PROVENANCE_LOCAL,
    SCHEMA_VERSION,
    SENSITIVITIES,
    TIER_FRONTIER,
    TIER_LOCAL_LARGE,
    TIER_LOCAL_SMALL,
    TIER_PCC,
    EnvelopeError,
    HandoffEnvelope,
    HandoffError,
    LocalAttempt,
    ReviewRecord,
    build_envelope,
    content_hash,
    now_iso,
)
from .ingest import (
    PROVENANCE_EXTERNAL,
    ExternalAnswer,
    IngestError,
    PromotionRefusedError,
    ingest_answer,
    promote_for_training,
    provenance_meta,
    training_eligible,
)
from .review import (
    DEFAULT_RULES,
    RedactionReport,
    RedactionRule,
    approve,
    redact_envelope,
    redact_text,
    reject,
    render_review,
)
from .store import HandoffStore, ReleaseRefusedError, default_root

__all__ = [
    "DEFAULT_RULES",
    "DESTINATION_TIERS",
    "PROVENANCE_EXTERNAL",
    "PROVENANCE_LOCAL",
    "SCHEMA_VERSION",
    "SENSITIVITIES",
    "TIER_FRONTIER",
    "TIER_LOCAL_LARGE",
    "TIER_LOCAL_SMALL",
    "TIER_PCC",
    "EnvelopeError",
    "ExternalAnswer",
    "HandoffEnvelope",
    "HandoffError",
    "HandoffStore",
    "IngestError",
    "LocalAttempt",
    "PromotionRefusedError",
    "RedactionReport",
    "RedactionRule",
    "ReleaseRefusedError",
    "ReviewRecord",
    "approve",
    "build_envelope",
    "content_hash",
    "default_root",
    "ingest_answer",
    "now_iso",
    "promote_for_training",
    "provenance_meta",
    "redact_envelope",
    "redact_text",
    "reject",
    "render_review",
    "training_eligible",
]
