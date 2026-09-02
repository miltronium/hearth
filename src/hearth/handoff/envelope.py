"""Handoff envelope — the artifact that crosses the boundary, so HEARTH never does.

**Invariant: this package contains no network code.** No HTTP client, no socket, no URL
fetch, no subprocess (a shell is an egress vector too). It imports the standard library
*only* — not even other HEARTH modules — so an auditor can confirm that property by reading
these files, without following an import chain. ``tests/test_handoff_no_network.py`` enforces
it mechanically. See ``docs/TIERS.md``.

A handoff envelope is what HEARTH produces when a task exceeds local capability (tiers 1-2)
and the operator wants to take it to a tier that is *not* on this machine (tier 3 Apple
Private Cloud Compute, tier 4 frontier). HEARTH writes a local file describing the task and
stops. The crossing is a deliberate human act, performed with tools outside HEARTH. That is
what keeps ``config/routing.private.yaml``'s "zero remotes" claim true *by construction* and
keeps ``lsof`` a valid proof (``docs/PRIVACY.md``).

An envelope records:

  * the task (class, prompt, inputs) — exactly the bytes that would cross;
  * **why local was insufficient** — which tier ran, on which model, what it produced, and
    what confidence it reported. Required, not optional: an envelope with no failure story is
    a request to leak for convenience.
  * the intended destination tier, plus an explicit **sensitivity** label;
  * a content hash, a created-at timestamp, and a provenance field;
  * a review record — approval is bound to the content hash, so an envelope edited after
    review is no longer releasable.

Construction fails closed: no default sensitivity, tier 4 requires explicitly-public content,
and tier 3 refuses confidential material until the operator settles ``docs/TIERS.md`` Q1.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

# Bump when the on-disk envelope shape changes. Every artifact carries this so a reader can
# reject a file it doesn't understand rather than silently mis-parse it (cf. training/dataset.py).
SCHEMA_VERSION = 1

# Marks a JSON file as a handoff envelope (vs. an ingested answer; see ingest.py).
KIND_ENVELOPE = "hearth.handoff.envelope"

# The tier ladder (docs/TIERS.md §1). Tiers 1-2 run *inside* HEARTH and never produce an
# envelope; only 3 and 4 are off-machine, and only they are valid destinations.
TIER_LOCAL_SMALL = 1
TIER_LOCAL_LARGE = 2
TIER_PCC = 3
TIER_FRONTIER = 4
DESTINATION_TIERS: tuple[int, ...] = (TIER_PCC, TIER_FRONTIER)

# Sensitivity of the payload. There is no default — the operator states it or the build fails.
SENSITIVITY_PUBLIC = "public"
SENSITIVITY_INTERNAL = "internal"
SENSITIVITY_CONFIDENTIAL = "confidential"
SENSITIVITIES: tuple[str, ...] = (
    SENSITIVITY_PUBLIC,
    SENSITIVITY_INTERNAL,
    SENSITIVITY_CONFIDENTIAL,
)

# Provenance of the *content* in this artifact. An envelope is always locally-authored;
# ingest.py owns the "external" side of the vocabulary.
PROVENANCE_LOCAL = "local"

# Review decisions.
DECISION_APPROVED = "approved"
DECISION_REJECTED = "rejected"

# Tier 3 (Apple PCC) is private but **not local** — bytes leave the machine. Whether
# confidential material may take that trip is a policy question the operator has not yet
# answered (docs/TIERS.md, open question Q1), so the code refuses it. Flip this only
# alongside a recorded decision.
PCC_ACCEPTS_CONFIDENTIAL = False


class HandoffError(RuntimeError):
    """Base class for every handoff failure."""


class EnvelopeError(HandoffError):
    """Raised when an envelope is malformed, unlabeled, or destined somewhere it may not go."""


def now_iso() -> str:
    """Return the current UTC time as a second-resolution ISO-8601 ``Z`` timestamp.

    Callers that need determinism (tests, reproducible builds) pass ``created_at``
    explicitly instead — nothing in this package reads the clock behind your back.
    """
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _canonical(obj: object) -> bytes:
    """Serialize ``obj`` to stable bytes (sorted keys, no whitespace) for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def content_hash(task_class: str, prompt: str, inputs: dict[str, str]) -> str:
    """Hash exactly the bytes that would cross the boundary.

    Deliberately covers the *payload only* — task class, prompt, inputs — and not the
    metadata around it. That makes the hash a stable identity for "this text went out", so a
    returned answer can be tied back to the request that produced it (``ingest.py``) and a
    review can be bound to the content it actually saw (:class:`ReviewRecord`).
    """
    payload = {"task_class": task_class, "prompt": prompt, "inputs": dict(inputs)}
    return "sha256:" + hashlib.sha256(_canonical(payload)).hexdigest()


@dataclass(frozen=True)
class LocalAttempt:
    """What tiers 1-2 did before the operator reached for an envelope.

    ``reason`` is the human sentence explaining why the local result was not good enough.
    It is required: the envelope's whole justification for existing lives in this field, and
    a reviewer reads it first.
    """

    tier: int
    model: str
    reason: str
    result: str = ""
    confidence: float | None = None

    def validate(self) -> None:
        """Raise :class:`EnvelopeError` unless this records a real local attempt."""
        if self.tier not in (TIER_LOCAL_SMALL, TIER_LOCAL_LARGE):
            raise EnvelopeError(
                f"local attempt tier must be {TIER_LOCAL_SMALL} or {TIER_LOCAL_LARGE}, "
                f"got {self.tier!r}"
            )
        if not self.reason.strip():
            raise EnvelopeError(
                "local attempt needs a reason — why was the local result not enough?"
            )
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise EnvelopeError(f"confidence must be in [0, 1], got {self.confidence!r}")

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, obj: dict) -> LocalAttempt:
        return cls(
            tier=int(obj["tier"]),
            model=obj.get("model", ""),
            reason=obj.get("reason", ""),
            result=obj.get("result", ""),
            confidence=obj.get("confidence"),
        )


@dataclass(frozen=True)
class ReviewRecord:
    """A human's decision about one envelope, bound to the content they saw.

    ``payload_hash`` is the :func:`content_hash` at review time. Release compares it to the
    envelope's current hash, so editing the prompt after approval silently invalidates the
    approval instead of riding on it.

    ``redactions`` counts what the redactor replaced, by rule name. It never carries the
    matched text — a review record must be safe to keep after the secret is gone.
    """

    reviewed_by: str
    reviewed_at: str
    payload_hash: str
    decision: str = DECISION_APPROVED
    redactions: dict[str, int] = field(default_factory=dict)
    note: str = ""

    def validate(self) -> None:
        """Raise :class:`EnvelopeError` unless the review is well-formed."""
        if not self.reviewed_by.strip():
            raise EnvelopeError("a review must name who performed it")
        if self.decision not in (DECISION_APPROVED, DECISION_REJECTED):
            raise EnvelopeError(f"unknown review decision {self.decision!r}")
        if not self.payload_hash.startswith("sha256:"):
            raise EnvelopeError("review payload_hash must be a sha256: digest")

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, obj: dict) -> ReviewRecord:
        return cls(
            reviewed_by=obj.get("reviewed_by", ""),
            reviewed_at=obj.get("reviewed_at", ""),
            payload_hash=obj.get("payload_hash", ""),
            decision=obj.get("decision", DECISION_APPROVED),
            redactions={str(k): int(v) for k, v in dict(obj.get("redactions", {})).items()},
            note=obj.get("note", ""),
        )


@dataclass(frozen=True)
class HandoffEnvelope:
    """One task, described locally, ready for a human to carry across the boundary.

    Immutable by design: :mod:`hearth.handoff.review` returns *new* envelopes rather than
    mutating one, so an approval can never be retro-fitted onto changed content.
    """

    id: str
    created_at: str
    task_class: str
    prompt: str
    destination_tier: int
    sensitivity: str
    local_attempt: LocalAttempt
    inputs: dict[str, str] = field(default_factory=dict)
    provenance: str = PROVENANCE_LOCAL
    review: ReviewRecord | None = None
    notes: str = ""
    schema_version: int = SCHEMA_VERSION

    @property
    def content_hash(self) -> str:
        """The current payload hash — recomputed, never stored, so it cannot go stale."""
        return content_hash(self.task_class, self.prompt, self.inputs)

    @property
    def is_approved(self) -> bool:
        """True when an approval exists *and* still matches the current payload."""
        r = self.review
        return bool(
            r and r.decision == DECISION_APPROVED and r.payload_hash == self.content_hash
        )

    def validate(self) -> None:
        """Raise :class:`EnvelopeError` unless this envelope is well-formed and permitted."""
        if self.schema_version != SCHEMA_VERSION:
            raise EnvelopeError(
                f"envelope schema {self.schema_version} != supported {SCHEMA_VERSION}"
            )
        if not self.task_class.strip():
            raise EnvelopeError("envelope needs a task class")
        if not self.prompt.strip():
            raise EnvelopeError("envelope needs a prompt — an empty handoff is a bug, not a task")
        if self.provenance != PROVENANCE_LOCAL:
            raise EnvelopeError(
                f"an envelope is locally authored; provenance must be {PROVENANCE_LOCAL!r}"
            )
        if self.destination_tier not in DESTINATION_TIERS:
            raise EnvelopeError(
                f"destination tier must be one of {DESTINATION_TIERS} (tiers 1-2 run inside "
                f"HEARTH and need no envelope), got {self.destination_tier!r}"
            )
        if self.sensitivity not in SENSITIVITIES:
            raise EnvelopeError(
                f"sensitivity must be explicitly one of {SENSITIVITIES}, got "
                f"{self.sensitivity!r} — there is no default; the operator labels the payload"
            )
        if self.destination_tier == TIER_FRONTIER and self.sensitivity != SENSITIVITY_PUBLIC:
            raise EnvelopeError(
                "tier 4 (frontier) accepts explicitly-public content only; this payload is "
                f"labeled {self.sensitivity!r} (docs/TIERS.md §2)"
            )
        if (
            self.destination_tier == TIER_PCC
            and self.sensitivity == SENSITIVITY_CONFIDENTIAL
            and not PCC_ACCEPTS_CONFIDENTIAL
        ):
            raise EnvelopeError(
                "tier 3 (Apple PCC) is private but NOT local — confidential material is "
                "refused pending the operator's decision (docs/TIERS.md open question Q1)"
            )
        self.local_attempt.validate()
        if self.review is not None:
            self.review.validate()

    def to_json(self) -> dict:
        obj = asdict(self)
        obj["kind"] = KIND_ENVELOPE
        obj["content_hash"] = self.content_hash
        return obj

    @classmethod
    def from_json(cls, obj: dict) -> HandoffEnvelope:
        """Rebuild an envelope from its JSON form, validating as it goes."""
        if obj.get("kind") not in (KIND_ENVELOPE, None):
            raise EnvelopeError(f"not a handoff envelope: kind={obj.get('kind')!r}")
        review = obj.get("review")
        env = cls(
            id=obj["id"],
            created_at=obj.get("created_at", ""),
            task_class=obj.get("task_class", ""),
            prompt=obj.get("prompt", ""),
            destination_tier=int(obj.get("destination_tier", 0)),
            sensitivity=obj.get("sensitivity", ""),
            local_attempt=LocalAttempt.from_json(obj["local_attempt"]),
            inputs={str(k): str(v) for k, v in dict(obj.get("inputs", {})).items()},
            provenance=obj.get("provenance", PROVENANCE_LOCAL),
            review=ReviewRecord.from_json(review) if review else None,
            notes=obj.get("notes", ""),
            schema_version=int(obj.get("schema_version", SCHEMA_VERSION)),
        )
        env.validate()
        # A stored hash that disagrees with the content means the file was edited in place.
        stored = obj.get("content_hash")
        if stored and stored != env.content_hash:
            raise EnvelopeError(
                f"envelope {env.id}: stored content_hash does not match its payload — "
                "the file was modified after it was written"
            )
        return env


def envelope_id(created_at: str, digest: str) -> str:
    """Derive a stable, human-sortable envelope id from its timestamp and content hash."""
    stamp = re.sub(r"[^0-9]", "", created_at)[:14]
    return f"hx-{stamp}-{digest.removeprefix('sha256:')[:8]}"


def build_envelope(
    *,
    task_class: str,
    prompt: str,
    destination_tier: int,
    sensitivity: str,
    local_attempt: LocalAttempt,
    inputs: dict[str, str] | None = None,
    created_at: str | None = None,
    notes: str = "",
) -> HandoffEnvelope:
    """Build and validate an envelope. Raises :class:`EnvelopeError` if it may not exist.

    Keyword-only on purpose: every field here is a decision, and a positional call site is
    a place to get ``sensitivity`` and ``destination_tier`` the wrong way round.
    """
    inputs = dict(inputs or {})
    created_at = created_at or now_iso()
    digest = content_hash(task_class, prompt, inputs)
    env = HandoffEnvelope(
        id=envelope_id(created_at, digest),
        created_at=created_at,
        task_class=task_class,
        prompt=prompt,
        destination_tier=destination_tier,
        sensitivity=sensitivity,
        local_attempt=local_attempt,
        inputs=inputs,
        notes=notes,
    )
    env.validate()
    return env
