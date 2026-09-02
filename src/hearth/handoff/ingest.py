"""The return path — bringing an outside answer back in, permanently marked as outside.

An answer that came from tier 3 or 4 is not a local result and must never become
indistinguishable from one. Two contaminations matter, and they have different half-lives
(``docs/TIERS.md`` §6):

  * **Session contamination** — the answer enters a sealed session's context and is then
    quoted, summarised, and re-used as if HEARTH produced it. Recoverable: it costs a session.
  * **Corpus contamination** — the answer is captured by the learning loop and distilled into
    a LoRA. Not recoverable: the provenance is gone the moment the weights are trained, and
    every downstream eval, adapter and claim about "local" output inherits the lie.

So an ingested answer is an :class:`ExternalAnswer` with ``provenance="external"`` — the
value is fixed at construction and cannot be built any other way — and
``training_eligible=False``. Promotion into a training corpus exists
(:func:`promote_for_training`) but is a separate, named, justified act that produces a new
record and keeps the external provenance forever. Nothing in this module can flip that flag
implicitly.

The honest limit, stated here because it belongs next to the mechanism: HEARTH cannot detect
an answer the operator retypes by hand into a prompt. This module makes the *recorded* path
safe and auditable; it does not police the keyboard.

As with the rest of the package: no network code. Ingest reads a string the operator already
has; it never goes and gets one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .envelope import (
    DESTINATION_TIERS,
    SCHEMA_VERSION,
    HandoffEnvelope,
    HandoffError,
    now_iso,
)

# Marks a JSON file as an ingested answer (vs. an envelope).
KIND_ANSWER = "hearth.handoff.answer"

# The only provenance an ingested answer may carry. Deliberately a different vocabulary from
# the envelope's ``local``: a grep for this string finds every off-machine-sourced record.
PROVENANCE_EXTERNAL = "external"


class IngestError(HandoffError):
    """Raised when an answer cannot be ingested (bad shape, wrong tier, broken linkage)."""


class PromotionRefusedError(HandoffError):
    """Raised when an external answer is promoted toward training without a real decision."""


@dataclass(frozen=True)
class ExternalAnswer:
    """One answer that came from outside this machine, tagged as such for good.

    ``envelope_content_hash`` ties the answer to the exact payload that crossed, so an answer
    can always be traced to the request it answered — and a mismatched pair is detectable
    rather than plausible.
    """

    id: str
    envelope_id: str
    envelope_content_hash: str
    source_tier: int
    source_label: str
    answer: str
    received_at: str
    operator: str
    provenance: str = PROVENANCE_EXTERNAL
    training_eligible: bool = False
    promotion: dict[str, str] = field(default_factory=dict)
    notes: str = ""
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        """Raise :class:`IngestError` unless the record is well-formed and honestly labeled."""
        if self.schema_version != SCHEMA_VERSION:
            raise IngestError(
                f"answer schema {self.schema_version} != supported {SCHEMA_VERSION}"
            )
        if self.provenance != PROVENANCE_EXTERNAL:
            raise IngestError(
                f"an ingested answer is always {PROVENANCE_EXTERNAL!r}; got {self.provenance!r}"
            )
        if self.source_tier not in DESTINATION_TIERS:
            raise IngestError(
                f"source tier must be one of {DESTINATION_TIERS}, got {self.source_tier!r}"
            )
        if not self.answer.strip():
            raise IngestError("an empty answer is not an answer")
        if not self.operator.strip():
            raise IngestError("ingest must name the operator who carried the answer back")
        if not self.envelope_content_hash.startswith("sha256:"):
            raise IngestError("answer must reference the envelope's sha256: content hash")
        if self.training_eligible and not self.promotion:
            raise IngestError(
                "training_eligible is set with no promotion record — an external answer only "
                "becomes eligible through promote_for_training()"
            )

    def to_json(self) -> dict:
        obj = asdict(self)
        obj["kind"] = KIND_ANSWER
        return obj

    @classmethod
    def from_json(cls, obj: dict) -> ExternalAnswer:
        if obj.get("kind") not in (KIND_ANSWER, None):
            raise IngestError(f"not a handoff answer: kind={obj.get('kind')!r}")
        record = cls(
            id=obj["id"],
            envelope_id=obj.get("envelope_id", ""),
            envelope_content_hash=obj.get("envelope_content_hash", ""),
            source_tier=int(obj.get("source_tier", 0)),
            source_label=obj.get("source_label", ""),
            answer=obj.get("answer", ""),
            received_at=obj.get("received_at", ""),
            operator=obj.get("operator", ""),
            provenance=obj.get("provenance", PROVENANCE_EXTERNAL),
            training_eligible=bool(obj.get("training_eligible", False)),
            promotion={str(k): str(v) for k, v in dict(obj.get("promotion", {})).items()},
            notes=obj.get("notes", ""),
            schema_version=int(obj.get("schema_version", SCHEMA_VERSION)),
        )
        record.validate()
        return record


def ingest_answer(
    envelope: HandoffEnvelope,
    *,
    answer: str,
    source_label: str,
    operator: str,
    source_tier: int | None = None,
    received_at: str | None = None,
    notes: str = "",
) -> ExternalAnswer:
    """Record an answer the operator carried back for ``envelope``.

    The result is always ``provenance="external"`` and ``training_eligible=False``. There is
    no parameter to change either — that is the point of the function.

    ``source_label`` is free text naming *what actually answered* ("PCC, reasoning=deep",
    "Claude Opus, web UI"), because the tier number alone loses the detail an audit needs.
    """
    tier = envelope.destination_tier if source_tier is None else source_tier
    record = ExternalAnswer(
        id=f"{envelope.id}-a",
        envelope_id=envelope.id,
        envelope_content_hash=envelope.content_hash,
        source_tier=tier,
        source_label=source_label,
        answer=answer,
        received_at=received_at or now_iso(),
        operator=operator,
        notes=notes,
    )
    record.validate()
    return record


def promote_for_training(
    record: ExternalAnswer,
    *,
    approved_by: str,
    justification: str,
    approved_at: str | None = None,
) -> ExternalAnswer:
    """Return a copy of ``record`` marked eligible for a training corpus.

    Deliberately awkward. Distilling a frontier answer into a local adapter is a legitimate
    thing to want; doing it *silently* is not, and for material covered by a confidentiality
    obligation the answer is often "no" for legal reasons rather than technical ones. So this
    demands a named approver and a written justification, records both, and keeps
    ``provenance="external"`` on the record forever — a promoted answer is still an outside
    answer, it is merely one someone signed for.
    """
    if not approved_by.strip():
        raise PromotionRefusedError("promotion must name who approved it")
    if not justification.strip():
        raise PromotionRefusedError(
            "promotion must state why this external answer may enter local training data"
        )
    promoted = ExternalAnswer(
        id=record.id,
        envelope_id=record.envelope_id,
        envelope_content_hash=record.envelope_content_hash,
        source_tier=record.source_tier,
        source_label=record.source_label,
        answer=record.answer,
        received_at=record.received_at,
        operator=record.operator,
        training_eligible=True,
        promotion={
            "approved_by": approved_by,
            "approved_at": approved_at or now_iso(),
            "justification": justification,
        },
        notes=record.notes,
        schema_version=record.schema_version,
    )
    promoted.validate()
    return promoted


def training_eligible(records: list[ExternalAnswer]) -> list[ExternalAnswer]:
    """Filter to the records a corpus builder may use. Un-promoted answers never appear."""
    return [r for r in records if r.training_eligible]


def provenance_meta(record: ExternalAnswer) -> dict[str, str]:
    """Provenance fields to attach to any training example derived from ``record``.

    Shaped as ``dict[str, str]`` so it drops straight into a dataset record's ``meta`` field
    (``hearth.training.dataset.DatasetRecord.meta``) without this package importing it. The
    tag travels with the example into the JSONL, so "which rows in this corpus came from
    outside?" stays answerable after the fact.
    """
    return {
        "provenance": record.provenance,
        "source_tier": str(record.source_tier),
        "source_label": record.source_label,
        "envelope_id": record.envelope_id,
        "envelope_content_hash": record.envelope_content_hash,
        "promoted_by": record.promotion.get("approved_by", ""),
    }
