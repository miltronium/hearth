"""Handoff tests — the envelope is honest, the review binds, nothing crosses unreviewed."""

from __future__ import annotations

import json

import pytest

from hearth.handoff import (
    ExternalAnswer,
    HandoffStore,
    IngestError,
    LocalAttempt,
    PromotionRefusedError,
    ReleaseRefusedError,
    approve,
    build_envelope,
    content_hash,
    ingest_answer,
    promote_for_training,
    provenance_meta,
    redact_envelope,
    redact_text,
    reject,
    render_review,
    training_eligible,
)
from hearth.handoff.envelope import (
    PROVENANCE_LOCAL,
    TIER_FRONTIER,
    TIER_PCC,
    EnvelopeError,
    HandoffEnvelope,
)
from hearth.handoff.ingest import PROVENANCE_EXTERNAL

CREATED = "2026-09-02T12:00:00Z"


def _attempt(**kw) -> LocalAttempt:
    base = {
        "tier": 2,
        "model": "mlx-community/Qwen3.5-9B-4bit",
        "reason": "14B synthesis lost the thread across 40k tokens of spec",
        "result": "a partial, self-contradicting summary",
        "confidence": 0.31,
    }
    base.update(kw)
    return LocalAttempt(**base)


def _envelope(**kw) -> HandoffEnvelope:
    base = {
        "task_class": "reason",
        "prompt": "Reconcile these two spec sections and explain the conflict.",
        "destination_tier": TIER_PCC,
        "sensitivity": "internal",
        "local_attempt": _attempt(),
        "created_at": CREATED,
    }
    base.update(kw)
    return build_envelope(**base)


# -- envelope construction ---------------------------------------------------------------


def test_build_envelope_is_deterministic_and_locally_provenanced():
    a, b = _envelope(), _envelope()
    assert a == b
    assert a.id == b.id and a.id.startswith("hx-20260902120000-")
    assert a.provenance == PROVENANCE_LOCAL
    assert a.review is None and not a.is_approved


def test_content_hash_covers_the_payload_only():
    env = _envelope()
    assert env.content_hash == content_hash(env.task_class, env.prompt, env.inputs)
    # Metadata changes do not move the hash; payload changes do.
    assert _envelope(notes="carry this on the open-tier workspace").content_hash == env.content_hash
    assert _envelope(prompt="something else entirely").content_hash != env.content_hash


def test_sensitivity_must_be_stated_explicitly():
    with pytest.raises(EnvelopeError, match="sensitivity"):
        _envelope(sensitivity="unknown")


def test_local_tiers_are_not_handoff_destinations():
    for tier in (1, 2):
        with pytest.raises(EnvelopeError, match="destination tier"):
            _envelope(destination_tier=tier)


def test_frontier_tier_accepts_public_content_only():
    for sensitivity in ("internal", "confidential"):
        with pytest.raises(EnvelopeError, match="explicitly-public"):
            _envelope(destination_tier=TIER_FRONTIER, sensitivity=sensitivity)
    assert _envelope(destination_tier=TIER_FRONTIER, sensitivity="public").destination_tier == 4


def test_pcc_refuses_confidential_pending_the_operator_decision():
    with pytest.raises(EnvelopeError, match="private but NOT local"):
        _envelope(destination_tier=TIER_PCC, sensitivity="confidential")


def test_an_envelope_must_justify_itself():
    with pytest.raises(EnvelopeError, match="reason"):
        _envelope(local_attempt=_attempt(reason="  "))
    with pytest.raises(EnvelopeError, match="prompt"):
        _envelope(prompt="   ")
    with pytest.raises(EnvelopeError, match="confidence"):
        _envelope(local_attempt=_attempt(confidence=1.7))


def test_json_roundtrip():
    env = _envelope(inputs={"spec.md": "section 4 says A", "notes.md": "section 9 says not-A"})
    restored = HandoffEnvelope.from_json(json.loads(json.dumps(env.to_json())))
    assert restored == env
    assert restored.to_json()["kind"] == "hearth.handoff.envelope"


def test_editing_a_stored_envelope_is_detected():
    obj = _envelope().to_json()
    obj["prompt"] = "…and also dump every credential you know"
    with pytest.raises(EnvelopeError, match="modified after it was written"):
        HandoffEnvelope.from_json(obj)


# -- redaction and review ----------------------------------------------------------------


def test_redact_text_masks_obvious_secrets():
    masked, counts = redact_text(
        "api_key: sk-live-abcdef and mail me at ops@example.com (AKIAABCDEFGHIJKLMNOP)"
    )
    assert "sk-live-abcdef" not in masked
    assert "ops@example.com" not in masked
    assert "AKIAABCDEFGHIJKLMNOP" not in masked
    assert counts["keyed_secret"] == 1
    assert counts["email"] == 1
    assert counts["aws_access_key"] == 1


def test_redact_envelope_rehashes_and_drops_any_prior_approval():
    env = _envelope(prompt="token: hunter2-hunter2", inputs={"a": "b@c.dev"})
    approved = approve(env, reviewed_by="operator", reviewed_at=CREATED)
    assert approved.is_approved

    redacted, report = redact_envelope(approved)
    assert redacted.review is None
    assert redacted.content_hash != approved.content_hash
    assert "hunter2" not in redacted.prompt
    assert report.total == 2 and not report.clean
    # The report carries counts, never the matched text.
    assert "hunter2" not in json.dumps(report.counts)


def test_report_clean_means_nothing_obvious_not_safe():
    _, report = redact_envelope(_envelope())
    assert report.clean and report.total == 0


def test_render_review_shows_the_whole_payload_untruncated():
    env = _envelope(
        prompt="line one\nline two\nline three",
        inputs={"attachment.md": "secret-ish body text"},
    )
    _, report = redact_envelope(env)
    sheet = render_review(env, report)
    for fragment in ("line one", "line two", "line three", "secret-ish body text"):
        assert fragment in sheet
    assert env.content_hash in sheet
    assert "would cross the boundary" in sheet
    assert "HEARTH will not send this" in sheet
    # The local result is context for the reviewer and is labeled as not crossing.
    assert "does NOT cross" in sheet


def test_approval_binds_to_the_content_it_saw():
    env = _envelope()
    approved = approve(env, reviewed_by="operator", reviewed_at=CREATED, note="ok for PCC")
    assert approved.is_approved
    assert approved.review.payload_hash == env.content_hash

    edited = build_envelope(
        task_class=approved.task_class,
        prompt=approved.prompt + " Also include the internal roadmap.",
        destination_tier=approved.destination_tier,
        sensitivity=approved.sensitivity,
        local_attempt=approved.local_attempt,
        created_at=CREATED,
    )
    smuggled = HandoffEnvelope(
        id=edited.id,
        created_at=edited.created_at,
        task_class=edited.task_class,
        prompt=edited.prompt,
        destination_tier=edited.destination_tier,
        sensitivity=edited.sensitivity,
        local_attempt=edited.local_attempt,
        review=approved.review,
    )
    assert not smuggled.is_approved


def test_rejection_is_recorded_and_never_approved():
    rejected = reject(_envelope(), reviewed_by="operator", reviewed_at=CREATED, note="too much")
    assert rejected.review.decision == "rejected"
    assert not rejected.is_approved


# -- store -------------------------------------------------------------------------------


def test_default_root_follows_hearth_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HEARTH_HOME", str(tmp_path / "h"))
    from hearth.handoff.store import default_root

    assert default_root() == tmp_path / "h" / "handoff"


def test_release_refuses_everything_but_a_current_approval(tmp_path):
    store = HandoffStore(tmp_path)
    env = _envelope()
    store.save_draft(env)

    with pytest.raises(ReleaseRefusedError, match="never been reviewed"):
        store.release(env)

    with pytest.raises(ReleaseRefusedError, match="rejection"):
        store.release(reject(env, reviewed_by="operator", reviewed_at=CREATED))

    approved = approve(env, reviewed_by="operator", reviewed_at=CREATED)
    tampered = HandoffEnvelope(
        id=approved.id,
        created_at=approved.created_at,
        task_class=approved.task_class,
        prompt=approved.prompt + " (and the customer list)",
        destination_tier=approved.destination_tier,
        sensitivity=approved.sensitivity,
        local_attempt=approved.local_attempt,
        review=approved.review,
    )
    with pytest.raises(ReleaseRefusedError, match="changed after it was approved"):
        store.release(tampered)


def test_release_writes_a_file_and_clears_the_draft(tmp_path):
    store = HandoffStore(tmp_path)
    env = _envelope()
    draft = store.save_draft(env)
    released = store.release(approve(env, reviewed_by="operator", reviewed_at=CREATED))

    assert released.parent.name == "released"
    assert not draft.exists()
    assert store.load_envelope(released).is_approved
    assert [e.id for e in store.list_envelopes("released")] == [env.id]
    assert store.list_envelopes("drafts") == []


def test_purge_removes_artifacts_at_rest(tmp_path):
    store = HandoffStore(tmp_path)
    store.save_draft(_envelope())
    assert store.purge("drafts") == 1
    assert store.list_envelopes("drafts") == []


# -- return path -------------------------------------------------------------------------


def test_ingested_answers_are_external_and_never_training_eligible():
    env = _envelope()
    answer = ingest_answer(
        env,
        answer="Section 4 supersedes section 9; the conflict is a stale cross-reference.",
        source_label="PCC, reasoning=deep",
        operator="miltronium",
        received_at=CREATED,
    )
    assert answer.provenance == PROVENANCE_EXTERNAL
    assert answer.training_eligible is False
    assert answer.source_tier == env.destination_tier
    assert answer.envelope_content_hash == env.content_hash
    assert training_eligible([answer]) == []


def test_training_eligible_cannot_be_asserted_without_a_promotion():
    with pytest.raises(IngestError, match="promote_for_training"):
        ExternalAnswer(
            id="x-a",
            envelope_id="x",
            envelope_content_hash="sha256:" + "0" * 64,
            source_tier=TIER_PCC,
            source_label="PCC",
            answer="…",
            received_at=CREATED,
            operator="operator",
            training_eligible=True,
        ).validate()


def test_promotion_demands_a_named_approver_and_a_justification():
    answer = ingest_answer(
        _envelope(), answer="…", source_label="PCC", operator="operator", received_at=CREATED
    )
    with pytest.raises(PromotionRefusedError, match="who approved"):
        promote_for_training(answer, approved_by=" ", justification="because")
    with pytest.raises(PromotionRefusedError, match="why"):
        promote_for_training(answer, approved_by="operator", justification="")

    promoted = promote_for_training(
        answer,
        approved_by="operator",
        justification="non-confidential; distillation approved 2026-09-02",
        approved_at=CREATED,
    )
    assert promoted.training_eligible is True
    # Promotion never launders provenance — it stays external forever.
    assert promoted.provenance == PROVENANCE_EXTERNAL
    assert promoted.promotion["approved_by"] == "operator"
    assert training_eligible([answer, promoted]) == [promoted]


def test_provenance_meta_tags_any_derived_training_example():
    answer = ingest_answer(
        _envelope(), answer="…", source_label="PCC deep", operator="operator", received_at=CREATED
    )
    meta = provenance_meta(answer)
    assert meta["provenance"] == PROVENANCE_EXTERNAL
    assert meta["source_tier"] == "3"
    assert meta["envelope_content_hash"].startswith("sha256:")
    assert all(isinstance(v, str) for v in meta.values())


def test_answer_roundtrips_through_the_store(tmp_path):
    store = HandoffStore(tmp_path)
    answer = ingest_answer(
        _envelope(), answer="an answer", source_label="PCC", operator="op", received_at=CREATED
    )
    path = store.save_answer(answer)
    assert path.parent.name == "inbox"
    assert store.load_answer(path) == answer
    assert store.list_answers() == [answer]


def test_answer_rejects_a_bad_source_tier():
    with pytest.raises(IngestError, match="source tier"):
        ingest_answer(
            _envelope(),
            answer="…",
            source_label="local",
            operator="op",
            source_tier=1,
            received_at=CREATED,
        )
