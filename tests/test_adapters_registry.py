"""Adapter registry lifecycle tests — register/promote/retire, gate, A/B, persistence."""

from __future__ import annotations

import pytest

from hearth.registry import (
    AdapterError,
    AdapterStore,
    GateNotPassedError,
)
from hearth.registry.adapters import STATUS_CANDIDATE, STATUS_PROMOTED, STATUS_RETIRED


def _store(tmp_path) -> AdapterStore:
    return AdapterStore(path=tmp_path / "adapters.json")


def _register(store, adapter_id="extract-1", task="extract"):
    return store.register(
        adapter_id,
        base_model="org/base",
        task=task,
        train_run_id="run-1",
        adapter_path=f"/adapters/{adapter_id}",
    )


def test_register_creates_candidate_and_persists(tmp_path):
    store = _store(tmp_path)
    entry = _register(store)
    assert entry.status == STATUS_CANDIDATE
    # A fresh store over the same file sees it (persisted to disk).
    assert _store(tmp_path).get("extract-1").status == STATUS_CANDIDATE


def test_register_rejects_duplicate(tmp_path):
    store = _store(tmp_path)
    _register(store)
    with pytest.raises(AdapterError):
        _register(store)


def test_promote_refused_without_gate(tmp_path):
    store = _store(tmp_path)
    _register(store)
    with pytest.raises(GateNotPassedError):
        store.promote("extract-1", gate_passed=False)
    # Still a candidate — the refusal did not mutate state.
    assert store.get("extract-1").status == STATUS_CANDIDATE


def test_promote_records_proof_and_retires_prior(tmp_path):
    store = _store(tmp_path)
    _register(store, "extract-1")
    _register(store, "extract-2")
    store.promote("extract-1", gate_passed=True, proof={"candidate_score": 0.9})
    assert store.get("extract-1").status == STATUS_PROMOTED
    assert store.get("extract-1").promotion_proof["candidate_score"] == 0.9

    # Promoting a second one for the same task retires the first (one promoted per task).
    store.promote("extract-2", gate_passed=True)
    assert store.get("extract-1").status == STATUS_RETIRED
    assert store.get("extract-2").status == STATUS_PROMOTED
    assert store.promoted_for("extract").id == "extract-2"


def test_retire_and_list_filters(tmp_path):
    store = _store(tmp_path)
    _register(store, "extract-1", task="extract")
    _register(store, "classify-1", task="classify")
    store.retire("classify-1")
    assert {e.id for e in store.list(task="extract")} == {"extract-1"}
    assert {e.id for e in store.list(status=STATUS_RETIRED)} == {"classify-1"}


def test_resolve_path_ab_flag_for_candidates(tmp_path):
    store = _store(tmp_path)
    _register(store, "extract-1")
    # A candidate serves only behind the A/B flag.
    assert store.resolve_path("extract-1", allow_candidate=True) == "/adapters/extract-1"
    with pytest.raises(AdapterError):
        store.resolve_path("extract-1", allow_candidate=False)
    # A retired adapter never resolves.
    store.retire("extract-1")
    with pytest.raises(AdapterError):
        store.resolve_path("extract-1")


def test_promote_unknown_and_retired(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(AdapterError):
        store.promote("nope", gate_passed=True)
    _register(store)
    store.retire("extract-1")
    with pytest.raises(AdapterError):
        store.promote("extract-1", gate_passed=True)
