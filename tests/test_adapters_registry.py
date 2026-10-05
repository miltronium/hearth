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


# -- B-082: lock, re-check under the lock, atomic write ----------------------------------


def test_promote_refuses_when_the_incumbent_is_not_the_one_beaten(tmp_path):
    from hearth.registry.adapters import IncumbentChangedError

    store = _store(tmp_path)
    _register(store, "extract-1")
    _register(store, "extract-0")
    store.promote("extract-0", gate_passed=True)
    with pytest.raises(IncumbentChangedError, match="measured against 'the base model'"):
        store.promote("extract-1", gate_passed=True, expected_incumbent=None)
    assert store.get("extract-0").status == STATUS_PROMOTED
    assert store.get("extract-1").status == STATUS_CANDIDATE
    # Naming the real incumbent is accepted, and retires it.
    store.promote("extract-1", gate_passed=True, expected_incumbent="extract-0")
    assert store.get("extract-0").status == STATUS_RETIRED


def test_promote_runs_the_precondition_and_refuses_on_any_problem(tmp_path):
    store = _store(tmp_path)
    _register(store, "extract-1")
    with pytest.raises(AdapterError, match="evidence changed while promoting — weights moved"):
        store.promote("extract-1", gate_passed=True, precondition=lambda: ["weights moved"])
    assert store.get("extract-1").status == STATUS_CANDIDATE


def test_concurrent_registrations_lose_no_update(tmp_path):
    """reload -> mutate -> rewrite with no lock lost updates under concurrency."""
    import threading

    errors = []

    def worker(n):
        try:
            for i in range(6):
                _register(_store(tmp_path), f"a-{n}-{i}")
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(_store(tmp_path).list()) == 48


def test_a_failed_write_leaves_the_registry_intact(tmp_path, monkeypatch):
    """The rewrite is temp + fsync + os.replace: a crash mid-write cannot truncate the file."""
    import os as _os

    store = _store(tmp_path)
    _register(store, "extract-1")
    before = store.path.read_bytes()

    def boom(fd):
        raise OSError("disk full")

    monkeypatch.setattr(_os, "fsync", boom)
    with pytest.raises(OSError, match="disk full"):
        _register(store, "extract-2")
    assert store.path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []


# -- B-086: the weights digest binds names and sizes, not just concatenated bytes ----------


def test_weights_digest_changes_when_a_file_is_renamed(tmp_path):
    from hearth.registry.adapters import adapter_weights_sha

    d = tmp_path / "w"
    d.mkdir()
    (d / "adapters.safetensors").write_bytes(b"tensor bytes")
    before = adapter_weights_sha(d)
    (d / "adapters.safetensors").rename(d / "0000100_adapters.safetensors")
    assert adapter_weights_sha(d) != before


def test_weights_digest_changes_when_bytes_move_between_files(tmp_path):
    from hearth.registry.adapters import adapter_weights_sha

    a, b = tmp_path / "a", tmp_path / "b"
    for d, (x, y) in ((a, (b"xy", b"z")), (b, (b"x", b"yz"))):
        d.mkdir()
        (d / "1.safetensors").write_bytes(x)
        (d / "2.safetensors").write_bytes(y)
    assert adapter_weights_sha(a) != adapter_weights_sha(b)


def test_weights_digest_ignores_dot_files_but_not_other_files(tmp_path):
    from hearth.registry.adapters import adapter_weights_sha

    d = tmp_path / "w"
    d.mkdir()
    (d / "adapters.safetensors").write_bytes(b"tensor bytes")
    before = adapter_weights_sha(d)
    (d / ".DS_Store").write_bytes(b"finder")
    assert adapter_weights_sha(d) == before
    (d / "adapter_config.json").write_text("{}")
    assert adapter_weights_sha(d) != before
