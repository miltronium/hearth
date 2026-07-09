"""ModelManager tests — memory-aware residency + LRU eviction (Phase 7).

Fully offline: fake providers report footprints; nothing is loaded for real. Verifies the
RAM-ceiling policy from ARCHITECTURE §5 — lazy load, LRU eviction when a new load would
overflow, thread safety, and refusal of an oversized model.
"""

from __future__ import annotations

import threading

import pytest

from hearth.serving import ModelManager, ModelTooLargeError


class FakeProvider:
    """A provider that reports a fixed footprint and counts load/unload calls."""

    def __init__(self, model_id: str, ram_gb: float) -> None:
        self.name = f"fake:{model_id}"
        self._model_id = model_id
        self._ram_gb = ram_gb
        self.loads = 0
        self.unloads = 0

    def load(self, model_id: str) -> None:
        self.loads += 1

    def unload(self, model_id: str) -> None:
        self.unloads += 1

    def footprint(self, model_id: str):
        class _E:
            ram_gb = self._ram_gb

        return _E()


def _factory(sizes: dict[str, float]):
    """Build a factory returning a FakeProvider sized per ``sizes`` (default 1 GB)."""
    made: dict[str, FakeProvider] = {}

    def factory(model_id: str) -> FakeProvider:
        p = FakeProvider(model_id, sizes.get(model_id, 1.0))
        made[model_id] = p
        return p

    factory.made = made  # type: ignore[attr-defined]
    return factory


def test_lazy_load_on_get():
    factory = _factory({"a": 2.0})
    mgr = ModelManager(factory, ram_ceiling_gb=10.0)
    assert not mgr.is_resident("a")
    p = mgr.get("a")
    assert mgr.is_resident("a")
    assert p.loads == 1


def test_hit_does_not_reload():
    factory = _factory({"a": 2.0})
    mgr = ModelManager(factory, ram_ceiling_gb=10.0)
    p1 = mgr.get("a")
    p2 = mgr.get("a")
    assert p1 is p2
    assert p1.loads == 1  # loaded exactly once


def test_two_models_resident_within_ceiling():
    factory = _factory({"a": 4.0, "b": 4.0})
    mgr = ModelManager(factory, ram_ceiling_gb=10.0)
    mgr.get("a")
    mgr.get("b")
    # Both fit under 10 GB → concurrent residency (the acceptance criterion).
    assert set(mgr.resident_ids()) == {"a", "b"}
    assert mgr.resident_ram_gb() == 8.0


def test_lru_eviction_when_ceiling_exceeded():
    factory = _factory({"a": 4.0, "b": 4.0, "c": 4.0})
    mgr = ModelManager(factory, ram_ceiling_gb=10.0)
    mgr.get("a")
    mgr.get("b")
    mgr.get("c")  # would be 12 GB → evict LRU ("a")
    assert set(mgr.resident_ids()) == {"b", "c"}
    assert not mgr.is_resident("a")
    assert mgr.resident_ram_gb() == 8.0


def test_lru_order_respects_recent_use():
    factory = _factory({"a": 4.0, "b": 4.0, "c": 4.0})
    mgr = ModelManager(factory, ram_ceiling_gb=10.0)
    mgr.get("a")
    mgr.get("b")
    mgr.get("a")  # touch "a" so "b" is now the LRU
    mgr.get("c")  # evicts "b", not "a"
    assert mgr.is_resident("a")
    assert not mgr.is_resident("b")
    assert mgr.is_resident("c")


def test_evicted_model_is_unloaded():
    factory = _factory({"a": 6.0, "b": 6.0})
    mgr = ModelManager(factory, ram_ceiling_gb=10.0)
    a = mgr.get("a")
    mgr.get("b")  # evicts "a"
    assert a.unloads == 1


def test_model_larger_than_ceiling_refused():
    factory = _factory({"huge": 100.0})
    mgr = ModelManager(factory, ram_ceiling_gb=10.0)
    with pytest.raises(ModelTooLargeError):
        mgr.get("huge")
    assert mgr.resident_ids() == []


def test_explicit_evict():
    factory = _factory({"a": 2.0})
    mgr = ModelManager(factory, ram_ceiling_gb=10.0)
    a = mgr.get("a")
    assert mgr.evict("a") is True
    assert not mgr.is_resident("a")
    assert a.unloads == 1
    assert mgr.evict("a") is False  # already gone


def test_failed_load_not_counted_as_resident():
    def factory(model_id: str):
        raise RuntimeError("load blew up")

    mgr = ModelManager(factory, ram_ceiling_gb=10.0)
    with pytest.raises(RuntimeError):
        mgr.get("a")
    # A failed load must not corrupt accounting or leave a phantom resident.
    assert mgr.resident_ids() == []
    assert mgr.resident_ram_gb() == 0.0


def test_thread_safe_concurrent_gets():
    factory = _factory({m: 1.0 for m in "abcde"})
    mgr = ModelManager(factory, ram_ceiling_gb=100.0)
    errors: list[Exception] = []

    def worker(model_id: str):
        try:
            for _ in range(50):
                mgr.get(model_id)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(m,)) for m in "abcde"]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert set(mgr.resident_ids()) == set("abcde")
    # Each model loaded exactly once despite concurrent access.
    assert all(p.loads == 1 for p in factory.made.values())  # type: ignore[attr-defined]
