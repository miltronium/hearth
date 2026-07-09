"""Observability rollup tests (ARCHITECTURE §8)."""

from __future__ import annotations

from hearth.observability.metrics import (
    MetricsStore,
    RequestRecord,
    estimated_tokens_saved,
)


def test_class_aware_savings_multiplier():
    # summarize (1.5x) saves more than a flat total; reason (0.8x) less.
    assert estimated_tokens_saved("summarize", 100, 100) == 300
    assert estimated_tokens_saved("reason", 100, 100) == 160
    assert estimated_tokens_saved("chat", 100, 100) == 200
    # unknown class defaults to 1.0x
    assert estimated_tokens_saved("mystery", 50, 50) == 100


def _rec(**kw) -> RequestRecord:
    base = dict(
        task_class="chat",
        backend="echo",
        model="m",
        served_by="local",
        prompt_tokens=10,
        completion_tokens=10,
        latency_ms=5.0,
        estimated_frontier_tokens_saved=20,
    )
    base.update(kw)
    return RequestRecord(**base)


def test_empty_rollup():
    roll = MetricsStore().rollup()
    assert roll["requests"] == 0
    assert roll["escalation_rate"] == 0.0


def test_rollup_aggregates():
    store = MetricsStore()
    store.record(_rec(latency_ms=10.0))
    store.record(_rec(latency_ms=20.0))
    store.record(
        _rec(served_by="remote", escalated=True, escalation_reason="class_policy",
             estimated_frontier_tokens_saved=0, latency_ms=100.0, task_class="reason")
    )
    roll = store.rollup()
    assert roll["requests"] == 3
    assert roll["estimated_frontier_tokens_saved"] == 40
    assert roll["escalations"] == 1
    assert round(roll["escalation_rate"], 3) == round(1 / 3, 3)
    assert roll["backend_mix"] == {"local": 2, "remote": 1}
    assert roll["class_mix"] == {"chat": 2, "reason": 1}
    assert roll["latency_ms"]["p95"] >= roll["latency_ms"]["p50"]


def test_ring_buffer_bounded():
    store = MetricsStore(capacity=5)
    for _ in range(20):
        store.record(_rec())
    assert store.rollup()["requests"] == 5
