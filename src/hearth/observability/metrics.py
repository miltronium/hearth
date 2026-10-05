"""Per-request telemetry and rollups (ARCHITECTURE §8).

Records one :class:`RequestRecord` per served request and computes the rollups that
justify the project: estimated frontier tokens saved, escalation rate, backend mix, and
p50/p95 latency. Phase 2 keeps records in an **in-memory ring buffer** (not persisted
across restarts; a follow-up can spool them to JSONL as ARCHITECTURE §8 anticipates).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from functools import lru_cache

# Class-aware savings multipliers (ARCHITECTURE §8). Replaces Phase 0's flat
# "saved = total tokens": a frontier model would spend roughly this many tokens per token
# a local model produces for the class, so a locally-served request "saves" that estimate.
# Cheap, high-volume classes (summarize/extract) save the most; reason least (it often
# *should* go to the frontier). Deliberately conservative — tune against real data later.
SAVINGS_MULTIPLIER: dict[str, float] = {
    "summarize": 1.5,
    "extract": 1.5,
    "classify": 1.2,
    "rank": 1.2,
    "draft": 1.3,
    "code": 1.4,
    "reason": 0.8,
    "chat": 1.0,
    "embed": 1.0,
}
_DEFAULT_MULTIPLIER = 1.0

# Bound the ring buffer so a long-lived daemon can't grow memory without limit.
_RING_CAPACITY = 10_000


def estimated_tokens_saved(task_class: str, prompt_tokens: int, completion_tokens: int) -> int:
    """Estimate frontier tokens saved by serving ``task_class`` locally (ARCHITECTURE §8).

    Returns 0 for remotely-served requests (the caller decides); this only prices the
    counterfactual "what a frontier call would have cost" for a local hit.
    """
    total = max(0, prompt_tokens) + max(0, completion_tokens)
    return round(total * SAVINGS_MULTIPLIER.get(task_class, _DEFAULT_MULTIPLIER))


@dataclass(frozen=True)
class RequestRecord:
    """One request's telemetry (ARCHITECTURE §8) — served, or failed (``failed`` set)."""

    task_class: str
    backend: str
    model: str
    # "local" | "remote": the tier that served — or, when ``failed`` is set, the tier that
    # was tried last and failed (nothing was served).
    served_by: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    escalated: bool = False
    escalation_reason: str | None = None
    # Set when the request TRIED to escalate and the remote failed, so it was served local
    # (router/route.py:degrade_to_local) — or, when ``failed`` is also set, the local
    # fallback failed too and nothing was served. Holds the remote's error. Without it a frontier
    # outage would read in the metrics as a policy that simply never escalated.
    escalation_failed: str | None = None
    adapter: str | None = None
    estimated_frontier_tokens_saved: int = 0
    # Set when the client got an ERROR instead of an answer (the local provider failed, or a
    # remote stream died mid-answer). Holds the error. A failed request used to write no
    # record at all, so an outage plus a broken local model read as zero traffic — and a
    # remote that had already received the prompt left no trace (docs/BUGS.md B-003).
    failed: str | None = None
    ts: float = field(default_factory=time.time)


class MetricsStore:
    """Thread-safe in-memory ring buffer of records plus rollup queries."""

    def __init__(self, capacity: int = _RING_CAPACITY) -> None:
        self._records: deque[RequestRecord] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def record(self, rec: RequestRecord) -> None:
        with self._lock:
            self._records.append(rec)

    def mark_failed(self, rec: RequestRecord, reason: str) -> None:
        """Re-mark an already-recorded request as failed (same record, ``failed`` set).

        For a failure discovered after the generation was recorded — JSON mode rejecting
        an unparseable object is the case: the client got an error, so the request must not
        count as served. Replaces the record in place (matched by identity) so the rollup
        counts it once, as a failure; a record no longer in the ring is a no-op.
        """
        failed = replace(rec, failed=reason)
        with self._lock:
            for i, r in enumerate(self._records):
                if r is rec:
                    self._records[i] = failed
                    return

    def _since(self, since_s: float | None) -> list[RequestRecord]:
        cutoff = 0.0 if since_s is None else time.time() - since_s
        with self._lock:
            return [r for r in self._records if r.ts >= cutoff]

    def rollup(self, since_s: float | None = None) -> dict:
        """Compute rollups over records newer than ``since_s`` seconds ago (all if None)."""
        records = self._since(since_s)
        total = len(records)
        if total == 0:
            return {
                "requests": 0,
                "estimated_frontier_tokens_saved": 0,
                "escalations": 0,
                "escalation_rate": 0.0,
                "escalations_failed": 0,
                "failed": 0,
                "failure_rate": 0.0,
                "backend_mix": {},
                "class_mix": {},
                "latency_ms": {"p50": 0.0, "p95": 0.0},
            }

        saved = sum(r.estimated_frontier_tokens_saved for r in records)
        escalations = sum(1 for r in records if r.escalated)
        escalations_failed = sum(1 for r in records if r.escalation_failed)
        failed = sum(1 for r in records if r.failed)
        backend_mix: dict[str, int] = {}
        class_mix: dict[str, int] = {}
        for r in records:
            # backend_mix is what SERVED answers; a failed request served nothing, so it is
            # counted under ``failed`` instead of inflating the tier it failed on.
            if not r.failed:
                backend_mix[r.served_by] = backend_mix.get(r.served_by, 0) + 1
            class_mix[r.task_class] = class_mix.get(r.task_class, 0) + 1
        latencies = sorted(r.latency_ms for r in records if not r.failed)
        return {
            "requests": total,
            "estimated_frontier_tokens_saved": saved,
            "escalations": escalations,
            "escalation_rate": round(escalations / total, 4),
            "escalations_failed": escalations_failed,
            "failed": failed,
            "failure_rate": round(failed / total, 4),
            "backend_mix": backend_mix,
            "class_mix": class_mix,
            "latency_ms": {
                "p50": round(_percentile(latencies, 50), 2),
                "p95": round(_percentile(latencies, 95), 2),
            },
        }


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile of an already-sorted, non-empty list."""
    if not sorted_values:
        return 0.0
    k = max(0, min(len(sorted_values) - 1, round((pct / 100.0) * len(sorted_values) + 0.5) - 1))
    return sorted_values[k]


@lru_cache(maxsize=1)
def get_metrics() -> MetricsStore:
    """Return the process-wide metrics store."""
    return MetricsStore()


__all__ = [
    "RequestRecord",
    "MetricsStore",
    "estimated_tokens_saved",
    "get_metrics",
    "SAVINGS_MULTIPLIER",
]
