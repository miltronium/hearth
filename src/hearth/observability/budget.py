"""Token-budget accountant (ADR-007, ARCHITECTURE §8).

Tracks remote tokens spent per UTC day against ``remote_budget_tokens_per_day`` so the
router can prefer local while remote budget is scarce and deny escalation when it is
exhausted. Phase 2 keeps this **in-memory** (not persisted across restarts — a follow-up
can back it with the metrics store); day boundaries key on the UTC date string.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from functools import lru_cache


class BudgetAccountant:
    """Per-day remote-token budget. Thread-safe; resets automatically at UTC midnight."""

    def __init__(self, tokens_per_day: int) -> None:
        self.tokens_per_day = tokens_per_day
        self._lock = threading.Lock()
        self._day = _utc_day()
        self._spent = 0

    def _roll_day(self) -> None:
        """Reset the counter when the UTC date has advanced. Caller holds the lock."""
        today = _utc_day()
        if today != self._day:
            self._day = today
            self._spent = 0

    def remaining(self) -> int:
        """Tokens still available for remote calls today (never negative)."""
        with self._lock:
            self._roll_day()
            return max(0, self.tokens_per_day - self._spent)

    def spent(self) -> int:
        """Remote tokens spent so far today."""
        with self._lock:
            self._roll_day()
            return self._spent

    def can_afford(self, tokens: int) -> bool:
        """True if at least ``tokens`` of remote budget remain today.

        A non-positive budget disables escalation entirely (nothing is affordable).
        """
        if self.tokens_per_day <= 0:
            return False
        return self.remaining() >= max(0, tokens)

    def spend(self, tokens: int) -> None:
        """Record ``tokens`` of remote spend against today's budget."""
        if tokens <= 0:
            return
        with self._lock:
            self._roll_day()
            self._spent += tokens


def _utc_day() -> str:
    """Today's UTC date as an ISO ``YYYY-MM-DD`` string (the budget's day key)."""
    return datetime.now(UTC).date().isoformat()


@lru_cache(maxsize=1)
def get_budget() -> BudgetAccountant:
    """Return the process budget accountant, sized from the routing policy."""
    from ..router.policy import get_policy

    return BudgetAccountant(get_policy().defaults.remote_budget_tokens_per_day)


__all__ = ["BudgetAccountant", "get_budget"]
