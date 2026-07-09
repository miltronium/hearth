"""Observability & token accounting (ARCHITECTURE §8, ADR-007).

Cross-cutting, present from Phase 2. Provides:

  * :mod:`hearth.observability.budget` — per-UTC-day remote token accountant.
  * :mod:`hearth.observability.metrics` — per-request records + rollups.
"""

from __future__ import annotations

from .budget import BudgetAccountant, get_budget
from .metrics import (
    MetricsStore,
    RequestRecord,
    estimated_tokens_saved,
    get_metrics,
)

__all__ = [
    "BudgetAccountant",
    "get_budget",
    "MetricsStore",
    "RequestRecord",
    "estimated_tokens_saved",
    "get_metrics",
]
