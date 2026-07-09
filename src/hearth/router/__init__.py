"""Router / policy layer (ARCHITECTURE §3, ADR-005, ADR-007).

Turns a request into a decision — which backend/model serves it and whether to escalate —
then executes it and records telemetry. Routing behavior is declarative data
(``config/routing.yaml``), not code; the engine here executes that policy.
"""

from __future__ import annotations

from .classify import TASK_CLASSES, classify
from .policy import RoutingPolicy, load_policy
from .route import BudgetExhaustedError, RouteDecision, Router, RouteResult

__all__ = [
    "TASK_CLASSES",
    "classify",
    "RoutingPolicy",
    "load_policy",
    "Router",
    "RouteDecision",
    "RouteResult",
    "BudgetExhaustedError",
]
