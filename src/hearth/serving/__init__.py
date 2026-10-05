"""Multi-model serving (ARCHITECTURE §5, Phase 7).

Memory-aware residency: :class:`ModelManager` keeps a bounded, LRU set of models loaded
within a configured RAM ceiling, lazily loading on demand and evicting the least-recently-
used model to make room. :class:`ModelPool` is the local provider built on it: one provider
per model id, validated against the registry, so the model a request names is the model that
generates it.
"""

from __future__ import annotations

from .manager import (
    DEFAULT_RAM_CEILING_GB,
    ModelManager,
    ModelTooLargeError,
    ProviderFactory,
    Resident,
)
from .pool import (
    ModelPool,
    UnknownModelError,
    check_model,
    resolve_model_id,
    servable_for,
    servable_ids,
)

__all__ = [
    "ModelManager",
    "ModelPool",
    "UnknownModelError",
    "check_model",
    "resolve_model_id",
    "servable_for",
    "servable_ids",
    "ModelTooLargeError",
    "Resident",
    "ProviderFactory",
    "DEFAULT_RAM_CEILING_GB",
]
