"""Multi-model serving (ARCHITECTURE §5, Phase 7).

Memory-aware residency: :class:`ModelManager` keeps a bounded, LRU set of models loaded
within a configured RAM ceiling, lazily loading on demand and evicting the least-recently-
used model to make room. The gateway asks the manager for a ready provider per request.
"""

from __future__ import annotations

from .manager import (
    DEFAULT_RAM_CEILING_GB,
    ModelManager,
    ModelTooLargeError,
    ProviderFactory,
    Resident,
)

__all__ = [
    "ModelManager",
    "ModelTooLargeError",
    "Resident",
    "ProviderFactory",
    "DEFAULT_RAM_CEILING_GB",
]
