"""Memory-aware multi-model serving (ARCHITECTURE §5, Phase 7).

A :class:`ModelManager` holds up to a configured RAM ceiling of resident models, lazily
loading them on demand and **LRU-evicting** the least-recently-used one when loading a new
model would exceed the ceiling. This is the "default one resident base model + N small
adapters; lazy load/unload driven by ``footprint()`` against a configured RAM ceiling"
policy (ARCHITECTURE §5), generalized to serve several models concurrently within the
budget.

Design notes:

  * A **provider factory** maps ``model_id -> ModelProvider``. The manager owns the
    residency policy; the factory owns backend construction (built-in ``select_provider``
    for a single backend, or per-model construction). This keeps the manager backend-
    agnostic and fully testable with fake providers that report footprints.
  * ``footprint().ram_gb`` is the sizing signal. A model larger than the whole ceiling is
    refused (:class:`ModelTooLargeError`) rather than silently evicting everything and
    still overflowing.
  * **Thread-safe.** A single re-entrant lock guards residency state; loads happen under
    the lock so two requests racing for the same cold model don't double-load or overflow.
  * **Graceful degradation.** A provider whose ``load()``/construction raises is not
    marked resident (its RAM is not counted), and the error propagates to the caller so
    the gateway can fall back — a failed load never corrupts the accounting.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from ..providers.base import ModelProvider

logger = logging.getLogger("hearth.serving")

# A factory constructs (but does not necessarily load) a provider for a model id.
ProviderFactory = Callable[[str], ModelProvider]

# Default resident-model RAM ceiling in GB. Sized for a 36 GB machine leaving headroom for
# the OS and the gateway itself; override with HEARTH_RAM_CEILING_GB.
DEFAULT_RAM_CEILING_GB = 24.0


class ModelTooLargeError(RuntimeError):
    """Raised when a single model's footprint exceeds the entire RAM ceiling."""


@dataclass(frozen=True)
class Resident:
    """One model currently resident under the manager."""

    model_id: str
    provider: ModelProvider
    ram_gb: float


class ModelManager:
    """Keeps a bounded, LRU set of models resident within a RAM ceiling (thread-safe).

    ``get(model_id)`` returns a ready provider, loading it (and evicting LRU residents as
    needed) on a miss. Construction is delegated to ``factory``; a provider's
    ``footprint(model_id).ram_gb`` sizes it against the ceiling.
    """

    def __init__(
        self,
        factory: ProviderFactory,
        ram_ceiling_gb: float = DEFAULT_RAM_CEILING_GB,
    ) -> None:
        self._factory = factory
        self.ram_ceiling_gb = ram_ceiling_gb
        # Insertion order == LRU order; move_to_end on access marks most-recently-used.
        self._resident: OrderedDict[str, Resident] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, model_id: str) -> ModelProvider:
        """Return a ready provider for ``model_id``, loading + evicting LRU as needed.

        On a hit the model is marked most-recently-used. On a miss the model is
        constructed, ``load()``-ed, and admitted after evicting enough LRU residents to
        keep the total footprint within the ceiling.
        """
        with self._lock:
            resident = self._resident.get(model_id)
            if resident is not None:
                self._resident.move_to_end(model_id)
                return resident.provider
            return self._load(model_id)

    def _load(self, model_id: str) -> ModelProvider:
        """Construct, load, and admit ``model_id`` (caller holds the lock)."""
        provider = self._factory(model_id)
        ram_gb = max(0.0, provider.footprint(model_id).ram_gb)
        if ram_gb > self.ram_ceiling_gb:
            raise ModelTooLargeError(
                f"{model_id} needs {ram_gb} GB > ceiling {self.ram_ceiling_gb} GB"
            )
        self._evict_until_fits(ram_gb)
        # Load only after we've made room, so a heavy load doesn't briefly overshoot.
        load = getattr(provider, "load", None)
        if callable(load):
            load(model_id)
        self._resident[model_id] = Resident(model_id, provider, ram_gb)
        self._resident.move_to_end(model_id)
        logger.info(
            "loaded %s (%.1f GB); resident=%.1f/%.1f GB",
            model_id,
            ram_gb,
            self.resident_ram_gb(),
            self.ram_ceiling_gb,
        )
        return provider

    def _evict_until_fits(self, incoming_ram_gb: float) -> None:
        """Evict LRU residents until ``incoming_ram_gb`` fits under the ceiling.

        The incoming model is known to fit on its own (checked in :meth:`_load`), so this
        loop always terminates — worst case it empties the resident set.
        """
        while self._resident and self.resident_ram_gb() + incoming_ram_gb > self.ram_ceiling_gb:
            victim_id, victim = self._resident.popitem(last=False)  # LRU end
            self._unload(victim)
            logger.info("evicted LRU model %s (%.1f GB)", victim_id, victim.ram_gb)

    def _unload(self, resident: Resident) -> None:
        """Best-effort unload of an evicted model; never propagate its failure."""
        unload = getattr(resident.provider, "unload", None)
        if callable(unload):
            try:
                unload(resident.model_id)
            except Exception as exc:  # noqa: BLE001 — eviction must not fail the new load
                logger.warning("unload of %s raised: %s", resident.model_id, exc)

    def evict(self, model_id: str) -> bool:
        """Explicitly evict ``model_id`` if resident; return whether it was present."""
        with self._lock:
            resident = self._resident.pop(model_id, None)
            if resident is None:
                return False
            self._unload(resident)
            return True

    def is_resident(self, model_id: str) -> bool:
        """Return whether ``model_id`` is currently loaded."""
        with self._lock:
            return model_id in self._resident

    def resident_ids(self) -> list[str]:
        """Resident model ids in LRU→MRU order (oldest first)."""
        with self._lock:
            return list(self._resident.keys())

    def resident_ram_gb(self) -> float:
        """Total RAM (GB) currently attributed to resident models."""
        with self._lock:
            return sum(r.ram_gb for r in self._resident.values())


__all__ = [
    "ModelManager",
    "ModelTooLargeError",
    "Resident",
    "ProviderFactory",
    "DEFAULT_RAM_CEILING_GB",
]
