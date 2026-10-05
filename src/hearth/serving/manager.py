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
  * **Thread-safe.** A load lock serializes admissions, so two requests racing for the same
    cold model don't double-load or overflow; a separate state lock guards the resident map
    and is only ever held briefly, so readers (``/ready``, the admin view) never wait out a
    multi-second model load.
  * **Graceful degradation.** A provider whose ``load()``/construction raises is not
    marked resident (its RAM is not counted), and the error propagates to the caller so
    the gateway can fall back — a failed load never corrupts the accounting.
  * **No eviction for a load that cannot start.** The ceiling check and the provider's
    ``preflight`` (weights resolve on disk, for MLX) run before any resident is evicted, so
    a missing model leaves the residents intact (B-072). A load that passes preflight and
    then fails (a corrupt checkpoint) has already freed its victims: restoring them would
    mean overshooting the ceiling on every load or another multi-GB reload that can fail
    too. They stay ``loaded_once`` with weights on disk, so they reload on demand.
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
        self._lock = threading.RLock()  # resident-map state; held briefly
        self._load_lock = threading.RLock()  # serializes construct/evict/load/admit
        # Load OUTCOMES per model id, for readiness: ids whose load completed with weights in
        # memory at least once, and the error of the most recent failed load attempt
        # (cleared by the next successful one). Eviction does not erase either: a model that
        # loaded and was evicted to make room is still known-loadable.
        self._loaded_once: set[str] = set()
        self._load_errors: dict[str, str] = {}

    def get(self, model_id: str) -> ModelProvider:
        """Return a ready provider for ``model_id``, loading + evicting LRU as needed.

        On a hit the model is marked most-recently-used. On a miss the model is
        constructed, ``load()``-ed, and admitted after evicting enough LRU residents to
        keep the total footprint within the ceiling.
        """
        hit = self._hit(model_id)
        if hit is not None:
            return hit
        with self._load_lock:
            hit = self._hit(model_id)  # another thread may have loaded it while we waited
            if hit is not None:
                return hit
            return self._load(model_id)

    def _hit(self, model_id: str) -> ModelProvider | None:
        with self._lock:
            resident = self._resident.get(model_id)
            if resident is None:
                return None
            self._resident.move_to_end(model_id)
            return resident.provider

    def peek(self, model_id: str) -> ModelProvider | None:
        """The resident provider for ``model_id`` without loading or touching LRU order."""
        with self._lock:
            resident = self._resident.get(model_id)
            return resident.provider if resident is not None else None

    def residents(self) -> list[Resident]:
        """Snapshot of resident entries in LRU→MRU order (oldest first)."""
        with self._lock:
            return list(self._resident.values())

    def _load(self, model_id: str) -> ModelProvider:
        """Construct, load, and admit ``model_id`` (caller holds the load lock)."""
        try:
            provider = self._admit(model_id)
        except Exception as exc:
            with self._lock:
                self._load_errors[model_id] = f"{type(exc).__name__}: {exc}"
            raise
        with self._lock:
            self._load_errors.pop(model_id, None)
            # Only a load that left weights in memory counts. A provider that cannot say
            # (no ``is_loaded``) is taken at its word, as the residency view does.
            if getattr(provider, "is_loaded", True) is not False:
                self._loaded_once.add(model_id)
        return provider

    def _admit(self, model_id: str) -> ModelProvider:
        provider = self._factory(model_id)
        ram_gb = max(0.0, provider.footprint(model_id).ram_gb)
        too_large = self.size_problem(model_id, ram_gb)
        if too_large is not None:
            raise ModelTooLargeError(too_large)
        # Nothing is evicted until the load is known to be able to start (B-072). A
        # provider's ``preflight`` checks what it can without loading (MLX: the weights
        # resolve on disk); it raises on a load that cannot work, leaving every resident in
        # place. A request for a registered-but-never-pulled model used to unload a working
        # 14B first and fail second.
        preflight = getattr(provider, "preflight", None)
        if callable(preflight):
            preflight(model_id)
        self._evict_until_fits(ram_gb)
        # Load only after we've made room, so a heavy load doesn't briefly overshoot.
        load = getattr(provider, "load", None)
        if callable(load):
            load(model_id)
        with self._lock:
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

    def size_problem(self, model_id: str, ram_gb: float) -> str | None:
        """Why a model of ``ram_gb`` can NEVER be admitted under this ceiling, or ``None``.

        The one comparison :meth:`_admit` refuses on (:class:`ModelTooLargeError`), exposed
        so readiness judges a rung by the same rule that will refuse its every request
        (B-100) rather than by a copy of it.
        """
        ram_gb = max(0.0, ram_gb)
        if ram_gb > self.ram_ceiling_gb:
            return f"{model_id} needs {ram_gb} GB > ceiling {self.ram_ceiling_gb} GB"
        return None

    def _evict_until_fits(self, incoming_ram_gb: float) -> None:
        """Evict LRU residents until ``incoming_ram_gb`` fits under the ceiling.

        The incoming model is known to fit on its own (checked in :meth:`_load`), so this
        loop always terminates — worst case it empties the resident set.
        """
        while True:
            with self._lock:
                if not self._resident or (
                    self.resident_ram_gb() + incoming_ram_gb <= self.ram_ceiling_gb
                ):
                    return
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
        with self._load_lock:
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

    def loaded_once(self, model_id: str) -> bool:
        """Whether a load of ``model_id`` has ever completed with weights in memory."""
        with self._lock:
            return model_id in self._loaded_once

    def last_load_error(self, model_id: str) -> str | None:
        """The error of the most recent load attempt of ``model_id``, if that attempt failed."""
        with self._lock:
            return self._load_errors.get(model_id)

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
