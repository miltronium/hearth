"""Model registry — the declarative catalog of servable models (ADR / ARCHITECTURE §5).

The registry is *data, not code*: it loads ``config/models.yaml``, exposes the entries,
and resolves the default model. The gateway's ``/v1/models`` and the ``hearth models``
CLI read from here so the catalog can change without code edits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelEntry:
    """One servable model in the catalog."""

    id: str
    backend: str
    quant: str
    context: int
    ram_gb: float
    capabilities: list[str] = field(default_factory=list)
    source: str = ""


class Registry:
    """An in-memory view of the model catalog loaded from a YAML file."""

    def __init__(self, entries: list[ModelEntry], default: str) -> None:
        self._entries = entries
        self._by_id = {e.id: e for e in entries}
        self._default = default

    def list(self) -> list[ModelEntry]:
        """Return all catalog entries in declaration order."""
        return list(self._entries)

    def get(self, model_id: str) -> ModelEntry | None:
        """Return the entry for ``model_id``, or ``None`` if unknown."""
        return self._by_id.get(model_id)

    def resolve(self, model_id: str) -> ModelEntry:
        """Resolve a model id (``"auto"``/``""`` → default) to a concrete entry.

        Raises :class:`KeyError` if the requested id is not in the catalog.
        """
        wanted = self._default if model_id in ("auto", "") else model_id
        entry = self._by_id.get(wanted)
        if entry is None:
            raise KeyError(f"model not in registry: {wanted!r}")
        return entry

    @property
    def default_id(self) -> str:
        return self._default


def default_registry_path() -> Path:
    """Path to the bundled ``config/models.yaml`` (override via ``HEARTH_MODELS_YAML``)."""
    override = os.environ.get("HEARTH_MODELS_YAML")
    if override:
        return Path(override)
    # repo root is three parents up from this file: src/hearth/registry/__init__.py
    return Path(__file__).resolve().parents[3] / "config" / "models.yaml"


def load_registry(path: Path | None = None) -> Registry:
    """Load the registry from ``path`` (or the bundled default)."""
    path = path or default_registry_path()
    data = yaml.safe_load(path.read_text()) or {}
    entries = [
        ModelEntry(
            id=m["id"],
            backend=m["backend"],
            quant=m.get("quant", "none"),
            context=int(m.get("context", 0)),
            ram_gb=float(m.get("ram_gb", 0.0)),
            capabilities=list(m.get("capabilities", [])),
            source=m.get("source", ""),
        )
        for m in data.get("models", [])
    ]
    default = data.get("default") or (entries[0].id if entries else "")
    return Registry(entries, default)


@lru_cache(maxsize=1)
def get_registry() -> Registry:
    """Return the cached process registry loaded from the default path."""
    return load_registry()


from .adapters import (  # noqa: E402  (re-export after the model-registry core above)
    AdapterEntry,
    AdapterError,
    AdapterStore,
    GateNotPassedError,
)

__all__ = [
    "ModelEntry",
    "Registry",
    "load_registry",
    "get_registry",
    "default_registry_path",
    "AdapterEntry",
    "AdapterStore",
    "AdapterError",
    "GateNotPassedError",
]
