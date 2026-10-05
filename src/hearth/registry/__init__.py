"""Model registry — the declarative catalog of servable models (ADR / ARCHITECTURE §5).

The registry is *data, not code*: it loads ``config/models.yaml``, exposes the entries,
and resolves the default model. The gateway's ``/v1/models`` and the ``hearth models``
CLI read from here so the catalog can change without code edits.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

logger = logging.getLogger("hearth.registry")


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

    def __init__(self, entries: list[ModelEntry], default: str, source: Path | None = None) -> None:
        self._entries = entries
        self._by_id = {e.id: e for e in entries}
        self._default = default
        #: The catalog file this registry was loaded from (``None`` when built in code).
        self.source = source
        self._warned_overrides: set[str] = set()

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
        """The model served when a request names none.

        ``HEARTH_DEFAULT_MODEL`` wins over the catalog's ``default:`` key when it names a
        model this registry actually holds. This property is the only reader of that
        variable: ``Settings`` has no ``default_model`` field (it had one, read by nothing,
        while every caller that asked the *registry* got the YAML value — so setting the
        variable produced a result from a different model than the one named, with nothing
        reporting the substitution; B-042). ``scripts/hearth_status.py`` lists it among the
        names read outside Settings. An override that names
        an unregistered model is ignored here rather than obeyed, so code that only needs an
        id keeps working; but the commands that serve (``serve``/``run``/``agent``/``mcp``)
        start through :meth:`require_default`, which refuses such an override (B-047).
        """
        override = os.environ.get("HEARTH_DEFAULT_MODEL", "").strip()
        if override and override in self._by_id:
            return override
        if override and override not in self._warned_overrides:
            # Once per registry per value: serve startup asks for this, and so does every
            # request that names no model (B-029). `hearth doctor` reports it too.
            self._warned_overrides.add(override)
            logger.warning(
                "HEARTH_DEFAULT_MODEL=%r is not in the model registry: `hearth serve`/`run`/"
                "`agent`/`mcp` refuse to start on it; anything asking only for an id gets "
                "the catalog default %r",
                override,
                self._default,
            )
        return self._default

    @property
    def ignored_default_override(self) -> str | None:
        """``HEARTH_DEFAULT_MODEL`` when it is set but :attr:`default_id` ignores it.

        The same test :attr:`default_id` applies, read from the same place, so a report
        built on this cannot disagree with the model that actually serves.
        """
        override = os.environ.get("HEARTH_DEFAULT_MODEL", "").strip()
        return override if override and override not in self._by_id else None

    def require_default(self) -> str:
        """The default model id, or :class:`UnregisteredDefaultModelError` (B-047).

        The strict form of :attr:`default_id`, for the places a command *starts*
        (``serve``, ``run``, ``agent``, ``mcp``). An explicitly set ``HEARTH_DEFAULT_MODEL``
        that names no registered model is an error here, not a fallback: falling back left
        ``/ready`` reporting the named model as failed while ``model=auto`` was answered by
        the catalog default — two parts of one server disagreeing about what serves. Unset
        (or empty) still means the catalog default. :attr:`default_id` keeps its lenient
        behaviour for code that only needs an id to display or compare.
        """
        ignored = self.ignored_default_override
        if ignored is not None:
            raise UnregisteredDefaultModelError(
                ignored, [e.id for e in self._entries], source=self.source
            )
        return self.default_id


class UnregisteredDefaultModelError(ValueError):
    """``HEARTH_DEFAULT_MODEL`` is set to an id the model registry does not hold."""

    def __init__(
        self, model_id: str, registered: list[str], source: Path | None = None
    ) -> None:
        self.model_id = model_id
        self.registered = list(registered)
        super().__init__(
            f"HEARTH_DEFAULT_MODEL={model_id!r} is not in the model registry "
            f"({source or 'config/models.yaml'}). Registered ids: "
            f"{', '.join(self.registered) or '(none)'}. "
            "Fix the id, register the model, or unset HEARTH_DEFAULT_MODEL to use the "
            "catalog default."
        )


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
    return Registry(entries, default, source=path)


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
    "UnregisteredDefaultModelError",
    "load_registry",
    "get_registry",
    "default_registry_path",
    "AdapterEntry",
    "AdapterStore",
    "AdapterError",
    "GateNotPassedError",
]
