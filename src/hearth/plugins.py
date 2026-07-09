"""Plugin API via packaging entry points (ARCHITECTURE §4/§5, Phase 7).

HEARTH grows without core edits: a third-party package declares an entry point in one
of three documented groups and, once installed, its backend resolves by name through the
existing selection functions (``select_provider`` / ``select_embedder`` / vector-store
selection). No router, gateway, or registry change is needed — this is the ADR-004
promise ("adding a backend = one new class + a registry entry") extended to *external*
packages.

Entry-point groups (see docs/PLUGINS.md):

  * ``hearth.providers``      → a :class:`~hearth.providers.base.ModelProvider` factory
  * ``hearth.vector_stores``  → a :class:`~hearth.memory.store.VectorStore` factory
  * ``hearth.embedders``      → an :class:`~hearth.memory.embed.EmbeddingProvider` factory

Each entry point resolves to a **factory**: a zero-arg callable returning an instance that
satisfies the group's Protocol. (A class works directly when its ``__init__`` takes no
required args.) Discovery is lazy and cached; a broken plugin (import error, missing attr,
or a returned object that fails the Protocol check) logs a warning and is skipped — one
bad plugin never crashes startup.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from typing import Any

from .memory.embed import EmbeddingProvider
from .memory.store import VectorStore
from .providers.base import ModelProvider

logger = logging.getLogger("hearth.plugins")

# Entry-point group → the runtime-checkable Protocol a plugin instance must satisfy.
PROVIDER_GROUP = "hearth.providers"
VECTOR_STORE_GROUP = "hearth.vector_stores"
EMBEDDER_GROUP = "hearth.embedders"

_PROTOCOLS: dict[str, type] = {
    PROVIDER_GROUP: ModelProvider,
    VECTOR_STORE_GROUP: VectorStore,
    EMBEDDER_GROUP: EmbeddingProvider,
}


@dataclass(frozen=True)
class DiscoveredPlugin:
    """One entry point found in a HEARTH plugin group.

    ``ok`` is ``True`` when the entry point imported, produced an instance, and that
    instance satisfied the group's Protocol; otherwise ``detail`` carries the reason it
    was skipped so ``hearth plugins list`` can show *why*.
    """

    name: str
    group: str
    value: str  # the "module:attr" target the entry point points at
    ok: bool
    detail: str = ""


def _iter_entry_points(group: str) -> list[metadata.EntryPoint]:
    """Return installed entry points in ``group`` (empty when none are registered).

    ``importlib.metadata.entry_points`` grew a group-selecting kwarg in 3.10; the
    supported range (>=3.11) has it, so no legacy fallback is needed.
    """
    return list(metadata.entry_points(group=group))


def _instantiate(ep: metadata.EntryPoint) -> Any:
    """Load an entry point and call its factory to get a backend instance.

    The loaded object is expected to be a zero-arg factory (a plain class with a no-arg
    ``__init__`` qualifies). Any exception propagates to the caller, which turns it into a
    skipped :class:`DiscoveredPlugin` rather than a crash.
    """
    factory: Callable[[], Any] = ep.load()
    return factory()


def discover(group: str) -> list[DiscoveredPlugin]:
    """Enumerate, import, and validate every plugin registered in ``group``.

    Never raises: a plugin that fails to import, instantiate, or satisfy the group's
    Protocol is reported with ``ok=False`` and a reason, and discovery continues.
    """
    protocol = _PROTOCOLS.get(group)
    if protocol is None:
        raise ValueError(f"unknown plugin group: {group!r}")

    found: list[DiscoveredPlugin] = []
    for ep in _iter_entry_points(group):
        try:
            instance = _instantiate(ep)
        except Exception as exc:  # noqa: BLE001 — a bad plugin must never crash startup
            logger.warning("plugin %r in %s failed to load: %s", ep.name, group, exc)
            found.append(
                DiscoveredPlugin(ep.name, group, ep.value, ok=False, detail=f"load error: {exc}")
            )
            continue
        if not isinstance(instance, protocol):
            detail = f"does not satisfy {protocol.__name__}"
            logger.warning("plugin %r in %s %s; skipping", ep.name, group, detail)
            found.append(DiscoveredPlugin(ep.name, group, ep.value, ok=False, detail=detail))
            continue
        found.append(DiscoveredPlugin(ep.name, group, ep.value, ok=True))
    return found


def discover_all() -> list[DiscoveredPlugin]:
    """Discover plugins across all three groups (for ``hearth plugins list``)."""
    out: list[DiscoveredPlugin] = []
    for group in _PROTOCOLS:
        out.extend(discover(group))
    return out


def load_plugin(group: str, name: str) -> Any | None:
    """Return a fresh backend instance for the plugin ``name`` in ``group``, or ``None``.

    Returns ``None`` (never raises) when no such plugin is registered or it fails
    validation, so callers can cleanly fall back to a built-in backend. A fresh instance
    is constructed on each call — callers (e.g. ``select_provider``) own the lifecycle,
    matching how the built-in providers are constructed per process.
    """
    protocol = _PROTOCOLS.get(group)
    if protocol is None:
        raise ValueError(f"unknown plugin group: {group!r}")
    for ep in _iter_entry_points(group):
        if ep.name != name:
            continue
        try:
            instance = _instantiate(ep)
        except Exception as exc:  # noqa: BLE001 — degrade to built-in on a bad plugin
            logger.warning("plugin %r in %s failed to load: %s", name, group, exc)
            return None
        if not isinstance(instance, protocol):
            logger.warning(
                "plugin %r in %s does not satisfy %s; ignoring", name, group, protocol.__name__
            )
            return None
        return instance
    return None


__all__ = [
    "DiscoveredPlugin",
    "PROVIDER_GROUP",
    "VECTOR_STORE_GROUP",
    "EMBEDDER_GROUP",
    "discover",
    "discover_all",
    "load_plugin",
]
