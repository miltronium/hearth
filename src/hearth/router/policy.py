"""Declarative routing policy (ADR-005).

Loads ``config/routing.yaml`` into a validated :class:`RoutingPolicy`. Routing is *data,
not code*: per-class backend + escalation rules, global defaults, and a map of named
remote endpoints. A bad or missing YAML must **never** take the server down — on any
load/validation error we log a warning and fall back to safe built-in defaults. The one
exception is a ``HEARTH_ROUTING_YAML`` that names a file which does not exist: that is the
operator selecting a profile by name, and it is a startup error (see :func:`load_policy`).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from .classify import TASK_CLASSES

logger = logging.getLogger("hearth.router.policy")

_BACKENDS = ("local", "remote")
_ESCALATE_MODES = ("never", "on_low_confidence", "always")


@dataclass(frozen=True)
class ClassRule:
    """Policy for one task class: where it runs, when it escalates, and on which model.

    ``local_model`` is the per-class rung of a **local model ladder**: a small fast model for
    high-volume structured classes (``classify``, ``extract``) and a larger one for synthesis
    (``summarize``, ``draft``, ``reason``, ``chat``). It is optional — ``None`` (or ``"auto"``)
    means "no opinion", and resolution falls through to ``defaults.local_model`` and then the
    registry default exactly as it did before the field existed. See
    :meth:`hearth.router.route.Router._local_model` for the full order.
    """

    backend: str = "local"
    escalate: str = "never"
    threshold: float = 0.6
    local_model: str | None = None


@dataclass(frozen=True)
class RemoteConfig:
    """One named remote endpoint the router can escalate to (see providers/remote.py)."""

    protocol: str  # "anthropic" | "openai"
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    thinking: bool = False


@dataclass(frozen=True)
class Defaults:
    """Global routing defaults."""

    local_model: str = "auto"
    remote: str = "default"  # names an entry in ``remotes``
    remote_budget_tokens_per_day: int = 200_000


@dataclass(frozen=True)
class RoutingPolicy:
    """The whole routing configuration: defaults, per-class rules, and remotes."""

    defaults: Defaults = field(default_factory=Defaults)
    classes: dict[str, ClassRule] = field(default_factory=dict)
    remotes: dict[str, RemoteConfig] = field(default_factory=dict)

    def rule_for(self, task_class: str) -> ClassRule:
        """Return the rule for ``task_class`` (safe local default if unspecified)."""
        return self.classes.get(task_class, ClassRule())

    def remote_for(self, name: str | None = None) -> RemoteConfig | None:
        """Return the named remote (or the configured default), or ``None`` if missing."""
        return self.remotes.get(name or self.defaults.remote)


# Safe built-in fallback used when routing.yaml is missing or invalid (ADR-005): keep every
# class local and never escalate, so a broken config degrades to "always local", never crash.
def _safe_defaults() -> RoutingPolicy:
    return RoutingPolicy(
        defaults=Defaults(),
        classes={c: ClassRule(backend="local", escalate="never") for c in TASK_CLASSES},
        remotes={},
    )


_ROUTING_ENV = "HEARTH_ROUTING_YAML"


class RoutingProfileNotFoundError(RuntimeError):
    """``HEARTH_ROUTING_YAML`` names a routing profile that does not exist."""


class RoutingPolicyError(RoutingProfileNotFoundError):
    """The routing profile parses, but a model rung in it cannot serve (B-065).

    Raised for ``defaults.local_model`` or a class ``local_model`` that names an id the
    model registry does not hold, a registered model that is not chat-capable (an embed
    model), or — checked against the live provider by :func:`check_policy_servable` — a
    model the active backend cannot serve (``echo`` under mlx).

    Unlike a structurally broken file this is NOT degraded to the safe defaults: the
    fallback is all-local and never leaks, but it silently replaces the operator's model
    ladder with the registry default, so the server would come up green answering with
    different weights than the profile names (the B-008 argument, CLAUDE.md §3).

    It subclasses :class:`RoutingProfileNotFoundError` so every place that already refuses
    to start on an unusable profile (the CLI's exit-2 handler, ``get_policy`` callers)
    refuses on this too, without each having to learn a second exception type.
    """


@dataclass(frozen=True)
class RoutingSelection:
    """Which routing profile is selected, and how.

    ``path`` is absolute. ``explicit`` is True when ``HEARTH_ROUTING_YAML`` chose it (the
    operator asked for that file by name); ``raw`` is the variable's value as written.
    """

    path: Path
    explicit: bool
    raw: str | None = None


def repo_root() -> Path:
    """The checkout this package was imported from (three parents up from this file)."""
    return Path(__file__).resolve().parents[3]


def resolve_routing_selection(environ: Mapping[str, str] | None = None) -> RoutingSelection:
    """The ONE resolver for ``HEARTH_ROUTING_YAML`` (B-008, B-025).

    Used by the router (:func:`load_policy` / :func:`get_policy`), the status probe and
    ``hearth doctor --offline``, so the path a report names is the path the router reads.

    Resolution:

    * unset or empty → the bundled ``<repo>/config/routing.yaml``;
    * ``~`` / ``~user`` is expanded;
    * an absolute path is used as is;
    * a **relative path is resolved against the repo root, not the current directory**.

    Why the repo root: the bundled default already lives there, and every shipped launcher
    does the same (``scripts/hearth_private.sh`` prefixes ``$REPO_ROOT``, ``cmux-open``
    passes ``$REPO_ROOT/config/...``), so ``HEARTH_ROUTING_YAML=config/routing.remote.yaml``
    means the same file in a hand-run ``hearth serve`` as it does there. CWD-relative made
    the selected profile depend on where the daemon happened to be started, and let a decoy
    ``config/routing*.yaml`` under that directory be loaded instead of the repo's.

    ``environ`` defaults to ``os.environ``; callers measuring another environment pass it.
    """
    env = os.environ if environ is None else environ
    raw = (env.get(_ROUTING_ENV) or "").strip()
    if not raw:
        return RoutingSelection(repo_root() / "config" / "routing.yaml", explicit=False)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = repo_root() / path
    return RoutingSelection(path, explicit=True, raw=raw)


def default_policy_path(environ: Mapping[str, str] | None = None) -> Path:
    """Absolute path of the routing profile the router loads (see
    :func:`resolve_routing_selection`)."""
    return resolve_routing_selection(environ).path


@dataclass(frozen=True)
class _ModelCatalog:
    """What a model rung is checked against: registered ids, and which of them chat.

    ``chat`` is ``None`` when only membership is known (a caller passed a bare id set).
    """

    known: frozenset[str]
    chat: frozenset[str] | None = None


def _registry_catalog() -> _ModelCatalog | None:
    """The model registry as a :class:`_ModelCatalog`, or ``None`` when it can't be read.

    Used to reject a ``local_model`` rung that names a model nobody can serve — the error
    belongs at config-load time, not at generation time. Deliberately best-effort: an
    unreadable *registry* must not veto an otherwise valid *routing* config, so we skip the
    check rather than fail closed on an unrelated file's problem.
    """
    try:
        from ..registry import get_registry

        entries = get_registry().list()
    except Exception as exc:  # noqa: BLE001 — registry trouble ⇒ skip the check, don't veto
        logger.warning("model registry unreadable; skipping local_model validation: %s", exc)
        return None
    return _ModelCatalog(
        known=frozenset(e.id for e in entries),
        chat=frozenset(e.id for e in entries if "chat" in e.capabilities),
    )


def _check_rung(where: str, model_id: str, catalog: _ModelCatalog | None) -> None:
    """Raise :class:`ValueError` when ``model_id`` (a ladder rung) cannot be a chat model.

    ``"auto"`` is the explicit "fall through" value and names no model.
    """
    if catalog is None or model_id == "auto":
        return
    if model_id not in catalog.known:
        raise _RungError(
            f"{where} {model_id!r} is not in the model registry (config/models.yaml); "
            f"known ids: {sorted(catalog.known)}"
        )
    if catalog.chat is not None and model_id not in catalog.chat:
        raise _RungError(
            f"{where} {model_id!r} is registered but not chat-capable, so no chat request "
            f"routed to it can be served; chat models: {sorted(catalog.chat)}"
        )


class _RungError(ValueError):
    """A model rung failed validation (kept distinct so load_policy does not swallow it)."""


def load_policy(path: Path | None = None, known_models: set[str] | None = None) -> RoutingPolicy:
    """Load and validate the routing policy, falling back to safe defaults on a broken file.

    ``known_models`` overrides the set of ids a ``local_model`` rung is checked against
    (default: the model registry, which also checks the rung is chat-capable). Tests inject
    it to stay off ``config/models.yaml``.

    A model rung that cannot serve — ``defaults.local_model`` or a class ``local_model``
    naming an unregistered id or a non-chat model — raises :class:`RoutingPolicyError`
    rather than falling back (B-065): the fallback would quietly replace the profile's
    ladder with the registry default.

    One exception to the fallback (B-008): with no ``path`` argument, when
    ``HEARTH_ROUTING_YAML`` names a file that **does not exist**, this raises
    :class:`RoutingProfileNotFoundError` instead of quietly substituting the safe defaults.
    The fallback is all-local, so it never leaks — but the operator asked for a specific
    profile by name and would otherwise be told nothing while running a different policy
    (e.g. believing escalation was on, or that the finance ladder was active). A missing
    file the operator *selected* is a startup error; a missing or broken *default*, and an
    unparseable selected file, still degrade to safe defaults per ADR-005.
    """
    if path is None:
        selection = resolve_routing_selection()
        if selection.explicit and not selection.path.is_file():
            raise RoutingProfileNotFoundError(
                f"{_ROUTING_ENV}={selection.raw!r} selects {selection.path}, which does not "
                "exist (relative paths resolve against the repo root, "
                f"{repo_root()}). Fix or unset {_ROUTING_ENV}."
            )
        path = selection.path
    try:
        raw = yaml.safe_load(path.read_text()) or {}
        catalog = (
            _ModelCatalog(known=frozenset(known_models))
            if known_models is not None
            else _registry_catalog()
        )
        return _parse(raw, catalog=catalog)
    except _RungError as exc:
        raise RoutingPolicyError(
            f"routing profile {path} names a model that cannot serve: {exc}. Fix the "
            "profile (or select another with HEARTH_ROUTING_YAML)."
        ) from None
    except (OSError, yaml.YAMLError, ValueError, KeyError, TypeError) as exc:
        logger.warning("invalid or missing routing.yaml (%s); using safe defaults: %s", path, exc)
        return _safe_defaults()


def _parse(
    raw: dict,
    known_models: set[str] | None = None,
    *,
    catalog: _ModelCatalog | None = None,
) -> RoutingPolicy:
    """Parse a raw dict into a validated policy. Raises on structural problems.

    ``catalog`` (or the bare id set ``known_models``) is what every model rung is checked
    against; neither means "skip the check" (registry unreadable).
    """
    if catalog is None and known_models is not None:
        catalog = _ModelCatalog(known=frozenset(known_models))
    d = raw.get("defaults", {}) or {}
    defaults = Defaults(
        local_model=str(d.get("local_model", "auto")),
        remote=str(d.get("remote", "default")),
        remote_budget_tokens_per_day=int(d.get("remote_budget_tokens_per_day", 200_000)),
    )
    # The rung every unpinned class falls through to: it serves exactly like a class rung,
    # so it is validated exactly like one (it used to be read verbatim, unchecked).
    _check_rung("defaults.local_model", defaults.local_model, catalog)

    classes: dict[str, ClassRule] = {}
    for name, spec in (raw.get("classes", {}) or {}).items():
        if name not in TASK_CLASSES:
            raise ValueError(f"unknown task class in routing.yaml: {name!r}")
        spec = spec or {}
        backend = str(spec.get("backend", "local"))
        escalate = str(spec.get("escalate", "never"))
        if backend not in _BACKENDS:
            raise ValueError(f"class {name!r}: backend must be one of {_BACKENDS}, got {backend!r}")
        if escalate not in _ESCALATE_MODES:
            raise ValueError(f"class {name!r}: escalate must be one of {_ESCALATE_MODES}")
        local_model = spec.get("local_model")
        if local_model is not None:
            local_model = str(local_model)
            # Fail here, not at generation time: a typo'd rung of the ladder is a config bug
            # and should read as one.
            _check_rung(f"class {name!r}: local_model", local_model, catalog)
        classes[name] = ClassRule(
            backend=backend,
            escalate=escalate,
            threshold=float(spec.get("threshold", 0.6)),
            local_model=local_model,
        )

    remotes: dict[str, RemoteConfig] = {}
    for name, spec in (raw.get("remotes", {}) or {}).items():
        spec = spec or {}
        protocol = str(spec.get("protocol", ""))
        if protocol not in ("anthropic", "openai"):
            raise ValueError(f"remote {name!r}: protocol must be 'anthropic' or 'openai'")
        remotes[name] = RemoteConfig(
            protocol=protocol,
            model=str(spec["model"]),
            base_url=spec.get("base_url"),
            api_key_env=spec.get("api_key_env"),
            thinking=bool(spec.get("thinking", False)),
        )

    return RoutingPolicy(defaults=defaults, classes=classes, remotes=remotes)


@lru_cache(maxsize=1)
def get_policy() -> RoutingPolicy:
    """Return the cached process routing policy loaded from the default path."""
    return load_policy()


__all__ = [
    "ClassRule",
    "RemoteConfig",
    "Defaults",
    "RoutingPolicy",
    "load_policy",
    "get_policy",
    "default_policy_path",
    "resolve_routing_selection",
    "RoutingSelection",
    "RoutingProfileNotFoundError",
    "RoutingPolicyError",
    "repo_root",
]
