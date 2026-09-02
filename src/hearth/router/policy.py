"""Declarative routing policy (ADR-005).

Loads ``config/routing.yaml`` into a validated :class:`RoutingPolicy`. Routing is *data,
not code*: per-class backend + escalation rules, global defaults, and a map of named
remote endpoints. A bad or missing YAML must **never** take the server down — on any
load/validation error we log a warning and fall back to safe built-in defaults.
"""

from __future__ import annotations

import logging
import os
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


def default_policy_path() -> Path:
    """Path to the bundled ``config/routing.yaml`` (override via ``HEARTH_ROUTING_YAML``)."""
    override = os.environ.get("HEARTH_ROUTING_YAML")
    if override:
        return Path(override)
    # repo root is three parents up from src/hearth/router/policy.py
    return Path(__file__).resolve().parents[3] / "config" / "routing.yaml"


def _known_model_ids() -> set[str] | None:
    """Servable model ids from the registry, or ``None`` when it can't be read.

    Used to reject a per-class ``local_model`` that names a model nobody can serve — the
    error belongs at config-load time, not at generation time. Deliberately best-effort: an
    unreadable *registry* must not veto an otherwise valid *routing* config, so we skip the
    check rather than fail closed on an unrelated file's problem.
    """
    try:
        from ..registry import get_registry

        return {entry.id for entry in get_registry().list()}
    except Exception as exc:  # noqa: BLE001 — registry trouble ⇒ skip the check, don't veto
        logger.warning("model registry unreadable; skipping local_model validation: %s", exc)
        return None


def load_policy(path: Path | None = None, known_models: set[str] | None = None) -> RoutingPolicy:
    """Load and validate the routing policy, falling back to safe defaults on any error.

    ``known_models`` overrides the set of ids a per-class ``local_model`` is checked against
    (default: the model registry). Tests inject it to stay off ``config/models.yaml``.
    """
    path = path or default_policy_path()
    try:
        raw = yaml.safe_load(path.read_text()) or {}
        known = known_models if known_models is not None else _known_model_ids()
        return _parse(raw, known_models=known)
    except (OSError, yaml.YAMLError, ValueError, KeyError, TypeError) as exc:
        logger.warning("invalid or missing routing.yaml (%s); using safe defaults: %s", path, exc)
        return _safe_defaults()


def _parse(raw: dict, known_models: set[str] | None = None) -> RoutingPolicy:
    """Parse a raw dict into a validated policy. Raises on structural problems."""
    d = raw.get("defaults", {}) or {}
    defaults = Defaults(
        local_model=str(d.get("local_model", "auto")),
        remote=str(d.get("remote", "default")),
        remote_budget_tokens_per_day=int(d.get("remote_budget_tokens_per_day", 200_000)),
    )

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
            # and should read as one. "auto" is the explicit "fall through to defaults" value
            # and names no model, so it is never looked up.
            unknown = known_models is not None and local_model not in known_models
            if unknown and local_model != "auto":
                raise ValueError(
                    f"class {name!r}: local_model {local_model!r} is not in the model "
                    f"registry (config/models.yaml); known ids: {sorted(known_models)}"
                )
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
]
