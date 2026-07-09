"""Provider registry and selection.

Selection logic: honor ``HEARTH_BACKEND`` (``auto`` | ``mlx`` | ``echo`` | a plugin name).
``auto`` prefers MLX when importable and falls back to the echo stub otherwise, so the
server always starts. The model id comes from the model registry's default (ARCHITECTURE
§5). Any other value is resolved against the ``hearth.providers`` plugin entry-point group
(Phase 7), so a third-party backend serves via ``HEARTH_BACKEND=<plugin-name>`` with zero
core edits.
"""

from __future__ import annotations

from ..config import Settings, get_settings
from ..registry import get_registry
from .base import ModelProvider
from .echo import EchoProvider
from .mlx import MLXProvider, mlx_available


def select_provider(settings: Settings | None = None) -> ModelProvider:
    """Return the active provider for this process based on configuration."""
    settings = settings or get_settings()
    choice = settings.backend.lower()
    default_model = get_registry().default_id

    if choice == "echo":
        return EchoProvider()
    if choice == "mlx":
        return MLXProvider(default_model)
    if choice == "auto":
        return MLXProvider(default_model) if mlx_available() else EchoProvider()

    # Fall through to a plugin-provided backend registered under `hearth.providers`.
    # Match the entry-point name as configured (case-sensitive), not the lowercased form.
    from ..plugins import PROVIDER_GROUP, load_plugin

    plugin = load_plugin(PROVIDER_GROUP, settings.backend)
    if plugin is not None:
        return plugin
    raise ValueError(
        f"Unknown HEARTH_BACKEND: {settings.backend!r} "
        "(use auto|mlx|echo, or install a plugin registering this name under hearth.providers)"
    )


__all__ = ["select_provider", "EchoProvider", "MLXProvider", "ModelProvider"]
