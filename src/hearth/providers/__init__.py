"""Provider registry and selection.

Phase 0 selection logic: honor ``HEARTH_BACKEND`` (``auto`` | ``mlx`` | ``echo``).
``auto`` prefers MLX when importable and falls back to the echo stub otherwise, so the
server always starts.
"""

from __future__ import annotations

from ..config import Settings, get_settings
from .base import ModelProvider
from .echo import EchoProvider
from .mlx import MLXProvider, mlx_available


def select_provider(settings: Settings | None = None) -> ModelProvider:
    """Return the active provider for this process based on configuration."""
    settings = settings or get_settings()
    choice = settings.backend.lower()

    if choice == "echo":
        return EchoProvider()
    if choice == "mlx":
        return MLXProvider(settings.default_model)
    if choice == "auto":
        return MLXProvider(settings.default_model) if mlx_available() else EchoProvider()
    raise ValueError(f"Unknown HEARTH_BACKEND: {settings.backend!r} (use auto|mlx|echo)")


__all__ = ["select_provider", "EchoProvider", "MLXProvider", "ModelProvider"]
