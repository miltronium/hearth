"""Runtime configuration and on-disk paths.

Settings resolve from (in order): explicit args -> environment (``HEARTH_*``) ->
defaults. Runtime state lives under ``~/.hearth`` unless ``HEARTH_HOME`` overrides it.
"""

from __future__ import annotations

import os
import secrets
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process-wide configuration, populated from ``HEARTH_*`` environment variables."""

    model_config = SettingsConfigDict(env_prefix="HEARTH_", extra="ignore")

    # Server
    host: str = "127.0.0.1"  # loopback-only by default; see ADR-002 / SECURITY posture
    port: int = 8080

    # Backend selection: "auto" uses MLX when importable, else falls back to "echo".
    # "mlx" forces the real backend; "echo" forces the deterministic stub.
    backend: str = "auto"

    # Default local model id. In Phase 0 this is the single hardcoded MLX model.
    default_model: str = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"

    # Root for runtime state (token, model cache, logs).
    home: Path = Path.home() / ".hearth"

    @property
    def token_path(self) -> Path:
        return self.home / "token"

    @property
    def models_dir(self) -> Path:
        return self.home / "models"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached process settings."""
    return Settings()


def ensure_home(settings: Settings | None = None) -> Path:
    """Create ``~/.hearth`` (and the models dir) if missing. Returns the home path."""
    settings = settings or get_settings()
    settings.home.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    return settings.home


def get_or_create_token(settings: Settings | None = None) -> str:
    """Return the local bearer token, generating a 0600 file on first use.

    The token gates the extension/admin routes. Because the server is loopback-only
    by default this is low-stakes, but it prevents other local users from driving it.
    """
    settings = settings or get_settings()
    ensure_home(settings)
    path = settings.token_path
    if path.exists():
        return path.read_text().strip()
    token = secrets.token_urlsafe(32)
    # Write then tighten permissions to owner read/write only.
    path.write_text(token + "\n")
    os.chmod(path, 0o600)
    return token
