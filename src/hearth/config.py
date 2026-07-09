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

    # Require a bearer token on all routes except /v1/hearth/admin/health. Tests set
    # this False (or pass the token); production keeps it on even on loopback.
    require_auth: bool = True

    # Default local model id. In Phase 0 this is the single hardcoded MLX model.
    default_model: str = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"

    # Embedding backend selection (Phase 3, ARCHITECTURE §6):
    #   "hash" — offline, dependency-free hashing embedder (default; used by tests/skeleton).
    #   "mlx"  — real local embeddings via the [embeddings] extra + a pre-pulled model.
    embedder: str = "hash"
    # Dimensionality for the offline hashing embedder.
    embed_dim: int = 256
    # Model id for the MLX embedder (only used when embedder="mlx").
    embed_model: str = "mlx-community/bge-small-en-v1.5-mlx"

    # Vector-store backend selection (Phase 7, ADR-008):
    #   "sqlite" — the embedded, file-based default (no extras, no service).
    #   any other value resolves a plugin registered under the `hearth.vector_stores`
    #   entry-point group, so a third-party store drops in with zero core edits.
    vector_store: str = "sqlite"

    # Multi-model serving (Phase 7, ARCHITECTURE §5). Total RAM (GB) budget for resident
    # models held by the ModelManager; a new load evicts the LRU model to stay under this.
    # Default leaves headroom for the OS + gateway on a 36 GB machine.
    ram_ceiling_gb: float = 24.0

    # Pre-load the default model on `hearth serve` start so the first request is warm
    # (Phase 7 hardening). Default on for the real backends, no-op for echo. A failed
    # warmup logs and continues in degraded mode — serve never blocks forever on it.
    warmup: bool = True

    # Root for runtime state (token, model cache, logs).
    home: Path = Path.home() / ".hearth"

    @property
    def token_path(self) -> Path:
        return self.home / "token"

    @property
    def models_dir(self) -> Path:
        return self.home / "models"

    @property
    def rag_dir(self) -> Path:
        return self.home / "rag"


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
