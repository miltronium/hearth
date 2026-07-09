"""Environment preflight — ``hearth doctor``.

Checks the things that make or break local inference on this machine: Apple Silicon,
usable RAM, whether the MLX backend is installed, and whether the state dir is writable.
Returns structured results so the CLI can render them and set an exit code.
"""

from __future__ import annotations

import platform
import shutil
from dataclasses import dataclass

from .config import Settings, ensure_home, get_settings
from .providers.mlx import mlx_available


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    # A failed check may still be non-fatal (e.g. MLX missing -> echo fallback works).
    fatal: bool = False


def _total_ram_gb() -> float | None:
    """Best-effort total RAM in GB, or None if it can't be determined."""
    try:
        import os

        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round(pages * page_size / (1024**3), 1)
    except (ValueError, OSError, AttributeError):
        return None


def run_checks(settings: Settings | None = None) -> list[Check]:
    """Run all environment checks and return their results."""
    settings = settings or get_settings()
    checks: list[Check] = []

    # Apple Silicon
    machine = platform.machine()
    is_arm = machine == "arm64"
    checks.append(
        Check(
            "apple_silicon",
            is_arm,
            f"machine={machine} ({'Apple Silicon' if is_arm else 'not arm64'})",
            fatal=False,
        )
    )

    # RAM (32 GB+ is the documented baseline)
    ram = _total_ram_gb()
    ram_ok = ram is not None and ram >= 16
    checks.append(
        Check(
            "memory",
            ram_ok,
            f"{ram} GB total" + ("" if ram is None else " (baseline 32 GB, min 16 GB)"),
        )
    )

    # MLX backend availability (non-fatal: echo fallback exists)
    has_mlx = mlx_available()
    checks.append(
        Check(
            "mlx_backend",
            has_mlx,
            "mlx-lm importable"
            if has_mlx
            else "mlx-lm not installed (echo fallback active; `uv sync --extra mlx`)",
            fatal=False,
        )
    )

    # State dir writable
    try:
        ensure_home(settings)
        writable = shutil.disk_usage(settings.home).free > 0
        detail = f"{settings.home} writable"
    except OSError as exc:
        writable = False
        detail = f"{settings.home}: {exc}"
    checks.append(Check("state_dir", writable, detail, fatal=True))

    return checks


def all_fatal_passed(checks: list[Check]) -> bool:
    """True if no fatal check failed (non-fatal failures are tolerated)."""
    return all(c.ok for c in checks if c.fatal)
