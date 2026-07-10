"""Hermetic smoke tests for the scripts/ + examples/ follow-up artifacts.

No network, no live daemon, no real model. These assert the shell harness is syntactically
sound and `--help`-clean, and that the Python offload example parses/dry-runs offline.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TRAIN_SCRIPT = _REPO_ROOT / "scripts" / "train_lora_real.sh"
_OFFLOAD_EXAMPLE = _REPO_ROOT / "examples" / "cambot_offload.py"


def test_train_script_exists_and_executable() -> None:
    assert _TRAIN_SCRIPT.is_file(), "scripts/train_lora_real.sh is missing"


def test_train_script_syntax_is_valid() -> None:
    """`bash -n` catches syntax errors without executing the script (no GPU/network)."""
    result = subprocess.run(
        ["bash", "-n", str(_TRAIN_SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"bash -n failed:\n{result.stderr}"


def test_train_script_help_is_clean() -> None:
    """`--help` must exit 0 and print usage without touching the network or a model."""
    result = subprocess.run(
        ["bash", str(_TRAIN_SCRIPT), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"--help exited {result.returncode}:\n{result.stderr}"
    assert "Usage:" in result.stdout


def test_train_script_requires_data() -> None:
    """Missing --data fails fast with a non-zero exit (a prereq guard, no downloads)."""
    result = subprocess.run(
        ["bash", str(_TRAIN_SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--data is required" in result.stderr


def test_offload_example_help_parses() -> None:
    """The Python example must import + parse args offline (guarded network call)."""
    result = subprocess.run(
        [sys.executable, str(_OFFLOAD_EXAMPLE), "--help"],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, f"--help exited {result.returncode}:\n{result.stderr}"
    assert "--live" in result.stdout


def test_offload_example_dry_run_makes_no_network_call() -> None:
    """Without --live the example prints a dry run and exits 0 (no daemon required)."""
    result = subprocess.run(
        [sys.executable, str(_OFFLOAD_EXAMPLE)],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, f"dry run exited {result.returncode}:\n{result.stderr}"
    assert "[dry run]" in result.stdout
