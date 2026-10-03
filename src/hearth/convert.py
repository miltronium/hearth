"""Quantization / conversion pipeline (ARCHITECTURE §5, Phase 7).

Wraps ``mlx_lm.convert`` to quantize/convert a checkpoint into an MLX-servable model, so a
new base model can be brought into the registry. Like the LoRA orchestrator
(:mod:`hearth.training.lora`), the heavy work is delegated to an **injectable runner** —
tests pass a fake and never launch a real conversion (model download is proxy-blocked and
slow). The default runner calls ``mlx_lm.convert`` behind the ``[mlx]`` extra.

Real path (needs the ``[mlx]`` extra and source weights on disk):

    uv sync --extra mlx --extra mcp --extra dev --extra files
    hearth models convert --source <hf-repo-or-path> --out ~/.hearth/models/<id> -q 4

Disk-only with no ``HF_HUB_OFFLINE`` export (:func:`convert_invocation`): ``--source`` is
resolved to a local path first, and ``mlx_lm convert`` runs in a child process with the hub
pinned offline unless ``HEARTH_ALLOW_DOWNLOADS=1``. The path matters beyond the weights:
handed a repo id, ``mlx_lm``'s ``save()`` also fetches that repo's model card from the hub.

``mlx_lm`` runs only in that child, so importing this module (and the whole test suite)
needs no extras.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# A runner performs the conversion for a resolved config and returns the output path.
# Injectable so tests fake it (never a real convert). The default calls mlx_lm.convert.
Runner = Callable[["ConvertConfig"], Path]

# MLX quantization presets: bits → group size. 4-bit is the workhorse (best size/quality
# tradeoff on Apple Silicon); 8-bit trades size for quality. Matches mlx_lm defaults.
_VALID_BITS = (2, 3, 4, 6, 8)


class ConvertUnavailableError(RuntimeError):
    """Raised when a real conversion is requested but ``mlx-lm`` isn't importable."""


@dataclass(frozen=True)
class ConvertConfig:
    """Inputs for one quantization/conversion run.

    ``source`` is an HF repo id or a local path to the source checkpoint; ``output_dir``
    is where the MLX-format model is written. ``quantize`` toggles quantization at
    ``q_bits`` (with ``q_group_size``); when False the model is only format-converted.
    """

    source: str
    output_dir: Path
    quantize: bool = True
    q_bits: int = 4
    q_group_size: int = 64

    def validate(self) -> None:
        """Raise :class:`ValueError` unless the config is convertible."""
        if not self.source:
            raise ValueError("source is required")
        if not self.output_dir:
            raise ValueError("output_dir is required")
        if self.quantize and self.q_bits not in _VALID_BITS:
            raise ValueError(f"q_bits must be one of {_VALID_BITS}, got {self.q_bits}")
        if self.q_group_size <= 0:
            raise ValueError("q_group_size must be positive")


@dataclass(frozen=True)
class ConvertOutcome:
    """Result of a conversion — the output path plus the settings used."""

    source: str
    output_dir: Path
    quantized: bool
    q_bits: int | None


def convert(config: ConvertConfig, *, runner: Runner | None = None) -> ConvertOutcome:
    """Orchestrate a conversion: validate, delegate to ``runner``, report the outcome.

    ``runner`` defaults to :func:`_mlx_convert_runner` (real conversion via
    ``mlx_lm.convert``); tests inject a fake that writes a stub model dir and returns it.
    """
    config.validate()
    run = runner or _mlx_convert_runner
    out = run(config)
    return ConvertOutcome(
        source=config.source,
        output_dir=Path(out),
        quantized=config.quantize,
        q_bits=config.q_bits if config.quantize else None,
    )


def convert_invocation(
    config: ConvertConfig, *, allow_downloads: bool | None = None
) -> tuple[list[str], dict[str, str]]:
    """The exact ``(command, child_env)`` the real runner launches — resolution included.

    ``--source`` becomes the local path :func:`resolve_local_model` finds (raising
    :class:`~hearth.providers.mlx.ModelNotOnDiskError` when it is not on disk and downloads
    are off), and the child env pins the hub offline. Separate from the runner so
    ``hearth doctor --offline`` can measure this path without converting anything.
    """
    from .providers.mlx import downloads_allowed, model_load_env, resolve_local_model

    allow_downloads = downloads_allowed(allow_downloads)
    source = resolve_local_model(config.source, allow_downloads=allow_downloads)
    command = [
        sys.executable, "-m", "mlx_lm", "convert",
        "--hf-path", source,
        "--mlx-path", str(config.output_dir),
    ]
    if config.quantize:
        command += ["-q", "--q-bits", str(config.q_bits),
                    "--q-group-size", str(config.q_group_size)]
    return command, model_load_env(allow_downloads)


def _mlx_convert_runner(config: ConvertConfig) -> Path:
    """Default runner: ``python -m mlx_lm convert`` in a child (needs the ``[mlx]`` extra).

    A child process rather than an in-process call so the hub's offline switches can be set
    for it without touching this process's environment. Raising with the fix hint mirrors
    :class:`hearth.providers.mlx.MLXUnavailableError`.
    """
    import importlib.util

    if importlib.util.find_spec("mlx_lm") is None:
        raise ConvertUnavailableError(
            "mlx-lm is not installed. Install the conversion backend with: "
            "uv sync --extra mlx --extra mcp --extra dev --extra files"
        )
    command, env = convert_invocation(config)
    # mlx_lm.convert refuses an existing --mlx-path ("Cannot save to the path … as it
    # already exists"), so create only the parent. This runner used to mkdir the output dir
    # itself, which made every real conversion fail at that check.
    if config.output_dir.exists():
        raise FileExistsError(
            f"output dir {config.output_dir} already exists; mlx_lm convert writes a fresh "
            "directory — remove it or choose another --out"
        )
    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True, env=env)
    return config.output_dir


__all__ = [
    "ConvertConfig",
    "ConvertOutcome",
    "ConvertUnavailableError",
    "Runner",
    "convert",
    "convert_invocation",
]
