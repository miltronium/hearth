"""LoRA/QLoRA training orchestrator (ARCHITECTURE §7, ADR-006, Phase 4).

A thin, testable wrapper around ``mlx_lm.lora``. It does the parts we can test with no
GPU-time and no model download — validate inputs, lay out the run directory, split the
dataset into the ``train.jsonl`` / ``valid.jsonl`` files mlx-lm expects, and assemble the
invocation — then delegates the actual (slow, heavy) training to an injectable
``runner``. Tests pass a FAKE runner and never launch a real run.

Real path (needs the ``[mlx]`` extra and a base model already on disk):

    uv sync --extra mlx --extra mcp --extra dev --extra files
    hearth train --task extract --base <model-id> --data dataset.jsonl

The default runner shells out to ``python -m mlx_lm.lora --train`` with the assembled
args. It is disk-only without any ``HF_HUB_OFFLINE`` export (:func:`runner_invocation`):
``--model`` is resolved to a local path first (``~/.hearth/models``, then the hub cache;
:class:`~hearth.providers.mlx.ModelNotOnDiskError` if in neither), and the child runs with
``HF_HUB_OFFLINE=1``/``TRANSFORMERS_OFFLINE=1`` unless ``HEARTH_ALLOW_DOWNLOADS=1``.
mlx-lm is imported/invoked only inside that default runner, so importing this module
(and the whole test suite) needs no extras.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .dataset import Dataset, DatasetError

# A runner takes the resolved mlx-lm argument list + the run directory and performs the
# training, returning the path to the produced adapter. Injectable so tests fake it.
Runner = Callable[[list[str], Path], Path]


@dataclass(frozen=True)
class LoRAConfig:
    """Inputs for one LoRA/QLoRA training run.

    ``dataset`` and ``base_model`` are required; the rest mirror ``mlx_lm.lora`` knobs
    with conservative small-model defaults (ADR-006: 3B–14B on Apple Silicon).
    """

    base_model: str
    task: str
    dataset: Dataset
    output_dir: Path
    iters: int = 200
    batch_size: int = 4
    learning_rate: float = 1e-5
    num_layers: int = 16
    valid_fraction: float = 0.1
    seed: int = 0
    extra_args: list[str] = field(default_factory=list)

    def validate(self) -> None:
        """Raise :class:`ValueError`/:class:`DatasetError` unless the config is trainable."""
        if not self.base_model:
            raise ValueError("base_model is required")
        if not self.task:
            raise ValueError("task is required")
        if self.iters <= 0 or self.batch_size <= 0:
            raise ValueError("iters and batch_size must be positive")
        if not (0.0 <= self.valid_fraction < 1.0):
            raise ValueError("valid_fraction must be in [0.0, 1.0)")
        self.dataset.validate()
        if len(self.dataset) < 2:
            raise DatasetError("need at least 2 records to split into train/valid")


@dataclass(frozen=True)
class TrainOutcome:
    """Result of a training run — enough to register the adapter as a candidate."""

    train_run_id: str
    base_model: str
    task: str
    adapter_path: Path
    args: list[str]
    num_records: int


def train(
    config: LoRAConfig,
    *,
    runner: Runner | None = None,
    train_run_id: str = "",
) -> TrainOutcome:
    """Orchestrate a LoRA/QLoRA run: validate, prepare files/args, delegate to ``runner``.

    ``runner`` defaults to :func:`_mlx_lm_runner` (real training via ``mlx_lm.lora``);
    tests inject a fake that writes a dummy adapter and returns its path. ``train_run_id``
    is passed in by the caller for determinism (the CLI derives one from a timestamp).
    """
    config.validate()
    if runner is None:
        # Real path: fail on a missing [mlx] extra or a base that is not on disk BEFORE
        # laying out the run dir, so a failed `hearth train` leaves nothing behind (B-026).
        _require_mlx_lm()
        from ..providers.mlx import resolve_local_model

        resolve_local_model(config.base_model)
    run_dir = _prepare_run_dir(config)
    args = _build_args(config, run_dir)
    run = runner or _mlx_lm_runner
    adapter_path = run(args, run_dir)
    return TrainOutcome(
        train_run_id=train_run_id or run_dir.name,
        base_model=config.base_model,
        task=config.task,
        adapter_path=Path(adapter_path),
        args=args,
        num_records=len(config.dataset),
    )


def _prepare_run_dir(config: LoRAConfig) -> Path:
    """Create the run/data dir and write the ``train.jsonl`` / ``valid.jsonl`` splits.

    mlx-lm's LoRA trainer reads ``train.jsonl`` and ``valid.jsonl`` from a ``--data``
    directory. The split is deterministic (order-preserving, no shuffling here) so the
    same dataset always lays out identically.
    """
    data_dir = config.output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_recs, valid_recs = _split(config.dataset.records, config.valid_fraction)
    _write_jsonl(data_dir / "train.jsonl", [r.to_json() for r in train_recs])
    _write_jsonl(data_dir / "valid.jsonl", [r.to_json() for r in valid_recs])
    return config.output_dir


def _split(records: Sequence, valid_fraction: float) -> tuple[list, list]:
    """Split records into (train, valid), holding out the tail as validation.

    Always leaves at least one record on each side (guaranteed by the >=2 check in
    :meth:`LoRAConfig.validate`).
    """
    n_valid = max(1, int(round(len(records) * valid_fraction)))
    n_valid = min(n_valid, len(records) - 1)
    cut = len(records) - n_valid
    return list(records[:cut]), list(records[cut:])


def _write_jsonl(path: Path, objs: list[dict]) -> None:
    path.write_text("".join(json.dumps(o, sort_keys=True) + "\n" for o in objs), encoding="utf-8")


def _build_args(config: LoRAConfig, run_dir: Path) -> list[str]:
    """Assemble the ``python -m mlx_lm.lora`` argument list (the invocation, not the run)."""
    adapter_path = run_dir / "adapters"
    args = [
        "--train",
        "--model", config.base_model,
        "--data", str(run_dir / "data"),
        "--adapter-path", str(adapter_path),
        "--iters", str(config.iters),
        "--batch-size", str(config.batch_size),
        "--learning-rate", str(config.learning_rate),
        "--num-layers", str(config.num_layers),
        "--seed", str(config.seed),
    ]
    args += list(config.extra_args)
    return args


def _preflight_batch_size(args: list[str], run_dir: Path) -> None:
    """Fail fast (before GPU time) if the validation split is smaller than ``batch_size``.

    ``mlx_lm.lora`` iterates the *validation* set in ``batch_size`` chunks and aborts with
    ``"Dataset must have at least batch_size=N examples but only has M"`` when the split is
    too small. Because HEARTH holds out ``valid_fraction`` of the records, a dataset that
    passes :meth:`LoRAConfig.validate` (>=2 records) can still yield a validation split
    below ``batch_size``. Surface that here with an actionable message instead of mlx-lm's
    opaque one. Real-path only (tests inject a fake runner and never reach this).
    """
    try:
        batch_size = int(args[args.index("--batch-size") + 1])
    except (ValueError, IndexError):
        return  # no/unparseable batch size — let mlx-lm speak for itself
    valid_path = run_dir / "data" / "valid.jsonl"
    if not valid_path.exists():
        return
    n_valid = sum(1 for line in valid_path.read_text().splitlines() if line.strip())
    if n_valid < batch_size:
        raise DatasetError(
            f"validation split has {n_valid} example(s) but batch_size is {batch_size}; "
            f"mlx-lm needs at least {batch_size}. Provide more records (roughly "
            f">= {batch_size}/valid_fraction total, e.g. ~{batch_size * 10} at the 0.1 "
            "default) or lower the batch size."
        )


def runner_invocation(
    args: list[str], *, allow_downloads: bool | None = None
) -> tuple[list[str], dict[str, str]]:
    """The exact ``(command, child_env)`` the real runner launches — resolution included.

    ``mlx_lm.lora --model <repo-id>`` calls ``snapshot_download`` with the network on, so
    ``--model`` is swapped for the local path :func:`resolve_local_model` finds (raising
    :class:`~hearth.providers.mlx.ModelNotOnDiskError` when the base is not on disk and
    downloads are off), and the child env pins the hub offline. The ``args`` HEARTH records
    keep the model *id*, which is what the adapter registry must store: the router matches
    a promoted adapter's ``base_model`` against the served model id, not a cache path.
    Separate from the runner so ``hearth doctor --offline`` can measure this path without
    training anything.
    """
    from ..providers.mlx import downloads_allowed, model_load_env, resolve_local_model

    allow_downloads = downloads_allowed(allow_downloads)
    resolved = list(args)
    try:
        at = resolved.index("--model") + 1
        model_id = resolved[at]
    except (ValueError, IndexError):
        raise ValueError("mlx_lm.lora args carry no --model value") from None
    resolved[at] = resolve_local_model(model_id, allow_downloads=allow_downloads)
    command = [sys.executable, "-m", "mlx_lm.lora", *resolved]
    return command, model_load_env(allow_downloads)


def _require_mlx_lm() -> None:
    """Raise with the install hint unless ``mlx_lm`` is importable (no import is done)."""
    import importlib.util

    if importlib.util.find_spec("mlx_lm") is None:
        raise RuntimeError(
            "mlx-lm is not installed. Install the training backend with: "
            "uv sync --extra mlx --extra mcp --extra dev --extra files"
        )


def _mlx_lm_runner(args: list[str], run_dir: Path) -> Path:
    """Default runner: shell out to ``python -m mlx_lm.lora`` (needs the ``[mlx]`` extra).

    Tests inject a fake runner for training; the launch itself is covered by faking
    ``subprocess.run`` (tests/test_offline_load_paths.py). Raising with the fix hint mirrors
    :class:`hearth.providers.mlx.MLXUnavailableError`.
    """
    _require_mlx_lm()
    # Resolve before the preflight: a base that is not on disk fails here, in a second.
    command, env = runner_invocation(args)
    _preflight_batch_size(args, run_dir)
    subprocess.run(command, check=True, env=env)
    return run_dir / "adapters"


__all__ = ["LoRAConfig", "TrainOutcome", "Runner", "runner_invocation", "train"]
