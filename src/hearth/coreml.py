"""Core ML / ANE export pipeline (Phase 6 extension point, ARCHITECTURE §5).

Converts a base model into a ``.mlpackage`` that the Swift ``CoreMLProvider`` (see
``swift/Sources/Hearth/CoreMLProvider.swift``) can load for fully-offline, ANE-accelerated
on-device inference. Like the quantization pipeline (:mod:`hearth.convert`) and the LoRA
orchestrator (:mod:`hearth.training.lora`), the heavy work is delegated to an **injectable
runner** — tests pass a fake and never launch a real export (model download is proxy-blocked
and Core ML conversion is slow). The default runner traces a Hugging Face model and converts
it via ``coremltools`` behind the ``[coreml]`` extra.

Real path (needs the ``[coreml]`` extra, source weights, and offline HF for cached inputs):

    uv sync --extra coreml
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    hearth models export-coreml --source <hf-repo-or-path> --out ~/.hearth/coreml/<id>

``coremltools`` / ``torch`` / ``transformers`` are imported only inside the default runner, so
importing this module (and the whole test suite) needs no extras.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# A runner performs the export for a resolved config and returns the output ``.mlpackage`` path.
# Injectable so tests fake it (never a real export). The default uses coremltools.
Runner = Callable[["CoreMLExportConfig"], Path]

# MLModel compute-unit placements exposed by coremltools. "cpuAndNeuralEngine" pins work to the
# ANE (+CPU fallback) for the offline/low-power path; "all" also allows the GPU. Names match the
# `MLComputeUnits` cases the Swift `CoreMLProvider` passes back through `MLModelConfiguration`.
_VALID_COMPUTE_UNITS = ("all", "cpuAndNeuralEngine", "cpuAndGPU", "cpuOnly")

# Weight precisions coremltools can emit. float16 is the on-device workhorse (half the size,
# ANE-native); int8 trades quality for size; float32 is debug-only.
_VALID_PRECISIONS = ("float16", "float32", "int8")


class CoreMLExportUnavailableError(RuntimeError):
    """Raised when a real export is requested but ``coremltools`` isn't importable."""


@dataclass(frozen=True)
class CoreMLExportConfig:
    """Inputs for one Core ML export run.

    ``source`` is an HF repo id or a local path to the source checkpoint; ``output_dir`` is
    where the ``.mlpackage`` is written. ``compute_units`` picks the runtime placement
    (default the ANE), ``precision`` the emitted weight dtype, and ``max_seq_len`` the fixed
    sequence length the model is traced/exported at (Core ML shapes are static).
    """

    source: str
    output_dir: Path
    compute_units: str = "cpuAndNeuralEngine"
    precision: str = "float16"
    max_seq_len: int = 512

    def validate(self) -> None:
        """Raise :class:`ValueError` unless the config is exportable."""
        if not self.source:
            raise ValueError("source is required")
        if not self.output_dir:
            raise ValueError("output_dir is required")
        if self.compute_units not in _VALID_COMPUTE_UNITS:
            raise ValueError(
                f"compute_units must be one of {_VALID_COMPUTE_UNITS}, got {self.compute_units!r}"
            )
        if self.precision not in _VALID_PRECISIONS:
            raise ValueError(
                f"precision must be one of {_VALID_PRECISIONS}, got {self.precision!r}"
            )
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")


@dataclass(frozen=True)
class CoreMLExportOutcome:
    """Result of an export — the output ``.mlpackage`` path plus the settings used."""

    source: str
    output_dir: Path
    compute_units: str
    precision: str
    max_seq_len: int


def export(
    config: CoreMLExportConfig, *, runner: Runner | None = None
) -> CoreMLExportOutcome:
    """Orchestrate a Core ML export: validate, delegate to ``runner``, report the outcome.

    ``runner`` defaults to :func:`_coreml_export_runner` (real conversion via
    ``coremltools``); tests inject a fake that writes a stub ``.mlpackage`` dir and returns it.
    """
    config.validate()
    run = runner or _coreml_export_runner
    out = run(config)
    return CoreMLExportOutcome(
        source=config.source,
        output_dir=Path(out),
        compute_units=config.compute_units,
        precision=config.precision,
        max_seq_len=config.max_seq_len,
    )


def _coreml_export_runner(config: CoreMLExportConfig) -> Path:
    """Default runner: convert an HF model to a ``.mlpackage`` (needs the ``[coreml]`` extra).

    Kept out of the tested path — tests always inject a fake runner. Raising with the fix hint
    mirrors :class:`hearth.convert.ConvertUnavailableError`. The conversion is minimal but
    plausible: load the HF model, trace it at a fixed sequence length, convert via
    ``coremltools`` and save the ``.mlpackage``.
    """
    import importlib.util

    if importlib.util.find_spec("coremltools") is None:
        raise CoreMLExportUnavailableError(
            "coremltools is not installed. Install the Core ML export backend with: "
            "uv sync --extra coreml"
        )

    # Deferred heavy imports — only reached on the real path, never in tests.
    import coremltools as ct
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    precision = {
        "float16": ct.precision.FLOAT16,
        "float32": ct.precision.FLOAT32,
        "int8": ct.precision.FLOAT16,  # int8 handled via post-conversion palettization below
    }[config.precision]
    compute_units = {
        "all": ct.ComputeUnit.ALL,
        "cpuAndNeuralEngine": ct.ComputeUnit.CPU_AND_NE,
        "cpuAndGPU": ct.ComputeUnit.CPU_AND_GPU,
        "cpuOnly": ct.ComputeUnit.CPU_ONLY,
    }[config.compute_units]

    tokenizer = AutoTokenizer.from_pretrained(config.source)
    model = AutoModelForCausalLM.from_pretrained(config.source, torchscript=True)
    model.eval()

    # Core ML needs a concrete traced graph with static shapes; trace a single forward at the
    # configured sequence length using a representative input.
    example = tokenizer("hello", return_tensors="pt", padding="max_length",
                        max_length=config.max_seq_len, truncation=True)
    input_ids = example["input_ids"]
    with torch.no_grad():
        traced = torch.jit.trace(model, input_ids)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input_ids", shape=input_ids.shape, dtype=int)],
        compute_precision=precision,
        compute_units=compute_units,
        minimum_deployment_target=ct.target.macOS13,
    )
    if config.precision == "int8":
        mlmodel = ct.optimize.coreml.palettize_weights(
            mlmodel, ct.optimize.coreml.OpPalettizerConfig(nbits=8)
        )

    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(config.output_dir))
    return config.output_dir


__all__ = [
    "CoreMLExportConfig",
    "CoreMLExportOutcome",
    "CoreMLExportUnavailableError",
    "Runner",
    "export",
]
