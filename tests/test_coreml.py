"""Core ML export pipeline tests — orchestration with a fake runner (Phase 6).

Never runs a real export (model download is proxy-blocked and Core ML conversion is slow). A
fake runner stands in for ``coremltools``; tests assert validation and that the config is
threaded through, and cover the unavailable-extra path by faking importability.
"""

from __future__ import annotations

import pytest

from hearth import coreml
from hearth.coreml import (
    CoreMLExportConfig,
    CoreMLExportOutcome,
    CoreMLExportUnavailableError,
    export,
)


def test_export_delegates_to_runner(tmp_path):
    seen = {}

    def fake_runner(config: CoreMLExportConfig):
        seen["source"] = config.source
        seen["compute_units"] = config.compute_units
        seen["precision"] = config.precision
        seen["max_seq_len"] = config.max_seq_len
        # A .mlpackage is a directory bundle; write a stub to prove the path is created.
        config.output_dir.mkdir(parents=True, exist_ok=True)
        (config.output_dir / "Manifest.json").write_text("{}")
        return config.output_dir

    out = tmp_path / "model.mlpackage"
    outcome = export(
        CoreMLExportConfig(source="org/model", output_dir=out, precision="float16"),
        runner=fake_runner,
    )
    assert isinstance(outcome, CoreMLExportOutcome)
    assert outcome.output_dir == out
    assert outcome.source == "org/model"
    assert outcome.compute_units == "cpuAndNeuralEngine"
    assert outcome.precision == "float16"
    assert outcome.max_seq_len == 512
    assert seen == {
        "source": "org/model",
        "compute_units": "cpuAndNeuralEngine",
        "precision": "float16",
        "max_seq_len": 512,
    }
    assert (out / "Manifest.json").exists()


def test_export_threads_custom_settings(tmp_path):
    outcome = export(
        CoreMLExportConfig(
            source="org/model",
            output_dir=tmp_path / "m.mlpackage",
            compute_units="all",
            precision="int8",
            max_seq_len=1024,
        ),
        runner=lambda c: c.output_dir,
    )
    assert outcome.compute_units == "all"
    assert outcome.precision == "int8"
    assert outcome.max_seq_len == 1024


def test_export_rejects_empty_source(tmp_path):
    with pytest.raises(ValueError):
        export(
            CoreMLExportConfig(source="", output_dir=tmp_path / "o"),
            runner=lambda c: c.output_dir,
        )


def test_export_rejects_bad_compute_units(tmp_path):
    with pytest.raises(ValueError):
        export(
            CoreMLExportConfig(source="x", output_dir=tmp_path / "o", compute_units="gpuOnly"),
            runner=lambda c: c.output_dir,
        )


def test_export_rejects_bad_precision(tmp_path):
    with pytest.raises(ValueError):
        export(
            CoreMLExportConfig(source="x", output_dir=tmp_path / "o", precision="bf16"),
            runner=lambda c: c.output_dir,
        )


def test_export_rejects_bad_max_seq_len(tmp_path):
    with pytest.raises(ValueError):
        export(
            CoreMLExportConfig(source="x", output_dir=tmp_path / "o", max_seq_len=0),
            runner=lambda c: c.output_dir,
        )


def test_default_runner_raises_when_coremltools_missing(tmp_path, monkeypatch):
    # Simulate the [coreml] extra not being installed: find_spec returns None. The default
    # runner must raise the typed error with the fix hint, never attempt a real import.
    import importlib.util

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "coremltools":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    config = CoreMLExportConfig(source="org/model", output_dir=tmp_path / "m.mlpackage")
    with pytest.raises(CoreMLExportUnavailableError) as excinfo:
        coreml._coreml_export_runner(config)
    assert "uv sync --extra coreml" in str(excinfo.value)


# --- CLI: `hearth models export-coreml` (mirrors tests/test_cli_phase7.py style) ---


def test_cli_export_coreml_rejects_bad_compute_units():
    from typer.testing import CliRunner

    from hearth.cli import app

    result = CliRunner().invoke(
        app,
        [
            "models",
            "export-coreml",
            "--source",
            "x",
            "--out",
            "/tmp/hearth-coreml",
            "--compute-units",
            "gpuOnly",
        ],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 1
    assert "Invalid Core ML export config" in result.stdout


def test_cli_export_coreml_unavailable_extra(tmp_path, monkeypatch):
    # With the [coreml] extra absent the command prints the fix hint and exits non-zero,
    # never attempting a real export.
    import importlib.util

    from typer.testing import CliRunner

    from hearth.cli import app

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "coremltools":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    result = CliRunner().invoke(
        app,
        ["models", "export-coreml", "--source", "org/m", "--out", str(tmp_path / "m.mlpackage")],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 1
    assert "uv sync --extra coreml" in result.stdout
