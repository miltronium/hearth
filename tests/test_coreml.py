"""Core ML export pipeline tests — orchestration + sidecar contract with a fake runner (ADR-011).

Never runs a real export (model download is proxy-blocked and Core ML conversion is slow). A
fake runner stands in for ``coremltools``; tests assert validation, that the config is threaded
through, that the sidecar (manifest + tokenizer files) is written, that the manifest round-trips
and rejects an incompatible schema, and cover the unavailable-extra path by faking importability.
"""

from __future__ import annotations

import json

import pytest

from hearth import coreml
from hearth.coreml import (
    MANIFEST_SCHEMA_VERSION,
    CoreMLExportConfig,
    CoreMLExportOutcome,
    CoreMLExportUnavailableError,
    CoreMLManifest,
    CoreMLRunResult,
    export,
    sidecar_paths,
    write_sidecar,
)


def _manifest(source: str = "org/model", **over) -> CoreMLManifest:
    base = dict(source=source, max_seq_len=512, vocab_size=100, eos_token_ids=[2, 7])
    base.update(over)
    return CoreMLManifest(**base)


def _fake_runner_factory(seen: dict, *, tokenizer_dir=None):
    def fake_runner(config: CoreMLExportConfig) -> CoreMLRunResult:
        seen["source"] = config.source
        seen["compute_units"] = config.compute_units
        seen["precision"] = config.precision
        seen["max_seq_len"] = config.max_seq_len
        # A .mlpackage is a directory bundle; write a stub to prove the path is created.
        config.output_dir.mkdir(parents=True, exist_ok=True)
        (config.output_dir / "Manifest.json").write_text("{}")
        return CoreMLRunResult(
            output_dir=config.output_dir,
            manifest=_manifest(
                source=config.source,
                max_seq_len=config.max_seq_len,
                compute_units=config.compute_units,
                precision=config.precision,
            ),
            tokenizer_dir=tokenizer_dir,
        )

    return fake_runner


# --- Orchestration ----------------------------------------------------------------------------


def test_export_delegates_to_runner(tmp_path):
    seen = {}
    out = tmp_path / "model.mlpackage"
    outcome = export(
        CoreMLExportConfig(source="org/model", output_dir=out, precision="float16"),
        runner=_fake_runner_factory(seen),
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
    seen = {}
    outcome = export(
        CoreMLExportConfig(
            source="org/model",
            output_dir=tmp_path / "m.mlpackage",
            compute_units="all",
            precision="int8",
            max_seq_len=1024,
        ),
        runner=_fake_runner_factory(seen),
    )
    assert outcome.compute_units == "all"
    assert outcome.precision == "int8"
    assert outcome.max_seq_len == 1024


# --- Sidecar contract -------------------------------------------------------------------------


def test_export_writes_manifest_sidecar(tmp_path):
    out = tmp_path / "qwen-coder.mlpackage"
    outcome = export(
        CoreMLExportConfig(source="org/qwen", output_dir=out),
        runner=_fake_runner_factory({}),
    )
    expected = sidecar_paths(out)["manifest"]
    assert outcome.manifest_path == expected
    assert expected.name == "qwen-coder.hearth-coreml.json"
    data = json.loads(expected.read_text())
    parsed = CoreMLManifest.from_dict(data)
    assert parsed.source == "org/qwen"
    assert parsed.eos_token_ids == [2, 7]
    assert parsed.schema_version == MANIFEST_SCHEMA_VERSION
    # No tokenizer_dir on the fake runner → no tokenizer files recorded.
    assert parsed.tokenizer_files == []
    assert outcome.tokenizer_paths == []


def test_export_copies_tokenizer_files(tmp_path):
    tok = tmp_path / "tok"
    tok.mkdir()
    (tok / "tokenizer.json").write_text('{"model": "fake-bpe"}')
    (tok / "tokenizer_config.json").write_text('{"chat_template": "chatml"}')

    out = tmp_path / "m.mlpackage"
    outcome = export(
        CoreMLExportConfig(source="org/m", output_dir=out),
        runner=_fake_runner_factory({}, tokenizer_dir=tok),
    )
    paths = sidecar_paths(out)
    assert paths["tokenizer"].exists()
    assert paths["tokenizer_config"].exists()
    assert paths["tokenizer"].read_text() == '{"model": "fake-bpe"}'
    assert set(outcome.tokenizer_paths) == {paths["tokenizer"], paths["tokenizer_config"]}
    # The written manifest records exactly the tokenizer files that shipped.
    manifest = CoreMLManifest.from_dict(json.loads(paths["manifest"].read_text()))
    assert set(manifest.tokenizer_files) == {
        "m.tokenizer.json",
        "m.tokenizer_config.json",
    }


def test_write_sidecar_requires_tokenizer_json(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()  # no tokenizer.json inside
    with pytest.raises(FileNotFoundError):
        write_sidecar(tmp_path / "m.mlpackage", _manifest(), tokenizer_dir=empty)


def test_write_sidecar_config_optional(tmp_path):
    tok = tmp_path / "tok"
    tok.mkdir()
    (tok / "tokenizer.json").write_text("{}")  # no tokenizer_config.json
    manifest_path, tokenizer_paths = write_sidecar(
        tmp_path / "m.mlpackage", _manifest(), tokenizer_dir=tok
    )
    assert len(tokenizer_paths) == 1
    manifest = CoreMLManifest.from_dict(json.loads(manifest_path.read_text()))
    assert manifest.tokenizer_files == ["m.tokenizer.json"]


def test_sidecar_paths_are_stem_prefixed_siblings():
    from pathlib import Path

    paths = sidecar_paths(Path("/a/b/qwen.mlpackage"))
    assert paths["manifest"] == Path("/a/b/qwen.hearth-coreml.json")
    assert paths["tokenizer"] == Path("/a/b/qwen.tokenizer.json")
    assert paths["tokenizer_config"] == Path("/a/b/qwen.tokenizer_config.json")


# --- Manifest round-trip / schema guard -------------------------------------------------------


def test_manifest_round_trips():
    m = _manifest(bos_token_id=1, tokenizer_files=["m.tokenizer.json"])
    assert CoreMLManifest.from_dict(m.to_dict()) == m


def test_manifest_rejects_unknown_schema_version():
    data = _manifest().to_dict()
    data["schema_version"] = 999
    with pytest.raises(ValueError, match="schema_version"):
        CoreMLManifest.from_dict(data)


def test_manifest_requires_an_eos_token():
    data = _manifest().to_dict()
    data["eos_token_ids"] = []
    with pytest.raises(ValueError, match="eos_token_id"):
        CoreMLManifest.from_dict(data)


# --- Config validation ------------------------------------------------------------------------


def test_export_rejects_empty_source(tmp_path):
    with pytest.raises(ValueError):
        export(
            CoreMLExportConfig(source="", output_dir=tmp_path / "o"),
            runner=_fake_runner_factory({}),
        )


def test_export_rejects_bad_compute_units(tmp_path):
    with pytest.raises(ValueError):
        export(
            CoreMLExportConfig(source="x", output_dir=tmp_path / "o", compute_units="gpuOnly"),
            runner=_fake_runner_factory({}),
        )


def test_export_rejects_bad_precision(tmp_path):
    with pytest.raises(ValueError):
        export(
            CoreMLExportConfig(source="x", output_dir=tmp_path / "o", precision="bf16"),
            runner=_fake_runner_factory({}),
        )


def test_export_rejects_bad_max_seq_len(tmp_path):
    with pytest.raises(ValueError):
        export(
            CoreMLExportConfig(source="x", output_dir=tmp_path / "o", max_seq_len=0),
            runner=_fake_runner_factory({}),
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
