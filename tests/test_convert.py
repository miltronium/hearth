"""Conversion pipeline tests — orchestration with a fake runner (Phase 7).

Never runs a real conversion (model download is proxy-blocked). A fake runner stands in
for ``mlx_lm.convert``; tests assert validation and that the config is threaded through.
"""

from __future__ import annotations

import pytest

from hearth.convert import ConvertConfig, ConvertOutcome, convert


def test_convert_delegates_to_runner(tmp_path):
    seen = {}

    def fake_runner(config: ConvertConfig):
        seen["source"] = config.source
        seen["bits"] = config.q_bits
        config.output_dir.mkdir(parents=True, exist_ok=True)
        (config.output_dir / "config.json").write_text("{}")
        return config.output_dir

    out = tmp_path / "converted"
    outcome = convert(
        ConvertConfig(source="org/model", output_dir=out, q_bits=4), runner=fake_runner
    )
    assert isinstance(outcome, ConvertOutcome)
    assert outcome.output_dir == out
    assert outcome.quantized is True
    assert outcome.q_bits == 4
    assert seen == {"source": "org/model", "bits": 4}
    assert (out / "config.json").exists()


def test_convert_no_quantize_reports_none_bits(tmp_path):
    outcome = convert(
        ConvertConfig(source="org/model", output_dir=tmp_path / "o", quantize=False),
        runner=lambda c: c.output_dir,
    )
    assert outcome.quantized is False
    assert outcome.q_bits is None


def test_convert_rejects_bad_bits(tmp_path):
    with pytest.raises(ValueError):
        convert(
            ConvertConfig(source="x", output_dir=tmp_path / "o", q_bits=5),
            runner=lambda c: c.output_dir,
        )


def test_convert_rejects_empty_source(tmp_path):
    with pytest.raises(ValueError):
        convert(ConvertConfig(source="", output_dir=tmp_path / "o"), runner=lambda c: c.output_dir)
