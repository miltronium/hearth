"""LoRA training orchestrator tests — a FAKE runner; never launches a real run."""

from __future__ import annotations

import json

import pytest

from hearth.training.dataset import DatasetError, build_dataset
from hearth.training.lora import LoRAConfig, train


def _dataset(n=4):
    pairs = [(f"prompt {i}", f"completion {i}") for i in range(n)]
    return build_dataset("extract", pairs, created_at="2026-07-08T00:00:00Z")


def test_train_delegates_to_runner_and_prepares_splits(tmp_path):
    seen = {}

    def fake_runner(args, run_dir):
        # The runner sees the assembled invocation and the run dir; write a dummy adapter.
        seen["args"] = args
        adapter = run_dir / "adapters"
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "adapters.safetensors").write_text("weights")
        return adapter

    config = LoRAConfig(
        base_model="org/base", task="extract", dataset=_dataset(4), output_dir=tmp_path / "run"
    )
    outcome = train(config, runner=fake_runner, train_run_id="run-1")

    assert outcome.train_run_id == "run-1"
    assert outcome.base_model == "org/base"
    assert outcome.num_records == 4
    assert outcome.adapter_path.exists()
    # mlx-lm invocation was assembled (not executed by us).
    assert "--train" in seen["args"]
    assert "org/base" in seen["args"]
    # Train/valid splits were written for mlx-lm to consume.
    data_dir = tmp_path / "run" / "data"
    train_lines = (data_dir / "train.jsonl").read_text().splitlines()
    valid_lines = (data_dir / "valid.jsonl").read_text().splitlines()
    assert len(train_lines) >= 1 and len(valid_lines) >= 1
    assert len(train_lines) + len(valid_lines) == 4
    # Records are valid JSON in mlx-lm's instruction shape.
    assert "prompt" in json.loads(train_lines[0])


def test_train_validates_inputs(tmp_path):
    with pytest.raises(ValueError):
        train(
            LoRAConfig(base_model="", task="extract", dataset=_dataset(), output_dir=tmp_path),
            runner=lambda a, d: d,
        )
    # Fewer than 2 records cannot be split into train/valid.
    with pytest.raises(DatasetError):
        train(
            LoRAConfig(
                base_model="b", task="extract", dataset=_dataset(1), output_dir=tmp_path
            ),
            runner=lambda a, d: d,
        )
