"""Training subsystem (ARCHITECTURE §7, ADR-006, Phase 4).

Parameter-efficient fine-tuning, local, gated by eval — explicitly *not* a general
training platform. The pipeline (ARCHITECTURE §7):

  curate dataset -> format (chat/instruction) -> LoRA/QLoRA train (mlx_lm.lora)
    -> eval vs incumbent on a golden set -> gate -> register adapter (candidate)
    -> human promote -> serve

Layout:
  * :mod:`hearth.training.dataset` — build/validate versioned JSONL with provenance.
  * :mod:`hearth.training.lora`    — thin, testable orchestrator around ``mlx_lm.lora``.
  * :mod:`hearth.training.eval`    — golden-set scoring + the promotion gate.

Everything here is testable with no extras and no real training: the heavy ``mlx``
imports are deferred behind the ``[mlx]`` extra and the trainer takes an injectable
runner (stubbed in tests). The adapter *registry* lifecycle lives in
:mod:`hearth.registry.adapters`.
"""

from __future__ import annotations

from .dataset import (
    Dataset,
    DatasetRecord,
    build_dataset,
    load_dataset,
    write_dataset,
)
from .eval import (
    EvalReport,
    GoldenExample,
    GoldenSet,
    beats_incumbent,
    exact_match_score,
    score_candidate,
    token_f1_score,
)
from .lora import LoRAConfig, TrainOutcome, train

__all__ = [
    "Dataset",
    "DatasetRecord",
    "build_dataset",
    "load_dataset",
    "write_dataset",
    "LoRAConfig",
    "TrainOutcome",
    "train",
    "GoldenExample",
    "GoldenSet",
    "EvalReport",
    "exact_match_score",
    "token_f1_score",
    "score_candidate",
    "beats_incumbent",
]
