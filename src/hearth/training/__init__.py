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
  * :mod:`hearth.training.stats`   — exact paired statistics behind the gate (stdlib only).
  * :mod:`hearth.training.prereg`  — pre-registration: the bar, declared and git-committed
    before the measurement.

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
    EvalConfig,
    EvalReport,
    GateProvenanceError,
    GateResult,
    GoldenExample,
    GoldenSet,
    baseline_reports,
    beats_incumbent,
    check_determinism,
    evaluate_gate,
    exact_match_score,
    score_candidate,
    token_f1_score,
)
from .lora import LoRAConfig, TrainOutcome, train
from .prereg import PreRegError, PreRegistration, load_prereg, require_prereg, verify_committed

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
    "EvalConfig",
    "EvalReport",
    "GateProvenanceError",
    "GateResult",
    "exact_match_score",
    "token_f1_score",
    "score_candidate",
    "baseline_reports",
    "check_determinism",
    "evaluate_gate",
    "beats_incumbent",
    "PreRegError",
    "PreRegistration",
    "load_prereg",
    "require_prereg",
    "verify_committed",
]
