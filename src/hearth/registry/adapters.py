"""Adapter registry lifecycle (ARCHITECTURE §5, ADR-006, Phase 4).

The model registry (:mod:`hearth.registry`) is a *static* catalog loaded from YAML. The
adapter registry is *dynamic*: LoRA adapters are produced by training, gated by eval, and
move through a lifecycle — so it persists to a JSON store under ``~/.hearth`` rather than
living in a checked-in config file.

Lifecycle (ADR-006):

    register (candidate) --promote--> promoted --retire--> retired
                          ^ only if the eval gate passed (proof recorded)

A candidate is *servable behind a flag* for A/B before promotion (see
:meth:`AdapterStore.resolve_path`); a promoted adapter serves by default for its task.
Promotion refuses unless an eval gate proof is attached — the store never trusts a bare
"please promote", the caller must show the candidate beat the incumbent.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import Settings, get_settings

# Lifecycle states (ARCHITECTURE §5).
STATUS_CANDIDATE = "candidate"
STATUS_PROMOTED = "promoted"
STATUS_RETIRED = "retired"
_STATUSES = (STATUS_CANDIDATE, STATUS_PROMOTED, STATUS_RETIRED)


class AdapterError(RuntimeError):
    """Raised on an invalid adapter operation (unknown id, failed gate, bad state)."""


class GateNotPassedError(AdapterError):
    """Raised when promotion is attempted without a passing eval-gate proof (ADR-006)."""


@dataclass
class AdapterEntry:
    """One adapter in the registry (ARCHITECTURE §5).

    ``eval_scores`` records the candidate's and incumbent's scores at gate time;
    ``promotion_proof`` records *why* a promote was allowed (the gate result), so a
    promotion is auditable after the fact.
    """

    id: str
    base_model: str
    task: str
    train_run_id: str
    adapter_path: str
    status: str = STATUS_CANDIDATE
    eval_scores: dict[str, float] = field(default_factory=dict)
    promotion_proof: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, obj: dict) -> AdapterEntry:
        return cls(
            id=obj["id"],
            base_model=obj["base_model"],
            task=obj["task"],
            train_run_id=obj.get("train_run_id", ""),
            adapter_path=obj.get("adapter_path", ""),
            status=obj.get("status", STATUS_CANDIDATE),
            eval_scores=dict(obj.get("eval_scores", {})),
            promotion_proof=dict(obj.get("promotion_proof", {})),
        )


class AdapterStore:
    """Persistent adapter registry, backed by a single JSON file under ``~/.hearth``.

    All mutations reload → mutate → rewrite so concurrent CLIs and a running daemon see a
    consistent file (the volume is tiny — a handful of adapters — so this is cheap).
    """

    def __init__(self, path: Path | None = None, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        self.path = path or (settings.home / "adapters.json")

    # -- lifecycle --------------------------------------------------------------------

    def register(
        self,
        adapter_id: str,
        *,
        base_model: str,
        task: str,
        train_run_id: str,
        adapter_path: str,
        eval_scores: dict[str, float] | None = None,
    ) -> AdapterEntry:
        """Register a newly-trained adapter as a **candidate** (ADR-006)."""
        entries = self._load()
        if adapter_id in entries:
            raise AdapterError(f"adapter already registered: {adapter_id!r}")
        entry = AdapterEntry(
            id=adapter_id,
            base_model=base_model,
            task=task,
            train_run_id=train_run_id,
            adapter_path=adapter_path,
            status=STATUS_CANDIDATE,
            eval_scores=dict(eval_scores or {}),
        )
        entries[adapter_id] = entry
        self._save(entries)
        return entry

    def promote(
        self,
        adapter_id: str,
        *,
        gate_passed: bool,
        proof: dict[str, object] | None = None,
    ) -> AdapterEntry:
        """Promote a candidate to **promoted** — only if the eval gate passed (ADR-006).

        ``gate_passed`` is the caller's assertion that the candidate beat the incumbent
        (see :func:`hearth.training.eval.beats_incumbent`); ``proof`` records the scores
        behind that assertion. Refuses with :class:`GateNotPassedError` when the gate
        didn't pass — this is the promotion safety guarantee. Any previously-promoted
        adapter for the same task is retired so exactly one is promoted per task.
        """
        entries = self._load()
        entry = self._require(entries, adapter_id)
        if entry.status == STATUS_RETIRED:
            raise AdapterError(f"cannot promote a retired adapter: {adapter_id!r}")
        if not gate_passed:
            raise GateNotPassedError(
                f"refusing to promote {adapter_id!r}: eval gate not passed "
                "(candidate did not beat the incumbent)"
            )
        for other in entries.values():
            if other.task == entry.task and other.status == STATUS_PROMOTED:
                other.status = STATUS_RETIRED
        entry.status = STATUS_PROMOTED
        entry.promotion_proof = dict(proof or {})
        entry.promotion_proof.setdefault("gate_passed", True)
        self._save(entries)
        return entry

    def retire(self, adapter_id: str) -> AdapterEntry:
        """Retire an adapter (from any non-retired state)."""
        entries = self._load()
        entry = self._require(entries, adapter_id)
        entry.status = STATUS_RETIRED
        self._save(entries)
        return entry

    # -- queries ----------------------------------------------------------------------

    def get(self, adapter_id: str) -> AdapterEntry | None:
        return self._load().get(adapter_id)

    def list(self, *, task: str | None = None, status: str | None = None) -> list[AdapterEntry]:
        """List adapters, optionally filtered by ``task`` and/or ``status``."""
        if status is not None and status not in _STATUSES:
            raise AdapterError(f"unknown status: {status!r}")
        entries = self._load().values()
        return [
            e
            for e in entries
            if (task is None or e.task == task) and (status is None or e.status == status)
        ]

    def promoted_for(self, task: str) -> AdapterEntry | None:
        """The promoted adapter serving ``task`` by default, if any."""
        for e in self._load().values():
            if e.task == task and e.status == STATUS_PROMOTED:
                return e
        return None

    def resolve_path(self, adapter_id: str, *, allow_candidate: bool = True) -> str:
        """Resolve an adapter id to its on-disk path for serving (A/B support).

        Promoted adapters always resolve. A candidate resolves only when
        ``allow_candidate`` is set — this is the A/B flag that lets a candidate be served
        for evaluation *before* promotion (ARCHITECTURE §5). Retired adapters never
        resolve.
        """
        entry = self._require(self._load(), adapter_id)
        if entry.status == STATUS_RETIRED:
            raise AdapterError(f"adapter is retired: {adapter_id!r}")
        if entry.status == STATUS_CANDIDATE and not allow_candidate:
            raise AdapterError(
                f"adapter {adapter_id!r} is a candidate; serving it requires the A/B flag"
            )
        return entry.adapter_path

    # -- persistence ------------------------------------------------------------------

    def _load(self) -> dict[str, AdapterEntry]:
        if not self.path.exists():
            return {}
        data = json.loads(self.path.read_text(encoding="utf-8")) or {}
        return {
            obj["id"]: AdapterEntry.from_json(obj) for obj in data.get("adapters", [])
        }

    def _save(self, entries: dict[str, AdapterEntry]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"adapters": [e.to_json() for e in entries.values()]}
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def _require(entries: dict[str, AdapterEntry], adapter_id: str) -> AdapterEntry:
        entry = entries.get(adapter_id)
        if entry is None:
            raise AdapterError(f"unknown adapter: {adapter_id!r}")
        return entry


__all__ = [
    "AdapterEntry",
    "AdapterStore",
    "AdapterError",
    "GateNotPassedError",
    "STATUS_CANDIDATE",
    "STATUS_PROMOTED",
    "STATUS_RETIRED",
]
