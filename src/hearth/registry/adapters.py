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

import hashlib
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


def adapter_weights_sha(adapter_path: str | Path) -> str:
    """SHA-256 over an adapter's on-disk weights; :class:`AdapterError` if there are none.

    The eval report records this at measurement time and promotion recomputes it (B-061):
    the thing promoted must be the bytes that were measured, not an id whose weights were
    swapped, retrained, or never existed. A registry entry's ``adapter_path`` is a
    configuration that *implies* weights; this asserts on the weights themselves.

    A directory (mlx_lm's ``adapters.safetensors`` + ``adapter_config.json``, plus any
    checkpoints) hashes every regular file under it in sorted relative-path order, each as
    ``path NUL size NUL bytes``, so a renamed, added, removed or edited file changes the
    digest. Dot-files (``.DS_Store``) are skipped: they are not weights. A single file
    hashes its bytes. A missing path, or a directory with no files, is an error — there is
    nothing to measure.
    """
    root = Path(adapter_path).expanduser() if str(adapter_path) else None
    if root is None or not root.exists():
        raise AdapterError(f"adapter weights not found: {str(adapter_path) or '<empty path>'!r}")
    digest = hashlib.sha256()
    if root.is_file():
        files = [(root.name, root)]
    else:
        files = sorted(
            (str(p.relative_to(root)), p)
            for p in root.rglob("*")
            if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts)
        )
    if not files:
        raise AdapterError(f"adapter weights not found: {str(adapter_path)!r} contains no files")
    for rel, path in files:
        digest.update(rel.encode("utf-8") + b"\x00" + str(path.stat().st_size).encode() + b"\x00")
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


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
        gate_passed: bool | None = None,
        proof: dict[str, object] | None = None,
        gate: object | None = None,
    ) -> AdapterEntry:
        """Promote a candidate to **promoted** — only if the eval gate passed (ADR-006).

        Two ways to show the gate passed, and the store records which one was used:

        * ``gate`` — a :class:`hearth.training.eval.GateResult`. The verdict and the whole
          statistical proof (test, p-value, interval, baselines, golden sha, config
          fingerprint) are taken from it. This is the verified path.
        * ``gate_passed`` + ``proof`` — the legacy caller assertion. Still honoured so
          existing callers keep working, but the stored proof is stamped
          ``gate: "unverified"`` so a weak gate is never indistinguishable from a strong one
          in the audit trail (LEARNING_plan §3.3).

        Refuses with :class:`GateNotPassedError` when the gate didn't pass — this is the
        promotion safety guarantee. Any previously-promoted adapter for the same task is
        retired so exactly one is promoted per task.
        """
        if gate is not None:
            if gate_passed is not None:
                raise AdapterError(
                    "pass either 'gate' (a GateResult) or 'gate_passed', not both"
                )
            gate_passed = bool(gate.passed)
            gate_proof = gate.as_proof() if hasattr(gate, "as_proof") else {}
            merged = dict(proof or {})
            merged.update(gate_proof)
            proof = merged
        if gate_passed is None:
            raise AdapterError("promote requires 'gate' (a GateResult) or 'gate_passed'")

        entries = self._load()
        entry = self._require(entries, adapter_id)
        if entry.status == STATUS_RETIRED:
            raise AdapterError(f"cannot promote a retired adapter: {adapter_id!r}")
        if not gate_passed:
            reason = ""
            if gate is not None:
                reason = f": {getattr(gate, 'reason', '')}"
            raise GateNotPassedError(
                f"refusing to promote {adapter_id!r}: eval gate not passed "
                f"(candidate did not beat the incumbent){reason}"
            )
        for other in entries.values():
            if other.task == entry.task and other.status == STATUS_PROMOTED:
                other.status = STATUS_RETIRED
        entry.status = STATUS_PROMOTED
        entry.promotion_proof = dict(proof or {})
        entry.promotion_proof.setdefault("gate", "unverified")
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
    "adapter_weights_sha",
]
