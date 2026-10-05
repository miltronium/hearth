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

import contextlib
import fcntl
import hashlib
import json
import os
import struct
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import Settings, get_settings

# Lifecycle states (ARCHITECTURE §5).
STATUS_CANDIDATE = "candidate"
STATUS_PROMOTED = "promoted"
STATUS_RETIRED = "retired"
_STATUSES = (STATUS_CANDIDATE, STATUS_PROMOTED, STATUS_RETIRED)

# "The caller did not say which incumbent it beat" — distinct from None ("the base model").
_UNCHECKED = object()


class AdapterError(RuntimeError):
    """Raised on an invalid adapter operation (unknown id, failed gate, bad state)."""


class GateNotPassedError(AdapterError):
    """Raised when promotion is attempted without a passing eval-gate proof (ADR-006)."""


class IncumbentChangedError(AdapterError):
    """The incumbent a promotion was measured against is no longer the incumbent (B-082)."""


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


# The files serving actually loads from an adapter directory. mlx_lm (0.29.1,
# ``mlx_lm/tuner/utils.py:load_adapters``) opens exactly two, by fixed name:
# ``adapter_config.json`` — of which it reads ``fine_tune_type`` (default "lora"),
# ``num_layers`` and ``lora_parameters`` — and ``adapters.safetensors``, via
# ``model.load_weights(..., strict=False)``. Nothing else in the directory (checkpoints,
# README, training logs) reaches the model.
SERVED_ADAPTER_FILES = ("adapter_config.json", "adapters.safetensors")
_SERVED_CONFIG_KEYS = ("fine_tune_type", "num_layers", "lora_parameters")


def _served_config_bytes(raw: bytes) -> bytes:
    """``adapter_config.json`` reduced to what mlx_lm reads, canonically serialised."""
    try:
        obj = json.loads(raw)
    except ValueError:
        return b"raw\x00" + raw
    if not isinstance(obj, dict):
        return b"raw\x00" + raw
    used = {k: obj.get(k) for k in _SERVED_CONFIG_KEYS}
    used["fine_tune_type"] = used["fine_tune_type"] or "lora"
    return b"cfg\x00" + json.dumps(used, sort_keys=True, separators=(",", ":")).encode()


def _served_tensor_digest(raw: bytes) -> bytes:
    """The tensors of a safetensors file — name, dtype, shape, bytes — sorted by name.

    The ``__metadata__`` block, the header's key order and its padding do not reach the
    model, so they do not change the identity. A file that does not parse as safetensors is
    hashed as raw bytes (mlx_lm could not load it; it is still a fixed identity).
    """
    digest = hashlib.sha256()
    try:
        (size,) = struct.unpack("<Q", raw[:8])
        header = json.loads(raw[8:8 + size])
        if not isinstance(header, dict):
            raise ValueError("header is not an object")
        data = memoryview(raw)[8 + size:]
        for name in sorted(k for k in header if k != "__metadata__"):
            info = header[name]
            begin, end = info["data_offsets"]
            digest.update(json.dumps([name, info["dtype"], info["shape"]]).encode() + b"\x00")
            digest.update(data[begin:end])
        return b"st\x00" + digest.digest()
    except (struct.error, ValueError, KeyError, TypeError):
        return b"raw\x00" + hashlib.sha256(raw).digest()


def adapter_served_sha(adapter_path: str | Path) -> str:
    """SHA-256 of what SERVING loads from an adapter — "" when it would load nothing (B-121).

    :func:`adapter_weights_sha` hashes every file and every file NAME, which is right for
    "are these still the bytes that were measured" but wrong for "is this the same adapter":
    the same weights plus a README under a new id hashed differently, so they were a "fresh,
    never-measured" adapter and the first-measurement rule (B-079) did not apply. This
    digest covers only :data:`SERVED_ADAPTER_FILES`, by role (never by what else is in the
    directory): the config keys mlx_lm reads, canonically serialised, and the safetensors
    tensors (name, dtype, shape, bytes), ignoring metadata and header layout. Each file
    present is labelled by its role, so a missing config differs from any present one. A
    single-file adapter hashes its bytes.

    What it cannot establish (residual, docs/BUGS.md B-121): behavioural identity. A weight
    nudged by one ulp, or an extra tensor that ``load_weights(strict=False)`` ignores, is a
    different digest for an effectively identical adapter.
    """
    root = Path(adapter_path).expanduser() if str(adapter_path) else None
    if root is None or not root.exists():
        return ""
    digest = hashlib.sha256(b"hearth.served-adapter/1\x00")
    if root.is_file():
        digest.update(b"file\x00" + root.read_bytes())
        return digest.hexdigest()
    found = False
    for name in SERVED_ADAPTER_FILES:
        path = root / name
        if not path.is_file():  # absent contributes nothing; each present file is labelled
            continue
        found = True
        raw = path.read_bytes()
        part = _served_config_bytes(raw) if name.endswith(".json") else _served_tensor_digest(raw)
        digest.update(name.encode() + b"\x00" + str(len(part)).encode() + b"\x00" + part)
    return digest.hexdigest() if found else ""


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

    Every mutation is reload → mutate → rewrite **under an exclusive ``fcntl.flock``** on a
    sibling ``.lock`` file, and the rewrite is atomic (temp file in the same directory,
    fsync, ``os.replace``), so concurrent CLIs and a running daemon never lose an update
    and a reader never sees a half-written file (B-082). Reads take no lock: with atomic
    replacement a read sees either the old file or the new one.
    """

    def __init__(self, path: Path | None = None, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        self.path = path or (settings.home / "adapters.json")

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.path.name + ".lock")
        with open(lock_path, "a", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

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
        with self._locked():
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
        expected_incumbent: object = _UNCHECKED,
        precondition: Callable[[], list[str]] | None = None,
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

        ``expected_incumbent`` is the id of the promoted adapter the candidate was measured
        against, or ``None`` when it beat the base model; ``precondition`` returns every
        reason the evidence no longer holds (e.g. weights re-hashed). Both are evaluated
        **under the registry lock, against the file as it is at write time** (B-082): the
        check and the write used to be separate steps, so an adapter promoted in between
        was silently retired by a report that had only ever beaten the base model.
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

        with self._locked():
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
            if expected_incumbent is not _UNCHECKED:
                current = next(
                    (e.id for e in entries.values() if e.task == entry.task
                     and e.status == STATUS_PROMOTED and e.id != adapter_id),
                    None,
                )
                if current != expected_incumbent:
                    beaten = expected_incumbent or "the base model"
                    raise IncumbentChangedError(
                        f"refusing to promote {adapter_id!r}: it was measured against "
                        f"{beaten!r}, but the promoted adapter for {entry.task!r} is now "
                        f"{current or 'none (the base model)'!r} — re-run `hearth eval`"
                    )
            if precondition is not None:
                problems = precondition()
                if problems:
                    raise IncumbentChangedError(
                        f"refusing to promote {adapter_id!r}: the evidence changed while "
                        "promoting — " + "; ".join(problems)
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
        with self._locked():
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
        """Atomically replace the registry file (caller holds :meth:`_locked`)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"adapters": [e.to_json() for e in entries.values()]}
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=f".{self.path.name}.",
                                   suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

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
    "IncumbentChangedError",
    "STATUS_CANDIDATE",
    "STATUS_PROMOTED",
    "SERVED_ADAPTER_FILES",
    "STATUS_RETIRED",
    "adapter_served_sha",
    "adapter_weights_sha",
]
