"""Dataset builder — versioned JSONL with provenance (ARCHITECTURE §7, Phase 4).

Curates instruction / chat-format training pairs into a single, versioned artifact:
a small JSON *header* record (schema version, task, provenance, timestamps, count)
followed by one JSON object per training example. The whole thing is deterministic —
timestamps are passed in by the caller, never generated here — so a build from the same
inputs yields byte-identical output and is trivially testable with in-memory data.

Two shapes are supported per record (both map cleanly onto ``mlx_lm.lora`` inputs):

  * **chat** — ``{"messages": [{"role": ..., "content": ...}, ...]}``
  * **instruction** — ``{"prompt": ..., "completion": ...}``

The default builder (:func:`build_dataset`) assembles instruction pairs from
``(prompt, completion)`` records; :func:`load_dataset` reads and validates a JSONL file
back into a :class:`Dataset`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

# Bump when the on-disk record shape changes. The header carries this so a reader can
# reject a file it doesn't understand rather than silently mis-parse it.
SCHEMA_VERSION = 1

# Marks the first line of a dataset file as the header (vs. an example record).
_HEADER_KIND = "hearth.dataset.header"


class DatasetError(ValueError):
    """Raised when a dataset fails validation (bad shape, empty, version mismatch)."""


@dataclass(frozen=True)
class DatasetRecord:
    """One training example, in either chat or instruction shape.

    Exactly one of (``messages``) or (``prompt`` + ``completion``) must be populated.
    ``meta`` carries optional per-record provenance (e.g. source file, accepted-at).
    """

    messages: list[dict[str, str]] | None = None
    prompt: str | None = None
    completion: str | None = None
    meta: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        """Raise :class:`DatasetError` unless this record is a well-formed pair."""
        is_chat = self.messages is not None
        is_instruction = self.prompt is not None or self.completion is not None
        if is_chat == is_instruction:
            raise DatasetError(
                "record must be either chat (messages) or instruction (prompt+completion), "
                "not both or neither"
            )
        if is_chat:
            if not self.messages:
                raise DatasetError("chat record has empty messages")
            for m in self.messages:
                if not m.get("role") or "content" not in m:
                    raise DatasetError(f"chat message missing role/content: {m!r}")
        else:
            if not (self.prompt and self.completion):
                raise DatasetError("instruction record needs a non-empty prompt and completion")

    def to_json(self) -> dict:
        """Render to the on-disk JSON object (omitting empty optional fields)."""
        if self.messages is not None:
            obj: dict = {"messages": self.messages}
        else:
            obj = {"prompt": self.prompt, "completion": self.completion}
        if self.meta:
            obj["meta"] = self.meta
        return obj


@dataclass(frozen=True)
class Dataset:
    """A versioned, provenance-tagged collection of training records.

    Timestamps are supplied by the caller (``created_at``) so builds are deterministic
    and reproducible; nothing here reads the clock.
    """

    task: str
    records: list[DatasetRecord]
    version: str = "v1"
    created_at: str = ""
    provenance: dict[str, str] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __len__(self) -> int:
        return len(self.records)

    def validate(self) -> None:
        """Validate the header and every record; raise :class:`DatasetError` on any issue."""
        if not self.task:
            raise DatasetError("dataset task must be non-empty")
        if not self.records:
            raise DatasetError("dataset has no records")
        for i, rec in enumerate(self.records):
            try:
                rec.validate()
            except DatasetError as exc:
                raise DatasetError(f"record {i}: {exc}") from exc

    def header(self) -> dict:
        """The header object written as the file's first line."""
        return {
            "kind": _HEADER_KIND,
            "schema_version": self.schema_version,
            "task": self.task,
            "version": self.version,
            "created_at": self.created_at,
            "count": len(self.records),
            "provenance": self.provenance,
        }

    def to_jsonl(self) -> str:
        """Serialize to JSONL text: one header line, then one line per record.

        Deterministic — ``sort_keys=True`` and caller-supplied timestamps mean the same
        dataset always produces byte-identical output.
        """
        self.validate()
        lines = [json.dumps(self.header(), sort_keys=True)]
        lines += [json.dumps(r.to_json(), sort_keys=True) for r in self.records]
        return "\n".join(lines) + "\n"


def build_dataset(
    task: str,
    pairs: Iterable[tuple[str, str]],
    *,
    version: str = "v1",
    created_at: str = "",
    provenance: dict[str, str] | None = None,
    metas: Sequence[dict[str, str]] | None = None,
) -> Dataset:
    """Build an instruction-format :class:`Dataset` from ``(prompt, completion)`` pairs.

    ``metas`` optionally supplies per-record provenance aligned to ``pairs``. The result
    is validated before return so a caller can't build a malformed dataset.
    """
    pair_list = list(pairs)
    if metas is not None and len(metas) != len(pair_list):
        raise DatasetError("metas must align 1:1 with pairs")
    records = [
        DatasetRecord(
            prompt=prompt,
            completion=completion,
            meta=dict(metas[i]) if metas is not None else {},
        )
        for i, (prompt, completion) in enumerate(pair_list)
    ]
    ds = Dataset(
        task=task,
        records=records,
        version=version,
        created_at=created_at,
        provenance=dict(provenance or {}),
    )
    ds.validate()
    return ds


def write_dataset(ds: Dataset, path: Path | str) -> Path:
    """Validate and write ``ds`` to ``path`` as JSONL. Returns the written path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ds.to_jsonl(), encoding="utf-8")
    return path


def load_dataset(path: Path | str) -> Dataset:
    """Read and validate a dataset JSONL file back into a :class:`Dataset`.

    Accepts files with or without a header line (a headerless file is treated as a bare
    list of records with default header fields). Raises :class:`DatasetError` on a
    schema-version mismatch or any malformed record.
    """
    text = Path(path).read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise DatasetError("dataset file is empty")

    first = json.loads(lines[0])
    if isinstance(first, dict) and first.get("kind") == _HEADER_KIND:
        header, record_lines = first, lines[1:]
    else:
        header, record_lines = {}, lines

    schema_version = int(header.get("schema_version", SCHEMA_VERSION))
    if schema_version != SCHEMA_VERSION:
        raise DatasetError(
            f"unsupported dataset schema_version {schema_version} (expected {SCHEMA_VERSION})"
        )

    records = [_record_from_json(json.loads(ln)) for ln in record_lines]
    ds = Dataset(
        task=header.get("task", ""),
        records=records,
        version=header.get("version", "v1"),
        created_at=header.get("created_at", ""),
        provenance=dict(header.get("provenance", {})),
        schema_version=schema_version,
    )
    ds.validate()
    return ds


def _record_from_json(obj: dict) -> DatasetRecord:
    """Parse one on-disk record object into a :class:`DatasetRecord`."""
    return DatasetRecord(
        messages=obj.get("messages"),
        prompt=obj.get("prompt"),
        completion=obj.get("completion"),
        meta=dict(obj.get("meta", {})),
    )


__all__ = [
    "SCHEMA_VERSION",
    "Dataset",
    "DatasetRecord",
    "DatasetError",
    "build_dataset",
    "write_dataset",
    "load_dataset",
]
