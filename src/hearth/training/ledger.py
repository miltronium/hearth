"""The measurement ledger: every `hearth eval` of an adapter, append-only and signed (B-079).

A pre-registration exists so that the bar is chosen BEFORE anyone sees whether it is
cleared. Requiring the prereg to predate *this* run did not deliver that: run ``hearth eval
X --golden G`` with no prereg, read PASS and the p-value, then write and commit a bar and
re-run — the second run is "after the prereg" and promotes. The bar was chosen after the
outcome was known, which is the exact failure pre-registration exists to prevent.

So every measurement is recorded here, before a single score is computed, and promotion
asks when the adapter was FIRST measured — on any golden set, with or without a prereg:

    <HEARTH_HOME>/measurements.jsonl      one signed JSON record per line

Each record carries the adapter id and weights digest, the golden set (content sha, repo,
and that repo's HEAD at the time), metric, decode fingerprint, backend and time, plus
``seq`` and ``prev`` (the previous record's MAC), and is HMAC-signed with the same
per-install key as eval reports (:mod:`hearth.training.attest`). Editing, reordering or
deleting a record in the middle breaks a signature or the chain, and a broken ledger
refuses every promotion (fail closed). Appends take an exclusive ``flock`` so two
concurrent evals cannot fork the chain.

What it does NOT stop (residual trust, documented in docs/BUGS.md B-079): a user who can
read the key can re-sign a rewritten ledger, and anyone who can write HEARTH_HOME can delete
the file or truncate its tail. Both are the same trust level as editing ``adapters.json``
directly; the ledger removes the *casual* path — measure, look, then register a bar.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
from collections.abc import Iterator
from pathlib import Path

from . import attest

LEDGER_FILENAME = "measurements.jsonl"
LEDGER_SCHEMA = "hearth.measurement/1"


class LedgerError(ValueError):
    """The ledger cannot be written, or what is on disk is not an intact signed chain."""


def ledger_path(home: Path) -> Path:
    return Path(home) / LEDGER_FILENAME


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(path.name + ".lock"), "a", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _mac(record: dict) -> str:
    return str((record.get(attest.SIGNATURE_FIELD) or {}).get("mac") or "")


def _read(path: Path, key: bytes) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    prev = ""
    # Records are separated by "\n" and by nothing else (B-124). str.splitlines() also
    # breaks on U+2028/U+2029/U+0085 (and \r, \v, \f, \x1c-\x1e): one such character in
    # an adapter id tore its record in two and bricked every later measurement and
    # promotion. Records are written ASCII-only, so no separator can appear raw in one; a
    # ledger written before that is still read correctly.
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LedgerError(f"{path}: not UTF-8, so not an intact ledger ({exc})") from None
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()  # the newline that terminates the last record
    for lineno, line in enumerate(lines, start=1):
        try:  # a blank or torn line is not JSON, and so not an intact record
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("not a JSON object")
            attest.verify(record, key)
        except (ValueError, attest.AttestationError) as exc:
            raise LedgerError(f"{path}:{lineno}: not an intact signed record ({exc})") from None
        if record.get("schema") != LEDGER_SCHEMA:
            raise LedgerError(f"{path}:{lineno}: schema {record.get('schema')!r}")
        if record.get("seq") != len(records) or record.get("prev") != prev:
            raise LedgerError(
                f"{path}:{lineno}: the chain is broken (seq {record.get('seq')!r}, expected "
                f"{len(records)}): a record was removed, reordered or inserted"
            )
        records.append(record)
        prev = _mac(record)
    return records


def read(home: Path, key: bytes) -> list[dict]:
    """Every record, verified: signatures, schema, ``seq`` and the ``prev`` chain."""
    return _read(ledger_path(home), key)


def append(home: Path, entry: dict, key: bytes) -> dict:
    """Append ``entry`` as the next signed record and return it (with ``seq`` / signature).

    Raises :class:`LedgerError` when the existing ledger does not verify — a measurement
    is never added on top of a chain that is already broken — or cannot be written.
    """
    path = ledger_path(home)
    with _locked(path):
        records = _read(path, key)
        body = {**entry, "schema": LEDGER_SCHEMA, "seq": len(records),
                "prev": _mac(records[-1]) if records else ""}
        signed = attest.sign(body, key)
        # ensure_ascii: every non-ASCII character is a \u escape, so no line separator
        # (U+2028, U+2029, U+0085) can appear raw inside a record (B-124).
        line = json.dumps(signed, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        try:
            fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        except OSError as exc:
            raise LedgerError(f"cannot append to {path}: {exc}") from None
    return signed


def find(records: list[dict], mac: object) -> dict | None:
    """The record whose signature MAC is ``mac``, if any."""
    if not isinstance(mac, str) or not mac:
        return None
    return next((r for r in records if _mac(r) == mac), None)


def first_measurement(
    records: list[dict], *, adapter_id: str, weights_sha: str, served_sha: str = ""
) -> dict | None:
    """The earliest record of this adapter — matched by id, weights digest OR served digest.

    Matching the weights too means registering the same weights under a fresh id does not
    produce a fresh, "never measured" adapter. ``weights_sha`` hashes every file in the
    adapter directory, names included, so the same weights plus a README were "new" (B-121);
    ``served_sha`` (:func:`hearth.registry.adapters.adapter_served_sha`) covers only what
    mlx_lm loads, so a junk file, a renamed checkpoint or reformatted config metadata does
    not reset the first measurement.
    """
    return next(
        (r for r in records
         if r.get("adapter_id") == adapter_id
         or (weights_sha and r.get("weights_sha") == weights_sha)
         or (served_sha and r.get("served_sha") == served_sha)),
        None,
    )


def binding_problems(record: dict | None, expected: dict) -> list[str]:
    """Every field of ``expected`` that ``record`` (this measurement's entry) disagrees with."""
    if record is None:
        return ["this measurement is not in the measurement ledger "
                f"({LEDGER_FILENAME}): re-run `hearth eval` on this install"]
    return [
        f"ledger record {record.get('seq')} says {k}={record.get(k)!r}, the report {v!r}"
        for k, v in expected.items() if record.get(k) != v
    ]


__all__ = [
    "LEDGER_FILENAME",
    "LEDGER_SCHEMA",
    "LedgerError",
    "append",
    "binding_problems",
    "find",
    "first_measurement",
    "ledger_path",
    "read",
]
