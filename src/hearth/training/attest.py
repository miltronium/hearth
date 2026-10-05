"""Eval-report attestation: a report `adapters promote` accepts must come from `hearth eval`.

`hearth adapters promote --report` used to accept any JSON with the right keys (B-061): a
report a human typed promoted an adapter that had no weights, was never run, and whose task
did not match. Recomputing the gate from the report's per-example vectors does not help when
the vectors are typed too — the check and the checked thing were different objects
(CLAUDE.md §3).

So `hearth eval --report-json` signs the canonical report with HMAC-SHA256 under a
per-install secret, and `adapters promote` verifies it before reading a single number:

    <HEARTH_HOME>/eval-report.key     32 random bytes (hex), created 0600 with O_EXCL

What this does and does not prove. A valid MAC proves the report was written by `hearth
eval` running as someone who can read that key, on this install, and has not been edited
since. It does not stop a user who deliberately reads the key and computes an HMAC — that
user can also edit ``adapters.json`` directly, so no file-based check could. What it removes
is every *casual* path: a hand-written, edited, copied-from-another-machine or
produced-by-some-other-tool report is refused. A key whose mode lets group/other read it is
refused too: a secret anyone on the machine can read authenticates nothing.

The signature is over :func:`canonical` (sorted keys, no whitespace), so it is stable across
re-serialisation and does not depend on the pretty-printing the file is written with.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import stat
from pathlib import Path

KEY_FILENAME = "eval-report.key"
SIGNATURE_FIELD = "signature"
ALGORITHM = "hmac-sha256"


class AttestationError(ValueError):
    """A report is unsigned, mis-signed, or the install has no usable signing key."""


def key_path(home: Path) -> Path:
    return Path(home) / KEY_FILENAME


def load_key(home: Path, *, create: bool = False) -> bytes:
    """Return this install's report-signing key, creating it 0600 when ``create`` is set.

    Creation uses ``O_CREAT | O_EXCL`` with mode 0600, so the secret is never on disk with
    looser permissions (write-then-chmod has a window) and two concurrent evals cannot both
    write one. Verification never creates a key: no key means no report was ever signed here.
    """
    path = key_path(home)
    if not path.exists():
        if not create:
            raise AttestationError(
                f"no report-signing key at {path}: this install has never written an eval "
                "report, so no report can be verified here — run `hearth eval ... "
                "--report-json` on this machine"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass  # a concurrent eval created it first; read theirs below
        else:
            with os.fdopen(fd, "w", encoding="ascii") as fh:
                fh.write(secrets.token_hex(32) + "\n")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise AttestationError(
            f"report-signing key {path} is readable by group/other (mode {mode:o}); a secret "
            "anyone on this machine can read authenticates nothing. `chmod 600` it."
        )
    key = path.read_text(encoding="ascii").strip()
    if len(key) < 64:
        raise AttestationError(f"report-signing key {path} is truncated or empty")
    return key.encode("ascii")


def canonical(payload: dict) -> bytes:
    """The bytes that are signed: ``payload`` minus its signature, sorted, compact JSON."""
    body = {k: v for k, v in payload.items() if k != SIGNATURE_FIELD}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def key_id(key: bytes) -> str:
    """A short public identifier of the key (a hash of it), so a mismatch is diagnosable."""
    return hashlib.sha256(b"hearth-eval-report-key\x00" + key).hexdigest()[:16]


def sign(payload: dict, key: bytes) -> dict:
    """Return a copy of ``payload`` with an HMAC-SHA256 signature block attached."""
    mac = hmac.new(key, canonical(payload), hashlib.sha256).hexdigest()
    signed = dict(payload)
    signed[SIGNATURE_FIELD] = {"alg": ALGORITHM, "key_id": key_id(key), "mac": mac}
    return signed


def verify(payload: dict, key: bytes) -> str:
    """Raise :class:`AttestationError` unless ``payload`` carries a valid signature.

    Returns the SHA-256 of the canonical report — the identity a promotion proof records,
    so an auditor holding the report file can confirm which report licensed the promotion.
    """
    block = payload.get(SIGNATURE_FIELD)
    if not isinstance(block, dict) or not isinstance(block.get("mac"), str):
        raise AttestationError(
            "report is not signed: only a report written by `hearth eval --report-json` on "
            "this install can license a promotion — a hand-written report is not evidence"
        )
    # No separate check of block["alg"]: the signature block is outside the MAC, so the
    # label is not evidence of anything — the MAC below is HMAC-SHA256 whatever it says.
    expected = hmac.new(key, canonical(payload), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, block["mac"]):
        if block.get("key_id") != key_id(key):
            raise AttestationError(
                "report was signed by a different install's key (key_id "
                f"{block.get('key_id')!r}, this install {key_id(key)!r})"
            )
        raise AttestationError(
            "report signature does not match its contents: the report was edited after "
            "`hearth eval` wrote it"
        )
    return hashlib.sha256(canonical(payload)).hexdigest()


__all__ = [
    "ALGORITHM",
    "AttestationError",
    "KEY_FILENAME",
    "SIGNATURE_FIELD",
    "canonical",
    "key_id",
    "key_path",
    "load_key",
    "sign",
    "verify",
]
