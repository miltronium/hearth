"""Unit tests for eval-report attestation (B-061): sign, verify, and the key's lifecycle."""

from __future__ import annotations

import pytest

from hearth.registry.adapters import AdapterError, adapter_weights_sha
from hearth.training import attest

PAYLOAD = {"candidate_id": "a", "candidate": {"per_example": [1.0, 0.0], "score": 0.5}}


def test_a_signed_report_verifies_and_returns_its_canonical_sha(tmp_path):
    key = attest.load_key(tmp_path, create=True)
    signed = attest.sign(PAYLOAD, key)
    digest = attest.verify(signed, key)
    assert len(digest) == 64
    # Stable across re-serialisation: the canonical form ignores key order and whitespace.
    reordered = dict(reversed(list(signed.items())))
    assert attest.verify(reordered, key) == digest


def test_the_key_is_created_0600_and_reused(tmp_path):
    first = attest.load_key(tmp_path, create=True)
    assert oct(attest.key_path(tmp_path).stat().st_mode & 0o777) == "0o600"
    assert attest.load_key(tmp_path) == first


def test_verifying_never_creates_a_key(tmp_path):
    with pytest.raises(attest.AttestationError, match="no report-signing key"):
        attest.load_key(tmp_path)
    assert not attest.key_path(tmp_path).exists()


def test_a_loosened_or_truncated_key_is_refused(tmp_path):
    attest.load_key(tmp_path, create=True)
    path = attest.key_path(tmp_path)
    path.chmod(0o640)
    with pytest.raises(attest.AttestationError, match="group/other"):
        attest.load_key(tmp_path)
    path.chmod(0o600)
    path.write_text("short\n")
    with pytest.raises(attest.AttestationError, match="truncated"):
        attest.load_key(tmp_path)


@pytest.mark.parametrize(
    "signature",
    [None, "deadbeef", {"alg": "hmac-sha256"}, {"alg": "hmac-sha256", "mac": 123},
     {"alg": "none", "mac": "00"}],
)
def test_an_unsigned_or_malformed_signature_is_refused(tmp_path, signature):
    key = attest.load_key(tmp_path, create=True)
    payload = dict(PAYLOAD)
    if signature is not None:
        payload["signature"] = signature
    with pytest.raises(attest.AttestationError):
        attest.verify(payload, key)


def test_an_edited_report_is_refused(tmp_path):
    key = attest.load_key(tmp_path, create=True)
    signed = attest.sign(PAYLOAD, key)
    signed["candidate"] = {"per_example": [1.0, 1.0], "score": 1.0}
    with pytest.raises(attest.AttestationError, match="edited"):
        attest.verify(signed, key)


def test_another_installs_key_is_refused(tmp_path):
    signed = attest.sign(PAYLOAD, attest.load_key(tmp_path / "a", create=True))
    with pytest.raises(attest.AttestationError, match="different install"):
        attest.verify(signed, attest.load_key(tmp_path / "b", create=True))


def test_weights_sha_covers_every_file_and_refuses_nothing(tmp_path):
    d = tmp_path / "ad"
    d.mkdir()
    with pytest.raises(AdapterError, match="contains no files"):
        adapter_weights_sha(d)
    with pytest.raises(AdapterError, match="not found"):
        adapter_weights_sha(tmp_path / "missing")
    with pytest.raises(AdapterError, match="not found"):
        adapter_weights_sha("")
    (d / "adapters.safetensors").write_bytes(b"w")
    first = adapter_weights_sha(d)
    (d / ".DS_Store").write_bytes(b"finder noise")
    assert adapter_weights_sha(d) == first  # dot-files are not weights
    (d / "adapter_config.json").write_text("{}")
    second = adapter_weights_sha(d)
    assert second != first  # an added file changes the digest
    (d / "adapter_config.json").rename(d / "renamed.json")
    assert adapter_weights_sha(d) != second  # so does a rename
    assert adapter_weights_sha(d / "adapters.safetensors")  # a single file hashes too
