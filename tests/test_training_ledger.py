"""The measurement ledger (B-079): append-only, signed, chained — and fail-closed when not."""

from __future__ import annotations

import json

import pytest

from hearth.training import attest
from hearth.training.ledger import (
    LedgerError,
    append,
    binding_problems,
    find,
    first_measurement,
    ledger_path,
    read,
)


def _key(tmp_path):
    return attest.load_key(tmp_path, create=True)


def _entry(adapter="a", weights="w1", at="2026-01-01T00:00:00+00:00"):
    return {"adapter_id": adapter, "weights_sha": weights, "measured_at": at, "golden_sha": "g"}


def test_records_are_signed_sequenced_and_chained(tmp_path):
    key = _key(tmp_path)
    r0 = append(tmp_path, _entry(), key)
    r1 = append(tmp_path, _entry(adapter="b", weights="w2"), key)
    assert (r0["seq"], r0["prev"]) == (0, "")
    assert (r1["seq"], r1["prev"]) == (1, r0["signature"]["mac"])
    assert read(tmp_path, key) == [r0, r1]
    assert find(read(tmp_path, key), r1["signature"]["mac"]) == r1
    assert oct(ledger_path(tmp_path).stat().st_mode & 0o777) == "0o600"


def test_first_measurement_matches_by_id_or_by_weights(tmp_path):
    key = _key(tmp_path)
    r0 = append(tmp_path, _entry(adapter="a", weights="w1"), key)
    append(tmp_path, _entry(adapter="b", weights="w2"), key)
    records = read(tmp_path, key)
    assert first_measurement(records, adapter_id="a", weights_sha="zz") == r0
    assert first_measurement(records, adapter_id="renamed", weights_sha="w1") == r0
    assert first_measurement(records, adapter_id="c", weights_sha="w3") is None
    assert first_measurement(records, adapter_id="c", weights_sha="") is None


@pytest.mark.parametrize(
    "tamper",
    [
        lambda lines: lines[1:],  # delete the first record
        lambda lines: [lines[1], lines[0], lines[2]],  # reorder
        lambda lines: [lines[0], lines[2]],  # delete from the middle
        lambda lines: [lines[0].replace('"adapter_id":"a"', '"adapter_id":"x"'), *lines[1:]],
        lambda lines: [*lines, ""],  # a blank line
        lambda lines: [*lines, "{not json"],
    ],
)
def test_any_edit_breaks_the_ledger(tmp_path, tamper):
    key = _key(tmp_path)
    for i in range(3):
        append(tmp_path, _entry(adapter="a" if i == 0 else f"n{i}"), key)
    path = ledger_path(tmp_path)
    path.write_text("\n".join(tamper(path.read_text().splitlines())) + "\n")
    with pytest.raises(LedgerError):
        read(tmp_path, key)
    with pytest.raises(LedgerError):
        append(tmp_path, _entry(), key)  # never extend a broken chain


def test_a_record_resigned_out_of_sequence_is_refused(tmp_path):
    """Even with the key, a record must carry the right seq and prev."""
    key = _key(tmp_path)
    append(tmp_path, _entry(), key)
    forged = attest.sign({**_entry(adapter="b"), "schema": "hearth.measurement/1", "seq": 5,
                          "prev": ""}, key)
    with ledger_path(tmp_path).open("a") as fh:
        fh.write(json.dumps(forged) + "\n")
    with pytest.raises(LedgerError, match="chain is broken"):
        read(tmp_path, key)


def test_a_record_of_another_schema_is_refused(tmp_path):
    key = _key(tmp_path)
    forged = attest.sign({**_entry(), "schema": "other/1", "seq": 0, "prev": ""}, key)
    ledger_path(tmp_path).write_text(json.dumps(forged) + "\n")
    with pytest.raises(LedgerError, match="schema"):
        read(tmp_path, key)


def test_another_installs_key_cannot_read_this_ledger(tmp_path):
    append(tmp_path, _entry(), _key(tmp_path))
    with pytest.raises(LedgerError):
        read(tmp_path, attest.load_key(tmp_path / "other", create=True))


def test_binding_problems_name_each_disagreement(tmp_path):
    key = _key(tmp_path)
    record = append(tmp_path, _entry(), key)
    assert binding_problems(record, {"adapter_id": "a", "golden_sha": "g"}) == []
    assert binding_problems(record, {"golden_sha": "h"})
    assert "not in the measurement ledger" in binding_problems(None, {})[0]
