"""Dataset builder tests — deterministic JSONL build, round-trip, and validation."""

from __future__ import annotations

import pytest

from hearth.training.dataset import (
    Dataset,
    DatasetError,
    DatasetRecord,
    build_dataset,
    load_dataset,
    write_dataset,
)


def test_build_instruction_dataset_is_deterministic():
    pairs = [("extract the emails", "a@b.com"), ("extract the dates", "2026-07-08")]
    ds1 = build_dataset("extract", pairs, created_at="2026-07-08T00:00:00Z")
    ds2 = build_dataset("extract", pairs, created_at="2026-07-08T00:00:00Z")
    assert len(ds1) == 2
    # Same inputs + same timestamp => byte-identical JSONL (no clock reads).
    assert ds1.to_jsonl() == ds2.to_jsonl()
    # Header carries provenance/count.
    assert ds1.header()["count"] == 2
    assert ds1.header()["task"] == "extract"


def test_build_with_metas_aligns_provenance():
    ds = build_dataset(
        "classify",
        [("a", "yes"), ("b", "no")],
        metas=[{"src": "f1"}, {"src": "f2"}],
    )
    assert ds.records[0].meta == {"src": "f1"}
    assert ds.records[1].meta == {"src": "f2"}


def test_build_rejects_misaligned_metas():
    with pytest.raises(DatasetError):
        build_dataset("classify", [("a", "yes")], metas=[{"src": "1"}, {"src": "2"}])


def test_round_trip_through_file(tmp_path):
    ds = build_dataset("extract", [("p1", "c1"), ("p2", "c2")], created_at="2026-01-01T00:00:00Z")
    path = write_dataset(ds, tmp_path / "data.jsonl")
    loaded = load_dataset(path)
    assert loaded.task == "extract"
    assert loaded.created_at == "2026-01-01T00:00:00Z"
    assert [(r.prompt, r.completion) for r in loaded.records] == [("p1", "c1"), ("p2", "c2")]


def test_chat_records_round_trip(tmp_path):
    rec = DatasetRecord(messages=[{"role": "user", "content": "hi"}])
    ds = Dataset(task="chat", records=[rec])
    path = write_dataset(ds, tmp_path / "chat.jsonl")
    loaded = load_dataset(path)
    assert loaded.records[0].messages == [{"role": "user", "content": "hi"}]


def test_validate_rejects_empty_and_mixed_records():
    with pytest.raises(DatasetError):
        Dataset(task="x", records=[]).validate()
    with pytest.raises(DatasetError):
        # Both chat and instruction populated => invalid.
        DatasetRecord(
            messages=[{"role": "u", "content": "c"}], prompt="p", completion="c"
        ).validate()
    with pytest.raises(DatasetError):
        DatasetRecord().validate()  # neither


def test_load_rejects_schema_version_mismatch(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"kind": "hearth.dataset.header", "schema_version": 999, "task": "x"}\n')
    with pytest.raises(DatasetError):
        load_dataset(path)
