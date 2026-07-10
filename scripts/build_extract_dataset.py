#!/usr/bin/env python
"""Build the real `extract` training dataset + golden set for the LoRA runbook.

Task class: extract a JIRA-style ticket id (PROJ-1234) from a one-line message.
Deterministic (caller-supplied timestamp) so the artifact is reproducible. Writes:
  data/extract.jsonl        — training set (instruction pairs)
  data/extract_golden.jsonl — held-out golden eval set (kept disjoint from training)

Size note: a real `mlx_lm.lora` run splits the set into train/valid (valid_fraction=0.1)
and REQUIRES the validation split to hold at least `batch_size` (default 4) examples —
so the dataset needs ~40+ records, not the bare 2 that LoRAConfig.validate() permits.
We generate a comfortable margin below.
"""

from __future__ import annotations

import json
from pathlib import Path

from hearth.training.dataset import build_dataset, write_dataset
from hearth.training.eval import as_golden_set

CREATED_AT = "2026-07-09T00:00:00Z"

# Deterministic generator: (project, number) → a varied one-line message. The templates
# and project pool are fixed, so the artifact is byte-reproducible. Golden ids are held
# out below so training and eval stay strictly disjoint.
_TEMPLATES = [
    "Fixed in {tid}, see PR #{n}",
    "Closes {tid} after review",
    "Blocked by {tid} until infra lands",
    "See {tid} for the migration plan",
    "Reverted {tid} due to a regression",
    "{tid} shipped in today's release",
    "Follow-up tracked in {tid}",
    "Root cause documented in {tid}",
    "Merged after CI green: {tid}",
    "Duplicate of {tid}, closing",
    "Hotfix for {tid} is live",
    "Design signed off in {tid}",
]
_PROJECTS = ["PROJ", "CORE", "DATA", "HEARTH", "WEB", "API", "OPS", "PLAT", "AUTH", "BILL"]

# Ids reserved for the golden set — never emitted into the training set.
_GOLDEN_IDS = {"PROJ-4242", "CORE-777", "HEARTH-31", "API-500", "OPS-13", "DATA-2024"}


def _gen_train_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    n = 100
    for i in range(60):  # generate a pool, filter golden collisions, keep the first 48
        proj = _PROJECTS[i % len(_PROJECTS)]
        num = 10 + (i * 37) % 990  # spread ids deterministically
        tid = f"{proj}-{num}"
        if tid in _GOLDEN_IDS:
            continue
        template = _TEMPLATES[i % len(_TEMPLATES)]
        message = template.format(tid=tid, n=n + i)
        pairs.append((f"Extract the ticket id from: '{message}'.", tid))
        if len(pairs) == 48:
            break
    return pairs


TRAIN_PAIRS = _gen_train_pairs()

# Golden set — disjoint messages/ids used only to score the candidate.
GOLDEN_PAIRS = [
    ("Extract the ticket id from: 'See PROJ-4242 for context'.", "PROJ-4242"),
    ("Extract the ticket id from: 'Resolved in CORE-777 last sprint'.", "CORE-777"),
    ("Extract the ticket id from: 'Tracking regression in HEARTH-31'.", "HEARTH-31"),
    ("Extract the ticket id from: 'Superseded by API-500'.", "API-500"),
    ("Extract the ticket id from: 'Deploy gated on OPS-13'.", "OPS-13"),
    ("Extract the ticket id from: 'Reopened as DATA-2024'.", "DATA-2024"),
]


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"

    ds = build_dataset(
        task="extract",
        pairs=TRAIN_PAIRS,
        created_at=CREATED_AT,
        provenance={"source": "scripts/build_extract_dataset.py", "hand_curated": "true"},
    )
    train_path = write_dataset(ds, data_dir / "extract.jsonl")

    # Golden set: write as a plain (prompt, expected) JSONL for the eval step to reload.
    golden = as_golden_set("extract", GOLDEN_PAIRS)
    golden_path = data_dir / "extract_golden.jsonl"
    golden_path.write_text(
        "".join(
            json.dumps({"prompt": ex.prompt, "expected": ex.expected}, sort_keys=True) + "\n"
            for ex in golden.examples
        ),
        encoding="utf-8",
    )

    print(f"wrote {train_path} ({len(ds)} training records)")
    print(f"wrote {golden_path} ({len(golden)} golden examples)")


if __name__ == "__main__":
    main()
