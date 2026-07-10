#!/usr/bin/env python
"""Build a `classify` (ticket-routing) dataset with real headroom for the LoRA runbook.

Why a second task: the base Qwen-Coder-7B already scores 1.0 on plain ticket-id
extraction, so there's no lift to demonstrate. Routing to an *arbitrary org convention*
(incident description -> one of our internal queue codes QX-1..QX-9) is something the base
model cannot guess from semantics — so a LoRA adapter that learns the mapping produces a
genuine, honest lift over base.

Chat format (messages) is used deliberately: mlx-lm applies the tokenizer chat template,
which ends the assistant turn with <|im_end|>, teaching the model to STOP. (Instruction
format `prompt`/`completion` does not, which makes short-answer adapters ramble — see
docs/RESULTS.md.)

Writes:
  data/route.jsonl        — training set (chat records)
  data/route_golden.jsonl — held-out golden eval set (new phrasings, same categories)
"""

from __future__ import annotations

import json
from pathlib import Path

from hearth.training.dataset import Dataset, DatasetRecord, write_dataset

CREATED_AT = "2026-07-09T00:00:00Z"
SYSTEM = "You are a ticket router. Reply with ONLY the queue code (e.g. QX-7), nothing else."

# Arbitrary org convention: category -> queue code. NOT guessable from semantics.
ROUTES = {
    "QX-7": [  # storage / disk
        "The primary volume is full and writes are failing.",
        "Disk usage hit 100% on the data node.",
        "Backup job aborted: no space left on device.",
        "S3 bucket lifecycle rules deleted files early.",
        "The NAS mount went read-only overnight.",
        "Snapshot retention is filling the array.",
        "The database ran out of storage mid-write.",
        "A runaway log file consumed the whole partition.",
        "Object storage replication is lagging by hours.",
        "The volume expansion request is stuck resizing.",
    ],
    "QX-2": [  # network / dns
        "DNS resolution is intermittently failing for the API host.",
        "Packet loss spiked between the two availability zones.",
        "The load balancer health checks are timing out.",
        "TLS handshakes are stalling on the edge nodes.",
        "Cross-region latency jumped to 400ms.",
        "A BGP flap knocked out east-1 connectivity.",
        "The VPN tunnel keeps dropping every few minutes.",
        "Firewall rules are blocking the new service port.",
        "The CDN is returning stale edge cache entries.",
        "Route tables lost the default gateway entry.",
    ],
    "QX-9": [  # auth / identity
        "Users can't log in; SSO returns an invalid assertion.",
        "OAuth token refresh is rejected with 401.",
        "The identity provider certificate expired.",
        "MFA prompts are looping and never complete.",
        "Service accounts lost their IAM role bindings.",
        "Password reset emails are not being accepted.",
        "Session cookies are being invalidated too early.",
        "The LDAP directory sync stopped importing users.",
        "API keys were rotated but clients still use old ones.",
        "Group membership changes aren't propagating to access.",
    ],
    "QX-4": [  # billing / invoice
        "A customer was double-charged on their invoice.",
        "The metering pipeline dropped yesterday's usage.",
        "Refunds are stuck in a pending state.",
        "Tax calculation is wrong for EU orders.",
        "The subscription renewal failed to bill.",
        "Invoice PDFs are generating with a zero total.",
        "Proration is miscalculated after a plan upgrade.",
        "The payment gateway webhook never marked paid.",
        "Credits from last month were not applied.",
        "Currency conversion on the receipt looks off.",
    ],
    "QX-1": [  # frontend / ui
        "The dashboard renders a blank white page on load.",
        "A CSS regression broke the mobile navbar.",
        "Buttons on the settings page are unclickable.",
        "The chart widget throws a console error and won't mount.",
        "Dark mode colors are unreadable after the deploy.",
        "The date picker shows the wrong month.",
        "The modal won't close after submitting the form.",
        "Table sorting reverses on every second click.",
        "Icons fail to load and show broken image boxes.",
        "The sidebar collapses unexpectedly on scroll.",
    ],
}

# Held-out phrasings (same categories, new sentences) — strictly disjoint from training.
GOLDEN = {
    "QX-7": "The write-ahead log partition ran out of space.",
    "QX-2": "Requests to the internal DNS resolver are timing out.",
    "QX-9": "SAML login fails with a signature validation error.",
    "QX-4": "The usage report shows charges for a cancelled plan.",
    "QX-1": "The nav menu overlaps the content on small screens.",
}


def _chat_record(desc: str, code: str) -> DatasetRecord:
    return DatasetRecord(
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": desc},
            {"role": "assistant", "content": code},
        ]
    )


def main() -> None:
    data_dir = Path(__file__).resolve().parent.parent / "data"

    records = [
        _chat_record(desc, code) for code, descs in ROUTES.items() for desc in descs
    ]
    ds = Dataset(
        task="classify",
        records=records,
        created_at=CREATED_AT,
        provenance={"source": "scripts/build_route_dataset.py", "convention": "arbitrary QX codes"},
    )
    train_path = write_dataset(ds, data_dir / "route.jsonl")

    golden_path = data_dir / "route_golden.jsonl"
    golden_path.write_text(
        "".join(
            json.dumps({"prompt": desc, "expected": code}, sort_keys=True) + "\n"
            for code, desc in GOLDEN.items()
        ),
        encoding="utf-8",
    )

    print(f"wrote {train_path} ({len(ds)} chat records, {len(ROUTES)} queue codes)")
    print(f"wrote {golden_path} ({len(GOLDEN)} golden examples)")


if __name__ == "__main__":
    main()
