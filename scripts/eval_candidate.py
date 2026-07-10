#!/usr/bin/env python
"""Evaluate a trained candidate adapter vs. the base model on a golden set.

Wires the candidate through MLXProvider's per-request adapter slot (GenRequest.adapter,
resolved via AdapterStore.resolve_path with the A/B flag) and scores both base and
candidate with the objective exact-match metric. Prints both scores and whether the
candidate beats the base (treated as the incumbent floor for the promote gate).

Usage:
    HF_HUB_OFFLINE=1 uv run python scripts/eval_candidate.py <adapter-id> \
        --golden data/route_golden.jsonl \
        --system "You are a ticket router. Reply with ONLY the queue code."
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hearth.providers.base import GenRequest, Message
from hearth.providers.mlx import MLXProvider
from hearth.registry.adapters import AdapterStore
from hearth.training.eval import GoldenExample, GoldenSet, beats_incumbent, score_candidate

BASE_MODEL = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"


def load_golden(path: Path) -> GoldenSet:
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    return GoldenSet(
        task=rows[0].get("task", "classify") if rows else "classify",
        examples=[GoldenExample(r["prompt"], r["expected"]) for r in rows],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter_id")
    ap.add_argument("--golden", type=Path, required=True)
    ap.add_argument("--system", default="Reply with only the answer, nothing else.")
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--metric", default="exact", choices=["exact", "f1"])
    args = ap.parse_args()

    store = AdapterStore()
    adapter_path = store.resolve_path(args.adapter_id, allow_candidate=True)
    print(f"candidate adapter: {args.adapter_id} -> {adapter_path}")

    golden = load_golden(args.golden)
    provider = MLXProvider(model_id=BASE_MODEL)

    def _run(prompt: str, adapter: str | None) -> str:
        req = GenRequest(
            messages=[Message("system", args.system), Message("user", prompt)],
            model=BASE_MODEL,
            max_tokens=args.max_tokens,
            adapter=adapter,
        )
        return provider.generate(req).text

    gen_base = lambda p: _run(p, None)  # noqa: E731
    gen_candidate = lambda p: _run(p, adapter_path)  # noqa: E731

    print("\n== scoring base (no adapter) ==")
    base_report = score_candidate(golden, gen_base, metric=args.metric)
    print("== scoring candidate (adapter via A/B slot) ==")
    cand_report = score_candidate(golden, gen_candidate, metric=args.metric)

    print("\n-- per-example (expected | base | candidate) --")
    for ex in golden.examples:
        b = gen_base(ex.prompt).replace("\n", " ")[:40]
        c = gen_candidate(ex.prompt).replace("\n", " ")[:40]
        print(f"  {ex.expected:10s} | base={b!r:30s} | cand={c!r}")

    passed = beats_incumbent(cand_report, base_report)
    m = args.metric
    print("\n==== RESULT ====")
    print(f"base      {m} score: {base_report.score:.4f}  per-example={base_report.per_example}")
    print(f"candidate {m} score: {cand_report.score:.4f}  per-example={cand_report.per_example}")
    print(f"beats_incumbent(candidate, base) = {passed}")
    print(
        f"\nPromote with:\n  uv run hearth adapters promote {args.adapter_id} "
        f"--candidate-score {cand_report.score:g} --incumbent-score {base_report.score:g}"
    )


if __name__ == "__main__":
    main()
