#!/usr/bin/env python3
"""Two-tier LOCAL model ladder, end to end, on synthetic statement data.

Proves that ``ClassRule.local_model`` (config/routing.finance.yaml) actually puts different
work on different local models, with nothing leaving the machine:

  stage 1  categorize  every transaction -> tier 1 (small, fast), one call per row
  stage 2  aggregate   every total       -> **Python**, never a model
  stage 3  narrate     one summary       -> tier 2 (larger), over stage 2's numbers

Stage 2 is the load-bearing rule. LLM arithmetic is unreliable and these are financial
figures, so no model on this ladder is ever asked to add anything up: Python computes the
totals and hands them to tier 2 as facts to describe. The tier-2 prompt says so explicitly.

The router picks the model for each stage purely from the routing policy — this script never
names a model. Which model actually served is printed at every stage so the ladder is visible.

Run it sealed (see README.md)::

    HEARTH_BACKEND=mlx HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HEARTH_ROUTING_YAML=config/routing.finance.yaml \
    uv run python examples/finance/run_finance_ladder.py

``--dry-run`` swaps in the echo provider: the same routing decisions, no weights loaded.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

# huggingface_hub reads its cache location at import time, and `hearth models pull` stores
# weights under ~/.hearth/models rather than the default ~/.cache/huggingface. Point it there
# before anything imports the hub, so offline loading finds the pre-pulled weights.
os.environ.setdefault("HF_HUB_CACHE", str(Path.home() / ".hearth" / "models"))

sys.path.insert(0, str(_REPO_ROOT / "src"))

from hearth.observability.budget import BudgetAccountant  # noqa: E402
from hearth.observability.metrics import MetricsStore  # noqa: E402
from hearth.providers.base import (  # noqa: E402
    Capabilities,
    GenRequest,
    GenResult,
    Message,
    ModelProvider,
    ResourceEstimate,
)
from hearth.router import Router  # noqa: E402
from hearth.router.policy import RoutingPolicy, load_policy  # noqa: E402
from hearth.serving import ModelManager  # noqa: E402

# The closed label set the tier-1 model must choose from. Kept small and mutually exclusive
# so a 3B model can hold it in one short prompt.
CATEGORIES: tuple[str, ...] = (
    "groceries",
    "dining",
    "transport",
    "utilities",
    "subscriptions",
    "income",
    "transfer",
    "healthcare",
    "shopping",
    "fees",
)

_UNCATEGORIZED = "uncategorized"

_SYSTEM = (
    "You are a bank-transaction categorizer. Answer with exactly one category name from "
    "the list and nothing else. No punctuation, no explanation."
)


# -- data ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Transaction:
    """One synthetic statement row, plus its hand-written answer key."""

    date: str
    description: str
    amount: float
    expected: str
    difficulty: str


@dataclass
class Categorized:
    """A transaction after the tier-1 pass."""

    txn: Transaction
    predicted: str
    model: str
    latency_ms: float

    @property
    def correct(self) -> bool:
        return self.predicted == self.txn.expected


@dataclass
class Aggregates:
    """Every figure in the report, computed in Python. No model contributes a number."""

    total_income: float = 0.0
    total_spend: float = 0.0
    net: float = 0.0
    transaction_count: int = 0
    by_category: dict[str, float] = field(default_factory=dict)
    counts_by_category: dict[str, int] = field(default_factory=dict)
    largest: tuple[str, float] | None = None


def load_transactions(path: Path) -> list[Transaction]:
    """Read the synthetic CSV, skipping the leading ``#`` provenance comments."""
    rows = [line for line in path.read_text().splitlines() if not line.startswith("#")]
    return [
        Transaction(
            date=r["date"],
            description=r["description"],
            amount=float(r["amount"]),
            expected=r["expected_category"],
            difficulty=r["difficulty"],
        )
        for r in csv.DictReader(rows)
    ]


# -- the local provider that makes a ladder possible --------------------------------------


class LadderProvider:
    """Serves whichever local model the router picked for this request.

    ``MLXProvider`` is bound to a single model id, so a per-class ladder needs a front door
    that maps ``GenRequest.model`` -> the right resident provider. :class:`ModelManager` owns
    residency (LRU under a RAM ceiling, ARCHITECTURE §5); this only delegates, and records
    which model served each call so the ladder is visible in the output.
    """

    name = "ladder"

    def __init__(self, manager: ModelManager) -> None:
        self._manager = manager
        self.calls_by_model: dict[str, int] = defaultdict(int)

    def capabilities(self) -> Capabilities:
        return Capabilities(chat=True, stream=True)

    def footprint(self, model_id: str) -> ResourceEstimate:
        return self._manager.get(model_id).footprint(model_id)

    def generate(self, req: GenRequest) -> GenResult:
        self.calls_by_model[req.model] += 1
        return self._manager.get(req.model).generate(req)

    def stream(self, req: GenRequest):
        self.calls_by_model[req.model] += 1
        yield from self._manager.get(req.model).stream(req)


def _mlx_factory(model_id: str) -> ModelProvider:
    from hearth.providers.mlx import MLXProvider

    return MLXProvider(model_id)


def _echo_factory(model_id: str) -> ModelProvider:
    from hearth.providers.echo import EchoProvider

    return EchoProvider()


# -- the seal ------------------------------------------------------------------------------


def verify_no_egress(policy: RoutingPolicy, path: Path) -> None:
    """Assert the same no-egress posture ``scripts/hearth_private.sh`` verifies, and exit if not.

    That script pins ``config/routing.private.yaml`` and takes no profile argument, so this
    example re-runs its three checks against whatever profile it was actually pointed at:
    zero remotes, no default remote resolves, and every class is local/never-escalate. The
    check runs *before* any weights load, so a leaky profile fails closed.
    """
    problems = []
    if policy.remotes:
        problems.append(f"remotes defined: {sorted(policy.remotes)}")
    if policy.remote_for() is not None:
        problems.append("a default remote resolves")
    escapable = [
        c for c, r in policy.classes.items() if r.backend != "local" or r.escalate != "never"
    ]
    if escapable:
        problems.append(f"classes can leave local: {escapable}")
    if problems:
        print(f"SEALED CHECK FAILED for {path}: {'; '.join(problems)}", file=sys.stderr)
        raise SystemExit(2)
    print(f"  seal: 0 remotes, no default remote, all {len(policy.classes)} classes local/never")


# -- stage 1: categorize (tier 1) ----------------------------------------------------------


def _prompt(txn: Transaction) -> list[Message]:
    direction = "credit (money in)" if txn.amount > 0 else "debit (money out)"
    return [
        Message(role="system", content=_SYSTEM),
        Message(
            role="user",
            content=(
                f"Categories: {', '.join(CATEGORIES)}\n\n"
                f"Transaction: {txn.description}\n"
                f"Amount: {abs(txn.amount):.2f} {direction}\n"
                "Category:"
            ),
        ),
    ]


def _parse_label(text: str) -> str:
    """Map a model's reply onto the closed label set (earliest match wins)."""
    lowered = text.strip().lower()
    hits = [(lowered.find(c), c) for c in CATEGORIES if c in lowered]
    return min(hits)[1] if hits else _UNCATEGORIZED


def categorize(router: Router, txns: list[Transaction]) -> list[Categorized]:
    """Route every row through the ``classify`` class — one tier-1 call per transaction."""
    out: list[Categorized] = []
    for i, txn in enumerate(txns, 1):
        req = GenRequest(messages=_prompt(txn), model="auto", max_tokens=8, temperature=0.0)
        started = time.perf_counter()
        routed = router.route(req, intent="classify")
        elapsed = (time.perf_counter() - started) * 1000.0
        predicted = _parse_label(routed.result.text)
        out.append(Categorized(txn, predicted, routed.result.model, elapsed))
        mark = "ok  " if predicted == txn.expected else "MISS"
        print(
            f"  [{i:>2}/{len(txns)}] {mark} {predicted:<13} "
            f"(want {txn.expected:<13}) {elapsed:7.0f} ms  {txn.description[:44]}"
        )
    return out


# -- stage 2: aggregate (PYTHON — never a model) -------------------------------------------


def aggregate(rows: list[Categorized]) -> Aggregates:
    """Compute every figure in the report. Deliberately model-free: this is the money.

    Credits (positive amounts) are income; debits are spend, accumulated as positive
    magnitudes per predicted category. Nothing here is ever handed to a model to calculate —
    tier 2 only receives the finished numbers.
    """
    agg = Aggregates(transaction_count=len(rows))
    by_category: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    largest: tuple[str, float] | None = None

    for row in rows:
        amount = row.txn.amount
        counts[row.predicted] += 1
        if amount > 0:
            agg.total_income += amount
        else:
            spend = -amount
            agg.total_spend += spend
            by_category[row.predicted] += spend
            if largest is None or spend > largest[1]:
                largest = (row.txn.description, spend)

    agg.net = agg.total_income - agg.total_spend
    agg.by_category = dict(sorted(by_category.items(), key=lambda kv: -kv[1]))
    agg.counts_by_category = dict(counts)
    agg.largest = largest
    return agg


def render_facts(agg: Aggregates) -> str:
    """The precomputed fact sheet handed to tier 2 — the only numbers it ever sees."""
    lines = [
        f"Transactions: {agg.transaction_count}",
        f"Total income: ${agg.total_income:,.2f}",
        f"Total spend: ${agg.total_spend:,.2f}",
        f"Net: ${agg.net:,.2f}",
        "Spend by category (already summed):",
    ]
    for category, total in agg.by_category.items():
        share = (total / agg.total_spend * 100.0) if agg.total_spend else 0.0
        count = agg.counts_by_category.get(category, 0)
        lines.append(f"  - {category}: ${total:,.2f} across {count} transactions ({share:.1f}%)")
    if agg.largest:
        lines.append(f"Largest single debit: {agg.largest[0]} at ${agg.largest[1]:,.2f}")
    return "\n".join(lines)


# -- stage 3: narrate (tier 2) -------------------------------------------------------------


def narrate(router: Router, agg: Aggregates, max_tokens: int) -> tuple[str, str, float]:
    """Route the fact sheet through the ``summarize`` class — one tier-2 call."""
    facts = render_facts(agg)
    messages = [
        Message(
            role="system",
            content=(
                "You are a personal-finance analyst. Every figure you need has already been "
                "computed and is given below. Do NOT calculate, re-add, or adjust any number: "
                "quote the given figures verbatim and never invent one that is not listed."
            ),
        ),
        Message(
            role="user",
            content=(
                "Here is a month of categorized spending, already totalled:\n\n"
                f"{facts}\n\n"
                "Write a short plain-English summary (4-6 sentences) of where the money went "
                "and what stands out. Quote figures exactly as given."
            ),
        ),
    ]
    req = GenRequest(messages=messages, model="auto", max_tokens=max_tokens, temperature=0.3)
    started = time.perf_counter()
    routed = router.route(req, intent="summarize")
    return routed.result.text, routed.result.model, (time.perf_counter() - started) * 1000.0


# -- reporting -----------------------------------------------------------------------------


def report_accuracy(rows: list[Categorized]) -> None:
    """Print the honest scoreboard: overall, and split by the ``difficulty`` column."""
    total = len(rows)
    hits = sum(r.correct for r in rows)
    hard = [r for r in rows if r.txn.difficulty == "hard"]
    easy = [r for r in rows if r.txn.difficulty != "hard"]
    print(f"\n  accuracy overall : {hits}/{total} = {hits / total:.1%}")
    if easy:
        e = sum(r.correct for r in easy)
        print(f"  accuracy easy    : {e}/{len(easy)} = {e / len(easy):.1%}")
    if hard:
        h = sum(r.correct for r in hard)
        print(f"  accuracy HARD    : {h}/{len(hard)} = {h / len(hard):.1%}")
    misses = [r for r in rows if not r.correct]
    if misses:
        print(f"\n  {len(misses)} miss(es):")
        for r in misses:
            print(
                f"    [{r.txn.difficulty:<4}] {r.txn.description[:46]:<46} "
                f"got {r.predicted:<14} want {r.txn.expected}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parent / "statements.csv",
        help="synthetic statement CSV (default: the bundled one)",
    )
    parser.add_argument(
        "--routing",
        type=Path,
        default=None,
        help="routing profile (default: $HEARTH_ROUTING_YAML, else config/routing.finance.yaml)",
    )
    parser.add_argument("--limit", type=int, default=0, help="only process the first N rows")
    parser.add_argument(
        "--max-tokens", type=int, default=320, help="tier-2 narrative budget (default 320)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="use the echo provider: same routing decisions, no weights loaded",
    )
    parser.add_argument(
        "--ram-ceiling-gb",
        type=float,
        default=24.0,
        help="resident-model RAM ceiling for the ModelManager (default 24)",
    )
    args = parser.parse_args(argv)

    routing = args.routing or Path(
        os.environ.get("HEARTH_ROUTING_YAML", _REPO_ROOT / "config" / "routing.finance.yaml")
    )

    print("=" * 78)
    print("HEARTH two-tier local ladder — synthetic finance harness")
    print("=" * 78)
    print(f"  routing : {routing}")
    print(f"  data    : {args.csv}  (synthetic; no real account data exists in this example)")
    print(f"  offline : HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE', '0')} "
          f"TRANSFORMERS_OFFLINE={os.environ.get('TRANSFORMERS_OFFLINE', '0')} "
          f"HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE')}")

    policy = load_policy(routing)
    verify_no_egress(policy, routing)

    txns = load_transactions(args.csv)
    if args.limit:
        txns = txns[: args.limit]

    factory = _echo_factory if args.dry_run else _mlx_factory
    provider = LadderProvider(ModelManager(factory, ram_ceiling_gb=args.ram_ceiling_gb))
    router = Router(
        local_provider=provider,
        policy=policy,
        # Zero remote budget: even a policy bug could not fund an escalation from here.
        budget=BudgetAccountant(policy.defaults.remote_budget_tokens_per_day),
        metrics=MetricsStore(),
    )

    tier1 = policy.rule_for("classify").local_model or policy.defaults.local_model
    tier2 = policy.rule_for("summarize").local_model or policy.defaults.local_model
    print(f"\n  ladder  : classify -> {tier1}")
    print(f"            summarize -> {tier2}")

    print(f"\n-- stage 1: categorize {len(txns)} transactions (tier 1) " + "-" * 20)
    stage1_started = time.perf_counter()
    rows = categorize(router, txns)
    stage1_ms = (time.perf_counter() - stage1_started) * 1000.0
    served1 = sorted({r.model for r in rows})
    print(f"\n  served by: {', '.join(served1)}")
    print(f"  stage 1 total: {stage1_ms / 1000:.1f} s "
          f"({stage1_ms / max(1, len(rows)):.0f} ms/transaction)")
    report_accuracy(rows)

    print("\n-- stage 2: aggregate (PYTHON — no model touches these numbers) " + "-" * 12)
    stage2_started = time.perf_counter()
    agg = aggregate(rows)
    stage2_ms = (time.perf_counter() - stage2_started) * 1000.0
    print(render_facts(agg))
    print(f"\n  stage 2 total: {stage2_ms:.2f} ms  (pure Python)")

    print("\n-- stage 3: narrate over the computed aggregates (tier 2) " + "-" * 17)
    print("  (loading the larger model — first call includes the cold load)")
    text, model3, stage3_ms = narrate(router, agg, args.max_tokens)
    print(f"\n  served by: {model3}")
    print(f"  stage 3 total: {stage3_ms / 1000:.1f} s (load + generate)\n")
    print(text)

    print("\n" + "=" * 78)
    print("  ladder actually exercised (calls per model):")
    for model_id, n in sorted(provider.calls_by_model.items()):
        print(f"    {n:>3} x {model_id}")
    rollup = router.metrics.rollup()
    print(f"  end to end: {(stage1_ms + stage2_ms + stage3_ms) / 1000:.1f} s")
    print(f"  backend mix: {rollup['backend_mix']}")
    print(f"  escalations: {rollup['escalations']} "
          "(the profile defines no remote to escalate to)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
