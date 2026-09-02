"""Eval harness + promotion gate (ARCHITECTURE §7, ADR-006, Phase 4; LEARNING_plan P1).

Scores a candidate adapter against a golden set per task class and encodes the gate that
protects promotion: **a candidate must BEAT the incumbent to be promotable** (ADR-006) —
and, since LEARNING_plan §3, must beat it *significantly*, on a golden set both sides were
provably scored against, under decode parameters recorded in the report.

Metrics (ARCHITECTURE §7):
  * ``extract`` / ``classify`` — exact-match / token-F1 (deterministic, no judge).
  * ``draft`` / ``code``       — win-rate via a pluggable judge hook (no default judge:
    :func:`default_judge` raises, because a gate that cannot run is safer than a gate that
    always passes — LEARNING_plan F1).

Scoring is done against a caller-supplied ``generate`` function ``(prompt) -> text`` so
this harness is fully testable with fakes and never touches a real model. Every report
carries the identity of what it measured (:class:`EvalReport`); the gate refuses to compare
two reports from different universes rather than silently comparing two bare floats.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field

from .stats import (
    DEFAULT_BOOTSTRAP_ITERATIONS,
    discordant_pairs,
    is_binary,
    mcnemar_exact_p,
    paired_bootstrap,
)

# Generates a candidate answer for a prompt. Injected — a real caller wires a provider;
# tests pass a dict-backed fake.
Generate = Callable[[str], str]

# Judges whether ``candidate`` beats ``reference`` for ``prompt``; returns True on a win.
# Injected for draft/code classes — there is deliberately no default (LEARNING_plan F1).
Judge = Callable[[str, str, str], bool]

# Classes scored by objective string metrics vs. a judge (ARCHITECTURE §7).
_OBJECTIVE_CLASSES = frozenset({"extract", "classify", "rank", "summarize"})

# Pre-registration defaults for the gate (LEARNING_plan §3.3).
DEFAULT_ALPHA = 0.05
# Policy floor on the golden set, well above the *mathematical* minimum of 5 (the size at
# which a clean sweep first reaches p = 0.5**5 = 0.031). Five items can only clear alpha
# when the incumbent misses every single item the candidate hits; 30 is the plan's working
# bar for a set that can license a promotion against a competent incumbent.
DEFAULT_MIN_N = 30

# Degenerate baselines every candidate must beat (LEARNING_plan §3.2.4).
BASELINE_EMPTY = "empty"
BASELINE_MAJORITY = "majority_label"
BASELINE_COPY_INPUT = "copy_input"


class GateProvenanceError(ValueError):
    """Raised when two reports are not comparable (different golden set, metric, config).

    This is deliberately an exception rather than a failed gate: a mismatch means the
    question was ill-posed, not that the candidate lost.
    """


@dataclass(frozen=True)
class GoldenExample:
    """One golden (prompt, expected) pair for a task class."""

    prompt: str
    expected: str


@dataclass(frozen=True)
class GoldenSet:
    """A golden evaluation set for one task class.

    ``version`` is the operator-declared label of the set ("v3"); ``sha`` is its content
    identity, and is what a report and a pre-registration actually pin (LEARNING_plan §3.1)
    — a renamed file with the same items compares equal, an edited file does not.
    """

    task: str
    examples: list[GoldenExample]
    version: str = ""

    def __len__(self) -> int:
        return len(self.examples)

    @property
    def sha(self) -> str:
        """SHA-256 over the canonicalized, sorted ``prompt\\x1fexpected`` items."""
        lines = sorted(
            f"{ex.prompt.strip()}\x1f{ex.expected.strip()}" for ex in self.examples
        )
        return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EvalConfig:
    """Decode parameters an eval ran under — the reproducibility fingerprint (F4).

    ``temperature`` defaults to 0.0 because a gate that can be re-rolled is not a gate:
    at 5 golden items the score granularity is 0.2, so one sampled token flip moves the
    result 20 points. Callers that genuinely want sampling must say so and wear the
    ``unverified`` stamp it earns them.
    """

    temperature: float = 0.0
    max_tokens: int = 64
    seed: int | None = None
    system_hash: str = ""

    @property
    def fingerprint(self) -> str:
        """Short SHA-256 over the decode parameters; equal fingerprints are comparable."""
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @property
    def deterministic(self) -> bool:
        """True when this config makes generation reproducible (greedy decode)."""
        return self.temperature == 0.0

    @classmethod
    def for_system(cls, system: str | None, **kwargs: object) -> EvalConfig:
        """Build a config, hashing ``system`` into ``system_hash`` (None -> empty)."""
        digest = (
            hashlib.sha256(system.encode("utf-8")).hexdigest()[:16] if system else ""
        )
        return cls(system_hash=digest, **kwargs)  # type: ignore[arg-type]


@dataclass(frozen=True)
class EvalReport:
    """Aggregate score of a candidate over a golden set, *plus who and what produced it*.

    ``score`` is in [0, 1] (mean per-example score). ``per_example`` keeps the individual
    scores — the gate needs the pairing, not just the mean, so this is load-bearing rather
    than decorative. ``metric`` names how it was computed.

    The provenance block (``golden_sha`` / ``golden_version`` / ``model_id`` / ``config``)
    is what makes two reports comparable at all: without it the gate cannot tell a 6-example
    set at ``max_tokens=24`` from a 5-example set at ``max_tokens=64`` (LEARNING_plan F3).
    """

    task: str
    metric: str
    score: float
    per_example: list[float] = field(default_factory=list)
    n: int = 0
    golden_sha: str = ""
    golden_version: str = ""
    model_id: str = ""
    config: EvalConfig | None = None
    measured_at: str = ""

    @property
    def config_fingerprint(self) -> str:
        """The decode-parameter fingerprint, or "" when the report carries no config."""
        return self.config.fingerprint if self.config is not None else ""

    @property
    def has_provenance(self) -> bool:
        """True when this report can be compared to another (it names its golden set)."""
        return bool(self.golden_sha) and self.config is not None

    def to_json(self) -> dict:
        obj = asdict(self)
        obj["config_fingerprint"] = self.config_fingerprint
        return obj

    @classmethod
    def from_json(cls, obj: dict) -> EvalReport:
        cfg = obj.get("config")
        return cls(
            task=obj["task"],
            metric=obj["metric"],
            score=float(obj["score"]),
            per_example=[float(v) for v in obj.get("per_example", [])],
            n=int(obj.get("n", 0)),
            golden_sha=obj.get("golden_sha", ""),
            golden_version=obj.get("golden_version", ""),
            model_id=obj.get("model_id", ""),
            config=EvalConfig(**cfg) if isinstance(cfg, dict) else None,
            measured_at=obj.get("measured_at", ""),
        )


@dataclass(frozen=True)
class GateResult:
    """The promotion gate's verdict — a decision *with its reasoning attached*.

    Replaces the old bare bool so ``promotion_proof`` has something worth persisting: what
    was compared, which test was used, the p-value/interval it produced, and every reason
    the gate refused. Truthy iff ``passed``, so ``if gate:`` reads naturally.
    """

    passed: bool
    test: str
    reasons: tuple[str, ...] = ()
    p_value: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    b: int = 0
    c: int = 0
    n: int = 0
    alpha: float = DEFAULT_ALPHA
    margin: float = 0.0
    min_n: int = DEFAULT_MIN_N
    candidate_score: float = 0.0
    incumbent_score: float | None = None
    incumbent_role: str = "none"
    candidate_id: str = ""
    incumbent_id: str = ""
    golden_sha: str = ""
    golden_version: str = ""
    metric: str = ""
    config_fingerprint: str = ""
    baselines: dict[str, float] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.passed

    @property
    def reason(self) -> str:
        """One-line summary: why it failed, or that it passed."""
        return "; ".join(self.reasons) if self.reasons else "gate passed"

    def as_proof(self) -> dict[str, object]:
        """The auditable ``promotion_proof`` payload (LEARNING_plan §3.6).

        Every field here is checkable after the fact by someone who does not trust the
        operator — which is the operational definition of eval authority.
        """
        return {
            "gate": "verified",
            "gate_passed": self.passed,
            "reason": self.reason,
            "test": self.test,
            "p_value": self.p_value,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "b": self.b,
            "c": self.c,
            "n": self.n,
            "alpha": self.alpha,
            "margin": self.margin,
            "min_n": self.min_n,
            "candidate_score": self.candidate_score,
            "incumbent_score": self.incumbent_score,
            "incumbent_role": self.incumbent_role,
            "candidate_id": self.candidate_id,
            "incumbent_id": self.incumbent_id,
            "golden_sha": self.golden_sha,
            "golden_version": self.golden_version,
            "metric": self.metric,
            "config": self.config_fingerprint,
            "baselines": dict(self.baselines),
        }


def exact_match_score(candidate: str, expected: str) -> float:
    """1.0 if the (stripped, case-folded) strings match exactly, else 0.0."""
    return 1.0 if candidate.strip().casefold() == expected.strip().casefold() else 0.0


def token_f1_score(candidate: str, expected: str) -> float:
    """Token-level F1 over whitespace tokens (case-folded). 0.0 when either side empty."""
    cand = candidate.casefold().split()
    gold = expected.casefold().split()
    if not cand or not gold:
        return 1.0 if not cand and not gold else 0.0
    # Multiset overlap (bounded by counts on each side).
    overlap = 0
    remaining = list(gold)
    for tok in cand:
        if tok in remaining:
            remaining.remove(tok)
            overlap += 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(cand)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def score_candidate(
    golden: GoldenSet,
    generate: Generate,
    *,
    metric: str | None = None,
    judge: Judge | None = None,
    model_id: str = "",
    config: EvalConfig | None = None,
    measured_at: str = "",
) -> EvalReport:
    """Score ``generate`` over ``golden`` and return an :class:`EvalReport`.

    ``metric`` selects the objective scorer for extract/classify-style tasks: ``"exact"``
    or ``"f1"`` (default). For subjective classes (draft/code) pass a ``judge`` *instead*;
    each example scores 1.0 on a judged win, else 0.0. Supplying both raises — the old
    behaviour silently let the judge override ``metric``, so a caller asking for exact-match
    got a judge and no warning (LEARNING_plan F1).

    ``model_id`` and ``config`` stamp the report with what produced it; pass them from the
    real eval path so the gate can refuse to compare unrelated runs.
    """
    if not golden.examples:
        raise ValueError("cannot score an empty golden set")
    if judge is not None and metric is not None:
        raise ValueError(
            "pass either 'metric' or 'judge', not both — a judge silently overriding the "
            "metric is how a length heuristic ends up gating an exact-match task"
        )

    if judge is not None:
        used_metric = "judge_win_rate"
        scorer = lambda cand, ex: 1.0 if judge(ex.prompt, cand, ex.expected) else 0.0  # noqa: E731
    elif metric in (None, "f1"):
        used_metric = "token_f1"
        scorer = lambda cand, ex: token_f1_score(cand, ex.expected)  # noqa: E731
    elif metric == "exact":
        used_metric = "exact_match"
        scorer = lambda cand, ex: exact_match_score(cand, ex.expected)  # noqa: E731
    else:
        raise ValueError(f"unknown metric: {metric!r} (use 'exact', 'f1', or pass a judge)")

    per_example = [scorer(generate(ex.prompt), ex) for ex in golden.examples]
    mean = sum(per_example) / len(per_example)
    return EvalReport(
        task=golden.task,
        metric=used_metric,
        score=mean,
        per_example=per_example,
        n=len(per_example),
        golden_sha=golden.sha,
        golden_version=golden.version,
        model_id=model_id,
        config=config,
        measured_at=measured_at,
    )


def baseline_reports(
    golden: GoldenSet,
    *,
    metric: str | None = None,
    config: EvalConfig | None = None,
    measured_at: str = "",
) -> dict[str, EvalReport]:
    """Score the three degenerate baselines on ``golden`` (LEARNING_plan §3.2.4).

    ``empty`` emits nothing, ``majority_label`` always emits the most common expected
    answer, ``copy_input`` echoes the prompt. A candidate that cannot beat all three has
    learned nothing, whatever the incumbent scored — this is what catches "token-F1 rewards
    an adapter that emits the word *the*" without needing a special case for it.

    Judged (draft/code) classes have no reference to be degenerate against, so this is only
    meaningful for the objective metrics; pass the same ``metric`` the candidate used.
    """
    if not golden.examples:
        raise ValueError("cannot score an empty golden set")
    majority = Counter(ex.expected.strip() for ex in golden.examples).most_common(1)[0][0]
    generators: dict[str, Generate] = {
        BASELINE_EMPTY: lambda prompt: "",
        BASELINE_MAJORITY: lambda prompt: majority,
        BASELINE_COPY_INPUT: lambda prompt: prompt,
    }
    return {
        name: score_candidate(
            golden,
            gen,
            metric=metric,
            model_id=f"baseline:{name}",
            config=config,
            measured_at=measured_at,
        )
        for name, gen in generators.items()
    }


def check_determinism(
    golden: GoldenSet, generate: Generate, *, sample: int = 3
) -> tuple[str, ...]:
    """Re-generate the first ``sample`` golden prompts and return the ones that differed.

    The eval-path equivalent of ``lora._preflight_batch_size``: it turns "this provider is
    sampling" from a silently re-rollable score into an actionable refusal *before* the
    number reaches a promotion. An empty tuple means the run is reproducible.
    """
    mismatched = []
    for ex in golden.examples[: max(0, sample)]:
        if generate(ex.prompt) != generate(ex.prompt):
            mismatched.append(ex.prompt)
    return tuple(mismatched)


def _assert_comparable(candidate: EvalReport, incumbent: EvalReport) -> None:
    """Raise :class:`GateProvenanceError` unless the two reports measure the same thing."""
    if candidate.task and incumbent.task and candidate.task != incumbent.task:
        raise GateProvenanceError(
            f"task mismatch: candidate {candidate.task!r} vs incumbent {incumbent.task!r}"
        )
    if candidate.metric != incumbent.metric:
        raise GateProvenanceError(
            f"metric mismatch: candidate {candidate.metric!r} vs incumbent {incumbent.metric!r}"
        )
    if candidate.golden_sha != incumbent.golden_sha:
        raise GateProvenanceError(
            "golden-set mismatch: candidate scored on "
            f"{candidate.golden_sha[:12] or '<unknown>'} "
            f"({candidate.golden_version or 'unversioned'}), incumbent on "
            f"{incumbent.golden_sha[:12] or '<unknown>'} "
            f"({incumbent.golden_version or 'unversioned'}) — these are different universes"
        )
    if candidate.config_fingerprint != incumbent.config_fingerprint:
        raise GateProvenanceError(
            "decode-config mismatch: candidate "
            f"{candidate.config_fingerprint or '<unknown>'} vs incumbent "
            f"{incumbent.config_fingerprint or '<unknown>'}"
        )
    if len(candidate.per_example) != len(incumbent.per_example):
        raise GateProvenanceError(
            "per-example vectors differ in length: "
            f"{len(candidate.per_example)} vs {len(incumbent.per_example)}"
        )


def evaluate_gate(
    candidate: EvalReport,
    incumbent: EvalReport | None,
    *,
    incumbent_role: str = "incumbent",
    baselines: dict[str, EvalReport] | None = None,
    margin: float = 0.0,
    alpha: float = DEFAULT_ALPHA,
    min_n: int = DEFAULT_MIN_N,
    test: str = "auto",
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = 0,
    candidate_id: str = "",
    incumbent_id: str = "",
) -> GateResult:
    """The promotion gate: is ``candidate``'s lift over ``incumbent`` real? (§3.3)

    The decision rule, in order — every clause must hold:

    1. **A comparison exists.** ``incumbent`` is required. There is no "first adapter is
       free" path: when no adapter is promoted for the task, **the base model is the
       incumbent** (pass it with ``incumbent_role="base"``). The old
       ``candidate.score > 0.0`` promoted an adapter that emitted the word "the".
    2. **The two sides are comparable** — same task, metric, golden-set sha and decode
       fingerprint, over equal-length per-example vectors. A mismatch *raises*
       :class:`GateProvenanceError`; it is not a losing candidate, it is a bad question.
    3. **The golden set is large enough** (``n >= min_n``) to license the claim.
    4. **The point estimate improves** by more than ``margin`` (a tie does not beat).
    5. **The improvement is significant at ``alpha``** under a paired test over the
       per-example vectors — exact one-sided McNemar for binary outcomes, paired bootstrap
       for continuous ones (``test="auto"`` picks; ``"mcnemar"`` / ``"bootstrap"`` force).
    6. **The candidate beats every degenerate baseline** it was given, by ``margin``.

    Returns a :class:`GateResult` carrying the verdict and all of its reasoning; it never
    returns a bare bool, so a promotion proof can be audited rather than believed.
    """
    reasons: list[str] = []
    baselines = baselines or {}
    baseline_scores = {name: rep.score for name, rep in baselines.items()}

    if incumbent is None:
        return GateResult(
            passed=False,
            test="none",
            reasons=(
                "no incumbent report: with no promoted adapter the BASE MODEL is the "
                "incumbent — score it and pass it as incumbent_role='base'",
            ),
            n=candidate.n,
            alpha=alpha,
            margin=margin,
            min_n=min_n,
            candidate_score=candidate.score,
            incumbent_role="none",
            candidate_id=candidate_id or candidate.model_id,
            golden_sha=candidate.golden_sha,
            golden_version=candidate.golden_version,
            metric=candidate.metric,
            config_fingerprint=candidate.config_fingerprint,
            baselines=baseline_scores,
        )

    _assert_comparable(candidate, incumbent)

    if not candidate.per_example:
        raise GateProvenanceError(
            "candidate report carries no per-example vector; the gate is paired and "
            "cannot run on a mean alone"
        )
    if not candidate.has_provenance:
        reasons.append(
            "candidate report has no provenance (golden_sha/config) — unverifiable"
        )

    n = len(candidate.per_example)
    if n < min_n:
        reasons.append(
            f"golden set too small: n={n} < min_n={min_n} "
            "(see stats.min_n_for_alpha for what a set this size can license)"
        )
    if candidate.score <= incumbent.score + margin:
        reasons.append(
            f"no lift: candidate {candidate.score:.4f} does not exceed "
            f"{incumbent_role} {incumbent.score:.4f} + margin {margin:g}"
        )

    # -- the paired test -----------------------------------------------------------------
    binary = is_binary(candidate.per_example) and is_binary(incumbent.per_example)
    if test == "auto":
        chosen = "mcnemar_exact" if binary else "paired_bootstrap"
    elif test == "mcnemar":
        chosen = "mcnemar_exact"
    elif test == "bootstrap":
        chosen = "paired_bootstrap"
    else:
        raise ValueError(f"unknown test: {test!r} (use 'auto', 'mcnemar', or 'bootstrap')")

    b = c = 0
    p_value: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    if chosen == "mcnemar_exact":
        b, c = discordant_pairs(candidate.per_example, incumbent.per_example)
        p_value = mcnemar_exact_p(b, c)
    else:
        boot = paired_bootstrap(
            candidate.per_example,
            incumbent.per_example,
            margin=margin,
            alpha=alpha,
            iterations=bootstrap_iterations,
            seed=seed,
        )
        p_value = boot.p_value
        ci_low, ci_high = boot.ci_low, boot.ci_high

    if p_value > alpha:
        detail = f"b={b}, c={c}" if chosen == "mcnemar_exact" else f"ci_low={ci_low:.4f}"
        reasons.append(
            f"not significant: {chosen} p={p_value:.4f} > alpha={alpha:g} ({detail})"
        )

    # -- degenerate baselines ------------------------------------------------------------
    for name, report in sorted(baselines.items()):
        if candidate.score <= report.score + margin:
            reasons.append(
                f"fails degenerate baseline {name!r}: candidate {candidate.score:.4f} "
                f"does not exceed {report.score:.4f} + margin {margin:g}"
            )

    return GateResult(
        passed=not reasons,
        test=chosen,
        reasons=tuple(reasons),
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        b=b,
        c=c,
        n=n,
        alpha=alpha,
        margin=margin,
        min_n=min_n,
        candidate_score=candidate.score,
        incumbent_score=incumbent.score,
        incumbent_role=incumbent_role,
        candidate_id=candidate_id or candidate.model_id,
        incumbent_id=incumbent_id or incumbent.model_id,
        golden_sha=candidate.golden_sha,
        golden_version=candidate.golden_version,
        metric=candidate.metric,
        config_fingerprint=candidate.config_fingerprint,
        baselines=baseline_scores,
    )


def beats_incumbent(
    candidate: EvalReport,
    incumbent: EvalReport | None,
    *,
    margin: float = 0.0,
) -> bool:
    """Legacy mean-only comparison — **not** the promotion gate. Use :func:`evaluate_gate`.

    Kept because callers outside this package (``scripts/eval_candidate.py``, the runbook)
    still ask the simple question "is the mean higher?". It answers exactly that and
    nothing more: no significance, no provenance check, no baselines. A promotion decided
    on this is stamped ``gate: "unverified"`` by the registry so a weak gate is never
    indistinguishable from a strong one in the audit trail.

    A missing incumbent now returns **False**, not "any score above zero wins": when there
    is no promoted adapter the base model is the incumbent and must be scored
    (LEARNING_plan F2).
    """
    if incumbent is None:
        return False
    return candidate.score > incumbent.score + margin


def objective_metric_for(task: str) -> str:
    """Default objective metric name for a task class (F1 for objective, else judge)."""
    return "f1" if task in _OBJECTIVE_CLASSES else "judge"


def default_judge(prompt: str, candidate: str, reference: str) -> bool:
    """There is no default judge — this always raises :class:`NotImplementedError`.

    It used to return ``len(candidate) >= len(reference)``: a verbosity contest that never
    read the prompt, never read the content, and awarded a perfect win-rate to a fixed 10 KB
    constant string — while being the documented default for four of nine task classes
    (LEARNING_plan F1). A gate that cannot run is safer than a gate that always passes, so
    callers wanting a judged class must inject a judge they have calibrated (§4.2).
    """
    raise NotImplementedError(
        "no default judge: judged task classes (draft/code/reason/chat) require an "
        "explicitly injected, calibrated judge — see docs/LEARNING_plan.md §4"
    )


def as_golden_set(
    task: str, pairs: Sequence[tuple[str, str]], *, version: str = ""
) -> GoldenSet:
    """Convenience: build a :class:`GoldenSet` from ``(prompt, expected)`` pairs."""
    return GoldenSet(
        task=task, examples=[GoldenExample(p, e) for p, e in pairs], version=version
    )


__all__ = [
    "BASELINE_COPY_INPUT",
    "BASELINE_EMPTY",
    "BASELINE_MAJORITY",
    "DEFAULT_ALPHA",
    "DEFAULT_MIN_N",
    "EvalConfig",
    "EvalReport",
    "GateProvenanceError",
    "GateResult",
    "Generate",
    "GoldenExample",
    "GoldenSet",
    "Judge",
    "as_golden_set",
    "baseline_reports",
    "beats_incumbent",
    "check_determinism",
    "default_judge",
    "evaluate_gate",
    "exact_match_score",
    "objective_metric_for",
    "score_candidate",
    "token_f1_score",
]
