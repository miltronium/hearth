"""Pre-registration for the promotion gate (LEARNING_plan §3.4).

The operator declares the bar **before** training: which golden set (by content sha), which
metric, which alpha, and the minimum effect that would count. The file is then committed to
git. ``promote`` refuses unless a matching pre-registration exists *and* is committed and
unmodified — so the bar provably predates the measurement and cannot be moved after seeing
the score.

This is the user's own APEX methodology, mechanized: the upgrade over discipline is that the
harness refuses to run without it, so the habit cannot erode under deadline pressure.

    evals/<task>/prereg-<YYYY-MM-DD>-<slug>.yaml

Git is consulted by shelling out (``git ls-files`` / ``git diff``) rather than by parsing
``.git`` — a repository is the source of truth about its own index, and the check must be
the same one a reviewer would run by hand.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from .eval import (
    BASELINE_COPY_INPUT,
    BASELINE_EMPTY,
    BASELINE_MAJORITY,
    DEFAULT_ALPHA,
    DEFAULT_MIN_N,
    EvalConfig,
    EvalReport,
    check_bar,
)

# Baselines a pre-registration requires by default (LEARNING_plan §3.2.4).
DEFAULT_BASELINES = (BASELINE_EMPTY, BASELINE_MAJORITY, BASELINE_COPY_INPUT)

# Fields with no sensible default — a prereg that omits one is not a prereg.
_REQUIRED = ("task", "golden_sha", "metric")


class PreRegError(ValueError):
    """Raised on a malformed, mismatched, or uncommitted pre-registration."""


@dataclass(frozen=True)
class GitStatus:
    """Whether a file is committed to git and unmodified since.

    ``commit`` is the commit that last changed the file — the one whose content is the bar
    in force — and ``committed_at`` its committer timestamp (ISO 8601). It is deliberately
    NOT ``rev-parse HEAD``: HEAD moves with every unrelated commit, so recording it said
    nothing about when the bar was written (B-061). ``introduced_commit`` is the commit
    that first added the file.
    """

    committed: bool
    reason: str
    commit: str = ""
    repo_root: str = ""
    committed_at: str = ""
    introduced_commit: str = ""


@dataclass(frozen=True)
class PreRegistration:
    """A parsed, validated pre-registration.

    ``sha`` is the SHA-256 of the file bytes and is what lands in ``promotion_proof`` as
    ``prereg_sha``: an auditor can re-hash the committed file and confirm the bar recorded
    in the proof is the bar that was actually registered.
    """

    path: Path
    sha: str
    task: str
    golden_sha: str
    golden_version: str
    metric: str
    alpha: float
    min_effect: float
    min_n: int
    test: str
    must_beat_baselines: tuple[str, ...]
    generation: EvalConfig
    hypothesis: str
    raw: dict

    def mismatches(self, report: EvalReport) -> tuple[str, ...]:
        """Every way ``report`` fails to be the measurement this prereg registered.

        Empty means the report is the declared experiment. Anything else means the run
        drifted from the plan — a different golden set, a different metric, different
        decode parameters — and the gate must not treat it as the registered test.
        """
        problems: list[str] = []
        if report.task and report.task != self.task:
            problems.append(f"task {report.task!r} != registered {self.task!r}")
        if report.golden_sha != self.golden_sha:
            problems.append(
                f"golden_sha {report.golden_sha[:12] or '<unknown>'} != registered "
                f"{self.golden_sha[:12]}"
            )
        expected_metric = _metric_name(self.metric)
        if report.metric != expected_metric:
            problems.append(f"metric {report.metric!r} != registered {expected_metric!r}")
        if report.config_fingerprint != self.generation.fingerprint:
            problems.append(
                f"decode config {report.config_fingerprint or '<unknown>'} != registered "
                f"{self.generation.fingerprint}"
            )
        return tuple(problems)

    def as_proof(self) -> dict[str, object]:
        """The pre-registration block of a ``promotion_proof``."""
        return {
            "prereg_path": str(self.path),
            "prereg_sha": self.sha,
            "alpha": self.alpha,
            "min_effect": self.min_effect,
            "min_n": self.min_n,
            "test": self.test,
            "must_beat_baselines": list(self.must_beat_baselines),
            "hypothesis": self.hypothesis,
        }


def load_prereg(path: Path | str) -> PreRegistration:
    """Parse and validate a pre-registration YAML file.

    Raises :class:`PreRegError` on anything malformed — an unreadable or half-written
    prereg is a failed gate, never a warning.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PreRegError(f"cannot read pre-registration {str(path)!r}: {exc}") from None
    try:
        obj = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise PreRegError(f"invalid YAML in {str(path)!r}: {exc}") from None
    if not isinstance(obj, dict):
        raise PreRegError(f"pre-registration must be a YAML mapping: {str(path)!r}")

    missing = [key for key in _REQUIRED if not obj.get(key)]
    if missing:
        raise PreRegError(
            f"pre-registration {str(path)!r} is missing required field(s): "
            + ", ".join(missing)
        )

    bar = obj.get("bar") or {}
    if not isinstance(bar, dict):
        raise PreRegError("'bar' must be a mapping (alpha, min_effect, min_n, test)")
    generation = obj.get("generation") or {}
    if not isinstance(generation, dict):
        raise PreRegError("'generation' must be a mapping (temperature, max_tokens, ...)")

    config = EvalConfig(
        temperature=float(generation.get("temperature", 0.0)),
        max_tokens=int(generation.get("max_tokens", 64)),
        seed=generation.get("seed"),
        system_hash=str(generation.get("system_hash", "")),
    )
    if not config.deterministic:
        raise PreRegError(
            "pre-registered generation.temperature must be 0.0: a re-rollable score is "
            "not a measurement (LEARNING_plan F4)"
        )

    baselines = bar.get("must_beat_baselines", list(DEFAULT_BASELINES))
    if isinstance(baselines, str):
        baselines = [baselines]
    # Beating the trivial baselines is part of the gate (CLAUDE.md §7), not an option a
    # prereg may switch off: a list may ADD baselines but never drop the defaults. An empty
    # list used to be accepted and silently removed the baseline check from the promotion.
    dropped = [b for b in DEFAULT_BASELINES if b not in [str(x) for x in baselines]]
    if dropped:
        raise PreRegError(
            f"bar.must_beat_baselines must include every default baseline; missing {dropped}. "
            "Beating the empty/majority/copy-input baselines is part of the gate, not optional."
        )

    # The written-down claim IS the pre-registration. `hearth prereg init` leaves these
    # blank on purpose; an unedited template used to load cleanly and, once committed, could
    # gate a promotion — a bar the tool wrote, not one the operator registered.
    blank = [k for k in ("hypothesis", "stopping_rule", "kill_condition")
             if not str(obj.get(k) or "").strip()]
    if blank:
        raise PreRegError(
            f"pre-registration has blank {blank}: write the hypothesis, when you will stop, "
            "and what result would kill the idea BEFORE training — an unedited template is "
            "not a pre-registration"
        )

    return PreRegistration(
        path=path,
        sha=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        task=str(obj["task"]),
        golden_sha=str(obj["golden_sha"]),
        golden_version=str(obj.get("golden_version", "")),
        metric=str(obj["metric"]),
        must_beat_baselines=tuple(str(b) for b in baselines),
        generation=config,
        hypothesis=str(obj.get("hypothesis", "")).strip(),
        raw=obj,
        **_checked_bar(bar),
    )


def _checked_bar(bar: dict) -> dict[str, object]:
    """Range-check the registered bar; raise :class:`PreRegError` on anything looser (B-062).

    YAML happily yields ``.nan``, ``1.0`` and ``-1`` here, and each used to load: NaN made
    every gate comparison False (so nothing refused), a negative ``min_effect`` made the
    baseline clause vacuous, ``alpha: 1.0`` / ``min_n: 1`` switched off significance and the
    size floor — a 0.033 candidate passed against a 1.0 incumbent. A pre-registration
    records the operator's bar; it may make the gate STRICTER than CLAUDE.md §7, never
    looser:

    * ``alpha`` in (0, 0.05] — §7's floors are derived at 0.05 (see ``eval.MAX_ALPHA``);
    * ``min_effect`` finite and >= 0;
    * ``min_n`` an integer >= ``DEFAULT_MIN_N`` (30). The gate's default IS the power
      floor; a prereg that could lower it would turn the floor into a per-experiment opt-out
      — and the operator writes the prereg after seeing how big the golden set is. Raise it,
      never lower it (the mathematical n>=5 floor stays available to library callers of
      ``evaluate_gate``, which cannot reach a promotion without a prereg);
    * ``test`` one of ``auto`` / ``mcnemar`` / ``bootstrap``.

    No coercion: ``float("nan")`` and ``int(30.5)`` are exactly how a bad value used to slip
    through, so the YAML value must already be the right kind of number.
    """
    alpha = bar.get("alpha", DEFAULT_ALPHA)
    min_effect = bar.get("min_effect", 0.0)
    min_n = bar.get("min_n", DEFAULT_MIN_N)
    test = bar.get("test", "auto")
    try:
        check_bar(alpha=alpha, margin=min_effect, min_n=min_n, test=test,
                  min_n_floor=DEFAULT_MIN_N)
    except ValueError as exc:
        raise PreRegError(f"pre-registered bar refused: {exc}") from None
    return {
        "alpha": float(alpha),
        "min_effect": float(min_effect),
        "min_n": int(min_n),
        "test": str(test),
    }


def verify_committed(path: Path | str) -> GitStatus:
    """Is ``path`` tracked by git and identical to its committed content?

    The two failure modes that matter are both covered: an untracked file (the bar was
    never registered) and a tracked-but-edited file (the bar was moved after the fact).
    Anything that prevents an answer — no git, no repository, a git error — is reported as
    *not committed*, because the gate must fail closed.
    """
    path = Path(path)
    if not path.exists():
        return GitStatus(committed=False, reason=f"file does not exist: {path}")
    directory = str(path.resolve().parent)

    try:
        root = _git(["rev-parse", "--show-toplevel"], cwd=directory)
    except FileNotFoundError:
        return GitStatus(committed=False, reason="git executable not found")
    except _GitError as exc:
        return GitStatus(committed=False, reason=f"not inside a git repository ({exc})")

    rel = str(path.resolve())
    try:
        _git(["ls-files", "--error-unmatch", "--", rel], cwd=root)
    except _GitError:
        return GitStatus(
            committed=False,
            reason=f"{path} is not tracked by git — commit the pre-registration first",
            repo_root=root,
        )
    try:
        _git(["diff", "--quiet", "HEAD", "--", rel], cwd=root)
    except _GitError:
        return GitStatus(
            committed=False,
            reason=(
                f"{path} has uncommitted modifications — the registered bar must match "
                "the committed one"
            ),
            repo_root=root,
        )
    try:
        last = _git(["log", "-1", "--format=%H %cI", "--", rel], cwd=root)
        introduced = _git(
            ["log", "--diff-filter=A", "--format=%H", "--", rel], cwd=root
        ).splitlines()
    except _GitError as exc:  # pragma: no cover - the file is tracked and unmodified
        return GitStatus(
            committed=False, reason=f"cannot read the file's history ({exc})", repo_root=root
        )
    commit, _, committed_at = last.partition(" ")
    if not commit or not committed_at:  # pragma: no cover - tracked implies a commit
        return GitStatus(
            committed=False, reason=f"{path} has no commit history", repo_root=root
        )
    return GitStatus(
        committed=True,
        reason="committed and unmodified",
        commit=commit,
        repo_root=root,
        committed_at=committed_at,
        introduced_commit=introduced[-1] if introduced else commit,
    )


def check_provenance(
    registration: PreRegistration, *, measured_at: str, golden_git: dict
) -> GitStatus:
    """Require the prereg to predate the measurement, in the repo that versions the golden set.

    Raises :class:`PreRegError`; returns the prereg's :class:`GitStatus` on success. Three
    things must hold, each one an outcome rather than a configuration (B-061):

    1. **Committed and unmodified** (:func:`verify_committed`).
    2. **Committed before the measurement.** The commit that last changed the file must be
       no later than ``measured_at`` (the time the eval started, recorded in the — signed —
       report). A bar committed seconds *after* the score was seen is the exact failure
       pre-registration exists to prevent. Committer timestamps are second-granular and
       set by whoever commits, so this stops the honest-but-post-hoc case and a casual
       forger, not one who deliberately backdates a commit.
    3. **In the repository that versions the golden set**, and the golden set itself
       committed and unmodified there at measurement time (``golden_git``, recorded by
       ``hearth eval``). Any-repo-will-do let a prereg be committed into a throwaway repo
       created for the purpose; requiring the golden set's own repository puts the bar and
       the evidence in one history an auditor can read, and makes "commit a bar somewhere"
       insufficient.
    """
    status = verify_committed(registration.path)
    if not status.committed:
        raise PreRegError(f"pre-registration is not git-committed: {status.reason}")
    committed = _parse_time(status.committed_at, "prereg commit time")
    measured = _parse_time(measured_at, "measurement time")
    if committed > measured:
        raise PreRegError(
            f"pre-registration was committed at {status.committed_at} (commit "
            f"{status.commit[:12]}), AFTER the measurement started at {measured_at}: the bar "
            "must be registered before the score is seen — re-run `hearth eval` now that it "
            "is committed"
        )
    if not golden_git.get("committed"):
        raise PreRegError(
            "the golden set was not committed and unmodified in git when it was measured "
            f"({golden_git.get('reason') or 'no git status recorded'}): commit it next to "
            "the pre-registration and re-run `hearth eval`"
        )
    golden_root = str(golden_git.get("repo_root") or "")
    if Path(golden_root).resolve() != Path(status.repo_root).resolve():
        raise PreRegError(
            f"the pre-registration lives in {status.repo_root}, but the golden set is "
            f"versioned in {golden_root}: register the bar in the repository that holds the "
            "golden set"
        )
    return status


def provenance_proof(status: GitStatus, golden_git: dict) -> dict[str, object]:
    """The git-provenance block of a ``promotion_proof`` (B-061).

    ``prereg_commit`` is the commit that last changed the prereg (the bar in force), with
    its timestamp — not HEAD, which says nothing about when the bar was written.
    """
    return {
        "prereg_committed": True,
        "prereg_commit": status.commit,
        "prereg_committed_at": status.committed_at,
        "prereg_introduced_commit": status.introduced_commit,
        "prereg_repo": status.repo_root,
        "golden_commit": str(golden_git.get("commit") or ""),
    }


def golden_git_status(path: Path | str) -> dict[str, object]:
    """The git status of a golden-set file, as recorded in an eval report."""
    status = verify_committed(path)
    return {
        "committed": status.committed,
        "reason": status.reason,
        "repo_root": status.repo_root,
        "commit": status.commit,
    }


def _parse_time(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        raise PreRegError(f"{label} {value!r} is not an ISO-8601 timestamp") from None
    if parsed.tzinfo is None:
        raise PreRegError(f"{label} {value!r} has no timezone; cannot order it")
    return parsed


def require_prereg(path: Path | str, report: EvalReport) -> PreRegistration:
    """Load ``path``, require it committed, and require ``report`` to match it.

    The single call a promotion path makes. Raises :class:`PreRegError` with the specific
    failure; returns the pre-registration when the run is the registered experiment.
    """
    prereg = load_prereg(path)
    status = verify_committed(prereg.path)
    if not status.committed:
        raise PreRegError(f"pre-registration is not git-committed: {status.reason}")
    problems = prereg.mismatches(report)
    if problems:
        raise PreRegError(
            "eval report does not match the pre-registration: " + "; ".join(problems)
        )
    return prereg


def template(
    *,
    task: str,
    golden_sha: str,
    golden_version: str = "",
    n: int = 0,
    metric: str = "exact",
    max_tokens: int = 64,
    system: str | None = None,
    alpha: float = DEFAULT_ALPHA,
    min_effect: float = 0.0,
    min_n: int = DEFAULT_MIN_N,
) -> str:
    """Render a pre-registration YAML skeleton for the operator to fill in and commit.

    The prose fields are left empty on purpose: a hypothesis and a stopping rule written
    by the tool are not a pre-registration, they are decoration.
    """
    config = EvalConfig.for_system(system, max_tokens=max_tokens)
    payload = {
        "task": task,
        "hypothesis": "",
        "golden_sha": golden_sha,
        "golden_version": golden_version,
        "n": n,
        "metric": metric,
        "generation": {
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "seed": config.seed,
            "system_hash": config.system_hash,
        },
        "bar": {
            "test": "auto",
            "alpha": alpha,
            "min_effect": min_effect,
            "min_n": min_n,
            "must_beat_baselines": list(DEFAULT_BASELINES),
        },
        "tie_rule": "a tie fails",
        "stopping_rule": "",
        "kill_condition": "",
    }
    header = (
        "# HEARTH pre-registration (docs/LEARNING_plan.md §3.4).\n"
        "# Declare the bar BEFORE training, then `git commit` this file. `hearth eval\n"
        "# --promote` refuses unless this file is committed and the run matches it.\n"
    )
    return header + yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)


class _GitError(RuntimeError):
    """A git invocation returned non-zero."""


def _git(args: list[str], *, cwd: str) -> str:
    """Run ``git <args>`` in ``cwd`` and return stripped stdout; raise on failure."""
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise _GitError((proc.stderr or proc.stdout).strip() or f"git {args[0]} failed")
    return proc.stdout.strip()


def _metric_name(metric: str) -> str:
    """Map a pre-registered metric name to the name an :class:`EvalReport` records."""
    return {
        "exact": "exact_match",
        "f1": "token_f1",
        "judge": "judge_win_rate",
    }.get(metric, metric)


__all__ = [
    "DEFAULT_BASELINES",
    "GitStatus",
    "PreRegError",
    "PreRegistration",
    "check_provenance",
    "golden_git_status",
    "provenance_proof",
    "load_prereg",
    "require_prereg",
    "template",
    "verify_committed",
]
