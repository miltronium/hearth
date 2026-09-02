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
)

# Baselines a pre-registration requires by default (LEARNING_plan §3.2.4).
DEFAULT_BASELINES = (BASELINE_EMPTY, BASELINE_MAJORITY, BASELINE_COPY_INPUT)

# Fields with no sensible default — a prereg that omits one is not a prereg.
_REQUIRED = ("task", "golden_sha", "metric")


class PreRegError(ValueError):
    """Raised on a malformed, mismatched, or uncommitted pre-registration."""


@dataclass(frozen=True)
class GitStatus:
    """Whether a pre-registration file is committed to git and unmodified since."""

    committed: bool
    reason: str
    commit: str = ""
    repo_root: str = ""


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

    return PreRegistration(
        path=path,
        sha=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        task=str(obj["task"]),
        golden_sha=str(obj["golden_sha"]),
        golden_version=str(obj.get("golden_version", "")),
        metric=str(obj["metric"]),
        alpha=float(bar.get("alpha", DEFAULT_ALPHA)),
        min_effect=float(bar.get("min_effect", 0.0)),
        min_n=int(bar.get("min_n", DEFAULT_MIN_N)),
        test=str(bar.get("test", "auto")),
        must_beat_baselines=tuple(str(b) for b in baselines),
        generation=config,
        hypothesis=str(obj.get("hypothesis", "")).strip(),
        raw=obj,
    )


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
        commit = _git(["rev-parse", "HEAD"], cwd=root)
    except _GitError as exc:  # pragma: no cover - HEAD exists if diff HEAD succeeded
        return GitStatus(committed=False, reason=f"cannot resolve HEAD ({exc})", repo_root=root)
    return GitStatus(
        committed=True,
        reason="committed and unmodified",
        commit=commit,
        repo_root=root,
    )


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
    "load_prereg",
    "require_prereg",
    "template",
    "verify_committed",
]
