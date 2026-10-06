"""Pre-registration for the promotion gate (LEARNING_plan §3.4).

The operator declares the bar **before** training: which golden set (by content sha), which
metric, which alpha, and the minimum effect that would count. The file is then committed to
git. ``promote`` refuses unless a matching pre-registration exists *and* is committed and
unmodified — so the bar provably predates the measurement and cannot be moved after seeing
the score.

This is the user's own APEX methodology, mechanized: the upgrade over discipline is that the
harness refuses to run without it, so the habit cannot erode under deadline pressure.

    evals/<task>/prereg-<YYYY-MM-DD>-<slug>.yaml

Git is consulted by shelling out rather than by parsing ``.git``, but never asked "is this
file modified?": that question is answered from the index, which ``--assume-unchanged``
can make lie (B-078). Instead the bytes on disk are hashed here and compared with the blob
HEAD records (``git rev-parse HEAD:<path>``) — a comparison a reviewer can repeat with
``git hash-object --no-filters``.
"""

from __future__ import annotations

import hashlib
import os
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


class _UniqueKeyLoader(yaml.SafeLoader):
    """``yaml.SafeLoader`` that refuses a mapping with a repeated key.

    PyYAML silently keeps the LAST of two equal keys, so ``min_n: 30`` followed further
    down by ``min_n: 5`` reads as 5 while a reviewer skimming the file sees 30 — the bar
    that is enforced is not the bar that is read. A duplicated key makes the file
    ambiguous, and an ambiguous bar is not a registered one.
    """

    def construct_mapping(self, node, deep=False):  # type: ignore[override]
        seen: set[object] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in seen
            except TypeError:  # unhashable key: let the base class report it
                break
            if duplicate:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping", node.start_mark,
                    f"found duplicate key {key!r} — a repeated key silently keeps only "
                    "the last value, so the file does not say one thing",
                    key_node.start_mark,
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


@dataclass(frozen=True)
class GitStatus:
    """Whether a file is committed to git and unmodified since.

    ``commit`` is the commit that last changed the file — the one whose content is the bar
    in force — and ``committed_at`` its committer timestamp (ISO 8601). It is deliberately
    NOT ``rev-parse HEAD``: HEAD moves with every unrelated commit, so recording it said
    nothing about when the bar was written (B-061). ``introduced_commit`` is the commit
    that first added the file.

    ``blob`` is the committed blob id the bytes on disk were compared against,
    ``rel_path`` the file's path inside the repository, and ``content_sha256`` the SHA-256
    of the exact bytes that were compared — so a caller that parsed the file can prove it
    parsed the bytes that were verified (B-078).
    """

    committed: bool
    reason: str
    commit: str = ""
    repo_root: str = ""
    committed_at: str = ""
    introduced_commit: str = ""
    blob: str = ""
    rel_path: str = ""
    content_sha256: str = ""


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
        return self.run_mismatches(task=report.task, golden_sha=report.golden_sha,
                                   metric=report.metric,
                                   config_fingerprint=report.config_fingerprint)

    def run_mismatches(
        self, *, task: str, golden_sha: str, metric: str, config_fingerprint: str
    ) -> tuple[str, ...]:
        """:meth:`mismatches` for a run that has not been scored yet (B-122).

        Everything the comparison needs is fixed before the first generation, so ``hearth
        eval --prereg`` checks it BEFORE recording the measurement: a run recorded as "made
        under" a prereg is then always the experiment that prereg registered.
        """
        problems: list[str] = []
        if task and task != self.task:
            problems.append(f"task {task!r} != registered {self.task!r}")
        if golden_sha != self.golden_sha:
            problems.append(
                f"golden_sha {golden_sha[:12] or '<unknown>'} != registered "
                f"{self.golden_sha[:12]}"
            )
        expected_metric = _metric_name(self.metric)
        if _metric_name(metric) != expected_metric:
            problems.append(
                f"metric {_metric_name(metric)!r} != registered {expected_metric!r}"
            )
        if config_fingerprint != self.generation.fingerprint:
            problems.append(
                f"decode config {config_fingerprint or '<unknown>'} != registered "
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
        data = path.read_bytes()
        text = data.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise PreRegError(f"cannot read pre-registration {str(path)!r}: {exc}") from None
    try:
        obj = yaml.load(text, Loader=_UniqueKeyLoader)  # noqa: S506 - a SafeLoader subclass
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
        sha=hashlib.sha256(data).hexdigest(),
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


def verify_committed(
    path: Path | str, *, data: bytes | None = None, role: str = "pre-registration"
) -> GitStatus:
    """Are ``path``'s bytes exactly the blob committed at HEAD? (B-061, B-078)

    Compares the bytes on disk (or ``data``, the bytes a caller already read and will use)
    with the blob HEAD records for the path: the blob id of ``data`` is computed here, in
    Python, and must equal ``git rev-parse HEAD:<path>``. It used to ask ``git diff --quiet
    HEAD``, which consults the INDEX: ``git update-index --assume-unchanged`` (or
    ``--skip-worktree``, or a clean filter) makes git report an edited file as unmodified,
    so an edited prereg or golden set passed as "committed and unmodified" — the check and
    the checked thing were different objects (CLAUDE.md §3). Hashing the bytes ourselves
    asks nothing of the index, attributes or filters; a file a filter legitimately
    rewrites on commit (LFS, eol conversion) is refused, which is the fail-closed side.

    An untracked file, a file absent from HEAD, edited bytes, no git, no repository, or
    any git error are all reported as *not committed*: the gate fails closed. ``role``
    names the file in the reasons ("golden set", "pre-registration"), so the operator is
    told which file to commit (B-128).
    """
    path = Path(path)
    if data is None:
        try:
            data = path.read_bytes()
        except OSError:
            return GitStatus(committed=False, reason=f"file does not exist: {path}")
    resolved = path.resolve()
    directory = str(resolved.parent)

    try:
        root = _git(["rev-parse", "--show-toplevel"], cwd=directory)
    except FileNotFoundError:
        return GitStatus(committed=False, reason="git executable not found")
    except _GitError as exc:
        return GitStatus(committed=False, reason=f"not inside a git repository ({exc})")

    try:
        rel = resolved.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return GitStatus(committed=False, reason=f"{path} is not inside {root}", repo_root=root)
    rewrites = _history_rewrites(root)
    if rewrites:
        return GitStatus(
            committed=False,
            reason=f"the repository {root} cannot vouch for when {path} was committed: "
            + "; ".join(rewrites),
            repo_root=root,
        )
    try:
        _git(["ls-files", "--error-unmatch", "--", rel], cwd=root)
    except _GitError:
        return GitStatus(
            committed=False,
            reason=f"{path} is not tracked by git — commit the {role} first",
            repo_root=root,
        )
    try:
        committed_blob = _git(["rev-parse", "--verify", "--quiet", f"HEAD:{rel}"], cwd=root)
    except _GitError:
        return GitStatus(
            committed=False,
            reason=f"{path} is staged but not in any commit — commit it first",
            repo_root=root,
        )
    if _blob_id(data, _object_format(root)) != committed_blob:
        return GitStatus(
            committed=False,
            reason=(
                f"{path} has uncommitted modifications — its bytes are not the committed "
                "blob (git's index is not consulted, so --assume-unchanged / "
                "--skip-worktree cannot hide an edit)"
            ),
            repo_root=root,
        )
    try:
        last = _git(["log", "-1", "--format=%H %cI", "--", rel], cwd=root)
        introduced = _git(
            ["log", "--diff-filter=A", "--format=%H", "--", rel], cwd=root
        ).splitlines()
    except _GitError as exc:  # pragma: no cover - the file is in HEAD
        return GitStatus(
            committed=False, reason=f"cannot read the file's history ({exc})", repo_root=root
        )
    commit, _, committed_at = last.partition(" ")
    if not commit or not committed_at:  # pragma: no cover - in HEAD implies a commit
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
        blob=committed_blob,
        rel_path=rel,
        content_sha256=hashlib.sha256(data).hexdigest(),
    )


def _object_format(root: str) -> str:
    """The repository's object hash (``sha1``, or ``sha256`` for a SHA-256 repository)."""
    try:
        fmt = _git(["rev-parse", "--show-object-format"], cwd=root)
    except _GitError:  # git too old to know the flag predates SHA-256 repositories
        return "sha1"
    return fmt if fmt in ("sha1", "sha256") else "sha1"


def _blob_id(data: bytes, fmt: str) -> str:
    """git's object id for ``data`` as a blob — computed here, never by asking git."""
    return hashlib.new(fmt, b"blob %d\x00" % len(data) + data).hexdigest()


def committed_golden_problems(golden_git: dict, *, task: str, golden_sha: str) -> list[str]:
    """Re-derive the golden set from the committed blob and compare it with the measurement.

    The report says which golden set was scored (``golden_sha``) and where it was committed
    (``golden_git``: repository, commit, path). This reads the blob out of git at that
    commit — not the working tree, not the report — parses it with the same parser the
    eval used, and requires its content sha to be the one measured and every prompt
    distinct (B-078, B-080). A golden set edited in the working tree and hidden from git
    is a different sha from the committed blob, so it cannot pass as the registered one.
    """
    from .eval import parse_golden_jsonl

    root = str(golden_git.get("repo_root") or "")
    commit = str(golden_git.get("commit") or "")
    rel = str(golden_git.get("rel_path") or "")
    if not (root and commit and rel):
        return ["the report does not say where the golden set was committed "
                "(repo_root/commit/rel_path): re-run `hearth eval` with this version"]
    try:
        text = _git(["cat-file", "blob", f"{commit}:{rel}"], cwd=root, strip=False)
    except (_GitError, FileNotFoundError, NotADirectoryError) as exc:
        return [f"cannot read the committed golden set {rel} at {commit[:12]} in {root}: {exc}"]
    try:
        committed = parse_golden_jsonl(text, task=task)
    except ValueError as exc:
        return [f"the committed golden set {rel} at {commit[:12]} does not parse: {exc}"]
    problems = []
    if committed.sha != golden_sha:
        problems.append(
            f"the golden set committed at {commit[:12]} ({rel}) hashes to "
            f"{committed.sha[:12]}, but the measurement scored {golden_sha[:12] or '<none>'}: "
            "what was scored is not what was committed"
        )
    if committed.duplicate_prompts():
        problems.append(f"the committed golden set {rel} repeats a prompt")
    return problems


def check_provenance(
    registration: PreRegistration,
    *,
    first_measurement: dict,
    golden_git: dict,
    golden_sha: str,
) -> GitStatus:
    """Require the prereg to predate the adapter's FIRST measurement, next to the golden set.

    ``first_measurement`` is the adapter's earliest record in the measurement ledger
    (:mod:`hearth.training.ledger`) — any golden set, with or without a prereg — and
    ``golden_git`` / ``golden_sha`` describe the measurement being promoted. Raises
    :class:`PreRegError`; returns the prereg's :class:`GitStatus` on success. Each check
    asserts an outcome rather than a configuration (B-061, B-078, B-079):

    1. **Committed and unmodified** (:func:`verify_committed`): the bytes on disk are the
       committed blob, and they are the bytes ``registration`` was parsed from.
    2. **Committed before the adapter was first measured.** It used to be "before THIS
       run", so an operator could measure with no prereg, read PASS, then commit a bar and
       re-measure (B-079). Two forms of "before", because each alone is weak:

       * the commit that last changed the prereg has a committer time no later than the
         first measurement's ``measured_at``;
       * that commit is an ancestor of the HEAD the golden set's repository was at when the
         first measurement started (recorded in the ledger then). Committer timestamps
         are second-granular and settable with ``GIT_COMMITTER_DATE``; a commit hash
         recorded at measurement time cannot contain a commit made afterwards.
    3. **In the repository that versions the golden set**, and the golden set itself
       committed and unmodified there (``golden_git``) — re-derived from the committed
       blob, whose content sha must be the ``golden_sha`` that was scored
       (:func:`committed_golden_problems`).
    4. **That repository is the anchored evals repository** recorded in the ledger at the
       first measurement (:func:`resolve_anchor`, B-081).
    5. **The first measurement was made under THIS prereg** (its ``prereg_sha``, B-122):
       a bar merely committed before the first measurement can be one of several, picked
       after the score was seen.
    """
    status = verify_committed(registration.path)
    if not status.committed:
        raise PreRegError(f"pre-registration is not git-committed: {status.reason}")
    if status.content_sha256 != registration.sha:
        raise PreRegError(
            f"{registration.path} changed between being read and being verified: the bar "
            "that was parsed is not the bar that is committed"
        )
    first_at = str(first_measurement.get("measured_at") or "")
    adapter = first_measurement.get("adapter_id")
    committed = _parse_time(status.committed_at, "prereg commit time")
    measured = _parse_time(first_at, "first measurement time")
    if committed > measured:
        raise PreRegError(
            f"pre-registration was committed at {status.committed_at} (commit "
            f"{status.commit[:12]}), AFTER the measurement started at {first_at} — the "
            f"first recorded measurement of {adapter!r} (ledger record "
            f"{first_measurement.get('seq')}). The bar must be registered before ANY score "
            "of the adapter is seen; an adapter measured before its bar existed cannot be "
            "promoted under it"
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
    # The repository is not "wherever the golden file happens to be": a copy of the golden
    # set in a throwaway `git init` satisfied that (B-081). It must be the evals repository
    # HEARTH was anchored to when the adapter was FIRST measured — a choice recorded in the
    # signed ledger before any score existed, so it cannot be made after seeing one.
    anchor = first_measurement.get("anchor") or {}
    anchor_dir = str(anchor.get("common_dir") or "")
    if not anchor_dir or _repo_identity(status.repo_root)[1] != anchor_dir:
        raise PreRegError(
            f"the pre-registration and golden set live in {status.repo_root}, which is not "
            f"the evals repository HEARTH was anchored to when {adapter!r} was first measured "
            f"({anchor.get('path') or 'none resolved'}): register the bar next to the golden "
            "set in the anchored repository (`hearth prereg anchor` shows it), before the "
            "adapter is first measured"
        )
    first_head = str((first_measurement.get("golden_git") or {}).get("head") or "")
    if not first_head or not _is_ancestor(status.commit, first_head, root=status.repo_root):
        raise PreRegError(
            f"pre-registration commit {status.commit[:12]} was not in the history of "
            f"{status.repo_root} when {adapter!r} was first measured at {first_at} (HEAD then: "
            f"{first_head[:12] or 'no commit'}): the bar was registered AFTER the measurement, "
            "or in another repository — an adapter measured before its bar existed cannot be "
            "promoted under it"
        )
    # The bar is the one the adapter was FIRST measured under — not merely one committed
    # before that measurement (B-122). Two bars committed up front (two golden sets, or one
    # strict and one loose) let the operator measure under one, see the verdict, and promote
    # under the other: the choice of bar was made after seeing a score. Each measurement
    # records the prereg it was made under (none for an exploratory run), so an adapter
    # first measured exploratory, or under another prereg, is not promotable under this one.
    first_prereg = str(first_measurement.get("prereg_sha") or "")
    if first_prereg != registration.sha:
        under = (f"under another pre-registration (sha {first_prereg[:12]}, "
                 f"{first_measurement.get('prereg_path') or 'path not recorded'})"
                 if first_prereg else "with no pre-registration (an exploratory run)")
        raise PreRegError(
            f"{adapter!r} was first measured at {first_at} (ledger record "
            f"{first_measurement.get('seq')}) {under}, not under {registration.path} (sha "
            f"{registration.sha[:12]}): an adapter is promotable only under the bar it was "
            "FIRST measured under, so that the bar cannot be picked after seeing a score — "
            "run the first `hearth eval` of an adapter with --prereg"
        )
    problems = committed_golden_problems(golden_git, task=registration.task,
                                         golden_sha=golden_sha)
    if problems:
        raise PreRegError("; ".join(problems))
    return status


def _is_ancestor(commit: str, head: str, *, root: str) -> bool:
    """Is ``commit`` reachable from ``head`` in the repository at ``root``? (False on error)."""
    try:
        _git(["merge-base", "--is-ancestor", commit, head], cwd=root)
    except (_GitError, FileNotFoundError, NotADirectoryError):
        return False
    return True


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


def golden_git_status(path: Path | str, *, data: bytes | None = None) -> dict[str, object]:
    """The git status of a golden-set file, as recorded in an eval report.

    Pass ``data`` — the bytes the eval actually parsed and scored — so the status is about
    those bytes, not a second read of the file that could differ.
    """
    status = verify_committed(path, data=data, role="golden set")
    head = ""
    if status.repo_root:
        try:  # the repository's HEAD at measurement time, committed file or not (B-079)
            head = _git(["rev-parse", "--verify", "--quiet", "HEAD"], cwd=status.repo_root)
        except (_GitError, FileNotFoundError):
            head = ""
    return {
        "committed": status.committed,
        "reason": status.reason,
        "repo_root": status.repo_root,
        "commit": status.commit,
        "rel_path": status.rel_path,
        "blob": status.blob,
        "head": head,
    }


# -- the evals-repository anchor (B-081) --------------------------------------------------

ANCHOR_FILENAME = "evals-repo"


def _repo_identity(directory: str | Path) -> tuple[str, str]:
    """``(toplevel, common_dir)`` of the repository containing ``directory`` ("" on failure).

    The identity is the resolved common git dir, so every worktree of one repository is the
    same repository, and a copy of its files in a fresh ``git init`` is not.
    """
    try:
        top = _git(["rev-parse", "--show-toplevel"], cwd=str(directory))
        common = _git(["rev-parse", "--git-common-dir"], cwd=str(directory))
    except (_GitError, FileNotFoundError, NotADirectoryError):
        return "", ""
    return top, str((Path(directory) / common).resolve())


def resolve_anchor(home: Path | str) -> dict[str, str]:
    """The evals repository promotions are anchored to, as of now.

    ``<HEARTH_HOME>/evals-repo`` (written by ``hearth prereg anchor``) names it; unset, it
    is HEARTH's own repository — the one holding the ``data/<task>_golden.jsonl`` sets the
    runbook registers bars next to. Returns ``{"path", "common_dir", "source"}``;
    ``common_dir`` is "" when the anchor does not resolve to a git repository.
    """
    pin = Path(home) / ANCHOR_FILENAME
    if pin.exists():
        target = pin.read_text(encoding="utf-8").strip()
        source = str(pin)
    else:
        target = str(Path(__file__).resolve().parent)
        source = "HEARTH's own repository (default)"
    top, common = _repo_identity(target) if Path(target).is_dir() else ("", "")
    return {"path": top or target, "common_dir": common, "source": source}


def pin_anchor(home: Path | str, repo: Path | str) -> dict[str, str]:
    """Anchor promotions to the git repository containing ``repo``; raise if it is none."""
    top, common = _repo_identity(repo) if Path(repo).is_dir() else ("", "")
    if not common:
        raise PreRegError(f"{repo} is not inside a git repository")
    Path(home).mkdir(parents=True, exist_ok=True)
    (Path(home) / ANCHOR_FILENAME).write_text(top + "\n", encoding="utf-8")
    return resolve_anchor(home)



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


# Every question the gate asks git is about content-addressed history: "is this blob in
# that commit", "is that commit an ancestor of this one". Content addressing is what makes
# the answers unforgeable — a commit id fixes its parents — so anything that lets git answer
# from somewhere OTHER than the objects themselves turns the gate back into a configuration
# check (CLAUDE.md §3, B-120). Three such mechanisms exist, and each is neutralised here:
#
# * ``refs/replace/*`` (``git replace``): git silently substitutes one object for another,
#   so a fabricated, backdated commit can be grafted under the HEAD recorded at a first
#   measurement. Off via ``GIT_NO_REPLACE_OBJECTS=1`` (config can only further disable
#   replace refs, never re-enable them over the environment variable).
# * ``info/grafts`` / ``GIT_GRAFT_FILE``: rewrites a commit's parents by fiat. git has no
#   switch to ignore a grafts file, so the gate refuses a repository that has one
#   (:func:`_history_rewrites`) and strips the env var.
# * the commit-graph cache (``objects/info/commit-graph``): parents, trees and dates are
#   read from it instead of from the commit objects, and its checksum is not verified on
#   read — a hand-edited graph makes ``git log`` report a parent the commit does not have.
#   Off via ``core.commitGraph=false`` (slower history walks; the gate walks one file).
#
# ``-c`` on the command line outranks every config file, so repository-local config cannot
# turn these back on. System and global config are skipped outright (``GIT_CONFIG_NOSYSTEM``
# / ``GIT_CONFIG_GLOBAL``): the gate's reads must not depend on ambient per-user settings.
# Repository-local config cannot be skipped (git offers no switch for it); the keys in it
# that change how objects or parents are read are exactly the ones overridden above, and
# the rest (aliases, hooks, fsmonitor) do not run for these plumbing reads. Every other
# ``GIT_*`` variable is dropped too — ``GIT_DIR``, ``GIT_OBJECT_DIRECTORY``,
# ``GIT_ALTERNATE_OBJECT_DIRECTORIES``, ``GIT_REPLACE_REF_BASE``, ``GIT_INDEX_FILE`` all
# redirect what is read. ``protocol.allow=never`` / ``GIT_NO_LAZY_FETCH`` keep a
# partial clone from fetching a missing object over the network mid-check.
_GIT_OVERRIDES = (
    "-c", "core.commitGraph=false",
    "-c", "protocol.allow=never",
)


def _git_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update({
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_TERMINAL_PROMPT": "0",
    })
    return env


def _history_rewrites(root: str) -> list[str]:
    """Every mechanism in the repository at ``root`` that rewrites history by fiat (B-120).

    Replace refs are already ignored by :func:`_git`; they are refused as well because a
    repository that carries them is one where a human reviewer running plain ``git log``
    sees a different history from the one the gate checked. A grafts file cannot be ignored
    at all, so it must be refused.
    """
    problems = []
    try:
        refs = _git(["for-each-ref", "--format=%(refname)", "refs/replace/"], cwd=root)
        common = _git(["rev-parse", "--git-common-dir"], cwd=root)
    except (_GitError, FileNotFoundError, NotADirectoryError) as exc:
        return [f"cannot inspect the repository for history rewrites ({exc})"]
    if refs:
        problems.append(
            f"it carries replace refs ({refs.splitlines()[0]}…): `git replace` substitutes "
            "one commit for another, so its history is not the committed history — "
            "remove them (`git replace -d`)"
        )
    if (Path(root) / common / "info" / "grafts").exists():
        problems.append(
            "it has an info/grafts file, which rewrites commit parents by fiat — remove it"
        )
    return problems


def _git(args: list[str], *, cwd: str, strip: bool = True) -> str:
    """Run ``git <args>`` in ``cwd`` and return its stdout (stripped); raise on failure.

    Always with history rewriting neutralised (see ``_GIT_OVERRIDES``).
    """
    proc = subprocess.run(
        ["git", *_GIT_OVERRIDES, *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env=_git_env(),
    )
    if proc.returncode != 0:
        raise _GitError((proc.stderr or proc.stdout).strip() or f"git {args[0]} failed")
    return proc.stdout.strip() if strip else proc.stdout


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
    "ANCHOR_FILENAME",
    "check_provenance",
    "committed_golden_problems",
    "pin_anchor",
    "resolve_anchor",
    "golden_git_status",
    "provenance_proof",
    "load_prereg",
    "require_prereg",
    "template",
    "verify_committed",
]
