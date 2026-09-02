"""The one place :mod:`hearth.status` leaves the process — read-only ``git`` plumbing.

Document staleness is a fact about *history*, and history exists nowhere except ``.git``.
Reimplementing packfile parsing to avoid a subprocess would trade a small, auditable
exec for a large, wrong parser, so this module shells out — under three constraints that
``tests/test_status_readonly.py`` enforces by reading this file:

  * it is the **only** module in the package that imports :mod:`subprocess`;
  * every invocation goes through :func:`git`, which refuses any subcommand outside
    :data:`READ_ONLY_SUBCOMMANDS` — so a future "just ``git add`` it while we're here"
    is a raised exception, not a mutation;
  * failures are swallowed into ``None``. A checkout without ``git``, or a directory that
    is not a repo, must degrade to "unverified", never to a traceback.

Nothing here writes to the index, the worktree, or the object store.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

# Plumbing that only reads. ``status``/``ls-files`` are included because "is this file even
# tracked?" and "is the worktree copy ahead of the last commit?" are both staleness facts.
READ_ONLY_SUBCOMMANDS = frozenset(
    {"log", "rev-list", "rev-parse", "status", "ls-files", "branch", "describe"}
)

_TIMEOUT_SECONDS = 10.0


class UnsafeGitCommand(ValueError):
    """Raised when a caller asks for a ``git`` subcommand that could mutate the repo."""


def git(args: list[str], *, cwd: Path, timeout: float = _TIMEOUT_SECONDS) -> str | None:
    """Run a read-only ``git`` command in ``cwd`` and return stdout, or ``None`` on failure.

    Raises :class:`UnsafeGitCommand` for anything outside :data:`READ_ONLY_SUBCOMMANDS` —
    that is a programming error in this package, not a runtime condition, so it is loud.
    """
    if not args or args[0] not in READ_ONLY_SUBCOMMANDS:
        raise UnsafeGitCommand(
            f"refusing git {args[0] if args else '<empty>'!r}: hearth.status is read-only "
            f"(allowed: {sorted(READ_ONLY_SUBCOMMANDS)})"
        )
    try:
        proc = subprocess.run(  # noqa: S603 — fixed argv, allowlisted read-only subcommand
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def is_repo(root: Path) -> bool:
    """True when ``root`` is inside a git worktree."""
    return git(["rev-parse", "--is-inside-work-tree"], cwd=root) == "true"


def head_commit(root: Path) -> tuple[str, str, str] | None:
    """``(short_sha, iso_date, subject)`` for HEAD, or ``None`` if unavailable."""
    out = git(["log", "-1", "--format=%h%x00%cI%x00%s"], cwd=root)
    if not out or "\x00" not in out:
        return None
    sha, date, subject = (out.split("\x00") + ["", "", ""])[:3]
    return sha, date, subject


def branch(root: Path) -> str | None:
    """Current branch name, or ``None`` (detached HEAD included)."""
    return git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=root) or None


def commit_count(root: Path) -> int | None:
    """Number of commits reachable from HEAD."""
    out = git(["rev-list", "--count", "HEAD"], cwd=root)
    return int(out) if out and out.isdigit() else None


def is_tracked(root: Path, relpath: str) -> bool:
    """True when ``relpath`` is in the index (i.e. git knows about it at all)."""
    return bool(git(["ls-files", "--", relpath], cwd=root))


def worktree_dirty(root: Path, relpath: str) -> bool | None:
    """True when ``relpath`` differs from HEAD in the index or worktree.

    ``None`` when it could not be determined. A dirty doc is *newer* than its last commit,
    so its commit date understates its freshness — the renderer says so rather than
    quietly reporting a stale-looking date.
    """
    out = git(["status", "--porcelain", "--", relpath], cwd=root)
    return None if out is None else bool(out)


def last_commit(root: Path, relpath: str) -> tuple[str, str] | None:
    """``(short_sha, iso_date)`` of the last commit touching ``relpath``, else ``None``."""
    out = git(["log", "-1", "--format=%h%x00%cI", "--", relpath], cwd=root)
    if not out or "\x00" not in out:
        return None
    sha, date = out.split("\x00", 1)
    return sha, date


def commits_since(root: Path, sha: str) -> int | None:
    """How many commits landed on HEAD after ``sha`` (0 = ``sha`` is HEAD)."""
    out = git(["rev-list", "--count", f"{sha}..HEAD"], cwd=root)
    return int(out) if out and out.isdigit() else None


__all__ = [
    "READ_ONLY_SUBCOMMANDS",
    "UnsafeGitCommand",
    "git",
    "is_repo",
    "head_commit",
    "branch",
    "commit_count",
    "is_tracked",
    "worktree_dirty",
    "last_commit",
    "commits_since",
]
