#!/usr/bin/env python3
"""cmux workspace tier classifier — implements ADR-C003 (config/cmux/tiers.yaml).

Resolves a repo/workspace to the SEALED (confidential) or OPEN (non-confidential) tier so the
`cmux-sealed` launcher can fail closed. This is cmux-integration tooling, deliberately kept OUT of
the HEARTH engine (ADR-C002/C005): HEARTH does not depend on cmux and vice versa.

INVARIANTS (see docs/cmux/DECISIONS.md ADR-C003; do not weaken):
  * default is SEALED — no matching rule ⇒ sealed.
  * OPEN requires an explicit `open:` match AND no `sealed_override:` match.
  * MOST-RESTRICTIVE-WINS — a sealed_override, no match, ambiguity, or ANY error ⇒ sealed.
  * UNKNOWN / unresolvable ⇒ sealed (fail safe, never fail open).

CLI:
  tier_classify.py <repo_path>                 # print "sealed" | "open"  (exit 0)
  tier_classify.py <repo_path> --require-sealed # exit 0 iff sealed, else 3  (launcher preflight)
  tier_classify.py <repo_path> --assert-open    # exit 0 iff open,   else 3  (guard before open tier)
  tier_classify.py <repo_path> --config PATH    # override tiers.yaml location

On ANY internal error the result is "sealed" — the safe default — and non-error exit codes reflect it.
"""

from __future__ import annotations

import fnmatch
import os
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except Exception:  # pragma: no cover - yaml is a project dep; absence ⇒ fail safe
    yaml = None

SEALED = "sealed"
OPEN = "open"


def default_config_path() -> Path:
    """`config/cmux/tiers.yaml` at the repo root (this file lives in scripts/cmux/)."""
    return Path(__file__).resolve().parents[2] / "config" / "cmux" / "tiers.yaml"


def load_config(path: Path | None = None) -> dict:
    """Load the tier policy. Missing/invalid ⇒ ``{}`` (which classifies everything sealed)."""
    p = path or default_config_path()
    if yaml is None or not p.exists():
        return {}
    try:
        data = yaml.safe_load(p.read_text()) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def normalize_remote(url: str | None) -> str | None:
    """Reduce a git remote URL to a ``host/org/repo`` string for matching. None if absent."""
    if not url:
        return None
    u = url.strip()
    if u.startswith("git@"):  # git@github.com:org/repo.git
        u = u[len("git@") :].replace(":", "/", 1)
    for pre in ("https://", "http://", "ssh://", "git://"):
        if u.startswith(pre):
            u = u[len(pre) :]
    if "@" in u.split("/", 1)[0]:  # strip user@host
        u = u.split("@", 1)[1]
    if u.endswith(".git"):
        u = u[: -len(".git")]
    return u or None


def repo_remote_host(repo_path: Path) -> str | None:
    """`git -C <path> remote get-url origin`, normalized. None on any failure (⇒ path-only match)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_path), "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return normalize_remote(out.stdout)
    except Exception:
        pass
    return None


def _expand(p: str) -> str:
    return os.path.expanduser(os.path.expandvars(p))


def _rule_matches(rule: dict, abs_path: str, remote: str | None, *, override: bool) -> bool:
    """True if an open/override rule matches. `path` glob (fnmatch), `remote_host`(open)/
    `remote_host_contains`(override). fnmatch `*`/`**` both match across separators."""
    if not isinstance(rule, dict):
        return False
    glob = rule.get("path")
    if glob and fnmatch.fnmatch(abs_path, _expand(str(glob))):
        return True
    if override:
        needle = rule.get("remote_host_contains")
        if needle and remote and str(needle).lower() in remote.lower():
            return True
    else:
        rhost = rule.get("remote_host")
        if rhost and remote and fnmatch.fnmatch(remote, str(rhost)):
            return True
    return False


def classify(repo_path: str | Path, config: dict, remote: str | None = None) -> str:
    """Resolve a repo to ``sealed``/``open`` per ADR-C003. Any ambiguity/error ⇒ ``sealed``."""
    try:
        abs_path = str(Path(_expand(str(repo_path))).resolve())
        overrides = config.get("sealed_override") or []
        opens = config.get("open") or []
        # sealed_override ALWAYS wins.
        if any(_rule_matches(r, abs_path, remote, override=True) for r in overrides):
            return SEALED
        # OPEN only by explicit opt-in.
        if any(_rule_matches(r, abs_path, remote, override=False) for r in opens):
            return OPEN
        # default; anything but an explicit 'open' default ⇒ sealed.
        return OPEN if str(config.get("default", SEALED)).lower() == OPEN else SEALED
    except Exception:
        return SEALED  # fail safe


def main(argv: list[str]) -> int:
    args = list(argv)
    config_path = None
    mode = "print"
    if "--config" in args:
        i = args.index("--config"); config_path = Path(args[i + 1]); del args[i : i + 2]
    if "--require-sealed" in args:
        mode = "require-sealed"; args.remove("--require-sealed")
    elif "--assert-open" in args:
        mode = "assert-open"; args.remove("--assert-open")
    if not args:
        print("usage: tier_classify.py <repo_path> [--require-sealed|--assert-open] [--config PATH]", file=sys.stderr)
        return 64
    repo = args[0]
    cfg = load_config(config_path)
    tier = classify(repo, cfg, remote=repo_remote_host(Path(_expand(repo))))
    if mode == "require-sealed":
        if tier != SEALED:
            print(f"NOT SEALED: {repo} classifies as '{tier}' — refusing sealed launch.", file=sys.stderr)
            return 3
        print(SEALED); return 0
    if mode == "assert-open":
        if tier != OPEN:
            print(f"NOT OPEN: {repo} classifies as '{tier}' (fail-closed to sealed).", file=sys.stderr)
            return 3
        print(OPEN); return 0
    print(tier); return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
