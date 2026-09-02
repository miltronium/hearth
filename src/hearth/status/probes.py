"""The probes — six read-only measurements of what is true on this machine.

Each ``probe_*`` returns one :class:`~hearth.status.report.Section` and never raises: a
subject that cannot be measured yields ``unverified`` facts, because a status tool that
dies on a broken machine is useless exactly when it is needed. Each section also carries
``limits`` — the things it did **not** measure — so silence is never read as a pass.

Where a probe needs to know what HEARTH itself would do (the resolved routing policy, the
model registry), it calls HEARTH's own loaders rather than re-parsing the YAML, so the
report describes the *outcome the running system gets* and not a second opinion. Those
imports are deferred and guarded: a sibling package mid-refactor degrades one section to
``unverified`` instead of taking the whole report down.
"""

from __future__ import annotations

import json
import os
import platform
import re
import stat
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from . import gitmeta
from .report import LEVEL_FAIL, LEVEL_OK, LEVEL_UNVERIFIED, LEVEL_WARN, Fact, Section

# ---------------------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------------------

_WEIGHT_SUFFIXES = frozenset({".safetensors", ".bin", ".gguf", ".npz"})


def _env(environ: dict[str, str] | None) -> dict[str, str]:
    """The environment to probe (injectable so tests never depend on the operator's shell)."""
    return dict(os.environ) if environ is None else dict(environ)


def _home_dir(environ: dict[str, str], home: Path | None) -> Path:
    """``~/.hearth`` — or ``HEARTH_HOME``, resolved the same way :mod:`hearth.config` does."""
    if home is not None:
        return Path(home)
    override = environ.get("HEARTH_HOME")
    return Path(override).expanduser() if override else Path.home() / ".hearth"


def _human_bytes(n: int) -> str:
    """Round bytes to the largest sensible **binary** unit, labelled honestly as GiB/MiB.

    Vendors quote decimal GB and ``mx.device_info()`` returns raw bytes, so a report that
    prints "GB" while dividing by 1024 invents a discrepancy of its own — the exact species
    of quiet wrongness this package exists to catch. Where the decimal figure is the one an
    operator will recognise, the caller prints both (see :func:`probe_environment`).
    """
    step = 1024.0
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < step or unit == "TiB":
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= step
    return f"{size:.1f} TiB"


def _decimal_gb(n: int) -> str:
    """Bytes as decimal GB — the unit Apple and the docs quote."""
    return f"{n / 1_000_000_000:.2f} GB"


def _read_text(path: Path) -> str | None:
    """File contents, or ``None`` if unreadable. Never raises — probes must not die on I/O."""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _mtime_iso(path: Path) -> str | None:
    """Modification time as an ISO-8601 UTC string, or ``None``."""
    try:
        ts = path.stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(ts, tz=UTC).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------------------
# 1. models on disk
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Weights:
    """One Hugging Face repo's footprint in one cache directory, measured by stat()."""

    repo: str
    path: Path
    total_bytes: int
    weight_files: int
    incomplete_files: int

    @property
    def present(self) -> bool:
        """True only when a *resolvable* weight file exists — not merely a directory."""
        return self.weight_files > 0


def hub_cache_dir(environ: dict[str, str]) -> tuple[Path, str]:
    """The hub cache ``huggingface_hub`` will actually use, and how it got resolved.

    Precedence mirrors the library's own: ``HF_HUB_CACHE`` > ``HF_HOME/hub`` >
    ``~/.cache/huggingface/hub``. This is the location the MLX provider loads from, which
    is *not* the location ``hearth models pull`` writes to — see :func:`probe_models`.
    """
    override = environ.get("HF_HUB_CACHE")
    if override:
        return Path(override).expanduser(), "HF_HUB_CACHE"
    hf_home = environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub", "HF_HOME/hub"
    return Path.home() / ".cache" / "huggingface" / "hub", "default (no HF_HUB_CACHE/HF_HOME)"


def scan_cache(cache_dir: Path) -> dict[str, Weights]:
    """Measure every ``models--org--name`` repo under a hub-layout cache directory.

    Sizes use ``lstat`` so the blob is counted once and its snapshot symlink costs nothing;
    presence uses ``stat`` so a *dangling* symlink counts as absent. That distinction is the
    whole point: a directory named after a model is a configuration, a resolvable weight
    file is an outcome.
    """
    found: dict[str, Weights] = {}
    if not cache_dir.is_dir():
        return found
    try:
        entries = sorted(cache_dir.iterdir())
    except OSError:
        return found
    for entry in entries:
        if not entry.name.startswith("models--") or not entry.is_dir():
            continue
        repo = entry.name[len("models--") :].replace("--", "/")
        total = weights = incomplete = 0
        try:
            children = list(entry.rglob("*"))
        except OSError:
            children = []
        for child in children:
            try:
                st = child.lstat()
            except OSError:
                continue
            if not (stat.S_ISREG(st.st_mode) or stat.S_ISLNK(st.st_mode)):
                continue
            total += st.st_size
            if child.name.endswith(".incomplete"):
                incomplete += 1
            elif child.suffix in _WEIGHT_SUFFIXES and child.exists():
                weights += 1
        found[repo] = Weights(repo, entry, total, weights, incomplete)
    return found


def _registry_entries(root: Path) -> list[tuple[str, str]] | None:
    """``(model_id, source_repo)`` from HEARTH's own registry loader, or ``None`` on failure."""
    try:
        from ..registry import load_registry

        registry = load_registry(root / "config" / "models.yaml")
        return [(e.id, e.source) for e in registry.list()]
    except Exception:  # noqa: BLE001 — a broken registry degrades the section, not the run
        return None


def probe_models(
    *,
    root: Path,
    home: Path | None = None,
    environ: dict[str, str] | None = None,
) -> Section:
    """Which model weights are physically on disk, where, and whether HEARTH can load them.

    HEARTH has **two** cache locations and they are not the same one:

      * ``~/.cache/huggingface/hub`` — the hub default, and what ``mlx_lm.load()`` reads;
      * ``~/.hearth/models``          — where ``hearth models pull`` writes (it passes
        ``cache_dir=settings.models_dir`` to ``snapshot_download``).

    So a model can be fully downloaded and still be invisible to the provider unless
    ``HF_HUB_CACHE`` points at ``~/.hearth/models``. "I pulled it" is a configuration;
    "the loader can find it" is the outcome, and only the second one serves a request.
    """
    env = _env(environ)
    hearth_home = _home_dir(env, home)
    models_dir = hearth_home / "models"
    hub_dir, hub_origin = hub_cache_dir(env)
    same_dir = hub_dir.resolve() == models_dir.resolve() if models_dir.exists() else False

    hub = scan_cache(hub_dir)
    pulled = scan_cache(models_dir)

    facts: list[Fact] = [
        Fact(
            "hub_cache",
            str(hub_dir),
            LEVEL_OK,
            f"resolved from {hub_origin}; this is what the MLX provider loads from",
            {"path": str(hub_dir), "origin": hub_origin, "repos": len(hub)},
        ),
        Fact(
            "hearth_models_dir",
            str(models_dir),
            LEVEL_OK,
            "where `hearth models pull` writes"
            + (" — same directory as the hub cache" if same_dir else ""),
            {"path": str(models_dir), "repos": len(pulled), "is_hub_cache": same_dir},
        ),
    ]

    entries = _registry_entries(root)
    if entries is None:
        facts.append(
            Fact(
                "registry",
                "unreadable",
                LEVEL_UNVERIFIED,
                "config/models.yaml could not be loaded; registry cross-check skipped",
            )
        )
        registered_sources: set[str] = set()
    else:
        registered_sources = {src for _, src in entries if src}
        for model_id, source in entries:
            if not source:
                continue  # the echo pseudo-model has nothing to download
            in_hub = hub.get(source)
            in_pulled = pulled.get(source)
            # The provider resolves HEARTH's own models_dir FIRST
            # (providers/mlx.py:resolve_local_model) and only then falls through to
            # huggingface_hub's HF_HUB_CACHE -> HF_HOME -> default chain. So weights in
            # either directory are genuinely loadable; this probe must agree with that
            # rule, or the status tool becomes another thing that reports a posture the
            # code does not actually have.
            in_pulled_ok = bool(in_pulled and in_pulled.present)
            loadable = bool(in_hub and in_hub.present) or in_pulled_ok
            if loadable:
                from_hearth = in_pulled_ok and not (in_hub and in_hub.present)
                w = in_pulled if from_hearth else in_hub
                level, value = LEVEL_OK, f"on disk, loadable ({_human_bytes(w.total_bytes)})"
                detail = f"{w.path}" + (
                    " (resolved from HEARTH's models_dir, not the hub cache)"
                    if from_hearth and not same_dir
                    else ""
                )
            else:
                level, value = LEVEL_WARN, "registered, NO weights on disk"
                detail = f"neither {hub_dir} nor {models_dir} holds resolvable weights"
            facts.append(
                Fact(
                    model_id,
                    value,
                    level,
                    detail,
                    {
                        "source": source,
                        "in_hub_cache": bool(in_hub and in_hub.present),
                        "in_hearth_models": bool(in_pulled and in_pulled.present),
                        "loadable_by_provider": loadable,
                        "bytes": (in_hub or in_pulled).total_bytes if (in_hub or in_pulled) else 0,
                    },
                )
            )

    for repo, w in sorted({**pulled, **hub}.items()):
        if repo in registered_sources or not w.present:
            continue
        where = "hub cache" if repo in hub and hub[repo].present else "~/.hearth/models"
        facts.append(
            Fact(
                repo,
                f"on disk but UNREGISTERED ({_human_bytes(w.total_bytes)})",
                LEVEL_WARN,
                f"weights in the {where} name no entry in config/models.yaml — "
                "nothing can serve them and nothing will garbage-collect them",
                {"path": str(w.path), "bytes": w.total_bytes, "registered": False},
            )
        )

    partial = [w for w in {**pulled, **hub}.values() if w.incomplete_files]
    if partial:
        facts.append(
            Fact(
                "partial_downloads",
                f"{len(partial)} repo(s) hold *.incomplete blobs",
                LEVEL_WARN,
                ", ".join(sorted(w.repo for w in partial)) + " — an interrupted pull",
                {"repos": sorted(w.repo for w in partial)},
            )
        )

    return Section(
        key="models",
        title="Models — weights actually on disk",
        facts=tuple(facts),
        limits=(
            "Presence is a stat() of a resolvable weight file. It does NOT prove the file "
            "is complete, uncorrupted, or that the model loads — only a real load does.",
            "Repo ids are recovered from the `models--org--name` directory convention; a "
            "repo name containing a literal '--' would be reported wrong.",
        ),
    )


# ---------------------------------------------------------------------------------------
# 2. egress posture
# ---------------------------------------------------------------------------------------


def _policy_outcome(path: Path) -> tuple[object | None, dict]:
    """Load a routing profile through HEARTH's own loader and diff it against the file.

    ``load_policy`` deliberately never raises: an invalid file falls back to safe built-in
    defaults. That is right for the server and dangerous for a status report — the fallback
    is itself no-egress, so a *broken* profile would read as a *good* profile. So the second
    return value diffs what the file declares against what the router resolved, and any
    disagreement is reported as a discrepancy rather than laundered into a green line.
    """
    try:
        from ..router.policy import load_policy

        policy = load_policy(path)
    except Exception:  # noqa: BLE001 — an unloadable router degrades this profile only
        return None, {"error": "policy loader unavailable"}

    raw_text = _read_text(path)
    if raw_text is None:
        return policy, {"error": "file unreadable"}
    try:
        raw = yaml.safe_load(raw_text) or {}
    except yaml.YAMLError as exc:
        return policy, {"error": f"invalid YAML: {exc.__class__.__name__}"}
    if not isinstance(raw, dict):
        return policy, {"error": "top level is not a mapping"}

    drift: list[str] = []
    declared_remotes = set((raw.get("remotes") or {}).keys())
    if declared_remotes != set(policy.remotes.keys()):
        drift.append(f"remotes declared {sorted(declared_remotes)} != resolved "
                     f"{sorted(policy.remotes.keys())}")
    for name, spec in (raw.get("classes") or {}).items():
        spec = spec or {}
        rule = policy.classes.get(name)
        if rule is None:
            drift.append(f"class {name} declared but not resolved")
            continue
        for key, default in (("backend", "local"), ("escalate", "never")):
            declared = str(spec.get(key, default))
            if declared != getattr(rule, key):
                drift.append(f"class {name}.{key}: file says {declared!r}, "
                             f"router resolved {getattr(rule, key)!r}")
    return policy, {"drift": drift}


def probe_egress(*, root: Path, environ: dict[str, str] | None = None) -> Section:
    """What each routing profile *structurally permits*, measured on the resolved policy.

    A profile is no-egress only when the loaded policy has zero remotes, resolves no default
    remote, and every class is ``backend: local`` with ``escalate: never``. That is read off
    the object the router will actually consult — not off the YAML, and not off a comment in
    the YAML claiming the file is private.

    This is the **router's** guarantee and nothing more. It is not machine-level containment:
    see ``limits``.
    """
    env = _env(environ)
    config_dir = root / "config"
    profiles = sorted(config_dir.glob("routing*.yaml")) if config_dir.is_dir() else []

    active = env.get("HEARTH_ROUTING_YAML")
    active_path = Path(active).expanduser() if active else config_dir / "routing.yaml"

    facts: list[Fact] = [
        Fact(
            "active_profile",
            str(active_path),
            LEVEL_OK if active_path.exists() else LEVEL_WARN,
            ("selected by HEARTH_ROUTING_YAML" if active else "the built-in default path")
            + ("" if active_path.exists() else " — but that file does not exist"),
            {"path": str(active_path), "from_env": bool(active), "exists": active_path.exists()},
        )
    ]

    if not profiles:
        facts.append(
            Fact("profiles", "none found", LEVEL_UNVERIFIED, f"no routing*.yaml under {config_dir}")
        )

    no_egress_profiles: list[str] = []
    for path in profiles:
        policy, meta = _policy_outcome(path)
        rel = path.relative_to(root).as_posix()
        if policy is None:
            facts.append(Fact(rel, "unmeasured", LEVEL_UNVERIFIED, str(meta.get("error", ""))))
            continue
        remote_classes = sorted(n for n, r in policy.classes.items() if r.backend == "remote")
        escalating = sorted(n for n, r in policy.classes.items() if r.escalate != "never")
        default_remote = policy.remote_for()
        no_egress = (
            not policy.remotes
            and not remote_classes
            and not escalating
            and default_remote is None
        )
        if no_egress:
            value = "NO EGRESS: 0 remotes, every class local/never"
            no_egress_profiles.append(rel)
        else:
            reasons = []
            if policy.remotes:
                reasons.append(f"{len(policy.remotes)} remote(s): {sorted(policy.remotes)}")
            if remote_classes:
                reasons.append(f"backend=remote for {remote_classes}")
            if escalating:
                reasons.append(f"escalates for {escalating}")
            value = "egress permitted — " + "; ".join(reasons)
        drift = meta.get("drift") or []
        error = meta.get("error")
        level = LEVEL_OK
        detail = ""
        if error:
            level, detail = LEVEL_WARN, f"file problem: {error}; router fell back to defaults"
        elif drift:
            level = LEVEL_WARN
            detail = "the router did NOT honour this file: " + "; ".join(drift)
        facts.append(
            Fact(
                rel,
                value,
                level,
                detail,
                {
                    "no_egress": no_egress,
                    "remotes": sorted(policy.remotes),
                    "remote_classes": remote_classes,
                    "escalating_classes": escalating,
                    "default_remote_resolves": default_remote is not None,
                    "budget_tokens_per_day": policy.defaults.remote_budget_tokens_per_day,
                    "drift": drift,
                },
            )
        )

    facts.append(
        Fact(
            "no_egress_profile_available",
            f"{len(no_egress_profiles)} profile(s)" if no_egress_profiles else "NONE",
            LEVEL_OK if no_egress_profiles else LEVEL_FAIL,
            ", ".join(no_egress_profiles)
            or "no profile in config/ resolves to a zero-remote policy — sealed mode has "
            "nothing to select",
            {"profiles": no_egress_profiles},
        )
    )

    offline = {k: env.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_ENDPOINT")}
    facts.append(
        Fact(
            "download_egress",
            "offline pinned" if offline["HF_HUB_OFFLINE"] else "downloads allowed",
            LEVEL_OK,
            "HF_HUB_OFFLINE="
            f"{offline['HF_HUB_OFFLINE'] or 'unset'}, "
            f"TRANSFORMERS_OFFLINE={offline['TRANSFORMERS_OFFLINE'] or 'unset'} — the router "
            "is not the only thing that can reach the network; a model load can too",
            {k: v for k, v in offline.items() if v is not None},
        )
    )

    return Section(
        key="egress",
        title="Egress posture — what the routing policy structurally permits",
        facts=tuple(facts),
        limits=(
            "This is the ROUTER'S guarantee only. It is NOT machine-level containment: "
            "nothing here inspects a firewall, a socket, or a process.",
            "A no-egress profile does not stop `providers/remote.py` being called directly, "
            "a non-loopback bind, a model download, telemetry from another process in the "
            "same terminal, or the calling agent leaking what it was shown.",
            "Whether the SERVER was started with the profile reported as active is "
            "unverified — this reads the environment of the status command, not of the daemon.",
        ),
    )


# ---------------------------------------------------------------------------------------
# 3. learning state
# ---------------------------------------------------------------------------------------

# alpha the promotion gate defaults to (hearth.training.eval.DEFAULT_ALPHA).
DEFAULT_ALPHA = 0.05


def smallest_achievable_p(n: int) -> float:
    """Best (smallest) one-sided exact McNemar p-value reachable on ``n`` paired items.

    The optimum is a clean sweep — candidate right on every item, incumbent wrong on every
    item — giving ``b = n``, ``c = 0`` and ``p = 0.5**n``. Mirrors
    ``hearth.training.stats.smallest_achievable_p``; ``tests/test_status_learning.py``
    asserts the two agree whenever that module is importable, so the report can never drift
    from the gate it is describing.
    """
    return 1.0 if n < 1 else 0.5**n


def min_n_for_alpha(alpha: float = DEFAULT_ALPHA) -> int:
    """Smallest golden-set size that could *ever* clear ``alpha``. At 0.05 this is 5.

    Solves ``0.5**n <= alpha``. Below it, no candidate — however much better — can produce
    a significant result, because the smallest p-value the test can emit is already larger
    than the bar. A golden set under this size cannot gate anything, and no amount of GPU
    time will change that.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    n = 1
    while 0.5**n > alpha:
        n += 1
    return n


def _jsonl_stats(path: Path) -> tuple[int, int]:
    """``(rows, malformed)`` for a JSONL file — blank lines ignored."""
    text = _read_text(path)
    if text is None:
        return 0, 0
    rows = malformed = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        rows += 1
        try:
            json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
    return rows, malformed


def _adapters(home: Path) -> list[dict] | None:
    """Entries from ``~/.hearth/adapters.json``, or ``None`` when it cannot be read.

    Read as plain JSON rather than through :class:`~hearth.registry.AdapterStore`: the store
    resolves settings and is a mutation API, and this package must not be one import away
    from a writer.
    """
    text = _read_text(home / "adapters.json")
    if text is None:
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    entries = obj.get("adapters") if isinstance(obj, dict) else None
    return entries if isinstance(entries, list) else None


def probe_learning(
    *,
    root: Path,
    home: Path | None = None,
    environ: dict[str, str] | None = None,
    alpha: float = DEFAULT_ALPHA,
) -> Section:
    """Golden sets, training corpora, adapters — and whether any of it can gate a promotion.

    The headline measurement is the **minimum detectable effect**. The promotion gate is a
    one-sided exact McNemar test on discordant pairs, so on ``n`` paired items the smallest
    p-value it can possibly emit is ``0.5**n``. At ``alpha = 0.05`` that means:

      * ``n < 5``  — no promotion can *ever* clear the bar. The golden set cannot gate.
      * ``n <= 10`` — it clears only if at least 5 of the ``n`` items flip in the
        candidate's favour with none flipping back, i.e. half the set or more.

    A golden set too small to ever gate is the most expensive thing in this report to
    discover late: it silently converts every training run into an unfalsifiable claim.
    """
    env = _env(environ)
    hearth_home = _home_dir(env, home)
    data_dir = root / "data"
    facts: list[Fact] = []

    required = min_n_for_alpha(alpha)
    golden = sorted(data_dir.glob("*_golden.jsonl")) if data_dir.is_dir() else []
    if not golden:
        facts.append(
            Fact(
                "golden_sets",
                "none found",
                LEVEL_UNVERIFIED,
                f"no *_golden.jsonl under {data_dir} — nothing can be gated",
            )
        )
    for path in golden:
        n, malformed = _jsonl_stats(path)
        p_best = smallest_achievable_p(n)
        can_gate = p_best <= alpha
        flip_fraction = (required / n) if n else 1.0
        if not can_gate:
            level = LEVEL_FAIL
            value = f"n={n} — CANNOT EVER GATE (best possible p={p_best:.4f} > alpha={alpha:g})"
            detail = (
                f"needs n >= {required} before any promotion is licensable at alpha={alpha:g}; "
                "every promotion decision made on this set is unfalsifiable"
            )
        elif flip_fraction >= 0.5:
            level = LEVEL_WARN
            value = f"n={n} — gates only in the near-degenerate case (best p={p_best:.4f})"
            detail = (
                f"clearing alpha={alpha:g} needs >= {required} discordant pairs won by the "
                f"candidate with none lost — {required}/{n} = {flip_fraction:.0%} of the set"
            )
        else:
            level = LEVEL_OK
            value = f"n={n} — can gate (best p={p_best:.4f}, needs >= {required} net wins)"
            detail = f"{required}/{n} = {flip_fraction:.0%} of items must flip to the candidate"
        if malformed:
            level = LEVEL_WARN
            detail += f"; {malformed} malformed JSON line(s)"
        facts.append(
            Fact(
                path.relative_to(root).as_posix(),
                value,
                level,
                detail,
                {
                    "n": n,
                    "malformed": malformed,
                    "alpha": alpha,
                    "smallest_achievable_p": p_best,
                    "can_ever_gate": can_gate,
                    "min_discordant_pairs_required": required,
                },
            )
        )

    corpora = (
        sorted(p for p in data_dir.glob("*.jsonl") if not p.name.endswith("_golden.jsonl"))
        if data_dir.is_dir()
        else []
    )
    for path in corpora:
        n, malformed = _jsonl_stats(path)
        facts.append(
            Fact(
                path.relative_to(root).as_posix(),
                f"{n} training rows",
                LEVEL_WARN if malformed else LEVEL_OK,
                f"{malformed} malformed line(s)" if malformed else "training corpus",
                {"rows": n, "malformed": malformed},
            )
        )

    entries = _adapters(hearth_home)
    if entries is None:
        facts.append(
            Fact(
                "adapters",
                "no registry",
                LEVEL_UNVERIFIED,
                f"{hearth_home / 'adapters.json'} missing or unreadable — nothing trained here, "
                "or state lives elsewhere",
            )
        )
    else:
        by_status: dict[str, int] = {}
        for entry in entries:
            status_name = str(entry.get("status", "?"))
            by_status[status_name] = by_status.get(status_name, 0) + 1
        facts.append(
            Fact(
                "adapters",
                ", ".join(f"{k}={v}" for k, v in sorted(by_status.items())) or "none registered",
                LEVEL_OK,
                str(hearth_home / "adapters.json"),
                {"counts": by_status, "total": len(entries)},
            )
        )
        for entry in entries:
            adapter_id = str(entry.get("id", "?"))
            status = str(entry.get("status", "?"))
            adapter_path = Path(str(entry.get("adapter_path", "")))
            on_disk = adapter_path.is_dir() if str(adapter_path) else False
            proof = entry.get("promotion_proof") or {}
            has_significance = isinstance(proof, dict) and "p_value" in proof
            problems = []
            level = LEVEL_OK
            if not on_disk:
                problems.append(f"adapter_path missing on disk ({adapter_path})")
                level = LEVEL_WARN
            if status == "promoted" and not has_significance:
                problems.append(
                    "promoted WITHOUT a significance proof (no p_value in promotion_proof) — "
                    "gated on a bare score comparison, which cannot distinguish a real "
                    "improvement from noise on a small golden set"
                )
                level = LEVEL_WARN
            facts.append(
                Fact(
                    f"adapter:{adapter_id}",
                    f"{status}, task={entry.get('task', '?')}"
                    + (", weights present" if on_disk else ", WEIGHTS ABSENT"),
                    level,
                    "; ".join(problems) or f"{adapter_path}",
                    {
                        "status": status,
                        "task": entry.get("task"),
                        "base_model": entry.get("base_model"),
                        "adapter_path": str(adapter_path),
                        "weights_present": on_disk,
                        "promotion_proof_keys": sorted(proof) if isinstance(proof, dict) else [],
                        "has_significance_proof": has_significance,
                    },
                )
            )

    return Section(
        key="learning",
        title="Learning state — golden sets, corpora, adapters, and what they can license",
        facts=tuple(facts),
        limits=(
            "Row counts are lines in a file. They do NOT measure label quality, duplication, "
            "leakage between the corpus and the golden set, or whether the golden set still "
            "matches the task being trained.",
            "The minimum detectable effect is an upper bound on what the test could show at "
            "best. Real discordance is almost always far below the clean-sweep optimum, so "
            "the practical requirement is larger than the number reported here.",
            "Adapter quality is unverified: this reports that weights exist and what proof "
            "was recorded, not that the adapter is any good.",
        ),
    )


# ---------------------------------------------------------------------------------------
# 4. test suite
# ---------------------------------------------------------------------------------------

_TEST_DEF = re.compile(r"^\s*(?:async\s+)?def\s+(test_\w+)", re.MULTILINE)


def probe_tests(*, root: Path) -> Section:
    """How many tests exist, and what the last recorded run said — never a fresh run.

    Running the suite from a status command would be a mutation (it writes caches, and here
    it would race three other agents' edits), so this counts declared test functions by
    reading source, and reports pytest's own on-disk cache as what it is: the residue of
    *some* earlier run at a known timestamp. Whether the suite passes **now** is reported as
    unverified, because that is the truth.
    """
    tests_dir = root / "tests"
    facts: list[Fact] = []
    if not tests_dir.is_dir():
        return Section(
            key="tests",
            title="Test suite",
            facts=(Fact("tests_dir", "absent", LEVEL_UNVERIFIED, str(tests_dir)),),
        )

    files = sorted(tests_dir.glob("test_*.py"))
    declared = 0
    for path in files:
        text = _read_text(path)
        if text:
            declared += len(_TEST_DEF.findall(text))
    facts.append(
        Fact(
            "declared",
            f"{declared} test functions in {len(files)} files",
            LEVEL_OK,
            "counted by parsing source; parametrised cases expand to more at runtime",
            {"files": len(files), "test_functions": declared},
        )
    )

    cache = root / ".pytest_cache" / "v" / "cache"
    nodeids_path, lastfailed_path = cache / "nodeids", cache / "lastfailed"
    if nodeids_path.exists() or lastfailed_path.exists():
        collected = failed = None
        text = _read_text(nodeids_path)
        if text:
            try:
                collected = len(json.loads(text))
            except (json.JSONDecodeError, TypeError):
                collected = None
        text = _read_text(lastfailed_path)
        if text:
            try:
                failed = len(json.loads(text))
            except (json.JSONDecodeError, TypeError):
                failed = None
        when = _mtime_iso(lastfailed_path) or _mtime_iso(nodeids_path)
        facts.append(
            Fact(
                "last_recorded_run",
                f"{collected if collected is not None else '?'} collected, "
                f"{failed if failed is not None else '?'} failing",
                LEVEL_WARN if failed else LEVEL_OK,
                f"from .pytest_cache, written {when} — a previous run on this machine, "
                "not necessarily of the current working tree",
                {"collected": collected, "failed": failed, "cache_mtime": when},
            )
        )
    facts.append(
        Fact(
            "passing_now",
            "unmeasured",
            LEVEL_UNVERIFIED,
            "this command never runs the suite (it would mutate caches and race concurrent "
            "edits); run `uv run pytest -q` for the only answer that counts",
        )
    )
    return Section(
        key="tests",
        title="Test suite — declared, and the last run on record",
        facts=tuple(facts),
        limits=(
            "A declared test count says nothing about coverage, and nothing about whether "
            "the assertions are load-bearing.",
            "The pytest cache is a cache: it can be stale, partial, or from a different "
            "checkout state entirely.",
        ),
    )


# ---------------------------------------------------------------------------------------
# 5. environment
# ---------------------------------------------------------------------------------------

# HEARTH_* names read directly from the environment rather than through Settings — the
# repo's own escape hatches. Anything set that is neither a Settings field nor in here is
# read by nothing and will be silently ignored (the HEARTH_MODEL/HEARTH_DEFAULT_MODEL trap).
_EXTRA_ENV_NAMES = frozenset(
    {
        "HEARTH_ROUTING_YAML",
        "HEARTH_MODELS_YAML",
        "HEARTH_BASE_MODEL",
        "HEARTH_TRAIN_DATA",
        "HEARTH_TRAIN_ITERS",
        "HEARTH_TRAIN_OUT",
        "HEARTH_TRAIN_TASK",
        "HEARTH_CANDIDATE_SCORE",
        "HEARTH_INCUMBENT_SCORE",
    }
)

_UPPER_BOUND = re.compile(r"^([A-Za-z0-9_.\-]+).*?<\s*([0-9][0-9.]*)")


def _installed_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    """Installed distribution versions, ``None`` where the package is absent."""
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def _version_tuple(v: str) -> tuple[int, ...]:
    """Numeric release segments of a version string (``"4.57.6rc1"`` -> ``(4, 57, 6)``)."""
    parts: list[int] = []
    for chunk in v.split("."):
        m = re.match(r"^(\d+)", chunk)
        if not m:
            break
        parts.append(int(m.group(1)))
    return tuple(parts)


def _upper_bound_violations(root: Path, installed: dict[str, str | None]) -> list[str]:
    """Declared ``<`` ceilings in pyproject that the installed version already exceeds.

    Only ``<`` bounds are checked, and only against packages already looked up. These are
    the pins that exist because a newer release *breaks* something (mlx-lm vs transformers
    5, coremltools vs torch 2.8), so exceeding one is a live incompatibility, not a nag.
    """
    text = _read_text(root / "pyproject.toml")
    if text is None:
        return []
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return []
    project = data.get("project", {})
    specs: list[str] = list(project.get("dependencies", []) or [])
    for extra in (project.get("optional-dependencies", {}) or {}).values():
        specs.extend(extra or [])

    violations: list[str] = []
    for spec in specs:
        m = _UPPER_BOUND.match(spec.strip())
        if not m:
            continue
        name, ceiling = m.group(1), m.group(2)
        have = installed.get(name)
        if not have:
            continue
        if _version_tuple(have) >= _version_tuple(ceiling):
            violations.append(f"{name} {have} >= declared ceiling <{ceiling}")
    return sorted(set(violations))


def _mlx_device_info() -> dict | None:
    """``mx.device_info()``, or ``None`` when MLX is absent or the call fails.

    Imported lazily and defensively: MLX is an optional extra, and the whole point of this
    field is that the number it returns disagrees with the number the machine advertises.
    """
    try:
        import mlx.core as mx

        info = mx.device_info()
    except Exception:  # noqa: BLE001 — no MLX, no Metal device, or an API change
        return None
    return dict(info) if isinstance(info, dict) else None


def probe_environment(*, environ: dict[str, str] | None = None) -> Section:
    """Chip, memory, the GPU working-set ceiling, library versions, and ignored env vars.

    The load-bearing number is ``max_recommended_working_set_size`` from
    ``mx.device_info()``: the driver's own ceiling on resident GPU memory, which is
    materially *lower* than the RAM the machine advertises. Sizing a model against the
    advertised figure is the same class of mistake as trusting a config — the outcome is
    what the driver will actually let you hold.
    """
    env = _env(environ)
    facts: list[Fact] = [
        Fact(
            "platform",
            f"{platform.system()} {platform.release()} / {platform.machine()}",
            LEVEL_OK if platform.machine() == "arm64" else LEVEL_WARN,
            "Apple Silicon" if platform.machine() == "arm64" else "not arm64 — MLX will not run",
            {"machine": platform.machine(), "python": platform.python_version()},
        )
    ]

    try:
        total_ram = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        total_ram = 0
    if total_ram:
        facts.append(
            Fact(
                "unified_memory",
                f"{_decimal_gb(total_ram)} ({_human_bytes(total_ram)})",
                LEVEL_OK,
                "total system memory as reported by sysconf — the advertised figure, which "
                "the GPU working-set ceiling below is materially lower than",
                {"bytes": total_ram},
            )
        )

    info = _mlx_device_info()
    if info is None:
        facts.append(
            Fact(
                "gpu_working_set",
                "unverified",
                LEVEL_UNVERIFIED,
                "mlx is not importable here, so the driver's working-set ceiling cannot be "
                "measured; do NOT size models against advertised RAM instead",
            )
        )
    else:
        ws = int(info.get("max_recommended_working_set_size", 0))
        mem = int(info.get("memory_size", 0)) or total_ram
        headroom = (ws / mem) if mem else 0.0
        facts.append(
            Fact(
                "gpu_working_set",
                f"{_decimal_gb(ws)} ({_human_bytes(ws)})",
                LEVEL_OK,
                f"{info.get('device_name', '?')} — the driver's recommended ceiling on "
                f"resident GPU memory: {headroom:.0%} of the {_decimal_gb(mem)} "
                f"({_human_bytes(mem)}) the machine advertises. Size models against THIS, "
                "not against advertised RAM",
                {
                    "device_name": info.get("device_name"),
                    "max_recommended_working_set_size": ws,
                    "memory_size": mem,
                    "fraction_of_advertised": round(headroom, 4),
                    "architecture": info.get("architecture"),
                },
            )
        )

    packages = ("mlx", "mlx-lm", "transformers", "huggingface-hub", "torch", "coremltools", "mcp")
    installed = _installed_versions(packages)
    facts.append(
        Fact(
            "versions",
            ", ".join(f"{k}={v}" for k, v in installed.items() if v) or "none of the extras",
            LEVEL_OK,
            "absent: " + (", ".join(k for k, v in installed.items() if not v) or "none"),
            {k: v for k, v in installed.items()},
        )
    )

    root = Path(__file__).resolve().parents[3]
    violations = _upper_bound_violations(root, installed)
    facts.append(
        Fact(
            "version_ceilings",
            "violated" if violations else "respected",
            LEVEL_WARN if violations else LEVEL_OK,
            "; ".join(violations)
            or "every installed package is below the '<' ceiling pyproject declares for it",
            {"violations": violations},
        )
    )

    try:
        from ..config import Settings

        known = {f"HEARTH_{name.upper()}" for name in Settings.model_fields} | _EXTRA_ENV_NAMES
    except Exception:  # noqa: BLE001 — settings unreadable: say so, don't guess
        known = None
    set_vars = sorted(k for k in env if k.startswith("HEARTH_"))
    if known is None:
        facts.append(
            Fact(
                "hearth_env",
                f"{len(set_vars)} set",
                LEVEL_UNVERIFIED,
                "hearth.config.Settings could not be imported, so it is unknown which of "
                f"{set_vars} are actually read",
                {"set": set_vars},
            )
        )
    else:
        ignored = [k for k in set_vars if k not in known]
        facts.append(
            Fact(
                "hearth_env",
                ", ".join(set_vars) or "none set",
                LEVEL_WARN if ignored else LEVEL_OK,
                (
                    f"SILENTLY IGNORED (no code reads them): {ignored} — a value set here has "
                    "no effect, and any result attributed to it describes a different "
                    "configuration than the operator believes"
                )
                if ignored
                else "every HEARTH_* variable set here is one the code actually reads",
                {"set": set_vars, "ignored": ignored, "recognised": sorted(known)},
            )
        )

    return Section(
        key="environment",
        title="Environment — chip, memory ceiling, versions, effective env",
        facts=tuple(facts),
        limits=(
            "Version ceilings are read from pyproject's '<' bounds only. Lower bounds, "
            "markers, and transitive conflicts are not checked — `uv sync` is the authority.",
            "A recognised HEARTH_* variable is one some code reads; that it has the value "
            "the operator intended, or that a running daemon inherited it, is unverified.",
        ),
    )


# ---------------------------------------------------------------------------------------
# 6. documentation staleness
# ---------------------------------------------------------------------------------------

# The docs that carry institutional memory — the ones a new session reads first, and so
# the ones whose rot costs the most.
KEY_DOCS = (
    "docs/RESULTS.md",
    "docs/cmux/HANDOFF.md",
    "docs/cmux/TODO.md",
    "docs/TIERS.md",
    "docs/LEARNING_plan.md",
    "docs/MODELS_local.md",
    "docs/APEX_seam.md",
)

# The one judgement call in this report: how many commits may land on top of a doc before
# it is *probably* out of date. It flags nothing more than "go re-read this"; it is not a
# measurement of correctness, and it is named here so a reader can discount it knowingly.
STALE_COMMITS = 20


def probe_staleness(*, root: Path, docs: tuple[str, ...] = KEY_DOCS) -> Section:
    """How far each memory doc has fallen behind the repo, measured from git history.

    A doc's own text cannot tell you whether it is still true. What *can* be measured is how
    much work has landed since anyone touched it — so this reports, per doc, the last commit
    that changed it and how many commits have landed on HEAD since. That number is evidence
    a human can act on, in place of a "last updated" line the last editor forgot to change.
    """
    facts: list[Fact] = []
    if not gitmeta.is_repo(root):
        return Section(
            key="staleness",
            title="Documentation staleness",
            facts=(
                Fact(
                    "git",
                    "unavailable",
                    LEVEL_UNVERIFIED,
                    f"{root} is not a git worktree (or git is not installed); document age "
                    "cannot be measured",
                ),
            ),
        )

    head = gitmeta.head_commit(root)
    total = gitmeta.commit_count(root)
    facts.append(
        Fact(
            "head",
            f"{head[0]} on {gitmeta.branch(root) or 'DETACHED'} ({total} commits)"
            if head
            else "unknown",
            LEVEL_OK if head else LEVEL_UNVERIFIED,
            f"{head[1]} — {head[2]}" if head else "",
            {
                "sha": head[0] if head else None,
                "date": head[1] if head else None,
                "branch": gitmeta.branch(root),
                "commits": total,
            },
        )
    )

    for rel in docs:
        path = root / rel
        exists = path.exists()
        commit = gitmeta.last_commit(root, rel) if exists else None
        if not exists:
            facts.append(Fact(rel, "MISSING", LEVEL_WARN, "referenced as a key doc but absent"))
            continue
        if commit is None:
            facts.append(
                Fact(
                    rel,
                    "uncommitted",
                    LEVEL_UNVERIFIED,
                    "no commit has ever touched this path — it exists only in the working "
                    "tree, so its age and its review status are both unmeasurable",
                    {"tracked": gitmeta.is_tracked(root, rel), "committed": False},
                )
            )
            continue
        sha, date = commit
        since = gitmeta.commits_since(root, sha)
        dirty = gitmeta.worktree_dirty(root, rel)
        level = LEVEL_WARN if (since is not None and since >= STALE_COMMITS) else LEVEL_OK
        suffix = " (+uncommitted edits)" if dirty else ""
        facts.append(
            Fact(
                rel,
                f"{date[:10]} @ {sha}, {since if since is not None else '?'} commits since"
                + suffix,
                level,
                (
                    f"{since} commits have landed since this doc was last written — treat its "
                    "claims as unverified until re-checked"
                )
                if level == LEVEL_WARN
                else "recently touched",
                {
                    "last_commit": sha,
                    "last_commit_date": date,
                    "commits_since": since,
                    "worktree_dirty": dirty,
                    "stale_threshold": STALE_COMMITS,
                },
            )
        )

    return Section(
        key="staleness",
        title="Documentation staleness — how far each memory doc has fallen behind",
        facts=tuple(facts),
        limits=(
            f"Commit distance is a proxy, not a verdict. The {STALE_COMMITS}-commit threshold "
            "is the only judgement call in this report; a doc under it can still be wrong, "
            "and one over it can still be right.",
            "Nothing here reads the documents. Whether a specific claim inside one is still "
            "true is exactly what a human must check — this only says where to look first.",
        ),
    )


__all__ = [
    "DEFAULT_ALPHA",
    "KEY_DOCS",
    "STALE_COMMITS",
    "Weights",
    "hub_cache_dir",
    "min_n_for_alpha",
    "probe_egress",
    "probe_environment",
    "probe_learning",
    "probe_models",
    "probe_staleness",
    "probe_tests",
    "scan_cache",
    "smallest_achievable_p",
]
