"""Environment preflight — ``hearth doctor`` and ``hearth doctor --offline``.

``hearth doctor`` checks the things that make or break local inference on this machine:
Apple Silicon, usable RAM, whether the MLX backend is installed, and whether the state dir
is writable. ``hearth doctor --offline`` (:func:`run_offline_checks`) answers "is it safe to
use HEARTH offline right now?" by measuring outcomes. Both return structured results so the
CLI can render them and set an exit code.
"""

from __future__ import annotations

import platform
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import Settings, ensure_home, get_settings
from .providers.mlx import mlx_available


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    # A failed check may still be non-fatal (e.g. MLX missing -> echo fallback works).
    fatal: bool = False


def _total_ram_gb() -> float | None:
    """Best-effort total RAM in GB, or None if it can't be determined."""
    try:
        import os

        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round(pages * page_size / (1024**3), 1)
    except (ValueError, OSError, AttributeError):
        return None


def run_checks(settings: Settings | None = None) -> list[Check]:
    """Run all environment checks and return their results."""
    settings = settings or get_settings()
    checks: list[Check] = []

    # Apple Silicon
    machine = platform.machine()
    is_arm = machine == "arm64"
    checks.append(
        Check(
            "apple_silicon",
            is_arm,
            f"machine={machine} ({'Apple Silicon' if is_arm else 'not arm64'})",
            fatal=False,
        )
    )

    # RAM (32 GB+ is the documented baseline)
    ram = _total_ram_gb()
    ram_ok = ram is not None and ram >= 16
    checks.append(
        Check(
            "memory",
            ram_ok,
            f"{ram} GB total" + ("" if ram is None else " (baseline 32 GB, min 16 GB)"),
        )
    )

    # MLX backend availability (non-fatal: echo fallback exists)
    has_mlx = mlx_available()
    checks.append(
        Check(
            "mlx_backend",
            has_mlx,
            "mlx-lm importable"
            if has_mlx
            else "mlx-lm not installed (echo fallback active; "
            "`uv sync --extra mlx --extra mcp --extra dev --extra files`)",
            fatal=False,
        )
    )

    # The backend the commands would actually build (B-059): construct it, as they do,
    # rather than pattern-matching the setting. An unknown HEARTH_BACKEND makes every
    # model-using command refuse to start, so it is fatal here, not a footnote.
    from .providers import UnknownBackendError, select_provider

    try:
        provider = select_provider(settings)
        checks.append(Check(
            "backend", True, f"HEARTH_BACKEND={settings.backend} -> {provider.name}", fatal=True
        ))
    except UnknownBackendError as exc:
        checks.append(Check("backend", False, str(exc), fatal=True))

    # State dir writable
    try:
        ensure_home(settings)
        writable = shutil.disk_usage(settings.home).free > 0
        detail = f"{settings.home} writable"
    except OSError as exc:
        writable = False
        detail = f"{settings.home}: {exc}"
    checks.append(Check("state_dir", writable, detail, fatal=True))

    default_model = check_default_model()
    if default_model is not None:
        checks.append(default_model)

    return checks


def check_default_model(registry=None, *, fatal: bool = True) -> Check | None:
    """FAIL when ``HEARTH_DEFAULT_MODEL`` is set but unregistered (B-029, B-047); else ``None``.

    The commands that serve start through ``Registry.require_default`` and refuse such an
    override (exit 2), so this asks the registry the same question (``require_default``)
    and reports the same answer. ``fatal`` is the caller's verdict: plain ``doctor`` asks
    "will HEARTH run?" — it will not, so fatal; ``doctor --offline`` asks "is it safe
    offline?" — a refusal to start is not a path off the machine, so it passes
    ``fatal=False`` and the row is shown (WARN) without changing the safety verdict.
    """
    from .registry import UnregisteredDefaultModelError

    if registry is None:
        from .registry import load_registry

        try:
            registry = load_registry()
        except Exception as exc:  # noqa: BLE001 — an unreadable catalog is reported, not raised
            return Check("default_model", False, f"model registry unreadable: {exc}")
    try:
        registry.require_default()
    except UnregisteredDefaultModelError as exc:
        return Check(
            "default_model",
            False,
            f"{exc} `hearth serve`/`run`/`agent`/`mcp` refuse to start until it is fixed.",
            fatal=fatal,
        )
    return None


def all_fatal_passed(checks: list[Check]) -> bool:
    """True if no fatal check failed (non-fatal failures are tolerated)."""
    return all(c.ok for c in checks if c.fatal)


# ---------------------------------------------------------------------------------------
# hearth doctor --offline: "is it safe to use HEARTH offline right now?"
# ---------------------------------------------------------------------------------------

#: An id that cannot be on disk. Resolving it must fail without a single connect attempt.
_ABSENT_MODEL = "hearth-doctor/no-such-model"
_OFFLINE_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
_BUILTIN_BACKENDS = ("auto", "mlx", "echo")

#: What ``--offline`` does not measure — printed with every report so silence is not a pass.
OFFLINE_LIMITS = (
    "Measured in THIS command's environment. A running `hearth serve` may have been started "
    "with a different HEARTH_ROUTING_YAML / HEARTH_HOST / HEARTH_ALLOW_DOWNLOADS, or --host.",
    "Covers HEARTH's router and loaders only — not machine-level containment, other "
    "processes, or a client that pins a model id per request. To prove no packet leaves, run "
    "under a deny-egress sandbox (docs/PRIVACY.md, 'Verifying no egress yourself').",
)


def _is_loopback(host: str) -> bool:
    """A literal loopback address, or ``localhost``. Never resolves a name (no DNS)."""
    import ipaddress

    host = host.strip().strip("[]")
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _has_weights(path: Path) -> bool:
    """True when ``path`` is (or directly holds) a resolvable weight file."""
    from .status.probes import _WEIGHT_SUFFIXES

    if path.is_file():
        return path.suffix in _WEIGHT_SUFFIXES
    try:
        return any(c.suffix in _WEIGHT_SUFFIXES and c.exists() for c in path.iterdir())
    except OSError:
        return False


def _check_routing(policy_path: Path | None, environ: dict[str, str] | None = None):
    """The profile the router would load NOW, judged on the policy it resolves to.

    The path comes from the router's own resolver (``resolve_routing_selection``), so ``~``
    and repo-root-relative values name the same file here as in ``hearth serve``. A
    ``HEARTH_ROUTING_YAML`` that names a missing file fails this check: the router refuses
    to start on it (B-008), so reporting the safe-default fallback as SAFE would describe a
    server that cannot run.
    """
    from .router.policy import resolve_routing_selection
    from .status.probes import _policy_outcome, policy_posture

    if policy_path is not None:
        path = Path(policy_path)
    else:
        selection = resolve_routing_selection(environ)
        path = selection.path
        if selection.explicit and not path.is_file():
            return None, Check(
                "routing_profile",
                False,
                f"HEARTH_ROUTING_YAML={selection.raw!r} selects {path}, which does not exist "
                "— the router refuses to start on it (relative paths resolve against the "
                "repo root)",
                fatal=True,
            )
    policy, meta = _policy_outcome(path)
    if policy is None:
        return None, Check("routing_profile", False, f"{path}: {meta.get('error')}", fatal=True)
    posture = policy_posture(policy)
    if posture["no_egress"]:
        detail = f"{path}: 0 remotes, every class local/never — the router has nowhere to send"
    else:
        detail = f"{path}: CAN ESCAPE — " + "; ".join(posture["reasons"])
    note = meta.get("error") or "; ".join(meta.get("drift") or [])
    if note:
        detail += (f" [the router did not take the file as written ({note}); judged on what it "
                   "resolved]")
    return policy, Check("routing_profile", posture["no_egress"], detail, fatal=True)


def _check_serving(environ: dict[str, str]) -> Check:
    """Serving resolution, measured by the status probe's connect-counting audit."""
    from .status.probes import _serving_load_fact
    from .status.report import LEVEL_OK

    fact = _serving_load_fact(environ)
    return Check(
        "serving_resolution", fact.level == LEVEL_OK, f"{fact.value}: {fact.detail}", fatal=True
    )


def _models_to_check(settings: Settings, policy, registry) -> dict[str, list[str]]:
    """Every model id a request can reach without naming one, with why it is reachable."""
    wanted: dict[str, list[str]] = {}
    wanted.setdefault(registry.default_id, []).append("default model")
    if policy is not None:
        if policy.defaults.local_model not in ("", "auto"):
            wanted.setdefault(policy.defaults.local_model, []).append("routing default")
        for name, rule in sorted(policy.classes.items()):
            if rule.local_model and rule.local_model != "auto":
                wanted.setdefault(rule.local_model, []).append(f"class {name}")
    if settings.embedder.lower() == "mlx":
        wanted.setdefault(settings.embed_model, []).append("embedder")
    return wanted


def _check_models(settings: Settings, policy, registry) -> tuple[list[Check], dict[str, str]]:
    """Each reachable model resolves to weights on disk, with zero connects attempted.

    Also returns ``{model_id: resolved_path}`` for the ones that did, so the load-path
    checks can probe with a real on-disk id rather than a path.
    """
    from .providers.mlx import audit_resolution

    backend = settings.backend.lower()
    if backend == "echo" or (backend == "auto" and not mlx_available()):
        return [Check(
            "models", True, f"backend={backend} serves the echo stub: no weights load", fatal=True
        )], {}
    from .serving.pool import UnknownModelError, resolve_model_id

    checks, resolved = [], {}
    for model_id, roles in _models_to_check(settings, policy, registry).items():
        why = ", ".join(roles)
        entry = registry.get(model_id)
        if any(role != "embedder" for role in roles):
            # The mlx backend serves through ModelPool, which refuses an id that is not a
            # registered chat model of its backend (echo, an embed model) — so such a rung is
            # a 404 at request time even though the routing loader accepted it. Judge the
            # rung by the pool's own resolver, not by "is it in the registry".
            try:
                resolve_model_id(registry, model_id, "mlx")
            except UnknownModelError as exc:
                checks.append(Check(
                    f"model {model_id}", False, f"{why}: NOT servable — {exc}", fatal=True
                ))
                continue
        if entry is not None and entry.backend == "echo":
            checks.append(Check(f"model {model_id}", True, f"{why}: echo, no weights", fatal=True))
            continue
        outcome, attempts = audit_resolution(model_id, allow_downloads=False)
        if attempts:
            ok, detail = False, f"resolving it tried {len(attempts)} connect(s) ({attempts[0]!r})"
        elif isinstance(outcome, Exception):
            ok, detail = False, f"NOT on disk — fetch it with `hearth models pull {model_id}`"
        elif not _has_weights(Path(str(outcome))):
            ok, detail = False, f"resolves to {outcome} but no weight file is there"
        else:
            ok, detail = True, f"on disk at {outcome}"
            resolved[model_id] = str(outcome)
        checks.append(Check(f"model {model_id}", ok, f"{why}: {detail}", fatal=True))
    return checks, resolved


def _load_plans(allow_downloads: bool, scratch: Path) -> dict:
    """The planning half of each non-serving load path — the code the real runners call."""
    from .convert import ConvertConfig, convert_invocation
    from .coreml import hf_load_plan
    from .training.lora import runner_invocation

    return {
        "hearth train": lambda src: runner_invocation(
            ["--train", "--model", src], allow_downloads=allow_downloads
        ),
        "hearth models convert": lambda src: convert_invocation(
            ConvertConfig(source=src, output_dir=scratch / "out"),
            allow_downloads=allow_downloads,
        ),
        "hearth models export-coreml": lambda src: hf_load_plan(
            src, allow_downloads=allow_downloads
        ),
    }


def _handed_and_offline(plan_outcome) -> tuple[str | None, bool]:
    """What the tool would be handed, and whether its load is pinned offline."""
    first, second = plan_outcome
    if isinstance(first, list):  # (command, child_env)
        flag = "--model" if "--model" in first else "--hf-path"
        return first[first.index(flag) + 1], all(second.get(v) == "1" for v in _OFFLINE_VARS)
    return first, second.get("local_files_only") is True  # (path, from_pretrained kwargs)


def _check_load_paths(settings: Settings, on_disk: dict[str, str]) -> list[Check]:
    """Exercise each non-serving load path's real planning code, with connects audited.

    Two probes per path: an absent model must raise ModelNotOnDiskError with no connect,
    and a model that IS on disk must be handed to the tool as its local path — never the
    bare id — with the load pinned offline (child env for subprocesses,
    ``local_files_only`` in-process). The on-disk probe uses a real model id found by the
    model checks (``on_disk``) when there is one, so "handed the path, not the id" is
    measured; with none on disk it falls back to a scratch directory passed as a path.
    """
    import tempfile

    from .providers.mlx import ModelNotOnDiskError, audit_connects

    checks = []
    with tempfile.TemporaryDirectory(prefix="hearth-doctor-") as tmp:
        if on_disk:
            source, expected = next(iter(on_disk.items()))
        else:
            scratch_model = Path(tmp) / "model"
            scratch_model.mkdir()
            source = expected = str(scratch_model)
        for name, plan in _load_plans(settings.allow_downloads, Path(tmp)).items():
            absent, absent_tries = audit_connects(plan, _ABSENT_MODEL)
            present, present_tries = audit_connects(plan, source)
            problems = []
            if absent_tries or present_tries:
                problems.append(f"{len(absent_tries) + len(present_tries)} connect(s) attempted")
            if not isinstance(absent, ModelNotOnDiskError):
                problems.append(f"a model not on disk did not fail (got {absent!r})")
            if isinstance(present, Exception):
                problems.append(f"a model on disk failed to plan: {present}")
            else:
                handed, offline = _handed_and_offline(present)
                if handed != expected:
                    problems.append(f"the tool is handed {handed!r}, not the local path")
                if not offline:
                    problems.append("the load is not pinned offline")
            checks.append(Check(
                f"load path: {name}",
                not problems,
                "; ".join(problems) or "disk-only: an absent model fails with no connect; "
                f"{source} is handed over as {expected} with the hub pinned offline",
                fatal=True,
            ))
    return checks


def _rag_component(name: str, var: str, raw: str, select, builtins: tuple, receives: str) -> Check:
    """Judge the RAG component HEARTH would actually construct, not the setting's spelling.

    ``select`` is the real selection function (``select_embedder`` / ``select_vector_store``),
    so aliases, case, and plugin resolution are exactly what ``hearth rag`` and the MCP
    server get. The verdict is on the TYPE of what came back: only an instance of a built-in
    class (exact type, so a plugin subclassing one is still a plugin) is vouched for. Anything
    else is third-party code that receives ``receives`` and whose network behaviour this
    command cannot measure, the same rule as a plugin ``HEARTH_BACKEND``.
    """
    try:
        chosen = select()
    except Exception as exc:  # unknown name, or a plugin that failed to load here
        return Check(
            name,
            False,
            f"{var}={raw} does not resolve to a built-in ({type(exc).__name__}: {exc}); "
            "cannot vouch for what RAG would use",
            fatal=True,
        )
    kind = type(chosen)
    if kind in builtins:
        return Check(name, True, f"{var}={raw} -> built-in {kind.__name__} (in-process)",
                     fatal=True)
    return Check(
        name,
        False,
        f"{var}={raw} -> plugin {kind.__module__}.{kind.__qualname__}: it receives "
        f"{receives}, and its network behaviour is not measured here",
        fatal=True,
    )


def _check_rag(settings: Settings) -> list[Check]:
    """The embedder and vector store RAG would use, resolved by their real selectors."""
    from .memory.embed import HashEmbedder, MLXEmbedder, select_embedder
    from .memory.store import SQLiteVectorStore, SqliteVecVectorStore, select_vector_store

    return [
        _rag_component(
            "embedder", "HEARTH_EMBEDDER", settings.embedder,
            lambda: select_embedder(settings), (HashEmbedder, MLXEmbedder),
            "every RAG chunk at ingest and every query",
        ),
        _rag_component(
            "vector_store", "HEARTH_VECTOR_STORE", settings.vector_store,
            lambda: select_vector_store(settings), (SQLiteVectorStore, SqliteVecVectorStore),
            "every RAG chunk's text and vector, and every query vector",
        ),
    ]


def run_offline_checks(
    settings: Settings | None = None,
    *,
    policy_path: Path | None = None,
    registry=None,
    environ: dict[str, str] | None = None,
) -> list[Check]:
    """Answer "is it safe to use HEARTH offline right now?" by measuring outcomes.

    Every fatal check exercises the code that would run — the router's own policy loader,
    the real resolver under a connect-refusing, connect-COUNTING audit, each load path's
    real planning function — rather than reading a setting that implies the answer
    (CLAUDE.md §3). Works with no network: nothing here opens a connection, and the audit
    would record it if anything tried. Injectable for tests; defaults are the live process.
    """
    import os

    from .registry import load_registry

    settings = settings or get_settings()
    env = dict(os.environ) if environ is None else dict(environ)
    registry = registry if registry is not None else load_registry()

    policy, routing = _check_routing(policy_path, env)
    model_checks, on_disk = _check_models(settings, policy, registry)
    builtin = settings.backend.lower() in _BUILTIN_BACKENDS
    loopback = _is_loopback(settings.host)
    # Non-fatal here: a misnamed default stops HEARTH starting; it opens no egress path.
    default_model = check_default_model(registry, fatal=False)
    return [
        routing,
        *([default_model] if default_model is not None else []),
        Check(
            "backend",
            builtin,
            f"HEARTH_BACKEND={settings.backend}"
            + ("" if builtin else " is a plugin — its network behaviour is not measured here"),
            fatal=True,
        ),
        *_check_rag(settings),
        Check(
            "bind_host",
            loopback,
            f"HEARTH_HOST={settings.host}"
            + (" (loopback)" if loopback
               else " is NOT a loopback address — the gateway would be reachable off-machine"),
            fatal=True,
        ),
        Check(
            "allow_downloads",
            not settings.allow_downloads,
            "HEARTH_ALLOW_DOWNLOADS is "
            + ("ON — a model not on disk would be fetched from the hub"
               if settings.allow_downloads else "off"),
            fatal=True,
        ),
        _check_serving(env),
        *model_checks,
        *_check_load_paths(settings, on_disk),
        Check(
            "hf_hub_offline",
            True,
            "HF_HUB_OFFLINE=" + (env.get("HF_HUB_OFFLINE") or "unset")
            + " — not required: every load path above is measured disk-only without it",
        ),
        Check(
            "models_pull",
            True,
            "`hearth models pull` is the one path that downloads, by design — run it only "
            "when you mean to fetch",
        ),
    ]
