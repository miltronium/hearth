"""The load-bearing invariant: ``hearth.status`` can only ever look, never touch.

A status command is the thing you run *first* on a machine you already suspect is broken,
and in a sealed no-egress session where anything leaving the box is the incident. It earns
that position only if it provably cannot write, mutate, or transmit. These tests make a
regression fail in CI instead of quietly in production.

They check the invariant twice, on purpose, because that duplication is the whole thesis of
this package:

  * **the configuration** — an AST scan of the package's own source for network imports and
    filesystem-write calls, which is a static claim about what the code *could* do;
  * **the outcome** — a real :func:`collect_status` run against a fixture tree, with the
    tree fingerprinted before and after, which is a dynamic fact about what it *did*.

The second is the one that counts. A source scan is exactly the kind of gate that passes
while the thing it guards is false (an ``importlib`` call, a C extension, a helper in
another package), so it is here as an early-warning tripwire and not as the proof.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import hearth.status
from hearth.status import collect_status
from hearth.status.gitmeta import READ_ONLY_SUBCOMMANDS, UnsafeGitCommand, git

PACKAGE_DIR = Path(hearth.status.__file__).parent
SOURCES = sorted(PACKAGE_DIR.glob("*.py"))

# Anything that could open a socket, directly or by proxy. A status report that can reach
# the network is a status report that can exfiltrate the thing it is describing.
BANNED_IMPORT_ROOTS = frozenset(
    {
        "aiohttp",
        "asyncio",
        "ftplib",
        "http",
        "httpx",
        "requests",
        "smtplib",
        "socket",
        "socketserver",
        "ssl",
        "telnetlib",
        "urllib",
        "urllib3",
        "webbrowser",
        "xmlrpc",
    }
)

# Calls that mutate the filesystem or the process's world. `replace` is deliberately absent:
# `Path.replace` writes but `str.replace` does not, and the AST cannot tell them apart — the
# runtime fixture test below is what actually covers that case.
BANNED_WRITE_CALLS = frozenset(
    {
        "chmod",
        "check_call",
        "check_output",
        "copyfile",
        "copytree",
        "dump",
        "makedirs",
        "mkdir",
        "mkdtemp",
        "move",
        "open",
        "Popen",
        "popen",
        "remove",
        "removedirs",
        "rmdir",
        "rmtree",
        "symlink_to",
        "system",
        "touch",
        "truncate",
        "unlink",
        "write",
        "write_bytes",
        "write_text",
        "writelines",
    }
)

# Qualified calls that share a name with a banned one but are harmless. Keep this list
# short and justified: every entry is a hole in the static scan, which is precisely why the
# runtime fingerprint test below — not this scan — is the actual proof of the invariant.
ALLOWED_QUALIFIED_CALLS = frozenset({("platform", "system")})

# The single module allowed to spawn a process, and the only function it may use.
SUBPROCESS_MODULE = "gitmeta.py"


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _import_roots(tree: ast.Module) -> list[str]:
    roots: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level:
            roots.append((node.module or "").split(".")[0])
    return roots


def test_the_package_has_sources_to_check():
    # A vacuous pass here would silently disarm every other test in this file.
    assert {p.name for p in SOURCES} >= {
        "__init__.py",
        "gitmeta.py",
        "probes.py",
        "render.py",
        "report.py",
    }


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_no_network_capable_import(path: Path):
    offenders = sorted(set(_import_roots(_parse(path))) & BANNED_IMPORT_ROOTS)
    assert not offenders, (
        f"{path.name} imports {offenders} — hearth.status must have no way to reach the "
        "network; it runs in sealed sessions and describes confidential state"
    )


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_no_write_or_mutation_call(path: Path):
    tree = _parse(path)
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            name = func.attr
            receiver = func.value.id if isinstance(func.value, ast.Name) else None
            if (receiver, name) in ALLOWED_QUALIFIED_CALLS:
                continue
        else:
            name = getattr(func, "id", None)
        if name in BANNED_WRITE_CALLS:
            offenders.append(name)
    assert not offenders, (
        f"{path.name} calls {sorted(set(offenders))} — hearth.status is read-only, so it "
        "must always be safe to run first on a machine you think is already broken"
    )


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_never_imports_the_handoff_package(path: Path):
    """``hearth.handoff`` holds a near-identical no-egress invariant. Keep them independent.

    Two packages that each prove they cannot reach the network prove nothing if one imports
    the other — the guarantee becomes a chain, and a chain is only ever as strong as the
    module nobody re-audited.
    """
    offenders: list[str] = []
    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.Import):
            offenders.extend(a.name for a in node.names if a.name.startswith("hearth.handoff"))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith("hearth.handoff") or (node.level and module.startswith("handoff")):
                offenders.append(module)
    assert not offenders, f"{path.name} imports {offenders} — keep the two invariants separate"


def test_subprocess_is_confined_to_one_module():
    users = [p.name for p in SOURCES if "subprocess" in _import_roots(_parse(p))]
    assert users == [SUBPROCESS_MODULE], (
        f"subprocess is imported by {users}; it must live only in {SUBPROCESS_MODULE}, "
        "behind the read-only subcommand allowlist"
    )


def test_the_only_process_ever_spawned_is_read_only_git():
    tree = _parse(PACKAGE_DIR / SUBPROCESS_MODULE)
    spawns = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
    ]
    assert len(spawns) == 1, f"expected exactly one subprocess call, found {len(spawns)}"
    call = spawns[0]
    assert call.func.attr == "run", "only subprocess.run is permitted (no Popen, no shell)"
    argv = call.args[0]
    assert isinstance(argv, ast.List) and isinstance(argv.elts[0], ast.Constant)
    assert argv.elts[0].value == "git", "the spawned program must be a literal 'git'"
    kwargs = {kw.arg for kw in call.keywords}
    assert "shell" not in kwargs, "never hand git's argv to a shell"


def test_git_helper_refuses_a_mutating_subcommand(tmp_path: Path):
    for subcommand in ("commit", "add", "push", "checkout", "gc", "fetch"):
        with pytest.raises(UnsafeGitCommand):
            git([subcommand], cwd=tmp_path)
    assert "commit" not in READ_ONLY_SUBCOMMANDS


def test_git_helper_degrades_to_none_outside_a_repo(tmp_path: Path):
    # No traceback on a machine without git, or a directory that is not a checkout.
    assert git(["rev-parse", "--is-inside-work-tree"], cwd=tmp_path) in (None, "false")


def _fingerprint(root: Path) -> set[tuple[str, int, int]]:
    """(relative path, size, mtime_ns) for every file under ``root``."""
    out: set[tuple[str, int, int]] = set()
    for path in sorted(root.rglob("*")):
        st = path.lstat()
        out.add((str(path.relative_to(root)), st.st_size, st.st_mtime_ns))
    return out


def test_a_real_run_changes_nothing_on_disk(tmp_path: Path):
    """The outcome check: run every probe for real, then diff the tree byte for byte.

    This is the assertion that would catch what the source scan cannot — a write through an
    alias, a cache file dropped by an imported library, a temp directory left behind.
    """
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "tests").mkdir()
    (root / "config" / "routing.yaml").write_text(
        "defaults: {local_model: auto, remote: none}\n"
        "classes:\n  chat: {backend: local, escalate: never}\nremotes: {}\n"
    )
    (root / "data" / "task_golden.jsonl").write_text('{"prompt": "a", "expected": "b"}\n')
    (root / "tests" / "test_example.py").write_text("def test_ok():\n    assert True\n")
    home = tmp_path / "home"
    (home / "models").mkdir(parents=True)

    before_root, before_home = _fingerprint(root), _fingerprint(home)
    report = collect_status(root=root, home=home, environ={"HF_HUB_CACHE": str(home / "models")})
    assert report.sections, "the probe must actually have run for this test to mean anything"

    assert _fingerprint(root) == before_root, "hearth.status wrote to the repo"
    assert _fingerprint(home) == before_home, "hearth.status wrote to ~/.hearth"


def test_no_transport_reachable_through_the_package_namespace():
    # Belt and braces: whatever the source says, the imported modules must not have pulled a
    # transport in under another name.
    for module in (hearth.status, hearth.status.probes, hearth.status.render):
        for name in ("socket", "http", "httpx", "requests", "urllib", "asyncio"):
            assert not hasattr(module, name), f"{module.__name__} exposes {name}"
