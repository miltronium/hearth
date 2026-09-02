"""The load-bearing invariant: ``hearth.handoff`` has no way to reach the network.

The whole tier-3/4 design (``docs/TIERS.md``) rests on HEARTH never being able to send
anything anywhere. The handoff package is the piece that *talks about* off-machine tiers, so
it is the piece most likely to grow a "just fetch it for me" convenience. These tests make
that regression fail loudly in CI instead of quietly in production:

  * every import in the package resolves to a small stdlib allowlist — no HTTP client, no
    socket, no third-party dependency, not even another HEARTH module;
  * no call escapes to a shell or a dynamic import (``os.system``, ``subprocess``, ``eval``),
    which would be egress by proxy;
  * no non-docstring string literal contains a URL scheme, so an endpoint cannot be smuggled
    in as data.

They read the package's own source, so a new file in ``src/hearth/handoff/`` is covered the
moment it lands.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import hearth.handoff

PACKAGE_DIR = Path(hearth.handoff.__file__).parent
SOURCES = sorted(PACKAGE_DIR.glob("*.py"))

# Everything the package is allowed to import. Deliberately tiny: an auditor should be able
# to confirm "no network" by reading this list and the four modules. Adding to it is a
# decision, and this test is where that decision gets made visible.
ALLOWED_IMPORTS = frozenset(
    {
        "__future__",
        "dataclasses",
        "datetime",
        "hashlib",
        "json",
        "os",
        "pathlib",
        "re",
        "typing",
    }
)

# Names that would make egress possible even without an obviously networky import.
BANNED_CALLS = frozenset(
    {
        "system",
        "popen",
        "execv",
        "execve",
        "execvp",
        "spawnl",
        "spawnv",
        "urlopen",
        "socket",
        "connect",
        "sendall",
        "eval",
        "exec",
        "__import__",
    }
)


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _docstring_nodes(tree: ast.Module) -> set[int]:
    """Return ``id()`` of every module/class/function docstring constant."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body:
            first = body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                if isinstance(first.value.value, str):
                    ids.add(id(first.value))
    return ids


def test_the_package_has_sources_to_check():
    # A vacuous pass here would silently disarm every other test in this file.
    assert len(SOURCES) >= 4
    assert {p.name for p in SOURCES} >= {"__init__.py", "envelope.py", "review.py", "store.py"}


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_only_allowlisted_stdlib_imports(path: Path):
    offenders: list[str] = []
    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    offenders.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative: stays inside this package
                continue
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                offenders.append(node.module or "?")
    assert not offenders, (
        f"{path.name} imports {offenders} — hearth.handoff must stay stdlib-only and "
        "network-free (docs/TIERS.md); if this is genuinely needed, the invariant needs "
        "an explicit decision, not an allowlist edit in passing"
    )


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_no_shell_or_dynamic_import_escape_hatch(path: Path):
    offenders: list[str] = []
    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.Attribute) and node.attr in BANNED_CALLS:
            offenders.append(node.attr)
        elif isinstance(node, ast.Name) and node.id in BANNED_CALLS:
            offenders.append(node.id)
    assert not offenders, f"{path.name} references {offenders} — a shell is an egress vector"


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_no_endpoint_smuggled_in_as_a_string_literal(path: Path):
    tree = _parse(path)
    docstrings = _docstring_nodes(tree)
    offenders = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
        and "://" in node.value
    ]
    assert not offenders, f"{path.name} contains a URL-shaped literal: {offenders}"


def test_no_network_module_reachable_through_the_package_namespace():
    # Belt and braces: whatever the source says, the imported module object must not have
    # pulled a transport in under a different name.
    for name in ("socket", "http", "httpx", "requests", "urllib", "subprocess", "asyncio"):
        assert not hasattr(hearth.handoff, name), f"hearth.handoff exposes {name}"
