"""The load-bearing invariant: ``hearth.agent`` has no way to reach the network or a shell.

Written to the pattern of :mod:`tests.test_handoff_no_network`, for the same reason and with
one difference. The reason: an agent package is *the* place a "just fetch that URL for me"
convenience gets added, because it is the one place where it would obviously be useful. These
tests make that land as a red build rather than as a quiet new capability, and they read the
package's own source, so a new file in ``src/hearth/agent/`` is covered the moment it lands.

The difference: this package cannot be stdlib-only — its whole job is to drive HEARTH's own
router and its own allowlisted file reader. So the allowlist below names the HEARTH modules it
may reach, *including relative imports*, which are resolved to their dotted path rather than
being skipped. Skipping them would leave the largest hole available: ``from ..providers.remote
import RemoteProvider`` is a relative import, and it is the one module in this repository that
can open a socket.

This is also the gate the loop's ``vetted_only`` default relies on. That default admits only
tools whose code lives in this directory; what makes that meaningful is that the code in this
directory is held network-free *here*.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import hearth.agent

PACKAGE = "hearth.agent"
PACKAGE_DIR = Path(hearth.agent.__file__).parent
SOURCES = sorted(PACKAGE_DIR.glob("*.py"))

# Standard-library modules the package may import. Deliberately tiny, and deliberately
# lacking every transport: no socket, no http, no urllib, no asyncio, no subprocess.
# Adding to this list is a decision, and this test is where that decision becomes visible.
ALLOWED_STDLIB = frozenset(
    {
        "__future__",
        "collections.abc",
        "dataclasses",
        "datetime",
        "json",
        "logging",
        "pathlib",
        "re",
        "time",
        "typing",
    }
)

# HEARTH modules the package may import. Named individually rather than by package prefix:
# `hearth.providers.base` is a dataclass contract, while `hearth.providers.remote` is the one
# module in this repository that can open a connection, and a prefix rule would admit both.
ALLOWED_HEARTH = frozenset(
    {
        "hearth.mcp.files",
        "hearth.providers.base",
        "hearth.router.route",
    }
)

# Names that would make egress possible even without an obviously networky import. The
# import allowlist above is the primary gate; this catches the same capability arriving under
# a different name. Deliberately excludes names that are ambiguous in ordinary Python
# (`run`, `compile`) — a ban that fires on `re.compile` teaches people to edit the list.
BANNED_CALLS = frozenset(
    {
        "system",
        "popen",
        "check_output",
        "execv",
        "execve",
        "execvp",
        "spawnl",
        "spawnv",
        "urlopen",
        "urlretrieve",
        "socket",
        "create_connection",
        "sendall",
        "eval",
        "exec",
        "__import__",
        "import_module",
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


def _resolve(module: str | None, level: int) -> str:
    """Resolve an import to its absolute dotted path, relative ones included.

    ``level`` 1 is ``hearth.agent``, 2 is ``hearth``. Resolving rather than skipping is the
    point: a relative import can reach anything in the repository.
    """
    if not level:
        return module or "?"
    parts = PACKAGE.split(".")
    base = parts[: len(parts) - (level - 1)]
    return ".".join([*base, module]) if module else ".".join(base)


def test_the_package_has_sources_to_check():
    # A vacuous pass here would silently disarm every other test in this file.
    assert len(SOURCES) >= 5
    assert {p.name for p in SOURCES} >= {
        "__init__.py",
        "builtins.py",
        "loop.py",
        "protocol.py",
        "tools.py",
    }


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_only_allowlisted_imports(path: Path):
    offenders: list[str] = []
    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_STDLIB and alias.name not in ALLOWED_HEARTH:
                    offenders.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            resolved = _resolve(node.module, node.level)
            if resolved.startswith(f"{PACKAGE}.") or resolved == PACKAGE:
                continue  # intra-package: covered by these same tests
            if resolved not in ALLOWED_STDLIB and resolved not in ALLOWED_HEARTH:
                offenders.append(resolved)
    assert not offenders, (
        f"{path.name} imports {offenders} — hearth.agent may reach only the standard-library "
        f"modules in ALLOWED_STDLIB and the HEARTH modules in ALLOWED_HEARTH. An agent "
        "package is exactly where a 'just fetch that URL' convenience gets added; if this is "
        "genuinely needed it wants an explicit decision, not an allowlist edit in passing "
        "(docs/AGENT.md, docs/TIERS.md)."
    )


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_no_shell_dynamic_import_or_socket_escape_hatch(path: Path):
    offenders: list[str] = []
    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.Attribute) and node.attr in BANNED_CALLS:
            offenders.append(node.attr)
        elif isinstance(node, ast.Name) and node.id in BANNED_CALLS:
            offenders.append(node.id)
    assert not offenders, (
        f"{path.name} references {offenders} — a shell, a dynamic import and a socket are all "
        "egress vectors, and a tool is not allowed to become one"
    )


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


def test_no_transport_reachable_through_the_package_namespace():
    # Belt and braces: whatever the source says, the imported module object must not have
    # pulled a transport in under a different name.
    for name in ("socket", "http", "httpx", "requests", "urllib", "subprocess", "asyncio", "os"):
        assert not hasattr(hearth.agent, name), f"hearth.agent exposes {name}"


def test_the_router_is_the_only_thing_that_talks_to_a_model():
    # The package reaches exactly one execution seam. If a provider were ever constructed
    # here directly, this package would stop inheriting the router's local-only routing.
    offenders: list[str] = []
    for path in SOURCES:
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.ImportFrom):
                resolved = _resolve(node.module, node.level)
                if resolved.startswith("hearth.providers") and resolved != "hearth.providers.base":
                    offenders.append(f"{path.name}: {resolved}")
    assert not offenders, (
        f"{offenders} — agent generations go through hearth.router.Router so that "
        "allow_escalation=False and the served-locally check both apply"
    )


def test_the_built_in_tools_expose_no_writing_capability():
    # The AST tests above cover egress; this one covers the other half of the security model.
    # A write tool is not a network call, so nothing above would catch one appearing.
    banned = ("write_text", "write_bytes", "mkdir", "unlink", "rmtree", "rename", "touch")
    offenders: list[str] = []
    for path in SOURCES:
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.Attribute) and node.attr in banned:
                offenders.append(f"{path.name}: {node.attr}")
    assert not offenders, (
        f"{offenders} — the built-in toolset is read-only by decision (docs/AGENT.md): a "
        "wrong read produces a checkable sentence, a wrong write produces a corrupted file "
        "the agent reports as done"
    )
