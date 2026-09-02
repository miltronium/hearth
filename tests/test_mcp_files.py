"""Tests for the allowlisted file reader and the path-taking MCP tools (docs/PRIVACY.md).

These close "the caller caveat": the ``*_file`` tools let an agent offload work on a
confidential file without reading it, which makes them an arbitrary-file-read primitive in
that agent's hands. So most of what follows is the *security* surface —
deny-by-default, traversal, symlink escape, size cap, wrong file type — plus the guarantee
that a refusal never hands the file's content back in the error message.

Like :mod:`tests.test_mcp_tools`, everything runs against the echo router with no ``mcp``
package installed.
"""

from __future__ import annotations

import pytest

from hearth.config import Settings
from hearth.mcp.files import (
    FileAccessError,
    allowed_roots,
    read_text_file,
    resolve_under_roots,
)
from hearth.mcp.tools import HearthTools
from hearth.memory import RagIndex, SQLiteVectorStore, select_embedder
from hearth.providers.echo import EchoProvider
from hearth.router import Router

SECRET = "CONFIDENTIAL-CANARY-9f3a"


def _settings(tmp_path, roots: list, **kwargs) -> Settings:
    """Echo-backed settings whose file allowlist is exactly ``roots``."""
    return Settings(
        backend="echo",
        home=tmp_path / ".hearth",
        require_auth=False,
        file_roots=":".join(str(r) for r in roots),
        **kwargs,
    )


def _tools(settings: Settings) -> HearthTools:
    router = Router(local_provider=EchoProvider())
    rag = RagIndex(
        embedder=select_embedder(settings),
        store=SQLiteVectorStore(settings=settings),
        router=router,
    )
    return HearthTools(router=router, rag=rag, settings=settings)


@pytest.fixture
def root(tmp_path):
    """An allowed root containing one readable file, plus a secret file OUTSIDE it."""
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (allowed / "note.txt").write_text("the quick brown fox\n")
    (tmp_path / "outside.txt").write_text(f"{SECRET}\n")
    return allowed


# -- the allowlist ---------------------------------------------------------------------


def test_unset_roots_denies_every_read(tmp_path, root):
    settings = _settings(tmp_path, [])  # HEARTH_FILE_ROOTS unset/empty
    with pytest.raises(FileAccessError) as excinfo:
        read_text_file(root / "note.txt", settings=settings)
    # The error must tell the operator how to turn it on.
    assert "HEARTH_FILE_ROOTS" in str(excinfo.value)


def test_path_inside_root_is_allowed(tmp_path, root):
    settings = _settings(tmp_path, [root])
    assert read_text_file(root / "note.txt", settings=settings) == "the quick brown fox\n"


def test_traversal_out_of_root_is_denied(tmp_path, root):
    settings = _settings(tmp_path, [root])
    escape = root / ".." / "outside.txt"  # resolves to tmp_path/outside.txt
    with pytest.raises(FileAccessError) as excinfo:
        read_text_file(escape, settings=settings)
    assert "outside every allowed root" in str(excinfo.value)
    assert SECRET not in str(excinfo.value)


def test_symlink_escaping_root_is_denied(tmp_path, root):
    settings = _settings(tmp_path, [root])
    link = root / "link.txt"
    link.symlink_to(tmp_path / "outside.txt")  # planted *inside* the root
    with pytest.raises(FileAccessError) as excinfo:
        read_text_file(link, settings=settings)
    assert "outside every allowed root" in str(excinfo.value)
    assert SECRET not in str(excinfo.value)


def test_symlink_staying_inside_root_is_allowed(tmp_path, root):
    settings = _settings(tmp_path, [root])
    link = root / "alias.txt"
    link.symlink_to(root / "note.txt")
    assert read_text_file(link, settings=settings) == "the quick brown fox\n"


def test_sibling_with_root_name_as_prefix_is_denied(tmp_path, root):
    # /…/allowed vs /…/allowed-evil: a string-prefix check would wrongly admit this.
    sibling = tmp_path / "allowed-evil"
    sibling.mkdir()
    (sibling / "note.txt").write_text(f"{SECRET}\n")
    settings = _settings(tmp_path, [root])
    with pytest.raises(FileAccessError):
        read_text_file(sibling / "note.txt", settings=settings)


def test_multiple_roots_are_each_honored(tmp_path, root):
    second = tmp_path / "also-allowed"
    second.mkdir()
    (second / "b.txt").write_text("second root\n")
    settings = _settings(tmp_path, [root, second])
    assert read_text_file(root / "note.txt", settings=settings)
    assert read_text_file(second / "b.txt", settings=settings) == "second root\n"


def test_roots_read_from_the_environment(tmp_path, root, monkeypatch):
    # Proves the HEARTH_FILE_ROOTS wiring, not just the Settings field.
    monkeypatch.setenv("HEARTH_FILE_ROOTS", str(root))
    monkeypatch.setenv("HEARTH_HOME", str(tmp_path / ".hearth"))
    assert allowed_roots(Settings()) == [root.resolve()]


def test_nonexistent_and_blank_roots_are_dropped(tmp_path, root):
    settings = _settings(tmp_path, [root, tmp_path / "nope", "", "  "])
    assert allowed_roots(settings) == [root.resolve()]


def test_all_roots_invalid_denies(tmp_path, root):
    settings = _settings(tmp_path, [tmp_path / "nope", root / "note.txt"])  # dir-only
    assert allowed_roots(settings) == []
    with pytest.raises(FileAccessError):
        resolve_under_roots(root / "note.txt", settings=settings)


def test_resolve_under_roots_returns_the_real_path(tmp_path, root):
    settings = _settings(tmp_path, [root])
    note = root / "note.txt"
    assert resolve_under_roots(note, settings=settings) == note.resolve()


# -- what may be read ------------------------------------------------------------------


def test_directory_is_rejected(tmp_path, root):
    settings = _settings(tmp_path, [root])
    with pytest.raises(FileAccessError) as excinfo:
        read_text_file(root, settings=settings)
    assert "directory" in str(excinfo.value)


def test_oversized_file_is_rejected(tmp_path, root):
    big = root / "big.txt"
    big.write_text(SECRET + "x" * 5000)
    settings = _settings(tmp_path, [root], file_max_bytes=1024)
    with pytest.raises(FileAccessError) as excinfo:
        read_text_file(big, settings=settings)
    message = str(excinfo.value)
    assert "too large" in message and "1024" in message
    assert SECRET not in message  # the refusal must not carry the content back


def test_unsupported_extension_names_the_format(tmp_path, root):
    pdf = root / "statement.pdf"
    pdf.write_bytes(b"%PDF-1.7 not really\n")
    settings = _settings(tmp_path, [root])
    with pytest.raises(FileAccessError) as excinfo:
        read_text_file(pdf, settings=settings)
    assert ".pdf" in str(excinfo.value)


def test_missing_file_reports_without_leaking(tmp_path, root):
    settings = _settings(tmp_path, [root])
    with pytest.raises(FileAccessError) as excinfo:
        read_text_file(root / "absent.txt", settings=settings)
    assert "not found" in str(excinfo.value)


def test_binary_content_in_a_text_file_is_rejected(tmp_path, root):
    blob = root / "blob.txt"
    blob.write_bytes(b"\x00\xff\xfe binary")
    settings = _settings(tmp_path, [root])
    with pytest.raises(FileAccessError) as excinfo:
        read_text_file(blob, settings=settings)
    assert "not UTF-8" in str(excinfo.value)


def test_extensionless_and_markdown_files_read_as_text(tmp_path, root):
    (root / "README").write_text("plain\n")
    (root / "doc.md").write_text("# heading\n")
    settings = _settings(tmp_path, [root])
    assert read_text_file(root / "README", settings=settings) == "plain\n"
    assert read_text_file(root / "doc.md", settings=settings) == "# heading\n"


def test_csv_is_normalized_to_rows(tmp_path, root):
    csv_path = root / "aug.csv"
    csv_path.write_text('date,amount,memo\n2026-08-01,12.50,"coffee, iced"\n\n')
    settings = _settings(tmp_path, [root])
    assert read_text_file(csv_path, settings=settings) == (
        "date | amount | memo\n2026-08-01 | 12.50 | coffee, iced"
    )


# -- the tools themselves --------------------------------------------------------------


def test_summarize_file_runs_locally(tmp_path, root):
    tools = _tools(_settings(tmp_path, [root]))
    out = tools.summarize_file(str(root / "note.txt"), max_words=10)
    # Echo backend prefixes [echo]; the file's text rode through the local prompt.
    assert "[echo]" in out
    assert "the quick brown fox" in out


def test_classify_file_returns_text(tmp_path, root):
    tools = _tools(_settings(tmp_path, [root]))
    out = tools.classify_file(str(root / "note.txt"), labels=["animal", "vegetable"])
    assert isinstance(out, str) and out


def test_extract_file_returns_field_map(tmp_path, root):
    (root / "ticket.txt").write_text("ticket ABC-1\n")
    tools = _tools(_settings(tmp_path, [root]))
    out = tools.extract_file(str(root / "ticket.txt"), fields=["ticket", "assignee"])
    assert set(out.keys()) == {"ticket", "assignee"}


def test_file_tools_refuse_outside_the_root(tmp_path, root):
    tools = _tools(_settings(tmp_path, [root]))
    outside = str(tmp_path / "outside.txt")
    for call in (
        lambda: tools.summarize_file(outside),
        lambda: tools.classify_file(outside, labels=["a", "b"]),
        lambda: tools.extract_file(outside, fields=["x"]),
    ):
        with pytest.raises(FileAccessError) as excinfo:
            call()
        assert SECRET not in str(excinfo.value)


def test_file_tools_deny_when_roots_unset(tmp_path, root):
    tools = _tools(_settings(tmp_path, []))
    with pytest.raises(FileAccessError):
        tools.summarize_file(str(root / "note.txt"))


def test_file_tools_validate_arguments_like_their_text_twins(tmp_path, root):
    tools = _tools(_settings(tmp_path, [root]))
    path = str(root / "note.txt")
    with pytest.raises(ValueError):
        tools.classify_file(path, labels=[])
    with pytest.raises(ValueError):
        tools.extract_file(path, fields=[])


def test_file_access_error_is_a_value_error():
    # The MCP layer surfaces ValueError-shaped tool errors uniformly.
    assert issubclass(FileAccessError, ValueError)
