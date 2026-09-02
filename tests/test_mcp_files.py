"""Tests for the allowlisted file reader and the path-taking MCP tools (docs/PRIVACY.md).

These close "the caller caveat": the ``*_file`` tools let an agent offload work on a
confidential file without reading it, which makes them an arbitrary-file-read primitive in
that agent's hands. So most of what follows is the *security* surface —
deny-by-default, traversal, symlink escape, size cap, wrong file type — plus the guarantee
that a refusal never hands the file's content back in the error message.

Like :mod:`tests.test_mcp_tools`, everything runs against the echo router with no ``mcp``
package installed. The XLSX/PDF parsers live behind the ``[files]`` extra, so tests that
need them are skipped (not failed) when it isn't installed — but their *refusal* paths, the
ones that matter for the security boundary, are exercised either way.
"""

from __future__ import annotations

import importlib
import importlib.util
import io

import pytest

from hearth.config import Settings
from hearth.mcp import files as files_module
from hearth.mcp.files import (
    FileAccessError,
    allowed_roots,
    read_table,
    read_text_file,
    resolve_under_roots,
)
from hearth.mcp.tools import HearthTools
from hearth.memory import RagIndex, SQLiteVectorStore, select_embedder
from hearth.providers.echo import EchoProvider
from hearth.router import Router

SECRET = "CONFIDENTIAL-CANARY-9f3a"


def _needs(module: str):
    """Skip a test when an optional ``[files]`` parser isn't installed."""
    return pytest.mark.skipif(
        importlib.util.find_spec(module) is None,
        reason=f"{module} is not installed (uv sync --extra files)",
    )


needs_openpyxl = _needs("openpyxl")
needs_pypdf = _needs("pypdf")


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


# -- builders for the binary formats ----------------------------------------------------
#
# Both are hand-built here rather than checked in as fixture files, so the exact bytes a
# test relies on are visible next to the assertion — and so the "scanned" PDF is provably
# the identical document minus its text-drawing operators, not a different file.


def _xlsx_bytes(sheets: dict[str, list[list]]) -> bytes:
    """Build an XLSX workbook in memory from ``{sheet title: rows}``.

    openpyxl never *computes* formulas, so any workbook it writes has no cached values —
    which is exactly the "written by a script, never opened in Excel" case the reader must
    refuse. A cell written as ``"=SUM(...)"`` therefore exercises the uncached path, and to
    stand in for a recalculated workbook a test writes the resulting value instead.
    """
    import openpyxl

    workbook = openpyxl.Workbook()
    workbook.remove(workbook.active)
    for title, rows in sheets.items():
        worksheet = workbook.create_sheet(title=title)
        for row in rows:
            worksheet.append(row)
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def _pdf_bytes(pages: list[list[str]]) -> bytes:
    """Build a minimal, valid multi-page PDF; a page with no lines gets no text layer."""
    count = len(pages)
    page_nums = [4 + 2 * i for i in range(count)]
    content_nums = [5 + 2 * i for i in range(count)]

    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: (
            b"<< /Type /Pages /Kids ["
            + b" ".join(f"{n} 0 R".encode() for n in page_nums)
            + b"] /Count "
            + str(count).encode()
            + b" >>"
        ),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    for index, lines in enumerate(pages):
        objects[page_nums[index]] = (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R >> >> /Contents "
            + str(content_nums[index]).encode()
            + b" 0 R >>"
        )
        ops = b""
        if lines:
            drawn = b"".join(b"(" + line.encode("ascii") + b") Tj T*\n" for line in lines)
            ops = b"BT /F1 12 Tf 72 720 Td 14 TL\n" + drawn + b"ET\n"
        objects[content_nums[index]] = (
            b"<< /Length " + str(len(ops)).encode() + b" >>\nstream\n" + ops + b"endstream"
        )

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for num in sorted(objects):
        offsets[num] = len(out)
        out += str(num).encode() + b" 0 obj\n" + objects[num] + b"\nendobj\n"

    xref_at = len(out)
    size = max(objects) + 1
    out += b"xref\n0 " + str(size).encode() + b"\n0000000000 65535 f \n"
    for num in range(1, size):
        out += f"{offsets[num]:010d} 00000 n \n".encode()
    out += b"trailer\n<< /Size " + str(size).encode() + b" /Root 1 0 R >>\n"
    out += b"startxref\n" + str(xref_at).encode() + b"\n%%EOF\n"
    return bytes(out)


def _hide_module(monkeypatch, name: str) -> None:
    """Make ``import <name>`` fail, to exercise the missing-``[files]``-extra path."""
    real = importlib.import_module

    def fake(module, *args, **kwargs):
        if module == name:
            raise ImportError(f"No module named {name!r}")
        return real(module, *args, **kwargs)

    monkeypatch.setattr(files_module.importlib, "import_module", fake)


# -- JSON --------------------------------------------------------------------------------


def test_json_is_pretty_printed_with_sorted_keys(tmp_path, root):
    # Compact input, minified and key-shuffled; the model must see the normalized form.
    (root / "acct.json").write_text('{"z":1,"a":{"n":2,"m":3}}')
    settings = _settings(tmp_path, [root])
    assert read_text_file(root / "acct.json", settings=settings) == (
        '{\n  "a": {\n    "m": 3,\n    "n": 2\n  },\n  "z": 1\n}'
    )


def test_json_keeps_non_ascii_readable(tmp_path, root):
    (root / "u.json").write_text('{"memo": "café"}', encoding="utf-8")
    settings = _settings(tmp_path, [root])
    assert "café" in read_text_file(root / "u.json", settings=settings)


def test_invalid_json_is_refused_without_quoting_it(tmp_path, root):
    (root / "bad.json").write_text(f'{{"memo": "{SECRET}", }}')
    settings = _settings(tmp_path, [root])
    with pytest.raises(FileAccessError) as excinfo:
        read_text_file(root / "bad.json", settings=settings)
    assert "not valid JSON" in str(excinfo.value)
    assert SECRET not in str(excinfo.value)


def test_json_array_of_objects_becomes_a_table_with_ragged_keys(tmp_path, root):
    # Ragged on purpose: 'memo' only on the first record, 'fee' only on the second. The
    # header is the sorted union, and the gaps are '' rather than a shifted row.
    (root / "tx.json").write_text(
        '[{"date": "2026-08-01", "amount": 12.5, "memo": "coffee"},'
        ' {"amount": 3, "date": "2026-08-02", "fee": null},'
        ' {"date": "2026-08-03", "amount": 1, "tags": ["a", "b"]}]'
    )
    settings = _settings(tmp_path, [root])
    assert read_table(root / "tx.json", settings=settings) == [
        ["amount", "date", "fee", "memo", "tags"],
        ["12.5", "2026-08-01", "", "coffee", ""],
        ["3", "2026-08-02", "", "", ""],
        ["1", "2026-08-03", "", "", '["a","b"]'],
    ]


def test_empty_json_array_is_an_empty_table(tmp_path, root):
    (root / "none.json").write_text("[]")
    settings = _settings(tmp_path, [root])
    # Honestly empty, and visibly so to a programmatic caller — no invented header.
    assert read_table(root / "none.json", settings=settings) == []


def test_json_table_refuses_a_top_level_object(tmp_path, root):
    (root / "obj.json").write_text(f'{{"memo": "{SECRET}"}}')
    settings = _settings(tmp_path, [root])
    with pytest.raises(FileAccessError) as excinfo:
        read_table(root / "obj.json", settings=settings)
    assert "not an array of objects" in str(excinfo.value)
    assert "object" in str(excinfo.value)
    assert SECRET not in str(excinfo.value)


def test_json_table_refuses_an_array_of_scalars(tmp_path, root):
    (root / "flat.json").write_text(f'["{SECRET}", 2]')
    settings = _settings(tmp_path, [root])
    with pytest.raises(FileAccessError) as excinfo:
        read_table(root / "flat.json", settings=settings)
    assert "non-object entries" in str(excinfo.value)
    assert SECRET not in str(excinfo.value)


# -- XLSX --------------------------------------------------------------------------------


@needs_openpyxl
def test_xlsx_renders_every_sheet_with_its_title(tmp_path, root):
    (root / "book.xlsx").write_bytes(
        _xlsx_bytes(
            {
                "Transactions": [["date", "amount"], ["2026-08-01", 12.5]],
                "Notes": [["memo"], ["annual fee"]],
            }
        )
    )
    settings = _settings(tmp_path, [root])
    assert read_text_file(root / "book.xlsx", settings=settings) == (
        "# Sheet: Transactions\ndate | amount\n2026-08-01 | 12.5\n\n"
        "# Sheet: Notes\nmemo\nannual fee"
    )


@needs_openpyxl
def test_xlsx_blank_rows_are_dropped_and_gaps_kept(tmp_path, root):
    (root / "gappy.xlsx").write_bytes(
        _xlsx_bytes({"S": [["a", "b", "c"], [None, None, None], ["x", None, "z"]]})
    )
    settings = _settings(tmp_path, [root])
    assert read_text_file(root / "gappy.xlsx", settings=settings) == (
        "# Sheet: S\na | b | c\nx |  | z"
    )


@needs_openpyxl
def test_xlsx_formula_cell_with_a_cached_value_reads_as_that_value(tmp_path, root):
    # A recalculated workbook: data_only=True hands back the cached number, and the
    # '=SUM(...)' text never reaches the model.
    (root / "totals.xlsx").write_bytes(_xlsx_bytes({"S": [["item", "amt"], ["total", 42]]}))
    settings = _settings(tmp_path, [root])
    out = read_text_file(root / "totals.xlsx", settings=settings)
    assert "total | 42" in out
    assert "=SUM" not in out


@needs_openpyxl
def test_xlsx_formula_without_a_cached_value_is_refused(tmp_path, root):
    # openpyxl writes the formula but never evaluates it, so the cached value is absent —
    # exactly the file that would otherwise render as a silently blank cell.
    (root / "uncached.xlsx").write_bytes(
        _xlsx_bytes({"S": [["item", "amt"], [SECRET, 1], ["total", "=SUM(B2:B2)"]]})
    )
    settings = _settings(tmp_path, [root])
    with pytest.raises(FileAccessError) as excinfo:
        read_text_file(root / "uncached.xlsx", settings=settings)
    message = str(excinfo.value)
    assert "no cached value" in message
    assert SECRET not in message


@needs_openpyxl
def test_xlsx_table_returns_padded_rows(tmp_path, root):
    (root / "one.xlsx").write_bytes(
        _xlsx_bytes({"S": [["date", "amount", "memo"], ["2026-08-01", 12.5]]})
    )
    settings = _settings(tmp_path, [root])
    assert read_table(root / "one.xlsx", settings=settings) == [
        ["date", "amount", "memo"],
        ["2026-08-01", "12.5", ""],
    ]


@needs_openpyxl
def test_xlsx_table_refuses_a_workbook_with_several_populated_sheets(tmp_path, root):
    (root / "two.xlsx").write_bytes(
        _xlsx_bytes({"Jan": [["a"], [SECRET]], "Feb": [["a"], ["2"]]})
    )
    settings = _settings(tmp_path, [root])
    with pytest.raises(FileAccessError) as excinfo:
        read_table(root / "two.xlsx", settings=settings)
    message = str(excinfo.value)
    assert "more than one worksheet" in message and "'Jan'" in message
    assert SECRET not in message  # sheet NAMES are fine to quote; cells are not


@needs_openpyxl
def test_xlsx_table_ignores_an_empty_second_sheet(tmp_path, root):
    (root / "plus-blank.xlsx").write_bytes(_xlsx_bytes({"Data": [["a"], ["1"]], "Blank": []}))
    settings = _settings(tmp_path, [root])
    assert read_table(root / "plus-blank.xlsx", settings=settings) == [["a"], ["1"]]


@needs_openpyxl
def test_corrupt_xlsx_is_refused_without_quoting_it(tmp_path, root):
    (root / "fake.xlsx").write_bytes(b"PK\x03\x04 not a workbook " + SECRET.encode())
    settings = _settings(tmp_path, [root])
    with pytest.raises(FileAccessError) as excinfo:
        read_text_file(root / "fake.xlsx", settings=settings)
    assert "not a readable XLSX" in str(excinfo.value)
    assert SECRET not in str(excinfo.value)


def test_xlsx_without_the_extra_names_the_format_and_the_extra(tmp_path, root, monkeypatch):
    (root / "book.xlsx").write_bytes(b"anything")
    settings = _settings(tmp_path, [root])
    _hide_module(monkeypatch, "openpyxl")
    with pytest.raises(FileAccessError) as excinfo:
        read_text_file(root / "book.xlsx", settings=settings)
    message = str(excinfo.value)
    assert "XLSX" in message and "openpyxl" in message
    assert "uv sync --extra files" in message


def test_xlsx_table_without_the_extra_also_refuses_cleanly(tmp_path, root, monkeypatch):
    (root / "book.xlsx").write_bytes(b"anything")
    settings = _settings(tmp_path, [root])
    _hide_module(monkeypatch, "openpyxl")
    with pytest.raises(FileAccessError) as excinfo:
        read_table(root / "book.xlsx", settings=settings)
    assert "uv sync --extra files" in str(excinfo.value)


# -- PDF ---------------------------------------------------------------------------------


@needs_pypdf
def test_pdf_text_layer_is_extracted(tmp_path, root):
    lines = ["Statement for August 2026", "Opening balance 1234.56", "Closing 1300.00"]
    (root / "aug.pdf").write_bytes(_pdf_bytes([lines]))
    settings = _settings(tmp_path, [root])
    out = read_text_file(root / "aug.pdf", settings=settings)
    assert "Statement for August 2026" in out
    assert "Closing 1300.00" in out


@needs_pypdf
def test_pdf_pages_are_concatenated_in_order(tmp_path, root):
    (root / "two.pdf").write_bytes(
        _pdf_bytes(
            [
                ["Page one line about the account and its opening balance figure"],
                ["Page two line about the account and its closing balance figure"],
            ]
        )
    )
    settings = _settings(tmp_path, [root])
    out = read_text_file(root / "two.pdf", settings=settings)
    assert out.index("Page one") < out.index("Page two")


@needs_pypdf
def test_scanned_pdf_is_refused_rather_than_returning_nothing(tmp_path, root):
    # Same document, minus every text-drawing operator: a valid PDF whose pages are images
    # as far as a text extractor is concerned. Returning "" here is the failure this guards.
    (root / "scan.pdf").write_bytes(_pdf_bytes([[], []]))
    settings = _settings(tmp_path, [root])
    with pytest.raises(FileAccessError) as excinfo:
        read_text_file(root / "scan.pdf", settings=settings)
    message = str(excinfo.value)
    assert "no text layer" in message
    assert "OCR" in message


@needs_pypdf
def test_pdf_with_only_a_stamp_of_text_is_still_refused(tmp_path, root):
    # A scan often carries a few characters from a page-number or "COPY" overlay; that must
    # not be mistaken for a text layer.
    (root / "stamped.pdf").write_bytes(_pdf_bytes([["p 1"], ["p 2"]]))
    settings = _settings(tmp_path, [root])
    with pytest.raises(FileAccessError) as excinfo:
        read_text_file(root / "stamped.pdf", settings=settings)
    assert "no text layer" in str(excinfo.value)


def test_corrupt_pdf_is_refused_without_quoting_it(tmp_path, root):
    (root / "broken.pdf").write_bytes(b"%PDF-1.7 " + SECRET.encode() + b" not really\n")
    settings = _settings(tmp_path, [root])
    with pytest.raises(FileAccessError) as excinfo:
        read_text_file(root / "broken.pdf", settings=settings)
    assert SECRET not in str(excinfo.value)


def test_pdf_without_the_extra_names_the_format_and_the_extra(tmp_path, root, monkeypatch):
    (root / "aug.pdf").write_bytes(b"%PDF-1.4\n")
    settings = _settings(tmp_path, [root])
    _hide_module(monkeypatch, "pypdf")
    with pytest.raises(FileAccessError) as excinfo:
        read_text_file(root / "aug.pdf", settings=settings)
    message = str(excinfo.value)
    assert "PDF" in message and "pypdf" in message
    assert "uv sync --extra files" in message


def test_pdf_is_not_readable_as_a_table(tmp_path, root):
    (root / "aug.pdf").write_bytes(b"%PDF-1.4\n")
    settings = _settings(tmp_path, [root])
    with pytest.raises(FileAccessError) as excinfo:
        read_table(root / "aug.pdf", settings=settings)
    assert "unsupported file type" in str(excinfo.value)


# -- read_table: the shared gate ---------------------------------------------------------


def test_read_table_goes_through_the_same_allowlist(tmp_path, root):
    (tmp_path / "outside.csv").write_text(f"memo\n{SECRET}\n")
    settings = _settings(tmp_path, [root])
    with pytest.raises(FileAccessError) as excinfo:
        read_table(tmp_path / "outside.csv", settings=settings)
    assert "outside every allowed root" in str(excinfo.value)
    assert SECRET not in str(excinfo.value)


def test_read_table_denies_when_roots_unset(tmp_path, root):
    (root / "a.csv").write_text("a\n1\n")
    with pytest.raises(FileAccessError) as excinfo:
        read_table(root / "a.csv", settings=_settings(tmp_path, []))
    assert "HEARTH_FILE_ROOTS" in str(excinfo.value)


def test_read_table_refuses_a_non_tabular_format(tmp_path, root):
    settings = _settings(tmp_path, [root])
    with pytest.raises(FileAccessError) as excinfo:
        read_table(root / "note.txt", settings=settings)
    message = str(excinfo.value)
    assert "unsupported file type" in message and ".csv" in message


def test_read_table_refuses_a_directory(tmp_path, root):
    with pytest.raises(FileAccessError):
        read_table(root, settings=_settings(tmp_path, [root]))


def test_read_table_pads_ragged_csv_rows(tmp_path, root):
    (root / "ragged.csv").write_text("a,b,c\n1,2\n\n3,4,5,6\n")
    settings = _settings(tmp_path, [root])
    assert read_table(root / "ragged.csv", settings=settings) == [
        ["a", "b", "c", ""],
        ["1", "2", "", ""],
        ["3", "4", "5", "6"],
    ]


@pytest.mark.parametrize("suffix", [".csv", ".json", ".xlsx"])
def test_oversized_file_is_rejected_before_any_parser_runs(tmp_path, root, monkeypatch, suffix):
    # Poison the optional-parser import: if the size cap ran *after* the handler, this blows
    # up with AssertionError instead of the refusal we assert on. The 500 MB workbook case.
    def never(*args, **kwargs):
        raise AssertionError("a parser was reached before the size cap")

    monkeypatch.setattr(files_module, "_import_optional", never)
    big = root / f"big{suffix}"
    big.write_text(SECRET + "x" * 5000)
    settings = _settings(tmp_path, [root], file_max_bytes=1024)
    for read in (read_text_file, read_table):
        with pytest.raises(FileAccessError) as excinfo:
            read(big, settings=settings)
        assert "too large" in str(excinfo.value)
        assert SECRET not in str(excinfo.value)


def test_oversized_pdf_is_rejected_before_pypdf_runs(tmp_path, root, monkeypatch):
    def never(*args, **kwargs):
        raise AssertionError("pypdf was reached before the size cap")

    monkeypatch.setattr(files_module, "_import_optional", never)
    big = root / "big.pdf"
    big.write_bytes(SECRET.encode() + b"x" * 5000)
    settings = _settings(tmp_path, [root], file_max_bytes=1024)
    with pytest.raises(FileAccessError) as excinfo:
        read_text_file(big, settings=settings)
    assert "too large" in str(excinfo.value)
    assert SECRET not in str(excinfo.value)


def test_no_refusal_for_any_format_ever_quotes_the_file(tmp_path, root):
    """The canary sweep: every new format's failure mode, checked for content leakage."""
    settings = _settings(tmp_path, [root])
    canary = SECRET.encode()
    cases = {
        "leak.txt": b"\x00" + canary,  # binary in a text file
        "leak.json": b'{"m": "' + canary + b'",}',  # trailing comma
        "leak.xlsx": b"PK\x03\x04" + canary,  # not a workbook
        "leak.pdf": b"%PDF-1.7 " + canary,  # not a PDF
        "leak.wat": canary,  # unsupported extension
    }
    for name, payload in cases.items():
        (root / name).write_bytes(payload)
        with pytest.raises(FileAccessError) as excinfo:
            read_text_file(root / name, settings=settings)
        assert SECRET not in str(excinfo.value), name
        assert repr(str(root / name)) in str(excinfo.value) or "unsupported" in str(
            excinfo.value
        ), name


# -- the MCP surface picks the new formats up for free -----------------------------------


def test_summarize_file_handles_json(tmp_path, root):
    (root / "acct.json").write_text('{"balance": 1300.0, "owner": "A. Person"}')
    tools = _tools(_settings(tmp_path, [root]))
    out = tools.summarize_file(str(root / "acct.json"))
    assert '"balance": 1300.0' in out  # pretty-printed form rode through the local prompt


@needs_openpyxl
def test_classify_file_handles_xlsx(tmp_path, root):
    (root / "book.xlsx").write_bytes(_xlsx_bytes({"S": [["memo"], ["annual fee"]]}))
    tools = _tools(_settings(tmp_path, [root]))
    out = tools.classify_file(str(root / "book.xlsx"), labels=["bank", "other"])
    assert "annual fee" in out


@needs_pypdf
def test_extract_file_handles_pdf(tmp_path, root):
    (root / "aug.pdf").write_bytes(
        _pdf_bytes([["Statement for August 2026 with a closing balance of 1300.00"]])
    )
    tools = _tools(_settings(tmp_path, [root]))
    out = tools.extract_file(str(root / "aug.pdf"), fields=["closing balance"])
    assert set(out.keys()) == {"closing balance"}


@needs_pypdf
def test_summarize_file_refuses_a_scanned_pdf_instead_of_summarizing_nothing(tmp_path, root):
    (root / "scan.pdf").write_bytes(_pdf_bytes([[], []]))
    tools = _tools(_settings(tmp_path, [root]))
    with pytest.raises(FileAccessError) as excinfo:
        tools.summarize_file(str(root / "scan.pdf"))
    assert "OCR" in str(excinfo.value)
