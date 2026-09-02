"""Allowlisted local file reading for the path-taking MCP tools (docs/PRIVACY.md).

This module exists to close **the caller caveat**: the text-taking tools
(:meth:`hearth.mcp.tools.HearthTools.summarize` and friends) require the calling agent to
read a file into *its own* context before handing it over, so a confidential file has
already left the machine before HEARTH ever sees it. The path-taking variants take a
*path* instead and let HEARTH open the file locally — the agent never holds a byte.

That inverts the trust model: a path-taking tool is an **arbitrary-file-read primitive
exposed to an agent**, so every read goes through :func:`_gated_bytes` — the one gate both
:func:`read_text_file` and :func:`read_table` sit behind — which enforces:

  * **Deny by default** — reads are refused entirely unless ``HEARTH_FILE_ROOTS``
    (``Settings.file_roots``) names at least one existing directory. There is no implicit
    root, not even the CWD or ``$HOME``.
  * **Full resolution before the check** — the request is ``expanduser()``-ed and
    ``resolve()``-d (which flattens ``..`` *and* follows every symlink) and the result must
    be genuinely inside a resolved root, so neither traversal nor a symlink planted inside
    a root can escape it.
  * **Regular files only, size-capped** — directories, devices and FIFOs are refused, and a
    file larger than ``Settings.file_max_bytes`` is refused rather than truncated. The cap
    is enforced *before* any parser runs, so a hostile 500 MB workbook is never handed to
    openpyxl in the first place.
  * **Errors that never quote the file** — every message here is built from the *requested*
    path, the reason, and sizes; file content never appears in an exception, because that
    exception travels back to the very agent we are keeping the content away from.

Two shapes come out of that gate. :func:`read_text_file` returns *prose for a prompt* —
text, CSV, JSON, XLSX and text-layer PDF, each normalized so the local model sees a clean
document instead of quoting noise. :func:`read_table` returns *rows of cells* for the
tabular subset (CSV, XLSX, and JSON that is an array of objects), for callers that want to
compute over the data rather than describe it.

An empty result is treated differently by the two: :func:`read_table` may honestly return
``[]`` (its caller can see that and decide), while :func:`read_text_file` **refuses**
wherever "empty" would be a lie — a scanned PDF with no text layer, or a workbook whose
formula cells were never evaluated. Handing a model a blank document produces a confident
summary of nothing, which is worse than an error.

Format handling dispatches on suffix through :data:`_READERS` and :data:`_TABLE_READERS`;
a new format is one handler plus one entry. Handlers are dependency-free or import their
library **lazily inside the handler**, so the core install stays light — ``openpyxl`` and
``pypdf`` live behind the ``[files]`` extra and a missing one surfaces as a
:class:`FileAccessError` naming the extra, never a raw ``ImportError``.
"""

from __future__ import annotations

import csv
import importlib
import io
import json
import stat
from collections.abc import Callable, Mapping
from datetime import date, datetime, time
from itertools import zip_longest
from pathlib import Path
from typing import TypeVar

from ..config import Settings, get_settings

# A text-layer PDF yields hundreds of characters per page. A scan yields nothing at all, or
# a handful of characters from a stamp, a page number or a signature overlay. 32
# non-whitespace characters per page sits far below any real page of prose and far above
# that noise floor, so it separates "machine-generated" from "needs OCR" without tuning.
_PDF_MIN_CHARS_PER_PAGE = 32

# The pip extra that carries the optional parsers, quoted in every missing-dependency error
# so the operator is told the fix rather than left with a traceback.
_FILES_EXTRA = "uv sync --extra files"

#: A handler that renders a file's bytes as text for a prompt.
TextReader = Callable[[bytes, str], str]
#: A handler that renders a file's bytes as rows of cells.
TableReader = Callable[[bytes, str], list[list[str]]]

_Reader = TypeVar("_Reader")


class FileAccessError(ValueError):
    """A file read was refused (outside the allowlist, too large, wrong type, unreadable).

    Subclasses :class:`ValueError` to match the argument-validation errors the other tools
    raise, so the MCP layer surfaces all of them the same way. The message is always safe
    to hand back to the caller: it names the path and the reason, never the content.
    """


def allowed_roots(settings: Settings | None = None) -> list[Path]:
    """Return the resolved directories file reads are permitted under.

    Parses the colon-separated ``HEARTH_FILE_ROOTS`` (``Settings.file_roots``), dropping
    blanks and any entry that isn't an existing directory — a typo'd root must not silently
    widen the allowlist, and a non-existent one can't contain a file anyway. An empty result
    means *deny everything*; callers treat it as the deny-by-default state.
    """
    settings = settings or get_settings()
    roots: list[Path] = []
    for entry in settings.file_roots.split(":"):
        entry = entry.strip()
        if not entry:
            continue
        root = Path(entry).expanduser().resolve()
        if root.is_dir() and root not in roots:
            roots.append(root)
    return roots


def resolve_under_roots(path: str | Path, settings: Settings | None = None) -> Path:
    """Fully resolve ``path`` and return it only if it lies inside an allowed root.

    This is the security gate; :func:`_gated_bytes` calls it before touching the file.
    ``resolve()`` collapses ``..`` and follows symlinks, so the containment check runs on
    the *real* target — a ``../../etc/passwd`` or a symlink planted inside a root both fail
    it. Raises :class:`FileAccessError` if roots are unconfigured or the path escapes them.
    """
    settings = settings or get_settings()
    roots = allowed_roots(settings)
    if not roots:
        raise FileAccessError(
            "file reads are disabled: no readable directory in HEARTH_FILE_ROOTS. "
            "Set it to a colon-separated list of directories HEARTH may read under, "
            "e.g. HEARTH_FILE_ROOTS=/Users/you/statements:/Users/you/work"
        )

    requested = str(path)
    resolved = Path(requested).expanduser().resolve()
    if not any(resolved == root or resolved.is_relative_to(root) for root in roots):
        # Report the path as asked for, not as resolved: where a symlink pointed is itself
        # information about the filesystem we're gating access to.
        raise FileAccessError(
            f"path is outside every allowed root in HEARTH_FILE_ROOTS: {requested!r}"
        )
    return resolved


def _gated_bytes(
    path: str | Path,
    settings: Settings | None,
    readers: Mapping[str, _Reader],
    what: str,
) -> tuple[_Reader, bytes, str]:
    """Run the whole security gate and return the format handler plus the file's bytes.

    Factored out so the two public readers cannot drift apart: there is exactly **one**
    place that resolves a path against the allowlist, refuses a non-regular file, and
    enforces the size cap — and it enforces that cap *before* a single byte reaches a
    parser, which is the difference between refusing a 500 MB workbook and feeding it to
    openpyxl. ``readers`` selects which format table applies and ``what`` phrases the
    unsupported-format refusal ("read" vs "read as a table").

    The format lookup deliberately happens first: an unsupported extension is a fact about
    the *request*, so it is worth reporting even for a file that also doesn't exist.
    """
    settings = settings or get_settings()
    requested = str(path)
    resolved = resolve_under_roots(requested, settings)

    reader = _reader_for(resolved.suffix, readers, what)

    try:
        info = resolved.stat()
    except OSError:
        # Covers missing, unreadable, and dangling-symlink targets alike; deliberately not
        # distinguishing them, so a denied caller can't probe the filesystem by error text.
        raise FileAccessError(f"file not found or not readable: {requested!r}") from None

    if stat.S_ISDIR(info.st_mode):
        raise FileAccessError(f"path is a directory, not a file: {requested!r}")
    if not stat.S_ISREG(info.st_mode):
        raise FileAccessError(f"path is not a regular file: {requested!r}")

    limit = settings.file_max_bytes
    if info.st_size > limit:
        raise FileAccessError(
            f"file is too large: {info.st_size} bytes exceeds the {limit}-byte limit "
            "(raise HEARTH_FILE_MAX_BYTES, or point the tool at a smaller file)"
        )

    try:
        with resolved.open("rb") as handle:
            # Read one byte past the cap so a file that grew (or misreported its size)
            # between stat() and open() is still refused rather than slurped whole.
            data = handle.read(limit + 1)
    except OSError:
        raise FileAccessError(f"file could not be read: {requested!r}") from None
    if len(data) > limit:
        raise FileAccessError(f"file is too large: exceeds the {limit}-byte limit")

    return reader, data, requested


def read_text_file(path: str | Path, settings: Settings | None = None) -> str:
    """Read ``path`` as text, enforcing the allowlist, the size cap, and the format table.

    The entry point the path-taking tools use. Returns the file's text, normalized per
    format: CSV and XLSX become one ``a | b | c`` line per row so quoting noise never
    reaches the model, JSON is re-rendered pretty-printed with sorted keys, and a PDF gives
    up its text layer. Raises :class:`FileAccessError` for every refusal — content never
    appears in the message.
    """
    reader, data, requested = _gated_bytes(path, settings, _READERS, "read")
    return reader(data, requested)


def read_table(path: str | Path, settings: Settings | None = None) -> list[list[str]]:
    """Read ``path`` as rows of cells, through the identical gate as :func:`read_text_file`.

    For the tabular formats — ``.csv``, ``.xlsx``, and ``.json`` that is an array of objects
    — this returns the grid rather than prose, for callers that compute over the data
    instead of describing it. Every row is padded to the same width, so ``rows[0]`` is a
    usable header and ``row[i]`` means the same column on every row; a cell that is absent
    or empty is ``""``. Wholly blank rows are dropped, exactly as :func:`read_text_file`
    drops them.

    A non-tabular format (text, Markdown, PDF) is refused by name rather than flattened
    into a one-column table. An input that is genuinely empty returns ``[]`` — unlike text
    headed for a prompt, an empty list is visibly empty to the caller, who can decide what
    that means.
    """
    reader, data, requested = _gated_bytes(path, settings, _TABLE_READERS, "read as a table")
    return reader(data, requested)


# -- format handlers ------------------------------------------------------------------
#
# Each handler takes the raw bytes plus the *requested* path (for error messages only) and
# returns text (or rows). To add a format later, write one handler and add its suffix to
# the table below; nothing else in this module or in tools.py needs to change. Keep
# handlers dependency-free or import their library lazily inside the handler, so the core
# install stays light (the same rule providers/remote.py and memory/embed.py follow).


def _read_plain(data: bytes, requested: str) -> str:
    """Decode ``data`` as UTF-8 text."""
    return _decode(data, requested)


def _read_csv(data: bytes, requested: str) -> str:
    """Decode ``data`` as CSV and render one ``a | b | c`` line per row.

    Normalizing here means the local model sees a clean table instead of raw quoting, and
    an embedded newline inside a quoted field can't masquerade as a row break.
    """
    return _render_rows(_csv_rows(data, requested))


def _read_json(data: bytes, requested: str) -> str:
    """Decode ``data`` as JSON and re-render it **pretty-printed with sorted keys**.

    The model is shown the normalized document, not the file's byte-for-byte formatting.
    Pretty over compact because this text is going into a *prompt*: a two-space indent and
    one value per line are what let a small local model see where an object ends, whereas a
    minified document is a single unreadable line that also tokenizes worse. ``sort_keys``
    makes two runs over equivalent data produce the identical prompt, so a summary doesn't
    drift just because the writer emitted its keys in a different order. Non-ASCII is left
    as itself rather than ``\\uXXXX``-escaped, for the same readability reason.
    """
    text = _decode(data, requested)
    return json.dumps(_parse_json(text, requested), indent=2, sort_keys=True, ensure_ascii=False)


def _read_xlsx(data: bytes, requested: str) -> str:
    """Render every worksheet of an XLSX workbook as ``a | b | c`` rows, like CSV.

    Each sheet is preceded by a ``# Sheet: <title>`` line — always, not just when there are
    several. The title is often the only thing saying what the grid *is* ("Transactions",
    "Q3 Actuals"), and a model reading a bare block of numbers has no other way to know.
    """
    blocks = []
    for title, rows in _xlsx_sheets(data, requested):
        body = _render_rows(rows)
        blocks.append(f"# Sheet: {title}\n{body}" if body else f"# Sheet: {title}")
    return "\n\n".join(blocks)


def _read_pdf(data: bytes, requested: str) -> str:
    """Extract a PDF's **text layer** — and refuse a scan rather than returning near-nothing.

    HEARTH does not OCR. For a machine-generated PDF (a bank statement, an invoice, an
    exported report) that costs nothing: the glyphs carry their characters and pypdf hands
    them back. For a *scanned* page there is no text layer at all, and ``extract_text()``
    cheerfully returns ``""``. Passing that on would hand the model an empty document and
    get back a fluent summary of nothing — a confabulation the caller has no way to detect,
    since the tool reported success. So we count what came out and refuse below
    :data:`_PDF_MIN_CHARS_PER_PAGE` non-whitespace characters per page, naming OCR as the
    missing step.
    """
    pypdf = _import_optional("pypdf", "PDF")
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            # An empty user password is common on statements; anything else we can't open.
            if not reader.decrypt(""):
                raise FileAccessError(
                    f"PDF is password-protected and cannot be opened: {requested!r}"
                )
        pages = [(page.extract_text() or "") for page in reader.pages]
    except FileAccessError:
        raise
    except Exception:
        # pypdf raises a wide family (PdfReadError, struct/zlib/KeyError...) on a damaged or
        # non-PDF file, and their messages can quote bytes from the file. Never re-raise one.
        raise FileAccessError(f"file is not a readable PDF: {requested!r}") from None

    if not pages:
        raise FileAccessError(f"PDF has no pages: {requested!r}")

    extracted = sum(len("".join(page.split())) for page in pages)
    if extracted < _PDF_MIN_CHARS_PER_PAGE * len(pages):
        raise FileAccessError(
            f"PDF appears to have no text layer (only {extracted} characters across "
            f"{len(pages)} page(s)): {requested!r}. It is most likely a scan, and reading it "
            "would require OCR, which HEARTH does not do — convert it to a text-layer PDF "
            "(or export the source data as CSV/XLSX) and try again"
        )
    return "\n\n".join(stripped for page in pages if (stripped := page.strip()))


# -- table handlers ---------------------------------------------------------------------


def _table_csv(data: bytes, requested: str) -> list[list[str]]:
    """Return a CSV's rows of cells, padded to a common width."""
    return _pad(_csv_rows(data, requested))


def _table_xlsx(data: bytes, requested: str) -> list[list[str]]:
    """Return the rows of the workbook's single data-bearing worksheet.

    A ``list[list[str]]`` is one grid, so a workbook with data on several sheets is
    *ambiguous*, not tabular — and quietly returning sheet 1 of 3 would drop the rest with
    no way for the caller to notice. Refuse instead and name the sheets;
    :func:`read_text_file` renders all of them when the caller wants everything.
    """
    populated = [(title, rows) for title, rows in _xlsx_sheets(data, requested) if rows]
    if len(populated) > 1:
        names = ", ".join(repr(title) for title, _ in populated)
        raise FileAccessError(
            f"workbook has data on more than one worksheet ({names}): {requested!r}. "
            "read_table returns a single grid — read it with read_text_file (which renders "
            "every sheet) or split the sheet you want into its own file"
        )
    return _pad(populated[0][1]) if populated else []


def _table_json(data: bytes, requested: str) -> list[list[str]]:
    """Return an array-of-objects JSON document as a header row plus one row per object.

    The header is the **union of every object's keys, sorted** — union so a field that only
    some records carry still gets a column instead of being silently dropped, sorted so the
    column order is a property of the data rather than of whichever record happened to come
    first. A key a given object lacks becomes ``""``, which is what makes the ragged input
    line up. Nested values are re-serialized compactly so one record still occupies one row.

    Anything else at the top level — an object, a bare array of scalars — is refused by
    *shape*, never by content: there is no honest header row to derive from it.
    """
    parsed = _parse_json(_decode(data, requested), requested)
    if not isinstance(parsed, list):
        raise FileAccessError(
            f"JSON is not an array of objects (top level is a {_shape(parsed)}): {requested!r}. "
            "read_table needs a list of records; read it with read_text_file instead"
        )
    if not parsed:
        return []
    if not all(isinstance(item, dict) for item in parsed):
        offenders = sorted({_shape(item) for item in parsed if not isinstance(item, dict)})
        raise FileAccessError(
            f"JSON array contains non-object entries ({', '.join(offenders)}): {requested!r}. "
            "read_table needs every entry to be an object with named fields"
        )

    header = sorted({str(key) for item in parsed for key in item})
    rows = [list(header)]
    rows.extend([_json_cell(item.get(key)) for key in header] for item in parsed)
    return rows


# -- shared helpers ---------------------------------------------------------------------


def _import_optional(module: str, fmt: str):
    """Import an optional parser, turning a missing ``[files]`` extra into a clear refusal.

    Deferred to call time so the core install carries neither openpyxl nor pypdf. A raw
    ``ImportError`` would surface to the agent as an internal failure it can't act on; this
    tells it (and the operator behind it) exactly which format wanted what, and the one
    command that fixes it.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise FileAccessError(
            f"reading {fmt} needs the optional {module!r} dependency, which is not "
            f"installed. Install the file-format extra with: {_FILES_EXTRA}"
        ) from exc


def _csv_rows(data: bytes, requested: str) -> list[list[str]]:
    """Parse ``data`` as CSV into stripped cells, dropping wholly blank rows."""
    text = _decode(data, requested)
    try:
        rows = list(csv.reader(io.StringIO(text, newline="")))
    except csv.Error:
        raise FileAccessError(f"file is not valid CSV: {requested!r}") from None
    return [
        [cell.strip() for cell in row] for row in rows if any(cell.strip() for cell in row)
    ]


def _xlsx_sheets(data: bytes, requested: str) -> list[tuple[str, list[list[str]]]]:
    """Return ``[(worksheet title, rows of cells)]`` for an XLSX workbook.

    The workbook is loaded **twice**, on purpose. ``data_only=True`` is the pass we take
    values from: a formula cell yields the value Excel cached the last time it recalculated,
    which is the only useful thing to show a reader — ``=SUM(B2:B40)`` says nothing about
    the total. But a workbook written by a library and never opened by a spreadsheet app has
    no cached values at all, and in that mode an uncached formula and an empty cell are
    *indistinguishable*: both come back ``None``. So we also load with ``data_only=False``,
    where a formula cell yields its ``=...`` text, and refuse the file if any cell is a
    formula whose value was never cached. Emitting those as blanks is how a model ends up
    confidently summarizing totals that were never computed.

    ``read_only=True`` streams each sheet instead of building the full object model, so the
    size cap (already enforced before we get here) genuinely bounds the work.
    """
    openpyxl = _import_optional("openpyxl", "XLSX workbooks")
    try:
        values_wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        formula_wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=False)
    except Exception:
        # openpyxl raises BadZipFile / InvalidFileException / KeyError on a non-workbook, and
        # those messages can quote file bytes. Replace, never wrap.
        raise FileAccessError(f"file is not a readable XLSX workbook: {requested!r}") from None

    try:
        sheets: list[tuple[str, list[list[str]]]] = []
        # Same file, loaded twice, so the sheet lists match by construction; strict=True
        # states that invariant instead of silently truncating if it ever stopped holding.
        for values_ws, formula_ws in zip(
            values_wb.worksheets, formula_wb.worksheets, strict=True
        ):
            rows: list[list[str]] = []
            value_rows = values_ws.iter_rows(values_only=True)
            formula_rows = formula_ws.iter_rows(values_only=True)
            # zip_longest, not zip: a read-only sheet can report a different used width in
            # the two passes, and the short one must not decide how many cells we keep.
            for value_row, formula_row in zip_longest(value_rows, formula_rows, fillvalue=()):
                cells = [
                    _xlsx_cell(value, formula, requested)
                    for value, formula in zip_longest(value_row, formula_row, fillvalue=None)
                ]
                if any(cell for cell in cells):
                    rows.append(cells)
            sheets.append((values_ws.title, rows))
        return sheets
    finally:
        values_wb.close()
        formula_wb.close()


def _xlsx_cell(value: object, formula: object, requested: str) -> str:
    """Render one workbook cell, refusing a formula whose value was never cached."""
    if value is None and isinstance(formula, str) and formula.startswith("="):
        raise FileAccessError(
            f"workbook contains formula cells with no cached value: {requested!r}. The file "
            "has not been recalculated (a workbook written by a script and never opened in "
            "Excel or LibreOffice stores no results), so those cells have no value to read "
            "— open and save it in a spreadsheet app, or export it to CSV, and try again"
        )
    return _cell_text(value)


def _cell_text(value: object) -> str:
    """Render one spreadsheet cell as text; a genuinely empty cell becomes ``''``."""
    if value is None:
        return ""
    if isinstance(value, datetime):
        # openpyxl gives a date-typed cell a midnight time it never had; don't show it.
        return value.date().isoformat() if value.time() == time.min else value.isoformat(" ")
    if isinstance(value, date):
        return value.isoformat()
    return str(value).strip()


def _json_cell(value: object) -> str:
    """Render one JSON value as a table cell; ``null``/absent becomes ``''``.

    A nested object or array is re-serialized **compactly** (and key-sorted) so a record
    still occupies exactly one row — the opposite choice from :func:`_read_json`, because
    here the cell is a grid position, not prose.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return str(value)


def _parse_json(text: str, requested: str) -> object:
    """Parse ``text`` as JSON, refusing without echoing the offending document."""
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        # json's own message quotes the position and can carry a fragment of the document;
        # ours names only the path.
        raise FileAccessError(f"file is not valid JSON: {requested!r}") from None


def _shape(value: object) -> str:
    """Name a JSON value's *type* for an error message — never its content."""
    return {
        dict: "object",
        list: "array",
        str: "string",
        bool: "boolean",
        int: "number",
        float: "number",
        type(None): "null",
    }.get(type(value), "value")


def _pad(rows: list[list[str]]) -> list[list[str]]:
    """Pad every row to the widest one, so ``row[i]`` is the same column on every row."""
    if not rows:
        return []
    width = max(len(row) for row in rows)
    return [row + [""] * (width - len(row)) for row in rows]


def _render_rows(rows: list[list[str]]) -> str:
    """Render rows of cells as one ``a | b | c`` line each (the shared CSV/XLSX shape)."""
    return "\n".join(" | ".join(row) for row in rows)


_READERS: dict[str, TextReader] = {
    "": _read_plain,  # extension-less files (README, LICENSE, a dumped log)
    ".txt": _read_plain,
    ".text": _read_plain,
    ".md": _read_plain,
    ".markdown": _read_plain,
    ".rst": _read_plain,
    ".log": _read_plain,
    ".csv": _read_csv,
    ".json": _read_json,
    ".xlsx": _read_xlsx,
    ".pdf": _read_pdf,
}

_TABLE_READERS: dict[str, TableReader] = {
    ".csv": _table_csv,
    ".json": _table_json,
    ".xlsx": _table_xlsx,
}


def _reader_for(suffix: str, readers: Mapping[str, _Reader], what: str) -> _Reader:
    """Return the handler for ``suffix``, or raise naming the unsupported format."""
    reader = readers.get(suffix.lower())
    if reader is None:
        supported = ", ".join(sorted(s for s in readers if s))
        extensionless = " (and extension-less text)" if "" in readers else ""
        raise FileAccessError(
            f"unsupported file type {suffix or '(none)'!r}: HEARTH can {what} {supported}"
            f"{extensionless} so far — no handler is registered for this format"
        )
    return reader


def _decode(data: bytes, requested: str) -> str:
    """Decode ``data`` as UTF-8, refusing binary rather than returning mojibake."""
    if b"\x00" in data:  # NUL byte ⇒ almost certainly binary
        raise FileAccessError(f"file is not UTF-8 text: {requested!r}")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        # Never echo the offending bytes — the decoder's own message would quote them.
        raise FileAccessError(f"file is not UTF-8 text: {requested!r}") from None


__all__ = [
    "FileAccessError",
    "allowed_roots",
    "read_table",
    "read_text_file",
    "resolve_under_roots",
]
