"""Allowlisted local file reading for the path-taking MCP tools (docs/PRIVACY.md).

This module exists to close **the caller caveat**: the text-taking tools
(:meth:`hearth.mcp.tools.HearthTools.summarize` and friends) require the calling agent to
read a file into *its own* context before handing it over, so a confidential file has
already left the machine before HEARTH ever sees it. The path-taking variants take a
*path* instead and let HEARTH open the file locally — the agent never holds a byte.

That inverts the trust model: a path-taking tool is an **arbitrary-file-read primitive
exposed to an agent**, so every read goes through :func:`read_text_file`, which enforces:

  * **Deny by default** — reads are refused entirely unless ``HEARTH_FILE_ROOTS``
    (``Settings.file_roots``) names at least one existing directory. There is no implicit
    root, not even the CWD or ``$HOME``.
  * **Full resolution before the check** — the request is ``expanduser()``-ed and
    ``resolve()``-d (which flattens ``..`` *and* follows every symlink) and the result must
    be genuinely inside a resolved root, so neither traversal nor a symlink planted inside
    a root can escape it.
  * **Regular files only, size-capped** — directories, devices and FIFOs are refused, and a
    file larger than ``Settings.file_max_bytes`` is refused rather than truncated.
  * **Errors that never quote the file** — every message here is built from the *requested*
    path, the reason, and sizes; file content never appears in an exception, because that
    exception travels back to the very agent we are keeping the content away from.

Format handling dispatches on suffix through :data:`_READERS`; plain text and CSV ship
today, and a new format (PDF/XLSX/JSON) is one ``bytes -> str`` handler plus one entry.
"""

from __future__ import annotations

import csv
import io
import stat
from collections.abc import Callable
from pathlib import Path

from ..config import Settings, get_settings


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

    This is the security gate; :func:`read_text_file` calls it before touching the file.
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


def read_text_file(path: str | Path, settings: Settings | None = None) -> str:
    """Read ``path`` as text, enforcing the allowlist, the size cap, and the format table.

    The single entry point the path-taking tools use. Returns the file's text (CSV is
    normalized to one ``a | b | c`` line per row so quoting noise doesn't reach the model).
    Raises :class:`FileAccessError` for every refusal — content never appears in the message.
    """
    settings = settings or get_settings()
    requested = str(path)
    resolved = resolve_under_roots(requested, settings)

    reader = _reader_for(resolved.suffix)

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

    return reader(data, requested)


# -- format handlers ------------------------------------------------------------------
#
# Each handler takes the raw bytes plus the *requested* path (for error messages only) and
# returns text. To add a format later — PDF, XLSX, JSON — write one handler and add its
# suffix here; nothing else in this module or in tools.py needs to change. Keep handlers
# dependency-free or import their library lazily inside the handler, so the core install
# stays light (the same rule providers/remote.py and memory/embed.py follow).


def _read_plain(data: bytes, requested: str) -> str:
    """Decode ``data`` as UTF-8 text."""
    return _decode(data, requested)


def _read_csv(data: bytes, requested: str) -> str:
    """Decode ``data`` as CSV and render one ``a | b | c`` line per row.

    Normalizing here means the local model sees a clean table instead of raw quoting, and
    an embedded newline inside a quoted field can't masquerade as a row break.
    """
    text = _decode(data, requested)
    try:
        rows = list(csv.reader(io.StringIO(text, newline="")))
    except csv.Error:
        raise FileAccessError(f"file is not valid CSV: {requested!r}") from None
    return "\n".join(
        " | ".join(cell.strip() for cell in row) for row in rows if any(c.strip() for c in row)
    )


_READERS: dict[str, Callable[[bytes, str], str]] = {
    "": _read_plain,  # extension-less files (README, LICENSE, a dumped log)
    ".txt": _read_plain,
    ".text": _read_plain,
    ".md": _read_plain,
    ".markdown": _read_plain,
    ".rst": _read_plain,
    ".log": _read_plain,
    ".csv": _read_csv,
}


def _reader_for(suffix: str) -> Callable[[bytes, str], str]:
    """Return the handler for ``suffix``, or raise naming the unsupported format."""
    reader = _READERS.get(suffix.lower())
    if reader is None:
        supported = ", ".join(sorted(s for s in _READERS if s))
        raise FileAccessError(
            f"unsupported file type {suffix or '(none)'!r}: HEARTH can read {supported} "
            "(and extension-less text) so far — no handler is registered for this format"
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


__all__ = ["FileAccessError", "allowed_roots", "read_text_file", "resolve_under_roots"]
