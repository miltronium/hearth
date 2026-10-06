"""The vetted local tools — thin wrappers over gates that already exist.

Nothing here is a new capability. Every tool routes to a HEARTH module that already owns the
relevant rule, so the agent inherits that rule instead of getting its own copy of it:

  * ``read_file`` and ``list_files`` go through :mod:`hearth.mcp.files`, so
    ``HEARTH_FILE_ROOTS`` still governs, deny-by-default still holds, symlinks are still
    resolved before the containment check, and the size cap still applies. If the allowlist
    is unset, the agent can read nothing — that is the same refusal an MCP client gets.
  * ``rag_search`` goes through the :class:`~hearth.memory.rag.RagIndex` the caller passes in.
  * The finance tools go through :class:`~hearth.finance.store.FinanceStore`, which computes
    in :class:`~decimal.Decimal` and refuses to return a figure its two independent sums
    disagree on. The model never sees an arithmetic problem; it sees a figure and the rows.

**Read-only, on purpose. There is no write tool and there will not be one by default.** A
bounded local model is acceptable at "find the number"; it is not acceptable at "and now
change something", because the two failure modes are not comparable. A wrong read produces a
wrong sentence the operator can check against the transcript. A wrong write produces a
corrupted file, a mis-categorised ledger, or a deleted note — and the agent will report it as
done. Writes belong to code the operator ran deliberately, or to a human acting on the
agent's transcript. ``docs/AGENT.md`` states this as a standing decision, not an oversight.

**No shell tool, no network tool, no `eval`.** ``tests/test_agent_no_network.py`` walks this
package's own source and fails if one appears — including as a string literal that looks like
a URL. The convenience that would break this is "just fetch that page for me"; the test is
there so that lands as a red build rather than as a quiet capability.

Every builder takes its collaborators as arguments. There is no module-level index, store or
settings object, so two agents in one process cannot reach each other's data.
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path
from typing import Any

from ..mcp.files import FileAccessError, allowed_roots, read_text_file, resolve_under_roots
from .tools import Tool, ToolParam, ToolRegistry

#: Cap on how many paths ``list_files`` will return. A listing is orientation, not data: a
#: model handed nine thousand paths spends its whole budget reading the listing.
DEFAULT_LIST_LIMIT = 200
# The tools gated by HEARTH_FILE_ROOTS: with no roots, every one of them refuses every call.
FILE_TOOLS = frozenset({"read_file", "list_files", "search_files"})
# search_files bounds: files opened, matches returned, characters per matching line.
DEFAULT_SEARCH_FILES = 500
DEFAULT_SEARCH_MATCHES = 40
SEARCH_SNIPPET_CHARS = 200
#: Most context lines search_files will show around a match. Each one costs observation space
#: that the loop caps (Budget.max_observation_chars), so this stays small.
MAX_SEARCH_CONTEXT = 3

#: Cap on how many ledger rows ``finance_rows`` will return in one call.
DEFAULT_ROW_LIMIT = 50


def _under_roots(path: str, settings: Any | None = None) -> str:
    """Resolve a possibly-relative ``path`` against the allowed roots.

    A model that has just seen ``BankA/2026/stmt-01.csv`` in a listing will naturally pass
    exactly that back, and a bare relative path resolves against the process's CWD — which is
    the repo, not the operator's statements — so the read was refused as "outside every
    allowed root" and the agent burned its budget arguing with a path it had been given. That
    is a usability bug pretending to be a security one.

    Relative candidates are tried against each root in order and the first that exists wins;
    an absolute path is passed through untouched. Nothing here widens the boundary: whatever
    comes out still goes through :func:`~hearth.mcp.files.resolve_under_roots`, so a
    ``../`` escape or a symlink leaving the root is refused exactly as before. This only
    supplies the prefix the model could not have known.
    """
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return path
    for root in allowed_roots(settings):
        if (root / candidate).exists():
            return str(root / candidate)
    return path  # unresolvable: let resolve_under_roots produce the real refusal


_RANGE_HEADER = re.compile(r"\[lines (\d+)-(\d+) of (\d+)\]")
_NUMBERED_LINE = re.compile(r"^(\d+): ", re.MULTILINE)


def _read_file_hint(shown: str, full: str) -> str:
    """Where to resume a cut-off ``read_file``, as a *file* line number.

    A whole-file read counts the newlines that were shown. A ranged read counts nothing
    itself: its lines carry their own numbers, and counting the observation's newlines said
    "line 60 of 60" for lines 253-312, which would send the model backwards. Either way the
    line the cut falls inside was not seen whole, so reading resumes *at* it.
    """
    header = _RANGE_HEADER.match(full)
    if header:
        numbered = _NUMBERED_LINE.findall(shown)
        next_line = int(numbered[-1]) if numbered else int(header.group(1))
        total = int(header.group(3))
    else:
        next_line = shown.count("\n") + 1
        total = full.count("\n") + (0 if full.endswith("\n") else 1)
    return (
        f"You saw up to part of line {next_line} of {total}. To read on, call read_file with "
        f"start_line={next_line} and max_lines=60 (start_line counts lines, not characters), "
        "or use search_files to find the line you need."
    )


def read_file_tool(*, settings: Any | None = None) -> Tool:
    """A tool that reads one allowlisted local file as text.

    The gate is :func:`hearth.mcp.files.read_text_file`, unchanged: outside
    ``HEARTH_FILE_ROOTS`` is a refusal, so is a directory, so is anything over the size cap,
    and the refusal message names the path and the reason but never the content — which
    matters more here than in MCP, because this message is written back into a prompt and
    then into a stored transcript.

    ``start_line`` / ``max_lines`` select a range, returned with line numbers that match
    ``search_files``'s. Without them a file longer than the loop's observation cap was readable
    only up to the cap: a fact at character 10,617 of a 37,562-character file could not be
    reached, and the truncation marker told the model to narrow a call that had nothing to
    narrow (B-130).
    """

    def read_file(path: str, start_line: int = 1, max_lines: int = 0) -> str:
        text = read_text_file(_under_roots(path, settings), settings=settings)
        if start_line == 1 and max_lines == 0:
            return text
        if start_line < 1:
            raise ValueError("start_line counts from 1")
        if max_lines < 0:
            raise ValueError("max_lines must be 0 (to the end) or more")
        lines = text.splitlines()
        if start_line > len(lines):
            raise ValueError(f"start_line {start_line} is past the end: the file has "
                             f"{len(lines)} line(s)")
        end = len(lines) if max_lines == 0 else min(len(lines), start_line + max_lines - 1)
        body = "\n".join(f"{n}: {lines[n - 1]}" for n in range(start_line, end + 1))
        return f"[lines {start_line}-{end} of {len(lines)}]\n{body}"

    return Tool(
        name="read_file",
        description=(
            "Read one local file as text. Only paths inside the operator's allowed roots "
            "can be read; use list_files first if you do not already know the exact path. "
            "For a long file, read part of it with start_line and max_lines."
        ),
        call=read_file,
        params=(
            ToolParam(
                name="path",
                type="string",
                description=(
                    "Path to the file, exactly as list_files reported it. A path relative "
                    "to an allowed root works; so does an absolute one."
                ),
            ),
            ToolParam(
                name="start_line",
                type="integer",
                description=(
                    "First line to return, counting from 1. To read around a search_files "
                    "match on line N, use a few lines before N."
                ),
                required=False,
                default=1,
            ),
            ToolParam(
                name="max_lines",
                type="integer",
                description="How many lines to return from start_line; 0 means to the end.",
                required=False,
                default=0,
            ),
        ),
        returns=(
            "the file's text (CSV and spreadsheets come back as 'a | b | c' rows); with "
            "start_line or max_lines, the selected lines, each prefixed 'N: '"
        ),
        truncation_hint=_read_file_hint,
    )


def _walk(root: str, pattern: str, settings: Any | None):
    """Yield resolved file paths under the allowed roots (or one root inside them).

    The one place containment is enforced for directory walks: every yielded path is
    re-checked after resolution, so a symlink inside a root pointing outside it is skipped
    whatever ``rglob`` does with symlinks on this Python version.
    """
    if root.strip():
        roots = [resolve_under_roots(_under_roots(root, settings), settings=settings)]
    else:
        roots = allowed_roots(settings)
    if not roots:
        raise ValueError(
            "file access is disabled: HEARTH_FILE_ROOTS names no readable directory, so "
            "there is nothing this agent may list or read"
        )
    allowed = allowed_roots(settings)
    for base in roots:
        if not base.is_dir():
            continue
        for entry in sorted(base.rglob(pattern)):
            try:
                if not entry.is_file():
                    continue
                resolved = entry.resolve()
            except OSError:
                continue  # a dangling symlink or an unreadable directory entry
            if any(resolved == r or resolved.is_relative_to(r) for r in allowed):
                yield str(resolved)


def _snippet(line: str) -> str:
    snippet = line.strip()
    if len(snippet) > SEARCH_SNIPPET_CHARS:
        snippet = snippet[:SEARCH_SNIPPET_CHARS] + "…"
    return snippet


def search_files_tool(
    *,
    settings: Any | None = None,
    max_files: int = DEFAULT_SEARCH_FILES,
    max_matches: int = DEFAULT_SEARCH_MATCHES,
) -> Tool:
    """A tool that finds which files (and lines) mention some text — B-013.

    Without it, finding a fact meant ``read_file`` on one file after another under the step
    cap; a live run read files 1, 2, 3, 4 in order and would never have reached file 9. Each
    file is read through :func:`hearth.mcp.files.read_text_file` — the same allowlist, size
    cap and format table as ``read_file`` — so this can reach nothing ``read_file`` cannot.
    Matching is a case-insensitive literal substring (predictable for a small model; no regex
    to get wrong). Files that cannot be read are counted, never quoted.
    """

    def search_files(
        text: str, root: str = "", pattern: str = "*", context: int = 0
    ) -> list[str]:
        needle = text.strip().casefold()
        if not needle:
            raise ValueError("text must not be empty: say what to look for")
        if not 0 <= context <= MAX_SEARCH_CONTEXT:
            raise ValueError(f"context must be between 0 and {MAX_SEARCH_CONTEXT}")
        hits: list[str] = []
        scanned = skipped = matches = 0
        for path in _walk(root, pattern, settings):
            if scanned >= max_files:
                hits.append(
                    f"[... stopped after {max_files} files. Use a narrower root or pattern.]"
                )
                break
            scanned += 1
            try:
                body = read_text_file(path, settings=settings)
            except FileAccessError:
                skipped += 1
                continue
            lines = body.splitlines()
            for number, line in enumerate(lines, start=1):
                if needle in line.casefold():
                    # Context lines use '-' where the match uses ':', as grep -C does, and do
                    # not count toward max_matches.
                    for n in range(max(1, number - context), number):
                        hits.append(f"{path}-{n}- {_snippet(lines[n - 1])}")
                    hits.append(f"{path}:{number}: {_snippet(line)}")
                    for n in range(number + 1, min(len(lines), number + context) + 1):
                        hits.append(f"{path}-{n}- {_snippet(lines[n - 1])}")
                    matches += 1
                    if matches >= max_matches:
                        hits.append(f"[... truncated at {max_matches} matches. Search for "
                                    "something more specific.]")
                        return hits
        if not hits and scanned == 0:
            # Not "the text is absent": no file name matched the glob, so nothing was searched.
            # Said plainly, because a model told "no match" retried *.txt, *.log, *.conf ... and
            # never the '*' that would have found the file (B-130 eval).
            hits.append(f"[no file name matches pattern {pattern!r}, so nothing was searched. "
                        "Use pattern '*' to search every file.]")
        elif not hits:
            hits.append(f"[no match in {scanned} file(s)"
                        + (f"; {skipped} could not be read" if skipped else "") + "]")
        elif skipped:
            hits.append(f"[{skipped} file(s) could not be read and were skipped]")
        return hits

    return Tool(
        name="search_files",
        description=(
            "Find which readable files mention some text, and on which lines. Use this "
            "instead of reading files one by one when you are looking for something."
        ),
        call=search_files,
        params=(
            ToolParam(
                name="text",
                type="string",
                description="Text to look for (case-insensitive, matched literally).",
            ),
            ToolParam(
                name="root",
                type="string",
                description="Directory to search, or leave empty to search every allowed root.",
                required=False,
                default="",
            ),
            ToolParam(
                name="pattern",
                type="string",
                description="Glob for file names to search, e.g. '*.txt'. '*' for everything.",
                required=False,
                default="*",
            ),
            ToolParam(
                name="context",
                type="integer",
                description=(
                    f"Lines to show before and after each match, 0 to {MAX_SEARCH_CONTEXT}. "
                    "Use 1 or 2 when the answer may sit on a line next to the match."
                ),
                required=False,
                default=0,
            ),
        ),
        returns=(
            f"'path:line: text' for each matching line, at most {max_matches} of them; with "
            "context, neighbouring lines as 'path-line- text'"
        ),
        truncation_hint=(
            "Search for something more specific, or pass a narrower root or pattern."
        ),
    )


def list_files_tool(*, settings: Any | None = None, limit: int = DEFAULT_LIST_LIMIT) -> Tool:
    """A tool that lists files under the allowed roots, so the agent can find a path.

    ``read_file`` alone would force the model to *guess* filenames, and a small model guesses
    plausibly and wrongly. This exists so a path in the transcript came from the filesystem
    rather than from the model's imagination.

    Containment is re-checked on every yielded path after resolution rather than trusting the
    walk: a symlink inside a root pointing outside it must not be listed, and whether a given
    Python version's ``rglob`` follows one is not a property worth depending on.
    """

    def list_files(root: str = "", pattern: str = "*") -> list[str]:
        found: list[str] = []
        for path in _walk(root, pattern, settings):
            if len(found) >= limit:
                found.append(f"[... truncated at {limit} paths. Use a narrower root or pattern.]")
                break
            found.append(path)
        return found

    return Tool(
        name="list_files",
        description=(
            "List readable files. With no root, lists everything under the operator's "
            "allowed roots; with a root, lists that directory recursively."
        ),
        call=list_files,
        params=(
            ToolParam(
                name="root",
                type="string",
                description=(
                    "Directory to list, or leave empty to list every allowed root. Must "
                    "itself be inside an allowed root."
                ),
                required=False,
                default="",
            ),
            ToolParam(
                name="pattern",
                type="string",
                description="Glob for the file name, e.g. '*.csv'. Use '*' for everything.",
                required=False,
                default="*",
            ),
        ),
        returns=f"a list of absolute paths, at most {limit} of them",
    )


def rag_search_tool(rag: Any, *, collection: str | None = None, k: int = 6) -> Tool:
    """A tool that retrieves passages from a local RAG collection.

    Retrieval only — ``answer=False``. Letting the tool answer would hide a second model call
    inside a step: its tokens would not be on the agent's budget, its output would not be in
    the transcript, and the agent would be reasoning over a summary of a summary. The agent is
    the reasoner; this tool hands it passages and their sources.

    Passing ``collection`` pins the agent to one collection and drops the parameter from the
    schema, which is both a smaller decision for the model and a real boundary: an agent built
    for the notes collection then has no way to name another one.
    """
    pinned = collection

    def rag_search(query: str, collection: str = "", k: int = k) -> list[str]:
        name = pinned or collection.strip()
        if not name:
            raise ValueError("rag_search needs a collection name")
        result = rag.query(name, query, k=max(1, k), answer=False)
        if not result.chunks:
            return []
        return [f"{chunk.source}: {chunk.text}" for chunk in result.chunks]

    params = [
        ToolParam(
            name="query",
            type="string",
            description="What to look for, in the words you would expect the source to use.",
        )
    ]
    if pinned is None:
        params.append(
            ToolParam(
                name="collection",
                type="string",
                description="Name of the indexed collection to search.",
            )
        )
    params.append(
        ToolParam(
            name="k",
            type="integer",
            description="How many passages to retrieve.",
            required=False,
            default=k,
        )
    )

    where = f" in the {pinned!r} collection" if pinned else ""
    return Tool(
        name="rag_search",
        description=(
            f"Search the operator's indexed documents{where} and return the passages that "
            "match, each with the file it came from. Retrieval only — it does not answer."
        ),
        call=rag_search,
        params=tuple(params),
        returns=(
            "a list of 'source: passage' strings, most relevant first; empty if nothing matched"
        ),
    )


def finance_tools(store: Any) -> tuple[Tool, ...]:
    """Read-only tools over the finance ledger: a total, its rows, and its audit rendering.

    The split is the point. ``finance_total`` gives a number the *store* computed in
    :class:`~decimal.Decimal` — the model never adds anything up, per ``CLAUDE.md`` §4.
    ``finance_explain`` gives that same number with every row behind it and a running sum, so
    a figure that reaches the operator can be walked back to the file and line it came from.
    An agent that quotes a total without ever calling one of these made it up, and the
    transcript shows that.

    There is no ingest, no categorise, no correct. Those write to the ledger.
    """

    def finance_total(start: str = "", end: str = "", category: str = "") -> dict[str, str]:
        figure = store.total(**_filters(start, end, category))
        return {
            "amount": str(figure.amount),
            "rows": str(figure.count),
            "complete": "yes" if figure.is_complete else "no",
            "note": (
                "computed by the ledger in Decimal from the rows below; call finance_explain "
                "for the rows"
                if figure.is_complete
                else "INCOMPLETE — statements were excluded; call finance_explain to see which"
            ),
        }

    def finance_explain(start: str = "", end: str = "", category: str = "") -> str:
        figure = store.total(**_filters(start, end, category))
        return store.explain(figure)

    def finance_rows(
        start: str = "",
        end: str = "",
        category: str = "",
        contains: str = "",
        limit: int = DEFAULT_ROW_LIMIT,
    ) -> list[str]:
        filters = _filters(start, end, category)
        if contains.strip():
            filters["contains"] = contains.strip()
        rows = store.rows(**filters)
        capped = rows[: max(1, min(limit, DEFAULT_ROW_LIMIT))]
        rendered = [
            f"{row.transaction_id}  {row.date.isoformat()}  {row.amount}  {row.description}"
            for row in capped
        ]
        if len(rows) > len(capped):
            rendered.append(
                f"[... {len(rows) - len(capped)} more rows not shown. Narrow the filter.]"
            )
        return rendered

    period = (
        ToolParam(
            name="start",
            type="string",
            description="Earliest date to include, YYYY-MM-DD. Empty for no lower bound.",
            required=False,
            default="",
        ),
        ToolParam(
            name="end",
            type="string",
            description="Latest date to include, YYYY-MM-DD. Empty for no upper bound.",
            required=False,
            default="",
        ),
        ToolParam(
            name="category",
            type="string",
            description="Restrict to one category. Empty for every category.",
            required=False,
            default="",
        ),
    )

    return (
        Tool(
            name="finance_total",
            description=(
                "Total the ledger over a period and optional category. The ledger does the "
                "arithmetic — never add these figures up yourself."
            ),
            call=finance_total,
            params=period,
            returns="amount, row count, and whether anything was excluded from it",
        ),
        Tool(
            name="finance_explain",
            description=(
                "Show a total together with every transaction behind it and a running sum, "
                "so the figure can be checked. Use this before quoting any number."
            ),
            call=finance_explain,
            params=period,
            returns="a plain-text audit listing: the figure, its rows, and anything excluded",
        ),
        Tool(
            name="finance_rows",
            description="List individual ledger transactions matching a filter.",
            call=finance_rows,
            params=period
            + (
                ToolParam(
                    name="contains",
                    type="string",
                    description="Substring the description must contain. Empty for any.",
                    required=False,
                    default="",
                ),
                ToolParam(
                    name="limit",
                    type="integer",
                    description="Maximum rows to return.",
                    required=False,
                    default=DEFAULT_ROW_LIMIT,
                ),
            ),
            returns="one 'id date amount description' line per transaction",
        ),
    )


def local_toolset(
    *,
    settings: Any | None = None,
    rag: Any | None = None,
    finance: Any | None = None,
    collection: str | None = None,
) -> ToolRegistry:
    """Assemble the built-in tools a caller actually has collaborators for.

    File tools are always present (they gate themselves on ``HEARTH_FILE_ROOTS``, which may
    well be empty — then the agent can read nothing and says so). RAG and finance tools appear
    only when their object is passed, so an agent's reach is decided by what the caller
    constructed, not by what happens to be installed.
    """
    registry = ToolRegistry((
        read_file_tool(settings=settings),
        list_files_tool(settings=settings),
        search_files_tool(settings=settings),
    ))
    if rag is not None:
        registry.register(rag_search_tool(rag, collection=collection))
    if finance is not None:
        for tool in finance_tools(finance):
            registry.register(tool)
    return registry


def _filters(start: str, end: str, category: str) -> dict[str, Any]:
    """Turn the model's string filters into the store's typed ones, refusing a bad date.

    An unparseable date is refused rather than dropped: silently ignoring ``start`` widens the
    query to the whole ledger and returns a much larger number that looks just as plausible.
    """
    filters: dict[str, Any] = {}
    if start.strip():
        filters["start"] = _date(start, "start")
    if end.strip():
        filters["end"] = _date(end, "end")
    if category.strip():
        filters["category"] = category.strip()
    return filters


def _date(value: str, field: str) -> datetime.date:
    """Parse ``YYYY-MM-DD``, naming the parameter when it doesn't parse."""
    try:
        return datetime.date.fromisoformat(value.strip())
    except ValueError:
        raise ValueError(
            f"{field} must be a date written YYYY-MM-DD (for example 2026-01-31); "
            f"got {value.strip()!r}"
        ) from None


__all__ = [
    "DEFAULT_LIST_LIMIT",
    "DEFAULT_ROW_LIMIT",
    "finance_tools",
    "list_files_tool",
    "local_toolset",
    "rag_search_tool",
    "read_file_tool",
]
