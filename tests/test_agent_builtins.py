"""The built-in tools: they add no capability, they only expose an existing gate.

So these tests are mostly about what the tools *cannot* do. ``HEARTH_FILE_ROOTS`` must still
govern (deny-by-default, no traversal, no symlink escape), the finance tools must be
read-only and must never make the model do arithmetic, and the default toolset must contain no
shell, no write and no network tool. The tools' callables are exercised directly — there is no
model in this file, because a schema-validated call is a plain Python call.
"""

from __future__ import annotations

import datetime
from types import SimpleNamespace

import pytest

from hearth.agent import (
    DEFAULT_LIST_LIMIT,
    ToolValidationError,
    finance_tools,
    list_files_tool,
    local_toolset,
    rag_search_tool,
    read_file_tool,
)
from hearth.config import Settings
from hearth.mcp.files import FileAccessError

SECRET = "AGENT-CANARY-4b17"


@pytest.fixture
def rooted(tmp_path):
    """A settings object whose one allowed root holds two readable files."""
    root = tmp_path / "docs"
    root.mkdir()
    (root / "march.txt").write_text("March total was 120.\n", encoding="utf-8")
    (root / "notes.md").write_text("a note\n", encoding="utf-8")
    outside = tmp_path / "private"
    outside.mkdir()
    (outside / "secrets.txt").write_text(SECRET, encoding="utf-8")
    return SimpleNamespace(
        settings=Settings(file_roots=str(root), home=tmp_path / ".hearth"),
        root=root,
        outside=outside,
    )


# -- read_file -----------------------------------------------------------------------------


def test_read_file_reads_inside_the_allowlist(rooted):
    tool = read_file_tool(settings=rooted.settings)
    args = tool.validate({"path": str(rooted.root / "march.txt")})
    assert "March total was 120." in tool.call(**args)


def test_read_file_refuses_outside_the_allowlist(rooted):
    tool = read_file_tool(settings=rooted.settings)
    with pytest.raises(FileAccessError, match="outside every allowed root"):
        tool.call(path=str(rooted.outside / "secrets.txt"))


def test_read_file_refuses_traversal_out_of_a_root(rooted):
    tool = read_file_tool(settings=rooted.settings)
    with pytest.raises(FileAccessError):
        tool.call(path=str(rooted.root / ".." / "private" / "secrets.txt"))


def test_a_refusal_never_hands_the_content_back_to_the_model(rooted):
    # The message goes into the next prompt and into the stored transcript, so it must carry
    # the path and the reason and nothing from inside the file.
    tool = read_file_tool(settings=rooted.settings)
    with pytest.raises(FileAccessError) as exc:
        tool.call(path=str(rooted.outside / "secrets.txt"))
    assert SECRET not in str(exc.value)


def test_with_no_roots_configured_the_agent_can_read_nothing(tmp_path):
    tool = read_file_tool(settings=Settings(file_roots="", home=tmp_path / ".hearth"))
    with pytest.raises(FileAccessError, match="HEARTH_FILE_ROOTS"):
        tool.call(path=str(tmp_path / "anything.txt"))


# -- list_files ----------------------------------------------------------------------------


def test_list_files_lists_only_what_is_inside_a_root(rooted):
    tool = list_files_tool(settings=rooted.settings)
    listed = tool.call(**tool.validate({}))
    assert sorted(listed) == sorted(
        [str(rooted.root / "march.txt"), str(rooted.root / "notes.md")]
    )
    assert not any("secrets" in path for path in listed)


def test_list_files_honours_a_glob(rooted):
    tool = list_files_tool(settings=rooted.settings)
    assert tool.call(root="", pattern="*.md") == [str(rooted.root / "notes.md")]


def test_list_files_refuses_a_root_outside_the_allowlist(rooted):
    tool = list_files_tool(settings=rooted.settings)
    with pytest.raises(FileAccessError):
        tool.call(root=str(rooted.outside), pattern="*")


def test_a_symlink_pointing_out_of_a_root_is_not_listed(rooted):
    # rglob's symlink behaviour is a Python-version detail; containment is re-checked on the
    # resolved path so the listing cannot depend on it.
    (rooted.root / "escape.txt").symlink_to(rooted.outside / "secrets.txt")
    listed = list_files_tool(settings=rooted.settings).call()
    assert not any("secrets" in path or "escape" in path for path in listed)


def test_the_listing_is_capped_and_says_that_it_was(rooted):
    for i in range(5):
        (rooted.root / f"f{i}.log").write_text("x", encoding="utf-8")
    listed = list_files_tool(settings=rooted.settings, limit=3).call()
    assert len(listed) == 4  # three paths plus the marker
    assert "truncated at 3 paths" in listed[-1]


def test_with_no_roots_configured_listing_says_file_access_is_disabled(tmp_path):
    tool = list_files_tool(settings=Settings(file_roots="", home=tmp_path / ".hearth"))
    with pytest.raises(ValueError, match="HEARTH_FILE_ROOTS"):
        tool.call()


# -- rag_search ----------------------------------------------------------------------------


class FakeRag:
    """Records what it was asked, and returns two chunks."""

    def __init__(self) -> None:
        self.queries: list[tuple] = []

    def query(self, collection, query, k=6, answer=False):
        self.queries.append((collection, query, k, answer))
        return SimpleNamespace(
            chunks=[
                SimpleNamespace(source="/docs/a.md", text="alpha", score=0.9),
                SimpleNamespace(source="/docs/b.md", text="beta", score=0.5),
            ]
        )


def test_rag_search_retrieves_and_never_asks_the_index_to_answer():
    # answer=True would hide a second model call inside a step: its tokens off the agent's
    # budget and its reasoning out of the transcript.
    rag = FakeRag()
    tool = rag_search_tool(rag)
    out = tool.call(**tool.validate({"query": "alpha", "collection": "notes"}))

    assert out == ["/docs/a.md: alpha", "/docs/b.md: beta"]
    assert rag.queries == [("notes", "alpha", 6, False)]


def test_pinning_a_collection_removes_it_from_the_schema_the_model_sees():
    rag = FakeRag()
    tool = rag_search_tool(rag, collection="notes")
    assert "collection" not in {p.name for p in tool.params}
    tool.call(**tool.validate({"query": "alpha"}))
    assert rag.queries[0][0] == "notes"
    with pytest.raises(ToolValidationError):
        tool.validate({"query": "alpha", "collection": "somewhere_else"})


# -- the finance tools -----------------------------------------------------------------------


class FakeStore:
    """A stand-in ledger: records the filters it was given, returns a fixed figure."""

    def __init__(self) -> None:
        self.filters: list[dict] = []
        self.figure = SimpleNamespace(
            label="total",
            amount="120.50",
            count=2,
            transaction_ids=(1, 2),
            excluded=(),
            is_complete=True,
        )

    def total(self, **filters):
        self.filters.append(filters)
        return self.figure

    def explain(self, figure):
        return (
            f"figure         : {figure.label}\n"
            f"amount         : {figure.amount}\n"
            "  1  2026-03-02  100.00"
        )

    def rows(self, **filters):
        self.filters.append(filters)
        return [
            SimpleNamespace(
                transaction_id=1,
                date=datetime.date(2026, 3, 2),
                amount="100.00",
                description="GROCER",
            ),
            SimpleNamespace(
                transaction_id=2,
                date=datetime.date(2026, 3, 9),
                amount="20.50",
                description="GROCER",
            ),
        ]


def _by_name(tools):
    return {t.name: t for t in tools}


def test_the_finance_tools_are_read_only():
    names = set(_by_name(finance_tools(FakeStore())))
    assert names == {"finance_total", "finance_explain", "finance_rows"}
    assert not any("ingest" in n or "assign" in n or "correct" in n for n in names)


def test_finance_total_returns_the_ledgers_number_not_the_models():
    store = FakeStore()
    tool = _by_name(finance_tools(store))["finance_total"]
    out = tool.call(**tool.validate({"start": "2026-03-01", "end": "2026-03-31"}))

    assert out["amount"] == "120.50"
    assert out["rows"] == "2"
    assert store.filters == [
        {"start": datetime.date(2026, 3, 1), "end": datetime.date(2026, 3, 31)}
    ]


def test_an_incomplete_figure_says_so_where_the_model_will_read_it():
    store = FakeStore()
    store.figure.is_complete = False
    out = _by_name(finance_tools(store))["finance_total"].call()
    assert "INCOMPLETE" in out["note"]


def test_finance_explain_returns_the_audit_listing_behind_the_figure():
    out = _by_name(finance_tools(FakeStore()))["finance_explain"].call()
    assert "amount         : 120.50" in out
    assert "2026-03-02" in out  # the rows, not just the headline


def test_an_unparseable_date_is_refused_rather_than_dropped():
    # Dropping `start` would silently widen the query to the whole ledger and return a much
    # larger number that looks exactly as plausible.
    tool = _by_name(finance_tools(FakeStore()))["finance_total"]
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        tool.call(start="March 2026")


def test_finance_rows_caps_what_it_returns_and_says_so():
    tool = _by_name(finance_tools(FakeStore()))["finance_rows"]
    rendered = tool.call(**tool.validate({"limit": 1}))
    assert rendered[0].startswith("1  2026-03-02  100.00  GROCER")
    assert "1 more rows not shown" in rendered[-1]


# -- the default toolset -----------------------------------------------------------------------


def test_the_default_toolset_has_no_shell_write_or_network_tool(rooted):
    registry = local_toolset(settings=rooted.settings, rag=FakeRag(), finance=FakeStore())
    assert set(registry.names) == {
        "read_file",
        "list_files",
        "rag_search",
        "finance_total",
        "finance_explain",
        "finance_rows",
    }
    forbidden = ("shell", "run", "exec", "write", "delete", "fetch", "http", "url", "request")
    assert not any(word in name for name in registry.names for word in forbidden)


def test_collaborators_a_caller_did_not_pass_produce_no_tools(rooted):
    registry = local_toolset(settings=rooted.settings)
    assert set(registry.names) == {"read_file", "list_files"}


def test_the_rendered_toolset_tells_the_model_what_each_tool_returns(rooted):
    rendered = local_toolset(settings=rooted.settings, finance=FakeStore()).render()
    assert "returns:" in rendered
    assert "never add these figures up yourself" in rendered
    assert str(DEFAULT_LIST_LIMIT) in rendered
