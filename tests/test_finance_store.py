"""Tests for the SQLite ledger (hearth.finance.store).

The store exists to turn "the model said £2,300" into "these 41 rows, in this file, at these
line numbers, add up to £2,300". So these tests are mostly about the properties that make that
sentence true, and each one is written against the failure it prevents rather than against the
implementation:

  * **idempotent re-ingest** — the same bytes twice must not double a total, whatever the file
    was called the second time. Silently doubling someone's spending is the plausible-wrong-
    number failure at its most expensive;
  * **a changed file makes a new version and keeps the old** — a corrected export must not
    erase what was previously believed;
  * **Decimal precision survives the round trip** — through SQLite and back out into
    :mod:`hearth.finance.aggregate`, digit for digit, with no REAL column anywhere;
  * **a failed reconciliation is recorded as failed** — an absent record is indistinguishable
    from "never tried", and only one of those means the money is accounted for;
  * **category history is kept** — a human correcting a model creates a row, never an
    overwrite, because those rows are the training set (docs/LEARNING_plan.md §2);
  * **the audit path returns the right ids** — the feature the whole design is for;
  * **foreign keys are on**, and the integrity gate actually fires when the derived integer
    column is tampered with (a gate that cannot fail is not a gate — CLAUDE.md §3);
  * **no network code**, enforced by walking this module's own source.

Every amount, merchant and file here is invented. No real statement is read by these tests,
and none may ever be (CLAUDE.md §4).
"""

from __future__ import annotations

import ast
import datetime
import sqlite3
from decimal import Decimal
from pathlib import Path

import pytest

import hearth.finance.store as store_module
from hearth.finance.aggregate import totals as aggregate_totals
from hearth.finance.parse import Transaction
from hearth.finance.store import (
    CategorySource,
    Figure,
    FinanceStore,
    StoreError,
    StoreIntegrityError,
    default_db_path,
    from_micros,
    hash_bytes,
    hash_file,
    mapping_fingerprint,
    to_micros,
)
from hearth.finance.validate import SUM_UNVERIFIED, SUM_VERIFIED, reconcile

D = Decimal

HASH_A = hash_bytes(b"synthetic statement A")
HASH_B = hash_bytes(b"synthetic statement B")


def tx(day: int, description: str, amount: str, row: int, category: str | None = None):
    return Transaction(
        date=datetime.date(2026, 8, day),
        description=description,
        amount=D(amount),
        row_index=row,
        currency="USD",
        category=category,
    )


def sample() -> list[Transaction]:
    """Four invented rows summing to 3050.15 exactly."""
    return [
        tx(2, "SQ *BLUE BOTTLE COFFEE", "-6.75", 1),
        tx(3, "SHELL OIL 574839201", "-52.10", 2),
        tx(5, "ACH DEPOSIT PAYROLL", "3250.00", 3),
        tx(7, "PACIFIC GAS + ELECTRIC", "-141.00", 4),
    ]


@pytest.fixture
def store(tmp_path) -> FinanceStore:
    return FinanceStore(tmp_path / "ledger.db")


def ingest(store: FinanceStore, txns, *, path="august.csv", digest=HASH_A, **kwargs):
    """Ingest with a passing reconciliation unless the caller says otherwise."""
    recon = kwargs.pop(
        "reconciliation",
        reconcile(txns, rows_read=len(txns), control_total=sum((t.amount for t in txns), D("0"))),
    )
    return store.ingest(
        txns,
        recon,
        source_path=path,
        content_sha256=digest,
        mapping_id=kwargs.pop("mapping_id", "acme-checking"),
        **kwargs,
    )


# -- idempotency: the same bytes never count twice -------------------------------------------


def test_a_first_ingest_stores_the_statement_and_its_rows(store):
    result = ingest(store, sample())
    assert result.skipped is False
    assert result.inserted == 4
    assert result.version == 1
    assert result.reconciled is True

    record = store.statement(result.statement_id)
    assert record.rows_read == 4
    assert record.rows_parsed == 4
    assert record.total == D("3050.15")
    assert record.control_total_supplied is True
    assert record.sum_status == SUM_VERIFIED
    assert store.total().amount == D("3050.15")


def test_reingesting_the_same_content_does_not_duplicate_anything(store):
    first = ingest(store, sample())
    again = ingest(store, sample())

    assert again.skipped is True
    assert again.inserted == 0
    assert again.statement_id == first.statement_id
    assert "already stored" in again.reason
    # The property that matters is not "skipped was reported" but "the total did not move".
    assert store.total().amount == D("3050.15")
    assert len(store.rows()) == 4
    assert len(store.statements()) == 1


def test_the_same_bytes_under_a_different_filename_are_still_the_same_statement(store):
    """A copy is not new data.

    ``august.csv`` and ``august-copy.csv`` holding identical bytes is the ordinary way an
    operator ends up double-counting a month. Identity here is the content hash, so the second
    ingest is refused however the file was named.
    """
    ingest(store, sample(), path="incoming/august.csv")
    duplicate = ingest(store, sample(), path="incoming/august-copy.csv")

    assert duplicate.skipped is True
    assert "incoming/august.csv" in duplicate.reason
    assert store.total().amount == D("3050.15")
    assert len(store.statements()) == 1


def test_a_changed_file_at_the_same_path_makes_a_new_version_and_keeps_the_old(store):
    first = ingest(store, sample(), digest=HASH_A)

    corrected = sample() + [tx(8, "REFUND — DUPLICATE CHARGE", "6.75", 5)]
    second = ingest(store, corrected, digest=HASH_B)

    assert second.skipped is False
    assert second.version == 2
    assert second.superseded == (first.statement_id,)

    # Both statements survive; only the newer one feeds a total.
    kept = {s.id: s for s in store.statements()}
    assert set(kept) == {first.statement_id, second.statement_id}
    assert kept[first.statement_id].superseded_by == second.statement_id
    assert kept[first.statement_id].is_current is False
    assert kept[second.statement_id].is_current is True

    assert store.total().amount == D("3056.90")
    assert len(store.rows()) == 5
    assert len(store.rows(include_superseded=True)) == 9


def test_ingest_refuses_a_reconciliation_computed_over_different_rows(store):
    """A verdict attached to rows it was not computed from is worse than no verdict."""
    txns = sample()
    wrong = reconcile(txns[:2], rows_read=2)
    with pytest.raises(StoreError, match="describes 2 parsed row"):
        store.ingest(
            txns, wrong, source_path="august.csv", content_sha256=HASH_A, mapping_id="acme"
        )


def test_ingest_requires_a_real_content_hash(store):
    with pytest.raises(StoreError, match="64-character hex"):
        store.ingest(
            sample(),
            reconcile(sample(), rows_read=4),
            source_path="august.csv",
            content_sha256="not-a-hash",
            mapping_id="acme",
        )


# -- money: exact, and never a float ---------------------------------------------------------


def test_no_column_in_this_schema_is_a_float(store):
    """The structural half of the money rule: there is nowhere for a float to be stored."""
    ingest(store, sample())
    conn = sqlite3.connect(store.path)
    try:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        assert set(tables) == {"statements", "transactions", "category_assignments"}
        for table in tables:
            types = {row[2].upper() for row in conn.execute(f"PRAGMA table_info({table})")}
            assert not types & {"REAL", "FLOAT", "DOUBLE"}, f"{table} has a float column: {types}"
    finally:
        conn.close()


def test_decimal_precision_survives_the_round_trip(store):
    """0.10 + 0.20 - 0.30 is exactly zero here, and 12.50 comes back as 12.50.

    In binary floating point the first is 5.55e-17 and the second prints as 12.5. Both errors
    are invisible per row and both compound under aggregation, which is why the canonical
    column is the decimal string the file wrote.
    """
    txns = [
        tx(1, "TENTHS A", "0.10", 1),
        tx(1, "TENTHS B", "0.20", 2),
        tx(1, "TENTHS C", "-0.30", 3),
        tx(2, "SCALE KEPT", "12.50", 4),
        tx(3, "BIG", "1234567.89", 5),
        tx(4, "SUB-CENT INTEREST", "0.004500", 6),
    ]
    ingest(store, txns)

    rows = {r.description: r for r in store.rows()}
    assert str(rows["SCALE KEPT"].amount) == "12.50"
    assert str(rows["SUB-CENT INTEREST"].amount) == "0.004500"
    assert all(isinstance(r.amount, Decimal) for r in rows.values())

    figure = store.total()
    assert figure.amount == D("1234580.394500")
    assert figure.amount == sum((t.amount for t in txns), D("0"))


def test_the_round_trip_feeds_aggregate_unchanged(store):
    """Store -> rows -> aggregate must produce what aggregate produced from the parse."""
    txns = sample()
    ingest(store, txns)

    direct = aggregate_totals(txns)
    through_store = aggregate_totals(r.to_transaction() for r in store.rows())

    assert through_store.total == direct.total == D("3050.15")
    assert through_store.income == direct.income
    assert through_store.spend == direct.spend
    assert through_store.count == direct.count
    assert (through_store.first_date, through_store.last_date) == (
        direct.first_date,
        direct.last_date,
    )


def test_an_amount_too_precise_to_store_exactly_is_refused_not_rounded(store):
    with pytest.raises(StoreError, match="refuses to round money"):
        to_micros(D("0.0000001"))


def test_a_float_amount_is_refused_at_the_boundary(store):
    with pytest.raises(StoreError, match="must be Decimal"):
        to_micros(12.5)


def test_micros_round_trip_is_exact():
    for text in ("0.10", "-141.00", "3250", "1234567.89", "0.000001"):
        assert from_micros(to_micros(D(text))) == D(text)


# -- a failed ingest is a fact -----------------------------------------------------------------


def test_a_failed_reconciliation_is_recorded_as_failed_not_omitted(store):
    """An absent record and a refused file look identical afterwards. Only one is honest."""
    txns = sample()
    failed = reconcile(txns, rows_read=6, control_total=D("3050.15"))  # two rows never parsed
    assert failed.passed is False

    result = store.ingest(
        txns,
        failed,
        source_path="august.csv",
        content_sha256=HASH_A,
        mapping_id="acme-checking",
    )
    assert result.reconciled is False
    assert "RECONCILIATION FAILED" in result.reason

    record = store.statement(result.statement_id)
    assert record.reconciled is False
    assert record.rows_read == 6
    assert record.rows_parsed == 4
    assert [s.id for s in store.failed_statements()] == [result.statement_id]

    # Its rows are stored (so the failure can be diagnosed) but kept out of every figure...
    assert store.rows() == []
    assert store.total().amount == D("0")
    assert len(store.rows(include_failed=True)) == 4


def test_a_figure_names_the_statements_it_left_out(store):
    """Excluding data silently is the mirror image of double-counting it."""
    ingest(store, sample(), path="july.csv", digest=HASH_A)

    bad = [tx(12, "MYSTERY DEBIT", "-40.00", 1)]
    store.ingest(
        bad,
        reconcile(bad, rows_read=3),  # a row count that does not tie
        source_path="august.csv",
        content_sha256=HASH_B,
        mapping_id="acme-checking",
    )

    figure = store.total()
    assert figure.amount == D("3050.15")
    assert figure.is_complete is False
    assert len(figure.excluded) == 1
    assert figure.excluded[0].source_path == "august.csv"
    assert "FAILED" in figure.excluded[0].reason
    assert "EXCLUDED" in store.explain(figure)


def test_an_unverified_sum_is_stored_as_unverified_not_as_a_pass(store):
    """The three states validate.py keeps apart must not collapse at the last step."""
    txns = sample()
    result = store.ingest(
        txns,
        reconcile(txns, rows_read=4),  # no control total
        source_path="august.csv",
        content_sha256=HASH_A,
        mapping_id="acme-checking",
    )
    record = store.statement(result.statement_id)
    assert record.sum_status == SUM_UNVERIFIED
    assert record.control_total_supplied is False
    assert record.control_total is None
    assert record.reconciled is True  # unverified is not a failure; it is not a pass either


# -- categories: provenance, history, and the flywheel ------------------------------------------


def test_a_category_arriving_without_provenance_is_refused(store):
    txns = [tx(2, "SQ *BLUE BOTTLE COFFEE", "-6.75", 1, category="dining")]
    with pytest.raises(StoreError, match="no category_source"):
        ingest(store, txns)


def test_a_model_assignment_must_name_its_model(store):
    with pytest.raises(StoreError, match="must name the model"):
        CategorySource(method="model")
    with pytest.raises(StoreError, match="must name the actor"):
        CategorySource(method="human")
    with pytest.raises(StoreError, match="must name the rule"):
        CategorySource(method="rule")


def test_a_human_correction_creates_a_new_record_and_keeps_the_model_s(store):
    """The history is the product. An UPDATE here would delete the training example."""
    txns = [tx(2, "SQ *BLUE BOTTLE COFFEE", "-6.75", 1, category="groceries")]
    ingest(
        store,
        txns,
        category_source=CategorySource.model("Qwen2.5-3B-Instruct-4bit", adapter_id="classify-7"),
    )
    row = store.rows()[0]
    assert row.category == "groceries"
    assert row.category_method == "model"

    store.assign_category(row.transaction_id, "dining", CategorySource.human("operator"))

    history = store.category_history(row.transaction_id)
    assert [(a.category, a.method) for a in history] == [
        ("groceries", "model"),
        ("dining", "human"),
    ]
    assert history[0].is_current is False
    assert history[0].superseded_by == history[1].id
    assert history[1].is_current is True
    assert history[0].model_id == "Qwen2.5-3B-Instruct-4bit"
    assert history[0].adapter_id == "classify-7"

    # Exactly one current assignment, so a query cannot double-count the row.
    current = store.rows()
    assert len(current) == 1
    assert current[0].category == "dining"
    assert current[0].category_method == "human"


def test_corrections_are_labelled_training_examples(store):
    """The flywheel substrate: input, the label a person stands behind, and what was wrong."""
    txns = [
        tx(2, "SQ *BLUE BOTTLE COFFEE", "-6.75", 1, category="groceries"),
        tx(3, "SAFEWAY #1042", "-88.10", 2, category="groceries"),
    ]
    ingest(store, txns, category_source=CategorySource.model("Qwen2.5-3B-Instruct-4bit"))
    coffee = next(r for r in store.rows() if r.description.startswith("SQ *"))
    store.assign_category(coffee.transaction_id, "dining", CategorySource.human("operator"))

    corrections = store.corrections()
    assert len(corrections) == 1  # only the overruled one, not the untouched agreement
    example = corrections[0]
    assert example.description == "SQ *BLUE BOTTLE COFFEE"
    assert example.category == "dining"
    assert example.previous_category == "groceries"
    assert example.previous_method == "model"
    assert example.previous_model_id == "Qwen2.5-3B-Instruct-4bit"
    assert example.actor == "operator"


def test_assigning_to_an_unknown_transaction_is_refused(store):
    with pytest.raises(StoreError, match="no transaction with id"):
        store.assign_category(999, "dining", CategorySource.human("operator"))


# -- queries return rows, and figures carry their evidence ---------------------------------------


def test_by_category_groups_current_assignments_and_keeps_the_unlabelled(store):
    txns = [
        tx(2, "SQ *BLUE BOTTLE COFFEE", "-6.75", 1, category="dining"),
        tx(3, "TST* SABOR LATINO", "-41.20", 2, category="dining"),
        tx(4, "PACIFIC GAS + ELECTRIC", "-141.00", 3, category="utilities"),
        tx(5, "MYSTERY DEBIT", "-9.99", 4),
    ]
    ingest(store, txns, category_source=CategorySource.rule("operator-table-v1"))

    figures = {f.label: f for f in store.by_category()}
    assert figures["dining"].amount == D("-47.95")
    assert figures["utilities"].amount == D("-141.00")
    assert figures["uncategorized"].amount == D("-9.99")
    assert figures["uncategorized"].count == 1


def test_by_month_and_date_filters(store):
    txns = [
        tx(2, "JULY-ADJACENT", "-10.00", 1),
        tx(20, "LATER", "-20.00", 2),
    ]
    september = Transaction(
        date=datetime.date(2026, 9, 3),
        description="SEPTEMBER",
        amount=D("-30.00"),
        row_index=3,
        currency="USD",
    )
    ingest(store, [*txns, september])

    months = {f.label: f.amount for f in store.by_month()}
    assert months == {"2026-08": D("-30.00"), "2026-09": D("-30.00")}
    assert [f.label for f in store.by_month()] == ["2026-08", "2026-09"]

    window = store.total(start=datetime.date(2026, 8, 10), end=datetime.date(2026, 9, 1))
    assert window.amount == D("-20.00")
    assert window.count == 1


def test_by_merchant_groups_the_exact_description_and_never_guesses(store):
    txns = [
        tx(2, "SQ *BLUE BOTTLE #4412", "-6.75", 1),
        tx(3, "SQ *BLUE BOTTLE #0031", "-4.25", 2),
        tx(4, "PACIFIC GAS + ELECTRIC", "-141.00", 3),
        tx(5, "ACH DEPOSIT PAYROLL", "3250.00", 4),
    ]
    ingest(store, txns)

    merchants = store.by_merchant(limit=3)
    assert [m.label for m in merchants] == [
        "PACIFIC GAS + ELECTRIC",
        "SQ *BLUE BOTTLE #4412",
        "SQ *BLUE BOTTLE #0031",
    ]
    assert merchants[0].amount == D("141.00")  # spend reported as a positive magnitude
    # The two store numbers stay apart: merging them would need knowledge this package
    # deliberately does not have, and a wrong merge is invisible in the number.
    assert len(merchants) == 3


def test_contains_is_a_substring_not_a_fuzzy_match(store):
    ingest(store, sample())
    assert [r.description for r in store.rows(contains="SHELL")] == ["SHELL OIL 574839201"]
    assert store.rows(contains="SHEL OIL") == []


# -- the audit path ------------------------------------------------------------------------------


def test_the_audit_path_returns_exactly_the_rows_behind_a_figure(store):
    """The feature the whole design is for: a figure resolves to specific lines of a file."""
    txns = [
        tx(2, "SQ *BLUE BOTTLE COFFEE", "-6.75", 1, category="dining"),
        tx(3, "TST* SABOR LATINO", "-41.20", 2, category="dining"),
        tx(4, "PACIFIC GAS + ELECTRIC", "-141.00", 3, category="utilities"),
    ]
    ingest(store, txns, path="incoming/august.csv", category_source=CategorySource.rule("v1"))

    dining = next(f for f in store.by_category() if f.label == "dining")
    assert dining.amount == D("-47.95")

    rows = store.rows_behind(dining)
    assert [r.description for r in rows] == ["SQ *BLUE BOTTLE COFFEE", "TST* SABOR LATINO"]
    assert {r.transaction_id for r in rows} == set(dining.transaction_ids)
    # The claim is checkable: the rows re-add to the figure, in Decimal, outside the store.
    assert sum((r.amount for r in rows), D("0")) == dining.amount
    # And each one names the exact line of the exact file.
    assert [(r.source_path, r.source_row) for r in rows] == [
        ("incoming/august.csv", 1),
        ("incoming/august.csv", 2),
    ]

    report = store.explain(dining)
    assert "incoming/august.csv#1" in report
    assert "SQ *BLUE BOTTLE COFFEE" in report
    assert "-47.95" in report
    assert "DISAGREES" not in report


def test_rows_behind_refuses_to_return_partial_evidence(store):
    ingest(store, sample())
    figure = store.total()
    broken = Figure(
        label="tampered",
        amount=figure.amount,
        count=figure.count,
        transaction_ids=(*figure.transaction_ids, 9999),
    )
    with pytest.raises(StoreError, match="evidence for this figure is incomplete"):
        store.rows_behind(broken)


def test_explain_of_an_unsupported_figure_says_so(store):
    empty = Figure(label="invented", amount=D("999.00"), count=0, transaction_ids=())
    assert "not supported by anything" in store.explain(empty)


# -- the gates are real ---------------------------------------------------------------------------


def test_foreign_keys_are_enforced(store):
    ingest(store, sample())
    conn = store._connect()
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO transactions (statement_id, source_row, txn_date, description,"
                " amount_text, amount_micros) VALUES (?, ?, ?, ?, ?, ?)",
                (4242, 1, "2026-08-01", "ORPHAN", "-1.00", -1_000_000),
            )
    finally:
        conn.close()


def test_the_integrity_gate_fires_when_the_derived_column_is_tampered_with(store):
    """Ask what the check reports if the thing it guards is broken (CLAUDE.md §3).

    Two computations agreeing is only a check if disagreement is detected. Corrupt the derived
    integer column and every figure over that row must refuse rather than return either number.
    """
    ingest(store, sample())
    assert store.total().amount == D("3050.15")

    conn = sqlite3.connect(store.path)
    try:
        conn.execute(
            "UPDATE transactions SET amount_micros = amount_micros - 1000000 "
            "WHERE description = 'SHELL OIL 574839201'"
        )
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(StoreIntegrityError, match="do not agree"):
        store.total()


def test_deleting_a_statement_takes_its_rows_and_assignments_with_it(store):
    result = ingest(
        store,
        [tx(2, "SQ *BLUE BOTTLE COFFEE", "-6.75", 1, category="dining")],
        category_source=CategorySource.rule("v1"),
    )
    conn = store._connect()
    try:
        conn.execute("DELETE FROM statements WHERE id = ?", (result.statement_id,))
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM category_assignments").fetchone()[0] == 0
    finally:
        conn.close()


# -- paths, hashing and fingerprints ---------------------------------------------------------------


def test_the_db_lives_under_the_hearth_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HEARTH_HOME", str(tmp_path / "home"))
    assert default_db_path() == tmp_path / "home" / "finance" / "ledger.db"

    class FakeSettings:
        home = tmp_path / "elsewhere"

    assert FinanceStore(settings=FakeSettings()).path == (
        tmp_path / "elsewhere" / "finance" / "ledger.db"
    )


def test_the_database_file_is_owner_only(store):
    ingest(store, sample())
    assert (store.path.stat().st_mode & 0o777) == 0o600


def test_hash_file_goes_through_the_allowlist(tmp_path):
    """The store gets its content identity through the same gate as every other read."""
    from hearth.config import Settings

    allowed = tmp_path / "statements"
    allowed.mkdir()
    synthetic = allowed / "august.csv"
    synthetic.write_bytes(b"Date,Description,Amount\n2026-08-02,SYNTHETIC,-1.00\n")
    settings = Settings(file_roots=str(allowed))

    digest = hash_file(synthetic, settings)
    assert digest == hash_bytes(synthetic.read_bytes())
    assert len(digest) == 64

    from hearth.mcp.files import FileAccessError

    outside = tmp_path / "elsewhere.csv"
    outside.write_bytes(b"x")
    with pytest.raises(FileAccessError):
        hash_file(outside, settings)


def test_a_mapping_fingerprint_changes_when_the_layout_does():
    from hearth.finance.mapping import ColumnMapping

    base = ColumnMapping(
        date_column="Posting Date",
        description_column="Description",
        amount_column="Amount",
        date_format="%Y-%m-%d",
        sign="as_written",
    )
    same_meaning = ColumnMapping(
        date_column="Posting Date",
        description_column="Description",
        amount_column="Amount",
        date_format="%Y-%m-%d",
        sign="as_written",
        bank="Acme",
        notes="prose changes nothing",
    )
    different = ColumnMapping(
        date_column="Posting Date",
        description_column="Description",
        amount_column="Amount",
        date_format="%Y-%m-%d",
        sign="negate",
    )
    assert mapping_fingerprint(base) == mapping_fingerprint(same_meaning)
    assert mapping_fingerprint(base) != mapping_fingerprint(different)


# -- the no-network invariant, enforced over this module's own source ------------------------------
#
# Following tests/test_handoff_no_network.py. The store is the finance module most likely to
# grow a "just sync it somewhere" convenience, and it is the one holding the operator's actual
# transactions, so the invariant is checked by reading the source rather than by trusting it.

SOURCE = Path(store_module.__file__)

# Deliberately tiny: an auditor should be able to confirm "no network" from this list plus one
# module. Adding to it is a decision, and this is where that decision becomes visible.
ALLOWED_IMPORTS = frozenset(
    {
        "__future__",
        "collections",
        "dataclasses",
        "datetime",
        "decimal",
        "hashlib",
        "json",
        "os",
        "pathlib",
        "sqlite3",
        "typing",
    }
)

# Names that would make egress possible even without an obviously networky import. ``connect``
# is banned outright, exactly as in tests/test_handoff_no_network.py and the package-wide check
# in tests/test_finance_aggregate.py — which is why the store opens its database with
# ``sqlite3.Connection(path)`` rather than ``sqlite3.connect(path)``. The two are the same
# call; carving an exemption into the ban so the second one could pass would also let through
# the socket call the ban exists to catch, and this module needs the ban more than it needs
# the conventional spelling.
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


def _parse() -> ast.Module:
    return ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))


def _docstring_nodes(tree: ast.Module) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body:
            first = body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                if isinstance(first.value.value, str):
                    ids.add(id(first.value))
    return ids


def test_there_is_a_source_to_check():
    # A vacuous pass here would silently disarm every other test in this section.
    assert SOURCE.name == "store.py"
    assert "class FinanceStore" in SOURCE.read_text(encoding="utf-8")


def test_only_allowlisted_stdlib_imports():
    offenders: list[str] = []
    for node in ast.walk(_parse()):
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
        f"store.py imports {offenders} — hearth.finance must stay stdlib-only and network-free "
        "(docs/TIERS.md); if this is genuinely needed, the invariant needs an explicit "
        "decision, not an allowlist edit in passing"
    )


# `connect` is banned to catch sockets, but the name is not unique to them: `sqlite3.connect`
# opens a local database file and reaches nothing. It is exempted by its full dotted spelling
# so the ban keeps its teeth for `socket.connect` while the store can be written the ordinary
# way — an over-broad rule that forces obscure workarounds only invites someone to delete it.
EXEMPT_DOTTED = frozenset({"sqlite3.connect"})


def _dotted(node: ast.AST) -> str:
    """Render an attribute/name chain as its dotted source spelling."""
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}" if node.value else node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def test_no_shell_or_dynamic_import_escape_hatch():
    offenders: list[str] = []
    for node in ast.walk(_parse()):
        if _dotted(node) in EXEMPT_DOTTED:
            continue
        if isinstance(node, ast.Attribute) and node.attr in BANNED_CALLS:
            offenders.append(node.attr)
        elif isinstance(node, ast.Name) and node.id in BANNED_CALLS:
            offenders.append(node.id)
    assert not offenders, f"store.py references {offenders} — a shell is an egress vector"


def test_the_exemption_does_not_blunt_the_socket_ban():
    """The exemption is by dotted spelling, so a real socket call is still caught.

    Pinning this because an exemption added to silence one false positive is exactly how a
    guard quietly stops guarding — the check has to keep failing on the thing it exists for.
    """
    tree = ast.parse("import socket\ns = socket.socket()\ns.connect(('example.com', 443))\n")
    offenders = [
        n.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute)
        and n.attr in BANNED_CALLS
        and _dotted(n) not in EXEMPT_DOTTED
    ]
    assert "connect" in offenders


def test_no_endpoint_smuggled_in_as_a_string_literal():
    tree = _parse()
    docstrings = _docstring_nodes(tree)
    offenders = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
        and "://" in node.value
    ]
    assert not offenders, f"store.py contains a URL-shaped literal: {offenders}"


def test_no_transport_reachable_through_the_module_namespace():
    # Belt and braces: whatever the source says, the imported module must not have pulled a
    # transport in under a different name.
    for name in ("socket", "http", "httpx", "requests", "urllib", "subprocess", "asyncio"):
        assert not hasattr(store_module, name), f"hearth.finance.store exposes {name}"
