"""Local SQLite ledger — the place a number stops being a claim and becomes a row.

Everything upstream of here produces figures. :mod:`~hearth.finance.parse` produces them from
a file, :mod:`~hearth.finance.validate` checks them against a control total, and
:mod:`~hearth.finance.aggregate` sums them. What none of them can do is answer the question an
operator actually has three weeks later, looking at a sentence a model wrote:

    *"Where did that number come from?"*

Without a store, the only available answer is "the model said so", and a model's confidence is
uncorrelated with its correctness — the failure this package exists to prevent is not an error
message, it is a **plausible wrong number**. With a store, the question changes shape entirely:
it becomes *"does this figure match a set of rows?"*, and that is arithmetic, so it is
checkable. Every figure this module returns carries the transaction ids that produced it
(:class:`Figure`), and :meth:`FinanceStore.rows_behind` turns those ids back into the exact
line of the exact file. **A number nobody can walk back to its rows is not a result here; it
is a rumour.**

Four design decisions carry that weight.

**1. Money is never a float, and never a REAL column.** ``amount_text`` is the canonical value:
the decimal string exactly as the file wrote it, so ``12.50`` stays ``12.50`` and a total can be
compared against a printed statement digit for digit. Binary floating point cannot represent
``0.10``; the error is invisible per row and compounds under aggregation until a reconciliation
that ties out by construction is off by cents nobody can find. ``amount_micros`` is a
**derived** integer (the value scaled by 10^6, exact by construction or the write is refused)
which exists so SQL can group, order and sum without touching a float.

**2. Two independent computations must agree before a figure exists.** Every query sums
``amount_micros`` in SQL *and* re-adds the ``amount_text`` values in :class:`~decimal.Decimal`
in Python, and refuses to return a :class:`Figure` unless the two match
(:class:`StoreIntegrityError`). This is the same principle the reconciliation applies to the
bank's own total, turned inward: a check that measures a thing against itself is not a check
(``CLAUDE.md`` §3). If the derived column ever drifts from the canonical text, every audit in
the system fails loudly rather than quietly returning the drifted number.

**3. Re-ingest is idempotent, and identity is the CONTENT, not the path.** The SHA-256 of the
file's bytes is unique across the whole store:

* *same bytes, already ingested* → **skipped**, whatever the path. Ingesting
  ``august.csv`` and then ``august-copy.csv`` does not double the operator's spending, which
  is precisely the plausible-wrong-number failure in its most expensive form.
* *same path, different bytes* → a **new version**. The previous statement is retained and
  marked ``superseded_by`` the new one, so a corrected export does not erase what was
  previously believed, and both remain auditable.

**4. A failed ingest is a fact, and is recorded as one.** A statement whose reconciliation did
not pass is written with ``reconciled = 0`` and kept out of totals by default — but it is
*written*. An absent record is indistinguishable from "never tried", and the two have opposite
implications for a total. Because excluding data silently is the mirror image of the bug being
guarded against, every :class:`Figure` carries the statements that were left out of it
(:attr:`Figure.excluded`) and says why.

**Category provenance is history, not a field.** Categorization is the one judgement in this
pipeline, and the only step a model is allowed to perform (``docs/APEX_seam.md``,
``CLAUDE.md`` §4). So a category is never overwritten: each assignment is an append-only row
recording *how* it was decided — ``rule``, ``model`` (with the model id and any adapter id) or
``human`` (with the actor) — and superseding one keeps it. That makes
:meth:`FinanceStore.corrections` return, for free, every case where a person disagreed with the
model and what they said instead: **labelled training examples, captured as a side effect of
the operator doing their own bookkeeping.** ``docs/LEARNING_plan.md`` §2 names the missing data
flywheel as this project's highest-leverage gap; this table is its substrate, and the measured
~70 % tier-1 categorizer in ``examples/finance/README.md`` is what it exists to improve.

**No network code, by construction.** Standard library only, no HTTP client, no socket, no
subprocess. ``tests/test_finance_store.py`` walks this module's own source and enforces it.
The one file read is :func:`hash_file`, which goes through the same allowlisted gate
(:func:`hearth.mcp.files.resolve_under_roots`) as every other path this package touches — it
has no reader of its own and no way to widen that allowlist.

Imported directly (``from hearth.finance.store import FinanceStore``) rather than re-exported
from the package, so ``import hearth.finance`` stays free of ``sqlite3`` and of any on-disk
state: parsing and reconciling a file must not require a database to exist.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from .parse import Transaction
from .validate import SUM_MISMATCH, SUM_UNVERIFIED, SUM_VERIFIED, Reconciliation

ZERO = Decimal("0")

#: Decimal places kept by the derived integer column. Six, not two: a hundredth is not a
#: universal minor unit (JPY has none, KWD has three) and assuming one is the same class of
#: silent misreading as guessing a column. Six covers every currency plus the fractional cents
#: that show up in interest and FX lines, and a value needing more precision is refused rather
#: than rounded — a rounded amount is a wrong amount that still adds up.
MINOR_SCALE = 6

_SCALE = Decimal(10) ** MINOR_SCALE
# SQLite integers are signed 64-bit. At 10^6 scaling this still spans ±9.2 x 10^12 units of
# currency, which is not a limit any statement will reach; overflowing it silently would be.
_MICROS_LIMIT = 2**63

#: How a category was decided. Stored on every assignment because a category with no
#: provenance cannot be corrected, audited, or used as a training label.
METHOD_RULE = "rule"
METHOD_MODEL = "model"
METHOD_HUMAN = "human"
CATEGORY_METHODS = (METHOD_RULE, METHOD_MODEL, METHOD_HUMAN)


class StoreError(ValueError):
    """A write was refused, or a query was asked for something the store cannot answer."""


class StoreIntegrityError(StoreError):
    """The canonical amounts and the derived integer column disagree.

    Raised instead of returning the figure. This is never expected: it means the derived
    ``amount_micros`` column has drifted from the ``amount_text`` it was computed from — a
    corrupted file, a hand-edited row, or a bug in this module. Returning either number would
    be returning one that nothing agrees with, and the whole point of the store is that a
    figure is checkable.
    """


# -- values --------------------------------------------------------------------------------


@dataclass(frozen=True)
class CategorySource:
    """*How* a category was decided — the provenance carried by every assignment.

    Constructed through :meth:`rule`, :meth:`model` or :meth:`human` rather than by hand, so
    the fields that make an assignment auditable cannot be omitted. A model assignment without
    a model id is unattributable (there is no way to tell later which model to blame or
    retrain), and a human correction without an actor is not a labelled example — it is an
    anonymous edit.
    """

    method: str
    model_id: str | None = None
    adapter_id: str | None = None
    rule_id: str | None = None
    actor: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.method not in CATEGORY_METHODS:
            raise StoreError(f"method must be one of {CATEGORY_METHODS}, got {self.method!r}")
        if self.method == METHOD_MODEL and not (self.model_id or "").strip():
            raise StoreError(
                "a model assignment must name the model that made it — without it the "
                "assignment cannot be attributed to a version, compared against a later one, "
                "or used as evidence for or against an adapter"
            )
        if self.method == METHOD_HUMAN and not (self.actor or "").strip():
            raise StoreError(
                "a human assignment must name the actor — a correction is a labelled training "
                "example (docs/LEARNING_plan.md §2) and an unattributed label is not one"
            )
        if self.method == METHOD_RULE and not (self.rule_id or "").strip():
            raise StoreError(
                "a rule assignment must name the rule, so the same rule can be found and "
                "changed when it turns out to be wrong"
            )

    @classmethod
    def rule(cls, rule_id: str, *, note: str = "") -> CategorySource:
        """A deterministic rule assigned it (a string match, an operator's own table)."""
        return cls(method=METHOD_RULE, rule_id=rule_id, note=note)

    @classmethod
    def model(
        cls, model_id: str, *, adapter_id: str | None = None, note: str = ""
    ) -> CategorySource:
        """A local model assigned it. ``adapter_id`` records the LoRA in play, if any."""
        return cls(method=METHOD_MODEL, model_id=model_id, adapter_id=adapter_id, note=note)

    @classmethod
    def human(cls, actor: str = "operator", *, note: str = "") -> CategorySource:
        """A person assigned it. These are the rows the flywheel is built from."""
        return cls(method=METHOD_HUMAN, actor=actor, note=note)


@dataclass(frozen=True)
class RowRef:
    """One stored transaction with everything needed to find it in the original file.

    ``source_path`` plus ``source_row`` is the audit coordinate. ``source_row`` is
    :attr:`Transaction.row_index` unchanged — the offset into the row sequence the reader
    returned, where 0 is the header — so it points at one specific line of one specific export
    and means the same thing here as in a :class:`~hearth.finance.parse.ParseError`.
    ``statement_version`` disambiguates a path that was re-ingested after the bank reissued it.
    """

    transaction_id: int
    statement_id: int
    source_path: str
    statement_version: int
    source_row: int
    date: datetime.date
    description: str
    amount: Decimal
    currency: str | None = None
    category: str | None = None
    category_method: str | None = None
    assignment_id: int | None = None

    def to_transaction(self) -> Transaction:
        """Return the :class:`~hearth.finance.parse.Transaction` shape for the aggregate layer.

        ``row_index`` becomes ``source_row``, which is only unique *within* a statement — so
        :func:`hearth.finance.validate.find_duplicates` over rows drawn from several
        statements would group by an index that means different things in each. Aggregation
        (which ignores the index) is safe; duplicate detection across statements is not, and
        wants :attr:`transaction_id` instead.
        """
        return Transaction(
            date=self.date,
            description=self.description,
            amount=self.amount,
            row_index=self.source_row,
            currency=self.currency,
            category=self.category,
        )


@dataclass(frozen=True)
class ExcludedStatement:
    """A statement deliberately left out of a figure, and why.

    Present on every :class:`Figure` because a total that quietly omits a file is wrong in
    exactly the same way as one that quietly double-counts it, and neither is visible in the
    number itself.
    """

    statement_id: int
    source_path: str
    version: int
    reason: str


@dataclass(frozen=True)
class Figure:
    """One number and the rows that produced it. The unit of trust in this module.

    A :class:`Figure` cannot be constructed by a query unless the SQL integer sum and an
    independent :class:`~decimal.Decimal` re-addition of the stored text agree, so ``amount``
    is a value two separate computations reached. ``transaction_ids`` is the evidence:
    :meth:`FinanceStore.rows_behind` turns it back into rows, and
    :meth:`FinanceStore.explain` renders both together for a person to check.

    ``excluded`` is repeated on every figure of a breakdown rather than reported once for the
    query, deliberately: a figure is routinely handed alone to a narrator or pasted into a
    note, and it has to carry its own caveats when it travels.
    """

    label: str
    amount: Decimal
    count: int
    transaction_ids: tuple[int, ...]
    excluded: tuple[ExcludedStatement, ...] = ()

    @property
    def is_complete(self) -> bool:
        """True when nothing was left out of this figure."""
        return not self.excluded


@dataclass(frozen=True)
class StatementRecord:
    """One ingested file as the store holds it — including the ones that failed.

    ``reconciled`` is the reconciliation's own verdict, and ``sum_status`` keeps the three
    states :mod:`~hearth.finance.validate` is careful to distinguish: verified, mismatched,
    and never checked at all. Collapsing "unverified" into "passed" here would undo that
    distinction at the last step.
    """

    id: int
    source_path: str
    content_sha256: str
    version: int
    mapping_id: str
    mapping_version: str
    mapping_fingerprint: str
    ingested_at: str
    rows_read: int
    rows_parsed: int
    total: Decimal
    control_total: Decimal | None
    control_total_supplied: bool
    sum_status: str
    reconciled: bool
    first_date: datetime.date | None
    last_date: datetime.date | None
    problems: tuple[str, ...]
    superseded_by: int | None
    note: str

    @property
    def is_current(self) -> bool:
        """True when no later ingest of the same path has replaced this one."""
        return self.superseded_by is None


@dataclass(frozen=True)
class IngestResult:
    """What one :meth:`FinanceStore.ingest` call did — including doing nothing.

    ``skipped`` being true is a success, not a failure: it is the idempotency guarantee
    reporting that these bytes were already in the store. ``reason`` says which statement
    they landed in the first time, so "nothing happened" is never ambiguous.
    """

    statement_id: int
    version: int
    inserted: int
    skipped: bool
    reason: str
    reconciled: bool
    sum_status: str
    superseded: tuple[int, ...] = ()


@dataclass(frozen=True)
class CategoryAssignment:
    """One point in a transaction's category history. Never deleted, never overwritten."""

    id: int
    transaction_id: int
    category: str
    method: str
    assigned_at: str
    model_id: str | None = None
    adapter_id: str | None = None
    rule_id: str | None = None
    actor: str | None = None
    note: str = ""
    superseded_by: int | None = None

    @property
    def is_current(self) -> bool:
        """True when this is the assignment in force."""
        return self.superseded_by is None


@dataclass(frozen=True)
class Correction:
    """A human overruling an earlier assignment — one labelled training example.

    ``description`` is the input, ``category`` the label a person stands behind, and
    ``previous_category`` what was believed before, with the model and adapter that believed
    it. That is exactly the shape a supervised example needs, and it accumulates as a
    by-product of the operator fixing their own books rather than from an annotation project
    nobody has time for (``docs/LEARNING_plan.md`` §2).
    """

    transaction_id: int
    description: str
    amount: Decimal
    category: str
    previous_category: str | None
    previous_method: str | None
    previous_model_id: str | None
    previous_adapter_id: str | None
    actor: str | None
    corrected_at: str


# -- schema --------------------------------------------------------------------------------

# Amounts are TEXT + a derived INTEGER, never REAL. `txn_date` is ISO 'YYYY-MM-DD', which
# sorts and range-compares correctly as text, so date filters stay in SQL without a date type.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS statements (
    id                     INTEGER PRIMARY KEY,
    source_path            TEXT    NOT NULL,
    content_sha256         TEXT    NOT NULL UNIQUE,
    version                INTEGER NOT NULL,
    mapping_id             TEXT    NOT NULL,
    mapping_version        TEXT    NOT NULL DEFAULT '',
    mapping_fingerprint    TEXT    NOT NULL DEFAULT '',
    ingested_at            TEXT    NOT NULL,
    rows_read              INTEGER NOT NULL,
    rows_parsed            INTEGER NOT NULL,
    total_text             TEXT    NOT NULL,
    total_micros           INTEGER NOT NULL,
    control_total_text     TEXT,
    control_total_supplied INTEGER NOT NULL CHECK (control_total_supplied IN (0, 1)),
    sum_status             TEXT    NOT NULL
                                   CHECK (sum_status IN ('verified', 'mismatch', 'unverified')),
    reconciled             INTEGER NOT NULL CHECK (reconciled IN (0, 1)),
    first_date             TEXT,
    last_date              TEXT,
    problems               TEXT    NOT NULL DEFAULT '[]',
    superseded_by          INTEGER REFERENCES statements(id),
    note                   TEXT    NOT NULL DEFAULT '',
    UNIQUE (source_path, version)
);

CREATE TABLE IF NOT EXISTS transactions (
    id           INTEGER PRIMARY KEY,
    statement_id INTEGER NOT NULL REFERENCES statements(id) ON DELETE CASCADE,
    source_row   INTEGER NOT NULL,
    txn_date     TEXT    NOT NULL,
    description  TEXT    NOT NULL,
    amount_text  TEXT    NOT NULL,
    amount_micros INTEGER NOT NULL,
    currency     TEXT,
    UNIQUE (statement_id, source_row)
);

CREATE TABLE IF NOT EXISTS category_assignments (
    id             INTEGER PRIMARY KEY,
    transaction_id INTEGER NOT NULL REFERENCES transactions(id) ON DELETE CASCADE,
    category       TEXT    NOT NULL,
    method         TEXT    NOT NULL CHECK (method IN ('rule', 'model', 'human')),
    model_id       TEXT,
    adapter_id     TEXT,
    rule_id        TEXT,
    actor          TEXT,
    note           TEXT    NOT NULL DEFAULT '',
    assigned_at    TEXT    NOT NULL,
    superseded_by  INTEGER REFERENCES category_assignments(id),
    CHECK (method <> 'model' OR model_id IS NOT NULL),
    CHECK (method <> 'human' OR actor IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS idx_statements_path ON statements(source_path);
CREATE INDEX IF NOT EXISTS idx_txn_statement ON transactions(statement_id);
CREATE INDEX IF NOT EXISTS idx_txn_date ON transactions(txn_date);
CREATE INDEX IF NOT EXISTS idx_txn_description ON transactions(description);
CREATE INDEX IF NOT EXISTS idx_assign_txn ON category_assignments(transaction_id);
CREATE INDEX IF NOT EXISTS idx_assign_current
    ON category_assignments(transaction_id) WHERE superseded_by IS NULL;

CREATE VIEW IF NOT EXISTS audit_rows AS
SELECT t.id            AS transaction_id,
       t.statement_id  AS statement_id,
       s.source_path   AS source_path,
       s.version       AS statement_version,
       s.superseded_by AS statement_superseded_by,
       s.reconciled    AS reconciled,
       t.source_row    AS source_row,
       t.txn_date      AS txn_date,
       t.description   AS description,
       t.amount_text   AS amount_text,
       t.amount_micros AS amount_micros,
       t.currency      AS currency,
       ca.id           AS assignment_id,
       ca.category     AS category,
       ca.method       AS category_method
  FROM transactions t
  JOIN statements s ON s.id = t.statement_id
  LEFT JOIN category_assignments ca
         ON ca.transaction_id = t.id AND ca.superseded_by IS NULL;
"""


# -- paths and hashing -----------------------------------------------------------------------


def default_db_path() -> Path:
    """Return ``$HEARTH_HOME/finance/ledger.db`` (default ``~/.hearth/finance/ledger.db``).

    Resolved from the environment rather than by importing :class:`hearth.config.Settings`,
    matching :mod:`hearth.handoff.store`: it is the same value ``Settings.home`` computes, and
    reading it here keeps this package's import graph to the standard library, which is what
    makes the no-network invariant checkable by reading the imports.
    """
    home = os.environ.get("HEARTH_HOME") or (Path.home() / ".hearth")
    return Path(home).expanduser() / "finance" / "ledger.db"


def hash_file(path: str | Path, settings: Any | None = None) -> str:
    """Return the SHA-256 of a statement file's bytes, for the idempotency check.

    The path goes through :func:`hearth.mcp.files.resolve_under_roots` — the same deny-by-
    default allowlist as every other file this package touches. This module has no reader of
    its own and must not become one; the import is local and late for the same reason it is in
    :func:`hearth.finance.parse.read_table`.

    The file is streamed rather than read whole, and the size cap that
    :func:`hearth.mcp.files.read_table` enforces is deliberately not applied: that cap exists
    to keep a huge file out of a parser and out of a model's context, whereas a digest is 64
    hex characters no matter what went in and carries none of the content.
    """
    try:
        from ..mcp.files import resolve_under_roots
    except ImportError:  # pragma: no cover - exercised by the skip guard in the tests
        raise StoreError(
            "hearth.mcp.files.resolve_under_roots is not available; this package deliberately "
            "has no file access of its own — reads go through the allowlist or not at all"
        ) from None

    resolved = resolve_under_roots(path, settings)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def hash_bytes(data: bytes) -> str:
    """Return the SHA-256 of content already in hand — the offline half of :func:`hash_file`."""
    return hashlib.sha256(data).hexdigest()


def mapping_fingerprint(mapping: Any) -> str:
    """Return a digest of a :class:`~hearth.finance.mapping.ColumnMapping`'s stated layout.

    A version *string* is a claim about which mapping was used; this is the mapping itself,
    so a layout edited without bumping its version is still visibly a different layout
    (``CLAUDE.md`` §3 — assert on the outcome, not on the configuration that implies it).
    Only fields that can change a parsed number are included; ``bank`` and ``notes`` are
    prose and are not.
    """
    fields = (
        "date_column",
        "description_column",
        "date_format",
        "sign",
        "amount_column",
        "debit_column",
        "credit_column",
        "currency",
        "negative_notation",
        "decimal_separator",
        "thousands_separator",
        "skip_rows",
    )
    payload = {name: _jsonable(getattr(mapping, name, None)) for name in fields}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _jsonable(value: Any) -> Any:
    """Render a mapping field for the fingerprint without inventing an ordering."""
    if isinstance(value, tuple | list):
        return list(value)
    return value


# -- Decimal <-> integer ----------------------------------------------------------------------


def to_micros(value: Decimal) -> int:
    """Scale an exact :class:`~decimal.Decimal` to the derived integer column, or refuse.

    Refusal rather than rounding is the whole point: a value with more precision than the
    column can hold is data this store cannot represent, and rounding it would produce a row
    that still sums cleanly to the wrong number.
    """
    if not isinstance(value, Decimal):
        raise StoreError(
            f"amounts must be Decimal, got {type(value).__name__} — a float amount is not the "
            "figure the file wrote (hearth.finance.parse keeps them exact from the string on)"
        )
    if not value.is_finite():
        raise StoreError(f"amount {value} is not a finite number")
    scaled = value.scaleb(MINOR_SCALE)
    as_int = int(scaled)
    if scaled != as_int:
        raise StoreError(
            f"amount {value} has more than {MINOR_SCALE} decimal places and cannot be stored "
            "exactly; this store refuses to round money"
        )
    if not -_MICROS_LIMIT < as_int < _MICROS_LIMIT:
        raise StoreError(f"amount {value} is out of range for a 64-bit minor-unit column")
    return as_int


def from_micros(micros: int) -> Decimal:
    """Return the exact :class:`~decimal.Decimal` for a stored integer amount."""
    return Decimal(micros).scaleb(-MINOR_SCALE)


# -- the store ------------------------------------------------------------------------------


class FinanceStore:
    """A SQLite ledger of ingested statements, their rows, and how each was categorized.

    The database file and its parent are created on first write, so constructing a store
    touches nothing. Connections are opened per call and closed again — the same shape as
    :class:`hearth.memory.store.SQLiteVectorStore` — because other processes on this machine
    (an editor, a second agent, a shell) may hold the file at the same time.
    """

    def __init__(self, path: str | Path | None = None, settings: Any | None = None) -> None:
        if path is not None:
            self.path = Path(path).expanduser()
        elif settings is not None and getattr(settings, "home", None) is not None:
            self.path = Path(settings.home).expanduser() / "finance" / "ledger.db"
        else:
            self.path = default_db_path()

    # -- connection -----------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Open the database, creating and locking down the file on first use.

        ``PRAGMA foreign_keys = ON`` is per-connection in SQLite and off by default, so it is
        set here rather than once at creation: a constraint that is only enforced on the
        connection that happened to create the schema is not a constraint.
        """
        created = not self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # The argument is always a filesystem path and never a URI, so sqlite3 has no way to
        # open anything but a local file. The package's no-network invariant
        # (tests/test_finance_aggregate.py) bans calls named `connect` to catch sockets and
        # exempts this one by its full dotted spelling, so the ordinary form is used here.
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        if created:
            # Transaction rows are the operator's own bank data at rest; owner-only, like the
            # handoff store's envelopes.
            os.chmod(self.path, 0o600)
        conn.executescript(_SCHEMA)
        return conn

    # -- ingest ---------------------------------------------------------------------------

    def ingest(
        self,
        transactions: Sequence[Transaction],
        reconciliation: Reconciliation,
        *,
        source_path: str | Path,
        content_sha256: str,
        mapping_id: str,
        mapping_version: str = "",
        fingerprint: str = "",
        category_source: CategorySource | None = None,
        ingested_at: datetime.datetime | None = None,
        note: str = "",
    ) -> IngestResult:
        """Record one parsed statement, its reconciliation, and its rows.

        ``content_sha256`` is the identity of the ingest — :func:`hash_file` of the bytes that
        produced ``transactions``. It is required, not optional: an ingest with no content
        identity cannot be made idempotent, and "I already have this file" is the check that
        stops a re-run doubling someone's spending.

        Three outcomes:

        * these bytes are already in the store → nothing is written and
          :attr:`IngestResult.skipped` is true, naming the statement they are already in
          (even if they arrived under a different filename — a copy is not new data);
        * this path has been ingested before with *different* bytes → a new statement at
          ``version + 1``, with every previous version of that path marked superseded and
          kept;
        * otherwise → a new statement at version 1.

        The reconciliation is stored whatever it says. **A failed one is recorded as failed**
        rather than dropped: an absent record and a refused file look identical afterwards,
        and only one of them means the money is accounted for. Failed statements are excluded
        from figures by default and reported in :attr:`Figure.excluded` when they overlap.

        ``category_source`` is required if any transaction already carries a ``category`` —
        a category with no stated provenance cannot be audited or corrected, so it is refused
        at the door rather than stored as an orphan.
        """
        txns = list(transactions)
        if not isinstance(reconciliation, Reconciliation):
            raise StoreError(
                "a reconciliation is required: a statement recorded without one is a set of "
                "numbers nobody checked, stored as though somebody had"
            )
        if reconciliation.rows_parsed != len(txns):
            raise StoreError(
                f"this reconciliation describes {reconciliation.rows_parsed} parsed row(s) but "
                f"{len(txns)} transaction(s) were handed over; storing them together would "
                "attach a verdict to rows it was not computed from"
            )
        digest = str(content_sha256).strip().lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise StoreError(
                f"content_sha256 must be a 64-character hex SHA-256 digest, got {content_sha256!r}"
            )
        if not str(mapping_id).strip():
            raise StoreError(
                "mapping_id is required — a stored row whose layout is unattributable cannot "
                "be re-derived from the file if the parse turns out to be wrong"
            )
        categorized = [t for t in txns if t.category]
        if categorized and category_source is None:
            raise StoreError(
                f"{len(categorized)} transaction(s) arrive with a category but no "
                "category_source; how a category was decided is not optional here "
                "(rule / model / human) — see CategorySource"
            )
        rows = sorted(txns, key=lambda t: t.row_index)
        if len({t.row_index for t in rows}) != len(rows):
            raise StoreError(
                "two transactions share a row_index; the source row is this store's audit "
                "coordinate and cannot be ambiguous"
            )

        path_text = str(source_path)
        stamp = _timestamp(ingested_at)

        conn = self._connect()
        try:
            existing = conn.execute(
                "SELECT id, version, source_path, reconciled, sum_status FROM statements "
                "WHERE content_sha256 = ?",
                (digest,),
            ).fetchone()
            if existing is not None:
                where = (
                    f"statement {existing['id']}"
                    if existing["source_path"] == path_text
                    else f"statement {existing['id']} ({existing['source_path']})"
                )
                return IngestResult(
                    statement_id=int(existing["id"]),
                    version=int(existing["version"]),
                    inserted=0,
                    skipped=True,
                    reason=f"identical content is already stored as {where}; nothing re-ingested",
                    reconciled=bool(existing["reconciled"]),
                    sum_status=str(existing["sum_status"]),
                )

            prior = conn.execute(
                "SELECT id, version, superseded_by FROM statements WHERE source_path = ? "
                "ORDER BY version",
                (path_text,),
            ).fetchall()
            version = (max(int(r["version"]) for r in prior) + 1) if prior else 1
            superseded = tuple(
                int(r["id"]) for r in prior if r["superseded_by"] is None
            )

            cursor = conn.execute(
                "INSERT INTO statements ("
                "  source_path, content_sha256, version, mapping_id, mapping_version,"
                "  mapping_fingerprint, ingested_at, rows_read, rows_parsed, total_text,"
                "  total_micros, control_total_text, control_total_supplied, sum_status,"
                "  reconciled, first_date, last_date, problems, note"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    path_text,
                    digest,
                    version,
                    str(mapping_id),
                    str(mapping_version),
                    str(fingerprint),
                    stamp,
                    int(reconciliation.rows_read),
                    int(reconciliation.rows_parsed),
                    str(reconciliation.total),
                    to_micros(reconciliation.total),
                    _text_or_none(reconciliation.control_total),
                    0 if reconciliation.control_total is None else 1,
                    reconciliation.sum_status,
                    1 if reconciliation.passed else 0,
                    _iso_date(reconciliation.first_date),
                    _iso_date(reconciliation.last_date),
                    json.dumps(list(reconciliation.problems)),
                    note,
                ),
            )
            statement_id = int(cursor.lastrowid or 0)

            for old_id in superseded:
                conn.execute(
                    "UPDATE statements SET superseded_by = ? WHERE id = ?", (statement_id, old_id)
                )

            for txn in rows:
                txn_cursor = conn.execute(
                    "INSERT INTO transactions ("
                    "  statement_id, source_row, txn_date, description, amount_text,"
                    "  amount_micros, currency"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        statement_id,
                        int(txn.row_index),
                        txn.date.isoformat(),
                        txn.description,
                        str(txn.amount),
                        to_micros(txn.amount),
                        txn.currency,
                    ),
                )
                if txn.category and category_source is not None:
                    _insert_assignment(
                        conn,
                        transaction_id=int(txn_cursor.lastrowid or 0),
                        category=txn.category,
                        source=category_source,
                        at=stamp,
                    )

            conn.commit()
        finally:
            conn.close()

        return IngestResult(
            statement_id=statement_id,
            version=version,
            inserted=len(rows),
            skipped=False,
            reason=(
                f"stored as statement {statement_id} version {version}"
                if reconciliation.passed
                else f"stored as statement {statement_id} version {version} — RECONCILIATION "
                "FAILED, recorded and kept out of totals"
            ),
            reconciled=reconciliation.passed,
            sum_status=reconciliation.sum_status,
            superseded=superseded,
        )

    # -- statements -----------------------------------------------------------------------

    def statements(
        self, *, source_path: str | Path | None = None, include_superseded: bool = True
    ) -> list[StatementRecord]:
        """Return stored statements, newest last. Failed ones are included — they are records."""
        sql = "SELECT * FROM statements"
        clauses: list[str] = []
        params: list[Any] = []
        if source_path is not None:
            clauses.append("source_path = ?")
            params.append(str(source_path))
        if not include_superseded:
            clauses.append("superseded_by IS NULL")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id"
        conn = self._connect()
        try:
            return [_statement_from_row(r) for r in conn.execute(sql, params).fetchall()]
        finally:
            conn.close()

    def statement(self, statement_id: int) -> StatementRecord:
        """Return one statement by id, or raise :class:`StoreError`."""
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM statements WHERE id = ?", (statement_id,)).fetchone()
        finally:
            conn.close()
        if row is None:
            raise StoreError(f"no statement with id {statement_id}")
        return _statement_from_row(row)

    def failed_statements(self) -> list[StatementRecord]:
        """Return current statements whose reconciliation did not pass.

        These are the files whose rows exist in the store but are kept out of every figure.
        An operator who never looks at this list has a total that is quietly incomplete, which
        is why it also surfaces on :attr:`Figure.excluded`.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM statements WHERE reconciled = 0 AND superseded_by IS NULL "
                "ORDER BY id"
            ).fetchall()
        finally:
            conn.close()
        return [_statement_from_row(r) for r in rows]

    # -- categories -----------------------------------------------------------------------

    def assign_category(
        self,
        transaction_id: int,
        category: str,
        source: CategorySource,
        *,
        at: datetime.datetime | None = None,
    ) -> int:
        """Record a category for one transaction and return the new assignment's id.

        The previous current assignment (if any) is marked ``superseded_by`` this one and
        **kept**. Nothing is ever updated in place: the history is the product. A human
        overruling a model is the single most valuable row in this database
        (``docs/LEARNING_plan.md`` §2), and an ``UPDATE`` would delete it.
        """
        if not isinstance(source, CategorySource):
            raise StoreError("source must be a CategorySource stating how this was decided")
        if not str(category).strip():
            raise StoreError("category must be a non-empty string")
        conn = self._connect()
        try:
            if conn.execute(
                "SELECT 1 FROM transactions WHERE id = ?", (transaction_id,)
            ).fetchone() is None:
                raise StoreError(f"no transaction with id {transaction_id}")
            new_id = _insert_assignment(
                conn,
                transaction_id=transaction_id,
                category=str(category),
                source=source,
                at=_timestamp(at),
            )
            conn.commit()
        finally:
            conn.close()
        return new_id

    def category_history(self, transaction_id: int) -> list[CategoryAssignment]:
        """Return every assignment ever made for a transaction, oldest first."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM category_assignments WHERE transaction_id = ? ORDER BY id",
                (transaction_id,),
            ).fetchall()
        finally:
            conn.close()
        return [_assignment_from_row(r) for r in rows]

    def corrections(self, *, since: str | None = None) -> list[Correction]:
        """Return every human assignment that overruled an earlier one — the training set.

        Only assignments that *superseded* something are returned: a human labelling an
        uncategorized row is a label, but a human replacing a model's answer is a label **and**
        a counter-example, which is what makes it worth training on. The pair (description ->
        corrected category) is a supervised example; the superseded row says which model and
        adapter produced the error, so a later evaluation can ask whether it was fixed.
        """
        sql = """
            SELECT new.transaction_id      AS transaction_id,
                   t.description           AS description,
                   t.amount_text           AS amount_text,
                   new.category            AS category,
                   new.actor               AS actor,
                   new.assigned_at         AS corrected_at,
                   old.category            AS previous_category,
                   old.method              AS previous_method,
                   old.model_id            AS previous_model_id,
                   old.adapter_id          AS previous_adapter_id
              FROM category_assignments new
              JOIN category_assignments old ON old.superseded_by = new.id
              JOIN transactions t ON t.id = new.transaction_id
             WHERE new.method = 'human'
        """
        params: list[Any] = []
        if since is not None:
            sql += " AND new.assigned_at >= ?"
            params.append(since)
        sql += " ORDER BY new.id"
        conn = self._connect()
        try:
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()
        return [
            Correction(
                transaction_id=int(r["transaction_id"]),
                description=str(r["description"]),
                amount=Decimal(r["amount_text"]),
                category=str(r["category"]),
                previous_category=r["previous_category"],
                previous_method=r["previous_method"],
                previous_model_id=r["previous_model_id"],
                previous_adapter_id=r["previous_adapter_id"],
                actor=r["actor"],
                corrected_at=str(r["corrected_at"]),
            )
            for r in rows
        ]

    # -- queries: rows --------------------------------------------------------------------

    def rows(
        self,
        *,
        start: datetime.date | None = None,
        end: datetime.date | None = None,
        category: str | None = None,
        uncategorized: bool = False,
        description: str | None = None,
        contains: str | None = None,
        statement_id: int | None = None,
        include_superseded: bool = False,
        include_failed: bool = False,
    ) -> list[RowRef]:
        """Return the rows matching a filter, oldest first. **Rows, never prose.**

        ``start`` and ``end`` are inclusive. ``category`` matches the *current* assignment;
        ``uncategorized`` selects rows that have none. ``description`` is an exact match and
        ``contains`` a substring — no fuzzy matching and no merchant normalization, for the
        reason :mod:`~hearth.finance.aggregate` gives: deciding two merchant strings mean the
        same thing is empirical knowledge this package does not have.

        By default only *current* statements that *reconciled* are visible. The two flags
        widen that for diagnosis, and are named so that a caller cannot turn them on without
        saying what they are asking for.
        """
        sql, params = _row_query(
            start=start,
            end=end,
            category=category,
            uncategorized=uncategorized,
            description=description,
            contains=contains,
            statement_id=statement_id,
            include_superseded=include_superseded,
            include_failed=include_failed,
        )
        conn = self._connect()
        try:
            found = conn.execute(
                f"SELECT * FROM audit_rows {sql} ORDER BY txn_date, transaction_id", params
            ).fetchall()
        finally:
            conn.close()
        return [_row_from_row(r) for r in found]

    def rows_behind(self, figure: Figure | Iterable[int]) -> list[RowRef]:
        """Return the exact rows a :class:`Figure` was computed from. **The audit path.**

        This is the method that makes the rest of the design mean anything. Given a number
        that appeared in a summary, a report, or a sentence a local model wrote, it returns
        the transactions behind it — each with the file and line it came from — so the claim
        can be checked instead of believed. A figure whose rows do not add up to it is a bug
        this call makes visible; a figure with no rows behind it was never real.

        Accepts a :class:`Figure` or a bare iterable of transaction ids, and returns them in
        date order. Ids that are not in the store raise rather than being dropped: a silently
        shorter list of evidence is a silently smaller total.
        """
        ids = tuple(figure.transaction_ids) if isinstance(figure, Figure) else tuple(figure)
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        conn = self._connect()
        try:
            found = conn.execute(
                f"SELECT * FROM audit_rows WHERE transaction_id IN ({placeholders}) "
                "ORDER BY txn_date, transaction_id",
                ids,
            ).fetchall()
        finally:
            conn.close()
        rows = [_row_from_row(r) for r in found]
        missing = set(ids) - {r.transaction_id for r in rows}
        if missing:
            raise StoreError(
                f"transaction id(s) {sorted(missing)} are not in this store; the evidence for "
                "this figure is incomplete and the figure should not be trusted"
            )
        return rows

    def explain(self, figure: Figure) -> str:
        """Render a figure, every row behind it, and a running total, for a person to check.

        The rendering re-adds the rows in :class:`~decimal.Decimal` as it goes and prints the
        running sum, so the last line of the listing is visibly the headline number rather
        than being asserted to equal it. Anything excluded from the figure is named at the
        bottom: a total that left a statement out has to say so where the total is read, not
        in a log nobody opens.
        """
        rows = self.rows_behind(figure)
        lines = [
            f"figure         : {figure.label}",
            f"amount         : {figure.amount}",
            f"rows           : {figure.count}",
        ]
        if not rows:
            lines.append("  (no rows — this figure is not supported by anything)")
        else:
            lines.append("")
            lines.append(f"  {'id':>6}  {'date':<10}  {'amount':>14}  {'running':>14}  source")
            running = ZERO
            for row in rows:
                running += row.amount
                origin = f"{row.source_path}#{row.source_row} v{row.statement_version}"
                lines.append(
                    f"  {row.transaction_id:>6}  {row.date.isoformat():<10}  "
                    f"{str(row.amount):>14}  {str(running):>14}  {origin}  {row.description}"
                )
            lines.append("")
            lines.append(f"  sum of the rows above : {running}")
            lines.append(
                f"  figure                : {figure.amount}"
                + ("" if running == figure.amount else "   <- DISAGREES WITH ITS OWN ROWS")
            )
        for excluded in figure.excluded:
            lines.append(
                f"  EXCLUDED: statement {excluded.statement_id} v{excluded.version} "
                f"{excluded.source_path} — {excluded.reason}"
            )
        return "\n".join(lines)

    # -- queries: figures -----------------------------------------------------------------

    def total(self, *, label: str = "total", **filters: Any) -> Figure:
        """Return the signed sum of the matching rows, with the ids that produced it."""
        return self._figure(label, filters)

    def by_category(self, **filters: Any) -> list[Figure]:
        """Return one :class:`Figure` per current category, largest absolute movement first.

        Signed, matching :func:`hearth.finance.aggregate.by_category`: a category holding a
        charge and its refund reports what happened to the balance rather than a spend figure
        that ignores the refund. Rows with no assignment are grouped under ``uncategorized``
        rather than dropped, because a breakdown that omits what it could not label
        understates every share it reports.
        """
        return self._grouped("COALESCE(category, 'uncategorized')", filters, key=lambda f: f.label)

    def by_month(self, **filters: Any) -> list[Figure]:
        """Return one :class:`Figure` per ``YYYY-MM``, chronologically.

        Months the data does not cover are absent rather than zero-filled — reporting a zero
        for a month no statement covers asserts that nothing happened in it.
        """
        figures = self._grouped("substr(txn_date, 1, 7)", filters, key=lambda f: f.label)
        return sorted(figures, key=lambda f: f.label)

    def by_merchant(
        self, *, limit: int = 10, spend_only: bool = True, **filters: Any
    ) -> list[Figure]:
        """Return the largest merchants by total, grouped on the **exact** description.

        No normalization, for the reason :func:`hearth.finance.aggregate.top_merchants` gives:
        collapsing two spellings of one shop requires knowing which characters are a store
        number, and a plausible guess at that produces a plausible merchant total. Two
        spellings appear as two rows, which a human can see; a bad merge is invisible.

        ``spend_only`` (the default) considers outflows only and reports positive magnitudes,
        which is what "top merchants" means.
        """
        if limit < 0:
            raise StoreError(f"limit must be >= 0, got {limit}")
        figures = self._grouped(
            "description", filters, key=lambda f: f.label, negative_only=spend_only
        )
        if spend_only:
            figures = [
                Figure(
                    label=f.label,
                    amount=-f.amount,
                    count=f.count,
                    transaction_ids=f.transaction_ids,
                    excluded=f.excluded,
                )
                for f in figures
            ]
        figures.sort(key=lambda f: (-abs(f.amount), f.label))
        return figures[:limit]

    # -- figure construction --------------------------------------------------------------

    def _figure(self, label: str, filters: dict[str, Any]) -> Figure:
        """Build one figure over a filter, verifying it against a second, independent sum."""
        found = self._grouped("''", filters)
        if not found:
            return Figure(
                label=label,
                amount=ZERO,
                count=0,
                transaction_ids=(),
                excluded=tuple(self.excluded_statements(**filters)),
            )
        one = found[0]
        return Figure(
            label=label,
            amount=one.amount,
            count=one.count,
            transaction_ids=one.transaction_ids,
            excluded=one.excluded,
        )

    def _grouped(
        self,
        group_expr: str,
        filters: dict[str, Any],
        *,
        key: Any = None,
        negative_only: bool = False,
    ) -> list[Figure]:
        """Group rows by ``group_expr`` and return verified figures, largest movement first.

        The verification is the point of this method. SQLite sums the derived integer column;
        Python re-adds the canonical decimal strings. Both must produce the same value or no
        figure is returned at all — a gate that asserts on the outcome (the two totals) rather
        than on the configuration that implies it (``CLAUDE.md`` §3).
        """
        where, params = _row_query(**filters)
        if negative_only:
            where += (" AND " if where else " WHERE ") + "amount_micros < 0"
        excluded = tuple(self.excluded_statements(**filters))

        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT {group_expr} AS bucket, transaction_id, amount_text, amount_micros "
                f"FROM audit_rows {where} ORDER BY bucket, txn_date, transaction_id",
                params,
            ).fetchall()
            sql_totals = {
                str(r["bucket"]): int(r["micros"])
                for r in conn.execute(
                    f"SELECT {group_expr} AS bucket, SUM(amount_micros) AS micros "
                    f"FROM audit_rows {where} GROUP BY bucket",
                    params,
                ).fetchall()
            }
        finally:
            conn.close()

        buckets: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            buckets.setdefault(str(row["bucket"]), []).append(row)

        figures: list[Figure] = []
        for bucket, members in buckets.items():
            amount = sum((Decimal(r["amount_text"]) for r in members), ZERO)
            if to_micros(amount) != sql_totals.get(bucket):
                raise StoreIntegrityError(
                    f"the stored amounts for {bucket!r} do not agree: adding the decimal text "
                    f"gives {amount}, the integer column gives "
                    f"{from_micros(sql_totals.get(bucket, 0))}. One of them has been corrupted "
                    "and neither number should be used until it is known which"
                )
            figures.append(
                Figure(
                    label=bucket,
                    amount=amount,
                    count=len(members),
                    transaction_ids=tuple(int(r["transaction_id"]) for r in members),
                    excluded=excluded,
                )
            )
        if key is not None:
            figures.sort(key=lambda f: (-abs(f.amount), key(f)))
        return figures

    def excluded_statements(self, **filters: Any) -> list[ExcludedStatement]:
        """Return the current statements deliberately left out of a figure over ``filters``.

        Only *failed* statements are reported. A superseded version is not an omission — it is
        data that was replaced by a later ingest of the same file, and counting both would
        double it — whereas a failed statement is real rows that a total is silently missing,
        which is the surprising case and therefore the one worth naming.
        """
        if filters.get("include_failed"):
            return []
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM statements WHERE reconciled = 0 AND superseded_by IS NULL "
                "ORDER BY id"
            ).fetchall()
        finally:
            conn.close()
        start, end = filters.get("start"), filters.get("end")
        excluded: list[ExcludedStatement] = []
        for row in rows:
            if not _overlaps(row["first_date"], row["last_date"], start, end):
                continue
            problems = json.loads(row["problems"]) or []
            detail = "; ".join(problems) if problems else f"sum {row['sum_status']}"
            excluded.append(
                ExcludedStatement(
                    statement_id=int(row["id"]),
                    source_path=str(row["source_path"]),
                    version=int(row["version"]),
                    reason=f"reconciliation FAILED ({detail}) — rows stored, kept out of totals",
                )
            )
        return excluded


# -- helpers --------------------------------------------------------------------------------


def _timestamp(when: datetime.datetime | None) -> str:
    """Return an ISO-8601 UTC timestamp. UTC so ordering survives a timezone change."""
    moment = when or datetime.datetime.now(datetime.UTC)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=datetime.UTC)
    return moment.astimezone(datetime.UTC).isoformat()


def _iso_date(value: datetime.date | None) -> str | None:
    return None if value is None else value.isoformat()


def _text_or_none(value: Decimal | None) -> str | None:
    """Render a Decimal for storage as its own exact digits, keeping ``None`` distinct.

    ``None`` means *no control total was supplied*, which is a different fact from a control
    total of zero, and the two must not collapse into one column value.
    """
    return None if value is None else str(value)


def _insert_assignment(
    conn: sqlite3.Connection,
    *,
    transaction_id: int,
    category: str,
    source: CategorySource,
    at: str,
) -> int:
    """Append an assignment and point the previous current one at it. Never an UPDATE of it."""
    cursor = conn.execute(
        "INSERT INTO category_assignments ("
        "  transaction_id, category, method, model_id, adapter_id, rule_id, actor, note,"
        "  assigned_at"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            transaction_id,
            category,
            source.method,
            source.model_id,
            source.adapter_id,
            source.rule_id,
            source.actor,
            source.note,
            at,
        ),
    )
    new_id = int(cursor.lastrowid or 0)
    conn.execute(
        "UPDATE category_assignments SET superseded_by = ? "
        "WHERE transaction_id = ? AND superseded_by IS NULL AND id <> ?",
        (new_id, transaction_id, new_id),
    )
    return new_id


def _row_query(
    *,
    start: datetime.date | None = None,
    end: datetime.date | None = None,
    category: str | None = None,
    uncategorized: bool = False,
    description: str | None = None,
    contains: str | None = None,
    statement_id: int | None = None,
    include_superseded: bool = False,
    include_failed: bool = False,
) -> tuple[str, list[Any]]:
    """Build the WHERE clause shared by every row and figure query.

    One builder, so the filters a figure was computed over and the filters
    :meth:`FinanceStore.rows_behind` would reproduce cannot drift apart. Dates compare as ISO
    text, which orders identically to the dates themselves.
    """
    clauses: list[str] = []
    params: list[Any] = []
    if not include_superseded:
        clauses.append("statement_superseded_by IS NULL")
    if not include_failed:
        clauses.append("reconciled = 1")
    if start is not None:
        clauses.append("txn_date >= ?")
        params.append(start.isoformat())
    if end is not None:
        clauses.append("txn_date <= ?")
        params.append(end.isoformat())
    if category is not None:
        clauses.append("category = ?")
        params.append(category)
    if uncategorized:
        clauses.append("category IS NULL")
    if description is not None:
        clauses.append("description = ?")
        params.append(description)
    if contains is not None:
        clauses.append("instr(description, ?) > 0")
        params.append(contains)
    if statement_id is not None:
        clauses.append("statement_id = ?")
        params.append(statement_id)
    return ("WHERE " + " AND ".join(clauses) if clauses else ""), params


def _overlaps(
    first: str | None,
    last: str | None,
    start: datetime.date | None,
    end: datetime.date | None,
) -> bool:
    """True when a statement's date span could contribute to a query's window.

    Unknown bounds count as overlapping: a statement whose dates are unrecorded might belong
    in the window, and reporting it as excluded when it does not is a harmless extra line,
    whereas omitting it when it does hides missing money.
    """
    if first is None or last is None:
        return True
    if start is not None and last < start.isoformat():
        return False
    if end is not None and first > end.isoformat():
        return False
    return True


def _statement_from_row(row: sqlite3.Row) -> StatementRecord:
    return StatementRecord(
        id=int(row["id"]),
        source_path=str(row["source_path"]),
        content_sha256=str(row["content_sha256"]),
        version=int(row["version"]),
        mapping_id=str(row["mapping_id"]),
        mapping_version=str(row["mapping_version"]),
        mapping_fingerprint=str(row["mapping_fingerprint"]),
        ingested_at=str(row["ingested_at"]),
        rows_read=int(row["rows_read"]),
        rows_parsed=int(row["rows_parsed"]),
        total=Decimal(row["total_text"]),
        control_total=(
            None if row["control_total_text"] is None else Decimal(row["control_total_text"])
        ),
        control_total_supplied=bool(row["control_total_supplied"]),
        sum_status=str(row["sum_status"]),
        reconciled=bool(row["reconciled"]),
        first_date=_date_or_none(row["first_date"]),
        last_date=_date_or_none(row["last_date"]),
        problems=tuple(json.loads(row["problems"]) or []),
        superseded_by=None if row["superseded_by"] is None else int(row["superseded_by"]),
        note=str(row["note"]),
    )


def _row_from_row(row: sqlite3.Row) -> RowRef:
    return RowRef(
        transaction_id=int(row["transaction_id"]),
        statement_id=int(row["statement_id"]),
        source_path=str(row["source_path"]),
        statement_version=int(row["statement_version"]),
        source_row=int(row["source_row"]),
        date=datetime.date.fromisoformat(row["txn_date"]),
        description=str(row["description"]),
        amount=Decimal(row["amount_text"]),
        currency=row["currency"],
        category=row["category"],
        category_method=row["category_method"],
        assignment_id=None if row["assignment_id"] is None else int(row["assignment_id"]),
    )


def _assignment_from_row(row: sqlite3.Row) -> CategoryAssignment:
    return CategoryAssignment(
        id=int(row["id"]),
        transaction_id=int(row["transaction_id"]),
        category=str(row["category"]),
        method=str(row["method"]),
        assigned_at=str(row["assigned_at"]),
        model_id=row["model_id"],
        adapter_id=row["adapter_id"],
        rule_id=row["rule_id"],
        actor=row["actor"],
        note=str(row["note"]),
        superseded_by=None if row["superseded_by"] is None else int(row["superseded_by"]),
    )


def _date_or_none(value: str | None) -> datetime.date | None:
    return None if value is None else datetime.date.fromisoformat(value)


__all__ = [
    "CATEGORY_METHODS",
    "METHOD_HUMAN",
    "METHOD_MODEL",
    "METHOD_RULE",
    "MINOR_SCALE",
    "SUM_MISMATCH",
    "SUM_UNVERIFIED",
    "SUM_VERIFIED",
    "CategoryAssignment",
    "CategorySource",
    "Correction",
    "ExcludedStatement",
    "Figure",
    "FinanceStore",
    "IngestResult",
    "RowRef",
    "StatementRecord",
    "StoreError",
    "StoreIntegrityError",
    "default_db_path",
    "from_micros",
    "hash_bytes",
    "hash_file",
    "mapping_fingerprint",
    "to_micros",
]
