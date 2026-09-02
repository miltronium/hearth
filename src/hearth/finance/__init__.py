"""Local ingestion of the operator's own bank exports — structure only, never semantics.

The operator has statements on this machine and wants HEARTH to validate and synthesize over
them without a byte leaving. That makes this package the most sensitive one in the repo, and
it is built around a single governing rule:

    **HEARTH may parse STRUCTURE, but must never INFER financial SEMANTICS.**

``docs/APEX_seam.md`` §5 is the argument. A sibling project already owns bank-export parsing,
and its processors encode knowledge that cannot be re-derived — alias tables with priority,
parenthesized and trailing-minus accounting negatives, per-bank layout quirks. Every entry in
those tables is a bank that formatted its export differently. A second implementation that
guesses will drift from the first, and **the drift is silent: a mis-parsed amount column
yields a plausible number, not an error.** Recommendation 10 of that document says plainly:
do not build a second parser.

So this package does not. It has no alias table, no header sniffing, no format detection and
no fallbacks. Instead:

* :mod:`~hearth.finance.mapping` — the operator states the layout once per bank, in YAML.
  Which column is the date and its exact format, which is the description, either one signed
  amount column or a debit/credit pair, and which direction of money the bank writes as
  positive. Construction fails on anything incomplete. :func:`inspect_header` reports what is
  mapped, unmapped and missing so the mapping is quick to author — it never picks a column.
* :mod:`~hearth.finance.parse` — rows plus a mapping become :class:`Transaction` records.
  Amounts are :class:`~decimal.Decimal` from the string onward, never float. **A row that
  cannot be parsed raises with its row index; nothing is ever skipped**, because a skipped
  row leaves a total that is wrong and internally consistent.
* :mod:`~hearth.finance.validate` — the reconciliation. Rows read against rows parsed, the
  sum against an operator-supplied control total, dates against the stated period, duplicates
  flagged and never removed. Arithmetic only, checkable by hand. With no control total it
  reports the sum as **unverified** rather than implying it was checked.
* :mod:`~hearth.finance.aggregate` — totals by category, by month, income against spend, top
  merchants. All Decimal, all in Python. **A model must never compute a number here.**

File access goes through :mod:`hearth.mcp.files`, the allowlisted reader, or not at all —
this package has no reader of its own and no way to widen that allowlist.

**No network code, by construction.** No HTTP client, no socket, no subprocess.
``tests/test_finance_no_network.py`` walks the package's own source and enforces it, so the
property survives a future file landing here.
"""

from __future__ import annotations

from .aggregate import (
    UNCATEGORIZED,
    MerchantTotal,
    Totals,
    by_category,
    by_month,
    month_key,
    render_totals,
    spend_by_category,
    top_merchants,
    totals,
)
from .mapping import (
    NEGATIVE_NOTATIONS,
    NOTATION_MINUS,
    NOTATION_PARENS,
    NOTATION_TRAILING_MINUS,
    PAIRED_SIGNS,
    SIGN_AS_WRITTEN,
    SIGN_CONVENTIONS,
    SIGN_DEBIT_NEGATIVE,
    SIGN_DEBIT_POSITIVE,
    SIGN_NEGATE,
    ColumnMapping,
    HeaderReport,
    MappingError,
    inspect_header,
    mapping_template,
)
from .parse import (
    ParseError,
    Transaction,
    data_row_count,
    parse_file,
    parse_money,
    parse_rows,
    read_table,
    resolve_columns,
)
from .validate import (
    SUM_MISMATCH,
    SUM_UNVERIFIED,
    SUM_VERIFIED,
    DuplicateGroup,
    Reconciliation,
    ReconciliationError,
    control_total_from_balances,
    find_duplicates,
    reconcile,
    require_pass,
)

__all__ = [
    "NEGATIVE_NOTATIONS",
    "NOTATION_MINUS",
    "NOTATION_PARENS",
    "NOTATION_TRAILING_MINUS",
    "PAIRED_SIGNS",
    "SIGN_AS_WRITTEN",
    "SIGN_CONVENTIONS",
    "SIGN_DEBIT_NEGATIVE",
    "SIGN_DEBIT_POSITIVE",
    "SIGN_NEGATE",
    "SUM_MISMATCH",
    "SUM_UNVERIFIED",
    "SUM_VERIFIED",
    "UNCATEGORIZED",
    "ColumnMapping",
    "DuplicateGroup",
    "HeaderReport",
    "MappingError",
    "MerchantTotal",
    "ParseError",
    "Reconciliation",
    "ReconciliationError",
    "Totals",
    "Transaction",
    "by_category",
    "by_month",
    "control_total_from_balances",
    "data_row_count",
    "find_duplicates",
    "inspect_header",
    "mapping_template",
    "month_key",
    "parse_file",
    "parse_money",
    "parse_rows",
    "read_table",
    "reconcile",
    "render_totals",
    "require_pass",
    "resolve_columns",
    "spend_by_category",
    "top_merchants",
    "totals",
]
