"""Parsing: exact Decimals, and a hard error with a row index for anything else.

The two properties under test are the ones that decide whether a total can be trusted:

* **no float ever exists.** ``0.1 + 0.2`` is famously not ``0.3`` in binary, and a statement
  of a few hundred rows will drift by cents that no one can account for. Amounts go from the
  file's characters straight into :class:`~decimal.Decimal` and stay there.
* **nothing is skipped.** A row that cannot be parsed raises :class:`ParseError` carrying its
  index. A skipped row would leave a total that is wrong *and* internally consistent, which
  is the one kind of wrong that survives every check downstream.

Every fixture below is synthetic — invented merchant strings, invented amounts, invented
banks. The messiness is deliberate: real exports carry store numbers, auth codes, truncated
names, doubled spaces and at least one accounting-negative convention per bank.
"""

from __future__ import annotations

import datetime
from decimal import Decimal

import pytest

from hearth.finance.mapping import (
    NOTATION_MINUS,
    NOTATION_PARENS,
    NOTATION_TRAILING_MINUS,
    SIGN_AS_WRITTEN,
    SIGN_DEBIT_NEGATIVE,
    SIGN_DEBIT_POSITIVE,
    SIGN_NEGATE,
    ColumnMapping,
    MappingError,
)
from hearth.finance.parse import (
    ParseError,
    Transaction,
    data_row_count,
    parse_money,
    parse_rows,
    resolve_columns,
)

# -- synthetic fixtures ----------------------------------------------------------------------
#
# "Synthetic Savings": one signed amount column, parenthesized accounting negatives, a running
# balance column the mapping deliberately ignores, and merchant strings with the noise real
# exports carry. Opening balance 1080.65, closing 6148.95, so the statement total is
# 5068.30. No value here corresponds to anything real.

SAVINGS_ROWS: list[list[str]] = [
    ["Posting Date", "Description", "Amount", "Running Balance", "Type"],
    ["03/01/2026", "PAYROLL DEP  ACME WIDGETS LLC   ID:8842", "3,120.44", "4,201.09", "CREDIT"],
    ["03/02/2026", "SQ *BLUE BOTTLE #4412  SEATTLE WA", "(6.75)", "4,194.34", "DEBIT"],
    ["03/02/2026", "SQ *BLUE BOTTLE #0031  SEATTLE WA", "(6.75)", "4,187.59", "DEBIT"],
    ["03/04/2026", "AMZN Mktp US*2R41K9TU3   AMZN.C", "(129.99)", "4,057.60", "DEBIT"],
    ["03/07/2026", "TST* PHO KING GOOD - BALL", "(41.20)", "4,016.40", "DEBIT"],
    ["03/09/2026", "RECURRING PMT  NORTHSTAR GYM 0223", "(59.00)", "3,957.40", "DEBIT"],
    ["03/12/2026", "CHECKCARD 0311 SHELL OIL 57445512", "(52.13)", "3,905.27", "DEBIT"],
    ["03/15/2026", "PAYROLL DEP  ACME WIDGETS LLC   ID:8842", "3,120.44", "7,025.71", "CREDIT"],
    ["03/18/2026", "SQ *BLUE BOTTLE #4412  SEATTLE WA", "(6.75)", "7,018.96", "DEBIT"],
    ["03/21/2026", "REFUND AMZN Mktp US*2R41K9TU3", "129.99", "7,148.95", "CREDIT"],
    ["03/28/2026", "ONLINE XFER TO SAVINGS ...4417", "(1,000.00)", "6,148.95", "DEBIT"],
]

SAVINGS_MAPPING = ColumnMapping(
    bank="Synthetic Savings",
    date_column="Posting Date",
    description_column="Description",
    amount_column="Amount",
    date_format="%m/%d/%Y",
    sign=SIGN_AS_WRITTEN,
    currency="USD",
    negative_notation=(NOTATION_MINUS, NOTATION_PARENS),
)

# "Ledger Mutual": a debit/credit pair, a two-line preamble above the header, and a bank that
# writes 0.00 into the unused column rather than leaving it blank.

MUTUAL_ROWS: list[list[str]] = [
    ["Ledger Mutual Bank", "", "", ""],
    ["Statement period 01 Apr 2026 - 30 Apr 2026", "", "", ""],
    ["Date", "Narrative", "Withdrawal", "Deposit"],
    ["01 Apr 2026", "OPENING TRANSFER", "0.00", "500.00"],
    ["03 Apr 2026", "DD  CITY WATER UTIL   REF 99120", "84.30", "0.00"],
    ["11 Apr 2026", "POS  GROCER  4 CORNERS MKT", "112.68", "0.00"],
    ["19 Apr 2026", "INTEREST", "", "1.02"],
]

MUTUAL_MAPPING = ColumnMapping(
    bank="Ledger Mutual",
    date_column="Date",
    description_column="Narrative",
    debit_column="Withdrawal",
    credit_column="Deposit",
    date_format="%d %b %Y",
    sign=SIGN_DEBIT_NEGATIVE,
    skip_rows=2,
)


# -- money -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ("12.50", "12.50"),
        ("  12.50  ", "12.50"),
        ("$12.50", "12.50"),
        ("-12.50", "-12.50"),
        ("+12.50", "12.50"),
        ("(12.50)", "-12.50"),
        ("($1,234.56)", "-1234.56"),
        ("1,234,567.89", "1234567.89"),
        ("0.00", "0.00"),
        ("0.1", "0.1"),
        ("7", "7"),
    ],
)
def test_parse_money_is_exact(cell, expected):
    value = parse_money(cell, SAVINGS_MAPPING)
    assert isinstance(value, Decimal)
    assert value == Decimal(expected)
    # Not merely equal: the written precision survives, so a total can be compared to a
    # printed control total digit for digit.
    assert str(value) == expected


def test_parse_money_never_produces_a_float():
    total = sum(
        (parse_money(c, SAVINGS_MAPPING) for c in ("0.10", "0.20")), Decimal("0")
    )
    assert total == Decimal("0.30")
    assert str(total) == "0.30"
    assert total != 0.1 + 0.2  # the float answer is 0.30000000000000004


def test_an_undeclared_accounting_negative_is_refused_not_read_as_positive():
    """Reading `(50.00)` as `50.00` turns a charge into a deposit and still balances."""
    plain = SAVINGS_MAPPING.replace_columns(negative_notation=(NOTATION_MINUS,))
    with pytest.raises(ParseError, match="parenthesized accounting negative"):
        parse_money("(50.00)", plain)


def test_a_trailing_minus_is_read_only_when_declared():
    plain = SAVINGS_MAPPING.replace_columns(negative_notation=(NOTATION_MINUS,))
    with pytest.raises(ParseError, match="trailing minus"):
        parse_money("1234.56-", plain)

    mainframe = SAVINGS_MAPPING.replace_columns(
        negative_notation=(NOTATION_MINUS, NOTATION_TRAILING_MINUS)
    )
    assert parse_money("1234.56-", mainframe) == Decimal("-1234.56")
    assert parse_money("1,234.56-", mainframe) == Decimal("-1234.56")


def test_a_separator_out_of_group_position_is_refused():
    """`1,50` under a comma-thousands mapping is not 150 and is not 1.50 — it is an error."""
    with pytest.raises(ParseError, match="grouping must be in threes"):
        parse_money("1,50", SAVINGS_MAPPING)


def test_european_separators_are_read_when_declared():
    euro = SAVINGS_MAPPING.replace_columns(decimal_separator=",", thousands_separator=".")
    assert parse_money("1.234,56", euro) == Decimal("1234.56")
    assert parse_money("1,50", euro) == Decimal("1.50")
    with pytest.raises(ParseError):
        parse_money("1,234.56", euro)


@pytest.mark.parametrize("cell", ["", "   ", "-", "n/a", "12.5.0", "1 234.56", "--5.00"])
def test_unreadable_money_cells_raise(cell):
    with pytest.raises(ParseError):
        parse_money(cell, SAVINGS_MAPPING)


def test_a_sign_stated_twice_is_refused():
    with pytest.raises(ParseError, match="states its sign twice"):
        parse_money("(-12.50)", SAVINGS_MAPPING)


# -- whole tables -------------------------------------------------------------------------------


def test_a_signed_amount_table_parses_whole():
    txns = parse_rows(SAVINGS_ROWS, SAVINGS_MAPPING)
    assert len(txns) == data_row_count(SAVINGS_ROWS, SAVINGS_MAPPING) == 11
    assert all(isinstance(t, Transaction) for t in txns)
    assert all(isinstance(t.amount, Decimal) for t in txns)
    assert txns[0].date == datetime.date(2026, 3, 1)
    assert txns[0].amount == Decimal("3120.44")
    assert txns[1].amount == Decimal("-6.75")
    assert txns[0].currency == "USD"


def test_row_indices_point_back_at_the_file():
    txns = parse_rows(SAVINGS_ROWS, SAVINGS_MAPPING)
    # Index 0 is the header, so the first transaction is row 1 of the table as read.
    assert [t.row_index for t in txns] == list(range(1, 12))
    assert SAVINGS_ROWS[txns[3].row_index][1].startswith("AMZN Mktp")


def test_descriptions_keep_their_mess_but_lose_their_padding():
    """Runs of whitespace collapse; the merchant string is otherwise untouched."""
    txns = parse_rows(SAVINGS_ROWS, SAVINGS_MAPPING)
    assert txns[0].description == "PAYROLL DEP ACME WIDGETS LLC ID:8842"
    assert txns[1].description == "SQ *BLUE BOTTLE #4412 SEATTLE WA"
    assert txns[3].description == "AMZN Mktp US*2R41K9TU3 AMZN.C"


def test_the_sum_of_the_synthetic_statement_is_exact():
    txns = parse_rows(SAVINGS_ROWS, SAVINGS_MAPPING)
    assert sum((t.amount for t in txns), Decimal("0")) == Decimal("5068.30")
    # And it ties to the balance column the mapping ignores: 6148.95 - 1080.65.
    assert Decimal("6148.95") - Decimal("1080.65") == Decimal("5068.30")


def test_a_negate_convention_flips_an_unsigned_spend_column():
    """A "Withdrawal amount" column that prints every charge as a positive number."""
    rows = [
        ["Date", "Description", "Withdrawal"],
        ["2026-05-02", "FOUR CORNERS MKT", "45.10"],
        ["2026-05-03", "NORTHSTAR GYM", "59.00"],
    ]
    mapping = ColumnMapping(
        date_column="Date",
        description_column="Description",
        amount_column="Withdrawal",
        date_format="%Y-%m-%d",
        sign=SIGN_NEGATE,
    )
    assert [t.amount for t in parse_rows(rows, mapping)] == [
        Decimal("-45.10"),
        Decimal("-59.00"),
    ]


# -- the debit/credit pair -----------------------------------------------------------------------


def test_a_debit_credit_pair_parses_with_a_preamble_above_the_header():
    txns = parse_rows(MUTUAL_ROWS, MUTUAL_MAPPING)
    assert [t.amount for t in txns] == [
        Decimal("500.00"),
        Decimal("-84.30"),
        Decimal("-112.68"),
        Decimal("1.02"),
    ]
    assert txns[0].date == datetime.date(2026, 4, 1)
    # skip_rows=2, so the header is row 2 and the first transaction is row 3.
    assert txns[0].row_index == 3


def test_a_zero_in_the_unused_column_is_treated_as_absent():
    """Exports routinely write 0.00 rather than leaving the cell blank; a zero cannot lie."""
    rows = [
        ["Date", "Narrative", "Withdrawal", "Deposit"],
        ["01 Apr 2026", "REFUND", "0.00", "22.15"],
    ]
    assert parse_rows(rows, MUTUAL_MAPPING.replace_columns(skip_rows=0))[0].amount == Decimal(
        "22.15"
    )


def test_two_non_zero_amounts_in_one_row_is_a_hard_error():
    rows = [
        ["Date", "Narrative", "Withdrawal", "Deposit"],
        ["01 Apr 2026", "CONTRADICTION", "10.00", "22.15"],
    ]
    with pytest.raises(ParseError, match="states two amounts") as exc:
        parse_rows(rows, MUTUAL_MAPPING.replace_columns(skip_rows=0))
    assert exc.value.row_index == 1


def test_a_signed_value_in_a_magnitude_column_is_a_hard_error():
    """A minus in a debit column means the declared convention is wrong, not that it is clever."""
    rows = [
        ["Date", "Narrative", "Withdrawal", "Deposit"],
        ["01 Apr 2026", "ALREADY SIGNED", "-10.00", ""],
    ]
    with pytest.raises(ParseError, match="carry magnitudes"):
        parse_rows(rows, MUTUAL_MAPPING.replace_columns(skip_rows=0))


def test_the_reversed_pair_convention_is_available_and_explicit():
    """`debit_positive` is the bookkeeping view — stated, never deduced from column names."""
    rows = [
        ["Date", "Narrative", "Withdrawal", "Deposit"],
        ["01 Apr 2026", "LEDGER STYLE", "10.00", ""],
    ]
    mapping = MUTUAL_MAPPING.replace_columns(skip_rows=0, sign=SIGN_DEBIT_POSITIVE)
    assert parse_rows(rows, mapping)[0].amount == Decimal("10.00")


# -- failure carries the row index -----------------------------------------------------------------


def test_an_unparseable_amount_raises_with_its_row_index():
    rows = [row[:] for row in SAVINGS_ROWS]
    rows[5][2] = "PENDING"
    with pytest.raises(ParseError) as exc:
        parse_rows(rows, SAVINGS_MAPPING)
    assert exc.value.row_index == 5
    assert exc.value.column == "Amount"
    assert "PENDING" in str(exc.value)


def test_a_date_that_does_not_match_the_format_raises_with_its_row_index():
    rows = [row[:] for row in SAVINGS_ROWS]
    rows[7][0] = "2026-03-09"
    with pytest.raises(ParseError) as exc:
        parse_rows(rows, SAVINGS_MAPPING)
    assert exc.value.row_index == 7
    assert exc.value.column == "Posting Date"
    assert "No other format is tried" in str(exc.value)


def test_an_empty_date_cell_raises_with_its_row_index():
    rows = [row[:] for row in SAVINGS_ROWS]
    rows[2][0] = ""
    with pytest.raises(ParseError) as exc:
        parse_rows(rows, SAVINGS_MAPPING)
    assert exc.value.row_index == 2


def test_a_blank_row_is_an_error_rather_than_a_skip():
    """The whole point: a silent skip leaves a wrong total that is internally consistent."""
    rows = [row[:] for row in SAVINGS_ROWS]
    rows.insert(4, ["", "", "", "", ""])
    with pytest.raises(ParseError, match="blank") as exc:
        parse_rows(rows, SAVINGS_MAPPING)
    assert exc.value.row_index == 4


def test_a_short_row_is_an_error_rather_than_a_shifted_read():
    rows = [row[:] for row in SAVINGS_ROWS]
    rows[6] = ["03/07/2026", "TRUNCATED EXPORT"]
    with pytest.raises(ParseError, match="short row") as exc:
        parse_rows(rows, SAVINGS_MAPPING)
    assert exc.value.row_index == 6


def test_parse_rows_is_all_or_nothing():
    """A partial result would be a silent skip wearing a list."""
    rows = [row[:] for row in SAVINGS_ROWS]
    rows[-1][2] = "???"
    with pytest.raises(ParseError):
        parse_rows(rows, SAVINGS_MAPPING)


def test_start_index_keeps_indices_meaningful_for_a_slice():
    chunk = [SAVINGS_ROWS[0], *SAVINGS_ROWS[6:8]]
    txns = parse_rows(chunk, SAVINGS_MAPPING, start_index=5)
    assert [t.row_index for t in txns] == [6, 7]


# -- the mapping is checked against the header first ---------------------------------------------


def test_a_mapping_that_does_not_fit_the_header_fails_before_any_row_is_read():
    other_bank = SAVINGS_MAPPING.replace_columns(amount_column="Transaction Amount")
    with pytest.raises(MappingError, match="not present in this header"):
        parse_rows(SAVINGS_ROWS, other_bank)


def test_a_repeated_mapped_column_name_fails_closed():
    rows = [
        ["Posting Date", "Description", "Amount", "Amount"],
        ["03/01/2026", "X", "1.00", "2.00"],
    ]
    with pytest.raises(MappingError, match="repeats mapped column"):
        parse_rows(rows, SAVINGS_MAPPING)


def test_resolve_columns_returns_indices_for_every_bound_role():
    assert resolve_columns(SAVINGS_ROWS[0], SAVINGS_MAPPING) == {
        "date": 0,
        "description": 1,
        "amount": 2,
    }


def test_a_table_with_no_data_rows_parses_to_nothing_without_inventing_any():
    assert parse_rows([SAVINGS_ROWS[0]], SAVINGS_MAPPING) == []
    assert data_row_count([SAVINGS_ROWS[0]], SAVINGS_MAPPING) == 0


def test_a_table_too_short_to_hold_a_header_raises():
    with pytest.raises(ParseError, match="not enough for skip_rows"):
        parse_rows([["Ledger Mutual Bank"]], MUTUAL_MAPPING)


def test_with_category_leaves_the_original_record_untouched():
    txn = parse_rows(SAVINGS_ROWS, SAVINGS_MAPPING)[1]
    labeled = txn.with_category("dining")
    assert labeled.category == "dining"
    assert txn.category is None
    assert labeled.amount == txn.amount
