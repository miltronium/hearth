"""A mapping is a decision the operator makes — these pin that HEARTH never makes it for them.

Every test here is a variation on one theme: an incomplete, ambiguous or misfitting mapping
must **fail loudly at construction**, before a single amount is read. The alternative is the
failure mode ``docs/APEX_seam.md`` §5.2 names — a plausible number instead of an error — and
by the time an amount has been read under a wrong mapping there is nothing left to detect it
with.

All data here is synthetic and invented for these tests.
"""

from __future__ import annotations

import pytest

from hearth.finance.mapping import (
    NOTATION_MINUS,
    NOTATION_PARENS,
    SIGN_AS_WRITTEN,
    SIGN_DEBIT_NEGATIVE,
    SIGN_NEGATE,
    ColumnMapping,
    MappingError,
    inspect_header,
    mapping_template,
    normalize_header,
)


def signed_mapping(**overrides) -> ColumnMapping:
    """A complete single-amount-column mapping; overrides break one field at a time."""
    kwargs = dict(
        date_column="Posting Date",
        description_column="Description",
        amount_column="Amount",
        date_format="%m/%d/%Y",
        sign=SIGN_AS_WRITTEN,
    )
    kwargs.update(overrides)
    return ColumnMapping(**kwargs)


# -- construction refuses anything incomplete ----------------------------------------------


def test_a_complete_mapping_constructs():
    mapping = signed_mapping(currency="USD", bank="Synthetic Savings")
    assert mapping.uses_pair is False
    assert mapping.columns() == {
        "date": "Posting Date",
        "description": "Description",
        "amount": "Amount",
    }


@pytest.mark.parametrize(
    "missing", ["date_column", "description_column", "date_format", "sign"]
)
def test_a_blank_required_field_is_refused(missing):
    with pytest.raises(MappingError, match=missing):
        signed_mapping(**{missing: ""})


def test_no_amount_source_at_all_is_refused():
    with pytest.raises(MappingError, match="no amount source"):
        ColumnMapping(
            date_column="Date",
            description_column="Description",
            date_format="%Y-%m-%d",
            sign=SIGN_AS_WRITTEN,
        )


def test_both_an_amount_column_and_a_pair_is_refused():
    """With both present there is no stated rule for which wins, so there is no mapping."""
    with pytest.raises(MappingError, match="never both"):
        ColumnMapping(
            date_column="Date",
            description_column="Description",
            amount_column="Amount",
            debit_column="Withdrawal",
            credit_column="Deposit",
            date_format="%Y-%m-%d",
            sign=SIGN_AS_WRITTEN,
        )


def test_half_a_debit_credit_pair_is_refused():
    with pytest.raises(MappingError, match="credit_column is missing"):
        ColumnMapping(
            date_column="Date",
            description_column="Description",
            debit_column="Withdrawal",
            date_format="%Y-%m-%d",
            sign=SIGN_DEBIT_NEGATIVE,
        )


def test_a_pair_convention_with_a_single_column_is_refused():
    with pytest.raises(MappingError, match="debit/credit pair"):
        signed_mapping(sign=SIGN_DEBIT_NEGATIVE)


def test_a_single_column_convention_with_a_pair_is_refused():
    with pytest.raises(MappingError, match="single amount column"):
        ColumnMapping(
            date_column="Date",
            description_column="Description",
            debit_column="Withdrawal",
            credit_column="Deposit",
            date_format="%Y-%m-%d",
            sign=SIGN_NEGATE,
        )


def test_an_unknown_sign_convention_is_refused():
    with pytest.raises(MappingError, match="sign must be one of"):
        signed_mapping(sign="whichever_looks_right")


def test_there_is_no_default_sign_convention():
    """The signature itself is the guarantee: omitting `sign` cannot construct anything."""
    with pytest.raises(TypeError):
        ColumnMapping(  # type: ignore[call-arg]
            date_column="Date",
            description_column="Description",
            amount_column="Amount",
            date_format="%Y-%m-%d",
        )


# -- date formats ---------------------------------------------------------------------------


def test_a_date_format_without_a_year_is_refused():
    """`%m/%d` parses fine and silently dates every row to 1900 — the classic plausible wrong."""
    with pytest.raises(MappingError, match="loses information"):
        signed_mapping(date_format="%m/%d")


def test_a_nonsense_date_format_is_refused():
    with pytest.raises(MappingError, match="not a usable strptime format"):
        signed_mapping(date_format="%Q-%Z%")


@pytest.mark.parametrize("fmt", ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%d %b %Y", "%m/%d/%y"])
def test_usable_date_formats_are_accepted(fmt):
    assert signed_mapping(date_format=fmt).date_format == fmt


# -- notations and separators ----------------------------------------------------------------


def test_an_unknown_negative_notation_is_refused():
    with pytest.raises(MappingError, match="unknown negative_notation"):
        signed_mapping(negative_notation=("parenthesis",))


def test_an_empty_negative_notation_is_refused():
    with pytest.raises(MappingError, match="cannot be empty"):
        signed_mapping(negative_notation=())


def test_identical_separators_are_refused():
    with pytest.raises(MappingError, match="nothing could be read unambiguously"):
        signed_mapping(decimal_separator=",", thousands_separator=",")


def test_a_multi_character_separator_is_refused():
    with pytest.raises(MappingError, match="single character"):
        signed_mapping(decimal_separator="..")


def test_an_empty_thousands_separator_means_no_grouping():
    mapping = signed_mapping(thousands_separator="")
    assert mapping.thousands_separator == ""


def test_a_bad_currency_code_is_refused():
    with pytest.raises(MappingError, match="three-letter code"):
        signed_mapping(currency="dollars")


def test_negative_skip_rows_is_refused():
    with pytest.raises(MappingError, match="skip_rows"):
        signed_mapping(skip_rows=-1)


# -- YAML round-trip ---------------------------------------------------------------------------


MAPPING_YAML = """
bank: Synthetic Savings
date_column: "Posting Date"
description_column: "Description"
amount_column: "Amount"
date_format: "%m/%d/%Y"
sign: as_written
currency: USD
negative_notation:
  - minus
  - parens
decimal_separator: "."
thousands_separator: ","
skip_rows: 2
"""


def test_a_mapping_loads_from_yaml(tmp_path):
    path = tmp_path / "synthetic_savings.yaml"
    path.write_text(MAPPING_YAML, encoding="utf-8")
    mapping = ColumnMapping.from_yaml(path)
    assert mapping.bank == "Synthetic Savings"
    assert mapping.amount_column == "Amount"
    assert mapping.negative_notation == (NOTATION_MINUS, NOTATION_PARENS)
    assert mapping.skip_rows == 2
    assert mapping.currency == "USD"


def test_a_mapping_round_trips_through_its_dict_form(tmp_path):
    path = tmp_path / "m.yaml"
    path.write_text(MAPPING_YAML, encoding="utf-8")
    mapping = ColumnMapping.from_yaml(path)
    assert ColumnMapping.from_dict(mapping.to_dict()) == mapping


def test_a_misspelled_yaml_key_is_refused_not_ignored():
    """`ammount_column` silently ignored would fall through to a different column entirely."""
    with pytest.raises(MappingError, match="unknown mapping key"):
        ColumnMapping.from_dict(
            {
                "date_column": "Date",
                "description_column": "Description",
                "ammount_column": "Amount",
                "date_format": "%Y-%m-%d",
                "sign": SIGN_AS_WRITTEN,
            }
        )


def test_a_yaml_mapping_missing_a_required_key_is_refused():
    with pytest.raises(MappingError, match="missing required key 'sign'"):
        ColumnMapping.from_dict(
            {
                "date_column": "Date",
                "description_column": "Description",
                "amount_column": "Amount",
                "date_format": "%Y-%m-%d",
            }
        )


def test_an_empty_yaml_file_is_refused(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("# nothing here yet\n", encoding="utf-8")
    with pytest.raises(MappingError, match="file is empty"):
        ColumnMapping.from_yaml(path)


def test_malformed_yaml_is_refused(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("date_column: [unclosed\n", encoding="utf-8")
    with pytest.raises(MappingError, match="not valid YAML"):
        ColumnMapping.from_yaml(path)


# -- the authoring aid reports, it does not choose ---------------------------------------------

HEADER = [
    "Posting Date",
    " Description ",
    "Amount",
    "Running Balance",
    "Check Number",
    "Type",
]


def test_header_cells_are_only_whitespace_trimmed():
    assert normalize_header(HEADER)[1] == "Description"


def test_inspect_header_lists_what_the_operator_has_not_accounted_for():
    report = inspect_header(HEADER, signed_mapping())
    assert report.fits
    assert report.mapped == {
        "date": "Posting Date",
        "description": "Description",
        "amount": "Amount",
    }
    assert report.unmapped == ("Running Balance", "Check Number", "Type")
    assert "Running Balance" in report.describe()


def test_inspect_header_with_no_mapping_is_a_blank_worksheet():
    """The from-scratch case: every column is listed, none is proposed for any role."""
    report = inspect_header(HEADER, None)
    assert report.mapped == {}
    assert report.unmapped == tuple(normalize_header(HEADER))


def test_inspect_header_reports_a_mapped_column_the_file_does_not_have():
    report = inspect_header(["Date", "Description", "Amount"], signed_mapping())
    assert not report.fits
    assert report.missing == {"date": "Posting Date"}
    assert "MISSING" in report.describe()


def test_inspect_header_reports_a_repeated_column_name_as_ambiguous():
    """Two columns called `Amount` and no rule for which wins — that is not a mapping."""
    report = inspect_header(["Posting Date", "Description", "Amount", "Amount"], signed_mapping())
    assert not report.fits
    assert report.ambiguous == ("Amount",)


def test_the_mapping_template_pre_fills_nothing():
    """The template hands over the vocabulary and the column list, and chooses no column."""
    template = mapping_template(HEADER)
    assert "'Running Balance'" in template
    assert "amount_column: \n" in template or "amount_column:\n" in template
    for column in ("Amount", "Posting Date"):
        assert f"amount_column: {column}" not in template
        assert f"date_column: {column}" not in template
