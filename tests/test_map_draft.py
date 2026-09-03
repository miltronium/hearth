"""``scripts/hearth_map_draft.py`` — the local mapping assistant, and the lines it must not cross.

These tests are about the *division of labour*, not about producing a nice draft. The three
properties they hold the script to:

**What is mechanically decidable is decided in code.** Above all the date format, scanned over
every row: day-first, month-first and ISO are determined, and a file where nothing
disambiguates comes back AMBIGUOUS rather than resolved by preference. That case matters more
than any other here, because reconciliation cannot catch a wrong date format — sums do not
depend on dates, so the wrong choice passes every arithmetic check and quietly moves
transactions between months.

**Nothing unverified is written.** A model proposal that contradicts the measurements is
dropped; a draft that cannot parse the file it was drafted from is not written at all; an
existing mapping is never overwritten without ``--force``, because the operator's reviewed
mapping outranks a fresh draft.

**No value ever reaches stdout.** A canary string planted in a fixture must not appear in the
tool's output — including on the failure path, where the parser's own error messages quote the
offending cell and must therefore be withheld.

No model is required: a stub proposer stands in for the local one, so the whole suite runs
offline with no weights on disk.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from hearth.config import Settings, get_settings
from hearth.finance.mapping import ColumnMapping, MappingError

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "hearth_map_draft.py"

_spec = importlib.util.spec_from_file_location("hearth_map_draft", _SCRIPT)
assert _spec and _spec.loader
md = importlib.util.module_from_spec(_spec)
# Registered before execution because @dataclass resolves annotations through sys.modules.
sys.modules["hearth_map_draft"] = md
_spec.loader.exec_module(md)


# -- fixtures ---------------------------------------------------------------------------------
#
# Every statement below is invented for these tests. CANARY is the string that must never be
# printed: it stands in for a merchant name on a real line item.

CANARY = "CANARY-MERCHANT-QX7"
CANARY_AMOUNT = "1313.13"

US_WITH_BALANCE = (
    "Posting Date,Description,Amount,Balance\n"
    "08/25/2026,COFFEE SHOP 118,-4.50,995.50\n"
    f"08/26/2026,{CANARY},-{CANARY_AMOUNT},-317.63\n"
    "08/27/2026,PAYROLL ACME,2000.00,1682.37\n"
    "08/28/2026,GROCERY 4471,-60.00,1622.37\n"
)

DAY_FIRST = (
    "Date,Description,Amount\n"
    "25/12/2026,MARKET,-10.00\n"
    "01/02/2026,REFUND,25.00\n"
)

MONTH_FIRST = (
    "Date,Description,Amount\n"
    "12/25/2026,MARKET,-10.00\n"
    "02/01/2026,REFUND,25.00\n"
)

ISO = (
    "Date,Description,Amount\n"
    "2026-12-25,MARKET,-10.00\n"
    "2026-02-01,REFUND,25.00\n"
)

AMBIGUOUS_DATES = (
    "Date,Description,Amount\n"
    "01/02/2026,MARKET,-10.00\n"
    "03/04/2026,REFUND,25.00\n"
)

PARENS = (
    "Date,Description,Amount\n"
    "01/25/2026,MARKET,(10.00)\n"
    "02/13/2026,REFUND,25.00\n"
)

TRAILING_MINUS = (
    "Date,Description,Amount\n"
    "01/25/2026,MARKET,10.00-\n"
    "02/13/2026,REFUND,25.00\n"
)

DEBIT_CREDIT = (
    "Date,Description,Type,Debit,Credit\n"
    "01/25/2026,MARKET,POS,10.00,\n"
    "02/13/2026,REFUND,ACH,,25.00\n"
    "03/14/2026,RENT,ACH,900.00,\n"
)

PAIR_WITH_BALANCE = (
    "Date,Description,Type,Debit,Credit,Balance\n"
    "2026-01-05,COFFEE,POS,4.50,,995.50\n"
    "2026-01-06,PAYROLL ACME,ACH,,2000.00,2995.50\n"
    "2026-01-31,RENT 4471,ACH,900.00,,2095.50\n"
)

# The date cell on the canary row is empty: the columns all classify, the mapping is complete,
# and parse_rows then fails on that row. That is the "drafted but unusable" path.
UNPARSEABLE = (
    "Posting Date,Description,Amount,Balance\n"
    "08/25/2026,COFFEE SHOP 118,-4.50,995.50\n"
    f",{CANARY},-{CANARY_AMOUNT},-317.63\n"
)


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _fresh_settings():
    """``get_settings`` is process-cached, so a test that sets HEARTH_FILE_ROOTS must clear it."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings whose file allowlist is exactly ``tmp_path`` — the reader's gate still applies."""
    return Settings(
        backend="echo", home=tmp_path / ".hearth", require_auth=False, file_roots=str(tmp_path)
    )


class StubModel:
    """A local model stand-in: records prompts, returns a canned answer, never loads weights."""

    name = "stub"

    def __init__(self, payload: str = "", *, unavailable: bool = False) -> None:
        self.payload = payload
        self.unavailable = unavailable
        self.prompts: list[str] = []

    def propose(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.unavailable:
            raise md.ModelUnavailable("mlx-lm is not installed")
        return self.payload


def _profile(tmp_path, settings, name: str, text: str):
    return md.profile_files([_write(tmp_path, name, text)], settings)


# -- 1. date format: the one semantic error with no downstream gate ---------------------------


def test_day_first_is_decided_from_a_row_no_sampler_would_have_to_see():
    finding = md.detect_date_format(["25/12/2026", "01/02/2026"])
    assert finding.status == md.DATE_DECIDED
    assert finding.chosen == "%d/%m/%Y"
    assert "day-first" in finding.evidence


def test_month_first_is_decided():
    finding = md.detect_date_format(["12/25/2026", "02/01/2026"])
    assert finding.status == md.DATE_DECIDED
    assert finding.chosen == "%m/%d/%Y"


def test_a_leading_four_digit_year_is_iso():
    finding = md.detect_date_format(["2026-12-25", "2026-02-01"])
    assert finding.status == md.DATE_DECIDED
    assert finding.chosen == "%Y-%m-%d"


def test_nothing_disambiguating_in_the_whole_file_is_ambiguous_not_a_preference():
    """The load-bearing case: no arithmetic downstream can catch this, so nobody may guess."""
    finding = md.detect_date_format(["01/02/2026", "03/04/2026"])
    assert finding.status == md.DATE_AMBIGUOUS
    assert set(finding.candidates) == {"%m/%d/%Y", "%d/%m/%Y"}
    assert finding.chosen is None
    assert "nothing in the data separates" in finding.evidence


def test_one_disambiguating_row_out_of_many_settles_the_whole_column():
    cells = ["01/02/2026"] * 200 + ["25/03/2026"] + ["04/05/2026"] * 200
    assert md.detect_date_format(cells).chosen == "%d/%m/%Y"


def test_a_two_digit_year_is_not_read_as_the_year_three():
    """strptime would happily call "03" the year 3; a statement is not from antiquity."""
    finding = md.detect_date_format(["01/02/03", "25/06/04"])
    assert "%d/%m/%Y" not in finding.candidates


def test_an_ambiguous_date_leaves_the_draft_unloadable(tmp_path, settings):
    profile = _profile(tmp_path, settings, "amb.csv", AMBIGUOUS_DATES)
    draft = md.draft_for_profile(profile, None, settings=settings)

    assert draft.fields["date_format"] == md.AMBIGUOUS
    assert not draft.complete
    assert any("date_format" in item and "AMBIGUOUS" in item for item in draft.confirm)

    # The rest of the mapping is still proved, under every surviving candidate.
    assert draft.verification.status == md.VERIFIED_EACH

    out = tmp_path / "mappings"
    path, outcome = md.write_draft(draft, out, force=False)
    assert outcome == "written"
    with pytest.raises(MappingError):
        ColumnMapping.from_yaml(path)


# -- 2. the other mechanical determinations ---------------------------------------------------


def test_parenthesized_negatives_are_detected_and_declared(tmp_path, settings):
    profile = _profile(tmp_path, settings, "parens.csv", PARENS)
    assert md.NOTATION_PARENS in profile.negative_notation
    assert md.NOTATION_TRAILING_MINUS not in profile.negative_notation


def test_trailing_minus_negatives_are_detected_and_declared(tmp_path, settings):
    profile = _profile(tmp_path, settings, "trailing.csv", TRAILING_MINUS)
    assert md.NOTATION_TRAILING_MINUS in profile.negative_notation
    assert md.NOTATION_PARENS not in profile.negative_notation


def test_a_debit_credit_pair_is_recognised_as_a_pair(tmp_path, settings):
    profile = _profile(tmp_path, settings, "pair.csv", DEBIT_CREDIT)
    assert profile.structure.pairs == (("Debit", "Credit"),)
    assert profile.structure.singles == ()


def test_a_single_signed_column_is_not_mistaken_for_half_a_pair(tmp_path, settings):
    profile = _profile(tmp_path, settings, "single.csv", MONTH_FIRST)
    assert profile.structure.pairs == ()
    assert profile.structure.singles == ("Amount",)


def test_an_empty_column_is_reported_as_empty(tmp_path, settings):
    profile = _profile(
        tmp_path,
        settings,
        "empty.csv",
        "Date,Description,Amount,Memo\n01/25/2026,MARKET,-10.00,\n02/13/2026,REFUND,25.00,\n",
    )
    memo = profile.by_name("Memo")
    assert memo is not None and memo.kind == md.KIND_EMPTY


def test_a_running_balance_settles_the_amount_column_and_the_sign_without_a_model(
    tmp_path, settings
):
    """Arithmetic, not opinion: the balance moves by the amount on every row, or it does not."""
    profile = _profile(tmp_path, settings, "bal.csv", US_WITH_BALANCE)
    draft = md.draft_for_profile(profile, None, settings=settings)

    assert draft.fields["amount_column"] == "Amount"
    assert draft.fields["sign"] == md.SIGN_AS_WRITTEN
    assert draft.complete
    sources = {p.source for p in draft.provenance if "amount_column" in p.field_name}
    assert sources == {md.SOURCE_MECHANICAL}
    # ...and it is still on the confirm list, because "that column is the balance" is a reading.
    assert any(item.startswith("sign") for item in draft.confirm)


# -- 3. the model's half, and the validation it has to survive --------------------------------


def test_the_model_is_never_asked_for_the_date_format(tmp_path, settings):
    """Step 1 already answered it. A model asked would answer plausibly and uncheckably."""
    profile = _profile(tmp_path, settings, "bal.csv", US_WITH_BALANCE)
    prompt = md.build_prompt(profile)
    for forbidden in ("date_format", "strptime", "%d", "%m", "%Y"):
        assert forbidden not in prompt


def test_the_prompt_carries_headers_and_counts_but_no_cell(tmp_path, settings):
    profile = _profile(tmp_path, settings, "bal.csv", US_WITH_BALANCE)
    prompt = md.build_prompt(profile)
    assert "Posting Date" in prompt
    assert CANARY not in prompt
    assert CANARY_AMOUNT not in prompt


def test_a_proposal_naming_a_column_that_is_not_there_is_dropped(tmp_path, settings):
    profile = _profile(tmp_path, settings, "pair.csv", DEBIT_CREDIT)
    proposal = md.validate_proposal(
        {"description_column": "Narrative", "amount_column": "Total", "format_name": "x"}, profile
    )
    assert "description_column" not in proposal
    assert "amount_column" not in proposal


def test_a_proposal_contradicting_the_measurements_is_dropped(tmp_path, settings):
    """'Description' exists, but it is not numeric — so it cannot be the amount column."""
    profile = _profile(tmp_path, settings, "pair.csv", DEBIT_CREDIT)
    proposal = md.validate_proposal({"amount_column": "Description"}, profile)
    assert "amount_column" not in proposal


def test_the_model_names_the_money_out_column_of_a_pair(tmp_path, settings):
    profile = _profile(tmp_path, settings, "pair.csv", DEBIT_CREDIT)
    model = StubModel(
        json.dumps(
            {
                "format_name": "acme checking",
                "date_column": "Date",
                "description_column": "Description",
                "debit_column": "Debit",
                "credit_column": "Credit",
            }
        )
    )
    draft = md.draft_for_profile(profile, model, settings=settings)

    assert draft.fields["debit_column"] == "Debit"
    assert draft.fields["sign"] == md.SIGN_DEBIT_NEGATIVE
    assert draft.complete
    assert draft.verification.status == md.VERIFIED
    assert any("direction of every amount" in item for item in draft.confirm)


def test_a_pair_with_no_model_and_no_balance_leaves_the_direction_to_the_operator(
    tmp_path, settings
):
    profile = _profile(tmp_path, settings, "pair.csv", DEBIT_CREDIT)
    draft = md.draft_for_profile(profile, None, settings=settings)

    assert draft.fields["debit_column"] == md.UNRESOLVED
    assert draft.fields["sign"] == md.UNRESOLVED
    assert not draft.complete
    assert draft.verification.status == md.NOT_ATTEMPTED


def test_prose_instead_of_json_is_treated_as_no_answer():
    assert md.parse_proposal("I think the amount column is probably Amount.") == {}


def test_the_local_model_runs_at_temperature_zero():
    """A mapping is not a creative act: the same file must draft the same way twice."""

    class FakeProvider:
        def __init__(self) -> None:
            self.requests: list = []

        def generate(self, request):
            self.requests.append(request)
            return type("R", (), {"text": "{}"})()

    provider = FakeProvider()
    md.LocalModel("some/model", provider=provider).propose("hello")
    assert provider.requests[0].temperature == 0.0


def test_a_model_that_cannot_load_is_reported_not_fatal(tmp_path, settings):
    """No mlx, no weights: the mechanical half still runs and the draft says what is missing."""
    profile = _profile(tmp_path, settings, "pair.csv", DEBIT_CREDIT)
    draft = md.draft_for_profile(profile, StubModel(unavailable=True), settings=settings)

    assert not draft.complete
    assert any("unavailable" in item for item in draft.confirm)
    rendered = md.render_draft(draft)
    assert md.UNRESOLVED in rendered

    path, outcome = md.write_draft(draft, tmp_path / "mappings", force=False)
    assert outcome == "written"
    with pytest.raises(MappingError):
        ColumnMapping.from_yaml(path)


def test_a_model_named_format_cannot_write_outside_the_output_directory(tmp_path, settings):
    profile = _profile(tmp_path, settings, "bal.csv", US_WITH_BALANCE)
    model = StubModel(json.dumps({"format_name": "../../etc/Passwd Bank!!"}))
    draft = md.draft_for_profile(profile, model, settings=settings)

    out = tmp_path / "mappings"
    path, outcome = md.write_draft(draft, out, force=False)
    assert outcome == "written"
    assert path.parent == out
    assert path.name == "etc-passwd-bank.yaml"


# -- 4. verification, and refusing to write ----------------------------------------------------


def test_a_draft_that_cannot_parse_its_own_file_is_not_written(tmp_path, settings):
    profile = _profile(tmp_path, settings, "bad.csv", UNPARSEABLE)
    draft = md.draft_for_profile(profile, None, settings=settings)

    assert draft.complete, "the mapping is complete; it is the file it fails on"
    assert draft.verification.status == md.FAILED

    out = tmp_path / "mappings"
    path, outcome = md.write_draft(draft, out, force=False)
    assert outcome == "refused"
    assert not path.exists()


def test_a_verified_draft_reports_rows_read_against_rows_parsed(tmp_path, settings):
    profile = _profile(tmp_path, settings, "bal.csv", US_WITH_BALANCE)
    draft = md.draft_for_profile(profile, None, settings=settings)

    assert draft.verification.status == md.VERIFIED
    assert draft.verification.rows_read == 4
    assert draft.verification.rows_parsed == 4
    assert draft.verification.total is not None
    # The draft carries the very mapping that parsed the file, not a rebuilt lookalike.
    assert draft.mapping is not None and draft.mapping.date_format == "%m/%d/%Y"


def test_the_arithmetic_half_is_verified_even_while_a_column_is_left_open(tmp_path, settings):
    """Two text columns and no model: the description stays open, the numbers are still proved.

    Neither the description column nor the date format changes a total, so the draft is tried
    under every candidate for them — which is worth doing, and is not worth calling a pass on
    the fields themselves.
    """
    profile = _profile(tmp_path, settings, "pairbal.csv", PAIR_WITH_BALANCE)
    draft = md.draft_for_profile(profile, None, settings=settings)

    assert draft.fields["debit_column"] == "Debit"
    assert draft.fields["description_column"] == md.UNRESOLVED
    assert draft.verification.status == md.VERIFIED_EACH
    assert draft.verification.rows_read == draft.verification.rows_parsed == 3
    assert "change no total" in draft.verification.detail


def test_the_draft_never_claims_a_control_total_it_was_not_given(tmp_path, settings):
    profile = _profile(tmp_path, settings, "bal.csv", US_WITH_BALANCE)
    rendered = md.render_draft(md.draft_for_profile(profile, None, settings=settings))
    assert "unverified" in rendered
    assert "NOT been checked" in rendered


def test_a_verified_draft_round_trips_through_from_yaml(tmp_path, settings):
    profile = _profile(tmp_path, settings, "bal.csv", US_WITH_BALANCE)
    draft = md.draft_for_profile(profile, None, settings=settings)
    path, _ = md.write_draft(draft, tmp_path / "mappings", force=False)

    mapping = ColumnMapping.from_yaml(path)
    assert mapping.amount_column == "Amount"
    assert mapping.date_format == "%m/%d/%Y"


def test_an_existing_mapping_is_not_overwritten_without_force(tmp_path, settings):
    """The operator's reviewed mapping outranks a fresh draft, every time."""
    profile = _profile(tmp_path, settings, "bal.csv", US_WITH_BALANCE)
    draft = md.draft_for_profile(profile, None, settings=settings)
    out = tmp_path / "mappings"
    out.mkdir()
    reviewed = out / f"{draft.name}.yaml"
    reviewed.write_text("# reviewed by a human\n", encoding="utf-8")

    path, outcome = md.write_draft(draft, out, force=False)
    assert outcome == "exists"
    assert path.read_text(encoding="utf-8") == "# reviewed by a human\n"

    path, outcome = md.write_draft(draft, out, force=True)
    assert outcome == "written"
    assert "DRAFT" in path.read_text(encoding="utf-8")


def test_two_formats_the_model_named_alike_do_not_collide(tmp_path, settings):
    """Observed on a live run: a 3B model called two different layouts the same thing."""
    claimed: set[str] = set()
    first = md.unique_name("bank-statement", claimed, 1)
    claimed.add(first)
    second = md.unique_name("bank-statement", claimed, 2)
    assert first != second


def test_files_sharing_a_header_need_one_mapping_not_one_each(tmp_path, settings):
    _write(tmp_path, "jan.csv", MONTH_FIRST)
    _write(tmp_path, "feb.csv", MONTH_FIRST)
    _write(tmp_path, "other.csv", DEBIT_CREDIT)
    groups, refused = md.group_by_signature(md.collect([str(tmp_path)], md.TABLE_EXTS), settings)
    assert not refused
    assert len(groups) == 2
    assert max(len(paths) for paths in groups.values()) == 2


# -- 5. privacy: the values stay on the machine ------------------------------------------------


def test_no_cell_value_reaches_stdout(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("HEARTH_FILE_ROOTS", str(tmp_path))
    _write(tmp_path, "bal.csv", US_WITH_BALANCE)
    out = tmp_path / "mappings"

    assert md.main([str(tmp_path), "--out", str(out), "--no-model"]) == 0
    printed = capsys.readouterr().out

    assert CANARY not in printed
    assert CANARY_AMOUNT not in printed
    assert "COFFEE SHOP" not in printed
    # Headers and classifications are safe, and are what the operator needs.
    assert "Posting Date" in printed
    assert "type=number" in printed


def test_a_failing_parse_withholds_the_message_that_quotes_the_cell(tmp_path, capsys, monkeypatch):
    """ParseError quotes the offending cell. Its coordinates are safe; its message is not."""
    monkeypatch.setenv("HEARTH_FILE_ROOTS", str(tmp_path))
    _write(tmp_path, "bad.csv", UNPARSEABLE)
    out = tmp_path / "mappings"

    assert md.main([str(tmp_path), "--out", str(out), "--no-model"]) == 1
    printed = capsys.readouterr().out

    assert CANARY not in printed
    assert CANARY_AMOUNT not in printed
    assert "withheld" in printed
    assert "NOT written" in printed
    assert not out.exists()


def test_the_total_is_kept_out_of_stdout_unless_asked_for(tmp_path, capsys, monkeypatch):
    """The sum is a figure; stdout is what gets pasted into a chat window. Opt in for it."""
    monkeypatch.setenv("HEARTH_FILE_ROOTS", str(tmp_path))
    _write(tmp_path, "bal.csv", US_WITH_BALANCE)
    out = tmp_path / "mappings"

    md.main([str(tmp_path), "--out", str(out), "--no-model"])
    quiet = capsys.readouterr().out
    assert "written into the draft" in quiet
    assert "1935.13" not in quiet

    md.main([str(tmp_path), "--out", str(out), "--no-model", "--force", "--show-total"])
    loud = capsys.readouterr().out
    assert "sum of amounts -" in loud or "sum of amounts " in loud

    # Either way the figure is recorded in the draft itself, where a reviewer needs it.
    draft = next(out.glob("*.yaml")).read_text(encoding="utf-8")
    assert "sum of amounts" in draft


def test_the_column_profile_never_renders_its_values(tmp_path, settings):
    """A dataclass repr is exactly what ends up in a log line, so the values are not in it."""
    profile = _profile(tmp_path, settings, "bal.csv", US_WITH_BALANCE)
    amount = profile.by_name("Amount")
    assert amount is not None
    assert amount.numbers  # they were measured...
    assert "1313.13" not in repr(amount)  # ...and they are not in the repr
    assert CANARY not in repr(profile)
