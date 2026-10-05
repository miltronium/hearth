"""scripts/hearth_peek.py must never print a cell — asserted on its actual OUTPUT.

The script's promise is "headers and type guesses, never a value", and its output is made to
be pasted to a cloud agent. It used to take rows[0] as the header, so a bank export with an
account-name/number preamble printed that preamble as "header names" and then announced "No
cell values were printed". These tests plant a unique marker in every preamble line and every
data cell, run the real script, and require that no marker appears anywhere in what it prints.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load_peek():
    path = REPO / "scripts" / "hearth_peek.py"
    spec = importlib.util.spec_from_file_location("hearth_peek", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(tmp_path, monkeypatch, capsys, files: dict[str, str]) -> str:
    for name, body in files.items():
        (tmp_path / name).write_text(body)
    monkeypatch.setenv("HEARTH_FILE_ROOTS", str(tmp_path))
    from hearth.config import get_settings

    get_settings.cache_clear()
    peek = _load_peek()
    monkeypatch.setattr(sys, "argv", ["hearth_peek.py", str(tmp_path)])
    try:
        peek.main()
    finally:
        get_settings.cache_clear()
    return capsys.readouterr().out


def _statement(preamble: list[str], header: str, rows: int = 6) -> tuple[str, list[str]]:
    """A CSV whose every non-header cell carries a unique marker; returns (text, markers)."""
    markers: list[str] = []
    lines = []
    for i, text in enumerate(preamble):
        tag = f"PRE{i}SECRET"
        markers.append(tag)
        lines.append(text.replace("{tag}", tag))
    lines.append(header)
    for r in range(rows):
        tag = f"ROW{r}SECRET"
        markers.append(tag)
        lines.append(f"2026-01-{r + 1:02d},Coffee {tag},-{r + 3}.50,{100 + r}.00")
    return "\n".join(lines) + "\n", markers


def test_a_preamble_above_the_header_is_never_printed(tmp_path, monkeypatch, capsys):
    body, markers = _statement(
        [
            "Account Name: Jane {tag} Doe,",
            "Account Number: XXXX-4417 {tag},",
            "Statement period,{tag}",
        ],
        "Posting Date,Description,Amount,Balance",
    )
    out = _run(tmp_path, monkeypatch, capsys, {"august.csv": body})
    leaked = [m for m in markers if m in out]
    assert leaked == [], f"cell values printed: {leaked}\n{out}"
    assert "4417" not in out and "Jane" not in out
    for name in ("Posting Date", "Description", "Amount", "Balance"):
        assert name in out  # the real header IS found and shown
    assert "3 preamble line(s)" in out
    assert "(6 rows" in out  # data rows counted from below the real header


def test_a_file_with_no_header_row_prints_no_names(tmp_path, monkeypatch, capsys):
    body, markers = _statement([], "2026-02-01,Coffee HDRSECRET,-1.00,10.00")
    markers.append("HDRSECRET")
    out = _run(tmp_path, monkeypatch, capsys, {"noheader.csv": body})
    assert [m for m in markers if m in out] == []
    assert "no header row identified" in out


def test_a_row_with_value_like_cells_is_never_taken_as_the_header(tmp_path, monkeypatch, capsys):
    # A first row carrying an account number and an e-mail has the right width, but a header
    # must read as labels in EVERY cell — so no header is identified and no name is printed.
    body, markers = _statement([], "Date,Acct 99887766,Amount,owner@example.com")
    out = _run(tmp_path, monkeypatch, capsys, {"odd.csv": body})
    assert [m for m in markers if m in out] == []
    assert "99887766" not in out and "owner@example.com" not in out
    assert "no header row identified" in out


def test_a_parser_error_message_is_not_printed(tmp_path, monkeypatch, capsys):
    peek = _load_peek()

    def explode(path):
        raise ValueError("could not parse line: 'Jane Doe,4417 MSGSECRET'")

    monkeypatch.setattr(peek, "_inspect", explode)
    (tmp_path / "broken.csv").write_text("x\n")
    monkeypatch.setenv("HEARTH_FILE_ROOTS", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["hearth_peek.py", str(tmp_path)])
    assert peek.main() == 1
    out = capsys.readouterr().out
    assert "MSGSECRET" not in out and "ValueError" in out


@pytest.mark.parametrize("header", ["Posting Date,Description,Amount,Balance"])
def test_a_plain_statement_still_shows_its_header(tmp_path, monkeypatch, capsys, header):
    body, markers = _statement([], header)
    out = _run(tmp_path, monkeypatch, capsys, {"plain.csv": body})
    assert [m for m in markers if m in out] == []
    assert "Posting Date" in out and "preamble" not in out


def test_an_account_number_row_of_header_width_is_not_taken_as_the_header(
    tmp_path, monkeypatch, capsys
):
    # Same width as the real header, and every other cell is plain text: only the digit-run
    # rule stands between this preamble row and being printed as "header names".
    body, markers = _statement(
        # "Acct ..." is neither date- nor money-shaped (a bare number would already be
        # caught as money-like), so this isolates the digit-run rule.
        ["Checking,Acct 4417123412341234,Open,Primary"],
        "Posting Date,Description,Amount,Balance",
    )
    out = _run(tmp_path, monkeypatch, capsys, {"acct.csv": body})
    assert "4417123412341234" not in out
    assert [m for m in markers if m in out] == []
    assert "Posting Date" in out and "1 preamble line(s)" in out
