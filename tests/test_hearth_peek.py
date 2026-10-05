"""scripts/hearth_peek.py must never print a cell value or a file name, asserted on its OUTPUT.

The output is made to be pasted to a cloud agent. Two earlier versions broke that promise:
the first printed a preamble as "header names" (B-044); the second guessed which text cells
were labels, and printed "Jane Q Public", "Premier Checking" and a headerless file's merchant
names (B-063). The script now prints a header cell only if it is made of a fixed vocabulary of
column words, and shows files as ordinal ids.

These tests run the real script and check what it printed: the reviewer's three synthetic
files (tests/fixtures/peek/), hand-written shapes with a marker planted in every non-header
cell, and a property test over many random synthetic tables whose every cell, header,
preamble line, file name and directory name is random text or a digit run.
"""

from __future__ import annotations

import importlib.util
import json
import random
import re
import shutil
import string
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "hearth_peek.py"
FIXTURES = REPO / "tests" / "fixtures" / "peek"
#: The shared privacy rule (VOCAB, header detection, file ids); its fixed strings print too.
SHAPE = REPO / "src" / "hearth" / "finance" / "shape.py"

_HOSTILE_SPEC = importlib.util.spec_from_file_location(
    "hostile_statements", REPO / "tests" / "fixtures" / "hostile_statements.py"
)
hostile = importlib.util.module_from_spec(_HOSTILE_SPEC)
_HOSTILE_SPEC.loader.exec_module(hostile)
#: Strings planted in workbook metadata; none may reach stdout or stderr.
META_SECRETS = ("Jane", "Public", "HOLDER", "SSN", "123-45", "6789", "987654321", "JANE_Q")


def _load_peek():
    spec = importlib.util.spec_from_file_location("hearth_peek", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_on(root: Path, monkeypatch, capsys, *extra: str) -> tuple[int, str]:
    """Run the real main() and return (exit code, stdout).

    stderr and the warnings channel are part of what an operator pastes, so every run also
    asserts both are empty (B-090). Warnings are recorded with every filter forced to
    "always": left to pytest's own capture, a ``warnings.warn`` quoting file metadata would be
    swallowed by the test harness and never fail anything, while reaching a real terminal.
    """
    monkeypatch.setenv("HEARTH_FILE_ROOTS", str(root))
    from hearth.config import get_settings

    get_settings.cache_clear()
    peek = _load_peek()
    monkeypatch.setattr(sys, "argv", ["hearth_peek.py", str(root), *extra])
    try:
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            code = peek.main()
    finally:
        get_settings.cache_clear()
    captured = capsys.readouterr()
    assert captured.err == "", f"stderr is not empty:\n{captured.err}"
    assert [str(w.message) for w in seen] == [], "a warning was emitted during the run"
    return code, captured.out


def _run(tmp_path, monkeypatch, capsys, files: dict[str, str]) -> str:
    for name, body in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
    return _run_on(tmp_path, monkeypatch, capsys)[1]


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


# -- the reviewer's B-063 reproductions --------------------------------------------------


def test_the_reviewer_fixtures_print_no_value(tmp_path, monkeypatch, capsys):
    for f in FIXTURES.iterdir():
        shutil.copy(f, tmp_path / f.name)
    code, out = _run_on(tmp_path, monkeypatch, capsys)
    assert code == 0
    for secret in (
        "Jane", "Public", "Premier", "Checking", "Holder", "SECRETMERCHANT", "PAYROLL",
        "ACME", "LANDLORD", "BOB", "Groceries", "Income", "Housing", "Jan 5",
        "headerless", "keys.json", "preamble.csv",  # file names
    ):
        assert secret not in out, f"{secret!r} printed:\n{out}"
    # Still useful: the real header below the full-width preamble is found and named,
    # and the one vocabulary JSON key is shown while the value-like key is withheld.
    for name in ("Date", "Description", "Amount", "Acct"):
        assert name in out
    assert "skip_rows: 1" in out
    assert "no header row identified" in out  # headerless.csv


def test_a_full_width_all_text_preamble_row_is_not_printed(tmp_path, monkeypatch, capsys):
    out = _run(tmp_path, monkeypatch, capsys, {
        "s.csv": "Account Holder,Jane Q Public,Premier Checking\n"
                 "Date,Description,Amount\n2024-01-02,SECRETMERCHANT ONE,-12.50\n",
    })
    assert "Jane" not in out and "Premier" not in out and "SECRETMERCHANT" not in out
    assert "skip_rows: 1" in out


def test_a_headerless_all_text_file_prints_no_names(tmp_path, monkeypatch, capsys):
    out = _run(tmp_path, monkeypatch, capsys, {
        "h.csv": "SECRETMERCHANT ONE,Groceries,Jan 5\nPAYROLL ACME,Income,Jan 6\n",
    })
    assert "SECRETMERCHANT" not in out and "Groceries" not in out and "Jan" not in out
    assert "no header row identified" in out


def test_json_keys_are_printed_only_when_in_the_vocabulary(tmp_path, monkeypatch, capsys):
    out = _run(tmp_path, monkeypatch, capsys, {
        "k.json": '[{"Jane Q Public": "x", "Amount": "1.00", "Date": "2024-01-01"}]',
    })
    assert "Jane" not in out
    assert "Amount" in out and "Date" in out and "(withheld)" in out


# -- B-090: workbook metadata on stderr ---------------------------------------------------


def _hostile_workbook_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "statement.xlsx").write_bytes(hostile.xlsx_with_metadata())
    return root


def test_workbook_metadata_reaches_neither_stdout_nor_stderr_in_a_real_process(tmp_path):
    """The integrator's reproduction, run as the operator runs it: a separate process.

    In-process, pytest owns sys.stderr and the warnings machinery, and either can hide a
    leak. A subprocess has neither: whatever reaches its stderr is what reaches a terminal.
    Before B-090 this printed ``UserWarning: Unknown type for HOLDER Jane Q Public SSN
    123-45-6789`` on stderr.
    """
    pytest.importorskip("openpyxl")
    root = _hostile_workbook_dir(tmp_path / "in")
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "HEARTH_FILE_ROOTS": str(root),
           "HEARTH_HOME": str(tmp_path / ".hearth"), "PYTHONWARNINGS": "always"}
    proc = subprocess.run([sys.executable, str(SCRIPT), str(root)], capture_output=True,
                          text=True, env=env, timeout=120, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stderr == "", f"stderr:\n{proc.stderr}"
    for secret in META_SECRETS + ("SECRETMERCHANT",):
        assert secret not in proc.stdout, f"{secret!r} printed:\n{proc.stdout}"
    assert "Date" in proc.stdout and "Amount" in proc.stdout  # still useful


def test_workbook_metadata_is_not_printed_in_process_either(tmp_path, monkeypatch, capsys):
    pytest.importorskip("openpyxl")
    root = _hostile_workbook_dir(tmp_path / "in")
    code, out = _run_on(root, monkeypatch, capsys)  # asserts stderr and warnings are empty
    assert code == 0
    for secret in META_SECRETS:
        assert secret not in out


# -- file names and paths ----------------------------------------------------------------


def test_file_and_directory_names_are_never_printed(tmp_path, monkeypatch, capsys):
    body, markers = _statement([], "Posting Date,Description,Amount,Balance")
    out = _run(tmp_path, monkeypatch, capsys, {
        "JaneDoe_4417/chk_99887766_2026-01.csv": body,
        "JaneDoe_4417/notes.pdf.txt": "SECRET TEXT LAYER 123456\n",
    })
    for secret in ("JaneDoe", "4417", "99887766", "chk_", "notes", "SECRET", str(tmp_path)):
        assert secret not in out, f"{secret!r} printed:\n{out}"
    assert "F1 (.csv)" in out and "F2 (.txt)" in out
    assert [m for m in markers if m in out] == []


def test_index_out_writes_the_mapping_locally_and_prints_none_of_it(
    tmp_path, monkeypatch, capsys
):
    data = tmp_path / "data"
    data.mkdir()
    (data / "acct_55512345.csv").write_text("Date,Amount\n2024-01-01,1.00\n")
    index = tmp_path / "index.tsv"
    _code, out = _run_on(data, monkeypatch, capsys, "--index-out", str(index))
    assert "55512345" not in out and "index.tsv" not in out
    assert index.read_text() == f"F1\t{data / 'acct_55512345.csv'}\n"


def test_a_gate_refusal_prints_a_fixed_reason_not_the_path(tmp_path, monkeypatch, capsys):
    inside = tmp_path / "in"
    inside.mkdir()
    outside = tmp_path / "out_4417"
    outside.mkdir()
    (outside / "acct_99887766.csv").write_text("Date,Amount\n")
    monkeypatch.setenv("HEARTH_FILE_ROOTS", str(inside))
    from hearth.config import get_settings

    get_settings.cache_clear()
    peek = _load_peek()
    monkeypatch.setattr(sys, "argv", ["hearth_peek.py", str(outside / "acct_99887766.csv")])
    try:
        assert peek.main() == 1
    finally:
        get_settings.cache_clear()
    out = capsys.readouterr().out
    assert "99887766" not in out and "4417" not in out
    assert "outside HEARTH_FILE_ROOTS" in out


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
    assert "MSGSECRET" not in out and "4417" not in out and "ValueError" in out


# -- earlier shapes (B-044) still hold ---------------------------------------------------


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
    assert [m for m in markers if m in out] == [], out
    assert "4417" not in out and "Jane" not in out
    for name in ("Posting Date", "Description", "Amount", "Balance"):
        assert name in out
    assert "skip_rows: 3" in out and "3 preamble line(s)" in out
    assert "6 rows," in out  # data rows counted from below the real header


def test_value_cells_in_a_header_width_row_are_withheld(tmp_path, monkeypatch, capsys):
    body, markers = _statement([], "Date,Acct 99887766,Amount,owner@example.com")
    out = _run(tmp_path, monkeypatch, capsys, {"odd.csv": body})
    assert [m for m in markers if m in out] == []
    assert "99887766" not in out and "owner" not in out
    assert "Date" in out and "Amount" in out
    assert len(re.findall(r"^ +\d+  \(withheld\) ", out, re.M)) == 2


def test_a_plain_statement_still_shows_its_header(tmp_path, monkeypatch, capsys):
    body, markers = _statement([], "Posting Date,Description,Amount,Balance")
    out = _run(tmp_path, monkeypatch, capsys, {"plain.csv": body})
    assert [m for m in markers if m in out] == []
    assert "Posting Date" in out and "skip_rows: 0" in out and "preamble line(s) counted" not in out


@pytest.mark.parametrize(
    ("cell", "known"),
    [
        ("Posting Date", True), ("  AMOUNT ($) ", True), ("Check #", True),
        ("running_balance", True), ("Debit/Credit", True), ("Jane", False),
        ("Amount 2024", False), ("Premier Checking", False), ("", False),
        ("Date Date Date Date Date", False), ("Ref1", False),
    ],
)
def test_the_allowlist_predicate(cell, known):
    assert _load_peek().is_known_column_name(cell) is known


# -- property: random synthetic tables ---------------------------------------------------

KNOWN_HEADERS = [
    "Date", "Posting Date", "Transaction Date", "Description", "Memo", "Payee", "Amount",
    "Debit", "Credit", "Balance", "Running Balance", "Type", "Category", "Check Number",
    "Reference", "Status",
]


def _make_token_source(rng: random.Random):
    peek = _load_peek()
    # Anything the script could print by itself is in its source; a random token that is a
    # substring of the source could match the output by coincidence, so it is re-drawn.
    source = (SCRIPT.read_text() + SHAPE.read_text()).lower()

    def token() -> str:
        while True:
            t = "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(5, 9)))
            if t not in peek.VOCAB and t not in source:
                return t
    return token


def _random_cell(rng: random.Random, token, secrets: list[str]) -> str:
    kind = rng.random()
    if kind < 0.45:
        words = [token() for _ in range(rng.randint(1, 3))]
        secrets.extend(words)
        return " ".join(w.capitalize() if rng.random() < 0.5 else w.upper() for w in words)
    if kind < 0.6:
        return f"{rng.randint(100, 99999)}.{rng.randint(10, 99)}"
    if kind < 0.7:
        return f"-{rng.randint(100, 9999)}.{rng.randint(10, 99)}"
    if kind < 0.8:
        return f"20{rng.randint(10, 29)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    if kind < 0.9:
        word = token()
        secrets.append(word)
        return f"{word.upper()} #{rng.randint(100, 9_999_999)}"
    return str(rng.randint(100, 10**12))


def _random_table(rng: random.Random, token) -> tuple[list[list[str]], list[str], list[str]]:
    """Rows, the random secrets in them, and the known header names a reader should see."""
    width = rng.randint(2, 6)
    secrets: list[str] = []
    rows: list[list[str]] = []
    for _ in range(rng.randint(0, 4)):  # preamble, sometimes full-width, all random text
        n = width if rng.random() < 0.5 else rng.randint(1, width)
        rows.append([_random_cell(rng, token, secrets) for _ in range(n)] + [""] * (width - n))
    shown: list[str] = []
    if rng.random() < 0.75:
        header = []
        for _ in range(width):
            if rng.random() < 0.6:
                name = rng.choice(KNOWN_HEADERS)
                header.append(name)
                shown.append(name)
            else:
                header.append(_random_cell(rng, token, secrets))
        rows.append(header)
        if sum(1 for c in header if c in KNOWN_HEADERS) < 2:
            shown = []  # not identifiable as a header; nothing need be shown
    for _ in range(rng.randint(1, 20)):
        rows.append([_random_cell(rng, token, secrets) for _ in range(width)])
    return rows, secrets, shown


def _csv(rows: list[list[str]]) -> str:
    return "\n".join(",".join(f'"{c}"' if "," in c else c for c in r) for r in rows) + "\n"


@pytest.mark.parametrize("seed", range(6))
def test_property_no_random_cell_text_or_digit_run_is_ever_printed(
    tmp_path, monkeypatch, capsys, seed
):
    rng = random.Random(1000 + seed)
    token = _make_token_source(rng)
    for case in range(25):
        root = tmp_path / f"case{case}"
        all_secrets: list[str] = []
        expected: list[str] = []
        for _ in range(rng.randint(1, 3)):
            rows, secrets, shown = _random_table(rng, token)
            dir_word, file_word = token(), token()
            all_secrets += secrets + [dir_word, file_word]
            folder = root / f"{dir_word}_{rng.randint(100, 99999)}"
            folder.mkdir(parents=True, exist_ok=True)
            name = f"{file_word}_{rng.randint(100, 10**10)}_{rng.randint(1000, 9999)}"
            if rng.random() < 0.2:
                keys = rows[-len(rows) // 2 - 1]
                objs = [dict(zip(keys, r, strict=False)) for r in rows[-len(rows) // 2:]]
                (folder / f"{name}.json").write_text(json.dumps(objs))
                all_secrets += [k.lower() for k in keys]
            elif rng.random() < 0.2:  # a workbook whose metadata also carries secrets
                (folder / f"{name}.xlsx").write_bytes(hostile.xlsx_with_metadata(rows))
                all_secrets += [s.lower() for s in META_SECRETS]
                expected += shown
            else:
                (folder / f"{name}.csv").write_text(_csv(rows))
                expected += shown
        _code, out = _run_on(root, monkeypatch, capsys)
        lowered = out.lower()
        leaked = sorted({s for s in all_secrets if s and s.lower() in lowered
                         and not _load_peek().is_known_column_name(s)})
        assert leaked == [], f"seed {seed} case {case} leaked {leaked}:\n{out}"
        runs = re.findall(r"\d{3,}", out)
        assert runs == [], f"seed {seed} case {case} printed digit runs {runs}:\n{out}"
        for name in expected:  # usefulness: identifiable headers are named
            assert name in out, f"seed {seed} case {case}: {name!r} not shown\n{out}"
