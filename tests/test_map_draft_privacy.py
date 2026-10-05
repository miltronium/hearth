"""``scripts/hearth_map_draft.py`` terminal output follows hearth_peek's rule, asserted on OUTPUT.

B-074: map_draft took ``rows[0]`` as the header. On a statement whose first rows were
"Account Holder,Jane Q Public,Premier Checking" and "Acct 4417123412341234,Open,x" it printed
'Jane Q Public' and 'Premier Checking' as column names, drafted a mapping with no skip_rows
(wrong for every file with a preamble), and closed with "No cell value was printed above."
It also printed file names and, under --show-total, a sum computed from the amounts.

The rule now shared with hearth_peek (``hearth.finance.shape``): a header name prints only if
it is made of the fixed column vocabulary; any other column is ``column N (withheld)``; the
header row is the first with >= 2 vocabulary names and the rows above become ``skip_rows``;
files print as ids; no figure computed from values; no exception message.

These tests run the real ``main()`` and read what it printed: the integrator's preamble file,
the model path with a fake local model whose every string is hostile, the failure paths, and a
property test over random synthetic tables in which every preamble line, data cell,
non-vocabulary header, file name and directory name is random text or a digit run.
"""

from __future__ import annotations

import importlib.util
import json
import random
import re
import string
import sys
from pathlib import Path

import pytest

from hearth.config import Settings, get_settings
from hearth.finance.mapping import ColumnMapping
from hearth.finance.parse import parse_rows
from hearth.finance.shape import VOCAB, is_known_column_name
from hearth.mcp.files import read_table

REPO = Path(__file__).resolve().parent.parent
_SCRIPT = REPO / "scripts" / "hearth_map_draft.py"

if "hearth_map_draft" in sys.modules:
    md = sys.modules["hearth_map_draft"]
else:
    _spec = importlib.util.spec_from_file_location("hearth_map_draft", _SCRIPT)
    assert _spec and _spec.loader
    md = importlib.util.module_from_spec(_spec)
    sys.modules["hearth_map_draft"] = md  # @dataclass resolves annotations through sys.modules
    _spec.loader.exec_module(md)

#: The integrator's reproduction (B-074), verbatim shape: two preamble rows above the header.
PREAMBLE = (
    "Account Holder,Jane Q Public,Premier Checking\n"
    "Acct 4417123412341234,Open,x\n"
    "Date,Description,Amount\n"
    "2024-01-02,SECRETMERCHANT ONE,-12.50\n"
    "2024-01-03,PAYROLL ACME,1500.00\n"
)
PREAMBLE_FILE = "acct_4417123412341234_JanePublic_20240131.csv"


@pytest.fixture(autouse=True)
def _fresh_settings():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _run(root: Path, capsys, monkeypatch, *extra: str) -> tuple[int, str]:
    monkeypatch.setenv("HEARTH_FILE_ROOTS", str(root))
    get_settings.cache_clear()
    out_dir = root.parent / f"{root.name}-mappings"
    code = md.main([str(root), "--out", str(out_dir), *extra])
    return code, capsys.readouterr().out


class FakeModel:
    """A local model stand-in whose answer is fixed by the test. Loads nothing."""

    name = "fake"

    def __init__(self, answer: dict | None = None, *, fail: str | None = None) -> None:
        self.answer = answer or {}
        self.fail = fail
        self.prompts: list[str] = []

    def propose(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.fail is not None:
            raise md.ModelUnavailable(self.fail)
        return json.dumps(self.answer)


def _with_model(monkeypatch, model: FakeModel) -> None:
    monkeypatch.setattr(md, "LocalModel", lambda model_id: model)


# -- the integrator's reproduction ------------------------------------------------------------


def test_the_preamble_file_prints_no_preamble_no_file_name_and_drafts_skip_rows(
    tmp_path, capsys, monkeypatch
):
    root = tmp_path / "in"
    root.mkdir()
    (root / PREAMBLE_FILE).write_text(PREAMBLE, encoding="utf-8")

    code, out = _run(root, capsys, monkeypatch, "--no-model")

    assert code == 0, out
    for secret in ("Jane", "Public", "Premier", "Checking", "Holder", "4417", "1234",
                   "20240131", "SECRETMERCHANT", "PAYROLL", "acct_", "Open", "12.50",
                   "1500", str(tmp_path)):
        assert secret not in out, f"{secret!r} printed:\n{out}"
    assert re.findall(r"\d{3,}", out) == [], out
    # Useful, and correct: the real header is named and its row is reported.
    for name in ("Date", "Description", "Amount"):
        assert name in out
    assert "header is row 3; skip_rows: 2" in out
    assert "F1 (.csv)" in out
    assert "verification: verified" in out

    # The draft file is right: real header names, skip_rows past the preamble...
    draft_path = tmp_path / "in-mappings" / "format-1.yaml"
    mapping = ColumnMapping.from_yaml(draft_path)
    assert mapping.skip_rows == 2
    assert (mapping.date_column, mapping.description_column, mapping.amount_column) == (
        "Date", "Description", "Amount")
    # ...and it parses its own file with the real parser.
    rows = read_table(root / PREAMBLE_FILE, Settings(file_roots=str(root)))
    parsed = parse_rows(rows, mapping)
    assert [str(t.amount) for t in parsed] == ["-12.50", "1500.00"]


def test_the_closing_line_states_what_was_printed_and_never_claims_more(
    tmp_path, capsys, monkeypatch
):
    root = tmp_path / "in"
    root.mkdir()
    (root / "s.csv").write_text(PREAMBLE, encoding="utf-8")
    _code, out = _run(root, capsys, monkeypatch, "--no-model")
    assert "No cell value was printed above" not in out  # the old, untrue sentence
    assert "Printed: file ids and extensions" in out
    assert "Not printed: any cell" in out
    assert "any sum or other figure computed from values" in out


def test_files_whose_header_sits_on_different_rows_are_different_formats(
    tmp_path, capsys, monkeypatch
):
    """One mapping has one skip_rows; the same header under a longer preamble needs its own."""
    root = tmp_path / "in"
    root.mkdir()
    (root / "a.csv").write_text(PREAMBLE, encoding="utf-8")
    (root / "b.csv").write_text("Statement for,Jane\n" + PREAMBLE, encoding="utf-8")
    groups, refused = md.group_by_signature(md.collect([str(root)], md.TABLE_EXTS),
                                            Settings(file_roots=str(root)))
    assert not refused
    assert sorted(len(v) for v in groups.values()) == [1, 1]
    code, out = _run(root, capsys, monkeypatch, "--no-model")
    assert code == 0, out
    assert "skip_rows: 2" in out and "skip_rows: 3" in out
    assert "Jane" not in out


def test_a_file_with_no_identifiable_header_is_refused_with_a_fixed_reason(
    tmp_path, capsys, monkeypatch
):
    root = tmp_path / "in"
    root.mkdir()
    (root / "h_99887766.csv").write_text(
        "SECRETMERCHANT ONE,Groceries,Jan 5\nPAYROLL ACME,Income,Jan 6\n", encoding="utf-8")
    code, out = _run(root, capsys, monkeypatch, "--no-model")
    assert code == 1
    assert "no header row identified" in out
    for secret in ("SECRETMERCHANT", "Groceries", "Jan", "PAYROLL", "99887766"):
        assert secret not in out


# -- columns outside the vocabulary -----------------------------------------------------------


def test_a_non_vocabulary_header_cell_is_withheld_everywhere_it_could_appear(
    tmp_path, capsys, monkeypatch
):
    """Header row qualifies (Date, Amount) but two of its cells are values."""
    root = tmp_path / "in"
    root.mkdir()
    (root / "odd.csv").write_text(
        "Date,Acct 99887766 Jane,Amount,owner@example.com\n"
        "2024-01-02,SECRETA,-1.50,SECRETB\n"
        "2024-01-03,SECRETC,2.00,SECRETD\n",
        encoding="utf-8",
    )
    _code, out = _run(root, capsys, monkeypatch, "--no-model")
    for secret in ("99887766", "Jane", "owner", "example", "SECRET"):
        assert secret not in out, f"{secret!r} printed:\n{out}"
    # Two text columns, no model: the description is open, and the verification sentence
    # lists the candidates by position, not by name.
    assert "column 1 (withheld)" in out and "column 3 (withheld)" in out
    assert len(re.findall(r"^ +\[\d\] \(withheld\) ", out, re.M)) == 2


def test_a_parse_failure_names_the_column_only_through_the_rule(tmp_path, capsys, monkeypatch):
    """ParseError carries the column; a non-vocabulary column prints by position only."""
    root = tmp_path / "in"
    root.mkdir()
    (root / "bad_4417.csv").write_text(
        "Valuta Jane,Description,Amount\n"
        "08/25/2026,COFFEE,4.50\n"
        ",SECRETCANARY,-13.13\n",
        encoding="utf-8",
    )
    code, out = _run(root, capsys, monkeypatch, "--no-model")
    assert code == 1
    assert "F1 (.csv): parse failed at row 2 column column 0 (withheld)" in out
    for secret in ("Valuta", "Jane", "SECRETCANARY", "13.13", "4417", "bad_"):
        assert secret not in out, f"{secret!r} printed:\n{out}"


def test_a_refused_file_prints_a_fixed_reason_not_the_path(tmp_path, capsys, monkeypatch):
    inside = tmp_path / "in"
    inside.mkdir()
    outside = tmp_path / "out_4417"
    outside.mkdir()
    (outside / "acct_99887766.csv").write_text(PREAMBLE, encoding="utf-8")
    monkeypatch.setenv("HEARTH_FILE_ROOTS", str(inside))
    get_settings.cache_clear()
    code = md.main([str(outside / "acct_99887766.csv"), "--out", str(tmp_path / "m"),
                    "--no-model"])
    out = capsys.readouterr().out
    assert code == 1
    assert "outside HEARTH_FILE_ROOTS" in out
    assert "99887766" not in out and "4417" not in out


def test_index_out_maps_ids_to_paths_locally_and_prints_none_of_it(
    tmp_path, capsys, monkeypatch
):
    root = tmp_path / "in"
    root.mkdir()
    (root / PREAMBLE_FILE).write_text(PREAMBLE, encoding="utf-8")
    index = tmp_path / "index.tsv"
    _code, out = _run(root, capsys, monkeypatch, "--no-model", "--index-out", str(index))
    assert "index.tsv" not in out and "4417" not in out
    lines = index.read_text(encoding="utf-8").splitlines()
    assert lines[0] == f"F1\t{root / PREAMBLE_FILE}"
    assert lines[1] == f"D1\t{tmp_path / 'in-mappings' / 'format-1.yaml'}"


# -- the model path: it may SEE what it needs; nothing it returns prints raw -------------------

HOSTILE_HEADER = (
    "Date,Description,Amount,Jane Public 4417\n"
    "2024-01-02,COFFEE,-4.50,7.25\n"
    "2024-01-03,PAYROLL,1500.00,3.10\n"
    "2024-01-04,RENT,-900.00,9.99\n"
)


def test_model_output_is_printed_only_through_the_rule(tmp_path, capsys, monkeypatch):
    root = tmp_path / "in"
    root.mkdir()
    (root / "s.csv").write_text(HOSTILE_HEADER, encoding="utf-8")
    model = FakeModel({
        "format_name": "Jane Public 4417 Premier",
        "date_column": "Date",
        "description_column": "Description",
        "amount_column": "Jane Public 4417",  # a real numeric column, non-vocabulary name
    })
    _with_model(monkeypatch, model)

    _code, out = _run(root, capsys, monkeypatch, "--model", "fake/local")

    # The local model was shown the real header (allowed: it is local)...
    assert model.prompts and "Jane Public 4417" in model.prompts[0]
    # ...but nothing it returned reached stdout raw.
    for secret in ("Jane", "Public", "4417", "Premier", "jane", "premier"):
        assert secret not in out, f"{secret!r} printed:\n{out}"
    assert "amount_column is column 3 (withheld) — a model proposal" in out
    assert "(file name withheld: proposed by the model; see --index-out)" in out
    # The local draft keeps the real name and the model's label, where the operator needs them.
    drafts = list((tmp_path / "in-mappings").glob("*.yaml"))
    assert [p.name for p in drafts] == ["jane-public-4417-premier.yaml"]
    assert 'amount_column: "Jane Public 4417"' in drafts[0].read_text(encoding="utf-8")


def test_a_vocabulary_only_model_name_is_printed(tmp_path, capsys, monkeypatch):
    root = tmp_path / "in"
    root.mkdir()
    (root / "s.csv").write_text(HOSTILE_HEADER, encoding="utf-8")
    _with_model(monkeypatch, FakeModel({"format_name": "card transactions"}))
    _code, out = _run(root, capsys, monkeypatch, "--model", "fake/local")
    assert "written into the --out directory: card-transactions.yaml" in out


def test_a_model_failure_message_is_kept_out_of_the_terminal(tmp_path, capsys, monkeypatch):
    root = tmp_path / "in"
    root.mkdir()
    (root / "s.csv").write_text(HOSTILE_HEADER, encoding="utf-8")
    _with_model(monkeypatch, FakeModel(fail="OSError: /Users/jane/SECRETPATH/weights echo"))
    _code, out = _run(root, capsys, monkeypatch, "--model", "fake/local")
    assert "SECRETPATH" not in out and "jane" not in out
    assert "the local model was unavailable (reason withheld from the terminal" in out
    draft = next((tmp_path / "in-mappings").glob("*.yaml")).read_text(encoding="utf-8")
    assert "SECRETPATH" in draft  # the operator still gets the reason, locally


# -- property: random synthetic tables ----------------------------------------------------------

KNOWN_HEADERS = [
    "Date", "Posting Date", "Transaction Date", "Description", "Memo", "Payee", "Amount",
    "Debit", "Credit", "Balance", "Running Balance", "Type", "Category", "Check Number",
    "Reference", "Status",
]


def _token_source(rng: random.Random):
    # Anything the script could print by itself is in its source (or the shared rule's, or a
    # finance module whose constants it prints); a random token that is a substring of those
    # could match the output by coincidence, so it is re-drawn.
    sources = [_SCRIPT, REPO / "src" / "hearth" / "finance" / "shape.py",
               *sorted((REPO / "src" / "hearth" / "finance").glob("*.py"))]
    corpus = "".join(p.read_text(encoding="utf-8") for p in sources).lower()

    def token() -> str:
        while True:
            t = "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(5, 9)))
            if t not in VOCAB and t not in corpus:
                return t
    return token


def _cell(rng: random.Random, token, secrets: list[str]) -> str:
    kind = rng.random()
    if kind < 0.4:
        words = [token() for _ in range(rng.randint(1, 3))]
        secrets.extend(words)
        return " ".join(w.capitalize() if rng.random() < 0.5 else w.upper() for w in words)
    if kind < 0.55:
        return f"{rng.randint(100, 99999)}.{rng.randint(10, 99)}"
    if kind < 0.65:
        return f"-{rng.randint(100, 9999)}.{rng.randint(10, 99)}"
    if kind < 0.8:
        return f"20{rng.randint(10, 29)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    if kind < 0.9:
        word = token()
        secrets.append(word)
        return f"{word.upper()} #{rng.randint(100, 9_999_999)}"
    return str(rng.randint(100, 10**12))


def _table(rng: random.Random, token):
    """Rows, their random secrets, the vocabulary names a reader should see, the preamble size.

    Columns are typed (date / text / number) about half the time so drafts get far enough to
    verify, name model-chosen columns, and fail parses; the rest of the time cells are mixed.
    """
    width = rng.randint(2, 6)
    secrets: list[str] = []
    rows: list[list[str]] = []
    for _ in range(rng.randint(0, 4)):  # preamble: random text, sometimes full width
        n = width if rng.random() < 0.5 else rng.randint(1, width)
        rows.append([_cell(rng, token, secrets) for _ in range(n)] + [""] * (width - n))
    preamble = len(rows)
    shown: list[str] = []
    identifiable = False
    if rng.random() < 0.8:
        header = []
        for _ in range(width):
            if rng.random() < 0.6:
                name = rng.choice(KNOWN_HEADERS)
                header.append(name)
            else:
                header.append(_cell(rng, token, secrets))
        rows.append(header)
        identifiable = sum(1 for c in header if is_known_column_name(c)) >= 2
        if identifiable:
            shown = [c for c in header if is_known_column_name(c)]
    typed = rng.random() < 0.7
    # Typed layouts: a date column, a text column, then signed amounts, magnitudes, or a
    # debit/credit pair (exactly one side filled per row), in shuffled order.
    kinds = ["d", "t", "n"][:width] + [rng.choice("ttm") for _ in range(width - 3)]
    if width >= 4 and rng.random() < 0.4:
        kinds[2], kinds[3] = "D", "C"
    rng.shuffle(kinds)
    n_rows = rng.randint(2, 15)
    for r in range(n_rows):
        row = []
        side = rng.choice("DC")
        for k in kinds:
            if not typed:
                row.append(_cell(rng, token, secrets))
            elif k == "d":
                row.append(f"20{rng.randint(10, 29)}-{rng.randint(1, 12):02d}-"
                           f"{rng.randint(1, 28):02d}")
            elif k == "n":  # signed: the first row negative, the second positive, then random
                sign = "-" if r == 0 else "" if r == 1 else rng.choice(["", "-"])
                row.append(f"{sign}{rng.randint(1, 99999)}.{rng.randint(10, 99)}")
            elif k == "m":
                row.append(f"{rng.randint(1, 99999)}.{rng.randint(10, 99)}")
            elif k in "DC":
                row.append(f"{rng.randint(1, 9999)}.{rng.randint(10, 99)}" if k == side else "")
            else:
                word = token()
                secrets.append(word)
                row.append(f"{word.upper()} {rng.randint(100, 99999)}")
        rows.append(row)
    if typed and n_rows > 2 and rng.random() < 0.35:  # a blank date: the trial parse fails
        rows[-1][kinds.index("d")] = ""
    return rows, secrets, shown, (preamble if identifiable else None)


def _csv(rows: list[list[str]]) -> str:
    return "\n".join(",".join(f'"{c}"' if "," in c else c for c in r) for r in rows) + "\n"


def _hostile_model(rng: random.Random, token, secrets: list[str]) -> FakeModel:
    """A fake local model that answers with whatever header text it was shown, plus junk."""

    class Echo(FakeModel):
        def propose(self, prompt: str) -> str:
            names = re.findall(r"\[\d+\] '([^']*)'", prompt)
            pick = (lambda: rng.choice(names)) if names else token
            junk = token()
            secrets.append(junk)
            single = rng.random() < 0.5
            return json.dumps({
                "format_name": f"{junk} {pick()} {rng.randint(100, 99999)}",
                "date_column": pick(), "description_column": pick(),
                "amount_column": pick() if single else None,
                "debit_column": None if single else pick(),
                "credit_column": None if single else pick(),
            })

    return Echo()


@pytest.mark.parametrize("seed", range(6))
def test_property_no_random_text_or_digit_run_reaches_stdout(
    tmp_path, capsys, monkeypatch, seed
):
    rng = random.Random(7000 + seed)
    token = _token_source(rng)
    for case in range(20):
        root = tmp_path / f"case{case}"
        all_secrets: list[str] = []
        expected_names: list[str] = []
        expected_skips: dict[Path, int] = {}
        for _ in range(rng.randint(1, 3)):
            rows, secrets, shown, skip = _table(rng, token)
            dir_word, file_word = token(), token()
            all_secrets += secrets + [dir_word, file_word]
            folder = root / f"{dir_word}_{rng.randint(100, 99999)}"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{file_word}_{rng.randint(100, 10**10)}.csv"
            path.write_text(_csv(rows), encoding="utf-8")
            expected_names += shown
            if skip is not None:
                expected_skips[path] = skip

        use_model = rng.random() < 0.5
        if use_model:
            _with_model(monkeypatch, _hostile_model(rng, token, all_secrets))
            _code, out = _run(root, capsys, monkeypatch, "--model", "fake/local")
        else:
            _code, out = _run(root, capsys, monkeypatch, "--no-model")

        lowered = out.lower()
        leaked = sorted({s for s in all_secrets if s.lower() in lowered})
        assert leaked == [], f"seed {seed} case {case} leaked {leaked}:\n{out}"
        runs = re.findall(r"\d{3,}", out)
        assert runs == [], f"seed {seed} case {case} printed digit runs {runs}:\n{out}"
        for name in expected_names:  # usefulness: identifiable headers are named
            assert name in out, f"seed {seed} case {case}: {name!r} not shown\n{out}"

        # correctness: every identifiable file is grouped under its real header row
        settings = Settings(file_roots=str(root))
        groups, _refused = md.group_by_signature(sorted(expected_skips), settings)
        for (at, _sig), paths in groups.items():
            for p in paths:
                assert at == expected_skips[p], f"seed {seed} case {case}: skip_rows wrong"
