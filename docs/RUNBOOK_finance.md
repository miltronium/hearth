# RUNBOOK — Your own bank statements, on this machine, with every number checkable

This is the operator's guide to the finance path: get a bank export onto disk, tell HEARTH its
layout once, parse it, **prove the parse against a figure the bank printed**, categorize it,
compute the totals in Python, and only then let a local model write prose about them. At the
end you can take any number in that prose and walk it back to the exact line of the exact file
it came from.

That last property is the point of the whole thing. A model asked about your finances will
produce a number that is right often enough to be trusted and wrong often enough to matter, and
**nothing in the output distinguishes the two**. The store (`src/hearth/finance/store.py`)
changes the question from *"do I trust what it said?"* to *"does this number match a set of
rows?"* — and only the second one is answerable.

> ## The one rule that is not negotiable
>
> **Never paste statement contents into a cloud agent.** Not a row, not a "sample", not "just
> the amounts to check my maths". A cloud agent that reads your file has already sent it,
> before HEARTH is involved at all — sealing HEARTH does not unsend it (`docs/PRIVACY.md`, *The
> caller caveat*).
>
> You do not have to. The path-taking tools exist precisely so a file can be *named* instead of
> *pasted*: you hand over a path, HEARTH opens it locally, and the contents never enter any
> agent's context. Column **headers** are safe to discuss ("Posting Date", "Amount", "Balance").
> The cells underneath are not.

---

## 1. Staging: `~/hearth-statements`

Keep exports in one directory that nothing else lives in:

```sh
mkdir -p ~/hearth-statements/incoming ~/hearth-statements/mappings
chmod 700 ~/hearth-statements
```

- `incoming/` — the exports, exactly as the bank produced them. Do not edit them by hand; if a
  row is wrong, fix it at the bank and re-export. Ingest is versioned (§6), so a corrected
  export is cheap and an edited one is untraceable.
- `mappings/` — one small YAML file per bank account, describing that bank's column layout.
  These contain no financial data, only column names, and are the one artefact here worth
  keeping in a backup you can read.

Now open the allowlist to exactly that directory and nothing else:

```sh
export HEARTH_FILE_ROOTS="$HOME/hearth-statements"
```

**Why it is scoped this tight.** The path-taking MCP tools are an arbitrary-file-read primitive
handed to an agent. `HEARTH_FILE_ROOTS` is the only thing standing between that agent and the
rest of your disk, and it is **deny-by-default**: unset, every file read is refused — there is
no implicit root, not even `$HOME` or the working directory. Paths are fully resolved (`..` is
collapsed, symlinks are followed) *before* the containment check, so neither traversal nor a
symlink planted inside the root escapes it. Widening this to `$HOME` would hand the same agent
your keys, your mail, and your other projects. Point it at the statements directory, and
nothing else.

---

## 2. Author a mapping — without ever reading a value

HEARTH refuses to guess which column is the amount. Guessing is the failure this whole package
is built around: **a mis-parsed statement produces a plausible number, not an error.** A
"Balance" column read as the amount still sums, still reconciles against a control total
derived from itself, and is wrong by an amount nobody can see. So you state the layout once.

To write the mapping you need the headers and a sense of what each column holds. You do **not**
need the values, and `scripts/hearth_peek.py` enforces that distinction mechanically — it
prints the header row, the row count, and a per-column *type guess*, and **never emits a cell**:

```sh
uv run --no-sync python scripts/hearth_peek.py ~/hearth-statements/incoming/august.csv
```

```
  ~/hearth-statements/incoming/august.csv
  128 data rows, 5 columns

    #  header                             looks like
  ---  ---------------------------------- ----------------------------------------
    0  Posting Date                       date-like
    1  Description                        text
    2  Amount                             number-like (accounting negatives present)
    3  Balance                            number-like
    4  Type                               text
```

That output is safe to read aloud, put in a note, or paste to a cloud agent for help writing
the YAML. Write the mapping into `~/hearth-statements/mappings/acme-checking.yaml`:

```yaml
bank: Acme Bank — checking
date_column: Posting Date
description_column: Description
amount_column: Amount            # NOT Balance. Nothing here will second-guess you.
date_format: "%m/%d/%Y"          # exactly what the file writes; no fallback is tried
sign: as_written                 # or: negate | debit_negative | debit_positive
negative_notation: [parens]      # this bank writes debits as (50.00)
decimal_separator: "."
thousands_separator: ","
skip_rows: 0                     # rows of preamble above the header
currency: USD
```

Every field is a decision and there are no defaults that guess. `sign` states which direction of
money this bank writes as positive — HEARTH normalizes everything to **money in positive, money
out negative** and cannot work that out from the data. `negative_notation` must list what the
bank actually uses; a `(50.00)` under a mapping that did not enable `parens` is **refused**,
because the alternative is reading it as a deposit.

Check the mapping fits the header before any number is computed from it:

```sh
uv run --no-sync python -c "
from hearth.config import Settings
from hearth.finance.mapping import ColumnMapping, inspect_header
from hearth.finance.parse import read_table
m = ColumnMapping.from_yaml('$HOME/hearth-statements/mappings/acme-checking.yaml')
rows = read_table('$HOME/hearth-statements/incoming/august.csv', Settings())
print(inspect_header(rows[0], m))"
```

`missing` must be empty. A mapping written for last year's export format fails here, on the
header, rather than half way down the file with a partial total already computed.

---

## 3. Run sealed

```sh
scripts/hearth_private.sh --profile config/routing.finance.yaml --check
```

`--check` verifies the posture and exits without starting anything; drop it to serve. It
asserts, against **the profile you actually named** (not a hardcoded one):

1. the bind address is loopback,
2. the routing policy parses to zero remotes, with every class `backend: local, escalate: never`,
3. `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` are set, so weights load from disk and are never
   fetched.

`config/routing.finance.yaml` satisfies (2) structurally — `remotes: {}`, a `defaults.remote`
naming an entry that does not exist, and a zero remote token budget. There is nowhere for a
transaction to go even if something explicitly asks to escalate. It also pins a two-tier local
ladder: a small fast model for per-row classification, a larger one for the final summary.

Confirm it independently while a run is in flight — a claim in a config file is not a
measurement:

```sh
lsof -nP -p "$(pgrep -f 'hearth serve')" | grep -E 'TCP|UDP'   # expect: loopback only
```

**What this seals and what it does not.** It seals HEARTH. It does not seal *you*, or an agent
you are talking to. See the box at the top.

---

## 4. The flow: ingest → reconcile → categorize → aggregate → synthesize

```
  file ──parse──▶ transactions ──reconcile──▶ VERDICT ──▶ store ──▶ figures ──▶ prose
                                    │                       │         │
                              arithmetic only          rows kept   Python only
                                                       + provenance
```

Each arrow is a different kind of claim, and they are kept apart deliberately.

### 4.1 Ingest and reconcile

```python
from decimal import Decimal

from hearth.config import Settings
from hearth.finance.mapping import ColumnMapping
from hearth.finance.parse import data_row_count, parse_rows, read_table
from hearth.finance.store import CategorySource, FinanceStore, hash_file, mapping_fingerprint
from hearth.finance.validate import reconcile

settings = Settings()                     # reads HEARTH_FILE_ROOTS from the environment
path = "~/hearth-statements/incoming/august.csv"
mapping = ColumnMapping.from_yaml("~/hearth-statements/mappings/acme-checking.yaml")

rows = read_table(path, settings)          # through the allowlist; no reader of its own
txns = parse_rows(rows, mapping)           # any unparseable row raises WITH ITS INDEX

recon = reconcile(
    txns,
    rows_read=data_row_count(rows, mapping),          # from the FILE, not from `txns`
    control_total=Decimal("-1284.37"),                # the figure the bank printed
)
print(recon.describe())
```

Two details do the work here:

- **`rows_read` comes from the table, not from the parsed list.** Comparing a list to itself is
  not a check. This is what makes "every row was read" a real comparison.
- **A row that cannot be parsed raises rather than being skipped.** A skipped row is invisible:
  the totals still add up, the reconciliation still ties to itself, and the number is simply
  wrong by whatever that row was worth. Fix the mapping or fix the export.

**On the control total.** It is the one figure that comes from outside the parse — the closing
total the bank printed, or `control_total_from_balances(opening, closing)` from the two printed
balances. Build it from the printed string as a `Decimal`; a control total built from a float is
not the number on the statement.

> **Without a control total, the sum is reported as `UNVERIFIED` — not as passing.** Three
> states are kept apart and never collapsed: *verified* (compared and matched), *mismatch*
> (compared and wrong), and *unverified* (**nobody checked**). An unverified sum is not a
> failure and it is not a pass; it is an unchecked number, and every rendering says so. If your
> pipeline must not act on one, call `require_pass(recon, require_control_total=True)`.

Then store it:

```python
store = FinanceStore(settings=settings)         # ~/.hearth/finance/ledger.db, mode 0600
result = store.ingest(
    txns, recon,
    source_path=path,
    content_sha256=hash_file(path, settings),   # the identity of this ingest
    mapping_id="acme-checking",
    mapping_version="1",
    fingerprint=mapping_fingerprint(mapping),   # the layout itself, not a claim about it
)
print(result.reason)
```

**A failed reconciliation is still stored**, marked failed, with its rows. An absent record and
a refused file look identical a month later, and only one of them means the money is accounted
for. Failed statements are kept out of every total *and named on every figure that would have
included them* (§7) — excluding data quietly is the same class of error as double-counting it.

### 4.2 Categorize

A category is the one judgement in this pipeline, and the only step a model is permitted to
perform. Three sources, and the store records which one decided:

```python
store.assign_category(txn_id, "dining", CategorySource.rule("acquirer-prefix-v1"))
store.assign_category(txn_id, "groceries", CategorySource.model("Qwen2.5-3B-Instruct-4bit"))
store.assign_category(txn_id, "dining", CategorySource.human("me"))     # a correction
```

Assignments are **append-only**. Correcting a model's answer writes a new row and keeps the old
one, so `store.category_history(txn_id)` shows what was believed, by what, and when.

Do the deterministic rows with rules, not with a model: `ACH DEPOSIT`, `ZELLE`, `VENMO`,
`*FEE`, your own recurring merchants. Only the residue needs a model at all — and the measured
tier-1 categorizer in `examples/finance/README.md` is ~70 % accurate, which is *not good enough
to ship unreviewed*. Read the ones it labelled; every time you correct one, you are writing a
labelled training example (§8).

### 4.3 Aggregate — in Python, never in the model

```python
august = store.total(start=date(2026, 8, 1), end=date(2026, 8, 31))
spend  = store.by_category(start=..., end=...)
months = store.by_month()
top    = store.by_merchant(limit=10)
```

**No model computes a figure anywhere in this pipeline.** Not the totals, not the per-category
sums, not the shares, not "roughly how much". This is a hard rule (`CLAUDE.md` §4), for the
reason that runs through this whole document: LLM arithmetic is wrong at a rate that is low
enough to be trusted and high enough to matter, and a wrong total is indistinguishable from a
right one. All arithmetic here is `Decimal` in Python and integer arithmetic in SQL. Money is
never a float and there is no `REAL` column in the schema — binary floating point cannot
represent `0.10`, and the error compounds under aggregation until a reconciliation that should
tie out by construction is off by cents nobody can find.

Every figure the store returns is computed **twice by independent means** — SQLite sums an
integer column, Python re-adds the stored decimal strings — and is refused if the two disagree.

### 4.4 Synthesize

Only now does a model see anything, and what it sees is a *finished fact sheet*: figures already
computed, with an instruction to quote them and never adjust or recompute one. That is what the
tier-2 rung of `config/routing.finance.yaml` is for, and `examples/finance/run_finance_ladder.py`
is a working end-to-end example of the shape (on synthetic data).

The model's output is prose about numbers it did not compute. Which is exactly why §7 exists.

---

## 5. Re-running: what happens if you ingest twice

Identity is the **content**, not the filename:

| you do this | HEARTH does this |
|---|---|
| ingest the same file again | **skips it.** Nothing is written; `result.skipped` is true |
| ingest a *copy* under a different name | **skips it**, naming the statement the bytes are already in |
| the bank reissues the export; same path, new bytes | **new version.** The old statement is retained, marked superseded; only the new one feeds totals |
| a genuinely different file | a new statement |

Re-running the whole month after a crash, or dragging `august.csv` and `august-copy.csv` into
`incoming/`, does not double your spending. That is not a convenience — silently doubling a
total is the single most expensive version of the plausible-wrong-number failure, and it is
invisible in the number itself.

To see both versions of a re-issued export:

```python
store.statements(source_path=path)                # every version, oldest first
store.rows(include_superseded=True)               # including the replaced ones
```

---

## 6. Where things live, and what to back up

| path | what | sensitivity |
|---|---|---|
| `~/hearth-statements/incoming/` | the bank exports | **your financial data** |
| `~/hearth-statements/mappings/` | column layouts, per account | column names only — safe |
| `~/.hearth/finance/ledger.db` | the store: statements, rows, categories | **your financial data**, mode 0600 |
| `~/.hearth/models/` | local weights | not sensitive |

Keep `~/.hearth` and `~/hearth-statements` on an encrypted volume. The ledger is a plaintext
SQLite file: it is real data at rest, and everything above about not pasting statement contents
applies equally to it.

---

## 7. Auditing a number back to its rows

This is the part that makes the rest of it worth doing. A summary says *"dining came to
$412.68 in August"*. Check it:

```python
figures = store.by_category(start=date(2026, 8, 1), end=date(2026, 8, 31))
dining = next(f for f in figures if f.label == "dining")

print(store.explain(dining))
```

```
figure         : dining
amount         : -412.68
rows           : 19

      id  date          amount         running  source
      41  2026-08-02     -6.75           -6.75  incoming/august.csv#12 v1  SQ *BLUE BOTTLE
      47  2026-08-03    -41.20          -47.95  incoming/august.csv#18 v1  TST* SABOR LATINO
      ...
  sum of the rows above : -412.68
  figure                : -412.68
```

Every line names **the file, the row index within it, and which version of that file** — so you
can open the export and look. (`#12` is the twelfth row the reader returned, counting the
header as row 0; it is the same index a parse error would quote.) The listing re-adds the rows
as it goes and prints the running sum, so the last line is visibly the headline number rather
than being asserted to equal it.

- `store.rows_behind(figure)` returns those rows as data instead of text. It **refuses** to
  return partial evidence: an id it cannot find raises rather than being dropped, because a
  quietly shorter list of evidence is a quietly smaller total.
- `figure.excluded` lists any statement deliberately left out — a failed reconciliation whose
  rows exist but do not tie. `store.explain` prints those at the bottom. A figure that is
  missing a file says so where the figure is read.
- `store.statement(id)` shows how that file was ingested: rows read vs parsed, the control
  total (or the fact that none was supplied), the mapping fingerprint, and the timestamp.

If a number in a summary has no rows behind it, the model invented it. That is now a detectable
event rather than a thing you find out about at tax time.

---

## 8. Corrections are training data

Every time you overrule a category, the store keeps both the old assignment (with the model and
adapter that produced it) and yours:

```python
for c in store.corrections():
    print(c.description, c.previous_category, "->", c.category)
```

That is a supervised example — input, correct label, and the specific error being corrected —
accumulated as a by-product of doing your own bookkeeping rather than from an annotation
project nobody has time for. `docs/LEARNING_plan.md` §2 names the missing data flywheel as this
project's highest-leverage gap; this table is its substrate. Once there are enough of them, they
train a LoRA adapter for the `classify` rung, and `hearth eval --promote` decides on evidence
whether the new one is actually better (`CLAUDE.md` §7 — it will refuse a promotion you cannot
support, which is the point).

---

## 9. When something goes wrong

| symptom | what it means | what to do |
|---|---|---|
| `file reads are disabled: no readable directory in HEARTH_FILE_ROOTS` | deny-by-default, working as intended | export `HEARTH_FILE_ROOTS` (§1) |
| `path is outside every allowed root` | the file is not under the root, after symlinks | move it into `incoming/` |
| `mapped column(s) not present in this header` | the export format changed | re-run `hearth_peek.py`, update the mapping, ingest as a new version |
| `'(50.00)' is a parenthesized accounting negative…` | the bank writes debits in parens | add `parens` to `negative_notation` |
| `row 47: …` on parse | one row does not fit the mapping | look at line 47 of the export. Do not skip it |
| reconciliation `FAIL`, rows read ≠ rows parsed | rows were dropped | fix before trusting any total; the statement is stored as failed |
| reconciliation `FAIL`, sum mismatch | the parse disagrees with the bank | usually the wrong amount column, or a sign convention |
| `sum UNVERIFIED` | you supplied no control total | supply one. Until then no total here has been checked against anything |
| `StoreIntegrityError: … do not agree` | the ledger's two amount columns disagree | the database has been corrupted or hand-edited; stop and restore it. No figure is returned |

---

## 10. The short version

1. Statements live in `~/hearth-statements`; `HEARTH_FILE_ROOTS` points there and nowhere else.
2. Author the mapping from **headers only** (`scripts/hearth_peek.py`) — it never prints a value.
3. Run sealed: `scripts/hearth_private.sh --profile config/routing.finance.yaml --check`.
4. Always supply a control total. Without one the sum is `UNVERIFIED`, not verified.
5. Rules first, model second, human last — and the store keeps all three.
6. **Python computes every figure. The model only ever writes prose about finished numbers.**
7. Any number you read, you can walk back to its rows: `store.explain(figure)`.
8. Never paste statement contents into a cloud agent. Hand over a path instead.

Related: `docs/PRIVACY.md` (the threat model and the caller caveat) · `docs/TIERS.md` (why
there is no remote backend to escalate to) · `docs/APEX_seam.md` §5 (why HEARTH parses
structure and never infers financial semantics) · `examples/finance/README.md` (the two-tier
ladder, measured, on synthetic data) · `docs/LEARNING_plan.md` §2 (the flywheel).

---

## Appendix A. Drafting a mapping with the local model — `scripts/hearth_map_draft.py`

§2 says to author the mapping from headers alone, and that stays true: **headers are safe to
share, values are not.** But some of the mapping cannot be settled from headers at all. Whether
`03/04/2026` is March or April, whether a "Debit" column carries magnitudes or signed values,
whether `(50.00)` occurs anywhere in the file — those live in the cells. A cloud agent
structurally cannot help you with them, because helping means reading your transactions. A
**local** model may read them freely. That asymmetry is the whole reason HEARTH exists, and
this tool is it pointed at the one job that needs it:

```sh
HEARTH_FILE_ROOTS=~/hearth-statements \
  uv run --no-sync python scripts/hearth_map_draft.py ~/hearth-statements/incoming
```

It walks the directory, groups files by header signature (one mapping per format, not per
file), and writes a **draft** YAML per format into `~/hearth-statements/mappings/` — or
wherever `--out` points. A draft is not a mapping. It is a proposal you review.

### What decides what

| Decided in code, from every row | Proposed by the local model | Left to you |
|---|---|---|
| `date_format` — exhaustively; **AMBIGUOUS** if the file never disambiguates | which numeric column is the amount vs a running balance | `sign`, always |
| numeric vs text vs date vs empty per column | which text column is the description | whichever fields came back `AMBIGUOUS` / `UNRESOLVED` |
| `negative_notation` — parens and trailing minus, as they actually occur | which of two date columns dates the transaction | whether a "Balance" column really is the account balance |
| `decimal_separator` / `thousands_separator`, by trial | a name for the format (a label; it changes no number) | |
| debit/credit pair vs one signed column | | |

The model is **never asked for the date format.** That is deliberate and it is the most
important line in the tool: reconciliation cannot catch a wrong date format. Sums do not depend
on dates, so `%m/%d/%Y` where `%d/%m/%Y` was meant passes every arithmetic check you have while
silently filing transactions in the wrong months. It is the one semantic error with no
downstream gate, so it is settled by scanning **every** row — one `25/12` anywhere in the file
settles it — and when nothing in the file settles it, the draft says `AMBIGUOUS` and **refuses
to load** until you choose. Do not "just pick the usual one".

Everything the model proposes is validated against the measurements before it is written: a
column name it did not copy exactly, or one whose measured type contradicts the role, is
dropped and the field is left `UNRESOLVED`.

### What runs before a draft is written

Every complete draft is trial-parsed against the real files with `parse_rows` and reconciled.
**A draft that cannot parse its own file is not written** — you get the row and column it
failed on, never the parser's message (that message quotes the cell). Where only the date
format or the description column is still open, the draft is parsed under *every* candidate for
them; neither changes a total, which is exactly why neither is proved by doing so.

No control total is ever invented. The sum in the draft is labelled `unverified`, and it is:
supply the real one when you ingest (§4.1).

### Reading a draft

The comment block at the top of each file lists every field under **MECHANICALLY DETERMINED**,
**PROPOSED BY THE LOCAL MODEL**, or **NOT SETTLED**, with the evidence for each — then the
verification, then a numbered list of what to confirm. Work through that list, replace every
`AMBIGUOUS` and `UNRESOLVED`, delete the `notes:` line, and the file becomes an ordinary
mapping. `sign` is on the confirm list every time, including when a running balance settled it:
arithmetic can only catch a reversed sign against a control total, and none was supplied.

An existing file is **never** overwritten without `--force`. Your reviewed mapping outranks a
fresh draft, permanently.

### Privacy, and the flags

The tool reads values — locally, and that is the point — but it never prints one. Stdout gets
headers, measured types, counts and the model's structural conclusions, because stdout is what
ends up pasted into a chat window. The trial-parse total is written into the draft (local,
where a reviewer needs it) and reaches the terminal only under `--show-total`.

- `--no-model` — mechanical determination only. Model-proposed fields stay `UNRESOLVED`; the
  draft is still written, still says so, and is never quietly worse.
- `--model <id>` — a specific local model (default: the registry default). It runs at
  temperature 0 so the same file drafts the same way twice.
- If mlx or the weights are missing, this is the `--no-model` path with a note saying why.

Reads go through `read_table`, so `HEARTH_FILE_ROOTS` gates this tool exactly as it gates
everything else. It has no privileges of its own.
