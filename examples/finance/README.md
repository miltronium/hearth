# A two-tier local model ladder, end to end

This example proves that HEARTH can express **different local models for different task
classes** and actually run a real workload across them, entirely on-device, with no egress
path in the configuration.

```
stage 1  categorize 46 transactions  ->  tier 1  Qwen2.5-3B-Instruct-4bit    (~2 GB)
stage 2  compute every total          ->  Python  (never a model)
stage 3  narrate the month            ->  tier 2  Qwen2.5-14B-Instruct-4bit  (~9 GB)
```

The script never names a model. It sets an `intent` (`classify` / `summarize`) and the
router resolves the model from `config/routing.finance.yaml`. Which model actually served
is printed at every stage, so the ladder is visible rather than asserted.

## The data is synthetic

`statements.csv` is invented for this example. Every merchant string, amount, and date was
written by hand to *imitate* the shapes real processors emit — `SQ *` and `TST*` acquirer
prefixes, store numbers, ACH descriptors, autopay memos — so the tier-1 model has realistic
noise to work through. No real account, statement, or export was read to build it, and the
harness reads nothing but this one file.

Each row carries an `expected_category` answer key and a `difficulty` flag, so the run
reports a real error rate instead of a vibe.

## Arithmetic is never the model's job

This is the hard rule of the example. LLM arithmetic is unreliable and these are financial
figures, so **stage 2 is pure Python**: totals, per-category sums, counts, percentage shares
and the largest debit are all computed in `aggregate()`. Tier 2 receives the finished fact
sheet and is told, in its system prompt, to quote the given figures and never compute or
adjust one. `tests/test_finance_example.py` pins that the aggregates are exact.

The model's only judgement is *which bucket a transaction belongs in* — a language task.
Everything downstream of that is arithmetic, and arithmetic stays in Python.

## Running it

```bash
# from the repo root
HEARTH_BACKEND=mlx HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
HEARTH_ROUTING_YAML=config/routing.finance.yaml \
uv run python examples/finance/run_finance_ladder.py
```

Useful flags: `--limit N` (first N rows), `--max-tokens` (tier-2 budget),
`--ram-ceiling-gb`, and `--dry-run` (echo provider — the same routing decisions with no
weights loaded, good for checking the plumbing in under a second).

Weights must already be on disk under `~/.hearth/models`:

```bash
hearth models pull mlx-community/Qwen2.5-3B-Instruct-4bit
hearth models pull mlx-community/Qwen2.5-14B-Instruct-4bit
```

The harness sets `HF_HUB_CACHE=~/.hearth/models` for you before importing anything, because
`hearth models pull` stores weights there rather than in the default `~/.cache/huggingface`.

### Running it sealed

`scripts/hearth_private.sh` is the sealed-mode entry point, but note what it actually does:
it **hardcodes `config/routing.private.yaml`** and takes no profile argument, so pointing
`HEARTH_ROUTING_YAML` at the finance profile before calling it has no effect — it verifies
and serves the private profile.

So use it for what it is good for — confirming the machine-level posture — and let the
harness verify the profile it is really using:

```bash
scripts/hearth_private.sh --check      # loopback + offline + private profile posture
```

`run_finance_ladder.py` re-runs that script's three checks against whatever profile it was
pointed at, *before* any weights load, and exits 2 if any fails:

1. zero remotes are defined,
2. no default remote resolves,
3. every class is `backend: local`, `escalate: never`.

`config/routing.finance.yaml` satisfies all three by construction: `remotes: {}`, a
`defaults.remote` naming an entry that does not exist, and a zero remote token budget. There
is nowhere for a transaction to go even if a request explicitly asks to escalate.

The caller caveat from `docs/PRIVACY.md` still applies: this seals HEARTH, not the agent
that hands HEARTH a file.

## Measured result

Run on 2026-09-02, Apple Silicon, `HEARTH_BACKEND=mlx`, both models resident, 46 synthetic
transactions.

| stage | model | work | time |
|---|---|---|---|
| 1 categorize | Qwen2.5-3B-Instruct-4bit | 46 calls | **14.2 s** (309 ms/txn) |
| 2 aggregate | *Python* | 46 rows | **0.04 ms** |
| 3 narrate | Qwen2.5-14B-Instruct-4bit | 1 call, 136 tokens | **~11 s warm / ~26 s cold** |

Isolated model benchmarks from the same session: 3B cold load 1.5 s; 14B ~15 s genuinely
cold, 1.3 s once the weights are in the page cache, generating at **12.6 tok/s**. Stage 3's
wall time varied between 21.6 s and 34.0 s across runs depending on cache state and memory
pressure with the 3B still resident. End to end: **38–48 s**.

Backend mix `{'local': 47}`, escalations `0`.

### Categorization accuracy — the honest number

| slice | correct | rate |
|---|---|---|
| overall | 32/46 | **69.6 %** |
| easy rows | 23/31 | 74.2 % |
| hard rows | 9/15 | **60.0 %** |

**This is not good enough to ship.** A ~70 % categorizer means roughly one dollar in seven
lands in the wrong bucket, and the narrative in stage 3 faithfully describes those wrong
buckets — the arithmetic is exact but it is exact over bad labels. The example is honest
about this on purpose; it demonstrates the *ladder*, not a finished categorizer.

The dominant failure is not ambiguity, it is a systematic blind spot: **the 3B calls small
`SQ *` card-present merchants `groceries` almost regardless of what they sell.** Dining
recall was 1/6.

```
[easy] SQ *BLUE RIDGE COFFEE ASHEVILLE NC   got groceries      want dining
[easy] SQ *THE DAILY GRIND                  got groceries      want dining
[easy] SQ *SWEETGREEN 1701 MARKET           got groceries      want dining
[hard] TST* SABOR LATINO - 15TH             got groceries      want dining
[easy] REI #018 BERKELEY CA                 got groceries      want shopping
[easy] VENMO PAYMENT ... J RIVERA           got groceries      want transfer
[hard] 76 - CIRCLE K #2298                  got groceries      want transport
```

Other misses: `CHEVRON 0093847 SAN JOSE CA` and `APPLE.COM/BILL` produced no usable label at
all (scored `uncategorized`); `AT&T *WIRELESS PAYMENT` went to `fees`; `ANNUAL MEMBERSHIP
FEE` went to `subscriptions`.

### The hard cases

15 rows are flagged `hard`. Two kinds:

*Solvable with better prompting or a tuned adapter* — the answer is genuinely recoverable
from the string. The 3B got `AMZN Mktp`, `SQ *FARMERS MKT STALL 12`, `PAYPAL *STEAMGAMES`,
`SQ *ROVING BARBER CO`, `PHARMACA`, `SAFEWAY FUEL` (correctly *transport*, not groceries),
`CVS/PHARMACY` and the Stripe ACH credit right. It missed `UBER *EATS` (called it
*transport* — the brand dominates the product), `TST* SABOR LATINO`, `76 - CIRCLE K`, and
`APPLE.COM/BILL`.

*Genuinely underdetermined* — a human would need the receipt. `COSTCO WHSE` and `TARGET`
are big-box merchants that sell across several buckets; the answer key picks one, and
counting these as errors is arguably unfair to the model. Roughly 2 of the 14 misses are of
this kind, so a charitable read is ~74 % rather than ~70 %. Still not shippable.

### What would fix it

Not a bigger model on stage 1 — that trades away the whole point of the ladder. The right
moves are a LoRA adapter for `classify` trained on the tier-1 base (HEARTH already has the
training path), a few-shot prompt covering the `SQ */TST*` acquirer-prefix pattern, and a
deterministic pre-pass for the rows that are pure string matching anyway (`ACH DEPOSIT`,
`ZELLE`, `VENMO`, `*FEE`). Only the residue needs a model at all.

## A bug this example found

The router auto-attaches the promoted LoRA adapter for a task class. Once a class pins its
own `local_model`, that adapter is no longer guaranteed to have been trained on the model
the class now routes to. On this machine a promoted `classify` adapter trained on
`Qwen2.5-Coder-7B` (hidden size 3584) was being layered onto the 3B rung (hidden size 2048),
failing with a matmul shape error on **every single transaction** and silently falling back
to base weights via the router's degrade-and-retry path — 92 generation attempts for 46 rows,
and no visible symptom beyond a log line. Removing the doomed attempt cut stage 1 from
364 ms to 309 ms per transaction (~15 %).

`Router._resolve_adapter` now skips a promoted adapter whose `base_model` differs from the
model actually being served. An explicitly requested adapter id still wins, since that is a
deliberate operator choice.

## Files

| file | what it is |
|---|---|
| `statements.csv` | 46 synthetic transactions + answer key + difficulty flags |
| `run_finance_ladder.py` | the harness: seal check, 3 stages, timings, accuracy report |
| `../../config/routing.finance.yaml` | the no-egress two-tier profile |
| `../../tests/test_finance_example.py` | hermetic tests (no model): exact aggregates, seal |
