# Measured status

> **This document contains no measurements, and must never contain any.**
> It explains why `hearth_status.py` exists, what each field means, and what the command
> cannot see. Every number lives in the command's output, where it is regenerated on demand
> and cannot go stale. If you catch yourself pasting a count, a date, or a "verified" claim
> into this file, that is the failure this whole document is about — put it in a probe.

```
uv run python scripts/hearth_status.py            # human-readable
uv run python scripts/hearth_status.py --json     # machine-readable
uv run python scripts/hearth_status.py --section learning egress
uv run python scripts/hearth_status.py --strict   # exit 1 on any warn/fail (CI)
```

---

## 1. Why this replaced the status docs

HEARTH's institutional memory used to live in `docs/RESULTS.md`, `docs/cmux/HANDOFF.md`
and `docs/cmux/TODO.md`. Those are hand-written, and hand-written status has one structural
property that no amount of discipline fixes: **it records what somebody believed at the
moment they typed it.** Everything after that moment is drift, and the drift is invisible —
a claim marked "verified" stays green for as long as nobody re-reads it.

That is not a documentation problem. It is a specific, recurring engineering bug, and this
project hit it six separate times in a single session:

| # | The gate that was green | What was actually true | The configuration it trusted |
|---|---|---|---|
| 1 | cmux reported the session **SEALED** | panes were egressing | the seal flag, not the traffic |
| 2 | HEARTH's no-egress mode **verified** | the calling agent leaked | the router's policy, not the caller |
| 3 | APEX's privacy gates **green** | `OLLAMA_HOST` could point anywhere | the gate's own config, not the endpoint |
| 4 | gateway returned `finish_reason: "stop"` | the output was truncated | the default value, not the token count |
| 5 | eval judge **passed** a candidate | it was merely longer | a proxy score, not the objective |
| 6 | operator set `HEARTH_MODEL`, reported a result | the real var is `HEARTH_DEFAULT_MODEL`; nothing read it | the variable being set, not it being read |

Every row is the same shape:

> **A gate must assert on the OUTCOME, never on a CONFIGURATION that implies the outcome.**

A status doc is row 7. "The golden set is big enough" is a belief; *counting the lines and
computing the smallest p-value the test could emit* is a measurement. "Sealed mode works"
is a belief; *loading the routing profile through the router's own loader and reading off
whether any remote resolves* is a measurement. `hearth.status` only does the second kind,
and where it cannot, it says `unverified` rather than assert.

Three rules follow, and they are the ones to preserve:

1. **Measure the outcome, not the setting.** A model directory existing is a setting; a
   resolvable weight file the provider's loader will actually find is the outcome.
2. **`unverified` is a first-class answer.** An honest gap beats a confident guess. The
   summary line counts them on purpose.
3. **Say what you did not measure.** Every section prints its `limits`. Silence gets read
   as a pass, so silence is not allowed.

---

## 2. What it reports, and what each field means

### Models — weights actually on disk

HEARTH has **two** cache locations and they are not the same one:

* `~/.cache/huggingface/hub` — the Hugging Face default, and what `mlx_lm.load()` reads;
* `~/.hearth/models` — where `hearth models pull` writes (`cli.py` passes
  `cache_dir=settings.models_dir` to `snapshot_download`).

So a model can be **fully downloaded and still invisible to the provider**, unless
`HF_HUB_CACHE` points at `~/.hearth/models`. This is the section's headline check: a pulled
model that the loader cannot find is reported as `pulled but INVISIBLE to the provider`,
not as present, because "downloaded" is the configuration and "loadable" is the outcome.

| Field | Meaning |
|---|---|
| `hub_cache` | the directory `huggingface_hub` will actually use, and how it resolved (`HF_HUB_CACHE` > `HF_HOME/hub` > default) |
| `hearth_models_dir` | where `hearth models pull` writes, and whether it *is* the hub cache |
| *one per registry entry* | `loadable` / `pulled but INVISIBLE` / `registered, NO weights on disk`, with measured size |
| *one per stray repo* | weights on disk naming no entry in `config/models.yaml` — nothing serves them, nothing collects them |
| `partial_downloads` | repos holding `*.incomplete` blobs (an interrupted pull) |

Presence means a weight file that *resolves* — a dangling snapshot symlink counts as absent.

### Egress posture — what the routing policy structurally permits

Per routing profile in `config/`, loaded through `hearth.router.policy.load_policy` (the
same call the running router makes). A profile is `NO EGRESS` only when the **resolved**
policy has zero remotes, resolves no default remote, and every class is `backend: local`
with `escalate: never`.

The subtle part: `load_policy` never raises — an invalid file silently falls back to safe
built-in defaults, and those defaults are *themselves* no-egress. So a broken profile would
resolve to a clean-looking policy. The probe therefore diffs what the file declares against
what the router resolved and reports any disagreement as **drift**, so a file the router
never honoured cannot read as a green line. (That is bug #1 in miniature, and it is
regression-tested.)

`no_egress_profile_available` fails when *no* profile in `config/` resolves to a zero-remote
policy — sealed mode would have nothing to select. `download_egress` reports
`HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE`, because the router is not the only thing on this
machine that can reach the network.

### Learning state — and the minimum detectable effect

The important field in the whole report. The promotion gate is a one-sided **exact McNemar**
test on discordant pairs, so on `n` paired golden items the smallest p-value it can *ever*
emit is `0.5**n` — the clean-sweep case where the candidate is right on every item and the
incumbent wrong on every one. Therefore, at `alpha = 0.05`:

| Golden set size | Smallest achievable p | Verdict |
|---|---|---|
| `n < 5` | `> 0.05` | **fail** — no promotion can *ever* clear the bar. The set cannot gate anything, at any effect size, with any amount of GPU time. |
| `n = 5` | `0.03125` | **warn** — clears only on a perfect 5-for-5 sweep with nothing lost. |
| `n <= 10` | | **warn** — needs ≥ 5 of `n` items to flip to the candidate with none flipping back: half the set or more. |
| `n > 10` | | **ok** — the bar is reachable without a degenerate result. |

**5 is the threshold**, because `0.5**5 = 0.03125 ≤ 0.05 < 0.0625 = 0.5**4`. A golden set
below it turns every training run into an unfalsifiable claim, which is the most expensive
thing in this report to discover late. The maths mirrors
`hearth.training.stats.smallest_achievable_p` / `min_n_for_alpha`, and a test asserts the
two agree, so the report can never drift from the gate it describes.

The section also reports training-corpus row counts, malformed JSONL lines, adapter counts
by lifecycle status, whether each adapter's weights still exist on disk, and — a second
outcome-vs-claim check — whether a **promoted** adapter carries a significance proof. An
adapter promoted on a bare score comparison (`candidate_score > incumbent_score`, no
`p_value`) is flagged: that comparison cannot distinguish a real improvement from noise on
a small golden set. It is bug #5 wearing a different hat.

### Test suite

Declared test functions, counted by reading source, plus whatever pytest's on-disk cache
records about *some* earlier run, with that cache's timestamp. Whether the suite passes
**now** is reported as `unverified`, because running it would mutate caches and race the
other agents editing this repo — and because a remembered pass is not a pass.

### Environment

Chip, unified memory, library versions, and the load-bearing number:
`max_recommended_working_set_size` from `mx.device_info()` — the driver's own ceiling on
resident GPU memory, which is **materially lower than the RAM the machine advertises**.
Size models against that number; sizing against advertised RAM is the same class of mistake
as trusting a config. Both figures print in decimal GB *and* binary GiB, because a report
that prints "GB" while dividing by 1024 invents a discrepancy of its own.

`version_ceilings` compares installed versions against the `<` upper bounds declared in
`pyproject.toml` — those pins exist because a newer release *breaks* something (mlx-lm
against transformers 5, coremltools against torch 2.8), so exceeding one is a live
incompatibility.

`hearth_env` lists every `HEARTH_*` variable set in the environment and flags any that **no
code reads**. This is bug #6 caught directly: a variable that is set but ignored means every
result attributed to it describes a different configuration than the operator believes.

### Staleness

For each memory doc, the last commit that touched it and how many commits have landed on
HEAD since. A doc's own text cannot tell you whether it is still true; commit distance is
evidence a human can act on, in place of a "last updated" line the last editor forgot to
change. A doc that exists only in the working tree is `unverified` — its age and its review
status are both unmeasurable.

The `STALE_COMMITS` threshold is the **only judgement call in the entire report**, and it is
named as one. It says "go re-read this", never "this is wrong".

---

## 3. What it deliberately CANNOT verify

This is the important half of the document. Every section prints its own `limits`; these are
the ones that matter most, and they are what a human still has to check.

* **Machine-level containment.** The egress section reports what the *router* structurally
  permits and nothing else. It does not inspect a firewall, a socket, a process, or a DNS
  query. A `NO EGRESS` profile does not stop `providers/remote.py` being called directly, a
  non-loopback bind, a model download, telemetry from another process in the same terminal,
  or **the calling agent leaking what it was shown** (bug #2 — the one this tool is
  structurally incapable of catching, because it runs inside that agent's blast radius).
* **Whether the running daemon has the posture reported.** The command reads *its own*
  environment, not a live server's. A daemon started an hour ago under a different profile
  will not show up here. Check the server's own startup log.
* **That a model loads.** Presence is a `stat()` of a resolvable weight file. Complete,
  uncorrupted, and compatible with the installed mlx-lm are all unverified — only a real
  load proves those.
* **That the tests pass.** Never run by default. `uv run pytest -q` is the only answer.
* **Golden-set quality.** Row counts do not measure label correctness, duplication, leakage
  between corpus and golden set, or whether the golden set still matches the task being
  trained. The minimum detectable effect is an *upper bound* on what the test could show at
  best; real discordance is far below the clean-sweep optimum, so the practical requirement
  is always larger than the number reported.
* **Adapter quality.** That weights exist and what proof was recorded, not that the adapter
  is any good.
* **Whether any doc's claims are true.** Nothing here reads the documents. Commit distance
  says where to look first, and stops there.

---

## 4. Read-only, and why that is enforced

`hearth.status` writes nothing, creates nothing, opens no socket, and mutates nothing. That
is what makes it safe to run *first* — on a machine you already suspect is broken, and
inside a sealed session where anything leaving the box is the incident.

`tests/test_status_readonly.py` enforces this twice, and the duplication is deliberate:

* **the configuration** — an AST scan of the package's own source for network imports and
  filesystem-write calls; and
* **the outcome** — a real `collect_status()` run against a fixture tree, fingerprinted
  (path, size, mtime) before and after, asserting nothing changed.

The second is the proof. A source scan is exactly the kind of gate that passes while the
thing it guards is false, so it is a tripwire, not evidence. (It already produced one false
positive — `platform.system()`, which shares a name with `os.system` — which is the point.)

The single exception is `hearth/status/gitmeta.py`, which shells out to `git`, because
history exists nowhere else. It is the only module allowed to import `subprocess`, it may
only call `subprocess.run` with a literal `git` argv and no shell, and every subcommand must
be in a read-only allowlist. All three constraints are asserted by tests.

The sibling `hearth/handoff/` package holds a near-identical no-egress invariant. The two
are kept **independent on purpose** — a test forbids importing one from the other — because
two packages that each prove they cannot reach the network prove nothing if one delegates to
the other.

---

## 5. How to extend it

Adding a measurement:

1. Write a `probe_*(...) -> Section` in `src/hearth/status/probes.py`. Take `root`, `home`
   and `environ` as keyword arguments so tests can point it at a fixture tree.
2. **Never raise.** Anything you cannot measure becomes a `LEVEL_UNVERIFIED` fact with a
   detail saying what a human must check instead.
3. Put the structured values in `Fact.data`, not only in the prose `value` — `--json`
   consumers must never have to parse English.
4. Fill in `Section.limits`. If you cannot name something the probe fails to measure, you
   have not thought about it hard enough yet.
5. Register it in `collect_status()` in `src/hearth/status/__init__.py`.
6. Add tests that build the exact on-disk situation and assert the probe *measured* it — not
   that it ran.

Before you write it, check the measurement against the rule at the top:

> Am I asserting on an outcome, or on a configuration that implies the outcome?

If a field would report "the config says X", it belongs nowhere in this package. Report what
X actually produced, or report `unverified`.

And the standing rule this file exists to defend: **do not answer "what is the state of the
project?" by writing it down.** Write a probe. A written answer is correct once; a probe is
correct every time it is run.
