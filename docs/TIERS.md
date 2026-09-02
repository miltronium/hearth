# HEARTH — The four-tier ladder, and why the top two are not backends

**Status:** Design. Tiers 1-2 are being implemented; tiers 3-4 are **deferred pending the
operator's decision** (§9). The handoff scaffolding in `src/hearth/handoff/` is implemented and
tested; nothing in it can reach the network, by construction.
**Date:** 2026-09-02.
**Scope:** the tier ladder, why tiers 3-4 are deliberately not `ModelProvider` backends, the
handoff-envelope mechanism, the return path and its provenance rules, and the questions that
must be answered before tier 3 or 4 is implemented at all.
**Constraint:** every proposal here must leave `scripts/hearth_private.sh --check` a *true*
statement about the running process, and must leave `lsof` a sufficient proof of no egress
(`docs/PRIVACY.md`). Nothing proposed here adds a byte of network capability to HEARTH.

---

## 0. TL;DR

There are four tiers. Two of them run on this machine and two do not, and that is not a
gradient — it is a boundary.

**HEARTH must never reach out.** When a task exceeds local capability, HEARTH writes a
**handoff envelope**: a local file describing the task, why the local tiers were insufficient,
and exactly what would cross. A human reads it, decides, and carries it across using tools that
live *outside* HEARTH. Answers come back as `provenance: external` records that are marked
forever and cannot silently enter a training corpus.

The reason is not squeamishness. Today HEARTH's no-egress property is **structural** — the
router has nowhere to send a task because `config/routing.private.yaml` defines zero remotes,
and `providers/remote.py` is the only code path that could — so a non-expert can verify it in
ten seconds with `lsof`. Adding a Private Cloud Compute backend would keep the *verifier*
passing (§2.1) while making the *guarantee* false. A guarantee that fails silently while its
check keeps saying OK is worse than no guarantee at all.

---

## 1. The ladder

| Tier | What it is | What it is genuinely for | Where it runs |
| --- | --- | --- | --- |
| **1** | Local small model (~3B) + task LoRAs | **Bulk structured work.** Extraction, classification into *your* taxonomy, ranking, tagging — thousands of items, overnight, at zero marginal cost. Its real advantage is that it retrains in ~30 s, so the correction loop can close daily (`docs/LEARNING_plan.md` §5.3). | On-device, sealed |
| **2** | Local large model (9-14B, 4-bit) | **Synthesis.** Multi-document summarisation, drafting, code work, the judge role — anything where one careful pass beats a thousand cheap ones. Slower to retrain, so it stays a generalist. | On-device, sealed |
| **3** | Apple Private Cloud Compute | **Hard reasoning on a long context**, occasionally. 32K context and explicit reasoning levels, with a per-user daily quota that makes it a scalpel, never a workhorse. **Private but not local.** | Apple's silicon |
| **4** | Frontier model | **Non-sensitive work only.** Novel reasoning, open synthesis, anything genuinely knowledge-bound. This is the tier HEARTH was built to *use less of* (ADR-001). | Someone else's silicon |

Two things about this table matter more than the capability ordering.

**The break is between 2 and 3, not between "small" and "large".** Tiers 1-2 differ in size;
tiers 2 and 3 differ in *whether your bytes left the building*. Everything in this document is
about that one edge.

**Tier 3 is not "tier 2 but better".** It is bounded by a daily per-user quota (§3). You cannot
run a nightly categorisation job on it, you cannot use it as the escalation target for a
`draft`/`code`/`chat` class under `on_low_confidence`, and you cannot rely on it under load. It
is for the handful of problems a week where the local 9-14B visibly loses the thread.

### 1.1 The decision rule that actually matters

Before reaching for tier 3 or 4, classify the *failure*, not the task. `docs/LEARNING_plan.md`
§5.4 measured this and the result generalises:

| The local model failed because… | Right fix | Escalating helps? |
| --- | --- | --- |
| It doesn't know **your** convention (queue codes, chart of accounts, field names) | Fine-tune tier 1 (measured 0.20 → 1.00) | **No.** A frontier model scores at chance too — the convention is in no pretraining corpus. |
| The output shape was wrong (invalid JSON, not one of N labels) | Constrained decoding at tier 1 | **No.** Costs nothing, cannot regress. |
| It is **guessing** — high dispersion across samples, contradicts itself, loses a long thread | Tier 2, then tier 3 | **Yes.** This is the only case worth an envelope. |
| It lacks knowledge that exists in the world but not on this disk | Tier 3/4, or RAG if the knowledge is a document you have | **Sometimes** — RAG first, it is cheaper and stays local. |

Most perceived "the local model is too weak" moments are the first two rows. Handing those to
the frontier buys nothing and spends the boundary crossing for free. **An envelope whose
`local_attempt.reason` is a convention failure should be rejected at review.**

---

## 2. Why tiers 3-4 are deliberately not backends

The tempting design is one class and one config line: a `PCCProvider` behind the
`ModelProvider` interface (ADR-004), a `remotes:` entry, done. Four arguments against, in
descending order of how much they should worry you.

### 2.1 It would silently falsify the guarantee *while the checker keeps passing*

`scripts/hearth_private.sh` verifies the routing **data**:

```
if policy.remotes:                  -> fail
if policy.remote_for() is not None: -> fail
if any class is not local/never:    -> fail
```

That check is sound today for exactly one reason: `providers/remote.py` is the only code in
the repo that can open an outbound connection with your content in it, and it is reachable
only through `policy.remotes`. The check is a proxy for a capability, and the proxy currently
holds.

A PCC provider breaks the proxy. PCC is *not* a "remote" in the policy's sense — it has no
base URL, no API key, no `remotes:` entry to write (§3: it is OS-integrated, there is nothing
to configure). It would be registered as a **backend**, exactly like `mlx`. So a class routed
to it reads `backend: pcc, escalate: never`, `policy.remotes` stays `{}`, `remote_for()` stays
`None`, and:

```
==> Posture verified: no router egress path exists.
```

…would be printed by a process that is at that moment holding a TLS connection to Apple. The
sealed check would keep saying OK about a box that is no longer sealed. **That is the single
strongest argument in this document.** An operator who trusts `--check` — which is precisely
what `docs/PRIVACY.md` tells them to do — would be trusting a statement that has quietly
become false.

Yes, the checker could be taught about backends too. Which leads to:

### 2.2 It trades a structural property for a configuration property

Today: *there is no code path*. Tomorrow, with a PCC backend: *there is a code path, and the
config currently doesn't take it*.

The first survives a bad merge, a typo in YAML, an env var set in the wrong shell, a future
refactor, and a concurrent agent editing `config/`. The second survives none of those. HEARTH
is a repo where **three agents edit concurrently** and the routing policy is deliberately data,
not code (ADR-005) — i.e. the exact conditions under which "correctly configured" degrades. The
whole value of "no-egress by construction" is that it does not depend on anybody, including
future you, getting a config file right.

### 2.3 It destroys `lsof` as a proof

Right now, *any* established outbound TCP connection from the `hearth serve` process is a bug.
That is binary, machine-checkable, needs no judgement, and can be run by someone who has never
read the code:

```sh
lsof -nP -iTCP -a -p "$(pgrep -f 'hearth serve')" -sTCP:ESTABLISHED
```

With a tier-3 backend the question becomes "is this connection to Apple, and was it for a task
the policy was supposed to allow, and did the content that went with it match?" That is not
answerable with `lsof`. It is answerable only by correlating application logs — and the logs
are written by the same process you are trying to audit. You would have replaced a proof with
an attestation.

### 2.4 "Private" and "local" are different guarantees, and only the operator can map them

Apple's PCC guarantee is about what happens to data **after** Apple receives it: not stored,
not accessible to Apple staff, independently verifiable images (§3). It is a strong guarantee
and it is not the same guarantee HEARTH sells, which is: **the bytes never left this machine.**

For Apple-confidential material the operative question is frequently the second, and it is
often a *policy or contractual* question rather than a technical one — "may this material be
transmitted off this device at all" has an answer that lives in an agreement, not in a threat
model. HEARTH cannot make that call and should not encode a guess about it. What it can do is
make the crossing impossible to perform by accident. That is what the envelope does.

### 2.5 The honest counter-arguments

This design has real costs and it would be dishonest to bury them.

- **A human is a rate limiter.** No autonomous agent loop can use tier 3 mid-task. If the
  operator later needs "escalate to PCC inside an agent loop", the envelope model cannot
  provide it and this decision has to be reopened, not worked around. That is the ADR's
  *Revisit-if*.
- **Manual carriage is error-prone in its own way.** Copy-paste picks up the wrong buffer;
  the wrong file gets carried; an answer gets attributed to the wrong envelope. The mechanism
  mitigates this with content hashes and an explicit linkage (§4), but a mitigated error is
  still an error the automated path would not have made.
- **Latency and ceremony.** Build, redact, review, release, carry, ingest. For a task worth
  thirty seconds of frontier time this is absurd overhead — which is a feature only if you
  agree that friction at a trust boundary is a feature. Under time pressure it will be
  tempting to skip HEARTH entirely and paste into a browser, which is *worse* because it
  leaves no record at all. §7 addresses this directly: the mechanism must be fast enough to
  beat the shortcut, or it will lose to the shortcut.
- **A middle path exists and is not obviously wrong.** A separate binary — call it a ferry —
  that reads `released/` and performs the crossing would keep the `hearth` process itself
  provably silent while removing most of the manual work. It weakens "a human act" to "a
  different program's act on the same machine": the *process* stays auditable, the *machine*
  no longer is. Under a sealed cmux workspace (`config/cmux/tiers.yaml`, most-restrictive-wins)
  the ferry would also have to run in the open tier, which is arguably the right place for it.
  This is open question **Q3**, not a settled no.

---

## 3. Apple Private Cloud Compute — what is actually true

Verified from Apple's WWDC26 sessions during this session. Verification status is marked,
because a design that leans on unverified capability is how a seal gets broken by surprise.

| Fact | Detail | Status |
| --- | --- | --- |
| **One-line adoption** | `LanguageModelSession(model: PrivateCloudComputeLanguageModel())` — a drop-in swap for the on-device model in the same API. | ✅ verified |
| **Context** | 32K tokens, vs. 4-8K on-device. | ✅ verified |
| **Reasoning levels** | `light` / `moderate` / `deep` — explicit deliberation control per request. | ✅ verified |
| **No credentials** | No API keys, no accounts, no billing setup. Integrated via the OS and iCloud identity. | ✅ verified |
| **Cost** | Free for developers under 2M first-time downloads. | ✅ verified |
| **Quota** | A **daily, per-user** cap. | ✅ verified |
| **Data handling** | User data is never stored; the guarantee is independently verifiable. | ✅ verified |
| **Requirements** | Internet connectivity and an Apple Intelligence-capable device. | ✅ verified |
| **Language** | Swift-first. `apple/python-apple-fm-sdk` provides Python bindings **to the on-device model**. | ✅ verified |
| **Python → PCC** | Whether PCC is reachable at all from those Python bindings. | ❌ **UNVERIFIED** |

Four consequences follow directly and shape everything below.

1. **The daily per-user quota disqualifies tier 3 from bulk work**, permanently and by design.
   It cannot be the escalation target for a task class; it can only ever be a per-item,
   operator-initiated act. This alone means a `PCCProvider` would spend a lot of machinery
   (budgets, retries, backpressure, the `remote_budget_tokens_per_day` plumbing) on something
   used a handful of times a week.
2. **"No API keys, OS-integrated" is worse for us, not better.** There is no credential to
   withhold, no endpoint to leave unconfigured, no `remotes:` entry to omit. The usual way to
   keep a remote path dormant — don't configure it — **does not exist for PCC**. Once the code
   is present, it is live. That is precisely why the code must not be present.
3. **The reasoning levels are a real reason to want it.** `deep` on a 32K context is a genuine
   capability step over a local 9-14B for exactly the "it is guessing / lost the thread" failure
   in §1.1. This tier is worth wanting; that is why it deserves a proper mechanism rather than a
   quiet exception.
4. **Swift-first means the crossing tool is a separate program anyway.** A tiny Swift CLI that
   takes an envelope's payload and prints the answer is the natural carrier — and it is
   naturally *outside* HEARTH, because HEARTH is Python and the binding for PCC-from-Python is
   unverified. The architecture and the ecosystem happen to agree here. Do not treat the Python
   binding as a fallback plan until someone has actually confirmed it reaches PCC (**Q6**).

---

## 4. The handoff envelope

### 4.1 The shape of it

```
tier 1 / tier 2 attempt
        │  insufficient — and the reason is row 3 or 4 of §1.1
        ▼
  build_envelope()          local file, drafts/            ← nothing has crossed
        │
        ▼
  redact_envelope()         mask obvious secrets, count them
        │
        ▼
  render_review()           the ENTIRE payload, printed, untruncated
        │
        ▼
  approve()                 signed, and BOUND to the content hash
        │
        ▼
  store.release()           writes released/<id>.json, returns a path
        │
        ╎  ← the boundary. A human carries the file. HEARTH has no code for this step.
        ▼
  tier 3 (Swift/PCC) or tier 4 (frontier), outside HEARTH
        │
        ╎  ← the human carries an answer back
        ▼
  ingest_answer()           inbox/<id>-a.json, provenance: external, training_eligible: FALSE
        │
        ▼
  promote_for_training()    named approver + written justification, or it never happens
```

Everything above the boundary line is `src/hearth/handoff/`. Everything below it is the
operator's business, deliberately. **There is no arrow HEARTH can draw across that line.**

### 4.2 What an envelope records

| Field | Why it exists |
| --- | --- |
| `task_class`, `prompt`, `inputs` | The payload. Exactly these bytes cross, and nothing else does. |
| `local_attempt` — `tier`, `model`, `result`, `confidence`, `reason` | **Why local was insufficient.** Required. `reason` is a sentence a reviewer reads first; an envelope with no failure story is a request to leak for convenience. The local `result` is shown to the reviewer but is *not* part of the payload — it stays here. |
| `destination_tier` | 3 or 4. Tiers 1-2 need no envelope; passing one is an error, not a no-op. |
| `sensitivity` | `public` / `internal` / `confidential`. **No default.** Construction fails without it, so labeling is a decision someone made rather than a field someone forgot. |
| `content_hash` | `sha256` over the payload only, recomputed on read. Ties an answer to the request that produced it, and binds a review to the content it actually saw. |
| `created_at` | Caller-supplied when determinism matters; nothing reads the clock behind your back (same discipline as `training/dataset.py`). |
| `provenance` | `local`. An envelope is always locally authored; `ingest.py` owns the `external` half of the vocabulary. |
| `review` | Who approved, when, the payload hash they saw, and per-rule redaction counts. |

### 4.3 The gates, and why each fails closed

- **Tier 4 accepts `public` only.** Not "internal, if you're sure". Frontier is the tier with
  the weakest privacy story and it is the one the whole project exists to use less of.
- **Tier 3 refuses `confidential`** while `PCC_ACCEPTS_CONFIDENTIAL = False`. This is **Q1**
  encoded in the code rather than in a comment. When the operator decides, the decision moves
  one boolean and gets recorded in an ADR — and if the answer is "no", nothing needs changing.
- **Release requires a *current* approval.** `is_approved` recomputes the content hash and
  compares it to the hash the reviewer signed. Editing the prompt after approval does not
  inherit the approval; it invalidates it. This closes the obvious time-of-check/time-of-use
  hole in "a human looked at it once".
- **Reading a stored envelope re-verifies its hash.** A file edited in place raises rather than
  loading, so hand-editing JSON to slip content past a recorded review does not work quietly.
- **`release()` writes a file. That is all it does.** The method's docstring says so and the
  package's import graph proves it: stdlib only, no HTTP, no socket, no subprocess (a shell is
  an egress vector), and no other HEARTH module — so the invariant is checkable by reading four
  files rather than by chasing an import tree. `tests/test_handoff_no_network.py` enforces it
  against the AST on every run, including for files that do not exist yet.

---

## 5. Redaction is an aid to review, not the control

`redact_envelope()` masks private-key blocks, AWS-shaped keys, `key: value` secrets, long hex
tokens, email addresses, and the operator's home path. It reports **counts by rule name and
never the matched text**, so a redaction report is safe to keep after the secret is gone.

Be clear about what that buys. A regex recognises *shapes*. It cannot recognise a project
codename, an unreleased product, an org chart, or a sentence that is confidential because of
what it implies rather than what it contains — which is most of what makes Apple-confidential
material confidential. **A clean report means "nothing obvious was found", never "safe to
send", and `render_review()` prints exactly that sentence** so nobody can misread a green tick.

The actual control is the review sheet: the whole payload, untruncated, with the byte count,
the hash, and the failure story, in one screen. `render_review()` must never elide — the
mechanism's entire claim is "a reviewer who read this has seen everything that crosses", and
one truncation makes that false.

The weak link is unavoidable and worth naming: the reviewer is the same person, under the same
time pressure, who wanted the answer. Two-person review is not available to a solo operator.
What the design can do instead is make the mistake *reconstructible* — hash-bound approval,
who and when, redaction counts, the envelope kept in `released/` — so a bad crossing can be
found and bounded afterwards rather than being invisible.

---

## 6. The return path — the part that is easy to get wrong

An answer from tier 3 or 4 is not a local result, and the danger is that within a week nobody
can tell which was which. Two contaminations, with very different half-lives.

**Session contamination** — the answer enters a sealed session's context and is thereafter
quoted, summarised and built on as if HEARTH produced it. Cost: one session. Recoverable.

**Corpus contamination** — the answer is captured by the learning loop and distilled into a
LoRA. **Not recoverable.** Four reasons this is the severe one:

1. **Weights carry no provenance field.** You can delete a row from a JSONL. You cannot delete
   a row from an adapter. Once trained, the only remedy is retraining from a corpus you must
   now prove is clean — and if capture was silent, you cannot prove that.
2. **It corrupts the evaluation.** If external answers become golden-set targets, the local
   model is being graded against frontier output; the gate stops measuring "is this good" and
   starts measuring "does this imitate". If the same answers are also in the training set you
   have train/test leakage on top, and the promotion gate — already an honour system
   (`docs/LEARNING_plan.md` §0) — starts certifying nonsense.
3. **It poisons the confidence signal, which is what the ladder runs on.** This is the
   non-obvious one. `docs/LEARNING_plan.md` §5.4 measured that fine-tuning a small model on
   knowledge-bound output does not transfer the capability — you get *style* transfer. So a
   3B distilled on frontier reasoning learns to produce fluent, confident-sounding answers to
   questions it still cannot do. That is not a neutral outcome: §6.3 of the same document wants
   to escalate on *low confidence*, and the plan's best proposal is self-consistency dispersion.
   A style-distilled model is confidently wrong with low dispersion — it agrees with itself
   about the wrong answer. **Silent distillation degrades exactly the signal the tier ladder
   uses to decide when to escalate.** The contamination attacks the immune system.
4. **Provenance you cannot disentangle is a liability, not just untidiness.** An answer
   produced by a third-party model, from confidential input, baked into local weights, is an
   artifact whose lineage cannot afterwards be separated or deleted on request.

### 6.1 What the code does about it

- `ExternalAnswer.provenance` is fixed at `"external"` and validation rejects anything else.
  There is no constructor argument that produces a differently-labeled record.
- `training_eligible` is `False` on ingest, with no parameter to change it — that is the point
  of `ingest_answer()`.
- Setting the flag without a promotion record fails validation, so a hand-edited JSON file does
  not sneak through `from_json`.
- `promote_for_training()` is **deliberately awkward**: a named approver and a written
  justification, both recorded. Distilling a frontier answer is a legitimate thing to want;
  doing it silently is not, and for material under a confidentiality obligation the answer is
  often "no" for legal reasons rather than technical ones.
- A promoted record **keeps `provenance: external` forever**. Promotion does not launder; it
  records that somebody signed.
- `provenance_meta()` returns `dict[str, str]` shaped to drop straight into
  `training.dataset.DatasetRecord.meta`, so the tag travels into the JSONL and "which rows came
  from outside?" stays answerable after the fact — without this package importing the training
  code and picking up its dependencies.

### 6.2 What the rest of the system must promise

These are requirements this design places on components that do not exist yet. They belong in
the capture design (`docs/LEARNING_plan.md` §2) before capture is built, because retrofitting
provenance onto a corpus is exactly the thing that cannot be done.

1. **Capture must never treat an ingested answer as a local completion.** Today nothing is
   captured at all (`RequestRecord` stores counters, no text), so there is no bug yet — there is
   a *design hole*, and it closes cheaply now and expensively later.
2. **Ingested answers should not be fed back through the gateway.** If they are, capture sees
   a normal request whose prompt happens to contain frontier output, and no downstream filter
   can tell. If the operator needs the content in-session, the right shape is a citation that
   carries a visible provenance banner and an answer id — not a paste.
3. **No external answer may ever be a golden-set target.** Flat rule. An eval built from
   frontier answers measures imitation, not quality.
4. **RAG is corpus-adjacent and needs the same treatment.** `hearth rag ingest` writes raw
   chunk text that is later retrieved and quoted; an external answer in a shared collection will
   come back looking local. At minimum: a separate collection, and provenance in the chunk
   metadata so the retriever can surface it. This is **Q5**.
5. **`hearth_private.sh --check` should eventually assert the handoff invariant too** — that no
   module under `src/hearth/` other than `providers/remote.py` can open a connection. That
   generalises §2.1's proxy into something that stays true as the repo grows. Not built here;
   it touches the script, which another concern owns.

### 6.3 The limit, stated plainly

HEARTH cannot detect an answer the operator retypes, re-words, or remembers. The recorded path
is safe and auditable; the keyboard is not policed. Every claim in this section is about the
path through `ingest.py`, and the mechanism's honesty depends on the operator using it rather
than routing around it — which is why §7 cares so much about the friction budget.

---

## 7. Moving between the tiers safely and proficiently

### 7.1 The loop, in practice

1. **Run it locally first, for real.** Tier 1 for volume, tier 2 for synthesis. An envelope
   whose `local_attempt` is fabricated or skipped is the failure mode that eats the whole
   design; the required `reason` field exists to make skipping visible.
2. **Diagnose the failure with §1.1.** Convention or format failure ⇒ do not build an envelope,
   fix it locally. Guessing, contradiction, lost thread, missing world-knowledge ⇒ continue.
3. **Label the sensitivity honestly.** This is the field that decides everything downstream and
   it has no default on purpose. When unsure, the answer is `confidential` — the ladder is
   most-restrictive-wins, the same rule as the cmux tier classifier.
4. **Minimise the payload before you build it.** The cheapest redaction is the paragraph you
   never put in the envelope. Send the conflicting two sections, not the whole spec.
5. **Redact, then read the review sheet end to end.** Not skim. The claim being made is
   "a human saw everything that crossed".
6. **Approve and release.** `released/<id>.json` is the pickup point. Nothing has moved.
7. **Cross deliberately, in the right place.** A sealed cmux workspace is sealed
   (`config/cmux/tiers.yaml`); the crossing happens in an **open**-tier workspace or on another
   device. The file in `released/` is still confidential material at rest — it is a plaintext
   copy of the payload, exactly like the RAG index (`docs/PRIVACY.md` § "Data at rest"). Keep
   `~/.hearth` on FileVault and `store.purge("released")` when a handoff is done.
8. **Ingest the answer, don't paste it.** `ingest_answer()` records the linkage by content hash.
   Reading the answer from `inbox/` costs one extra command and buys the entire provenance story.
9. **Decide about training separately, later, in daylight.** Never in the same motion as
   consuming the answer.

### 7.2 A friction budget, because the shortcut is the real competitor

Under deadline pressure the alternative to this loop is not a better loop — it is pasting into
a browser, which leaves no envelope, no hash, no review record and no provenance. So the
mechanism has to be *fast*. Concretely, the operator-facing surface should be about three
commands (build+review, approve+release, ingest); if it is not, the shortcut wins and the
design has failed in the only way that matters. That is **Q3**, and it is not a cosmetic
question.

### 7.3 What "proficiently" looks like after a month

- Tier 3 gets used a few times a week, on `deep`, with a hand-trimmed 5-10K payload — never on
  bulk, because the quota makes bulk impossible anyway.
- Tier 4 sees only `public`-labeled work, and the label was assigned before the task started,
  not argued into place afterwards.
- `inbox/` shows every answer that ever came back, and none of them are in a training corpus.
- The interesting metric is not how often you escalate. It is **how often an envelope's
  `local_attempt.reason` turns out to be a §1.1 row-1 or row-2 failure** — that number is your
  backlog of local work: the LoRAs you have not trained and the constrained decoding you have
  not wired. The ladder should be getting *shorter* over time, not busier.

---

## 8. Open questions for the operator

These block implementation. Each names what it blocks.

**Q1 — May confidential material go to Apple PCC at all?**
PCC is private but not local; the bytes leave the machine. This is a policy/contractual
question, not a threat-modelling one, and only the operator can answer it. *Blocks:* the
`PCC_ACCEPTS_CONFIDENTIAL` constant, currently `False`. *Note:* "no" is a perfectly good answer
and costs nothing — tier 3 remains available for `internal` and `public` work.

**Q2 — Where does the crossing physically happen?**
A Swift CLI on this same Mac makes the seal a *process* boundary, not a *machine* boundary: the
`hearth` process stays `lsof`-clean, the machine does not. A different device keeps the machine
boundary but adds real friction. *Blocks:* what the sealed-mode claim in `docs/PRIVACY.md`
actually says once tier 3 is in use, and where the carrier tool lives.

**Q3 — Manual carriage, or a ferry?**
Fully manual (copy the file, use the tool, paste the answer back) versus a separate,
non-HEARTH binary that reads `released/` and does the crossing. The ferry is far less
error-prone and much faster — see §7.2, where speed is a safety property — but it converts "a
human act" into "a program's act". *Blocks:* whether `release()` is the end of HEARTH's
involvement or the start of another component's, and the shape of the operator CLI.

**Q4 — May an ingested answer influence routing, confidence, or the escalation threshold?**
An external answer is evidence about the *task*, not about the local model. Letting it feed the
confidence function would be a subtle form of the §6 contamination. *Blocks:* whether `inbox/`
is readable by the (not yet built) capture and router-learning subsystems at all.

**Q5 — May an ingested answer enter RAG?**
RAG is not weights, but it is a corpus that shapes output and gets quoted. Separate collection
with provenance metadata, or a flat no? *Blocks:* §6.2 item 4 and any RAG-side schema change.

**Q6 — Verify Python → PCC, or accept Swift-only?**
`apple/python-apple-fm-sdk` binds the **on-device** model; PCC reachability from Python is
unverified. Someone should confirm before any plan depends on it. *Blocks:* the carrier tool's
language, and whether a future ferry (Q3) could be Python.

**Q7 — Is redaction advisory or fail-closed?**
Today it masks and reports. Should a hit on `private_key_block` or `aws_access_key` block
release outright rather than relying on the reviewer noticing the count? Fail-closed is safer
and will produce false positives on, e.g., a code review of key-handling code. *Blocks:*
whether `release()` grows a rule-severity gate.

**Q8 — Retention and purge policy for `~/.hearth/handoff/`.**
Envelopes and answers are plaintext copies of the material they describe, sitting at rest
indefinitely. Purge after N days? On release? Never, for auditability? Auditability and minimal
data-at-rest genuinely conflict here. *Blocks:* whether `purge()` gets a scheduler and what
`docs/PRIVACY.md` § "Data at rest" must say.

**Q9 — Who assigns `sensitivity`, and can it ever be automatic?**
Currently the human, always. A classifier would reduce friction and add a silent failure mode
in the one place the design cannot tolerate one. *Blocks:* any automation of envelope
construction; also interacts with the cmux tier classifier, which already resolves a
sealed/open decision from the workspace and could plausibly supply a floor.

---

## 9. Deliberately out of scope here

Named so the gaps are choices, not oversights:

- **No CLI surface.** No `hearth handoff` command group. That means editing `cli.py`, which
  another concern owns this session. The package's API is arranged so the commands are thin.
- **No gateway route and no MCP tool.** Deliberate: an agent should not be able to *originate*
  a boundary crossing. Envelope construction is an operator action.
- **No PCC or frontier client of any kind**, in any language, anywhere in this repo. That is
  the entire point.
- **No encryption of artifacts at rest.** They inherit FileVault, like the RAG index. Revisit
  under Q8.
- **No automatic sensitivity classification** (Q9), **no batching of multiple tasks into one
  envelope** (it would make the review sheet unreadable, which breaks §5), and **no tier
  config schema** — `config/` belongs to another concern, and there is nothing to configure
  until Q1-Q3 are answered.
- **No change to `scripts/hearth_private.sh`** despite §6.2 item 5 proposing one.

---

## Appendix — draft ADR, ready to move into `docs/DECISIONS.md`

> Formatted to match the existing ADRs. `ADR-011` is the last one in `DECISIONS.md` today
> (the cmux ADRs use a separate `ADR-C0xx` namespace), so this drafts as **ADR-012**. Renumber
> if another concern lands one first.

---

## ADR-012 — Tiers 3-4 reach the outside via a handoff envelope, never a backend

**Context.** HEARTH's local tiers (1: small model + LoRAs, 2: 9-14B) cannot do everything.
Two off-machine tiers are genuinely attractive: Apple Private Cloud Compute (32K context,
reasoning levels, no API keys, free, per-user daily quota) and a frontier model. The obvious
implementation is a `ModelProvider` for each (ADR-004) plus routing entries (ADR-005). But
HEARTH's differentiating property is that it is no-egress **by construction**:
`config/routing.private.yaml` defines zero remotes, `providers/remote.py` is the only code that
could carry content off-box, and `scripts/hearth_private.sh --check` verifies the posture before
serving. That makes `lsof` a complete proof for a non-expert. Critically, a PCC provider would
*not* be a "remote" in the policy's sense — it has no URL and no key to configure — so it would
register as a backend, `policy.remotes` would stay empty, and `--check` would keep printing
"no router egress path exists" about a process holding a live TLS connection. The verifier would
keep passing while the guarantee became false.

**Decision.** Tiers 3 and 4 are **not backends and never appear behind `ModelProvider`**. No
PCC or frontier client code enters this repo in any language. Instead HEARTH emits a **handoff
envelope** (`src/hearth/handoff/`): a local JSON artifact recording the task, why the local
tiers were insufficient (required, with tier/model/result/confidence), the destination tier, an
explicitly-stated sensitivity, and a content hash. Redaction masks obvious secrets and reports
counts; `render_review()` prints the entire payload untruncated; approval is bound to the
content hash so post-approval edits invalidate it; `release()` writes a file and nothing else.
A human carries it across using tools outside HEARTH. Answers return through `ingest_answer()`
as `provenance: external`, `training_eligible: False`, and reach a training corpus only via
`promote_for_training()` with a named approver and written justification — which records the
decision and keeps the external provenance forever. Tier 4 accepts `public` payloads only;
tier 3 refuses `confidential` pending an operator decision. The package imports the standard
library only — no HTTP, no socket, no subprocess, not even another HEARTH module — and a test
enforces that against its own AST.

**Consequences.** `--check` and `lsof` stay true and stay sufficient; the no-egress property
remains structural rather than configuration-dependent, which matters in a repo with concurrent
editors and data-driven routing. Every crossing is deliberate, reviewed, hashed and recorded,
and no frontier output can be silently distilled into a local adapter — which also protects the
confidence signal the ladder uses to decide when to escalate (style transfer from frontier
answers produces a model that is confidently wrong with low dispersion). Costs are real: no
agent loop can use tier 3 mid-task; manual carriage introduces its own errors; the ceremony is
heavy for small tasks, and if it is heavier than pasting into a browser the operator will paste
into a browser and the design loses. Tier 3/4 remain unimplemented pending open questions Q1-Q9
in `docs/TIERS.md`.

**Revisit if.** Tier 3 is needed *inside* an autonomous loop (the envelope model cannot provide
that — reopen rather than work around it). Or a separate, non-HEARTH "ferry" process proves
necessary to keep the friction below the shortcut threshold, in which case the guarantee narrows
from "this machine is silent" to "this process is silent" and `docs/PRIVACY.md` must say so
explicitly. Or Apple ships a PCC path with a verifiable local-attestation story strong enough
that the operator's answer to Q1 changes.
