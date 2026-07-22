# HEARTH × cmux — Branch & work discipline

**Status:** Active. How this effort is organized in git so the standalone HEARTH stays safe, work
is isolated, and nothing is lost. This is the process contract for every contributor and every
sub-agent.

---

## Branch topology

```
main                     ← standalone HEARTH (Phases 0–7). Protected. cmux merges here ONLY at C6.
│
archive/hearth-pre-cmux  ← dormant frozen snapshot of the pre-cmux original (branch)
   └── tag archive/hearth-pre-cmux-2026-07-21   ← immutable frozen point (preferred restore anchor)
│
cmux/integration         ← the WORK TRUNK for all cmux work. Never merges to main until C6.
   ├── cmux/planning-docs     ← this doc set
   ├── cmux/egress-audit      ← C0
   ├── cmux/adr               ← C1
   ├── cmux/wiring            ← C2
   ├── cmux/sealed-profile    ← C3
   ├── cmux/orchestrator      ← C4
   ├── cmux/open-tier         ← C5
   └── cmux/graduation        ← C6
```

### Why the `cmux/` namespace (not `cmux-integration/…`)

git stores refs as files, so it **cannot** have a branch `cmux/integration` *and* branches nested
literally under it. All cmux branches are therefore **siblings under `cmux/`**. They are still
branched **from** `cmux/integration` (the ancestry that matters); the flat naming is the only scheme
git allows. **Never create a bare branch named `cmux`** — it would block the whole namespace.

---

## The two merge gates

1. **Sub-branch → `cmux/integration`.** A `cmux/<task>` branch merges up when *its own* acceptance
   gate (in [ROADMAP.md](ROADMAP.md)) is green. Update the README status tracker in the same merge.
2. **`cmux/integration` → `main`.** Happens once, at **C6**, only when the whole integration is
   **proven, working, successful, and beneficial** (all Proposal success criteria). Never piecemeal,
   never early.

> The standalone guarantee is the point: `main` must, at every moment before C6, still be the
> working standalone HEARTH. If in doubt, do not touch `main`.

---

## Rules

- **One unit of work per sub-branch.** A phase (or a well-scoped slice of one). Keep them small
  enough to review and gate independently.
- **Sub-agents branch too.** Any sub-agent doing build/test work gets its own `cmux/<task>`
  sub-branch (or a worktree on one); it never commits straight to `cmux/integration` or `main`.
- **Docs travel with the work.** A phase isn't done until its doc (RUNBOOK/AUDIT/RESULTS/ADR) is
  written and its README status row is updated. "Nothing gets lost" is enforced here.
- **Third-party code stays out of the repo.** cmux itself is cloned to a scratch location for the
  audit/build, never committed into HEARTH. We commit *our* wiring, launchers, docs — not cmux.
- **Every gate is verifiable.** Prefer a command that proves the claim (`--check`, `lsof`, a test)
  over a prose assertion. Privacy claims especially.
- **Keep `main` restorable.** Nothing merged at C6 may hard-require cmux to be installed for
  standalone HEARTH to run; its conformance suite must still pass with no cmux present.

---

## Reverting / bailing out

If the integration proves not worth it at any point:

- **Before C6:** nothing in `main` changed — just stop. `cmux/integration` and its sub-branches can
  stay for reference or be deleted. Standalone HEARTH is untouched by construction.
- **After C6, if we regret it:** restore from the immutable tag —
  `git switch -c restore-pre-cmux archive/hearth-pre-cmux-2026-07-21` — the frozen original is
  exactly as it was at the start of this effort (Phases 0–7, 224 tests green).

The archive branch `archive/hearth-pre-cmux` is the convenient handle; the **tag** is the
authoritative immutable anchor (branches can move, tags don't).

---

## Definition of done (per phase)

A phase merges to `cmux/integration` only when **all** are true:

1. Its ROADMAP acceptance gate passes (with the proving command/output captured).
2. Its doc artifact exists and is accurate.
3. The README status tracker row is updated to `☑`.
4. Any privacy-relevant claim is `--check`/`lsof`-verified, not asserted.
5. Standalone HEARTH's test suite still passes on the sub-branch (no regression leaked in).
