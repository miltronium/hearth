# DESIGN — C7 workspace containment (closes B-002)

> **Status:** DRAFT 2026-10-15 · **Nothing here is measured.** Every claim about macOS, pf or cmux
> behaviour is marked MEASURED (a recorded result), READ (from a file in this repo) or UNKNOWN
> (a prototype must answer it). Inputs: [FINDING_pane_egress.md](FINDING_pane_egress.md),
> [ADR-C009](DECISIONS.md), BUGS B-002.

## 1. Decisions already made

- **Posture: loopback-only** (operator, 2026-10-15). A process in a sealed workspace may reach
  `127.0.0.0/8` and `::1` and nothing else. The "internal-only" tier (FINDING §7) is out of scope.
- **Enforcement is not yet chosen.** §4 compares the two candidates and §6 says what to measure.
- **Compliance caveat stands** (FINDING §11): a self-built seal over third-party GPL software is
  not an approved control for regulated confidential data, whatever the probes show. This design
  makes the seal *true*; it does not make it *approved*.

## 2. The invariant

> A sealed workspace is contained iff a probe run **as a child of a pane of that workspace**
> cannot reach a non-loopback address, **and** the same probe run on the same machine **outside**
> the seal can.

The second clause is what makes it a gate (CLAUDE.md §3). A probe that only asserts failure also
passes on a machine with Wi-Fi off, in the wrong pane, or with the probe binary missing. Result
states are three, never two: `CONTAINED`, `LEAKS`, `INCONCLUSIVE`. Only `CONTAINED` launches.

## 3. What the existing seal gets wrong (READ)

| Finding | Source | Consequence for the design |
|---|---|---|
| Rule is keyed on the cmux bundle id | FINDING §3.1 | Demote to defense-in-depth; never the primary control |
| LuLu is allow-by-default | FINDING §3.2 | Do not build on LuLu rules |
| The pf reference rules match `proto { tcp udp }` only | `scripts/cmux/cmux-sealed.pf.conf` | **ICMP and other IP protocols are not blocked.** The new rule must be `block drop out all` with an explicit loopback pass, and the probe must try ICMP |
| `block drop` gives a timeout, not a refusal | pf semantics, UNKNOWN on this host | A blocked probe and a dead network look identical. Prefer `block return` for the sealed uid so denial is fast and distinguishable. For TCP the refusal is visible; for UDP and ICMP a returned block is still only partly distinguishable from silence, so those attempts lean on the paired control. A UDP attempt that "fails" in the pane and also fails in the control is `INCONCLUSIVE`, never evidence of containment |
| `--install-firewall` in `cmux-sealed` generates the same tcp/udp-only rules | `scripts/cmux/cmux-sealed` | **Still unchanged.** Until the generator emits `block return out all` for the sealed uid plus a loopback pass, ICMP is open, and the probe's ICMP attempt will report `LEAKS`. That is the intended outcome of the gate, not a bug in it. Changing the generator is deliverable D1 in §8 and needs §6 to validate it |
| Anchor rules are dormant unless `/etc/pf.conf` references the anchor | pf conf comment | "Anchor loaded" proves nothing; only the probe does |
| `cmux_egress_probe.sh` greps processes matching `-c cmux` | script header | Cannot see a pane's `curl`. Replace with a probe that runs *inside* the pane |

## 4. Candidates

### A. Dedicated sealed uid + pf `user <uid>` rule

Run sealed cmux as a separate local user. pf blocks every non-loopback packet from that uid, so
every descendant is covered whatever binary it is.

- **For:** structural, not enumerative. A new binary in a pane is contained with no change. The
  operator's login uid is a natural **control** for the paired probe (§2).
- **Against:** needs a second account, a launch path (`sudo -u` / `launchctl asuser`) and shared
  file access to the repo being worked on. Nothing but the probe proves it works.
- **UNKNOWN, must be measured:**
  1. Does cmux start and keep state under another `$HOME`? (session store, socket path)
  2. Does pf attribute a pane child's socket to the sealed uid? (setuid binaries, `sudo`, and
     anything launched through the login user's launchd could escape the uid)
  3. Can the sealed uid reach the operator's uid through loopback (a local proxy the operator
     runs would be a hole)? Loopback-only means any loopback listener is an egress path.
  4. Does the repo checkout need to be readable by the sealed uid, and what does that expose?

### B. Panes in `docker --network none`

Every sealed pane is a container with no network namespace peers.

- **For:** containment is the kernel's, per container. Already the design intent (ARCHITECTURE §3).
- **Against:** drops native panes entirely for sealed work, so `claude`/`node` tooling and the
  operator's usual shell have to live in the image. A bind mount of the repo is the data path.
- **UNKNOWN:** whether the engine in use gives each container a truly empty network namespace on
  this host, and whether HEARTH on `127.0.0.1:8080` is reachable from inside (a `--network none`
  container cannot see host loopback, so the pane agent would need a mounted unix socket).
  That last point may make B incompatible with "panes talk to HEARTH". Check it first.

### Disqualified

LuLu per-binary rules (enumerative, bypassed by `allowApple`/`allowInstalled`), LuLu
deny-by-default (machine-wide, expected to be reverted), proxy allowlist (different tier).

### Recommendation

Prototype **A** first. It is the only candidate that keeps native panes. Take **B** if A fails
question 2 above, or if question 3 shows loopback cannot be kept clean. Do not mix them in one
gate until one of them has passed §5 on its own.

## 5. The gate: `workspace-containment`

New mandatory gate in `cmux-sealed --check`, and a standalone `scripts/cmux/pane_egress_probe.sh`.

**Procedure**

1. **Machinery control (loopback).** From inside the pane, connect to a throwaway listener the
   gate opens on `127.0.0.1`. Must succeed, or the pane cannot run the probe at all →
   `INCONCLUSIVE`.
2. **Sealed probe.** From inside the pane, attempt each of: `curl` (Apple-signed), a non-Apple
   interpreter doing TCP to an IP literal, UDP to an IP literal, ICMP, an IPv6 literal, and a
   DNS lookup. Every attempt must fail. IP literals keep the result independent of DNS.
3. **Unsealed control.** Run the same probe set from the control context. Candidate A: the
   operator's own uid, outside any pane. Candidate B: a throwaway container on the default
   bridge network (`docker run` without `--network none`), same image and probe, so the only
   difference from the sealed pane is the network. Every attempt that failed in
   step 2 must **succeed** here. If any also fails, the network or target is down →
   `INCONCLUSIVE`, never `CONTAINED`.
4. **Identity check.** Record `id -u` from inside the pane and assert it equals the expected
   sealed uid. Then assert the probe's parent chain reaches the pane: the probe walks its own
   `ppid` chain (`ps -o ppid= -p`) and the gate requires it to hit the pane shell's pid. **How the
   gate learns the pane shell's pid is UNKNOWN** (does the cmux CLI expose it?); §6 step 4
   measures it. Until it is measured the ancestry assertion is weak, and the implementation
   (`CmuxPaneRunner.identity()`) only confirms the surface answered. The uid half is real, and
   with candidate A it is the stronger check, because the pf rule is keyed on the uid. Do not
   describe the gate as verifying ancestry until §6 step 4 has an answer.
5. **Verdict.** `CONTAINED` only if 1 passes, 2 all fail, 3 all succeed, 4 matches.

**Where the gate gets its inputs.** The gate must be told which pane to probe and which uid is
sealed, and it must not guess either (a default of "the current uid" would make the check
describe the configuration, not the workspace). For now both are environment variables read by
`cmux-sealed`: `CMUX_SEALED_SURFACE` (the pane surface id) and `CMUX_SEALED_UID`. If either is
unset the gate reports `INCONCLUSIVE` and fails closed, which is the true state today. Note that
`--install-firewall` already reads `CMUX_SEALED_UID` and defaults it to the caller's uid; that
default is acceptable for installing a rule but must not carry over to the gate. **Open
decision:** whether these belong in `config/cmux/tiers.yaml` instead. Environment variables are
fine for the prototype and wrong for a launcher that should not depend on the operator's shell.

**What it reports if broken the most plausible way**

| Plausible breakage | Reported |
|---|---|
| Anchor not referenced from `/etc/pf.conf` (dormant) | step 2 succeeds → `LEAKS` |
| Machine offline | step 3 fails → `INCONCLUSIVE` |
| Probe sent to a pane running as the wrong uid | step 4 mismatch → `INCONCLUSIVE` |
| New binary appears in a pane | covered by uid scope; no change needed |
| ICMP left open by a tcp/udp-only rule | the ICMP attempt succeeds → `LEAKS` |
| Rule loaded, then flushed mid-session | gate re-runs at launch only → **not covered**; see §7 |

The gate must be unit-testable with an injected runner and fake outcomes, with a mutation check
that each plausible breakage above flips the verdict. The real probe is operator-run.

## 6. Measurement plan (operator, on a throwaway account; agents cannot `bind()` or touch pf)

For candidate A, in order, stopping at the first failure:

1. Reproduce the finding: FINDING §9 steps 1–2 (`http=200` while the gates say `SEALED`).
2. Create a throwaway user. Does cmux launch and keep working as that user?
3. Load the anchor, add the `anchor` line to `/etc/pf.conf`, then run the §5 probe set as that
   user from a terminal (not yet a pane). Record every attempt's outcome.
4. Repeat from inside a cmux pane. Then a pane child that is `sudo`, a launchd-spawned job and a
   setuid binary.
5. Leave a loopback listener forwarding outward and confirm the gate cannot see it (this is the
   known residual risk, question 3).

Write results into `RESULTS.md` §1.8 only after the probe has also been run against the **current**
seal and found to reach the internet — the "can tell the two apart" requirement from B-002.

## 7. Residual risk, stated plainly

- The gate runs at launch. A rule flushed afterwards is not detected. A periodic re-probe or a
  `pfctl` state check would narrow it but a state check is a configuration check; only the
  probe is an outcome check.
- Loopback is trusted. Any local listener that forwards (an SSH tunnel, a proxy, HEARTH's own
  remote provider if a remote profile were ever active) is an egress path. The sealed launcher
  must refuse a non-no-egress HEARTH routing profile; it already has a `hearth-routing` gate.
- Pane-agent choice (FINDING §4) is a separate gate: `pane-agent` refuses to launch sealed if
  the configured pane agent is a frontier agent. With loopback-only enforced that agent cannot
  reach its endpoint, so it fails closed on its own — but the gate should say so rather than
  letting the operator discover it at the first prompt.

## 8. Work split

| Item | Who | Needs |
|---|---|---|
| This design reviewed and the candidates agreed | operator + agent | — |
| `pane_egress_probe.sh` and the verdict logic, with fake-runner tests and mutation checks | agent (isolated worktree) | no sockets |
| `workspace-containment` and `pane-agent` gates wired into `cmux-sealed --check` | agent | no sockets |
| §6 measurement on a throwaway account | operator | pf, a second user, cmux.app |
| D1: change the `--install-firewall` generator and `cmux-sealed.pf.conf` to `block return out all` for the sealed uid with a loopback pass; tcp/udp-only is the cause of an open ICMP | agent writes, operator validates in §6 | pf to validate |
| Pane-agent gate: needs a config field naming the pane agent first; without it a refusal would be a guess | decision, then agent | operator decision |
| ADR-C009 follow-up, AUDIT §4 sixth condition, PRIVACY.md rewrite, RESULTS §1.8 | agent, after §6 | measured results |

## 9. Acceptance (restates B-002)

The §5 probe, run from a child of a sealed pane, returns `CONTAINED`; the same probe against the
current seal returns `LEAKS`; and an offline machine returns `INCONCLUSIVE`. Until then the
sealed tier must not be described as protecting confidential work.
