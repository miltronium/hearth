# FINDING — the sealed tier does not contain the workspace (pane-child egress)

> **Status:** OPEN · **Severity:** high (invalidates the privacy claim people will *read into* §1.3)
> **Found:** 2026-08-19 · **Branch:** `cmux/pane-egress-finding` · **ADR:** [ADR-C009](DECISIONS.md#adr-c009--the-seal-is-scoped-to-the-emitter-not-the-workspace)
> **Tracking issue:** [miltronium/hearth#7](https://github.com/miltronium/hearth/issues/7) (this doc is the source of truth; the issue summarizes it)
>
> **Read this before putting any confidential material through cmux.** This is the resume-here doc
> for the follow-up work; it is written so a fresh agent needs no other context.

---

## 1. TL;DR

The sealed tier blocks **cmux's own binary** from reaching the network. It does **not** block the
processes a pane spawns. A cmux pane has ordinary, unrestricted internet access **while every seal
gate in this repo reports `SEALED` / `PASS`.**

`RESULTS.md §1.3` is not wrong — it proved what it says: `com.cmuxterm.app` made no off-box
connection. The error is one of **scope**: the seal was designed against "cmux the app phones home"
(the C0 audit's threat model) and has been informally read as "confidential work in a sealed
workspace cannot leave the machine." The second claim was never tested. It is false.

**The one-line rule:** the seal covers the **emitter** (cmux.app), not the **workspace** (everything
running in it). A terminal's entire purpose is to run arbitrary programs; a per-app firewall rule
cannot contain that.

---

## 2. The measurement (2026-08-19, cmux 0.64.20, pid 96728)

Driven over the socket into the existing pane at `examples/repos/oss_repo`
(`surface 8F670BF1-224F-4316-B88E-8029BC8A3782`):

```sh
CX=/Applications/cmux.app/Contents/Resources/bin/cmux
"$CX" send --surface "$SID" 'curl -sS -m 8 -o /dev/null -w "http=%{http_code}\n" https://example.com'
"$CX" send-key --surface "$SID" Enter
"$CX" --json read-screen --surface "$SID" --lines 60
```

Observed on the pane:

```
==EGRESS-TEST-START==
PANE_curl_example http=200                 ← public internet, full HTTP round-trip
PANE_python_1.1.1.1:443 CONNECTED          ← raw TCP, non-Apple-signed interpreter path
==EGRESS-TEST-END==
```

**Simultaneously, in the same minute:**

```
$ python3 scripts/cmux/lulu_rule_check.py
SEALED: app is BLOCKED for all endpoints by 1 rule(s)
  filter: LuLu's network extension is loaded and running
  BLOCK  *:*  [com.cmuxterm.app:Developer ID Application: Manaflow, Inc. (7WLXT3NR37)]
exit=0

$ scripts/cmux/cmux-sealed --check --strict .
  PASS  firewall: LuLu blocks 'com.cmuxterm.app' for all endpoints
```

For contrast, the *agent's own* shell (outside cmux, inside the corp Claude Code sandbox) got
`curl: (56) CONNECT tunnel failed, response 403` for the same URL. **The cmux pane was the least
contained execution context on the machine** — less contained than the agent session driving it.

---

## 3. Root cause — three independent layers, all needed for the fix

### 3.1 The rule is scoped to a bundle ID; panes are other binaries

LuLu rules key on the executable/signing identity. A pane's `node`, `claude`, `curl`, `git`, `ssh`,
`python3` are separate processes evaluated on their own identity, not cmux's. Verified — **no rule
exists for any of them**:

```sh
for app in node claude curl git ssh python3; do
  printf '%-8s ' "$app"; python3 scripts/cmux/lulu_rule_check.py --app "$app" --quiet; echo "exit=$?"
done
# every one: exit=1  ("no LuLu rule found for this app")
```

### 3.2 LuLu on this machine is allow-by-default, not deny-by-default

`/Library/Objective-See/LuLu/preferences.plist`:

| Key | Value | Effect |
| --- | --- | --- |
| `blockMode` | `false` | unmatched traffic is not blocked |
| `passiveMode` | `true` (`passiveModeAction` `0`) | **no alerts** — unmatched connections auto-allowed, silently |
| `allowApple` | `true` | Apple-signed binaries (`curl`, `git`, `ssh`, `python3`) bypass rules entirely |
| `allowInstalled` | `true` | anything installed before LuLu (`installTime` 2026-07-23) bypasses rules |
| `allowDNS`, `allowLocalHost` | `true` | expected/benign |

**ADR-C006 #5 requires "a loopback-only firewall profile" — a deny-by-default allowlist.** What is
deployed is an allow-by-default firewall carrying exactly one deny rule. It satisfies the *letter* of
the gate (`lulu_rule_check.py` reads the rule store and finds `BLOCK *:*`) while violating the
*intent*. This is the same failure class as ADR-C008 — one layer down: **scope lies too.**

### 3.3 Nothing in the seal has a concept of a process tree

`cmux-sealed --check`'s seven gates are: tier · hearth-routing · firewall(app) · telemetry ·
auto-update · cli-telemetry · sign-in. None of them ask "what is allowed to run in the pane" or
"what can the pane's children reach". AUDIT §4's five-point invariant is likewise entirely about the
cmux process. The blind spot is structural in the audit's framing, not a coding slip.

---

## 4. The second gap (independent of the firewall): no agent-choice enforcement

`PRIVACY.md` states: *"a sealed workspace's panes run local/sealed HEARTH agents … the gate is
designed so you can't"* run a bare frontier agent over confidential files.

**Nothing implements that.** No check inspects what the pane launches. A frontier coding agent in a
"sealed" pane reads confidential files into a cloud context, and every gate still says PASS. Even a
perfect network fix in §3 would not address this unless the agent's own endpoint is what gets blocked
— which is precisely what §3's fix *would* do, making these two gaps one fix with two acceptance
criteria.

## 5. Third, minor: a `--check` artifact to not misread

```
$ scripts/cmux/cmux-sealed --check --strict .
  FAIL  cli-telemetry: export CMUX_CLI_SENTRY_DISABLED=1 and CMUX_CLAUDE_HOOK_SENTRY_DISABLED=1
==> RESULT: NOT SEALED — failing closed, will not launch.   exit=2
```

**This is a check-mode artifact, not a posture gap** — the gate reads two env vars that the launcher
exports itself (`cmux-sealed:170`), so a bare shell always fails it. Pass them inline:
`CMUX_CLI_SENTRY_DISABLED=1 CMUX_CLAUDE_HOOK_SENTRY_DISABLED=1 scripts/cmux/cmux-sealed --check --strict .`
→ all 7 PASS. Already documented in HANDOFF; noted here so the next session doesn't read the exit 2
above as evidence of anything.

---

## 6. What this does and does not invalidate

| Claim | Verdict |
| --- | --- |
| §1.3 "cmux made no off-box connection under the block" | ✅ **stands** — verified, reproducible, correctly scoped to `com.cmuxterm.app` |
| ADR-C008 (verify enforcement, not config) | ✅ **stands** — and this finding extends it with a *scope* layer |
| §1.5 negative control, Sparkle finding, C2/C4 functional results | ✅ **unaffected** |
| "confidential work in a sealed workspace cannot leave the machine" | ❌ **false as implemented** |
| "sealed tier = airtight *inside that workspace*" (PRIVACY.md) | ❌ **false as implemented** — aspirational, not built |
| "`git push` from inside a cmux pane will fail while the seal holds" (HANDOFF, 2026-08-17) | ❌ **wrong** — corrected; it works, same root cause |

---

## 7. The posture actually wanted (new requirement, 2026-08-19)

The operator's ask was: *"sensitive information … will not be shared or transmitted to any outside
network other than internal networks I am a part of."* **That posture does not exist in the current
model.** Sealed = loopback-only (stricter than asked); open = unrestricted. Nothing anywhere
distinguishes *internal* from *public internet*, and the measurement shows a pane reaching the public
internet.

An "internal-only" tier is a **third posture** and a design decision, not a config tweak. It needs an
egress allowlist at the network layer (proxy/CIDR), which is a different enforcement primitive from
either existing tier. Note the corp Claude Code sandbox already demonstrates the pattern on this
machine (domain allowlist + `dangerous_allowed_domains.csv` + a 403 on deny).

---

## 8. Proposed work — C7 "workspace containment"

Acceptance gate for the phase: **the §2 in-pane probe fails closed** (no connection, from `curl`,
from a non-Apple-signed binary, and from a spawned agent) while cmux remains usable.

### 8.1 Enforcement options (recommendation first)

| # | Approach | Contains pane children? | Cost / risk |
| --- | --- | --- | --- |
| **1** | **Dedicated sealed UID + the existing pf uid anchor** (`scripts/cmux/cmux-sealed.pf.conf` already blocks non-loopback per-uid) — run sealed cmux as that user | **Yes** — uid scope covers every descendant regardless of binary | Medium: needs a second account, file-permission plumbing, `su`/`launchctl` launch path. **Recommended** — it is the only option here that is structural rather than enumerative |
| 2 | LuLu rules for each pane binary | Partially | Unbounded enumeration; `allowApple`/`allowInstalled` bypass it; a new binary defeats it silently |
| 3 | LuLu deny-by-default (`blockMode`, `passiveMode` off, `allowApple`/`allowInstalled` off) | Yes, machine-wide | Blocks the operator's whole machine; high friction; likely reverted in a week — the ADR-C008 "flags regress" failure mode |
| 4 | Run the sealed workspace's panes in `docker --network none` | Yes, per container | Already the design intent for Docker panes (ARCHITECTURE §3); doesn't cover native panes |
| 5 | Proxy allowlist (serves §7's internal-only tier) | Yes, if enforced at the network layer | Different, larger project; the right answer for "internal-only" but not for "sealed" |

pf caveat already recorded in the repo: a named anchor is only enforced if `/etc/pf.conf` references
`anchor "cmux-sealed"` — the ADR-C008 dormant-anchor shape. Verify enforcement, don't assume load.

### 8.2 Tooling deliverables

- [ ] `scripts/cmux/pane_egress_probe.sh` — the §2 test as a **first-class gate**: send a probe into a
      live pane from *two* binary identities (Apple-signed + a local interpreter), assert failure.
      This is the layer-3 outcome check for the workspace, the analog of `cmux_egress_probe.sh` for
      the app.
- [ ] Extend `cmux_egress_probe.sh` to watch **cmux's descendants**, not just processes matching
      `-c cmux` (today it cannot see a pane's `curl` at all).
- [ ] New `cmux-sealed --check` gate: `workspace-containment` — fails closed unless the pane probe
      fails to egress. Must be **mandatory**, not `--strict`-only.
- [ ] New gate: `pane-agent` — refuse to launch sealed if the configured pane agent is a frontier
      agent (closes §4).
- [ ] Scrollback purge on sealed exit (`cmux-sealed --purge`), per PRIVACY.md "Data at rest" — still
      unimplemented.

### 8.3 Doc deliverables

- [ ] ADR-C009 ratified (drafted in this branch) + ADR-C006 amended to say the containment boundary is
      the **process tree / uid**, not the app.
- [ ] AUDIT §4 invariant gains a 6th condition (workspace containment).
- [ ] PRIVACY.md "honest tradeoff" rewritten: today the machine offers **neither** airtight nor gated
      containment for pane work — only app-level quieting.
- [ ] RESULTS §1.8 filled with the fixed-state re-run.

---

## 9. Next steps for whoever picks this up (in order)

```sh
cd /Users/miltronix/Claude/apps/HEARTH
git switch cmux/integration && git pull

# 1. REPRODUCE the finding first — do not trust this doc, it is 30 minutes of one session's work.
#    cmux must be running; socket auth is already `password` on this machine.
. scripts/cmux/cmux-auth-env
CX=/Applications/cmux.app/Contents/Resources/bin/cmux
"$CX" --json --id-format both list-workspaces          # get a workspace UUID
# ... then the §2 send/read-screen sequence. Expect http=200 = finding reproduced.

# 2. Confirm the seal gates still claim SEALED at the same moment (the contradiction is the point).
python3 scripts/cmux/lulu_rule_check.py                # expect exit 0, "SEALED"

# 3. Then pick an option from §8.1. Recommendation: prototype option 1 (sealed uid + pf uid anchor)
#    on a throwaway account before touching the operator's login user.
```

Order matters: **reproduce → confirm the contradiction → then fix.** If step 1 no longer reproduces,
find out *why* before celebrating (a LuLu pref change? a new rule? `passiveMode` off?) — an
unexplained pass is the ADR-C008 failure mode wearing a friendly face.

---

## 10. Do not re-derive (facts established 2026-08-19)

1. **Pane children egress freely** under the current seal — `curl`→`example.com` = 200,
   `python3`→`1.1.1.1:443` connected, with all gates green.
2. **LuLu here is allow-by-default** — `passiveMode=true`, `allowApple=true`, `allowInstalled=true`,
   `blockMode=false`. A block rule exists **only** for `com.cmuxterm.app`.
3. **`git push` from a pane works** — the 2026-08-17 note to the contrary is wrong.
4. **cmux.app itself is genuinely blocked** (rule + live extension) and had **no** established TCP
   connections during this session. §1.3 stands as scoped.
5. **`SULastCheckTime = 2026-08-18`** with `SUEnableAutomaticChecks=0` — Sparkle still *attempts*
   checks; further evidence for ADR-C006/C008, and evidence the block is doing real work.
6. **Data at rest, partial good news:** FileVault **On**. Grepping
   `~/Library/Application Support/cmux/` (session store, `search.db-wal`) and `~/.cmuxterm/events.jsonl`
   for a marker string typed into a pane found **nothing**; `events.jsonl` holds event names
   (`surface.input_sent` etc.), not pane text. **Not** a clearance: PRIVACY.md documents scrollback
   surviving restarts by design, so assume it is written at some point (likely on quit) and treat
   purge-on-exit as still required.
7. **`cmux-sealed --check --strict` exits 2** on the `cli-telemetry` env gate from a bare shell — a
   check-mode artifact (the launcher exports those vars itself), not a regression. Pass them inline.

---

## 11. Open questions

- Does a dedicated sealed uid break cmux's own operation (session store, socket path, app support dir
  are all `$HOME`-relative)? Unknown — prototype needed.
- Does the pf uid anchor actually catch a pane child, or does macOS attribute the socket differently?
  Must be measured with the §2 probe, not reasoned about.
- Is there an upstream cmux appetite for a `--sealed` / local-only mode (ADR-C006 already floats
  upstreaming)? A pane-level network-namespace option would be the clean fix.
- For §7's internal-only tier: is there a sanctioned corp egress proxy this could point at, so the
  allowlist is not hand-maintained?
- **Compliance, out of scope for this repo but not for the operator:** a self-built seal over
  third-party GPL software is not an approved control for confidential data regardless of what these
  probes show. Decide the intended data class before investing in C7.
