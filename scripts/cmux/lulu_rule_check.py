#!/usr/bin/env python3
"""lulu_rule_check.py — verify that LuLu actually BLOCKS an app's outbound traffic.

Why this exists (found live 2026-08-17): `cmux-sealed --check` used to pass its mandatory firewall
gate whenever a LuLu/Little Snitch process was *running*. On this machine LuLu was running **and had
an explicit ALLOW `*:*` rule for cmux** — so the gate reported a seal that did not exist. "A firewall
is installed" is not "this app is blocked"; ADR-C006 requires the structural seal, so the gate has to
read the rule store, not the process table.

LuLu keeps its rules in an NSKeyedArchiver plist at /Library/Objective-See/LuLu/rules.plist
(world-readable). Rule dicts carry `action`: 0 = block, 1 = allow.

A SECOND live finding, 2026-08-17, minutes after the first: the rule store said BLOCK *:* and this
script reported SEALED — while LuLu's network extension was `[terminated waiting to uninstall on
reboot]` and LuLu.app was not running. cmux connected out anyway. **A rule that nothing enforces is
not a seal.** So the check has three layers, all of which must hold:

  1. a block-everything rule exists for the app,
  2. no allow rule contradicts it,
  3. LuLu's network filter is actually LOADED AND RUNNING.

Exit codes (FAIL-CLOSED — anything but 0 means "do not treat as sealed"):
  0  blocked by rule AND the filter is live
  1  the app is NOT blocked (an allow rule exists, or no rule at all)
  2  undetermined — LuLu not installed, store unreadable, or an unrecognized format
  3  rule says block, but LuLu's filter is NOT running — nothing is enforcing it

Usage:
  lulu_rule_check.py [--app com.cmuxterm.app] [--rules /path/to/rules.plist] [--json] [--quiet]
                     [--skip-liveness]

Caveat: LuLu writes this store asynchronously after a GUI edit. If you just changed a rule and the
verdict looks stale, re-run after a moment. Rule state is still not proof of *outcome* — the
authoritative confirmation is always scripts/cmux/cmux_egress_probe.sh reporting loopback-only.
"""
from __future__ import annotations

import argparse
import json
import plistlib
import subprocess
import sys

DEFAULT_RULES = "/Library/Objective-See/LuLu/rules.plist"
BLOCK, ALLOW = 0, 1
ACTION_NAME = {BLOCK: "BLOCK", ALLOW: "ALLOW"}

# LuLu marks "any endpoint" with these; treat them as a wildcard match.
WILDCARDS = {"*", "any", None, ""}


def _deref(obj, objects, depth=0, seen=frozenset()):
    """Resolve an NSKeyedArchiver object graph into plain Python."""
    if depth > 12:
        return "<max-depth>"
    if isinstance(obj, plistlib.UID):
        idx = obj.data
        if idx in seen:
            return "<cycle>"
        return _deref(objects[idx], objects, depth + 1, seen | {idx})
    if isinstance(obj, dict):
        if "NS.keys" in obj and "NS.objects" in obj:
            return {
                str(_deref(k, objects, depth + 1, seen)): _deref(v, objects, depth + 1, seen)
                for k, v in zip(obj["NS.keys"], obj["NS.objects"])
            }
        if "NS.objects" in obj:
            return [_deref(v, objects, depth + 1, seen) for v in obj["NS.objects"]]
        return {
            str(k): _deref(v, objects, depth + 1, seen)
            for k, v in obj.items()
            if not str(k).startswith("$")
        }
    if isinstance(obj, list):
        return [_deref(v, objects, depth + 1, seen) for v in obj]
    return obj


def load_rules(path: str) -> dict:
    with open(path, "rb") as fh:
        raw = plistlib.load(fh)
    objects = raw.get("$objects")
    if not objects:
        raise ValueError("not an NSKeyedArchiver store (no $objects)")
    top = _deref(objects[1], objects)
    if not isinstance(top, dict):
        raise ValueError("unexpected rule-store shape")
    return top


def rules_for(store: dict, app: str) -> list[tuple[str, dict]]:
    """Every rule belonging to an app whose key/contents mention `app`."""
    needle = app.lower()
    out: list[tuple[str, dict]] = []
    for key, val in store.items():
        if needle not in (str(key) + str(val)).lower():
            continue
        entries = val.get("rules", val) if isinstance(val, dict) else val
        if isinstance(entries, list):
            for r in entries:
                if isinstance(r, dict):
                    out.append((str(key), r))
    return out


def is_block_all(rule: dict) -> bool:
    """A rule that blocks every endpoint (not just one host/port)."""
    if rule.get("action") != BLOCK:
        return False
    addr = rule.get("endpointAddr")
    port = rule.get("endpointPort")
    return (addr in WILDCARDS or str(addr) == "*") and (port in WILDCARDS or str(port) == "*")


EXT_ID = "com.objective-see.lulu.extension"
# States that mean the filter is not filtering, whatever the rules say.
DEAD_STATES = ("terminated", "uninstall", "disabled", "waiting to activate", "activated waiting")


def filter_liveness(list_output: str | None = None, procs: str | None = None) -> tuple[bool, str]:
    """Is LuLu's network extension actually loaded and running?

    Two independent signals, because either alone can lie: `systemextensionsctl list` reports the
    registered state, and the process table reports whether anything is actually there. Both must
    agree before we call the filter live.
    """
    if list_output is None:
        try:
            list_output = subprocess.run(
                ["systemextensionsctl", "list"], capture_output=True, text=True, timeout=15
            ).stdout
        except Exception as exc:
            return False, f"cannot query system extensions: {exc}"

    line = next((ln for ln in list_output.splitlines() if EXT_ID in ln), None)
    if line is None:
        return False, "LuLu's network extension is not registered with the system"

    low = line.lower()
    for bad in DEAD_STATES:
        if bad in low:
            state = line.split("[")[-1].rstrip("]") if "[" in line else low.strip()
            return False, f"LuLu's network extension is not running (state: {state})"

    # `enabled` and `active` are the first two columns, marked with '*' when true.
    cols = line.split("\t")
    if len(cols) >= 2 and not (cols[0].strip() == "*" and cols[1].strip() == "*"):
        return False, "LuLu's network extension is registered but not enabled+active"

    if procs is None:
        try:
            procs = subprocess.run(["pgrep", "-fl", EXT_ID], capture_output=True, text=True, timeout=10).stdout
        except Exception:
            procs = ""
    if EXT_ID not in procs:
        return False, "LuLu's network extension process is not running"

    return True, "LuLu's network extension is loaded and running"


def evaluate(rules: list[tuple[str, dict]]) -> tuple[int, str]:
    if not rules:
        return 1, "no LuLu rule found for this app — LuLu will prompt (or default-allow) rather than block"
    allows = [r for _, r in rules if r.get("action") == ALLOW]
    blocks_all = [r for _, r in rules if is_block_all(r)]
    if allows:
        # An explicit allow is the dangerous case: the firewall is running but permits the app.
        return 1, f"app is ALLOWED by {len(allows)} rule(s) — NOT sealed"
    if blocks_all:
        return 0, f"app is BLOCKED for all endpoints by {len(blocks_all)} rule(s)"
    return 1, "only narrow block rule(s) present — no block-everything rule, so egress paths remain"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--app", default="com.cmuxterm.app", help="bundle id or path substring (default: cmux)")
    ap.add_argument("--rules", default=DEFAULT_RULES, help=f"rule store (default: {DEFAULT_RULES})")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--quiet", action="store_true", help="exit code only")
    ap.add_argument("--skip-liveness", action="store_true",
                    help="check rules only, do NOT verify the filter is running (unsafe; for diagnostics)")
    args = ap.parse_args()

    try:
        store = load_rules(args.rules)
    except FileNotFoundError:
        if not args.quiet:
            print(f"UNDETERMINED: no LuLu rule store at {args.rules} (LuLu not installed?)", file=sys.stderr)
        return 2
    except Exception as exc:  # unreadable or format changed -> fail closed, never guess
        if not args.quiet:
            print(f"UNDETERMINED: cannot read {args.rules}: {exc}", file=sys.stderr)
        return 2

    matched = rules_for(store, args.app)
    code, reason = evaluate(matched)

    # A correct rule that nothing enforces is not a seal — this overrides a clean rule verdict.
    live, live_reason = (True, "liveness check skipped") if args.skip_liveness else filter_liveness()
    if code == 0 and not live:
        code, reason = 3, f"rule blocks the app, but {live_reason}"

    if args.json:
        print(json.dumps({
            "app": args.app,
            "sealed": code == 0,
            "reason": reason,
            "filter_live": live,
            "filter_status": live_reason,
            "rules": [
                {
                    "key": k,
                    "action": ACTION_NAME.get(r.get("action"), r.get("action")),
                    "endpoint": f"{r.get('endpointAddr')}:{r.get('endpointPort')}",
                }
                for k, r in matched
            ],
        }, indent=2))
    elif not args.quiet:
        verdict = "SEALED" if code == 0 else "NOT SEALED"
        print(f"{verdict}: {reason}")
        print(f"  filter: {live_reason}")
        for k, r in matched:
            print(f"  {ACTION_NAME.get(r.get('action'), r.get('action')):5}  "
                  f"{r.get('endpointAddr')}:{r.get('endpointPort')}  [{k}]")
    return code


if __name__ == "__main__":
    sys.exit(main())
