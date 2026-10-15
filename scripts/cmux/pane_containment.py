#!/usr/bin/env python3
"""pane_containment.py — verdict logic for the `workspace-containment` gate (B-002, DESIGN_C7 §5).

Why this exists: the sealed tier reported SEALED while a pane's `curl` reached the public internet
(docs/cmux/FINDING_pane_egress.md). Every earlier check inspected a CONFIGURATION (a rule, a process,
a loaded anchor). This one asserts on the OUTCOME: what a probe run as a child of the pane can reach,
compared against the same probe run outside the seal.

Result states are three, never two:
  CONTAINED     every attempt was blocked in the pane AND reached in the control, uid matched and a cmux ancestor was seen
  LEAKS         at least one attempt reached the network from the pane
  INCONCLUSIVE  the evidence cannot distinguish "sealed" from "offline / wrong pane / tool missing"

Verdict order (first match wins — the order is load-bearing):
  (a) loopback machinery failed, uid != expected sealed uid, or ancestry check failed -> INCONCLUSIVE
  (b) any pane attempt REACHED                                                         -> LEAKS
      (a positive reach proves a leak even if the control is down)
  (c) any pane attempt UNAVAILABLE or missing, or BLOCKED in the pane but not REACHED
      in the control                                                                   -> INCONCLUSIVE
  (d) otherwise                                                                        -> CONTAINED

Exit codes (fail-closed — only 0 means "contained"):
  0  CONTAINED
  3  LEAKS
  4  INCONCLUSIVE
  2  usage error

Usage:
  pane_containment.py --surface SURFACE_ID --sealed-uid UID     # operator-run, real pane
  (both may come from CMUX_SEALED_SURFACE / CMUX_SEALED_UID; with neither configured the result is
   INCONCLUSIVE — containment is not built yet.)

NOT MEASURED, always (printed on every run): rules flushed after this run (the gate runs at launch);
loopback listeners that forward outward; setuid/sudo/launchd escapes of the sealed uid; any protocol
or destination outside the six attempts; whether the control's success reflects the same route the
pane would take. Nothing in the unit tests touches a network: they inject a fake Runner.
"""
from __future__ import annotations

import argparse
import enum
import os
import re
import secrets
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Mapping, Protocol

ATTEMPTS: tuple[str, ...] = (
    "curl_https", "tcp_ip_literal", "udp_ip_literal", "icmp", "tcp_ipv6_literal", "dns_lookup",
)
CONTEXTS = ("pane", "control")

EXIT_CONTAINED, EXIT_USAGE, EXIT_LEAKS, EXIT_INCONCLUSIVE = 0, 2, 3, 4

NOT_MEASURED = (
    "that the probed surface is the workspace being launched (the gate probes a surface it is given)",
    "ancestry beyond 'an ancestor process's executable path contains cmux' (ssh/su into the same uid passes)",
    "that results are read back from a screen scrape of the pane, not from the process itself",
    "that the pane and the control share PATH, proxy and certificate environment (they may not)",
    "that IPv6, ICMP and DNS are reachable by design on this network (an IPv4-only net is INCONCLUSIVE)",
    "rules flushed or changed after this run (gate runs at launch only)",
    "loopback listeners that forward outward (loopback is trusted)",
    "sudo / setuid / launchd-spawned escapes from the sealed uid",
    "protocols and destinations beyond the six attempts",
    "that the control used the same route the pane would have",
)

DESIGN_DOC = "docs/cmux/DESIGN_C7_containment.md"


class Outcome(enum.Enum):
    REACHED = "REACHED"
    BLOCKED = "BLOCKED"
    UNAVAILABLE = "UNAVAILABLE"  # probe tool missing/errored in a way that says nothing about the network


class Verdict(enum.Enum):
    CONTAINED = "CONTAINED"
    LEAKS = "LEAKS"
    INCONCLUSIVE = "INCONCLUSIVE"


class Runner(Protocol):
    def loopback_ok(self) -> bool: ...
    def identity(self) -> tuple[int, bool]: ...  # (uid, ancestry_reaches_pane)
    def attempt(self, context: str, name: str) -> Outcome: ...


@dataclass(frozen=True)
class Evidence:
    loopback_ok: bool
    uid: int
    ancestry_ok: bool
    pane: Mapping[str, Outcome] = field(default_factory=dict)
    control: Mapping[str, Outcome] = field(default_factory=dict)


def verdict(ev: Evidence, expected_uid: int) -> tuple[Verdict, str]:
    # (a) the probe may not have run where we think it ran
    if not ev.loopback_ok:
        return Verdict.INCONCLUSIVE, "loopback machinery check failed — the pane cannot run the probe"
    if ev.uid != expected_uid:
        return Verdict.INCONCLUSIVE, f"probe ran as uid {ev.uid}, expected sealed uid {expected_uid}"
    if not ev.ancestry_ok:
        return Verdict.INCONCLUSIVE, "no ancestor of the probe is a cmux process (wrong pane, or ssh/su out of it)"
    # (b) a positive reach is a leak regardless of the control
    reached = [n for n in ATTEMPTS if ev.pane.get(n) is Outcome.REACHED]
    if reached:
        return Verdict.LEAKS, "pane reached the network: " + ", ".join(reached)
    # (c) anything we cannot interpret
    missing = [n for n in ATTEMPTS if n not in ev.pane]
    if missing:
        return Verdict.INCONCLUSIVE, "no pane result for: " + ", ".join(missing)
    unavailable = [n for n in ATTEMPTS if ev.pane[n] is Outcome.UNAVAILABLE]
    if unavailable:
        return Verdict.INCONCLUSIVE, "probe tool unavailable in pane: " + ", ".join(unavailable)
    uncontrolled = [n for n in ATTEMPTS if ev.control.get(n) is not Outcome.REACHED]
    if uncontrolled:
        return Verdict.INCONCLUSIVE, (
            "blocked in the pane but not shown reachable in the control (offline or target down?): "
            + ", ".join(uncontrolled))
    # (d)
    return Verdict.CONTAINED, "all six attempts blocked in the pane and reached in the control"


def collect(runner: Runner) -> Evidence:
    """Gather evidence. A runner exception means the instrument failed: UNAVAILABLE, never BLOCKED."""
    try:
        lo = bool(runner.loopback_ok())
    except Exception:
        lo = False
    try:
        uid, anc = runner.identity()
    except Exception:
        uid, anc = -1, False
    res: dict[str, dict[str, Outcome]] = {c: {} for c in CONTEXTS}
    for ctx in CONTEXTS:
        for name in ATTEMPTS:
            try:
                res[ctx][name] = runner.attempt(ctx, name)
            except Exception:
                res[ctx][name] = Outcome.UNAVAILABLE
    return Evidence(lo, uid, anc, res["pane"], res["control"])


def render(ev: Evidence, v: Verdict, reason: str) -> str:
    lines = [f"  {'attempt':<18} {'pane':<12} control"]
    for n in ATTEMPTS:
        p, c = ev.pane.get(n), ev.control.get(n)
        lines.append(f"  {n:<18} {p.value if p else 'MISSING':<12} {c.value if c else 'MISSING'}")
    lines.append(f"  loopback={ev.loopback_ok} uid={ev.uid} ancestor_is_cmux={ev.ancestry_ok}")
    lines.append(f"{v.value}: {reason}")
    lines.append("NOT MEASURED:")
    lines.extend(f"  - {x}" for x in NOT_MEASURED)
    return "\n".join(lines)


_EXIT = {Verdict.CONTAINED: EXIT_CONTAINED, Verdict.LEAKS: EXIT_LEAKS, Verdict.INCONCLUSIVE: EXIT_INCONCLUSIVE}


def run(runner: Runner, expected_uid: int) -> tuple[int, str]:
    ev = collect(runner)
    v, reason = verdict(ev, expected_uid)
    return _EXIT[v], render(ev, v, reason)


# --- OPERATOR-RUN ONLY ------------------------------------------------------------------------------
# CmuxPaneRunner is never constructed by a test. It makes real network attempts from a real pane.
# The pure helpers around it (_py_src, _probe_line, _wrap, _classify) are tested with fake commands.
#
# Exit-code contract for every probe fragment: 0 = reached, 10 = positive evidence of a block
# (connection refused / unreachable / timeout / no reply), anything else = the instrument failed
# (missing tool, certificate error, bad proxy, stub binary) and is UNAVAILABLE, never BLOCKED.
EXIT_BLOCKED = 10

_GUARD = "command -v {tool} >/dev/null || exit 127; "
_UDP_QUERY = ("\\x00\\x01\\x01\\x00\\x00\\x01\\x00\\x00\\x00\\x00\\x00\\x00"
              "\\x07example\\x03com\\x00\\x00\\x01\\x00\\x01")


def _py_src(body: str) -> str:
    """Python source that exits 0 if `body` completes and 10 if it raises OSError (incl. timeouts, gaierror)."""
    return f"import socket,sys\ntry:\n {body}\nexcept OSError:\n sys.exit({EXIT_BLOCKED})"


def _py_probe(body: str) -> str:
    # exec(repr(src)) keeps the whole probe on ONE line: a real newline typed into a pane would press Enter.
    return _GUARD.format(tool="python3") + "python3 -c " + shlex.quote(f"exec({_py_src(body)!r})")


def _tool_probe(tool: str, command: str, blocked_codes: str) -> str:
    return (_GUARD.format(tool=tool) + f"{command}; r=$?; case $r in 0) exit 0;; {blocked_codes}) "
            f"exit {EXIT_BLOCKED};; *) exit 99;; esac")


_PROBES = {
    # curl 6 = could not resolve, 7 = could not connect, 28 = timed out. Others (60/77 CA bundle, 5 proxy) are NOT a block.
    "curl_https": _tool_probe("curl", "curl -sS -m 6 -o /dev/null https://example.com", "6|7|28"),
    "tcp_ip_literal": _py_probe("socket.create_connection(('1.1.1.1',443),5)"),
    "udp_ip_literal": _py_probe(
        "s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.settimeout(4);s.connect(('1.1.1.1',53));"
        f"s.send(b'{_UDP_QUERY}');s.recv(512)"),
    "icmp": _tool_probe("ping", "ping -c1 -t4 1.1.1.1 >/dev/null", "2"),
    "tcp_ipv6_literal": _py_probe("socket.create_connection(('2606:4700:4700::1111',443),5)"),
    "dns_lookup": _py_probe("socket.getaddrinfo('example.com',443)"),
}
_LOOPBACK = ("command -v python3 >/dev/null && python3 -c \"import socket;l=socket.socket();l.bind(('127.0.0.1',0));"
             "l.listen(1);socket.create_connection(l.getsockname(),2)\" >/dev/null 2>&1")
_ANCESTRY = ('p=$$; a=NO; while [ "$p" -gt 1 ] 2>/dev/null; do case "$(ps -o comm= -p $p)" in *cmux*) a=YES;; esac; '
             "p=$(ps -o ppid= -p $p | tr -d ' '); done")


def _classify(rc: int) -> Outcome:
    return Outcome.REACHED if rc == 0 else Outcome.BLOCKED if rc == EXIT_BLOCKED else Outcome.UNAVAILABLE


def _wrap(name: str, nonce: str) -> str:
    # The result goes through printf %s with the nonce as an argument, so the echoed command text never
    # matches the marker regex, and a marker from an earlier run (other nonce) can never be mistaken for this one.
    return (f"( {_PROBES[name]} ) >/dev/null 2>&1; r=$?; "
            f"case $r in 0) o=REACHED;; {EXIT_BLOCKED}) o=BLOCKED;; *) o=UNAVAILABLE;; esac; "
            f"printf 'PCX%s:%s:%s\\n' {nonce} {name} $o")


def _markers(screen: str, nonce: str) -> list[tuple[str, ...]]:
    return re.findall(rf"PCX{re.escape(nonce)}:(\w+):(\w+)(?::(\w+))?", screen)


class CmuxPaneRunner:
    """Sends probes into a live pane via the cmux CLI; the control runs locally as the operator."""

    CX = "/Applications/cmux.app/Contents/Resources/bin/cmux"

    def __init__(self, surface: str) -> None:
        self.surface = surface

    @staticmethod
    def _nonce() -> str:
        return secrets.token_hex(6)

    def _pane(self, shell_line: str, wait: float = 12.0) -> str:
        subprocess.run([self.CX, "send", "--surface", self.surface, shell_line], check=True, timeout=15)
        subprocess.run([self.CX, "send-key", "--surface", self.surface, "Enter"], check=True, timeout=15)
        time.sleep(wait)
        out = subprocess.run([self.CX, "--json", "read-screen", "--surface", self.surface, "--lines", "80"],
                             capture_output=True, text=True, timeout=15, check=True)
        return out.stdout

    def loopback_ok(self) -> bool:
        n = self._nonce()
        line = (f"{_LOOPBACK} && printf 'PCX%s:%s:%s\\n' {n} loopback REACHED "
                f"|| printf 'PCX%s:%s:%s\\n' {n} loopback BLOCKED")
        found = [m for m in _markers(self._pane(line, 4), n) if m[0] == "loopback"]
        return bool(found) and found[-1][1] == "REACHED"

    def identity(self) -> tuple[int, bool]:
        n = self._nonce()
        line = f"{_ANCESTRY}; printf 'PCX%s:%s:%s:%s\\n' {n} ident $(id -u) $a"
        found = [m for m in _markers(self._pane(line, 4), n) if m[0] == "ident"]
        if not found or not found[-1][1].isdigit():
            return -1, False
        return int(found[-1][1]), found[-1][2] == "YES"

    def attempt(self, context: str, name: str) -> Outcome:
        if context == "control":
            return _classify(subprocess.run(["sh", "-c", _PROBES[name]], capture_output=True, timeout=20).returncode)
        n = self._nonce()
        found = [m for m in _markers(self._pane(_wrap(name, n)), n) if m[0] == name]
        if not found or found[-1][1] not in Outcome.__members__:
            return Outcome.UNAVAILABLE
        return Outcome(found[-1][1])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--surface", default=os.environ.get("CMUX_SEALED_SURFACE"), help="cmux surface id of a sealed pane")
    ap.add_argument("--sealed-uid", default=os.environ.get("CMUX_SEALED_UID"), help="uid the sealed pane must run as")
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code in (0, None) else EXIT_USAGE
    if not args.surface or args.sealed_uid is None:
        print("INCONCLUSIVE: workspace containment is not built yet — no sealed pane / sealed uid configured "
              f"(see {DESIGN_DOC}). Failing closed.")
        print("NOT MEASURED: everything.")
        return EXIT_INCONCLUSIVE
    try:
        uid = int(args.sealed_uid)
    except ValueError:
        print(f"usage: --sealed-uid must be an integer, got {args.sealed_uid!r}", file=sys.stderr)
        return EXIT_USAGE
    code, text = run(CmuxPaneRunner(args.surface), uid)
    print(text)
    return code


if __name__ == "__main__":
    sys.exit(main())
