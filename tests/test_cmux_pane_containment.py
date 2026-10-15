"""Tests for the workspace-containment verdict (scripts/cmux/pane_containment.py) — B-002, DESIGN_C7 §5.

Every test uses an injected fake runner; nothing here touches a network, a pane or cmux.
Each row of the design doc's "what it reports if broken the most plausible way" table has a test, and
each verdict rule has at least one test that fails when the rule is removed (see the mutation table in
the commit message / report).

Loaded by file path — cmux tooling lives under scripts/, not in the HEARTH package (ADR-C002/C005).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_MOD_PATH = Path(__file__).resolve().parents[1] / "scripts" / "cmux" / "pane_containment.py"
_spec = importlib.util.spec_from_file_location("pane_containment", _MOD_PATH)
pc = importlib.util.module_from_spec(_spec)
sys.modules["pane_containment"] = pc  # dataclasses resolves string annotations via sys.modules
_spec.loader.exec_module(pc)

R, B, U = pc.Outcome.REACHED, pc.Outcome.BLOCKED, pc.Outcome.UNAVAILABLE
SEALED_UID = 599


class FakeRunner:
    """Healthy sealed world by default; each test breaks exactly one thing."""

    def __init__(self, *, loopback=True, uid=SEALED_UID, ancestry=True, pane=None, control=None, raises=()):
        self._loopback, self._id = loopback, (uid, ancestry)
        self.table = {
            "pane": {n: B for n in pc.ATTEMPTS},
            "control": {n: R for n in pc.ATTEMPTS},
        }
        self.table["pane"].update(pane or {})
        self.table["control"].update(control or {})
        self.raises = set(raises)
        self.calls: list[tuple[str, str]] = []

    def loopback_ok(self):
        return self._loopback

    def identity(self):
        return self._id

    def attempt(self, context, name):
        self.calls.append((context, name))
        if (context, name) in self.raises:
            raise OSError("instrument failure")
        return self.table[context][name]


def verdict_of(runner, expected_uid=SEALED_UID):
    ev = pc.collect(runner)
    return pc.verdict(ev, expected_uid)[0]


def evidence(**kw):
    base = dict(loopback_ok=True, uid=SEALED_UID, ancestry_ok=True,
                pane={n: B for n in pc.ATTEMPTS}, control={n: R for n in pc.ATTEMPTS})
    base.update(kw)
    return pc.Evidence(**base)


# --- the happy path, so the failures below mean something --------------------------------------------

def test_healthy_sealed_world_is_contained():
    assert verdict_of(FakeRunner()) is pc.Verdict.CONTAINED


def test_attempt_names_are_the_six_fixed_ones():
    assert pc.ATTEMPTS == ("curl_https", "tcp_ip_literal", "udp_ip_literal", "icmp",
                           "tcp_ipv6_literal", "dns_lookup")


# --- one test per row of the design table ------------------------------------------------------------

def test_row_dormant_anchor_pane_reaches_everything_is_leaks():
    """Anchor not referenced from /etc/pf.conf: every pane attempt succeeds."""
    r = FakeRunner(pane={n: R for n in pc.ATTEMPTS})
    assert verdict_of(r) is pc.Verdict.LEAKS


def test_row_offline_machine_is_inconclusive_not_contained():
    """Wi-Fi off: pane blocked AND control blocked. Must never read as sealed."""
    r = FakeRunner(control={n: B for n in pc.ATTEMPTS})
    assert verdict_of(r) is pc.Verdict.INCONCLUSIVE


def test_row_wrong_uid_pane_is_inconclusive():
    """Probe landed in a pane running as the operator, not the sealed uid."""
    r = FakeRunner(uid=501)
    assert verdict_of(r) is pc.Verdict.INCONCLUSIVE


def test_row_icmp_left_open_is_leaks():
    """tcp/udp-only rule: five attempts blocked, ICMP reaches."""
    r = FakeRunner(pane={"icmp": R})
    v, reason = pc.verdict(pc.collect(r), SEALED_UID)
    assert v is pc.Verdict.LEAKS
    assert "icmp" in reason


def test_row_new_missing_probe_tool_is_inconclusive():
    """A probe binary disappears (e.g. no python3): the attempt says nothing about the network."""
    r = FakeRunner(pane={"tcp_ip_literal": U})
    assert verdict_of(r) is pc.Verdict.INCONCLUSIVE


# --- missing attempt, precedence, and the remaining rules --------------------------------------------

def test_missing_attempt_is_inconclusive_never_contained():
    ev = evidence(pane={n: B for n in pc.ATTEMPTS if n != "dns_lookup"})
    assert pc.verdict(ev, SEALED_UID)[0] is pc.Verdict.INCONCLUSIVE


def test_precedence_a_over_b_wrong_uid_beats_a_reach():
    """Wrong pane AND a reach: the reach says nothing about the sealed pane, so INCONCLUSIVE."""
    ev = evidence(uid=501, pane={**{n: B for n in pc.ATTEMPTS}, "curl_https": R})
    assert pc.verdict(ev, SEALED_UID)[0] is pc.Verdict.INCONCLUSIVE


def test_precedence_a_over_b_loopback_failure_beats_a_reach():
    ev = evidence(loopback_ok=False, pane={**{n: B for n in pc.ATTEMPTS}, "icmp": R})
    assert pc.verdict(ev, SEALED_UID)[0] is pc.Verdict.INCONCLUSIVE


def test_ancestry_failure_is_inconclusive():
    assert verdict_of(FakeRunner(ancestry=False)) is pc.Verdict.INCONCLUSIVE


def test_reach_is_leaks_even_when_control_is_down():
    """b before c: a positive reach proves a leak without needing the control."""
    r = FakeRunner(pane={"udp_ip_literal": R}, control={n: B for n in pc.ATTEMPTS})
    assert verdict_of(r) is pc.Verdict.LEAKS


def test_one_attempt_dead_in_control_blocks_contained():
    """Only dns blocked in the control: that attempt proves nothing."""
    r = FakeRunner(control={"dns_lookup": B})
    assert verdict_of(r) is pc.Verdict.INCONCLUSIVE


def test_missing_control_result_is_not_a_reach():
    ev = evidence(control={n: R for n in pc.ATTEMPTS if n != "icmp"})
    assert pc.verdict(ev, SEALED_UID)[0] is pc.Verdict.INCONCLUSIVE


def test_unavailable_control_is_inconclusive():
    assert verdict_of(FakeRunner(control={"curl_https": U})) is pc.Verdict.INCONCLUSIVE


def test_runner_exception_is_unavailable_not_blocked():
    """A crashed instrument must not be read as a blocked network."""
    r = FakeRunner(raises={("pane", "icmp")})
    ev = pc.collect(r)
    assert ev.pane["icmp"] is U
    assert pc.verdict(ev, SEALED_UID)[0] is pc.Verdict.INCONCLUSIVE


def test_identity_exception_is_inconclusive():
    class Boom(FakeRunner):
        def identity(self):
            raise RuntimeError("cmux gone")

    assert verdict_of(Boom()) is pc.Verdict.INCONCLUSIVE


# --- CLI ---------------------------------------------------------------------------------------------

def test_run_exit_codes_and_output():
    code, text = pc.run(FakeRunner(), SEALED_UID)
    assert code == 0 and "CONTAINED" in text and "NOT MEASURED" in text
    assert all(n in text for n in pc.ATTEMPTS)
    assert pc.run(FakeRunner(pane={"icmp": R}), SEALED_UID)[0] == 3
    assert pc.run(FakeRunner(uid=1), SEALED_UID)[0] == 4


def test_cli_unconfigured_fails_closed_pointing_at_design(monkeypatch, capsys):
    monkeypatch.delenv("CMUX_SEALED_SURFACE", raising=False)
    monkeypatch.delenv("CMUX_SEALED_UID", raising=False)
    assert pc.main([]) == 4
    out = capsys.readouterr().out
    assert "INCONCLUSIVE" in out and "not built yet" in out and "DESIGN_C7_containment.md" in out


def test_cli_usage_errors_exit_2(monkeypatch):
    monkeypatch.delenv("CMUX_SEALED_SURFACE", raising=False)
    monkeypatch.delenv("CMUX_SEALED_UID", raising=False)
    assert pc.main(["--bogus"]) == 2
    assert pc.main(["--surface", "S", "--sealed-uid", "abc"]) == 2


# --- the operator-run layer: everything except the cmux CLI calls ------------------------------------
# Review of 320564d found the bug class lives here (stale screen reads, any non-zero exit scored BLOCKED,
# ancestry == "a marker appeared"). These tests run the real shell fragments against FAKE tools in a temp
# PATH — no network, no pane, no cmux.

import os
import subprocess
import tempfile


def _run_wrapped(name, tool_rcs, nonce="n0nce"):
    """Execute _wrap(name) in sh with fake curl/ping/python3 that exit with the given codes."""
    with tempfile.TemporaryDirectory() as d:
        for tool, rc in tool_rcs.items():
            path = Path(d) / tool
            path.write_text(f"#!/bin/sh\nexit {rc}\n")
            path.chmod(0o755)
        out = subprocess.run(["/bin/sh", "-c", pc._wrap(name, nonce)], capture_output=True, text=True,
                             env={"PATH": d}, timeout=20).stdout
    return pc._markers(out, nonce)


def test_curl_block_exit_codes_are_blocked_but_cert_and_proxy_errors_are_not():
    for rc, want in [(0, "REACHED"), (6, "BLOCKED"), (7, "BLOCKED"), (28, "BLOCKED"),
                     (60, "UNAVAILABLE"), (77, "UNAVAILABLE"), (5, "UNAVAILABLE"), (1, "UNAVAILABLE")]:
        (m,) = _run_wrapped("curl_https", {"curl": rc})
        assert m[:2] == ("curl_https", want), (rc, m)


def test_a_stub_python_that_exits_1_is_unavailable_not_blocked():
    (m,) = _run_wrapped("tcp_ip_literal", {"python3": 1})
    assert m[:2] == ("tcp_ip_literal", "UNAVAILABLE")
    (m,) = _run_wrapped("tcp_ip_literal", {"python3": pc.EXIT_BLOCKED})
    assert m[:2] == ("tcp_ip_literal", "BLOCKED")
    (m,) = _run_wrapped("tcp_ip_literal", {"python3": 0})
    assert m[:2] == ("tcp_ip_literal", "REACHED")


def test_missing_tool_is_unavailable():
    (m,) = _run_wrapped("icmp", {})  # no ping on PATH
    assert m[:2] == ("icmp", "UNAVAILABLE")


def test_ping_no_reply_is_blocked_other_failures_are_not():
    assert _run_wrapped("icmp", {"ping": 2})[0][1] == "BLOCKED"
    assert _run_wrapped("icmp", {"ping": 68})[0][1] == "UNAVAILABLE"


def test_every_python_probe_compiles_and_stays_on_one_line():
    for name, line in pc._PROBES.items():
        assert "\n" not in line and "\n" not in pc._wrap(name, "x"), name
    for body in ("socket.getaddrinfo('example.com',443)", f"s.send(b'{pc._UDP_QUERY}')"):
        compile(pc._py_src(body), "probe", "exec")


def test_the_echoed_command_text_never_matches_the_marker_regex():
    # What the pane echoes back is the command we typed; it must not itself look like a result.
    assert pc._markers(pc._wrap("curl_https", "abc123"), "abc123") == []


def test_classify_control_exit_codes():
    assert pc._classify(0) is R and pc._classify(pc.EXIT_BLOCKED) is B
    assert pc._classify(127) is U and pc._classify(1) is U and pc._classify(99) is U


class ScreenRunner(pc.CmuxPaneRunner):
    """CmuxPaneRunner with the cmux CLI replaced by a canned screen."""

    def __init__(self, screen_for):
        super().__init__("surface-x")
        self._screen_for, self.nonce = screen_for, "fresh"

    def _nonce(self):
        return self.nonce

    def _pane(self, shell_line, wait=12.0):
        return self._screen_for(shell_line)


def test_a_stale_marker_from_an_earlier_run_cannot_shadow_this_run():
    screen = "PCXstale:curl_https:BLOCKED\nPCXfresh:curl_https:REACHED\n"
    assert ScreenRunner(lambda _l: screen).attempt("pane", "curl_https") is R


def test_a_stale_marker_alone_is_unavailable_never_blocked():
    assert ScreenRunner(lambda _l: "PCXstale:curl_https:BLOCKED\n").attempt("pane", "curl_https") is U


def test_the_last_marker_of_this_run_wins():
    screen = "PCXfresh:curl_https:BLOCKED\nPCXfresh:curl_https:REACHED\n"
    assert ScreenRunner(lambda _l: screen).attempt("pane", "curl_https") is R


def test_garbled_or_unknown_marker_is_unavailable():
    assert ScreenRunner(lambda _l: "garbage\n").attempt("pane", "icmp") is U
    assert ScreenRunner(lambda _l: "PCXfresh:icmp:WEIRD\n").attempt("pane", "icmp") is U
    assert ScreenRunner(lambda _l: "PCXfresh:other:REACHED\n").attempt("pane", "icmp") is U


def test_a_stale_loopback_reached_does_not_satisfy_the_loopback_check():
    assert ScreenRunner(lambda _l: "PCXstale:loopback:REACHED\n").loopback_ok() is False
    assert ScreenRunner(lambda _l: "PCXfresh:loopback:BLOCKED\n").loopback_ok() is False
    assert ScreenRunner(lambda _l: "PCXfresh:loopback:REACHED\n").loopback_ok() is True


def test_identity_reads_uid_and_a_measured_cmux_ancestor():
    assert ScreenRunner(lambda _l: "PCXfresh:ident:599:YES\n").identity() == (599, True)
    assert ScreenRunner(lambda _l: "PCXfresh:ident:599:NO\n").identity() == (599, False)
    assert ScreenRunner(lambda _l: "").identity() == (-1, False)
    assert ScreenRunner(lambda _l: "PCXstale:ident:599:YES\n").identity() == (-1, False)


def _ancestry_flag(cmux_pid_comm):
    """Run the real ancestry fragment with a fake `ps`: the shell's chain is $$ -> 500 -> 1.
    `cmux_pid_comm` is the executable path reported for pid 500."""
    with tempfile.TemporaryDirectory() as d:
        ps = Path(d) / "ps"
        ps.write_text(
            "#!/bin/sh\n"
            'for last; do :; done\n'  # POSIX: the pid is the final argument
            'case "$*" in\n'
            f'  *comm=*) if [ "$last" = 500 ]; then echo "{cmux_pid_comm}"; else echo zsh; fi;;\n'
            '  *ppid=*) if [ "$last" = 500 ]; then echo 1; else echo 500; fi;;\n'
            "esac\n")
        ps.chmod(0o755)
        return subprocess.run(["/bin/sh", "-c", pc._ANCESTRY + "; echo $a"], capture_output=True, text=True,
                              env={"PATH": f"{d}:/usr/bin:/bin"}, timeout=20).stdout.strip()


def test_ancestry_fragment_finds_a_cmux_ancestor_and_not_otherwise():
    assert _ancestry_flag("/Applications/cmux.app/Contents/MacOS/cmux") == "YES"
    assert _ancestry_flag("/usr/sbin/sshd") == "NO"


class _LoopbackRaises(FakeRunner):
    def loopback_ok(self):
        raise OSError("screen read failed")


class _IdentityRaises(FakeRunner):
    def identity(self):
        raise OSError("screen read failed")


def test_a_loopback_exception_is_a_failed_check_not_ok():
    ev = pc.collect(_LoopbackRaises())
    assert ev.loopback_ok is False
    assert verdict_of(_LoopbackRaises()) is pc.Verdict.INCONCLUSIVE


def test_an_identity_exception_fails_both_uid_and_ancestry():
    ev = pc.collect(_IdentityRaises())
    assert ev.uid == -1 and ev.ancestry_ok is False
    assert verdict_of(_IdentityRaises()) is pc.Verdict.INCONCLUSIVE
