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
