"""The sealed-uid pf rule TEXT (`cmux-sealed --print-rules`) — B-002 deliverable D1, DESIGN_C7 §3.

This tests what the generator emits, not whether pf enforces it. Enforcement and the pane-child question
are outcomes, measured only by scripts/cmux/pane_egress_probe.sh on real hardware (ADR-C008).
Nothing here loads a rule or touches pf.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "cmux" / "cmux-sealed"


def rules(uid: str | None, **env_extra):
    env = {"PATH": os.environ["PATH"], **env_extra}
    if uid is not None:
        env["CMUX_SEALED_UID"] = uid
    return subprocess.run(["/bin/bash", str(SCRIPT), "--print-rules"], capture_output=True, text=True,
                          env=env, timeout=20)


def test_rules_block_every_protocol_not_just_tcp_and_udp():
    r = rules("599")
    assert r.returncode == 0
    assert "proto" not in r.stdout  # a proto list is what left ICMP open
    assert "block return out quick all user 599" in r.stdout


def test_loopback_is_passed_before_the_block_and_quickly():
    lines = rules("599").stdout.splitlines()
    assert lines[0] == "pass quick on lo0 all"
    assert lines.index("pass quick on lo0 all") < next(i for i, l in enumerate(lines) if l.startswith("block"))


def test_the_block_returns_rather_than_drops_so_it_is_distinguishable_from_a_dead_network():
    assert "block drop" not in rules("599").stdout


def test_the_uid_is_the_one_asked_for_and_only_that_one():
    out = rules("1234").stdout
    assert "user 1234" in out and "user 599" not in out


def test_a_non_numeric_uid_is_refused_not_spliced_into_a_rule():
    for bad in ("5;x", "root", "5 pass all", "-1", ""):
        r = rules(bad)
        assert r.returncode == 64 and r.stdout == "", bad


def test_no_uid_configured_is_refused_not_defaulted_to_the_callers_uid():
    r = rules(None)
    assert r.returncode == 64 and r.stdout == ""


def test_install_path_validates_the_uid_before_it_would_call_sudo():
    # --install-firewall must reject a bad uid before reaching `sudo pfctl`. A sentinel `sudo` on PATH proves it.
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        sudo = Path(d) / "sudo"
        sudo.write_text("#!/bin/sh\necho SUDO-WAS-CALLED\nexit 0\n")
        sudo.chmod(0o755)
        r = subprocess.run(["/bin/bash", str(SCRIPT), "--install-firewall"], capture_output=True, text=True,
                           env={"PATH": f"{d}:{os.environ['PATH']}", "CMUX_SEALED_UID": "5;x"}, timeout=20)
    assert r.returncode == 64 and "SUDO-WAS-CALLED" not in r.stdout


def test_pfctl_accepts_the_rule_text_when_pfctl_can_tell_good_from_bad(tmp_path):
    # `pfctl -n -f` parses without loading. It is only an instrument if it rejects a known-bad file here.
    import shutil
    pfctl = shutil.which("pfctl") or ("/sbin/pfctl" if Path("/sbin/pfctl").exists() else None)
    if not pfctl:
        import pytest; pytest.skip("pfctl not available")
    bad = tmp_path / "bad.conf"; bad.write_text("block return out quick all usr 599\n")
    if subprocess.run([pfctl, "-nf", str(bad)], capture_output=True, timeout=20).returncode == 0:
        import pytest; pytest.skip("pfctl -n does not reject a syntax error here, so it cannot validate anything")
    good = tmp_path / "good.conf"; good.write_text(rules("599").stdout)
    r = subprocess.run([pfctl, "-nf", str(good)], capture_output=True, text=True, timeout=20)
    assert r.returncode == 0, r.stderr
