"""Tests for the LuLu rule verifier (scripts/cmux/lulu_rule_check.py) — ADR-C006 seal invariant.

These pin the fail-closed behaviour that motivated the script: on 2026-08-17 `cmux-sealed --check`
passed its firewall gate because a LuLu *process* was running, while LuLu in fact held an explicit
ALLOW *:* rule for cmux. The gate now reads rules, so the rule reader must never report "sealed"
on anything short of a real block-everything rule.

Loaded by file path — cmux tooling lives under scripts/, not in the HEARTH package (ADR-C002/C005).
"""

from __future__ import annotations

import importlib.util
import plistlib
from pathlib import Path

_MOD_PATH = Path(__file__).resolve().parents[1] / "scripts" / "cmux" / "lulu_rule_check.py"
_spec = importlib.util.spec_from_file_location("lulu_rule_check", _MOD_PATH)
lrc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lrc)

BLOCK_ALL = {"action": 0, "endpointAddr": "*", "endpointPort": "*"}
ALLOW_ALL = {"action": 1, "endpointAddr": "*", "endpointPort": "*"}
BLOCK_ONE = {"action": 0, "endpointAddr": "example.com", "endpointPort": "443"}


# --- the regression this whole script exists for ------------------------------------------------

def test_explicit_allow_is_not_sealed():
    """The live failure: firewall running, app explicitly permitted."""
    code, reason = lrc.evaluate([("com.cmuxterm.app", ALLOW_ALL)])
    assert code == 1
    assert "ALLOWED" in reason


def test_block_all_is_sealed():
    code, _ = lrc.evaluate([("com.cmuxterm.app", BLOCK_ALL)])
    assert code == 0


def test_allow_beats_block_when_both_present():
    """A stray allow alongside a block must NOT be reported as sealed."""
    code, _ = lrc.evaluate([("app", BLOCK_ALL), ("app", ALLOW_ALL)])
    assert code == 1


# --- fail-closed on everything ambiguous --------------------------------------------------------

def test_no_rules_is_not_sealed():
    code, reason = lrc.evaluate([])
    assert code == 1
    assert "no LuLu rule" in reason


def test_narrow_block_is_not_sealed():
    """Blocking one host leaves every other egress path open."""
    code, reason = lrc.evaluate([("app", BLOCK_ONE)])
    assert code == 1
    assert "narrow" in reason


def test_is_block_all_rejects_narrow_and_allow():
    assert lrc.is_block_all(BLOCK_ALL)
    assert not lrc.is_block_all(BLOCK_ONE)
    assert not lrc.is_block_all(ALLOW_ALL)


def test_missing_store_exits_undetermined(tmp_path, monkeypatch, capsys):
    """Absent rule store => exit 2 (undetermined), never a pass."""
    monkeypatch.setattr(
        "sys.argv", ["lulu_rule_check.py", "--rules", str(tmp_path / "nope.plist"), "--quiet"]
    )
    assert lrc.main() == 2


def test_unreadable_store_exits_undetermined(tmp_path, monkeypatch):
    """A corrupt/foreign plist must fail closed rather than guess."""
    bad = tmp_path / "rules.plist"
    with open(bad, "wb") as fh:
        plistlib.dump({"not": "an archive"}, fh)
    monkeypatch.setattr("sys.argv", ["lulu_rule_check.py", "--rules", str(bad), "--quiet"])
    assert lrc.main() == 2


# --- filter liveness: a rule nothing enforces is not a seal --------------------------------------
# Found live 2026-08-17: rules.plist said BLOCK *:*, LuLu's extension was "terminated waiting to
# uninstall on reboot", and cmux connected out regardless.

LIVE_LINE = "*\t*\tVBG97UB4TA\tcom.objective-see.lulu.extension (4.3.2/4.3.2)\tLuLu\t[activated enabled]"
DEAD_LINE = "\t\tVBG97UB4TA\tcom.objective-see.lulu.extension (4.3.2/4.3.2)\tLuLu\t[terminated waiting to uninstall on reboot]"
PROCS = "652 /Library/SystemExtensions/X/com.objective-see.lulu.extension.systemextension/Contents/MacOS/com.objective-see.lulu.extension"


def test_liveness_true_when_active_and_running():
    live, _ = lrc.filter_liveness(list_output=LIVE_LINE, procs=PROCS)
    assert live


def test_liveness_false_when_terminated():
    """The exact live failure state."""
    live, reason = lrc.filter_liveness(list_output=DEAD_LINE, procs=PROCS)
    assert not live
    assert "not running" in reason


def test_liveness_false_when_process_absent():
    """Registered and enabled, but nothing actually running."""
    live, reason = lrc.filter_liveness(list_output=LIVE_LINE, procs="")
    assert not live
    assert "process is not running" in reason


def test_liveness_false_when_not_registered():
    live, reason = lrc.filter_liveness(list_output="1 extension(s)\n", procs=PROCS)
    assert not live
    assert "not registered" in reason


def test_liveness_false_when_enabled_columns_blank():
    line = "\t\tVBG97UB4TA\tcom.objective-see.lulu.extension (4.3.2/4.3.2)\tLuLu\t[activated enabled]"
    live, reason = lrc.filter_liveness(list_output=line, procs=PROCS)
    assert not live
    assert "not enabled+active" in reason


# --- the archive decoder ------------------------------------------------------------------------

def test_rules_for_matches_by_substring():
    store = {
        "com.cmuxterm.app:Developer ID": {"rules": [BLOCK_ALL]},
        "com.other.app:Developer ID": {"rules": [ALLOW_ALL]},
    }
    matched = lrc.rules_for(store, "com.cmuxterm.app")
    assert len(matched) == 1
    assert matched[0][1]["action"] == 0


def test_rules_for_is_case_insensitive():
    store = {"com.cmuxterm.app": {"rules": [BLOCK_ALL]}}
    assert len(lrc.rules_for(store, "CMUXTERM")) == 1
