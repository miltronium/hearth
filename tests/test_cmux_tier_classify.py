"""Tests for the cmux tier classifier (scripts/cmux/tier_classify.py) — ADR-C003 invariants.

The classifier is cmux-integration tooling under scripts/ (not a HEARTH package, per ADR-C002/C005),
so it is loaded by file path here. These tests pin the fail-safe invariants: default sealed, open by
explicit opt-in only, sealed_override always wins, unknown/error ⇒ sealed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_MOD_PATH = Path(__file__).resolve().parents[1] / "scripts" / "cmux" / "tier_classify.py"
_spec = importlib.util.spec_from_file_location("tier_classify", _MOD_PATH)
tc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tc)


def test_default_sealed_when_no_rules():
    assert tc.classify("/Users/x/anything", {}) == tc.SEALED


def test_unknown_repo_is_sealed():
    cfg = {"default": "sealed", "open": [{"path": "/Users/x/oss/*"}]}
    assert tc.classify("/Users/x/secret/repo", cfg) == tc.SEALED


def test_open_opt_in_by_path_glob():
    cfg = {"open": [{"path": "/Users/x/oss/*"}]}
    assert tc.classify("/Users/x/oss/proj", cfg) == tc.OPEN


def test_open_opt_in_by_remote_host():
    cfg = {"open": [{"remote_host": "github.com/public-org/*"}]}
    assert tc.classify("/tmp/w", cfg, remote="github.com/public-org/repo") == tc.OPEN


def test_remote_host_no_match_stays_sealed():
    cfg = {"open": [{"remote_host": "github.com/public-org/*"}]}
    assert tc.classify("/tmp/w", cfg, remote="github.com/other/repo") == tc.SEALED


def test_sealed_override_beats_open_path():
    cfg = {
        "open": [{"path": "/Users/x/*"}],  # broad open
        "sealed_override": [{"path": "/Users/x/apple/*"}],
    }
    assert tc.classify("/Users/x/apple/proj", cfg) == tc.SEALED
    assert tc.classify("/Users/x/oss/proj", cfg) == tc.OPEN


def test_sealed_override_by_remote_host_contains():
    cfg = {
        "open": [{"remote_host": "*"}],  # everything open by remote
        "sealed_override": [{"remote_host_contains": "apple"}],
    }
    assert tc.classify("/tmp/w", cfg, remote="ghe.apple.com/team/repo") == tc.SEALED
    assert tc.classify("/tmp/w", cfg, remote="github.com/oss/repo") == tc.OPEN


def test_default_open_is_honored_but_override_still_wins():
    cfg = {"default": "open", "sealed_override": [{"path": "/Users/x/apple/*"}]}
    assert tc.classify("/Users/x/random", cfg) == tc.OPEN
    assert tc.classify("/Users/x/apple/proj", cfg) == tc.SEALED


def test_bad_config_type_falls_back_to_sealed():
    # a malformed rule set must not throw open — most-restrictive-wins
    assert tc.classify("/x", {"open": "not-a-list"}) == tc.SEALED


def test_normalize_remote_variants():
    assert tc.normalize_remote("git@github.com:org/repo.git") == "github.com/org/repo"
    assert tc.normalize_remote("https://github.com/org/repo.git") == "github.com/org/repo"
    assert tc.normalize_remote("ssh://git@ghe.apple.com/team/repo") == "ghe.apple.com/team/repo"
    assert tc.normalize_remote("") is None
    assert tc.normalize_remote(None) is None


def test_tilde_and_env_expansion_in_rule(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", "/Users/tester")
    cfg = {"open": [{"path": "~/oss/*"}]}
    assert tc.classify("/Users/tester/oss/p", cfg) == tc.OPEN
    assert tc.classify("/Users/tester/work/p", cfg) == tc.SEALED


# --- main() exit-code contract the cmux-sealed launcher depends on -----------------------------

def _write_cfg(tmp_path, body: str):
    p = tmp_path / "tiers.yaml"
    p.write_text(body)
    return p


def test_main_require_sealed_exit_codes(tmp_path):
    open_dir = (tmp_path / "openproj").resolve()
    open_dir.mkdir()
    cfg = _write_cfg(tmp_path, f'default: sealed\nopen:\n  - path: "{open_dir}*"\n')
    # open-classified repo => require-sealed refuses (exit 3)
    assert tc.main([str(open_dir), "--require-sealed", "--config", str(cfg)]) == 3
    # a non-matching path stays sealed (exit 0)
    assert tc.main([str(tmp_path / "secret"), "--require-sealed", "--config", str(cfg)]) == 0


def test_main_assert_open_exit_codes(tmp_path):
    open_dir = (tmp_path / "oss").resolve()
    open_dir.mkdir()
    cfg = _write_cfg(tmp_path, f'default: sealed\nopen:\n  - path: "{open_dir}*"\n')
    assert tc.main([str(open_dir), "--assert-open", "--config", str(cfg)]) == 0
    # unclassified path fails the open assertion (fail-closed to sealed) => exit 3
    assert tc.main([str(tmp_path / "secret"), "--assert-open", "--config", str(cfg)]) == 3


def test_main_usage_error_without_repo():
    assert tc.main([]) == 64
