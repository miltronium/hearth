"""Doctor tests — checks run and report structurally, regardless of the host machine."""

from __future__ import annotations

from hearth.doctor import Check, all_fatal_passed, run_checks


def test_run_checks_returns_expected_names(settings):
    names = {c.name for c in run_checks(settings)}
    assert {"apple_silicon", "memory", "mlx_backend", "state_dir"} <= names


def test_state_dir_check_passes_with_tmp_home(settings):
    checks = run_checks(settings)
    state = next(c for c in checks if c.name == "state_dir")
    assert state.ok is True
    # With a writable tmp home, no fatal check should fail.
    assert all_fatal_passed(checks) is True


def test_all_fatal_passed_ignores_nonfatal():
    checks = [
        Check("a", ok=False, detail="", fatal=False),
        Check("b", ok=True, detail="", fatal=True),
    ]
    assert all_fatal_passed(checks) is True
