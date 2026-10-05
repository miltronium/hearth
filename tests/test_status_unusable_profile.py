"""B-108: the status report shows an unusable active profile as unusable — wherever it lives.

Before: ``active_profile`` was ``[ok]`` whenever the file existed, so a bad-rung profile that
``hearth serve`` refuses (exit 2) read green; and a profile selected from outside
``config/`` was never loaded at all, so its routing error never appeared. Unusable is
reported as unusable (FAIL: not safe to run), not dressed as an egress finding.
"""

from __future__ import annotations

from pathlib import Path

from hearth.status.probes import probe_egress
from hearth.status.report import LEVEL_FAIL, LEVEL_OK

BAD_RUNG = (
    "classes:\n  chat: { backend: local, escalate: never, "
    "local_model: mlx-community/bge-small-en-v1.5-bf16 }\n"
)
NO_EGRESS = (
    "defaults: {local_model: auto, remote: none, remote_budget_tokens_per_day: 0}\n"
    "classes:\n  chat: {backend: local, escalate: never}\nremotes: {}\n"
)


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    (root / "config" / "routing.yaml").write_text(NO_EGRESS)
    return root


def _fact(section, name):
    return next(f for f in section.facts if f.name == name)


def test_an_unusable_active_profile_in_config_fails_and_says_why(tmp_path):
    root = _repo(tmp_path)
    bad = root / "config" / "routing.bad.yaml"
    bad.write_text(BAD_RUNG)
    section = probe_egress(root=root, environ={"HEARTH_ROUTING_YAML": str(bad)})
    active = _fact(section, "active_profile")
    assert active.level == LEVEL_FAIL
    assert "UNUSABLE" in active.detail and "bge-small" in active.detail
    assert "not an egress finding" in active.detail
    assert active.data["unusable"] is True
    listed = _fact(section, "config/routing.bad.yaml")
    assert listed.value.startswith("UNUSABLE") and "egress" not in listed.value


def test_an_unusable_active_profile_outside_config_shows_its_error(tmp_path):
    root = _repo(tmp_path)
    outside = tmp_path / "elsewhere" / "routing.mine.yaml"
    outside.parent.mkdir()
    outside.write_text(BAD_RUNG)
    section = probe_egress(root=root, environ={"HEARTH_ROUTING_YAML": str(outside)})
    active = _fact(section, "active_profile")
    assert active.level == LEVEL_FAIL and "bge-small" in active.detail
    listed = _fact(section, str(outside))  # judged like the profiles in config/
    assert listed.value.startswith("UNUSABLE") and "bge-small" in listed.detail


def test_a_good_profile_outside_config_is_judged_on_its_posture(tmp_path):
    root = _repo(tmp_path)
    outside = tmp_path / "elsewhere" / "routing.mine.yaml"
    outside.parent.mkdir()
    outside.write_text(NO_EGRESS)
    section = probe_egress(root=root, environ={"HEARTH_ROUTING_YAML": str(outside)})
    active = _fact(section, "active_profile")
    assert active.level == LEVEL_OK and "unusable" not in active.data
    assert _fact(section, str(outside)).value.startswith("NO EGRESS")
