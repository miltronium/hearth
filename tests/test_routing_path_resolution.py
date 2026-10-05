"""B-008 / B-025: one resolver for ``HEARTH_ROUTING_YAML``.

The status probe, ``hearth doctor --offline`` and the router must name the SAME file. These
tests assert on what ``load_policy()`` actually read (a spy on ``Path.read_text`` plus a
sentinel in the profile), not on what any of them say they would read (CLAUDE.md §3).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hearth.doctor import _check_routing
from hearth.router import policy as policy_mod
from hearth.router.policy import (
    RoutingProfileNotFoundError,
    default_policy_path,
    load_policy,
)
from hearth.status.probes import probe_egress

_REPO = Path(__file__).resolve().parent.parent

_SENTINEL_PROFILE = """\
defaults: {local_model: auto, remote: sentinel_remote}
classes:
  reason: {backend: remote, escalate: always}
remotes:
  sentinel_remote: {protocol: anthropic, model: sentinel-model-from-%s}
"""


@pytest.fixture
def reads(monkeypatch) -> list[Path]:
    """Every routing file ``load_policy`` opens, absolute."""
    seen: list[Path] = []
    real = Path.read_text

    def spy(self, *args, **kwargs):
        if self.suffix == ".yaml" and "routing" in self.name:
            seen.append(Path(self).absolute())
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", spy)
    return seen


def _probe_path(env: dict[str, str]) -> Path:
    section = probe_egress(root=_REPO, environ=env)
    fact = next(f for f in section.facts if f.name == "active_profile")
    return Path(fact.data["path"])


def test_tilde_path_probe_doctor_and_router_agree(tmp_path, monkeypatch, reads):
    home = tmp_path / "home"
    home.mkdir()
    (home / "my-routing.yaml").write_text(_SENTINEL_PROFILE % "home")
    monkeypatch.setenv("HOME", str(home))
    env = {"HEARTH_ROUTING_YAML": "~/my-routing.yaml"}
    monkeypatch.setenv("HEARTH_ROUTING_YAML", env["HEARTH_ROUTING_YAML"])

    policy = load_policy(known_models=set())

    assert reads == [home / "my-routing.yaml"]
    assert policy.remote_for().model == "sentinel-model-from-home"  # it really read that file
    assert _probe_path(env) == reads[0]
    assert default_policy_path(env) == reads[0]
    _, check = _check_routing(None, env)
    assert str(reads[0]) in check.detail
    assert "CAN ESCAPE" in check.detail  # judged on the sentinel, not the safe defaults


def test_relative_path_resolves_against_repo_root_not_cwd(tmp_path, monkeypatch, reads):
    # A decoy profile under the cwd, at the same relative path, with a sentinel remote.
    decoy_dir = tmp_path / "elsewhere" / "config"
    decoy_dir.mkdir(parents=True)
    (decoy_dir / "routing.remote.yaml").write_text(_SENTINEL_PROFILE % "decoy")
    monkeypatch.chdir(tmp_path / "elsewhere")
    env = {"HEARTH_ROUTING_YAML": "config/routing.remote.yaml"}
    monkeypatch.setenv("HEARTH_ROUTING_YAML", env["HEARTH_ROUTING_YAML"])

    policy = load_policy()

    expected = _REPO / "config" / "routing.remote.yaml"
    assert reads == [expected]
    assert "sentinel_remote" not in policy.remotes, "loaded the cwd decoy, not the repo profile"
    assert policy.remote_for().model == load_policy(expected).remote_for().model
    assert _probe_path(env) == expected
    _, check = _check_routing(None, env)
    assert str(expected) in check.detail


def test_missing_explicit_profile_is_loud_everywhere(tmp_path, monkeypatch):
    env = {"HEARTH_ROUTING_YAML": str(tmp_path / "typo.yaml")}
    monkeypatch.setenv("HEARTH_ROUTING_YAML", env["HEARTH_ROUTING_YAML"])

    with pytest.raises(RoutingProfileNotFoundError, match="typo.yaml"):
        load_policy()
    policy_mod.get_policy.cache_clear()
    try:
        with pytest.raises(RoutingProfileNotFoundError):
            policy_mod.get_policy()  # what Router() calls at serve startup
    finally:
        policy_mod.get_policy.cache_clear()

    policy, check = _check_routing(None, env)
    assert policy is None
    assert not check.ok and check.fatal
    assert "does not exist" in check.detail


def test_explicit_path_argument_and_unset_env_keep_the_safe_fallback(tmp_path, monkeypatch):
    # ADR-005 is unchanged where the operator did not name a file via the environment.
    monkeypatch.delenv("HEARTH_ROUTING_YAML", raising=False)
    assert default_policy_path() == _REPO / "config" / "routing.yaml"
    assert load_policy(tmp_path / "absent.yaml").remotes == {}
