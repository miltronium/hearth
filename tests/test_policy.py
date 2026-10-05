"""Routing policy tests (ADR-005): load, validation, and fallback-on-bad-yaml."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hearth.registry import get_registry
from hearth.router.policy import RoutingPolicyError, _parse, load_policy

_CONFIG = Path(__file__).resolve().parent.parent / "config"


def _write(tmp_path, text: str):
    p = tmp_path / "routing.yaml"
    p.write_text(text)
    return p


def test_bundled_default_is_no_egress(monkeypatch):
    """The profile HEARTH uses when nothing is selected cannot send a task anywhere.

    Asserted on the OUTCOME — no remote exists, and no task class can reach the remote
    backend — not on the file's name or a comment that says "no-egress".
    """
    from hearth.router.classify import TASK_CLASSES
    from hearth.router.policy import default_policy_path

    monkeypatch.delenv("HEARTH_ROUTING_YAML", raising=False)
    assert default_policy_path() == _CONFIG / "routing.yaml"
    policy = load_policy()
    assert policy.remotes == {}
    assert policy.remote_for() is None
    assert policy.defaults.remote_budget_tokens_per_day == 0
    for task_class in TASK_CLASSES:
        rule = policy.rule_for(task_class)
        assert (rule.backend, rule.escalate) == ("local", "never"), task_class


def test_bundled_remote_profile_is_the_opt_in_escalation_table():
    # config/routing.remote.yaml carries the ARCHITECTURE §3 escalation table, opt-in only.
    policy = load_policy(_CONFIG / "routing.remote.yaml")
    assert policy.rule_for("reason").backend == "remote"
    assert policy.rule_for("reason").escalate == "always"
    assert policy.rule_for("summarize").escalate == "never"
    assert policy.rule_for("code").threshold == 0.7
    remote = policy.remote_for()
    assert remote is not None
    assert remote.protocol == "anthropic"
    assert remote.model == "claude-opus-4-8"


def test_valid_custom_policy(tmp_path):
    path = _write(
        tmp_path,
        """
        defaults:
          local_model: my-local
          remote: lan
          remote_budget_tokens_per_day: 500
        classes:
          chat: { backend: local, escalate: on_low_confidence, threshold: 0.5 }
          reason: { backend: remote, escalate: always }
        remotes:
          lan:
            protocol: openai
            model: llama-70b
            base_url: http://host:8000/v1
            api_key_env: LAN_KEY
        """,
    )
    # defaults.local_model is a model rung, validated like a class rung (B-065).
    policy = load_policy(path, known_models={"my-local"})
    assert policy.defaults.local_model == "my-local"
    assert policy.defaults.remote_budget_tokens_per_day == 500
    assert policy.rule_for("chat").threshold == 0.5
    lan = policy.remote_for("lan")
    assert lan.protocol == "openai"
    assert lan.base_url == "http://host:8000/v1"
    assert lan.api_key_env == "LAN_KEY"


def test_missing_file_falls_back_to_safe_defaults(tmp_path):
    policy = load_policy(tmp_path / "does-not-exist.yaml")
    # Safe defaults: every known class local, never escalate, no remotes.
    assert policy.rule_for("reason").backend == "local"
    assert policy.rule_for("reason").escalate == "never"
    assert policy.remotes == {}


def test_invalid_backend_falls_back(tmp_path):
    path = _write(
        tmp_path,
        """
        classes:
          chat: { backend: bogus, escalate: never }
        """,
    )
    policy = load_policy(path)
    # Validation failed -> safe defaults, service stays up (ADR-005).
    assert policy.rule_for("chat").backend == "local"


def test_unknown_class_falls_back(tmp_path):
    path = _write(tmp_path, "classes:\n  wat: { backend: local }\n")
    policy = load_policy(path)
    assert policy.rule_for("chat").escalate == "never"


def test_malformed_yaml_falls_back(tmp_path):
    path = _write(tmp_path, "classes: [this: is: not valid")
    policy = load_policy(path)
    assert policy.rule_for("summarize").backend == "local"


# -- per-class local_model: the local model ladder (ClassRule.local_model) ----------------

_KNOWN = {"small-3b", "big-14b"}


def test_per_class_local_model_parses(tmp_path):
    path = _write(
        tmp_path,
        """
        defaults:
          local_model: big-14b
        classes:
          classify:  { backend: local, escalate: never, local_model: small-3b }
          summarize: { backend: local, escalate: never, local_model: big-14b }
        """,
    )
    policy = load_policy(path, known_models=_KNOWN)
    assert policy.rule_for("classify").local_model == "small-3b"
    assert policy.rule_for("summarize").local_model == "big-14b"


def test_local_model_defaults_to_none(tmp_path):
    """Backwards compatibility: a rule without the field is indistinguishable from before."""
    path = _write(tmp_path, "classes:\n  chat: { backend: local, escalate: never }\n")
    policy = load_policy(path, known_models=_KNOWN)
    assert policy.rule_for("chat").local_model is None
    # An unspecified class gets the safe default rule, also unpinned.
    assert policy.rule_for("summarize").local_model is None


def test_unknown_local_model_is_rejected_at_load(tmp_path):
    """A model id nobody can serve is a config bug — caught here, not at generation time."""
    path = _write(
        tmp_path,
        "classes:\n  classify: { backend: local, local_model: mlx-community/typo-7B }\n",
    )
    with pytest.raises(ValueError, match="not in the model registry"):
        _parse(yaml.safe_load(path.read_text()), known_models=_KNOWN)
    # B-065: NOT degraded to the safe defaults at the load_policy boundary. The fallback
    # silently replaced the whole ladder with the registry default; a bad rung is loud.
    with pytest.raises(RoutingPolicyError, match="mlx-community/typo-7B"):
        load_policy(path, known_models=_KNOWN)


def test_local_model_auto_is_allowed_and_unvalidated(tmp_path):
    """"auto" means "fall through to defaults" — it names no model, so nothing is looked up."""
    path = _write(tmp_path, "classes:\n  classify: { backend: local, local_model: auto }\n")
    policy = load_policy(path, known_models=_KNOWN)
    assert policy.rule_for("classify").local_model == "auto"


def test_local_model_unchecked_when_registry_unavailable(tmp_path):
    """An unreadable registry skips the check rather than vetoing a valid routing config."""
    path = _write(tmp_path, "classes:\n  classify: { local_model: whatever-id }\n")
    policy = _parse(yaml.safe_load(path.read_text()), known_models=None)
    assert policy.rule_for("classify").local_model == "whatever-id"


def test_bundled_profiles_are_unpinned():
    """The shipped default/private profiles predate the field and must stay unpinned."""
    for name in ("routing.yaml", "routing.remote.yaml", "routing.private.yaml"):
        policy = load_policy(_CONFIG / name)
        assert all(r.local_model is None for r in policy.classes.values()), name


def test_bundled_finance_profile_is_a_sealed_two_tier_ladder():
    """config/routing.finance.yaml: real registry ids, tiered, and structurally no-egress."""
    policy = load_policy(_CONFIG / "routing.finance.yaml")
    tier1 = "mlx-community/Qwen2.5-3B-Instruct-4bit"
    tier2 = "mlx-community/Qwen2.5-14B-Instruct-4bit"
    assert policy.rule_for("classify").local_model == tier1
    assert policy.rule_for("extract").local_model == tier1
    assert policy.rule_for("summarize").local_model == tier2
    assert policy.rule_for("draft").local_model == tier2
    assert policy.rule_for("reason").local_model == tier2
    assert policy.rule_for("chat").local_model == tier2
    # Every pinned id is really servable (the profile loaded against the real registry).
    known = {e.id for e in get_registry().list()}
    assert {r.local_model for r in policy.classes.values()} <= known
    # The no-egress property scripts/hearth_private.sh verifies must hold here too.
    assert policy.remotes == {}
    assert policy.remote_for() is None
    assert all(
        r.backend == "local" and r.escalate == "never" for r in policy.classes.values()
    )
