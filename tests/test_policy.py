"""Routing policy tests (ADR-005): load, validation, and fallback-on-bad-yaml."""

from __future__ import annotations

from hearth.router.policy import load_policy


def _write(tmp_path, text: str):
    p = tmp_path / "routing.yaml"
    p.write_text(text)
    return p


def test_loads_bundled_default():
    # The shipped config/routing.yaml parses and seeds the ARCHITECTURE §3 table.
    policy = load_policy()
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
    policy = load_policy(path)
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
