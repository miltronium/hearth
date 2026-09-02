"""Probe behaviour, exercised against fixture trees rather than the operator's machine.

Each test builds the exact on-disk situation it is about and asserts the probe *measured*
it — not that the probe ran. The cases are chosen from the failures this package exists to
catch: weights that exist but cannot be loaded, a routing file the router silently refused,
a golden set too small to ever license a promotion, and an env var nothing reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hearth.status.probes import (
    KEY_DOCS,
    min_n_for_alpha,
    probe_egress,
    probe_environment,
    probe_learning,
    probe_models,
    probe_staleness,
    probe_tests,
    scan_cache,
    smallest_achievable_p,
)
from hearth.status.report import LEVEL_FAIL, LEVEL_OK, LEVEL_UNVERIFIED, LEVEL_WARN

# ---------------------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------------------


def _fake_repo(source: str = "org/model-a") -> str:
    return (
        "default: org/model-a\n"
        "models:\n"
        f"  - id: {source}\n"
        "    backend: mlx\n"
        f"    source: {source}\n"
        "  - id: echo\n"
        "    backend: echo\n"
        "    source: ''\n"
    )


def _plant_weights(cache: Path, repo: str, *, size: int = 2048) -> Path:
    """Create a hub-layout cache entry with one real weight file."""
    snapshot = cache / f"models--{repo.replace('/', '--')}" / "snapshots" / "deadbeef"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"\0" * size)
    return snapshot


# ---------------------------------------------------------------------------------------
# 1. models
# ---------------------------------------------------------------------------------------


def test_weights_in_the_hub_cache_are_loadable(tmp_path: Path):
    root, hub, home = tmp_path / "repo", tmp_path / "hub", tmp_path / "home"
    (root / "config").mkdir(parents=True)
    (root / "config" / "models.yaml").write_text(_fake_repo())
    _plant_weights(hub, "org/model-a")
    (home / "models").mkdir(parents=True)

    section = probe_models(root=root, home=home, environ={"HF_HUB_CACHE": str(hub)})
    fact = next(f for f in section.facts if f.name == "org/model-a")
    assert fact.level == LEVEL_OK
    assert fact.data["loadable_by_provider"] is True


def test_a_pulled_model_outside_the_hub_cache_is_flagged_invisible(tmp_path: Path):
    """The trap: ``hearth models pull`` writes to ~/.hearth/models, mlx-lm reads elsewhere.

    Downloaded is a configuration; loadable is the outcome. Only the second serves a request,
    so a probe that reported "present" here would be reporting the wrong thing.
    """
    root, hub, home = tmp_path / "repo", tmp_path / "hub", tmp_path / "home"
    (root / "config").mkdir(parents=True)
    (root / "config" / "models.yaml").write_text(_fake_repo())
    hub.mkdir()
    _plant_weights(home / "models", "org/model-a")

    section = probe_models(root=root, home=home, environ={"HF_HUB_CACHE": str(hub)})
    fact = next(f for f in section.facts if f.name == "org/model-a")
    assert fact.level == LEVEL_WARN
    assert "INVISIBLE" in fact.value
    assert fact.data["in_hearth_models"] is True
    assert fact.data["loadable_by_provider"] is False
    assert "HF_HUB_CACHE" in fact.detail


def test_pointing_hf_hub_cache_at_the_hearth_dir_makes_it_loadable(tmp_path: Path):
    root, home = tmp_path / "repo", tmp_path / "home"
    (root / "config").mkdir(parents=True)
    (root / "config" / "models.yaml").write_text(_fake_repo())
    _plant_weights(home / "models", "org/model-a")

    section = probe_models(
        root=root, home=home, environ={"HF_HUB_CACHE": str(home / "models")}
    )
    fact = next(f for f in section.facts if f.name == "org/model-a")
    assert fact.level == LEVEL_OK
    assert fact.data["loadable_by_provider"] is True


def test_registered_model_with_no_weights_anywhere(tmp_path: Path):
    root, hub, home = tmp_path / "repo", tmp_path / "hub", tmp_path / "home"
    (root / "config").mkdir(parents=True)
    (root / "config" / "models.yaml").write_text(_fake_repo())
    hub.mkdir()
    (home / "models").mkdir(parents=True)

    section = probe_models(root=root, home=home, environ={"HF_HUB_CACHE": str(hub)})
    fact = next(f for f in section.facts if f.name == "org/model-a")
    assert fact.level == LEVEL_WARN
    assert "NO weights" in fact.value


def test_unregistered_weights_are_surfaced(tmp_path: Path):
    root, hub, home = tmp_path / "repo", tmp_path / "hub", tmp_path / "home"
    (root / "config").mkdir(parents=True)
    (root / "config" / "models.yaml").write_text(_fake_repo())
    _plant_weights(hub, "org/model-a")
    _plant_weights(hub, "someone/stray-model")
    (home / "models").mkdir(parents=True)

    section = probe_models(root=root, home=home, environ={"HF_HUB_CACHE": str(hub)})
    fact = next(f for f in section.facts if f.name == "someone/stray-model")
    assert fact.level == LEVEL_WARN
    assert "UNREGISTERED" in fact.value


def test_a_dangling_symlink_does_not_count_as_present(tmp_path: Path):
    """A directory named after a model is configuration; a resolvable weight is the outcome."""
    cache = tmp_path / "hub"
    snapshot = cache / "models--org--ghost" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").symlink_to(cache / "blobs" / "missing")

    found = scan_cache(cache)
    assert "org/ghost" in found
    assert found["org/ghost"].present is False


def test_interrupted_downloads_are_reported(tmp_path: Path):
    root, hub, home = tmp_path / "repo", tmp_path / "hub", tmp_path / "home"
    (root / "config").mkdir(parents=True)
    (root / "config" / "models.yaml").write_text(_fake_repo())
    _plant_weights(hub, "org/model-a")
    blobs = hub / "models--org--model-a" / "blobs"
    blobs.mkdir()
    (blobs / "0123.incomplete").write_bytes(b"partial")
    (home / "models").mkdir(parents=True)

    section = probe_models(root=root, home=home, environ={"HF_HUB_CACHE": str(hub)})
    fact = next(f for f in section.facts if f.name == "partial_downloads")
    assert fact.level == LEVEL_WARN


def test_an_unreadable_registry_is_unverified_not_assumed_empty(tmp_path: Path):
    root, home = tmp_path / "repo", tmp_path / "home"
    (root / "config").mkdir(parents=True)  # no models.yaml at all
    (home / "models").mkdir(parents=True)

    section = probe_models(root=root, home=home, environ={"HF_HUB_CACHE": str(tmp_path / "hub")})
    fact = next(f for f in section.facts if f.name == "registry")
    assert fact.level == LEVEL_UNVERIFIED


# ---------------------------------------------------------------------------------------
# 2. egress
# ---------------------------------------------------------------------------------------

_NO_EGRESS = (
    "defaults: {local_model: auto, remote: none, remote_budget_tokens_per_day: 0}\n"
    "classes:\n"
    "  chat: {backend: local, escalate: never}\n"
    "  code: {backend: local, escalate: never}\n"
    "remotes: {}\n"
)

_EGRESS = (
    "defaults: {local_model: auto, remote: default}\n"
    "classes:\n"
    "  chat: {backend: local, escalate: on_low_confidence, threshold: 0.6}\n"
    "  reason: {backend: remote, escalate: always}\n"
    "remotes:\n"
    "  default: {protocol: anthropic, model: claude-opus-4-8}\n"
)


def _egress_section(tmp_path: Path, files: dict[str, str], **kwargs):
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    for name, body in files.items():
        (root / "config" / name).write_text(body)
    return probe_egress(root=root, **kwargs)


def test_a_zero_remote_profile_reads_as_no_egress(tmp_path: Path):
    section = _egress_section(tmp_path, {"routing.private.yaml": _NO_EGRESS}, environ={})
    fact = next(f for f in section.facts if f.name.endswith("routing.private.yaml"))
    assert fact.data["no_egress"] is True
    assert fact.data["default_remote_resolves"] is False
    assert "NO EGRESS" in fact.value


def test_a_profile_with_a_remote_reads_as_permitting_egress(tmp_path: Path):
    section = _egress_section(tmp_path, {"routing.yaml": _EGRESS}, environ={})
    fact = next(f for f in section.facts if f.name.endswith("routing.yaml"))
    assert fact.data["no_egress"] is False
    assert fact.data["remotes"] == ["default"]
    assert fact.data["remote_classes"] == ["reason"]
    assert "chat" in fact.data["escalating_classes"]


def test_a_broken_profile_is_reported_as_drift_not_as_no_egress(tmp_path: Path):
    """The trap this whole package is about, in miniature.

    ``load_policy`` swallows an invalid file and substitutes safe built-in defaults — which
    are themselves no-egress. So a *broken* profile resolves to a *clean-looking* policy. A
    probe that read only the resolved object would print "NO EGRESS" for a file the router
    never actually honoured, which is a green light with no measurement behind it.
    """
    broken = (
        "defaults: {local_model: auto}\n"
        "classes:\n"
        "  not_a_real_task_class: {backend: local, escalate: never}\n"
        "remotes: {}\n"
    )
    section = _egress_section(tmp_path, {"routing.broken.yaml": broken}, environ={})
    fact = next(f for f in section.facts if f.name.endswith("routing.broken.yaml"))
    assert fact.level == LEVEL_WARN
    assert fact.data["drift"], "the loader's silent fallback must be reported, not laundered"
    assert "not_a_real_task_class" in " ".join(fact.data["drift"])


def test_no_no_egress_profile_available_is_a_failure(tmp_path: Path):
    section = _egress_section(tmp_path, {"routing.yaml": _EGRESS}, environ={})
    fact = next(f for f in section.facts if f.name == "no_egress_profile_available")
    assert fact.level == LEVEL_FAIL


def test_the_active_profile_comes_from_the_environment(tmp_path: Path):
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    (root / "config" / "routing.private.yaml").write_text(_NO_EGRESS)
    chosen = root / "config" / "routing.private.yaml"
    section = probe_egress(root=root, environ={"HEARTH_ROUTING_YAML": str(chosen)})
    fact = next(f for f in section.facts if f.name == "active_profile")
    assert fact.data == {"path": str(chosen), "from_env": True, "exists": True}


def test_an_active_profile_that_does_not_exist_is_a_warning(tmp_path: Path):
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    section = probe_egress(root=root, environ={"HEARTH_ROUTING_YAML": "/nope/routing.yaml"})
    fact = next(f for f in section.facts if f.name == "active_profile")
    assert fact.level == LEVEL_WARN


def test_egress_section_states_it_is_not_machine_containment(tmp_path: Path):
    section = _egress_section(tmp_path, {"routing.private.yaml": _NO_EGRESS}, environ={})
    joined = " ".join(section.limits)
    assert "ROUTER'S guarantee" in joined
    assert "containment" in joined


# ---------------------------------------------------------------------------------------
# 3. learning + the minimum detectable effect
# ---------------------------------------------------------------------------------------


def test_min_n_for_alpha_is_five_at_the_default_bar():
    assert min_n_for_alpha(0.05) == 5
    assert smallest_achievable_p(4) == pytest.approx(0.0625)  # > 0.05: can never gate
    assert smallest_achievable_p(5) == pytest.approx(0.03125)  # <= 0.05: can, only just


def test_the_mde_math_matches_the_gate_it_describes():
    """If the gate's statistics move, this report must move with them — or fail loudly."""
    stats = pytest.importorskip("hearth.training.stats")
    for n in range(0, 12):
        assert smallest_achievable_p(n) == stats.smallest_achievable_p(n)
    for alpha in (0.1, 0.05, 0.01):
        assert min_n_for_alpha(alpha) == stats.min_n_for_alpha(alpha)


def _learning_section(tmp_path: Path, golden_rows: int, **kwargs):
    root, home = tmp_path / "repo", tmp_path / "home"
    (root / "data").mkdir(parents=True)
    home.mkdir()
    body = "".join(
        json.dumps({"prompt": f"p{i}", "expected": f"e{i}"}) + "\n" for i in range(golden_rows)
    )
    (root / "data" / "task_golden.jsonl").write_text(body)
    return probe_learning(root=root, home=home, environ={}, **kwargs)


def test_a_golden_set_below_five_can_never_gate(tmp_path: Path):
    section = _learning_section(tmp_path, 4)
    fact = next(f for f in section.facts if f.name.endswith("task_golden.jsonl"))
    assert fact.level == LEVEL_FAIL
    assert "CANNOT EVER GATE" in fact.value
    assert fact.data["can_ever_gate"] is False
    assert fact.data["min_discordant_pairs_required"] == 5


def test_a_golden_set_of_exactly_five_gates_only_on_a_clean_sweep(tmp_path: Path):
    section = _learning_section(tmp_path, 5)
    fact = next(f for f in section.facts if f.name.endswith("task_golden.jsonl"))
    assert fact.level == LEVEL_WARN
    assert fact.data["can_ever_gate"] is True
    assert fact.data["smallest_achievable_p"] == pytest.approx(0.03125)


def test_a_comfortable_golden_set_reads_ok(tmp_path: Path):
    section = _learning_section(tmp_path, 20)
    fact = next(f for f in section.facts if f.name.endswith("task_golden.jsonl"))
    assert fact.level == LEVEL_OK


def test_a_stricter_alpha_raises_the_bar(tmp_path: Path):
    section = _learning_section(tmp_path, 6, alpha=0.01)
    fact = next(f for f in section.facts if f.name.endswith("task_golden.jsonl"))
    assert fact.data["min_discordant_pairs_required"] == 7
    assert fact.data["can_ever_gate"] is False


def test_malformed_golden_lines_are_counted(tmp_path: Path):
    root, home = tmp_path / "repo", tmp_path / "home"
    (root / "data").mkdir(parents=True)
    home.mkdir()
    rows = [json.dumps({"prompt": "a", "expected": "b"})] * 20 + ["{not json}"]
    (root / "data" / "task_golden.jsonl").write_text("\n".join(rows) + "\n")

    section = probe_learning(root=root, home=home, environ={})
    fact = next(f for f in section.facts if f.name.endswith("task_golden.jsonl"))
    assert fact.data["malformed"] == 1
    assert fact.level == LEVEL_WARN


def test_a_promoted_adapter_without_a_significance_proof_is_flagged(tmp_path: Path):
    root, home = tmp_path / "repo", tmp_path / "home"
    (root / "data").mkdir(parents=True)
    adapters = home / "train" / "run1" / "adapters"
    adapters.mkdir(parents=True)
    (home / "adapters.json").write_text(
        json.dumps(
            {
                "adapters": [
                    {
                        "id": "a-1",
                        "task": "classify",
                        "status": "promoted",
                        "adapter_path": str(adapters),
                        "promotion_proof": {"gate_passed": True, "candidate_score": 1.0},
                    }
                ]
            }
        )
    )
    section = probe_learning(root=root, home=home, environ={})
    fact = next(f for f in section.facts if f.name == "adapter:a-1")
    assert fact.level == LEVEL_WARN
    assert fact.data["has_significance_proof"] is False
    assert "without a significance proof" in fact.detail.lower()


def test_an_adapter_whose_weights_vanished_is_flagged(tmp_path: Path):
    root, home = tmp_path / "repo", tmp_path / "home"
    (root / "data").mkdir(parents=True)
    home.mkdir()
    (home / "adapters.json").write_text(
        json.dumps(
            {
                "adapters": [
                    {
                        "id": "a-2",
                        "task": "extract",
                        "status": "candidate",
                        "adapter_path": str(home / "gone"),
                        "promotion_proof": {},
                    }
                ]
            }
        )
    )
    section = probe_learning(root=root, home=home, environ={})
    fact = next(f for f in section.facts if f.name == "adapter:a-2")
    assert fact.level == LEVEL_WARN
    assert fact.data["weights_present"] is False


def test_a_missing_adapter_registry_is_unverified(tmp_path: Path):
    section = _learning_section(tmp_path, 8)
    fact = next(f for f in section.facts if f.name == "adapters")
    assert fact.level == LEVEL_UNVERIFIED


# ---------------------------------------------------------------------------------------
# 4. tests
# ---------------------------------------------------------------------------------------


def test_test_probe_counts_declared_functions_without_running_them(tmp_path: Path):
    root = tmp_path / "repo"
    (root / "tests").mkdir(parents=True)
    (root / "tests" / "test_a.py").write_text(
        "def test_one():\n    assert True\n\nasync def test_two():\n    assert True\n"
    )
    (root / "tests" / "test_b.py").write_text("def test_three():\n    raise SystemExit(1)\n")

    section = probe_tests(root=root)
    declared = next(f for f in section.facts if f.name == "declared")
    assert declared.data == {"files": 2, "test_functions": 3}
    # The suite in the fixture would fail if run; the probe must not have run it.
    passing = next(f for f in section.facts if f.name == "passing_now")
    assert passing.level == LEVEL_UNVERIFIED


def test_the_pytest_cache_is_reported_as_a_cache(tmp_path: Path):
    root = tmp_path / "repo"
    cache = root / ".pytest_cache" / "v" / "cache"
    cache.mkdir(parents=True)
    (root / "tests").mkdir()
    (cache / "nodeids").write_text(json.dumps(["tests/test_a.py::test_one"] * 7))
    (cache / "lastfailed").write_text(json.dumps({"tests/test_a.py::test_one": True}))

    section = probe_tests(root=root)
    fact = next(f for f in section.facts if f.name == "last_recorded_run")
    assert fact.data["collected"] == 7
    assert fact.data["failed"] == 1
    assert fact.level == LEVEL_WARN
    assert "not necessarily of the current working tree" in fact.detail


# ---------------------------------------------------------------------------------------
# 5. environment
# ---------------------------------------------------------------------------------------


def test_an_unread_hearth_env_var_is_flagged_as_silently_ignored():
    """The HEARTH_MODEL trap: the real setting is HEARTH_DEFAULT_MODEL, and Settings ignores
    the rest — so an operator can set a variable, run a benchmark, and report a number for a
    model that never loaded."""
    section = probe_environment(environ={"HEARTH_MODEL": "some/model"})
    fact = next(f for f in section.facts if f.name == "hearth_env")
    assert fact.level == LEVEL_WARN
    assert fact.data["ignored"] == ["HEARTH_MODEL"]
    assert "SILENTLY IGNORED" in fact.detail


def test_a_real_settings_var_is_not_flagged():
    section = probe_environment(environ={"HEARTH_DEFAULT_MODEL": "some/model"})
    fact = next(f for f in section.facts if f.name == "hearth_env")
    assert fact.level == LEVEL_OK
    assert fact.data["ignored"] == []


def test_env_vars_read_outside_settings_are_recognised():
    section = probe_environment(environ={"HEARTH_ROUTING_YAML": "config/routing.private.yaml"})
    fact = next(f for f in section.facts if f.name == "hearth_env")
    assert fact.level == LEVEL_OK


def test_environment_reports_the_gpu_ceiling_or_says_it_could_not():
    section = probe_environment(environ={})
    fact = next(f for f in section.facts if f.name == "gpu_working_set")
    if fact.level == LEVEL_UNVERIFIED:
        assert "mlx is not importable" in fact.detail
    else:
        ws = fact.data["max_recommended_working_set_size"]
        assert ws > 0
        # The whole reason this field exists: the driver's ceiling is below advertised RAM.
        assert ws <= fact.data["memory_size"]


# ---------------------------------------------------------------------------------------
# 6. staleness
# ---------------------------------------------------------------------------------------


def test_staleness_is_unverified_outside_a_git_repo(tmp_path: Path):
    section = probe_staleness(root=tmp_path)
    assert [f.name for f in section.facts] == ["git"]
    assert section.facts[0].level == LEVEL_UNVERIFIED


def test_a_missing_key_doc_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo_root = Path(__file__).resolve().parent.parent
    section = probe_staleness(root=repo_root, docs=("docs/definitely-not-a-real-doc.md",))
    fact = next(f for f in section.facts if f.name.endswith("not-a-real-doc.md"))
    assert fact.level == LEVEL_WARN
    assert fact.value == "MISSING"


def test_staleness_measures_real_history_for_a_committed_doc():
    repo_root = Path(__file__).resolve().parent.parent
    section = probe_staleness(root=repo_root, docs=("README.md",))
    head = next(f for f in section.facts if f.name == "head")
    assert head.data["sha"], "HEAD should be measurable in this checkout"
    readme = next(f for f in section.facts if f.name == "README.md")
    if readme.level != LEVEL_UNVERIFIED:  # committed at least once
        assert readme.data["commits_since"] is not None
        assert readme.data["last_commit_date"].startswith("20")


def test_key_docs_are_the_memory_docs_this_package_replaces():
    assert "docs/RESULTS.md" in KEY_DOCS
    assert "docs/cmux/HANDOFF.md" in KEY_DOCS
