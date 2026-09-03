"""Registry tests — the catalog loads, lists, and resolves the default model."""

from __future__ import annotations

from hearth.registry import ModelEntry, get_registry, load_registry


def test_bundled_registry_loads():
    reg = get_registry()
    entries = reg.list()
    assert len(entries) >= 3
    assert all(isinstance(e, ModelEntry) for e in entries)
    ids = {e.id for e in entries}
    assert "echo" in ids
    assert reg.default_id in ids


def test_resolve_auto_and_explicit():
    reg = get_registry()
    assert reg.resolve("auto").id == reg.default_id
    assert reg.resolve("").id == reg.default_id
    assert reg.resolve("echo").id == "echo"


def test_load_registry_from_file(tmp_path):
    yaml_path = tmp_path / "models.yaml"
    yaml_path.write_text(
        "default: a\n"
        "models:\n"
        "  - {id: a, backend: mlx, quant: 4bit, context: 4096, ram_gb: 2.0, "
        "capabilities: [chat], source: org/a}\n"
        "  - {id: b, backend: echo, quant: none, context: 8192, ram_gb: 0.0, "
        "capabilities: [chat], source: ''}\n"
    )
    reg = load_registry(yaml_path)
    assert [e.id for e in reg.list()] == ["a", "b"]
    assert reg.default_id == "a"
    assert reg.get("b").backend == "echo"
    assert reg.get("missing") is None


def test_the_env_override_reaches_the_registry(monkeypatch):
    """HEARTH_DEFAULT_MODEL must decide what the registry serves by default.

    It reached Settings.default_model but not the registry, and every caller that asked the
    *registry* — the CLI agent command among them — got the catalog's YAML default instead.
    So naming a model produced output from a different one, with nothing reporting the
    substitution. Two notions of "the default model" that disagree is the same shape as the
    other silent-substitution bugs in this codebase.
    """
    from hearth.registry import load_registry

    registry = load_registry()
    catalog_default = registry.default_id
    other = next(e.id for e in registry.list() if e.id != catalog_default and e.backend != "echo")

    monkeypatch.setenv("HEARTH_DEFAULT_MODEL", other)
    assert load_registry().default_id == other


def test_an_override_naming_an_unregistered_model_is_ignored(monkeypatch):
    """A typo must not take the catalog's default away.

    Obeying an unknown id would defer the failure to resolve(), far from its cause; ignoring
    it keeps the daemon serving something real and lets the model-presence probe in
    scripts/hearth_status.py be the place that reports the mistake.
    """
    from hearth.registry import load_registry

    catalog_default = load_registry().default_id
    monkeypatch.setenv("HEARTH_DEFAULT_MODEL", "not/a-real-model")
    assert load_registry().default_id == catalog_default
