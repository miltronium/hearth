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
