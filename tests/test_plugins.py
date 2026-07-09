"""Plugin API tests — entry-point discovery, validation, and graceful skip (Phase 7).

Entry points are simulated by patching :func:`hearth.plugins._iter_entry_points`, so these
tests need nothing installed. They assert the ADR-004 promise extended to third parties: a
plugin backend resolves through the existing selection functions with zero core edits, and
a broken plugin is skipped (logged), never crashing discovery or startup.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import pytest

from hearth import plugins
from hearth.config import Settings
from hearth.memory.embed import select_embedder
from hearth.memory.store import select_vector_store
from hearth.plugins import (
    EMBEDDER_GROUP,
    PROVIDER_GROUP,
    VECTOR_STORE_GROUP,
    discover,
    discover_all,
    load_plugin,
)
from hearth.providers import select_provider


class _FakeEntryPoint:
    """Stand-in for importlib.metadata.EntryPoint: ``load()`` returns a factory."""

    def __init__(self, name: str, value: str, factory) -> None:
        self.name = name
        self.value = value
        self._factory = factory

    def load(self):
        return self._factory


# -- fake backends satisfying each Protocol structurally ------------------------------


@dataclass(frozen=True)
class _Caps:
    chat: bool = True
    embed: bool = False
    stream: bool = True
    adapters: bool = False


@dataclass(frozen=True)
class _Estimate:
    ram_gb: float = 0.0
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _Result:
    text: str
    model: str
    backend: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class PluginProvider:
    name = "pluginbackend"

    def capabilities(self) -> _Caps:
        return _Caps()

    def generate(self, req) -> _Result:
        return _Result(text="[plugin] ok", model=req.model, backend=self.name)

    def stream(self, req) -> Iterator[str]:
        yield "[plugin] ok"

    def footprint(self, model_id: str) -> _Estimate:
        return _Estimate(ram_gb=1.0)


class PluginEmbedder:
    name = "pluginembed"
    dim = 3

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


class PluginStore:
    name = "pluginstore"

    def add(self, collection, chunks, vectors) -> int:
        return len(chunks)

    def query(self, collection, vector, k) -> list:
        return []

    def count(self, collection) -> int:
        return 0


class NotAProvider:
    """Missing generate/footprint — must fail the Protocol check."""

    name = "broken"


def _patch_group(monkeypatch, group: str, entries: list[_FakeEntryPoint]) -> None:
    real = plugins._iter_entry_points

    def fake(g: str):
        return entries if g == group else real(g)

    monkeypatch.setattr(plugins, "_iter_entry_points", fake)


# -- discovery + validation -----------------------------------------------------------


def test_discover_valid_provider(monkeypatch):
    _patch_group(monkeypatch, PROVIDER_GROUP, [_FakeEntryPoint("foo", "m:build", PluginProvider)])
    found = discover(PROVIDER_GROUP)
    assert len(found) == 1
    assert found[0].name == "foo" and found[0].ok
    assert found[0].group == PROVIDER_GROUP


def test_discover_skips_protocol_violator(monkeypatch):
    _patch_group(monkeypatch, PROVIDER_GROUP, [_FakeEntryPoint("bad", "m:X", NotAProvider)])
    found = discover(PROVIDER_GROUP)
    assert len(found) == 1
    assert not found[0].ok
    assert "satisfy" in found[0].detail


def test_discover_skips_import_error(monkeypatch):
    ep = _FakeEntryPoint("explodes", "m:boom", None)
    ep.load = lambda: (_ for _ in ()).throw(ImportError("boom on load"))
    _patch_group(monkeypatch, PROVIDER_GROUP, [ep])
    found = discover(PROVIDER_GROUP)
    assert not found[0].ok
    assert "load error" in found[0].detail


def test_discover_factory_raising_is_skipped_not_crash(monkeypatch):
    def bad_factory():
        raise RuntimeError("construction failed")

    _patch_group(monkeypatch, PROVIDER_GROUP, [_FakeEntryPoint("crash", "m:f", bad_factory)])
    found = discover(PROVIDER_GROUP)  # must not raise
    assert not found[0].ok
    assert "construction failed" in found[0].detail


def test_discover_mixed_good_and_bad_keeps_going(monkeypatch):
    _patch_group(
        monkeypatch,
        PROVIDER_GROUP,
        [
            _FakeEntryPoint("good", "m:build", PluginProvider),
            _FakeEntryPoint("bad", "m:X", NotAProvider),
        ],
    )
    found = discover(PROVIDER_GROUP)
    by_name = {p.name: p for p in found}
    assert by_name["good"].ok
    assert not by_name["bad"].ok


def test_discover_unknown_group_raises():
    with pytest.raises(ValueError):
        discover("hearth.not_a_group")


def test_discover_all_covers_three_groups(monkeypatch):
    _patch_group(monkeypatch, EMBEDDER_GROUP, [_FakeEntryPoint("e", "m:E", PluginEmbedder)])
    found = discover_all()
    assert any(p.group == EMBEDDER_GROUP and p.name == "e" and p.ok for p in found)


# -- load_plugin ----------------------------------------------------------------------


def test_load_plugin_returns_instance(monkeypatch):
    _patch_group(monkeypatch, PROVIDER_GROUP, [_FakeEntryPoint("foo", "m:build", PluginProvider)])
    inst = load_plugin(PROVIDER_GROUP, "foo")
    assert isinstance(inst, PluginProvider)


def test_load_plugin_missing_returns_none(monkeypatch):
    _patch_group(monkeypatch, PROVIDER_GROUP, [])
    assert load_plugin(PROVIDER_GROUP, "absent") is None


def test_load_plugin_bad_returns_none(monkeypatch):
    _patch_group(monkeypatch, PROVIDER_GROUP, [_FakeEntryPoint("bad", "m:X", NotAProvider)])
    assert load_plugin(PROVIDER_GROUP, "bad") is None


# -- wired into the selection functions (zero-core-edit path) -------------------------


def test_select_provider_resolves_plugin_backend(monkeypatch):
    _patch_group(
        monkeypatch, PROVIDER_GROUP, [_FakeEntryPoint("pluginbackend", "m:b", PluginProvider)]
    )
    provider = select_provider(Settings(backend="pluginbackend"))
    assert isinstance(provider, PluginProvider)


def test_select_provider_unknown_backend_raises(monkeypatch):
    _patch_group(monkeypatch, PROVIDER_GROUP, [])
    with pytest.raises(ValueError):
        select_provider(Settings(backend="does-not-exist"))


def test_select_embedder_resolves_plugin(monkeypatch):
    _patch_group(monkeypatch, EMBEDDER_GROUP, [_FakeEntryPoint("pe", "m:e", PluginEmbedder)])
    emb = select_embedder(Settings(embedder="pe"))
    assert isinstance(emb, PluginEmbedder)


def test_select_vector_store_resolves_plugin(monkeypatch):
    _patch_group(monkeypatch, VECTOR_STORE_GROUP, [_FakeEntryPoint("ps", "m:s", PluginStore)])
    store = select_vector_store(Settings(vector_store="ps"))
    assert isinstance(store, PluginStore)
