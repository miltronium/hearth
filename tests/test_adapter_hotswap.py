"""Adapter hot-swap plumbing tests — the router resolves an adapter id to a path and
passes it to the provider; MLXProvider caches per-adapter loads. All with fakes; no MLX."""

from __future__ import annotations

from hearth.observability.budget import BudgetAccountant
from hearth.observability.metrics import MetricsStore
from hearth.providers.base import Capabilities, GenRequest, GenResult, Message
from hearth.registry import AdapterStore
from hearth.router import Router, RoutingPolicy
from hearth.router.classify import TASK_CLASSES
from hearth.router.policy import ClassRule, Defaults


class RecordingProvider:
    """A fake local provider that records the adapter path each request carried."""

    name = "fake"

    def __init__(self) -> None:
        self.seen_adapters: list[str | None] = []

    def capabilities(self) -> Capabilities:
        return Capabilities(chat=True, adapters=True)

    def generate(self, req: GenRequest) -> GenResult:
        self.seen_adapters.append(req.adapter)
        return GenResult(text="ok", model=req.model, backend=self.name)

    def stream(self, req):  # pragma: no cover - not exercised here
        yield "ok"

    def footprint(self, model_id):  # pragma: no cover
        from hearth.providers.base import ResourceEstimate

        return ResourceEstimate()


def _local_policy() -> RoutingPolicy:
    return RoutingPolicy(
        defaults=Defaults(),
        classes={c: ClassRule(backend="local", escalate="never") for c in TASK_CLASSES},
        remotes={},
    )


def _router(provider, store) -> Router:
    return Router(
        local_provider=provider,
        policy=_local_policy(),
        budget=BudgetAccountant(1000),
        metrics=MetricsStore(),
        adapters=store,
    )


def _req():
    return GenRequest(messages=[Message(role="user", content="extract the names")], model="auto")


def test_explicit_candidate_adapter_is_resolved_and_passed(tmp_path):
    store = AdapterStore(path=tmp_path / "adapters.json")
    store.register(
        "extract-1", base_model="b", task="extract", train_run_id="r", adapter_path="/a/extract-1"
    )
    provider = RecordingProvider()
    router = _router(provider, store)

    router.route(_req(), intent="extract", adapter="extract-1")
    # Candidate served behind the A/B flag: its path reached the provider.
    assert provider.seen_adapters == ["/a/extract-1"]


def test_promoted_adapter_serves_by_default_for_its_class(tmp_path):
    store = AdapterStore(path=tmp_path / "adapters.json")
    store.register(
        "extract-1", base_model="b", task="extract", train_run_id="r", adapter_path="/a/extract-1"
    )
    store.promote("extract-1", gate_passed=True)
    provider = RecordingProvider()
    router = _router(provider, store)

    # No explicit adapter → the promoted adapter for the classified task is selected.
    router.route(_req(), intent="extract")
    assert provider.seen_adapters == ["/a/extract-1"]


def test_no_adapter_serves_base_weights(tmp_path):
    store = AdapterStore(path=tmp_path / "adapters.json")
    provider = RecordingProvider()
    router = _router(provider, store)
    router.route(_req(), intent="extract")
    assert provider.seen_adapters == [None]


def test_unresolvable_adapter_degrades_to_base(tmp_path):
    store = AdapterStore(path=tmp_path / "adapters.json")
    provider = RecordingProvider()
    router = _router(provider, store)
    # Requesting an unknown id must not fail the request — it serves base weights.
    router.route(_req(), intent="extract", adapter="does-not-exist")
    assert provider.seen_adapters == [None]


def test_mlx_provider_caches_per_adapter_loads():
    """MLXProvider loads each distinct adapter path once and caches it (fake mlx_lm.load)."""
    from hearth.providers import mlx as mlx_mod

    calls: list[str | None] = []

    class FakeTokenizer:
        def encode(self, text):
            return list(text)

        def apply_chat_template(self, chat, tokenize, add_generation_prompt):
            return "PROMPT"

    def fake_load(model_id, **kwargs):
        calls.append(kwargs.get("adapter_path"))
        return object(), FakeTokenizer()

    provider = mlx_mod.MLXProvider("org/model")
    # Bypass availability + heavy import by loading variants directly through the cache.
    provider._cache[""] = (object(), FakeTokenizer())
    # First load of an adapter path calls through; the second is a cache hit.
    import importlib.machinery
    import types

    fake_mlx_lm = types.ModuleType("mlx_lm")
    fake_mlx_lm.load = fake_load
    # A real spec so importlib.util.find_spec("mlx_lm") reports it as available.
    fake_mlx_lm.__spec__ = importlib.machinery.ModuleSpec("mlx_lm", loader=None)
    import sys

    sys.modules["mlx_lm"] = fake_mlx_lm
    try:
        provider._load_variant("/a/one")
        provider._load_variant("/a/one")
        provider._load_variant("/a/two")
    finally:
        del sys.modules["mlx_lm"]
    # Two distinct paths => two loads; the repeat was cached.
    assert calls == ["/a/one", "/a/two"]
