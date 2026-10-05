"""The response names the adapter that SERVED, never just the one requested (B-034).

An explicitly requested unknown adapter used to be answered by base weights with
``hearth.adapter: "<requested>"`` and a 200 — so an A/B comparison could compare base weights
against themselves and believe an adapter had run (CLAUDE.md §3). Now an explicit unknown
adapter is a 404 before anything runs, like an unknown model, and the telemetry is derived
from the generation: the fake local below tags its output with the adapter path it was
actually given, so "served by adapter X" is checked against the text, not against itself.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from hearth.config import Settings
from hearth.gateway import create_app
from hearth.observability.budget import BudgetAccountant
from hearth.observability.metrics import MetricsStore
from hearth.providers.base import Capabilities, GenRequest, GenResult, ResourceEstimate
from hearth.providers.echo import EchoProvider
from hearth.registry import AdapterStore
from hearth.router import Router, RoutingPolicy
from hearth.router.classify import TASK_CLASSES
from hearth.router.policy import ClassRule, Defaults

EXTRACT = "extract the names from this list"  # intent=extract is passed explicitly anyway


class TaggingLocal:
    """A local backend that applies adapters and says which one in every answer."""

    name = "mlx"

    def __init__(self, broken: set[str] | None = None) -> None:
        self.calls: list[str | None] = []
        self.broken = broken or set()

    def capabilities(self) -> Capabilities:
        return Capabilities(chat=True, stream=True, adapters=True)

    def footprint(self, model_id: str) -> ResourceEstimate:
        return ResourceEstimate()

    def _text(self, req: GenRequest) -> str:
        self.calls.append(req.adapter)
        if req.adapter in self.broken:
            raise RuntimeError("bad adapter shape")
        return f"<<adapter={req.adapter}>>"

    def generate(self, req: GenRequest) -> GenResult:
        return GenResult(text=self._text(req), model=req.model, backend=self.name)

    def stream(self, req: GenRequest):
        yield self._text(req)


def _store(tmp_path) -> AdapterStore:
    store = AdapterStore(path=tmp_path / "adapters.json")
    store.register("ab-1", base_model="", task="draft", train_run_id="r", adapter_path="/a/ab-1")
    store.register(
        "extract-p", base_model="", task="extract", train_run_id="r", adapter_path="/a/extract-p"
    )
    store.promote("extract-p", gate_passed=True)
    return store


def _client(tmp_path, local, store) -> TestClient:
    metrics = MetricsStore()
    router = Router(
        local_provider=local,
        policy=RoutingPolicy(
            defaults=Defaults(),
            classes={c: ClassRule(backend="local", escalate="never") for c in TASK_CLASSES},
            remotes={},
        ),
        budget=BudgetAccountant(1000),
        metrics=metrics,
        adapters=store,
    )
    settings = Settings(backend="echo", home=tmp_path / ".hearth", require_auth=False)
    return TestClient(create_app(provider=local, settings=settings, router=router,
                                 metrics=metrics))


def _ask(client, *, stream: bool, adapter: str | None = None, intent: str = "extract"):
    hearth = {"intent": intent}
    if adapter is not None:
        hearth["adapter"] = adapter
    r = client.post(
        "/v1/chat/completions",
        json={"stream": stream, "messages": [{"role": "user", "content": EXTRACT}],
              "hearth": hearth},
    )
    if not stream or r.status_code != 200:
        return r, r.json()
    events = [
        json.loads(line[6:]) for line in r.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    text = "".join(
        e["choices"][0]["delta"].get("content") or "" for e in events if e.get("choices")
    )
    final = next(e for e in reversed(events) if e.get("hearth"))
    return r, {"text": text, "hearth": final["hearth"]}


def _text_and_adapter(body: dict, stream: bool) -> tuple[str, str | None]:
    if stream:
        return body["text"], body["hearth"].get("adapter")
    return body["choices"][0]["message"]["content"], body["hearth"].get("adapter")


@pytest.mark.parametrize("stream", [False, True])
def test_an_explicit_unknown_adapter_is_a_404_and_nothing_runs(tmp_path, stream):
    local = TaggingLocal()
    client = _client(tmp_path, local, _store(tmp_path))
    r, body = _ask(client, stream=stream, adapter="no-such-adapter")
    assert r.status_code == 404
    assert body["error"]["code"] == "adapter_not_found"
    assert body["error"]["param"] == "hearth.adapter"
    assert "no-such-adapter" in body["error"]["message"]
    assert local.calls == []  # base weights did not quietly answer it


@pytest.mark.parametrize("stream", [False, True])
def test_an_explicit_known_adapter_is_reported_because_it_ran(tmp_path, stream):
    local = TaggingLocal()
    r, body = _ask(_client(tmp_path, local, _store(tmp_path)), stream=stream, adapter="ab-1")
    assert r.status_code == 200
    text, adapter = _text_and_adapter(body, stream)
    assert text == "<<adapter=/a/ab-1>>"  # the adapter's weights generated it
    assert adapter == "ab-1"


@pytest.mark.parametrize("stream", [False, True])
def test_the_promoted_adapter_still_serves_by_default_and_is_named(tmp_path, stream):
    local = TaggingLocal()
    r, body = _ask(_client(tmp_path, local, _store(tmp_path)), stream=stream)
    assert r.status_code == 200
    text, adapter = _text_and_adapter(body, stream)
    assert text == "<<adapter=/a/extract-p>>"
    assert adapter == "extract-p"  # resolved id, though none was requested


@pytest.mark.parametrize("stream", [False, True])
def test_an_adapter_that_fails_and_falls_back_to_base_is_not_named(tmp_path, stream):
    local = TaggingLocal(broken={"/a/ab-1"})
    client = _client(tmp_path, local, _store(tmp_path))
    r, body = _ask(client, stream=stream, adapter="ab-1")
    assert r.status_code == 200
    text, adapter = _text_and_adapter(body, stream)
    assert text == "<<adapter=None>>"  # base weights answered
    assert adapter is None
    (rec,) = client.app.state.metrics._records
    assert rec.adapter is None


@pytest.mark.parametrize("stream", [False, True])
def test_a_backend_that_ignores_adapters_is_not_credited_with_one(tmp_path, stream):
    """echo declares adapters=False and ignores GenRequest.adapter: report none."""
    r, body = _ask(
        _client(tmp_path, EchoProvider(), _store(tmp_path)), stream=stream, adapter="ab-1"
    )
    assert r.status_code == 200
    assert _text_and_adapter(body, stream)[1] is None


def test_no_adapter_store_and_an_explicit_adapter_is_a_404(tmp_path):
    local = TaggingLocal()
    client = _client(tmp_path, local, None)
    client.app.state.router._adapter_store = lambda: None
    r, body = _ask(client, stream=False, adapter="ab-1")
    assert r.status_code == 404
    assert local.calls == []
