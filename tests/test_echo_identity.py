"""B-068: the echo stub names itself and never claims frontier-token savings.

Before: with ``HEARTH_BACKEND=auto`` fallen back to echo (a pruned venv), ``/ready`` was 503
— correct — but ``POST model=Qwen-14B`` answered 200 with ``model: ...14B`` and the text
``[echo] hi``, and the metrics credited ``estimated_frontier_tokens_saved`` for it: a stub's
echo was labelled, recorded and priced as real inference by the model the client asked for
(CLAUDE.md §3: the label came from the request, not from what generated).

Asserted on every surface a reader could take the identity from: the response body, the
``hearth`` telemetry block, the streamed chunks, the metrics record and the rollup.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from hearth.config import Settings
from hearth.gateway import create_app
from hearth.observability.budget import BudgetAccountant
from hearth.observability.metrics import MetricsStore
from hearth.providers import _auto_fallback_stub
from hearth.providers.base import GenRequest, Message
from hearth.providers.echo import EchoProvider
from hearth.router import Router

BIG = "mlx-community/Qwen2.5-14B-Instruct-4bit"


def _app(tmp_path, provider, local_policy, backend: str):
    metrics = MetricsStore()
    router = Router(local_provider=provider, policy=local_policy, budget=BudgetAccountant(0),
                    metrics=metrics)
    settings = Settings(backend=backend, home=tmp_path / ".hearth", require_auth=False,
                        warmup=False)
    app = create_app(provider=provider, settings=settings, router=router, metrics=metrics)
    return TestClient(app), metrics


@pytest.mark.parametrize(
    ("make", "backend"), [(_auto_fallback_stub, "auto"), (EchoProvider, "echo")]
)
def test_the_stub_names_itself_not_the_requested_model(tmp_path, local_policy, make, backend):
    client, metrics = _app(tmp_path, make(), local_policy, backend)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": BIG, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "[echo] hi"
    assert body["model"] == "echo"
    assert body["hearth"]["model"] == "echo" and body["hearth"]["backend"] == "echo"
    assert body["hearth"]["estimated_frontier_tokens_saved"] == 0
    (rec,) = list(metrics._records)
    assert (rec.model, rec.backend, rec.estimated_frontier_tokens_saved) == ("echo", "echo", 0)
    assert metrics.rollup()["estimated_frontier_tokens_saved"] == 0


@pytest.mark.parametrize(
    ("make", "backend"), [(_auto_fallback_stub, "auto"), (EchoProvider, "echo")]
)
def test_the_streamed_stub_names_itself_and_saves_nothing(tmp_path, local_policy, make, backend):
    client, metrics = _app(tmp_path, make(), local_policy, backend)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": BIG, "stream": True,
              "messages": [{"role": "user", "content": "hello there friend"}]},
    )
    events = [json.loads(line[6:]) for line in resp.text.splitlines()
              if line.startswith("data: ") and line != "data: [DONE]"]
    final = next(e for e in reversed(events) if e.get("hearth"))
    assert final["model"] == "echo" and final["hearth"]["model"] == "echo"
    assert final["hearth"]["estimated_frontier_tokens_saved"] == 0
    # EVERY chunk names the stub — the role chunk and each content chunk too, not just the
    # final one (B-101: they used to carry the requested 14B's id).
    chunks = [e for e in events if "choices" in e]
    assert len(chunks) >= 3  # role, at least one content delta, final
    assert [c["model"] for c in chunks] == ["echo"] * len(chunks)
    (rec,) = list(metrics._records)
    assert (rec.model, rec.estimated_frontier_tokens_saved) == ("echo", 0)


class _BrokenEcho(EchoProvider):
    """An echo stub whose generation dies: what was ATTEMPTED is still the stub."""

    def generate(self, req):
        raise RuntimeError("stub broke")

    def stream_deltas(self, req):
        raise RuntimeError("stub broke")
        yield  # pragma: no cover — makes this a generator


@pytest.mark.parametrize("stream", [False, True])
def test_a_failed_stub_request_is_recorded_as_the_stub(tmp_path, local_policy, stream):
    """B-101: a failure record names the model that was attempted, as a success would."""
    client, metrics = _app(tmp_path, _BrokenEcho(), local_policy, "echo")
    resp = client.post(
        "/v1/chat/completions",
        json={"model": BIG, "stream": stream,
              "messages": [{"role": "user", "content": "hello there friend"}]},
    )
    if stream:
        events = [json.loads(line[6:]) for line in resp.text.splitlines()
                  if line.startswith("data: ") and line != "data: [DONE]"]
        assert [e["model"] for e in events if "choices" in e] == ["echo"]  # the role chunk
        assert events[-1]["error"]["code"] == "hearth.provider.unavailable"
    else:
        assert resp.status_code == 503
    (rec,) = list(metrics._records)
    assert rec.failed and (rec.model, rec.backend) == ("echo", "echo")


def test_router_never_prices_an_echo_as_saved_tokens(local_policy):
    router = Router(local_provider=EchoProvider(), policy=local_policy,
                    budget=BudgetAccountant(0), metrics=MetricsStore())
    routed = router.route(
        GenRequest(messages=[Message(role="user", content="summarize " * 50)], model=BIG)
    )
    assert routed.decision.model == BIG  # what was asked for...
    assert routed.result.model == "echo"  # ...is not what generated
    assert routed.record.estimated_frontier_tokens_saved == 0
