"""All MLX work runs on one thread, whoever calls (providers/mlx.py, the MLX thread).

MLX's GPU stream is thread-local: weights loaded on one thread raise "There is no
Stream(gpu, 0) in current thread" when run from another. The fake ``mlx_lm`` below models
exactly that rule — generation on any thread but the one that loaded raises the real
message — and then the provider is driven from many request threads at once, the way
FastAPI's threadpool drives it. Measured live before the fix: 3 concurrent chat requests
against a 14B gave 2 of these errors and 1 answer.
"""

from __future__ import annotations

import importlib.machinery
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor

import pytest

from hearth.providers import mlx as mlx_mod
from hearth.providers.base import GenRequest, Message
from hearth.providers.mlx import MLXProvider, on_mlx_thread, run_on_mlx_thread


class _Response:
    def __init__(self, text: str, finish_reason: str | None = None) -> None:
        self.text = text
        self.finish_reason = finish_reason


class _Tokenizer:
    eos_token = "<|im_end|>"

    def encode(self, text):
        return list(text)

    def apply_chat_template(self, chat, tokenize, add_generation_prompt):
        return chat[-1]["content"]


class ThreadLocalMLX:
    """A fake ``mlx_lm`` that enforces MLX's one-thread rule, with the real error text."""

    def __init__(self, tokens: int = 3, delay: float = 0.0) -> None:
        self.loaded_on: threading.Thread | None = None
        self.generated_on: set[str] = set()
        self.tokens = tokens
        self.delay = delay
        self.emitted = 0

    def load(self, path, **kwargs):
        self.loaded_on = threading.current_thread()
        return object(), _Tokenizer()

    def stream_generate(self, model, tokenizer, prompt, max_tokens):
        if threading.current_thread() is not self.loaded_on:
            raise RuntimeError("There is no Stream(gpu, 0) in current thread.")
        self.generated_on.add(threading.current_thread().name)
        for i in range(self.tokens):
            time.sleep(self.delay)
            self.emitted += 1
            last = i == self.tokens - 1
            yield _Response(f"{prompt}-{i} ", "stop" if last else None)

    def install(self, monkeypatch) -> None:
        module = types.ModuleType("mlx_lm")
        module.load = self.load
        module.stream_generate = self.stream_generate
        module.__spec__ = importlib.machinery.ModuleSpec("mlx_lm", loader=None)
        monkeypatch.setitem(sys.modules, "mlx_lm", module)
        monkeypatch.setattr(mlx_mod, "resolve_local_model", lambda model_id: model_id)


def _req(text: str) -> GenRequest:
    return GenRequest(messages=[Message(role="user", content=text)], model="m", max_tokens=8)


def test_concurrent_requests_from_many_threads_all_succeed(monkeypatch):
    fake = ThreadLocalMLX(delay=0.005)
    fake.install(monkeypatch)
    provider = MLXProvider("org/model")
    with ThreadPoolExecutor(max_workers=6) as pool:  # FastAPI's threadpool, in miniature
        results = list(pool.map(lambda i: provider.generate(_req(f"q{i}")), range(12)))
    assert [r.text.split("-")[0] for r in results] == [f"q{i}" for i in range(12)]
    assert all(r.finish_reason == "stop" for r in results)
    # Outcome, not configuration: every generation actually ran on one MLX thread.
    assert len(fake.generated_on) == 1
    assert next(iter(fake.generated_on)).startswith("hearth-mlx")


def test_streaming_from_other_threads_streams_and_succeeds(monkeypatch):
    fake = ThreadLocalMLX()
    fake.install(monkeypatch)
    provider = MLXProvider("org/model")

    def consume(i):
        return [d.text for d in provider.stream_deltas(_req(f"s{i}")) if d.text]

    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(consume, range(8)))
    for i, parts in enumerate(outputs):
        assert "".join(parts).startswith(f"s{i}-0")
        assert len(parts) >= 2  # deltas arrived incrementally, not as one blob


def test_errors_raised_on_the_mlx_thread_reach_the_caller(monkeypatch):
    fake = ThreadLocalMLX()
    fake.install(monkeypatch)

    def broken(*a, **k):
        raise ValueError("model exploded")
        yield  # pragma: no cover

    monkeypatch.setattr(fake, "stream_generate", broken)
    sys.modules["mlx_lm"].stream_generate = broken
    with pytest.raises(ValueError, match="model exploded"):
        MLXProvider("org/model").generate(_req("x"))


def test_a_consumer_that_stops_early_frees_the_mlx_thread(monkeypatch):
    fake = ThreadLocalMLX(tokens=10_000, delay=0.001)
    fake.install(monkeypatch)
    provider = MLXProvider("org/model")
    stream = provider.stream_deltas(_req("long"))
    next(stream)
    stream.close()  # a client disconnecting mid-answer
    # The next request must run promptly, not wait out 10k tokens of an abandoned stream.
    started = time.perf_counter()
    fake.tokens = 2
    run_on_mlx_thread(lambda: None)
    assert time.perf_counter() - started < 2.0
    assert fake.emitted < 10_000


def test_calls_already_on_the_mlx_thread_do_not_deadlock(monkeypatch):
    fake = ThreadLocalMLX()
    fake.install(monkeypatch)
    provider = MLXProvider("org/model")
    assert on_mlx_thread() is False
    result = run_on_mlx_thread(lambda: (on_mlx_thread(), provider.generate(_req("nested"))))
    assert result[0] is True
    assert result[1].text.startswith("nested")
