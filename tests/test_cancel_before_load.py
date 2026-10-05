"""B-103: a job abandoned while queued loads no weights — on every path that could load.

B-071 added "do not load weights for a job abandoned while queued" checks in
``ModelPool.generate`` and in ``MLXProvider``'s generation, but no test failed with either
removed. And the pool's STREAMING path had no check of its own: the provider's check ran only
after ``manager.get()`` had already loaded (and evicted for) the weights. Each test below is
the outcome — the fake mlx_lm's load log stays empty — on one path.
"""

from __future__ import annotations

import threading

import pytest
from test_model_selection import BIG, CODER7, _settings, fake  # noqa: F401

from hearth.providers import mlx_pool
from hearth.providers.base import GenerationCancelledError, GenRequest, Message, cancel_scope
from hearth.providers.mlx import MLXProvider


def _gone() -> threading.Event:
    event = threading.Event()
    event.set()  # the client left before the job reached the MLX thread
    return event


def _req(model: str) -> GenRequest:
    return GenRequest(messages=[Message(role="user", content="hi")], model=model)


def test_pool_generate_abandoned_while_queued_loads_nothing(fake, tmp_path):  # noqa: F811
    pool = mlx_pool(_settings(tmp_path))
    with cancel_scope(_gone()), pytest.raises(GenerationCancelledError):
        pool.generate(_req(BIG))
    assert fake.loaded_paths() == []
    assert pool.manager.resident_ids() == []
    assert pool.manager.last_load_error(BIG) is None  # not a failed load either


def test_pool_stream_abandoned_while_queued_loads_nothing(fake, tmp_path):  # noqa: F811
    pool = mlx_pool(_settings(tmp_path))
    with cancel_scope(_gone()), pytest.raises(GenerationCancelledError):
        list(pool.stream_deltas(_req(BIG)))
    assert fake.loaded_paths() == []
    assert pool.manager.resident_ids() == []


def test_pool_abandoned_job_does_not_evict_a_resident(fake, tmp_path):  # noqa: F811
    pool = mlx_pool(_settings(tmp_path, ram_ceiling_gb=10.0))
    pool.generate(_req(BIG))  # 9 GB resident; CODER7 (4.5 GB) would evict it
    for run in (pool.generate, lambda r: list(pool.stream_deltas(r))):
        with cancel_scope(_gone()), pytest.raises(GenerationCancelledError):
            run(_req(CODER7))
    assert pool.manager.resident_ids() == [BIG]
    assert fake.loaded_paths() == [f"/weights/{BIG}"]


@pytest.mark.parametrize("stream", [False, True])
def test_bare_provider_abandoned_while_queued_loads_nothing(fake, stream):  # noqa: F811
    provider = MLXProvider(CODER7)
    with cancel_scope(_gone()), pytest.raises(GenerationCancelledError):
        if stream:
            list(provider.stream_deltas(_req(CODER7)))
        else:
            provider.generate(_req(CODER7))
    assert fake.loaded_paths() == [] and not provider.is_loaded
