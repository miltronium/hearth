"""B-069: every loaded LoRA adapter variant is counted as the full base reload it is.

Before: ``MLXProvider._load_variant`` loaded base+adapter per variant into the provider's own
cache, invisible to the ModelManager. A 14B serving three adapters held ~36 GB of weights
while ``resident_ram_gb`` said 9.0 — above the 24 GB ceiling and the 30.15 GB GPU working set
on this machine (CLAUDE.md §5), with nothing evicting.

Now the pool keeps each (model, adapter) variant as its own resident under the manager, sized
at the model's full ``ram_gb``, LRU-evicted like any model. The fake mlx_lm records every
``load`` call (and its adapter_path), so the counts below are of loads that happened.

Why not share the base weights: mlx_lm 0.29.1 can apply an adapter in place
(``tuner.utils.load_adapters``), but its ``remove_lora_layers`` restores only ``LoRALinear``
— not DoRA, LoRA-embedding or switch layers — and a ``fine_tune_type: full`` adapter
overwrites base weights outright. A swap that silently leaves the previous adapter applied
would answer with weights nobody asked for, which is worse than paying the memory.
"""

from __future__ import annotations

import pytest
from test_model_selection import BIG, _settings, fake  # noqa: F401

from hearth.providers import mlx_pool
from hearth.providers.base import GenRequest, Message
from hearth.serving import ModelTooLargeError


def _gen(pool, adapter=None):
    return pool.generate(
        GenRequest(messages=[Message(role="user", content="x")], model=BIG, adapter=adapter)
    )


def _adapters(tmp_path, n: int) -> list[str]:
    out = []
    for i in range(n):
        path = tmp_path / f"ad{i}"
        path.mkdir()
        out.append(str(path))
    return out


def test_resident_ram_counts_every_variant(fake, tmp_path):  # noqa: F811
    pool = mlx_pool(_settings(tmp_path, ram_ceiling_gb=40.0))
    adapters = _adapters(tmp_path, 3)
    for adapter in (None, *adapters):
        _gen(pool, adapter)
    assert len(fake.loads) == 4  # each variant really was a full load
    assert pool.manager.resident_ram_gb() == pytest.approx(36.0)  # was 9.0


def test_variants_are_lru_evicted_under_the_ceiling(fake, tmp_path):  # noqa: F811
    pool = mlx_pool(_settings(tmp_path, ram_ceiling_gb=24.0))
    adapters = _adapters(tmp_path, 3)
    for adapter in (None, *adapters):
        _gen(pool, adapter)
    # 4 x 9 GB cannot fit in 24: the two least-recently-used variants were evicted.
    assert pool.manager.resident_ram_gb() <= 24.0
    assert len(pool.manager.resident_ids()) == 2
    held = [r.provider for r in pool.manager.residents()]
    assert [p.adapter for p in held] == adapters[1:]
    # An evicted variant's weights are really gone, not just unaccounted.
    assert all(p.is_loaded for p in held)


def test_a_variant_that_cannot_fit_is_refused(fake, tmp_path):  # noqa: F811
    pool = mlx_pool(_settings(tmp_path, ram_ceiling_gb=8.0))  # the 14B alone is 9 GB
    (adapter,) = _adapters(tmp_path, 1)
    with pytest.raises(ModelTooLargeError):
        _gen(pool, adapter)
    assert fake.loads == []


def test_a_variant_generates_with_its_adapter_and_the_base_with_none(fake, tmp_path):  # noqa: F811
    seen: list[str | None] = []
    real_load = fake.load

    def load(path, **kwargs):
        seen.append(kwargs.get("adapter_path"))
        return real_load(path, **kwargs)

    fake.load = load
    import sys

    sys.modules["mlx_lm"].load = load
    pool = mlx_pool(_settings(tmp_path, ram_ceiling_gb=40.0))
    (adapter,) = _adapters(tmp_path, 1)
    _gen(pool, adapter)
    _gen(pool, None)
    _gen(pool, adapter)  # cached: no third load
    assert seen == [adapter, None]


def test_a_missing_adapter_is_refused_before_anything_is_evicted(fake, tmp_path):  # noqa: F811
    pool = mlx_pool(_settings(tmp_path, ram_ceiling_gb=10.0))
    _gen(pool)
    with pytest.raises(FileNotFoundError):
        _gen(pool, str(tmp_path / "no-such-adapter"))
    assert pool.manager.resident_ids() == [BIG]
