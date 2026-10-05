"""B-107: evicting a base model evicts the adapter variants layered over it.

Every (model, adapter) variant is its own resident holding a full copy of the base's weights
(B-069). ``ModelPool.evict(base)`` evicted only the bare id, so "evict the 14B" left a 9 GB
14B+adapter resident and counted. Ports reviewer probe p5.
"""

from __future__ import annotations

from test_model_selection import BIG, SMALL, _settings, fake  # noqa: F401

from hearth.providers import mlx_pool
from hearth.providers.base import GenRequest, Message


def _gen(pool, model, adapter=None):
    pool.generate(GenRequest(messages=[Message(role="user", content="x")], model=model,
                             adapter=adapter))


def test_evicting_a_base_evicts_its_variants_and_nothing_else(fake, tmp_path):  # noqa: F811
    first, second = tmp_path / "ad1", tmp_path / "ad2"
    first.mkdir()
    second.mkdir()
    pool = mlx_pool(_settings(tmp_path, ram_ceiling_gb=40.0))
    _gen(pool, BIG, str(first))
    _gen(pool, BIG, str(second))
    _gen(pool, BIG)
    _gen(pool, SMALL)
    assert len(pool.manager.resident_ids()) == 4
    assert pool.evict(BIG) is True
    assert pool.manager.resident_ids() == [SMALL]
    assert pool.manager.resident_ram_gb() == 2.0


def test_evicting_a_base_that_is_gone_still_evicts_its_variants(fake, tmp_path):  # noqa: F811
    adapter = tmp_path / "ad"
    adapter.mkdir()
    pool = mlx_pool(_settings(tmp_path))
    _gen(pool, BIG, str(adapter))  # only the variant is resident
    assert pool.evict(BIG) is True
    assert pool.manager.resident_ids() == []
    assert pool.evict(BIG) is False  # nothing left to evict
