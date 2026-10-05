"""B-072: the ModelManager does not evict residents before it knows the new load can work.

Before: ``_admit`` evicted LRU residents to make room and only THEN called ``load``. A request
for a registered-but-never-pulled model therefore unloaded a working 14B, then failed with
"not on disk" — the next request for the 14B paid a full reload for nothing. Now the provider's
``preflight`` (for MLX: the weights resolve on disk, without loading) runs before anything is
evicted, so a load that cannot start leaves the residents exactly as they were.

Asserted on the outcome: the fake mlx_lm records every load and the providers count unloads.
"""

from __future__ import annotations

import pytest
from test_model_selection import (  # noqa: F401
    BIG,
    CODER7,
    SMALL,
    _settings,
    fake,
    weights,
)

from hearth.providers import mlx_pool
from hearth.providers.base import GenRequest, Message
from hearth.providers.mlx import ModelNotOnDiskError
from hearth.serving import ModelManager


def _req(model: str) -> GenRequest:
    return GenRequest(messages=[Message(role="user", content="hi")], model=model)


def test_a_model_not_on_disk_evicts_nothing(fake, tmp_path):  # noqa: F811
    # Ceiling 10 GB: the 14B (9 GB) is resident; the 7B (4.5 GB) would need it evicted.
    pool = mlx_pool(_settings(tmp_path, ram_ceiling_gb=10.0))
    pool.generate(_req(BIG))
    resident = pool.manager.peek(BIG)
    fake.missing.add(CODER7)
    with pytest.raises(ModelNotOnDiskError):
        pool.generate(_req(CODER7))
    assert pool.manager.resident_ids() == [BIG]
    assert resident.is_loaded  # never unloaded
    assert pool.manager.last_load_error(CODER7)  # the failure is still reported
    pool.generate(_req(BIG))
    assert fake.loaded_paths() == [weights(BIG)]  # no reload of the 14B was needed


def test_a_model_that_is_on_disk_still_evicts_to_make_room(fake, tmp_path):  # noqa: F811
    pool = mlx_pool(_settings(tmp_path, ram_ceiling_gb=10.0))
    pool.generate(_req(BIG))
    pool.generate(_req(CODER7))
    assert pool.manager.resident_ids() == [CODER7]


class _Provider:
    def __init__(self, model_id: str, ram_gb: float, ok: bool = True) -> None:
        self.model_id, self.ram_gb, self.ok = model_id, ram_gb, ok
        self.unloads = 0

    def footprint(self, model_id):
        from hearth.providers.base import ResourceEstimate

        return ResourceEstimate(ram_gb=self.ram_gb)

    def preflight(self, model_id):
        if not self.ok:
            raise FileNotFoundError(f"{model_id}: no weights")

    def load(self, model_id):
        assert self.ok, "load called after a failed preflight"

    def unload(self, model_id):
        self.unloads += 1


def test_manager_runs_preflight_before_evicting():
    made: dict[str, _Provider] = {}

    def factory(model_id):
        made[model_id] = _Provider(model_id, 6.0, ok=model_id != "absent")
        return made[model_id]

    manager = ModelManager(factory, ram_ceiling_gb=10.0)
    manager.get("a")
    with pytest.raises(FileNotFoundError):
        manager.get("absent")
    assert manager.resident_ids() == ["a"] and made["a"].unloads == 0
    assert "no weights" in manager.last_load_error("absent")
