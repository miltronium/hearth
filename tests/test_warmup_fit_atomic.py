"""B-110: warmup's fit check and its load are one step — it never evicts a rung it warmed.

Before: ``_warmup`` asked ``_fits_beside_residents`` on the warmup thread, then called
``warm()``, which loaded on the MLX thread. A request that loaded another model in between
made the "fitting" load evict the LRU resident — the rung warmup had just warmed. The check
now runs inside the load, under the manager's load lock (``ModelManager.get_if_fits``).

The race is made deterministic by landing the request's load exactly in the gap: the
wrapper around ``warm`` performs it before delegating. Under the old code the fit had
already been judged by then; under the new code the judgement is part of the delegated call.
"""

from __future__ import annotations

from test_model_selection import BIG, CODER7, SMALL, _app, fake, weights  # noqa: F401

from hearth.gateway import app as app_mod
from hearth.providers.base import GenRequest, Message, ResourceEstimate
from hearth.providers.mlx import run_on_mlx_thread
from hearth.router.classify import TASK_CLASSES
from hearth.router.policy import load_policy
from hearth.serving import ModelManager


def _ladder(tmp_path):
    # SMALL serves most classes (warmed first), CODER7 a few (warmed second).
    lines = "".join(
        f"  {c}: {{backend: local, escalate: never, "
        f"local_model: {CODER7 if c in ('code', 'reason') else SMALL}}}\n"
        for c in TASK_CLASSES
    )
    path = tmp_path / "routing.yaml"
    path.write_text("classes:\n" + lines)
    return load_policy(path)


def test_a_request_load_in_the_gap_does_not_make_warmup_evict_its_rung(fake, tmp_path):  # noqa: F811
    policy = _ladder(tmp_path)
    # 12 GB: SMALL (2) + BIG (9) fit; adding CODER7 (4.5) would evict the LRU — SMALL.
    app, pool, _ = _app(tmp_path, policy, ram_ceiling_gb=12.0)
    real_warm = pool.warm
    calls: list[str] = []

    def warm_with_a_request_in_the_gap(model_id=None, **kw):
        calls.append(model_id)
        if len(calls) == 2:  # between rung 1 and rung 2: a client asks for the 14B
            pool.generate(GenRequest(messages=[Message(role="user", content="x")], model=BIG))
        return real_warm(model_id, **kw)

    pool.warm = warm_with_a_request_in_the_gap
    state = app_mod._WarmupState()
    app_mod._warmup(pool, pool.manager, app.state.registry, state, policy)
    assert calls == [SMALL, CODER7]
    assert SMALL in pool.manager.resident_ids()  # the rung warmup warmed is still there
    assert CODER7 not in pool.manager.resident_ids() and CODER7 in state.skipped
    assert fake.loaded_paths() == [weights(SMALL), weights(BIG)]


def test_get_if_fits_loads_only_without_evicting(fake, tmp_path):  # noqa: F811
    _, pool, _ = _app(tmp_path, _ladder(tmp_path), ram_ceiling_gb=12.0)
    manager = pool.manager
    get_if_fits = pool.manager.get_if_fits

    assert run_on_mlx_thread(get_if_fits, BIG) is not None
    assert run_on_mlx_thread(get_if_fits, SMALL) is not None  # 11 GB: fits
    assert run_on_mlx_thread(get_if_fits, CODER7) is None  # 15.5 GB: would evict
    assert manager.resident_ids() == [BIG, SMALL]
    assert manager.last_load_error(CODER7) is None  # not a failed load


def test_get_if_fits_on_a_plain_manager():
    class Fake:
        def __init__(self, ram):
            self.ram = ram

        def footprint(self, model_id):
            return ResourceEstimate(ram_gb=self.ram)

    sizes = {"a": 5.0, "b": 5.0, "c": 1.0}
    manager = ModelManager(factory=lambda m: Fake(sizes[m]), ram_ceiling_gb=10.0)
    assert manager.get_if_fits("a") is not None
    assert manager.get_if_fits("b") is not None
    assert manager.get_if_fits("c") is None
    assert manager.resident_ids() == ["a", "b"]
