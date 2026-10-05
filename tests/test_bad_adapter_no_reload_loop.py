"""B-102: a bad adapter does not cost a full base reload on every request.

Before: a (model, adapter) variant whose load failed was retried on every request, and its
admission evicted the resident base first (both are full models; together over the
ceiling). The router's degrade-and-retry then reloaded the base. Three requests = three
failed variant loads + three base loads of a 14B. The failure is now remembered against
the adapter's on-disk fingerprint until the adapter changes. Ports reviewer probe p7.
"""

from __future__ import annotations

import os
import sys

from test_model_selection import BIG, _app, fake, weights  # noqa: F401

from hearth.providers.base import GenRequest, Message


class _Store:
    def __init__(self, path: str) -> None:
        self.path = path

    def resolve_path(self, adapter_id, allow_candidate=False):
        return self.path

    def promoted_for(self, task_class):
        return None


def _bad_adapter_world(fake, tmp_path, local_policy):  # noqa: F811
    adapter = tmp_path / "bad_adapter"
    adapter.mkdir()
    (adapter / "adapters.safetensors").write_text("corrupt")
    attempts: list[str] = []
    real_load = fake.load

    def load(path, **kw):
        if kw.get("adapter_path"):
            attempts.append(kw["adapter_path"])
            raise RuntimeError("corrupt adapter")
        return real_load(path, **kw)

    sys.modules["mlx_lm"].load = load
    # 10 GB: the 14B fits alone, base + variant (2 x 9 GB) do not.
    _, pool, router = _app(tmp_path, local_policy, ram_ceiling_gb=10.0)
    router._adapters = _Store(str(adapter))
    return adapter, attempts, pool, router


def _ask(router):
    return router.route(
        GenRequest(messages=[Message(role="user", content="hi there")], model=BIG),
        adapter="a1",
    )


def test_three_requests_one_failed_variant_load_one_base_load(fake, tmp_path, local_policy):  # noqa: F811
    _, attempts, pool, router = _bad_adapter_world(fake, tmp_path, local_policy)
    for _ in range(3):
        routed = _ask(router)
        assert routed.record.adapter is None  # base weights answered, and the record says so
        assert routed.result.text == f"<<{weights(BIG)}>>"
    assert len(attempts) == 1
    assert fake.loaded_paths() == [weights(BIG)]  # the base, loaded once
    assert pool.manager.resident_ids() == [BIG]


def test_a_changed_adapter_is_tried_again(fake, tmp_path, local_policy):  # noqa: F811
    adapter, attempts, _, router = _bad_adapter_world(fake, tmp_path, local_policy)
    _ask(router)
    _ask(router)
    assert len(attempts) == 1
    # The operator fixes/retrains the adapter: its files change, so one more load is due.
    weights_file = adapter / "adapters.safetensors"
    weights_file.write_text("retrained, different size")
    st = weights_file.stat()
    os.utime(weights_file, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    _ask(router)
    assert len(attempts) == 2
    _ask(router)
    assert len(attempts) == 2  # ...and the new failure is remembered in turn
