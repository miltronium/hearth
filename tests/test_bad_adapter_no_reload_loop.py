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

import pytest
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


# -- B-127: transient failures are not remembered; a disguised overwrite is noticed -------


def _flaky_world(fake, tmp_path, local_policy, error):  # noqa: F811
    """Like _bad_adapter_world, but the adapter load fails with ``error``."""
    adapter, attempts, pool, router = _bad_adapter_world(fake, tmp_path, local_policy)
    real_load = fake.load

    def load(path, **kw):
        if kw.get("adapter_path"):
            attempts.append(kw["adapter_path"])
            raise error
        return real_load(path, **kw)

    sys.modules["mlx_lm"].load = load
    return adapter, attempts, pool, router


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("[metal::malloc] Attempting to allocate 9663676416 bytes which is greater "
                     "than the maximum allowed buffer size"),
        RuntimeError("[METAL] Command buffer execution failed: Insufficient Memory "
                     "(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"),
        MemoryError(),
        OSError(12, "Cannot allocate memory"),
    ],
)
def test_a_transient_load_failure_is_not_remembered(fake, tmp_path, local_policy, error):  # noqa: F811
    _, attempts, _, router = _flaky_world(fake, tmp_path, local_policy, error)
    for _ in range(3):
        routed = _ask(router)
        assert routed.record.adapter is None  # base weights still answer
    assert len(attempts) == 3  # ...but each request tried the adapter again


def test_a_same_size_overwrite_that_restores_the_mtime_is_tried_again(fake, tmp_path, local_policy):  # noqa: F811
    """`cp -p` / `rsync -t` / os.utime: same size, same mtime — but the bytes are new."""
    adapter, attempts, _, router = _bad_adapter_world(fake, tmp_path, local_policy)
    _ask(router)
    _ask(router)
    assert len(attempts) == 1
    weights_file = adapter / "adapters.safetensors"
    before = weights_file.stat()
    weights_file.write_text("CORRUPT")  # same length as "corrupt"
    os.utime(weights_file, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = weights_file.stat()
    assert (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns)
    _ask(router)
    assert len(attempts) == 2


def test_an_atomic_replace_keeping_size_and_mtime_is_tried_again(fake, tmp_path, local_policy):  # noqa: F811
    adapter, attempts, _, router = _bad_adapter_world(fake, tmp_path, local_policy)
    _ask(router)
    weights_file = adapter / "adapters.safetensors"
    before = weights_file.stat()
    fresh = adapter / ".fresh"
    fresh.write_text("CORRUPT")
    os.utime(fresh, ns=(before.st_atime_ns, before.st_mtime_ns))
    os.replace(fresh, weights_file)  # new inode, old size and mtime
    _ask(router)
    assert len(attempts) == 2


def test_transient_classification():
    from hearth.serving.manager import ModelTooLargeError
    from hearth.serving.pool import is_transient_load_error

    assert is_transient_load_error(ModelTooLargeError("too big"))
    assert is_transient_load_error(OSError(35, "Resource temporarily unavailable"))
    for message in ("GPU: Insufficient Memory for the command buffer", "CUDA out of memory",
                    "failed to allocate 4096 bytes", "allocation failed in pool",
                    "resource exhausted: buffer", "Cannot allocate memory (wired limit)"):
        assert is_transient_load_error(RuntimeError(message)), message
    assert not is_transient_load_error(RuntimeError("corrupt adapter"))
    assert not is_transient_load_error(ValueError("[load_safetensors] Invalid json header"))
    assert not is_transient_load_error(FileNotFoundError(2, "No such file"))


def test_the_fingerprint_covers_inode_and_ctime(tmp_path, monkeypatch):
    """Coarse-timestamp filesystems (FAT, SMB: 1-2 s) can give a replacement file the old
    mtime AND ctime; the inode still differs. Each field alone must change the stamp."""
    from pathlib import Path
    from types import SimpleNamespace

    from hearth.serving.pool import adapter_fingerprint

    f = tmp_path / "adapters.safetensors"
    f.write_text("x")
    fields = {"st_mtime_ns": 1, "st_ctime_ns": 2, "st_ino": 3, "st_size": 4}

    def stamp(**changed):
        monkeypatch.setattr(Path, "stat", lambda self, **kw: SimpleNamespace(
            **{**fields, **changed}, st_mode=0o100644))
        return adapter_fingerprint(str(f))

    base = stamp()
    for name in fields:
        assert stamp(**{name: 99}) != base, name
