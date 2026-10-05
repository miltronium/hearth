"""A cancellation during the base-weights retry stays a cancellation (review finding 6).

``Router._generate`` retries once on base weights when an adapter fails. The retry sat under
a bare ``except Exception``, so a client that went away while the adapter was failing — its
cancellation surfacing in the retry — came back as ``ProviderError`` (a 503, a "provider
outage") instead of :class:`GenerationCancelledError`. The first attempt already let
cancellation through; the retry did not, so the same event had two meanings depending on
when it landed.
"""

from __future__ import annotations

import sys
import threading

import pytest
from test_model_selection import BIG, _app, fake  # noqa: F401

from hearth.providers.base import GenerationCancelledError, GenRequest, Message, cancel_scope
from hearth.router import ProviderError


class _Store:
    def __init__(self, path: str) -> None:
        self.path = path

    def resolve_path(self, adapter_id, allow_candidate=False):
        return self.path

    def promoted_for(self, task_class):
        return None


def test_cancel_landing_in_the_base_retry_is_not_a_provider_error(fake, tmp_path, local_policy):  # noqa: F811
    adapter = tmp_path / "ad"
    adapter.mkdir()
    gone = threading.Event()
    real_load = fake.load

    def load(path, **kw):
        if kw.get("adapter_path"):
            gone.set()  # the client goes away while the adapter load is failing
            raise RuntimeError("corrupt adapter")
        return real_load(path, **kw)

    sys.modules["mlx_lm"].load = load
    _, _, router = _app(tmp_path, local_policy)
    router._adapters = _Store(str(adapter))
    with cancel_scope(gone), pytest.raises(GenerationCancelledError) as caught:
        router.route(GenRequest(messages=[Message(role="user", content="hi")], model=BIG),
                     adapter="a1")
    assert not isinstance(caught.value, ProviderError)
    # Nothing was generated for a caller that left: the base was never loaded for it.
    assert fake.loaded_paths() == []
    (rec,) = list(router.metrics._records)
    assert rec.failed and "cancel" in rec.failed
