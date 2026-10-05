"""B-070: an unregistered HEARTH_DEFAULT_MODEL is refused where the model is CHOSEN.

B-047 made ``serve``/``run``/``agent``/``mcp`` refuse to start on it, but at their call
sites. Where the model is actually chosen — ``ModelPool.resolve("")`` / ``("auto")`` and the
router's fall-through to the registry default — it still fell back to the catalog default
with a warning. So ``hearth eval``, the example scripts and any direct API user asked for one
model and were answered by another (CLAUDE.md §3). Now the choosing code itself raises
:class:`UnregisteredDefaultModelError`, so every entry point is covered.
"""

from __future__ import annotations

import pytest
from test_model_selection import CODER7, _settings, fake, tag  # noqa: F401

from hearth.observability.budget import BudgetAccountant
from hearth.observability.metrics import MetricsStore
from hearth.providers import mlx_pool
from hearth.providers.base import GenRequest, Message
from hearth.registry import UnregisteredDefaultModelError
from hearth.router import Router

TYPO = "nope/typo-model"


def _req(model: str) -> GenRequest:
    return GenRequest(messages=[Message(role="user", content="hello")], model=model)


@pytest.fixture
def typo_default(monkeypatch):
    monkeypatch.setenv("HEARTH_DEFAULT_MODEL", TYPO)


def test_pool_resolve_of_auto_or_empty_refuses(fake, tmp_path, typo_default):  # noqa: F811
    pool = mlx_pool(_settings(tmp_path))
    for wanted in ("", "auto", None):
        with pytest.raises(UnregisteredDefaultModelError, match=TYPO):
            pool.resolve(wanted)
    with pytest.raises(UnregisteredDefaultModelError):
        pool.generate(_req("auto"))
    assert fake.loads == []  # the catalog default was NOT loaded in its place


def test_an_explicit_registered_model_still_serves(fake, tmp_path, typo_default):  # noqa: F811
    pool = mlx_pool(_settings(tmp_path))
    assert pool.generate(_req(CODER7)).text == tag(CODER7)


def test_the_router_fall_through_to_the_default_refuses(
    fake, tmp_path, typo_default, local_policy  # noqa: F811
):
    """A direct API user: Router over the pool, a request naming no model."""
    pool = mlx_pool(_settings(tmp_path))
    router = Router(local_provider=pool, policy=local_policy, budget=BudgetAccountant(0),
                    metrics=MetricsStore())
    with pytest.raises(UnregisteredDefaultModelError, match=TYPO):
        router.route(_req("auto"))
    assert fake.loads == []


def test_unset_still_means_the_catalog_default(fake, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.delenv("HEARTH_DEFAULT_MODEL", raising=False)
    assert mlx_pool(_settings(tmp_path)).resolve("auto") == CODER7
