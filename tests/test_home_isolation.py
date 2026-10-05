"""The suite never touches the operator's real ~/.hearth (tests/conftest.py autouse fixture).

Asserted on the OUTCOME: where a default-built store actually points, not on the fixture.
"""

from __future__ import annotations

import os
from pathlib import Path

from hearth.providers.echo import EchoProvider
from hearth.router import Router


def test_a_default_adapter_store_resolves_inside_a_throwaway_home():
    real_home = (Path.home() / ".hearth").resolve()
    store = Router(local_provider=EchoProvider())._adapter_store()
    assert store is not None
    resolved = store.path.resolve()
    assert real_home not in resolved.parents and resolved != real_home / "adapters.json"
    assert str(resolved).startswith(str(Path(os.environ["HEARTH_HOME"]).resolve()))
