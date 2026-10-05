"""B-109: ``defaults: {local_model: null}`` means "unset", exactly like ``auto``.

YAML ``null`` (or a key with no value) is how an operator says "no value". A class rule
already reads ``local_model: null`` as "no opinion"; the defaults block stringified it to
``'None'`` and refused the whole profile with "'None' is not in the model registry". It now
falls through to the registry default, as ``auto`` and an absent key do. Ports probe p4.
"""

from __future__ import annotations

import pytest

from hearth.router.policy import load_policy
from hearth.router.route import policy_rungs


@pytest.mark.parametrize(
    "defaults",
    ["{local_model: null}", "{local_model: ~}", "{local_model: auto}", "{}", "{local_model: }"],
)
def test_null_auto_and_absent_all_fall_through(tmp_path, defaults):
    path = tmp_path / "routing.yaml"
    path.write_text(f"defaults: {defaults}\nclasses:\n  chat: {{local_model: null}}\n")
    policy = load_policy(path)
    assert policy.defaults.local_model == "auto"
    rungs = policy_rungs(policy, "some/registry-default")
    assert list(rungs) == ["some/registry-default"]
    assert all(s.endswith("(registry default)") for s in rungs["some/registry-default"])
