"""B-111: ``policy_rungs`` honours ``defaults.local_model`` — readiness and warmup follow it.

Every unpinned class is served by ``defaults.local_model`` when it is set
(``Router._local_model`` step 3). A ``policy_rungs`` that ignored it and listed the registry
default instead survived every test: readiness would judge (and warmup load) a model the
profile never serves, while the one it does serve could be missing. Asserted on the outcome:
which model /ready judges and fails, which weights warmup loads, which model a request runs.
"""

from __future__ import annotations

from fastapi.testclient import TestClient
from test_model_selection import CODER7, SMALL, _app, _chat, fake, tag, weights  # noqa: F401

from hearth.router import RoutingPolicy
from hearth.router.classify import TASK_CLASSES
from hearth.router.policy import ClassRule, Defaults
from hearth.router.route import policy_rungs

READY = "/v1/hearth/admin/ready"


def _unpinned_with_default(model: str) -> RoutingPolicy:
    return RoutingPolicy(
        defaults=Defaults(local_model=model),
        classes={c: ClassRule(backend="local", escalate="never") for c in TASK_CLASSES},
        remotes={},
    )


def test_policy_rungs_lists_the_defaults_rung_not_the_registry_default():
    rungs = policy_rungs(_unpinned_with_default(SMALL), CODER7)
    assert list(rungs) == [SMALL]
    assert all(s.endswith("(defaults.local_model)") for s in rungs[SMALL])


def test_ready_judges_the_defaults_rung_and_fails_when_it_is_missing(fake, tmp_path):  # noqa: F811
    fake.missing.add(SMALL)
    app, _, _ = _app(tmp_path, _unpinned_with_default(SMALL))
    client = TestClient(app)
    body = client.get(READY).json()
    assert body["status"] == "failed" and SMALL in body["reason"]
    assert list(body["models"]) == [SMALL]  # the registry default (Coder-7B) serves nothing
    assert _chat(client, "auto").status_code == 503  # ...which is what a request sees


def test_warmup_loads_the_defaults_rung(fake, tmp_path):  # noqa: F811
    app, _, _ = _app(tmp_path, _unpinned_with_default(SMALL), warmup=True)
    app.state.warmup.thread.join(10)
    assert fake.loaded_paths() == [weights(SMALL)]
    client = TestClient(app)
    assert client.get(READY).status_code == 200
    assert _chat(client, "auto").json()["choices"][0]["message"]["content"] == tag(SMALL)
