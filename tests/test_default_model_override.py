"""B-029 / B-047: an explicitly set ``HEARTH_DEFAULT_MODEL`` that names no registered model.

B-029 made the silent fallback visible (doctor WARN, one registry log line). B-047 found the
fallback itself was the bug: ``/ready`` reported the named model failed while ``model=auto``
was answered by the catalog default. The contract is now:

* unset (or empty) -> the catalog default, as before;
* set and registered -> that model;
* set and unregistered -> ``serve``/``run``/``agent``/``mcp`` (and ``rag query --answer``)
  refuse to start, exit 2, naming the bad id and the registered ids; ``hearth doctor`` FAILs
  (fatal). ``Registry.default_id`` stays lenient for code that only needs an id.
"""

from __future__ import annotations

import logging

import pytest
from typer.testing import CliRunner

from hearth.cli import app
from hearth.config import get_settings
from hearth.registry import UnregisteredDefaultModelError, get_registry, load_registry
from hearth.router import policy as policy_mod

CATALOG_DEFAULT = "org/catalog-default"
OTHER = "org/other-registered"
UNREGISTERED = "org/not-in-the-catalog"

_MODELS_YAML = f"""
default: {CATALOG_DEFAULT}
models:
  - {{id: {CATALOG_DEFAULT}, backend: echo}}
  - {{id: {OTHER}, backend: echo}}
"""

_LOCAL_PROFILE = """
defaults: {local_model: auto, remote: none, remote_budget_tokens_per_day: 0}
classes:
  chat: {backend: local, escalate: never}
remotes: {}
"""


@pytest.fixture(autouse=True)
def fresh_process_caches():
    """The CLI reads the process-cached registry/settings/policy; each test sets its own."""
    caches = (get_registry, get_settings, policy_mod.get_policy)
    for cache in caches:
        cache.cache_clear()
    yield
    for cache in caches:
        cache.cache_clear()


@pytest.fixture
def no_server(monkeypatch):
    """Record `uvicorn.run` instead of binding; restore the `hearth` logger serve touches."""
    import uvicorn

    log = logging.getLogger("hearth")
    handlers, level = list(log.handlers), log.level
    calls: list = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: calls.append((a, k)))
    yield calls
    log.handlers[:] = handlers
    log.setLevel(level)


def _write_models(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(_MODELS_YAML)
    return path


def _env(tmp_path, override: str | None) -> dict[str, str]:
    profile = tmp_path / "routing.yaml"
    profile.write_text(_LOCAL_PROFILE)
    roots = tmp_path / "roots"
    roots.mkdir(exist_ok=True)
    env = {
        "COLUMNS": "400",
        "HEARTH_HOME": str(tmp_path / "home"),
        "HEARTH_MODELS_YAML": str(_write_models(tmp_path)),
        "HEARTH_BACKEND": "echo",
        "HEARTH_ROUTING_YAML": str(profile),
        "HEARTH_FILE_ROOTS": str(roots),
        "HEARTH_EMBEDDER": "hash",
    }
    # CliRunner's env: None removes the variable for the invocation.
    env["HEARTH_DEFAULT_MODEL"] = override  # type: ignore[assignment]
    return env


# -- the commands that serve --------------------------------------------------------------

SERVING_COMMANDS = [
    pytest.param(["serve"], id="serve"),
    pytest.param(["run", "hello"], id="run"),
    pytest.param(["agent", "hello"], id="agent"),
    pytest.param(["mcp"], id="mcp"),
    pytest.param(["rag", "query", "hello", "--answer"], id="rag-query-answer"),
]


@pytest.mark.parametrize("argv", SERVING_COMMANDS)
def test_an_unregistered_default_refuses_to_start_naming_it_and_the_registered_ids(
    argv, tmp_path, no_server
):
    result = CliRunner().invoke(app, argv, env=_env(tmp_path, UNREGISTERED))
    assert result.exit_code == 2, (result.exit_code, result.output, result.exception)
    assert isinstance(result.exception, SystemExit), repr(result.exception)
    flat = " ".join(result.output.split())
    assert "Refusing to start" in flat
    assert UNREGISTERED in flat
    assert CATALOG_DEFAULT in flat and OTHER in flat  # the registered ids
    assert "Traceback" not in result.output
    assert no_server == []  # serve never bound a port


def test_serve_starts_with_the_override_unset_or_registered(tmp_path, no_server):
    """The refusal is specific to an unregistered id: unset -> catalog default, a registered
    id -> that id. Both reach uvicorn (stubbed) and name the model that will serve."""
    unset = CliRunner().invoke(app, ["serve"], env=_env(tmp_path, None))
    assert unset.exit_code == 0, unset.output
    assert f"model={CATALOG_DEFAULT}" in unset.output
    get_registry.cache_clear()
    named = CliRunner().invoke(app, ["serve"], env=_env(tmp_path, OTHER))
    assert named.exit_code == 0, named.output
    assert f"model={OTHER}" in named.output
    assert len(no_server) == 2


def test_run_serves_with_the_override_unset(tmp_path):
    result = CliRunner().invoke(app, ["run", "hello"], env=_env(tmp_path, None))
    assert result.exit_code == 0, result.output
    assert "Refusing to start" not in result.output


def test_rag_query_without_answer_is_not_gated(tmp_path):
    """Retrieval alone never generates, so it does not depend on the default model."""
    result = CliRunner().invoke(
        app, ["rag", "query", "hello"], env=_env(tmp_path, UNREGISTERED)
    )
    assert result.exit_code == 0, result.output


# -- doctor --------------------------------------------------------------------------------


def test_hearth_doctor_fails_fatally_naming_the_bad_id(tmp_path):
    result = CliRunner().invoke(app, ["doctor"], env=_env(tmp_path, UNREGISTERED))
    line = next((ln for ln in result.output.splitlines() if "default_model" in ln), "")
    assert "FAIL" in line, result.output
    assert UNREGISTERED in line, result.output
    assert result.exit_code == 1, result.output
    assert "Fatal checks failed" in result.output


def test_hearth_doctor_is_quiet_when_the_override_is_registered(tmp_path):
    result = CliRunner().invoke(app, ["doctor"], env=_env(tmp_path, OTHER))
    assert "default_model" not in result.output


# -- the registry ---------------------------------------------------------------------------


def test_require_default_raises_and_default_id_stays_lenient(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("HEARTH_DEFAULT_MODEL", UNREGISTERED)
    registry = load_registry(_write_models(tmp_path))
    with pytest.raises(UnregisteredDefaultModelError) as excinfo:
        registry.require_default()
    assert excinfo.value.model_id == UNREGISTERED
    assert excinfo.value.registered == [CATALOG_DEFAULT, OTHER]
    assert str(tmp_path / "models.yaml") in str(excinfo.value)  # names the file it read
    with caplog.at_level(logging.WARNING, logger="hearth.registry"):
        assert registry.default_id == CATALOG_DEFAULT  # lenient: an id for display code
        assert registry.default_id == CATALOG_DEFAULT
    warnings = [r.getMessage() for r in caplog.records if r.name == "hearth.registry"]
    assert len(warnings) == 1, warnings
    assert UNREGISTERED in warnings[0] and "refuse to start" in warnings[0]
    assert registry.ignored_default_override == UNREGISTERED


@pytest.mark.parametrize("value, expected", [
    (None, CATALOG_DEFAULT), ("", CATALOG_DEFAULT), ("  ", CATALOG_DEFAULT), (OTHER, OTHER),
])
def test_require_default_accepts_unset_empty_and_registered(
    value, expected, tmp_path, monkeypatch
):
    if value is None:
        monkeypatch.delenv("HEARTH_DEFAULT_MODEL", raising=False)
    else:
        monkeypatch.setenv("HEARTH_DEFAULT_MODEL", value)
    assert load_registry(_write_models(tmp_path)).require_default() == expected
