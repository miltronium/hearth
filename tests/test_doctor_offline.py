"""`hearth doctor --offline` — "is it safe to use HEARTH offline right now?"

Each test builds a fixture machine (isolated HEARTH home + hub cache, a routing profile on
disk, an injected registry) and asserts the verdict. The two mutants the command exists to
catch are here explicitly: a remote added to the routing profile, and a resolver that
reaches for the network — the latter still ends in ModelNotOnDiskError, so only the connect
count the doctor takes can see it.
"""

from __future__ import annotations

import socket

import pytest

pytest.importorskip("huggingface_hub")

import huggingface_hub  # noqa: E402
import huggingface_hub.constants  # noqa: E402

from hearth import doctor as doctor_mod  # noqa: E402
from hearth.config import Settings  # noqa: E402
from hearth.doctor import all_fatal_passed, run_offline_checks  # noqa: E402
from hearth.providers import mlx as mlx_mod  # noqa: E402
from hearth.registry import ModelEntry, Registry  # noqa: E402

DEFAULT = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
SMALL = "mlx-community/Qwen2.5-3B-Instruct-4bit"  # in the real registry: local_model validates

SAFE_PROFILE = """
defaults: {local_model: auto, remote: none, remote_budget_tokens_per_day: 0}
classes:
  chat:    {backend: local, escalate: never}
  extract: {backend: local, escalate: never}
remotes: {}
"""

LEAKY_PROFILE = """
defaults: {local_model: auto, remote: frontier}
classes:
  chat:   {backend: local, escalate: never}
  reason: {backend: remote, escalate: always}
remotes:
  frontier: {protocol: anthropic, model: some-frontier-model, api_key_env: X_KEY}
"""


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Any connect outside the doctor's own audit is refused and fails the test."""
    attempts: list = []

    def refuse(self, address, *args, **kwargs):
        attempts.append(address)
        raise OSError(f"network connect refused by test: {address!r}")

    for var in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY",
                "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HEARTH_DEFAULT_MODEL",
                "HEARTH_ROUTING_YAML"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    yield
    assert attempts == [], f"network connect attempted outside the audit: {attempts!r}"


def _plant(cache, repo):
    repo_dir = cache / f"models--{repo.replace('/', '--')}"
    snapshot = repo_dir / "snapshots" / "deadbeef"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_text("weights")
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text("deadbeef")
    return snapshot


@pytest.fixture
def machine(tmp_path, monkeypatch):
    """A fixture machine whose default model is on disk and whose profile is no-egress."""
    home, hub = tmp_path / "home", tmp_path / "hub"
    (home / "models").mkdir(parents=True)
    hub.mkdir()
    _plant(home / "models", DEFAULT)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(hub))
    profile = tmp_path / "routing.yaml"
    profile.write_text(SAFE_PROFILE)

    class Machine:
        def __init__(self):
            self.home, self.hub, self.profile = home, hub, profile
            self.registry = Registry(
                [ModelEntry(DEFAULT, "mlx", "4bit", 0, 0.0, ["chat"], DEFAULT),
                 ModelEntry(SMALL, "mlx", "4bit", 0, 0.0, ["chat"], SMALL),
                 ModelEntry("echo", "echo", "none", 0, 0.0, ["chat"], "")],
                DEFAULT,
            )

        def run(self, **settings_kw):
            settings = Settings(home=home, backend="mlx", **settings_kw)
            # The resolver reads process settings; point it at the same fixture machine.
            monkeypatch.setattr(mlx_mod, "get_settings", lambda: settings)
            environ = {"HEARTH_ALLOW_DOWNLOADS": "1"} if settings.allow_downloads else {}
            checks = run_offline_checks(
                settings, policy_path=profile, registry=self.registry, environ=environ
            )
            return {c.name: c for c in checks}, all_fatal_passed(checks)

    return Machine()


def test_a_no_egress_profile_with_every_model_on_disk_is_safe(machine):
    checks, safe = machine.run()
    assert safe, [f"{c.name}: {c.detail}" for c in checks.values() if not c.ok]
    assert checks["routing_profile"].ok
    assert checks["serving_resolution"].ok
    assert checks[f"model {DEFAULT}"].ok
    for path in ("hearth train", "hearth models convert", "hearth models export-coreml"):
        assert checks[f"load path: {path}"].ok, checks[f"load path: {path}"].detail


def test_mutant_a_remote_in_the_profile_is_unsafe_and_names_who_escapes(machine):
    machine.profile.write_text(LEAKY_PROFILE)
    checks, safe = machine.run()
    assert not safe
    routing = checks["routing_profile"]
    assert not routing.ok and routing.fatal
    assert "frontier" in routing.detail and "reason" in routing.detail


def test_mutant_only_a_remote_added_with_every_class_still_local_is_unsafe(machine):
    """Adding just a `remotes:` entry (and pointing the default at it) is the minimal edit
    that gives the router somewhere to send; it must go red even with no class changed."""
    machine.profile.write_text(SAFE_PROFILE.replace("remote: none", "remote: frontier").replace(
        "remotes: {}",
        "remotes:\n  frontier: {protocol: openai, model: m, base_url: 'https://example.invalid'}",
    ))
    checks, safe = machine.run()
    assert not safe
    assert "frontier" in checks["routing_profile"].detail


def test_mutant_a_resolver_that_attempts_a_connect_is_unsafe(machine, monkeypatch):
    """The regression the exception type cannot show: a lookup with the network ON still
    ends in ModelNotOnDiskError for a missing model, and still finds a cached one."""
    real = huggingface_hub.snapshot_download

    def online(*args, **kwargs):
        kwargs["local_files_only"] = False
        return real(*args, **kwargs)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", online)
    checks, safe = machine.run()
    assert not safe
    assert not checks["serving_resolution"].ok
    assert "attempted the network" in checks["serving_resolution"].detail
    assert "connect" in checks[f"model {DEFAULT}"].detail
    assert not checks["load path: hearth train"].ok
    assert "connect(s) attempted" in checks["load path: hearth train"].detail


def test_a_default_model_not_on_disk_is_unsafe(machine):
    machine.registry = Registry(machine.registry.list(), SMALL)  # SMALL is not planted
    checks, safe = machine.run()
    assert not safe
    assert f"hearth models pull {SMALL}" in checks[f"model {SMALL}"].detail


def test_every_class_local_model_must_be_on_disk(machine):
    machine.profile.write_text(SAFE_PROFILE.replace(
        "extract: {backend: local, escalate: never}",
        f"extract: {{backend: local, escalate: never, local_model: {SMALL}}}",
    ))
    checks, safe = machine.run()
    assert not safe
    assert "class extract" in checks[f"model {SMALL}"].detail
    _plant(machine.hub, SMALL)  # now it is on disk (in the hub cache) -> safe
    checks, safe = machine.run()
    assert safe and checks[f"model {SMALL}"].ok


def test_a_snapshot_without_weights_is_not_loadable(machine):
    weights = next((machine.home / "models").rglob("model.safetensors"))
    weights.unlink()
    checks, safe = machine.run()
    assert not safe
    assert "no weight file" in checks[f"model {DEFAULT}"].detail


def test_a_load_path_whose_child_is_not_pinned_offline_is_unsafe(machine, monkeypatch):
    monkeypatch.setattr(mlx_mod, "model_load_env", lambda allow=None, base=None: {})
    checks, safe = machine.run()
    assert not safe
    for path in ("hearth train", "hearth models convert"):
        assert "not pinned offline" in checks[f"load path: {path}"].detail
    assert checks["load path: hearth models export-coreml"].ok  # in-process, unaffected


def test_a_load_path_that_resolves_but_hands_over_the_bare_id_is_unsafe(machine, monkeypatch):
    """Resolution ran (so an absent model still fails) but its result was discarded: the
    tool would get the repo id and download. Only the on-disk probe can see this."""
    from hearth.training import lora

    real = lora.runner_invocation

    def discards(args, allow_downloads=None):
        command, env = real(args, allow_downloads=allow_downloads)
        command[command.index("--model") + 1] = args[args.index("--model") + 1]
        return command, env

    monkeypatch.setattr(lora, "runner_invocation", discards)
    checks, safe = machine.run()
    assert not safe
    assert f"handed '{DEFAULT}'" in checks["load path: hearth train"].detail


def test_downloads_opted_in_is_unsafe_everywhere_it_matters(machine):
    checks, safe = machine.run(allow_downloads=True)
    assert not safe
    assert not checks["allow_downloads"].ok
    assert not checks["serving_resolution"].ok
    # Measured, not inferred from the flag: each load path really would hand on a repo id.
    assert "did not fail" in checks["load path: hearth models convert"].detail


@pytest.mark.parametrize("host, ok", [
    ("127.0.0.1", True), ("localhost", True), ("::1", True), ("127.0.0.2", True),
    ("0.0.0.0", False), ("192.168.1.10", False), ("myhost.local", False),
])
def test_bind_host_must_be_loopback(machine, host, ok):
    checks, _ = machine.run(host=host)
    assert checks["bind_host"].ok is ok


def test_a_plugin_backend_is_not_vouched_for(machine):
    settings = Settings(home=machine.home, backend="some-plugin")
    checks = {c.name: c for c in run_offline_checks(
        settings, policy_path=machine.profile, registry=machine.registry, environ={}
    )}
    assert not checks["backend"].ok


def test_cli_exits_nonzero_when_unsafe_and_zero_when_safe(machine, monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from hearth.cli import app

    settings = Settings(home=machine.home, backend="echo")
    monkeypatch.setattr(doctor_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(mlx_mod, "get_settings", lambda: settings)
    monkeypatch.setenv("HEARTH_ROUTING_YAML", str(machine.profile))  # what the router reads
    runner = CliRunner()
    safe = runner.invoke(app, ["doctor", "--offline"], env={"COLUMNS": "250"})
    assert safe.exit_code == 0, safe.output
    assert "SAFE offline" in safe.output

    machine.profile.write_text(LEAKY_PROFILE)
    unsafe = runner.invoke(app, ["doctor", "--offline"], env={"COLUMNS": "250"})
    assert unsafe.exit_code == 1, unsafe.output
    assert "UNSAFE offline" in unsafe.output and "routing_profile" in unsafe.output


def test_a_rung_the_model_pool_would_refuse_is_not_safe(machine):
    """The routing loader accepts any REGISTERED id as a rung (echo is registered), but the
    mlx backend serves through ModelPool, which 404s an id that is not a chat model of its
    backend. Such a rung must fail here, not at the first classified request."""
    machine.profile.write_text(SAFE_PROFILE.replace(
        "extract: {backend: local, escalate: never}",
        "extract: {backend: local, escalate: never, local_model: echo}",
    ))
    checks, safe = machine.run()
    assert not safe
    assert not checks["model echo"].ok
    assert "NOT servable" in checks["model echo"].detail
    assert "class extract" in checks["model echo"].detail
    assert checks[f"model {DEFAULT}"].ok
