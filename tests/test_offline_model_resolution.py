"""A model load must never be a download by accident (providers/mlx.py:resolve_local_model).

``mlx_lm.load`` given a bare repo id calls ``snapshot_download`` with the network ON. These
tests plant real hub-layout caches on disk and refuse every socket connect — and COUNT them,
failing any test that attempted one. Counting is the instrument because the exception type
is not: huggingface_hub turns a refused connection into ``LocalEntryNotFoundError``, the very
"not cached" error a cache miss raises, so a resolver that reached for the network can still
end in ModelNotOnDiskError looking exactly like a disk-only one (measured: 8 connects).
"""

from __future__ import annotations

import socket
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("huggingface_hub")

import huggingface_hub.constants  # noqa: E402

from hearth.config import Settings  # noqa: E402
from hearth.providers import mlx as mlx_mod  # noqa: E402
from hearth.providers.mlx import (  # noqa: E402
    MLXProvider,
    ModelNotOnDiskError,
    resolve_local_model,
)

REPO = "org/model-a"


@pytest.fixture(autouse=True)
def connects(monkeypatch):
    """Refuse every socket connect the way a dead network does (OSError), and record it.

    Proxy variables are cleared so a request would connect directly rather than to a local
    proxy. Any test that leaves an attempt recorded fails at teardown.
    """
    attempts: list = []

    def refuse(self, address, *args, **kwargs):
        attempts.append(address)
        raise OSError(f"network connect refused by test: {address!r}")

    for var in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    yield attempts
    assert attempts == [], f"network connect attempted: {attempts!r}"


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """An isolated HEARTH home and an isolated (empty) default hub cache."""
    home, hub = tmp_path / "home", tmp_path / "hub"
    (home / "models").mkdir(parents=True)
    hub.mkdir()
    settings = Settings(home=home)
    monkeypatch.setattr(mlx_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(hub))
    return home, hub, settings


def _plant(cache: Path, repo: str = REPO) -> Path:
    """A minimal real hub-layout entry that ``snapshot_download(local_files_only)`` resolves."""
    repo_dir = cache / f"models--{repo.replace('/', '--')}"
    snapshot = repo_dir / "snapshots" / "deadbeef"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text("deadbeef")
    return snapshot


def test_a_model_in_hearths_directory_resolves_there(isolated):
    home, hub, _ = isolated
    snapshot = _plant(home / "models")
    _plant(hub)  # present in both: HEARTH's own directory wins
    assert Path(resolve_local_model(REPO)) == snapshot


def test_a_model_only_in_the_hub_cache_resolves_to_a_local_path(isolated):
    _, hub, _ = isolated
    snapshot = _plant(hub)
    resolved = resolve_local_model(REPO)
    # The outcome that matters: mlx_lm.load gets a PATH (no download), not the bare repo id.
    assert Path(resolved) == snapshot
    assert resolved != REPO


def test_a_missing_model_raises_instead_of_reaching_the_network(isolated):
    with pytest.raises(ModelNotOnDiskError) as excinfo:
        resolve_local_model(REPO)
    message = str(excinfo.value)
    assert REPO in message
    assert "hearth models pull" in message
    assert "HEARTH_ALLOW_DOWNLOADS" in message


def test_downloads_are_off_unless_the_operator_opts_in(isolated, monkeypatch):
    _, _, settings = isolated
    assert settings.allow_downloads is False
    # Opted in: the bare id is handed on so mlx_lm may fetch it (the explicit, chosen path).
    assert resolve_local_model(REPO, allow_downloads=True) == REPO
    monkeypatch.setenv("HEARTH_ALLOW_DOWNLOADS", "1")
    assert Settings().allow_downloads is True


def test_an_existing_path_is_used_as_is(isolated, tmp_path):
    local = tmp_path / "some-converted-model"
    local.mkdir()
    assert resolve_local_model(str(local)) == str(local)


def _fake_mlx_lm(monkeypatch, calls: list):
    """Install a fake ``mlx_lm`` whose ``load`` records exactly what it was handed."""
    import importlib.machinery

    class FakeTokenizer:
        eos_token_ids = set()

        def encode(self, text):
            return list(text)

    def fake_load(path_or_repo, **kwargs):
        calls.append(path_or_repo)
        return object(), FakeTokenizer()

    module = types.ModuleType("mlx_lm")
    module.load = fake_load
    module.__spec__ = importlib.machinery.ModuleSpec("mlx_lm", loader=None)
    monkeypatch.setitem(sys.modules, "mlx_lm", module)
    monkeypatch.setattr(MLXProvider, "_ensure_stop_tokens", lambda self, tok: None)


def test_the_provider_never_hands_mlx_lm_a_bare_repo_id(isolated, monkeypatch):
    _, hub, _ = isolated
    snapshot = _plant(hub)
    calls: list = []
    _fake_mlx_lm(monkeypatch, calls)
    MLXProvider(REPO)._load_variant(None)
    assert calls == [str(snapshot)]


def test_the_provider_refuses_to_load_a_missing_model(isolated, monkeypatch):
    calls: list = []
    _fake_mlx_lm(monkeypatch, calls)
    with pytest.raises(ModelNotOnDiskError):
        MLXProvider(REPO)._load_variant(None)
    assert calls == []  # mlx_lm.load — the thing that downloads — was never reached


def test_the_connect_counter_catches_a_resolver_that_goes_online(isolated, monkeypatch, connects):
    """Validate the instrument: a regressed resolver that looks up with the network ON IS
    detected by the connect count — even though it still ends in ModelNotOnDiskError, which
    is why no test here may rely on the exception type alone."""
    import huggingface_hub

    real = huggingface_hub.snapshot_download

    def online(*args, **kwargs):  # the regression: a lookup with the network ON
        kwargs["local_files_only"] = False
        return real(*args, **kwargs)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", online)
    with pytest.raises(Exception):
        resolve_local_model(REPO)
    assert connects, "the instrument failed to see a lookup that went online"
    connects.clear()  # the attempt was the point of this test, not a failure of it
