"""MLXEmbedder: disk-only model resolution, and all MLX work on the one MLX thread.

The embedder had no tests (adversarial review finding F8), so neither of the two properties
it gained on this branch — no download on load, no cross-thread MLX use — was checked. The
fake ``mlx_lm.load`` below enforces MLX's one-thread rule with the real error text.
"""

from __future__ import annotations

import importlib.machinery
import socket
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("huggingface_hub")

import huggingface_hub.constants  # noqa: E402

from hearth.config import Settings  # noqa: E402
from hearth.memory.embed import EmbeddingUnavailableError, MLXEmbedder  # noqa: E402
from hearth.providers import mlx as mlx_mod  # noqa: E402

REPO = "org/embedder"


class _Tokenizer:
    def encode(self, text):
        return [1 + (ord(c) % 7) for c in text] or [1]


class ThreadBoundModel:
    """Raises MLX's real error when called off the thread that loaded it."""

    def __init__(self) -> None:
        self.loaded_on = threading.current_thread()
        self.called_on: set[str] = set()

    def __call__(self, tokens):
        if threading.current_thread() is not self.loaded_on:
            raise RuntimeError("There is no Stream(gpu, 0) in current thread.")
        self.called_on.add(threading.current_thread().name)
        return mx.ones((1, tokens.shape[1], 4))


@pytest.fixture
def fake_mlx(monkeypatch, tmp_path):
    loads: list = []
    models: list[ThreadBoundModel] = []

    def load(path, **kwargs):
        loads.append(path)
        model = ThreadBoundModel()
        models.append(model)
        return model, _Tokenizer()

    module = types.ModuleType("mlx_lm")
    module.load = load
    module.__spec__ = importlib.machinery.ModuleSpec("mlx_lm", loader=None)
    monkeypatch.setitem(sys.modules, "mlx_lm", module)

    attempts: list = []

    def refuse(self, address, *args, **kwargs):
        attempts.append(address)
        raise OSError("network connect refused by test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    settings = Settings(home=tmp_path / "home")
    (tmp_path / "home" / "models").mkdir(parents=True)
    (tmp_path / "hub").mkdir()
    monkeypatch.setattr(mlx_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    yield loads, models, tmp_path
    assert attempts == [], f"network connect attempted: {attempts!r}"


def _plant(cache, repo=REPO):
    repo_dir = cache / f"models--{repo.replace('/', '--')}"
    snapshot = repo_dir / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text("abc123")
    return snapshot


def test_a_missing_embedding_model_is_refused_without_loading(fake_mlx):
    loads, _, _ = fake_mlx
    with pytest.raises(EmbeddingUnavailableError, match="not on disk"):
        MLXEmbedder(REPO).embed(["hello"])
    assert loads == []  # mlx_lm.load — the call that downloads a bare id — never ran


def test_the_load_error_reads_cleanly_and_names_the_real_fix(fake_mlx):
    """It used to render "...HEARTH_ALLOW_DOWNLOADS=1.. Pre-pull it from an unrestricted
    terminal (network is blocked here)." — a doubled period and sandbox advice that does not
    apply on the operator's machine."""
    with pytest.raises(EmbeddingUnavailableError) as excinfo:
        MLXEmbedder(REPO).embed(["hello"])
    message = str(excinfo.value)
    assert ".." not in message
    assert "unrestricted terminal" not in message and "network is blocked" not in message
    assert "hearth models pull" in message
    assert "B-011" in message and "HEARTH_EMBEDDER=hash" in message


def test_a_non_disk_load_failure_says_weights_must_be_on_disk(fake_mlx, monkeypatch):
    """mlx-lm itself refusing (B-011: "Model type bert not supported.") gets the on-disk
    requirement and the B-011 note, with one period."""
    loads, _, tmp_path = fake_mlx
    _plant(tmp_path / "hub")

    def bert_refused(path):
        raise ValueError("Model type bert not supported.")

    monkeypatch.setattr(sys.modules["mlx_lm"], "load", bert_refused)
    with pytest.raises(EmbeddingUnavailableError) as excinfo:
        MLXEmbedder(REPO).embed(["hello"])
    message = str(excinfo.value)
    assert "Model type bert not supported." in message and ".." not in message
    assert "`hearth models pull <registry id>`" in message and "B-011" in message


def test_a_cached_embedding_model_loads_from_its_path(fake_mlx):
    loads, _, tmp_path = fake_mlx
    snapshot = _plant(tmp_path / "hub")
    vectors = MLXEmbedder(REPO).embed(["hello"])
    assert loads == [str(snapshot)]
    assert len(vectors) == 1 and len(vectors[0]) == 4


def test_embedding_from_many_threads_runs_on_the_one_mlx_thread(fake_mlx):
    _, models, tmp_path = fake_mlx
    _plant(tmp_path / "hub")
    embedder = MLXEmbedder(REPO)
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda i: embedder.embed([f"text {i}"]), range(12)))
    assert all(len(r) == 1 for r in results)
    assert len(models) == 1  # loaded once
    assert len(models[0].called_on) == 1
    assert next(iter(models[0].called_on)).startswith("hearth-mlx")
