"""B-024: ``train_lora_real.sh``'s base-model pre-check sees ``hearth models pull`` output.

The script delegates to ``scripts/check_base_on_disk.py``; these run its ``main`` with every
socket connect refused AND counted (the tests/test_offline_model_resolution.py instrument —
an exception type cannot tell a disk-only lookup from one that tried the network).
"""

from __future__ import annotations

import importlib.util
import socket
from pathlib import Path

import pytest

pytest.importorskip("huggingface_hub")

import huggingface_hub.constants  # noqa: E402

from hearth.config import get_settings  # noqa: E402

_REPO = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO / "scripts" / "train_lora_real.sh"
_CHECK = _REPO / "scripts" / "check_base_on_disk.py"
REPO_ID = "org/pulled-only-model"


def _load_check():
    spec = importlib.util.spec_from_file_location("check_base_on_disk", _CHECK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def connects(monkeypatch):
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
def hearth_home(tmp_path, monkeypatch):
    """A temp HEARTH_HOME (what a fresh script process would read) and an EMPTY hub cache."""
    home, hub = tmp_path / "hearth-home", tmp_path / "hub"
    (home / "models").mkdir(parents=True)
    hub.mkdir()
    monkeypatch.setenv("HEARTH_HOME", str(home))
    monkeypatch.delenv("HEARTH_ALLOW_DOWNLOADS", raising=False)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(hub))
    get_settings.cache_clear()
    yield home
    get_settings.cache_clear()


def _plant(cache: Path, repo: str = REPO_ID) -> Path:
    repo_dir = cache / f"models--{repo.replace('/', '--')}"
    snapshot = repo_dir / "snapshots" / "deadbeef"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text("deadbeef")
    return snapshot


def test_a_model_only_in_hearths_models_dir_passes(hearth_home, capsys):
    snapshot = _plant(hearth_home / "models")
    assert _load_check().main([REPO_ID]) == 0
    assert capsys.readouterr().out.strip() == str(snapshot)


def test_an_absent_model_fails_with_the_pull_hint(hearth_home, capsys):
    assert _load_check().main([REPO_ID]) == 3
    assert f"hearth models pull {REPO_ID}" in capsys.readouterr().err


def test_the_script_uses_the_checker_not_a_hub_only_lookup():
    text = _SCRIPT.read_text()
    assert "check_base_on_disk.py" in text
    assert "snapshot_download" not in text, "a hub-only cache check is back in the script"


# -- B-015: --promote goes through the real gate, and is checked before any training --------

import subprocess as _sp  # noqa: E402
from pathlib import Path as _P  # noqa: E402

_SCRIPT = _P(__file__).resolve().parent.parent / "scripts" / "train_lora_real.sh"


def _run_script(tmp_path, *args):
    data = tmp_path / "d.jsonl"
    data.write_text("{}\n")
    return _sp.run(["bash", str(_SCRIPT), "--data", str(data), *args],
                   capture_output=True, text=True, timeout=60)


def test_promote_without_a_golden_set_and_prereg_refuses_before_training(tmp_path):
    r = _run_script(tmp_path, "--promote")
    assert r.returncode == 2
    assert "requires --golden and --prereg" in r.stderr
    assert "Training" not in r.stdout  # no GPU spent on a run that could not be promoted


def test_the_removed_typed_score_flags_are_refused(tmp_path):
    for flag in ("--candidate-score", "--incumbent-score"):
        r = _run_script(tmp_path, flag, "0.9")
        assert r.returncode == 2 and "was removed" in r.stderr, flag
