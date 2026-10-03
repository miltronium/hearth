"""Every load path outside serving is disk-only too: `hearth train`, `hearth models convert`,
`hearth models export-coreml` (and the census that keeps it that way).

Same instrument as tests/test_offline_model_resolution.py: real hub-layout caches planted on
disk, every socket connect refused AND counted, and any recorded attempt fails the test at
teardown. The exception type proves nothing on its own — huggingface_hub reports a refused
connection as the same LocalEntryNotFoundError a cache miss raises — so each test asserts
the outcome directly: the tool received a local PATH (never a bare repo id), and a child
process received an env with the hub pinned offline. ``subprocess.run`` and
``transformers`` are faked at the boundary, so nothing is trained or converted.
"""

from __future__ import annotations

import ast
import importlib.machinery
import importlib.util
import os
import socket
import subprocess
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("huggingface_hub")

import huggingface_hub.constants  # noqa: E402

from hearth import coreml  # noqa: E402
from hearth.config import Settings  # noqa: E402
from hearth.convert import ConvertConfig, convert  # noqa: E402
from hearth.providers import mlx as mlx_mod  # noqa: E402
from hearth.providers.mlx import ModelNotOnDiskError  # noqa: E402
from hearth.training.dataset import build_dataset  # noqa: E402
from hearth.training.lora import LoRAConfig, train  # noqa: E402

REPO = "org/model-a"
OFFLINE_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def connects(monkeypatch):
    """Refuse every socket connect the way a dead network does, and record it."""
    attempts: list = []

    def refuse(self, address, *args, **kwargs):
        attempts.append(address)
        raise OSError(f"network connect refused by test: {address!r}")

    for var in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY",
                *OFFLINE_VARS):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    yield attempts
    assert attempts == [], f"network connect attempted: {attempts!r}"


def _isolate(tmp_path, monkeypatch, *, allow_downloads=False):
    home, hub = tmp_path / "home", tmp_path / "hub"
    (home / "models").mkdir(parents=True)
    hub.mkdir()
    settings = Settings(home=home, allow_downloads=allow_downloads)
    monkeypatch.setattr(mlx_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(hub))
    return home, hub


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    return _isolate(tmp_path, monkeypatch)


def _plant(cache: Path, repo: str = REPO) -> Path:
    repo_dir = cache / f"models--{repo.replace('/', '--')}"
    snapshot = repo_dir / "snapshots" / "deadbeef"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text("deadbeef")
    return snapshot


@pytest.fixture
def launches(monkeypatch):
    """Fake ``subprocess.run`` (record command + env) and make ``mlx_lm`` look installed."""
    calls: list[tuple[list[str], dict | None]] = []

    def fake_run(command, *args, env=None, **kwargs):
        calls.append((list(command), env))
        return subprocess.CompletedProcess(command, 0)

    real_find_spec = importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        if name == "mlx_lm":
            return importlib.machinery.ModuleSpec("mlx_lm", loader=None)
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    return calls


def _flag(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def _assert_offline_child(env: dict | None) -> None:
    assert env is not None, "the child inherited the parent env wholesale (env=None)"
    for var in OFFLINE_VARS:
        assert env.get(var) == "1", f"child env lacks {var}=1"
        # ...and it was set for the CHILD only, never by mutating this process.
        assert var not in os.environ, f"parent os.environ was mutated: {var}"


# --- hearth train (mlx_lm.lora in a child process) -----------------------------------------


def _lora_config(tmp_path, base=REPO) -> LoRAConfig:
    pairs = [(f"prompt {i}", f"completion {i}") for i in range(40)]  # valid split = 4
    dataset = build_dataset("extract", pairs, created_at="2026-10-02T00:00:00Z")
    return LoRAConfig(base_model=base, task="extract", dataset=dataset,
                      output_dir=tmp_path / "run", iters=1)


def test_train_hands_mlx_lm_lora_a_local_path_and_an_offline_child_env(
    isolated, tmp_path, launches
):
    _, hub = isolated
    snapshot = _plant(hub)
    outcome = train(_lora_config(tmp_path))  # the REAL default runner
    [(command, env)] = launches
    assert command[1:3] == ["-m", "mlx_lm.lora"]
    assert _flag(command, "--model") == str(snapshot)  # a path, never the bare repo id
    assert REPO not in command
    _assert_offline_child(env)
    # What HEARTH records keeps the model ID: the adapter registry matches base_model
    # against the served model id, so a cache path there would orphan the adapter.
    assert outcome.base_model == REPO
    assert _flag(outcome.args, "--model") == REPO


def test_train_on_a_model_not_on_disk_fails_before_launching_anything(
    isolated, tmp_path, launches
):
    with pytest.raises(ModelNotOnDiskError, match="hearth models pull"):
        train(_lora_config(tmp_path))
    assert launches == []  # mlx_lm.lora — the thing that downloads — never started


def test_train_honours_the_download_opt_in(tmp_path, monkeypatch, launches):
    _isolate(tmp_path, monkeypatch, allow_downloads=True)
    train(_lora_config(tmp_path))
    [(command, env)] = launches
    # Opted in: the id is handed on and the child is NOT pinned offline (the chosen path).
    assert _flag(command, "--model") == REPO
    assert not any(var in env for var in OFFLINE_VARS)


def test_cli_train_reports_a_missing_base_cleanly(isolated, tmp_path, launches):
    from typer.testing import CliRunner

    from hearth.cli import app
    from hearth.training.dataset import write_dataset

    data = write_dataset(_lora_config(tmp_path).dataset, tmp_path / "data.jsonl")
    result = CliRunner().invoke(
        app,
        ["train", "--task", "extract", "--base", REPO, "--data", str(data),
         "--out", str(tmp_path / "run"), "--no-register"],
        env={"COLUMNS": "300", "HEARTH_HOME": str(tmp_path / "home")},
    )
    assert result.exit_code == 1, result.output
    assert "not on disk" in result.output
    assert launches == []


# --- hearth models convert (mlx_lm convert in a child process) ------------------------------


def test_convert_hands_mlx_lm_a_local_path_and_an_offline_child_env(
    isolated, tmp_path, launches
):
    home, _ = isolated
    snapshot = _plant(home / "models")  # pulled by `hearth models pull` -> HEARTH's dir
    out = tmp_path / "out" / "converted"
    convert(ConvertConfig(source=REPO, output_dir=out, q_bits=4))  # the REAL default runner
    [(command, env)] = launches
    assert command[1:4] == ["-m", "mlx_lm", "convert"]
    assert _flag(command, "--hf-path") == str(snapshot)
    assert REPO not in command
    assert _flag(command, "--mlx-path") == str(out)
    _assert_offline_child(env)
    # mlx_lm.convert refuses an existing --mlx-path, so only the parent may exist.
    assert out.parent.is_dir() and not out.exists()


def test_convert_of_a_model_not_on_disk_fails_before_launching_anything(
    isolated, tmp_path, launches
):
    with pytest.raises(ModelNotOnDiskError):
        convert(ConvertConfig(source=REPO, output_dir=tmp_path / "out"))
    assert launches == []


def test_convert_honours_the_download_opt_in(tmp_path, monkeypatch, launches):
    _isolate(tmp_path, monkeypatch, allow_downloads=True)
    convert(ConvertConfig(source=REPO, output_dir=tmp_path / "out", quantize=False))
    [(command, env)] = launches
    assert _flag(command, "--hf-path") == REPO
    assert not any(var in env for var in OFFLINE_VARS)


# --- hearth models export-coreml (transformers, in-process) ---------------------------------


@pytest.fixture
def fake_transformers(monkeypatch):
    """A ``transformers`` whose ``from_pretrained`` records exactly what it was handed."""
    calls: list[tuple[str, str, dict]] = []

    def auto(kind):
        class Auto:
            @staticmethod
            def from_pretrained(path, **kwargs):
                calls.append((kind, path, kwargs))
                return object()

        return Auto

    module = types.ModuleType("transformers")
    module.AutoConfig = auto("config")
    module.AutoTokenizer = auto("tokenizer")
    module.AutoModelForCausalLM = auto("model")
    monkeypatch.setitem(sys.modules, "transformers", module)
    return calls


def test_export_coreml_loads_transformers_from_a_local_path_only(isolated, fake_transformers):
    _, hub = isolated
    snapshot = _plant(hub)
    coreml.load_hf_source(REPO, attn_implementation="eager")
    assert [kind for kind, _, _ in fake_transformers] == ["config", "tokenizer", "model"]
    for kind, path, kwargs in fake_transformers:
        assert path == str(snapshot), f"{kind} got {path!r}, not the local path"
        assert kwargs.get("local_files_only") is True, f"{kind} may reach the hub"
    assert fake_transformers[2][2]["attn_implementation"] == "eager"  # caller kwargs kept


def test_export_coreml_of_a_model_not_on_disk_never_touches_transformers(
    isolated, fake_transformers
):
    with pytest.raises(ModelNotOnDiskError):
        coreml.load_hf_source(REPO)
    assert fake_transformers == []


def test_cli_export_coreml_reports_a_missing_source_cleanly(isolated, tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from hearth.cli import app

    # Run the real export dispatch up to the load: coremltools "installed", Approach A.
    monkeypatch.setattr(coreml, "_plain_export_runner",
                        lambda config: coreml.load_hf_source(config.source))
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name, *a, **k: object() if name == "coremltools" else real_find_spec(name, *a, **k),
    )
    result = CliRunner().invoke(
        app, ["models", "export-coreml", "--source", REPO, "--out", str(tmp_path / "m.mlpackage")],
        env={"COLUMNS": "300"},
    )
    assert result.exit_code == 1, result.output
    assert "not on disk" in result.output


# --- the census: `hearth models pull` is the only download path ----------------------------

_HUB_FETCHERS = {"snapshot_download", "hf_hub_download"}


def _calls(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            owner = node
            while owner in parents and not isinstance(owner, ast.FunctionDef):
                owner = parents[owner]
            yield name, node, getattr(owner, "name", "<module>")


def _local_only(call: ast.Call) -> bool:
    return any(
        kw.arg == "local_files_only"
        and isinstance(kw.value, ast.Constant) and kw.value.value is True
        for kw in call.keywords
    )


def test_hearth_models_pull_is_the_only_call_that_may_download():
    """Every hub fetch in src/ and scripts/ is local_files_only — except `models pull`.

    A census, not an outcome: it exists so a NEW load path cannot be added silently. The
    outcome tests above are what prove the existing paths actually stay off the network.
    """
    downloading = []
    for path in sorted([*ROOT.glob("src/hearth/**/*.py"), *ROOT.glob("scripts/*.py")]):
        for name, call, owner in _calls(path):
            if name in _HUB_FETCHERS and not _local_only(call):
                downloading.append(f"{path.relative_to(ROOT)}:{call.lineno} {owner}")
            if name == "from_pretrained" and not (
                _local_only(call) or (path.name == "coreml.py" and owner == "load_hf_source")
            ):
                downloading.append(f"{path.relative_to(ROOT)}:{call.lineno} {owner}")
    assert [d.split(" ")[1] for d in downloading] == ["models_pull"], downloading


def test_every_mlx_lm_load_is_handed_a_resolved_path():
    """``mlx_lm.load(<repo-id>)`` downloads; each call site must wrap resolve_local_model."""
    unresolved = []
    for path in sorted([*ROOT.glob("src/hearth/**/*.py"), *ROOT.glob("scripts/*.py")]):
        source = path.read_text(encoding="utf-8")
        if "mlx_lm" not in source:
            continue
        for name, call, owner in _calls(path):
            if name != "load" or not isinstance(call.func, ast.Name) or not call.args:
                continue
            first = call.args[0]
            resolved = (
                isinstance(first, ast.Call)
                and getattr(first.func, "id", None) == "resolve_local_model"
            )
            if not resolved:
                unresolved.append(f"{path.relative_to(ROOT)}:{call.lineno} {owner}")
    assert unresolved == [], unresolved
