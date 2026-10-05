"""Third-round attacks on the promotion gate (B-120..) — each must now be REFUSED.

A reviewer's attack suite (round 4 of review, reusing ``test_promotion_evidence.World``)
asserted that each of these PROMOTED on the code at 3caecb0. Every test below is that attack
ported to assert the refusal; each was run against 3caecb0 first and promoted there.

    H2  `git replace` / info/grafts / a forged commit-graph fake "the prereg commit is an
        ancestor of the HEAD recorded at the first measurement" (B-120)
    H3  the same served weights re-registered under a new id with a junk file beside them
        were a "fresh, never-measured" adapter (B-121)
"""

from __future__ import annotations

import os
import struct
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import test_promotion_evidence as pe
from test_promotion_evidence import World

from hearth.providers.base import GenResult


class _WeightsProvider:
    """Answers correctly iff the adapter dir's adapters.safetensors holds the GOOD weights."""

    name = "fake"

    def generate(self, req):
        prompt = req.messages[-1].content
        good = False
        if req.adapter:
            w = Path(req.adapter) / "adapters.safetensors"
            good = w.exists() and w.read_bytes() == b"GOOD"
        return GenResult(text=pe.ANSWERS[prompt] if good else "A", model=req.model,
                         backend="fake")


@pytest.fixture(autouse=True)
def fake_provider(monkeypatch):
    monkeypatch.setattr("hearth.cli.select_provider", lambda settings: _WeightsProvider())
    pe.allow_test_backends(monkeypatch)


def _flat(result) -> str:
    return " ".join(result.output.split())


def _adapter(world: World, aid: str, *, extra_file: str | None = None) -> Path:
    path = world.tmp / "weights" / aid
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapters.safetensors").write_bytes(b"GOOD")
    if extra_file:
        (path / extra_file).write_text("notes")
    world.store.register(aid, base_model=pe.BASE, task="extract", train_run_id="r",
                         adapter_path=str(path))
    return path


def _g(repo: Path, *args: str, env: dict | None = None, inp: str | None = None) -> str:
    full = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@e",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@e", **(env or {})}
    return subprocess.run(["git", *args], cwd=str(repo), check=True, capture_output=True,
                          text=True, env=full, input=inp).stdout.strip()


def _refused(result, world: World, aid: str, text: str = "") -> None:
    flat = _flat(result)
    assert result.exit_code == 1, flat
    assert "Promoted" not in flat
    assert world.status(aid) == "candidate", flat
    if text:
        assert text in flat, flat


def _peek(world: World, aid: str = "a1"):
    """The exploratory first measurement: the operator sees PASS before any bar exists."""
    peek = world.eval(aid)
    assert peek.exit_code == 0 and "PASS" in _flat(peek), _flat(peek)
    return peek


def _backdated_bar_commit(world: World) -> tuple[str, str]:
    """AFTER seeing PASS: the bar, in a fabricated commit dated a month ago (parentless)."""
    world.write_prereg()
    _g(world.repo, "add", "prereg.yaml")
    tree = _g(world.repo, "write-tree")
    old = (datetime.now(tz=UTC) - timedelta(days=30)).isoformat()
    dates = {"GIT_COMMITTER_DATE": old, "GIT_AUTHOR_DATE": old}
    return tree, _g(world.repo, "commit-tree", tree, "-m", "bar", env=dates)


# -- H2: history rewriting cannot fake "committed before the first measurement" (B-120) ----


def test_H2_git_replace_cannot_graft_a_backdated_bar_under_the_recorded_head(tmp_path):
    w = World(tmp_path)
    _adapter(w, "a1")
    h1 = w.commit("golden.jsonl")
    _peek(w)  # first measurement: HEAD = h1, no bar anywhere
    tree, bar = _backdated_bar_commit(w)
    old = (datetime.now(tz=UTC) - timedelta(days=30)).isoformat()
    fake_h1 = _g(w.repo, "commit-tree", tree, "-p", bar, "-m", "golden",
                 env={"GIT_COMMITTER_DATE": old, "GIT_AUTHOR_DATE": old})
    _g(w.repo, "replace", h1, fake_h1)
    assert _g(w.repo, "rev-parse", "HEAD") == h1  # plain git now "sees" the bar in h1
    result = w.eval("a1", "--prereg", str(w.prereg), "--promote")
    _refused(result, w, "a1", "replace refs")


def test_H2_replace_objects_are_ignored_even_where_the_refusal_is_not_reached(tmp_path):
    """Layer two on its own: every gate git read runs with replace objects switched off."""
    from hearth.training import prereg

    w = World(tmp_path)
    h1 = w.commit("golden.jsonl")
    tree, bar = _backdated_bar_commit(w)
    fake_h1 = _g(w.repo, "commit-tree", tree, "-p", bar, "-m", "golden")
    _g(w.repo, "replace", h1, fake_h1)
    _g(w.repo, "config", "core.useReplaceRefs", "true")  # repo config cannot turn it back on
    assert not prereg._is_ancestor(bar, h1, root=str(w.repo))
    assert "prereg.yaml" not in prereg._git(["ls-tree", "--name-only", h1], cwd=str(w.repo))


def test_H2_an_info_grafts_file_cannot_rewrite_the_recorded_heads_parents(tmp_path):
    w = World(tmp_path)
    _adapter(w, "a1")
    h1 = w.commit("golden.jsonl")
    _peek(w)
    tree, bar = _backdated_bar_commit(w)
    old = (datetime.now(tz=UTC) - timedelta(days=30)).isoformat()
    # A merge of h1 and the bar, tree-identical to the bar: `git log -- prereg.yaml` follows
    # the bar (TREESAME parent), and the graft makes the bar an ancestor of h1.
    merge = _g(w.repo, "commit-tree", tree, "-p", h1, "-p", bar, "-m", "merge",
               env={"GIT_COMMITTER_DATE": old, "GIT_AUTHOR_DATE": old})
    _g(w.repo, "reset", "-q", "--hard", merge)
    common = Path(w.repo) / _g(w.repo, "rev-parse", "--git-common-dir")
    (common / "info").mkdir(exist_ok=True)
    (common / "info" / "grafts").write_text(f"{h1} {bar}\n")
    result = w.eval("a1", "--prereg", str(w.prereg), "--promote")
    _refused(result, w, "a1", "info/grafts")


def _forge_commit_graph(repo: Path, child: str, parent: str) -> None:
    """Hand-edit the commit-graph so ``child`` records ``parent`` (its checksum is unchecked)."""
    _g(repo, "commit-graph", "write", "--stdin-commits", inp=f"{child}\n{parent}\n")
    path = repo / ".git" / "objects" / "info" / "commit-graph"
    data = bytearray(path.read_bytes())
    chunks = {}
    for i in range(data[6] + 1):
        at = 8 + 12 * i
        chunks[bytes(data[at:at + 4])] = struct.unpack(">Q", data[at + 4:at + 12])[0]
    oids = [bytes(data[chunks[b"OIDL"] + 20 * i:chunks[b"OIDL"] + 20 * i + 20]).hex()
            for i in range(2)]
    record = chunks[b"CDAT"] + 36 * oids.index(child)
    struct.pack_into(">I", data, record + 20, oids.index(parent))
    os.chmod(path, 0o644)
    path.write_bytes(bytes(data))


def test_H2_a_forged_commit_graph_is_not_believed(tmp_path):
    from hearth.training import prereg

    w = World(tmp_path)
    h1 = w.commit("golden.jsonl")
    _, bar = _backdated_bar_commit(w)
    _g(w.repo, "reset", "-q")
    _forge_commit_graph(w.repo, h1, bar)
    # The forgery works on plain git — the graph is believed over the commit object...
    assert bar in _g(w.repo, "log", "--format=%H", h1)
    # ...and is not believed by the gate's reads.
    assert prereg._git(["log", "--format=%H", h1], cwd=str(w.repo)) == h1


def test_H2_ambient_git_environment_cannot_redirect_the_gate(tmp_path, monkeypatch):
    """GIT_DIR (or any GIT_*) pointing elsewhere must not change which repository is read."""
    from hearth.training import prereg

    w = World(tmp_path)
    w.registered()
    other = tmp_path / "other"
    other.mkdir()
    _g(other, "init", "-q")
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    status = prereg.verify_committed(w.prereg)
    assert status.committed, status.reason


# -- H3: the same served weights under a new id are not fresh, whatever sits beside them ---


def test_H3_same_weights_new_id_plus_a_junk_file_is_not_a_fresh_adapter(tmp_path):
    """Reviewer R1: peek a1 (PASS), then register its weights + a README as a2."""
    w = World(tmp_path)
    _adapter(w, "a1")
    w.commit("golden.jsonl")
    _peek(w)
    _adapter(w, "a2", extra_file="README")  # identical served weights, new id, extra file
    w.write_prereg()
    w.commit("prereg.yaml")
    result = w.eval("a2", "--prereg", str(w.prereg), "--promote")
    _refused(result, w, "a2")


def _safetensors(tensors: dict[str, bytes], metadata: dict | None = None,
                 reverse: bool = False) -> bytes:
    import json

    header, offset, blobs = {}, 0, []
    for name in sorted(tensors, reverse=reverse):
        data = tensors[name]
        header[name] = {"dtype": "U8", "shape": [len(data)],
                        "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        blobs.append(data)
    if metadata is not None:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode()
    return struct.pack("<Q", len(raw)) + raw + b"".join(blobs)


def _served_dir(root: Path, name: str, *, st: bytes, cfg: str | None) -> Path:
    path = root / name
    path.mkdir()
    (path / "adapters.safetensors").write_bytes(st)
    if cfg is not None:
        (path / "adapter_config.json").write_text(cfg)
    return path


def test_H3_served_digest_ignores_what_mlx_lm_does_not_load(tmp_path):
    from hearth.registry.adapters import adapter_served_sha, adapter_weights_sha

    tensors = {"layers.0.lora_a": b"\x01\x02", "layers.0.lora_b": b"\x03\x04"}
    cfg = '{"fine_tune_type": "lora", "num_layers": 8, "lora_parameters": {"rank": 8}}'
    a = _served_dir(tmp_path, "a", st=_safetensors(tensors, {"format": "mlx"}), cfg=cfg)
    b = _served_dir(tmp_path, "b", st=_safetensors(tensors, {"note": "x"}, reverse=True),
                    cfg='{"num_layers": 8, "lora_parameters": {"rank": 8}, "seed": 7,\n'
                        ' "data": "/elsewhere"}')
    (b / "README.md").write_text("same adapter")
    (b / "0000100_adapters.safetensors").write_bytes(b"a checkpoint mlx_lm never opens")
    assert adapter_weights_sha(a) != adapter_weights_sha(b)  # different bytes on disk...
    assert adapter_served_sha(a) == adapter_served_sha(b) != ""  # ...same adapter served


@pytest.mark.parametrize(
    ("tensors", "cfg"),
    [
        ({"layers.0.lora_a": b"\x01\x02", "layers.0.lora_b": b"\x03\x05"}, None),  # a weight
        ({"layers.0.lora_a": b"\x01\x02", "layers.0.lora_c": b"\x03\x04"}, None),  # a name
        ({"layers.0.lora_a": b"\x01\x02", "layers.0.lora_b": b"\x03\x04"},
         '{"num_layers": 16, "lora_parameters": {"rank": 8}}'),  # the config mlx_lm reads
        ({"layers.0.lora_a": b"\x01\x02", "layers.0.lora_b": b"\x03\x04"},
         '{"fine_tune_type": "dora", "num_layers": 8, "lora_parameters": {"rank": 8}}'),
    ],
)
def test_H3_served_digest_changes_with_anything_mlx_lm_loads(tmp_path, tensors, cfg):
    from hearth.registry.adapters import adapter_served_sha

    base_cfg = '{"num_layers": 8, "lora_parameters": {"rank": 8}}'
    base = _served_dir(tmp_path, "base", cfg=base_cfg,
                       st=_safetensors({"layers.0.lora_a": b"\x01\x02",
                                        "layers.0.lora_b": b"\x03\x04"}))
    other = _served_dir(tmp_path, "other", st=_safetensors(tensors), cfg=cfg or base_cfg)
    assert adapter_served_sha(base) != adapter_served_sha(other)


def test_H3_a_missing_config_is_not_the_same_as_a_present_one(tmp_path):
    from hearth.registry.adapters import adapter_served_sha

    st = _safetensors({"t": b"\x01"})
    with_cfg = _served_dir(tmp_path, "c", st=st, cfg='{"num_layers": 8}')
    without = _served_dir(tmp_path, "n", st=st, cfg=None)
    assert adapter_served_sha(with_cfg) != adapter_served_sha(without) != ""
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "notes.txt").write_text("nothing mlx_lm loads")
    assert adapter_served_sha(empty) == ""
    assert adapter_served_sha(tmp_path / "missing") == ""
