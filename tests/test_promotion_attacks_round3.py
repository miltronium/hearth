"""Third-round attacks on the promotion gate (B-120..) — each must now be REFUSED.

A reviewer's attack suite (round 4 of review, reusing ``test_promotion_evidence.World``)
asserted that each of these PROMOTED on the code at 3caecb0. Every test below is that attack
ported to assert the refusal; each was run against 3caecb0 first and promoted there.

    H2  `git replace` / info/grafts / a forged commit-graph fake "the prereg commit is an
        ancestor of the HEAD recorded at the first measurement" (B-120)
    H3  the same served weights re-registered under a new id with a junk file beside them
        were a "fresh, never-measured" adapter (B-121)
    M1  a "menu" of bars committed before the first measurement: measure under one, see the
        verdict, promote under another (B-122)
    M4  an adapter id holding U+2028 tore its ledger record in two, bricking every later
        measurement and promotion on the install (B-124)
    M5  near-duplicate prompts passed as distinct via zero-width / soft hyphen / word joiner,
        fullwidth forms, NFC vs NFD, or trailing punctuation, inflating n (B-125)
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


# -- M1: the bar is the one the adapter was FIRST measured under (B-122) -------------------


def _menu(world: World) -> Path:
    """Two bars committed up front: the real one, and one on a golden set the adapter loses."""
    import json

    import yaml

    from hearth.training.eval import as_golden_set

    lose = world.repo / "lose.jsonl"
    lose.write_text("".join(json.dumps({"prompt": r["prompt"], "expected": "Z"}) + "\n"
                            for r in pe.ROWS))
    world.write_prereg()
    body = yaml.safe_load(world.prereg.read_text())
    body["golden_sha"] = as_golden_set("extract", [(r["prompt"], "Z") for r in pe.ROWS]).sha
    (world.repo / "prereg_lose.yaml").write_text(yaml.safe_dump(body))
    world.commit("golden.jsonl", "lose.jsonl", "prereg.yaml", "prereg_lose.yaml")
    return lose


def test_M1_a_menu_of_bars_committed_up_front_cannot_be_picked_from_after_scoring(tmp_path):
    """Reviewer R3: measure under prereg_lose (FAIL), then promote under prereg (PASS)."""
    w = World(tmp_path)
    _adapter(w, "a1")
    lose = _menu(w)
    first = pe.runner.invoke(pe.app, ["eval", "a1", "--golden", str(lose), "--metric", "exact",
                                      "--max-tokens", "24", "--prereg",
                                      str(w.repo / "prereg_lose.yaml")], env=w.env)
    assert first.exit_code == 0 and "FAIL" in _flat(first), _flat(first)
    result = w.eval("a1", "--prereg", str(w.prereg), "--promote")
    _refused(result, w, "a1", "under another pre-registration")
    # ...and the offline path refuses it for the same reason.
    w.eval_report("a1")
    _refused(w.promote("a1"), w, "a1", "under another pre-registration")


def test_M1_an_exploratory_first_measurement_is_unpromotable_even_after_the_bar(tmp_path):
    """Bar committed first, then a peek with no --prereg: the bar was not the one in force."""
    w = World(tmp_path)
    _adapter(w, "a1")
    w.registered()
    peek = _peek(w)
    assert "cannot be promoted" in _flat(peek)
    result = w.eval("a1", "--prereg", str(w.prereg), "--promote")
    _refused(result, w, "a1", "with no pre-registration (an exploratory run)")


def test_M1_first_measured_under_the_bar_stays_promotable_after_a_later_peek(tmp_path):
    w = World(tmp_path)
    _adapter(w, "a1")
    w.registered()
    first = w.eval("a1", "--prereg", str(w.prereg))
    assert first.exit_code == 0 and "promote with:" in _flat(first), _flat(first)
    later = _peek(w)  # exploratory, but not the first measurement
    assert "stays promotable only under" in _flat(later)
    result = w.eval("a1", "--prereg", str(w.prereg), "--promote")
    assert result.exit_code == 0 and w.status("a1") == "promoted", _flat(result)


def test_M1_the_ledger_records_which_prereg_each_measurement_was_made_under(tmp_path):
    from hearth.training import attest, ledger

    w = World(tmp_path)
    _adapter(w, "a1")
    w.registered()
    w.eval("a1", "--prereg", str(w.prereg))
    _peek(w)
    first, second = ledger.read(w.home, attest.load_key(w.home))
    from hearth.training.prereg import load_prereg

    assert first["prereg_sha"] == load_prereg(w.prereg).sha
    assert first["prereg_path"] == str(w.prereg.resolve())
    assert second["prereg_sha"] == second["prereg_path"] == ""


# -- M4: a separator character cannot tear a ledger line and brick the install (B-124) -----


@pytest.mark.parametrize("ch", ["\u2028", "\u2029", "\x85", "\u200b", "\u00ad", "\u202e",
                                "\x00", "\n", "\r", "\t", "\ue000"])
def test_M4_registration_refuses_control_format_and_separator_characters(tmp_path, ch):
    from hearth.registry.adapters import AdapterError, AdapterStore

    store = AdapterStore(path=tmp_path / "adapters.json")
    with pytest.raises(AdapterError, match="control, format or separator"):
        store.register(f"a{ch}b", base_model=pe.BASE, task="extract", train_run_id="r",
                       adapter_path=str(tmp_path))
    assert store.get(f"a{ch}b") is None


@pytest.mark.parametrize("aid", ["", "   "])
def test_M4_registration_refuses_an_empty_id(tmp_path, aid):
    from hearth.registry.adapters import AdapterError, AdapterStore

    with pytest.raises(AdapterError, match="non-empty"):
        AdapterStore(path=tmp_path / "adapters.json").register(
            aid, base_model=pe.BASE, task="extract", train_run_id="r", adapter_path="x")


def test_M4_ordinary_unicode_ids_still_register(tmp_path):
    from hearth.registry.adapters import AdapterStore

    store = AdapterStore(path=tmp_path / "adapters.json")
    store.register("extract-café v2", base_model=pe.BASE, task="extract", train_run_id="r",
                   adapter_path="x")
    assert store.get("extract-café v2") is not None


def _hand_registered(world: World, aid: str) -> None:
    """An id that predates validation (older version, or adapters.json edited by hand)."""
    import json

    path = world.tmp / "weights" / "legacy"
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapters.safetensors").write_bytes(b"GOOD")
    reg = world.store.path
    data = json.loads(reg.read_text()) if reg.exists() else {"adapters": []}
    data["adapters"].append({"id": aid, "base_model": pe.BASE, "task": "extract",
                             "train_run_id": "r", "adapter_path": str(path)})
    reg.parent.mkdir(parents=True, exist_ok=True)
    reg.write_text(json.dumps(data))


def test_M4_a_u2028_adapter_id_does_not_brick_the_ledger(tmp_path):
    """Reviewer R7: measuring an id holding U+2028 made every later measurement refuse."""
    from hearth.training.ledger import ledger_path

    w = World(tmp_path)
    w.commit("golden.jsonl")
    _hand_registered(w, "a\u2028b")
    _adapter(w, "ok")
    first = w.eval("a\u2028b")
    assert first.exit_code == 0 and "not an intact" not in _flat(first), _flat(first)
    second = w.eval("ok")
    assert second.exit_code == 0 and "PASS" in _flat(second), _flat(second)
    raw = ledger_path(w.home).read_bytes()
    assert raw.isascii() and raw.count(b"\n") == 2


def test_M4_a_ledger_written_before_the_fix_is_read_whole(tmp_path):
    """An install already "bricked" by a raw U+2028 reads again: records split on \\n only."""
    import json

    from hearth.training import attest, ledger

    key = attest.load_key(tmp_path, create=True)
    first = attest.sign({"adapter_id": "a\u2028b\u0085c", "schema": ledger.LEDGER_SCHEMA,
                         "seq": 0, "prev": ""}, key)
    second = attest.sign({"adapter_id": "d", "schema": ledger.LEDGER_SCHEMA, "seq": 1,
                          "prev": first["signature"]["mac"]}, key)
    ledger.ledger_path(tmp_path).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in (first, second)),
        encoding="utf-8")
    records = ledger.read(tmp_path, key)
    assert [r["adapter_id"] for r in records] == ["a\u2028b\u0085c", "d"]
    ledger.append(tmp_path, {"adapter_id": "e"}, key)
    assert len(ledger.read(tmp_path, key)) == 3


def test_M4_a_ledger_that_is_not_utf8_refuses_cleanly(tmp_path):
    from hearth.training import attest, ledger

    key = attest.load_key(tmp_path, create=True)
    ledger.ledger_path(tmp_path).write_bytes(b"\xff\xfe{}\n")
    with pytest.raises(ledger.LedgerError, match="not UTF-8"):
        ledger.read(tmp_path, key)


# -- M5: near-duplicate prompts cannot pass as distinct by changing bytes, not text (B-125) -


@pytest.mark.parametrize(
    "twin",
    [
        "what is the capital of france\u200b",          # zero-width space (Cf)
        "what is the capi\u00adtal of france",           # soft hyphen (Cf)
        "what is the\u2060 capital of france",           # word joiner (Cf)
        "\ufeffwhat is the capital of france",           # BOM / ZWNBSP (Cf)
        "ｗｈａｔ ｉｓ ｔｈｅ ｃａｐｉｔａｌ ｏｆ ｆｒａｎｃｅ",  # fullwidth (NFKC)
        "What is the capital of France?",                # trailing punctuation + case
        "what is the capital of france .",               # trailing punctuation after space
        "what\u00a0is the capital of france",            # no-break space (NFKC -> space)
    ],
)
def test_M5_near_duplicates_are_one_prompt(twin):
    from hearth.training.eval import as_golden_set

    base = "what is the capital of france"
    golden = as_golden_set("extract", [(base, "Paris"), (twin, "Paris")])
    assert golden.duplicate_prompts(), repr(twin)


def test_M5_composed_and_decomposed_accents_are_one_prompt():
    from hearth.training.eval import normalize_prompt

    assert normalize_prompt("café menu") == normalize_prompt("cafe\u0301 menu")


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("3-1", "31"),                     # inner punctuation can carry meaning: kept
        ("?what", "what"),                 # leading punctuation: kept
        ("label: spam", "label: ham"),
    ],
)
def test_M5_genuinely_different_prompts_stay_distinct(a, b):
    from hearth.training.eval import normalize_prompt

    assert normalize_prompt(a) != normalize_prompt(b)


def test_M5_a_near_duplicate_golden_set_is_refused_end_to_end(tmp_path):
    """30 distinct rows + 10 zero-width twins: n is 30 observations, not 40."""
    import json

    w = World(tmp_path)
    _adapter(w, "a1")
    rows = pe.ROWS[:30] + [{"prompt": r["prompt"] + "\u200b", "expected": r["expected"]}
                           for r in pe.ROWS[:10]]
    w.golden.write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = w.eval("a1")
    assert result.exit_code == 1 and "repeats" in _flat(result), _flat(result)
