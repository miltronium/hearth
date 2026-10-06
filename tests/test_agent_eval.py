"""scripts/agent_eval.py: the answer key and the scorer must be able to fail (B-013).

No model runs here. These tests pin the two places the eval could report a pass it did not
earn: an answer key that no longer holds at the pinned commit, and a scorer lenient enough
that a wrong or shotgun answer counts as correct.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("agent_eval", _REPO / "scripts" / "agent_eval.py")
ae = importlib.util.module_from_spec(_spec)
sys.modules["agent_eval"] = ae  # dataclasses resolve their module while it loads
_spec.loader.exec_module(ae)

NAMES = {"AGENT.md", "API.md", "GUIDE.md", "HANDOFF.md", "README.md"}


def _item(kind="find", answer="AGENT.md", term="AgentIncompleteError", file="AGENT.md"):
    return ae.Item("t", kind, "q?", answer, term, file)


# --- the answer key, checked against the pinned corpus -----------------------------------


def test_the_shipped_key_holds_at_its_pinned_commit():
    commit, path, items = ae.load_spec(ae.DEFAULT_SPEC)
    with tempfile.TemporaryDirectory() as tmp:
        root = ae.extract_corpus(commit, path, Path(tmp))
        assert ae.verify_key(items, ae.corpus_files(root)) == []
    assert len(items) == 10


def test_the_pin_is_a_full_commit_hash_that_exists():
    commit, _, _ = ae.load_spec(ae.DEFAULT_SPEC)
    assert len(commit) == 40
    out = subprocess.run(
        ["git", "-C", str(_REPO), "cat-file", "-t", commit], capture_output=True, text=True
    )
    assert out.stdout.strip() == "commit"


def test_key_check_refuses_a_term_missing_from_its_file():
    files = {"AGENT.md": "nothing here"}
    assert ae.verify_key([_item()], files)


def test_key_check_refuses_a_find_term_present_in_two_files():
    files = {"AGENT.md": "AgentIncompleteError", "BUGS.md": "AgentIncompleteError"}
    problems = ae.verify_key([_item()], files)
    assert any("also in" in p for p in problems)


def test_key_check_refuses_a_find_answer_that_is_not_the_evidence_file():
    files = {"AGENT.md": "AgentIncompleteError"}
    assert ae.verify_key([_item(answer="API.md")], files)


def test_key_check_allows_a_read_term_found_in_several_files():
    files = {"MODELS_local.md": "30.15 GB", "GUIDE.md": "30.15 GB"}
    item = _item(kind="read", answer="30.15", term="30.15 GB", file="MODELS_local.md")
    assert ae.verify_key([item], files) == []


# --- the scorer ----------------------------------------------------------------------------


def test_find_passes_with_the_right_file_in_a_path():
    assert ae.score(_item(), "It is /tmp/x/docs/AGENT.md.", NAMES) == (True, "ok")


def test_find_fails_when_the_answer_also_names_other_files():
    ok, reason = ae.score(_item(), "AGENT.md, API.md and GUIDE.md all match", NAMES)
    assert not ok and "also names" in reason


def test_find_fails_on_the_wrong_file_and_on_no_answer():
    assert not ae.score(_item(), "API.md", NAMES)[0]
    assert not ae.score(_item(), None, NAMES)[0]


def test_numbers_are_tokens_not_substrings():
    item = _item(kind="read", answer="30", term="x", file="x")
    assert ae.score(item, "The default min_n is 30.", NAMES)[0]
    assert not ae.score(item, "It is 300.", NAMES)[0]
    assert not ae.score(item, "The ceiling is 30.15 GB.", NAMES)[0]


def test_words_are_case_insensitive_tokens():
    item = _item(kind="two-step", answer="false", term="x", file="x")
    assert ae.score(item, "They set it to False.", NAMES)[0]
    assert not ae.score(item, "a falsey value", NAMES)[0]


def test_the_corpus_directory_is_the_same_on_every_run():
    # The path is in every observation the model reads; a random name per run made the same
    # code score differently from run to run.
    commit, _, _ = ae.load_spec(ae.DEFAULT_SPEC)
    first, second = ae.corpus_dir(commit), ae.corpus_dir(commit)
    assert first == second
    assert commit[:12] in first.name
    first.rmdir()
