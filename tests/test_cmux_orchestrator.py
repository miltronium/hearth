"""Tests for the cmux×HEARTH orchestrator (scripts/cmux/orchestrator.py).

Deterministic: the triage/decision logic and the CLI client's command construction are tested with a
stub classifier and a monkeypatched subprocess (no model, no cmux). The real-model triage is exercised
by scripts/cmux/orchestrator_demo.py. Loaded by path (cmux tooling isn't a HEARTH package; ADR-C005).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

_MOD = Path(__file__).resolve().parents[1] / "scripts" / "cmux" / "orchestrator.py"
_spec = importlib.util.spec_from_file_location("orchestrator", _MOD)
orch = importlib.util.module_from_spec(_spec)
sys.modules["orchestrator"] = orch  # dataclasses + future-annotations need the module registered
_spec.loader.exec_module(orch)


def _classifier(mapping):
    """Stub classify(text, labels) that returns a state based on a substring match in the text."""
    def classify(text, labels):
        for needle, state in mapping.items():
            if needle in text:
                return state
        return "working"
    return classify


def test_priority_mapping():
    assert orch.priority_for("waiting") == "attention"
    assert orch.priority_for("error") == "attention"
    assert orch.priority_for("done") == "info"
    assert orch.priority_for("working") == "none"
    assert orch.priority_for("garbage") == "none"  # unknown ⇒ never a spurious notify


def test_triage_text_tolerant_and_summarizes_on_attention():
    classify = lambda t, labels: "the state is ERROR here"  # noqa: E731 - model-ish noisy output
    summarize = lambda t: "compile failed"
    state, priority, message = orch.triage_text("boom", classify, summarize)
    assert state == "error" and priority == "attention" and message == "compile failed"


def test_triage_text_no_summary_when_quiet():
    state, priority, message = orch.triage_text("running", _classifier({"running": "working"}), lambda t: "x")
    assert state == "working" and priority == "none" and message == ""


def test_run_once_notifies_only_attention_and_info():
    screens = {
        "s1": "compiling files",              # working -> quiet
        "s2": "Proceed? [y/N] ",              # waiting -> attention
        "s3": "All tests passed. $ ",         # done -> info
        "s4": "Build failed (exit 65)",       # error -> attention
    }
    client = orch.FakeCmuxClient(screens, titles={k: k for k in screens})
    classify = _classifier({"compiling": "working", "[y/N]": "waiting", "passed": "done", "failed": "error"})
    results = orch.run_once(client, classify, summarize=lambda t: "note")

    by_id = {t.surface_id: t for t in results}
    assert by_id["s1"].notified is False and by_id["s1"].priority == "none"
    assert by_id["s2"].notified is True and by_id["s2"].state == "waiting"
    assert by_id["s3"].notified is True and by_id["s3"].priority == "info"
    assert by_id["s4"].notified is True and by_id["s4"].state == "error"
    # exactly the three non-working panes were notified
    assert {n[0] for n in client.notifications} == {"s2", "s3", "s4"}


def test_run_once_skips_browser_surfaces():
    client = orch.FakeCmuxClient(
        {"term": "Build failed", "web": "Build failed"},
        kinds={"term": "terminal", "web": "browser"},
    )
    results = orch.run_once(client, _classifier({"failed": "error"}), summarize=lambda t: "n")
    assert [t.surface_id for t in results] == ["term"]  # browser skipped
    assert {n[0] for n in client.notifications} == {"term"}


def test_run_once_dry_run_does_not_notify():
    client = orch.FakeCmuxClient({"s1": "Build failed"})
    orch.run_once(client, _classifier({"failed": "error"}), summarize=lambda t: "n", do_notify=False)
    assert client.notifications == []


def test_cli_client_builds_correct_argv(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        stdout = '{"text": "hello"}' if "read-screen" in argv else "{}"
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(orch.subprocess, "run", fake_run)
    client = orch.CmuxCliClient(cmux_bin="cmux", socket_path="/tmp/cmux.sock")

    assert client.read_screen("surface:2", lines=120) == "hello"
    client.notify("surface:2", "cmux: t", "body")
    client.send_key("surface:2", "ctrl+c")

    read_argv = next(c for c in calls if "read-screen" in c)
    assert read_argv == ["cmux", "--socket", "/tmp/cmux.sock", "--json", "read-screen", "--surface", "surface:2", "--lines", "120"]
    notify_argv = next(c for c in calls if "notify" in c)
    assert notify_argv == ["cmux", "--socket", "/tmp/cmux.sock", "notify", "--surface", "surface:2", "--title", "cmux: t", "--body", "body"]
    sendkey_argv = next(c for c in calls if "send-key" in c)
    assert sendkey_argv == ["cmux", "--socket", "/tmp/cmux.sock", "send-key", "--surface", "surface:2", "ctrl+c"]


def test_cli_client_list_surfaces_parses_workspaces(monkeypatch):
    def fake_run(argv, **kwargs):
        if "list-workspaces" in argv:
            out = '{"workspaces": [{"id": "ws1"}]}'
        elif "list-pane-surfaces" in argv:
            out = '{"surfaces": [{"id": "s1", "title": "build", "type": "terminal"}, {"id": "b1", "type": "browser"}]}'
        else:
            out = "{}"
        return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")

    monkeypatch.setattr(orch.subprocess, "run", fake_run)
    surfaces = orch.CmuxCliClient().list_surfaces()
    assert [(s.surface_id, s.kind) for s in surfaces] == [("s1", "terminal"), ("b1", "browser")]
