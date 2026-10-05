"""``hearth stats`` shows failed requests, and its escalation label is true in every case.

WHY: B-003 made a failed request write a ``RequestRecord`` and added ``failed`` /
``failure_rate`` to the rollup, but ``hearth stats`` never printed them, and it labelled
``escalations_failed`` as "(served local)" although that count also includes requests whose
local fallback failed too (nothing was served). These tests feed the real ``MetricsStore``
and assert on the rendered table.
"""

from __future__ import annotations

import re

import pytest
from typer.testing import CliRunner

import hearth.observability as observability
from hearth.cli import app
from hearth.observability.metrics import MetricsStore, RequestRecord

runner = CliRunner()
WIDE = {"COLUMNS": "200", "NO_COLOR": "1"}


def _rec(**kw) -> RequestRecord:
    base = dict(
        task_class="chat", backend="echo", model="echo", served_by="local",
        prompt_tokens=3, completion_tokens=4, latency_ms=1.0,
    )
    base.update(kw)
    return RequestRecord(**base)


@pytest.fixture
def store(monkeypatch):
    s = MetricsStore()
    monkeypatch.setattr(observability, "get_metrics", lambda: s)
    return s


def _row(output: str, label: str) -> str:
    for line in output.splitlines():
        cells = [c.strip() for c in re.split(r"[│┃]", line) if c.strip()]
        if len(cells) == 2 and cells[0] == label:
            return cells[1]
    raise AssertionError(f"no row {label!r} in:\n{output}")


def test_stats_shows_failed_and_failure_rate(store):
    store.record(_rec())  # answered
    store.record(_rec(failed="provider 'mlx' failed: weights missing"))  # local failure
    store.record(  # escalation failed, then the local fallback failed too
        _rec(escalated=True, escalation_failed="remote 503", failed="local failed")
    )
    store.record(_rec(escalated=True, escalation_failed="remote 503"))  # served local
    result = runner.invoke(app, ["stats"], env=WIDE)
    assert result.exit_code == 0, result.output
    assert _row(result.output, "requests") == "4"
    assert _row(result.output, "failed (error, no answer)") == "2"
    assert _row(result.output, "failure rate") == "50.00%"
    label = "escalations failed (remote errored; prompt may have left)"
    assert _row(result.output, label) == "2"


def test_escalation_label_does_not_claim_served_local(store):
    # The only failed escalation here was NOT served locally: the fallback failed too.
    store.record(_rec(escalated=True, escalation_failed="remote 503", failed="local failed"))
    result = runner.invoke(app, ["stats"], env=WIDE)
    assert result.exit_code == 0, result.output
    assert "served local" not in result.output
    assert _row(result.output, "failed (error, no answer)") == "1"
