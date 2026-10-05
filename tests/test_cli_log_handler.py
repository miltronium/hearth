"""``serve``'s stderr log handler is installed once, on the current stderr, every time.

``_log_hearth_to_stderr`` runs on every ``hearth serve``. Called twice in one process it must
leave exactly one CLI handler, writing to the stderr that is current NOW: a handler kept from
an earlier call writes to a stream that may be closed (CliRunner closes it after each
invocation, the "--- Logging error ---" trap noted in test_cli_startup_errors.py).
"""

from __future__ import annotations

import io
import logging
import sys

import pytest

from hearth.cli import _CLI_LOG_HANDLER_TAG, _log_hearth_to_stderr


@pytest.fixture
def hearth_logger():
    log = logging.getLogger("hearth")
    saved = (list(log.handlers), log.level)
    log.handlers[:] = []
    yield log
    log.handlers[:], log.level = saved[0], saved[1]


def _ours(log: logging.Logger) -> list[logging.Handler]:
    return [h for h in log.handlers if getattr(h, _CLI_LOG_HANDLER_TAG, False)]


def test_two_calls_leave_one_handler_on_the_current_stderr(hearth_logger, monkeypatch):
    first, second = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stderr", first)
    _log_hearth_to_stderr()
    monkeypatch.setattr(sys, "stderr", second)
    _log_hearth_to_stderr()
    assert len(hearth_logger.handlers) == 1 and len(_ours(hearth_logger)) == 1
    logging.getLogger("hearth.test").info("which stream?")
    assert "which stream?" in second.getvalue()
    assert "which stream?" not in first.getvalue()


def test_a_foreign_handler_does_not_stop_ours_or_the_info_level(hearth_logger, monkeypatch):
    foreign = logging.NullHandler()
    hearth_logger.addHandler(foreign)
    hearth_logger.setLevel(logging.WARNING)
    out = io.StringIO()
    monkeypatch.setattr(sys, "stderr", out)
    _log_hearth_to_stderr()
    _log_hearth_to_stderr()
    assert foreign in hearth_logger.handlers
    assert len(_ours(hearth_logger)) == 1
    logging.getLogger("hearth.test").info("weights loaded from /x")
    assert "weights loaded from /x" in out.getvalue()
