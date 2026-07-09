"""RemoteProvider tests — verify protocol mapping with a FAKE client (no network).

The anthropic SDK is not installed in the test env and api.anthropic.com is blocked; we
inject a fake ``anthropic`` module so we can assert the request mapping (system split,
text-block extraction, no forbidden params) without any real call.
"""

from __future__ import annotations

import sys
import types

from hearth.providers.base import GenRequest, Message
from hearth.providers.remote import RemoteProvider
from hearth.router.policy import RemoteConfig


class _FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeMessage:
    def __init__(self, text: str) -> None:
        self.content = [_FakeTextBlock(text)]
        self.usage = types.SimpleNamespace(input_tokens=7, output_tokens=11)


class _FakeMessages:
    def __init__(self, recorder: dict) -> None:
        self._recorder = recorder

    def create(self, **kwargs):
        self._recorder.update(kwargs)
        return _FakeMessage("hello from fake claude")


class _FakeAnthropic:
    """Stand-in for anthropic.Anthropic — records the constructor + create kwargs."""

    last_init: dict = {}

    def __init__(self, **kwargs):
        _FakeAnthropic.last_init = kwargs
        self.recorder: dict = {}
        self.messages = _FakeMessages(self.recorder)


def _install_fake_anthropic(monkeypatch):
    module = types.ModuleType("anthropic")
    module.Anthropic = _FakeAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", module)


def test_anthropic_generate_maps_request(monkeypatch):
    _install_fake_anthropic(monkeypatch)
    provider = RemoteProvider(RemoteConfig(protocol="anthropic", model="claude-opus-4-8"))
    req = GenRequest(
        messages=[
            Message(role="system", content="be terse"),
            Message(role="user", content="hi"),
        ],
        model="ignored-by-remote",
    )
    result = provider.generate(req)
    assert result.text == "hello from fake claude"
    assert result.backend == "remote"
    assert result.model == "claude-opus-4-8"
    assert result.prompt_tokens == 7
    assert result.completion_tokens == 11


def test_anthropic_splits_system_and_omits_forbidden_params(monkeypatch):
    _install_fake_anthropic(monkeypatch)
    provider = RemoteProvider(RemoteConfig(protocol="anthropic", model="claude-opus-4-8"))
    kwargs = provider._anthropic_kwargs(
        GenRequest(
            messages=[
                Message(role="system", content="sys prompt"),
                Message(role="user", content="u1"),
                Message(role="assistant", content="a1"),
            ],
            model="m",
        )
    )
    assert kwargs["system"] == "sys prompt"
    assert kwargs["messages"] == [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]
    # Forbidden on Opus 4.8 — must never be sent.
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert "thinking" not in kwargs


def test_anthropic_thinking_opt_in(monkeypatch):
    _install_fake_anthropic(monkeypatch)
    provider = RemoteProvider(
        RemoteConfig(protocol="anthropic", model="claude-opus-4-8", thinking=True)
    )
    kwargs = provider._anthropic_kwargs(
        GenRequest(messages=[Message(role="user", content="x")], model="m")
    )
    assert kwargs["thinking"] == {"type": "adaptive"}


def test_anthropic_uses_api_key_env(monkeypatch):
    _install_fake_anthropic(monkeypatch)
    monkeypatch.setenv("MY_REMOTE_KEY", "sk-test-123")
    provider = RemoteProvider(
        RemoteConfig(protocol="anthropic", model="m", api_key_env="MY_REMOTE_KEY")
    )
    provider._anthropic_client()
    assert _FakeAnthropic.last_init == {"api_key": "sk-test-123"}


def test_missing_sdk_raises_clean_error(monkeypatch):
    from hearth.providers.remote import RemoteUnavailableError

    # Ensure importing anthropic fails.
    monkeypatch.setitem(sys.modules, "anthropic", None)
    provider = RemoteProvider(RemoteConfig(protocol="anthropic", model="m"))
    try:
        provider._anthropic_client()
    except RemoteUnavailableError:
        pass
    else:
        raise AssertionError("expected RemoteUnavailableError when anthropic is missing")
