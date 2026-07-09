"""Provider contract tests — every backend must honor the ModelProvider interface.

This is the client-agnostic guarantee in miniature (ADR-004): the echo and MLX
providers are checked against the same structural contract, so backends stay swappable.
"""

from __future__ import annotations

from hearth.providers.base import (
    Capabilities,
    GenRequest,
    Message,
    ModelProvider,
    ResourceEstimate,
)
from hearth.providers.echo import EchoProvider
from hearth.providers.mlx import MLXProvider


def test_echo_satisfies_protocol():
    assert isinstance(EchoProvider(), ModelProvider)


def test_mlx_satisfies_protocol():
    # Constructing MLXProvider must not import mlx-lm (deferred to load time),
    # so this holds even without the mlx extra installed.
    assert isinstance(MLXProvider("some/model"), ModelProvider)


def test_echo_generate_contract():
    p = EchoProvider()
    assert isinstance(p.capabilities(), Capabilities)
    assert isinstance(p.footprint("x"), ResourceEstimate)
    result = p.generate(
        GenRequest(messages=[Message(role="user", content="ping")], model="m")
    )
    assert result.backend == "echo"
    assert result.text == "[echo] ping"
    assert result.prompt_tokens > 0
    assert result.completion_tokens > 0
