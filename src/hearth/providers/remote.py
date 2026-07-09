"""Remote escalation provider (ARCHITECTURE §4, ADR-007).

The escalation target. HEARTH stays provider-agnostic: endpoint, model, and auth are
config (``remotes:`` in routing.yaml). Two protocols are supported:

  * ``anthropic`` — the **official** ``anthropic`` SDK (an optional ``[remote]`` extra;
    imported lazily so core install/tests don't need it). Auto-resolves credentials
    (``ANTHROPIC_API_KEY`` / ``ant auth login``) unless the config names ``api_key_env``.
  * ``openai`` — generic OpenAI-compatible endpoint over ``httpx`` (core dep). For any
    non-Claude model or LAN server; reads a bearer token from ``api_key_env``.

Secrets never live in files — only the *name* of an env var does.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

from ..router.policy import RemoteConfig
from .base import Capabilities, GenRequest, GenResult, Message, ResourceEstimate


class RemoteUnavailableError(RuntimeError):
    """Raised when a remote backend is requested but its dependency/config is unusable."""


class RemoteProvider:
    """Serves completions by calling a configured remote endpoint (frontier or internal)."""

    name = "remote"

    def __init__(self, config: RemoteConfig) -> None:
        self.config = config

    def capabilities(self) -> Capabilities:
        return Capabilities(chat=True, embed=False, stream=True, adapters=False)

    def footprint(self, model_id: str) -> ResourceEstimate:
        # A remote call has no local resident footprint.
        return ResourceEstimate(ram_gb=0.0)

    def generate(self, req: GenRequest) -> GenResult:
        if self.config.protocol == "anthropic":
            return self._anthropic_generate(req)
        if self.config.protocol == "openai":
            return self._openai_generate(req)
        raise RemoteUnavailableError(f"unknown remote protocol: {self.config.protocol!r}")

    def stream(self, req: GenRequest) -> Iterator[str]:
        if self.config.protocol == "anthropic":
            yield from self._anthropic_stream(req)
        elif self.config.protocol == "openai":
            yield from self._openai_stream(req)
        else:
            raise RemoteUnavailableError(f"unknown remote protocol: {self.config.protocol!r}")

    # -- anthropic (official SDK) -----------------------------------------------------

    def _anthropic_client(self):
        """Construct an ``anthropic.Anthropic`` client. Import is deferred (optional extra)."""
        try:
            import anthropic  # deferred; only needed on the anthropic escalation path
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
            raise RemoteUnavailableError(
                "the anthropic SDK is not installed. Install it with: uv sync --extra remote"
            ) from exc
        # When no explicit key is configured the SDK auto-resolves ANTHROPIC_API_KEY / the
        # `ant auth login` profile. Only pass api_key when the config names an env var.
        if self.config.api_key_env:
            key = os.environ.get(self.config.api_key_env)
            if not key:
                raise RemoteUnavailableError(
                    f"remote api_key_env {self.config.api_key_env!r} is unset"
                )
            return anthropic.Anthropic(api_key=key)
        return anthropic.Anthropic()

    def _anthropic_kwargs(self, req: GenRequest) -> dict:
        """Map a HEARTH request to Anthropic's create/stream kwargs.

        System messages fold into the top-level ``system=`` param; user/assistant become
        ``messages``. We deliberately omit temperature/top_p/top_k and only send
        ``thinking`` (adaptive) when explicitly enabled — those otherwise 400 on Opus 4.8.
        """
        system, messages = _split_system(req.messages)
        kwargs: dict = {
            "model": self.config.model,
            "max_tokens": req.max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if self.config.thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        return kwargs

    def _anthropic_generate(self, req: GenRequest) -> GenResult:
        client = self._anthropic_client()
        msg = client.messages.create(**self._anthropic_kwargs(req))
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        usage = getattr(msg, "usage", None)
        return GenResult(
            text=text.strip(),
            model=self.config.model,
            backend=self.name,
            prompt_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
        )

    def _anthropic_stream(self, req: GenRequest) -> Iterator[str]:
        client = self._anthropic_client()
        with client.messages.stream(**self._anthropic_kwargs(req)) as stream:
            yield from stream.text_stream

    # -- openai-compatible (httpx) ----------------------------------------------------

    def _openai_url(self) -> str:
        base = (self.config.base_url or "").rstrip("/")
        if not base:
            raise RemoteUnavailableError("openai remote requires a base_url")
        return f"{base}/chat/completions"

    def _openai_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key_env:
            key = os.environ.get(self.config.api_key_env)
            if not key:
                raise RemoteUnavailableError(
                    f"remote api_key_env {self.config.api_key_env!r} is unset"
                )
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _openai_body(self, req: GenRequest, stream: bool) -> dict:
        return {
            "model": self.config.model,
            "max_tokens": req.max_tokens,
            "stream": stream,
            "messages": [{"role": m.role, "content": m.content} for m in req.messages],
        }

    def _openai_generate(self, req: GenRequest) -> GenResult:
        import httpx  # core dep, but imported lazily to keep module import cheap

        resp = httpx.post(
            self._openai_url(),
            headers=self._openai_headers(),
            json=self._openai_body(req, stream=False),
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return GenResult(
            text=(text or "").strip(),
            model=self.config.model,
            backend=self.name,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
        )

    def _openai_stream(self, req: GenRequest) -> Iterator[str]:
        import json

        import httpx

        with httpx.stream(
            "POST",
            self._openai_url(),
            headers=self._openai_headers(),
            json=self._openai_body(req, stream=True),
            timeout=120.0,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[len("data: ") :]
                if payload == "[DONE]":
                    break
                delta = json.loads(payload)["choices"][0]["delta"].get("content")
                if delta:
                    yield delta


def _split_system(messages: list[Message]) -> tuple[str, list[dict]]:
    """Split ``system`` content out for Anthropic's top-level ``system=`` param.

    Returns ``(system_text, [{"role","content"}, ...])`` for user/assistant turns.
    """
    system_parts = [m.content for m in messages if m.role == "system"]
    turns = [
        {"role": m.role, "content": m.content}
        for m in messages
        if m.role in ("user", "assistant")
    ]
    return "\n\n".join(system_parts), turns


__all__ = ["RemoteProvider", "RemoteUnavailableError"]
