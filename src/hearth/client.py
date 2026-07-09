"""Thin Python client for the HEARTH HTTP API (Phase 5, ADR-002).

A convenience wrapper over the OpenAI-compatible gateway for Python callers who'd rather
not hand-roll ``httpx`` calls. It is deliberately thin — every method maps to one endpoint
and returns the parsed JSON (or a stream of text). ``httpx`` is a core dependency, so this
needs no extras.

    from hearth.client import HearthClient

    hearth = HearthClient("http://127.0.0.1:8080", token=my_token)
    print(hearth.summarize("...long text..."))

The Swift SDK (``swift/``) mirrors these method shapes against the same endpoints so the
two clients stay at parity (docs/INTEGRATION.md, "SDK parity").
"""

from __future__ import annotations

from collections.abc import Iterator

import httpx

# Reuse the router's task-class names as intent hints so summarize/classify/extract pin the
# right class without re-running classification server-side.
_DEFAULT_TIMEOUT = 60.0


class HearthClient:
    """Synchronous HEARTH API client over ``httpx``.

    Args:
        base_url: HEARTH root, e.g. ``http://127.0.0.1:8080`` (with or without ``/v1``).
        token: bearer token; sent as ``Authorization: Bearer <token>`` when provided.
        timeout: per-request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        # Normalize so both ".../v1" and bare roots work; endpoints are appended as "/v1/...".
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        self.token = token
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._client = httpx.Client(
            base_url=self.base_url, headers=headers, timeout=timeout
        )

    # -- context manager ------------------------------------------------------------------

    def __enter__(self) -> HearthClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    # -- chat ------------------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str = "auto",
        stream: bool = False,
        intent: str | None = None,
        allow_escalation: bool = True,
        **kwargs: object,
    ) -> dict | Iterator[str]:
        """Call ``/v1/chat/completions``.

        Non-streaming returns the parsed JSON response. Streaming returns an iterator of
        text deltas (the assistant content, reassembled chunk by chunk).
        """
        payload: dict = {"model": model, "messages": messages, "stream": stream, **kwargs}
        if intent is not None or not allow_escalation:
            payload["hearth"] = {"intent": intent, "allow_escalation": allow_escalation}
        if stream:
            return self._stream_chat(payload)
        r = self._client.post("/v1/chat/completions", json=payload)
        r.raise_for_status()
        return r.json()

    def _stream_chat(self, payload: dict) -> Iterator[str]:
        """Yield assistant text deltas from a streaming chat completion (SSE)."""
        import json

        with self._client.stream("POST", "/v1/chat/completions", json=payload) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[len("data: ") :]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                if content := delta.get("content"):
                    yield content

    # -- convenience wrappers over chat ---------------------------------------------------

    def summarize(self, text: str, max_words: int | None = None) -> str:
        """Summarize ``text`` locally via a chat completion with intent=summarize."""
        limit = f" in at most {max_words} words" if max_words else ""
        return self._one_shot(f"Summarize the following text{limit}:\n\n{text}", "summarize")

    def classify(self, text: str, labels: list[str]) -> str:
        """Classify ``text`` into one of ``labels`` locally (intent=classify)."""
        options = ", ".join(labels)
        prompt = (
            f"Classify the following text into exactly one of these labels: {options}.\n"
            f"Reply with only the label.\n\nText:\n{text}"
        )
        return self._one_shot(prompt, "classify")

    def _one_shot(self, prompt: str, intent: str) -> str:
        """Send a single user turn and return the assistant text (hard-local)."""
        resp = self.chat(
            [{"role": "user", "content": prompt}],
            intent=intent,
            allow_escalation=False,
        )
        assert isinstance(resp, dict)  # non-streaming path
        return resp["choices"][0]["message"]["content"].strip()

    # -- embeddings + RAG ------------------------------------------------------------------

    def embed(self, texts: str | list[str], model: str = "auto") -> list[list[float]]:
        """Call ``/v1/embeddings`` and return the raw embedding vectors in input order."""
        r = self._client.post("/v1/embeddings", json={"model": model, "input": texts})
        r.raise_for_status()
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data]

    def rag_query(
        self, collection: str, query: str, k: int = 6, answer: bool = False
    ) -> dict:
        """Call ``/v1/hearth/rag/query`` and return the parsed response (chunks + answer)."""
        r = self._client.post(
            "/v1/hearth/rag/query",
            json={"collection": collection, "query": query, "k": k, "answer": answer},
        )
        r.raise_for_status()
        return r.json()


__all__ = ["HearthClient"]
