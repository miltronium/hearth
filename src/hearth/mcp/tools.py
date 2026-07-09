"""MCP tool logic — pure, dependency-free functions (ADR-010, Phase 5).

These are the *tools* the HEARTH MCP server exposes, but they carry **no MCP dependency**
so they can be unit-tested against the echo router with only core deps installed. Each one
drives HEARTH's router in-process with ``allow_escalation=False`` — the delegation is
strictly local, so an agent offloading work here never spends its frontier budget.

:func:`build_toolset` wires a :class:`HearthTools` bound to a shared router + RAG index;
:mod:`hearth.mcp.server` registers each bound method as an MCP tool. Every function returns
plain strings / small dicts so the MCP layer can serialize them trivially.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings, get_settings
from ..memory import RagIndex, SQLiteVectorStore, select_embedder
from ..providers import select_provider
from ..providers.base import GenRequest, Message
from ..router import Router


def _route_local(router: Router, prompt: str, intent: str) -> str:
    """Run one hard-local completion through the router and return its text.

    ``allow_escalation=False`` is the whole point: an MCP client offloading a subtask must
    never trigger a surprise remote call on HEARTH's side.
    """
    routed = router.route(
        GenRequest(messages=[Message(role="user", content=prompt)], model="auto"),
        intent=intent,
        allow_escalation=False,
    )
    return routed.result.text.strip()


@dataclass
class HearthTools:
    """Bound HEARTH tools sharing one router + RAG index.

    Constructed via :func:`build_toolset`. Kept as an object (not module-level functions)
    so the router/RAG index are injectable — tests pass an echo router, the server passes
    the configured one, and both exercise the identical logic.
    """

    router: Router
    rag: RagIndex

    def summarize(self, text: str, max_words: int | None = None) -> str:
        """Summarize ``text`` locally. ``max_words`` caps the requested length when given."""
        limit = f" in at most {max_words} words" if max_words else ""
        prompt = f"Summarize the following text{limit}:\n\n{text}"
        return _route_local(self.router, prompt, intent="summarize")

    def classify(self, text: str, labels: list[str]) -> str:
        """Classify ``text`` into exactly one of ``labels`` (returns the chosen label)."""
        if not labels:
            raise ValueError("classify requires at least one label")
        options = ", ".join(labels)
        prompt = (
            f"Classify the following text into exactly one of these labels: {options}.\n"
            f"Reply with only the label.\n\nText:\n{text}"
        )
        return _route_local(self.router, prompt, intent="classify")

    def extract(self, text: str, fields: list[str]) -> dict[str, str]:
        """Extract ``fields`` from ``text`` locally; returns ``{field: value}`` (empty if absent).

        The local model produces free-form text; this maps each requested field to a value
        by asking for a ``field: value`` line per field and parsing them back, so the return
        shape is stable regardless of the model's phrasing.
        """
        if not fields:
            raise ValueError("extract requires at least one field")
        wanted = ", ".join(fields)
        prompt = (
            f"Extract these fields from the text: {wanted}.\n"
            "Reply with one 'field: value' per line, using an empty value if a field is "
            f"absent.\n\nText:\n{text}"
        )
        raw = _route_local(self.router, prompt, intent="extract")
        return _parse_fields(raw, fields)

    def draft(self, instruction: str, context: str | None = None) -> str:
        """Draft text (commit message, prose, boilerplate) locally from ``instruction``."""
        prompt = instruction if not context else f"{instruction}\n\nContext:\n{context}"
        return _route_local(self.router, prompt, intent="draft")

    def rag_query(
        self, collection: str, query: str, k: int = 6, answer: bool = False
    ) -> dict:
        """Retrieve top-``k`` chunks from ``collection``; optionally answer locally.

        Returns ``{"chunks": [{text, source, score}, ...], "answer": str | None}`` — a small
        dict the MCP layer serializes directly. ``answer=True`` runs the local model over
        the retrieved chunks (still ``allow_escalation=False`` inside :class:`RagIndex`).
        """
        result = self.rag.query(collection, query, k=k, answer=answer)
        return {
            "chunks": [
                {"text": c.text, "source": c.source, "score": c.score}
                for c in result.chunks
            ],
            "answer": result.answer,
        }


def build_toolset(
    settings: Settings | None = None,
    router: Router | None = None,
    rag: RagIndex | None = None,
) -> HearthTools:
    """Assemble a :class:`HearthTools` with a local-only router + offline RAG index.

    Mirrors the gateway's wiring (:func:`hearth.gateway.create_app`) so MCP tools and the
    HTTP API share identical routing/RAG behavior. All parts are injectable for tests.
    """
    settings = settings or get_settings()
    router = router or Router(local_provider=select_provider(settings))
    rag = rag or RagIndex(
        embedder=select_embedder(settings),
        store=SQLiteVectorStore(settings=settings),
        router=router,
    )
    return HearthTools(router=router, rag=rag)


def _parse_fields(raw: str, fields: list[str]) -> dict[str, str]:
    """Parse ``field: value`` lines back into ``{field: value}`` for every requested field.

    Matching is case-insensitive on the field name; requested fields the model didn't emit
    default to ``""`` so the returned dict always has exactly ``fields`` as its keys.
    """
    found: dict[str, str] = {}
    lower_to_field = {f.lower(): f for f in fields}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        canonical = lower_to_field.get(key.strip().lower())
        if canonical is not None and canonical not in found:
            found[canonical] = value.strip()
    return {f: found.get(f, "") for f in fields}


__all__ = ["HearthTools", "build_toolset"]
