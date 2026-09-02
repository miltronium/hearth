"""On-disk handoff store — three directories and a gate, no transport.

Layout under ``~/.hearth/handoff`` (``HEARTH_HOME`` moves it, matching
``hearth.config.Settings.home``; resolved from the environment here so this package keeps
its stdlib-only import graph):

    drafts/    envelopes being written and reviewed
    released/  envelopes a human approved — the pickup point
    inbox/     answers carried back in (ingest.py)

**"Release" does not send anything.** There is no code in HEARTH that can. :meth:`release`
verifies an approval that still matches the payload, writes the file into ``released/``, and
returns the path. Moving those bytes off the machine is a separate act performed by a human
with tools outside HEARTH — which is exactly why ``lsof`` on the HEARTH process remains a
valid proof of no egress (``docs/PRIVACY.md``).

Envelopes and answers are plaintext copies of the material they describe. Like the RAG index,
they are real data at rest: keep ``~/.hearth`` on an encrypted volume and purge with
:meth:`purge` when a handoff is done.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .envelope import HandoffEnvelope, HandoffError
from .ingest import ExternalAnswer

DRAFTS = "drafts"
RELEASED = "released"
INBOX = "inbox"
_SUBDIRS = (DRAFTS, RELEASED, INBOX)


class ReleaseRefusedError(HandoffError):
    """Raised when an envelope is released without a valid, current human approval."""


def default_root() -> Path:
    """Return ``$HEARTH_HOME/handoff`` (default ``~/.hearth/handoff``)."""
    home = os.environ.get("HEARTH_HOME") or (Path.home() / ".hearth")
    return Path(home).expanduser() / "handoff"


class HandoffStore:
    """Filesystem store for envelopes and returned answers.

    Directories are created lazily on first write, so constructing a store touches nothing.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_root()

    def dir(self, name: str) -> Path:
        """Return one of the store's subdirectories, creating it if needed."""
        if name not in _SUBDIRS:
            raise HandoffError(f"unknown handoff directory {name!r}; expected one of {_SUBDIRS}")
        path = self.root / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    # -- envelopes ---------------------------------------------------------------------

    def save_draft(self, envelope: HandoffEnvelope) -> Path:
        """Write ``envelope`` to ``drafts/``. Overwrites an earlier draft of the same id."""
        envelope.validate()
        return _write_json(self.dir(DRAFTS) / f"{envelope.id}.json", envelope.to_json())

    def release(self, envelope: HandoffEnvelope) -> Path:
        """Move an approved envelope into ``released/`` and return its path.

        Refuses unless :attr:`HandoffEnvelope.is_approved` — i.e. a human approved it *and*
        the payload has not changed since. Editing a prompt after review invalidates the
        review rather than inheriting it.

        This writes a file. It does not transmit, upload, or connect to anything.
        """
        envelope.validate()
        if envelope.review is None:
            raise ReleaseRefusedError(
                f"envelope {envelope.id} has never been reviewed — nothing crosses the "
                "boundary without a human having seen it (hearth.handoff.review)"
            )
        if not envelope.is_approved:
            reason = (
                "the review was a rejection"
                if envelope.review.decision != "approved"
                else "the payload changed after it was approved"
            )
            raise ReleaseRefusedError(f"envelope {envelope.id} is not releasable: {reason}")
        path = _write_json(self.dir(RELEASED) / f"{envelope.id}.json", envelope.to_json())
        draft = self.root / DRAFTS / f"{envelope.id}.json"
        draft.unlink(missing_ok=True)
        return path

    def load_envelope(self, path: Path) -> HandoffEnvelope:
        """Read and validate an envelope from ``path``."""
        return HandoffEnvelope.from_json(json.loads(Path(path).read_text(encoding="utf-8")))

    def list_envelopes(self, name: str = DRAFTS) -> list[HandoffEnvelope]:
        """Return the envelopes in one directory, sorted by id (which sorts by time)."""
        return [self.load_envelope(p) for p in sorted(self.dir(name).glob("*.json"))]

    # -- returned answers --------------------------------------------------------------

    def save_answer(self, record: ExternalAnswer) -> Path:
        """Write an ingested answer to ``inbox/``."""
        record.validate()
        return _write_json(self.dir(INBOX) / f"{record.id}.json", record.to_json())

    def load_answer(self, path: Path) -> ExternalAnswer:
        """Read and validate an ingested answer from ``path``."""
        return ExternalAnswer.from_json(json.loads(Path(path).read_text(encoding="utf-8")))

    def list_answers(self) -> list[ExternalAnswer]:
        """Return every ingested answer, sorted by id."""
        return [self.load_answer(p) for p in sorted(self.dir(INBOX).glob("*.json"))]

    # -- housekeeping ------------------------------------------------------------------

    def purge(self, name: str) -> int:
        """Delete every artifact in one directory. Returns how many files were removed.

        These files are plaintext copies of the material they describe; purging a finished
        handoff is the cheapest way to shrink what is sitting at rest on the machine.
        """
        removed = 0
        for path in self.dir(name).glob("*.json"):
            path.unlink()
            removed += 1
        return removed


def _write_json(path: Path, obj: dict) -> Path:
    """Write ``obj`` as indented JSON with owner-only permissions."""
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)
    return path
