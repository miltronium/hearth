#!/usr/bin/env python3
"""cmux × HEARTH local orchestrator (C4) — read panes, triage with on-device HEARTH, notify.

A control loop that, for each terminal pane in a running cmux, reads the pane's recent screen
output, asks a LOCAL HEARTH model to classify what state the pane's agent is in (working / waiting
on input / done / error), and drives `cmux notify` for the panes that need a human — so you can run
many agents in parallel and be told which one to look at.

PRIVACY (docs/cmux/PRIVACY.md, ADR-C002): pane text goes to exactly one place — the local HEARTH
router built via ``build_toolset`` with ``allow_escalation=False`` (mcp/tools.py). The cmux client
only speaks to the local Unix socket. There is no other network path; run it under `cmux-sealed` and
`cmux_egress_probe.sh` stays loopback-only. Do not add a transport that ships pane text off-box.

cmux CLI protocol used (mapped from the cmux source): `cmux --json list-workspaces` /
`list-pane-surfaces --workspace <id>` to enumerate, `read-screen --surface <id> --lines N` -> {"text"},
`notify --surface <id> --title .. --body ..`, `send-key --surface <id> <key>`. Socket via
`--socket $CMUX_SOCKET_PATH`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Protocol

# cmux ships its CLI *inside* the app bundle, not on PATH. Auto-locate it so the orchestrator works
# from a pane without manual setup. NOTE on access: by default (automation.socketControlMode =
# "cmuxOnly") the socket only accepts processes DESCENDED FROM cmux — it injects
# CMUX_SOCKET_PASSWORD into pane shells — so the orchestrator has to run inside a pane. Setting the
# mode to "password" lifts that: any local process presenting the socket password may connect, which
# is how this runs from an ordinary terminal. See scripts/cmux/cmux-auth-env.
_CMUX_BUNDLE_BINS = [
    "/Applications/cmux.app/Contents/Resources/bin/cmux",
    os.path.expanduser("~/Applications/cmux.app/Contents/Resources/bin/cmux"),
]


def default_cmux_bin() -> str:
    """Resolve the cmux CLI: $CMUX_BIN, then PATH, then the app bundle, else bare 'cmux'."""
    env = os.environ.get("CMUX_BIN")
    if env:
        return env
    on_path = shutil.which("cmux")
    if on_path:
        return on_path
    for candidate in _CMUX_BUNDLE_BINS:
        if os.path.exists(candidate):
            return candidate
    return "cmux"

# Pane states the local model classifies into, and how each maps to a triage priority.
STATES = ["working", "waiting", "done", "error"]
_PRIORITY = {"waiting": "attention", "error": "attention", "done": "info", "working": "none"}


@dataclass
class Surface:
    """A cmux surface (pane content). ``kind`` is the cmux surface type (terminal/browser/…)."""
    surface_id: str
    title: str = ""
    kind: str = "terminal"


@dataclass
class Triage:
    surface_id: str
    title: str
    state: str          # one of STATES
    priority: str       # attention | info | none
    message: str = ""   # short note used as the notification body
    notified: bool = False


# --- transport abstraction: a real CLI client and a fake, behind one Protocol ------------------

class CmuxClient(Protocol):
    def list_surfaces(self) -> list[Surface]: ...
    def read_screen(self, surface_id: str, lines: int = 80) -> str: ...
    def notify(self, surface_id: str, title: str, body: str) -> None: ...
    def send_key(self, surface_id: str, key: str) -> None: ...


class CmuxCliClient:
    """Talks to a running cmux via its `cmux` CLI over the local Unix socket."""

    def __init__(self, cmux_bin: str | None = None, socket_path: str | None = None, timeout: float = 15.0):
        self.base = [cmux_bin or default_cmux_bin()]
        sp = socket_path or os.environ.get("CMUX_SOCKET_PATH")
        if sp:
            self.base += ["--socket", sp]
        self.timeout = timeout

    def _json(self, *args: str) -> dict:
        # --id-format both: cmux defaults to short refs ("surface:1") and omits UUIDs entirely.
        # Refs are positional and renumber as workspaces/surfaces come and go, so a ref captured
        # during enumeration can resolve to a different surface — or nothing — by the time we
        # notify against it. Asking for UUIDs gives every surface a stable identity.
        out = subprocess.run(
            self.base + ["--json", "--id-format", "both", *args],
            capture_output=True, text=True, timeout=self.timeout,
        )
        out.check_returncode()
        return json.loads(out.stdout or "{}")

    def _run(self, *args: str) -> None:
        proc = subprocess.run(self.base + list(args), capture_output=True, text=True, timeout=self.timeout)
        if proc.returncode != 0:
            # cmux reports failures like "Surface ref not found" on stdout with rc=1; surface that
            # text instead of a bare CalledProcessError that says only "returned non-zero".
            detail = (proc.stderr or proc.stdout or "").strip().splitlines()
            raise subprocess.CalledProcessError(
                proc.returncode, proc.args, output=proc.stdout, stderr=detail[0] if detail else "",
            )

    def list_surfaces(self) -> list[Surface]:
        surfaces: list[Surface] = []
        for ws in self._json("list-workspaces").get("workspaces", []):
            wid = ws.get("id") or ws.get("ref")
            if not wid:
                continue
            for s in self._json("list-pane-surfaces", "--workspace", str(wid)).get("surfaces", []):
                sid = s.get("id") or s.get("ref")
                if sid:
                    surfaces.append(Surface(str(sid), title=str(s.get("title", "")), kind=str(s.get("type", "terminal"))))
        return surfaces

    def read_screen(self, surface_id: str, lines: int = 80) -> str:
        return str(self._json("read-screen", "--surface", surface_id, "--lines", str(lines)).get("text", ""))

    def notify(self, surface_id: str, title: str, body: str) -> None:
        self._run("notify", "--surface", surface_id, "--title", title, "--body", body)

    def send_key(self, surface_id: str, key: str) -> None:
        self._run("send-key", "--surface", surface_id, key)


class FakeCmuxClient:
    """In-memory cmux for tests/demo. Seed ``screens``; inspect ``notifications`` after a run."""

    def __init__(self, screens: dict[str, str], titles: dict[str, str] | None = None, kinds: dict[str, str] | None = None):
        self._screens = screens
        self._titles = titles or {}
        self._kinds = kinds or {}
        self.notifications: list[tuple[str, str, str]] = []
        self.keys_sent: list[tuple[str, str]] = []

    def list_surfaces(self) -> list[Surface]:
        return [Surface(sid, self._titles.get(sid, sid), self._kinds.get(sid, "terminal")) for sid in self._screens]

    def read_screen(self, surface_id: str, lines: int = 80) -> str:
        return self._screens.get(surface_id, "")

    def notify(self, surface_id: str, title: str, body: str) -> None:
        self.notifications.append((surface_id, title, body))

    def send_key(self, surface_id: str, key: str) -> None:
        self.keys_sent.append((surface_id, key))


# --- triage core (pure; injectable classify/summarize so it is deterministic in tests) ---------

def priority_for(state: str) -> str:
    """Map a pane state to a triage priority. Unknown ⇒ 'none' (never a spurious notify)."""
    return _PRIORITY.get(state, "none")


def triage_text(
    text: str,
    classify: Callable[[str, list[str]], str],
    summarize: Callable[[str], str] | None = None,
) -> tuple[str, str, str]:
    """Return (state, priority, message). ``classify`` picks a state; ``summarize`` (optional)
    produces a one-line body for panes that warrant a notification."""
    raw = (classify(_triage_prompt(text), STATES) or "").strip().lower()
    state = next((s for s in STATES if s in raw), "working")  # tolerant; default no-attention
    priority = priority_for(state)
    message = ""
    if priority != "none" and summarize is not None:
        message = (summarize(text) or "").strip().replace("\n", " ")[:160]
    return state, priority, message


def _triage_prompt(text: str) -> str:
    tail = text[-2000:]
    return (
        "You are triaging a terminal pane running a coding agent. Based on the recent output, reply "
        "with exactly one word describing its state: 'working' (actively running, no input needed), "
        "'waiting' (paused, awaiting user input or confirmation), 'done' (task finished / idle at a "
        "prompt), or 'error' (failed / crashed). Recent output:\n\n" + tail
    )


def run_once(
    client: CmuxClient,
    classify: Callable[[str, list[str]], str],
    summarize: Callable[[str], str] | None = None,
    *,
    lines: int = 80,
    do_notify: bool = True,
) -> list[Triage]:
    """One triage sweep: read every terminal pane, classify, notify the ones needing attention."""
    results: list[Triage] = []
    for surf in client.list_surfaces():
        if surf.kind != "terminal":
            continue  # browsers etc. are not agent panes
        text = client.read_screen(surf.surface_id, lines)
        state, priority, message = triage_text(text, classify, summarize)
        t = Triage(surf.surface_id, surf.title, state, priority, message)
        if do_notify and priority != "none":
            body = message or f"pane is {state}"
            client.notify(surf.surface_id, f"cmux: {surf.title or surf.surface_id} — {state}", body)
            t.notified = True
        results.append(t)
    return results


# --- HEARTH-backed classify/summarize (local, escalation off) ----------------------------------

def hearth_callables():
    """Build local HEARTH classify/summarize callables (same path as the MCP offload)."""
    from hearth.mcp.tools import build_toolset

    tools = build_toolset()
    return (lambda text, labels: tools.classify(text, labels)), (lambda text: tools.summarize(text, max_words=25))


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="cmux × HEARTH local orchestrator (one triage sweep).")
    ap.add_argument("--lines", type=int, default=80, help="screen lines per pane to read")
    ap.add_argument("--dry-run", action="store_true", help="triage but do not send notifications")
    ap.add_argument("--socket", default=None, help="cmux socket path (else $CMUX_SOCKET_PATH)")
    ap.add_argument("--cmux-bin", default=None, help="path to the cmux CLI (else $CMUX_BIN / PATH / app bundle)")
    args = ap.parse_args()

    # Two ways to be allowed on the socket (see docs/cmux/RUNBOOK_orchestrator.md):
    #   1. run INSIDE a cmux pane           — works under the default socketControlMode "cmuxOnly"
    #   2. present the socket password      — requires socketControlMode "password"
    # The cmux CLI also auto-reads the password from ~/.local/state/cmux/socket-control-password,
    # so mode "password" often just works; this note only fires when neither route is evident.
    _pw_store = os.path.expanduser("~/.local/state/cmux/socket-control-password")
    if not (os.environ.get("CMUX_SOCKET_PASSWORD") or os.environ.get("CMUX_SURFACE_ID")
            or os.path.exists(_pw_store)):
        print("note: not inside a cmux pane and no socket password found — either run this in a "
              "pane, or set Settings ▸ Automation ▸ socketControlMode to 'password' and "
              "`source scripts/cmux/cmux-auth-env`.")

    client = CmuxCliClient(cmux_bin=args.cmux_bin, socket_path=args.socket)
    classify, summarize = hearth_callables()
    try:
        results = run_once(client, classify, summarize, lines=args.lines, do_notify=not args.dry_run)
    except (subprocess.SubprocessError, OSError, json.JSONDecodeError) as exc:
        print(f"orchestrator: could not reach cmux ({exc}). Run inside a cmux pane; check --socket/--cmux-bin.")
        return 1
    for t in results:
        flag = "🔔" if t.notified else ("· " if t.priority == "none" else "  ")
        print(f"{flag} [{t.state:>7}] {t.title or t.surface_id}  {('- ' + t.message) if t.message else ''}")
    # Count what WARRANTS attention, not what was sent — under --dry-run nothing is ever sent, so
    # counting `notified` reported a flat 0/N and made a working sweep look like it found nothing.
    flagged = sum(1 for t in results if t.priority != "none")
    verb = "would be flagged" if args.dry_run else "flagged"
    print(f"\n{flagged}/{len(results)} panes {verb} for attention (local triage, 0 frontier tokens).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
