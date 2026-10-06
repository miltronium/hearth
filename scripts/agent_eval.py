#!/usr/bin/env python
"""Run the agent tool-use eval (B-013) over a pinned corpus and report what happened.

The spec (``data/agent_eval.yaml``) pins a commit; ``docs/`` at that commit is extracted with
``git archive`` into a temporary directory, which becomes the agent's only file root. Before any
model runs, the answer key is checked against that corpus (each evidence term must occur in
its file, and for ``find`` items in no other file), so a bad pin cannot be scored as a model
failure, or worse, as a pass.

Scoring is per item, on the final answer only:

* every kind: the expected answer must appear as a token (word-bounded, case-insensitive);
* ``find``: the answer must also name no *other* corpus file, so listing every file in the
  directory does not pass.

This is a measurement, not a gate. n=10 is below the promotion gate's ``min_n`` (30), so the
summary says so; it can show failures, it cannot license a step-cap claim.

Usage:
    uv run --no-sync python scripts/agent_eval.py --check-only
    uv run --no-sync python scripts/agent_eval.py --max-iterations 6 --out /tmp/agent_eval.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SPEC = REPO / "data" / "agent_eval.yaml"
KINDS = ("find", "read", "two-step")
GATE_MIN_N = 30


@dataclass(frozen=True)
class Item:
    id: str
    kind: str
    question: str
    answer: str
    evidence_term: str
    evidence_file: str


@dataclass
class Outcome:
    item: Item
    correct: bool
    reason: str
    stopped_reason: str
    answer: str | None
    iterations: int
    tools: list[str] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    total_tokens: int = 0
    seconds: float = 0.0


def load_spec(path: Path) -> tuple[str, str, list[Item]]:
    spec = yaml.safe_load(path.read_text())
    corpus = spec["corpus"]
    items = []
    seen = set()
    for raw in spec["items"]:
        item = Item(
            id=raw["id"],
            kind=raw["kind"],
            question=raw["question"],
            answer=str(raw["answer"]),
            evidence_term=raw["evidence"]["term"],
            evidence_file=raw["evidence"]["file"],
        )
        if item.kind not in KINDS:
            raise ValueError(f"{item.id}: unknown kind {item.kind!r}")
        if item.id in seen:
            raise ValueError(f"duplicate id {item.id!r}")
        seen.add(item.id)
        items.append(item)
    return corpus["commit"], corpus["path"], items


def extract_corpus(commit: str, path: str, dest: Path) -> Path:
    """``git archive <commit> <path>`` into ``dest``; returns the extracted corpus root."""
    archive = subprocess.run(
        ["git", "-C", str(REPO), "archive", "--format=tar", commit, path],
        check=True,
        capture_output=True,
    )
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive.stdout, check=True)
    root = dest / path
    if not root.is_dir():
        raise RuntimeError(f"{path!r} is not a directory at {commit}")
    return root


def corpus_dir(commit: str) -> Path:
    """A fixed extraction directory for ``commit``, recreated empty on every run.

    Not a random ``mkdtemp`` name: the corpus path appears in every tool observation, so a
    different directory name per run is different prompt text, and greedy decoding diverged on
    it. With random names the same code scored 9/10 and then 8/10, which made every comparison
    between tool versions noise.
    """
    path = Path(tempfile.gettempdir()) / f"hearth-agent-eval-{commit[:12]}"
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path


def corpus_files(root: Path) -> dict[str, str]:
    """Relative path -> text, for every file under the corpus root."""
    return {
        str(p.relative_to(root)): p.read_text(errors="replace")
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def verify_key(items: list[Item], files: dict[str, str]) -> list[str]:
    """Problems with the answer key against the pinned corpus; empty means the key holds."""
    problems = []
    for item in items:
        text = files.get(item.evidence_file)
        if text is None:
            problems.append(f"{item.id}: {item.evidence_file} is not in the corpus")
            continue
        if item.evidence_term not in text:
            problems.append(f"{item.id}: {item.evidence_term!r} not in {item.evidence_file}")
        if item.kind == "find":
            others = sorted(
                rel for rel, t in files.items()
                if rel != item.evidence_file and item.evidence_term in t
            )
            if others:
                problems.append(f"{item.id}: {item.evidence_term!r} is also in {others}")
            if Path(item.evidence_file).name != item.answer:
                problems.append(
                    f"{item.id}: answer {item.answer!r} is not the evidence file's name"
                )
    return problems


def _token(text: str) -> re.Pattern[str]:
    # Bounded so "30" is not found inside "300" or "30.15", while a sentence-final period
    # ("the default is 30.") still counts.
    return re.compile(r"(?<![\w.])" + re.escape(text) + r"(?!\w|\.\w)", re.IGNORECASE)


def score(item: Item, answer: str | None, names: set[str]) -> tuple[bool, str]:
    """Whether a final answer is correct for ``item``, and why not when it is not."""
    if answer is None:
        return False, "no answer"
    if not _token(item.answer).search(answer):
        return False, f"expected {item.answer!r} not in the answer"
    if item.kind == "find":
        rest = _token(item.answer).sub(" ", answer)
        named = sorted(n for n in names if n != item.answer and _token(n).search(rest))
        if named:
            return False, f"also names {named}"
    return True, "ok"


def run_items(items: list[Item], root: Path, args: argparse.Namespace) -> list[Outcome]:
    from hearth.agent import Agent, Budget, local_toolset
    from hearth.config import Settings
    from hearth.providers import select_provider
    from hearth.router import Router

    settings = Settings(file_roots=str(root))
    provider = select_provider(settings)
    router = Router(local_provider=provider)
    tools = local_toolset(settings=settings)
    budget = Budget(
        max_iterations=args.max_iterations,
        max_seconds=args.max_seconds,
        max_total_tokens=args.max_tokens,
    )
    names = {Path(rel).name for rel in corpus_files(root)}
    outcomes = []
    for item in items:
        run = Agent(router, tools, budget=budget, model=args.model).run(item.question)
        correct, reason = score(item, run.answer, names)
        outcomes.append(
            Outcome(
                item=item,
                correct=correct,
                reason=reason,
                stopped_reason=str(run.stopped_reason),
                answer=run.answer,
                iterations=run.iterations,
                tools=[s.tool for s in run.steps if s.tool],
                # What the model was shown at each step, so a failure can be traced to the
                # tool (a truncated read, a context-free search hit) or to the model.
                steps=[
                    {
                        "kind": str(s.kind),
                        "tool": s.tool,
                        "arguments": s.arguments,
                        "observation": s.observation,
                        "error": s.error,
                    }
                    for s in run.steps
                ],
                models=sorted({s.model for s in run.steps if s.model}),
                total_tokens=run.total_tokens,
                seconds=round(run.elapsed_seconds, 2),
            )
        )
        mark = "PASS" if correct else "FAIL"
        print(
            f"{mark}  {item.id:<28} {run.iterations} step(s)  "
            f"{' > '.join(outcomes[-1].tools) or '-':<40} {reason}",
            flush=True,
        )
    return outcomes


def summarize(outcomes: list[Outcome], args: argparse.Namespace, commit: str) -> dict:
    n = len(outcomes)
    correct = [o for o in outcomes if o.correct]
    by_kind = {
        k: f"{sum(o.correct for o in outcomes if o.item.kind == k)}"
        f"/{sum(o.item.kind == k for o in outcomes)}"
        for k in KINDS
    }
    return {
        "corpus_commit": commit,
        "model": args.model,
        "served_by": sorted({m for o in outcomes for m in o.models}),
        "max_iterations": args.max_iterations,
        "n": n,
        "correct": len(correct),
        "by_kind": by_kind,
        "stopped_reasons": {
            r: sum(o.stopped_reason == r for o in outcomes)
            for r in sorted({o.stopped_reason for o in outcomes})
        },
        "steps_when_correct": sorted(o.iterations for o in correct),
        "hit_step_cap": sum(o.iterations >= args.max_iterations for o in outcomes),
        "below_gate_min_n": n < GATE_MIN_N,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--model", default="auto")
    ap.add_argument("--max-iterations", type=int, default=8)
    ap.add_argument("--max-seconds", type=float, default=180.0)
    ap.add_argument("--max-tokens", type=int, default=24000)
    ap.add_argument("--out", type=Path, help="write the full JSON report here")
    ap.add_argument("--check-only", action="store_true", help="verify the key; run no model")
    args = ap.parse_args()

    commit, path, items = load_spec(args.spec)
    workdir = corpus_dir(commit)
    try:
        root = extract_corpus(commit, path, workdir)
        problems = verify_key(items, corpus_files(root))
        if problems:
            print("ANSWER KEY DOES NOT HOLD at the pinned commit:", file=sys.stderr)
            for p in problems:
                print(f"  {p}", file=sys.stderr)
            return 2
        print(f"key ok: {len(items)} items verified against {path}/ at {commit[:12]}")
        if args.check_only:
            return 0
        outcomes = run_items(items, root, args)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    summary = summarize(outcomes, args, commit)
    print(json.dumps(summary, indent=2))
    if summary["below_gate_min_n"]:
        print(
            f"n={summary['n']} is below the gate's min_n={GATE_MIN_N}: a smoke measurement. "
            "It can show failures; it cannot license a step-cap or tool-style claim."
        )
    if args.out:
        report = {
            "summary": summary,
            "items": [
                {**o.__dict__, "item": o.item.__dict__} for o in outcomes
            ],
        }
        args.out.write_text(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
