#!/usr/bin/env python3
"""Generate ``man/hearth.1`` (roff) from the Typer app and ``hearth.config.Settings``.

WHAT: every command, subcommand, argument and option in the man page is read off the live
Typer app (``hearth.cli.app``): its name, usage, docstring, option names, types, defaults and
help text. The ENVIRONMENT section is generated from ``Settings.model_fields`` plus
``hearth.status.probes._EXTRA_ENV_NAMES`` (the ``HEARTH_*`` names read outside Settings).
The prose sections (DESCRIPTION, QUICK START, PRIVACY AND OFFLINE USE, FILES, EXIT STATUS,
EXAMPLES, SEE ALSO) are written here.

WHY: a hand-maintained man page drifts from the CLI the first time an option changes.
``tests/test_manpage.py`` regenerates this in memory and compares it with the committed
file, so a command or option change that was not regenerated fails the suite.

Run from the repo root (never a bare ``uv run``; it prunes the venv):

    uv run --no-sync python scripts/gen_manpage.py          # write man/hearth.1
    uv run --no-sync python scripts/gen_manpage.py --check  # exit 1 if it is stale
    man ./man/hearth.1                                      # read it

The output is deterministic: no dates, no machine paths (``Path.home()`` renders as ``~``),
nothing that depends on the terminal width.
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "man" / "hearth.1"

# Generate from THIS checkout's source, whatever the venv has installed.
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import typer  # noqa: E402

#: Order of the COMMANDS section: the CLI's own help panels, in learning order.
PANEL_ORDER = ("Start here", "Use", "Train and evaluate", "Extend")

#: One line per Settings field. A new field without an entry renders as UNDOCUMENTED, and
#: ``tests/test_manpage.py`` fails on that text, so the man page cannot silently omit it.
SETTINGS_DOCS: dict[str, str] = {
    "host": "Bind address for hearth serve. Anything but loopback (127.0.0.1, ::1, "
    "localhost) exposes the API to the network and makes doctor --offline UNSAFE. "
    "--host overrides it.",
    "port": "Bind port for hearth serve. --port overrides it.",
    "backend": "Inference backend: auto (mlx when mlx-lm imports, otherwise the echo stub, "
    "silently), mlx (fail loudly if MLX is missing), echo (deterministic test stub, no "
    "weights), or a plugin name. Check backend= in the serve banner.",
    "require_auth": "Require 'Authorization: Bearer TOKEN' on every /v1/* route. false is "
    "for throwaway local testing only.",
    "embedder": "RAG embedder: hash (offline, lexical; the default), mlx (does not work "
    "today, docs/BUGS.md B-011), or a plugin name.",
    "embed_dim": "Vector dimension of the hash embedder.",
    "embed_model": "Model id of the mlx embedder (only used when HEARTH_EMBEDDER=mlx).",
    "vector_store": "RAG vector store: sqlite (default), sqlite-vec (needs the vec extra), "
    "or a plugin name.",
    "ram_ceiling_gb": "RAM budget (GB, registry estimates) for resident models. Loading a "
    "model that would exceed it evicts the least-recently-used one. The real GPU "
    "working-set ceiling of a 36 GB M3 Pro is about 30 GB; scripts/hearth_status.py "
    "measures it.",
    "warmup": "Load the default model in the background when hearth serve starts, so the "
    "first request is warm and /v1/hearth/admin/ready can turn 200. No effect on echo.",
    "file_roots": "Colon-separated directories the agent's read_file / list_files and the "
    "MCP *_file tools may read under. Deny-by-default: empty refuses every read; there is "
    "no implicit root. ~ is expanded; entries that are not existing directories are "
    "dropped.",
    "file_max_bytes": "Largest single file those tools will read; bigger files are refused, "
    "not truncated.",
    "allow_downloads": "Let a model load fetch weights that are not on disk. Leave it off: "
    "with it on, doctor --offline reports UNSAFE. hearth models pull downloads regardless.",
    "home": "Root of HEARTH's state: token, models/, rag/, adapters.json, train/, "
    "finance/ledger.db, handoff/. Point it at a scratch directory for experiments.",
}

#: The HEARTH_* names read outside Settings (status/probes.py:_EXTRA_ENV_NAMES).
EXTRA_DOCS: dict[str, tuple[str, str]] = {
    "HEARTH_DEFAULT_MODEL": (
        "the default: key of config/models.yaml",
        "Registry id served when a request names no model (or 'auto' finds no per-class "
        "rung). Must be a registered id: an unregistered value is currently ignored with a "
        "warning, doctor shows a WARN row, and /v1/hearth/admin/ready reports failed "
        "(docs/BUGS.md B-047 tracks making it a startup error). The name is "
        "HEARTH_DEFAULT_MODEL, not HEARTH_MODEL; misspelled HEARTH_* names are silently "
        "ignored. Confirm with hearth models list.",
    ),
    "HEARTH_ROUTING_YAML": (
        "<repo>/config/routing.yaml",
        "Routing profile: which task classes run locally and which may escalate. ~ is "
        "expanded and a relative path resolves against the repo root, not the current "
        "directory. A named file that does not exist is an error (serve, run and agent "
        "refuse to start; doctor --offline fails). The shipped profiles "
        "routing.yaml, routing.private.yaml and routing.finance.yaml define zero remotes.",
    ),
    "HEARTH_MODELS_YAML": ("<repo>/config/models.yaml", "Alternate model registry file."),
    "HEARTH_BASE_MODEL": (
        "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        "Base model for scripts/train_lora_real.sh only.",
    ),
    "HEARTH_TRAIN_DATA": ("unset", "Dataset for scripts/train_lora_real.sh only."),
    "HEARTH_TRAIN_TASK": ("extract", "Task class for scripts/train_lora_real.sh only."),
    "HEARTH_TRAIN_ITERS": ("200", "Iterations for scripts/train_lora_real.sh only."),
    "HEARTH_TRAIN_OUT": ("unset", "Output directory for scripts/train_lora_real.sh only."),
    "HEARTH_CANDIDATE_SCORE": (
        "unset",
        "scripts/train_lora_real.sh only; feeds the removed typed-score promote path.",
    ),
    "HEARTH_INCUMBENT_SCORE": (
        "unset",
        "scripts/train_lora_real.sh only; feeds the removed typed-score promote path.",
    ),
}

OTHER_ENV: tuple[tuple[str, str], ...] = (
    ("HF_HUB_CACHE, HF_HOME", "Where the model resolver's Hugging Face cache lookup looks "
     "(HF_HUB_CACHE, else HF_HOME/hub, else ~/.cache/huggingface/hub), after "
     "$HEARTH_HOME/models. Do not export them globally to paper over a split."),
    ("HF_ENDPOINT", "Mirror used by hearth models pull."),
    ("HF_HUB_OFFLINE", "Not needed by HEARTH's own load paths, which are disk-only "
     "already; scripts/hearth_private.sh sets it for any other hub call."),
    ("ANTHROPIC_API_KEY", "Only for the opt-in escalation profile config/routing.remote.yaml "
     "(needs the remote extra)."),
)

NAME_LINE = "hearth \\- local\\-first, no\\-egress LLM gateway for Apple Silicon"

DESCRIPTION = """\
HEARTH runs local MLX models (4\\-bit Qwen2.5, 3B to 14B) behind an OpenAI\\-compatible
HTTP API on 127.0.0.1:8080, so any OpenAI client can use it by changing base_url. It adds
a chat page (/chat), a bounded read\\-only tool\\-using agent, local RAG, an MCP server so
Claude Code can hand routine work to the local model, a finance pipeline whose arithmetic
is Python Decimal, and LoRA fine\\-tuning behind a statistical promotion gate.

Run it from the repository checkout as
.B uv run \\-\\-no\\-sync hearth
.IR command .
A bare
.B uv run
or a partial
.B uv sync
uninstalls mlx, mlx\\-lm and mcp, and HEARTH then silently falls back to the echo stub.
Install or repair with one command:
.B uv sync \\-\\-extra mlx \\-\\-extra mcp \\-\\-extra dev \\-\\-extra files
"""

QUICK_START = [
    ("uv run --no-sync python -c \"import mlx_lm, mcp, openpyxl, pypdf; print('ok')\"",
     "Verify the venv; must print ok."),
    ("hearth doctor --offline", "Is it safe to use HEARTH offline right now? Exit 0 = SAFE."),
    ("hearth models list", "Which models exist; (default) marks the one that serves."),
    ('hearth run "Say hello in five words."', "One local completion."),
    ("hearth serve", "API on http://127.0.0.1:8080/v1 and the chat page on /chat."),
    ('HEARTH_FILE_ROOTS=~/notes hearth agent "how many markdown files are there?"',
     "A bounded agent over directories you allow."),
    ("hearth stats", "Token savings for this process (zeros in a fresh shell)."),
]

PRIVACY = """\
The default routing profile (config/routing.yaml) defines zero remote models: every task
class runs locally, so the router has nowhere to send a prompt. Escalation to a frontier
model exists only under the opt\\-in config/routing.remote.yaml. hearth run, hearth agent
and every MCP tool never escalate, under any profile.

Model loads are disk\\-only. serve, chat, agent, MCP, RAG, train, models convert and models
export\\-coreml resolve weights from $HEARTH_HOME/models, then the Hugging Face hub cache,
and fail with ModelNotOnDiskError instead of downloading. hearth models pull is the only
command that downloads.

The gateway binds to loopback and requires a bearer token. Metrics hold token counts and
metadata, never prompt or response text, in memory only.

.B hearth doctor \\-\\-offline
measures this posture for the current shell's environment and exits 1 when it is unsafe.
It does not cover the calling agent (a cloud agent that read a file has already sent it;
hand HEARTH a path via the *_file MCP tools instead), machine\\-level containment
(firewalls, other processes), or data at rest (RAG collections, training runs and the
ledger hold real copies of your text). See docs/PRIVACY.md.
"""

FILES = [
    ("~/.hearth/token", "Bearer token, mode 0600, created on first serve."),
    ("~/.hearth/models/", "Weights downloaded by hearth models pull (hub layout)."),
    ("~/.hearth/rag/COLLECTION.db", "A RAG collection, including the raw chunk text."),
    ("~/.hearth/adapters.json", "The adapter registry (candidate / promoted / retired)."),
    ("~/.hearth/train/RUN_ID/", "Training runs and their adapters."),
    ("~/.hearth/finance/ledger.db", "The finance ledger; its presence enables the agent's "
     "finance tools."),
    ("config/models.yaml", "The model registry (override: HEARTH_MODELS_YAML)."),
    ("config/routing*.yaml", "Routing profiles (select: HEARTH_ROUTING_YAML)."),
    ("man/hearth.1", "This page, generated by scripts/gen_manpage.py."),
]

EXIT_STATUS = [
    ("0", "Success. For hearth agent: the model answered. For hearth doctor --offline: SAFE. "
     "For hearth eval without --promote: the gate was measured (read its PASS/FAIL line)."),
    ("1", "Failure or refusal: a fatal doctor check, UNSAFE offline, an empty prompt, an "
     "agent run that stopped at a bound, a refused promotion, a model not on disk, an "
     "invalid option value."),
    ("2", "Usage error (unknown option or command), or the work never started: an unknown "
     "--model for run or agent, an agent with nothing it can reach, the removed "
     "--candidate-score flags."),
]

EXAMPLES = [
    ("Check offline safety and see which file each model resolves from:",
     ["hearth doctor --offline; echo $?"]),
    ("Serve the two-tier local finance ladder (3B for classify, 14B for summarize):",
     ["HEARTH_ROUTING_YAML=config/routing.finance.yaml hearth doctor --offline",
      "HEARTH_ROUTING_YAML=config/routing.finance.yaml hearth serve"]),
    ("Call the API with curl:",
     ['export HEARTH_TOKEN="$(cat ~/.hearth/token)"',
      "curl -s http://127.0.0.1:8080/v1/chat/completions \\",
      '  -H "Authorization: Bearer $HEARTH_TOKEN" -H "Content-Type: application/json" \\',
      "  -d '{\"model\":\"auto\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}'"]),
    ("Pick a model per request, and see what is loaded:",
     ['hearth run --model mlx-community/Qwen2.5-3B-Instruct-4bit "hello"',
      'curl -s -H "Authorization: Bearer $HEARTH_TOKEN" \\',
      "  http://127.0.0.1:8080/v1/hearth/admin/models"]),
    ("Ask the agent about a directory, with the run as JSON:",
     ['HEARTH_FILE_ROOTS=~/notes hearth agent --json "how many CSV files are there?"']),
    ("Index notes and query them:",
     ["hearth rag ingest ~/notes --collection notes",
      'hearth rag query "descale the kettle" --collection notes --answer']),
    ("The promotion path for a trained adapter:",
     ["hearth prereg init --task classify --golden golden.jsonl --out prereg/classify.yaml",
      "$EDITOR prereg/classify.yaml   # write hypothesis, stopping_rule, kill_condition",
      "hearth prereg check prereg/classify.yaml --golden golden.jsonl   # fails: uncommitted",
      "git add prereg/classify.yaml && git commit -m 'prereg: classify'",
      "hearth eval ADAPTER_ID --golden golden.jsonl --prereg prereg/classify.yaml --promote"]),
    ("Try anything without a model (deterministic stub, scratch state):",
     ['HEARTH_BACKEND=echo HEARTH_HOME=/tmp/hearth-scratch hearth run "hello"']),
]

SEE_ALSO = (
    "docs/GUIDE.md (the user guide; start with \"Learn HEARTH in 15 minutes\"), "
    "docs/PRIVACY.md, docs/API.md, docs/AGENT.md, docs/RUNBOOK_finance.md, "
    "docs/RUNBOOK_training.md, docs/BUGS.md, docs/STATUS.md, scripts/hearth_status.py "
    "(measured status), uv(1)."
)


# ---------------------------------------------------------------------------------------
# roff helpers
# ---------------------------------------------------------------------------------------


def esc(text: str) -> str:
    """Escape plain text for roff: backslashes, hyphens, and control chars at line start."""
    out = text.replace("\\", "\\e").replace("-", "\\-")
    lines = []
    for line in out.split("\n"):
        if line.startswith((".", "'")):
            line = "\\&" + line
        lines.append(line)
    return "\n".join(lines)


def _code_block(lines: list[str]) -> list[str]:
    return [".RS 4", ".nf", *(esc(ln) for ln in lines), ".fi", ".RE"]


def doc_to_roff(doc: str) -> list[str]:
    """Render a command docstring (after its first line) as roff paragraphs.

    The CLI's docstrings follow one convention (enforced for examples by the help test):
    prose paragraphs separated by blank lines, and blocks like ``Examples:`` whose
    following lines are indented — those become no-fill code blocks.
    """
    body = inspect.cleandoc(doc or "").partition("\f")[0]
    paragraphs = body.split("\n\n")[1:]
    out: list[str] = []
    for para in paragraphs:
        lines = para.split("\n")
        prose: list[str] = []
        code: list[str] = []
        for line in lines:
            if line.startswith("  "):
                code.append(line[2:])
            else:
                if code:  # prose after code: flush the code first
                    out += [".PP", esc(" ".join(prose))] if prose else []
                    out += _code_block(code)
                    prose, code = [], []
                prose.append(line.strip())
        if prose:
            out += [".PP", esc(" ".join(prose))]
        if code:
            out += _code_block(code)
    return out


def _first_line(doc: str) -> str:
    return inspect.cleandoc(doc or "").split("\n\n")[0].replace("\n", " ").strip()


def _param_roff(param) -> list[str]:
    help_text = (getattr(param, "help", None) or "").strip()
    # Duck-typed: Typer ships its own vendored click, so isinstance against `click` fails.
    if param.param_type_name == "argument":
        meta = (param.metavar or param.name or "").upper().strip("[]")
        tag = f"\\fI{esc(meta)}\\fR"
        note = "required" if param.required else "optional"
        return [".TP", tag, esc(f"{help_text} ({note})".strip())]
    names = " , ".join(f"\\fB{esc(o)}\\fR" for o in param.opts)
    if param.secondary_opts:
        names += " / " + " , ".join(f"\\fB{esc(o)}\\fR" for o in param.secondary_opts)
    if not param.is_flag:
        names += f" \\fI{esc(param.type.name.upper())}\\fR"
    extras: list[str] = []
    if param.required:
        extras.append("required")
    default = param.default
    if param.is_flag and param.secondary_opts:
        chosen = param.opts[0] if default else param.secondary_opts[0]
        extras.append(f"default: {chosen}")
    elif not param.is_flag and default is not None and not callable(default):
        extras.append(f"default: {default}")
    text = help_text + (f" ({'; '.join(extras)})" if extras else "")
    return [".TP", names, esc(text)]


def _walk(cmd, path: list[str]):
    """Yield (path, command) for every leaf and group under ``cmd``, groups first."""
    yield path, cmd
    if hasattr(cmd, "commands"):
        for name, sub in cmd.commands.items():
            yield from _walk(sub, [*path, name])


def iter_commands(app: typer.Typer | None = None):
    """(path, click command) for every command below the root, in panel order."""
    if app is None:
        from hearth.cli import app as hearth_app

        app = hearth_app
    root = typer.main.get_command(app)
    assert hasattr(root, "commands"), "hearth's Typer app should build a command group"

    def panel_rank(item):
        _name, cmd = item
        panel = getattr(cmd, "rich_help_panel", None)
        return PANEL_ORDER.index(panel) if panel in PANEL_ORDER else len(PANEL_ORDER)

    top = sorted(root.commands.items(), key=panel_rank)  # stable: keeps app order in a panel
    for name, cmd in top:
        if getattr(cmd, "hidden", False):
            continue
        yield from _walk(cmd, ["hearth", name])


def _usage(cmd, path: list[str]) -> str:
    ctx = cmd.context_class(cmd, info_name=" ".join(path))
    return " ".join(cmd.collect_usage_pieces(ctx))


def _command_roff(path: list[str], cmd) -> list[str]:
    usage = _usage(cmd, path)
    out = [f'.SS "{esc(" ".join(path))}"']
    out += [".PP", f"\\fB{esc(' '.join(path))}\\fR {esc(usage)}".rstrip()]
    out += [".PP", esc(_first_line(cmd.help or ""))]
    out += doc_to_roff(cmd.help or "")
    params = [p for p in cmd.params if not getattr(p, "hidden", False)]
    if params:
        out += [".PP", "Arguments and options:", ".RS 4"]
        for p in params:
            out += _param_roff(p)
        out += [".RE"]
    return out


def _env_default(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Path):
        home = str(Path.home())
        text = str(value)
        return "~" + text[len(home):] if text.startswith(home) else text
    if value == "":
        return '"" (empty)'
    return str(value)


def environment_roff() -> list[str]:
    from hearth.config import Settings
    from hearth.status.probes import _EXTRA_ENV_NAMES

    out = [
        ".PP",
        esc(
            "Settings are read from HEARTH_* variables (env_prefix HEARTH_, extra ignored): a "
            "misspelled name is silently ignored; scripts/hearth_status.py --section "
            "environment flags it. hearth serve and hearth mcp read settings once at "
            "startup. Booleans accept 1/0/true/false/yes/no."
        ),
    ]
    for name, field in Settings.model_fields.items():
        doc = SETTINGS_DOCS.get(name, "UNDOCUMENTED: add it to SETTINGS_DOCS in "
                                      "scripts/gen_manpage.py.")
        out += [
            ".TP",
            f"\\fB{esc('HEARTH_' + name.upper())}\\fR "
            f"(default: {esc(_env_default(field.default))})",
            esc(doc),
        ]
    out += [".PP", esc("Read directly from the environment, outside Settings:")]
    for name in sorted(_EXTRA_ENV_NAMES):
        default, doc = EXTRA_DOCS.get(
            name, ("unknown", "UNDOCUMENTED: add it to EXTRA_DOCS in scripts/gen_manpage.py.")
        )
        out += [".TP", f"\\fB{esc(name)}\\fR (default: {esc(default)})", esc(doc)]
    out += [".PP", esc("Not HEARTH's, but they matter:")]
    for name, doc in OTHER_ENV:
        out += [".TP", f"\\fB{esc(name)}\\fR", esc(doc)]
    return out


def render() -> str:
    from hearth import __version__

    lines: list[str] = [
        '.\\" Generated by scripts/gen_manpage.py from the Typer app (src/hearth/cli.py),',
        '.\\" hearth.config.Settings and hearth.status.probes. Do not edit by hand; run:',
        '.\\"   uv run \\-\\-no\\-sync python scripts/gen_manpage.py',
        f'.TH HEARTH 1 "" "hearth {esc(__version__)}" "HEARTH Manual"',
        ".SH NAME",
        NAME_LINE,
        ".SH SYNOPSIS",
        ".B uv run \\-\\-no\\-sync hearth",
        ".I command",
        "[\\fIoptions\\fR] [\\fIarguments\\fR]",
        ".br",
        ".B hearth",
        ".I command",
        "\\fB\\-\\-help\\fR",
        ".SH DESCRIPTION",
        DESCRIPTION.rstrip("\n"),
        ".SH QUICK START",
        ".PP",
        "From the repo root; prefix each hearth command with",
        ".BR \"uv run \\-\\-no\\-sync\" :",
    ]
    for command, what in QUICK_START:
        lines += [".TP", f"\\fB{esc(command)}\\fR", esc(what)]
    lines += [".SH COMMANDS"]
    for path, cmd in iter_commands():
        lines += _command_roff(path, cmd)
    lines += [".SH PRIVACY AND OFFLINE USE", ".PP", PRIVACY.rstrip("\n")]
    lines += [".SH ENVIRONMENT", *environment_roff()]
    lines += [".SH FILES", ".PP", esc("Everything under ~/.hearth moves with HEARTH_HOME. "
                                      "config/ paths are relative to the repo root.")]
    for path, what in FILES:
        lines += [".TP", f"\\fI{esc(path)}\\fR", esc(what)]
    lines += [".SH EXIT STATUS"]
    for code, what in EXIT_STATUS:
        lines += [".TP", f"\\fB{code}\\fR", esc(what)]
    lines += [".PP", esc("Each command's own exit codes are listed under COMMANDS.")]
    lines += [".SH EXAMPLES"]
    for what, cmds in EXAMPLES:
        lines += [".PP", esc(what), *_code_block(cmds)]
    lines += [".SH SEE ALSO", ".PP", esc(SEE_ALSO)]
    text = "\n".join(lines)
    # A paragraph macro straight after a (sub)section heading is redundant (mandoc -T lint).
    tidy: list[str] = []
    for line in text.split("\n"):
        if line == ".PP" and tidy and tidy[-1].startswith((".SH", ".SS")):
            continue
        tidy.append(line)
    return "\n".join(tidy) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=OUT, help="where to write (man/hearth.1)")
    parser.add_argument(
        "--check", action="store_true", help="exit 1 if the file differs from a fresh render"
    )
    args = parser.parse_args(argv)
    text = render()
    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
        if current != text:
            print(f"{args.out} is stale: run scripts/gen_manpage.py", file=sys.stderr)
            return 1
        print(f"{args.out} is up to date")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
