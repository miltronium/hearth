"""Core ML / ANE export pipeline (Phase 6 extension point, ARCHITECTURE §5; ADR-011).

Converts a base model into a **stateful** ``.mlpackage`` that the Swift ``CoreMLProvider`` (see
``swift/Sources/Hearth/CoreMLProvider.swift``) can load for fully-offline, ANE-accelerated
on-device generation. Like the quantization pipeline (:mod:`hearth.convert`) and the LoRA
orchestrator (:mod:`hearth.training.lora`), the heavy work is delegated to an **injectable
runner** — tests pass a fake and never launch a real export (model download is proxy-blocked and
Core ML conversion is slow). The default runner traces a Hugging Face model with a KV-cache and
converts it via ``coremltools`` behind the ``[coreml]`` extra.

The Swift generation loop can't run a bare graph — it needs a **contract** telling it how to
tokenize, frame the chat, and stop. So every export also writes a *sidecar* next to the
``.mlpackage`` (see :func:`sidecar_paths`):

* ``<stem>.hearth-coreml.json`` — a :class:`CoreMLManifest`: tensor names, ``max_seq_len``,
  ``vocab_size``, ``bos``/``eos`` token ids (the Finding-2b terminator set), and which tokenizer
  files to read.
* ``<stem>.tokenizer.json`` / ``<stem>.tokenizer_config.json`` — the tokenizer, so
  ``swift-transformers`` can tokenize and apply the model's own chat template (ADR-011).

The manifest + sidecar wiring is pure and fully offline-tested; the real stateful conversion is
the hardware-validated piece (``docs/HANDOFF.md`` → Task C).

Real path (needs the ``[coreml]`` extra, source weights, and offline HF for cached inputs):

    uv sync --extra coreml
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    hearth models export-coreml --source <hf-repo-or-path> --out ~/.hearth/coreml/<id>.mlpackage

``coremltools`` / ``torch`` / ``transformers`` are imported only inside the default runner, so
importing this module (and the whole test suite) needs no extras.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

# Bump when the manifest shape changes so the Swift loader can reject an incompatible sidecar.
MANIFEST_SCHEMA_VERSION = 1

# MLModel compute-unit placements exposed by coremltools. "cpuAndNeuralEngine" pins work to the
# ANE (+CPU fallback) for the offline/low-power path; "all" also allows the GPU. Names match the
# `MLComputeUnits` cases the Swift `CoreMLProvider` passes back through `MLModelConfiguration`.
_VALID_COMPUTE_UNITS = ("all", "cpuAndNeuralEngine", "cpuAndGPU", "cpuOnly")

# Weight precisions coremltools can emit. float16 is the on-device workhorse (half the size,
# ANE-native); int8 trades quality for size; float32 is debug-only.
_VALID_PRECISIONS = ("float16", "float32", "int8")

# Chat-template markers a model can carry. "chatml" is Qwen/ChatML (the model validated in
# RESULTS); Swift prefers the tokenizer's own `chat_template`, falling back to this id.
_DEFAULT_CHAT_TEMPLATE = "chatml"


class CoreMLExportUnavailableError(RuntimeError):
    """Raised when a real export is requested but ``coremltools`` isn't importable."""


@dataclass(frozen=True)
class CoreMLManifest:
    """The contract the Swift generation loop reads to drive a loaded Core ML model (ADR-011).

    Written as ``<stem>.hearth-coreml.json`` next to the ``.mlpackage``. Holds everything the
    graph itself doesn't encode: which tensors to feed/read, how long a window it accepts, the
    vocabulary size, the tokens that start/stop generation, and which tokenizer files ship
    alongside. ``eos_token_ids`` is a *list* on purpose — Finding-2b showed Qwen terminates on
    ``<|im_end|>`` (151645) as well as ``<|endoftext|>`` (151643), and the loop must stop on any.
    """

    source: str
    max_seq_len: int
    vocab_size: int
    eos_token_ids: list[int]
    bos_token_id: int | None = None
    stateful: bool = True
    # Feature names follow swift-transformers' stateful Core ML LLM contract (Apple convention):
    # input `inputIds` (+ optional `causalMask`), states `keyCache`/`valueCache`, output `logits`.
    input_name: str = "inputIds"
    output_name: str = "logits"
    chat_template_id: str = _DEFAULT_CHAT_TEMPLATE
    tokenizer_files: list[str] = field(default_factory=list)
    compute_units: str = "cpuAndNeuralEngine"
    precision: str = "float16"
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def to_dict(self) -> dict:
        """JSON-serializable form (stable key order for reproducible sidecars)."""
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "stateful": self.stateful,
            "input_name": self.input_name,
            "output_name": self.output_name,
            "max_seq_len": self.max_seq_len,
            "vocab_size": self.vocab_size,
            "bos_token_id": self.bos_token_id,
            "eos_token_ids": list(self.eos_token_ids),
            "chat_template_id": self.chat_template_id,
            "tokenizer_files": list(self.tokenizer_files),
            "compute_units": self.compute_units,
            "precision": self.precision,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CoreMLManifest:
        """Parse a manifest dict, rejecting an unknown/newer schema version."""
        version = data.get("schema_version")
        if version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported Core ML manifest schema_version {version!r} "
                f"(this build understands {MANIFEST_SCHEMA_VERSION})"
            )
        if not data.get("eos_token_ids"):
            raise ValueError("manifest must list at least one eos_token_id")
        return cls(
            source=data["source"],
            max_seq_len=data["max_seq_len"],
            vocab_size=data["vocab_size"],
            eos_token_ids=list(data["eos_token_ids"]),
            bos_token_id=data.get("bos_token_id"),
            stateful=data.get("stateful", True),
            input_name=data.get("input_name", "inputIds"),
            output_name=data.get("output_name", "logits"),
            chat_template_id=data.get("chat_template_id", _DEFAULT_CHAT_TEMPLATE),
            tokenizer_files=list(data.get("tokenizer_files", [])),
            compute_units=data.get("compute_units", "cpuAndNeuralEngine"),
            precision=data.get("precision", "float16"),
        )


@dataclass(frozen=True)
class CoreMLRunResult:
    """What a runner returns: the exported ``.mlpackage`` plus the metadata needed for a sidecar.

    ``manifest`` carries the model-specific facts only the runner can know (vocab size, eos ids);
    ``tokenizer_dir`` is a directory holding ``tokenizer.json`` (and optionally
    ``tokenizer_config.json``) to copy alongside — ``None`` when a fake runner has no tokenizer.
    :func:`export` writes the sidecar centrally from this, so sidecar emission is offline-testable.
    """

    output_dir: Path
    manifest: CoreMLManifest
    tokenizer_dir: Path | None = None


# A runner performs the export for a resolved config and returns a :class:`CoreMLRunResult`.
# Injectable so tests fake it (never a real export). The default uses coremltools.
Runner = Callable[["CoreMLExportConfig"], CoreMLRunResult]


@dataclass(frozen=True)
class CoreMLExportConfig:
    """Inputs for one Core ML export run.

    ``source`` is an HF repo id or a local path to the source checkpoint; ``output_dir`` is
    where the ``.mlpackage`` is written. ``compute_units`` picks the runtime placement
    (default the ANE), ``precision`` the emitted weight dtype, and ``max_seq_len`` the fixed
    context window the stateful model is exported at (Core ML shapes are static).
    """

    source: str
    output_dir: Path
    compute_units: str = "cpuAndNeuralEngine"
    precision: str = "float16"
    max_seq_len: int = 512

    def validate(self) -> None:
        """Raise :class:`ValueError` unless the config is exportable."""
        if not self.source:
            raise ValueError("source is required")
        if not self.output_dir:
            raise ValueError("output_dir is required")
        if self.compute_units not in _VALID_COMPUTE_UNITS:
            raise ValueError(
                f"compute_units must be one of {_VALID_COMPUTE_UNITS}, got {self.compute_units!r}"
            )
        if self.precision not in _VALID_PRECISIONS:
            raise ValueError(
                f"precision must be one of {_VALID_PRECISIONS}, got {self.precision!r}"
            )
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")


@dataclass(frozen=True)
class CoreMLExportOutcome:
    """Result of an export — the ``.mlpackage`` path, the sidecar paths, and the settings used."""

    source: str
    output_dir: Path
    compute_units: str
    precision: str
    max_seq_len: int
    manifest_path: Path
    tokenizer_paths: list[Path]


def sidecar_paths(mlpackage: Path) -> dict[str, Path]:
    """Where the sidecar files live for a given ``.mlpackage``.

    They sit **next to** the package, prefixed with its stem, so several models can share a
    directory without colliding and the Swift side can derive every path from the model URL:

        ~/.hearth/coreml/qwen-coder.mlpackage
        ~/.hearth/coreml/qwen-coder.hearth-coreml.json
        ~/.hearth/coreml/qwen-coder.tokenizer.json
        ~/.hearth/coreml/qwen-coder.tokenizer_config.json
    """
    base = mlpackage.parent / mlpackage.stem
    return {
        "manifest": base.with_name(base.name + ".hearth-coreml.json"),
        "tokenizer": base.with_name(base.name + ".tokenizer.json"),
        "tokenizer_config": base.with_name(base.name + ".tokenizer_config.json"),
    }


def write_sidecar(
    mlpackage: Path, manifest: CoreMLManifest, *, tokenizer_dir: Path | None = None
) -> tuple[Path, list[Path]]:
    """Write the manifest (and copy tokenizer files) next to ``mlpackage``.

    Returns ``(manifest_path, tokenizer_paths)``. Pure filesystem work — no model, no extras —
    so it is fully offline-testable. Copies ``tokenizer.json`` (required when ``tokenizer_dir``
    is given) and ``tokenizer_config.json`` (optional; carries the chat template) from
    ``tokenizer_dir`` to the stem-prefixed sidecar names, and records their basenames on the
    written manifest so the loader knows what shipped.
    """
    paths = sidecar_paths(mlpackage)
    tokenizer_paths: list[Path] = []
    written_names: list[str] = []

    if tokenizer_dir is not None:
        src_tok = Path(tokenizer_dir) / "tokenizer.json"
        if not src_tok.exists():
            raise FileNotFoundError(
                f"tokenizer.json not found in {tokenizer_dir} — needed for the Core ML sidecar"
            )
        paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_tok, paths["tokenizer"])
        tokenizer_paths.append(paths["tokenizer"])
        written_names.append(paths["tokenizer"].name)

        src_cfg = Path(tokenizer_dir) / "tokenizer_config.json"
        if src_cfg.exists():
            shutil.copyfile(src_cfg, paths["tokenizer_config"])
            tokenizer_paths.append(paths["tokenizer_config"])
            written_names.append(paths["tokenizer_config"].name)

    # Record which tokenizer files actually shipped (a fake/tokenizer-less runner writes none).
    manifest_out = CoreMLManifest(
        source=manifest.source,
        max_seq_len=manifest.max_seq_len,
        vocab_size=manifest.vocab_size,
        eos_token_ids=manifest.eos_token_ids,
        bos_token_id=manifest.bos_token_id,
        stateful=manifest.stateful,
        input_name=manifest.input_name,
        output_name=manifest.output_name,
        chat_template_id=manifest.chat_template_id,
        tokenizer_files=written_names,
        compute_units=manifest.compute_units,
        precision=manifest.precision,
    )
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    paths["manifest"].write_text(json.dumps(manifest_out.to_dict(), indent=2) + "\n")
    return paths["manifest"], tokenizer_paths


def export(
    config: CoreMLExportConfig, *, runner: Runner | None = None
) -> CoreMLExportOutcome:
    """Orchestrate a Core ML export: validate, delegate to ``runner``, write the sidecar, report.

    ``runner`` defaults to :func:`_coreml_export_runner` (real stateful conversion via
    ``coremltools``); tests inject a fake that returns a :class:`CoreMLRunResult` with a stub
    ``.mlpackage`` and manifest. Sidecar writing happens here (not in the runner) so it is
    exercised on the fake path without any extras installed.
    """
    config.validate()
    run = runner or _coreml_export_runner
    result = run(config)
    manifest_path, tokenizer_paths = write_sidecar(
        result.output_dir, result.manifest, tokenizer_dir=result.tokenizer_dir
    )
    return CoreMLExportOutcome(
        source=config.source,
        output_dir=Path(result.output_dir),
        compute_units=config.compute_units,
        precision=config.precision,
        max_seq_len=config.max_seq_len,
        manifest_path=manifest_path,
        tokenizer_paths=tokenizer_paths,
    )


def _terminator_ids(tokenizer, model_config) -> list[int]:
    """The Finding-2b stop set: every id that ends a turn for this model.

    mlx-lm only stopped on ``<|endoftext|>`` and a tuned Qwen adapter rambled on ``<|im_end|>``;
    the offline loop must stop on *any* terminator. Collect the config eos (int or list), the
    tokenizer eos, and the ChatML end marker, de-duped and minus the unk id.
    """
    ids: list[int] = []
    cfg_eos = getattr(model_config, "eos_token_id", None)
    if isinstance(cfg_eos, (list, tuple)):
        ids.extend(int(x) for x in cfg_eos)
    elif cfg_eos is not None:
        ids.append(int(cfg_eos))
    if getattr(tokenizer, "eos_token_id", None) is not None:
        ids.append(int(tokenizer.eos_token_id))
    for marker in ("<|im_end|>", "<|endoftext|>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(marker)
        except Exception:
            tid = None
        unk = getattr(tokenizer, "unk_token_id", None)
        if tid is not None and tid != unk and tid >= 0:
            ids.append(int(tid))
    # De-dupe, preserve first-seen order.
    seen: set[int] = set()
    return [i for i in ids if not (i in seen or seen.add(i))]


def _coreml_export_runner(config: CoreMLExportConfig) -> CoreMLRunResult:
    """Default runner: convert an HF model to a **stateful** ``.mlpackage`` (needs ``[coreml]``).

    Kept out of the tested path — tests always inject a fake runner. Raising with the fix hint
    mirrors :class:`hearth.convert.ConvertUnavailableError`. The stateful KV-cache export
    (single-token input + per-layer cache read/write via coremltools' ``States``, ADR-011) is the
    model-specific piece validated on real hardware in ``docs/HANDOFF.md`` → Task C; here we set
    up the graph, gather the tokenizer/manifest contract, and hand both back to :func:`export`.
    """
    import importlib.util
    import tempfile

    if importlib.util.find_spec("coremltools") is None:
        raise CoreMLExportUnavailableError(
            "coremltools is not installed. Install the Core ML export backend with: "
            "uv sync --extra coreml"
        )

    # Deferred heavy imports — only reached on the real path, never in tests.
    import coremltools as ct
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    precision = {
        "float16": ct.precision.FLOAT16,
        "float32": ct.precision.FLOAT32,
        "int8": ct.precision.FLOAT16,  # int8 handled via post-conversion palettization below
    }[config.precision]
    compute_units = {
        "all": ct.ComputeUnit.ALL,
        "cpuAndNeuralEngine": ct.ComputeUnit.CPU_AND_NE,
        "cpuAndGPU": ct.ComputeUnit.CPU_AND_GPU,
        "cpuOnly": ct.ComputeUnit.CPU_ONLY,
    }[config.compute_units]

    hf_config = AutoConfig.from_pretrained(config.source)
    tokenizer = AutoTokenizer.from_pretrained(config.source)
    model = AutoModelForCausalLM.from_pretrained(config.source, torchscript=True)
    model.eval()

    # --- Stateful KV-cache export to swift-transformers' contract (ADR-011) -----------------
    # `LanguageModelWithStatefulKVCache` (swift-transformers) drives a Core ML model shaped:
    #   inputs : `inputIds`  [1, RangeDim(1..max)]   (+ `causalMask` [1,1,q,max] when required)
    #   states : `keyCache`, `valueCache`  — combined-across-layers tensors, updated in place
    #   output : `logits`    [1, seq, vocab]
    # The ranged `inputIds` lets one graph both prefill the prompt and extend one token at a time
    # (the O(1)-per-token win). We back the cache with fixed buffers slice-updated in place
    # (Apple's stateful-LLM recipe) so coremltools captures them as `ct.StateType`s. The concrete
    # StaticCache/attention wiring is architecture-specific (Qwen2/Llama-family) and is finalized
    # + validated on real weights in docs/HANDOFF.md → Task C.
    from transformers.cache_utils import StaticCache

    num_layers = hf_config.num_hidden_layers
    num_kv_heads = getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads)
    head_dim = hf_config.hidden_size // hf_config.num_attention_heads
    # One tensor per cache, stacked over layers, matching swift-transformers' two states.
    cache_shape = (num_layers, 1, num_kv_heads, config.max_seq_len, head_dim)

    class _StatefulCausalLM(torch.nn.Module):  # pragma: no cover - hardware path (Task C)
        """HF model wrapped with fixed `keyCache`/`valueCache` buffers exposed as Core ML states.

        ``forward(inputIds, causalMask)`` seeds a StaticCache from the buffers, runs one
        prefill-or-extend step, writes the updated cache back in place, and returns logits.
        """

        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.register_buffer("keyCache", torch.zeros(cache_shape, dtype=torch.float16))
            self.register_buffer("valueCache", torch.zeros(cache_shape, dtype=torch.float16))

        def forward(self, inputIds, causalMask):
            cache = StaticCache(
                config=hf_config,
                max_batch_size=1,
                max_cache_len=config.max_seq_len,
                device="cpu",
                dtype=torch.float16,
            )
            for layer in range(num_layers):
                cache.key_cache[layer] = self.keyCache[layer]
                cache.value_cache[layer] = self.valueCache[layer]
            out = self.inner(
                input_ids=inputIds, attention_mask=causalMask, past_key_values=cache, use_cache=True
            )
            for layer in range(num_layers):
                self.keyCache[layer].copy_(cache.key_cache[layer])
                self.valueCache[layer].copy_(cache.value_cache[layer])
            return out.logits

    wrapper = _StatefulCausalLM(model).eval()
    seq = min(8, config.max_seq_len)
    example_ids = torch.zeros((1, seq), dtype=torch.int32)
    example_mask = torch.zeros((1, 1, seq, config.max_seq_len), dtype=torch.float16)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (example_ids, example_mask))

    query = ct.RangeDim(lower_bound=1, upper_bound=config.max_seq_len, default=seq)
    def _state(name):
        return ct.StateType(
            wrapped_type=ct.TensorType(shape=cache_shape, dtype=np.float16), name=name
        )

    states = [_state("keyCache"), _state("valueCache")]
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="inputIds", shape=(1, query), dtype=np.int32),
            ct.TensorType(
                name="causalMask",
                shape=(1, 1, query, config.max_seq_len),
                dtype=np.float16,
            ),
        ],
        outputs=[ct.TensorType(name="logits", dtype=np.float16)],
        states=states,
        compute_precision=precision,
        compute_units=compute_units,
        minimum_deployment_target=ct.target.macOS15,
    )
    if config.precision == "int8":
        mlmodel = ct.optimize.coreml.palettize_weights(
            mlmodel, ct.optimize.coreml.OpPalettizerConfig(nbits=8)
        )

    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(config.output_dir))

    # Stage the tokenizer so `export()` can copy it into the sidecar.
    tok_dir = Path(tempfile.mkdtemp(prefix="hearth-coreml-tok-"))
    tokenizer.save_pretrained(str(tok_dir))

    manifest = CoreMLManifest(
        source=config.source,
        max_seq_len=config.max_seq_len,
        vocab_size=int(getattr(hf_config, "vocab_size", len(tokenizer))),
        eos_token_ids=_terminator_ids(tokenizer, hf_config),
        bos_token_id=getattr(tokenizer, "bos_token_id", None),
        stateful=True,
        input_name="inputIds",
        output_name="logits",
        compute_units=config.compute_units,
        precision=config.precision,
    )
    return CoreMLRunResult(
        output_dir=config.output_dir, manifest=manifest, tokenizer_dir=tok_dir
    )


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "CoreMLExportConfig",
    "CoreMLExportOutcome",
    "CoreMLExportUnavailableError",
    "CoreMLManifest",
    "CoreMLRunResult",
    "Runner",
    "export",
    "sidecar_paths",
    "write_sidecar",
]
