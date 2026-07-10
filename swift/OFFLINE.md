# HEARTH Swift SDK — Embedded / Offline mode (Phase 6)

The SDK offers two interchangeable transports behind one interface, `HearthInference`:

| Transport | Type | Network | Daemon | Backend |
|-----------|------|---------|--------|---------|
| HTTP daemon | `HearthClient` | yes (localhost) | yes | HEARTH gateway → routed providers |
| Embedded / offline | `FoundationModelsProvider` | **no** | **no** | Apple's on-device system model, in-process |

Both conform to `HearthInference` (`generate(messages:options:)` and
`generateStream(messages:options:)`), so an app picks a transport once and the rest of the
code is transport-agnostic (ADR-009 deployment model #3; ADR-001 client-agnostic).

```swift
import Hearth

// Choose a transport once — everything downstream takes `any HearthInference`.
let engine: any HearthInference
if FoundationModelsProvider.isAvailable {
    engine = try FoundationModelsProvider(instructions: "You are a terse assistant.")
} else {
    engine = HearthClient(baseURL: URL(string: "http://127.0.0.1:8080")!, token: token)
}

let reply = try await engine.generate(
    messages: [ChatMessage(role: "user", content: "Summarize: …")],
    options: InferenceOptions(temperature: 0.3, maxTokens: 200)
)

for try await delta in engine.generateStream(messages: msgs, options: .default) {
    print(delta, terminator: "")   // incremental text deltas, like the HTTP client
}
```

## Requirements & availability

`FoundationModelsProvider` runs Apple's built-in on-device model (Apple Intelligence). It is
only usable when **all** of these hold:

- Built with an SDK where `FoundationModels` is importable (Xcode 26 / Swift 6 era toolchain).
- **macOS 26+ / iOS 26+ / visionOS 26+** at runtime (`@available` gated).
- Apple Intelligence enabled on an eligible device, and the model downloaded.

The package is designed to **build and test on any toolchain**, including ones with no
FoundationModels and no Apple Intelligence:

- The whole real implementation is behind `#if canImport(FoundationModels)` + an
  `@available(macOS 26.0, iOS 26.0, visionOS 26.0, *)` gate.
- On toolchains without the framework, a fallback stub compiles instead.
- In every unavailable case — framework absent, OS too old, Apple Intelligence off, model not
  ready — construction and calls throw a clear `HearthError.onDeviceUnavailable(reason)`.
- Probe without throwing via `FoundationModelsProvider.isAvailable` /
  `.unavailableReason`, then fall back to `HearthClient` (as in the snippet above).

## Open Question #3 — are fine-tuned MLX adapters portable to embedded mode?

**Decision: embedded mode is base-model-only in v1.**

**Rationale.** HEARTH's fine-tuning path (elsewhere in the project) produces **MLX LoRA
adapters**. The embedded path here is Apple's `FoundationModels` framework, which exposes the
**system** language model through `LanguageModelSession` — it does **not** accept an
externally-trained MLX LoRA adapter, and provides no API to load arbitrary third-party
weights. FoundationModels' own adapter mechanism (rank-tuned adapters trained with Apple's
toolchain) is a different artifact than an MLX LoRA and is not interchangeable with it. There
is therefore no supported, offline way to run a HEARTH MLX adapter inside
`FoundationModelsProvider` in v1.

**What this means for callers.**

- Need a fine-tuned adapter → use the **daemon / HTTP** transport (`HearthClient`), where the
  MLX runtime serves the adapter. This is the intended home for adapter-specialized behavior.
- Need fully offline, no-daemon inference → use `FoundationModelsProvider` with the **base**
  on-device model, and steer it with `instructions` (system prompt) + few-shot context in the
  messages, rather than weights.

**Revisit if** Apple ships a public API to load custom adapters into the on-device model, or a
Core ML export path (below) makes a HEARTH-fine-tuned small model runnable on-device — at
which point adapter portability to *some* embedded backend becomes feasible.

## Core ML — extension point (now implemented, generation loop pending)

The Core ML / ANE-accelerated small-model path is a **second** `HearthInference` conformer,
`CoreMLProvider`, wrapping an `MLModel`. It is gated the same `#if canImport(CoreML)` /
`@available` way as `FoundationModelsProvider`, so the package still builds and tests on any
toolchain (a fallback stub compiles where Core ML can't be imported, and every entry point
throws `HearthError.onDeviceUnavailable`).

### Export a model

Produce a `.mlpackage` from a base checkpoint with the Python side (needs the `[coreml]`
extra — `coremltools` / `torch` / `transformers`):

```sh
uv sync --extra coreml
HF_HUB_OFFLINE=1 hearth models export-coreml \
    --source <hf-repo-or-path> \
    --out ~/.hearth/coreml/my-model.mlpackage \
    --compute-units cpuAndNeuralEngine \
    --precision float16 \
    --max-seq-len 512
```

Like the MLX quantization pipeline, the export is delegated to an injectable runner
(`hearth.coreml.export`) and the heavy `coremltools` work lives in a lazily-imported default
runner, so the module imports with no extras installed.

### Construct the provider

`CoreMLProvider` lives in the **opt-in `HearthCoreML` product** (ADR-011), so daemon-only
consumers of `Hearth` never pull swift-transformers or its transitive deps:

```swift
import Hearth        // the `HearthInference` protocol + core types
import HearthCoreML  // the offline Core ML provider

let url = URL(fileURLWithPath: "…/my-model.mlpackage")   // or a compiled .mlmodelc
if CoreMLProvider.isAvailable {
    let engine: any HearthInference =
        try CoreMLProvider(modelURL: url, computeUnits: .cpuAndNeuralEngine)
    // engine slots in behind `any HearthInference` exactly like the other transports.
}
```

Construction compiles the `.mlpackage` (or loads a `.mlmodelc`) into an `MLModel` and throws
`HearthError.onDeviceUnavailable(reason)` on a missing file, compile failure, or incompatible
device.

### Current state (honest)

**What is real and shipping:** the `CoreMLProvider` type, the `#if canImport(CoreML)` gating +
fallback stub, availability probes (`isAvailable` / `unavailableReason`), init-from-URL, model
compilation/loading, and full `HearthInference` conformance. All of this compiles and is tested
on any toolchain.

**What is not yet wired:** the actual token-generation loop (tokenizer + KV-cache + sampling)
is a large, model-specific pipeline and the exported `.mlpackage` does not yet bundle a
tokenizer contract. Until that lands, `generate` / `generateStream` throw
`HearthError.onDeviceUnavailable(...)` explaining the state and pointing callers at
`HearthClient` (daemon) or `FoundationModelsProvider` for on-device generation. The seam,
export path, and model loading are the real deliverable here; the generation loop is the
remaining follow-up.

