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

## Core ML — extension point (not built in v1)

A Core ML / ANE-accelerated small-model path would be a **second** `HearthInference`
conformer wrapping an `MLModel`, gated the same `#if canImport` / `@available` way as
`FoundationModelsProvider`. It is intentionally **not** implemented in v1: a real Core ML LLM
path needs a full export pipeline (weight conversion + tokenizer + KV-cache/generation loop),
which is far more than a thin stub and out of scope for Phase 6.

The seam is marked in `Sources/Hearth/FoundationModelsProvider.swift` (the `#else` branch
comment). To add it later: introduce e.g. `CoreMLProvider: HearthInference`, keep it behind
`#if canImport(CoreML)` + an availability gate, throw `HearthError.onDeviceUnavailable` when
the compiled model/tokenizer aren't bundled, and let callers select it exactly like the
FoundationModels provider. No changes to `HearthInference` or existing call sites are needed —
that is the point of the protocol.
