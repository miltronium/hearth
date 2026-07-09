# HEARTH Swift SDK

A generic async Swift client for the [HEARTH](../README.md) HTTP API. Any Swift app (macOS
13+) can point it at a local HEARTH gateway and offload routine work — summarizing,
classifying, extracting, drafting, RAG retrieval — to the on-device model, keeping that
work off any frontier budget.

The package is **generic**: it exposes no consumer-specific types (ADR-001). CAMBOT is just
one consumer; the SDK never knows about it.

## Add the dependency

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/miltronix/hearth.git", from: "0.1.0"),
]
// ... and in your target:
.target(name: "MyApp", dependencies: [.product(name: "Hearth", package: "hearth")])
```

Locally (before a tagged release), point at the checkout:

```swift
.package(path: "../hearth/swift")
```

## Usage

```swift
import Hearth

let hearth = HearthClient(baseURL: URL(string: "http://127.0.0.1:8080")!, token: token)

// Summarize a device log locally — 0 frontier tokens.
let summary = try await hearth.summarize(text: deviceLog, maxWords: 120)

// Classify a command intent.
let label = try await hearth.classify(text: userCommand, labels: ["query", "action", "config"])

// Stream a chat completion.
for try await delta in hearth.chatStream(messages: [ChatMessage(role: "user", content: "hi")]) {
    print(delta, terminator: "")
}

// Retrieve grounded chunks from a local RAG collection.
let hits = try await hearth.ragQuery(collection: "mydocs", query: "where is auth handled?")
```

`HearthClient` is a `Sendable` value type over `URLSession` with no third-party
dependencies. Method shapes mirror the Python client (`src/hearth/client.py`) so both SDKs
stay at parity against the same endpoints.

## Embedded / offline mode (no daemon, no network)

For fully offline on-device inference, use `FoundationModelsProvider` — it runs Apple's
built-in on-device model in-process via the FoundationModels framework, with no daemon and no
network. Both it and `HearthClient` conform to one interface, `HearthInference`, so an app can
swap "call the daemon over HTTP" for "run on-device" behind a single type (ADR-009):

```swift
import Hearth

// Pick a transport once; downstream code takes `any HearthInference`.
let engine: any HearthInference = FoundationModelsProvider.isAvailable
    ? try FoundationModelsProvider(instructions: "You are a terse assistant.")
    : HearthClient(baseURL: URL(string: "http://127.0.0.1:8080")!, token: token)

let reply = try await engine.generate(
    messages: [ChatMessage(role: "user", content: "Summarize: …")],
    options: InferenceOptions(temperature: 0.3, maxTokens: 200)
)
for try await delta in engine.generateStream(messages: msgs, options: .default) {
    print(delta, terminator: "")
}
```

**Availability:** the embedded path needs a FoundationModels-capable toolchain and, at
runtime, **macOS 26+ / iOS 26+ / visionOS 26+** with Apple Intelligence enabled and the model
downloaded. The package still **builds and tests on any toolchain** — the implementation is
behind `#if canImport(FoundationModels)` + `@available`, and every unavailable case throws a
clear `HearthError.onDeviceUnavailable(reason)` (probe with `FoundationModelsProvider.isAvailable`).

See **[OFFLINE.md](OFFLINE.md)** for the full requirements, the Open-Question-#3 decision
(embedded mode is base-model-only in v1 — MLX LoRA adapters are not portable to the
FoundationModels path), and the Core ML extension point.

## Build & test

```bash
cd swift
swift build
swift test
```

The tests are offline unit tests (request encoding, URL construction, auth headers, SSE
delta parsing) — they need no running HEARTH server.
