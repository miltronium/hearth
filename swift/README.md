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

## Build & test

```bash
cd swift
swift build
swift test
```

The tests are offline unit tests (request encoding, URL construction, auth headers, SSE
delta parsing) — they need no running HEARTH server.
