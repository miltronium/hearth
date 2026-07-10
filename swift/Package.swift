// swift-tools-version: 5.9
// HEARTH Swift SDK — a generic async client for the HEARTH HTTP API (Phase 5), plus an
// opt-in, fully-offline Core ML generation path (Phase 6 / ADR-011).
// GENERIC by design: no CAMBOT (or any other consumer) types appear here (ADR-001).
//
// Two products so the dependency cost is opt-in (ADR-011):
//   • "Hearth"        — core: HTTP client, the `HearthInference` protocol, FoundationModels.
//                       ZERO external dependencies, macOS 13. Every HTTP/daemon consumer uses
//                       only this and inherits nothing below.
//   • "HearthCoreML"  — the offline Core ML stateful generation loop. Depends on `Hearth` +
//                       swift-transformers (tokenizer + chat template). The KV-cache path needs
//                       macOS 15 / iOS 18 at runtime; the code is `@available`-gated so the
//                       target still *builds* on older SDKs and the core floor stays at v13.

import PackageDescription

let package = Package(
    name: "Hearth",
    platforms: [
        .macOS(.v13),
        .iOS(.v16)
    ],
    products: [
        .library(name: "Hearth", targets: ["Hearth"]),
        .library(name: "HearthCoreML", targets: ["HearthCoreML"])
    ],
    dependencies: [
        // Tokenizer + chat-template rendering for the offline Core ML path. Declares macOS 13 /
        // iOS 16 itself, so it does not raise the core `Hearth` floor.
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.0.0"),
        // Toolchain workaround: swift-collections 1.6.0 gates `@inline(always)` behind
        // `#if compiler(>=6.3)`, which the current pre-release Swift 6.3 rejects as experimental.
        // Hold it at 1.5.x (a transitive dep of swift-transformers/NIO). Remove once the
        // toolchain stabilizes.
        .package(url: "https://github.com/apple/swift-collections.git", "1.1.0" ..< "1.6.0")
    ],
    targets: [
        // Core: zero external dependencies.
        .target(name: "Hearth"),
        // Opt-in offline Core ML generation. Isolated so `Hearth` consumers don't pull
        // swift-transformers or its transitive deps. Links the `Transformers` product; source
        // imports its `Tokenizers` / `Models` / `Generation` modules.
        .target(
            name: "HearthCoreML",
            dependencies: [
                "Hearth",
                .product(name: "Transformers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers")
            ]
        ),
        .testTarget(name: "HearthTests", dependencies: ["Hearth"]),
        .testTarget(name: "HearthCoreMLTests", dependencies: ["HearthCoreML"])
    ]
)
