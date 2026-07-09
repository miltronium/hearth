// swift-tools-version: 5.9
// HEARTH Swift SDK — a generic async client for the HEARTH HTTP API (Phase 5).
// GENERIC by design: no CAMBOT (or any other consumer) types appear here (ADR-001).

import PackageDescription

let package = Package(
    name: "Hearth",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .library(name: "Hearth", targets: ["Hearth"])
    ],
    targets: [
        .target(name: "Hearth"),
        .testTarget(name: "HearthTests", dependencies: ["Hearth"])
    ]
)
