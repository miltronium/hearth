// CoreMLProvider — fully-offline, ANE-accelerated on-device inference via a compiled Core ML
// model (Phase 6, ADR-009 deployment model #3). This is the "Extension point — Core ML" seam
// documented in the `#else` branch of FoundationModelsProvider.swift, now realized.
//
// No daemon, no downloads, no network: a `.mlpackage` / `.mlmodelc` exported by
// `hearth models export-coreml` (see src/hearth/coreml.py) is loaded into an `MLModel` and run
// in-process. This is the second `HearthInference` conformer wrapping an `MLModel`, gated the
// same way as FoundationModelsProvider so `swift build` / `swift test` stay green on any
// toolchain — including ones without Core ML or with no exported model.
//
// GENERIC by design: no CAMBOT (or any consumer) types appear here (ADR-001).
//
// SCOPE (honest current state): construction, availability gating, and loading an `MLModel`
// from a URL are real and fully wired. A complete LLM generation loop (tokenizer + KV-cache +
// sampling) is a large, model-specific pipeline and is intentionally NOT implemented here —
// the exported `.mlpackage` does not yet bundle a tokenizer contract. Until that is wired, the
// generate paths throw `HearthError.onDeviceUnavailable(...)` explaining the state, exactly as
// the extension-point comment describes. Everything that CAN be real (type, gating, init, model
// load, protocol conformance) is real and compiles. See swift/OFFLINE.md.

import Foundation
import Hearth

#if canImport(CoreML)

@preconcurrency import CoreML

/// Sendable wrapper for the loaded `MLModel`. Core ML models are safe to call `prediction`
/// on concurrently, but `MLModel` isn't marked `Sendable`; this box lets ``CoreMLProvider``
/// satisfy the `Sendable` requirement of ``HearthInference`` without leaking the annotation.
private final class LoadedModel: @unchecked Sendable {
    let model: MLModel
    init(_ model: MLModel) { self.model = model }
}

/// On-device, offline implementation of ``HearthInference`` backed by a compiled Core ML model
/// (`.mlpackage` or `.mlmodelc`) exported via `hearth models export-coreml`. Runs on the ANE
/// when the model was exported for `cpuAndNeuralEngine`.
///
/// ```swift
/// let url = URL(fileURLWithPath: "~/.hearth/coreml/my-model.mlpackage")
/// let provider = try CoreMLProvider(modelURL: url)   // throws if the model can't be loaded
/// ```
///
/// Construction fails fast with ``HearthError/onDeviceUnavailable(_:)`` when the compiled model
/// can't be loaded (missing file, incompatible device, unsupported compute units), so callers
/// can fall back to ``HearthClient`` (the HTTP daemon) without inspecting internals.
///
/// - Note: The token-generation loop is not yet wired (no bundled tokenizer contract); the
///   `generate` paths currently throw ``HearthError/onDeviceUnavailable(_:)`` describing that.
///   Model loading, availability, and gating are fully real.
@available(macOS 13.0, iOS 16.0, visionOS 1.0, *)
public struct CoreMLProvider: HearthInference {
    /// The loaded compiled Core ML model, in a `Sendable` box. Held so a future generation
    /// loop can invoke it.
    private let loaded: LoadedModel

    /// The URL the model was loaded from (for diagnostics).
    public let modelURL: URL

    /// Load a compiled Core ML model from `modelURL`, verifying it can be instantiated *now*.
    ///
    /// A `.mlpackage` is compiled on the fly via ``MLModel/compileModel(at:)``; an already
    /// compiled `.mlmodelc` is loaded directly.
    ///
    /// - Parameters:
    ///   - modelURL: A `.mlpackage` or `.mlmodelc` on disk.
    ///   - computeUnits: Runtime placement; defaults to `.cpuAndNeuralEngine` to match the
    ///     offline/ANE export default.
    /// - Throws: ``HearthError/onDeviceUnavailable(_:)`` if the model can't be loaded.
    public init(modelURL: URL, computeUnits: MLComputeUnits = .cpuAndNeuralEngine) throws {
        self.modelURL = modelURL

        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw HearthError.onDeviceUnavailable(
                "no Core ML model at \(modelURL.path) — export one with `hearth models export-coreml`"
            )
        }

        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits

        do {
            let compiledURL = try Self.compiledURL(for: modelURL)
            self.loaded = LoadedModel(try MLModel(contentsOf: compiledURL, configuration: configuration))
        } catch let error as HearthError {
            throw error
        } catch {
            throw HearthError.onDeviceUnavailable(
                "failed to load Core ML model at \(modelURL.path): \(error)"
            )
        }
    }

    /// Whether a Core ML provider *could* be constructed on this build/platform. Core ML is
    /// present (this is the `canImport(CoreML)` branch), but availability of an actual model is
    /// per-URL and checked in ``init(modelURL:computeUnits:)``.
    public static var isAvailable: Bool { true }

    /// A human-readable reason the provider is unavailable, or `nil` when Core ML is present.
    /// Model-file availability is per-URL, so this only reports build-time framework absence.
    public static var unavailableReason: String? { nil }

    // MARK: HearthInference

    public func generate(
        messages: [ChatMessage],
        options: InferenceOptions = .default
    ) async throws -> String {
        throw HearthError.onDeviceUnavailable(Self.noGenerationLoopReason)
    }

    public func generateStream(
        messages: [ChatMessage],
        options: InferenceOptions = .default
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream {
            $0.finish(throwing: HearthError.onDeviceUnavailable(Self.noGenerationLoopReason))
        }
    }

    // MARK: Internals

    static let noGenerationLoopReason =
        "Core ML model loaded, but the token-generation loop (tokenizer + KV-cache + sampling) " +
        "is not yet wired for this exported model; use HearthClient (daemon) or " +
        "FoundationModelsProvider for on-device generation. See swift/OFFLINE.md."

    /// Resolve a compiled-model URL: compile a `.mlpackage`/`.mlmodel` on the fly, or pass a
    /// `.mlmodelc` through unchanged.
    static func compiledURL(for modelURL: URL) throws -> URL {
        if modelURL.pathExtension == "mlmodelc" {
            return modelURL
        }
        do {
            return try MLModel.compileModel(at: modelURL)
        } catch {
            throw HearthError.onDeviceUnavailable(
                "failed to compile Core ML model at \(modelURL.path): \(error)"
            )
        }
    }
}

#else

// Fallback for toolchains/platforms where Core ML can't be imported. Keeps the package building
// everywhere; every entry point reports the same clear error — mirroring the FoundationModels
// stub so callers can fall back to the HTTP daemon uniformly.

/// Stub used when Core ML is unavailable at build time. Any use reports
/// ``HearthError/onDeviceUnavailable(_:)`` so callers can fall back to the HTTP daemon.
public struct CoreMLProvider: HearthInference {
    private static let reason =
        "Core ML is not available in this build (framework not importable on this platform)"

    public let modelURL: URL

    public static var isAvailable: Bool { false }
    public static var unavailableReason: String? { reason }

    public init(modelURL: URL) throws {
        self.modelURL = modelURL
        throw HearthError.onDeviceUnavailable(Self.reason)
    }

    public func generate(
        messages: [ChatMessage],
        options: InferenceOptions = .default
    ) async throws -> String {
        throw HearthError.onDeviceUnavailable(Self.reason)
    }

    public func generateStream(
        messages: [ChatMessage],
        options: InferenceOptions = .default
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { $0.finish(throwing: HearthError.onDeviceUnavailable(Self.reason)) }
    }
}

#endif
