// FoundationModelsProvider — fully-offline, on-device inference via Apple's FoundationModels
// framework (Phase 6, ADR-009 deployment model #3).
//
// No daemon, no downloads, no network: the ~3B on-device model that ships with Apple
// Intelligence is invoked in-process. This is the embedded path CAMBOT (or any Swift app)
// links when it wants inference with networking disabled — see swift/OFFLINE.md.
//
// BUILD SAFETY (a hard requirement): the whole file is guarded by
//   #if canImport(FoundationModels)
// with a compile-time `@available` gate, and a fallback stub for platforms where the
// framework can't be imported at all. So `swift build` / `swift test` stay green on any
// toolchain — including ones with no Apple Intelligence — and unavailability surfaces as a
// clear `HearthError.onDeviceUnavailable(...)` instead of a build break.
//
// GENERIC by design: no CAMBOT (or any consumer) types appear here (ADR-001).

import Foundation

#if canImport(FoundationModels)

import FoundationModels

/// On-device, offline implementation of ``HearthInference`` backed by Apple's built-in
/// system language model. Requires macOS 26+ (and equivalent OS versions on other Apple
/// platforms) with Apple Intelligence enabled and the model downloaded.
///
/// ```swift
/// let provider = try FoundationModelsProvider()          // throws if unavailable
/// let reply = try await provider.generate(
///     messages: [ChatMessage(role: "user", content: "Summarize this log: …")]
/// )
/// ```
///
/// Construction fails fast with ``HearthError/onDeviceUnavailable(_:)`` when the model can't
/// serve requests (Apple Intelligence off, model not yet downloaded, unsupported device),
/// so callers can fall back to ``HearthClient`` (the HTTP daemon) without inspecting internals.
@available(macOS 26.0, iOS 26.0, visionOS 26.0, *)
public struct FoundationModelsProvider: HearthInference {
    /// Optional system prompt applied to every session (maps to FoundationModels instructions).
    public let instructions: String?

    /// Create a provider, verifying the on-device model can serve requests *now*.
    /// - Throws: ``HearthError/onDeviceUnavailable(_:)`` if the model is not available.
    public init(instructions: String? = nil) throws {
        let availability = SystemLanguageModel.default.availability
        guard case .available = availability else {
            throw HearthError.onDeviceUnavailable(Self.describe(availability))
        }
        self.instructions = instructions
    }

    /// Whether the on-device model is currently available, without throwing.
    public static var isAvailable: Bool {
        if case .available = SystemLanguageModel.default.availability { return true }
        return false
    }

    /// A human-readable reason the model is unavailable, or `nil` if it is available.
    public static var unavailableReason: String? {
        let availability = SystemLanguageModel.default.availability
        if case .available = availability { return nil }
        return describe(availability)
    }

    // MARK: HearthInference

    public func generate(
        messages: [ChatMessage],
        options: InferenceOptions = .default
    ) async throws -> String {
        let session = makeSession()
        do {
            let response = try await session.respond(
                to: Self.prompt(from: messages),
                options: Self.generationOptions(options)
            )
            return response.content
        } catch {
            throw HearthError.onDeviceUnavailable("on-device generation failed: \(error)")
        }
    }

    public func generateStream(
        messages: [ChatMessage],
        options: InferenceOptions = .default
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let session = makeSession()
            let task = Task {
                do {
                    // FoundationModels streams *cumulative* snapshots; diff against what we've
                    // already emitted so callers see incremental deltas (matching HearthClient).
                    var emitted = ""
                    let stream = session.streamResponse(
                        to: Self.prompt(from: messages),
                        options: Self.generationOptions(options)
                    )
                    for try await partial in stream {
                        let full = partial.content
                        if full.count > emitted.count, full.hasPrefix(emitted) {
                            continuation.yield(String(full.dropFirst(emitted.count)))
                            emitted = full
                        } else if full != emitted {
                            // Non-monotonic snapshot (rare): re-emit the whole thing.
                            continuation.yield(full)
                            emitted = full
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(
                        throwing: HearthError.onDeviceUnavailable("on-device stream failed: \(error)")
                    )
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    // MARK: Internals

    private func makeSession() -> LanguageModelSession {
        if let instructions {
            return LanguageModelSession(instructions: instructions)
        }
        return LanguageModelSession()
    }

    /// Flatten HEARTH chat messages into a single prompt. A leading `system` message becomes
    /// session instructions only when the provider wasn't already given fixed `instructions`;
    /// otherwise roles are inlined so multi-turn context is preserved in one call.
    static func prompt(from messages: [ChatMessage]) -> String {
        messages
            .map { message in
                switch message.role {
                case "system": return "System: \(message.content)"
                case "assistant": return "Assistant: \(message.content)"
                default: return message.content
                }
            }
            .joined(separator: "\n\n")
    }

    static func generationOptions(_ options: InferenceOptions) -> GenerationOptions {
        switch (options.temperature, options.maxTokens) {
        case let (temperature?, maxTokens?):
            return GenerationOptions(temperature: temperature, maximumResponseTokens: maxTokens)
        case let (temperature?, nil):
            return GenerationOptions(temperature: temperature)
        case let (nil, maxTokens?):
            return GenerationOptions(maximumResponseTokens: maxTokens)
        case (nil, nil):
            return GenerationOptions()
        }
    }

    static func describe(_ availability: SystemLanguageModel.Availability) -> String {
        switch availability {
        case .available:
            return "available"
        case .unavailable(let reason):
            switch reason {
            case .deviceNotEligible:
                return "this device is not eligible for Apple Intelligence"
            case .appleIntelligenceNotEnabled:
                return "Apple Intelligence is not enabled in Settings"
            case .modelNotReady:
                return "the on-device model is downloading or not yet ready"
            @unknown default:
                return "the on-device model is unavailable (\(reason))"
            }
        @unknown default:
            return "the on-device model is unavailable"
        }
    }
}

#else

// Fallback for toolchains/platforms where FoundationModels can't be imported. Keeps the
// package building everywhere; every entry point reports the same clear error.
//
// Extension point — Core ML: a small-model / ANE-accelerated offline path would plug in here
// (a second `HearthInference` conformer wrapping an `MLModel`), gated the same way. Kept out
// of v1 on purpose: a full Core ML export/tokenizer pipeline is more than a thin stub (see
// swift/OFFLINE.md, "Core ML extension point"). This struct is the seam.

/// Stub used when FoundationModels is unavailable at build time. Any use reports
/// ``HearthError/onDeviceUnavailable(_:)`` so callers can fall back to the HTTP daemon.
public struct FoundationModelsProvider: HearthInference {
    private static let reason =
        "FoundationModels is not available in this build (framework not importable on this platform)"

    public static var isAvailable: Bool { false }
    public static var unavailableReason: String? { reason }

    public init(instructions: String? = nil) throws {
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
