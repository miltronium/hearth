// HearthInference — the one interface a Swift app codes against, regardless of *where*
// inference happens (Phase 6, ADR-009 deployment models).
//
// Both transports conform to it:
//   • `HearthClient`            — calls a local HEARTH daemon over HTTP.
//   • `FoundationModelsProvider` — runs Apple's on-device model in-process, no daemon,
//                                  no network (see FoundationModelsProvider.swift).
//
// A consumer can therefore swap "talk to the daemon" for "run fully offline on-device"
// behind a single `any HearthInference` without touching call sites. The protocol stays
// GENERIC: no CAMBOT (or any consumer) types leak in (ADR-001).

import Foundation

// MARK: - Inference abstraction

/// A source of local chat-style inference. Method shapes mirror ``HearthClient`` so the two
/// transports are interchangeable.
///
/// ```swift
/// // Pick a transport once; the rest of the app doesn't care which.
/// let engine: any HearthInference = useDaemon
///     ? HearthClient(baseURL: url, token: token)
///     : try FoundationModelsProvider()          // fully offline, no daemon
///
/// let reply = try await engine.generate(messages: [.init(role: "user", content: "hi")])
/// ```
public protocol HearthInference: Sendable {
    /// Run a chat completion and return the assistant's full text.
    func generate(
        messages: [ChatMessage],
        options: InferenceOptions
    ) async throws -> String

    /// Run a chat completion, yielding assistant-text deltas as they are produced.
    func generateStream(
        messages: [ChatMessage],
        options: InferenceOptions
    ) -> AsyncThrowingStream<String, Error>
}

/// Transport-neutral generation knobs. Each conformer maps these onto its own mechanism
/// (HTTP request fields for the daemon; `GenerationOptions` for on-device).
public struct InferenceOptions: Sendable, Equatable {
    /// Sampling temperature, if the backend supports it. `nil` = backend default.
    public var temperature: Double?
    /// Soft cap on generated tokens, if the backend supports it. `nil` = backend default.
    public var maxTokens: Int?

    public init(temperature: Double? = nil, maxTokens: Int? = nil) {
        self.temperature = temperature
        self.maxTokens = maxTokens
    }

    /// Backend defaults.
    public static let `default` = InferenceOptions()
}

// MARK: - HearthClient conformance (HTTP transport)

extension HearthClient: HearthInference {
    /// Non-streaming generation over HTTP. `temperature`/`maxTokens` are advisory: the daemon
    /// applies its own routing/limits. Escalation is disabled so embedded and HTTP callers get
    /// the same "stay local" behavior by default.
    public func generate(
        messages: [ChatMessage],
        options: InferenceOptions = .default
    ) async throws -> String {
        let response = try await chat(
            messages: messages,
            hearth: HearthOptions(allowEscalation: false)
        )
        return response.text
    }

    /// Streaming generation over HTTP — a thin pass-through to ``HearthClient/chatStream(messages:model:hearth:)``.
    public func generateStream(
        messages: [ChatMessage],
        options: InferenceOptions = .default
    ) -> AsyncThrowingStream<String, Error> {
        chatStream(messages: messages, hearth: HearthOptions(allowEscalation: false))
    }
}
