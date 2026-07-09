// Tests for the Phase 6 embedded/offline path: the `HearthInference` abstraction and the
// on-device `FoundationModelsProvider`. These stay offline and need no live model — where the
// real on-device model isn't ready we assert the *unavailable* contract instead.

import XCTest
@testable import Hearth

final class HearthInferenceTests: XCTestCase {

    // MARK: HearthInference abstraction (generic, no CAMBOT types)

    /// A trivial in-memory conformer proves the protocol is usable without either transport,
    /// and that call sites can be written against `any HearthInference`.
    private struct EchoEngine: HearthInference {
        func generate(messages: [ChatMessage], options: InferenceOptions) async throws -> String {
            "echo: " + (messages.last?.content ?? "")
        }
        func generateStream(
            messages: [ChatMessage], options: InferenceOptions
        ) -> AsyncThrowingStream<String, Error> {
            AsyncThrowingStream { continuation in
                continuation.yield("echo: " + (messages.last?.content ?? ""))
                continuation.finish()
            }
        }
    }

    func testInferenceProtocolIsUsableGenerically() async throws {
        let engine: any HearthInference = EchoEngine()
        let out = try await engine.generate(
            messages: [ChatMessage(role: "user", content: "hi")], options: .default
        )
        XCTAssertEqual(out, "echo: hi")
    }

    func testHearthClientConformsToHearthInference() {
        // Compile-time proof the HTTP client is swappable behind the same interface.
        let engine: any HearthInference = HearthClient(baseURL: URL(string: "http://127.0.0.1:8080")!)
        XCTAssertNotNil(engine)
    }

    func testInferenceOptionsDefaults() {
        XCTAssertNil(InferenceOptions.default.temperature)
        XCTAssertNil(InferenceOptions.default.maxTokens)
        XCTAssertEqual(InferenceOptions(temperature: 0.5), InferenceOptions(temperature: 0.5))
    }

    // MARK: FoundationModelsProvider — availability contract

    func testProviderStaticAvailabilityIsConsistent() {
        // isAvailable and unavailableReason must agree, on any platform/build.
        #if canImport(FoundationModels)
        guard #available(macOS 26.0, iOS 26.0, visionOS 26.0, *) else {
            // Framework importable but OS too old: nothing to assert here.
            return
        }
        #endif
        if FoundationModelsProvider.isAvailable {
            XCTAssertNil(FoundationModelsProvider.unavailableReason)
        } else {
            XCTAssertNotNil(FoundationModelsProvider.unavailableReason)
        }
    }

    func testProviderInitFailsWithClearErrorWhenUnavailable() throws {
        // Only assert the *unavailable* path — it's the one guaranteed reproducible offline
        // (CI, no Apple Intelligence, or a build without the framework).
        #if canImport(FoundationModels)
        guard #available(macOS 26.0, iOS 26.0, visionOS 26.0, *) else {
            throw XCTSkip("FoundationModels present but OS too old to instantiate the provider.")
        }
        #endif
        guard !FoundationModelsProvider.isAvailable else {
            throw XCTSkip("On-device model is available on this host; unavailable path not exercised.")
        }
        XCTAssertThrowsError(try FoundationModelsProvider()) { error in
            guard case HearthError.onDeviceUnavailable(let reason) = error else {
                return XCTFail("expected .onDeviceUnavailable, got \(error)")
            }
            XCTAssertFalse(reason.isEmpty)
        }
    }

    // MARK: FoundationModelsProvider — pure helpers (only where the framework is present)

    #if canImport(FoundationModels)
    @available(macOS 26.0, iOS 26.0, visionOS 26.0, *)
    func testPromptFlattensRolesInOrder() {
        let prompt = FoundationModelsProvider.prompt(from: [
            ChatMessage(role: "system", content: "Be terse."),
            ChatMessage(role: "user", content: "Hello"),
            ChatMessage(role: "assistant", content: "Hi"),
            ChatMessage(role: "user", content: "Bye"),
        ])
        XCTAssertEqual(prompt, "System: Be terse.\n\nHello\n\nAssistant: Hi\n\nBye")
    }
    #endif
}

