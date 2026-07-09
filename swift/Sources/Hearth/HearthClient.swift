// HearthClient — a generic async Swift client for the HEARTH HTTP API (Phase 5, ADR-001).
//
// Mirrors the Python `HearthClient` (src/hearth/client.py) against the same OpenAI-compatible
// endpoints, so both SDKs stay at parity. Uses only Foundation's URLSession — no third-party
// dependencies, no CAMBOT (or any consumer) types. Any Swift app can depend on this.

import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

// MARK: - Errors

/// Errors surfaced by ``HearthClient``.
public enum HearthError: Error, Sendable {
    /// The server returned a non-2xx status. Carries the HTTP status and any response body.
    case httpStatus(code: Int, body: String)
    /// The response body could not be decoded into the expected shape.
    case decoding(String)
    /// A streaming response emitted a malformed event.
    case malformedStream(String)
}

// MARK: - Wire types (mirror src/hearth/gateway/schemas.py)

/// One chat message. `role` is "system" | "user" | "assistant".
public struct ChatMessage: Codable, Sendable, Equatable {
    public var role: String
    public var content: String

    public init(role: String, content: String) {
        self.role = role
        self.content = content
    }
}

/// Optional HEARTH-specific request hints (ignored by pure-OpenAI servers).
public struct HearthOptions: Codable, Sendable, Equatable {
    public var intent: String?
    public var allowEscalation: Bool

    public init(intent: String? = nil, allowEscalation: Bool = true) {
        self.intent = intent
        self.allowEscalation = allowEscalation
    }

    enum CodingKeys: String, CodingKey {
        case intent
        case allowEscalation = "allow_escalation"
    }
}

/// A `/v1/chat/completions` request body.
public struct ChatRequest: Codable, Sendable, Equatable {
    public var model: String
    public var messages: [ChatMessage]
    public var stream: Bool
    public var hearth: HearthOptions?

    public init(
        model: String = "auto",
        messages: [ChatMessage],
        stream: Bool = false,
        hearth: HearthOptions? = nil
    ) {
        self.model = model
        self.messages = messages
        self.stream = stream
        self.hearth = hearth
    }
}

/// The (non-streaming) chat completion response — just the fields callers usually read.
public struct ChatResponse: Codable, Sendable {
    public struct Choice: Codable, Sendable {
        public struct Message: Codable, Sendable {
            public var role: String
            public var content: String
        }
        public var message: Message
    }
    public var id: String
    public var model: String
    public var choices: [Choice]

    /// The assistant text of the first choice (the common case).
    public var text: String { choices.first?.message.content ?? "" }
}

/// A `/v1/embeddings` response.
public struct EmbeddingResponse: Codable, Sendable {
    public struct Item: Codable, Sendable {
        public var embedding: [Double]
        public var index: Int
    }
    public var data: [Item]
}

/// A `/v1/hearth/rag/query` response.
public struct RagQueryResponse: Codable, Sendable {
    public struct Chunk: Codable, Sendable {
        public var text: String
        public var source: String
        public var score: Double
    }
    public var chunks: [Chunk]
    public var answer: String?
}

// MARK: - Client

/// Async client for a local HEARTH gateway.
///
/// ```swift
/// let hearth = HearthClient(baseURL: URL(string: "http://127.0.0.1:8080")!, token: token)
/// let summary = try await hearth.summarize(text: log, maxWords: 120)  // local, 0 frontier tokens
/// ```
public struct HearthClient: Sendable {
    /// Root of the HEARTH server, e.g. `http://127.0.0.1:8080` (a trailing `/v1` is trimmed).
    public let baseURL: URL
    /// Optional bearer token; sent as `Authorization: Bearer <token>` when present.
    public let token: String?
    private let session: URLSession

    public init(baseURL: URL, token: String? = nil, session: URLSession = .shared) {
        // Normalize away a trailing "/v1" so endpoint paths ("/v1/...") don't double up.
        if baseURL.lastPathComponent == "v1" {
            // deletingLastPathComponent leaves a trailing slash; re-parse to drop it.
            let trimmed = baseURL.deletingLastPathComponent().absoluteString
            self.baseURL = URL(string: trimmed.hasSuffix("/") ? String(trimmed.dropLast()) : trimmed) ?? baseURL
        } else {
            self.baseURL = baseURL
        }
        self.token = token
        self.session = session
    }

    // MARK: Request building

    /// Build a POST request to `path` with a JSON `body`. Exposed for parity/URL tests.
    public func makeRequest(path: String, body: Encodable) throws -> URLRequest {
        let url = baseURL.appendingPathComponent(path.hasPrefix("/") ? String(path.dropFirst()) : path)
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let token {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        request.httpBody = try Self.encoder.encode(AnyEncodable(body))
        return request
    }

    // MARK: Chat

    /// Call `/v1/chat/completions` (non-streaming) and return the decoded response.
    public func chat(
        messages: [ChatMessage],
        model: String = "auto",
        hearth: HearthOptions? = nil
    ) async throws -> ChatResponse {
        let request = try makeRequest(
            path: "v1/chat/completions",
            body: ChatRequest(model: model, messages: messages, stream: false, hearth: hearth)
        )
        return try await send(request)
    }

    /// Call `/v1/chat/completions` with streaming; yields assistant text deltas as they arrive.
    public func chatStream(
        messages: [ChatMessage],
        model: String = "auto",
        hearth: HearthOptions? = nil
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let request = try makeRequest(
                        path: "v1/chat/completions",
                        body: ChatRequest(
                            model: model, messages: messages, stream: true, hearth: hearth
                        )
                    )
                    let (bytes, response) = try await session.bytes(for: request)
                    try Self.checkStatus(response, body: "<stream>")
                    for try await line in bytes.lines {
                        guard line.hasPrefix("data: ") else { continue }
                        let payload = String(line.dropFirst("data: ".count))
                        if payload == "[DONE]" { break }
                        if let delta = Self.deltaContent(from: payload) {
                            continuation.yield(delta)
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    // MARK: Convenience helpers

    /// Summarize `text` on the local model (intent=summarize, escalation disabled).
    public func summarize(text: String, maxWords: Int? = nil) async throws -> String {
        let limit = maxWords.map { " in at most \($0) words" } ?? ""
        return try await oneShot(
            "Summarize the following text\(limit):\n\n\(text)", intent: "summarize"
        )
    }

    /// Classify `text` into one of `labels` on the local model (intent=classify).
    public func classify(text: String, labels: [String]) async throws -> String {
        let options = labels.joined(separator: ", ")
        let prompt = """
        Classify the following text into exactly one of these labels: \(options).
        Reply with only the label.

        Text:
        \(text)
        """
        return try await oneShot(prompt, intent: "classify")
    }

    private func oneShot(_ prompt: String, intent: String) async throws -> String {
        let response = try await chat(
            messages: [ChatMessage(role: "user", content: prompt)],
            hearth: HearthOptions(intent: intent, allowEscalation: false)
        )
        return response.text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Call `/v1/embeddings` and return vectors in input order.
    public func embed(texts: [String], model: String = "auto") async throws -> [[Double]] {
        struct Body: Encodable { let model: String; let input: [String] }
        let request = try makeRequest(
            path: "v1/embeddings", body: Body(model: model, input: texts)
        )
        let response: EmbeddingResponse = try await send(request)
        return response.data.sorted { $0.index < $1.index }.map(\.embedding)
    }

    /// Call `/v1/hearth/rag/query` and return the retrieved chunks (+ optional answer).
    public func ragQuery(
        collection: String, query: String, k: Int = 6, answer: Bool = false
    ) async throws -> RagQueryResponse {
        struct Body: Encodable {
            let collection: String; let query: String; let k: Int; let answer: Bool
        }
        let request = try makeRequest(
            path: "v1/hearth/rag/query",
            body: Body(collection: collection, query: query, k: k, answer: answer)
        )
        return try await send(request)
    }

    // MARK: Transport internals

    private func send<T: Decodable>(_ request: URLRequest) async throws -> T {
        let (data, response) = try await session.data(for: request)
        try Self.checkStatus(response, body: String(data: data, encoding: .utf8) ?? "")
        do {
            return try Self.decoder.decode(T.self, from: data)
        } catch {
            throw HearthError.decoding("\(error)")
        }
    }

    static func checkStatus(_ response: URLResponse, body: String) throws {
        guard let http = response as? HTTPURLResponse else { return }
        guard (200..<300).contains(http.statusCode) else {
            throw HearthError.httpStatus(code: http.statusCode, body: body)
        }
    }

    /// Pull `choices[0].delta.content` out of one SSE chunk payload, if present.
    static func deltaContent(from payload: String) -> String? {
        guard let data = payload.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let choices = object["choices"] as? [[String: Any]],
              let delta = choices.first?["delta"] as? [String: Any],
              let content = delta["content"] as? String
        else { return nil }
        return content
    }

    static let encoder = JSONEncoder()
    static let decoder = JSONDecoder()
}

/// Type-eraser so `makeRequest` can accept any `Encodable` body.
private struct AnyEncodable: Encodable {
    private let encodeFunc: (Encoder) throws -> Void
    init(_ wrapped: Encodable) { self.encodeFunc = wrapped.encode }
    func encode(to encoder: Encoder) throws { try encodeFunc(encoder) }
}
