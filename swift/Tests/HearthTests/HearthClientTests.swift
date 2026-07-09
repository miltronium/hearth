// Unit tests for the HEARTH Swift SDK that don't require a live server:
// request encoding, URL construction, auth headers, and SSE delta parsing.

import XCTest
@testable import Hearth

final class HearthClientTests: XCTestCase {
    private let base = URL(string: "http://127.0.0.1:8080")!

    func testBaseURLTrimsTrailingV1() {
        let a = HearthClient(baseURL: URL(string: "http://127.0.0.1:8080/v1")!)
        let b = HearthClient(baseURL: base)
        XCTAssertEqual(a.baseURL, b.baseURL)
        XCTAssertEqual(a.baseURL.absoluteString, "http://127.0.0.1:8080")
    }

    func testRequestURLAndMethod() throws {
        let client = HearthClient(baseURL: base)
        let request = try client.makeRequest(
            path: "v1/chat/completions",
            body: ChatRequest(messages: [ChatMessage(role: "user", content: "hi")])
        )
        XCTAssertEqual(request.url?.absoluteString, "http://127.0.0.1:8080/v1/chat/completions")
        XCTAssertEqual(request.httpMethod, "POST")
        XCTAssertEqual(request.value(forHTTPHeaderField: "Content-Type"), "application/json")
    }

    func testAuthHeaderPresentOnlyWithToken() throws {
        let withToken = HearthClient(baseURL: base, token: "secret")
        let noToken = HearthClient(baseURL: base)
        let a = try withToken.makeRequest(
            path: "v1/embeddings",
            body: ChatRequest(messages: [])
        )
        let b = try noToken.makeRequest(
            path: "v1/embeddings",
            body: ChatRequest(messages: [])
        )
        XCTAssertEqual(a.value(forHTTPHeaderField: "Authorization"), "Bearer secret")
        XCTAssertNil(b.value(forHTTPHeaderField: "Authorization"))
    }

    func testChatRequestEncodesHearthOptions() throws {
        let client = HearthClient(baseURL: base)
        let request = try client.makeRequest(
            path: "v1/chat/completions",
            body: ChatRequest(
                messages: [ChatMessage(role: "user", content: "x")],
                stream: false,
                hearth: HearthOptions(intent: "summarize", allowEscalation: false)
            )
        )
        let body = try XCTUnwrap(request.httpBody)
        let json = try XCTUnwrap(
            JSONSerialization.jsonObject(with: body) as? [String: Any]
        )
        XCTAssertEqual(json["model"] as? String, "auto")
        let hearth = try XCTUnwrap(json["hearth"] as? [String: Any])
        XCTAssertEqual(hearth["intent"] as? String, "summarize")
        // snake_case key must match the Python API's `allow_escalation`.
        XCTAssertEqual(hearth["allow_escalation"] as? Bool, false)
    }

    func testDeltaContentParsing() {
        let payload = """
        {"choices":[{"delta":{"content":"hello"}}]}
        """
        XCTAssertEqual(HearthClient.deltaContent(from: payload), "hello")
        // A role-only first chunk has no content delta.
        XCTAssertNil(HearthClient.deltaContent(from: #"{"choices":[{"delta":{"role":"assistant"}}]}"#))
        XCTAssertNil(HearthClient.deltaContent(from: "[DONE]"))
    }

    func testChatResponseDecodesText() throws {
        let json = """
        {"id":"chatcmpl-1","model":"echo","choices":[
          {"message":{"role":"assistant","content":"[echo] hi"}}]}
        """.data(using: .utf8)!
        let decoded = try JSONDecoder().decode(ChatResponse.self, from: json)
        XCTAssertEqual(decoded.text, "[echo] hi")
    }
}
