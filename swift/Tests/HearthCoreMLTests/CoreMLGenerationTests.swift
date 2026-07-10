// Offline tests for the Core ML generation contract: locating and parsing the sidecar the
// Python exporter writes next to a `.mlpackage`. The token-generation loop itself needs a real
// exported model and is validated on real hardware (docs/HANDOFF.md → Task C); here we prove the
// Swift side reads exactly what `hearth models export-coreml` writes.

import Foundation
import XCTest

@testable import HearthCoreML

#if canImport(CoreML)

final class CoreMLGenerationTests: XCTestCase {

    private func tempDir() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("hearth-coreml-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// Writes a sidecar in the exact shape `hearth.coreml.CoreMLManifest.to_dict` produces.
    private func writeSidecar(
        stem: String,
        in dir: URL,
        schemaVersion: Int = 1,
        withTokenizer: Bool = true,
        withTokenizerConfig: Bool = true
    ) throws {
        let manifest = """
        {
          "schema_version": \(schemaVersion),
          "source": "org/model",
          "stateful": true,
          "input_name": "inputIds",
          "output_name": "logits",
          "max_seq_len": 512,
          "vocab_size": 100,
          "bos_token_id": 1,
          "eos_token_ids": [2, 7],
          "chat_template_id": "chatml",
          "tokenizer_files": ["\(stem).tokenizer.json"],
          "compute_units": "cpuAndNeuralEngine",
          "precision": "float16"
        }
        """
        try manifest.write(
            to: dir.appendingPathComponent("\(stem).hearth-coreml.json"),
            atomically: true, encoding: .utf8
        )
        if withTokenizer {
            try "{}".write(
                to: dir.appendingPathComponent("\(stem).tokenizer.json"),
                atomically: true, encoding: .utf8
            )
        }
        if withTokenizerConfig {
            try #"{"chat_template": "x"}"#.write(
                to: dir.appendingPathComponent("\(stem).tokenizer_config.json"),
                atomically: true, encoding: .utf8
            )
        }
    }

    func testLocatesAndParsesSidecarNextToModel() throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeSidecar(stem: "qwen", in: dir)

        let sidecar = try XCTUnwrap(
            CoreMLSidecar.locate(modelURL: dir.appendingPathComponent("qwen.mlpackage"))
        )
        XCTAssertEqual(sidecar.manifest.source, "org/model")
        XCTAssertEqual(sidecar.manifest.inputName, "inputIds")
        XCTAssertEqual(sidecar.manifest.outputName, "logits")
        XCTAssertEqual(sidecar.manifest.eosTokenIds, [2, 7])
        XCTAssertEqual(sidecar.manifest.bosTokenId, 1)
        XCTAssertEqual(sidecar.manifest.chatTemplateId, "chatml")
        XCTAssertTrue(sidecar.manifest.stateful)
        XCTAssertEqual(sidecar.tokenizerJSON.lastPathComponent, "qwen.tokenizer.json")
        XCTAssertEqual(sidecar.tokenizerConfig?.lastPathComponent, "qwen.tokenizer_config.json")
    }

    func testTokenizerConfigIsOptional() throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeSidecar(stem: "m", in: dir, withTokenizerConfig: false)

        let sidecar = try XCTUnwrap(
            CoreMLSidecar.locate(modelURL: dir.appendingPathComponent("m.mlpackage"))
        )
        XCTAssertNil(sidecar.tokenizerConfig)
    }

    func testAbsentManifestYieldsNil() throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertNil(CoreMLSidecar.locate(modelURL: dir.appendingPathComponent("nope.mlpackage")))
    }

    func testIncompatibleSchemaYieldsNil() throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeSidecar(stem: "m", in: dir, schemaVersion: 999)
        XCTAssertNil(CoreMLSidecar.locate(modelURL: dir.appendingPathComponent("m.mlpackage")))
    }

    func testMissingTokenizerJSONYieldsNil() throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeSidecar(stem: "m", in: dir, withTokenizer: false)
        XCTAssertNil(CoreMLSidecar.locate(modelURL: dir.appendingPathComponent("m.mlpackage")))
    }
}

#endif
