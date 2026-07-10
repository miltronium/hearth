// CoreMLGeneration — the offline token-generation loop for ``CoreMLProvider`` (ADR-011).
//
// The heavy lifting (stateful KV-cache decode, sampling, prefill→extend) is swift-transformers'
// `LanguageModelWithStatefulKVCache` + `Generation`. This file is the thin bridge: read the
// sidecar contract written by `hearth models export-coreml` (see src/hearth/coreml.py), build a
// `Tokenizer` from the bundled tokenizer files, wrap the already-loaded `MLModel` in the right
// `LanguageModel`, map HEARTH's transport-neutral `InferenceOptions` onto a `GenerationConfig`,
// and decode. No network, no daemon.
//
// GENERIC by design: no CAMBOT (or any consumer) types appear here (ADR-001).

import Foundation

#if canImport(CoreML)

@preconcurrency import CoreML
import Generation
import Hearth
import Hub
import Models
import Tokenizers

/// The sidecar manifest emitted next to a `.mlpackage` (`<stem>.hearth-coreml.json`). Mirrors
/// `hearth.coreml.CoreMLManifest`; the snake_case keys match the Python writer.
struct CoreMLSidecarManifest: Codable, Sendable {
    var schemaVersion: Int
    var source: String
    var stateful: Bool
    var inputName: String
    var outputName: String
    var maxSeqLen: Int
    var vocabSize: Int
    var bosTokenId: Int?
    var eosTokenIds: [Int]
    var chatTemplateId: String
    var tokenizerFiles: [String]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case source
        case stateful
        case inputName = "input_name"
        case outputName = "output_name"
        case maxSeqLen = "max_seq_len"
        case vocabSize = "vocab_size"
        case bosTokenId = "bos_token_id"
        case eosTokenIds = "eos_token_ids"
        case chatTemplateId = "chat_template_id"
        case tokenizerFiles = "tokenizer_files"
    }

    /// The schema version this build understands (kept in sync with the Python
    /// `MANIFEST_SCHEMA_VERSION`).
    static let supportedSchemaVersion = 1
}

/// A located sidecar: the parsed manifest plus the on-disk tokenizer files that ship with a model.
struct CoreMLSidecar: Sendable {
    let manifest: CoreMLSidecarManifest
    let tokenizerJSON: URL
    let tokenizerConfig: URL?

    /// Find and parse the sidecar for a model URL, or return `nil` if it's absent/incompatible.
    ///
    /// Sidecar files are stem-prefixed siblings of the `.mlpackage` (see
    /// `hearth.coreml.sidecar_paths`): `<stem>.hearth-coreml.json`, `<stem>.tokenizer.json`,
    /// `<stem>.tokenizer_config.json`.
    static func locate(modelURL: URL) -> CoreMLSidecar? {
        let dir = modelURL.deletingLastPathComponent()
        let stem = modelURL.deletingPathExtension().lastPathComponent
        let manifestURL = dir.appendingPathComponent("\(stem).hearth-coreml.json")
        guard
            let data = try? Data(contentsOf: manifestURL),
            let manifest = try? JSONDecoder().decode(CoreMLSidecarManifest.self, from: data),
            manifest.schemaVersion == CoreMLSidecarManifest.supportedSchemaVersion
        else { return nil }

        let tokenizer = dir.appendingPathComponent("\(stem).tokenizer.json")
        guard FileManager.default.fileExists(atPath: tokenizer.path) else { return nil }
        let config = dir.appendingPathComponent("\(stem).tokenizer_config.json")
        let configURL = FileManager.default.fileExists(atPath: config.path) ? config : nil
        return CoreMLSidecar(manifest: manifest, tokenizerJSON: tokenizer, tokenizerConfig: configURL)
    }

    /// Build a `PreTrainedTokenizer` from the bundled files — fully offline (no Hub download).
    func loadTokenizer() throws -> Tokenizer {
        let dataDict = try Self.jsonObject(at: tokenizerJSON)
        let configDict = try tokenizerConfig.map(Self.jsonObject(at:)) ?? [:]
        return try PreTrainedTokenizer(
            tokenizerConfig: Config(configDict),
            tokenizerData: Config(dataDict)
        )
    }

    private static func jsonObject(at url: URL) throws -> [NSString: Any] {
        let data = try Data(contentsOf: url)
        guard let dict = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw HearthError.onDeviceUnavailable(
                "malformed tokenizer JSON at \(url.lastPathComponent)"
            )
        }
        var out: [NSString: Any] = [:]
        for (key, value) in dict { out[key as NSString] = value }
        return out
    }
}

@available(macOS 15.0, iOS 18.0, visionOS 2.0, *)
extension CoreMLProvider {
    /// Run the offline generation loop against a loaded model + sidecar. When `stream` is
    /// non-nil, it receives assistant-text deltas as they are produced; the full assistant text
    /// is always returned.
    func runGeneration(
        messages: [ChatMessage],
        options: InferenceOptions,
        model: MLModel,
        sidecar: CoreMLSidecar,
        stream: (@Sendable (String) -> Void)?
    ) async throws -> String {
        let tokenizer = try sidecar.loadTokenizer()
        let languageModel = try Self.makeLanguageModel(model: model, tokenizer: tokenizer)
        await languageModel.resetState()

        let promptTokens = try Self.encodePrompt(messages, tokenizer: tokenizer)
        let promptCount = promptTokens.count
        let config = Self.generationConfig(options, tokenizer: tokenizer, manifest: sidecar.manifest)

        // Decode the growing sequence each step and emit only the newly-appended suffix. Redecoding
        // (rather than per-token decode) avoids splitting a multibyte token across deltas.
        let cursor = TextCursor()
        let callback: PredictionTokensCallback? = stream.map { emit in
            { outputTokenIDs in
                let generated = Array(outputTokenIDs.dropFirst(promptCount))
                let text = tokenizer.decode(tokens: generated, skipSpecialTokens: true)
                if text.count > cursor.emittedCount {
                    let delta = String(text.dropFirst(cursor.emittedCount))
                    cursor.emittedCount = text.count
                    emit(delta)
                }
            }
        }

        let output = try await languageModel.generate(config: config, tokens: promptTokens, callback: callback)
        let generated = Array(output.dropFirst(promptCount))
        return tokenizer.decode(tokens: generated, skipSpecialTokens: true)
    }

    /// Pick the concrete `LanguageModel` for a compiled model: the stateful KV-cache subclass when
    /// the model exposes `keyCache`/`valueCache` states with a ranged `inputIds`, else the base.
    /// Guards the ranged-shape precondition so a malformed export throws instead of trapping inside
    /// swift-transformers' `fatalError`.
    static func makeLanguageModel(model: MLModel, tokenizer: Tokenizer) throws -> LanguageModel {
        let description = model.modelDescription
        let hasKeyCache = description.stateDescriptionsByName["keyCache"] != nil
        let hasValueCache = description.stateDescriptionsByName["valueCache"] != nil
        guard hasKeyCache == hasValueCache else {
            throw HearthError.onDeviceUnavailable(
                "Core ML model has only one of keyCache/valueCache states — export is malformed"
            )
        }
        if hasKeyCache {
            guard
                let constraint = description.inputDescriptionsByName["inputIds"]?.multiArrayConstraint,
                constraint.shapeConstraint.type == .range
            else {
                throw HearthError.onDeviceUnavailable(
                    "stateful Core ML model needs a ranged `inputIds` sequence dimension "
                        + "(prefill + extend); re-export with a RangeDim length"
                )
            }
            return LanguageModelWithStatefulKVCache(model: model, tokenizer: tokenizer)
        }
        return LanguageModel(model: model, tokenizer: tokenizer)
    }

    /// Tokenize the chat: apply the model's own chat template when it has one, else flatten roles.
    static func encodePrompt(_ messages: [ChatMessage], tokenizer: Tokenizer) throws -> [Int] {
        if tokenizer.hasChatTemplate {
            let chat: [Message] = messages.map { ["role": $0.role, "content": $0.content] }
            return try tokenizer.applyChatTemplate(messages: chat)
        }
        let flattened = messages.map { "\($0.role): \($0.content)" }.joined(separator: "\n") + "\nassistant:"
        return tokenizer.encode(text: flattened)
    }

    /// Map HEARTH's transport-neutral options onto a swift-transformers `GenerationConfig`.
    /// Temperature 0 (or nil) ⇒ greedy; the stop token prefers the ChatML turn terminator so a
    /// tuned model halts at end-of-turn (Finding-2b) rather than running to `maxNewTokens`.
    static func generationConfig(
        _ options: InferenceOptions,
        tokenizer: Tokenizer,
        manifest: CoreMLSidecarManifest
    ) -> GenerationConfig {
        var config = GenerationConfig(maxNewTokens: options.maxTokens ?? 256)
        let temperature = options.temperature ?? 0
        config.temperature = Float(temperature)
        config.doSample = temperature > 0
        config.bosTokenId = manifest.bosTokenId
        config.eosTokenId = stopTokenID(tokenizer: tokenizer, manifest: manifest)
        config.padTokenId = manifest.eosTokenIds.first
        return config
    }

    /// The single token that ends generation. swift-transformers' loop stops on one `eosTokenId`;
    /// for ChatML that's `<|im_end|>` (the turn terminator the tuned model actually emits), else
    /// the manifest's first terminator, else the tokenizer's eos.
    static func stopTokenID(tokenizer: Tokenizer, manifest: CoreMLSidecarManifest) -> Int? {
        if manifest.chatTemplateId == "chatml" {
            let id = tokenizer.convertTokenToId("<|im_end|>")
            if let id, id != tokenizer.unknownTokenId { return id }
        }
        return manifest.eosTokenIds.first ?? tokenizer.eosTokenId
    }
}

/// Mutable cursor for streaming deltas, in a reference type so the escaping token callback can
/// advance it across generation steps without a mutable-capture data race.
private final class TextCursor: @unchecked Sendable {
    var emittedCount = 0
}

#endif
