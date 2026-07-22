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
    // --- HEARTH stateful KV-cache contract (schema v2; nil/defaulted on a v1 sidecar) ----------
    // When `writePos` names an input and `stateLayers` is set, the model uses the HEARTH native
    // stateful contract (`inputIds [1,1]`, fixed-width `causalMask`, per-layer `keyCache{i}`/
    // `valueCache{i}` states) that swift-transformers cannot drive — see the native decode path.
    var causalMaskName: String?
    var writePosName: String?
    var stateLayers: Int?
    var keyCachePrefix: String?
    var valueCachePrefix: String?

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
        case causalMaskName = "causal_mask_name"
        case writePosName = "write_pos_name"
        case stateLayers = "state_layers"
        case keyCachePrefix = "key_cache_prefix"
        case valueCachePrefix = "value_cache_prefix"
    }

    /// The newest schema version this build understands. v1 (Approach A) sidecars still decode:
    /// the v2-only fields are optional and absent keys stay `nil`.
    static let supportedSchemaVersion = 2

    /// Whether this manifest describes the HEARTH native stateful contract (writePos + per-layer
    /// states) that needs the native decode loop rather than swift-transformers' `LanguageModel`.
    var usesHearthStatefulContract: Bool {
        stateful && writePosName != nil
    }

    var causalMaskFeature: String { causalMaskName ?? "causalMask" }
    var writePosFeature: String { writePosName ?? "writePos" }
    var keyCacheStatePrefix: String { keyCachePrefix ?? "keyCache" }
    var valueCacheStatePrefix: String { valueCachePrefix ?? "valueCache" }
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
            manifest.schemaVersion >= 1,
            manifest.schemaVersion <= CoreMLSidecarManifest.supportedSchemaVersion
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

        // The HEARTH stateful contract (writePos + per-layer keyCache{i}/valueCache{i} states) is a
        // fully-static single-token graph swift-transformers can't drive; use the native loop.
        // Detected from the manifest, or (defensively) from the model's own I/O description.
        if sidecar.manifest.usesHearthStatefulContract
            || Self.exposesHearthStatefulContract(model: model, manifest: sidecar.manifest)
        {
            return try Self.runNativeStatefulGeneration(
                messages: messages,
                options: options,
                model: model,
                tokenizer: tokenizer,
                manifest: sidecar.manifest,
                stream: stream
            )
        }

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

    // MARK: HEARTH native stateful decode (ADR-011 Approach B)

    /// Fallback detection for the HEARTH stateful contract from the model's own description, for
    /// when a manifest predates the v2 fields: a `writePos` input plus a per-layer `keyCache0` state.
    static func exposesHearthStatefulContract(model: MLModel, manifest: CoreMLSidecarManifest) -> Bool {
        let description = model.modelDescription
        let hasWritePos = description.inputDescriptionsByName[manifest.writePosFeature] != nil
        let hasPerLayerState =
            description.stateDescriptionsByName["\(manifest.keyCacheStatePrefix)0"] != nil
        return hasWritePos && hasPerLayerState
    }

    /// Drive the fully-static single-token stateful model with a native decode loop.
    ///
    /// Contract (matches `hearth.coreml._stateful_export_runner`): feed `inputIds [1,1]` (int32),
    /// `causalMask [1,1,1,STATE_LEN]` (fp16; 0 for slots ≤ pos, else -1e4), `writePos [1]` (int32);
    /// read `logits[0, 0, :]`. Per-layer `keyCache{i}`/`valueCache{i}` states live in the `MLState`
    /// from `model.makeState()`. Prefill one token at a time (writePos = 0…promptCount-1), then
    /// decode greedily (temperature 0) or by sampling, stopping on any manifest eos, up to maxTokens.
    static func runNativeStatefulGeneration(
        messages: [ChatMessage],
        options: InferenceOptions,
        model: MLModel,
        tokenizer: Tokenizer,
        manifest: CoreMLSidecarManifest,
        stream: (@Sendable (String) -> Void)?
    ) throws -> String {
        let stateLen = manifest.maxSeqLen
        let promptTokens = try encodePrompt(messages, tokenizer: tokenizer)
        guard !promptTokens.isEmpty else { return "" }
        guard promptTokens.count <= stateLen else {
            throw HearthError.onDeviceUnavailable(
                "prompt is \(promptTokens.count) tokens but the stateful model's fixed window is "
                    + "\(stateLen); export with a larger --max-seq-len or shorten the prompt"
            )
        }

        let state = model.makeState()
        let maxTokens = options.maxTokens ?? 256
        let temperature = options.temperature ?? 0
        let stopIDs = Set(manifest.eosTokenIds)

        // Prefill: run every prompt token so its KV lands in the cache. The last prefill step's
        // logits seed the first generated token.
        var lastLogits: MLMultiArray?
        for (pos, token) in promptTokens.enumerated() {
            lastLogits = try predictStep(
                model: model, state: state, token: token, pos: pos,
                stateLen: stateLen, manifest: manifest
            )
        }

        var generated: [Int] = []
        var pos = promptTokens.count - 1
        let cursor = TextCursor()

        func emitDelta() {
            guard let stream else { return }
            let text = tokenizer.decode(tokens: generated, skipSpecialTokens: true)
            if text.count > cursor.emittedCount {
                let delta = String(text.dropFirst(cursor.emittedCount))
                cursor.emittedCount = text.count
                stream(delta)
            }
        }

        var current = try nextToken(from: lastLogits, temperature: temperature, vocab: manifest.vocabSize)
        while generated.count < maxTokens {
            if stopIDs.contains(current) { break }
            generated.append(current)
            emitDelta()
            if generated.count >= maxTokens || pos + 1 >= stateLen { break }
            pos += 1
            let logits = try predictStep(
                model: model, state: state, token: current, pos: pos,
                stateLen: stateLen, manifest: manifest
            )
            current = try nextToken(from: logits, temperature: temperature, vocab: manifest.vocabSize)
        }

        return tokenizer.decode(tokens: generated, skipSpecialTokens: true)
    }

    /// One single-token stateful prediction; returns the `logits` multi-array (`[1, 1, vocab]`).
    private static func predictStep(
        model: MLModel,
        state: MLState,
        token: Int,
        pos: Int,
        stateLen: Int,
        manifest: CoreMLSidecarManifest
    ) throws -> MLMultiArray {
        let inputIds = try MLMultiArray(shape: [1, 1], dataType: .int32)
        inputIds[0] = NSNumber(value: Int32(token))

        // causalMask [1,1,1,STATE_LEN]: 0 attend for slots 0…pos, -1e4 (fp16-safe) for the rest.
        let mask = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: stateLen)], dataType: .float16)
        for slot in 0..<stateLen {
            mask[slot] = NSNumber(value: Float(slot <= pos ? 0 : -1e4))
        }

        let writePos = try MLMultiArray(shape: [1], dataType: .int32)
        writePos[0] = NSNumber(value: Int32(pos))

        let features = try MLDictionaryFeatureProvider(dictionary: [
            manifest.inputName: inputIds,
            manifest.causalMaskFeature: mask,
            manifest.writePosFeature: writePos,
        ])
        let out = try model.prediction(from: features, using: state)
        guard let logits = out.featureValue(for: manifest.outputName)?.multiArrayValue else {
            throw HearthError.onDeviceUnavailable(
                "stateful Core ML model produced no `\(manifest.outputName)` output"
            )
        }
        return logits
    }

    /// Pick the next token id from `[1, 1, vocab]` logits: argmax (greedy) when temperature ≤ 0,
    /// else temperature-scaled softmax sampling. The single-token contract means we read row 0.
    private static func nextToken(
        from logits: MLMultiArray?,
        temperature: Double,
        vocab: Int
    ) throws -> Int {
        guard let logits else {
            throw HearthError.onDeviceUnavailable("no logits to sample from (empty prefill?)")
        }
        let count = logits.count
        let width = min(vocab, count)
        // Logits are `[1, 1, vocab]`; the last dimension is contiguous, so slot i lives at offset i.
        let base = count - width

        if temperature <= 0 {
            var bestIndex = 0
            var bestValue = -Double.infinity
            for i in 0..<width {
                let v = logits[base + i].doubleValue
                if v > bestValue {
                    bestValue = v
                    bestIndex = i
                }
            }
            return bestIndex
        }

        var maxValue = -Double.infinity
        for i in 0..<width {
            maxValue = max(maxValue, logits[base + i].doubleValue)
        }
        var probs = [Double](repeating: 0, count: width)
        var sum = 0.0
        for i in 0..<width {
            let p = exp((logits[base + i].doubleValue - maxValue) / temperature)
            probs[i] = p
            sum += p
        }
        var r = Double.random(in: 0..<1) * sum
        for i in 0..<width {
            r -= probs[i]
            if r <= 0 { return i }
        }
        return width - 1
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
