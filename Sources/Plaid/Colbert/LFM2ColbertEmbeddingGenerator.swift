import CoreML
import Foundation

public enum LFM2ColbertGeneratorError: Error, LocalizedError {
    case modelNotFound(URL)
    case missingOutput(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let url):
            return "No compiled Core ML model found at \(url.path)."
        case .missingOutput(let name):
            return "Core ML output \(name) was not found in the prediction result."
        }
    }
}

/// ColBERT encoder for the LFM2.5 model family.
///
/// Unlike the earlier single-model `LFM2Colbert`, LFM2.5 ships **two** compiled
/// Core ML encoders in one repository — a query encoder and a document encoder —
/// each with a *fixed* input shape (no shape flexibility): the query model takes
/// `[1, 32]` and the doc model `[1, 512]`. This generator loads both, routes each
/// request to the matching model by `isQuery`, and pads the token sequence to that
/// model's exact required length (read from the model description at load time).
public final class LFM2ColbertEmbeddingGenerator: ColbertEmbeddingGenerator, @unchecked Sendable {
    private let queryModel: MLModel
    private let docModel: MLModel
    private let tokenizer: ColbertTokenizer
    private let skiplistTokenIds: Set<Int>
    /// Fixed `input_ids` length the query model requires (typically 32).
    private let queryLength: Int
    /// Fixed `input_ids` length the doc model requires (typically 512).
    private let docLength: Int

    static let defaultSkiplistCharacters: [String] = {
        let punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
        return punctuation.map { String($0) }
    }()

    /// Hugging Face repository that hosts the compiled query/doc encoders.
    public static let defaultRepoId = "smdesai/LFM2.5-ColBERT-350M"
    /// Compiled query-encoder directory inside `defaultRepoId` (without `.mlmodelc`).
    public static let defaultQueryModelName = "LFM25-ColBERT-query-6bit"
    /// Compiled document-encoder directory inside `defaultRepoId` (without `.mlmodelc`).
    public static let defaultDocModelName = "LFM25-ColBERT-doc-6bit"
    /// Fallback input lengths, used only if a model omits its `input_ids` shape.
    public static let defaultQuerySequenceLength = 32
    public static let defaultDocSequenceLength = 512

    /// Downloads (or reuses the cached copies of) both Core ML encoders from the
    /// Hugging Face Hub, then builds a generator on top of them. Usual entry point.
    public static func download(
        tokenizer: ColbertTokenizer,
        repoId: String = defaultRepoId,
        queryModelName: String = defaultQueryModelName,
        docModelName: String = defaultDocModelName,
        revision: String = "main",
        configuration: MLModelConfiguration = MLModelConfiguration(),
        skiplistWords: [String]? = nil,
        progressHandler: ColbertModelDownloader.ProgressHandler? = nil
    ) async throws -> LFM2ColbertEmbeddingGenerator {
        // Both encoders live in the same repo; the snapshot is cached, so the
        // second call is a local lookup of the other bundle.
        let queryModelURL = try await ColbertModelDownloader.download(
            repoId: repoId, modelName: queryModelName, revision: revision,
            progressHandler: progressHandler)
        let docModelURL = try await ColbertModelDownloader.download(
            repoId: repoId, modelName: docModelName, revision: revision,
            progressHandler: progressHandler)
        return try LFM2ColbertEmbeddingGenerator(
            tokenizer: tokenizer, queryModelURL: queryModelURL, docModelURL: docModelURL,
            configuration: configuration, skiplistWords: skiplistWords)
    }

    /// Builds a generator from already-available compiled models (`.mlmodelc`) or
    /// `.mlpackage`s, which are compiled on the fly.
    public init(
        tokenizer: ColbertTokenizer,
        queryModelURL: URL,
        docModelURL: URL,
        configuration: MLModelConfiguration = MLModelConfiguration(),
        skiplistWords: [String]? = nil
    ) throws {
        let queryURL = try Self.resolveModelURL(queryModelURL)
        let docURL = try Self.resolveModelURL(docModelURL)
        configuration.computeUnits = .cpuAndGPU
        self.queryModel = try MLModel(contentsOf: queryURL, configuration: configuration)
        self.docModel = try MLModel(contentsOf: docURL, configuration: configuration)
        self.tokenizer = tokenizer
        self.queryLength =
            Self.inputLength(of: queryModel) ?? Self.defaultQuerySequenceLength
        self.docLength =
            Self.inputLength(of: docModel) ?? Self.defaultDocSequenceLength
        let words = skiplistWords ?? Self.defaultSkiplistCharacters
        self.skiplistTokenIds = Self.buildSkiplist(tokenizer: tokenizer, words: words)
    }

    private func model(isQuery: Bool) -> MLModel { isQuery ? queryModel : docModel }
    private func sequenceLength(isQuery: Bool) -> Int { isQuery ? queryLength : docLength }

    public func generateEmbeddings(
        for sentence: String,
        isQuery: Bool,
        maxLength: Int
    ) throws -> ColbertEmbeddingBatch {
        let inputIds = tokenizer.buildModelTokens(
            sentence: sentence, isQuery: isQuery,
            sequenceLength: sequenceLength(isQuery: isQuery))
        return try generateEmbeddingsFromInputIds(inputIds, isQuery: isQuery, maxLength: maxLength)
    }

    public func generateEmbeddings(
        fromTokenIds tokenIds: [Int],
        isQuery: Bool,
        maxLength: Int
    ) throws -> ColbertEmbeddingBatch {
        let inputIds = tokenizer.buildModelTokensFromIds(
            tokenIds: tokenIds, isQuery: isQuery,
            sequenceLength: sequenceLength(isQuery: isQuery))
        return try generateEmbeddingsFromInputIds(inputIds, isQuery: isQuery, maxLength: maxLength)
    }

    private func generateEmbeddingsFromInputIds(
        _ inputIds: [Int],
        isQuery: Bool,
        maxLength: Int
    ) throws -> ColbertEmbeddingBatch {
        // `inputIds` is already padded to the model's fixed length by the caller.
        let effectiveLength = min(maxLength, inputIds.count)
        let padTokenId =
            isQuery ? tokenizer.queryPadTokenIdentifier : tokenizer.docPadTokenIdentifier

        let attentionMask = Self.buildAttentionMask(
            inputIds: inputIds, effectiveLength: effectiveLength, padTokenId: padTokenId,
            skiplistTokenIds: skiplistTokenIds)

        let inputIdsArray = try MLMultiArray.makeInt32Batch(values: inputIds)
        let attentionArray = try MLMultiArray.makeInt32Batch(values: attentionMask)

        let inputs = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: inputIdsArray),
            "attention_mask": MLFeatureValue(multiArray: attentionArray),
        ])

        let prediction = try model(isQuery: isQuery).prediction(from: inputs)
        guard
            let tokenEmbeddings = prediction.featureValue(for: "token_embeddings")?.multiArrayValue
        else {
            throw LFM2ColbertGeneratorError.missingOutput("token_embeddings")
        }

        let validTokenCount = max(attentionMask.reduce(0, +), 1)
        let embeddings = Self.extractEmbeddings(from: tokenEmbeddings, limit: validTokenCount)
        let boolMask = Array(attentionMask.prefix(validTokenCount)).map { $0 != 0 }

        return ColbertEmbeddingBatch(embeddings: embeddings, attentionMask: boolMask)
    }

    /// Batch processing: encode multiple sentences in a single model pass
    /// Uses CoreML's predictions(inputs:) API for true batch processing
    public func generateEmbeddingsBatch(
        for sentences: [String],
        isQuery: Bool,
        maxLength: Int
    ) throws -> [ColbertEmbeddingBatch] {
        guard !sentences.isEmpty else { return [] }

        // Single sentence - use standard path
        if sentences.count == 1 {
            return try [
                generateEmbeddings(for: sentences[0], isQuery: isQuery, maxLength: maxLength)
            ]
        }

        let tokenIdBatches = sentences.map {
            tokenizer.buildModelTokens(
                sentence: $0, isQuery: isQuery, sequenceLength: sequenceLength(isQuery: isQuery))
        }
        return try runInputIdBatch(tokenIdBatches, isQuery: isQuery, maxLength: maxLength)
    }

    /// Batch processing from pre-tokenized IDs (most efficient)
    public func generateEmbeddingsBatch(
        fromTokenIds tokenIdBatch: [[Int]],
        isQuery: Bool,
        maxLength: Int
    ) throws -> [ColbertEmbeddingBatch] {
        guard !tokenIdBatch.isEmpty else { return [] }

        // Single batch - use standard path
        if tokenIdBatch.count == 1 {
            return try [
                generateEmbeddings(
                    fromTokenIds: tokenIdBatch[0], isQuery: isQuery, maxLength: maxLength)
            ]
        }

        let inputIdBatches = tokenIdBatch.map {
            tokenizer.buildModelTokensFromIds(
                tokenIds: $0, isQuery: isQuery, sequenceLength: sequenceLength(isQuery: isQuery))
        }
        return try runInputIdBatch(inputIdBatches, isQuery: isQuery, maxLength: maxLength)
    }

    private func runInputIdBatch(
        _ inputIdBatches: [[Int]],
        isQuery: Bool,
        maxLength: Int
    ) throws -> [ColbertEmbeddingBatch] {
        let prepared = try prepareModelReadyBatch(
            inputIdBatches, isQuery: isQuery, maxLength: maxLength)
        return try runPreparedBatch(prepared)
    }

    /// Build the model inputs for a batch of *already model-ready* token-id
    /// sequences (each padded to this encoder's fixed length). CPU-only, no
    /// inference — extracted so the encode loop can run it ahead of time on a
    /// background thread while the GPU works on the previous batch.
    private func prepareModelReadyBatch(
        _ inputIdBatches: [[Int]],
        isQuery: Bool,
        maxLength: Int
    ) throws -> PreparedColbertBatch {
        let padTokenId =
            isQuery ? tokenizer.queryPadTokenIdentifier : tokenizer.docPadTokenIdentifier

        var batchInputs: [MLDictionaryFeatureProvider] = []
        var allAttentionMasks: [[Int]] = []
        batchInputs.reserveCapacity(inputIdBatches.count)
        allAttentionMasks.reserveCapacity(inputIdBatches.count)

        for inputIds in inputIdBatches {
            let effectiveLength = min(maxLength, inputIds.count)
            let attentionMask = Self.buildAttentionMask(
                inputIds: inputIds, effectiveLength: effectiveLength, padTokenId: padTokenId,
                skiplistTokenIds: skiplistTokenIds)

            let inputIdsArray = try MLMultiArray.makeInt32Batch(values: inputIds)
            let attentionArray = try MLMultiArray.makeInt32Batch(values: attentionMask)

            let input = try MLDictionaryFeatureProvider(dictionary: [
                "input_ids": MLFeatureValue(multiArray: inputIdsArray),
                "attention_mask": MLFeatureValue(multiArray: attentionArray),
            ])

            batchInputs.append(input)
            allAttentionMasks.append(attentionMask)
        }

        return PreparedColbertBatch(
            batchProvider: MLArrayBatchProvider(array: batchInputs),
            attentionMasks: allAttentionMasks,
            isQuery: isQuery)
    }

    public func prepareBatch(
        for sentences: [String],
        isQuery: Bool,
        maxLength: Int
    ) throws -> PreparedColbertBatch {
        let inputIdBatches = sentences.map {
            tokenizer.buildModelTokens(
                sentence: $0, isQuery: isQuery, sequenceLength: sequenceLength(isQuery: isQuery))
        }
        return try prepareModelReadyBatch(inputIdBatches, isQuery: isQuery, maxLength: maxLength)
    }

    public func runPreparedBatch(_ prepared: PreparedColbertBatch) throws -> [ColbertEmbeddingBatch]
    {
        // Route to the encoder the inputs were prepared for.
        let predictions = try model(isQuery: prepared.isQuery).predictions(
            from: prepared.batchProvider, options: MLPredictionOptions())

        var results: [ColbertEmbeddingBatch] = []
        results.reserveCapacity(predictions.count)

        for index in 0 ..< predictions.count {
            let prediction = predictions.features(at: index)
            guard
                let tokenEmbeddings = prediction.featureValue(for: "token_embeddings")?
                    .multiArrayValue
            else {
                throw LFM2ColbertGeneratorError.missingOutput("token_embeddings")
            }

            let validTokenCount = max(prepared.attentionMasks[index].reduce(0, +), 1)
            let embeddings = Self.extractEmbeddings(from: tokenEmbeddings, limit: validTokenCount)
            let boolMask = Array(prepared.attentionMasks[index].prefix(validTokenCount)).map {
                $0 != 0
            }

            results.append(ColbertEmbeddingBatch(embeddings: embeddings, attentionMask: boolMask))
        }

        return results
    }

    /// Build the attention mask for a model-ready token sequence: attend to the
    /// first `effectiveLength` non-pad tokens, then drop skiplist punctuation that
    /// follows the two prefix tokens ([BOS], [Q]/[D]).
    private static func buildAttentionMask(
        inputIds: [Int],
        effectiveLength: Int,
        padTokenId: Int,
        skiplistTokenIds: Set<Int>
    ) -> [Int] {
        var attentionMask: [Int] = []
        attentionMask.reserveCapacity(inputIds.count)
        for (idx, token) in inputIds.enumerated() {
            let withinAllowedLength = idx < effectiveLength
            attentionMask.append((withinAllowedLength && token != padTokenId) ? 1 : 0)
        }

        // Skip masking for prefix tokens ([BOS], [Q]/[D])
        for index in 2 ..< min(effectiveLength, inputIds.count) {
            if skiplistTokenIds.contains(inputIds[index]) {
                attentionMask[index] = 0
            }
        }

        return attentionMask
    }

    private static func resolveModelURL(_ url: URL) throws -> URL {
        let fm = FileManager.default
        guard fm.fileExists(atPath: url.path) else {
            throw LFM2ColbertGeneratorError.modelNotFound(url)
        }
        if url.pathExtension == "mlpackage" {
            return try MLModel.compileModel(at: url)
        }
        return url
    }

    /// The fixed `input_ids` length declared by a compiled model, or `nil` if the
    /// model doesn't expose a constrained multi-array shape for that input.
    private static func inputLength(of model: MLModel) -> Int? {
        guard
            let constraint = model.modelDescription
                .inputDescriptionsByName["input_ids"]?.multiArrayConstraint
        else {
            return nil
        }
        return constraint.shape.last?.intValue
    }

    public func tokenizeToIds(text: String) -> [Int] {
        return tokenizer.tokenizeToIds(text: text)
    }

    private static func buildSkiplist(tokenizer: ColbertTokenizer, words: [String])
        -> Set<Int>
    {
        var set = Set<Int>()
        for word in words {
            if let tokenId = tokenizer.tokenId(for: word) {
                set.insert(tokenId)
            }
        }
        return set
    }

    private static func extractEmbeddings(from array: MLMultiArray, limit: Int) -> [[Float]] {
        MLMultiArray.colbertRowMajorFloats(from: array, limit: limit)
    }
}
