import CoreML
import Foundation

public enum MXBAIEdgeColbertGeneratorError: Error, LocalizedError {
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

public final class MXBAIEdgeColbertEmbeddingGenerator: ColbertEmbeddingGenerator {
    private let model: MLModel
    private let tokenizer: ColbertTokenizer
    private let skiplistTokenIds: Set<Int>
    private let maxSequenceLength: Int

    static let defaultSkiplistCharacters: [String] = {
        let punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
        return punctuation.map { String($0) }
    }()

    /// Hugging Face repository that hosts the compiled `MXBAIEdgeColbert.mlmodelc`.
    public static let defaultRepoId = "smdesai/MXBAIEdgeColbert"
    /// Name of the compiled model directory inside `defaultRepoId` (without `.mlmodelc`).
    public static let defaultModelName = "MXBAIEdgeColbert"

    /// Downloads (or reuses the cached copy of) the Core ML encoder from the Hugging Face
    /// Hub, then builds a generator on top of it. This is the usual entry point.
    public static func download(
        tokenizer: ColbertTokenizer,
        repoId: String = defaultRepoId,
        modelName: String = defaultModelName,
        revision: String = "main",
        configuration: MLModelConfiguration = MLModelConfiguration(),
        skiplistWords: [String]? = nil,
        progressHandler: ColbertModelDownloader.ProgressHandler? = nil
    ) async throws -> MXBAIEdgeColbertEmbeddingGenerator {
        let modelURL = try await ColbertModelDownloader.download(
            repoId: repoId, modelName: modelName, revision: revision,
            progressHandler: progressHandler)
        return try MXBAIEdgeColbertEmbeddingGenerator(
            tokenizer: tokenizer, modelURL: modelURL, configuration: configuration,
            skiplistWords: skiplistWords)
    }

    /// Builds a generator from an already-available compiled model (`.mlmodelc`) or a
    /// `.mlpackage`, which is compiled on the fly.
    public init(
        tokenizer: ColbertTokenizer,
        modelURL: URL,
        configuration: MLModelConfiguration = MLModelConfiguration(),
        skiplistWords: [String]? = nil
    ) throws {
        let modelURL = try Self.resolveModelURL(modelURL)
        configuration.computeUnits = .cpuAndGPU
        self.model = try MLModel(contentsOf: modelURL, configuration: configuration)
        self.tokenizer = tokenizer
        self.maxSequenceLength = tokenizer.maxSequenceLength
        let words = skiplistWords ?? Self.defaultSkiplistCharacters
        self.skiplistTokenIds = Self.buildSkiplist(tokenizer: tokenizer, words: words)
    }

    public func generateEmbeddings(
        for sentence: String,
        isQuery: Bool,
        maxLength: Int
    ) throws -> ColbertEmbeddingBatch {
        let inputIds = tokenizer.buildModelTokens(sentence: sentence, isQuery: isQuery)
        return try generateEmbeddingsFromInputIds(inputIds, isQuery: isQuery, maxLength: maxLength)
    }

    public func generateEmbeddings(
        fromTokenIds tokenIds: [Int],
        isQuery: Bool,
        maxLength: Int
    ) throws -> ColbertEmbeddingBatch {
        let inputIds = tokenizer.buildModelTokensFromIds(tokenIds: tokenIds, isQuery: isQuery)
        return try generateEmbeddingsFromInputIds(inputIds, isQuery: isQuery, maxLength: maxLength)
    }

    private func generateEmbeddingsFromInputIds(
        _ inputIds: [Int],
        isQuery: Bool,
        maxLength: Int
    ) throws -> ColbertEmbeddingBatch {
        let effectiveLength = min(maxLength, maxSequenceLength)
        let padTokenId =
            isQuery ? tokenizer.queryPadTokenIdentifier : tokenizer.docPadTokenIdentifier

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

        let inputIdsArray = try MLMultiArray.makeInt32Batch(values: inputIds)
        let attentionArray = try MLMultiArray.makeInt32Batch(values: attentionMask)

        let inputs = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: inputIdsArray),
            "attention_mask": MLFeatureValue(multiArray: attentionArray),
        ])

        let prediction = try model.prediction(from: inputs)
        guard
            let tokenEmbeddings = prediction.featureValue(for: "token_embeddings")?.multiArrayValue
        else {
            throw MXBAIEdgeColbertGeneratorError.missingOutput("token_embeddings")
        }

        let validTokenCount = max(attentionMask.reduce(0, +), 1)
        let embeddings = Self.extractEmbeddings(from: tokenEmbeddings, limit: validTokenCount)
        let boolMask = Array(attentionMask.prefix(validTokenCount)).map { $0 != 0 }

        //print("generateEmbeddings preview:\n\(EmbeddingFormatting.formatEmbeddingsPreview(embeddings))")

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
            tokenizer.buildModelTokens(sentence: $0, isQuery: isQuery)
        }
        return try generateEmbeddingsBatchFromInputIds(
            tokenIdBatches, isQuery: isQuery, maxLength: maxLength)
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
            tokenizer.buildModelTokensFromIds(tokenIds: $0, isQuery: isQuery)
        }
        return try generateEmbeddingsBatchFromInputIds(
            inputIdBatches, isQuery: isQuery, maxLength: maxLength)
    }

    private func generateEmbeddingsBatchFromInputIds(
        _ inputIdBatches: [[Int]],
        isQuery: Bool,
        maxLength: Int
    ) throws -> [ColbertEmbeddingBatch] {
        let effectiveLength = min(maxLength, maxSequenceLength)
        let padTokenId =
            isQuery ? tokenizer.queryPadTokenIdentifier : tokenizer.docPadTokenIdentifier

        // Build individual inputs for each token sequence
        var batchInputs: [MLDictionaryFeatureProvider] = []
        var allAttentionMasks: [[Int]] = []
        batchInputs.reserveCapacity(inputIdBatches.count)
        allAttentionMasks.reserveCapacity(inputIdBatches.count)

        for inputIds in inputIdBatches {
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

            // Create individual MLMultiArray for this sequence
            let inputIdsArray = try MLMultiArray.makeInt32Batch(values: inputIds)
            let attentionArray = try MLMultiArray.makeInt32Batch(values: attentionMask)

            let input = try MLDictionaryFeatureProvider(dictionary: [
                "input_ids": MLFeatureValue(multiArray: inputIdsArray),
                "attention_mask": MLFeatureValue(multiArray: attentionArray),
            ])

            batchInputs.append(input)
            allAttentionMasks.append(attentionMask)
        }

        // Use predictions(from:options:) for batch processing - CoreML handles batching!
        let batchProvider = MLArrayBatchProvider(array: batchInputs)
        let predictions = try model.predictions(from: batchProvider, options: MLPredictionOptions())

        // Extract embeddings from batch predictions
        var results: [ColbertEmbeddingBatch] = []
        results.reserveCapacity(predictions.count)

        for index in 0 ..< predictions.count {
            let prediction = predictions.features(at: index)
            guard
                let tokenEmbeddings = prediction.featureValue(for: "token_embeddings")?
                    .multiArrayValue
            else {
                throw MXBAIEdgeColbertGeneratorError.missingOutput("token_embeddings")
            }

            let validTokenCount = max(allAttentionMasks[index].reduce(0, +), 1)
            let embeddings = Self.extractEmbeddings(from: tokenEmbeddings, limit: validTokenCount)
            let boolMask = Array(allAttentionMasks[index].prefix(validTokenCount)).map { $0 != 0 }

            results.append(ColbertEmbeddingBatch(embeddings: embeddings, attentionMask: boolMask))
        }

        return results
    }

    private static func resolveModelURL(_ url: URL) throws -> URL {
        let fm = FileManager.default
        guard fm.fileExists(atPath: url.path) else {
            throw MXBAIEdgeColbertGeneratorError.modelNotFound(url)
        }
        if url.pathExtension == "mlpackage" {
            return try MLModel.compileModel(at: url)
        }
        return url
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
        let shape = array.shape.map { $0.intValue }
        guard shape.count >= 2 else { return [] }
        let embeddingDim = shape.last ?? 0
        let tokenCount = shape.count >= 2 ? shape[shape.count - 2] : 0
        let totalTokens = min(tokenCount, limit)
        var vectors: [[Float]] = []
        vectors.reserveCapacity(totalTokens)

        for tokenIndex in 0 ..< totalTokens {
            var vector: [Float] = []
            vector.reserveCapacity(embeddingDim)
            for dim in 0 ..< embeddingDim {
                let flatIndex = tokenIndex * embeddingDim + dim
                vector.append(array[flatIndex].floatValue)
            }
            vectors.append(vector)
        }

        return vectors
    }
}

extension MLMultiArray {
    fileprivate static func makeInt32Batch(values: [Int]) throws -> MLMultiArray {
        let shape: [NSNumber] = [1, NSNumber(value: values.count)]
        let array = try MLMultiArray(shape: shape, dataType: .int32)
        for (index, value) in values.enumerated() {
            array[index] = NSNumber(value: value)
        }
        return array
    }
}
