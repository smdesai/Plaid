import CoreML
import Foundation

public enum MXBAIEdgeColbertGeneratorError: Error, LocalizedError {
    case modelNotFound
    case missingOutput(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound:
            return "Unable to locate MXVBAIEdgeColbert Core ML model in the Plaid bundle."
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

    public init(
        tokenizer: ColbertTokenizer,
        configuration: MLModelConfiguration = MLModelConfiguration(),
        skiplistWords: [String]? = nil
    ) throws {
        let modelURL = try Self.locateModelURL()
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

        //print("generateEmbeddings preview:\n\(Self.formatEmbeddingsPreview(embeddings))")

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

        let effectiveLength = min(maxLength, maxSequenceLength)
        let padTokenId =
            isQuery ? tokenizer.queryPadTokenIdentifier : tokenizer.docPadTokenIdentifier

        // Build individual inputs for each sentence
        var batchInputs: [MLDictionaryFeatureProvider] = []
        var allAttentionMasks: [[Int]] = []
        batchInputs.reserveCapacity(sentences.count)
        allAttentionMasks.reserveCapacity(sentences.count)

        for sentence in sentences {
            let inputIds = tokenizer.buildModelTokens(sentence: sentence, isQuery: isQuery)

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

            // Create individual MLMultiArray for this sentence
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

    private static func locateModelURL() throws -> URL {
        #if SWIFT_PACKAGE
            if let compiled = Bundle.module.url(
                forResource: "MXBAIEdgeColbert", withExtension: "mlmodelc")
            {
                return compiled
            }
            if let packageURL = Bundle.module.url(
                forResource: "MXBAIEdgeColbert", withExtension: "mlpackage")
            {
                let compiled = try MLModel.compileModel(at: packageURL)
                return compiled
            }
            if let modelURL = Bundle.module.url(
                forResource: "MXBAIEdgeColbert", withExtension: "mlmodel")
            {
                let compiled = try MLModel.compileModel(at: modelURL)
                return compiled
            }
        #endif
        throw MXBAIEdgeColbertGeneratorError.modelNotFound
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

    private static func formatEmbeddingsPreview(
        _ embeddings: [[Float]],
        headCount: Int = 5,
        tailCount: Int = 5
    ) -> String {
        guard !embeddings.isEmpty else { return "(empty)" }
        return embeddings.enumerated().map { index, vector in
            let formatted = formatVector(vector, headCount: headCount, tailCount: tailCount)
            return "  [token #\(index)] \(formatted)"
        }.joined(separator: "\n")
    }

    private static func formatVector(
        _ values: [Float],
        headCount: Int,
        tailCount: Int
    ) -> String {
        let formatter: (Float) -> String = { String(format: "%.9f", $0) }
        if values.count <= headCount + tailCount {
            return "[" + values.map(formatter).joined(separator: ", ") + "]"
        }

        let head = values.prefix(headCount).map(formatter)
        let tail = values.suffix(tailCount).map(formatter)
        return "[" + head.joined(separator: ", ") + ", …, " + tail.joined(separator: ", ") + "]"
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
