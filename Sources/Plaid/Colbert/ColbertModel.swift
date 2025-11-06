import Foundation

public enum ColbertModelError: Error, LocalizedError {
    case emptyInput
    case maskMismatch(expected: Int, actual: Int)
    case dimensionMismatch(expected: Int, actual: Int)

    public var errorDescription: String? {
        switch self {
        case .emptyInput:
            return "Input sentences cannot be empty."
        case .maskMismatch(let expected, let actual):
            return "Attention mask length mismatch. Expected \(expected) values, got \(actual)."
        case .dimensionMismatch(let expected, let actual):
            return "Embedding dimension mismatch. Expected \(expected), got \(actual)."
        }
    }
}

/// Swift port of the PyLate ColBERT orchestration logic.
///
/// This type encapsulates batching, normalization, padding behaviour and
/// similarity scoring using a pluggable embedding generator.
public struct ColbertModel {
    public struct Configuration {
        public var batchSize: Int
        public var embeddingDimension: Int
        public var queryLength: Int
        public var documentLength: Int
        public var doQueryExpansion: Bool
        public var attendToExpansionTokens: Bool

        public init(
            batchSize: Int = 32,
            embeddingDimension: Int,
            queryLength: Int = 32,
            documentLength: Int = 180,
            doQueryExpansion: Bool = true,
            attendToExpansionTokens: Bool = false
        ) {
            self.batchSize = batchSize
            self.embeddingDimension = embeddingDimension
            self.queryLength = queryLength
            self.documentLength = documentLength
            self.doQueryExpansion = doQueryExpansion
            self.attendToExpansionTokens = attendToExpansionTokens
        }
    }

    private let generator: ColbertEmbeddingGenerator
    private var config: Configuration
    private let chunker: SentenceChunker?

    public init(
        generator: ColbertEmbeddingGenerator,
        configuration: Configuration,
        chunker: SentenceChunker? = nil
    ) {
        self.generator = generator
        self.config = configuration
        self.chunker = chunker
    }

    public var batchSize: Int {
        get { config.batchSize }
        set { config.batchSize = newValue }
    }

    /// Encodes a single query or document into ColBERT embeddings.
    ///
    /// For large documents, this method automatically chunks the text using the configured
    /// `SentenceChunker` and processes each chunk independently, concatenating the results.
    /// Queries are never chunked as they are designed to be short.
    public func encode(sentence: String, isQuery: Bool) throws -> [[Float]] {
        let trimmed = sentence.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw ColbertModelError.emptyInput }

        // Queries are never chunked - they're short by design
        // Documents without a chunker use the original single-pass encoding
        guard !isQuery, let chunker = chunker else {
            return try encodeSinglePass(trimmed, isQuery: isQuery)
        }

        // For documents with a chunker: intelligently split and process each chunk
        return try encodeWithChunking(trimmed, chunker: chunker)
    }

    /// Encodes text in a single pass without chunking (original behavior).
    private func encodeSinglePass(_ text: String, isQuery: Bool) throws -> [[Float]] {
        let maxLength = isQuery ? config.queryLength : config.documentLength
        let generated = try generator.generateEmbeddings(
            for: text,
            isQuery: isQuery,
            maxLength: maxLength
        )

        guard !generated.embeddings.isEmpty else { throw ColbertModelError.emptyInput }

        return try processBatch(
            generated.embeddings,
            attentionMask: generated.attentionMask,
            isQuery: isQuery
        )
    }

    /// Encodes large documents by chunking them intelligently and concatenating embeddings.
    private func encodeWithChunking(_ text: String, chunker: SentenceChunker) throws -> [[Float]] {
        // Split document into manageable chunks using smart punctuation-aware splitting
        let chunks = chunker.chunk(
            for: text,
            chunkSize: config.documentLength,
            overlapSize: 64  // Overlap helps maintain context across chunk boundaries
        )

        guard !chunks.isEmpty else {
            throw ColbertModelError.emptyInput
        }

        print("📄 Document chunking: split into \(chunks.count) chunk(s)")

        // Process each chunk independently and concatenate all embeddings
        var allEmbeddings: [[Float]] = []
        allEmbeddings.reserveCapacity(chunks.count * config.documentLength / 2)  // Estimate

        for (index, chunk) in chunks.enumerated() {
            guard !chunk.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                continue  // Skip empty chunks
            }

            let preview = chunk.prefix(60)
            print(
                "  Chunk \(index + 1)/\(chunks.count): \"\(preview)\(chunk.count > 60 ? "..." : "")\""
            )

            let chunkEmbeddings = try encodeSinglePass(chunk, isQuery: false)
            allEmbeddings.append(contentsOf: chunkEmbeddings)

            print(
                "    → Encoded \(chunkEmbeddings.count) tokens, total: \(allEmbeddings.count) embeddings"
            )
        }

        guard !allEmbeddings.isEmpty else {
            throw ColbertModelError.emptyInput
        }

        print(
            "✅ Chunking complete: \(allEmbeddings.count) total embeddings from \(chunks.count) chunk(s)"
        )

        return allEmbeddings
    }

    /// Computes the ColBERT similarity score for a single query/document pair.
    public func similarity(
        query: [[Float]],
        document: [[Float]]
    ) throws -> Float {
        guard !query.isEmpty, !document.isEmpty else { throw ColbertModelError.emptyInput }
        try validateEmbeddingDimensions(query)
        try validateEmbeddingDimensions(document)

        var total: Float = 0
        for queryVector in query {
            var best: Float = -.infinity
            for documentVector in document {
                let dot = dotProduct(queryVector, documentVector)
                if dot > best {
                    best = dot
                }
            }
            total += best
        }

        return total
    }

    /// Produces the full interaction matrix for a single query/document pair.
    public func rawSimilarity(
        query: [[Float]],
        document: [[Float]]
    ) throws -> [[Float]] {
        guard !query.isEmpty, !document.isEmpty else { throw ColbertModelError.emptyInput }
        try validateEmbeddingDimensions(query)
        try validateEmbeddingDimensions(document)

        return try interactionMatrix(query, document)
    }
}

// MARK: - Internal helpers

extension ColbertModel {
    private func processBatch(
        _ embeddings: [[Float]],
        attentionMask: [Bool],
        isQuery: Bool
    ) throws -> [[Float]] {
        guard embeddings.count == attentionMask.count else {
            throw ColbertModelError.maskMismatch(
                expected: embeddings.count,
                actual: attentionMask.count
            )
        }

        if isQuery {
            return try normalizeAndPadQueries(embeddings: embeddings, attentionMask: attentionMask)
        }

        return try filterNormalizeAndPad(embeddings: embeddings, attentionMask: attentionMask)
    }

    private func filterNormalizeAndPad(
        embeddings: [[Float]],
        attentionMask: [Bool]
    ) throws -> [[Float]] {
        guard attentionMask.count == embeddings.count else {
            throw ColbertModelError.maskMismatch(
                expected: embeddings.count,
                actual: attentionMask.count
            )
        }

        var kept: [[Float]] = []
        kept.reserveCapacity(embeddings.count)

        for (vector, shouldKeep) in zip(embeddings, attentionMask) where shouldKeep {
            kept.append(try normalizeVector(vector))
        }

        if kept.isEmpty {
            let zero = Array(repeating: Float(0), count: config.embeddingDimension)
            kept.append(zero)
        }

        return kept
    }

    private func normalizeAndPadQueries(
        embeddings: [[Float]],
        attentionMask: [Bool]
    ) throws -> [[Float]] {
        let normalized = try embeddings.map { try normalizeVector($0) }
        let targetLength = config.queryLength
        var result: [[Float]] = []
        result.reserveCapacity(targetLength)

        let kept = zip(normalized, attentionMask)
            .compactMap { vector, shouldKeep in shouldKeep ? vector : nil }

        if kept.isEmpty {
            let zero = Array(repeating: Float(0), count: config.embeddingDimension)
            return Array(repeating: zero, count: targetLength)
        }

        result.append(contentsOf: kept)

        if result.count < targetLength {
            let zero = Array(repeating: Float(0), count: config.embeddingDimension)
            result.append(contentsOf: Array(repeating: zero, count: targetLength - result.count))
        } else if result.count > targetLength {
            result = Array(result.prefix(targetLength))
        }

        return result
    }

    private func validateEmbeddingDimensions(_ tokenVectors: [[Float]]) throws {
        for vector in tokenVectors {
            guard vector.count == config.embeddingDimension else {
                throw ColbertModelError.dimensionMismatch(
                    expected: config.embeddingDimension,
                    actual: vector.count
                )
            }
        }
    }

    private func normalizeVector(_ vector: [Float]) throws -> [Float] {
        guard vector.count == config.embeddingDimension else {
            throw ColbertModelError.dimensionMismatch(
                expected: config.embeddingDimension,
                actual: vector.count
            )
        }

        let norm = sqrt(vector.reduce(Float(0)) { $0 + $1 * $1 })
        if norm == 0 {
            return Array(repeating: 0, count: vector.count)
        }
        let invNorm = 1.0 / norm
        return vector.map { $0 * invNorm }
    }

    private func dotProduct(_ lhs: [Float], _ rhs: [Float]) -> Float {
        precondition(lhs.count == rhs.count, "Vector dimension mismatch")
        var result: Float = 0
        for i in 0 ..< lhs.count {
            result += lhs[i] * rhs[i]
        }
        return result
    }

    private func interactionMatrix(
        _ queryTokens: [[Float]],
        _ documentTokens: [[Float]]
    ) throws -> [[Float]] {
        guard
            let queryDim = queryTokens.first?.count,
            let documentDim = documentTokens.first?.count,
            queryDim == documentDim,
            queryDim == config.embeddingDimension
        else {
            throw ColbertModelError.dimensionMismatch(
                expected: config.embeddingDimension,
                actual: documentTokens.first?.count ?? 0
            )
        }

        return queryTokens.map { queryVector in
            documentTokens.map { documentVector in
                dotProduct(queryVector, documentVector)
            }
        }
    }
}
