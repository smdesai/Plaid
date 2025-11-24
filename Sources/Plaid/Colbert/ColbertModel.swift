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

/// Represents a single chunk of a document with its text and embeddings
public struct DocumentChunk: Sendable {
    /// The index of this chunk within the original document (0-indexed)
    public let chunkIndex: Int

    /// The text content of this chunk
    public let text: String

    /// The ColBERT token embeddings for this chunk
    public let embeddings: [[Float]]

    public init(chunkIndex: Int, text: String, embeddings: [[Float]]) {
        self.chunkIndex = chunkIndex
        self.text = text
        self.embeddings = embeddings
    }
}

/// Result of encoding a document with chunk information preserved
public struct ChunkedDocumentEmbeddings: Sendable {
    /// Individual chunks with their text and embeddings
    public let chunks: [DocumentChunk]

    /// Total number of embeddings across all chunks
    public var totalEmbeddingCount: Int {
        chunks.reduce(0) { $0 + $1.embeddings.count }
    }

    /// All embeddings concatenated (for Plaid indexing)
    public var allEmbeddings: [[Float]] {
        chunks.flatMap { $0.embeddings }
    }

    public init(chunks: [DocumentChunk]) {
        self.chunks = chunks
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
    private let specialTokenCount = 2  // Accounts for [BOS] + [Q]/[D]

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

    /// Encodes a document and returns individual chunks with their text and embeddings.
    ///
    /// Unlike `encode()` which concatenates all embeddings, this method preserves
    /// the chunk boundaries so each chunk can be stored and retrieved separately.
    /// This enables returning the specific relevant chunk in search results.
    ///
    /// - Parameter text: The document text to encode
    /// - Returns: ChunkedDocumentEmbeddings containing individual chunks with text and embeddings
    public func encodeDocument(_ text: String) throws -> ChunkedDocumentEmbeddings {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw ColbertModelError.emptyInput }

        // If no chunker configured, treat entire document as single chunk
        guard let chunker = chunker else {
            let embeddings = try encodeSinglePass(trimmed, isQuery: false)
            let chunk = DocumentChunk(chunkIndex: 0, text: trimmed, embeddings: embeddings)
            return ChunkedDocumentEmbeddings(chunks: [chunk])
        }

        // Get text-based chunks so we preserve the actual chunk text
        let contentTokenBudget = max(config.documentLength - specialTokenCount, 1)
        let overlapTokens = min(64, max(contentTokenBudget - 1, 0))
        let textChunks = chunker.chunk(
            for: trimmed, chunkSize: contentTokenBudget, overlapSize: overlapTokens)

        guard !textChunks.isEmpty else {
            throw ColbertModelError.emptyInput
        }

        print("✅ Document chunked: \(textChunks.count) chunk(s) created")

        // Process each chunk and preserve its text
        var documentChunks: [DocumentChunk] = []
        documentChunks.reserveCapacity(textChunks.count)

        for (index, chunkText) in textChunks.enumerated() {
            guard !chunkText.isEmpty else { continue }

            let generated = try generator.generateEmbeddings(
                for: chunkText,
                isQuery: false,
                maxLength: config.documentLength
            )

            let chunkEmbeddings = try processBatch(
                generated.embeddings,
                attentionMask: generated.attentionMask,
                isQuery: false
            )

            let chunk = DocumentChunk(
                chunkIndex: index,
                text: chunkText,
                embeddings: chunkEmbeddings
            )
            documentChunks.append(chunk)
        }

        guard !documentChunks.isEmpty else {
            throw ColbertModelError.emptyInput
        }

        let totalEmbeddings = documentChunks.reduce(0) { $0 + $1.embeddings.count }
        print(
            "✅ Encoding complete: \(totalEmbeddings) total embeddings from \(documentChunks.count) chunk(s)"
        )

        return ChunkedDocumentEmbeddings(chunks: documentChunks)
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
        let textSize = text.count
        if textSize > 100_000 {
            print("📄 Chunking large document (\(textSize) chars)...")
        }

        // PERFORMANCE OPTIMIZATION: Use token ID-based chunking to avoid re-tokenization
        // Tokenize once, chunk the IDs, then pass directly to embedding generator
        let tokenIds = generator.tokenizeToIds(text: text)
        let contentTokenBudget = max(config.documentLength - specialTokenCount, 1)
        let overlapTokens = min(64, max(contentTokenBudget - 1, 0))
        let tokenChunks = chunker.chunkToIds(
            tokenIds: tokenIds,
            chunkSize: contentTokenBudget,
            overlapSize: overlapTokens
        )

        guard !tokenChunks.isEmpty else {
            throw ColbertModelError.emptyInput
        }

        print("✅ Document chunked: \(tokenChunks.count) chunk(s) created")

        // Use batched processing for large documents (>100 chunks)
        if tokenChunks.count > 100 {
            return try encodeChunksBatchedOptimized(tokenChunks)
        }

        // Process each chunk independently and concatenate all embeddings
        var allEmbeddings: [[Float]] = []
        allEmbeddings.reserveCapacity(tokenChunks.count * config.documentLength / 2)  // Estimate

        for tokenChunk in tokenChunks {
            guard !tokenChunk.isEmpty else {
                continue  // Skip empty chunks
            }

            let generated = try generator.generateEmbeddings(
                fromTokenIds: tokenChunk,
                isQuery: false,
                maxLength: config.documentLength
            )

            let chunkEmbeddings = try processBatch(
                generated.embeddings,
                attentionMask: generated.attentionMask,
                isQuery: false
            )
            allEmbeddings.append(contentsOf: chunkEmbeddings)
        }

        guard !allEmbeddings.isEmpty else {
            throw ColbertModelError.emptyInput
        }

        print(
            "✅ Chunking complete: \(allEmbeddings.count) total embeddings from \(tokenChunks.count) chunk(s)"
        )

        return allEmbeddings
    }

    /// Process chunks in batches for optimal performance (optimized token ID path)
    private func encodeChunksBatchedOptimized(_ tokenChunks: [[Int]]) throws -> [[Float]] {
        var allEmbeddings: [[Float]] = []
        allEmbeddings.reserveCapacity(tokenChunks.count * config.documentLength / 2)

        let batchSize = config.batchSize  // Use configured batch size (default 32)
        let totalBatches = (tokenChunks.count + batchSize - 1) / batchSize

        var lastReportedPercent = 0

        for batchIndex in 0 ..< totalBatches {
            let startIdx = batchIndex * batchSize
            let endIdx = min(startIdx + batchSize, tokenChunks.count)
            let batchTokenChunks = Array(tokenChunks[startIdx ..< endIdx])

            // Filter empty chunks
            let validChunks = batchTokenChunks.filter { !$0.isEmpty }

            guard !validChunks.isEmpty else { continue }

            // Process entire batch in single model call (optimized token ID path)
            let batchResults = try generator.generateEmbeddingsBatch(
                fromTokenIds: validChunks,
                isQuery: false,
                maxLength: config.documentLength
            )

            // Process each result
            for generated in batchResults {
                let processed = try processBatch(
                    generated.embeddings,
                    attentionMask: generated.attentionMask,
                    isQuery: false
                )
                allEmbeddings.append(contentsOf: processed)
            }

            // Show progress
            let percentComplete = ((batchIndex + 1) * 100) / totalBatches
            if percentComplete >= lastReportedPercent + 10 {
                let chunksProcessed = endIdx
                print(
                    "   Encoding progress: \(percentComplete)% (\(chunksProcessed)/\(tokenChunks.count) chunks) [batch \(batchIndex + 1)/\(totalBatches)]"
                )
                lastReportedPercent = percentComplete
            }
        }

        guard !allEmbeddings.isEmpty else {
            throw ColbertModelError.emptyInput
        }

        print(
            "✅ Chunking complete: \(allEmbeddings.count) total embeddings from \(tokenChunks.count) chunk(s)"
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
