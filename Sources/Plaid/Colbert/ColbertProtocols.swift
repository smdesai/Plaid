import Foundation

/// Batch of token-level embeddings returned by a ColBERT-compatible encoder.
///
/// `embeddings` follow `[batch, sequenceLength, embeddingDim]`.
/// `attentionMask` mirrors the token layout and is used to drop padded tokens.
public struct ColbertEmbeddingBatch {
    public let embeddings: [[Float]]
    public let attentionMask: [Bool]

    public init(embeddings: [[Float]], attentionMask: [Bool]) {
        self.embeddings = embeddings
        self.attentionMask = attentionMask
    }
}

/// Abstraction over the model that produces ColBERT token embeddings.
///
/// Different backends (MLX, Core ML, Metal, etc.) can conform to this protocol
/// to supply token-level embeddings alongside their attention masks.
public protocol ColbertEmbeddingGenerator {
    /// Generate embeddings for a single sentence
    func generateEmbeddings(
        for sentence: String,
        isQuery: Bool,
        maxLength: Int
    ) throws -> ColbertEmbeddingBatch

    /// Generate embeddings for multiple sentences in a single batch (more efficient)
    /// Default implementation falls back to single-sentence processing
    func generateEmbeddingsBatch(
        for sentences: [String],
        isQuery: Bool,
        maxLength: Int
    ) throws -> [ColbertEmbeddingBatch]
}

/// Default implementation processes sentences individually
/// Conformers should override with batched implementation for better performance
extension ColbertEmbeddingGenerator {
    public func generateEmbeddingsBatch(
        for sentences: [String],
        isQuery: Bool,
        maxLength: Int
    ) throws -> [ColbertEmbeddingBatch] {
        try sentences.map { sentence in
            try generateEmbeddings(for: sentence, isQuery: isQuery, maxLength: maxLength)
        }
    }
}

/// Defines how input sentences are chunked before encoding.
public protocol SentenceChunker {
    func chunk(for sentence: String, chunkSize: Int, overlapSize: Int) -> [String]
}

/// Default chunker that slices inputs into fixed-size batches.
public struct TokenSplitter: SentenceChunker {
    let tokenizer: any TokenizerProtocol

    public init(withTokenizer: any TokenizerProtocol) {
        self.tokenizer = withTokenizer
    }

    public func chunk(for sentence: String, chunkSize: Int = 180, overlapSize: Int = 64) -> [String]
    {
        let whitespace = CharacterSet.whitespacesAndNewlines
        guard !sentence.trimmingCharacters(in: whitespace).isEmpty else {
            return []
        }

        let effectiveChunkSize = min(chunkSize, 180)
        let effectiveOverlap = min(overlapSize, effectiveChunkSize - 1)

        // Show progress for large documents
        let textSize = sentence.count
        if textSize > 100_000 {
            print("   Tokenizing document...")
        }

        let tokens = tokenizer.tokenize(text: sentence)

        guard !tokens.isEmpty else { return [] }

        var chunks: [String] = []
        var position = 0

        // Estimate chunk count for progress reporting
        let step = max(1, effectiveChunkSize - effectiveOverlap)
        let estimatedChunks = (tokens.count + step - 1) / step
        let showProgress = estimatedChunks > 1000
        var lastReportedPercent = 0

        // Process tokens in sliding windows with overlap
        while position < tokens.count {
            // Calculate chunk boundaries
            let end = min(position + effectiveChunkSize, tokens.count)
            let chunkTokens = Array(tokens[position ..< end])

            // Convert tokens to text
            var chunkText = tokenizer.detokenize(tokens: chunkTokens)

            // Skip empty chunks
            guard !chunkText.trimmingCharacters(in: whitespace).isEmpty else {
                position = end
                continue
            }

            // Normalize whitespace and clean up
            chunkText = chunkText.replacingOccurrences(of: "\n", with: " ")
                .trimmingCharacters(in: whitespace)

            // Only append non-empty chunks
            if !chunkText.isEmpty {
                chunks.append(chunkText)
            }

            // Move position forward with overlap
            // Step size = chunkSize - overlap (ensures overlap between consecutive chunks)
            let step = max(1, effectiveChunkSize - effectiveOverlap)
            position += step

            // Report progress for very large documents
            if showProgress {
                let percentComplete = (position * 100) / tokens.count
                if percentComplete >= lastReportedPercent + 10 {
                    print(
                        "   Chunking progress: \(percentComplete)% (\(chunks.count) chunks created)"
                    )
                    lastReportedPercent = percentComplete
                }
            }
        }

        return chunks
    }
}
