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

    /// Generate embeddings from pre-tokenized token IDs (performance optimized)
    /// Skips tokenization step for improved performance when processing chunks
    func generateEmbeddings(
        fromTokenIds tokenIds: [Int],
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

    /// Generate embeddings for multiple pre-tokenized token ID sequences in a batch
    /// Most efficient for processing pre-chunked documents
    func generateEmbeddingsBatch(
        fromTokenIds tokenIdBatch: [[Int]],
        isQuery: Bool,
        maxLength: Int
    ) throws -> [ColbertEmbeddingBatch]

    /// Tokenize text to IDs without adding special tokens (for chunking)
    func tokenizeToIds(text: String) -> [Int]
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

    public func generateEmbeddingsBatch(
        fromTokenIds tokenIdBatch: [[Int]],
        isQuery: Bool,
        maxLength: Int
    ) throws -> [ColbertEmbeddingBatch] {
        try tokenIdBatch.map { tokenIds in
            try generateEmbeddings(fromTokenIds: tokenIds, isQuery: isQuery, maxLength: maxLength)
        }
    }
}

/// Defines how input sentences are chunked before encoding.
public protocol SentenceChunker {
    func chunk(for sentence: String, chunkSize: Int, overlapSize: Int) -> [String]

    /// Chunk token IDs directly without intermediate text conversion (performance optimized)
    func chunkToIds(tokenIds: [Int], chunkSize: Int, overlapSize: Int) -> [[Int]]
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

    /// Performance-optimized chunking that operates directly on token IDs
    /// Avoids the expensive tokenize→detokenize→re-tokenize cycle
    public func chunkToIds(tokenIds: [Int], chunkSize: Int, overlapSize: Int) -> [[Int]] {
        guard !tokenIds.isEmpty else { return [] }

        let effectiveChunkSize = min(chunkSize, 180)
        let effectiveOverlap = min(overlapSize, effectiveChunkSize - 1)

        var chunks: [[Int]] = []
        var position = 0
        let step = max(1, effectiveChunkSize - effectiveOverlap)

        // Estimate chunk count for capacity reservation
        let estimatedChunks = (tokenIds.count + step - 1) / step
        chunks.reserveCapacity(estimatedChunks)

        // Process token IDs in sliding windows with overlap
        while position < tokenIds.count {
            let end = min(position + effectiveChunkSize, tokenIds.count)
            let chunkTokenIds = Array(tokenIds[position ..< end])

            chunks.append(chunkTokenIds)
            position += step
        }

        return chunks
    }
}

#if canImport(NaturalLanguage)
    import NaturalLanguage

    /// Sentence-boundary-aware chunker that preserves complete sentences in each chunk.
    ///
    /// Unlike `TokenSplitter` which uses fixed-size sliding windows that can split mid-sentence,
    /// this implementation respects linguistic boundaries by:
    /// - Detecting sentences using NaturalLanguage framework (handles abbreviations like Dr., Mr.)
    /// - Packing complete sentences into token budgets without splitting them
    /// - Using sentence-aligned overlap for semantic continuity
    /// - Falling back to clause-level splitting for very long sentences
    ///
    /// Trade-offs vs TokenSplitter:
    /// - Better semantic coherence in search results (complete thoughts)
    /// - ~10-20% overhead for sentence detection
    /// - ~5-15% lower token utilization (gaps at chunk boundaries)
    /// - Significantly improved search quality
    public struct SentenceBoundarySplitter: SentenceChunker {
        let tokenizer: any TokenizerProtocol

        public init(withTokenizer: any TokenizerProtocol) {
            self.tokenizer = withTokenizer
        }

        public func chunk(for text: String, chunkSize: Int = 180, overlapSize: Int = 64) -> [String]
        {
            let whitespace = CharacterSet.whitespacesAndNewlines
            guard !text.trimmingCharacters(in: whitespace).isEmpty else {
                return []
            }

            let effectiveChunkSize = min(chunkSize, 180)
            let effectiveOverlap = min(overlapSize, effectiveChunkSize - 1)

            // Show progress for large documents
            let textSize = text.count
            if textSize > 100_000 {
                print("   Detecting sentences in document...")
            }

            // Step 1: Detect sentence boundaries using NaturalLanguage
            let sentences = detectSentences(in: text)

            guard !sentences.isEmpty else { return [] }

            // Step 2: Calculate token counts for each sentence
            let sentenceTokenCounts = sentences.map { sentence in
                tokenizer.tokenize(text: sentence).count
            }

            // Step 3: Pack sentences into chunks respecting token budgets
            var chunks: [String] = []
            var currentSentences: [String] = []
            var currentTokenCount = 0
            var sentenceIndex = 0

            let showProgress = sentences.count > 1000
            var lastReportedPercent = 0

            while sentenceIndex < sentences.count {
                let sentence = sentences[sentenceIndex]
                let sentenceTokens = sentenceTokenCounts[sentenceIndex]

                // Handle very long sentences that exceed chunk size
                if sentenceTokens > effectiveChunkSize {
                    // Flush current chunk if not empty
                    if !currentSentences.isEmpty {
                        let chunkText = currentSentences.joined(separator: " ")
                            .trimmingCharacters(in: whitespace)
                        if !chunkText.isEmpty {
                            chunks.append(chunkText)
                        }
                        currentSentences = []
                        currentTokenCount = 0
                    }

                    // Split long sentence at clause boundaries
                    let longSentenceChunks = splitLongSentence(
                        sentence,
                        maxTokens: effectiveChunkSize
                    )
                    chunks.append(contentsOf: longSentenceChunks)
                    sentenceIndex += 1
                    continue
                }

                // Check if adding this sentence would exceed token budget
                if currentTokenCount + sentenceTokens > effectiveChunkSize
                    && !currentSentences.isEmpty
                {
                    // Flush current chunk
                    let chunkText = currentSentences.joined(separator: " ")
                        .trimmingCharacters(in: whitespace)
                    if !chunkText.isEmpty {
                        chunks.append(chunkText)
                    }

                    // Start new chunk with overlap
                    let overlapSentences = calculateOverlapSentences(
                        sentences: currentSentences,
                        tokenCounts: Array(
                            sentenceTokenCounts[
                                max(0, sentenceIndex - currentSentences.count) ..< sentenceIndex]),
                        maxOverlapTokens: effectiveOverlap
                    )

                    currentSentences = overlapSentences
                    currentTokenCount = overlapSentences.reduce(0) { count, sent in
                        count + (tokenizer.tokenize(text: sent).count)
                    }
                }

                // Add sentence to current chunk
                currentSentences.append(sentence)
                currentTokenCount += sentenceTokens
                sentenceIndex += 1

                // Report progress for very large documents
                if showProgress {
                    let percentComplete = (sentenceIndex * 100) / sentences.count
                    if percentComplete >= lastReportedPercent + 10 {
                        print(
                            "   Chunking progress: \(percentComplete)% (\(chunks.count) chunks created)"
                        )
                        lastReportedPercent = percentComplete
                    }
                }
            }

            // Flush remaining sentences
            if !currentSentences.isEmpty {
                let chunkText = currentSentences.joined(separator: " ")
                    .trimmingCharacters(in: whitespace)
                if !chunkText.isEmpty {
                    chunks.append(chunkText)
                }
            }

            return chunks
        }

        /// Detect sentence boundaries using NaturalLanguage framework
        private func detectSentences(in text: String) -> [String] {
            let tokenizer = NLTokenizer(unit: .sentence)
            tokenizer.string = text

            var sentences: [String] = []
            tokenizer.enumerateTokens(in: text.startIndex ..< text.endIndex) { range, _ in
                let sentence = String(text[range]).trimmingCharacters(in: .whitespacesAndNewlines)
                if !sentence.isEmpty {
                    sentences.append(sentence)
                }
                return true
            }

            return sentences
        }

        /// Calculate which sentences from the previous chunk should be included in overlap
        private func calculateOverlapSentences(
            sentences: [String],
            tokenCounts: [Int],
            maxOverlapTokens: Int
        ) -> [String] {
            guard !sentences.isEmpty else { return [] }

            var overlapSentences: [String] = []
            var overlapTokenCount = 0

            // Add sentences from the end until we hit the overlap budget
            for i in stride(from: sentences.count - 1, through: 0, by: -1) {
                let sentenceTokens = i < tokenCounts.count ? tokenCounts[i] : 0
                if overlapTokenCount + sentenceTokens > maxOverlapTokens {
                    break
                }
                overlapSentences.insert(sentences[i], at: 0)
                overlapTokenCount += sentenceTokens
            }

            return overlapSentences
        }

        /// Split a very long sentence at clause boundaries (commas, semicolons, etc.)
        /// Falls back to token-based splitting if no suitable boundaries found
        private func splitLongSentence(_ sentence: String, maxTokens: Int) -> [String] {
            // Try splitting at clause boundaries
            let clauseDelimiters = [";", ":", ",", " - ", " — "]

            for delimiter in clauseDelimiters {
                let parts = sentence.components(separatedBy: delimiter)
                if parts.count > 1 {
                    // Try packing clauses
                    var chunks: [String] = []
                    var currentParts: [String] = []
                    var currentTokens = 0

                    for (index, part) in parts.enumerated() {
                        let partWithDelimiter = index < parts.count - 1 ? part + delimiter : part
                        let partTokens = tokenizer.tokenize(text: partWithDelimiter).count

                        if currentTokens + partTokens > maxTokens && !currentParts.isEmpty {
                            chunks.append(currentParts.joined())
                            currentParts = [partWithDelimiter]
                            currentTokens = partTokens
                        } else {
                            currentParts.append(partWithDelimiter)
                            currentTokens += partTokens
                        }
                    }

                    if !currentParts.isEmpty {
                        chunks.append(currentParts.joined())
                    }

                    // If we successfully split, return
                    if chunks.count > 1
                        || (chunks.count == 1
                            && tokenizer.tokenize(text: chunks[0]).count <= maxTokens)
                    {
                        return chunks.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                            .filter { !$0.isEmpty }
                    }
                }
            }

            // Fallback: token-based splitting for sentences with no suitable boundaries
            let tokens = tokenizer.tokenize(text: sentence)
            var chunks: [String] = []
            var position = 0

            while position < tokens.count {
                let end = min(position + maxTokens, tokens.count)
                let chunkTokens = Array(tokens[position ..< end])
                let chunkText = tokenizer.detokenize(tokens: chunkTokens)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if !chunkText.isEmpty {
                    chunks.append(chunkText)
                }
                position = end
            }

            return chunks
        }

        /// Performance-optimized chunking that operates directly on token IDs
        ///
        /// Note: This implementation falls back to token-based chunking because we need
        /// the original text to detect sentence boundaries. For optimal performance with
        /// sentence boundaries, use the text-based `chunk()` method.
        public func chunkToIds(tokenIds: [Int], chunkSize: Int, overlapSize: Int) -> [[Int]] {
            // Sentence boundary detection requires original text
            // Fall back to token-based chunking (same as TokenSplitter)
            guard !tokenIds.isEmpty else { return [] }

            let effectiveChunkSize = min(chunkSize, 180)
            let effectiveOverlap = min(overlapSize, effectiveChunkSize - 1)

            var chunks: [[Int]] = []
            var position = 0
            let step = max(1, effectiveChunkSize - effectiveOverlap)

            let estimatedChunks = (tokenIds.count + step - 1) / step
            chunks.reserveCapacity(estimatedChunks)

            while position < tokenIds.count {
                let end = min(position + effectiveChunkSize, tokenIds.count)
                let chunkTokenIds = Array(tokenIds[position ..< end])
                chunks.append(chunkTokenIds)
                position += step
            }

            return chunks
        }
    }
#endif
