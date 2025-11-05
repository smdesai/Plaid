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
    func generateEmbeddings(
        for sentence: String,
        isQuery: Bool,
        maxLength: Int
    ) throws -> ColbertEmbeddingBatch
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
        if sentence.isEmpty || sentence.trimmingCharacters(in: whitespace).isEmpty {
            return []
        }

        let chunkSize = min(chunkSize, 180)
        let tokens = tokenizer.tokenize(text: sentence)

        var chunks: [String] = []

        // Initialize a counter for the number of chunks
        var numChunks = 0

        // Create a variable to store the remaining tokens
        var remainingTokens = tokens

        // Loop until all tokens are consumed
        while !remainingTokens.isEmpty {
            // Take the first chunkSize tokens as a chunk
            let chunk = Array(remainingTokens.prefix(chunkSize))

            // Decode the chunk into text
            let chunkText = tokenizer.detokenize(tokens: chunk)

            // Skip the chunk if it is empty or whitespace
            if chunkText.isEmpty || chunkText.trimmingCharacters(in: whitespace).isEmpty {
                // Remove the tokens corresponding to the chunk text from the remaining tokens
                remainingTokens.removeFirst(chunk.count)
                // Continue to the next iteration of the loop
                continue
            }

            // Find the last period or punctuation mark in the chunk
            let punctuationMarks: [Character] = [".", "?", "!", "\n"]
            let lastPunctuation =
                punctuationMarks.compactMap {
                    chunkText.lastIndex(of: $0)?.utf16Offset(in: chunkText)
                }.max() ?? -1

            var chunkTextToAppend = chunkText

            // If there is a punctuation mark
            if lastPunctuation != -1 {
                // Ensure the index is within the chunkText bounds
                let safeIndex = min(chunkText.count - 1, lastPunctuation + 1)
                // Truncate the chunk text at the punctuation mark
                chunkTextToAppend = String(
                    chunkText[..<chunkText.index(chunkText.startIndex, offsetBy: safeIndex)])
            }

            // Remove any newline characters and strip any leading or trailing whitespace
            chunkTextToAppend = chunkTextToAppend.replacingOccurrences(of: "\n", with: " ")
                .trimmingCharacters(in: whitespace)

            // Append the chunk text to the list of chunks
            chunks.append(chunkTextToAppend)

            // Remove the tokens corresponding to the chunk text from the remaining tokens
            remainingTokens.removeFirst(tokenizer.tokenize(text: chunkTextToAppend).count)

            // Increment the number of chunks
            numChunks += 1
        }

        // Handle the remaining tokens
        if !remainingTokens.isEmpty {
            let remainingText = tokenizer.detokenize(tokens: remainingTokens).replacingOccurrences(
                of: "\n", with: " "
            ).trimmingCharacters(in: whitespace)

            chunks.append(remainingText)
        }

        return chunks
    }
}
